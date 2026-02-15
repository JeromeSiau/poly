"""Database-backed risk guard with circuit breaker, daily halt, and staleness detection.

Each strategy process instantiates its own RiskGuard. Shared state lives in the
``risk_state`` table so independent daemons can enforce a global daily loss
limit cooperatively.

Uses SQLAlchemy ORM — works with both SQLite and MySQL.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import func as sa_func, select
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    create_async_engine,
    async_sessionmaker,
)
import structlog

from src.db.models import Base, RiskState

logger = structlog.get_logger()

# Module-level engine cache: db_url -> (engine, session_factory)
_engines: dict[str, tuple] = {}


def _get_factory(db_url: str) -> async_sessionmaker[AsyncSession]:
    """Get or create an async session factory for the given URL."""
    if db_url not in _engines:
        engine = create_async_engine(db_url, echo=False, pool_pre_ping=True)
        factory = async_sessionmaker(
            bind=engine, class_=AsyncSession, expire_on_commit=False,
        )
        _engines[db_url] = (engine, factory)
    return _engines[db_url][1]


class RiskGuard:
    """Per-strategy risk guard backed by a shared database.

    Parameters
    ----------
    strategy_tag:
        Unique identifier for this strategy instance.
    db_url:
        SQLAlchemy async database URL (e.g. ``mysql+aiomysql://...``
        or ``sqlite+aiosqlite:///data/arb.db``).
    max_consecutive_losses:
        Circuit-break after this many consecutive losses.
    max_drawdown_usd:
        Circuit-break when session PnL drops to (or below) this value.
        Must be negative.
    stale_seconds:
        Maximum age of the last book update before we pause trading
        (temporary, not a circuit break).
    daily_loss_limit_usd:
        Global daily loss limit summed across **all** strategies in the
        same database.  Must be negative.
    telegram_alerter:
        Optional ``TelegramAlerter`` (or duck-typed object with
        ``async send_custom_alert(msg) -> bool``).
    """

    def __init__(
        self,
        strategy_tag: str,
        db_url: str,
        max_consecutive_losses: int = 5,
        max_drawdown_usd: float = -50.0,
        stale_seconds: float = 30.0,
        stale_cancel_seconds: float = 120.0,
        stale_exit_seconds: float = 300.0,
        daily_loss_limit_usd: float = -200.0,
        telegram_alerter: Optional[object] = None,
        # Legacy alias — ignored, kept for backwards compat during migration
        db_path: str = "",
    ) -> None:
        self.strategy_tag = strategy_tag
        self.db_url = db_url
        self.max_consecutive_losses = max_consecutive_losses
        self.max_drawdown_usd = max_drawdown_usd
        self.stale_seconds = stale_seconds
        self.stale_cancel_seconds = stale_cancel_seconds
        self.stale_exit_seconds = stale_exit_seconds
        self.daily_loss_limit_usd = daily_loss_limit_usd
        self._alerter = telegram_alerter
        self._factory: Optional[async_sessionmaker] = None

        # Local (in-memory) state — fast reads, persisted on writes.
        self.daily_pnl: float = 0.0
        self.session_pnl: float = 0.0
        self.consecutive_losses: int = 0
        self.circuit_broken: bool = False
        self.circuit_reason: str = ""
        self._alert_sent: bool = False
        self._last_stale_log: float = 0.0
        # Stale escalation state (reset when book recovers).
        self.should_cancel_orders: bool = False
        self._stale_cancel_alerted: bool = False

    def _get_session_factory(self) -> async_sessionmaker[AsyncSession]:
        if self._factory is None:
            self._factory = _get_factory(self.db_url)
        return self._factory

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Create the ``risk_state`` table (if needed) and restore or reset state.

        * Same UTC calendar day  ->  keep ``daily_pnl`` from previous run.
        * Different day (or no row) ->  reset ``daily_pnl`` to 0.
        * ``session_pnl`` and ``consecutive_losses`` always start at 0
          on a new process start.
        """
        now = time.time()
        factory = self._get_session_factory()

        # Ensure table exists
        engine = _engines[self.db_url][0]
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all, tables=[RiskState.__table__])

        async with factory() as session:
            # Read existing state
            result = await session.execute(
                select(RiskState).where(RiskState.strategy_tag == self.strategy_tag)
            )
            existing = result.scalar_one_or_none()

            if existing is not None:
                if self._same_utc_day(existing.updated_at or 0, now):
                    self.daily_pnl = existing.daily_pnl or 0.0
                else:
                    self.daily_pnl = 0.0
            else:
                self.daily_pnl = 0.0

            # Session state always resets on new process.
            self.session_pnl = 0.0
            self.consecutive_losses = 0
            self.circuit_broken = False
            self.circuit_reason = ""

            # Upsert via merge
            row = RiskState(
                strategy_tag=self.strategy_tag,
                daily_pnl=self.daily_pnl,
                session_pnl=self.session_pnl,
                consec_losses=self.consecutive_losses,
                last_heartbeat=now,
                circuit_broken=False,
                circuit_reason="",
                updated_at=now,
            )
            await session.merge(row)
            await session.commit()

        logger.info(
            "risk_guard_initialized",
            strategy_tag=self.strategy_tag,
            daily_pnl=self.daily_pnl,
        )

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    async def record_result(self, *, pnl: float, won: bool) -> bool:
        """Record a trade result and check circuit-breaker conditions.

        Returns ``True`` if trading is still allowed, ``False`` otherwise.
        """
        self.daily_pnl += pnl
        self.session_pnl += pnl

        if won:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1

        now = time.time()

        # Persist to DB first (so global halt sees fresh numbers).
        await self._persist_state(now)

        # --- Circuit-breaker checks ---

        if self.consecutive_losses >= self.max_consecutive_losses:
            reason = (
                f"consecutive losses: {self.consecutive_losses} "
                f"(limit {self.max_consecutive_losses})"
            )
            await self._trigger_circuit_break(reason, now)
            return False

        if self.session_pnl <= self.max_drawdown_usd:
            reason = (
                f"session drawdown: ${self.session_pnl:.2f} "
                f"(limit ${self.max_drawdown_usd:.2f})"
            )
            await self._trigger_circuit_break(reason, now)
            return False

        # Global daily halt (sum across all strategies).
        if await self._check_global_halt():
            return False

        return True

    async def is_trading_allowed(self, *, last_book_update: float) -> bool:
        """Check whether this strategy is allowed to place orders right now.

        * Circuit broken  ->  ``False`` (permanent until restart).
        * Stale book      ->  ``False`` (temporary — not a circuit break).
          - ``> stale_seconds`` (30s default): block new orders.
          - ``> stale_cancel_seconds`` (120s): set ``should_cancel_orders``
            flag so the caller can pull live orders. Sends Telegram alert.
          - ``> stale_exit_seconds`` (300s): raise ``SystemExit`` so the
            process supervisor restarts the strategy cleanly.
        * Global halt     ->  ``False`` (circuit break).
        * Otherwise       ->  ``True``.
        """
        if self.circuit_broken:
            return False

        now = time.time()
        stale_age = now - last_book_update

        if stale_age > self.stale_seconds:
            # --- Escalation 3: exit process for clean restart ---
            if self.stale_exit_seconds > 0 and stale_age > self.stale_exit_seconds:
                logger.error(
                    "stale_exit",
                    strategy_tag=self.strategy_tag,
                    age_s=round(stale_age, 0),
                    threshold_s=self.stale_exit_seconds,
                )
                await self._send_stale_alert(
                    f"STALE EXIT: book stale {stale_age:.0f}s, killing process"
                )
                raise SystemExit(1)

            # --- Escalation 2: signal caller to cancel live orders ---
            if self.stale_cancel_seconds > 0 and stale_age > self.stale_cancel_seconds:
                if not self._stale_cancel_alerted:
                    self._stale_cancel_alerted = True
                    self.should_cancel_orders = True
                    logger.warning(
                        "stale_cancel_orders",
                        strategy_tag=self.strategy_tag,
                        age_s=round(stale_age, 0),
                        threshold_s=self.stale_cancel_seconds,
                    )
                    await self._send_stale_alert(
                        f"STALE: book stale {stale_age:.0f}s, cancelling orders"
                    )

            # --- Escalation 1: block new orders (rate-limited log) ---
            if now - self._last_stale_log >= 30.0:
                self._last_stale_log = now
                logger.warning(
                    "book_stale",
                    strategy_tag=self.strategy_tag,
                    age_s=round(stale_age, 1),
                    threshold_s=self.stale_seconds,
                )
            return False

        # Book is fresh — reset stale escalation state.
        if self._stale_cancel_alerted:
            self._stale_cancel_alerted = False
            self.should_cancel_orders = False
            logger.info("book_recovered", strategy_tag=self.strategy_tag)

        if await self._check_global_halt():
            return False

        return True

    async def heartbeat(self) -> None:
        """Update ``last_heartbeat`` in the database."""
        now = time.time()
        factory = self._get_session_factory()
        async with factory() as session:
            result = await session.execute(
                select(RiskState).where(RiskState.strategy_tag == self.strategy_tag)
            )
            row = result.scalar_one_or_none()
            if row:
                row.last_heartbeat = now
                row.updated_at = now
                await session.commit()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _check_global_halt(self) -> bool:
        """Return ``True`` if the global daily loss limit has been breached.

        The limit is checked by summing ``daily_pnl`` across **all** rows
        in ``risk_state``.
        """
        factory = self._get_session_factory()
        async with factory() as session:
            result = await session.execute(
                select(sa_func.coalesce(sa_func.sum(RiskState.daily_pnl), 0.0))
            )
            total = result.scalar()

        if total <= self.daily_loss_limit_usd:
            reason = f"global daily halt: total ${total:.2f}"
            now = time.time()
            await self._trigger_circuit_break(reason, now)
            return True
        return False

    async def _trigger_circuit_break(self, reason: str, now: float) -> None:
        """Set circuit-broken state, persist, log, and alert."""
        self.circuit_broken = True
        self.circuit_reason = reason
        await self._persist_circuit_break(reason, now)
        await self._send_alert(reason)

    async def _persist_circuit_break(self, reason: str, now: float) -> None:
        """Write circuit-broken flag and reason to the database."""
        logger.warning(
            "circuit_break_triggered",
            strategy_tag=self.strategy_tag,
            reason=reason,
        )
        factory = self._get_session_factory()
        async with factory() as session:
            result = await session.execute(
                select(RiskState).where(RiskState.strategy_tag == self.strategy_tag)
            )
            row = result.scalar_one_or_none()
            if row:
                row.circuit_broken = True
                row.circuit_reason = reason
                row.updated_at = now
                await session.commit()

    async def _persist_state(self, now: float) -> None:
        """Persist the full local state to the database."""
        factory = self._get_session_factory()
        async with factory() as session:
            row = RiskState(
                strategy_tag=self.strategy_tag,
                daily_pnl=self.daily_pnl,
                session_pnl=self.session_pnl,
                consec_losses=self.consecutive_losses,
                last_heartbeat=now,
                circuit_broken=self.circuit_broken,
                circuit_reason=self.circuit_reason,
                updated_at=now,
            )
            await session.merge(row)
            await session.commit()

    async def _send_alert(self, reason: str) -> None:
        """Send a one-shot Telegram alert (at most one per circuit break)."""
        if self._alert_sent or self._alerter is None:
            return
        self._alert_sent = True

        msg = (
            f"CIRCUIT BREAKER {self.strategy_tag}\n"
            f"Reason: {reason}\n"
            f"Session PnL: ${self.session_pnl:.2f}\n"
            f"Daily PnL: ${self.daily_pnl:.2f}"
        )
        try:
            await self._alerter.send_custom_alert(msg)
        except Exception:
            logger.exception("telegram_alert_failed", strategy_tag=self.strategy_tag)

    async def _send_stale_alert(self, message: str) -> None:
        """Send a Telegram alert for stale escalation (not gated by _alert_sent)."""
        if self._alerter is None:
            return
        try:
            await self._alerter.send_custom_alert(f"⚠️ {self.strategy_tag}\n{message}")
        except Exception:
            logger.exception("telegram_stale_alert_failed", strategy_tag=self.strategy_tag)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _same_utc_day(ts_a: float, ts_b: float) -> bool:
        """Return ``True`` if two Unix timestamps fall on the same UTC date."""
        day_a = datetime.fromtimestamp(ts_a, tz=timezone.utc).date()
        day_b = datetime.fromtimestamp(ts_b, tz=timezone.utc).date()
        return day_a == day_b
