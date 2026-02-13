"""SQLite-backed risk guard with circuit breaker, daily halt, and staleness detection.

Each strategy process instantiates its own RiskGuard. Shared state lives in the
``risk_state`` table so independent daemons can enforce a global daily loss
limit cooperatively.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional

import aiosqlite
import structlog

logger = structlog.get_logger()

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS risk_state (
    strategy_tag    TEXT PRIMARY KEY,
    daily_pnl       REAL DEFAULT 0.0,
    session_pnl     REAL DEFAULT 0.0,
    consec_losses   INTEGER DEFAULT 0,
    last_heartbeat  REAL,
    circuit_broken  INTEGER DEFAULT 0,
    circuit_reason  TEXT,
    updated_at      REAL
);
"""

_UPSERT_SQL = """
INSERT INTO risk_state (
    strategy_tag, daily_pnl, session_pnl, consec_losses,
    last_heartbeat, circuit_broken, circuit_reason, updated_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(strategy_tag) DO UPDATE SET
    daily_pnl      = excluded.daily_pnl,
    session_pnl    = excluded.session_pnl,
    consec_losses  = excluded.consec_losses,
    last_heartbeat = excluded.last_heartbeat,
    circuit_broken = excluded.circuit_broken,
    circuit_reason = excluded.circuit_reason,
    updated_at     = excluded.updated_at;
"""


class RiskGuard:
    """Per-strategy risk guard backed by a shared SQLite database.

    Parameters
    ----------
    strategy_tag:
        Unique identifier for this strategy instance.
    db_path:
        Path to the SQLite database file (e.g. ``data/arb.db``).
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
        db_path: str,
        max_consecutive_losses: int = 5,
        max_drawdown_usd: float = -50.0,
        stale_seconds: float = 30.0,
        daily_loss_limit_usd: float = -200.0,
        telegram_alerter: Optional[object] = None,
    ) -> None:
        self.strategy_tag = strategy_tag
        self.db_path = db_path
        self.max_consecutive_losses = max_consecutive_losses
        self.max_drawdown_usd = max_drawdown_usd
        self.stale_seconds = stale_seconds
        self.daily_loss_limit_usd = daily_loss_limit_usd
        self._alerter = telegram_alerter

        # Local (in-memory) state — fast reads, persisted on writes.
        self.daily_pnl: float = 0.0
        self.session_pnl: float = 0.0
        self.consecutive_losses: int = 0
        self.circuit_broken: bool = False
        self.circuit_reason: str = ""
        self._alert_sent: bool = False

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
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute(_CREATE_TABLE_SQL)
            await conn.commit()

            cursor = await conn.execute(
                "SELECT daily_pnl, updated_at FROM risk_state WHERE strategy_tag = ?",
                (self.strategy_tag,),
            )
            row = await cursor.fetchone()

            if row is not None:
                prev_daily_pnl, updated_at = row
                if self._same_utc_day(updated_at, now):
                    self.daily_pnl = prev_daily_pnl
                else:
                    self.daily_pnl = 0.0
            else:
                self.daily_pnl = 0.0

            # Session state always resets on new process.
            self.session_pnl = 0.0
            self.consecutive_losses = 0
            self.circuit_broken = False
            self.circuit_reason = ""

            await conn.execute(
                _UPSERT_SQL,
                (
                    self.strategy_tag,
                    self.daily_pnl,
                    self.session_pnl,
                    self.consecutive_losses,
                    now,
                    int(self.circuit_broken),
                    self.circuit_reason,
                    now,
                ),
            )
            await conn.commit()

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
        * Global halt     ->  ``False`` (circuit break).
        * Otherwise       ->  ``True``.
        """
        if self.circuit_broken:
            return False

        now = time.time()
        if now - last_book_update > self.stale_seconds:
            logger.warning(
                "book_stale",
                strategy_tag=self.strategy_tag,
                age_s=round(now - last_book_update, 1),
                threshold_s=self.stale_seconds,
            )
            return False

        if await self._check_global_halt():
            return False

        return True

    async def heartbeat(self) -> None:
        """Update ``last_heartbeat`` in the database."""
        now = time.time()
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute(
                "UPDATE risk_state SET last_heartbeat = ?, updated_at = ? "
                "WHERE strategy_tag = ?",
                (now, now, self.strategy_tag),
            )
            await conn.commit()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _check_global_halt(self) -> bool:
        """Return ``True`` if the global daily loss limit has been breached.

        The limit is checked by summing ``daily_pnl`` across **all** rows
        in ``risk_state``.
        """
        async with aiosqlite.connect(self.db_path) as conn:
            cursor = await conn.execute(
                "SELECT COALESCE(SUM(daily_pnl), 0.0) FROM risk_state"
            )
            row = await cursor.fetchone()
            total = row[0]

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
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute(
                "UPDATE risk_state SET circuit_broken = 1, circuit_reason = ?, "
                "updated_at = ? WHERE strategy_tag = ?",
                (reason, now, self.strategy_tag),
            )
            await conn.commit()

    async def _persist_state(self, now: float) -> None:
        """Persist the full local state to the database."""
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute(
                _UPSERT_SQL,
                (
                    self.strategy_tag,
                    self.daily_pnl,
                    self.session_pnl,
                    self.consecutive_losses,
                    now,
                    int(self.circuit_broken),
                    self.circuit_reason,
                    now,
                ),
            )
            await conn.commit()

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

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _same_utc_day(ts_a: float, ts_b: float) -> bool:
        """Return ``True`` if two Unix timestamps fall on the same UTC date."""
        day_a = datetime.fromtimestamp(ts_a, tz=timezone.utc).date()
        day_b = datetime.fromtimestamp(ts_b, tz=timezone.utc).date()
        return day_a == day_b
