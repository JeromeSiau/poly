"""Two-sided paper trading recorder and replay.

Adapts two-sided ``TradeIntent`` / ``FillResult`` objects into the
unified ``TradeRecorder`` pipeline, and provides ``replay_into_engine``
to restore inventory state from persisted fills on restart.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import timezone
from typing import Any, Optional

from sqlalchemy import select

from src.arb.two_sided_inventory import (
    FillResult,
    MarketSnapshot,
    TradeIntent,
    TwoSidedInventoryEngine,
)
from src.db.database import get_sync_session
from src.db.models import LiveObservation, PaperTrade
from src.execution.models import FillResult as ExecFillResult
from src.execution.models import TradeIntent as ExecTradeIntent
from src.execution.trade_recorder import TradeRecorder
from src.utils.parsing import _ensure_sync_db_url, _to_float

TWO_SIDED_EVENT_TYPE = "two_sided_inventory"


@dataclass(slots=True)
class PendingMakerOrder:
    intent: TradeIntent
    placed_at: float


def _to_exec_intent(intent: TradeIntent) -> ExecTradeIntent:
    """Convert two-sided TradeIntent to execution TradeIntent."""
    return ExecTradeIntent(
        condition_id=intent.condition_id,
        token_id=intent.token_id,
        outcome=intent.outcome,
        side=intent.side,
        price=intent.price,
        size_usd=intent.size_usd,
        reason=intent.reason,
        title=intent.title,
        edge_pct=intent.edge_pct,
        timestamp=intent.timestamp,
    )


def _to_exec_fill(fill: FillResult) -> ExecFillResult:
    """Convert two-sided FillResult to execution FillResult."""
    return ExecFillResult(
        filled=fill.shares > 0,
        shares=fill.shares,
        avg_price=fill.fill_price,
        pnl_delta=fill.realized_pnl_delta,
    )


def _build_extra_state(
    *,
    intent: TradeIntent,
    fill: FillResult,
    snapshot: Optional[MarketSnapshot],
    min_edge_pct: float = 0.0,
    exit_edge_pct: float = 0.0,
) -> dict[str, Any]:
    """Build the two-sided-specific extra game_state fields."""
    quote = snapshot.outcomes.get(intent.outcome) if snapshot else None
    extra: dict[str, Any] = {
        "min_edge_pct": min_edge_pct,
        "exit_edge_pct": exit_edge_pct,
        "slug": snapshot.slug if snapshot else "",
        "inventory_avg_price": fill.avg_price,
        "inventory_remaining_shares": fill.remaining_shares,
        "liquidity": snapshot.liquidity if snapshot else None,
        "volume_24h": snapshot.volume_24h if snapshot else None,
        "market_bid": quote.bid if quote else None,
        "market_ask": quote.ask if quote else None,
    }
    if intent.reason.startswith("settlement_"):
        extra["is_settlement"] = True
    if intent.reason == "pair_merge":
        extra["is_pair_merge"] = True
    return extra


class TwoSidedPaperRecorder:
    """Persist and replay two-sided paper fills.

    Persistence is now delegated to :class:`TradeRecorder`; this class
    retains the two-sided-specific ``replay_into_engine`` logic and a
    backward-compatible ``persist_fill`` wrapper used by ``run_sniper.py``.
    """

    def __init__(
        self,
        database_url: str,
        *,
        strategy_tag: str,
        run_id: str,
        min_edge_pct: float,
        exit_edge_pct: float,
        event_type: str = TWO_SIDED_EVENT_TYPE,
    ) -> None:
        self._database_url = _ensure_sync_db_url(database_url)
        self._strategy_tag = strategy_tag
        self._run_id = run_id
        self._min_edge_pct = min_edge_pct
        self._exit_edge_pct = exit_edge_pct
        self._event_type = event_type
        self._recorder = TradeRecorder(
            db_url=database_url,
            strategy_tag=strategy_tag,
            event_type=event_type,
            run_id=run_id,
        )

    def bootstrap(self) -> None:
        self._recorder.bootstrap()

    def replay_into_engine(self, engine: TwoSidedInventoryEngine) -> int:
        stmt = (
            select(PaperTrade, LiveObservation)
            .join(LiveObservation, LiveObservation.id == PaperTrade.observation_id)
            .where(LiveObservation.event_type == self._event_type)
            .order_by(PaperTrade.created_at.asc(), PaperTrade.id.asc())
        )
        session = get_sync_session(self._database_url)
        try:
            rows = session.execute(stmt).all()
        finally:
            session.close()

        restored = 0
        for trade, observation in rows:
            game_state = observation.game_state if isinstance(observation.game_state, dict) else {}
            strategy_tag = str(game_state.get("strategy_tag") or "default")
            if strategy_tag != self._strategy_tag:
                continue
            condition_id = str(game_state.get("condition_id") or observation.match_id or "")
            outcome = str(game_state.get("outcome") or "")
            token_id = str(game_state.get("token_id") or "")
            title = str(game_state.get("title") or condition_id or "restored")
            side = str(trade.side or game_state.get("side") or "").upper()
            price = _to_float(
                trade.simulated_fill_price if trade.simulated_fill_price is not None else trade.entry_price,
                default=0.0,
            )
            size_usd = _to_float(trade.size, default=0.0)
            if not condition_id or not outcome or side not in {"BUY", "SELL"}:
                continue

            timestamp = trade.created_at or observation.timestamp
            if timestamp is None:
                ts = time.time()
            else:
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                ts = timestamp.timestamp()

            reason = str(game_state.get("reason") or "restore")
            if side == "SELL" and reason.startswith("settlement_"):
                fill = engine.settle_position(
                    condition_id=condition_id,
                    outcome=outcome,
                    settlement_price=max(0.0, price),
                    timestamp=ts,
                )
                if fill.shares > 0:
                    restored += 1
                continue

            if price <= 0 or size_usd <= 0:
                continue
            restored_intent = TradeIntent(
                condition_id=condition_id,
                title=title,
                outcome=outcome,
                token_id=token_id,
                side=side,
                price=price,
                size_usd=size_usd,
                edge_pct=_to_float(trade.edge_theoretical, default=0.0),
                reason=reason,
                timestamp=ts,
            )
            fill = engine.apply_fill(restored_intent)
            if fill.shares > 0:
                restored += 1
        return restored

    def persist_fill(
        self,
        *,
        intent: TradeIntent,
        fill: FillResult,
        snapshot: Optional[MarketSnapshot],
        fair_prices: dict[str, float],
        execution_mode: str,
    ) -> None:
        """Backward-compatible wrapper: delegates to TradeRecorder."""
        if fill.shares <= 0:
            return

        extra = _build_extra_state(
            intent=intent,
            fill=fill,
            snapshot=snapshot,
            min_edge_pct=self._min_edge_pct,
            exit_edge_pct=self._exit_edge_pct,
        )
        exec_intent = _to_exec_intent(intent)
        exec_fill = _to_exec_fill(fill)
        is_settle = intent.reason.startswith("settlement_") or intent.reason == "pair_merge"
        if is_settle:
            self._recorder.record_settle(
                intent=exec_intent,
                fill=exec_fill,
                fair_prices=fair_prices,
                extra_state=extra,
            )
        else:
            self._recorder.record_fill(
                intent=exec_intent,
                fill=exec_fill,
                fair_prices=fair_prices,
                execution_mode=execution_mode,
                extra_state=extra,
            )
