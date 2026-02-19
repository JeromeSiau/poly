# src/td_maker/sizing.py
from __future__ import annotations
import time
import uuid
from typing import Optional, Any
from src.td_maker.state import MarketState, MarketRegistry, PassiveOrder


def compute_rung_prices(lo: float, hi: float, n_rungs: int) -> list[float]:
    """Evenly-spaced rung prices in [lo, hi], snapped to 1c, deduplicated."""
    if n_rungs <= 1:
        return [round((lo + hi) / 2, 2)]
    raw = [lo + i * (hi - lo) / (n_rungs - 1) for i in range(n_rungs)]
    seen: set[float] = set()
    result: list[float] = []
    for p in raw:
        r = round(p, 2)
        if r not in seen:
            seen.add(r)
            result.append(r)
    return result


def model_size_scale(model: Any, p_win: float) -> float:
    """Map model P(win) → position size multiplier [0.2, 2.0]."""
    lo, hi = 0.55, 0.90
    clamped = max(lo, min(hi, p_win))
    t = (clamped - lo) / (hi - lo)
    return round(0.2 + t * (2.0 - 0.2), 3)


class Sizing:
    def __init__(self, config, model: Optional[Any] = None) -> None:
        self.config = config
        self.model = model
        self.rung_prices = compute_rung_prices(
            config.target_bid, config.max_bid, config.ladder_rungs
        )

    def available_budget(self, registry: MarketRegistry) -> float:
        return (
            self.config.max_exposure
            - registry.total_exposure()
            - registry.total_pending()
        )

    def build_order(
        self,
        market: MarketState,
        filter_result: Any,
        budget: float,
    ) -> Optional[PassiveOrder]:
        rung_idx = market.fill_count
        if rung_idx >= len(self.rung_prices):
            return None
        price = self.rung_prices[rung_idx]
        size = self._compute_size(filter_result)
        if size > budget:
            return None
        outcome = (
            market.position.outcome if market.position
            else getattr(filter_result, "outcome", next(iter(market.token_ids)))
        )
        token_id = market.token_ids.get(outcome, "")
        return PassiveOrder(
            order_id=f"_placing_{uuid.uuid4().hex[:8]}",
            condition_id=market.condition_id,
            outcome=outcome,
            token_id=token_id,
            price=price,
            size_usd=size,
            placed_at=time.time(),
        )

    def _compute_size(self, filter_result: Any = None) -> float:
        base = self.config.order_size
        if self.model is not None and filter_result is not None:
            p_win = getattr(filter_result, "p_win", None)
            if p_win is not None:
                return base * model_size_scale(self.model, p_win)
        return base
