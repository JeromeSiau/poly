# src/td_maker/bidding.py
from __future__ import annotations
from typing import Any
import structlog
from src.td_maker.state import MarketState, MarketRegistry, PassiveOrder

logger = structlog.get_logger()


class BiddingEngine:

    def __init__(
        self, *, registry: MarketRegistry, filters: Any, order_mgr: Any,
        sizing: Any, config: Any, poly_feed: Any,
    ) -> None:
        self.registry = registry
        self.filters = filters
        self.order_mgr = order_mgr
        self.sizing = sizing
        self.config = config
        self.poly_feed = poly_feed

    async def scan_and_place(self, registry: MarketRegistry) -> None:
        budget = self.sizing.available_budget(registry)

        for market in registry.active_markets():
            if market.awaiting_settlement:
                continue
            if market.fill_count >= self.config.ladder_rungs:
                continue

            for outcome in market.token_ids:
                await self._check_outcome(market, outcome, budget)

    async def _check_outcome(
        self, market: MarketState, outcome: str, budget: float
    ) -> None:
        try:
            bid, bid_sz, ask, ask_sz = self.poly_feed.get_best_levels(
                market.condition_id, outcome)
        except Exception:
            return

        if bid is None:
            return

        # Track last known bid for settlement fallback
        market.last_bids[outcome] = bid

        result = self.filters.should_bid(market, outcome=outcome,
                                          bid=bid, ask=ask)
        if result.is_skip:
            return

        # Guard: skip if already have an active order for this outcome
        if any(o.outcome == outcome
               for o in market.active_orders.values()
               if not MarketState.is_placeholder(o.order_id)):
            return

        # Rung dedup based on the actual rung price (not filter result / bid price)
        rung_idx = market.fill_count
        if rung_idx >= len(self.sizing.rung_prices):
            return
        rung_price_cents = int(round(self.sizing.rung_prices[rung_idx] * 100))
        rung_key = (outcome, rung_price_cents)
        if rung_key in market.rungs_placed:
            return

        order = self.sizing.build_order(market, result, budget)
        if order is None:
            return

        market.rungs_placed.add(rung_key)
        placed_id = await self.order_mgr.place_order(market, order)
        if placed_id is None:
            market.rungs_placed.discard(rung_key)
