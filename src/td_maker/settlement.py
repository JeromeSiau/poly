# src/td_maker/settlement.py
from __future__ import annotations

import time
from typing import Optional, Any

import httpx
import structlog

from src.td_maker.state import MarketState, MarketRegistry
from src.td_maker.resilience import clob_retry

logger = structlog.get_logger()

_SETTLE_DEFER_MAX = 3600  # 1 hour


class SettlementManager:

    def __init__(
        self, *, registry: MarketRegistry, executor: Any,
        trade_manager: Any, shadow: Any, guard: Any,
        db: Any, config: Any, order_mgr: Any,
    ) -> None:
        self.registry = registry
        self.executor = executor
        self.trade_manager = trade_manager
        self.shadow = shadow
        self.guard = guard
        self.db = db
        self.config = config
        self.order_mgr = order_mgr

    # ── DB Startup Recovery ─────────────────────────────────────────

    async def load_db_state(self, registry: MarketRegistry) -> None:
        """Restore pending/filled orders and positions from DB on startup."""
        from src.db.td_orders import load_orders
        rows = await load_orders(
            db_url=self.config.db_url,
            platform="polymarket",
            strategy_tag=self.config.strategy_tag)
        now = time.time()
        restored_orders = 0
        restored_positions = 0
        for row in rows:
            if row.placed_at and (now - row.placed_at) > 1800:
                from src.db.td_orders import delete_order
                await delete_order(db_url=self.config.db_url,
                                   order_id=row.order_id)
                continue
            market = registry.get(row.condition_id)
            if not market:
                slot_duration = getattr(self.config, "slot_duration", 15 * 60)
                market = MarketState.orphan(row, slot_duration=slot_duration)
                registry.register(market)
            if row.status == "pending":
                from src.td_maker.state import PassiveOrder
                o = PassiveOrder(
                    order_id=row.order_id,
                    condition_id=row.condition_id,
                    outcome=row.outcome,
                    token_id=row.token_id,
                    price=row.price,
                    size_usd=row.size_usd,
                    placed_at=row.placed_at or now,
                )
                market.add_order(o)
                restored_orders += 1
            elif row.status == "filled" and market.position is None:
                from src.td_maker.state import OpenPosition
                market.position = OpenPosition(
                    condition_id=row.condition_id,
                    outcome=row.outcome,
                    token_id=row.token_id,
                    entry_price=row.price,
                    size_usd=row.size_usd,
                    shares=row.shares or round(row.size_usd / row.price, 4),
                    filled_at=row.filled_at or now,
                )
                restored_positions += 1
        logger.info("db_state_loaded", orders=restored_orders,
                    positions=restored_positions)

    # ── Pruning ─────────────────────────────────────────────────────

    async def prune_expired(self, registry: MarketRegistry) -> None:
        """Settle and clean up markets past slot_end + grace."""
        for market in registry.expired_markets(time.time()):
            for oid in list(market.active_orders):
                await self.order_mgr.cancel_order(market, oid)
            if market.position:
                await self._settle(market)
            elif not market.awaiting_settlement:
                registry.remove(market.condition_id)

    # ── Settlement ──────────────────────────────────────────────────

    async def _settle(self, market: MarketState) -> None:
        resolution = await self._query_resolution(market)

        if resolution is None:
            if market.settlement_deferred_until is None:
                market.settlement_deferred_until = time.time() + _SETTLE_DEFER_MAX
                market.awaiting_settlement = True
                logger.info("settlement_deferred", cid=market.condition_id)
                return
            if time.time() < market.settlement_deferred_until:
                return
            resolution = self._force_from_last_bid(market)
            logger.warning("settlement_forced", cid=market.condition_id,
                           resolution=resolution)

        pos = market.position
        if pos is None:
            self.registry.remove(market.condition_id)
            return

        pnl = (pos.shares - pos.size_usd) if resolution == "win" else -pos.size_usd
        exit_price = 1.0 if resolution == "win" else 0.0

        if not self.config.paper_mode:
            from src.execution.models import TradeIntent, FillResult as TradeFillResult
            intent = TradeIntent(
                condition_id=market.condition_id,
                token_id=pos.token_id,
                outcome=pos.outcome,
                side="BUY",
                price=pos.entry_price,
                size_usd=pos.size_usd,
                reason="settlement",
                title=market.slug,
            )
            trade_fill = TradeFillResult(
                filled=True, shares=pos.shares,
                avg_price=exit_price, pnl_delta=pnl)
            await self.trade_manager.record_settle_direct(intent, trade_fill)

        await self.guard.record_result(pnl=pnl, won=(resolution == "win"))
        self.shadow.settle(market.condition_id, won=(resolution == "win"))
        self.db.fire(self.db.mark_settled(pos, pnl))

        logger.info("settled", cid=market.condition_id, resolution=resolution,
                    pnl=round(pnl, 4))
        self.registry.remove(market.condition_id)

    async def _query_resolution(self, market: MarketState) -> Optional[str]:
        """Try Gamma API, then CLOB API. Returns 'win'|'loss'|None."""
        # 1. Gamma API via slug
        try:
            res = await clob_retry(
                lambda: self._query_gamma(market.slug),
                max_attempts=2, base_delay=0.5,
                operation="resolve_gamma")
            if res:
                return res
        except Exception:
            pass

        # 2. CLOB API via condition_id
        try:
            res = await clob_retry(
                lambda: self._query_clob(market.condition_id),
                max_attempts=2, base_delay=0.5,
                operation="resolve_clob")
            if res:
                return res
        except Exception:
            pass

        return None

    async def _query_gamma(self, slug: str) -> Optional[str]:
        url = f"https://gamma-api.polymarket.com/events?slug={slug}"
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url)
            r.raise_for_status()
            data = r.json()
        if not data:
            return None
        event = data[0] if isinstance(data, list) else data
        markets = event.get("markets", [])
        for m in markets:
            if m.get("closed") and m.get("outcomePrices"):
                prices = m["outcomePrices"]
                if isinstance(prices, str):
                    import json
                    prices = json.loads(prices)
                if prices and float(prices[0]) >= 0.99:
                    return "win"
                elif prices and float(prices[1]) >= 0.99:
                    return "loss"
        return None

    async def _query_clob(self, condition_id: str) -> Optional[str]:
        url = f"https://clob.polymarket.com/markets/{condition_id}"
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url)
            r.raise_for_status()
            data = r.json()
        tokens = data.get("tokens", [])
        for token in tokens:
            price = float(token.get("price", 0))
            if price >= 0.99:
                outcome = token.get("outcome", "").lower()
                return "win" if "yes" in outcome or "up" in outcome else "loss"
        return None

    def _force_from_last_bid(self, market: MarketState) -> str:
        if market.position is None:
            return "loss"
        outcome = market.position.outcome
        last_bid = market.last_bids.get(outcome, 0.5)
        if last_bid >= 0.9:
            return "win"
        elif last_bid <= 0.1:
            return "loss"
        else:
            logger.error("settlement_ambiguous_bid", bid=last_bid,
                         cid=market.condition_id)
            return "loss"  # conservative
