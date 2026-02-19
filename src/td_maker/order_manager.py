# src/td_maker/order_manager.py
from __future__ import annotations

import asyncio
import time
from typing import Optional, Any

import structlog

from src.td_maker.state import MarketState, MarketRegistry, PassiveOrder
from src.td_maker.resilience import clob_retry

logger = structlog.get_logger()

_PLACEHOLDER_TIMEOUT = 15.0
_PENDING_CANCEL_TTL = 30.0


class OrderManager:

    def __init__(
        self, *, executor: Any, registry: MarketRegistry,
        db: Any, trade_manager: Any, config: Any,
    ) -> None:
        self.executor = executor
        self.registry = registry
        self.db = db
        self.trade_manager = trade_manager
        self.config = config

    async def place_order(
        self, market: MarketState, order: PassiveOrder
    ) -> Optional[str]:
        """Place order with placeholder safety and 15s timeout."""
        # Pre-register placeholder
        market.add_order(order)
        placeholder_id = order.order_id

        try:
            real_id = await asyncio.wait_for(
                clob_retry(
                    lambda: self.executor.place_order(
                        token_id=order.token_id,
                        side="BUY",
                        size=order.size_usd,
                        price=order.price,
                        outcome=order.outcome,
                    ),
                    operation="place_order"),
                timeout=_PLACEHOLDER_TIMEOUT)
        except (asyncio.TimeoutError, Exception) as e:
            market.active_orders.pop(placeholder_id, None)
            await self._check_ghost_order(market, order)
            logger.error("place_order_failed",
                         cid=market.condition_id, error=str(e))
            return None

        market.replace_order_id(placeholder_id, real_id)
        self.db.fire(self.db.save_order(order_id=real_id, market=market,
                                        order=order))
        logger.info("order_placed", cid=market.condition_id,
                    outcome=order.outcome, price=order.price,
                    size_usd=order.size_usd, order_id=real_id)
        return real_id

    async def cancel_order(self, market: MarketState, order_id: str) -> None:
        """Cancel order — moves to pending_cancels, calls CLOB."""
        order = market.move_to_pending_cancel(order_id)
        if order is None:
            return
        try:
            await clob_retry(
                lambda: self.executor.cancel_order(order_id),
                operation="cancel_order")
        except Exception as e:
            logger.warning("cancel_order_failed", oid=order_id, error=str(e))

    async def cancel_other_side(
        self, market: MarketState, *, filled_outcome: str
    ) -> None:
        """Cancel all orders on the opposite outcome."""
        to_cancel = [
            oid for oid, o in list(market.active_orders.items())
            if o.outcome != filled_outcome
        ]
        for oid in to_cancel:
            await self.cancel_order(market, oid)

    async def cancel_batch(self, pairs: list[tuple[MarketState, str]]) -> None:
        for market, oid in pairs:
            await self.cancel_order(market, oid)

    async def place_batch(
        self, pairs: list[tuple[MarketState, PassiveOrder]]
    ) -> None:
        for market, order in pairs:
            await self.place_order(market, order)

    def expire_stale_cancels(self) -> None:
        """Clean pending_cancels older than 30s — schedule CLOB check."""
        now = time.time()
        for market in self.registry.active_markets():
            stale = [
                oid for oid, o in list(market.pending_cancels.items())
                if o.cancelled_at and (now - o.cancelled_at) > _PENDING_CANCEL_TTL
            ]
            for oid in stale:
                market.pending_cancels.pop(oid, None)
                self.db.fire(self.db.delete_order(oid))

    async def cancel_orphaned_orders(self) -> None:
        """Startup: cancel every open CLOB order from previous runs."""
        try:
            await clob_retry(
                lambda: self.executor.cancel_all_orders(),
                operation="cancel_orphans")
            logger.info("orphaned_orders_cancelled")
        except Exception as e:
            logger.error("cancel_orphans_failed", error=str(e))

    async def _check_ghost_order(
        self, market: MarketState, order: PassiveOrder
    ) -> None:
        """After placement timeout: verify order doesn't exist on CLOB."""
        try:
            results = await self.executor.get_open_orders(
                market=market.condition_id)
            for o in (results or []):
                o_price = float(o.get("price", 0) if isinstance(o, dict)
                                else getattr(o, "price", 0))
                o_token = (o.get("asset_id", "") if isinstance(o, dict)
                           else getattr(o, "asset_id", ""))
                if abs(o_price - order.price) < 0.005 and o_token == order.token_id:
                    real_id = (o.get("id") if isinstance(o, dict)
                               else getattr(o, "id", None))
                    if real_id:
                        logger.warning("ghost_order_detected",
                                       cid=market.condition_id, real_id=real_id)
                        order.order_id = real_id
                        market.add_order(order)
                        return
        except Exception as e:
            logger.warning("ghost_check_failed",
                           cid=market.condition_id, error=str(e))
