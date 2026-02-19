# src/td_maker/engine.py
from __future__ import annotations
import asyncio
import time
from typing import Any
import structlog
from src.td_maker.state import MarketRegistry

logger = structlog.get_logger()


class TDMakerEngine:
    """Orchestrator — zero business logic, wires components and runs loops."""

    def __init__(
        self, *,
        registry: MarketRegistry,
        discovery: Any,
        bidding: Any,
        order_mgr: Any,
        fill_detector: Any,
        stop_loss: Any,
        settlement: Any,
        status: Any,
        guard: Any,
        poly_feed: Any,
        user_feed: Any,
        chainlink_feed: Any,
        config: Any,
    ) -> None:
        self.registry = registry
        self.discovery = discovery
        self.bidding = bidding
        self.order_mgr = order_mgr
        self.fill_detector = fill_detector
        self.stop_loss = stop_loss
        self.settlement = settlement
        self.status = status
        self.guard = guard
        self.poly_feed = poly_feed
        self.user_feed = user_feed
        self.chainlink_feed = chainlink_feed
        self.config = config
        self._last_book_update: float = time.time()
        self._all_blind_since: float = 0.0

    async def run(self) -> None:
        await self._startup()
        await asyncio.gather(
            self._discovery_loop(),
            self._maker_loop(),
            self._fill_listener(),
        )

    async def _startup(self) -> None:
        logger.info("engine_startup")
        await self.settlement.load_db_state(self.registry)
        await self.poly_feed.connect()
        await self.user_feed.connect()
        if self.chainlink_feed:
            await self.chainlink_feed.connect()
            # Wait for initial Chainlink prices before using ref-price filters.
            await asyncio.sleep(2)
        if not getattr(self.config, "paper_mode", True):
            await self.order_mgr.cancel_orphaned_orders()
            await self.fill_detector.reconcile()
        # Clear reconnect flag from initial connect.
        if hasattr(self.user_feed, "reconnected"):
            self.user_feed.reconnected.clear()
        logger.info("engine_ready")

    async def _discovery_loop(self) -> None:
        while True:
            await self.discovery.discover(self.registry)
            await self.settlement.prune_expired(self.registry)
            await asyncio.sleep(self.config.discovery_interval)

    async def _maker_loop(self) -> None:
        while True:
            try:
                await asyncio.wait_for(
                    self.poly_feed.book_updated.wait(),
                    timeout=self.config.maker_interval)
            except asyncio.TimeoutError:
                pass
            self.poly_feed.book_updated.clear()

            # Track last book update for stale detection.
            feed_ts = getattr(self.poly_feed, "last_update_ts", 0.0)
            if feed_ts > self._last_book_update:
                self._last_book_update = feed_ts

            await self.guard.heartbeat()
            try:
                await self._tick()
            except Exception as exc:
                logger.error("maker_tick_error", error=str(exc))

    async def _tick(self) -> None:
        """Core tick. Order is intentional — stop-loss before CB gate."""
        now = time.time()

        # 1. Cleanup stale cancels
        self.order_mgr.expire_stale_cancels()

        # 2. Stop-loss MUST run before CB gate (detect crash even when stale)
        await self.stop_loss.check_all(self.registry)

        # 3. Circuit breaker gate — pass last_book_update for stale detection
        if not await self.guard.is_trading_allowed(
                last_book_update=self._last_book_update):
            # Stale escalation: cancel live orders to avoid adverse fills.
            if self.guard.should_cancel_orders:
                all_orders = [
                    (m, oid)
                    for m in self.registry.active_markets()
                    for oid in list(m.active_orders)
                ]
                if all_orders:
                    await self.order_mgr.cancel_batch(all_orders)
            self.status.print_if_due()
            return

        # 4. Watchdog: if all books empty >30s, force resubscription.
        all_cids = [m.condition_id for m in self.registry.active_markets()]
        if all_cids:
            has_any_price = any(
                self.poly_feed.get_best_prices(cid, out) != (None, None)
                for m in self.registry.active_markets()
                for out in m.token_ids
                for cid in [m.condition_id]
            )
            if not has_any_price:
                if self._all_blind_since == 0.0:
                    self._all_blind_since = now
                elif now - self._all_blind_since > 30.0:
                    logger.warning("watchdog_all_books_empty",
                                   seconds=round(now - self._all_blind_since, 1))
                    self._all_blind_since = now
                    try:
                        await self.poly_feed.flush_subscriptions()
                        logger.info("watchdog_resubscribed")
                    except Exception as exc:
                        logger.warning("watchdog_resubscribe_failed",
                                       error=str(exc)[:80])
            else:
                self._all_blind_since = 0.0

        # 5. Paper fill simulation
        if self.config.paper_mode:
            await self.fill_detector.check_paper_fills(self.registry)

        # 6. Periodic CLOB reconciliation (live only, triggered on reconnect too)
        if not self.config.paper_mode:
            if (hasattr(self.user_feed, "reconnected")
                    and self.user_feed.reconnected.is_set()):
                self.user_feed.reconnected.clear()
                await self.fill_detector.reconcile()
            else:
                await self.fill_detector.periodic_reconcile()

        # 7. New orders
        await self.bidding.scan_and_place(self.registry)

        # 8. Status line
        self.status.print_if_due()

    async def _fill_listener(self) -> None:
        await self.fill_detector.listen(self.registry)
