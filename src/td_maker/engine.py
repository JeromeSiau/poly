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
        await self.chainlink_feed.connect()
        await self.order_mgr.cancel_orphaned_orders()
        await self.fill_detector.reconcile()
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
            self.guard.heartbeat()
            await self._tick()

    async def _tick(self) -> None:
        """Core tick. Order is intentional — stop-loss before CB gate."""
        # 1. Cleanup stale cancels
        self.order_mgr.expire_stale_cancels()

        # 2. Stop-loss MUST run before CB gate (detect crash even when stale)
        await self.stop_loss.check_all(self.registry)

        # 3. Circuit breaker gate
        if not self.guard.is_trading_allowed():
            self.status.print_if_due()
            return

        # 4. Paper fill simulation
        if self.config.paper_mode:
            await self.fill_detector.check_paper_fills(self.registry)

        # 5. Periodic CLOB reconciliation
        await self.fill_detector.periodic_reconcile()

        # 6. New orders
        await self.bidding.scan_and_place(self.registry)

        # 7. Status line
        self.status.print_if_due()

    async def _fill_listener(self) -> None:
        await self.fill_detector.listen(self.registry)
