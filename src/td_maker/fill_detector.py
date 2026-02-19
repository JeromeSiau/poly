# src/td_maker/fill_detector.py
from __future__ import annotations

import asyncio
import time
from typing import Optional, Any

import structlog

from src.td_maker.state import MarketState, MarketRegistry, PassiveOrder
from src.td_maker.resilience import clob_retry

logger = structlog.get_logger()

_FILL_DEDUP_TTL = 1800  # 30 minutes


class FillDetector:

    def __init__(
        self, *, registry: MarketRegistry, order_mgr: Any,
        poly_feed: Any, user_feed: Any, trade_manager: Any,
        shadow: Any, db: Any, config: Any, executor: Any,
    ) -> None:
        self.registry = registry
        self.order_mgr = order_mgr
        self.poly_feed = poly_feed
        self.user_feed = user_feed
        self.trade_manager = trade_manager
        self.shadow = shadow
        self.db = db
        self.config = config
        self.executor = executor
        self._processed_fills: dict[str, float] = {}
        self._last_reconcile: float = 0.0

    def _fill_key(self, cid: str, order_id: str, trade_id: str) -> str:
        return f"{cid}:{order_id}:{trade_id}"

    def _is_duplicate(self, key: str) -> bool:
        return key in self._processed_fills

    def _purge_old_fills(self) -> None:
        cutoff = time.time() - _FILL_DEDUP_TTL
        self._processed_fills = {
            k: v for k, v in self._processed_fills.items() if v > cutoff
        }

    # ── WSS Real-time (live) ────────────────────────────────────────

    async def listen(self, registry: MarketRegistry) -> None:
        """Infinite loop draining User WS fills."""
        while True:
            try:
                evt = await self.user_feed.fills.get()
            except asyncio.CancelledError:
                raise
            except Exception:
                await asyncio.sleep(0.1)
                continue
            # evt is a UserTradeEvent: .market (cid), .size (shares), .maker_order_id
            cid = getattr(evt, "market", None)
            if not cid:
                continue
            market = registry.get(cid)
            if not market:
                continue
            key = self._fill_key(
                cid,
                getattr(evt, "order_id", ""),
                getattr(evt, "asset_id", ""))
            if self._is_duplicate(key):
                continue
            order = self._match_order(market, evt)
            if order:
                shares = float(getattr(evt, "size", 0))
                await self._process_fill(market, order, shares=shares)
                self._processed_fills[key] = time.time()

    def _match_order(self, market: MarketState, msg: Any) -> Optional[PassiveOrder]:
        """4-priority matching: exact → placeholder → pending-cancel → broad."""
        maker_oid = getattr(msg, "maker_order_id", None)

        # 1. Exact maker_order_id
        if maker_oid and maker_oid in market.active_orders:
            return market.active_orders[maker_oid]

        # 2. Placeholder by cid
        for oid, o in market.active_orders.items():
            if MarketState.is_placeholder(oid):
                return o

        # 3. Pending cancel
        if maker_oid and maker_oid in market.pending_cancels:
            return market.pending_cancels[maker_oid]

        # 4. Broad cid match (only if no maker_order_id)
        if not maker_oid:
            if market.active_orders:
                return next(iter(market.active_orders.values()))

        return None

    # ── Paper fills ──────────────────────────────────────────────────

    async def check_paper_fills(self, registry: MarketRegistry) -> None:
        for market in registry.markets_with_orders():
            for order in list(market.active_orders.values()):
                if MarketState.is_placeholder(order.order_id):
                    continue
                if self._paper_fill_triggered(market, order):
                    shares = round(order.size_usd / order.price, 4)
                    key = self._fill_key(market.condition_id, order.order_id,
                                        "paper")
                    if not self._is_duplicate(key):
                        await self._process_fill(market, order, shares=shares,
                                                  paper=True)
                        self._processed_fills[key] = time.time()

    def _paper_fill_triggered(self, market: MarketState,
                               order: PassiveOrder) -> bool:
        try:
            bid, bid_sz, ask, ask_sz = self.poly_feed.get_best_levels(
                market.condition_id, order.outcome)
        except Exception:
            return False
        if ask is None:
            return False
        # Condition 1: ask crossed down to our bid
        if ask <= order.price:
            return True
        # Condition 2: bid dropped below our price
        if bid is not None and bid < order.price:
            return True
        return False

    # ── CLOB Reconciliation ──────────────────────────────────────────

    async def reconcile(self) -> None:
        """Poll CLOB for all active + pending-cancel orders."""
        for market in self.registry.markets_with_orders():
            for oid, order in list(market.active_orders.items()):
                if MarketState.is_placeholder(oid):
                    continue
                await self._check_clob_order(market, oid, order)
            for oid, order in list(market.pending_cancels.items()):
                await self._check_clob_order(market, oid, order)

    async def periodic_reconcile(self) -> None:
        if time.time() - self._last_reconcile < 60:
            return
        self._last_reconcile = time.time()
        self._purge_old_fills()
        await self.reconcile()

    async def _check_clob_order(self, market: MarketState,
                                 oid: str, order: PassiveOrder) -> None:
        try:
            result = await clob_retry(
                lambda: self.executor.get_order(oid),
                operation="reconcile_order")
            status = getattr(result, "status", result) if result else None
            if status in ("MATCHED", "FILLED"):
                shares = getattr(result, "size_matched", None) or \
                         round(order.size_usd / order.price, 4)
                key = self._fill_key(market.condition_id, oid, "recon")
                if not self._is_duplicate(key):
                    await self._process_fill(market, order, shares=shares)
                    self._processed_fills[key] = time.time()
        except Exception as e:
            logger.warning("reconcile_failed", oid=oid, error=str(e))

    # ── Shared fill processing ───────────────────────────────────────

    async def _process_fill(
        self, market: MarketState, order: PassiveOrder,
        shares: float, paper: bool = False,
    ) -> None:
        """Central fill handler — shared by WS, paper, reconciliation."""
        ok = market.record_fill(order.order_id, shares)
        if not ok:
            logger.warning("fill_rejected_state_inconsistency",
                           cid=market.condition_id, oid=order.order_id)
            return

        # Cancel other side on first fill
        if market.fill_count == 1:
            await self.order_mgr.cancel_other_side(market, filled_outcome=order.outcome)

        # Shadow taker entry
        try:
            bid, _, ask, _ = self.poly_feed.get_best_levels(
                market.condition_id, order.outcome)
            if ask:
                self.shadow.record(
                    market.condition_id, order.outcome, ask, order.size_usd)
        except Exception:
            pass

        # Telegram + DB
        from src.execution.models import TradeIntent, FillResult as TradeFillResult
        intent = TradeIntent(
            condition_id=market.condition_id,
            token_id=order.token_id,
            outcome=order.outcome,
            side="BUY",
            price=order.price,
            size_usd=order.size_usd,
            reason="td_maker_fill",
            title=market.slug,
        )
        trade_fill = TradeFillResult(
            filled=True, shares=shares, avg_price=order.price)
        await self.trade_manager.record_fill_direct(
            intent, trade_fill,
            execution_mode="paper" if paper else "maker")
        self.db.fire(self.db.mark_filled(order.order_id, shares=shares))

        logger.info("fill_processed", cid=market.condition_id,
                    outcome=order.outcome, price=order.price,
                    shares=shares, paper=paper)
