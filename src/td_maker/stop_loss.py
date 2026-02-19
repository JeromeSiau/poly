# src/td_maker/stop_loss.py
from __future__ import annotations

import time
from typing import Optional, Any

import structlog

from src.td_maker.state import MarketState, MarketRegistry
from src.td_maker.resilience import clob_retry

logger = structlog.get_logger()


class StopLossManager:

    def __init__(
        self, *, registry: MarketRegistry, order_mgr: Any,
        executor: Any, chainlink_feed: Any, trade_manager: Any,
        shadow: Any, db: Any, config: Any,
        exit_model: Optional[Any] = None,
        poly_feed: Optional[Any] = None,
    ) -> None:
        self.registry = registry
        self.order_mgr = order_mgr
        self.executor = executor
        self.chainlink = chainlink_feed
        self.trade_manager = trade_manager
        self.shadow = shadow
        self.db = db
        self.config = config
        self.exit_model = exit_model
        self.poly_feed = poly_feed
        self._consecutive_failures: dict[str, int] = {}

    async def check_all(self, registry: MarketRegistry) -> None:
        """Check all positions. MUST run even when feed is stale."""
        for market in registry.markets_with_positions():
            await self._check_one(market)

    async def _check_one(self, market: MarketState) -> None:
        cfg = self.config
        if cfg.stoploss_peak <= 0 and self.exit_model is None:
            return

        bid = self._get_current_bid(market)
        if bid is None:
            self._handle_empty_book(market)
            return

        # Track bid max
        if bid > market.bid_max:
            market.bid_max = bid

        if self.exit_model:
            triggered = self._check_ml_exit(market, bid)
        else:
            triggered = self._check_rule_based(market, bid)

        if triggered:
            await self._execute(market)

    def _check_rule_based(self, market: MarketState, current_bid: float) -> bool:
        cfg = self.config
        if market.bid_max < cfg.stoploss_peak:
            return False
        if current_bid > cfg.stoploss_exit:
            market.bid_below_exit_since = None
            return False

        # Fair value override: suppress trigger when Chainlink says price is OK
        fair = self._estimate_fair_value(market)
        if fair is not None and fair > cfg.stoploss_exit + cfg.stoploss_fair_margin:
            now = time.time()
            if market.bid_below_exit_since is None:
                market.bid_below_exit_since = now
            if now - market.bid_below_exit_since < 10.0:
                return False

        return True

    def _check_ml_exit(self, market: MarketState, bid: float) -> bool:
        """ML-based exit decision. Returns True to trigger sell."""
        if self.exit_model is None:
            return False
        try:
            p_exit = self.exit_model.predict_exit(market, bid)
            threshold = getattr(self.config, "exit_threshold", 0.35)
            return p_exit >= threshold
        except Exception as e:
            logger.warning("exit_model_error", cid=market.condition_id,
                           error=str(e))
            return False

    def _handle_empty_book(self, market: MarketState) -> None:
        """Empty book = market resolved/expired. Don't sell, wait for settlement."""
        market.awaiting_settlement = True
        logger.info("stoploss_book_empty_awaiting_settlement",
                    cid=market.condition_id)

    async def _execute(self, market: MarketState) -> None:
        """Sell with 3-level escalation."""
        # Re-check bid before selling
        bid = self._get_current_bid(market)
        if bid is not None and bid > self.config.stoploss_exit:
            logger.info("stoploss_aborted_recovered",
                        cid=market.condition_id, bid=bid)
            return

        pos = market.position
        try:
            await clob_retry(
                lambda: self.executor.cancel_and_sell(
                    token_id=pos.token_id,
                    shares=pos.shares,
                    price=0.01,
                    force_taker=True),
                max_attempts=3,
                base_delay=1.0,
                operation="stop_loss")

            # Success: cleanup
            pnl = (1.0 - pos.entry_price) * pos.shares - pos.size_usd
            await self.trade_manager.record_settle_direct(
                condition_id=market.condition_id,
                outcome=pos.outcome,
                entry_price=pos.entry_price,
                exit_price=0.01,
                size_usd=pos.size_usd,
                pnl=pnl,
                context=f"STOP-LOSS | peak {market.bid_max:.2f} -> {bid or 0:.2f}",
            )
            self.shadow.settle(market.condition_id, won=False)
            market.position = None
            market.bid_max = 0.0
            self._consecutive_failures.pop(market.condition_id, None)
            logger.info("stoploss_executed", cid=market.condition_id, pnl=pnl)

        except Exception as e:
            count = self._consecutive_failures.get(market.condition_id, 0) + 1
            self._consecutive_failures[market.condition_id] = count
            logger.error("stoploss_failed", cid=market.condition_id,
                         attempt=count, error=str(e))
            if count >= 3:
                await self.trade_manager.notify_critical(
                    f"STOP-LOSS FAILED x{count} — INTERVENTION REQUISE\n"
                    f"Market: {market.slug}\n"
                    f"Position: {pos.size_usd:.2f} USD @ {pos.entry_price:.2f}")
            # Never cleanup position on failure — retry next tick

    def _get_current_bid(self, market: MarketState) -> Optional[float]:
        if self.poly_feed is None:
            return None
        if market.position is None:
            return None
        outcome = market.position.outcome
        try:
            bid, _, _, _ = self.poly_feed.get_best_levels(
                market.condition_id, outcome)
            if bid and bid > 0:
                market.last_bids[outcome] = bid
                return bid
        except Exception:
            pass
        return None

    def _estimate_fair_value(self, market: MarketState) -> Optional[float]:
        current = self.chainlink.get_price(market.chainlink_symbol)
        if current is None or market.ref_price <= 0 or market.position is None:
            return None
        from src.utils.fair_value import estimate_fair_value
        slot_remaining = max(0, market.slot_end_ts() - time.time())
        return estimate_fair_value(
            current, market.ref_price, market.position.outcome, slot_remaining)
