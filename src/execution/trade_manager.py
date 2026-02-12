"""Generic trade execution manager for all strategies.

Orchestrates: order placement (paper/live), paper fill detection,
DB persistence, and Telegram notifications.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Optional

import structlog

from config.settings import settings
from src.execution.executor import ExecutorProtocol, adapt_polymarket_response
from src.execution.models import (
    FillResult,
    OrderResult,
    PendingOrder,
    TradeIntent,
)
from src.execution.trade_recorder import TradeRecorder
from src.paper_trading.alerts import TelegramAlerter

logger = structlog.get_logger()

# Emoji prefixes
_PAPER = "\U0001f4dd"  # 📝
_LIVE = "\U0001f525"   # 🔥
_BID = "\U0001f4ca"    # 📊
_FILL = "\u2705"       # ✅
_WIN = "\U0001f7e2"    # 🟢
_LOSS = "\U0001f534"   # 🔴


class TradeManager:
    """Generic trade manager for all strategies."""

    def __init__(
        self,
        *,
        executor: Optional[ExecutorProtocol] = None,
        strategy: str,
        paper: bool = True,
        db_url: str = "",
        event_type: str = "",
        run_id: str = "",
        notify_bids: bool = True,
        notify_fills: bool = True,
        notify_closes: bool = True,
    ) -> None:
        self.executor = executor
        self.strategy = strategy
        self.paper = paper
        self.notify_bids = notify_bids
        self.notify_fills = notify_fills
        self.notify_closes = notify_closes

        # Internal state
        self._pending: dict[str, PendingOrder] = {}
        self._positions: dict[tuple[str, str], PendingOrder] = {}  # (cid, outcome) -> filled order
        self._paper_counter: int = 0
        self._wins: int = 0
        self._losses: int = 0
        self._total_pnl: float = 0.0

        # Recorder (public — strategies may need direct access for extra_state)
        self.recorder: Optional[TradeRecorder] = None
        if db_url or event_type:
            self.recorder = TradeRecorder(
                db_url=db_url or settings.DATABASE_URL,
                strategy_tag=strategy,
                event_type=event_type,
                run_id=run_id,
            )
            self.recorder.bootstrap()

        # Telegram
        self._alerter = TelegramAlerter()

    # --- place ---

    async def place(self, intent: TradeIntent) -> PendingOrder:
        """Place an order. Paper: fake ID. Live: via executor."""
        if self.paper or self.executor is None:
            order_id = self._next_paper_id()
        else:
            raw = await self.executor.place_order(
                token_id=intent.token_id,
                side=intent.side,
                size=intent.size_usd,
                price=intent.price,
                outcome=intent.outcome,
            )
            result = self._adapt_result(raw)
            if result.status == "error":
                logger.warning("order_failed", strategy=self.strategy, error=result.error)
                return PendingOrder(order_id="", intent=intent, placed_at=time.time())
            order_id = result.order_id

        now = time.time()
        pending = PendingOrder(order_id=order_id, intent=intent, placed_at=now)
        self._pending[order_id] = pending

        # Telegram
        if self.notify_bids:
            await self.notify_bid(intent)

        logger.info(
            "order_placed",
            strategy=self.strategy,
            outcome=intent.outcome,
            price=intent.price,
            size=intent.size_usd,
            paper=self.paper,
            order_id=order_id,
        )
        return pending

    # --- check_paper_fills ---

    def check_paper_fills(
        self,
        get_levels: Callable[[str, str], tuple[Any, ...]],
    ) -> list[FillResult]:
        """Check pending paper orders against orderbook levels.

        get_levels(condition_id, outcome) -> (bid, bid_sz, ask, ask_sz)
        BUY fills when ask <= order price.
        SELL fills when bid >= order price.
        """
        fills: list[FillResult] = []
        filled_ids: list[str] = []

        for oid, pending in self._pending.items():
            intent = pending.intent
            bid, _, ask, _ = get_levels(intent.condition_id, intent.outcome)

            filled = False
            if intent.side == "BUY" and ask is not None:
                filled = ask <= intent.price
            elif intent.side == "SELL" and bid is not None:
                filled = bid >= intent.price

            if filled:
                fill = FillResult(
                    filled=True,
                    shares=intent.shares,
                    avg_price=intent.price,
                )
                fills.append(fill)
                filled_ids.append(oid)
                self._positions[(intent.condition_id, intent.outcome)] = pending

                # Persist
                if self.recorder:
                    try:
                        self.recorder.record_fill(
                            intent=intent, fill=fill,
                            fair_prices={intent.outcome: intent.price},
                            execution_mode="paper_fill",
                        )
                    except Exception as exc:
                        logger.warning("record_fill_failed", error=str(exc))

                # Telegram (fire-and-forget)
                if self.notify_fills:
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(self.notify_fill(intent, fill))
                    except RuntimeError:
                        pass

                logger.info(
                    "paper_fill",
                    strategy=self.strategy,
                    outcome=intent.outcome,
                    price=intent.price,
                    shares=round(fill.shares, 2),
                )

        for oid in filled_ids:
            del self._pending[oid]

        return fills

    # --- settle ---

    async def settle(
        self,
        condition_id: str,
        outcome: str,
        settlement_price: float,
        won: bool,
        extra_state: dict[str, Any] | None = None,
    ) -> float:
        """Settle a position. Returns PnL."""
        pos = self._positions.pop((condition_id, outcome), None)
        if pos is None:
            return 0.0

        intent = pos.intent
        pnl = intent.shares * (settlement_price - intent.price)
        if won:
            self._wins += 1
        else:
            self._losses += 1
        self._total_pnl += pnl

        # Persist settlement
        if self.recorder:
            settle_intent = TradeIntent(
                condition_id=condition_id,
                token_id=intent.token_id,
                outcome=outcome,
                side="SELL",
                price=settlement_price,
                size_usd=intent.shares * settlement_price,
                reason="settlement",
                title=intent.title,
                edge_pct=0.0,
                timestamp=time.time(),
            )
            settle_fill = FillResult(
                filled=True,
                shares=intent.shares,
                avg_price=settlement_price,
                pnl_delta=pnl,
            )
            try:
                self.recorder.record_settle(
                    intent=settle_intent, fill=settle_fill,
                    fair_prices={outcome: settlement_price},
                    extra_state=extra_state,
                )
            except Exception as exc:
                logger.warning("record_settle_failed", error=str(exc))

        # Telegram
        if self.notify_closes:
            await self.notify_settle(intent, settlement_price, pnl, won)

        logger.info(
            "position_settled",
            strategy=self.strategy,
            outcome=outcome,
            entry=intent.price,
            exit=settlement_price,
            won=won,
            pnl=round(pnl, 4),
            record=f"{self._wins}W-{self._losses}L",
        )
        return pnl

    # --- cancel ---

    async def cancel(self, order_id: str) -> bool:
        """Cancel a pending order."""
        if order_id not in self._pending:
            return False
        del self._pending[order_id]
        if not self.paper and self.executor:
            try:
                await self.executor.cancel_order(order_id)
            except Exception:
                pass
        return True

    # --- queries ---

    def get_pending_orders(self) -> dict[str, PendingOrder]:
        return dict(self._pending)

    def get_stats(self) -> dict[str, Any]:
        return {
            "wins": self._wins,
            "losses": self._losses,
            "total_pnl": self._total_pnl,
            "pending_orders": len(self._pending),
            "open_positions": len(self._positions),
        }

    async def close(self) -> None:
        await self._alerter.close()

    # --- public notifications (strategies can call directly) ---

    async def notify_bid(self, intent: TradeIntent) -> None:
        """Send Telegram BID notification."""
        mode = self._mode_emoji()
        msg = (
            f"{mode}{_BID} {self.strategy}\n"
            f"BID {intent.outcome} @ {intent.price:.2f} | ${intent.size_usd:.0f}\n"
            f"{intent.title}"
        )
        try:
            await self._alerter.send_custom_alert(msg)
        except Exception:
            pass

    async def notify_fill(self, intent: TradeIntent, fill: FillResult) -> None:
        """Send Telegram FILL notification."""
        mode = self._mode_emoji()
        msg = (
            f"{mode}{_FILL} {self.strategy}\n"
            f"FILL {intent.outcome} @ {fill.avg_price:.2f} | {fill.shares:.1f} shares\n"
            f"{intent.title}"
        )
        try:
            await self._alerter.send_custom_alert(msg)
        except Exception:
            pass

    async def notify_settle(
        self, intent: TradeIntent, exit_price: float, pnl: float, won: bool,
    ) -> None:
        """Send Telegram WIN/LOSS notification."""
        mode = self._mode_emoji()
        result_emoji = _WIN if won else _LOSS
        result_text = "WIN" if won else "LOSS"
        record = f"{self._wins}W-{self._losses}L"
        msg = (
            f"{mode}{result_emoji} {self.strategy}\n"
            f"{result_text} {intent.outcome} {intent.price:.2f} \u2192 {exit_price:.2f} | ${pnl:+.2f}\n"
            f"{record} | Total: ${self._total_pnl:+.2f}"
        )
        try:
            await self._alerter.send_custom_alert(msg)
        except Exception:
            pass

    # --- internal ---

    def _next_paper_id(self) -> str:
        self._paper_counter += 1
        return f"paper_{self._paper_counter}"

    def _adapt_result(self, raw: Any) -> OrderResult:
        if isinstance(raw, OrderResult):
            return raw
        if isinstance(raw, dict):
            return adapt_polymarket_response(raw)
        return OrderResult(order_id="", status="error", error="unexpected response")

    def _mode_emoji(self) -> str:
        return _PAPER if self.paper else _LIVE
