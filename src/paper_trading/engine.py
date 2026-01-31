# src/paper_trading/engine.py
"""Paper trading engine orchestrator."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import structlog

from src.feeds.base import FeedEvent
from src.paper_trading.execution_sim import ExecutionSimulator
from src.paper_trading.market_observer import MarketObserver, PriceCapture
from src.paper_trading.position_manager import PositionManager

logger = structlog.get_logger()


@dataclass
class PaperTradingEngine:
    """Main orchestrator for paper trading.

    Takes events from data feeds, gets ML predictions,
    captures market prices, and creates paper trades when edge is sufficient.
    """

    model: Any  # ImpactModel or similar with predict_single method
    capital: float = 10000.0
    min_edge: float = 0.05

    # Created in __post_init__
    position_manager: PositionManager = field(init=False)
    execution_sim: ExecutionSimulator = field(init=False)
    market_observer: MarketObserver = field(init=False)

    # Follow-up intervals in seconds
    followup_intervals: list[float] = field(
        default_factory=lambda: [30.0, 60.0, 120.0]
    )

    # Track pending followup tasks
    _pending_followups: list[asyncio.Task] = field(default_factory=list, init=False)

    def __post_init__(self):
        """Initialize dependent components."""
        self.position_manager = PositionManager(
            capital=self.capital,
            min_edge=self.min_edge,
        )
        self.execution_sim = ExecutionSimulator()
        self.market_observer = MarketObserver()

    async def process_event(
        self,
        event: FeedEvent,
        market_id: str,
        outcome: str = "YES",
    ) -> dict[str, Any]:
        """Process an event and potentially create a paper trade.

        Args:
            event: The feed event to process
            market_id: Polymarket market ID to trade on
            outcome: Market outcome to trade ("YES" or "NO")

        Returns:
            Dict with model_prediction, market_price, edge, and trade info
        """
        # Extract features from event data
        features = self._extract_features(event)

        # Get model prediction
        model_prediction = self.model.predict_single(features)

        # Capture current market price
        price_capture = await self.market_observer.capture_price(market_id, outcome)
        market_price = price_capture.price if price_capture.price is not None else 0.50

        # Calculate edge
        edge = model_prediction - market_price

        # Prepare result
        result: dict[str, Any] = {
            "event": event,
            "market_id": market_id,
            "outcome": outcome,
            "model_prediction": model_prediction,
            "market_price": market_price,
            "edge": edge,
            "trade": None,
            "timestamp": datetime.now(timezone.utc),
        }

        # Create trade if edge is sufficient
        if edge >= self.min_edge:
            trade = self._create_trade(
                model_prediction=model_prediction,
                market_price=market_price,
                market_id=market_id,
                outcome=outcome,
                event=event,
            )
            result["trade"] = trade

            logger.info(
                "paper_trade_created",
                market_id=market_id,
                prediction=model_prediction,
                market_price=market_price,
                edge=edge,
                size=trade["size"],
            )

        # Schedule follow-up captures
        await self._schedule_followups(market_id, outcome, result)

        return result

    def _extract_features(self, event: FeedEvent) -> dict[str, Any]:
        """Extract features from event for model prediction.

        Args:
            event: The feed event

        Returns:
            Feature dict for the model
        """
        # Use event data directly as features
        # The model expects specific feature names
        features = dict(event.data)

        # Add event metadata that might be useful
        if "game_time_minutes" not in features and "game_time" in features:
            features["game_time_minutes"] = features.get("game_time", 0)

        return features

    def _create_trade(
        self,
        model_prediction: float,
        market_price: float,
        market_id: str,
        outcome: str,
        event: FeedEvent,
    ) -> dict[str, Any]:
        """Create a paper trade.

        Args:
            model_prediction: Our predicted probability
            market_price: Current market price
            market_id: Market to trade
            outcome: Outcome to trade
            event: Source event

        Returns:
            Trade details dict
        """
        # Calculate position size using PositionManager
        size = self.position_manager.calculate_position_size(
            our_prob=model_prediction,
            market_price=market_price,
        )

        # Simulate execution with slippage
        fill_price = self.execution_sim.simulate_fill(
            target_price=market_price,
            size=size,
            side="BUY",
        )

        return {
            "market_id": market_id,
            "outcome": outcome,
            "side": "BUY",
            "size": size,
            "target_price": market_price,
            "fill_price": fill_price,
            "model_prediction": model_prediction,
            "edge": model_prediction - market_price,
            "event_type": event.event_type,
            "match_id": event.match_id,
            "timestamp": datetime.now(timezone.utc),
        }

    async def _schedule_followups(
        self,
        market_id: str,
        outcome: str,
        initial_result: dict[str, Any],
    ) -> None:
        """Schedule follow-up price captures.

        Captures prices at T+30s, T+60s, T+120s to track market reaction.

        Args:
            market_id: Market to observe
            outcome: Outcome to observe
            initial_result: Initial processing result
        """
        task = asyncio.create_task(
            self._run_followups(market_id, outcome, initial_result)
        )
        self._pending_followups.append(task)

        # Clean up completed tasks
        self._pending_followups = [t for t in self._pending_followups if not t.done()]

    async def _run_followups(
        self,
        market_id: str,
        outcome: str,
        initial_result: dict[str, Any],
    ) -> list[PriceCapture]:
        """Run the follow-up price captures.

        Args:
            market_id: Market to observe
            outcome: Outcome to observe
            initial_result: Initial processing result

        Returns:
            List of price captures
        """
        captures = []
        initial_price = initial_result["market_price"]

        for interval in self.followup_intervals:
            await asyncio.sleep(interval)

            capture = await self.market_observer.capture_price(market_id, outcome)
            captures.append(capture)

            if capture.price is not None:
                price_change = capture.price - initial_price
                logger.info(
                    "followup_capture",
                    market_id=market_id,
                    interval=interval,
                    price=capture.price,
                    price_change=price_change,
                )

        return captures

    async def shutdown(self) -> None:
        """Cancel any pending follow-up tasks."""
        for task in self._pending_followups:
            if not task.done():
                task.cancel()

        if self._pending_followups:
            await asyncio.gather(*self._pending_followups, return_exceptions=True)

        self._pending_followups = []
