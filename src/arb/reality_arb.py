# src/arb/reality_arb.py
"""Reality Arbitrage Engine - Core trading logic.

This is the main trading logic that:
1. Receives significant game events from EventDetector
2. Looks up corresponding Polymarket market via MarketMapper
3. Compares estimated fair price vs current market price
4. If edge > threshold, generates trading opportunity
5. Executes trade if in autopilot mode, or alerts for manual approval
"""

from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Any, Optional

import structlog

from src.realtime.event_detector import EventDetector, SignificantEvent
from src.realtime.market_mapper import MarketMapper, MarketMapping
from config.settings import settings

logger = structlog.get_logger()


@dataclass
class ArbOpportunity:
    """Represents an arbitrage opportunity detected by the engine."""

    market_id: str
    side: str  # "BUY" or "SELL"
    outcome: str  # "YES" or "NO"
    current_price: float
    estimated_fair_price: float
    edge_pct: float
    trigger_event: str
    timestamp: float

    @property
    def is_valid(self) -> bool:
        """Check if the opportunity is still valid.

        An opportunity is valid if:
        - Edge is above minimum threshold
        - Not past anomaly threshold (too good to be true)
        - Timestamp is recent
        """
        min_edge = settings.MIN_EDGE_PCT
        max_edge = settings.ANOMALY_THRESHOLD_PCT

        return (
            self.edge_pct >= min_edge
            and self.edge_pct < max_edge
            and self.current_price > 0
            and self.current_price < 1
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert opportunity to dictionary for serialization."""
        return {
            "market_id": self.market_id,
            "side": self.side,
            "outcome": self.outcome,
            "current_price": self.current_price,
            "estimated_fair_price": self.estimated_fair_price,
            "edge_pct": self.edge_pct,
            "trigger_event": self.trigger_event,
            "timestamp": self.timestamp,
            "is_valid": self.is_valid,
        }


class RealityArbEngine:
    """Reality Arbitrage Engine - Core trading logic.

    Detects arbitrage opportunities by comparing game events
    with market prices on Polymarket.
    """

    def __init__(
        self,
        polymarket_feed: Optional[Any] = None,
        event_detector: Optional[EventDetector] = None,
        market_mapper: Optional[MarketMapper] = None,
    ):
        """Initialize the Reality Arbitrage Engine.

        Args:
            polymarket_feed: Feed for Polymarket price data
            event_detector: Detector for significant game events
            market_mapper: Mapper for linking events to markets
        """
        self.polymarket_feed = polymarket_feed
        self.event_detector = event_detector or EventDetector()
        self.market_mapper = market_mapper or MarketMapper()

        # Risk parameters (from settings)
        self.capital: float = 10000.0
        self.max_position_pct: float = settings.MAX_POSITION_PCT
        self.min_edge_pct: float = settings.MIN_EDGE_PCT
        self.anomaly_threshold: float = settings.ANOMALY_THRESHOLD_PCT

        # Executor for trade execution
        self.executor: Optional[Any] = None

        # Pending opportunities
        self._pending_opportunities: dict[str, ArbOpportunity] = {}

    def evaluate_opportunity(
        self,
        event: SignificantEvent,
        mapping: MarketMapping,
    ) -> Optional[ArbOpportunity]:
        """Evaluate if an event creates an arbitrage opportunity.

        Args:
            event: Significant game event detected
            mapping: Market mapping linking event to Polymarket

        Returns:
            ArbOpportunity if edge exists, None otherwise
        """
        if not event.is_significant or not event.favored_team:
            return None

        # Get the outcome for the favored team
        outcome = mapping.get_outcome_for_team(event.favored_team)
        if not outcome:
            logger.warning(
                "no_outcome_for_team",
                team=event.favored_team,
                mapping=mapping.team_to_outcome,
            )
            return None

        # Get current market prices
        if self.polymarket_feed:
            best_bid, best_ask = self.polymarket_feed.get_best_prices(
                mapping.polymarket_id, outcome
            )
        else:
            best_bid, best_ask = None, None

        if best_bid is None or best_ask is None:
            logger.warning(
                "no_prices_available",
                market_id=mapping.polymarket_id,
                outcome=outcome,
            )
            return None

        # Current price is the midpoint
        current_price = (best_bid + best_ask) / 2

        # Estimate fair price based on event impact
        # Use the event detector's price impact estimation
        estimated_fair_price = self.event_detector.estimate_price_impact(
            event, current_price
        )

        # Calculate edge
        edge_pct = estimated_fair_price - current_price

        # Determine side based on edge direction
        if edge_pct > 0:
            side = "BUY"
        elif edge_pct < 0:
            side = "SELL"
            edge_pct = abs(edge_pct)
        else:
            return None

        # Check if edge meets minimum threshold
        if edge_pct < self.min_edge_pct:
            logger.debug(
                "edge_below_threshold",
                edge_pct=edge_pct,
                min_edge=self.min_edge_pct,
            )
            return None

        # Create opportunity
        opportunity = ArbOpportunity(
            market_id=mapping.polymarket_id,
            side=side,
            outcome=outcome,
            current_price=current_price,
            estimated_fair_price=estimated_fair_price,
            edge_pct=edge_pct,
            trigger_event=event.event_description,
            timestamp=datetime.now(UTC).timestamp(),
        )

        # Store in pending opportunities
        self._pending_opportunities[mapping.polymarket_id] = opportunity

        logger.info(
            "opportunity_detected",
            market_id=opportunity.market_id,
            side=opportunity.side,
            outcome=opportunity.outcome,
            edge_pct=opportunity.edge_pct,
            current_price=opportunity.current_price,
            fair_price=opportunity.estimated_fair_price,
        )

        return opportunity

    def calculate_position_size(self, opportunity: ArbOpportunity) -> float:
        """Calculate position size using Kelly criterion.

        The Kelly criterion optimizes long-term growth by betting a fraction
        of capital proportional to edge and probability.

        Kelly fraction = edge / odds

        For simplicity, we use a fractional Kelly (1/4) to reduce variance.

        Args:
            opportunity: The arbitrage opportunity

        Returns:
            Position size in dollars (capped by max_position_pct)
        """
        # Max position based on capital and risk limit
        max_position = self.capital * self.max_position_pct

        # Kelly criterion: f* = edge / odds
        # For binary markets, odds = 1 / (1 - p) where p is probability
        # Simplified: f* = edge / variance
        prob = opportunity.estimated_fair_price
        variance = prob * (1 - prob)

        if variance <= 0:
            return 0.0

        # Full Kelly
        kelly_fraction = opportunity.edge_pct / variance

        # Use fractional Kelly (1/4) for safety
        fractional_kelly = kelly_fraction * 0.25

        # Calculate position size
        position = self.capital * fractional_kelly

        # Cap at maximum allowed position
        position = min(position, max_position)

        # Ensure positive
        position = max(0.0, position)

        logger.debug(
            "position_size_calculated",
            kelly_fraction=kelly_fraction,
            fractional_kelly=fractional_kelly,
            position=position,
            max_position=max_position,
        )

        return position

    async def execute(
        self,
        opportunity: ArbOpportunity,
        size: float,
    ) -> dict[str, Any]:
        """Execute a trade for the given opportunity.

        Args:
            opportunity: The arbitrage opportunity to execute
            size: Position size in dollars

        Returns:
            Execution result dictionary
        """
        if not self.executor:
            return {
                "status": "NO_EXECUTOR",
                "message": "No executor configured",
            }

        if not opportunity.is_valid:
            return {
                "status": "INVALID",
                "message": "Opportunity is no longer valid",
            }

        logger.info(
            "executing_trade",
            market_id=opportunity.market_id,
            side=opportunity.side,
            outcome=opportunity.outcome,
            size=size,
            price=opportunity.current_price,
        )

        try:
            result = await self.executor.place_order(
                market_id=opportunity.market_id,
                side=opportunity.side,
                outcome=opportunity.outcome,
                size=size,
                price=opportunity.current_price,
            )

            logger.info(
                "trade_executed",
                result=result,
            )

            # Clear the opportunity after execution
            self.clear_opportunity(opportunity.market_id)

            return result

        except Exception as e:
            logger.error(
                "trade_execution_failed",
                error=str(e),
                market_id=opportunity.market_id,
            )
            return {
                "status": "ERROR",
                "message": str(e),
            }

    async def process_event(
        self,
        event: SignificantEvent,
        mapping: MarketMapping,
    ) -> Optional[dict[str, Any]]:
        """Process a game event and potentially execute a trade.

        This is the main entry point for the engine. It:
        1. Evaluates if the event creates an opportunity
        2. Calculates position size
        3. Executes if in autopilot mode

        Args:
            event: Significant game event
            mapping: Market mapping for the event

        Returns:
            Trade result if executed, opportunity dict if pending, None if no opportunity
        """
        opportunity = self.evaluate_opportunity(event, mapping)

        if not opportunity:
            return None

        size = self.calculate_position_size(opportunity)

        if size <= 0:
            logger.info(
                "position_size_zero",
                opportunity=opportunity.to_dict(),
            )
            return opportunity.to_dict()

        # Check if autopilot mode is enabled
        if settings.AUTOPILOT_MODE and self.executor:
            result = await self.execute(opportunity, size)
            return {
                **opportunity.to_dict(),
                "execution": result,
                "size": size,
            }

        # Return opportunity for manual approval
        return {
            **opportunity.to_dict(),
            "size": size,
            "status": "PENDING_APPROVAL",
        }

    def get_pending_opportunities(self) -> list[ArbOpportunity]:
        """Get all pending arbitrage opportunities.

        Returns:
            List of pending ArbOpportunity instances
        """
        return list(self._pending_opportunities.values())

    def clear_opportunity(self, market_id: str) -> None:
        """Clear a pending opportunity.

        Args:
            market_id: Market ID to clear
        """
        self._pending_opportunities.pop(market_id, None)
