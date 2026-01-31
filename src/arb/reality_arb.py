# src/arb/reality_arb.py
"""Reality Arbitrage Engine - Core trading logic.

This is the main trading logic that:
1. Receives significant game events from EventDetector
2. Looks up corresponding Polymarket market via MarketMapper
3. Compares estimated fair price vs current market price
4. If edge > threshold, generates trading opportunity
5. Manages positions and handles reversals via PositionManager
6. Executes trade if in autopilot mode, or alerts for manual approval
"""

from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Any, Optional

import structlog

from src.realtime.event_detector import EventDetector, SignificantEvent
from src.realtime.market_mapper import MarketMapper, MarketMapping
from src.arb.position_manager import PositionManager, PositionAction, PositionDecision
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
        position_manager: Optional[PositionManager] = None,
    ):
        """Initialize the Reality Arbitrage Engine.

        Args:
            polymarket_feed: Feed for Polymarket price data
            event_detector: Detector for significant game events
            market_mapper: Mapper for linking events to markets
            position_manager: Manager for tracking open positions
        """
        self.polymarket_feed = polymarket_feed

        # Load ML model if configured
        if event_detector:
            self.event_detector = event_detector
        elif settings.ML_USE_MODEL:
            from pathlib import Path
            model_path = Path(settings.ML_MODEL_PATH)
            self.event_detector = EventDetector(
                model_path=model_path if model_path.exists() else None
            )
        else:
            self.event_detector = EventDetector()

        self.market_mapper = market_mapper or MarketMapper()

        # Risk parameters (from settings)
        self.capital: float = 10000.0
        self.max_position_pct: float = settings.MAX_POSITION_PCT
        self.min_edge_pct: float = settings.MIN_EDGE_PCT
        self.anomaly_threshold: float = settings.ANOMALY_THRESHOLD_PCT

        # Position manager for tracking open positions and handling reversals
        self.position_manager = position_manager or PositionManager(
            max_position_per_market=self.capital * self.max_position_pct,
        )

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
        2. Consults PositionManager for action (open/hold/close/reverse)
        3. Calculates position size
        4. Executes if in autopilot mode

        Args:
            event: Significant game event
            mapping: Market mapping for the event

        Returns:
            Trade result if executed, opportunity dict if pending, None if no opportunity
        """
        # ALWAYS update existing position prices for P&L tracking
        # This must happen regardless of whether there's an opportunity
        existing = self.position_manager.get_position(mapping.polymarket_id)
        if existing and self.polymarket_feed:
            outcome = mapping.get_outcome_for_team(existing.team)
            if outcome:
                best_bid, best_ask = self.polymarket_feed.get_best_prices(
                    mapping.polymarket_id, outcome
                )
                if best_bid and best_ask:
                    existing.update_price((best_bid + best_ask) / 2)

        opportunity = self.evaluate_opportunity(event, mapping)

        if not opportunity:
            return None

        size = self.calculate_position_size(opportunity)

        # Consult position manager for what action to take
        decision = self.position_manager.evaluate(
            market_id=opportunity.market_id,
            favored_team=event.favored_team,
            fair_price=opportunity.estimated_fair_price,
            current_market_price=opportunity.current_price,
            edge_pct=opportunity.edge_pct,
            min_edge=self.min_edge_pct,
            suggested_size=size,
        )

        logger.info(
            "position_decision",
            action=decision.action.value,
            market_id=decision.market_id,
            team=decision.team,
            reason=decision.reason,
        )

        # Handle different actions
        if decision.action == PositionAction.NO_ACTION:
            return {
                **opportunity.to_dict(),
                "position_action": decision.to_dict(),
                "status": "NO_ACTION",
            }

        if decision.action == PositionAction.HOLD:
            return {
                **opportunity.to_dict(),
                "position_action": decision.to_dict(),
                "status": "HOLDING",
            }

        if decision.action == PositionAction.CLOSE:
            # Need to close position (reversal detected)
            if settings.AUTOPILOT_MODE and self.executor:
                # Execute SELL order
                close_result = await self._execute_close(
                    decision.existing_position,
                    opportunity.current_price,
                    decision.reason,
                )
                return {
                    **opportunity.to_dict(),
                    "position_action": decision.to_dict(),
                    "execution": close_result,
                    "status": "CLOSED",
                }
            else:
                return {
                    **opportunity.to_dict(),
                    "position_action": decision.to_dict(),
                    "status": "CLOSE_PENDING_APPROVAL",
                }

        if decision.action in (PositionAction.OPEN, PositionAction.ADD):
            trade_size = decision.size

            if trade_size <= 0:
                logger.info(
                    "position_size_zero",
                    opportunity=opportunity.to_dict(),
                )
                return {
                    **opportunity.to_dict(),
                    "position_action": decision.to_dict(),
                    "status": "SIZE_ZERO",
                }

            # Check if autopilot mode is enabled
            if settings.AUTOPILOT_MODE and self.executor:
                # CRITICAL: Record position BEFORE execution to prevent loss tracking
                # on crash. If execution fails, we roll back the position.
                if decision.action == PositionAction.OPEN:
                    self.position_manager.open_position(
                        market_id=opportunity.market_id,
                        team=event.favored_team,
                        entry_price=opportunity.current_price,
                        size=trade_size,
                        trigger_event=opportunity.trigger_event,
                    )
                elif decision.action == PositionAction.ADD:
                    self.position_manager.add_to_position(
                        market_id=opportunity.market_id,
                        additional_price=opportunity.current_price,
                        additional_size=trade_size,
                    )

                result = await self.execute(opportunity, trade_size)

                # Roll back position if execution failed
                if result.get("status") in ("ERROR", "NO_EXECUTOR", "INVALID"):
                    logger.warning(
                        "rolling_back_position",
                        market_id=opportunity.market_id,
                        reason=result.get("status"),
                    )
                    if decision.action == PositionAction.OPEN:
                        # Remove the position we just created
                        self.position_manager.close_position(
                            market_id=opportunity.market_id,
                            exit_price=opportunity.current_price,
                            reason="execution_failed_rollback",
                        )
                    # For ADD, we'd need to track the previous state - for now log warning
                    elif decision.action == PositionAction.ADD:
                        logger.error(
                            "cannot_rollback_add",
                            market_id=opportunity.market_id,
                            message="ADD position rollback not implemented",
                        )

                return {
                    **opportunity.to_dict(),
                    "position_action": decision.to_dict(),
                    "execution": result,
                    "size": trade_size,
                }

            # Return opportunity for manual approval
            return {
                **opportunity.to_dict(),
                "position_action": decision.to_dict(),
                "size": trade_size,
                "status": "PENDING_APPROVAL",
            }

        return None

    async def _execute_close(
        self,
        position: Any,
        exit_price: float,
        reason: str,
    ) -> dict[str, Any]:
        """Execute a position close (sell).

        Args:
            position: Position to close
            exit_price: Price to sell at
            reason: Why we're closing

        Returns:
            Execution result
        """
        if not self.executor or not position:
            return {"status": "NO_EXECUTOR"}

        try:
            result = await self.executor.place_order(
                market_id=position.market_id,
                side="SELL",
                outcome=position.team,
                size=position.size,
                price=exit_price,
            )

            # Update position manager
            closed = self.position_manager.close_position(
                market_id=position.market_id,
                exit_price=exit_price,
                reason=reason,
            )

            logger.info(
                "position_closed",
                market_id=position.market_id,
                realized_pnl=closed.get("realized_pnl") if closed else None,
            )

            return {
                **result,
                "closed_position": closed,
            }

        except Exception as e:
            logger.error("close_execution_failed", error=str(e))
            return {"status": "ERROR", "message": str(e)}

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

    async def close_match_position(
        self,
        market_id: str,
        winner: str,
        final_price: float = 1.0,
    ) -> Optional[dict]:
        """Close position when match ends.

        Called when a match concludes to settle the position.

        Args:
            market_id: Market that just concluded
            winner: Team that won
            final_price: Final settlement price (1.0 for winner, 0.0 for loser)

        Returns:
            Closed position details with P&L
        """
        position = self.position_manager.get_position(market_id)

        if not position:
            return None

        # Determine settlement price based on whether we bet on winner
        # In binary markets: winner's shares settle at 1.0, loser's at 0.0
        if position.team == winner:
            settlement_price = final_price  # We won, shares settle at ~1.0
        else:
            settlement_price = 0.0  # We lost, shares are worthless

        closed = self.position_manager.close_position(
            market_id=market_id,
            exit_price=settlement_price,
            reason=f"match_ended_{winner}_won",
        )

        logger.info(
            "match_position_settled",
            market_id=market_id,
            our_team=position.team,
            winner=winner,
            realized_pnl=closed.get("realized_pnl") if closed else None,
        )

        return closed

    def get_positions_summary(self) -> dict[str, Any]:
        """Get summary of all positions.

        Returns:
            Dictionary with position summary
        """
        return self.position_manager.to_dict()
