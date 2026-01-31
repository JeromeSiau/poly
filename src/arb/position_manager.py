# src/arb/position_manager.py
"""Position Manager - Tracks open positions and handles reversals.

Manages the lifecycle of positions:
1. Opening new positions when edge is detected
2. Holding positions when events confirm our side
3. Closing positions on reversal (cut losses)
4. Closing positions when match ends
"""

from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import Enum
from typing import Any, Optional

import structlog

logger = structlog.get_logger()


class PositionAction(Enum):
    """Actions the position manager can recommend."""

    OPEN = "open"           # No position, open new one
    HOLD = "hold"           # Position exists, same direction, do nothing
    ADD = "add"             # Position exists, same direction, add more
    CLOSE = "close"         # Close position (reversal or match end)
    REVERSE = "reverse"     # Close and open opposite (aggressive)
    NO_ACTION = "no_action" # No edge, no position, do nothing


@dataclass
class Position:
    """Represents an open position on a market."""

    market_id: str
    team: str               # Team we're betting ON (the one we bought)
    entry_price: float      # Price we paid (0-1)
    size: float             # Position size in dollars
    opened_at: float        # Timestamp when opened
    trigger_event: str      # What event triggered the position

    # Running P&L tracking
    current_price: float = 0.0
    unrealized_pnl: float = 0.0

    def update_price(self, new_price: float) -> None:
        """Update current price and calculate unrealized P&L.

        P&L calculation for binary markets:
        - Bought at 0.55, current 0.70 → unrealized +0.15 per share
        - Position size $100 at 0.55 → ~182 shares
        - Unrealized P&L = 182 * (0.70 - 0.55) = $27.27
        """
        self.current_price = new_price

        # Number of shares = size / entry_price
        if self.entry_price > 0:
            shares = self.size / self.entry_price
            self.unrealized_pnl = shares * (new_price - self.entry_price)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "market_id": self.market_id,
            "team": self.team,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "size": self.size,
            "unrealized_pnl": self.unrealized_pnl,
            "opened_at": self.opened_at,
            "trigger_event": self.trigger_event,
        }


@dataclass
class PositionDecision:
    """Decision made by PositionManager for an event."""

    action: PositionAction
    market_id: str
    team: Optional[str] = None
    size: float = 0.0
    reason: str = ""
    existing_position: Optional[Position] = None
    estimated_pnl: float = 0.0  # Expected P&L if we take this action

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "action": self.action.value,
            "market_id": self.market_id,
            "team": self.team,
            "size": self.size,
            "reason": self.reason,
            "existing_position": self.existing_position.to_dict() if self.existing_position else None,
            "estimated_pnl": self.estimated_pnl,
        }


class PositionManager:
    """Manages open positions and makes trading decisions.

    Key responsibilities:
    - Track all open positions by market_id
    - Decide whether to open, hold, add, or close based on new events
    - Handle reversals (when favored team changes)
    - Calculate P&L and risk metrics
    """

    def __init__(
        self,
        max_position_per_market: float = 500.0,
        reversal_threshold: float = 0.15,  # Close if fair price drops by 15%
        add_threshold: float = 0.10,       # Add if edge increases by 10%
    ):
        """Initialize PositionManager.

        Args:
            max_position_per_market: Maximum $ position per market
            reversal_threshold: Price drop threshold to trigger close
            add_threshold: Edge increase threshold to add to position
        """
        self._positions: dict[str, Position] = {}
        self.max_position_per_market = max_position_per_market
        self.reversal_threshold = reversal_threshold
        self.add_threshold = add_threshold

        # Closed positions history for P&L tracking
        self._closed_positions: list[dict] = []

    def get_position(self, market_id: str) -> Optional[Position]:
        """Get current position for a market."""
        return self._positions.get(market_id)

    def get_all_positions(self) -> list[Position]:
        """Get all open positions."""
        return list(self._positions.values())

    def evaluate(
        self,
        market_id: str,
        favored_team: str,
        fair_price: float,
        current_market_price: float,
        edge_pct: float,
        min_edge: float = 0.05,
        suggested_size: float = 0.0,
    ) -> PositionDecision:
        """Evaluate what action to take given a new event.

        Args:
            market_id: Polymarket market ID
            favored_team: Team that model says should win
            fair_price: Model's estimated fair price for favored_team
            current_market_price: Current Polymarket price
            edge_pct: Calculated edge percentage
            min_edge: Minimum edge to act on
            suggested_size: Suggested position size from Kelly

        Returns:
            PositionDecision with recommended action
        """
        existing = self._positions.get(market_id)

        # Case 1: No existing position
        if existing is None:
            if edge_pct >= min_edge:
                return PositionDecision(
                    action=PositionAction.OPEN,
                    market_id=market_id,
                    team=favored_team,
                    size=min(suggested_size, self.max_position_per_market),
                    reason=f"New opportunity: {favored_team} at {fair_price:.1%} vs market {current_market_price:.1%}",
                )
            else:
                return PositionDecision(
                    action=PositionAction.NO_ACTION,
                    market_id=market_id,
                    reason=f"Edge {edge_pct:.1%} below threshold {min_edge:.1%}",
                )

        # Case 2: Existing position on SAME team
        if existing.team == favored_team:
            # Update price for P&L tracking
            existing.update_price(current_market_price)

            # Check if we should add to position
            price_improvement = fair_price - existing.entry_price

            if price_improvement >= self.add_threshold:
                # Edge increased significantly, consider adding
                room_to_add = self.max_position_per_market - existing.size
                add_size = min(suggested_size, room_to_add)

                if add_size > 50:  # Minimum $50 to add
                    return PositionDecision(
                        action=PositionAction.ADD,
                        market_id=market_id,
                        team=favored_team,
                        size=add_size,
                        reason=f"Edge increased: fair {fair_price:.1%} vs entry {existing.entry_price:.1%}",
                        existing_position=existing,
                    )

            # Otherwise just hold
            return PositionDecision(
                action=PositionAction.HOLD,
                market_id=market_id,
                team=favored_team,
                reason=f"Holding {existing.team}: unrealized P&L ${existing.unrealized_pnl:.2f}",
                existing_position=existing,
            )

        # Case 3: Existing position on OPPOSITE team (REVERSAL!)
        # This is the critical case - we bet on Team A but now Team B is favored
        #
        # IMPORTANT: Use current_market_price, not fair_price for reversal decision
        # Because we'd sell at market price, not model's estimate
        our_team_market_price = 1 - current_market_price
        existing.update_price(our_team_market_price)

        # Calculate actual price drop based on market, not model
        price_drop = existing.entry_price - our_team_market_price

        logger.warning(
            "reversal_detected",
            market_id=market_id,
            our_team=existing.team,
            new_favored=favored_team,
            entry_price=existing.entry_price,
            our_team_market_price=our_team_market_price,
            price_drop=price_drop,
        )

        # If price dropped significantly, cut losses
        if price_drop >= self.reversal_threshold:
            # Calculate expected loss if we close now (at our team's market price)
            shares = existing.size / existing.entry_price
            expected_loss = shares * (our_team_market_price - existing.entry_price)

            return PositionDecision(
                action=PositionAction.CLOSE,
                market_id=market_id,
                team=existing.team,
                size=existing.size,
                reason=f"REVERSAL: {favored_team} now favored. Cut loss on {existing.team}",
                existing_position=existing,
                estimated_pnl=expected_loss,
            )

        # Price hasn't dropped enough to close yet - hold and monitor
        return PositionDecision(
            action=PositionAction.HOLD,
            market_id=market_id,
            team=existing.team,
            reason=f"Reversal brewing but holding: drop {price_drop:.1%} < threshold {self.reversal_threshold:.1%}",
            existing_position=existing,
        )

    def open_position(
        self,
        market_id: str,
        team: str,
        entry_price: float,
        size: float,
        trigger_event: str,
    ) -> Position:
        """Open a new position.

        Args:
            market_id: Market identifier
            team: Team we're betting on
            entry_price: Price we're buying at
            size: Position size in dollars
            trigger_event: Event that triggered this position

        Returns:
            The created Position
        """
        position = Position(
            market_id=market_id,
            team=team,
            entry_price=entry_price,
            size=size,
            opened_at=datetime.now(UTC).timestamp(),
            trigger_event=trigger_event,
            current_price=entry_price,
        )

        self._positions[market_id] = position

        logger.info(
            "position_opened",
            market_id=market_id,
            team=team,
            entry_price=entry_price,
            size=size,
        )

        return position

    def close_position(
        self,
        market_id: str,
        exit_price: float,
        reason: str = "manual",
    ) -> Optional[dict]:
        """Close an existing position.

        Args:
            market_id: Market to close
            exit_price: Price we're selling at
            reason: Why we're closing

        Returns:
            Closed position details with realized P&L, or None if no position
        """
        position = self._positions.pop(market_id, None)

        if position is None:
            return None

        # Calculate realized P&L
        shares = position.size / position.entry_price
        realized_pnl = shares * (exit_price - position.entry_price)

        closed_record = {
            **position.to_dict(),
            "exit_price": exit_price,
            "realized_pnl": realized_pnl,
            "closed_at": datetime.now(UTC).timestamp(),
            "close_reason": reason,
        }

        self._closed_positions.append(closed_record)

        logger.info(
            "position_closed",
            market_id=market_id,
            team=position.team,
            entry_price=position.entry_price,
            exit_price=exit_price,
            realized_pnl=realized_pnl,
            reason=reason,
        )

        return closed_record

    def add_to_position(
        self,
        market_id: str,
        additional_price: float,
        additional_size: float,
    ) -> Optional[Position]:
        """Add to an existing position.

        Calculates new average entry price.

        Args:
            market_id: Market to add to
            additional_price: Price of new shares
            additional_size: Dollar amount to add

        Returns:
            Updated position, or None if no existing position
        """
        position = self._positions.get(market_id)

        if position is None:
            return None

        # Calculate new average entry price
        old_shares = position.size / position.entry_price
        new_shares = additional_size / additional_price
        total_shares = old_shares + new_shares
        total_cost = position.size + additional_size

        new_avg_price = total_cost / total_shares

        position.entry_price = new_avg_price
        position.size = total_cost

        logger.info(
            "position_increased",
            market_id=market_id,
            team=position.team,
            new_avg_price=new_avg_price,
            new_size=total_cost,
        )

        return position

    def get_total_exposure(self) -> float:
        """Get total dollar exposure across all positions."""
        return sum(p.size for p in self._positions.values())

    def get_total_unrealized_pnl(self) -> float:
        """Get total unrealized P&L across all positions."""
        return sum(p.unrealized_pnl for p in self._positions.values())

    def get_total_realized_pnl(self) -> float:
        """Get total realized P&L from closed positions."""
        return sum(p.get("realized_pnl", 0) for p in self._closed_positions)

    def get_closed_positions(self) -> list[dict]:
        """Get history of closed positions."""
        return list(self._closed_positions)

    def to_dict(self) -> dict[str, Any]:
        """Get full state as dictionary."""
        return {
            "open_positions": [p.to_dict() for p in self._positions.values()],
            "total_exposure": self.get_total_exposure(),
            "total_unrealized_pnl": self.get_total_unrealized_pnl(),
            "total_realized_pnl": self.get_total_realized_pnl(),
            "closed_count": len(self._closed_positions),
        }
