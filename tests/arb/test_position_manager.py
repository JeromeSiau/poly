# tests/arb/test_position_manager.py
"""Tests for PositionManager - position tracking and reversal handling."""

import pytest
from src.arb.position_manager import (
    PositionManager,
    PositionAction,
    Position,
)


class TestPositionManager:
    """Tests for PositionManager."""

    def test_no_position_with_edge_opens(self):
        """Should recommend OPEN when no position and edge exists."""
        pm = PositionManager()

        decision = pm.evaluate(
            market_id="market_123",
            favored_team="T1",
            fair_price=0.75,
            current_market_price=0.55,
            edge_pct=0.20,
            min_edge=0.05,
            suggested_size=100.0,
        )

        assert decision.action == PositionAction.OPEN
        assert decision.team == "T1"
        assert decision.size == 100.0

    def test_no_position_without_edge_no_action(self):
        """Should recommend NO_ACTION when no position and no edge."""
        pm = PositionManager()

        decision = pm.evaluate(
            market_id="market_123",
            favored_team="T1",
            fair_price=0.56,
            current_market_price=0.55,
            edge_pct=0.01,
            min_edge=0.05,
            suggested_size=100.0,
        )

        assert decision.action == PositionAction.NO_ACTION

    def test_same_team_position_holds(self):
        """Should HOLD when new event confirms our existing position."""
        pm = PositionManager()

        # Open position on T1
        pm.open_position(
            market_id="market_123",
            team="T1",
            outcome="YES",
            entry_price=0.55,
            size=100.0,
            trigger_event="Baron kill",
        )

        # New event also favors T1
        decision = pm.evaluate(
            market_id="market_123",
            favored_team="T1",
            fair_price=0.78,
            current_market_price=0.60,
            edge_pct=0.18,
            min_edge=0.05,
            suggested_size=50.0,
        )

        assert decision.action == PositionAction.HOLD
        assert decision.existing_position is not None
        assert decision.existing_position.team == "T1"

    def test_reversal_triggers_close(self):
        """Should recommend CLOSE when favored team changes significantly."""
        pm = PositionManager(reversal_threshold=0.15)

        # Open position on T1
        pm.open_position(
            market_id="market_123",
            team="T1",
            outcome="YES",
            entry_price=0.55,
            size=100.0,
            trigger_event="Baron kill",
        )

        # New event favors GenG with significant shift
        # T1's fair price = 1 - 0.80 = 0.20 (was 0.55)
        # Price drop = 0.55 - 0.20 = 0.35 > 0.15 threshold
        decision = pm.evaluate(
            market_id="market_123",
            favored_team="GenG",
            fair_price=0.80,  # GenG's fair price
            current_market_price=0.70,
            edge_pct=0.10,
            min_edge=0.05,
            suggested_size=50.0,
        )

        assert decision.action == PositionAction.CLOSE
        assert "REVERSAL" in decision.reason
        assert decision.existing_position.team == "T1"

    def test_small_reversal_holds(self):
        """Should HOLD on small reversal below threshold."""
        pm = PositionManager(reversal_threshold=0.15)

        # Open position on T1
        pm.open_position(
            market_id="market_123",
            team="T1",
            outcome="YES",
            entry_price=0.55,
            size=100.0,
            trigger_event="Baron kill",
        )

        # Small reversal - GenG barely favored
        # T1's fair price = 1 - 0.52 = 0.48 (was 0.55)
        # Price drop = 0.55 - 0.48 = 0.07 < 0.15 threshold
        decision = pm.evaluate(
            market_id="market_123",
            favored_team="GenG",
            fair_price=0.52,  # GenG barely favored
            current_market_price=0.50,
            edge_pct=0.02,
            min_edge=0.05,
            suggested_size=50.0,
        )

        assert decision.action == PositionAction.HOLD
        assert "brewing" in decision.reason.lower()

    def test_open_and_close_tracks_pnl(self):
        """Should track P&L correctly through position lifecycle."""
        pm = PositionManager()

        # Open position
        pm.open_position(
            market_id="market_123",
            team="T1",
            outcome="YES",
            entry_price=0.55,
            size=100.0,
            trigger_event="Baron kill",
        )

        assert pm.get_total_exposure() == 100.0

        # Close at profit
        closed = pm.close_position(
            market_id="market_123",
            exit_price=0.70,
            reason="match_end",
        )

        assert closed is not None
        # Shares = 100 / 0.55 = 181.82
        # P&L = 181.82 * (0.70 - 0.55) = 27.27
        assert closed["realized_pnl"] == pytest.approx(27.27, rel=0.01)
        assert pm.get_total_exposure() == 0.0
        assert pm.get_total_realized_pnl() == pytest.approx(27.27, rel=0.01)

    def test_close_at_loss(self):
        """Should correctly calculate loss on close."""
        pm = PositionManager()

        pm.open_position(
            market_id="market_123",
            team="T1",
            outcome="YES",
            entry_price=0.55,
            size=100.0,
            trigger_event="Baron kill",
        )

        # Close at loss (reversal happened)
        closed = pm.close_position(
            market_id="market_123",
            exit_price=0.30,
            reason="reversal",
        )

        # Shares = 100 / 0.55 = 181.82
        # P&L = 181.82 * (0.30 - 0.55) = -45.45
        assert closed["realized_pnl"] == pytest.approx(-45.45, rel=0.01)

    def test_add_to_position(self):
        """Should correctly average entry price when adding."""
        pm = PositionManager(max_position_per_market=500.0, add_threshold=0.05)

        pm.open_position(
            market_id="market_123",
            team="T1",
            outcome="YES",
            entry_price=0.50,
            size=100.0,
            trigger_event="First event",
        )

        # Add more at higher price
        pm.add_to_position(
            market_id="market_123",
            additional_price=0.60,
            additional_size=100.0,
        )

        position = pm.get_position("market_123")

        # Old: 100 / 0.50 = 200 shares
        # New: 100 / 0.60 = 166.67 shares
        # Total: 366.67 shares, cost $200
        # Avg price: 200 / 366.67 = 0.5455
        assert position.size == 200.0
        assert position.entry_price == pytest.approx(0.5455, rel=0.01)

    def test_multiple_positions(self):
        """Should track multiple positions independently."""
        pm = PositionManager()

        pm.open_position("market_1", "T1", "YES", 0.55, 100.0, "Event 1")
        pm.open_position("market_2", "GenG", "YES", 0.60, 150.0, "Event 2")

        assert len(pm.get_all_positions()) == 2
        assert pm.get_total_exposure() == 250.0

        pm.close_position("market_1", 0.70, "done")

        assert len(pm.get_all_positions()) == 1
        assert pm.get_total_exposure() == 150.0


class TestPosition:
    """Tests for Position dataclass."""

    def test_unrealized_pnl_calculation(self):
        """Should calculate unrealized P&L correctly."""
        position = Position(
            market_id="market_123",
            team="T1",
            outcome="YES",
            entry_price=0.50,
            size=100.0,
            opened_at=0.0,
            trigger_event="Test",
        )

        position.update_price(0.70)

        # 100 / 0.50 = 200 shares
        # 200 * (0.70 - 0.50) = 40
        assert position.unrealized_pnl == pytest.approx(40.0)

    def test_unrealized_loss(self):
        """Should calculate unrealized loss correctly."""
        position = Position(
            market_id="market_123",
            team="T1",
            outcome="YES",
            entry_price=0.60,
            size=120.0,
            opened_at=0.0,
            trigger_event="Test",
        )

        position.update_price(0.40)

        # 120 / 0.60 = 200 shares
        # 200 * (0.40 - 0.60) = -40
        assert position.unrealized_pnl == pytest.approx(-40.0)
