"""Tests for position manager."""

import pytest

from src.paper_trading.position_manager import PositionManager, kelly_fraction


class TestKellyFraction:
    def test_positive_edge_positive_fraction(self):
        """Positive edge should give positive Kelly fraction."""
        # Our prob: 60%, market: 50% → edge = 10%
        fraction = kelly_fraction(our_prob=0.60, market_price=0.50)
        assert fraction > 0
        assert fraction < 1

    def test_no_edge_zero_fraction(self):
        """No edge should give zero fraction."""
        fraction = kelly_fraction(our_prob=0.50, market_price=0.50)
        assert fraction == pytest.approx(0.0, abs=0.01)

    def test_negative_edge_zero_fraction(self):
        """Negative edge should return 0 (don't bet)."""
        fraction = kelly_fraction(our_prob=0.40, market_price=0.50)
        assert fraction == 0

    def test_capped_at_max(self):
        """Kelly fraction is capped at 25%."""
        # Huge edge
        fraction = kelly_fraction(our_prob=0.99, market_price=0.10)
        assert fraction <= 0.25


class TestPositionManager:
    @pytest.fixture
    def manager(self):
        return PositionManager(
            capital=10000,
            max_position_pct=0.25,
            min_edge=0.05,
            max_daily_loss_pct=0.10,
        )

    def test_calculate_position_size(self, manager):
        """Calculate position size based on Kelly."""
        size = manager.calculate_position_size(
            our_prob=0.70,
            market_price=0.55,
        )
        assert size > 0
        assert size <= 2500  # Max 25% of 10k

    def test_position_size_zero_below_min_edge(self, manager):
        """No position if edge below minimum."""
        size = manager.calculate_position_size(
            our_prob=0.52,  # 2% edge, below 5% min
            market_price=0.50,
        )
        assert size == 0

    def test_position_respects_daily_loss_limit(self, manager):
        """Position size reduced when approaching daily loss limit."""
        manager.record_loss(800)  # Already lost 8%

        size = manager.calculate_position_size(
            our_prob=0.80,
            market_price=0.50,
        )
        # Should be capped at remaining 2% = $200
        assert size <= 200

    def test_can_trade_checks_limits(self, manager):
        """can_trade returns False when at limit."""
        manager.record_loss(1000)  # Hit 10% daily limit

        assert manager.can_trade() is False

    def test_record_pnl_updates_state(self, manager):
        """Recording P&L updates daily totals."""
        manager.record_win(100)
        manager.record_loss(50)

        assert manager.daily_pnl == 50
        assert manager.capital == 10050
