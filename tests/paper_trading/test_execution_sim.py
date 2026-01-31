# tests/paper_trading/test_execution_sim.py
"""Tests for execution simulation."""

import pytest

from src.paper_trading.execution_sim import ExecutionSimulator


class TestExecutionSimulator:
    @pytest.fixture
    def simulator(self):
        return ExecutionSimulator(
            default_depth=10000,  # $10k default depth
            max_slippage=0.05,
        )

    def test_small_order_minimal_slippage(self, simulator):
        """Small orders should have minimal slippage."""
        fill_price = simulator.simulate_fill(
            target_price=0.50,
            size=100,
            side="BUY",
        )

        # Slippage should be tiny
        slippage = (fill_price - 0.50) / 0.50
        assert slippage < 0.01

    def test_large_order_more_slippage(self, simulator):
        """Large orders relative to depth should have more slippage."""
        fill_price = simulator.simulate_fill(
            target_price=0.50,
            size=5000,  # 50% of depth
            side="BUY",
        )

        slippage = (fill_price - 0.50) / 0.50
        assert slippage > 0.01

    def test_buy_slippage_increases_price(self, simulator):
        """Buying should fill at higher price."""
        fill_price = simulator.simulate_fill(
            target_price=0.50,
            size=1000,
            side="BUY",
        )

        assert fill_price >= 0.50

    def test_sell_slippage_decreases_price(self, simulator):
        """Selling should fill at lower price."""
        fill_price = simulator.simulate_fill(
            target_price=0.50,
            size=1000,
            side="SELL",
        )

        assert fill_price <= 0.50

    def test_max_slippage_cap(self, simulator):
        """Slippage should be capped."""
        fill_price = simulator.simulate_fill(
            target_price=0.50,
            size=100000,  # Huge order
            side="BUY",
        )

        slippage = (fill_price - 0.50) / 0.50
        assert slippage <= 0.05 + 1e-9  # Allow for floating point imprecision

    def test_custom_depth(self, simulator):
        """Can specify custom orderbook depth."""
        fill_price = simulator.simulate_fill(
            target_price=0.50,
            size=1000,
            side="BUY",
            orderbook_depth=50000,  # Deeper book
        )

        slippage = (fill_price - 0.50) / 0.50
        assert slippage < 0.005  # Very small
