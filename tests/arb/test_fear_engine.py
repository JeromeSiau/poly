"""Tests for FearSellingEngine — core orchestrator for fear-selling strategy."""

import pytest

from src.arb.fear_engine import FearSellingEngine, FearTradeSignal
from src.arb.fear_scanner import FearMarketCandidate


@pytest.fixture
def engine():
    """Create a FearSellingEngine with 100k allocated capital."""
    return FearSellingEngine(allocated_capital=100_000.0)


class TestKellySizing:
    def test_kelly_size_moderate_edge(self, engine):
        """Moderate edge should produce a positive but bounded size."""
        size = engine.compute_kelly_size(
            estimated_no_prob=0.85, no_price=0.65, available_capital=100_000.0
        )
        assert 0 < size < 30_000

    def test_kelly_size_zero_when_no_edge(self, engine):
        """When NO probability barely exceeds the price, Kelly returns zero."""
        size = engine.compute_kelly_size(
            estimated_no_prob=0.60, no_price=0.65, available_capital=100_000.0
        )
        assert size == 0.0

    def test_kelly_size_capped(self, engine):
        """Even with extreme edge, size must be capped at max_position_pct."""
        size = engine.compute_kelly_size(
            estimated_no_prob=0.99, no_price=0.50, available_capital=100_000.0
        )
        assert size <= 10_000.0


class TestClusterLimits:
    def test_cluster_exposure_within_limits(self, engine):
        """Fresh cluster should have room for a modest position."""
        assert engine.check_cluster_limit("iran", 10_000.0) is True

    def test_cluster_exposure_exceeds_limit(self, engine):
        """Adding to an already-loaded cluster should be rejected."""
        engine._cluster_exposure["iran"] = 28_000.0
        assert engine.check_cluster_limit("iran", 5_000.0) is False


class TestEntryRules:
    def test_generate_signal_from_candidate(self, engine):
        """A valid high-fear candidate should produce a BUY NO signal."""
        candidate = FearMarketCandidate(
            condition_id="0xabc123",
            token_id="tok_no_1",
            title="Will Iran be attacked by June 2026?",
            yes_price=0.35,
            no_price=0.65,
            estimated_no_probability=0.90,
            edge_pct=0.25,
            volume_24h=120_000,
            liquidity=50_000,
            end_date_iso="2026-06-30",
            fear_score=0.82,
            cluster="iran",
        )
        signal = engine.evaluate_candidate(candidate)
        assert signal is not None
        assert signal.side == "BUY"
        assert signal.outcome == "NO"
        assert signal.size_usd > 0

    def test_reject_candidate_low_fear_score(self, engine):
        """Candidates with low fear score should be rejected."""
        candidate = FearMarketCandidate(
            condition_id="0xdef456",
            token_id="tok_no_2",
            title="Will tariffs increase?",
            yes_price=0.30,
            no_price=0.70,
            estimated_no_probability=0.85,
            edge_pct=0.15,
            volume_24h=80_000,
            liquidity=30_000,
            end_date_iso="2026-09-30",
            fear_score=0.3,
            cluster="other",
        )
        signal = engine.evaluate_candidate(candidate)
        assert signal is None


class TestExitRules:
    def test_should_exit_take_profit(self, engine):
        """NO price at or above exit threshold triggers take-profit."""
        should_exit, reason = engine.check_exit(
            entry_price=0.65, current_no_price=0.96, current_yes_price=0.04
        )
        assert should_exit is True
        assert "profit" in reason.lower()

    def test_should_exit_stop_loss(self, engine):
        """YES price at or above stop threshold triggers stop-loss."""
        should_exit, reason = engine.check_exit(
            entry_price=0.65, current_no_price=0.28, current_yes_price=0.72
        )
        assert should_exit is True
        assert "stop" in reason.lower()

    def test_should_hold(self, engine):
        """When neither exit condition is met, engine should hold."""
        should_exit, reason = engine.check_exit(
            entry_price=0.65, current_no_price=0.75, current_yes_price=0.25
        )
        assert should_exit is False
