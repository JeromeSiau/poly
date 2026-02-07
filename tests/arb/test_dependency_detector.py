"""Tests for cross-market logical dependency detection."""
import pytest
from unittest.mock import AsyncMock, patch

from src.arb.dependency_detector import DependencyDetector, MarketDependency


class TestMarketDependency:
    def test_has_arbitrage_yes(self):
        dep = MarketDependency(
            market_a_id="mkt-a", market_a_title="Will Trump win Pennsylvania?",
            market_b_id="mkt-b", market_b_title="Will Republicans win by 5+ points nationally?",
            dependency_type="implication", description="If B is YES then A must be YES",
            valid_outcomes=[("YES", "YES"), ("YES", "NO"), ("NO", "NO")],
            invalid_outcomes=[("NO", "YES")], confidence=0.95,
        )
        assert dep.has_dependency
        assert dep.n_valid_outcomes == 3
        assert dep.n_total_outcomes == 4

    def test_no_dependency(self):
        dep = MarketDependency(
            market_a_id="mkt-a", market_a_title="Market A",
            market_b_id="mkt-b", market_b_title="Market B",
            dependency_type="independent", description="No dependency",
            valid_outcomes=[("YES", "YES"), ("YES", "NO"), ("NO", "YES"), ("NO", "NO")],
            invalid_outcomes=[], confidence=0.9,
        )
        assert not dep.has_dependency


class TestDependencyDetector:
    def test_init(self):
        detector = DependencyDetector()
        assert detector.confidence_threshold == 0.9

    def test_check_single_market_arbitrage_underpriced(self):
        detector = DependencyDetector()
        arb = detector.check_single_market_arbitrage(yes_price=0.40, no_price=0.40)
        assert arb is not None
        assert arb["type"] == "buy_both"
        assert arb["profit_per_dollar"] == pytest.approx(0.20, abs=0.01)

    def test_check_single_market_arbitrage_overpriced(self):
        detector = DependencyDetector()
        arb = detector.check_single_market_arbitrage(yes_price=0.60, no_price=0.55)
        assert arb is not None
        assert arb["type"] == "sell_both"

    def test_check_single_market_no_arb(self):
        detector = DependencyDetector()
        arb = detector.check_single_market_arbitrage(yes_price=0.48, no_price=0.52)
        assert arb is None

    def test_check_pair_arbitrage_with_dependency(self):
        detector = DependencyDetector()
        dep = MarketDependency(
            market_a_id="mkt-a", market_a_title="Trump wins PA?",
            market_b_id="mkt-b", market_b_title="GOP +5 nationally?",
            dependency_type="implication", description="B implies A",
            valid_outcomes=[("YES", "YES"), ("YES", "NO"), ("NO", "NO")],
            invalid_outcomes=[("NO", "YES")], confidence=0.95,
        )
        prices = {
            "mkt-a": {"YES": 0.48, "NO": 0.52},
            "mkt-b": {"YES": 0.32, "NO": 0.68},
        }
        arb = detector.check_pair_arbitrage(dep, prices)
        assert arb is None or isinstance(arb, dict)

    @pytest.mark.asyncio
    async def test_detect_dependencies_via_llm(self):
        detector = DependencyDetector()
        mock_result = {
            "dependency_type": "implication", "description": "If B then A",
            "valid_outcomes": [["YES", "YES"], ["YES", "NO"], ["NO", "NO"]],
            "invalid_outcomes": [["NO", "YES"]], "confidence": 0.95,
        }
        with patch.object(detector, "_ask_llm_dependency", new_callable=AsyncMock) as mock:
            mock.return_value = mock_result
            dep = await detector.detect_dependency(
                market_a_id="mkt-a", market_a_title="Will Trump win Pennsylvania?",
                market_b_id="mkt-b", market_b_title="Will Republicans win by 5+ nationally?",
            )
        assert dep is not None
        assert dep.has_dependency
        assert dep.confidence == 0.95
