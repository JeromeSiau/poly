"""Tests for contrarian NO bet scanner (NeverYES / DidiTrading strategy)."""
import pytest
from unittest.mock import MagicMock

from src.arb.no_bet_scanner import NoBetScanner, NoBetOpportunity


class TestNoBetOpportunity:
    def test_expected_return(self):
        # NeverYES style: buy NO at 0.50 on a hype market, true prob of NO ~0.65
        opp = NoBetOpportunity(
            market_id="mkt-001",
            token_id="tok-no-001",
            title="Will TOKEN FDV be above $1B at launch?",
            yes_price=0.50,
            no_price=0.50,
            estimated_no_probability=0.65,
            edge_pct=0.15,
            volume_24h=80000.0,
            liquidity=20000.0,
        )
        # EV = 0.65 * 0.50 - 0.35 * 0.50 = 0.325 - 0.175 = 0.15
        assert opp.expected_return == pytest.approx(0.15, abs=0.01)

    def test_is_valid(self):
        opp = NoBetOpportunity(
            market_id="mkt-001",
            token_id="tok-no-001",
            title="Will TOKEN hit $100 at TGE?",
            yes_price=0.55,
            no_price=0.45,
            estimated_no_probability=0.65,
            edge_pct=0.10,
            volume_24h=50000.0,
            liquidity=10000.0,
        )
        assert opp.is_valid

    def test_is_invalid_low_liquidity(self):
        opp = NoBetOpportunity(
            market_id="mkt-001",
            token_id="tok-no-001",
            title="Test",
            yes_price=0.50,
            no_price=0.50,
            estimated_no_probability=0.65,
            edge_pct=0.10,
            volume_24h=100.0,
            liquidity=50.0,
        )
        assert not opp.is_valid


class TestNoBetScanner:
    def test_init_defaults(self):
        scanner = NoBetScanner()
        assert scanner.min_yes_price == 0.35
        assert scanner.max_yes_price == 0.65
        assert scanner.min_liquidity == 1000.0

    def test_score_market_hype_sweet_spot(self):
        """FDV market at YES=0.50, high volume, short resolution = top score."""
        scanner = NoBetScanner()
        score = scanner.score_market(
            title="Will TOKEN FDV be above $500M at launch?",
            yes_price=0.50,
            volume_24h=200000.0,
            end_date_days=5,
        )
        # base 0.3 + sweet_spot 0.25 + hype_kw 0.25 + volume 0.10 + short 0.10 = 1.0
        assert score >= 0.9

    def test_score_market_moderate_hype(self):
        """Moderate hype keyword, edge of sweet spot."""
        scanner = NoBetScanner()
        score = scanner.score_market(
            title="Will the token airdrop happen before March?",
            yes_price=0.58,
            volume_24h=30000.0,
            end_date_days=60,
        )
        # base 0.3 + price 0.15 (0.40-0.60 band) + moderate_kw 0.10 = 0.55
        assert 0.4 < score < 0.7

    def test_score_market_no_hype_keywords(self):
        """No hype keywords, still in price range."""
        scanner = NoBetScanner()
        score = scanner.score_market(
            title="Will it rain in Paris on March 1?",
            yes_price=0.50,
            volume_24h=10000.0,
            end_date_days=30,
        )
        # base 0.3 + sweet_spot 0.25 + no_kw 0 + low_vol 0 + short 0.05 = 0.60
        assert 0.5 < score < 0.7

    def test_score_penalizes_long_resolution(self):
        scanner = NoBetScanner()
        score = scanner.score_market(
            title="Will TOKEN reach $1B by 2028?",
            yes_price=0.50,
            volume_24h=10000.0,
            end_date_days=365,
        )
        # base 0.3 + sweet_spot 0.25 + hype_kw 0.25 + long_penalty -0.10 = 0.70
        score_short = scanner.score_market(
            title="Will TOKEN reach $1B by 2028?",
            yes_price=0.50,
            volume_24h=10000.0,
            end_date_days=5,
        )
        assert score_short > score

    def test_filter_candidates_price_range(self):
        """Only markets with YES in [0.35, 0.65] pass the filter."""
        scanner = NoBetScanner()
        markets = [
            {"condition_id": "a", "title": "FDV above $1B?", "tokens": [
                {"outcome": "Yes", "price": 0.50},
                {"outcome": "No", "price": 0.50, "token_id": "tok-a"},
            ]},
            {"condition_id": "b", "title": "Near certainty", "tokens": [
                {"outcome": "Yes", "price": 0.92},
                {"outcome": "No", "price": 0.08, "token_id": "tok-b"},
            ]},
            {"condition_id": "c", "title": "Very unlikely", "tokens": [
                {"outcome": "Yes", "price": 0.05},
                {"outcome": "No", "price": 0.93, "token_id": "tok-c"},
            ]},
            {"condition_id": "d", "title": "Edge of range", "tokens": [
                {"outcome": "Yes", "price": 0.40},
                {"outcome": "No", "price": 0.58, "token_id": "tok-d"},
            ]},
        ]
        candidates = scanner.filter_candidates(markets)
        ids = [c["condition_id"] for c in candidates]
        assert "a" in ids  # 0.50 in range
        assert "d" in ids  # 0.40 in range
        assert "b" not in ids  # 0.92 too high
        assert "c" not in ids  # 0.05 too low
