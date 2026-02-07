"""Tests for LLM-powered market screener."""
import pytest
from src.screening.market_screener import MarketScreener, ScreenedMarket


class TestScreenedMarket:
    def test_alpha_score(self):
        m = ScreenedMarket(
            market_id="mkt-001", title="Will X happen?", volume_24h=100000,
            liquidity=50000, price_yes=0.50, category="politics",
            end_date="2026-03-01", alpha_score=0.75,
        )
        assert m.alpha_score == 0.75

    def test_is_interesting_high_score(self):
        m = ScreenedMarket(
            market_id="mkt-001", title="Test", volume_24h=100000,
            liquidity=50000, price_yes=0.50, category="politics",
            end_date="2026-03-01", alpha_score=0.80,
        )
        assert m.is_interesting

    def test_is_not_interesting_low_score(self):
        m = ScreenedMarket(
            market_id="mkt-001", title="Test", volume_24h=100000,
            liquidity=50000, price_yes=0.50, category="politics",
            end_date="2026-03-01", alpha_score=0.30,
        )
        assert not m.is_interesting


class TestMarketScreener:
    def test_init(self):
        screener = MarketScreener()
        assert screener.min_alpha_score == 0.6

    def test_compute_alpha_score_high_volume_midprice(self):
        screener = MarketScreener()
        score = screener.compute_alpha_score(
            volume_24h=200000, liquidity=100000, price_yes=0.50,
            category="politics", hours_to_resolution=48,
        )
        assert score > 0.6

    def test_compute_alpha_score_low_volume(self):
        screener = MarketScreener()
        score = screener.compute_alpha_score(
            volume_24h=500, liquidity=200, price_yes=0.50,
            category="crypto", hours_to_resolution=2,
        )
        assert score < 0.5

    def test_compute_alpha_score_extreme_price(self):
        screener = MarketScreener()
        score = screener.compute_alpha_score(
            volume_24h=100000, liquidity=50000, price_yes=0.98,
            category="politics", hours_to_resolution=48,
        )
        assert score < 0.5

    def test_screen_markets(self):
        screener = MarketScreener(min_alpha_score=0.5)
        markets = [
            {
                "condition_id": "a", "title": "Interesting politics market?",
                "volume_24h": 200000, "liquidity": 100000,
                "tokens": [{"outcome": "Yes", "price": 0.50}],
                "category": "politics", "end_date": "2026-03-01",
            },
            {
                "condition_id": "b", "title": "Dead market?",
                "volume_24h": 100, "liquidity": 50,
                "tokens": [{"outcome": "Yes", "price": 0.50}],
                "category": "other", "end_date": "2026-03-01",
            },
        ]
        results = screener.screen_markets(markets)
        interesting = [r for r in results if r.is_interesting]
        assert len(interesting) >= 1
        assert interesting[0].market_id == "a"
