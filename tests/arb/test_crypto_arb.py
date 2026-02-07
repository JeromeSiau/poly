"""Tests for CryptoArbEngine — exploits CEX-to-Polymarket lag on 15-min markets."""
import pytest
import time
from unittest.mock import MagicMock

from src.arb.crypto_arb import CryptoArbEngine, CryptoArbOpportunity


class TestCryptoArbOpportunity:
    def test_opportunity_valid(self):
        opp = CryptoArbOpportunity(
            symbol="BTCUSDT",
            market_id="mkt-001",
            token_id="tok-001",
            side="BUY",
            outcome="Yes",
            polymarket_price=0.45,
            fair_value_price=0.55,
            cex_direction="UP",
            edge_pct=0.10,
            timestamp=time.time(),
        )
        assert opp.is_valid

    def test_opportunity_edge_too_low(self):
        opp = CryptoArbOpportunity(
            symbol="BTCUSDT",
            market_id="mkt-001",
            token_id="tok-001",
            side="BUY",
            outcome="Yes",
            polymarket_price=0.50,
            fair_value_price=0.51,
            cex_direction="UP",
            edge_pct=0.01,
            timestamp=time.time(),
        )
        assert not opp.is_valid

    def test_opportunity_stale(self):
        opp = CryptoArbOpportunity(
            symbol="BTCUSDT",
            market_id="mkt-001",
            token_id="tok-001",
            side="BUY",
            outcome="Yes",
            polymarket_price=0.45,
            fair_value_price=0.62,
            cex_direction="UP",
            edge_pct=0.17,
            timestamp=time.time() - 120,  # 2 minutes old
        )
        assert not opp.is_valid


class TestCryptoArbEngine:
    def test_init(self):
        engine = CryptoArbEngine()
        assert engine.min_edge_pct == 0.02
        assert engine.stale_seconds == 45.0

    def test_estimate_fair_price_up(self):
        engine = CryptoArbEngine()
        fair = engine.estimate_fair_price(
            direction="UP",
            current_polymarket_price=0.50,
            cex_pct_move=0.005,
        )
        assert fair > 0.50

    def test_estimate_fair_price_down(self):
        engine = CryptoArbEngine()
        fair = engine.estimate_fair_price(
            direction="DOWN",
            current_polymarket_price=0.50,
            cex_pct_move=-0.005,
        )
        assert fair < 0.50

    def test_estimate_fair_price_clamped(self):
        engine = CryptoArbEngine()
        fair = engine.estimate_fair_price(
            direction="UP",
            current_polymarket_price=0.98,
            cex_pct_move=0.05,
        )
        assert fair <= 0.99

    def test_evaluate_opportunity_with_edge(self):
        binance_feed = MagicMock()
        binance_feed.get_fair_value.return_value = 65500.0
        binance_feed.get_price_direction.return_value = "UP"
        binance_feed.get_recent_trades.return_value = [
            MagicMock(price=65000.0), MagicMock(price=65500.0)
        ]

        polymarket_feed = MagicMock()
        polymarket_feed.get_best_prices.return_value = (0.45, 0.47)

        crypto_mapper = MagicMock()
        crypto_mapper.get_active_market.return_value = {
            "condition_id": "mkt-001",
            "tokens": [
                {"token_id": "tok-yes", "outcome": "Yes"},
                {"token_id": "tok-no", "outcome": "No"},
            ],
        }
        crypto_mapper.get_token_for_direction.return_value = ("tok-yes", "Yes")

        engine = CryptoArbEngine(
            binance_feed=binance_feed,
            polymarket_feed=polymarket_feed,
            crypto_mapper=crypto_mapper,
        )

        opp = engine.evaluate_opportunity("BTCUSDT")
        assert opp is not None
        assert opp.side == "BUY"
        assert opp.outcome == "Yes"
        assert opp.edge_pct > 0

    def test_evaluate_no_opportunity_neutral(self):
        binance_feed = MagicMock()
        binance_feed.get_price_direction.return_value = "NEUTRAL"

        engine = CryptoArbEngine(binance_feed=binance_feed)
        opp = engine.evaluate_opportunity("BTCUSDT")
        assert opp is None
