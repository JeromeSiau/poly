"""Tests for Binance WebSocket price feed."""
import pytest
import json
from collections import deque
from unittest.mock import AsyncMock, MagicMock, patch

from src.feeds.binance import BinanceFeed, BinanceTick


class TestBinanceTick:
    def test_from_raw_trade(self):
        raw = {
            "e": "trade",
            "s": "BTCUSDT",
            "p": "65432.10",
            "q": "0.5",
            "T": 1707300000000,
        }
        tick = BinanceTick.from_raw(raw)
        assert tick.symbol == "BTCUSDT"
        assert tick.price == 65432.10
        assert tick.quantity == 0.5
        assert tick.timestamp == 1707300000.0

    def test_from_raw_missing_fields(self):
        tick = BinanceTick.from_raw({})
        assert tick.symbol == ""
        assert tick.price == 0.0


class TestBinanceFeed:
    def test_init_default_symbols(self):
        feed = BinanceFeed()
        assert "BTCUSDT" in feed.symbols
        assert "ETHUSDT" in feed.symbols
        assert "SOLUSDT" in feed.symbols

    def test_init_custom_symbols(self):
        feed = BinanceFeed(symbols=["XRPUSDT"])
        assert feed.symbols == ["XRPUSDT"]

    def test_fair_value_empty(self):
        feed = BinanceFeed()
        fv = feed.get_fair_value("BTCUSDT")
        assert fv is None

    def test_fair_value_with_trades(self):
        feed = BinanceFeed(fair_value_window=3)
        feed._recent_trades["BTCUSDT"] = deque([
            BinanceTick("BTCUSDT", 100.0, 1.0, 1.0),
            BinanceTick("BTCUSDT", 102.0, 2.0, 2.0),
            BinanceTick("BTCUSDT", 104.0, 1.0, 3.0),
        ], maxlen=3)
        fv = feed.get_fair_value("BTCUSDT")
        assert fv == pytest.approx(102.0)

    def test_price_direction_up(self):
        feed = BinanceFeed(fair_value_window=3)
        feed._recent_trades["BTCUSDT"] = deque([
            BinanceTick("BTCUSDT", 100.0, 1.0, 1.0),
            BinanceTick("BTCUSDT", 100.0, 1.0, 2.0),
            BinanceTick("BTCUSDT", 105.0, 1.0, 3.0),
        ], maxlen=3)
        direction = feed.get_price_direction("BTCUSDT")
        assert direction == "UP"

    def test_price_direction_down(self):
        feed = BinanceFeed(fair_value_window=3)
        feed._recent_trades["BTCUSDT"] = deque([
            BinanceTick("BTCUSDT", 105.0, 1.0, 1.0),
            BinanceTick("BTCUSDT", 105.0, 1.0, 2.0),
            BinanceTick("BTCUSDT", 100.0, 1.0, 3.0),
        ], maxlen=3)
        direction = feed.get_price_direction("BTCUSDT")
        assert direction == "DOWN"

    def test_price_direction_neutral(self):
        feed = BinanceFeed(fair_value_window=3)
        feed._recent_trades["BTCUSDT"] = deque([
            BinanceTick("BTCUSDT", 100.0, 1.0, 1.0),
            BinanceTick("BTCUSDT", 100.0, 1.0, 2.0),
        ], maxlen=3)
        direction = feed.get_price_direction("BTCUSDT")
        assert direction == "NEUTRAL"
