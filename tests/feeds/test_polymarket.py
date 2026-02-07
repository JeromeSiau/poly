# tests/feeds/test_polymarket.py
import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch

from src.feeds.polymarket import PolymarketFeed, OrderBookUpdate


class TestOrderBookUpdate:
    def test_parse_price_update(self):
        raw = {
            "type": "price_change",
            "market_id": "0x123abc",
            "outcome": "YES",
            "price": 0.65,
            "timestamp": 1234567890
        }

        update = OrderBookUpdate.from_raw(raw)

        assert update.market_id == "0x123abc"
        assert update.outcome == "YES"
        assert update.price == 0.65

    def test_parse_trade_update(self):
        raw = {
            "type": "trade",
            "market_id": "0x123abc",
            "outcome": "YES",
            "price": 0.66,
            "size": 100.0,
            "side": "BUY",
            "timestamp": 1234567891
        }

        update = OrderBookUpdate.from_raw(raw)

        assert update.event_type == "trade"
        assert update.data["size"] == 100.0
        assert update.data["side"] == "BUY"


class TestPolymarketFeed:
    def test_initialization(self):
        feed = PolymarketFeed()
        assert feed.is_connected is False
        assert feed._local_orderbook == {}

    @pytest.mark.asyncio
    async def test_subscribe_market(self):
        feed = PolymarketFeed()
        feed._ws = AsyncMock()
        feed._connected = True

        await feed.subscribe_market("0x123abc")

        feed._ws.send.assert_called_once()
        call_args = json.loads(feed._ws.send.call_args[0][0])
        assert call_args["type"] == "subscribe"
        assert "0x123abc" in call_args["markets"]

    def test_get_best_price(self):
        feed = PolymarketFeed()
        feed._local_orderbook = {
            "0x123abc": {
                "YES": {"bids": [(0.64, 100), (0.63, 200)], "asks": [(0.66, 150)]},
                "NO": {"bids": [(0.34, 100)], "asks": [(0.36, 150)]}
            }
        }

        best_bid, best_ask = feed.get_best_prices("0x123abc", "YES")

        assert best_bid == 0.64
        assert best_ask == 0.66

    def test_get_best_levels(self):
        feed = PolymarketFeed()
        feed._local_orderbook = {
            "0x123abc": {
                "YES": {"bids": [(0.64, 100), (0.63, 200)], "asks": [(0.66, 150)]},
            }
        }

        best_bid, bid_size, best_ask, ask_size = feed.get_best_levels("0x123abc", "YES")

        assert best_bid == 0.64
        assert bid_size == 100
        assert best_ask == 0.66
        assert ask_size == 150

    def test_calculate_implied_probability(self):
        feed = PolymarketFeed()

        # Mid price of 0.65 = 65% implied probability
        prob = feed.calculate_implied_probability(0.64, 0.66)

        assert prob == 0.65

    def test_merge_levels_add_new(self):
        """Test adding a new price level."""
        feed = PolymarketFeed()
        existing = [(0.64, 100.0), (0.63, 200.0)]
        updates = [(0.65, 50.0)]  # New level
        result = feed._merge_levels(existing, updates, descending=True)
        assert (0.65, 50.0) in result
        assert len(result) == 3

    def test_merge_levels_remove_zero_size(self):
        """Test removing a level with size 0."""
        feed = PolymarketFeed()
        existing = [(0.64, 100.0), (0.63, 200.0)]
        updates = [(0.64, 0.0)]  # Remove this level
        result = feed._merge_levels(existing, updates, descending=True)
        assert (0.64, 100.0) not in result
        assert len(result) == 1

    def test_get_best_prices_unknown_market(self):
        """Test get_best_prices returns (None, None) for unknown market."""
        feed = PolymarketFeed()
        best_bid, best_ask = feed.get_best_prices("unknown", "YES")
        assert best_bid is None
        assert best_ask is None
