# tests/feeds/test_polymarket.py
import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch

from src.feeds.polymarket import (
    PolymarketFeed,
    PolymarketUserFeed,
    OrderBookUpdate,
    UserTradeEvent,
)


class TestOrderBookUpdate:
    def test_parse_price_update(self):
        raw = {
            "event_type": "price_change",
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

    def test_parse_book_snapshot(self):
        raw = {
            "event_type": "book",
            "asset_id": "token123",
            "last_trade_price": "0.49",
            "bids": [{"price": "0.49", "size": "100"}],
            "asks": [{"price": "0.51", "size": "50"}],
            "timestamp": 1234567892,
        }
        update = OrderBookUpdate.from_raw(raw)
        assert update.event_type == "book"
        assert update.market_id == "token123"
        assert update.price == 0.49


class TestPolymarketFeed:
    def test_initialization(self):
        feed = PolymarketFeed()
        assert feed.is_connected is False
        assert feed._local_orderbook == {}
        assert feed._token_map == {}

    @pytest.mark.asyncio
    async def test_subscribe_market_with_token_map(self):
        feed = PolymarketFeed()
        feed._ws = AsyncMock()
        feed._connected = True

        await feed.subscribe_market(
            "cond_123",
            token_map={"Up": "token_up", "Down": "token_down"},
        )

        feed._ws.send.assert_called_once()
        call_args = json.loads(feed._ws.send.call_args[0][0])
        assert call_args["type"] == "MARKET"
        assert "token_up" in call_args["assets_ids"]
        assert "token_down" in call_args["assets_ids"]

    @pytest.mark.asyncio
    async def test_subscribe_market_without_token_map(self):
        """subscribe_market without token_map is a no-op."""
        feed = PolymarketFeed()
        feed._ws = AsyncMock()
        feed._connected = True

        await feed.subscribe_market("cond_123")

        feed._ws.send.assert_not_called()

    def test_get_best_prices_via_token_map(self):
        feed = PolymarketFeed()
        feed._token_map = {
            ("cond_123", "YES"): "token_yes",
            ("cond_123", "NO"): "token_no",
        }
        feed._local_orderbook = {
            "token_yes": {"bids": [(0.64, 100), (0.63, 200)], "asks": [(0.66, 150)]},
            "token_no": {"bids": [(0.34, 100)], "asks": [(0.36, 150)]},
        }
        feed._best_cache = {
            "token_yes": (0.64, 100, 0.66, 150),
            "token_no": (0.34, 100, 0.36, 150),
        }

        best_bid, best_ask = feed.get_best_prices("cond_123", "YES")
        assert best_bid == 0.64
        assert best_ask == 0.66

        best_bid, best_ask = feed.get_best_prices("cond_123", "NO")
        assert best_bid == 0.34
        assert best_ask == 0.36

    def test_get_best_levels_via_token_map(self):
        feed = PolymarketFeed()
        feed._token_map = {("cond_123", "Up"): "token_up"}
        feed._local_orderbook = {
            "token_up": {"bids": [(0.64, 100), (0.63, 200)], "asks": [(0.66, 150)]},
        }
        feed._best_cache = {"token_up": (0.64, 100, 0.66, 150)}

        best_bid, bid_size, best_ask, ask_size = feed.get_best_levels("cond_123", "Up")
        assert best_bid == 0.64
        assert bid_size == 100
        assert best_ask == 0.66
        assert ask_size == 150

    def test_calculate_implied_probability(self):
        feed = PolymarketFeed()
        prob = feed.calculate_implied_probability(0.64, 0.66)
        assert prob == 0.65

    def test_merge_levels_add_new(self):
        feed = PolymarketFeed()
        existing = [(0.64, 100.0), (0.63, 200.0)]
        updates = [(0.65, 50.0)]
        result = feed._merge_levels(existing, updates, descending=True)
        assert (0.65, 50.0) in result
        assert len(result) == 3

    def test_merge_levels_remove_zero_size(self):
        feed = PolymarketFeed()
        existing = [(0.64, 100.0), (0.63, 200.0)]
        updates = [(0.64, 0.0)]
        result = feed._merge_levels(existing, updates, descending=True)
        assert (0.64, 100.0) not in result
        assert len(result) == 1

    def test_get_best_prices_unknown_market(self):
        feed = PolymarketFeed()
        best_bid, best_ask = feed.get_best_prices("unknown", "YES")
        assert best_bid is None
        assert best_ask is None

    def test_handle_book_snapshot(self):
        """Test that book snapshots update the local orderbook."""
        feed = PolymarketFeed()
        feed._subscribed_tokens.add("token_123")
        feed._local_orderbook["token_123"] = {"bids": [], "asks": []}

        item = {
            "event_type": "book",
            "asset_id": "token_123",
            "market": "0xcond",
            "bids": [
                {"price": "0.49", "size": "100"},
                {"price": "0.48", "size": "200"},
            ],
            "asks": [
                {"price": "0.51", "size": "150"},
                {"price": "0.52", "size": "50"},
            ],
        }
        feed._handle_book_snapshot(item)

        book = feed._local_orderbook["token_123"]
        # Bids sorted descending (best first)
        assert book["bids"][0] == (0.49, 100.0)
        assert book["bids"][1] == (0.48, 200.0)
        # Asks sorted ascending (best first)
        assert book["asks"][0] == (0.51, 150.0)
        assert book["asks"][1] == (0.52, 50.0)

    def test_handle_price_change(self):
        """Test that price_change messages incrementally update the book."""
        feed = PolymarketFeed()
        feed._local_orderbook["token_123"] = {
            "bids": [(0.49, 100.0)],
            "asks": [(0.51, 150.0)],
        }

        data = {
            "event_type": "price_change",
            "market": "0xcond",
            "price_changes": [
                {"asset_id": "token_123", "price": "0.50", "size": "200", "side": "BUY"},
                {"asset_id": "token_123", "price": "0.51", "size": "0", "side": "SELL"},
            ],
        }
        feed._handle_price_change(data)

        book = feed._local_orderbook["token_123"]
        assert (0.50, 200.0) in book["bids"]
        assert len(book["asks"]) == 0  # removed the only ask

    @pytest.mark.asyncio
    async def test_unsubscribe_market_cleans_up(self):
        feed = PolymarketFeed()
        feed._ws = AsyncMock()
        feed._connected = True
        feed._token_map = {
            ("cond_1", "Up"): "tok_up",
            ("cond_1", "Down"): "tok_down",
            ("cond_2", "Up"): "tok_other",
        }
        feed._subscribed_tokens = {"tok_up", "tok_down", "tok_other"}
        feed._local_orderbook = {
            "tok_up": {"bids": [], "asks": []},
            "tok_down": {"bids": [], "asks": []},
            "tok_other": {"bids": [], "asks": []},
        }
        feed._subscriptions = {("prediction", "cond_1"), ("prediction", "cond_2")}

        await feed.unsubscribe_market("cond_1")

        assert ("cond_1", "Up") not in feed._token_map
        assert ("cond_1", "Down") not in feed._token_map
        assert ("cond_2", "Up") in feed._token_map
        assert "tok_up" not in feed._subscribed_tokens
        assert "tok_down" not in feed._subscribed_tokens
        assert "tok_other" in feed._subscribed_tokens
        assert ("prediction", "cond_1") not in feed._subscriptions
        # Should have sent unsubscribe message
        feed._ws.send.assert_called_once()
        unsub_msg = json.loads(feed._ws.send.call_args[0][0])
        assert unsub_msg["operation"] == "unsubscribe"
        assert "tok_up" in unsub_msg["assets_ids"]
        assert "tok_down" in unsub_msg["assets_ids"]

    def test_get_market_prices(self):
        feed = PolymarketFeed()
        feed._token_map = {
            ("cond_1", "Up"): "tok_up",
            ("cond_1", "Down"): "tok_down",
        }
        feed._local_orderbook = {
            "tok_up": {"bids": [(0.49, 100)], "asks": [(0.51, 50)]},
            "tok_down": {"bids": [(0.48, 80)], "asks": [(0.52, 60)]},
        }
        feed._best_cache = {
            "tok_up": (0.49, 100, 0.51, 50),
            "tok_down": (0.48, 80, 0.52, 60),
        }

        prices = feed.get_market_prices("cond_1")
        assert "Up" in prices
        assert "Down" in prices
        assert prices["Up"] == (0.49, 0.51)
        assert prices["Down"] == (0.48, 0.52)

    @pytest.mark.asyncio
    async def test_process_message_list(self):
        """Test that list messages are processed as book snapshots."""
        feed = PolymarketFeed()
        feed._subscribed_tokens = {"tok_a", "tok_b"}
        feed._local_orderbook = {
            "tok_a": {"bids": [], "asks": []},
            "tok_b": {"bids": [], "asks": []},
        }

        msg = [
            {
                "event_type": "book",
                "asset_id": "tok_a",
                "bids": [{"price": "0.49", "size": "100"}],
                "asks": [{"price": "0.51", "size": "50"}],
            },
            {
                "event_type": "book",
                "asset_id": "tok_b",
                "bids": [{"price": "0.30", "size": "80"}],
                "asks": [{"price": "0.70", "size": "60"}],
            },
        ]
        await feed._process_message(msg)

        assert feed._local_orderbook["tok_a"]["bids"][0] == (0.49, 100.0)
        assert feed._local_orderbook["tok_b"]["asks"][0] == (0.70, 60.0)

    @pytest.mark.asyncio
    async def test_process_message_dict(self):
        """Test that dict messages with price_change are processed."""
        feed = PolymarketFeed()
        feed._local_orderbook = {
            "tok_a": {"bids": [(0.49, 100.0)], "asks": [(0.51, 50.0)]},
        }

        msg = {
            "event_type": "price_change",
            "market": "0xcond",
            "price_changes": [
                {"asset_id": "tok_a", "price": "0.50", "size": "200", "side": "BUY"},
            ],
        }
        await feed._process_message(msg)

        assert (0.50, 200.0) in feed._local_orderbook["tok_a"]["bids"]


class TestUserTradeEvent:
    def test_from_raw_taker(self):
        raw = {
            "event_type": "trade",
            "taker_order_id": "0xabc123",
            "market": "0xcond_1",
            "asset_id": "token_up",
            "side": "BUY",
            "price": "0.49",
            "size": "100",
            "status": "MATCHED",
            "timestamp": "1672290701",
        }
        evt = UserTradeEvent.from_raw(raw)
        assert evt.order_id == "0xabc123"
        assert evt.market == "0xcond_1"
        assert evt.asset_id == "token_up"
        assert evt.side == "BUY"
        assert evt.price == 0.49
        assert evt.size == 100.0
        assert evt.status == "MATCHED"

    def test_from_raw_maker_fallback(self):
        raw = {
            "event_type": "trade",
            "taker_order_id": "",
            "maker_orders": [{"order_id": "0xmaker1"}],
            "market": "0xcond_2",
            "asset_id": "token_down",
            "side": "SELL",
            "price": "0.51",
            "size": "50",
            "status": "CONFIRMED",
            "timestamp": "1672290800",
        }
        evt = UserTradeEvent.from_raw(raw)
        assert evt.order_id == "0xmaker1"
        assert evt.status == "CONFIRMED"


class TestPolymarketUserFeed:
    def test_initialization(self):
        feed = PolymarketUserFeed("key", "secret", "pass")
        assert feed.is_connected is False
        assert feed._subscribed_markets == set()
        assert feed.fills.empty()

    @pytest.mark.asyncio
    async def test_resubscribe_all_includes_auth(self):
        feed = PolymarketUserFeed("mykey", "mysecret", "mypass")
        feed._ws = AsyncMock()
        feed._connected = True
        feed._subscribed_markets = {"0xcond_1", "0xcond_2"}

        await feed._resubscribe_all()

        feed._ws.send.assert_called_once()
        msg = json.loads(feed._ws.send.call_args[0][0])
        assert msg["type"] == "user"
        assert msg["auth"]["apiKey"] == "mykey"
        assert msg["auth"]["secret"] == "mysecret"
        assert msg["auth"]["passphrase"] == "mypass"
        assert set(msg["markets"]) == {"0xcond_1", "0xcond_2"}

    @pytest.mark.asyncio
    async def test_subscribe_markets_sends_plain_subscribe(self):
        feed = PolymarketUserFeed("key", "secret", "pass")
        feed._ws = AsyncMock()
        feed._connected = True

        await feed.subscribe_markets(["0xcond_1", "0xcond_2"])

        feed._ws.send.assert_called_once()
        msg = json.loads(feed._ws.send.call_args[0][0])
        assert "auth" not in msg
        assert msg["operation"] == "subscribe"
        assert set(msg["markets"]) == {"0xcond_1", "0xcond_2"}

    @pytest.mark.asyncio
    async def test_subscribe_markets_queued_when_disconnected(self):
        feed = PolymarketUserFeed("key", "secret", "pass")
        feed._connected = False

        await feed.subscribe_markets(["0xcond_1"])

        assert "0xcond_1" in feed._subscribed_markets

    @pytest.mark.asyncio
    async def test_subscribe_dedup(self):
        feed = PolymarketUserFeed("key", "secret", "pass")
        feed._ws = AsyncMock()
        feed._connected = True
        feed._subscribed_markets = {"0xcond_1"}

        await feed.subscribe_markets(["0xcond_1"])
        feed._ws.send.assert_not_called()

    def test_process_message_trade_matched(self):
        feed = PolymarketUserFeed("key", "secret", "pass")
        data = {
            "event_type": "trade",
            "taker_order_id": "0xabc",
            "market": "0xcond_1",
            "asset_id": "token_up",
            "side": "BUY",
            "price": "0.49",
            "size": "100",
            "status": "MATCHED",
            "timestamp": "1672290701",
        }
        feed._process_message(data)
        assert not feed.fills.empty()
        evt = feed.fills.get_nowait()
        assert evt.order_id == "0xabc"
        assert evt.status == "MATCHED"

    def test_process_message_trade_retrying_ignored(self):
        feed = PolymarketUserFeed("key", "secret", "pass")
        data = {
            "event_type": "trade",
            "taker_order_id": "0xabc",
            "market": "0xcond_1",
            "asset_id": "token_up",
            "side": "BUY",
            "price": "0.49",
            "size": "100",
            "status": "RETRYING",
            "timestamp": "1672290701",
        }
        feed._process_message(data)
        assert feed.fills.empty()

    def test_process_message_order_ignored(self):
        feed = PolymarketUserFeed("key", "secret", "pass")
        data = {
            "event_type": "order",
            "id": "0xord1",
            "type": "PLACEMENT",
        }
        feed._process_message(data)
        assert feed.fills.empty()

    @pytest.mark.asyncio
    async def test_unsubscribe_markets(self):
        feed = PolymarketUserFeed("key", "secret", "pass")
        feed._ws = AsyncMock()
        feed._connected = True
        feed._subscribed_markets = {"0xcond_1", "0xcond_2"}

        await feed.unsubscribe_markets(["0xcond_1"])

        feed._ws.send.assert_called_once()
        msg = json.loads(feed._ws.send.call_args[0][0])
        assert msg["operation"] == "unsubscribe"
        assert "0xcond_1" in msg["markets"]
        assert "0xcond_1" not in feed._subscribed_markets
        assert "0xcond_2" in feed._subscribed_markets
