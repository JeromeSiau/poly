import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.td_maker.state import MarketRegistry
from src.td_maker.discovery import MarketDiscovery


@pytest.mark.asyncio
async def test_discover_registers_new_markets():
    poly_feed = MagicMock()
    poly_feed.subscribe_batch = AsyncMock()
    chainlink = MagicMock()
    chainlink.get_price.return_value = 95000.0

    config = MagicMock()
    config.symbols = ["BTCUSDT"]

    discovery = MarketDiscovery(poly_feed, chainlink, config)
    registry = MarketRegistry()

    fake_market = MagicMock()
    fake_market.condition_id = "cid1"
    fake_market.slug = "btc-up-15m-1771079400"
    fake_market.token_ids = {"Up": "tok_up", "Down": "tok_dn"}

    with patch("src.td_maker.discovery.fetch_crypto_markets",
               new=AsyncMock(return_value=[fake_market])):
        await discovery.discover(registry)

    assert registry.get("cid1") is not None


@pytest.mark.asyncio
async def test_discover_skips_already_known():
    poly_feed = MagicMock()
    poly_feed.subscribe_batch = AsyncMock()
    chainlink = MagicMock()
    chainlink.get_price.return_value = 95000.0

    config = MagicMock()
    config.symbols = ["BTCUSDT"]

    discovery = MarketDiscovery(poly_feed, chainlink, config)
    registry = MarketRegistry()

    fake_market = MagicMock()
    fake_market.condition_id = "cid1"
    fake_market.slug = "btc-up-15m-1771079400"
    fake_market.token_ids = {"Up": "tok_up"}

    with patch("src.td_maker.discovery.fetch_crypto_markets",
               new=AsyncMock(return_value=[fake_market])):
        await discovery.discover(registry)
        initial_market = registry.get("cid1")
        await discovery.discover(registry)

    assert registry.get("cid1") is initial_market  # same object, not re-registered
