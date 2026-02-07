"""Tests for crypto market mapper — links CEX symbols to Polymarket 15-min markets."""
import pytest
from unittest.mock import AsyncMock

from src.realtime.crypto_mapper import CryptoMarketMapper


class TestCryptoMarketMapper:
    def test_map_symbol_to_market(self):
        mapper = CryptoMarketMapper()
        mapper._active_markets = {
            "btc-up-15min-001": {
                "condition_id": "btc-up-15min-001",
                "title": "Will Bitcoin go up in the next 15 minutes?",
                "tokens": [
                    {"token_id": "tok-yes-001", "outcome": "Yes"},
                    {"token_id": "tok-no-001", "outcome": "No"},
                ],
            }
        }
        mapper._symbol_to_markets = {"BTCUSDT": ["btc-up-15min-001"]}

        result = mapper.get_active_market("BTCUSDT")
        assert result is not None
        assert result["condition_id"] == "btc-up-15min-001"

    def test_map_symbol_no_market(self):
        mapper = CryptoMarketMapper()
        result = mapper.get_active_market("DOGEUSDT")
        assert result is None

    def test_get_token_for_direction_up(self):
        mapper = CryptoMarketMapper()
        market = {
            "tokens": [
                {"token_id": "tok-yes", "outcome": "Yes"},
                {"token_id": "tok-no", "outcome": "No"},
            ]
        }
        token_id, outcome = mapper.get_token_for_direction(market, "UP")
        assert token_id == "tok-yes"
        assert outcome == "Yes"

    def test_get_token_for_direction_down(self):
        mapper = CryptoMarketMapper()
        market = {
            "tokens": [
                {"token_id": "tok-yes", "outcome": "Yes"},
                {"token_id": "tok-no", "outcome": "No"},
            ]
        }
        token_id, outcome = mapper.get_token_for_direction(market, "DOWN")
        assert token_id == "tok-no"
        assert outcome == "No"

    def test_get_token_for_neutral_returns_none(self):
        mapper = CryptoMarketMapper()
        market = {"tokens": []}
        result = mapper.get_token_for_direction(market, "NEUTRAL")
        assert result is None

    @pytest.mark.asyncio
    async def test_sync_markets_matches_keywords(self):
        mapper = CryptoMarketMapper()
        mock_feed = AsyncMock()
        mock_feed.get_markets.return_value = [
            {
                "title": "Will Bitcoin go up in the next 15 minutes?",
                "condition_id": "btc-123",
                "tokens": [
                    {"token_id": "tok-yes", "outcome": "Yes"},
                    {"token_id": "tok-no", "outcome": "No"},
                ],
            },
            {
                "title": "Will Ethereum price increase?",
                "condition_id": "eth-456",
                "tokens": [
                    {"token_id": "tok-yes-2", "outcome": "Yes"},
                    {"token_id": "tok-no-2", "outcome": "No"},
                ],
            },
        ]

        count = await mapper.sync_markets(mock_feed)
        assert count == 2
        assert "BTCUSDT" in mapper._symbol_to_markets
        assert "ETHUSDT" in mapper._symbol_to_markets
        assert mapper.get_active_market("BTCUSDT")["condition_id"] == "btc-123"

    @pytest.mark.asyncio
    async def test_sync_markets_deduplicates(self):
        mapper = CryptoMarketMapper()
        mock_feed = AsyncMock()
        mock_feed.get_markets.return_value = [
            {"title": "Bitcoin up?", "condition_id": "btc-123", "tokens": []},
        ]
        count1 = await mapper.sync_markets(mock_feed)
        count2 = await mapper.sync_markets(mock_feed)
        assert count1 == 1
        assert count2 == 0
        assert len(mapper._symbol_to_markets["BTCUSDT"]) == 1
