"""Tests for market observer."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from src.paper_trading.market_observer import MarketObserver, PriceCapture


class TestMarketObserver:
    @pytest.fixture
    def observer(self):
        return MarketObserver()

    @pytest.mark.asyncio
    async def test_capture_price_returns_price_capture(self, observer):
        """capture_price returns PriceCapture object."""
        with patch.object(observer, "_fetch_price", return_value=0.58):
            capture = await observer.capture_price("market_123", "TeamA")

        assert isinstance(capture, PriceCapture)
        assert capture.market_id == "market_123"
        assert capture.price == 0.58

    @pytest.mark.asyncio
    async def test_schedule_followups_captures_at_intervals(self, observer):
        """schedule_followups captures prices at 30s, 60s, 120s."""
        prices = [0.58, 0.62, 0.65, 0.68]
        call_count = 0

        async def mock_fetch(*args):
            nonlocal call_count
            price = prices[call_count]
            call_count += 1
            return price

        with patch.object(observer, "_fetch_price", side_effect=mock_fetch):
            # Use very short intervals for testing
            captures = await observer.capture_with_followups(
                "market_123",
                "TeamA",
                intervals=[0.01, 0.02, 0.03],  # 10ms, 20ms, 30ms for testing
            )

        assert len(captures) == 4
        assert captures[0].price == 0.58
        assert captures[1].price == 0.62
        assert captures[2].price == 0.65
        assert captures[3].price == 0.68

    @pytest.mark.asyncio
    async def test_handles_fetch_failure(self, observer):
        """Gracefully handles API failures."""
        with patch.object(observer, "_fetch_price", side_effect=Exception("API error")):
            capture = await observer.capture_price("market_123", "TeamA")

        assert capture.price is None
        assert capture.error is not None


class TestPriceCapture:
    def test_price_capture_has_timestamp(self):
        capture = PriceCapture(
            market_id="123",
            outcome="TeamA",
            price=0.55,
            timestamp=datetime.utcnow(),
        )
        assert capture.timestamp is not None

    def test_price_capture_optional_fields(self):
        capture = PriceCapture(
            market_id="123",
            outcome="TeamA",
            price=None,
            timestamp=datetime.utcnow(),
            error="Fetch failed",
        )
        assert capture.error == "Fetch failed"
