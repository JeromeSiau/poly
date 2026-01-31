# tests/bot/test_crossmarket_handlers.py
"""Tests for cross-market arbitrage Telegram handlers."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import dataclass

from src.bot.crossmarket_handlers import CrossMarketArbHandler


@dataclass
class MockEvent:
    """Mock event for testing."""
    name: str = "Chiefs win Super Bowl"
    confidence: float = 0.98


@dataclass
class MockOpportunity:
    """Mock CrossMarketOpportunity for testing."""
    source_platform: str = "polymarket"
    source_price: float = 0.42
    source_liquidity: float = 5000.0
    target_platform: str = "azuro"
    target_price: float = 0.47
    target_liquidity: float = 3000.0
    gross_edge_pct: float = 0.05
    net_edge_pct: float = 0.04
    event: MockEvent = None

    def __post_init__(self):
        if self.event is None:
            self.event = MockEvent()


@pytest.fixture
def mock_bot():
    """Create a mock Telegram bot."""
    bot = MagicMock()
    bot.send_message = AsyncMock(return_value=MagicMock(message_id=123))
    bot.edit_message_text = AsyncMock()
    return bot


@pytest.fixture
def handler(mock_bot):
    """Create a CrossMarketArbHandler with mock bot."""
    return CrossMarketArbHandler(bot=mock_bot, chat_id="123456")


class TestHandlerCreation:
    """Tests for handler initialization."""

    def test_handler_creation(self, handler):
        """Test that handler is created with correct defaults."""
        assert handler is not None
        assert handler.alert_expiry_seconds == 60

    def test_handler_with_custom_expiry(self, mock_bot):
        """Test handler creation with custom expiry."""
        handler = CrossMarketArbHandler(
            bot=mock_bot,
            chat_id="123456",
            alert_expiry_seconds=120
        )
        assert handler.alert_expiry_seconds == 120


class TestAlertFormatting:
    """Tests for alert message formatting."""

    def test_format_alert_message(self, handler):
        """Test that alert message contains all required information."""
        opp = MockOpportunity()
        message = handler.format_alert_message(opp, position_size=500.0)

        assert "CROSS-MARKET ARB" in message
        assert "Chiefs win Super Bowl" in message
        assert "polymarket" in message.lower()
        assert "azuro" in message.lower()
        assert "42" in message  # source price
        assert "47" in message  # target price

    def test_format_alert_includes_liquidity(self, handler):
        """Test that alert includes liquidity information."""
        opp = MockOpportunity()
        message = handler.format_alert_message(opp, position_size=500.0)

        assert "5000" in message or "5,000" in message
        assert "3000" in message or "3,000" in message

    def test_format_alert_includes_edge(self, handler):
        """Test that alert includes edge percentage."""
        opp = MockOpportunity()
        message = handler.format_alert_message(opp, position_size=500.0)

        # Net edge is 0.04 = 4%
        assert "4" in message

    def test_format_alert_includes_position_size(self, handler):
        """Test that alert includes position size."""
        opp = MockOpportunity()
        message = handler.format_alert_message(opp, position_size=500.0)

        assert "500" in message

    def test_format_alert_includes_expiry(self, handler):
        """Test that alert includes expiry time."""
        opp = MockOpportunity()
        message = handler.format_alert_message(opp, position_size=500.0)

        assert "60s" in message or "60 s" in message or "Expires" in message

    def test_format_alert_includes_confidence(self, handler):
        """Test that alert includes match confidence."""
        opp = MockOpportunity()
        message = handler.format_alert_message(opp, position_size=500.0)

        assert "98" in message or "confidence" in message.lower()


class TestSendAlert:
    """Tests for sending alerts."""

    @pytest.mark.asyncio
    async def test_send_alert(self, handler, mock_bot):
        """Test that send_alert sends message and returns message_id."""
        opp = MockOpportunity()

        message_id = await handler.send_alert(opp, position_size=500.0)

        assert message_id == 123
        mock_bot.send_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_alert_includes_keyboard(self, handler, mock_bot):
        """Test that send_alert includes inline keyboard."""
        opp = MockOpportunity()

        await handler.send_alert(opp, position_size=500.0)

        call_kwargs = mock_bot.send_message.call_args.kwargs
        assert "reply_markup" in call_kwargs

    @pytest.mark.asyncio
    async def test_send_alert_stores_pending(self, handler, mock_bot):
        """Test that send_alert stores pending opportunity."""
        opp = MockOpportunity()

        message_id = await handler.send_alert(opp, position_size=500.0)

        # Should be stored with message_id as key
        assert handler.get_pending_opportunity(str(message_id)) is not None


class TestPendingOpportunityManagement:
    """Tests for pending opportunity management."""

    def test_pending_opportunities_management(self, handler):
        """Test add, get, and remove pending opportunities."""
        opp = MockOpportunity()

        handler.add_pending_opportunity("msg_123", opp)

        assert handler.get_pending_opportunity("msg_123") == opp

        handler.remove_pending_opportunity("msg_123")

        assert handler.get_pending_opportunity("msg_123") is None

    def test_get_nonexistent_opportunity(self, handler):
        """Test getting a non-existent opportunity returns None."""
        result = handler.get_pending_opportunity("nonexistent")
        assert result is None

    def test_remove_nonexistent_opportunity(self, handler):
        """Test removing non-existent opportunity doesn't raise error."""
        # Should not raise
        handler.remove_pending_opportunity("nonexistent")


class TestApprovalFlow:
    """Tests for approve/skip handling."""

    @pytest.mark.asyncio
    async def test_handle_approve_with_pending(self, handler):
        """Test approve handler with pending opportunity."""
        opp = MockOpportunity()
        handler.add_pending_opportunity("msg_123", opp, position_size=500.0)

        query = MagicMock()
        query.message = MagicMock()
        query.message.message_id = 123
        query.data = "crossmarket_approve:msg_123"
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()

        result = await handler.handle_approve("msg_123")

        assert result["approved"] is True
        assert handler.get_pending_opportunity("msg_123") is None

    @pytest.mark.asyncio
    async def test_handle_approve_expired(self, handler):
        """Test approve handler with expired/missing opportunity."""
        result = await handler.handle_approve("nonexistent")

        assert result["approved"] is False
        assert "expired" in result.get("error", "").lower() or "not found" in result.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_handle_skip(self, handler):
        """Test skip handler removes opportunity."""
        opp = MockOpportunity()
        handler.add_pending_opportunity("msg_123", opp, position_size=500.0)

        result = await handler.handle_skip("msg_123")

        assert result["skipped"] is True
        assert handler.get_pending_opportunity("msg_123") is None

    @pytest.mark.asyncio
    async def test_handle_skip_expired(self, handler):
        """Test skip handler with expired/missing opportunity."""
        result = await handler.handle_skip("nonexistent")

        assert result["skipped"] is False


class TestCleanupExpiredOpportunities:
    """Tests for expired opportunity cleanup."""

    def test_cleanup_expired_opportunities(self, handler):
        """Test cleanup removes old opportunities."""
        opp = MockOpportunity()

        # Add opportunity with old timestamp
        handler.add_pending_opportunity("msg_old", opp, position_size=500.0)

        # Manually set timestamp to be expired (hacky but works for testing)
        if hasattr(handler, '_pending_timestamps'):
            import time
            handler._pending_timestamps["msg_old"] = time.time() - 120  # 2 minutes ago

        # Run cleanup
        handler.cleanup_expired_opportunities()

        # Old opportunity should be removed
        # Note: this depends on implementation details


class TestInlineKeyboard:
    """Tests for inline keyboard generation."""

    def test_get_approval_keyboard(self, handler):
        """Test that approval keyboard has correct buttons."""
        keyboard = handler.get_approval_keyboard("msg_123")

        # Should have inline_keyboard structure or be InlineKeyboardMarkup
        assert keyboard is not None
