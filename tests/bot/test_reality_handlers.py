# tests/bot/test_reality_handlers.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from src.bot.reality_handlers import RealityArbHandler
from src.arb.reality_arb import ArbOpportunity


class TestAlertFormatting:
    @pytest.fixture
    def handler(self):
        return RealityArbHandler(bot=MagicMock(), engine=MagicMock())

    def test_format_opportunity_alert(self, handler):
        opportunity = ArbOpportunity(
            market_id="0x123abc",
            side="BUY",
            outcome="YES",
            current_price=0.52,
            estimated_fair_price=0.65,
            edge_pct=0.12,
            trigger_event="Baron kill by T1",
            timestamp=datetime.utcnow().timestamp()
        )

        alert = handler.format_alert(opportunity, market_title="T1 vs Gen.G Winner")

        assert "REALITY ARB" in alert
        assert "T1 vs Gen.G" in alert
        assert "12%" in alert or "12.0%" in alert
        assert "Baron kill" in alert
        assert "BUY YES" in alert

    def test_format_includes_expiry(self, handler):
        opportunity = ArbOpportunity(
            market_id="0x123",
            side="BUY",
            outcome="YES",
            current_price=0.50,
            estimated_fair_price=0.60,
            edge_pct=0.10,
            trigger_event="Dragon kill",
            timestamp=datetime.utcnow().timestamp()
        )

        alert = handler.format_alert(opportunity, market_title="Test", expiry_seconds=30)

        assert "30s" in alert or "30 sec" in alert.lower()


class TestApprovalFlow:
    @pytest.fixture
    def handler(self):
        handler = RealityArbHandler(bot=AsyncMock(), engine=MagicMock())
        handler.engine.execute = AsyncMock(return_value={"status": "FILLED"})
        return handler

    @pytest.mark.asyncio
    async def test_approve_executes_trade(self, handler):
        opportunity = ArbOpportunity(
            market_id="0x123",
            side="BUY",
            outcome="YES",
            current_price=0.50,
            estimated_fair_price=0.60,
            edge_pct=0.10,
            trigger_event="test",
            timestamp=datetime.utcnow().timestamp()
        )
        handler._pending = {"0x123": (opportunity, 100)}

        result = await handler.handle_approve("0x123")

        assert result["executed"] is True
        handler.engine.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_skip_removes_opportunity(self, handler):
        opportunity = ArbOpportunity(
            market_id="0x123",
            side="BUY",
            outcome="YES",
            current_price=0.50,
            estimated_fair_price=0.60,
            edge_pct=0.10,
            trigger_event="test",
            timestamp=datetime.utcnow().timestamp()
        )
        handler._pending = {"0x123": (opportunity, 100)}

        result = await handler.handle_skip("0x123")

        assert result["skipped"] is True
        assert "0x123" not in handler._pending
