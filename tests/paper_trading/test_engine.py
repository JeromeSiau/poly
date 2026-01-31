# tests/paper_trading/test_engine.py
"""Tests for paper trading engine."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from src.paper_trading.engine import PaperTradingEngine
from src.feeds.base import FeedEvent


class TestPaperTradingEngine:
    @pytest.fixture
    def mock_model(self):
        model = MagicMock()
        model.predict_single.return_value = 0.72
        return model

    @pytest.fixture
    def engine(self, mock_model):
        return PaperTradingEngine(
            model=mock_model,
            capital=10000,
            min_edge=0.05,
        )

    @pytest.fixture
    def sample_event(self):
        return FeedEvent(
            source="pandascore",
            event_type="baron_kill",
            game="lol",
            data={
                "team": "T1",
                "game_time_minutes": 25,
                "gold_diff": 5000,
                "kill_diff": 3,
                "tower_diff": 2,
                "dragon_diff": 1,
                "baron_diff": 1,
            },
            timestamp=datetime.now(timezone.utc).timestamp(),
            match_id="12345",
        )

    @pytest.mark.asyncio
    async def test_process_event_creates_observation(self, engine, sample_event):
        """Processing an event creates a LiveObservation."""
        with patch.object(engine.market_observer, "capture_price") as mock_capture:
            mock_capture.return_value = MagicMock(price=0.58)

            result = await engine.process_event(sample_event, market_id="market_123")

        assert result is not None
        assert result["model_prediction"] == 0.72
        assert result["market_price"] == 0.58
        assert result["edge"] == pytest.approx(0.14, rel=0.01)

    @pytest.mark.asyncio
    async def test_process_event_creates_trade_if_edge(self, engine, sample_event):
        """Creates paper trade if edge exceeds minimum."""
        with patch.object(engine.market_observer, "capture_price") as mock_capture:
            mock_capture.return_value = MagicMock(price=0.58)

            result = await engine.process_event(sample_event, market_id="market_123")

        assert result["trade"] is not None
        assert result["trade"]["size"] > 0

    @pytest.mark.asyncio
    async def test_no_trade_if_edge_too_small(self, engine, sample_event):
        """No trade if edge below minimum."""
        engine.model.predict_single.return_value = 0.60  # 2% edge

        with patch.object(engine.market_observer, "capture_price") as mock_capture:
            mock_capture.return_value = MagicMock(price=0.58)

            result = await engine.process_event(sample_event, market_id="market_123")

        assert result["trade"] is None

    @pytest.mark.asyncio
    async def test_schedules_followup_captures(self, engine, sample_event):
        """Engine schedules follow-up price captures."""
        with patch.object(engine.market_observer, "capture_price") as mock_capture:
            mock_capture.return_value = MagicMock(price=0.58)

            with patch.object(
                engine, "_schedule_followups", new_callable=AsyncMock
            ) as mock_schedule:
                await engine.process_event(sample_event, market_id="market_123")

                mock_schedule.assert_called_once()
