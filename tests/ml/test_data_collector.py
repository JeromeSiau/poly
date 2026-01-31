"""Tests for historical data collection from PandaScore."""

import pytest
from unittest.mock import AsyncMock, MagicMock
import httpx

from src.ml.data_collector import HistoricalDataCollector, MatchData, EventData


class TestHistoricalDataCollector:
    """Tests for HistoricalDataCollector."""

    @pytest.fixture
    def collector(self):
        return HistoricalDataCollector(api_key="test_key")

    @pytest.fixture
    def mock_match_response(self):
        """Sample PandaScore match response."""
        return {
            "id": 12345,
            "name": "T1 vs Gen.G",
            "winner": {"id": 1, "name": "T1"},
            "opponents": [
                {"opponent": {"id": 1, "name": "T1"}},
                {"opponent": {"id": 2, "name": "Gen.G"}},
            ],
            "games": [
                {
                    "id": 111,
                    "winner": {"id": 1, "name": "T1"},
                    "length": 1920,  # 32 minutes in seconds
                }
            ],
        }

    @pytest.fixture
    def mock_events_response(self):
        """Sample PandaScore events response."""
        return [
            {
                "type": "kill",
                "timestamp": 300,
                "payload": {
                    "killer": {"name": "Faker", "team": "T1"},
                    "victim": {"name": "Chovy", "team": "Gen.G"},
                },
            },
            {
                "type": "baron_kill",
                "timestamp": 1200,
                "payload": {"team": "T1"},
            },
        ]

    @pytest.mark.asyncio
    async def test_fetch_past_matches_returns_match_list(
        self, collector, mock_match_response
    ):
        """fetch_past_matches returns list of MatchData."""
        collector._client = AsyncMock()
        collector._client.get = AsyncMock(
            return_value=MagicMock(
                json=lambda: [mock_match_response],
                raise_for_status=lambda: None,
            )
        )

        matches = await collector.fetch_past_matches(game="lol", limit=10)

        assert len(matches) == 1
        assert matches[0].match_id == 12345
        assert matches[0].winner == "T1"

    @pytest.mark.asyncio
    async def test_fetch_match_events_returns_event_list(
        self, collector, mock_events_response
    ):
        """fetch_match_events returns list of EventData."""
        collector._client = AsyncMock()
        collector._client.get = AsyncMock(
            return_value=MagicMock(
                json=lambda: mock_events_response,
                raise_for_status=lambda: None,
            )
        )

        events = await collector.fetch_match_events(
            game="lol", match_id=12345, game_id=111
        )

        assert len(events) == 2
        assert events[0].event_type == "kill"
        assert events[1].event_type == "baron_kill"

    def test_event_data_has_game_state(self):
        """EventData includes game state snapshot."""
        event = EventData(
            event_type="baron_kill",
            timestamp=1200,
            team="T1",
            game_time_minutes=20.0,
            gold_diff=5000,
            kill_diff=4,
            tower_diff=2,
            dragon_diff=1,
            baron_diff=1,
            winner="T1",
        )

        assert event.gold_diff == 5000
        assert event.winner == "T1"

    def test_match_data_has_teams(self):
        """MatchData includes team names and winner."""
        match = MatchData(
            match_id=12345,
            game="lol",
            team_a="T1",
            team_b="Gen.G",
            winner="T1",
            game_length_minutes=32.0,
        )

        assert match.team_a == "T1"
        assert match.winner == "T1"
