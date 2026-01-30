# tests/feeds/test_pandascore.py
import pytest
import respx
from httpx import Response
from datetime import datetime

from src.feeds.pandascore import PandaScoreFeed, PandaScoreEvent


@pytest.fixture
def mock_api():
    with respx.mock:
        yield respx


class TestPandaScoreEventParsing:
    def test_parse_lol_kill_event(self):
        raw_data = {
            "type": "kill",
            "timestamp": 1234567890,
            "payload": {
                "killer": {"name": "Faker", "team": "T1"},
                "victim": {"name": "Chovy", "team": "Gen.G"},
                "assists": [{"name": "Keria"}]
            }
        }

        event = PandaScoreEvent.from_raw("lol", "match_123", raw_data)

        assert event.event_type == "kill"
        assert event.game == "lol"
        assert event.data["killer"] == "Faker"
        assert event.data["killer_team"] == "T1"

    def test_parse_lol_tower_event(self):
        raw_data = {
            "type": "tower_destroyed",
            "timestamp": 1234567891,
            "payload": {
                "team": "T1",
                "tower": "top_outer"
            }
        }

        event = PandaScoreEvent.from_raw("lol", "match_123", raw_data)

        assert event.event_type == "tower_destroyed"
        assert event.data["team"] == "T1"

    def test_parse_csgo_round_event(self):
        raw_data = {
            "type": "round_end",
            "timestamp": 1234567892,
            "payload": {
                "winner": "Navi",
                "score": {"Navi": 10, "FaZe": 8}
            }
        }

        event = PandaScoreEvent.from_raw("csgo", "match_456", raw_data)

        assert event.event_type == "round_end"
        assert event.data["winner"] == "Navi"


class TestPandaScoreFeed:
    @pytest.mark.asyncio
    async def test_fetch_live_matches(self, mock_api):
        mock_api.get("https://api.pandascore.co/lol/matches/running").mock(
            return_value=Response(200, json=[
                {
                    "id": 123,
                    "name": "T1 vs Gen.G",
                    "league": {"name": "LCK"},
                    "status": "running"
                }
            ])
        )

        feed = PandaScoreFeed(api_key="test_key")
        matches = await feed.get_live_matches("lol")

        assert len(matches) == 1
        assert matches[0]["name"] == "T1 vs Gen.G"

    @pytest.mark.asyncio
    async def test_latency_measurement(self, mock_api):
        mock_api.get("https://api.pandascore.co/lol/matches/running").mock(
            return_value=Response(200, json=[{"id": 123, "status": "running"}])
        )

        feed = PandaScoreFeed(api_key="test_key")
        latency_ms = await feed.measure_latency()

        assert latency_ms > 0
        assert latency_ms < 5000  # Should be under 5 seconds
