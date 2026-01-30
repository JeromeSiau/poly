# tests/realtime/test_event_detector.py
import pytest
from datetime import datetime

from src.feeds.base import FeedEvent
from src.realtime.event_detector import EventDetector, SignificantEvent


class TestSignificantEventClassification:
    @pytest.fixture
    def detector(self):
        return EventDetector()

    def test_lol_kill_is_significant_late_game(self, detector):
        """Kill at 35 minutes is significant (late game)."""
        event = FeedEvent(
            source="pandascore",
            event_type="kill",
            game="lol",
            data={"killer": "Faker", "killer_team": "T1", "game_time_minutes": 35},
            timestamp=datetime.utcnow().timestamp()
        )
        result = detector.classify(event)
        assert result is not None
        assert result.is_significant is True
        assert result.impact_score >= 0.5

    def test_lol_kill_not_significant_early_game(self, detector):
        """Single kill at 5 minutes is not very significant."""
        event = FeedEvent(
            source="pandascore",
            event_type="kill",
            game="lol",
            data={"killer": "Faker", "killer_team": "T1", "game_time_minutes": 5},
            timestamp=datetime.utcnow().timestamp()
        )
        result = detector.classify(event)
        assert result.impact_score < 0.3

    def test_lol_baron_is_highly_significant(self, detector):
        """Baron kill is always highly significant."""
        event = FeedEvent(
            source="pandascore",
            event_type="baron_kill",
            game="lol",
            data={"team": "T1"},
            timestamp=datetime.utcnow().timestamp()
        )
        result = detector.classify(event)
        assert result.is_significant is True
        assert result.impact_score >= 0.8

    def test_csgo_round_end_calculates_momentum(self, detector):
        """Round end with score change affects momentum."""
        event = FeedEvent(
            source="pandascore",
            event_type="round_end",
            game="csgo",
            data={"winner": "Navi", "score": {"Navi": 12, "FaZe": 8}},
            timestamp=datetime.utcnow().timestamp()
        )
        result = detector.classify(event)
        assert result is not None
        assert result.favored_team == "Navi"
        assert result.impact_score > 0.5


class TestEventDetectorPriceImpact:
    @pytest.fixture
    def detector(self):
        return EventDetector()

    def test_estimate_price_impact_baron(self, detector):
        """Baron should swing price by ~10-15%."""
        event = SignificantEvent(
            original_event=None,
            is_significant=True,
            impact_score=0.85,
            favored_team="T1",
            event_description="Baron kill by T1"
        )
        current_price = 0.50
        estimated_new_price = detector.estimate_price_impact(event, current_price)
        assert estimated_new_price >= 0.60
        assert estimated_new_price <= 0.70

    def test_estimate_price_impact_late_kill(self, detector):
        """Late game kill should have smaller impact."""
        event = SignificantEvent(
            original_event=None,
            is_significant=True,
            impact_score=0.55,
            favored_team="T1",
            event_description="Kill by Faker"
        )
        current_price = 0.60
        estimated_new_price = detector.estimate_price_impact(event, current_price)
        assert estimated_new_price > 0.60
        assert estimated_new_price < 0.70
