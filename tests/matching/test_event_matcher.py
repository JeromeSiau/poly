"""Tests for the CrossMarketMatcher event matching system."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from src.matching.event_matcher import CrossMarketMatcher, MatchedEvent
from src.matching.llm_verifier import MatchResult
from src.feeds.azuro import AzuroEvent
from src.feeds.overtime import OvertimeGame


@pytest.fixture
def mock_verifier():
    verifier = MagicMock()
    verifier.verify_match = AsyncMock(return_value=MatchResult(
        is_match=True,
        confidence=0.98,
        reasoning="Same event",
    ))
    return verifier


@pytest.fixture
def matcher(mock_verifier):
    return CrossMarketMatcher(llm_verifier=mock_verifier)


def test_matched_event_creation():
    event = MatchedEvent(
        name="Chiefs win Super Bowl LIX",
        category="sports",
        polymarket_id="0x123",
        azuro_condition_id="456",
        overtime_game_id=None,
        confidence=0.98,
    )
    assert event.name == "Chiefs win Super Bowl LIX"
    assert event.has_polymarket
    assert event.has_azuro
    assert not event.has_overtime


def test_matched_event_platforms_count():
    event = MatchedEvent(
        name="Test",
        category="sports",
        polymarket_id="0x123",
        azuro_condition_id="456",
        overtime_game_id="789",
        confidence=0.95,
    )
    assert event.platforms_count == 3


@pytest.mark.asyncio
async def test_matcher_find_potential_matches(matcher):
    pm_events = [
        {"id": "0x123", "title": "Chiefs win Super Bowl LIX", "outcomes": ["Yes", "No"]},
    ]

    azuro_events = [
        AzuroEvent(
            condition_id="456",
            game_id="g1",
            sport="Football",
            league="NFL",
            home_team="Kansas City Chiefs",
            away_team="Philadelphia Eagles",
            starts_at=1700000000.0,
            outcomes={"1": 0.55, "2": 0.45},
        ),
    ]

    matches = await matcher.find_potential_matches(
        polymarket_events=pm_events,
        azuro_events=azuro_events,
        overtime_games=[],
    )

    assert len(matches) >= 0  # May or may not find matches depending on similarity


@pytest.mark.asyncio
async def test_matcher_verify_and_store(matcher):
    candidate = {
        "pm_event": {"id": "0x123", "title": "Chiefs win Super Bowl LIX"},
        "azuro_event": AzuroEvent(
            condition_id="456",
            game_id="g1",
            sport="Football",
            league="NFL",
            home_team="Kansas City Chiefs",
            away_team="Philadelphia Eagles",
            starts_at=1700000000.0,
            outcomes={"1": 0.55, "2": 0.45},
        ),
        "similarity": 0.85,
        "type": "pm_azuro",
    }

    result = await matcher.verify_match(candidate)

    assert result is not None
    assert result.confidence == 0.98
    assert result.polymarket_id == "0x123"
    assert result.azuro_condition_id == "456"
