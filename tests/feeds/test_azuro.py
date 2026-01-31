import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.feeds.azuro import AzuroFeed, AzuroEvent


def test_azuro_event_creation():
    event = AzuroEvent(
        condition_id="123",
        game_id="456",
        sport="football",
        league="NFL",
        home_team="Chiefs",
        away_team="Eagles",
        starts_at=1700000000.0,
        outcomes={"home": 0.55, "away": 0.45},
    )
    assert event.condition_id == "123"
    assert event.outcomes["home"] == 0.55


def test_azuro_feed_extends_base_feed():
    from src.feeds.base import BaseFeed
    feed = AzuroFeed()
    assert isinstance(feed, BaseFeed)


@pytest.mark.asyncio
async def test_azuro_feed_connect():
    feed = AzuroFeed()
    with patch.object(feed, "_init_client") as mock_init:
        await feed.connect()
        assert feed.is_connected
        mock_init.assert_called_once()


@pytest.mark.asyncio
async def test_azuro_feed_get_active_events():
    feed = AzuroFeed()

    mock_response = {
        "conditions": [
            {
                "conditionId": "123",
                "gameId": "456",
                "outcomes": [
                    {"outcomeId": "1", "odds": "1.8"},
                    {"outcomeId": "2", "odds": "2.2"},
                ],
                "game": {
                    "sport": {"name": "Football"},
                    "league": {"name": "NFL"},
                    "participants": [
                        {"name": "Kansas City Chiefs"},
                        {"name": "Philadelphia Eagles"},
                    ],
                    "startsAt": "1700000000",
                },
            }
        ]
    }

    with patch.object(feed, "_execute_query", new_callable=AsyncMock) as mock_query:
        mock_query.return_value = mock_response
        feed._connected = True

        events = await feed.get_active_events()

        assert len(events) == 1
        assert events[0].condition_id == "123"
        assert events[0].home_team == "Kansas City Chiefs"


@pytest.mark.asyncio
async def test_azuro_feed_get_odds():
    feed = AzuroFeed()

    mock_response = {
        "condition": {
            "outcomes": [
                {"outcomeId": "1", "odds": "1.85"},
                {"outcomeId": "2", "odds": "2.15"},
            ]
        }
    }

    with patch.object(feed, "_execute_query", new_callable=AsyncMock) as mock_query:
        mock_query.return_value = mock_response
        feed._connected = True

        odds = await feed.get_odds("123")

        assert "1" in odds
        assert "2" in odds
        assert abs(odds["1"] - 0.54) < 0.01  # 1/1.85 ≈ 0.54
