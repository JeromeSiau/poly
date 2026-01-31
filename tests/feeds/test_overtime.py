import pytest
from unittest.mock import AsyncMock, patch
from src.feeds.overtime import OvertimeFeed, OvertimeGame


def test_overtime_game_creation():
    game = OvertimeGame(
        game_id="123",
        sport="NFL",
        home_team="Chiefs",
        away_team="Eagles",
        starts_at=1700000000.0,
        home_odds=0.55,
        away_odds=0.45,
        is_resolved=False,
    )
    assert game.game_id == "123"
    assert game.home_odds == 0.55


def test_overtime_feed_extends_base_feed():
    from src.feeds.base import BaseFeed
    feed = OvertimeFeed()
    assert isinstance(feed, BaseFeed)


@pytest.mark.asyncio
async def test_overtime_feed_connect():
    feed = OvertimeFeed()
    with patch.object(feed, "_init_client") as mock_init:
        await feed.connect()
        assert feed.is_connected
        mock_init.assert_called_once()


@pytest.mark.asyncio
async def test_overtime_feed_get_active_games():
    feed = OvertimeFeed()

    mock_response = {
        "sportMarkets": [
            {
                "id": "0x123",
                "gameId": "456",
                "tags": ["9004"],  # NFL tag
                "homeTeam": "Kansas City Chiefs",
                "awayTeam": "Philadelphia Eagles",
                "maturityDate": "1700000000",
                "homeOdds": "550000000000000000",  # 0.55 in wei-like format
                "awayOdds": "450000000000000000",
                "isResolved": False,
            }
        ]
    }

    with patch.object(feed, "_execute_query", new_callable=AsyncMock) as mock_query:
        mock_query.return_value = mock_response
        feed._connected = True

        games = await feed.get_active_games()

        assert len(games) == 1
        assert games[0].game_id == "456"
        assert games[0].home_team == "Kansas City Chiefs"


@pytest.mark.asyncio
async def test_overtime_feed_get_odds():
    feed = OvertimeFeed()

    mock_response = {
        "sportMarket": {
            "homeOdds": "550000000000000000",
            "awayOdds": "450000000000000000",
            "drawOdds": "0",
        }
    }

    with patch.object(feed, "_execute_query", new_callable=AsyncMock) as mock_query:
        mock_query.return_value = mock_response
        feed._connected = True

        odds = await feed.get_odds("0x123")

        assert "home" in odds
        assert "away" in odds
        assert abs(odds["home"] - 0.55) < 0.01
