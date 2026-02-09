import pytest
import httpx

from src.feeds.odds_api import OddsApiClient, ScoreTracker, LiveGame, _devig


def test_devig_binary_probs_sum_to_one() -> None:
    probs = _devig({"home": 1.80, "away": 2.20})
    assert "home" in probs
    assert "away" in probs
    assert abs(sum(probs.values()) - 1.0) < 1e-9
    assert 0 < probs["home"] < 1


def test_parse_event_row_builds_moneyline_and_totals() -> None:
    client = OddsApiClient(api_key="test-key")
    row = {
        "id": "evt-123",
        "home_team": "AC Monza",
        "away_team": "US Avellino 1912",
        "commence_time": "2026-02-08T19:45:00Z",
        "sport_title": "Soccer Italy Serie B",
        "bookmakers": [
            {
                "key": "book-a",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "AC Monza", "price": 2.40},
                            {"name": "Draw", "price": 3.10},
                            {"name": "US Avellino 1912", "price": 2.90},
                        ],
                    },
                    {
                        "key": "totals",
                        "outcomes": [
                            {"name": "Over", "price": 1.95, "point": 2.5},
                            {"name": "Under", "price": 1.85, "point": 2.5},
                        ],
                    },
                ],
            },
            {
                "key": "book-b",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "AC Monza", "price": 2.35},
                            {"name": "Draw", "price": 3.20},
                            {"name": "US Avellino 1912", "price": 2.95},
                        ],
                    },
                ],
            },
        ],
    }

    event = client._parse_event_row(row=row, sport="upcoming")
    assert event is not None
    assert event.event_id == "upcoming:evt-123"
    assert event.home_team == "AC Monza"
    assert event.away_team == "US Avellino 1912"
    assert len(event.markets) == 2

    h2h = next(m for m in event.markets if m.market_type == "moneyline")
    totals = next(m for m in event.markets if m.market_type == "totals")

    assert "home" in h2h.outcomes
    assert "away" in h2h.outcomes
    assert "draw" in h2h.outcomes
    assert abs(sum(h2h.outcomes.values()) - 1.0) < 1e-9

    assert totals.line == 2.5
    assert set(totals.outcomes.keys()) == {"over", "under"}
    assert abs(sum(totals.outcomes.values()) - 1.0) < 1e-9


@pytest.mark.asyncio
async def test_fetch_scores_returns_live_games(respx_mock):
    """fetch_scores should parse completed and in-progress games."""
    payload = [
        {
            "id": "abc123",
            "sport_key": "soccer_epl",
            "sport_title": "EPL",
            "commence_time": "2026-02-09T15:00:00Z",
            "home_team": "Liverpool",
            "away_team": "Arsenal",
            "completed": False,
            "scores": [
                {"name": "Liverpool", "score": "2"},
                {"name": "Arsenal", "score": "1"},
            ],
            "last_updated": "2026-02-09T15:45:00Z",
        },
        {
            "id": "def456",
            "sport_key": "soccer_epl",
            "sport_title": "EPL",
            "commence_time": "2026-02-09T13:00:00Z",
            "home_team": "Chelsea",
            "away_team": "Spurs",
            "completed": True,
            "scores": [
                {"name": "Chelsea", "score": "0"},
                {"name": "Spurs", "score": "1"},
            ],
            "last_updated": "2026-02-09T14:52:00Z",
        },
    ]
    respx_mock.get("https://api.the-odds-api.com/v4/sports/soccer_epl/scores").mock(
        return_value=httpx.Response(
            200,
            json=payload,
            headers={"x-requests-remaining": "9500", "x-requests-used": "500"},
        )
    )

    client = OddsApiClient(api_key="test-key")
    async with httpx.AsyncClient() as http:
        result = await client.fetch_scores(http, sports=["soccer_epl"])

    assert len(result.games) == 2
    liv = result.games[0]
    assert liv.home_team == "Liverpool"
    assert liv.away_team == "Arsenal"
    assert liv.home_score == 2
    assert liv.away_score == 1
    assert liv.completed is False
    che = result.games[1]
    assert che.completed is True
    assert che.away_score == 1
    assert result.usage.remaining == 9500


def test_fetch_scores_detects_change():
    """ScoreTracker should detect when a score changes between polls."""
    tracker = ScoreTracker()
    game = LiveGame(
        event_id="abc123", sport="soccer_epl", home_team="Liverpool",
        away_team="Arsenal", home_score=1, away_score=0, completed=False,
    )
    changes = tracker.update([game])
    assert len(changes) == 0  # first observation, no change

    game_updated = LiveGame(
        event_id="abc123", sport="soccer_epl", home_team="Liverpool",
        away_team="Arsenal", home_score=2, away_score=0, completed=False,
    )
    changes = tracker.update([game_updated])
    assert len(changes) == 1
    assert changes[0].event_id == "abc123"
    assert changes[0].home_score == 2
    assert changes[0].prev_home_score == 1
    assert changes[0].change_type == "score_change"


def test_fetch_scores_detects_completion():
    """ScoreTracker should detect game completion."""
    tracker = ScoreTracker()
    game = LiveGame(
        event_id="abc123", sport="soccer_epl", home_team="Liverpool",
        away_team="Arsenal", home_score=2, away_score=1, completed=False,
    )
    tracker.update([game])

    game_ended = LiveGame(
        event_id="abc123", sport="soccer_epl", home_team="Liverpool",
        away_team="Arsenal", home_score=2, away_score=1, completed=True,
    )
    changes = tracker.update([game_ended])
    assert len(changes) == 1
    assert changes[0].change_type == "completed"
