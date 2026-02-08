from src.feeds.odds_api import OddsApiClient, _devig


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
