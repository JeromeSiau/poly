from datetime import datetime, timezone

from src.matching.bookmaker_matcher import (
    BookmakerEvent,
    BookmakerMarket,
    BookmakerMatcher,
)


def test_match_draw_binary_market() -> None:
    matcher = BookmakerMatcher(min_event_confidence=0.50)

    polymarket = {
        "conditionId": "cond-draw",
        "question": "Will AC Monza vs US Avellino 1912 end in a draw?",
        "outcomes": '["Yes", "No"]',
        "startDate": "2026-02-08T19:45:00Z",
    }
    events = [
        BookmakerEvent(
            event_id="bk-1",
            home_team="AC Monza",
            away_team="US Avellino 1912",
            starts_at=datetime(2026, 2, 8, 19, 45, tzinfo=timezone.utc),
            markets=[
                BookmakerMarket(
                    market_type="moneyline",
                    outcomes={"home": 0.43, "draw": 0.29, "away": 0.28},
                )
            ],
        )
    ]

    result = matcher.match_market(polymarket, events)
    assert result is not None
    assert result.bookmaker_event_id == "bk-1"
    assert result.outcome_map == {"Yes": "draw", "No": "1-draw"}


def test_match_ou_line_to_totals_market() -> None:
    matcher = BookmakerMatcher(min_event_confidence=0.50)

    polymarket = {
        "conditionId": "cond-ou",
        "question": "SV Darmstadt 98 vs. 1. FC Kaiserslautern: O/U 3.5",
        "outcomes": '["Over", "Under"]',
        "startDate": "2026-02-08T19:30:00Z",
    }
    events = [
        BookmakerEvent(
            event_id="bk-2",
            home_team="SV Darmstadt 98",
            away_team="1. FC Kaiserslautern",
            starts_at=datetime(2026, 2, 8, 19, 30, tzinfo=timezone.utc),
            markets=[
                BookmakerMarket(
                    market_type="totals",
                    line=2.5,
                    outcomes={"over": 0.52, "under": 0.48},
                    market_id="tot-25",
                ),
                BookmakerMarket(
                    market_type="totals",
                    line=3.5,
                    outcomes={"over": 0.33, "under": 0.67},
                    market_id="tot-35",
                ),
            ],
        )
    ]

    result = matcher.match_market(polymarket, events)
    assert result is not None
    assert result.bookmaker_market_type == "totals"
    assert result.bookmaker_market_id == "tot-35"
    assert result.outcome_map == {"Over": "over", "Under": "under"}


def test_match_yes_no_team_win_market() -> None:
    matcher = BookmakerMatcher(min_event_confidence=0.45)

    polymarket = {
        "conditionId": "cond-win",
        "question": "Will Arsenal FC win on 2026-02-15?",
        "outcomes": '["Yes", "No"]',
        "startDate": "2026-02-15T16:00:00Z",
    }
    events = [
        BookmakerEvent(
            event_id="bk-3",
            home_team="Arsenal FC",
            away_team="Chelsea FC",
            starts_at=datetime(2026, 2, 15, 16, 0, tzinfo=timezone.utc),
            markets=[
                BookmakerMarket(
                    market_type="moneyline",
                    outcomes={"home": 0.57, "draw": 0.23, "away": 0.20},
                )
            ],
        )
    ]

    result = matcher.match_market(polymarket, events)
    assert result is not None
    assert result.outcome_map["Yes"] == "home"
    assert result.outcome_map["No"] == "1-home"


def test_match_named_binary_outcomes_tennis() -> None:
    matcher = BookmakerMatcher(min_event_confidence=0.45)

    polymarket = {
        "conditionId": "cond-tennis",
        "question": "Argentina Open, Qualification: Andrea Collarini vs Thiago Seyboth Wild",
        "outcomes": '["Collarini", "Wild"]',
        "startDate": "2026-02-08T13:00:00Z",
    }
    events = [
        BookmakerEvent(
            event_id="bk-4",
            home_team="Andrea Collarini",
            away_team="Thiago Seyboth Wild",
            starts_at=datetime(2026, 2, 8, 13, 0, tzinfo=timezone.utc),
            markets=[
                BookmakerMarket(
                    market_type="moneyline",
                    outcomes={"andrea collarini": 0.47, "thiago seyboth wild": 0.53},
                )
            ],
        )
    ]

    result = matcher.match_market(polymarket, events)
    assert result is not None
    assert result.outcome_map == {"Collarini": "home", "Wild": "away"}
