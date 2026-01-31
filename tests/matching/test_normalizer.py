import pytest
from src.matching.normalizer import EventNormalizer


def test_normalize_team_name_aliases():
    normalizer = EventNormalizer()

    assert normalizer.normalize_team("KC Chiefs") == "kansas city chiefs"
    assert normalizer.normalize_team("Kansas City Chiefs") == "kansas city chiefs"
    assert normalizer.normalize_team("Chiefs") == "kansas city chiefs"


def test_normalize_team_name_special_chars():
    normalizer = EventNormalizer()

    assert normalizer.normalize_team("L.A. Lakers") == "los angeles lakers"
    assert normalizer.normalize_team("LA Lakers") == "los angeles lakers"


def test_normalize_event_name():
    normalizer = EventNormalizer()

    result = normalizer.normalize_event(
        "Will the Kansas City Chiefs win Super Bowl LIX?"
    )
    assert "kansas city chiefs" in result
    assert "super bowl" in result


def test_extract_teams_from_event():
    normalizer = EventNormalizer()

    teams = normalizer.extract_teams(
        "Kansas City Chiefs vs Philadelphia Eagles - Super Bowl LIX"
    )
    assert "kansas city chiefs" in teams
    assert "philadelphia eagles" in teams


def test_extract_date_from_event():
    normalizer = EventNormalizer()

    date = normalizer.extract_date("Super Bowl LIX - February 9, 2025")
    assert date is not None
    assert date.year == 2025
    assert date.month == 2


def test_calculate_similarity():
    normalizer = EventNormalizer()

    # High similarity
    score = normalizer.calculate_similarity(
        "Chiefs win Super Bowl LIX",
        "Kansas City Chiefs to win Super Bowl LIX"
    )
    assert score > 0.7

    # Low similarity
    score = normalizer.calculate_similarity(
        "Chiefs win Super Bowl",
        "Lakers win NBA Finals"
    )
    assert score < 0.3
