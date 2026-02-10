"""Tests for the Weather Oracle strategy."""

import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from src.arb.weather_oracle import (
    OpenMeteoFetcher,
    ForecastData,
    CITY_STATIONS,
    WeatherMarketScanner,
    WeatherMarket,
)


@pytest.mark.asyncio
async def test_open_meteo_fetcher_parses_response():
    """Fetcher should parse Open-Meteo JSON into ForecastData objects."""
    mock_response = {
        "daily": {
            "time": ["2026-02-10", "2026-02-11", "2026-02-12"],
            "temperature_2m_max": [55.2, 60.1, 58.7],
            "temperature_2m_min": [40.1, 45.3, 42.0],
        },
    }

    fetcher = OpenMeteoFetcher()

    with patch("aiohttp.ClientSession") as mock_session_cls:
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=mock_response)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session_cls.return_value = mock_session

        forecasts = await fetcher.fetch_city("Dallas")

    assert len(forecasts) == 3
    assert forecasts[0].city == "Dallas"
    assert forecasts[0].date == "2026-02-10"
    assert forecasts[0].temp_max == 55.2
    assert forecasts[0].unit == "fahrenheit"


@pytest.mark.asyncio
async def test_open_meteo_fetcher_handles_error():
    """Fetcher should return empty list on HTTP error."""
    fetcher = OpenMeteoFetcher()

    with patch("aiohttp.ClientSession") as mock_session_cls:
        mock_resp = AsyncMock()
        mock_resp.status = 500
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session_cls.return_value = mock_session

        forecasts = await fetcher.fetch_city("Dallas")

    assert forecasts == []


@pytest.mark.asyncio
async def test_open_meteo_fetcher_caches_results():
    """Fetcher should cache results and make them available via get_forecast."""
    mock_response = {
        "daily": {
            "time": ["2026-02-10"],
            "temperature_2m_max": [55.2],
            "temperature_2m_min": [40.1],
        },
    }

    fetcher = OpenMeteoFetcher()

    with patch("aiohttp.ClientSession") as mock_session_cls:
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=mock_response)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session_cls.return_value = mock_session

        await fetcher.fetch_city("Dallas")

    # Should be cached
    fc = fetcher.get_forecast("Dallas", "2026-02-10")
    assert fc is not None
    assert fc.temp_max == 55.2

    # Non-existent date should return None
    assert fetcher.get_forecast("Dallas", "2026-02-20") is None

    # Non-existent city should return None
    assert fetcher.get_forecast("Unknown", "2026-02-10") is None


def test_open_meteo_fetcher_unknown_city():
    """Fetcher should handle unknown cities gracefully."""
    fetcher = OpenMeteoFetcher()
    assert fetcher.get_forecast("UnknownCity", "2026-02-10") is None


def test_scanner_generates_slugs_for_all_cities():
    """Scanner should build slug for each city × upcoming date."""
    scanner = WeatherMarketScanner()
    slugs = scanner._generate_event_slugs()
    assert len(slugs) > 0

    # NYC should use "nyc" in slug, not "new-york"
    nyc_slugs = [(s, c, d) for s, c, d in slugs if c == "New York"]
    assert len(nyc_slugs) > 0
    assert all("nyc" in s for s, _, _ in nyc_slugs)

    # Buenos Aires should use "buenos-aires"
    ba_slugs = [(s, c, d) for s, c, d in slugs if c == "Buenos Aires"]
    assert len(ba_slugs) > 0
    assert all("buenos-aires" in s for s, _, _ in ba_slugs)

    # All slugs should start with "highest-temperature-in-"
    for slug, _, _ in slugs:
        assert slug.startswith("highest-temperature-in-")


def test_scanner_parses_event():
    """Scanner should parse a Gamma events API response into a WeatherMarket."""
    event = {
        "title": "Highest temperature in Dallas on February 10?",
        "slug": "highest-temperature-in-dallas-on-february-10-2026",
        "id": 202127,
        "endDate": "2026-02-10T12:00:00Z",
        "description": "Resolution source: Weather Underground",
        "markets": [
            {
                "question": "Will the highest temperature in Dallas be 61°F or below on February 10?",
                "outcomePrices": '["0.0005","0.9995"]',
                "clobTokenIds": '["tok1","tok2"]',
            },
            {
                "question": "Will the highest temperature in Dallas be between 62-63°F on February 10?",
                "outcomePrices": '["0.003","0.997"]',
                "clobTokenIds": '["tok3","tok4"]',
            },
            {
                "question": "Will the highest temperature in Dallas be 72°F or higher on February 10?",
                "outcomePrices": '["0.9765","0.0235"]',
                "clobTokenIds": '["tok5","tok6"]',
            },
        ],
    }

    scanner = WeatherMarketScanner()
    market = scanner._parse_event(event, "Dallas", "2026-02-10")

    assert market is not None
    assert market.city == "Dallas"
    assert market.target_date == "2026-02-10"
    assert market.condition_id == "highest-temperature-in-dallas-on-february-10-2026"
    assert "61°F or below" in market.outcome_prices
    assert "between 62-63°F" in market.outcome_prices
    assert "72°F or higher" in market.outcome_prices
    assert market.outcome_prices["61°F or below"] == 0.0005
    assert market.outcome_prices["72°F or higher"] == 0.9765
    assert market.outcomes["61°F or below"] == "tok1"  # YES token


@pytest.mark.asyncio
async def test_scanner_scan_discovers_events():
    """Scanner.scan() should query events API and discover weather markets."""
    mock_event = [{
        "title": "Highest temperature in Dallas on February 15?",
        "slug": "highest-temperature-in-dallas-on-february-15-2026",
        "endDate": "2026-02-15T12:00:00Z",
        "description": "Weather Underground",
        "markets": [
            {
                "question": "Will the highest temperature in Dallas be 58°F or below on February 15?",
                "outcomePrices": '["0.03","0.97"]',
                "clobTokenIds": '["tok1","tok2"]',
            },
            {
                "question": "Will the highest temperature in Dallas be 65°F or higher on February 15?",
                "outcomePrices": '["0.80","0.20"]',
                "clobTokenIds": '["tok3","tok4"]',
            },
        ],
    }]

    scanner = WeatherMarketScanner()

    # Return event for any slug, empty for most
    def make_resp(return_value):
        resp = AsyncMock()
        resp.status = 200
        resp.json = AsyncMock(return_value=return_value)
        resp.__aenter__ = AsyncMock(return_value=resp)
        resp.__aexit__ = AsyncMock(return_value=False)
        return resp

    call_count = 0

    def side_effect_get(url, **kwargs):
        nonlocal call_count
        call_count += 1
        slug = kwargs.get("params", {}).get("slug", "")
        if "dallas" in slug:
            return make_resp(mock_event)
        return make_resp([])

    with patch("aiohttp.ClientSession") as mock_cls:
        mock_session = AsyncMock()
        mock_session.get = MagicMock(side_effect=side_effect_get)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_cls.return_value = mock_session

        markets = await scanner.scan()

    # Should discover at least one Dallas market
    assert len(markets) >= 1
    assert markets[0].city == "Dallas"
    assert "58°F or below" in markets[0].outcome_prices


# ---------------------------------------------------------------------------
# WeatherOracleEngine tests
# ---------------------------------------------------------------------------
import time
from src.arb.weather_oracle import (
    WeatherOracleEngine,
    WeatherPaperTrade,
    WeatherSignal,
)


def test_engine_detects_cheap_certain_outcome():
    """Engine should signal when forecast makes a cheap outcome near-certain."""
    market = WeatherMarket(
        condition_id="0xabc",
        slug="highest-temp-dallas-feb-12",
        title="Highest temperature in Dallas on February 12?",
        city="Dallas",
        target_date="2026-02-12",
        outcomes={"58°F or higher": "tok1", "56-57°F": "tok2", "54-55°F": "tok3", "52-53°F": "tok4"},
        outcome_prices={"58°F or higher": 0.03, "56-57°F": 0.12, "54-55°F": 0.35, "52-53°F": 0.50},
        end_date="2026-02-13T00:00:00Z",
        resolution_source="Weather Underground",
    )

    forecast = ForecastData(
        city="Dallas", date="2026-02-12",
        temp_max=66.0, temp_min=45.0, unit="fahrenheit", fetched_at=time.time(),
    )

    engine = WeatherOracleEngine.__new__(WeatherOracleEngine)
    engine.max_entry_price = 0.05
    engine.min_confidence = 0.90
    engine._entered_markets = set()

    signals = engine.evaluate_market(market, forecast)
    assert len(signals) == 1
    assert signals[0].outcome == "58°F or higher"
    assert signals[0].entry_price == 0.03
    assert signals[0].confidence >= 0.90


def test_engine_skips_expensive_outcomes():
    """Engine should NOT signal outcomes priced above max_entry_price."""
    market = WeatherMarket(
        condition_id="0xdef",
        slug="highest-temp-miami-feb-12",
        title="Highest temperature in Miami on February 12?",
        city="Miami",
        target_date="2026-02-12",
        outcomes={"70°F or higher": "tok1", "68-69°F": "tok2"},
        outcome_prices={"70°F or higher": 0.85, "68-69°F": 0.10},
        end_date="2026-02-13T00:00:00Z",
        resolution_source="Weather Underground",
    )

    forecast = ForecastData(
        city="Miami", date="2026-02-12",
        temp_max=75.0, temp_min=65.0, unit="fahrenheit", fetched_at=time.time(),
    )

    engine = WeatherOracleEngine.__new__(WeatherOracleEngine)
    engine.max_entry_price = 0.05
    engine.min_confidence = 0.90
    engine._entered_markets = set()

    signals = engine.evaluate_market(market, forecast)
    assert len(signals) == 0


def test_engine_handles_range_outcomes():
    """Engine should detect when forecast lands in a specific range bucket."""
    market = WeatherMarket(
        condition_id="0xghi",
        slug="highest-temp-nyc-feb-12",
        title="Highest temperature in New York on February 12?",
        city="New York",
        target_date="2026-02-12",
        outcomes={"40°F or higher": "tok1", "38-39°F": "tok2", "36-37°F": "tok3"},
        outcome_prices={"40°F or higher": 0.60, "38-39°F": 0.02, "36-37°F": 0.03},
        end_date="2026-02-13T00:00:00Z",
        resolution_source="Weather Underground",
    )

    forecast = ForecastData(
        city="New York", date="2026-02-12",
        temp_max=38.5, temp_min=28.0, unit="fahrenheit", fetched_at=time.time(),
    )

    engine = WeatherOracleEngine.__new__(WeatherOracleEngine)
    engine.max_entry_price = 0.05
    engine.min_confidence = 0.90
    engine._entered_markets = set()

    signals = engine.evaluate_market(market, forecast)
    outcome_labels = [s.outcome for s in signals]
    assert "38-39°F" in outcome_labels


def test_engine_skips_already_entered():
    """Engine should not signal markets already entered."""
    market = WeatherMarket(
        condition_id="0xabc",
        slug="test",
        title="Highest temperature in Dallas on February 12?",
        city="Dallas",
        target_date="2026-02-12",
        outcomes={"58°F or higher": "tok1"},
        outcome_prices={"58°F or higher": 0.03},
        end_date="2026-02-13T00:00:00Z",
        resolution_source="",
    )

    forecast = ForecastData(
        city="Dallas", date="2026-02-12",
        temp_max=65.0, temp_min=45.0, unit="fahrenheit", fetched_at=time.time(),
    )

    engine = WeatherOracleEngine.__new__(WeatherOracleEngine)
    engine.max_entry_price = 0.05
    engine.min_confidence = 0.90
    engine._entered_markets = {"0xabc:58°F or higher"}

    signals = engine.evaluate_market(market, forecast)
    assert len(signals) == 0


def test_outcome_confidence_threshold_high():
    """Test confidence scoring for 'X degrees F or higher' outcomes."""
    engine = WeatherOracleEngine.__new__(WeatherOracleEngine)

    # Forecast well above threshold -> high confidence
    conf = engine._outcome_confidence("58°F or higher", 66.0)
    assert conf >= 0.90

    # Forecast at threshold -> 0
    conf = engine._outcome_confidence("58°F or higher", 58.0)
    assert conf == 0.0  # margin is 0, so confidence is 0

    # Forecast below threshold -> 0
    conf = engine._outcome_confidence("58°F or higher", 55.0)
    assert conf == 0.0


def test_outcome_confidence_threshold_low():
    """Test confidence scoring for 'X°F or lower' and 'X°F or below' outcomes."""
    engine = WeatherOracleEngine.__new__(WeatherOracleEngine)

    # "or lower" - Forecast well below threshold -> high confidence
    conf = engine._outcome_confidence("47°F or lower", 38.0)
    assert conf >= 0.90

    # "or below" (used by Polymarket) - same logic
    conf = engine._outcome_confidence("47°F or below", 38.0)
    assert conf >= 0.90

    # Forecast above threshold -> 0
    conf = engine._outcome_confidence("47°F or lower", 50.0)
    assert conf == 0.0
    conf = engine._outcome_confidence("47°F or below", 50.0)
    assert conf == 0.0


def test_engine_enter_paper_trade():
    """Engine should record a paper trade from a signal."""
    market = WeatherMarket(
        condition_id="0xabc",
        slug="highest-temp-dallas-feb-12",
        title="Highest temperature in Dallas on February 12?",
        city="Dallas",
        target_date="2026-02-12",
        outcomes={"58°F or higher": "tok1"},
        outcome_prices={"58°F or higher": 0.03},
        end_date="2026-02-13T00:00:00Z",
        resolution_source="Weather Underground",
    )
    forecast = ForecastData(
        city="Dallas", date="2026-02-12",
        temp_max=65.0, temp_min=45.0, unit="fahrenheit", fetched_at=time.time(),
    )
    signal = WeatherSignal(
        market=market, outcome="58°F or higher", entry_price=0.03,
        forecast=forecast, confidence=0.95, reason="test",
    )

    engine = WeatherOracleEngine.__new__(WeatherOracleEngine)
    engine.paper_size = 3.0
    engine.max_daily_spend = 50.0
    engine._open_trades = []
    engine._entered_markets = set()
    engine._daily_spend = 0.0
    engine._daily_spend_date = ""
    engine._stats = {"trades": 0, "wins": 0, "pnl": 0.0}

    trade = engine.enter_paper_trade(signal)
    assert trade is not None
    assert trade.city == "Dallas"
    assert trade.entry_price == 0.03
    assert trade.size_usd == 3.0
    assert "0xabc:58°F or higher" in engine._entered_markets
    assert len(engine._open_trades) == 1


def test_engine_daily_spend_limit():
    """Engine should respect daily spend limit."""
    from datetime import datetime, timezone

    engine = WeatherOracleEngine.__new__(WeatherOracleEngine)
    engine.paper_size = 30.0
    engine.max_daily_spend = 50.0
    engine._open_trades = []
    engine._entered_markets = set()
    engine._daily_spend = 40.0
    engine._daily_spend_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    engine._stats = {"trades": 0, "wins": 0, "pnl": 0.0}

    market = WeatherMarket(
        condition_id="0x1", slug="test", title="test",
        city="Dallas", target_date="2026-02-12",
        outcomes={"58°F or higher": "tok1"},
        outcome_prices={"58°F or higher": 0.03},
        end_date="2026-02-13T00:00:00Z", resolution_source="",
    )
    forecast = ForecastData(
        city="Dallas", date="2026-02-12",
        temp_max=65.0, temp_min=45.0, unit="fahrenheit", fetched_at=time.time(),
    )
    signal = WeatherSignal(
        market=market, outcome="58°F or higher", entry_price=0.03,
        forecast=forecast, confidence=0.95, reason="test",
    )

    result = engine.enter_paper_trade(signal)
    assert result is None  # exceeded daily limit


@pytest.mark.asyncio
async def test_full_cycle_scan_evaluate_enter():
    """Integration: scan -> forecast -> evaluate -> enter trade."""
    from pathlib import Path

    mock_event_resp = [{
        "title": "Highest temperature in Dallas on February 15?",
        "slug": "highest-temperature-in-dallas-on-february-15-2026",
        "endDate": "2026-02-16T00:00:00Z",
        "description": "Resolution source: Weather Underground",
        "markets": [
            {
                "question": "Will the highest temperature in Dallas be 60°F or higher on February 15?",
                "outcomePrices": '["0.02","0.98"]',
                "clobTokenIds": '["t1","t2"]',
            },
            {
                "question": "Will the highest temperature in Dallas be between 58-59°F on February 15?",
                "outcomePrices": '["0.08","0.92"]',
                "clobTokenIds": '["t3","t4"]',
            },
            {
                "question": "Will the highest temperature in Dallas be 57°F or below on February 15?",
                "outcomePrices": '["0.15","0.85"]',
                "clobTokenIds": '["t5","t6"]',
            },
        ],
    }]

    mock_forecast_resp = {
        "daily": {
            "time": ["2026-02-15"],
            "temperature_2m_max": [78.0],
            "temperature_2m_min": [42.0],
        },
    }

    engine = WeatherOracleEngine.__new__(WeatherOracleEngine)
    engine.fetcher = OpenMeteoFetcher()
    engine.scanner = WeatherMarketScanner()
    engine.max_entry_price = 0.05
    engine.min_confidence = 0.90
    engine.paper_size = 3.0
    engine.max_daily_spend = 50.0
    engine.paper_file = Path("/tmp/test_weather_oracle.jsonl")
    engine._open_trades = []
    engine._entered_markets = set()
    engine._daily_spend = 0.0
    engine._daily_spend_date = ""
    engine._stats = {"trades": 0, "wins": 0, "pnl": 0.0}
    engine._database_url = "sqlite:///data/arb.db"

    def make_resp(return_value):
        resp = AsyncMock()
        resp.status = 200
        resp.json = AsyncMock(return_value=return_value)
        resp.__aenter__ = AsyncMock(return_value=resp)
        resp.__aexit__ = AsyncMock(return_value=False)
        return resp

    with patch("aiohttp.ClientSession") as mock_cls:
        # Scanner calls use events API; fetcher calls use open-meteo
        def side_effect_get(url, **kwargs):
            if "gamma" in url or "events" in str(kwargs):
                slug = kwargs.get("params", {}).get("slug", "")
                if "dallas" in slug and "february-15" in slug:
                    return make_resp(mock_event_resp)
                return make_resp([])
            # Open-Meteo forecast
            return make_resp(mock_forecast_resp)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(side_effect=side_effect_get)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_cls.return_value = mock_session

        # Scan
        await engine.scanner.scan()
        assert len(engine.scanner.markets) >= 1

        # Fetch forecast
        await engine.fetcher.fetch_city("Dallas")
        forecast = engine.fetcher.get_forecast("Dallas", "2026-02-15")
        assert forecast is not None

    # Evaluate
    market = list(engine.scanner.markets.values())[0]
    signals = engine.evaluate_market(market, forecast)
    assert len(signals) >= 1
    assert signals[0].outcome == "60°F or higher"

    # Enter trade
    trade = engine.enter_paper_trade(signals[0])
    assert trade is not None
    assert trade.city == "Dallas"
    assert trade.entry_price == 0.02
