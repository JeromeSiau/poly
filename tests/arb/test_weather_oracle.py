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


def test_scanner_extracts_city_from_title():
    """Scanner should extract city name from market title."""
    scanner = WeatherMarketScanner()
    assert scanner._extract_city("Highest temperature in Dallas on February 12?") == "Dallas"
    assert scanner._extract_city("Highest temperature in New York on February 12?") == "New York"
    assert scanner._extract_city("Highest temperature in Buenos Aires on February 12?") == "Buenos Aires"
    assert scanner._extract_city("Unrelated market title") is None


def test_scanner_extracts_date_from_title():
    """Scanner should extract target date from market title."""
    scanner = WeatherMarketScanner()
    result = scanner._extract_target_date("Highest temperature in Dallas on February 12?")
    assert result is not None
    assert result.endswith("-02-12")


def test_scanner_extracts_date_various_months():
    """Scanner should handle different month names."""
    scanner = WeatherMarketScanner()
    assert scanner._extract_target_date("Highest temperature in Miami on January 5?").endswith("-01-05")
    assert scanner._extract_target_date("Highest temperature in London on March 20?").endswith("-03-20")
    assert scanner._extract_target_date("Highest temperature in Seoul on December 1?").endswith("-12-01")


@pytest.mark.asyncio
async def test_scanner_parses_weather_markets():
    """Scanner should parse Gamma API response into WeatherMarket objects."""
    mock_markets = [
        {
            "conditionId": "0xabc123",
            "slug": "highest-temperature-in-dallas-on-february-12",
            "question": "Highest temperature in Dallas on February 12?",
            "outcomes": '["58\u00b0F or higher","56-57\u00b0F","54-55\u00b0F","52-53\u00b0F","50-51\u00b0F","48-49\u00b0F","47\u00b0F or lower"]',
            "outcomePrices": '[0.03,0.05,0.12,0.25,0.30,0.15,0.10]',
            "clobTokenIds": '["tok1","tok2","tok3","tok4","tok5","tok6","tok7"]',
            "endDate": "2026-02-13T00:00:00Z",
            "description": "Resolution source: Weather Underground, station KDAL",
        },
    ]

    scanner = WeatherMarketScanner()

    with patch("aiohttp.ClientSession") as mock_session_cls:
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=mock_markets)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session_cls.return_value = mock_session

        markets = await scanner.scan()

    assert len(markets) >= 1
    m = markets[0]
    assert m.city == "Dallas"
    assert m.condition_id == "0xabc123"
    assert "58°F or higher" in m.outcome_prices
    assert m.outcome_prices["58°F or higher"] == 0.03


# ---------------------------------------------------------------------------
# WeatherOracleEngine tests
# ---------------------------------------------------------------------------
import time
from src.arb.weather_oracle import (
    WeatherOracleEngine,
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
    """Test confidence scoring for 'X degrees F or lower' outcomes."""
    engine = WeatherOracleEngine.__new__(WeatherOracleEngine)

    # Forecast well below threshold -> high confidence
    conf = engine._outcome_confidence("47°F or lower", 38.0)
    assert conf >= 0.90

    # Forecast above threshold -> 0
    conf = engine._outcome_confidence("47°F or lower", 50.0)
    assert conf == 0.0
