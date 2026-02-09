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
    assert "58\u00b0F or higher" in m.outcome_prices
    assert m.outcome_prices["58\u00b0F or higher"] == 0.03
