# Weather Oracle Strategy — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Automated strategy that scans Polymarket weather markets, fetches forecasts from Open-Meteo (ECMWF), detects mispriced outcomes, and buys cheap lottery tickets ($2–5) for 100x+ returns.

**Architecture:** An engine (`WeatherOracleEngine`) polls Polymarket Gamma API for active weather markets, fetches temperature forecasts from Open-Meteo using exact airport-station coordinates, compares forecast vs market prices, and enters paper trades when a market outcome is near-certain but still priced cheap. Trades persist to the existing `LiveObservation` + `PaperTrade` DB tables using `event_type="weather_oracle"`. A dedicated Streamlit tab shows P&L, city breakdown, and forecast accuracy.

**Tech Stack:** Python 3.11+, aiohttp, Open-Meteo REST API (free, no key), Polymarket Gamma API, SQLAlchemy, Streamlit, Plotly

---

## Task 1: Settings & Constants

**Files:**
- Modify: `config/settings.py:140-170`
- Create: `src/arb/weather_oracle.py`

**Step 1: Add settings to `config/settings.py`**

Add before `model_config = {"env_file": ".env"}` (line 170):

```python
    # === Weather Oracle Strategy ===
    WEATHER_ORACLE_ENABLED: bool = True
    WEATHER_ORACLE_SCAN_INTERVAL: float = 300.0  # 5 min between scans
    WEATHER_ORACLE_GAMMA_URL: str = "https://gamma-api.polymarket.com"
    WEATHER_ORACLE_OPEN_METEO_URL: str = "https://api.open-meteo.com/v1/forecast"
    WEATHER_ORACLE_MAX_ENTRY_PRICE: float = 0.05  # max 5¢ entry
    WEATHER_ORACLE_MIN_FORECAST_CONFIDENCE: float = 0.90  # 90% confidence threshold
    WEATHER_ORACLE_PAPER_SIZE_USD: float = 3.0  # $3 per ticket
    WEATHER_ORACLE_MAX_DAILY_SPEND: float = 50.0  # max $50/day
    WEATHER_ORACLE_FORECAST_DAYS: int = 7  # look ahead 7 days
    WEATHER_ORACLE_PAPER_FILE: str = "data/weather_oracle_paper.jsonl"
```

**Step 2: Create `src/arb/weather_oracle.py` with constants and dataclasses**

```python
"""Weather Oracle Strategy.

Buys ultra-cheap outcomes (1-5¢) on Polymarket weather markets
when Open-Meteo forecasts indicate near-certain resolution.
Uses exact airport station coordinates to match Weather Underground
resolution source.

Edge: weather forecasts are accurate 3-7 days out, but Polymarket
markets often don't fully price in forecast data.
"""

import asyncio
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import structlog

from config.settings import settings
from src.db.database import get_sync_session, init_db
from src.db.models import LiveObservation
from src.db.models import PaperTrade as PaperTradeDB

logger = structlog.get_logger()

WEATHER_ORACLE_EVENT_TYPE = "weather_oracle"

# Airport station coordinates used by Polymarket (Weather Underground).
# Keys match city names as they appear in Polymarket market titles.
CITY_STATIONS: dict[str, dict[str, Any]] = {
    "Dallas": {
        "lat": 32.8471, "lon": -96.8518,
        "station": "KDAL", "tz": "America/Chicago", "unit": "fahrenheit",
    },
    "Miami": {
        "lat": 25.7959, "lon": -80.2870,
        "station": "KMIA", "tz": "America/New_York", "unit": "fahrenheit",
    },
    "Atlanta": {
        "lat": 33.6407, "lon": -84.4277,
        "station": "KATL", "tz": "America/New_York", "unit": "fahrenheit",
    },
    "New York": {
        "lat": 40.7769, "lon": -73.8740,
        "station": "KLGA", "tz": "America/New_York", "unit": "fahrenheit",
    },
    "Chicago": {
        "lat": 41.9742, "lon": -87.9073,
        "station": "KORD", "tz": "America/Chicago", "unit": "fahrenheit",
    },
    "Seattle": {
        "lat": 47.4502, "lon": -122.3088,
        "station": "KSEA", "tz": "America/Los_Angeles", "unit": "fahrenheit",
    },
    "London": {
        "lat": 51.5048, "lon": 0.0495,
        "station": "EGLC", "tz": "Europe/London", "unit": "celsius",
    },
    "Toronto": {
        "lat": 43.6777, "lon": -79.6248,
        "station": "CYYZ", "tz": "America/Toronto", "unit": "celsius",
    },
    "Seoul": {
        "lat": 37.4602, "lon": 126.4407,
        "station": "RKSI", "tz": "Asia/Seoul", "unit": "celsius",
    },
    "Buenos Aires": {
        "lat": -34.8222, "lon": -58.5358,
        "station": "SAEZ", "tz": "America/Argentina/Buenos_Aires", "unit": "celsius",
    },
    "Ankara": {
        "lat": 40.1281, "lon": 32.9951,
        "station": "LTAC", "tz": "Europe/Istanbul", "unit": "celsius",
    },
}


@dataclass
class WeatherMarket:
    """A Polymarket weather market discovered via Gamma API."""
    condition_id: str
    slug: str
    title: str
    city: str
    target_date: str  # "YYYY-MM-DD"
    outcomes: dict[str, str]  # outcome_label -> token_id
    outcome_prices: dict[str, float]  # outcome_label -> price
    end_date: str  # ISO datetime
    resolution_source: str  # e.g. "Weather Underground"


@dataclass
class ForecastData:
    """Temperature forecast for a city on a specific date."""
    city: str
    date: str  # "YYYY-MM-DD"
    temp_max: float  # in the unit for that city (F or C)
    temp_min: float
    unit: str  # "fahrenheit" or "celsius"
    fetched_at: float  # unix timestamp


@dataclass
class WeatherSignal:
    """A detected entry signal on a weather market."""
    market: WeatherMarket
    outcome: str  # the outcome label to buy
    entry_price: float
    forecast: ForecastData
    confidence: float  # how certain the forecast makes this outcome
    reason: str  # human-readable explanation


@dataclass
class WeatherPaperTrade:
    """A paper trade record for the weather oracle strategy."""
    id: str = ""
    timestamp: str = ""
    city: str = ""
    target_date: str = ""
    market_slug: str = ""
    condition_id: str = ""
    outcome: str = ""
    entry_price: float = 0.0
    size_usd: float = 0.0
    forecast_temp_max: float = 0.0
    forecast_temp_min: float = 0.0
    confidence: float = 0.0
    reason: str = ""
    resolved: bool = False
    won: bool = False
    pnl_usd: float = 0.0
    resolution_price: float = 0.0
    resolution_time: str = ""
```

**Step 3: Run tests to check import**

```bash
python -c "from src.arb.weather_oracle import CITY_STATIONS, WEATHER_ORACLE_EVENT_TYPE; print('OK')"
```

Expected: `OK`

**Step 4: Commit**

```bash
git add config/settings.py src/arb/weather_oracle.py
git commit -m "feat(weather_oracle): add settings, constants, and dataclasses"
```

---

## Task 2: Open-Meteo Forecast Fetcher

**Files:**
- Modify: `src/arb/weather_oracle.py`
- Create: `tests/arb/test_weather_oracle.py`

**Step 1: Write the failing test**

```python
# tests/arb/test_weather_oracle.py
"""Tests for the Weather Oracle strategy."""

import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from src.arb.weather_oracle import (
    OpenMeteoFetcher,
    ForecastData,
    CITY_STATIONS,
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
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/arb/test_weather_oracle.py -v
```

Expected: FAIL (OpenMeteoFetcher not defined)

**Step 3: Implement `OpenMeteoFetcher`**

Add to `src/arb/weather_oracle.py` after the dataclasses:

```python
class OpenMeteoFetcher:
    """Fetches temperature forecasts from Open-Meteo (free, no API key)."""

    def __init__(self, base_url: str = ""):
        self._base_url = base_url or settings.WEATHER_ORACLE_OPEN_METEO_URL
        self._cache: dict[str, list[ForecastData]] = {}  # city -> forecasts
        self._last_fetch: dict[str, float] = {}  # city -> timestamp

    async def fetch_city(self, city: str) -> list[ForecastData]:
        """Fetch daily high/low forecast for a city using airport coordinates."""
        import aiohttp

        station = CITY_STATIONS.get(city)
        if not station:
            logger.warning("unknown_city", city=city)
            return []

        params = {
            "latitude": station["lat"],
            "longitude": station["lon"],
            "daily": "temperature_2m_max,temperature_2m_min",
            "temperature_unit": station["unit"],
            "timezone": station["tz"],
            "forecast_days": settings.WEATHER_ORACLE_FORECAST_DAYS,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self._base_url,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status != 200:
                        logger.warning("open_meteo_error", city=city, status=resp.status)
                        return []
                    data = await resp.json()
        except Exception as e:
            logger.warning("open_meteo_fetch_error", city=city, error=str(e))
            return []

        return self._parse_response(city, station["unit"], data)

    def _parse_response(
        self, city: str, unit: str, data: dict[str, Any],
    ) -> list[ForecastData]:
        """Parse Open-Meteo daily response into ForecastData list."""
        daily = data.get("daily", {})
        dates = daily.get("time", [])
        maxes = daily.get("temperature_2m_max", [])
        mins = daily.get("temperature_2m_min", [])

        now = time.time()
        forecasts = []
        for i, date_str in enumerate(dates):
            if i < len(maxes) and i < len(mins):
                forecasts.append(ForecastData(
                    city=city,
                    date=date_str,
                    temp_max=maxes[i],
                    temp_min=mins[i],
                    unit=unit,
                    fetched_at=now,
                ))

        self._cache[city] = forecasts
        self._last_fetch[city] = now
        return forecasts

    async def fetch_all_cities(self) -> dict[str, list[ForecastData]]:
        """Fetch forecasts for all known cities."""
        results: dict[str, list[ForecastData]] = {}
        for city in CITY_STATIONS:
            forecasts = await self.fetch_city(city)
            if forecasts:
                results[city] = forecasts
        return results

    def get_forecast(self, city: str, date: str) -> Optional[ForecastData]:
        """Get cached forecast for a city on a specific date."""
        for fc in self._cache.get(city, []):
            if fc.date == date:
                return fc
        return None
```

**Step 4: Run tests**

```bash
pytest tests/arb/test_weather_oracle.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/arb/weather_oracle.py tests/arb/test_weather_oracle.py
git commit -m "feat(weather_oracle): add Open-Meteo forecast fetcher with tests"
```

---

## Task 3: Polymarket Weather Market Scanner

**Files:**
- Modify: `src/arb/weather_oracle.py`
- Modify: `tests/arb/test_weather_oracle.py`

**Step 1: Write the failing test**

Add to `tests/arb/test_weather_oracle.py`:

```python
from src.arb.weather_oracle import WeatherMarketScanner, WeatherMarket


@pytest.mark.asyncio
async def test_scanner_parses_weather_markets():
    """Scanner should parse Gamma API response into WeatherMarket objects."""
    mock_markets = [
        {
            "conditionId": "0xabc123",
            "slug": "highest-temperature-in-dallas-on-february-12",
            "question": "Highest temperature in Dallas on February 12?",
            "outcomes": '["58°F or higher","56-57°F","54-55°F","52-53°F","50-51°F","48-49°F","47°F or lower"]',
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
    # The method needs current year context, so we test the pattern
    result = scanner._extract_target_date("Highest temperature in Dallas on February 12?")
    assert result is not None
    assert result.endswith("-02-12")
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/arb/test_weather_oracle.py::test_scanner_parses_weather_markets -v
```

Expected: FAIL (WeatherMarketScanner not defined)

**Step 3: Implement `WeatherMarketScanner`**

Add to `src/arb/weather_oracle.py`:

```python
import re
from datetime import date as date_type


class WeatherMarketScanner:
    """Discovers active weather markets on Polymarket via Gamma API."""

    # Pattern: "Highest temperature in <City> on <Month> <Day>"
    _TITLE_PATTERN = re.compile(
        r"(?:Highest|High)\s+temperature\s+in\s+(.+?)\s+on\s+(\w+)\s+(\d{1,2})",
        re.IGNORECASE,
    )

    _MONTH_MAP = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
    }

    def __init__(self, gamma_url: str = ""):
        self._gamma_url = gamma_url or settings.WEATHER_ORACLE_GAMMA_URL
        self._markets: dict[str, WeatherMarket] = {}  # condition_id -> market

    @property
    def markets(self) -> dict[str, WeatherMarket]:
        return dict(self._markets)

    async def scan(self) -> list[WeatherMarket]:
        """Fetch active weather markets from Gamma API."""
        import aiohttp

        discovered: list[WeatherMarket] = []
        search_terms = ["temperature", "fahrenheit", "celsius", "weather"]

        try:
            async with aiohttp.ClientSession() as session:
                for term in search_terms:
                    url = f"{self._gamma_url}/markets"
                    params = {"_q": term, "closed": "false", "limit": 100}
                    async with session.get(
                        url,
                        params=params,
                        headers={"User-Agent": "Mozilla/5.0"},
                        timeout=aiohttp.ClientTimeout(total=15),
                    ) as resp:
                        if resp.status != 200:
                            continue
                        raw_markets = await resp.json()

                        for mkt in raw_markets:
                            market = self._parse_market(mkt)
                            if market and market.condition_id not in self._markets:
                                self._markets[market.condition_id] = market
                                discovered.append(market)

        except Exception as e:
            logger.warning("weather_scan_error", error=str(e))

        return discovered

    def _parse_market(self, mkt: dict[str, Any]) -> Optional[WeatherMarket]:
        """Parse a Gamma API market into a WeatherMarket."""
        try:
            question = mkt.get("question", "") or mkt.get("title", "")
            city = self._extract_city(question)
            if not city:
                return None

            target_date = self._extract_target_date(question)
            if not target_date:
                return None

            condition_id = mkt.get("conditionId", "")
            slug = mkt.get("slug", "")
            outcomes_raw = json.loads(mkt.get("outcomes", "[]"))
            prices_raw = json.loads(mkt.get("outcomePrices", "[]"))
            tokens_raw = json.loads(mkt.get("clobTokenIds", "[]"))
            end_date = mkt.get("endDate", "")
            description = mkt.get("description", "")

            outcomes = {}
            outcome_prices = {}
            for i, outcome_label in enumerate(outcomes_raw):
                token_id = tokens_raw[i] if i < len(tokens_raw) else ""
                price = float(prices_raw[i]) if i < len(prices_raw) else 0.0
                outcomes[outcome_label] = token_id
                outcome_prices[outcome_label] = price

            return WeatherMarket(
                condition_id=condition_id,
                slug=slug,
                title=question,
                city=city,
                target_date=target_date,
                outcomes=outcomes,
                outcome_prices=outcome_prices,
                end_date=end_date,
                resolution_source=description,
            )
        except Exception as e:
            logger.warning("parse_weather_market_error", error=str(e))
            return None

    def _extract_city(self, title: str) -> Optional[str]:
        """Extract city name from market title."""
        match = self._TITLE_PATTERN.search(title)
        if not match:
            return None
        city_raw = match.group(1).strip()
        # Normalize to known city
        for known_city in CITY_STATIONS:
            if known_city.lower() == city_raw.lower():
                return known_city
        return None

    def _extract_target_date(self, title: str) -> Optional[str]:
        """Extract target date as YYYY-MM-DD from market title."""
        match = self._TITLE_PATTERN.search(title)
        if not match:
            return None
        month_str = match.group(2).lower()
        day = int(match.group(3))
        month = self._MONTH_MAP.get(month_str)
        if not month:
            return None
        year = datetime.now(timezone.utc).year
        try:
            target = date_type(year, month, day)
            return target.isoformat()
        except ValueError:
            return None
```

**Step 4: Run tests**

```bash
pytest tests/arb/test_weather_oracle.py -v
```

Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/arb/weather_oracle.py tests/arb/test_weather_oracle.py
git commit -m "feat(weather_oracle): add Polymarket weather market scanner"
```

---

## Task 4: Signal Detection Engine

**Files:**
- Modify: `src/arb/weather_oracle.py`
- Modify: `tests/arb/test_weather_oracle.py`

**Step 1: Write the failing test**

Add to `tests/arb/test_weather_oracle.py`:

```python
from src.arb.weather_oracle import (
    WeatherOracleEngine,
    WeatherMarket,
    ForecastData,
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
        outcomes={
            "58°F or higher": "tok1",
            "56-57°F": "tok2",
            "54-55°F": "tok3",
            "52-53°F": "tok4",
        },
        outcome_prices={
            "58°F or higher": 0.03,  # cheap!
            "56-57°F": 0.12,
            "54-55°F": 0.35,
            "52-53°F": 0.50,
        },
        end_date="2026-02-13T00:00:00Z",
        resolution_source="Weather Underground",
    )

    forecast = ForecastData(
        city="Dallas",
        date="2026-02-12",
        temp_max=62.5,  # well above 58°F
        temp_min=45.0,
        unit="fahrenheit",
        fetched_at=time.time(),
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
        city="Miami",
        date="2026-02-12",
        temp_max=75.0,
        temp_min=65.0,
        unit="fahrenheit",
        fetched_at=time.time(),
    )

    engine = WeatherOracleEngine.__new__(WeatherOracleEngine)
    engine.max_entry_price = 0.05
    engine.min_confidence = 0.90
    engine._entered_markets = set()

    signals = engine.evaluate_market(market, forecast)
    assert len(signals) == 0  # 70°F or higher is priced at 85¢ — too expensive


def test_engine_handles_range_outcomes():
    """Engine should detect when forecast lands in a specific range bucket."""
    market = WeatherMarket(
        condition_id="0xghi",
        slug="highest-temp-nyc-feb-12",
        title="Highest temperature in New York on February 12?",
        city="New York",
        target_date="2026-02-12",
        outcomes={
            "40°F or higher": "tok1",
            "38-39°F": "tok2",
            "36-37°F": "tok3",
        },
        outcome_prices={
            "40°F or higher": 0.60,
            "38-39°F": 0.02,  # cheap
            "36-37°F": 0.03,  # cheap
        },
        end_date="2026-02-13T00:00:00Z",
        resolution_source="Weather Underground",
    )

    forecast = ForecastData(
        city="New York",
        date="2026-02-12",
        temp_max=38.5,  # falls in 38-39°F range
        temp_min=28.0,
        unit="fahrenheit",
        fetched_at=time.time(),
    )

    engine = WeatherOracleEngine.__new__(WeatherOracleEngine)
    engine.max_entry_price = 0.05
    engine.min_confidence = 0.90
    engine._entered_markets = set()

    signals = engine.evaluate_market(market, forecast)
    # Should signal 38-39°F (cheap and matches forecast)
    outcome_labels = [s.outcome for s in signals]
    assert "38-39°F" in outcome_labels
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/arb/test_weather_oracle.py::test_engine_detects_cheap_certain_outcome -v
```

Expected: FAIL (WeatherOracleEngine not defined)

**Step 3: Implement `WeatherOracleEngine`**

Add to `src/arb/weather_oracle.py`:

```python
class WeatherOracleEngine:
    """Main engine: matches forecasts to markets and generates buy signals."""

    # Pattern for parsing outcome labels like "58°F or higher", "54-55°F", "47°F or lower"
    _THRESHOLD_HIGH = re.compile(r"(\d+)\s*°[FC]\s+or\s+higher", re.IGNORECASE)
    _THRESHOLD_LOW = re.compile(r"(\d+)\s*°[FC]\s+or\s+lower", re.IGNORECASE)
    _RANGE = re.compile(r"(\d+)\s*[-–]\s*(\d+)\s*°[FC]", re.IGNORECASE)
    _EXACT = re.compile(r"^(\d+)\s*°[FC]$", re.IGNORECASE)

    def __init__(
        self,
        fetcher: Optional[OpenMeteoFetcher] = None,
        scanner: Optional[WeatherMarketScanner] = None,
        database_url: str = "sqlite:///data/arb.db",
    ):
        self.fetcher = fetcher or OpenMeteoFetcher()
        self.scanner = scanner or WeatherMarketScanner()
        self._database_url = database_url

        self.max_entry_price = settings.WEATHER_ORACLE_MAX_ENTRY_PRICE
        self.min_confidence = settings.WEATHER_ORACLE_MIN_FORECAST_CONFIDENCE
        self.paper_size = settings.WEATHER_ORACLE_PAPER_SIZE_USD
        self.max_daily_spend = settings.WEATHER_ORACLE_MAX_DAILY_SPEND
        self.paper_file = Path(settings.WEATHER_ORACLE_PAPER_FILE)

        self._open_trades: list[WeatherPaperTrade] = []
        self._entered_markets: set[str] = set()  # "condition_id:outcome"
        self._daily_spend: float = 0.0
        self._daily_spend_date: str = ""

        self._stats = {"trades": 0, "wins": 0, "pnl": 0.0}

    def evaluate_market(
        self, market: WeatherMarket, forecast: ForecastData,
    ) -> list[WeatherSignal]:
        """Evaluate a single market against a forecast, return buy signals."""
        signals: list[WeatherSignal] = []
        temp_max = forecast.temp_max

        for outcome_label, price in market.outcome_prices.items():
            # Skip expensive outcomes
            if price > self.max_entry_price or price <= 0:
                continue

            # Skip already entered
            entry_key = f"{market.condition_id}:{outcome_label}"
            if entry_key in self._entered_markets:
                continue

            # Determine if forecast supports this outcome
            confidence = self._outcome_confidence(outcome_label, temp_max)

            if confidence >= self.min_confidence:
                signals.append(WeatherSignal(
                    market=market,
                    outcome=outcome_label,
                    entry_price=price,
                    forecast=forecast,
                    confidence=confidence,
                    reason=f"Forecast {temp_max:.0f}° → {outcome_label} (conf={confidence:.0%})",
                ))

        return signals

    def _outcome_confidence(self, outcome_label: str, forecast_temp: float) -> float:
        """Estimate confidence that a forecast supports this outcome.

        Uses simple heuristics based on forecast distance from thresholds.
        Open-Meteo ECMWF forecast MAE is ~3-4°F at 3 days, ~5-6°F at 7 days.
        We use 4°F as a conservative standard deviation.
        """
        forecast_std = 4.0  # °F typical forecast error

        # "X°F or higher" → confident if forecast is well above X
        match = self._THRESHOLD_HIGH.search(outcome_label)
        if match:
            threshold = float(match.group(1))
            margin = forecast_temp - threshold
            if margin <= 0:
                return 0.0
            # How many std devs above threshold?
            z_score = margin / forecast_std
            return min(0.99, 0.5 + 0.5 * min(z_score / 2.0, 1.0))

        # "X°F or lower" → confident if forecast is well below X
        match = self._THRESHOLD_LOW.search(outcome_label)
        if match:
            threshold = float(match.group(1))
            margin = threshold - forecast_temp
            if margin <= 0:
                return 0.0
            z_score = margin / forecast_std
            return min(0.99, 0.5 + 0.5 * min(z_score / 2.0, 1.0))

        # "X-Y°F" range → confident if forecast is clearly in range
        match = self._RANGE.search(outcome_label)
        if match:
            low = float(match.group(1))
            high = float(match.group(2))
            mid = (low + high) / 2.0
            # Distance from forecast to range midpoint
            dist_to_mid = abs(forecast_temp - mid)
            range_half = (high - low) / 2.0
            if dist_to_mid <= range_half:
                # Forecast is inside range
                return min(0.95, 0.6 + 0.35 * (1.0 - dist_to_mid / max(forecast_std, 1.0)))
            else:
                # Forecast is outside range
                return max(0.0, 0.5 - dist_to_mid / (2 * forecast_std))

        # "X°F" exact
        match = self._EXACT.search(outcome_label)
        if match:
            exact = float(match.group(1))
            dist = abs(forecast_temp - exact)
            if dist <= 0.5:
                return 0.85
            return max(0.0, 0.5 - dist / (2 * forecast_std))

        return 0.0
```

**Step 4: Run tests**

```bash
pytest tests/arb/test_weather_oracle.py -v
```

Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/arb/weather_oracle.py tests/arb/test_weather_oracle.py
git commit -m "feat(weather_oracle): add signal detection engine with confidence scoring"
```

---

## Task 5: Paper Trading & DB Persistence

**Files:**
- Modify: `src/arb/weather_oracle.py`
- Modify: `tests/arb/test_weather_oracle.py`

**Step 1: Write the failing test**

Add to `tests/arb/test_weather_oracle.py`:

```python
import time
from src.arb.weather_oracle import WeatherPaperTrade


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
    assert trade.city == "Dallas"
    assert trade.entry_price == 0.03
    assert trade.size_usd == 3.0
    assert "0xabc:58°F or higher" in engine._entered_markets
    assert len(engine._open_trades) == 1
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/arb/test_weather_oracle.py::test_engine_enter_paper_trade -v
```

Expected: FAIL (enter_paper_trade not defined)

**Step 3: Implement paper trading and persistence methods**

Add to `WeatherOracleEngine`:

```python
    def enter_paper_trade(self, signal: WeatherSignal) -> WeatherPaperTrade:
        """Record a paper trade entry."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._daily_spend_date != today:
            self._daily_spend = 0.0
            self._daily_spend_date = today

        if self._daily_spend + self.paper_size > self.max_daily_spend:
            logger.info("daily_spend_limit", spend=self._daily_spend, limit=self.max_daily_spend)
            return None

        trade = WeatherPaperTrade(
            id=str(uuid.uuid4())[:8],
            timestamp=datetime.now(timezone.utc).isoformat(),
            city=signal.market.city,
            target_date=signal.market.target_date,
            market_slug=signal.market.slug,
            condition_id=signal.market.condition_id,
            outcome=signal.outcome,
            entry_price=signal.entry_price,
            size_usd=self.paper_size,
            forecast_temp_max=signal.forecast.temp_max,
            forecast_temp_min=signal.forecast.temp_min,
            confidence=signal.confidence,
            reason=signal.reason,
        )

        self._open_trades.append(trade)
        entry_key = f"{signal.market.condition_id}:{signal.outcome}"
        self._entered_markets.add(entry_key)
        self._daily_spend += self.paper_size

        logger.info(
            "weather_paper_trade_entered",
            city=trade.city,
            outcome=trade.outcome,
            entry_price=trade.entry_price,
            confidence=trade.confidence,
            reason=trade.reason,
        )
        return trade

    async def resolve_trades(self) -> list[WeatherPaperTrade]:
        """Check open trades against current market prices for resolution."""
        resolved: list[WeatherPaperTrade] = []
        still_open: list[WeatherPaperTrade] = []

        for trade in self._open_trades:
            market = self.scanner.markets.get(trade.condition_id)
            if not market:
                still_open.append(trade)
                continue

            # Check if market end date has passed
            try:
                end_dt = datetime.fromisoformat(market.end_date.replace("Z", "+00:00"))
                if datetime.now(timezone.utc) < end_dt:
                    still_open.append(trade)
                    continue
            except (ValueError, AttributeError):
                still_open.append(trade)
                continue

            # Market has ended — check resolution price
            final_price = market.outcome_prices.get(trade.outcome, 0.0)

            # Resolved if price is near 0 or 1
            if final_price > 0.9:
                trade.won = True
                shares = trade.size_usd / trade.entry_price
                trade.pnl_usd = round(shares * (1.0 - trade.entry_price), 2)
            elif final_price < 0.1:
                trade.won = False
                trade.pnl_usd = round(-trade.size_usd, 2)
            else:
                # Not yet resolved
                still_open.append(trade)
                continue

            trade.resolved = True
            trade.resolution_price = final_price
            trade.resolution_time = datetime.now(timezone.utc).isoformat()

            self._stats["trades"] += 1
            if trade.won:
                self._stats["wins"] += 1
            self._stats["pnl"] += trade.pnl_usd

            self._save_trade(trade)
            resolved.append(trade)

            logger.info(
                "weather_trade_resolved",
                city=trade.city,
                outcome=trade.outcome,
                won=trade.won,
                pnl=trade.pnl_usd,
                entry=trade.entry_price,
            )

        self._open_trades = still_open
        return resolved

    def _save_trade(self, trade: WeatherPaperTrade) -> None:
        """Persist trade to DB (LiveObservation + PaperTrade) and JSONL backup."""
        now = datetime.now(timezone.utc)
        strategy_tag = "weather_oracle"

        game_state = {
            "strategy": "weather_oracle",
            "strategy_tag": strategy_tag,
            "city": trade.city,
            "target_date": trade.target_date,
            "condition_id": trade.condition_id,
            "outcome": trade.outcome,
            "side": "BUY",
            "slug": trade.market_slug,
            "forecast_temp_max": trade.forecast_temp_max,
            "forecast_temp_min": trade.forecast_temp_min,
            "confidence": trade.confidence,
            "reason": trade.reason,
            "title": f"{trade.city} {trade.target_date} → {trade.outcome}",
        }

        edge = trade.pnl_usd / trade.size_usd if trade.size_usd > 0 else 0.0

        observation = LiveObservation(
            timestamp=now,
            match_id=trade.condition_id,
            event_type=WEATHER_ORACLE_EVENT_TYPE,
            game_state=game_state,
            model_prediction=trade.confidence,
            polymarket_price=trade.entry_price,
        )

        db_trade = PaperTradeDB(
            observation_id=0,
            side="BUY",
            entry_price=trade.entry_price,
            simulated_fill_price=trade.entry_price,
            size=trade.size_usd,
            edge_theoretical=edge,
            edge_realized=edge,
            exit_price=1.0 if trade.won else 0.0,
            pnl=trade.pnl_usd,
            created_at=now,
        )

        try:
            session = get_sync_session(self._database_url)
            try:
                session.add(observation)
                session.flush()
                db_trade.observation_id = int(observation.id)
                session.add(db_trade)
                session.commit()
            except Exception:
                session.rollback()
                logger.warning("db_save_error", trade_id=trade.id)
            finally:
                session.close()
        except Exception as e:
            logger.warning("db_connect_error", error=str(e))

        # JSONL backup
        self.paper_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.paper_file, "a") as f:
            f.write(json.dumps(asdict(trade)) + "\n")

    def get_stats(self) -> dict[str, Any]:
        """Return current paper trading stats."""
        total = self._stats["trades"]
        wins = self._stats["wins"]
        return {
            "trades": total,
            "wins": wins,
            "winrate": round(wins / total * 100, 1) if total > 0 else 0.0,
            "pnl_usd": round(self._stats["pnl"], 2),
            "open": len(self._open_trades),
            "daily_spend": self._daily_spend,
        }
```

**Step 4: Run tests**

```bash
pytest tests/arb/test_weather_oracle.py -v
```

Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/arb/weather_oracle.py tests/arb/test_weather_oracle.py
git commit -m "feat(weather_oracle): add paper trading, resolution, and DB persistence"
```

---

## Task 6: Main Loop (run method)

**Files:**
- Modify: `src/arb/weather_oracle.py`

**Step 1: Add the `run` async method to `WeatherOracleEngine`**

```python
    async def run(self) -> None:
        """Main loop: scan markets, fetch forecasts, enter trades, resolve."""
        scan_interval = settings.WEATHER_ORACLE_SCAN_INTERVAL
        logger.info(
            "weather_oracle_started",
            scan_interval=scan_interval,
            max_entry_price=self.max_entry_price,
            min_confidence=self.min_confidence,
            paper_size=self.paper_size,
        )

        while True:
            try:
                # 1. Scan for weather markets
                new_markets = await self.scanner.scan()
                if new_markets:
                    logger.info("weather_markets_discovered", count=len(new_markets))

                # 2. Fetch forecasts for all cities with active markets
                cities_needed = {m.city for m in self.scanner.markets.values()}
                forecasts: dict[str, list[ForecastData]] = {}
                for city in cities_needed:
                    fc = await self.fetcher.fetch_city(city)
                    if fc:
                        forecasts[city] = fc

                # 3. Evaluate each market
                total_signals = 0
                for market in self.scanner.markets.values():
                    forecast = self.fetcher.get_forecast(market.city, market.target_date)
                    if not forecast:
                        continue

                    signals = self.evaluate_market(market, forecast)
                    for signal in signals:
                        trade = self.enter_paper_trade(signal)
                        if trade:
                            total_signals += 1

                if total_signals:
                    logger.info("weather_signals_entered", count=total_signals)

                # 4. Resolve expired trades
                resolved = await self.resolve_trades()
                if resolved:
                    logger.info("weather_trades_resolved", count=len(resolved))

                # 5. Log stats
                stats = self.get_stats()
                logger.info("weather_oracle_stats", **stats)

            except Exception as e:
                logger.error("weather_oracle_cycle_error", error=str(e))

            await asyncio.sleep(scan_interval)
```

**Step 2: Verify import still works**

```bash
python -c "from src.arb.weather_oracle import WeatherOracleEngine; print('OK')"
```

Expected: `OK`

**Step 3: Commit**

```bash
git add src/arb/weather_oracle.py
git commit -m "feat(weather_oracle): add main async run loop"
```

---

## Task 7: Runner Script

**Files:**
- Create: `scripts/run_weather_oracle.py`

**Step 1: Create the runner script**

```python
#!/usr/bin/env python3
"""Weather Oracle strategy runner.

Scans Polymarket weather markets, fetches Open-Meteo forecasts,
and buys cheap outcomes when forecasts indicate near-certainty.

Usage:
    python scripts/run_weather_oracle.py              # scan once
    python scripts/run_weather_oracle.py watch         # continuous
    python scripts/run_weather_oracle.py watch --interval 120
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog

from config.settings import settings
from src.arb.weather_oracle import (
    WeatherOracleEngine,
    OpenMeteoFetcher,
    WeatherMarketScanner,
)
from src.db.database import init_db

logger = structlog.get_logger()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Weather Oracle strategy runner")
    parser.add_argument("mode", choices=["scan", "watch"], default="scan", nargs="?")
    parser.add_argument(
        "--interval", type=float,
        default=settings.WEATHER_ORACLE_SCAN_INTERVAL,
        help="Watch mode polling interval (seconds).",
    )
    parser.add_argument(
        "--max-entry-price", type=float,
        default=settings.WEATHER_ORACLE_MAX_ENTRY_PRICE,
        help="Max entry price (e.g. 0.05 = 5 cents).",
    )
    parser.add_argument(
        "--min-confidence", type=float,
        default=settings.WEATHER_ORACLE_MIN_FORECAST_CONFIDENCE,
        help="Min forecast confidence to enter (0-1).",
    )
    parser.add_argument(
        "--paper-size", type=float,
        default=settings.WEATHER_ORACLE_PAPER_SIZE_USD,
        help="Dollar size per paper trade.",
    )
    parser.add_argument(
        "--max-daily-spend", type=float,
        default=settings.WEATHER_ORACLE_MAX_DAILY_SPEND,
        help="Max daily spend on paper trades.",
    )
    parser.add_argument(
        "--db-url", type=str,
        default="sqlite:///data/arb.db",
        help="Database URL for persistence.",
    )
    return parser


async def run_scan_once(engine: WeatherOracleEngine) -> None:
    """Run a single scan cycle and print results."""
    new_markets = await engine.scanner.scan()
    print(f"\nDiscovered {len(engine.scanner.markets)} weather markets")

    cities = {m.city for m in engine.scanner.markets.values()}
    for city in sorted(cities):
        await engine.fetcher.fetch_city(city)

    print(f"Fetched forecasts for {len(cities)} cities\n")

    total_signals = 0
    for market in engine.scanner.markets.values():
        forecast = engine.fetcher.get_forecast(market.city, market.target_date)
        if not forecast:
            continue

        signals = engine.evaluate_market(market, forecast)
        for signal in signals:
            print(
                f"  SIGNAL: {signal.market.city} {signal.market.target_date} "
                f"| {signal.outcome} @ {signal.entry_price:.3f} "
                f"| forecast={signal.forecast.temp_max:.0f}° "
                f"| conf={signal.confidence:.0%} "
                f"| {signal.reason}"
            )
            trade = engine.enter_paper_trade(signal)
            if trade:
                total_signals += 1

    print(f"\nTotal signals: {total_signals}")
    print(f"Open trades: {len(engine._open_trades)}")
    print(f"Stats: {engine.get_stats()}")


async def main():
    args = build_parser().parse_args()

    init_db(args.db_url)

    engine = WeatherOracleEngine(
        database_url=args.db_url,
    )
    engine.max_entry_price = args.max_entry_price
    engine.min_confidence = args.min_confidence
    engine.paper_size = args.paper_size
    engine.max_daily_spend = args.max_daily_spend

    if args.mode == "scan":
        await run_scan_once(engine)
    else:
        # Override scan interval from args
        original_interval = settings.WEATHER_ORACLE_SCAN_INTERVAL
        settings.WEATHER_ORACLE_SCAN_INTERVAL = args.interval  # type: ignore
        try:
            await engine.run()
        except KeyboardInterrupt:
            print("\nStopped.")
            print(f"Final stats: {engine.get_stats()}")


if __name__ == "__main__":
    asyncio.run(main())
```

**Step 2: Test it runs**

```bash
python scripts/run_weather_oracle.py scan 2>&1 | head -30
```

Expected: Should discover weather markets and print forecasts (or print 0 if none active).

**Step 3: Commit**

```bash
git add scripts/run_weather_oracle.py
git commit -m "feat(weather_oracle): add runner script with scan/watch modes"
```

---

## Task 8: Streamlit Dashboard Tab

**Files:**
- Modify: `src/paper_trading/dashboard.py`

**Step 1: Add the event type constant**

At line ~24, add:

```python
WEATHER_ORACLE_EVENT_TYPE = "weather_oracle"
```

**Step 2: Add `extract_weather_oracle_rows` helper**

Add after `extract_crypto_minute_rows` (around line 843):

```python
def extract_weather_oracle_rows(
    observations: list[LiveObservation],
    trades: list[PaperTrade],
) -> list[dict[str, Any]]:
    """Build rows for weather oracle strategy analysis."""
    obs_by_id = {obs.id: obs for obs in observations}
    rows: list[dict[str, Any]] = []

    for trade in trades:
        obs = obs_by_id.get(trade.observation_id)
        if obs is None or obs.event_type != WEATHER_ORACLE_EVENT_TYPE:
            continue
        gs = _observation_game_state(obs)
        rows.append({
            "timestamp": trade.created_at or obs.timestamp,
            "city": gs.get("city", ""),
            "target_date": gs.get("target_date", ""),
            "outcome": gs.get("outcome", ""),
            "entry_price": _safe_float(trade.entry_price),
            "exit_price": _safe_float(trade.exit_price),
            "size_usd": _safe_float(trade.size),
            "pnl": _safe_float(trade.pnl),
            "confidence": _safe_float(gs.get("confidence")),
            "forecast_temp_max": _safe_float(gs.get("forecast_temp_max")),
            "forecast_temp_min": _safe_float(gs.get("forecast_temp_min")),
            "reason": gs.get("reason", ""),
            "slug": gs.get("slug", ""),
            "won": _safe_float(trade.exit_price) >= 0.5,
        })
    return rows
```

**Step 3: Add `_render_weather_oracle_tab`**

Add after the `_render_crypto_minute_tab` function:

```python
def _render_weather_oracle_tab(
    observations: list[LiveObservation],
    trades: list[PaperTrade],
) -> None:
    """Render the Weather Oracle tab content."""
    rows = extract_weather_oracle_rows(observations, trades)

    if not rows:
        st.info("No weather oracle trades found. Run: python scripts/run_weather_oracle.py watch")
        return

    # Summary metrics
    pnls = [r["pnl"] for r in rows]
    total_pnl = sum(pnls)
    wins = sum(1 for p in pnls if p > 0)
    losses = sum(1 for p in pnls if p <= 0)
    total_invested = sum(r["size_usd"] for r in rows)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Trades", len(rows))
    c2.metric("Win Rate", f"{wins / len(rows):.1%}" if rows else "N/A")
    c3.metric("Total P&L", f"${total_pnl:,.2f}")
    c4.metric("ROI", f"{total_pnl / total_invested:.0%}" if total_invested else "N/A")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Wins / Losses", f"{wins} / {losses}")
    c6.metric("Total Invested", f"${total_invested:,.2f}")
    avg_win = sum(p for p in pnls if p > 0) / wins if wins else 0
    c7.metric("Avg Win", f"${avg_win:,.2f}")
    avg_conf = sum(r["confidence"] for r in rows) / len(rows) if rows else 0
    c8.metric("Avg Confidence", f"{avg_conf:.0%}")

    st.divider()

    # P&L by city
    st.subheader("By City")
    cities = sorted({r["city"] for r in rows if r["city"]})
    city_data: list[dict[str, Any]] = []
    for city in cities:
        c_rows = [r for r in rows if r["city"] == city]
        c_pnls = [r["pnl"] for r in c_rows]
        c_wins = sum(1 for p in c_pnls if p > 0)
        city_data.append({
            "City": city,
            "Trades": len(c_rows),
            "Win Rate": f"{c_wins / len(c_rows):.1%}" if c_rows else "N/A",
            "Total P&L": f"${sum(c_pnls):,.2f}",
            "Invested": f"${sum(r['size_usd'] for r in c_rows):,.2f}",
        })
    if city_data:
        st.dataframe(pd.DataFrame(city_data), use_container_width=True, hide_index=True)

    st.divider()

    # Cumulative P&L chart
    st.subheader("Cumulative P&L")
    sorted_rows = sorted(rows, key=lambda r: r.get("timestamp") or datetime.min)
    if sorted_rows:
        wo_df = pd.DataFrame(sorted_rows)
        wo_df = wo_df.sort_values("timestamp")
        wo_df["cumulative_pnl"] = wo_df["pnl"].cumsum()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=wo_df["timestamp"], y=wo_df["cumulative_pnl"],
            mode="lines+markers", name="Cumulative P&L",
            line=dict(color="#f39c12", width=2), marker=dict(size=5),
        ))
        fig.update_layout(
            xaxis_title="Time", yaxis_title="Cumulative P&L ($)",
            height=350, template="plotly_dark",
            margin=dict(l=50, r=50, t=30, b=50),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Recent trades table
    st.subheader("Recent Trades")
    recent = sorted(rows, key=lambda r: r.get("timestamp") or datetime.min, reverse=True)[:50]
    if recent:
        df = pd.DataFrame([{
            "Time": r["timestamp"].strftime("%m-%d %H:%M") if isinstance(r.get("timestamp"), datetime) else "N/A",
            "City": r["city"],
            "Date": r["target_date"],
            "Outcome": r["outcome"],
            "Entry": f"{r['entry_price']:.3f}",
            "Exit": f"{r['exit_price']:.3f}",
            "Won": "Y" if r["won"] else "N",
            "Size": f"${r['size_usd']:.2f}",
            "P&L": f"${r['pnl']:.2f}",
            "Conf": f"{r['confidence']:.0%}",
            "Forecast High": f"{r['forecast_temp_max']:.0f}°",
        } for r in recent])
        st.dataframe(df, use_container_width=True, hide_index=True)
```

**Step 4: Add the tab to `main()`**

In `main()` around line 1102, change:

```python
    tab_two_sided, tab_sniper, tab_crypto = st.tabs(["Two-Sided", "Sniper Sports", "Crypto Minute"])
```

to:

```python
    tab_two_sided, tab_sniper, tab_crypto, tab_weather = st.tabs(
        ["Two-Sided", "Sniper Sports", "Crypto Minute", "Weather Oracle"]
    )
```

Then add after the `with tab_crypto:` block:

```python
    with tab_weather:
        _render_weather_oracle_tab(observations, trades)
```

**Step 5: Verify dashboard loads**

```bash
python -c "from src.paper_trading.dashboard import extract_weather_oracle_rows; print('OK')"
```

Expected: `OK`

**Step 6: Commit**

```bash
git add src/paper_trading/dashboard.py
git commit -m "feat(weather_oracle): add Streamlit dashboard tab with city breakdown and P&L chart"
```

---

## Task 9: Integration Test

**Files:**
- Modify: `tests/arb/test_weather_oracle.py`

**Step 1: Write an end-to-end integration test**

```python
@pytest.mark.asyncio
async def test_full_cycle_scan_evaluate_enter():
    """Integration: scan → forecast → evaluate → enter trade."""
    # Mock Gamma API response
    mock_markets_resp = [
        {
            "conditionId": "0xintegration",
            "slug": "highest-temp-dallas-feb-15",
            "question": "Highest temperature in Dallas on February 15?",
            "outcomes": '["60°F or higher","58-59°F","56-57°F"]',
            "outcomePrices": '[0.02,0.08,0.15]',
            "clobTokenIds": '["t1","t2","t3"]',
            "endDate": "2026-02-16T00:00:00Z",
            "description": "Resolution source: Weather Underground",
        },
    ]

    # Mock Open-Meteo response
    mock_forecast_resp = {
        "daily": {
            "time": ["2026-02-15"],
            "temperature_2m_max": [65.0],
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

    with patch("aiohttp.ClientSession") as mock_cls:
        # We'll need to handle multiple calls (scanner + fetcher)
        mock_resp_markets = AsyncMock()
        mock_resp_markets.status = 200
        mock_resp_markets.json = AsyncMock(return_value=mock_markets_resp)
        mock_resp_markets.__aenter__ = AsyncMock(return_value=mock_resp_markets)
        mock_resp_markets.__aexit__ = AsyncMock(return_value=False)

        mock_resp_forecast = AsyncMock()
        mock_resp_forecast.status = 200
        mock_resp_forecast.json = AsyncMock(return_value=mock_forecast_resp)
        mock_resp_forecast.__aenter__ = AsyncMock(return_value=mock_resp_forecast)
        mock_resp_forecast.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        # First N calls are scanner (markets), then fetcher (forecast)
        mock_session.get = MagicMock(side_effect=[
            mock_resp_markets, mock_resp_markets, mock_resp_markets, mock_resp_markets,
            mock_resp_forecast,
        ])
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
```

**Step 2: Run all tests**

```bash
pytest tests/arb/test_weather_oracle.py -v
```

Expected: ALL PASS

**Step 3: Run full test suite**

```bash
pytest -x -q
```

Expected: No regressions.

**Step 4: Commit**

```bash
git add tests/arb/test_weather_oracle.py
git commit -m "test(weather_oracle): add integration test for full scan-evaluate-enter cycle"
```

---

## Summary

| Task | Component | Files |
|------|-----------|-------|
| 1 | Settings & dataclasses | `config/settings.py`, `src/arb/weather_oracle.py` |
| 2 | Open-Meteo fetcher | `src/arb/weather_oracle.py`, `tests/arb/test_weather_oracle.py` |
| 3 | Market scanner | `src/arb/weather_oracle.py`, `tests/arb/test_weather_oracle.py` |
| 4 | Signal engine | `src/arb/weather_oracle.py`, `tests/arb/test_weather_oracle.py` |
| 5 | Paper trading & DB | `src/arb/weather_oracle.py`, `tests/arb/test_weather_oracle.py` |
| 6 | Main async loop | `src/arb/weather_oracle.py` |
| 7 | Runner script | `scripts/run_weather_oracle.py` |
| 8 | Streamlit dashboard tab | `src/paper_trading/dashboard.py` |
| 9 | Integration test | `tests/arb/test_weather_oracle.py` |
