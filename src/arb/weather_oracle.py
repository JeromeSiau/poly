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
