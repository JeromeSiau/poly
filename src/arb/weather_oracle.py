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
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date as date_type
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import structlog

from config.settings import settings

if TYPE_CHECKING:
    from src.execution.trade_manager import TradeManager
    from src.risk.guard import RiskGuard

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
        "slug_name": "nyc",
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

FORECAST_MODELS = ["best_match", "ecmwf_ifs025", "gfs_seamless"]


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
    model_temps: dict[str, float] = field(default_factory=dict)  # model_name -> temp_max


@dataclass
class WeatherSignal:
    """A detected entry signal on a weather market."""
    market: WeatherMarket
    outcome: str  # the outcome label to buy
    entry_price: float
    forecast: ForecastData
    confidence: float  # how certain the forecast makes this outcome
    reason: str  # human-readable explanation
    side: str = "BUY_YES"  # "BUY_YES" or "BUY_NO"
    trade_type: str = "lottery"  # "lottery", "yield_yes", "yield_no"


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
    side: str = "BUY_YES"  # "BUY_YES" or "BUY_NO"
    trade_type: str = "lottery"  # "lottery", "yield_yes", "yield_no"
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
            "models": ",".join(FORECAST_MODELS),
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

        # Collect per-model temp_max arrays
        model_maxes: dict[str, list[float]] = {}
        for model in FORECAST_MODELS:
            key = f"temperature_2m_max_{model}"
            if key in daily:
                model_maxes[model] = daily[key]

        now = time.time()
        forecasts = []
        for i, date_str in enumerate(dates):
            if i < len(maxes) and i < len(mins):
                # Build per-model temps for this day
                model_temps: dict[str, float] = {}
                for model, model_vals in model_maxes.items():
                    if i < len(model_vals) and model_vals[i] is not None:
                        model_temps[model] = model_vals[i]

                # Use average of models if available, otherwise the default
                if model_temps:
                    avg_max = sum(model_temps.values()) / len(model_temps)
                else:
                    avg_max = maxes[i]

                forecasts.append(ForecastData(
                    city=city,
                    date=date_str,
                    temp_max=avg_max,
                    temp_min=mins[i],
                    unit=unit,
                    fetched_at=now,
                    model_temps=model_temps,
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


class WeatherMarketScanner:
    """Discovers active weather markets on Polymarket via Gamma events API.

    Polymarket weather markets are structured as *events*, each containing
    multiple Yes/No sub-markets (one per temperature outcome).  The Gamma
    ``_q`` full-text search does **not** reliably return weather events, so
    we construct deterministic slugs for each city × date combination and
    query ``/events?slug=<slug>`` directly.
    """

    # Extract outcome label from sub-market question, e.g.
    # "Will the highest temperature in Dallas be 61°F or below on February 10?"
    # → "61°F or below"
    _OUTCOME_EXTRACT = re.compile(
        r"\bbe\s+(.+?)\s+on\s+\w+\s+\d+",
        re.IGNORECASE,
    )

    def __init__(self, gamma_url: str = ""):
        self._gamma_url = gamma_url or settings.WEATHER_ORACLE_GAMMA_URL
        self._markets: dict[str, WeatherMarket] = {}

    @property
    def markets(self) -> dict[str, WeatherMarket]:
        return dict(self._markets)

    def _generate_event_slugs(self) -> list[tuple[str, str, str]]:
        """Build ``(slug, city, YYYY-MM-DD)`` for each city × upcoming date."""
        today = datetime.now(timezone.utc).date()
        results: list[tuple[str, str, str]] = []
        for city, info in CITY_STATIONS.items():
            slug_name = info.get("slug_name", city.lower().replace(" ", "-"))
            for offset in range(0, settings.WEATHER_ORACLE_FORECAST_DAYS + 1):
                target = today + timedelta(days=offset)
                month_name = target.strftime("%B").lower()
                slug = (
                    f"highest-temperature-in-{slug_name}"
                    f"-on-{month_name}-{target.day}-{target.year}"
                )
                results.append((slug, city, target.isoformat()))
        return results

    async def scan(self) -> list[WeatherMarket]:
        """Fetch active weather events from Gamma API."""
        import aiohttp

        slugs = self._generate_event_slugs()
        discovered: list[WeatherMarket] = []
        sem = asyncio.Semaphore(5)

        async with aiohttp.ClientSession(
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=aiohttp.ClientTimeout(total=15),
        ) as session:

            async def _fetch_one(
                slug: str, city: str, date_str: str,
            ) -> Optional[WeatherMarket]:
                async with sem:
                    try:
                        url = f"{self._gamma_url}/events"
                        params = {"slug": slug}
                        async with session.get(url, params=params) as resp:
                            if resp.status != 200:
                                return None
                            data = await resp.json()
                            if not data:
                                return None
                            event = data[0] if isinstance(data, list) else data
                            return self._parse_event(event, city, date_str)
                    except Exception:
                        return None

            tasks = [_fetch_one(s, c, d) for s, c, d in slugs]
            results = await asyncio.gather(*tasks)

        for market in results:
            if not market:
                continue
            if market.condition_id in self._markets:
                self._markets[market.condition_id].outcome_prices = (
                    market.outcome_prices
                )
            else:
                self._markets[market.condition_id] = market
                discovered.append(market)

        logger.info(
            "weather_scan_complete",
            slugs_queried=len(slugs),
            new=len(discovered),
            total=len(self._markets),
        )
        return discovered

    def _parse_event(
        self, event: dict[str, Any], city: str, date_str: str,
    ) -> Optional[WeatherMarket]:
        """Turn a Gamma event (with sub-markets) into a WeatherMarket."""
        sub_markets = event.get("markets", [])
        if not sub_markets:
            return None

        outcomes: dict[str, str] = {}
        outcome_prices: dict[str, float] = {}

        for sm in sub_markets:
            question = sm.get("question", "")
            match = self._OUTCOME_EXTRACT.search(question)
            if not match:
                continue
            outcome_label = match.group(1).strip()

            prices_raw = json.loads(sm.get("outcomePrices", "[]"))
            tokens_raw = json.loads(sm.get("clobTokenIds", "[]"))

            yes_price = float(prices_raw[0]) if prices_raw else 0.0
            yes_token = tokens_raw[0] if tokens_raw else ""

            outcomes[outcome_label] = yes_token
            outcome_prices[outcome_label] = yes_price

        if not outcomes:
            return None

        event_slug = event.get("slug", "")
        event_title = event.get("title", "")
        end_date = event.get("endDate", "")
        if not end_date and sub_markets:
            end_date = sub_markets[0].get("endDate", "")
        description = event.get("description", "")

        return WeatherMarket(
            condition_id=event_slug,
            slug=event_slug,
            title=event_title,
            city=city,
            target_date=date_str,
            outcomes=outcomes,
            outcome_prices=outcome_prices,
            end_date=end_date,
            resolution_source=description,
        )


class WeatherOracleEngine:
    """Main engine: matches forecasts to markets and generates buy signals."""

    _THRESHOLD_HIGH = re.compile(r"(-?\d+)\s*°[FC]\s+or\s+higher", re.IGNORECASE)
    _THRESHOLD_LOW = re.compile(r"(-?\d+)\s*°[FC]\s+or\s+(?:lower|below)", re.IGNORECASE)
    _RANGE = re.compile(r"(?:between\s+)?(-?\d+)\s*[-–]\s*(-?\d+)\s*°[FC]", re.IGNORECASE)
    _EXACT = re.compile(r"^(-?\d+)\s*°[FC]$", re.IGNORECASE)

    def __init__(
        self,
        fetcher=None,
        scanner=None,
        database_url: str = "",
        manager: Optional["TradeManager"] = None,
        guard: Optional["RiskGuard"] = None,
    ):
        self.fetcher = fetcher or OpenMeteoFetcher()
        self.scanner = scanner or WeatherMarketScanner()
        self._database_url = database_url
        self.manager = manager
        self.guard = guard
        self._last_book_update: float = time.time()

        # Days-to-resolution filter
        self.min_days_to_resolution = settings.WEATHER_ORACLE_MIN_DAYS_TO_RESOLUTION
        self.max_days_to_resolution = settings.WEATHER_ORACLE_MAX_DAYS_TO_RESOLUTION

        # Lottery YES (Type 3) – cheap tail bets
        self.max_entry_price = settings.WEATHER_ORACLE_MAX_ENTRY_PRICE
        self.min_confidence = settings.WEATHER_ORACLE_MIN_FORECAST_CONFIDENCE
        self.paper_size = settings.WEATHER_ORACLE_PAPER_SIZE_USD

        # Lottery NO (Type 4) – the 0x594ed strategy: buy NO at <5¢ on mispriced markets
        self.lottery_no_enabled = settings.WEATHER_ORACLE_LOTTERY_NO_ENABLED
        self.lottery_no_size = settings.WEATHER_ORACLE_LOTTERY_NO_SIZE_USD
        self.lottery_no_min_yes_price = settings.WEATHER_ORACLE_LOTTERY_NO_MIN_YES_PRICE
        self.lottery_no_min_no_confidence = settings.WEATHER_ORACLE_LOTTERY_NO_MIN_NO_CONFIDENCE

        # Yield YES (Type 1) – buy likely outcome at 80-97¢
        self.yield_enabled = settings.WEATHER_ORACLE_YIELD_ENABLED
        self.yield_size = settings.WEATHER_ORACLE_YIELD_SIZE_USD
        self.yield_min_confidence = settings.WEATHER_ORACLE_YIELD_MIN_CONFIDENCE
        self.yield_min_yes_price = settings.WEATHER_ORACLE_YIELD_MIN_YES_PRICE
        self.yield_max_yes_price = settings.WEATHER_ORACLE_YIELD_MAX_YES_PRICE

        # Yield NO (Type 2) – buy NO on unlikely outcomes
        self.no_enabled = settings.WEATHER_ORACLE_NO_ENABLED
        self.no_size = settings.WEATHER_ORACLE_NO_SIZE_USD
        self.no_max_confidence = settings.WEATHER_ORACLE_NO_MAX_CONFIDENCE
        self.no_max_yes_price = settings.WEATHER_ORACLE_NO_MAX_YES_PRICE

        self.max_daily_spend = settings.WEATHER_ORACLE_MAX_DAILY_SPEND
        self.paper_file = Path(settings.WEATHER_ORACLE_PAPER_FILE)

        self._open_trades: list[WeatherPaperTrade] = []
        self._entered_markets: set[str] = set()  # "condition_id:side:outcome"
        self._daily_spend: float = 0.0
        self._daily_spend_date: str = ""

        self._stats = {"trades": 0, "wins": 0, "pnl": 0.0}

    def _days_to_resolution(self, market: WeatherMarket) -> Optional[float]:
        """Calculate days until market resolves. Returns None if unparseable."""
        try:
            end_str = market.end_date.replace("Z", "+00:00") if market.end_date else ""
            if not end_str:
                return None
            end_dt = datetime.fromisoformat(end_str)
            now = datetime.now(timezone.utc)
            return (end_dt - now).total_seconds() / 86400.0
        except (ValueError, AttributeError):
            return None

    def evaluate_market(
        self, market: WeatherMarket, forecast: ForecastData,
    ) -> list[WeatherSignal]:
        """Evaluate a single market against a forecast, return buy signals.

        Generates up to 4 signal types per outcome:
        - Type 3 (lottery YES): cheap YES on high-confidence tail outcomes
        - Type 4 (lottery NO): cheap NO when market is wrong (the 0x594ed strategy)
        - Type 1 (yield YES): buy YES at 80-97¢ on the most likely outcome
        - Type 2 (yield NO): buy NO on very unlikely outcomes (YES ≤ 5¢)
        """
        # Days-to-resolution filter: only trade in the sweet spot
        days = self._days_to_resolution(market)
        if days is not None:
            if days < self.min_days_to_resolution or days > self.max_days_to_resolution:
                return []

        signals: list[WeatherSignal] = []
        temp_max = forecast.temp_max

        for outcome_label, yes_price in market.outcome_prices.items():
            if yes_price <= 0:
                continue

            confidence = self._outcome_confidence(outcome_label, temp_max)

            # --- Type 3: Lottery YES (cheap tail bet) ---
            if yes_price <= self.max_entry_price:
                key = f"{market.condition_id}:BUY_YES:lottery:{outcome_label}"
                if (
                    key not in self._entered_markets
                    and confidence >= self.min_confidence
                    and self._check_consensus(forecast, outcome_label, self.min_confidence, require_all=False)
                ):
                    signals.append(WeatherSignal(
                        market=market,
                        outcome=outcome_label,
                        entry_price=yes_price,
                        forecast=forecast,
                        confidence=confidence,
                        reason=f"Lottery YES {temp_max:.0f}° → {outcome_label} @{yes_price:.3f} (conf={confidence:.0%})",
                        side="BUY_YES",
                        trade_type="lottery",
                    ))

            # --- Type 4: Lottery NO (the 0x594ed strategy) ---
            # Market prices YES at 95%+ but forecast says outcome is unlikely.
            # Buy NO at <5¢ for massive asymmetric upside.
            if self.lottery_no_enabled:
                no_price = 1.0 - yes_price
                if (
                    yes_price >= self.lottery_no_min_yes_price
                    and no_price <= self.max_entry_price
                ):
                    no_confidence = self._outcome_no_confidence(outcome_label, temp_max)
                    key = f"{market.condition_id}:BUY_NO:lottery_no:{outcome_label}"
                    if (
                        key not in self._entered_markets
                        and no_confidence >= self.lottery_no_min_no_confidence
                        and self._check_no_consensus(forecast, outcome_label)
                    ):
                        signals.append(WeatherSignal(
                            market=market,
                            outcome=outcome_label,
                            entry_price=no_price,
                            forecast=forecast,
                            confidence=no_confidence,
                            reason=(
                                f"Lottery NO {temp_max:.0f}° → {outcome_label} "
                                f"YES@{yes_price:.3f} NO@{no_price:.3f} "
                                f"(no_conf={no_confidence:.0%})"
                            ),
                            side="BUY_NO",
                            trade_type="lottery_no",
                        ))

            # --- Type 1: Yield YES (buy likely outcome at 80-97¢) ---
            if self.yield_enabled:
                if self.yield_min_yes_price <= yes_price <= self.yield_max_yes_price:
                    key = f"{market.condition_id}:BUY_YES:yield_yes:{outcome_label}"
                    if (
                        key not in self._entered_markets
                        and confidence >= self.yield_min_confidence
                        and self._check_consensus(forecast, outcome_label, self.yield_min_confidence, require_all=True)
                    ):
                        signals.append(WeatherSignal(
                            market=market,
                            outcome=outcome_label,
                            entry_price=yes_price,
                            forecast=forecast,
                            confidence=confidence,
                            reason=f"Yield YES {temp_max:.0f}° → {outcome_label} @{yes_price:.3f} (conf={confidence:.0%})",
                            side="BUY_YES",
                            trade_type="yield_yes",
                        ))

            # --- Type 2: Yield NO (buy NO on unlikely outcomes) ---
            if self.no_enabled:
                if (
                    yes_price <= self.no_max_yes_price
                    and confidence <= self.no_max_confidence
                    and self._check_no_consensus(forecast, outcome_label)
                ):
                    no_price = 1.0 - yes_price
                    key = f"{market.condition_id}:BUY_NO:yield_no:{outcome_label}"
                    if key not in self._entered_markets:
                        signals.append(WeatherSignal(
                            market=market,
                            outcome=outcome_label,
                            entry_price=no_price,
                            forecast=forecast,
                            confidence=confidence,
                            reason=f"Yield NO {temp_max:.0f}° → {outcome_label} YES@{yes_price:.3f} NO@{no_price:.3f} (conf={confidence:.0%})",
                            side="BUY_NO",
                            trade_type="yield_no",
                        ))

        return signals

    def _outcome_confidence(self, outcome_label: str, forecast_temp: float) -> float:
        """Estimate confidence that a forecast supports this outcome.

        Uses simple heuristics based on forecast distance from thresholds.
        Open-Meteo ECMWF forecast MAE is ~3-4°F at 3 days, ~5-6°F at 7 days.
        We use 4°F (or ~2.2°C) as a conservative standard deviation.
        """
        is_celsius = "°C" in outcome_label or "°c" in outcome_label
        forecast_std = 2.2 if is_celsius else 4.0

        match = self._THRESHOLD_HIGH.search(outcome_label)
        if match:
            threshold = float(match.group(1))
            margin = forecast_temp - threshold
            if margin <= 0:
                return 0.0
            z_score = margin / forecast_std
            return min(0.99, 0.5 + 0.5 * min(z_score / 2.0, 1.0))

        match = self._THRESHOLD_LOW.search(outcome_label)
        if match:
            threshold = float(match.group(1))
            margin = threshold - forecast_temp
            if margin <= 0:
                return 0.0
            z_score = margin / forecast_std
            return min(0.99, 0.5 + 0.5 * min(z_score / 2.0, 1.0))

        match = self._RANGE.search(outcome_label)
        if match:
            low = float(match.group(1))
            high = float(match.group(2))
            mid = (low + high) / 2.0
            dist_to_mid = abs(forecast_temp - mid)
            range_half = (high - low) / 2.0
            if dist_to_mid <= range_half:
                return min(0.95, 0.6 + 0.35 * (1.0 - dist_to_mid / max(forecast_std, 1.0)))
            else:
                return max(0.0, 0.5 - dist_to_mid / (2 * forecast_std))

        match = self._EXACT.search(outcome_label)
        if match:
            exact = float(match.group(1))
            dist = abs(forecast_temp - exact)
            if dist <= 0.5:
                return 0.85
            return max(0.0, 0.5 - dist / (2 * forecast_std))

        return 0.0

    def _outcome_no_confidence(self, outcome_label: str, forecast_temp: float) -> float:
        """Confidence that this outcome will NOT happen (for lottery NO trades).

        Mirrors _outcome_confidence but measures how far the forecast is on
        the WRONG side of the threshold. Used by Type 4 (Lottery NO) to
        confirm that buying NO is supported by forecast data.
        """
        is_celsius = "°C" in outcome_label or "°c" in outcome_label
        forecast_std = 2.2 if is_celsius else 4.0

        # "X°F or higher" → NO wins when temp is BELOW X
        match = self._THRESHOLD_HIGH.search(outcome_label)
        if match:
            threshold = float(match.group(1))
            margin_below = threshold - forecast_temp
            if margin_below <= 0:
                return 0.0  # forecast is above threshold, YES might happen
            z_score = margin_below / forecast_std
            return min(0.99, 0.5 + 0.5 * min(z_score / 2.0, 1.0))

        # "X°F or below" → NO wins when temp is ABOVE X
        match = self._THRESHOLD_LOW.search(outcome_label)
        if match:
            threshold = float(match.group(1))
            margin_above = forecast_temp - threshold
            if margin_above <= 0:
                return 0.0  # forecast is below threshold, YES might happen
            z_score = margin_above / forecast_std
            return min(0.99, 0.5 + 0.5 * min(z_score / 2.0, 1.0))

        # Range "X-Y°F" → NO wins when temp is OUTSIDE the range
        match = self._RANGE.search(outcome_label)
        if match:
            low = float(match.group(1))
            high = float(match.group(2))
            mid = (low + high) / 2.0
            range_half = (high - low) / 2.0
            dist_to_mid = abs(forecast_temp - mid)
            if dist_to_mid <= range_half:
                return 0.0  # forecast is IN the range, YES might happen
            excess = dist_to_mid - range_half
            z_score = excess / forecast_std
            return min(0.99, 0.5 + 0.5 * min(z_score / 2.0, 1.0))

        # Exact "X°F" → NO wins when temp is NOT at this value
        match = self._EXACT.search(outcome_label)
        if match:
            exact = float(match.group(1))
            dist = abs(forecast_temp - exact)
            if dist <= 0.5:
                return 0.0  # forecast is at this exact temp
            z_score = (dist - 0.5) / forecast_std
            return min(0.99, 0.5 + 0.5 * min(z_score / 2.0, 1.0))

        return 0.0

    def _check_consensus(
        self,
        forecast: ForecastData,
        outcome_label: str,
        min_confidence: float,
        require_all: bool = True,
    ) -> bool:
        """Check model consensus that an outcome WILL happen.

        Args:
            forecast: Forecast with per-model temps.
            outcome_label: The outcome to check confidence for.
            min_confidence: Minimum confidence threshold per model.
            require_all: If True, ALL models must meet threshold.
                         If False, 2/3 majority is enough.
        Returns:
            True if consensus is met or no multi-model data available.
        """
        if not forecast.model_temps:
            return True  # single model, skip consensus

        confident_count = 0
        for model_temp in forecast.model_temps.values():
            conf = self._outcome_confidence(outcome_label, model_temp)
            if conf >= min_confidence:
                confident_count += 1

        total = len(forecast.model_temps)
        if require_all:
            return confident_count == total
        # Majority: at least 2/3
        return confident_count >= (total * 2 / 3)

    def _check_no_consensus(
        self,
        forecast: ForecastData,
        outcome_label: str,
    ) -> bool:
        """Check that ALL models agree the outcome is unlikely (for NO trades).

        Returns True if every model's confidence for this outcome is ≤ no_max_confidence.
        Returns True if no multi-model data (backward compat).
        """
        if not forecast.model_temps:
            return True

        for model_temp in forecast.model_temps.values():
            conf = self._outcome_confidence(outcome_label, model_temp)
            if conf > self.no_max_confidence:
                return False  # at least one model thinks YES might happen
        return True

    def _size_for_signal(self, signal: WeatherSignal) -> float:
        """Return position size in USD based on trade type."""
        if signal.trade_type == "yield_yes":
            return self.yield_size
        if signal.trade_type == "yield_no":
            return self.no_size
        if signal.trade_type == "lottery_no":
            return self.lottery_no_size
        return self.paper_size  # lottery YES

    async def enter_paper_trade(self, signal: WeatherSignal) -> Optional[WeatherPaperTrade]:
        """Record a paper trade entry."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._daily_spend_date != today:
            self._daily_spend = 0.0
            self._daily_spend_date = today

        size = self._size_for_signal(signal)

        if self._daily_spend + size > self.max_daily_spend:
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
            side=signal.side,
            trade_type=signal.trade_type,
            entry_price=signal.entry_price,
            size_usd=size,
            forecast_temp_max=signal.forecast.temp_max,
            forecast_temp_min=signal.forecast.temp_min,
            confidence=signal.confidence,
            reason=signal.reason,
        )

        self._open_trades.append(trade)
        entry_key = f"{signal.market.condition_id}:{signal.side}:{signal.trade_type}:{signal.outcome}"
        self._entered_markets.add(entry_key)
        self._daily_spend += size

        # Persist entry via TradeManager
        await self._record_entry(trade)

        logger.info(
            "weather_paper_trade_entered",
            city=trade.city,
            outcome=trade.outcome,
            side=trade.side,
            trade_type=trade.trade_type,
            entry_price=trade.entry_price,
            size_usd=trade.size_usd,
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

            try:
                end_dt = datetime.fromisoformat(market.end_date.replace("Z", "+00:00"))
                if datetime.now(timezone.utc) < end_dt:
                    still_open.append(trade)
                    continue
            except (ValueError, AttributeError):
                still_open.append(trade)
                continue

            final_yes_price = market.outcome_prices.get(trade.outcome, 0.0)

            if trade.side == "BUY_NO":
                # BUY_NO wins when YES resolves to 0 (NO resolves to 1)
                if final_yes_price < 0.1:
                    trade.won = True
                    shares = trade.size_usd / trade.entry_price
                    trade.pnl_usd = round(shares * (1.0 - trade.entry_price), 2)
                elif final_yes_price > 0.9:
                    trade.won = False
                    trade.pnl_usd = round(-trade.size_usd, 2)
                else:
                    still_open.append(trade)
                    continue
            else:
                # BUY_YES wins when YES resolves to 1
                if final_yes_price > 0.9:
                    trade.won = True
                    shares = trade.size_usd / trade.entry_price
                    trade.pnl_usd = round(shares * (1.0 - trade.entry_price), 2)
                elif final_yes_price < 0.1:
                    trade.won = False
                    trade.pnl_usd = round(-trade.size_usd, 2)
                else:
                    still_open.append(trade)
                    continue

            trade.resolved = True
            trade.resolution_price = final_yes_price
            trade.resolution_time = datetime.now(timezone.utc).isoformat()

            self._stats["trades"] += 1
            if trade.won:
                self._stats["wins"] += 1
            self._stats["pnl"] += trade.pnl_usd

            # Record result in RiskGuard
            if self.guard:
                try:
                    await self.guard.record_result(pnl=trade.pnl_usd, won=trade.won)
                except Exception:
                    pass

            await self._save_trade(trade)
            resolved.append(trade)

            logger.info(
                "weather_trade_resolved",
                city=trade.city, outcome=trade.outcome,
                won=trade.won, pnl=trade.pnl_usd, entry=trade.entry_price,
            )

        self._open_trades = still_open
        return resolved

    def _extra_state(self, trade: WeatherPaperTrade) -> dict[str, Any]:
        """Build strategy-specific extra_state for TradeRecorder."""
        return {
            "city": trade.city,
            "target_date": trade.target_date,
            "slug": trade.market_slug,
            "trade_type": trade.trade_type,
            "forecast_temp_max": trade.forecast_temp_max,
            "forecast_temp_min": trade.forecast_temp_min,
            "confidence": trade.confidence,
        }

    async def _record_entry(self, trade: WeatherPaperTrade) -> None:
        """Persist entry via TradeManager recorder + Telegram."""
        if self.manager:
            from src.execution.models import FillResult, TradeIntent

            intent = TradeIntent(
                condition_id=trade.condition_id,
                token_id="",
                outcome=trade.outcome,
                side=trade.side,
                price=trade.entry_price,
                size_usd=trade.size_usd,
                reason=trade.reason,
                title=f"{trade.city} {trade.target_date} \u2192 {trade.side} {trade.outcome}",
                edge_pct=trade.confidence,
                timestamp=time.time(),
            )
            fill = FillResult(
                filled=True,
                shares=trade.size_usd / trade.entry_price if trade.entry_price > 0 else 0.0,
                avg_price=trade.entry_price,
            )
            await self.manager.record_fill_direct(
                intent=intent,
                fill=fill,
                fair_prices={trade.outcome: trade.entry_price},
                execution_mode="paper",
                extra_state=self._extra_state(trade),
            )

    async def _save_trade(self, trade: WeatherPaperTrade) -> None:
        """Persist resolved trade via TradeManager recorder + Telegram + JSONL backup."""
        if self.manager:
            from src.execution.models import FillResult, TradeIntent

            title = f"{trade.city} {trade.target_date} \u2192 {trade.side} {trade.outcome}"
            settlement_price = 1.0 if trade.won else 0.0
            settle_intent = TradeIntent(
                condition_id=trade.condition_id,
                token_id="",
                outcome=trade.outcome,
                side="SELL",
                price=settlement_price,
                size_usd=trade.size_usd,
                reason="settlement",
                title=title,
                edge_pct=0.0,
                timestamp=time.time(),
            )
            pnl = trade.pnl_usd
            settle_fill = FillResult(
                filled=True,
                shares=trade.size_usd / trade.entry_price if trade.entry_price > 0 else 0.0,
                avg_price=settlement_price,
                pnl_delta=pnl,
            )
            await self.manager.record_settle_direct(
                intent=settle_intent,
                fill=settle_fill,
                fair_prices={trade.outcome: settlement_price},
                extra_state=self._extra_state(trade),
            )

        # JSONL backup
        try:
            self.paper_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.paper_file, "a") as f:
                f.write(json.dumps(asdict(trade)) + "\n")
        except Exception as e:
            logger.warning("jsonl_save_error", trade_id=trade.id, error=str(e))

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

    async def run(self) -> None:
        """Main loop: scan markets, fetch forecasts, enter trades, resolve."""
        scan_interval = settings.WEATHER_ORACLE_SCAN_INTERVAL
        logger.info(
            "weather_oracle_started",
            scan_interval=scan_interval,
            days_filter=f"{self.min_days_to_resolution}-{self.max_days_to_resolution}",
            lottery_yes_max_price=self.max_entry_price,
            lottery_yes_min_conf=self.min_confidence,
            lottery_yes_size=self.paper_size,
            lottery_no_enabled=self.lottery_no_enabled,
            lottery_no_size=self.lottery_no_size,
            lottery_no_min_yes=self.lottery_no_min_yes_price,
            lottery_no_min_no_conf=self.lottery_no_min_no_confidence,
            yield_yes_enabled=self.yield_enabled,
            yield_no_enabled=self.no_enabled,
        )

        while True:
            try:
                # Heartbeat + circuit breaker gate
                if self.guard:
                    await self.guard.heartbeat()
                    self._last_book_update = time.time()  # REST-based
                    if not await self.guard.is_trading_allowed(last_book_update=self._last_book_update):
                        # Still resolve trades
                        await self.resolve_trades()
                        await asyncio.sleep(scan_interval)
                        continue

                new_markets = await self.scanner.scan()
                if new_markets:
                    for m in new_markets:
                        logger.info(
                            "weather_event_found",
                            city=m.city, date=m.target_date,
                            outcomes=len(m.outcomes),
                            prices={k: v for k, v in m.outcome_prices.items() if v <= 0.10},
                        )

                cities_needed = {m.city for m in self.scanner.markets.values()}
                for city in cities_needed:
                    await self.fetcher.fetch_city(city)

                total_signals = 0
                total_evaluated = 0
                for market in self.scanner.markets.values():
                    forecast = self.fetcher.get_forecast(market.city, market.target_date)
                    if not forecast:
                        continue

                    total_evaluated += 1
                    signals = self.evaluate_market(market, forecast)
                    for signal in signals:
                        logger.info(
                            "weather_signal_candidate",
                            city=signal.market.city,
                            outcome=signal.outcome,
                            side=signal.side,
                            trade_type=signal.trade_type,
                            price=signal.entry_price,
                            confidence=round(signal.confidence, 3),
                            forecast_max=signal.forecast.temp_max,
                        )
                        trade = await self.enter_paper_trade(signal)
                        if trade:
                            total_signals += 1

                logger.info(
                    "weather_eval_complete",
                    markets=len(self.scanner.markets),
                    evaluated=total_evaluated,
                    signals=total_signals,
                )

                resolved = await self.resolve_trades()
                if resolved:
                    logger.info("weather_trades_resolved", count=len(resolved))

                stats = self.get_stats()
                logger.info("weather_oracle_stats", **stats)

            except Exception as e:
                logger.error("weather_oracle_cycle_error", error=str(e))

            await asyncio.sleep(scan_interval)
