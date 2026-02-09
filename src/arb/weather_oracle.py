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


class WeatherMarketScanner:
    """Discovers active weather markets on Polymarket via Gamma API."""

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
        self._markets: dict[str, WeatherMarket] = {}

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


class WeatherOracleEngine:
    """Main engine: matches forecasts to markets and generates buy signals."""

    _THRESHOLD_HIGH = re.compile(r"(\d+)\s*°[FC]\s+or\s+higher", re.IGNORECASE)
    _THRESHOLD_LOW = re.compile(r"(\d+)\s*°[FC]\s+or\s+lower", re.IGNORECASE)
    _RANGE = re.compile(r"(\d+)\s*[-–]\s*(\d+)\s*°[FC]", re.IGNORECASE)
    _EXACT = re.compile(r"^(\d+)\s*°[FC]$", re.IGNORECASE)

    def __init__(
        self,
        fetcher=None,
        scanner=None,
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
            if price > self.max_entry_price or price <= 0:
                continue

            entry_key = f"{market.condition_id}:{outcome_label}"
            if entry_key in self._entered_markets:
                continue

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
        forecast_std = 4.0

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

    def enter_paper_trade(self, signal: WeatherSignal) -> Optional[WeatherPaperTrade]:
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

            try:
                end_dt = datetime.fromisoformat(market.end_date.replace("Z", "+00:00"))
                if datetime.now(timezone.utc) < end_dt:
                    still_open.append(trade)
                    continue
            except (ValueError, AttributeError):
                still_open.append(trade)
                continue

            final_price = market.outcome_prices.get(trade.outcome, 0.0)

            if final_price > 0.9:
                trade.won = True
                shares = trade.size_usd / trade.entry_price
                trade.pnl_usd = round(shares * (1.0 - trade.entry_price), 2)
            elif final_price < 0.1:
                trade.won = False
                trade.pnl_usd = round(-trade.size_usd, 2)
            else:
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
                city=trade.city, outcome=trade.outcome,
                won=trade.won, pnl=trade.pnl_usd, entry=trade.entry_price,
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
                new_markets = await self.scanner.scan()
                if new_markets:
                    logger.info("weather_markets_discovered", count=len(new_markets))

                cities_needed = {m.city for m in self.scanner.markets.values()}
                for city in cities_needed:
                    await self.fetcher.fetch_city(city)

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

                resolved = await self.resolve_trades()
                if resolved:
                    logger.info("weather_trades_resolved", count=len(resolved))

                stats = self.get_stats()
                logger.info("weather_oracle_stats", **stats)

            except Exception as e:
                logger.error("weather_oracle_cycle_error", error=str(e))

            await asyncio.sleep(scan_interval)
