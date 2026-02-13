"""Tests for the Weather Oracle strategy."""

import json
import pytest
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, patch, MagicMock

from src.arb.weather_oracle import (
    OpenMeteoFetcher,
    ForecastData,
    CITY_STATIONS,
    WeatherMarketScanner,
    WeatherMarket,
)


def _future_end_date(days: int = 3) -> str:
    """Return an end_date string N days in the future (passes days-to-resolution filter)."""
    return (datetime.now(timezone.utc) + timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _future_target_date(days: int = 3) -> str:
    """Return a target_date string N days in the future."""
    return (datetime.now(timezone.utc) + timedelta(days=days)).strftime("%Y-%m-%d")


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


def _make_engine(**overrides):
    """Helper to build a minimal WeatherOracleEngine for unit tests."""
    engine = WeatherOracleEngine.__new__(WeatherOracleEngine)
    engine.min_days_to_resolution = overrides.get("min_days_to_resolution", 0.0)
    engine.max_days_to_resolution = overrides.get("max_days_to_resolution", 999.0)
    engine.max_entry_price = overrides.get("max_entry_price", 0.05)
    engine.min_confidence = overrides.get("min_confidence", 0.90)
    engine.lottery_no_enabled = overrides.get("lottery_no_enabled", False)
    engine.lottery_no_size = overrides.get("lottery_no_size", 3.0)
    engine.lottery_no_min_yes_price = overrides.get("lottery_no_min_yes_price", 0.95)
    engine.lottery_no_min_no_confidence = overrides.get("lottery_no_min_no_confidence", 0.60)
    engine.yield_enabled = overrides.get("yield_enabled", False)
    engine.yield_size = overrides.get("yield_size", 50.0)
    engine.yield_min_confidence = overrides.get("yield_min_confidence", 0.95)
    engine.yield_min_yes_price = overrides.get("yield_min_yes_price", 0.80)
    engine.yield_max_yes_price = overrides.get("yield_max_yes_price", 0.97)
    engine.no_enabled = overrides.get("no_enabled", False)
    engine.no_size = overrides.get("no_size", 50.0)
    engine.no_max_confidence = overrides.get("no_max_confidence", 0.10)
    engine.no_max_yes_price = overrides.get("no_max_yes_price", 0.05)
    engine.paper_size = overrides.get("paper_size", 3.0)
    engine.max_daily_spend = overrides.get("max_daily_spend", 50.0)
    engine._entered_markets = overrides.get("_entered_markets", set())
    engine._open_trades = overrides.get("_open_trades", [])
    engine._daily_spend = overrides.get("_daily_spend", 0.0)
    engine._daily_spend_date = overrides.get("_daily_spend_date", "")
    engine._stats = overrides.get("_stats", {"trades": 0, "wins": 0, "pnl": 0.0})
    engine.manager = overrides.get("manager", None)
    return engine


def test_engine_detects_cheap_certain_outcome():
    """Engine should signal when forecast makes a cheap outcome near-certain (lottery)."""
    market = WeatherMarket(
        condition_id="0xabc",
        slug="highest-temp-dallas-feb-12",
        title="Highest temperature in Dallas on February 12?",
        city="Dallas",
        target_date=_future_target_date(),
        outcomes={"58°F or higher": "tok1", "56-57°F": "tok2", "54-55°F": "tok3", "52-53°F": "tok4"},
        outcome_prices={"58°F or higher": 0.03, "56-57°F": 0.12, "54-55°F": 0.35, "52-53°F": 0.50},
        end_date=_future_end_date(),
        resolution_source="Weather Underground",
    )

    forecast = ForecastData(
        city="Dallas", date="2026-02-12",
        temp_max=66.0, temp_min=45.0, unit="fahrenheit", fetched_at=time.time(),
    )

    engine = _make_engine()

    signals = engine.evaluate_market(market, forecast)
    lottery = [s for s in signals if s.trade_type == "lottery"]
    assert len(lottery) == 1
    assert lottery[0].outcome == "58°F or higher"
    assert lottery[0].entry_price == 0.03
    assert lottery[0].confidence >= 0.90
    assert lottery[0].side == "BUY_YES"


def test_engine_skips_expensive_outcomes():
    """Engine should NOT signal lottery for outcomes priced above max_entry_price."""
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

    engine = _make_engine()  # yield disabled by default

    signals = engine.evaluate_market(market, forecast)
    assert len(signals) == 0


def test_engine_handles_range_outcomes():
    """Engine should detect when forecast lands in a specific range bucket."""
    market = WeatherMarket(
        condition_id="0xghi",
        slug="highest-temp-nyc-feb-12",
        title="Highest temperature in New York on February 12?",
        city="New York",
        target_date=_future_target_date(),
        outcomes={"40°F or higher": "tok1", "38-39°F": "tok2", "36-37°F": "tok3"},
        outcome_prices={"40°F or higher": 0.60, "38-39°F": 0.02, "36-37°F": 0.03},
        end_date=_future_end_date(),
        resolution_source="Weather Underground",
    )

    forecast = ForecastData(
        city="New York", date=_future_target_date(),
        temp_max=38.5, temp_min=28.0, unit="fahrenheit", fetched_at=time.time(),
    )

    engine = _make_engine()

    signals = engine.evaluate_market(market, forecast)
    lottery_outcomes = [s.outcome for s in signals if s.trade_type == "lottery"]
    assert "38-39°F" in lottery_outcomes


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

    engine = _make_engine(
        _entered_markets={"0xabc:BUY_YES:lottery:58°F or higher"},
    )

    signals = engine.evaluate_market(market, forecast)
    lottery = [s for s in signals if s.trade_type == "lottery"]
    assert len(lottery) == 0


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


@pytest.mark.asyncio
async def test_engine_enter_paper_trade():
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
        side="BUY_YES", trade_type="lottery",
    )

    engine = _make_engine()

    trade = await engine.enter_paper_trade(signal)
    assert trade is not None
    assert trade.city == "Dallas"
    assert trade.entry_price == 0.03
    assert trade.size_usd == 3.0
    assert trade.side == "BUY_YES"
    assert trade.trade_type == "lottery"
    assert "0xabc:BUY_YES:lottery:58°F or higher" in engine._entered_markets
    assert len(engine._open_trades) == 1


@pytest.mark.asyncio
async def test_engine_daily_spend_limit():
    """Engine should respect daily spend limit."""
    from datetime import datetime, timezone

    engine = _make_engine(
        paper_size=30.0,
        max_daily_spend=50.0,
        _daily_spend=40.0,
        _daily_spend_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    )

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
        side="BUY_YES", trade_type="lottery",
    )

    result = await engine.enter_paper_trade(signal)
    assert result is None  # exceeded daily limit


def test_engine_yield_yes_signal():
    """Engine should generate yield YES signal for high-confidence likely outcome."""
    market = WeatherMarket(
        condition_id="0xabc",
        slug="highest-temp-dallas-feb-12",
        title="Highest temperature in Dallas on February 12?",
        city="Dallas",
        target_date=_future_target_date(),
        outcomes={"58°F or higher": "tok1", "56-57°F": "tok2"},
        outcome_prices={"58°F or higher": 0.92, "56-57°F": 0.05},
        end_date=_future_end_date(),
        resolution_source="Weather Underground",
    )
    forecast = ForecastData(
        city="Dallas", date=_future_target_date(),
        temp_max=66.0, temp_min=45.0, unit="fahrenheit", fetched_at=time.time(),
    )

    engine = _make_engine(yield_enabled=True)

    signals = engine.evaluate_market(market, forecast)
    yield_yes = [s for s in signals if s.trade_type == "yield_yes"]
    assert len(yield_yes) == 1
    assert yield_yes[0].outcome == "58°F or higher"
    assert yield_yes[0].side == "BUY_YES"
    assert yield_yes[0].entry_price == 0.92


def test_engine_yield_no_signal():
    """Engine should generate yield NO signal for unlikely outcome with low YES price."""
    market = WeatherMarket(
        condition_id="0xabc",
        slug="highest-temp-dallas-feb-12",
        title="Highest temperature in Dallas on February 12?",
        city="Dallas",
        target_date=_future_target_date(),
        outcomes={"45°F or below": "tok1", "58°F or higher": "tok2"},
        # 45°F or below is very unlikely if forecast says 66°F
        outcome_prices={"45°F or below": 0.01, "58°F or higher": 0.92},
        end_date=_future_end_date(),
        resolution_source="Weather Underground",
    )
    forecast = ForecastData(
        city="Dallas", date=_future_target_date(),
        temp_max=66.0, temp_min=45.0, unit="fahrenheit", fetched_at=time.time(),
    )

    engine = _make_engine(no_enabled=True)

    signals = engine.evaluate_market(market, forecast)
    yield_no = [s for s in signals if s.trade_type == "yield_no"]
    assert len(yield_no) == 1
    assert yield_no[0].outcome == "45°F or below"
    assert yield_no[0].side == "BUY_NO"
    assert yield_no[0].entry_price == 0.99  # 1 - 0.01


@pytest.mark.asyncio
async def test_engine_yield_sizing():
    """Yield trades should use yield_size, lottery should use paper_size."""
    engine = _make_engine(yield_enabled=True, yield_size=50.0, paper_size=3.0)

    market = WeatherMarket(
        condition_id="0xabc", slug="test", title="test",
        city="Dallas", target_date=_future_target_date(),
        outcomes={"58°F or higher": "tok1"},
        outcome_prices={"58°F or higher": 0.92},
        end_date=_future_end_date(), resolution_source="",
    )
    forecast = ForecastData(
        city="Dallas", date=_future_target_date(),
        temp_max=66.0, temp_min=45.0, unit="fahrenheit", fetched_at=time.time(),
    )

    signals = engine.evaluate_market(market, forecast)
    yield_yes = [s for s in signals if s.trade_type == "yield_yes"]
    assert len(yield_yes) == 1

    trade = await engine.enter_paper_trade(yield_yes[0])
    assert trade is not None
    assert trade.size_usd == 50.0


@pytest.mark.asyncio
async def test_resolve_buy_no_trade_wins():
    """BUY_NO trade should win when YES price drops below 0.1."""
    engine = _make_engine()
    engine.scanner = WeatherMarketScanner()
    # Manually add a market that has resolved (YES → 0)
    engine.scanner._markets["slug1"] = WeatherMarket(
        condition_id="slug1", slug="slug1", title="test",
        city="Dallas", target_date="2026-02-12",
        outcomes={"45°F or below": "tok1"},
        outcome_prices={"45°F or below": 0.0},  # YES resolved to 0
        end_date="2026-02-10T00:00:00Z", resolution_source="",
    )

    trade = WeatherPaperTrade(
        id="t1", condition_id="slug1", outcome="45°F or below",
        side="BUY_NO", trade_type="yield_no",
        entry_price=0.99, size_usd=50.0,
    )
    engine._open_trades = [trade]

    resolved = await engine.resolve_trades()
    assert len(resolved) == 1
    assert resolved[0].won is True
    assert resolved[0].pnl_usd > 0  # profit = shares * (1 - 0.99)


@pytest.mark.asyncio
async def test_resolve_buy_no_trade_loses():
    """BUY_NO trade should lose when YES price goes above 0.9."""
    engine = _make_engine()
    engine.scanner = WeatherMarketScanner()
    engine.scanner._markets["slug1"] = WeatherMarket(
        condition_id="slug1", slug="slug1", title="test",
        city="Dallas", target_date="2026-02-12",
        outcomes={"45°F or below": "tok1"},
        outcome_prices={"45°F or below": 1.0},  # YES resolved to 1
        end_date="2026-02-10T00:00:00Z", resolution_source="",
    )

    trade = WeatherPaperTrade(
        id="t2", condition_id="slug1", outcome="45°F or below",
        side="BUY_NO", trade_type="yield_no",
        entry_price=0.99, size_usd=50.0,
    )
    engine._open_trades = [trade]

    resolved = await engine.resolve_trades()
    assert len(resolved) == 1
    assert resolved[0].won is False
    assert resolved[0].pnl_usd == -50.0


@pytest.mark.asyncio
async def test_full_cycle_scan_evaluate_enter():
    """Integration: scan -> forecast -> evaluate -> enter trade (all 3 types)."""
    from pathlib import Path

    mock_event_resp = [{
        "title": "Highest temperature in Dallas on February 15?",
        "slug": "highest-temperature-in-dallas-on-february-15-2026",
        "endDate": "2026-02-16T00:00:00Z",
        "description": "Resolution source: Weather Underground",
        "markets": [
            {
                "question": "Will the highest temperature in Dallas be 60°F or higher on February 15?",
                "outcomePrices": '["0.92","0.08"]',
                "clobTokenIds": '["t1","t2"]',
            },
            {
                "question": "Will the highest temperature in Dallas be between 58-59°F on February 15?",
                "outcomePrices": '["0.04","0.96"]',
                "clobTokenIds": '["t3","t4"]',
            },
            {
                "question": "Will the highest temperature in Dallas be 57°F or below on February 15?",
                "outcomePrices": '["0.01","0.99"]',
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

    engine = _make_engine(yield_enabled=True, no_enabled=True)
    engine.fetcher = OpenMeteoFetcher()
    engine.scanner = WeatherMarketScanner()
    engine.paper_file = Path("/tmp/test_weather_oracle.jsonl")
    engine._database_url = "sqlite:///data/arb.db"

    def make_resp(return_value):
        resp = AsyncMock()
        resp.status = 200
        resp.json = AsyncMock(return_value=return_value)
        resp.__aenter__ = AsyncMock(return_value=resp)
        resp.__aexit__ = AsyncMock(return_value=False)
        return resp

    with patch("aiohttp.ClientSession") as mock_cls:
        def side_effect_get(url, **kwargs):
            if "gamma" in url or "events" in str(kwargs):
                slug = kwargs.get("params", {}).get("slug", "")
                if "dallas" in slug and "february-15" in slug:
                    return make_resp(mock_event_resp)
                return make_resp([])
            return make_resp(mock_forecast_resp)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(side_effect=side_effect_get)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_cls.return_value = mock_session

        await engine.scanner.scan()
        assert len(engine.scanner.markets) >= 1

        await engine.fetcher.fetch_city("Dallas")
        forecast = engine.fetcher.get_forecast("Dallas", "2026-02-15")
        assert forecast is not None

    market = list(engine.scanner.markets.values())[0]
    signals = engine.evaluate_market(market, forecast)

    # Should have all 3 types
    types = {s.trade_type for s in signals}
    assert "yield_yes" in types  # 60°F or higher at 0.92 with high conf
    assert "yield_no" in types   # 57°F or below at 0.01 YES, low conf

    # Enter a yield YES trade
    yield_yes = [s for s in signals if s.trade_type == "yield_yes"][0]
    trade = await engine.enter_paper_trade(yield_yes)
    assert trade is not None
    assert trade.side == "BUY_YES"
    assert trade.size_usd == 50.0


# ---------------------------------------------------------------------------
# Lottery NO (Type 4) tests — the 0x594ed strategy
# ---------------------------------------------------------------------------

def test_outcome_no_confidence_threshold_high():
    """NO confidence for 'X°F or higher' when forecast is BELOW threshold."""
    engine = WeatherOracleEngine.__new__(WeatherOracleEngine)

    # Forecast 55°F, threshold 58°F → 3°F below → should be confident in NO
    conf = engine._outcome_no_confidence("58°F or higher", 55.0)
    assert conf >= 0.60
    assert conf < 1.0

    # Forecast 50°F, threshold 58°F → 8°F below → very confident
    conf = engine._outcome_no_confidence("58°F or higher", 50.0)
    assert conf >= 0.90

    # Forecast 60°F, threshold 58°F → ABOVE threshold → NO confidence = 0
    conf = engine._outcome_no_confidence("58°F or higher", 60.0)
    assert conf == 0.0

    # Forecast exactly at threshold → NO confidence = 0
    conf = engine._outcome_no_confidence("58°F or higher", 58.0)
    assert conf == 0.0


def test_outcome_no_confidence_threshold_low():
    """NO confidence for 'X°F or below' when forecast is ABOVE threshold."""
    engine = WeatherOracleEngine.__new__(WeatherOracleEngine)

    # Forecast 66°F, threshold 61°F → 5°F above → confident in NO
    conf = engine._outcome_no_confidence("61°F or below", 66.0)
    assert conf >= 0.60

    # Forecast 55°F, threshold 61°F → below threshold → NO confidence = 0
    conf = engine._outcome_no_confidence("61°F or below", 55.0)
    assert conf == 0.0


def test_outcome_no_confidence_range():
    """NO confidence for range 'X-Y°F' when forecast is OUTSIDE the range."""
    engine = WeatherOracleEngine.__new__(WeatherOracleEngine)

    # Forecast 66°F, range 44-45°F → way outside → very confident in NO
    conf = engine._outcome_no_confidence("44-45°F", 66.0)
    assert conf >= 0.90

    # Forecast 44.5°F, range 44-45°F → inside range → NO confidence = 0
    conf = engine._outcome_no_confidence("44-45°F", 44.5)
    assert conf == 0.0

    # Forecast 47°F, range 44-45°F → 2.5° outside → moderate confidence
    conf = engine._outcome_no_confidence("44-45°F", 47.0)
    assert conf > 0.0


def test_lottery_no_threshold_high():
    """Lottery NO: buy NO at <5¢ when 'X or higher' is wrong per forecast.

    This is the core 0x594ed trade:
    - Market: "58°F or higher?" → YES at 99.9%, NO at 0.1%
    - Forecast: 50°F (well below 58)
    - Action: BUY NO at $0.001
    """
    market = WeatherMarket(
        condition_id="0xdallas",
        slug="highest-temp-dallas-feb-14",
        title="Highest temperature in Dallas on February 14?",
        city="Dallas",
        target_date="2026-02-14",
        outcomes={"58°F or higher": "tok1", "56-57°F": "tok2"},
        # YES at 99.9% → NO at 0.1% — massive mispricing!
        outcome_prices={"58°F or higher": 0.999, "56-57°F": 0.001},
        end_date="2026-02-15T00:00:00Z",
        resolution_source="Weather Underground",
    )

    forecast = ForecastData(
        city="Dallas", date="2026-02-14",
        temp_max=50.0, temp_min=35.0, unit="fahrenheit", fetched_at=time.time(),
    )

    engine = _make_engine(lottery_no_enabled=True)

    signals = engine.evaluate_market(market, forecast)
    lottery_no = [s for s in signals if s.trade_type == "lottery_no"]

    assert len(lottery_no) >= 1
    # Should buy NO on "58°F or higher" since forecast (50) is well below 58
    no_58 = [s for s in lottery_no if s.outcome == "58°F or higher"]
    assert len(no_58) == 1
    assert no_58[0].side == "BUY_NO"
    assert no_58[0].entry_price == pytest.approx(0.001)  # NO price = 1 - 0.999
    assert no_58[0].confidence >= 0.60


def test_lottery_no_threshold_low():
    """Lottery NO on 'X or below' when forecast is well above threshold."""
    market = WeatherMarket(
        condition_id="0xatl",
        slug="highest-temp-atlanta-feb-14",
        title="Highest temperature in Atlanta on February 14?",
        city="Atlanta",
        target_date="2026-02-14",
        outcomes={"45°F or below": "tok1"},
        # Market thinks 45°F or below is almost certain — but forecast says 60°F
        outcome_prices={"45°F or below": 0.98},
        end_date="2026-02-15T00:00:00Z",
        resolution_source="Weather Underground",
    )

    forecast = ForecastData(
        city="Atlanta", date="2026-02-14",
        temp_max=60.0, temp_min=42.0, unit="fahrenheit", fetched_at=time.time(),
    )

    engine = _make_engine(lottery_no_enabled=True)

    signals = engine.evaluate_market(market, forecast)
    lottery_no = [s for s in signals if s.trade_type == "lottery_no"]

    assert len(lottery_no) == 1
    assert lottery_no[0].outcome == "45°F or below"
    assert lottery_no[0].side == "BUY_NO"
    assert lottery_no[0].entry_price == pytest.approx(0.02)


def test_lottery_no_skips_when_forecast_agrees_with_market():
    """Should NOT buy NO when forecast actually supports the outcome."""
    market = WeatherMarket(
        condition_id="0xmia",
        slug="highest-temp-miami-feb-14",
        title="Highest temperature in Miami on February 14?",
        city="Miami",
        target_date="2026-02-14",
        outcomes={"70°F or higher": "tok1"},
        outcome_prices={"70°F or higher": 0.99},
        end_date="2026-02-15T00:00:00Z",
        resolution_source="Weather Underground",
    )

    # Forecast AGREES with market: 80°F, well above 70°F threshold
    forecast = ForecastData(
        city="Miami", date="2026-02-14",
        temp_max=80.0, temp_min=68.0, unit="fahrenheit", fetched_at=time.time(),
    )

    engine = _make_engine(lottery_no_enabled=True)

    signals = engine.evaluate_market(market, forecast)
    lottery_no = [s for s in signals if s.trade_type == "lottery_no"]
    assert len(lottery_no) == 0  # NO confidence = 0, should not trigger


@pytest.mark.asyncio
async def test_lottery_no_sizing():
    """Lottery NO trades should use lottery_no_size."""
    engine = _make_engine(lottery_no_enabled=True, lottery_no_size=5.0)

    market = WeatherMarket(
        condition_id="0xtest", slug="test", title="test",
        city="Dallas", target_date="2026-02-14",
        outcomes={"58°F or higher": "tok1"},
        outcome_prices={"58°F or higher": 0.999},
        end_date="2026-02-15T00:00:00Z", resolution_source="",
    )
    forecast = ForecastData(
        city="Dallas", date="2026-02-14",
        temp_max=50.0, temp_min=35.0, unit="fahrenheit", fetched_at=time.time(),
    )

    signals = engine.evaluate_market(market, forecast)
    lottery_no = [s for s in signals if s.trade_type == "lottery_no"]
    assert len(lottery_no) == 1

    trade = await engine.enter_paper_trade(lottery_no[0])
    assert trade is not None
    assert trade.size_usd == 5.0
    assert trade.side == "BUY_NO"
    assert trade.trade_type == "lottery_no"


@pytest.mark.asyncio
async def test_resolve_lottery_no_wins():
    """Lottery NO trade wins when YES resolves to 0 (NO pays $1)."""
    engine = _make_engine()
    engine.scanner = WeatherMarketScanner()
    engine.scanner._markets["slug_ln"] = WeatherMarket(
        condition_id="slug_ln", slug="slug_ln", title="test",
        city="Dallas", target_date="2026-02-14",
        outcomes={"58°F or higher": "tok1"},
        outcome_prices={"58°F or higher": 0.0},  # YES resolved to 0 → NO wins
        end_date="2026-02-10T00:00:00Z", resolution_source="",
    )

    trade = WeatherPaperTrade(
        id="tln1", condition_id="slug_ln", outcome="58°F or higher",
        side="BUY_NO", trade_type="lottery_no",
        entry_price=0.001, size_usd=3.0,  # Bought NO at 0.1¢
    )
    engine._open_trades = [trade]

    resolved = await engine.resolve_trades()
    assert len(resolved) == 1
    assert resolved[0].won is True
    # $3 at 0.001 = 3000 shares → profit = 3000 * (1 - 0.001) = $2997
    assert resolved[0].pnl_usd == pytest.approx(2997.0, rel=0.01)


# ---------------------------------------------------------------------------
# Days-to-resolution filter tests
# ---------------------------------------------------------------------------

def test_days_to_resolution_filters_too_close():
    """Markets resolving in <2 days should be filtered out."""
    from datetime import datetime, timezone, timedelta

    # Market resolving in 1 day
    end_date = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    market = WeatherMarket(
        condition_id="0xclose", slug="test", title="test",
        city="Dallas", target_date="2026-02-11",
        outcomes={"58°F or higher": "tok1"},
        outcome_prices={"58°F or higher": 0.03},
        end_date=end_date, resolution_source="",
    )

    forecast = ForecastData(
        city="Dallas", date="2026-02-11",
        temp_max=66.0, temp_min=45.0, unit="fahrenheit", fetched_at=time.time(),
    )

    engine = _make_engine(min_days_to_resolution=2.0, max_days_to_resolution=8.0)
    signals = engine.evaluate_market(market, forecast)
    assert len(signals) == 0


def test_days_to_resolution_filters_too_far():
    """Markets resolving in >8 days should be filtered out."""
    from datetime import datetime, timezone, timedelta

    end_date = (datetime.now(timezone.utc) + timedelta(days=10)).isoformat()
    market = WeatherMarket(
        condition_id="0xfar", slug="test", title="test",
        city="Dallas", target_date="2026-02-20",
        outcomes={"58°F or higher": "tok1"},
        outcome_prices={"58°F or higher": 0.03},
        end_date=end_date, resolution_source="",
    )

    forecast = ForecastData(
        city="Dallas", date="2026-02-20",
        temp_max=66.0, temp_min=45.0, unit="fahrenheit", fetched_at=time.time(),
    )

    engine = _make_engine(min_days_to_resolution=2.0, max_days_to_resolution=8.0)
    signals = engine.evaluate_market(market, forecast)
    assert len(signals) == 0


def test_days_to_resolution_passes_sweet_spot():
    """Markets resolving in 3-7 days should pass the filter."""
    from datetime import datetime, timezone, timedelta

    end_date = (datetime.now(timezone.utc) + timedelta(days=5)).isoformat()
    market = WeatherMarket(
        condition_id="0xsweet", slug="test", title="test",
        city="Dallas", target_date="2026-02-15",
        outcomes={"58°F or higher": "tok1"},
        outcome_prices={"58°F or higher": 0.03},
        end_date=end_date, resolution_source="",
    )

    forecast = ForecastData(
        city="Dallas", date="2026-02-15",
        temp_max=66.0, temp_min=45.0, unit="fahrenheit", fetched_at=time.time(),
    )

    engine = _make_engine(min_days_to_resolution=2.0, max_days_to_resolution=8.0)
    signals = engine.evaluate_market(market, forecast)
    lottery = [s for s in signals if s.trade_type == "lottery"]
    assert len(lottery) == 1
