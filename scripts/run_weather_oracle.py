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
from src.arb.weather_oracle import WeatherOracleEngine, WEATHER_ORACLE_EVENT_TYPE
from src.execution import TradeManager

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
                f"| forecast={signal.forecast.temp_max:.0f}\u00b0 "
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

    from datetime import datetime, timezone

    run_id = f"weather_oracle-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    manager = TradeManager(
        strategy="WeatherOracle",
        paper=True,
        db_url=args.db_url,
        event_type=WEATHER_ORACLE_EVENT_TYPE,
        run_id=run_id,
        notify_bids=True,
        notify_fills=False,
        notify_closes=True,
    )

    engine = WeatherOracleEngine(database_url=args.db_url, manager=manager)
    engine.max_entry_price = args.max_entry_price
    engine.min_confidence = args.min_confidence
    engine.paper_size = args.paper_size
    engine.max_daily_spend = args.max_daily_spend

    try:
        if args.mode == "scan":
            await run_scan_once(engine)
        else:
            await engine.run()
    finally:
        await manager.close()


if __name__ == "__main__":
    asyncio.run(main())
