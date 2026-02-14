#!/usr/bin/env python3
"""Run the Last-Penny Sniper strategy.

Buys quasi-certain outcomes (0.95-0.999) across all fee-free Polymarket
markets, holds to resolution. Inspired by Sharky6999 (99.3% win rate,
$597K profit).

The Polymarket price IS the signal — no external feeds needed.
If something trades at 0.99, the market has decided the outcome.

Usage:
    ./run scripts/run_sniper.py                    # paper mode (default)
    ./run scripts/run_sniper.py --live             # live trading
    ./run scripts/run_sniper.py --min-price 0.99   # higher threshold
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from typing import Optional

from src.utils.logging import configure_logging

configure_logging()

try:
    import uvloop
except ImportError:
    uvloop = None

import structlog

from config.settings import settings
from src.arb.polymarket_executor import PolymarketExecutor
from src.arb.sniper_engine import SniperEngine
from src.execution import TradeManager
from src.feeds.polymarket import PolymarketFeed, PolymarketUserFeed
from src.feeds.polymarket_scanner import MarketScanner
from src.risk.guard import RiskGuard

logger = structlog.get_logger()

SNIPER_EVENT_TYPE = "last_penny_sniper"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Last-Penny Sniper — buy quasi-certain outcomes, hold to resolution"
    )
    p.add_argument("--paper", action="store_true", default=True,
                    help="Paper trading mode (default)")
    p.add_argument("--live", action="store_true", default=False,
                    help="Live trading with real orders")
    p.add_argument(
        "--min-price", type=float,
        default=settings.SNIPER_MIN_PRICE,
        help=f"Minimum ask price to snipe (default: {settings.SNIPER_MIN_PRICE})",
    )
    p.add_argument(
        "--capital", type=float, default=500.0,
        help="Starting capital in USD (default: 500)",
    )
    p.add_argument(
        "--risk-pct", type=float,
        default=settings.SNIPER_RISK_PCT,
        help=f"Risk per trade as fraction of capital (default: {settings.SNIPER_RISK_PCT})",
    )
    p.add_argument(
        "--max-per-market", type=float,
        default=settings.SNIPER_MAX_PER_MARKET_PCT,
        help=f"Max pct of capital per market (default: {settings.SNIPER_MAX_PER_MARKET_PCT})",
    )
    p.add_argument(
        "--scan-interval", type=float,
        default=settings.SNIPER_SCAN_INTERVAL,
        help=f"Seconds between REST scans (default: {settings.SNIPER_SCAN_INTERVAL})",
    )
    p.add_argument(
        "--max-end-hours", type=float, default=1.0,
        help="Max hours to resolution for sports/other (default: 1). Crypto uses fixed limits.",
    )
    p.add_argument("--strategy-tag", type=str, default="sniper_engine")
    p.add_argument("--db-url", type=str, default=settings.DATABASE_URL)
    p.add_argument("--cb-max-losses", type=int, default=3,
                    help="Circuit breaker: max consecutive losses (default: 3)")
    p.add_argument("--cb-max-drawdown", type=float, default=-50.0,
                    help="Circuit breaker: max session drawdown USD (default: -50)")
    p.add_argument("--cb-stale-seconds", type=float, default=60.0,
                    help="Circuit breaker: book staleness threshold (default: 60)")
    p.add_argument("--cb-daily-limit", type=float, default=-100.0,
                    help="Global daily loss limit USD (default: -100)")
    return p


async def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    paper_mode = not args.live
    strategy_tag = args.strategy_tag.strip() or "sniper_engine"
    run_id = f"{strategy_tag}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    # Executor (live only)
    executor: Optional[PolymarketExecutor] = None
    if not paper_mode:
        executor = PolymarketExecutor.from_settings()

    # Trade manager
    manager = TradeManager(
        executor=executor,
        strategy="SniperEngine",
        paper=paper_mode,
        db_url=args.db_url,
        event_type=SNIPER_EVENT_TYPE,
        run_id=run_id,
        notify_bids=False,
        notify_fills=not paper_mode,
        notify_closes=not paper_mode,
    )

    # Risk guard
    guard = RiskGuard(
        strategy_tag=strategy_tag,
        db_path=args.db_url.replace("sqlite+aiosqlite:///", "").replace("sqlite:///", ""),
        max_consecutive_losses=args.cb_max_losses,
        max_drawdown_usd=args.cb_max_drawdown,
        stale_seconds=args.cb_stale_seconds,
        stale_cancel_seconds=180.0,
        stale_exit_seconds=600.0,
        daily_loss_limit_usd=args.cb_daily_limit,
        telegram_alerter=manager._alerter if not paper_mode else None,
    )
    await guard.initialize()

    # Feeds
    polymarket = PolymarketFeed()

    user_feed: Optional[PolymarketUserFeed] = None
    if not paper_mode and settings.POLYMARKET_API_KEY:
        api_key = settings.POLYMARKET_API_KEY
        api_secret = settings.POLYMARKET_API_SECRET
        api_passphrase = settings.POLYMARKET_API_PASSPHRASE
        if api_key and api_secret and api_passphrase:
            user_feed = PolymarketUserFeed(
                api_key=api_key,
                api_secret=api_secret,
                api_passphrase=api_passphrase,
            )

    # Scanner
    scanner = MarketScanner(
        min_price=args.min_price,
        scan_interval=args.scan_interval,
        fee_ok_above=0.99,
        max_end_hours=args.max_end_hours,
    )

    # Engine
    engine = SniperEngine(
        polymarket=polymarket,
        user_feed=user_feed,
        manager=manager,
        guard=guard,
        scanner=scanner,
        capital=args.capital,
        risk_pct=args.risk_pct,
        max_per_market_pct=args.max_per_market,
        scan_interval=args.scan_interval,
        paper=paper_mode,
    )

    mode = "PAPER" if paper_mode else "LIVE"
    print(f"=== Last-Penny Sniper ({mode}) ===")
    print(f"  Min price:    {args.min_price}")
    print(f"  Max end:      {args.max_end_hours}h")
    print(f"  Capital:      ${args.capital:.0f}")
    print(f"  Risk/trade:   {args.risk_pct * 100:.1f}%")
    print(f"  Max/market:   {args.max_per_market * 100:.1f}%")
    print(f"  Scan interval: {args.scan_interval}s")
    print(f"  Circuit break: {args.cb_max_losses} losses / ${args.cb_max_drawdown} drawdown")
    print()

    try:
        await engine.run()
    finally:
        await polymarket.disconnect()
        if user_feed:
            await user_feed.disconnect()
        stats = engine.get_stats()
        logger.info("sniper_final_stats", **stats)


if __name__ == "__main__":
    if uvloop is not None:
        uvloop.install()
    asyncio.run(main())
