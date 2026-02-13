"""Runner for crypto 15-minute binary market strategies (paper trading).

Two strategies:
- Time Decay: buy the expensive side when gap is large
- Long Vol: buy the cheap side when gap is small

Usage:
    python scripts/run_crypto_minute.py
    python scripts/run_crypto_minute.py --symbols BTCUSDT,ETHUSDT
    python scripts/run_crypto_minute.py --td-threshold 0.90 --lv-threshold 0.12
"""

import argparse
import asyncio

import structlog

from src.utils.logging import configure_logging

configure_logging()

from config.settings import settings
from src.arb.crypto_minute import CryptoMinuteEngine, CRYPTO_MINUTE_EVENT_TYPE
from src.execution import TradeManager

logger = structlog.get_logger()

DB_URL = "sqlite:///data/arb.db"


async def main() -> None:
    from datetime import datetime, timezone

    run_id = f"crypto_minute-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    manager = TradeManager(
        strategy="CryptoMinute",
        paper=True,
        db_url=DB_URL,
        event_type=CRYPTO_MINUTE_EVENT_TYPE,
        run_id=run_id,
        notify_bids=False,
        notify_fills=False,
        notify_closes=False,
    )

    engine = CryptoMinuteEngine(database_url=DB_URL, manager=manager)
    try:
        await engine.run()
    finally:
        await manager.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crypto 15-min Strategies (Paper)")
    parser.add_argument("--symbols", default=settings.CRYPTO_MINUTE_SYMBOLS)
    parser.add_argument("--td-threshold", type=float, default=settings.CRYPTO_MINUTE_TD_THRESHOLD)
    parser.add_argument("--lv-threshold", type=float, default=settings.CRYPTO_MINUTE_LV_THRESHOLD)
    parser.add_argument("--scan-interval", type=float, default=settings.CRYPTO_MINUTE_SCAN_INTERVAL)
    args = parser.parse_args()

    # Override settings from CLI args
    settings.CRYPTO_MINUTE_SYMBOLS = args.symbols
    settings.CRYPTO_MINUTE_TD_THRESHOLD = args.td_threshold
    settings.CRYPTO_MINUTE_LV_THRESHOLD = args.lv_threshold
    settings.CRYPTO_MINUTE_SCAN_INTERVAL = args.scan_interval

    logger.info(
        "crypto_minute_runner",
        symbols=args.symbols,
        td_threshold=args.td_threshold,
        lv_threshold=args.lv_threshold,
        scan_interval=args.scan_interval,
    )

    asyncio.run(main())
