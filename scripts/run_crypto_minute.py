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

from config.settings import settings
from src.arb.crypto_minute import CryptoMinuteEngine

logger = structlog.get_logger()


async def main() -> None:
    engine = CryptoMinuteEngine()
    await engine.run()


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
