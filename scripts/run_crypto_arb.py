"""Runner for crypto reality arbitrage on 15-minute Polymarket markets.

Usage:
    python scripts/run_crypto_arb.py [--symbols BTCUSDT,ETHUSDT] [--autopilot]
"""

import argparse
import asyncio

import structlog

from src.utils.logging import configure_logging

configure_logging()

from config.settings import settings
from src.feeds.binance import BinanceFeed
from src.feeds.polymarket import PolymarketFeed
from src.realtime.crypto_mapper import CryptoMarketMapper
from src.arb.crypto_arb import CryptoArbEngine
from src.arb.position_manager import PositionManager
from src.risk.guard import RiskGuard

logger = structlog.get_logger()


async def main(symbols: list[str], autopilot: bool) -> None:
    logger.info("crypto_arb_starting", symbols=symbols, autopilot=autopilot)

    # Init feeds
    binance = BinanceFeed(
        symbols=symbols,
        fair_value_window=settings.CRYPTO_ARB_FAIR_VALUE_WINDOW,
    )
    polymarket = PolymarketFeed()
    mapper = CryptoMarketMapper()

    # Init RiskGuard
    allocated_capital = (
        settings.GLOBAL_CAPITAL * (settings.CAPITAL_ALLOCATION_CRYPTO_PCT / 100.0)
    )
    guard = RiskGuard(
        strategy_tag="crypto_arb",
        db_url=settings.DATABASE_URL,
        daily_loss_limit_usd=-(
            settings.GLOBAL_CAPITAL * settings.DAILY_LOSS_LIMIT_PCT
        ),
    )
    await guard.initialize()

    # Init engine
    engine = CryptoArbEngine(
        binance_feed=binance,
        polymarket_feed=polymarket,
        crypto_mapper=mapper,
        guard=guard,
        allocated_capital=allocated_capital,
    )

    # Connect feeds
    await binance.connect()
    await polymarket.connect()
    await mapper.sync_markets(polymarket)

    # Scan loop
    async def scan_loop():
        while True:
            for symbol in symbols:
                opp = engine.evaluate_opportunity(symbol)
                if opp and opp.is_valid:
                    logger.info(
                        "crypto_arb_opportunity",
                        symbol=opp.symbol,
                        direction=opp.cex_direction,
                        edge=f"{opp.edge_pct:.2%}",
                        pm_price=opp.polymarket_price,
                        fair_price=opp.fair_value_price,
                    )
            await asyncio.sleep(settings.CRYPTO_ARB_SCAN_INTERVAL)

    # Run feed listener and scanner concurrently
    await asyncio.gather(binance.listen(), scan_loop())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crypto Reality Arb Bot")
    parser.add_argument("--symbols", default=settings.CRYPTO_ARB_SYMBOLS)
    parser.add_argument("--autopilot", action="store_true")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")]
    asyncio.run(main(symbols, args.autopilot))
