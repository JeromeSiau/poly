"""Runner for LLM-powered market screener.

Scans Polymarket, scores markets for alpha, researches top N via LLM.

Usage:
    python scripts/run_screener.py [--top-n 10] [--once]
"""

import argparse
import asyncio

import structlog

from config.settings import settings
from src.screening.market_screener import MarketScreener
from src.screening.llm_researcher import LLMResearcher

logger = structlog.get_logger()


async def main(top_n: int, once: bool) -> None:
    screener = MarketScreener(min_alpha_score=settings.SCREENER_MIN_ALPHA_SCORE)
    researcher = LLMResearcher()

    while True:
        logger.info("screener_scan_starting")
        # TODO: fetch all active markets from Polymarket REST API
        if once:
            break
        await asyncio.sleep(settings.SCREENER_SCAN_INTERVAL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Market Screener")
    parser.add_argument("--top-n", type=int, default=settings.SCREENER_TOP_N)
    parser.add_argument("--once", action="store_true", help="Run once then exit")
    args = parser.parse_args()
    asyncio.run(main(args.top_n, args.once))
