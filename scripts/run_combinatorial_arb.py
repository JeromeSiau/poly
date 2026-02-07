"""Runner for combinatorial arbitrage on logically dependent markets.

Usage:
    python scripts/run_combinatorial_arb.py [--scan-interval 60] [--autopilot]
"""

import argparse
import asyncio

import structlog

from config.settings import settings
from src.arb.dependency_detector import DependencyDetector

logger = structlog.get_logger()


async def main(scan_interval: float, autopilot: bool) -> None:
    logger.info("combinatorial_arb_starting", scan_interval=scan_interval)

    detector = DependencyDetector(
        confidence_threshold=settings.COMBO_ARB_DEPENDENCY_CONFIDENCE,
    )

    while True:
        logger.info("combinatorial_arb_scan_starting")
        # TODO: Phase 1 — single market arb scan
        # TODO: Phase 2 — dependency detection
        # TODO: Phase 3 — pair arbitrage on cached dependencies
        await asyncio.sleep(scan_interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combinatorial Arb Scanner")
    parser.add_argument("--scan-interval", type=float, default=settings.COMBO_ARB_SCAN_INTERVAL)
    parser.add_argument("--autopilot", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(args.scan_interval, args.autopilot))
