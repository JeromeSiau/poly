#!/usr/bin/env python3
"""Auto-redeem resolved Polymarket positions.

Periodically calls redeem_all() via the Builder Relayer (gas-free).
Converts winning outcome tokens back to USDC automatically.

Usage:
    ./run run_auto_redeem.py              # single pass
    ./run run_auto_redeem.py --loop       # continuous (every 15 min)
    ./run run_auto_redeem.py --loop --interval 600
"""

import argparse
import asyncio

import structlog

from src.utils.logging import configure_logging

configure_logging()

from src.execution.redeemer import PolymarketRedeemer
from src.paper_trading.alerts import TelegramAlerter

logger = structlog.get_logger()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Auto-redeem resolved Polymarket positions")
    parser.add_argument("--loop", action="store_true", help="Run continuously")
    parser.add_argument(
        "--interval", type=float, default=900,
        help="Seconds between scans in loop mode (default: 900 = 15 min)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=10,
        help="Max positions per redeem batch (default: 10)",
    )
    return parser


async def run_once(
    redeemer: PolymarketRedeemer,
    batch_size: int,
    alerter: TelegramAlerter | None = None,
) -> int:
    """Single redeem pass. Returns number of successful redeems."""
    logger.info("redeem_scan_start")
    try:
        results = await redeemer.redeem_all(batch_size=batch_size)
    except Exception as exc:
        logger.error("redeem_scan_failed", error=str(exc))
        if alerter:
            await alerter.send_custom_alert(
                f"AUTO-REDEEM ERROR\nScan failed: {exc}"
            )
        return 0

    if not results:
        logger.info("redeem_scan_done", redeemed=0, msg="no redeemable positions")
        return 0

    ok = sum(1 for r in results if r.get("status") != "failed")
    failed = len(results) - ok
    logger.info("redeem_scan_done", redeemed=ok, failed=failed, results=results)

    if alerter and ok > 0:
        lines = [f"AUTO-REDEEM: {ok} position(s) redeemed"]
        if failed:
            lines.append(f"({failed} failed)")
        for r in results:
            if r.get("status") != "failed":
                lines.append(f"  - {r}")
        await alerter.send_custom_alert("\n".join(lines))
    elif alerter and failed > 0:
        await alerter.send_custom_alert(
            f"AUTO-REDEEM: {failed} redemption(s) failed\n"
            + "\n".join(f"  - {r}" for r in results)
        )

    return ok


async def main() -> None:
    args = build_parser().parse_args()
    redeemer = PolymarketRedeemer.from_settings()
    alerter = TelegramAlerter()

    if not args.loop:
        await run_once(redeemer, args.batch_size, alerter)
        return

    logger.info("redeem_loop_start", interval=args.interval)
    while True:
        await run_once(redeemer, args.batch_size, alerter)
        await asyncio.sleep(args.interval)


if __name__ == "__main__":
    asyncio.run(main())
