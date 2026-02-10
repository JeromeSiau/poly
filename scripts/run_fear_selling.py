#!/usr/bin/env python3
"""Fear Selling strategy runner.

Scans Polymarket for fear-driven tail-risk markets, evaluates contrarian
NO bets using Kelly criterion and cluster-based risk limits, and optionally
places orders via the CLOB API in autopilot mode.

Usage:
    python scripts/run_fear_selling.py                        # scan once, log signals
    python scripts/run_fear_selling.py --scan-interval 120    # continuous loop
    python scripts/run_fear_selling.py --autopilot            # live execution
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.dev.ConsoleRenderer(),
    ]
)
logger = structlog.get_logger()

from config.settings import settings
from src.arb.fear_classifier import FearClassifier
from src.arb.fear_engine import FearSellingEngine, FearTradeSignal
from src.arb.fear_scanner import FearMarketScanner
from src.arb.fear_spike_detector import FearSpikeDetector
from src.arb.polymarket_executor import PolymarketExecutor
from src.db.database import get_sync_session, init_db
from src.db.models import FearPosition
from src.risk.manager import UnifiedRiskManager


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Fear Selling strategy runner")
    parser.add_argument(
        "--autopilot",
        action="store_true",
        help="Enable autopilot mode (auto-execute trades via Polymarket CLOB)",
    )
    parser.add_argument(
        "--scan-interval",
        type=float,
        default=settings.FEAR_SELLING_SCAN_INTERVAL,
        help=f"Seconds between scan cycles (default: {settings.FEAR_SELLING_SCAN_INTERVAL})",
    )
    return parser.parse_args()


def _persist_signal(sig: FearTradeSignal) -> None:
    """Save a trade signal as an open FearPosition in the database."""
    try:
        session = get_sync_session()
        # Skip if we already have an open position for this market
        existing = (
            session.query(FearPosition)
            .filter_by(condition_id=sig.condition_id, is_open=True)
            .first()
        )
        if existing:
            logger.debug("position_already_open", condition_id=sig.condition_id)
            session.close()
            return

        pos = FearPosition(
            condition_id=sig.condition_id,
            token_id=sig.token_id,
            title=sig.title,
            cluster=sig.cluster,
            side=sig.outcome,
            entry_price=sig.price,
            size_usd=sig.size_usd,
            shares=sig.size_usd / sig.price if sig.price > 0 else 0.0,
            fear_score=sig.fear_score,
            yes_price_at_entry=1.0 - sig.price,
            entry_trigger=sig.trigger,
            is_open=True,
        )
        session.add(pos)
        session.commit()
        logger.info(
            "fear_position_persisted",
            condition_id=sig.condition_id,
            size_usd=round(sig.size_usd, 2),
        )
    except Exception as exc:
        logger.warning("fear_position_persist_error", error=str(exc))
    finally:
        try:
            session.close()
        except Exception:
            pass


async def main(args: argparse.Namespace) -> None:
    """Main entry point for the fear selling bot."""

    # Shutdown coordination
    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _signal_handler() -> None:
        logger.info("shutdown_signal_received")
        shutdown_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    # --- Risk manager ---------------------------------------------------
    risk_manager = UnifiedRiskManager(
        global_capital=settings.GLOBAL_CAPITAL,
        reality_allocation_pct=settings.CAPITAL_ALLOCATION_REALITY_PCT,
        crossmarket_allocation_pct=settings.CAPITAL_ALLOCATION_CROSSMARKET_PCT,
        max_position_pct=settings.MAX_POSITION_PCT,
        daily_loss_limit_pct=settings.DAILY_LOSS_LIMIT_PCT,
        fear_allocation_pct=settings.CAPITAL_ALLOCATION_FEAR_PCT,
    )

    # --- Executor (only in autopilot with a private key) ----------------
    executor = None
    if args.autopilot and settings.POLYMARKET_PRIVATE_KEY:
        try:
            executor = PolymarketExecutor(
                host=settings.POLYMARKET_CLOB_HTTP,
                chain_id=settings.POLYMARKET_CHAIN_ID,
                private_key=settings.POLYMARKET_PRIVATE_KEY,
                funder=settings.POLYMARKET_WALLET_ADDRESS,
                api_key=settings.POLYMARKET_API_KEY,
                api_secret=settings.POLYMARKET_API_SECRET,
                api_passphrase=settings.POLYMARKET_API_PASSPHRASE,
            )
            logger.info("polymarket_executor_ready")
        except Exception as exc:
            logger.error("polymarket_executor_init_failed", error=str(exc))
            executor = None
    elif args.autopilot:
        logger.warning("autopilot_requested_but_no_private_key")

    # --- LLM classifier (GPT-5-nano, optional) --------------------------
    classifier = None
    if settings.OPENAI_API_KEY:
        classifier = FearClassifier(api_key=settings.OPENAI_API_KEY)
        logger.info("fear_classifier_enabled", model="gpt-5-nano")
    else:
        logger.info("fear_classifier_disabled_no_openai_key")

    # --- Fear selling engine --------------------------------------------
    engine = FearSellingEngine(
        risk_manager=risk_manager,
        executor=executor,
        max_cluster_pct=settings.FEAR_SELLING_MAX_CLUSTER_PCT,
        max_position_pct=settings.FEAR_SELLING_MAX_POSITION_PCT,
        kelly_fraction=settings.FEAR_SELLING_KELLY_FRACTION,
        exit_no_price=settings.FEAR_SELLING_EXIT_NO_PRICE,
        stop_yes_price=settings.FEAR_SELLING_STOP_YES_PRICE,
        min_fear_score=settings.FEAR_SELLING_MIN_FEAR_SCORE,
        classifier=classifier,
    )

    # --- Database (paper trade persistence) -------------------------------
    init_db()
    logger.info("fear_selling_db_initialized")

    logger.info(
        "fear_selling_bot_started",
        autopilot=args.autopilot,
        scan_interval=args.scan_interval,
        min_fear_score=settings.FEAR_SELLING_MIN_FEAR_SCORE,
    )

    # --- Main scan loop -------------------------------------------------
    cycle = 0
    while not shutdown_event.is_set():
        cycle += 1
        logger.info("scan_cycle_start", cycle=cycle)

        try:
            candidates = await engine._scanner.discover_markets()
            logger.info("candidates_discovered", count=len(candidates), cycle=cycle)

            signals_count = 0
            for candidate in candidates:
                signal_result = engine.evaluate_candidate(candidate)
                if signal_result is None:
                    continue

                signals_count += 1
                logger.info(
                    "trade_signal",
                    condition_id=signal_result.condition_id,
                    title=signal_result.title,
                    outcome=signal_result.outcome,
                    price=signal_result.price,
                    size_usd=round(signal_result.size_usd, 2),
                    edge_pct=round(signal_result.edge_pct, 4),
                    fear_score=round(signal_result.fear_score, 3),
                    cluster=signal_result.cluster,
                )

                # Persist paper position to DB for dashboard visibility
                _persist_signal(signal_result)

                if args.autopilot and executor is not None:
                    try:
                        result = await executor.place_order(
                            token_id=signal_result.token_id,
                            side=signal_result.side,
                            size=signal_result.size_usd,
                            price=signal_result.price,
                            outcome=signal_result.outcome,
                        )
                        logger.info(
                            "order_placed",
                            condition_id=signal_result.condition_id,
                            status=result.get("status"),
                        )
                    except Exception as exc:
                        logger.error(
                            "order_placement_failed",
                            condition_id=signal_result.condition_id,
                            error=str(exc),
                        )

            logger.info("scan_cycle_complete", cycle=cycle, signals=signals_count)

        except Exception as exc:
            logger.error("scan_cycle_error", cycle=cycle, error=str(exc))

        # Wait for next cycle, but break early on shutdown
        try:
            await asyncio.wait_for(shutdown_event.wait(), timeout=args.scan_interval)
        except asyncio.TimeoutError:
            pass  # Normal: timeout means it's time for the next cycle

    logger.info("fear_selling_bot_stopped")


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
