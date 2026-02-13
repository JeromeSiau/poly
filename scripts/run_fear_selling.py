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

import json
from datetime import datetime, timezone

import httpx
import structlog

from src.utils.logging import configure_logging

configure_logging()
logger = structlog.get_logger()

from config.settings import settings
from src.arb.fear_classifier import FearClassifier
from src.arb.fear_engine import FearSellingEngine, FearTradeSignal
from src.arb.fear_scanner import FearMarketScanner
from src.arb.fear_spike_detector import FearSpikeDetector
from src.arb.polymarket_executor import PolymarketExecutor
from src.db.database import get_sync_session, init_db
from src.db.models import FearPosition
import time

from src.execution import TradeManager
from src.execution import TradeIntent as ExecTradeIntent
from src.execution import FillResult as ExecFillResult
from src.risk.guard import RiskGuard


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
    parser.add_argument("--cb-max-losses", type=int, default=5, help="Circuit breaker: max consecutive losses")
    parser.add_argument("--cb-max-drawdown", type=float, default=-50.0, help="Circuit breaker: max session drawdown USD")
    parser.add_argument("--cb-stale-seconds", type=float, default=300.0, help="Circuit breaker: book staleness threshold")
    parser.add_argument("--cb-daily-limit", type=float, default=-200.0, help="Global daily loss limit USD")
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


GAMMA_MARKETS_API = "https://gamma-api.polymarket.com/markets"


async def _check_exits(
    engine: FearSellingEngine,
    manager: TradeManager | None = None,
    guard: RiskGuard | None = None,
) -> int:
    """Check open FearPositions for exit signals or market resolution.

    Returns number of positions closed.
    """
    session = get_sync_session()
    try:
        open_positions = (
            session.query(FearPosition)
            .filter_by(is_open=True)
            .all()
        )
        if not open_positions:
            return 0

        # Fetch current prices from Gamma API for all open condition_ids
        cid_to_pos: dict[str, list[FearPosition]] = {}
        for pos in open_positions:
            cid_to_pos.setdefault(pos.condition_id, []).append(pos)

        closed_count = 0
        now = datetime.now(timezone.utc)

        async with httpx.AsyncClient(timeout=15.0) as client:
            cids = list(cid_to_pos.keys())
            # Fetch in chunks of 20
            for i in range(0, len(cids), 20):
                chunk = cids[i:i + 20]
                try:
                    params = [("condition_ids", cid) for cid in chunk]
                    resp = await client.get(GAMMA_MARKETS_API, params=params)
                    if resp.status_code != 200:
                        continue
                    rows = resp.json()
                except Exception as exc:
                    logger.warning("fear_exit_fetch_error", error=str(exc))
                    continue

                if not isinstance(rows, list):
                    continue

                for raw in rows:
                    cid = str(raw.get("conditionId", ""))
                    positions = cid_to_pos.get(cid, [])
                    if not positions:
                        continue

                    # Parse outcome prices
                    try:
                        prices_raw = raw.get("outcomePrices", [])
                        if isinstance(prices_raw, str):
                            prices_raw = json.loads(prices_raw)
                        outcomes = raw.get("outcomes", [])
                        if isinstance(outcomes, str):
                            outcomes = json.loads(outcomes)
                        if len(prices_raw) < 2 or len(outcomes) < 2:
                            continue

                        price_map = {}
                        for outcome, price in zip(outcomes, prices_raw):
                            price_map[str(outcome)] = float(price)
                    except (ValueError, TypeError):
                        continue

                    yes_price = price_map.get("Yes", 0.0)
                    no_price = price_map.get("No", 0.0)

                    # Check market resolution (prices near 0/1)
                    is_resolved = (
                        max(yes_price, no_price) >= 0.985
                        and min(yes_price, no_price) <= 0.015
                    )

                    for pos in positions:
                        should_exit = False
                        reason = ""

                        if is_resolved:
                            should_exit = True
                            reason = f"Market resolved: YES={yes_price:.2f} NO={no_price:.2f}"
                        else:
                            should_exit, reason = engine.check_exit(
                                entry_price=pos.entry_price,
                                current_no_price=no_price,
                                current_yes_price=yes_price,
                            )

                        if should_exit:
                            exit_price = no_price  # fear selling is always NO side
                            realized_pnl = (exit_price - pos.entry_price) * pos.shares

                            pos.is_open = False
                            pos.closed_at = now
                            pos.exit_price = exit_price
                            pos.realized_pnl = round(realized_pnl, 4)

                            logger.info(
                                "fear_position_closed",
                                condition_id=cid,
                                title=pos.title,
                                reason=reason,
                                entry=pos.entry_price,
                                exit=exit_price,
                                pnl=pos.realized_pnl,
                            )
                            closed_count += 1

                            # Record result in RiskGuard
                            if guard:
                                try:
                                    await guard.record_result(pnl=realized_pnl, won=realized_pnl > 0)
                                except Exception:
                                    pass

                            # Settle via TradeManager for Telegram + LiveObservation
                            if manager is not None:
                                try:
                                    settle_intent = ExecTradeIntent(
                                        condition_id=cid,
                                        token_id=pos.token_id or "",
                                        outcome=pos.side,  # "No"
                                        side="SELL",
                                        price=exit_price,
                                        size_usd=pos.shares * exit_price,
                                        reason=f"fear_exit:{reason}",
                                        title=pos.title or "",
                                        edge_pct=0.0,
                                    )
                                    settle_fill = ExecFillResult(
                                        filled=True,
                                        shares=pos.shares,
                                        avg_price=exit_price,
                                        pnl_delta=realized_pnl,
                                    )
                                    await manager.record_settle_direct(
                                        intent=settle_intent,
                                        fill=settle_fill,
                                        fair_prices={pos.side: exit_price},
                                        extra_state={
                                            "cluster": pos.cluster,
                                            "fear_score": pos.fear_score,
                                        },
                                    )
                                except Exception as exc:
                                    logger.warning(
                                        "fear_settle_manager_error",
                                        condition_id=cid,
                                        error=str(exc),
                                    )

        session.commit()
        return closed_count
    except Exception as exc:
        session.rollback()
        logger.warning("fear_exit_check_error", error=str(exc))
        return 0
    finally:
        session.close()


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

    # --- Executor (only in autopilot with a private key) ----------------
    executor = None
    if args.autopilot:
        try:
            executor = PolymarketExecutor.from_settings()
            logger.info("polymarket_executor_ready")
        except Exception as exc:
            logger.error("polymarket_executor_init_failed", error=str(exc))
            executor = None

    # --- LLM classifier (GPT-5-nano, optional) --------------------------
    classifier = None
    if settings.OPENAI_API_KEY:
        classifier = FearClassifier(api_key=settings.OPENAI_API_KEY)
        logger.info("fear_classifier_enabled", model="gpt-5-nano")
    else:
        logger.info("fear_classifier_disabled_no_openai_key")

    # --- Fear selling engine --------------------------------------------
    engine = FearSellingEngine(
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

    # --- TradeManager for Telegram + unified LiveObservation DB ----------
    manager = TradeManager(
        strategy="fear_selling",
        paper=not args.autopilot,
        db_url=settings.DATABASE_URL or "sqlite+aiosqlite:///data/arb.db",
        event_type="fear_selling",
        run_id="",
        notify_bids=False,
        notify_fills=args.autopilot,
        notify_closes=args.autopilot,
    )
    logger.info("trade_manager_ready", strategy="fear_selling")

    # --- RiskGuard (replaces UnifiedRiskManager) -------------------------
    db_url = settings.DATABASE_URL or "sqlite+aiosqlite:///data/arb.db"
    db_path = db_url.replace("sqlite+aiosqlite:///", "").replace("sqlite:///", "")
    guard = RiskGuard(
        strategy_tag="fear_selling",
        db_path=db_path,
        max_consecutive_losses=args.cb_max_losses,
        max_drawdown_usd=args.cb_max_drawdown,
        stale_seconds=args.cb_stale_seconds,
        daily_loss_limit_usd=args.cb_daily_limit,
        telegram_alerter=manager._alerter,
    )
    await guard.initialize()
    logger.info("risk_guard_ready", strategy="fear_selling")

    _last_book_update = time.time()

    logger.info(
        "fear_selling_bot_started",
        autopilot=args.autopilot,
        scan_interval=args.scan_interval,
        min_fear_score=settings.FEAR_SELLING_MIN_FEAR_SCORE,
    )

    # --- Main scan loop -------------------------------------------------
    cycle = 0
    try:
        while not shutdown_event.is_set():
            cycle += 1

            # Circuit breaker gate
            await guard.heartbeat()
            _last_book_update = time.time()  # REST-based: staleness = time since last successful API call
            if not await guard.is_trading_allowed(last_book_update=_last_book_update):
                logger.info("scan_cycle_skipped_circuit_breaker", cycle=cycle)
                try:
                    await asyncio.wait_for(shutdown_event.wait(), timeout=args.scan_interval)
                except asyncio.TimeoutError:
                    pass
                # Still check exits even when circuit broken
                closed = await _check_exits(engine, manager=manager, guard=guard)
                if closed:
                    logger.info("fear_exits_triggered", cycle=cycle, closed=closed)
                continue

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

                    # Persist paper position to FearPosition DB (fear-specific queries)
                    _persist_signal(signal_result)

                    # Record entry via TradeManager for Telegram + LiveObservation
                    try:
                        exec_intent = ExecTradeIntent(
                            condition_id=signal_result.condition_id,
                            token_id=signal_result.token_id,
                            outcome=signal_result.outcome,
                            side=signal_result.side,
                            price=signal_result.price,
                            size_usd=signal_result.size_usd,
                            reason=f"fear_scan:{signal_result.trigger}",
                            title=signal_result.title,
                            edge_pct=signal_result.edge_pct,
                        )
                        shares = signal_result.size_usd / signal_result.price if signal_result.price > 0 else 0.0
                        exec_fill = ExecFillResult(
                            filled=True,
                            shares=shares,
                            avg_price=signal_result.price,
                        )
                        await manager.record_fill_direct(
                            intent=exec_intent,
                            fill=exec_fill,
                            fair_prices={signal_result.outcome: signal_result.price},
                            execution_mode="paper" if not args.autopilot else "live",
                            extra_state={
                                "cluster": signal_result.cluster,
                                "fear_score": signal_result.fear_score,
                            },
                        )
                    except Exception as exc:
                        logger.warning("fear_manager_record_error", error=str(exc))

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

                # Check exit conditions for open positions
                closed = await _check_exits(engine, manager=manager, guard=guard)
                if closed:
                    logger.info("fear_exits_triggered", cycle=cycle, closed=closed)

            except Exception as exc:
                logger.error("scan_cycle_error", cycle=cycle, error=str(exc))

            # Wait for next cycle, but break early on shutdown
            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=args.scan_interval)
            except asyncio.TimeoutError:
                pass  # Normal: timeout means it's time for the next cycle
    finally:
        await manager.close()

    logger.info("fear_selling_bot_stopped")


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
