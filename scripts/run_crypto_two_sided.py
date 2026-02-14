#!/usr/bin/env python3
"""Run crypto two-sided arbitrage on Polymarket up/down markets.

Strategy: buy both Up and Down outcomes at market open when the combined ask
prices plus fees leave a structural edge (ask_up + ask_down + 2*fee < 1.0).
Resolution happens automatically after the time slot expires.

Default mode is paper execution. Enable ``--autopilot`` to place real orders.

Usage:
    python scripts/run_crypto_two_sided.py
    python scripts/run_crypto_two_sided.py --symbols BTCUSDT,ETHUSDT --budget 500
    python scripts/run_crypto_two_sided.py --min-edge 0.005 --autopilot
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _ROOT)

import structlog

from src.utils.logging import configure_logging

configure_logging()

from config.settings import settings
from src.arb.crypto_two_sided import (
    SYMBOL_TO_SLUG,
    CryptoTwoSidedEngine,
    MarketPosition,
    SlotScanner,
    compute_edge,
    compute_sweep,
    next_slots,
)
from src.arb.polymarket_executor import PolymarketExecutor
from src.execution import TradeManager
from src.execution.models import FillResult, TradeIntent
from src.feeds.polymarket import PolymarketFeed
from src.risk.guard import RiskGuard

logger = structlog.get_logger()

CRYPTO_TWO_SIDED_EVENT_TYPE = "crypto_two_sided"
DB_URL = "sqlite:///data/arb.db"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_symbols(raw: str) -> list[str]:
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


def _parse_timeframes(raw: str) -> list[int]:
    return [int(t.strip()) for t in raw.split(",") if t.strip()]


def _get_asks_from_feed(
    feed: PolymarketFeed, condition_id: str, outcome: str
) -> list[tuple[float, float]]:
    """Get full ask-side depth from the feed's internal orderbook."""
    token_id = feed._token_map.get((condition_id, outcome))
    if not token_id:
        return []
    book = feed._local_orderbook.get(token_id)
    if not book:
        return []
    return list(book.get("asks", []))


# ---------------------------------------------------------------------------
# Slot loop: one per (symbol, timeframe)
# ---------------------------------------------------------------------------


async def slot_loop(
    symbol: str,
    timeframe: int,
    engine: CryptoTwoSidedEngine,
    scanner: SlotScanner,
    feed: PolymarketFeed,
    manager: TradeManager,
    guard: RiskGuard,
    *,
    paper: bool,
    budget: float,
    fee_rate: float,
    discovery_lead_s: int,
    poll_interval_s: float,
    tag: str,
    shutdown_event: asyncio.Event,
) -> None:
    """Continuously discover and trade upcoming slots for one symbol x timeframe."""
    slug_prefix = SYMBOL_TO_SLUG.get(symbol, "")
    if not slug_prefix:
        logger.warning("unknown_symbol", symbol=symbol)
        return

    logger.info(
        "slot_loop_started",
        symbol=symbol,
        timeframe=timeframe,
        budget=budget,
    )

    while not shutdown_event.is_set():
        try:
            now = time.time()

            # Pre-compute next slot
            slots = next_slots(now, [symbol], [timeframe])
            if not slots:
                await asyncio.sleep(poll_interval_s)
                continue

            slot = slots[0]
            event_start = float(slot["event_start"])
            slug = str(slot["slug"])

            # Skip if already entered
            if engine.already_entered(slug):
                wait = event_start + timeframe - now + 5
                if wait > 0:
                    await _interruptible_sleep(wait, shutdown_event)
                continue

            # Wait until discovery_lead_s before slot opens
            wait_until_discover = event_start - discovery_lead_s - now
            if wait_until_discover > 0:
                await _interruptible_sleep(wait_until_discover, shutdown_event)
                if shutdown_event.is_set():
                    break

            # Poll for market discovery
            market = None
            discovery_deadline = event_start + engine.entry_window_s
            while time.time() < discovery_deadline and not shutdown_event.is_set():
                market = await scanner.discover_slot(slug, symbol, timeframe)
                if market:
                    break
                await _interruptible_sleep(poll_interval_s, shutdown_event)

            if not market or shutdown_event.is_set():
                if not market:
                    logger.debug("slot_not_found", slug=slug)
                continue

            # Determine outcome labels early (before subscribe)
            outcomes = list(market.token_ids.keys())
            if len(outcomes) < 2:
                logger.warning("insufficient_outcomes", slug=slug)
                continue

            # Identify Up/Down sides
            up_outcome = None
            down_outcome = None
            for o in outcomes:
                ol = o.lower()
                if "up" in ol or "yes" in ol:
                    up_outcome = o
                elif "down" in ol or "no" in ol:
                    down_outcome = o
            if not up_outcome or not down_outcome:
                up_outcome, down_outcome = outcomes[0], outcomes[1]

            # Subscribe WebSocket once (batched, like TD maker)
            await feed.subscribe_market(
                market.condition_id,
                token_map=market.token_ids,
                send=False,
            )
            await feed.flush_subscriptions()

            # Wait for book snapshot with event-driven approach
            # (like TD maker's book_updated.wait())
            entry_deadline = market.event_start + engine.entry_window_s
            entered = False

            while time.time() < entry_deadline and not shutdown_event.is_set():
                # Wait for book update or timeout
                feed.book_updated.clear()
                try:
                    await asyncio.wait_for(
                        feed.book_updated.wait(),
                        timeout=2.0,
                    )
                except asyncio.TimeoutError:
                    pass

                # Risk check
                if not await guard.is_trading_allowed(
                    last_book_update=feed.last_update_ts or time.time()
                ):
                    logger.info("trading_blocked_by_guard", slug=slug)
                    break

                # Get best ask levels
                _, _, ask_up, _ = feed.get_best_levels(market.condition_id, up_outcome)
                _, _, ask_down, _ = feed.get_best_levels(market.condition_id, down_outcome)

                if ask_up is None or ask_down is None:
                    continue  # keep waiting for book data

                now = time.time()
                market_age_s = now - market.event_start

                # Check edge
                if not engine.should_enter(ask_up, ask_down, market_age_s):
                    edge = compute_edge(ask_up, ask_down, fee_rate)
                    logger.debug(
                        "edge_insufficient",
                        slug=slug,
                        ask_up=ask_up,
                        ask_down=ask_down,
                        edge=round(edge, 4),
                        market_age_s=round(market_age_s, 1),
                    )
                    continue  # keep watching — edge may improve

                # Compute sweep budget
                up_asks = _get_asks_from_feed(feed, market.condition_id, up_outcome)
                down_asks = _get_asks_from_feed(feed, market.condition_id, down_outcome)
                up_budget, down_budget, best_edge = compute_sweep(
                    up_asks, down_asks, fee_rate, budget
                )

                if up_budget <= 0 or down_budget <= 0:
                    logger.debug("sweep_budget_zero", slug=slug)
                    continue  # keep watching

                entered = True
                break

            if not entered:
                await feed.unsubscribe_market(market.condition_id)
                continue

            # Execute both sides
            up_shares = up_budget / ask_up if ask_up > 0 else 0.0
            down_shares = down_budget / ask_down if ask_down > 0 else 0.0

            up_intent = TradeIntent(
                condition_id=market.condition_id,
                token_id=market.token_ids[up_outcome],
                outcome=up_outcome,
                side="BUY",
                price=ask_up,
                size_usd=up_budget,
                reason="crypto_two_sided_entry",
                title=slug,
                edge_pct=best_edge,
            )
            down_intent = TradeIntent(
                condition_id=market.condition_id,
                token_id=market.token_ids[down_outcome],
                outcome=down_outcome,
                side="BUY",
                price=ask_down,
                size_usd=down_budget,
                reason="crypto_two_sided_entry",
                title=slug,
                edge_pct=best_edge,
            )

            up_fill = FillResult(filled=True, shares=up_shares, avg_price=ask_up)
            down_fill = FillResult(filled=True, shares=down_shares, avg_price=ask_down)

            if paper:
                await manager.record_fill_direct(
                    up_intent,
                    up_fill,
                    execution_mode="paper",
                    extra_state={"slug": slug, "edge": best_edge},
                )
                await manager.record_fill_direct(
                    down_intent,
                    down_fill,
                    execution_mode="paper",
                    extra_state={"slug": slug, "edge": best_edge},
                )
            else:
                await manager.place(up_intent)
                await manager.place(down_intent)

            # Record position in engine
            position = MarketPosition(
                condition_id=market.condition_id,
                slug=slug,
                symbol=symbol,
                timeframe=timeframe,
                up_token_id=market.token_ids[up_outcome],
                down_token_id=market.token_ids[down_outcome],
                up_shares=up_shares,
                down_shares=down_shares,
                up_cost=up_budget,
                down_cost=down_budget,
                entered_at=time.time(),
                end_time=market.end_time,
                entry_edge=best_edge,
            )
            engine.record_entry(position)

            logger.info(
                "position_entered",
                slug=slug,
                symbol=symbol,
                edge=round(best_edge, 4),
                up_budget=round(up_budget, 2),
                down_budget=round(down_budget, 2),
                ask_up=ask_up,
                ask_down=ask_down,
                total_cost=round(up_budget + down_budget, 2),
            )

        except asyncio.CancelledError:
            break
        except Exception:
            logger.exception("slot_loop_error", symbol=symbol, timeframe=timeframe)
            await _interruptible_sleep(poll_interval_s, shutdown_event)


# ---------------------------------------------------------------------------
# Resolution loop
# ---------------------------------------------------------------------------


async def resolution_loop(
    engine: CryptoTwoSidedEngine,
    scanner: SlotScanner,
    feed: PolymarketFeed,
    manager: TradeManager,
    guard: RiskGuard,
    *,
    shutdown_event: asyncio.Event,
) -> None:
    """Every 10s, check pending resolutions and settle."""
    logger.info("resolution_loop_started")

    while not shutdown_event.is_set():
        try:
            now = time.time()
            pending = engine.get_pending_resolutions(now)

            for pos in pending:
                # Refresh prices from Gamma
                refreshed = await scanner.refresh_prices(pos.slug)
                if not refreshed:
                    logger.debug("refresh_failed", slug=pos.slug)
                    continue

                up_final = refreshed.outcome_prices.get("Up", 0.0)
                down_final = refreshed.outcome_prices.get("Down", 0.0)

                # Also try alternative labels
                if up_final == 0.0 and down_final == 0.0:
                    for key, val in refreshed.outcome_prices.items():
                        kl = key.lower()
                        if "up" in kl or "yes" in kl:
                            up_final = val
                        elif "down" in kl or "no" in kl:
                            down_final = val

                # Need decisive resolution
                if up_final < 0.9 and down_final < 0.9:
                    logger.debug(
                        "not_yet_resolved",
                        slug=pos.slug,
                        up_final=up_final,
                        down_final=down_final,
                    )
                    continue

                # Resolve
                pnl = engine.resolve(pos.condition_id, up_final, down_final)

                # Record settlement via TradeManager
                settle_intent = TradeIntent(
                    condition_id=pos.condition_id,
                    token_id=pos.up_token_id,
                    outcome="settlement",
                    side="SELL",
                    price=1.0,
                    size_usd=pos.total_cost,
                    reason="crypto_two_sided_resolution",
                    title=pos.slug,
                    edge_pct=pos.entry_edge,
                )
                settle_fill = FillResult(
                    filled=True,
                    shares=pos.up_shares + pos.down_shares,
                    avg_price=1.0,
                    pnl_delta=pnl,
                )
                await manager.record_settle_direct(
                    settle_intent,
                    settle_fill,
                    extra_state={
                        "slug": pos.slug,
                        "up_final": up_final,
                        "down_final": down_final,
                        "entry_edge": pos.entry_edge,
                    },
                )

                # Record with RiskGuard
                await guard.record_result(pnl=pnl, won=pnl > 0)

                # Unsubscribe WebSocket
                await feed.unsubscribe_market(pos.condition_id)

                logger.info(
                    "position_resolved",
                    slug=pos.slug,
                    pnl=round(pnl, 4),
                    up_final=up_final,
                    down_final=down_final,
                    total_cost=round(pos.total_cost, 2),
                    realized_pnl=round(engine.realized_pnl, 2),
                )

            # Cleanup resolved positions
            engine.cleanup_resolved()

        except asyncio.CancelledError:
            break
        except Exception:
            logger.exception("resolution_loop_error")

        await _interruptible_sleep(10.0, shutdown_event)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


async def _interruptible_sleep(duration: float, shutdown_event: asyncio.Event) -> None:
    """Sleep that can be interrupted by the shutdown event."""
    try:
        await asyncio.wait_for(shutdown_event.wait(), timeout=duration)
    except asyncio.TimeoutError:
        pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Crypto Two-Sided Arbitrage runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Market selection
    p.add_argument(
        "--symbols",
        type=str,
        default=settings.CRYPTO_TWO_SIDED_SYMBOLS,
        help="Comma-separated crypto symbols (e.g. BTCUSDT,ETHUSDT).",
    )
    p.add_argument(
        "--timeframes",
        type=str,
        default=settings.CRYPTO_TWO_SIDED_TIMEFRAMES,
        help="Comma-separated timeframes in seconds (e.g. 300,900).",
    )

    # Trading params
    p.add_argument(
        "--budget",
        type=float,
        default=settings.CRYPTO_TWO_SIDED_BUDGET_PER_MARKET,
        help="Max budget per market (USD).",
    )
    p.add_argument(
        "--min-edge",
        type=float,
        default=settings.CRYPTO_TWO_SIDED_MIN_EDGE_PCT,
        help="Minimum edge to enter (decimal, e.g. 0.01 = 1%%).",
    )
    p.add_argument(
        "--entry-window",
        type=int,
        default=settings.CRYPTO_TWO_SIDED_ENTRY_WINDOW_S,
        help="Seconds after slot open to enter.",
    )
    p.add_argument(
        "--fee-bps",
        type=int,
        default=settings.CRYPTO_TWO_SIDED_FEE_BPS,
        help="Fee in basis points (100 = 1%%).",
    )
    p.add_argument(
        "--max-concurrent",
        type=int,
        default=settings.CRYPTO_TWO_SIDED_MAX_CONCURRENT,
        help="Max concurrent open positions.",
    )
    p.add_argument(
        "--discovery-lead",
        type=int,
        default=settings.CRYPTO_TWO_SIDED_DISCOVERY_LEAD_S,
        help="Seconds before slot to start discovery polling.",
    )
    p.add_argument(
        "--poll-interval",
        type=float,
        default=settings.CRYPTO_TWO_SIDED_POLL_INTERVAL_S,
        help="Discovery poll interval (seconds).",
    )

    # Strategy tag
    p.add_argument(
        "--tag",
        type=str,
        default="crypto_2s",
        help="Strategy tag for DB grouping.",
    )

    # Execution mode
    p.add_argument(
        "--paper",
        action="store_true",
        default=True,
        help="Paper trading mode (default).",
    )
    p.add_argument(
        "--autopilot",
        action="store_true",
        default=False,
        help="Live trading mode via CLOB executor.",
    )

    # Risk params
    p.add_argument(
        "--cb-max-losses",
        type=int,
        default=5,
        help="Circuit breaker: max consecutive losses.",
    )
    p.add_argument(
        "--cb-max-drawdown",
        type=float,
        default=-50.0,
        help="Circuit breaker: max session drawdown USD.",
    )
    p.add_argument(
        "--cb-stale-seconds",
        type=float,
        default=300.0,
        help="Circuit breaker: book staleness threshold.",
    )
    p.add_argument(
        "--cb-daily-limit",
        type=float,
        default=-200.0,
        help="Global daily loss limit USD.",
    )

    # DB
    p.add_argument(
        "--db-url",
        type=str,
        default=settings.DATABASE_URL,
        help="Database URL for persistence.",
    )

    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    symbols = _parse_symbols(args.symbols)
    timeframes = _parse_timeframes(args.timeframes)
    fee_rate = args.fee_bps / 10_000.0
    paper = not args.autopilot
    tag = args.tag.strip() or "crypto_2s"
    run_id = f"{tag}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    logger.info(
        "crypto_two_sided_config",
        symbols=symbols,
        timeframes=timeframes,
        budget=args.budget,
        min_edge=args.min_edge,
        fee_rate=fee_rate,
        entry_window=args.entry_window,
        paper=paper,
        tag=tag,
        run_id=run_id,
    )

    # Engine
    engine = CryptoTwoSidedEngine(
        min_edge_pct=args.min_edge,
        budget_per_market=args.budget,
        max_concurrent=args.max_concurrent,
        entry_window_s=args.entry_window,
        fee_rate=fee_rate,
    )

    # Scanner
    scanner = SlotScanner(
        symbols=symbols,
        timeframes=timeframes,
    )

    # PolymarketFeed
    feed = PolymarketFeed()

    # Executor (live mode only)
    executor = None
    if not paper:
        executor = PolymarketExecutor.from_settings()

    # TradeManager
    manager = TradeManager(
        executor=executor,
        strategy=tag,
        paper=paper,
        db_url=args.db_url,
        event_type=CRYPTO_TWO_SIDED_EVENT_TYPE,
        run_id=run_id,
        notify_bids=False,
        notify_fills=not paper,
        notify_closes=True,
    )

    # RiskGuard
    db_path = args.db_url.replace("sqlite+aiosqlite:///", "").replace("sqlite:///", "")
    guard = RiskGuard(
        strategy_tag=tag,
        db_path=db_path,
        max_consecutive_losses=args.cb_max_losses,
        max_drawdown_usd=args.cb_max_drawdown,
        stale_seconds=args.cb_stale_seconds,
        daily_loss_limit_usd=args.cb_daily_limit,
        telegram_alerter=manager._alerter,
    )
    await guard.initialize()

    # Shutdown event
    shutdown_event = asyncio.Event()

    def _signal_handler() -> None:
        logger.info("shutdown_signal_received")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    # Connect WebSocket feed
    await feed.connect()

    try:
        # Build slot loop tasks
        slot_tasks = []
        for sym in symbols:
            for tf in timeframes:
                task = asyncio.create_task(
                    slot_loop(
                        symbol=sym,
                        timeframe=tf,
                        engine=engine,
                        scanner=scanner,
                        feed=feed,
                        manager=manager,
                        guard=guard,
                        paper=paper,
                        budget=args.budget,
                        fee_rate=fee_rate,
                        discovery_lead_s=args.discovery_lead,
                        poll_interval_s=args.poll_interval,
                        tag=tag,
                        shutdown_event=shutdown_event,
                    ),
                    name=f"slot_{sym}_{tf}",
                )
                slot_tasks.append(task)

        # Resolution loop task
        resolution_task = asyncio.create_task(
            resolution_loop(
                engine=engine,
                scanner=scanner,
                feed=feed,
                manager=manager,
                guard=guard,
                shutdown_event=shutdown_event,
            ),
            name="resolution",
        )

        all_tasks = slot_tasks + [resolution_task]

        logger.info(
            "all_loops_started",
            slot_loops=len(slot_tasks),
            symbols=symbols,
            timeframes=timeframes,
        )

        # Wait for shutdown
        await shutdown_event.wait()

        # Cancel all tasks
        for task in all_tasks:
            task.cancel()
        await asyncio.gather(*all_tasks, return_exceptions=True)

    finally:
        await feed.disconnect()
        await manager.close()
        logger.info(
            "shutdown_complete",
            realized_pnl=round(engine.realized_pnl, 4),
            stats=manager.get_stats(),
        )


if __name__ == "__main__":
    asyncio.run(main())
