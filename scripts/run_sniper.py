#!/usr/bin/env python3
"""Event-driven sports sniper daemon.

Detects live sports events via Polymarket CLOB orderbook price spikes
and Odds API score changes, then buys the winning outcome across all
related conditions before the market fully converges.

Usage:
    .venv/bin/python scripts/run_sniper.py \
        --scores-sports soccer_epl,soccer_la_liga \
        --scores-interval 120 \
        --spike-threshold 0.15 \
        --max-order 10 \
        --strategy-tag sniper_sports
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import httpx
import structlog

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import Settings
from src.analysis.event_condition_mapper import EventConditionMapper
from src.arb.sniper_router import SniperRouter, SniperAction
from src.arb.two_sided_inventory import TradeIntent, FillResult, TwoSidedInventoryEngine
from src.feeds.odds_api import OddsApiClient, ScoreTracker
from src.feeds.spike_detector import SpikeDetector

# Lazy import to avoid circular deps at module level
TwoSidedPaperRecorder: Any = None

logger = structlog.get_logger()

GAMMA_API = "https://gamma-api.polymarket.com/markets"
CLOB_BOOK_URL = "https://clob.polymarket.com/book"


async def fetch_sport_markets(
    client: httpx.AsyncClient,
    limit: int = 1500,
    event_prefixes: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """Fetch active binary sport markets from Gamma API."""
    all_markets: list[dict[str, Any]] = []
    batch = 100
    offset = 0

    while len(all_markets) < limit:
        try:
            resp = await client.get(
                GAMMA_API,
                params={
                    "limit": min(batch, limit - len(all_markets)),
                    "offset": offset,
                    "active": "true",
                    "closed": "false",
                },
            )
            if resp.status_code != 200:
                break
            rows = resp.json()
        except Exception as exc:
            logger.warning("gamma_fetch_error", error=str(exc))
            break

        if not isinstance(rows, list) or not rows:
            break

        for raw in rows:
            outcomes = json.loads(raw.get("outcomes", "[]")) if isinstance(raw.get("outcomes"), str) else (raw.get("outcomes") or [])
            clob_ids = json.loads(raw.get("clobTokenIds", "[]")) if isinstance(raw.get("clobTokenIds"), str) else (raw.get("clobTokenIds") or [])
            if len(outcomes) != 2 or len(clob_ids) < 2:
                continue
            # Skip already-resolved markets (any outcome price >= 0.95)
            try:
                raw_prices = json.loads(raw.get("outcomePrices", "[]")) if isinstance(raw.get("outcomePrices"), str) else (raw.get("outcomePrices") or [])
                if raw_prices and any(float(p) >= 0.95 for p in raw_prices):
                    continue
            except (ValueError, TypeError):
                pass
            # Event prefix filter
            events = raw.get("events", [])
            if isinstance(events, str):
                try:
                    events = json.loads(events)
                except Exception:
                    events = []
            slug = events[0].get("slug", "") if isinstance(events, list) and events and isinstance(events[0], dict) else ""
            if event_prefixes and slug:
                prefix = slug.split("-", 1)[0]
                if prefix not in event_prefixes:
                    continue
            elif event_prefixes and not slug:
                continue
            all_markets.append(raw)
        offset += batch
        if len(rows) < batch:
            break

    return all_markets


async def fetch_orderbook(
    client: httpx.AsyncClient,
    token_id: str,
) -> dict[str, Any]:
    """Fetch CLOB orderbook for a single token."""
    try:
        resp = await client.get(CLOB_BOOK_URL, params={"token_id": token_id})
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {}


def action_to_intent(
    action: SniperAction,
    mapper: EventConditionMapper,
    price: float,
    size_usd: float,
) -> TradeIntent:
    """Convert a SniperAction to a TradeIntent for the engine."""
    token_ids = mapper.token_ids_for(action.condition_id)
    entry = mapper._cid_to_entry.get(action.condition_id, {})
    outcomes = entry.get("outcomes", [])
    token_id = ""
    if outcomes and token_ids:
        try:
            idx = outcomes.index(action.outcome)
            token_id = token_ids[idx] if idx < len(token_ids) else ""
        except ValueError:
            pass

    return TradeIntent(
        condition_id=action.condition_id,
        title=entry.get("question", ""),
        outcome=action.outcome,
        token_id=token_id,
        side="BUY",
        price=price,
        size_usd=size_usd,
        edge_pct=0.0,
        reason=f"sniper:{action.reason}",
    )


async def scores_loop(
    client: httpx.AsyncClient,
    odds_client: OddsApiClient,
    tracker: ScoreTracker,
    router: SniperRouter,
    action_queue: asyncio.Queue,
    sports: list[str],
    interval: float,
) -> None:
    """Poll Odds API /scores and emit actions on changes."""
    while True:
        try:
            snapshot = await odds_client.fetch_scores(client, sports=sports)
            changes = tracker.update(snapshot.games)
            for change in changes:
                actions = router.route_score_change(change)
                for action in actions:
                    await action_queue.put(action)
                if actions:
                    logger.info(
                        "score_change_detected",
                        event_id=change.event_id,
                        score=f"{change.home_score}-{change.away_score}",
                        change_type=change.change_type,
                        actions=len(actions),
                    )
            if snapshot.usage.remaining is not None:
                logger.debug("odds_api_credits", remaining=snapshot.usage.remaining)
        except Exception as exc:
            logger.warning("scores_loop_error", error=str(exc))
        await asyncio.sleep(interval)


async def spike_monitor_loop(
    client: httpx.AsyncClient,
    detector: SpikeDetector,
    router: SniperRouter,
    mapper: EventConditionMapper,
    action_queue: asyncio.Queue,
    poll_interval: float = 2.0,
    book_concurrency: int = 40,
) -> None:
    """Poll CLOB orderbooks for all tracked tokens and detect spikes."""
    while True:
        try:
            all_entries = list(mapper._cid_to_entry.values())
            sem = asyncio.Semaphore(book_concurrency)
            now = time.time()

            async def check_condition(entry: dict) -> None:
                cid = entry["conditionId"]
                token_ids = entry.get("clobTokenIds", [])
                outcomes = entry.get("outcomes", [])

                for i, tid in enumerate(token_ids):
                    if i >= len(outcomes):
                        break
                    async with sem:
                        book = await fetch_orderbook(client, tid)
                    bids = book.get("bids", [])
                    asks = book.get("asks", [])
                    if not bids and not asks:
                        continue
                    best_bid = float(bids[0]["price"]) if bids else 0.0
                    best_ask = float(asks[0]["price"]) if asks else 0.0
                    mid = (best_bid + best_ask) / 2 if best_bid > 0 and best_ask > 0 else max(best_bid, best_ask)
                    if mid <= 0:
                        continue

                    spikes = detector.observe(cid, outcomes[i], mid, now)
                    for spike in spikes:
                        actions = router.route_spike(spike)
                        for action in actions:
                            await action_queue.put(action)
                        if actions:
                            logger.info(
                                "spike_detected",
                                condition_id=cid,
                                outcome=outcomes[i],
                                delta=f"{spike.delta:.3f}",
                                direction=spike.direction,
                                actions=len(actions),
                            )

            tasks = [check_condition(e) for e in all_entries]
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as exc:
            logger.warning("spike_loop_error", error=str(exc))
        await asyncio.sleep(poll_interval)


async def execution_loop(
    client: httpx.AsyncClient,
    action_queue: asyncio.Queue,
    mapper: EventConditionMapper,
    engine: TwoSidedInventoryEngine,
    max_order_usd: float,
    strategy_tag: str,
    paper_recorder: Any = None,
    dry_run: bool = False,
) -> None:
    """Consume actions from queue, fetch live ask, and paper-fill."""
    while True:
        action: SniperAction = await action_queue.get()
        try:
            # Fetch live ask price for the target token
            token_ids = mapper.token_ids_for(action.condition_id)
            entry = mapper._cid_to_entry.get(action.condition_id, {})
            outcomes = entry.get("outcomes", [])
            token_id = ""
            ask_price = 0.0

            if outcomes and token_ids:
                try:
                    idx = outcomes.index(action.outcome)
                    token_id = token_ids[idx] if idx < len(token_ids) else ""
                except ValueError:
                    pass

            if token_id:
                book = await fetch_orderbook(client, token_id)
                asks = book.get("asks", [])
                if asks:
                    ask_price = float(asks[0]["price"])

            if ask_price <= 0 or ask_price >= 0.99:
                logger.debug("sniper_skip_no_ask", condition_id=action.condition_id, ask=ask_price)
                continue

            intent = action_to_intent(action, mapper, price=ask_price, size_usd=max_order_usd)

            if dry_run:
                logger.info(
                    "sniper_dry_run",
                    condition_id=action.condition_id,
                    outcome=action.outcome,
                    ask=f"{ask_price:.4f}",
                    reason=action.reason,
                    event_slug=action.source_event_slug,
                    tag=strategy_tag,
                )
            else:
                fill = engine.apply_fill(intent)
                if fill and fill.shares > 0:
                    logger.info(
                        "sniper_fill",
                        condition_id=action.condition_id,
                        outcome=action.outcome,
                        side="BUY",
                        shares=f"{fill.shares:.4f}",
                        price=f"{fill.fill_price:.4f}",
                        reason=action.reason,
                        event_slug=action.source_event_slug,
                        tag=strategy_tag,
                    )
                    if paper_recorder is not None:
                        try:
                            paper_recorder.persist_fill(
                                intent=intent,
                                fill=fill,
                                snapshot=None,
                                fair_prices={intent.outcome: intent.price},
                                execution_mode="paper",
                            )
                        except Exception as exc:
                            logger.warning("paper_db_persist_failed", error=repr(exc))
        except Exception as exc:
            logger.warning("execution_error", error=str(exc), condition_id=action.condition_id)
        finally:
            action_queue.task_done()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Event-driven sports sniper")
    p.add_argument("--scores-sports", type=str,
                    default="soccer_epl,soccer_spain_la_liga,soccer_italy_serie_a,soccer_brazil_campeonato,soccer_argentina_primera_division")
    p.add_argument("--scores-interval", type=float, default=120.0,
                    help="Seconds between Odds API score polls")
    p.add_argument("--spike-threshold", type=float, default=0.15,
                    help="Min price move to trigger spike (0.15 = 15 cents)")
    p.add_argument("--spike-window", type=float, default=60.0,
                    help="Spike detection window in seconds")
    p.add_argument("--spike-cooldown", type=float, default=120.0,
                    help="Min seconds between spikes on same condition")
    p.add_argument("--spike-poll-interval", type=float, default=2.0,
                    help="Seconds between CLOB orderbook polls")
    p.add_argument("--book-concurrency", type=int, default=40)
    p.add_argument("--market-limit", type=int, default=1500)
    p.add_argument("--event-prefixes", type=str,
                    default="epl,lal,sea,fl1,por,bun,tur,arg,col1,nba,nfl,cbb,atp,wta,ucl,cs2,super")
    p.add_argument("--max-order", type=float, default=10.0)
    p.add_argument("--mapper-refresh-seconds", type=float, default=300.0,
                    help="Re-fetch Gamma markets every N seconds")
    p.add_argument("--strategy-tag", type=str, default="sniper_sports")
    p.add_argument("--dry-run", action="store_true",
                    help="Log actions without recording paper fills")
    p.add_argument("--db-url", type=str, default="sqlite+aiosqlite:///data/arb.db")
    p.add_argument("--wallet-usd", type=float, default=200.0)
    p.add_argument("--max-outcome-inv", type=float, default=100.0)
    p.add_argument("--max-market-net", type=float, default=80.0)
    return p


async def main() -> None:
    args = build_parser().parse_args()
    settings = Settings()

    odds_client = OddsApiClient(api_key=settings.ODDS_API_KEY)
    tracker = ScoreTracker()
    detector = SpikeDetector(
        threshold_pct=args.spike_threshold,
        window_seconds=args.spike_window,
        cooldown_seconds=args.spike_cooldown,
    )
    mapper = EventConditionMapper()
    action_queue: asyncio.Queue[SniperAction] = asyncio.Queue()

    engine = TwoSidedInventoryEngine(
        max_order_usd=args.max_order,
        max_outcome_inventory_usd=args.max_outcome_inv,
        max_market_net_usd=args.max_market_net,
    )

    # Paper recorder for DB persistence (same as run_two_sided_inventory.py)
    paper_recorder = None
    if not args.dry_run:
        from scripts.run_two_sided_inventory import TwoSidedPaperRecorder
        run_id = f"sniper_{uuid.uuid4().hex[:8]}"
        paper_recorder = TwoSidedPaperRecorder(
            args.db_url,
            strategy_tag=args.strategy_tag,
            run_id=run_id,
            min_edge_pct=0.0,
            exit_edge_pct=0.0,
            event_type="sniper_sports",
        )
        paper_recorder.bootstrap()
        logger.info("paper_recorder_ready", strategy_tag=args.strategy_tag, run_id=run_id)

    prefixes = [p.strip() for p in args.event_prefixes.split(",") if p.strip()] or None
    scores_sports = [s.strip() for s in args.scores_sports.split(",") if s.strip()]

    async with httpx.AsyncClient(timeout=20.0) as client:
        # Initial market fetch + mapper build
        logger.info("sniper_fetching_markets", limit=args.market_limit)
        raw_markets = await fetch_sport_markets(client, limit=args.market_limit, event_prefixes=prefixes)
        mapper.build(raw_markets)
        n_conditions = len(mapper._cid_to_entry)
        n_events = len(mapper._slug_to_cids)
        logger.info("sniper_mapper_built", conditions=n_conditions, events=n_events)

        if n_conditions == 0:
            logger.warning("sniper_no_conditions_found")

        router = SniperRouter(mapper=mapper)

        # Launch parallel loops
        loop_tasks = [
            asyncio.create_task(scores_loop(
                client, odds_client, tracker, router, action_queue,
                sports=scores_sports, interval=args.scores_interval,
            )),
            asyncio.create_task(spike_monitor_loop(
                client, detector, router, mapper, action_queue,
                poll_interval=args.spike_poll_interval,
                book_concurrency=args.book_concurrency,
            )),
            asyncio.create_task(execution_loop(
                client, action_queue, mapper, engine,
                max_order_usd=args.max_order,
                strategy_tag=args.strategy_tag,
                paper_recorder=paper_recorder,
                dry_run=args.dry_run,
            )),
        ]

        # Periodic mapper refresh
        async def refresh_mapper():
            while True:
                await asyncio.sleep(args.mapper_refresh_seconds)
                try:
                    new_markets = await fetch_sport_markets(client, limit=args.market_limit, event_prefixes=prefixes)
                    mapper.build(new_markets)
                    logger.info("sniper_mapper_refreshed", conditions=len(mapper._cid_to_entry))
                except Exception as exc:
                    logger.warning("sniper_mapper_refresh_error", error=str(exc))

        loop_tasks.append(asyncio.create_task(refresh_mapper()))

        logger.info(
            "sniper_started",
            scores_sports=scores_sports,
            spike_threshold=args.spike_threshold,
            strategy_tag=args.strategy_tag,
            dry_run=args.dry_run,
        )
        await asyncio.gather(*loop_tasks)


if __name__ == "__main__":
    asyncio.run(main())
