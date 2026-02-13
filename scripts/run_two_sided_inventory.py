#!/usr/bin/env python3
"""Run two-sided inventory strategy on Polymarket.

Strategy intent:
- exploit temporary mispricing between two outcomes of the same condition
- trade both sides over time (inventory-aware)
- prioritize fast exits when outcomes become over-fair or positions age out

Default mode is paper execution (fills applied locally, no live order).
Enable ``--autopilot`` to place real orders through Polymarket CLOB.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
import structlog

from src.utils.logging import configure_logging

configure_logging()

from config.settings import settings
from src.arb.polymarket_executor import PolymarketExecutor
from src.arb.two_sided_inventory import (
    FillResult,
    MarketSnapshot,
    OutcomeQuote,
    TradeIntent,
    TwoSidedInventoryEngine,
)
from src.execution import TradeManager

# ---------------------------------------------------------------------------
# Extracted modules (Phase 6)
# ---------------------------------------------------------------------------

from src.arb.external_fair import (  # noqa: E402
    ExternalFairRuntime,
    ExternalFairStats,
    _build_gamma_timing_fair,
    _find_market_by_id,
    _lookup_outcome_price,
    _resolve_probability,
)
from src.execution.paper_recorder import (  # noqa: E402
    PendingMakerOrder,
    TwoSidedPaperRecorder,
    _build_extra_state,
    _to_exec_fill,
    _to_exec_intent,
)
from src.feeds.gamma_api import (  # noqa: E402
    CLOB_API,
    DEFAULT_BOOK_404_COOLDOWN_SECONDS,
    GAMMA_API,
    SPORT_HINT_PATTERNS,
    _BOOK_404_SUPPRESS_UNTIL,
    _best_orderbook_level,
    _looks_sports,
    _parse_orderbook_level,
    build_snapshots,
    fetch_book,
    fetch_markets,
)

# Re-exported utilities (now in src/utils/)
from src.utils.parsing import (  # noqa: E402
    _clamp,
    _ensure_sync_db_url,
    _extract_outcome_price_map,
    _first_event_slug,
    _normalize_outcome_label,
    _parse_csv_values,
    _parse_datetime,
    _to_bool,
    _to_float,
    parse_json_list,
)
from src.utils.crypto_markets import CRYPTO_SYMBOL_TO_SLUG, fetch_crypto_markets  # noqa: E402

logger = structlog.get_logger()

GAMMA_EVENTS_API = "https://gamma-api.polymarket.com/events"
TWO_SIDED_EVENT_TYPE = "two_sided_inventory"
DEFAULT_SETTLEMENT_WINNER_MIN_PRICE = 0.985
DEFAULT_SETTLEMENT_LOSER_MAX_PRICE = 0.015
DEFAULT_SETTLEMENT_FETCH_CHUNK = 40
DEFAULT_SETTLEMENT_ENDDATE_GRACE_SECONDS = 300.0
DEFAULT_PAIR_MERGE_PRICE = 0.5
DEFAULT_PAIR_MERGE_MIN_EDGE = 0.002
PAIR_BUNDLE_REASONS = {"pair_arb_entry", "pair_arb_exit", "pair_merge", "maker_pair_arb_entry"}


@dataclass(slots=True)
class ResolvedCondition:
    condition_id: str
    title: str
    outcome_prices: dict[str, float]


def _parse_resolved_binary_market(
    raw: dict[str, Any],
    *,
    winner_min_price: float,
    loser_max_price: float,
    now_ts: float,
    allow_ended_open: bool,
    enddate_grace_seconds: float,
) -> Optional[ResolvedCondition]:
    condition_id = str(raw.get("conditionId") or "")
    if not condition_id:
        return None

    is_closed = _to_bool(raw.get("closed"))
    is_ended_open = False
    if not is_closed and allow_ended_open:
        end_dt = _parse_datetime(raw.get("endDate"))
        if end_dt is not None:
            grace = max(0.0, enddate_grace_seconds)
            is_ended_open = now_ts >= (end_dt.timestamp() + grace)

    if not is_closed and not is_ended_open:
        return None

    outcomes = [str(item) for item in parse_json_list(raw.get("outcomes", []))]
    prices_raw = parse_json_list(raw.get("outcomePrices", []))
    if len(outcomes) != 2 or len(prices_raw) != len(outcomes):
        return None

    outcome_prices: dict[str, float] = {}
    for outcome, price_raw in zip(outcomes, prices_raw):
        price = _to_float(price_raw, default=-1.0)
        if not (0.0 <= price <= 1.0):
            return None
        outcome_prices[outcome] = price

    winner_price = max(outcome_prices.values())
    loser_price = min(outcome_prices.values())
    if winner_price < winner_min_price or loser_price > loser_max_price:
        return None

    return ResolvedCondition(
        condition_id=condition_id,
        title=str(raw.get("question") or condition_id),
        outcome_prices=outcome_prices,
    )


def inventory_mark_summary(
    engine: TwoSidedInventoryEngine,
    snapshots: list[MarketSnapshot],
    fair_cache: Optional[dict[str, dict[str, float]]] = None,
) -> dict[str, float]:
    mark: dict[tuple[str, str], float] = {}
    for snapshot in snapshots:
        fair = fair_cache.get(snapshot.condition_id) if fair_cache else None
        if fair is None:
            fair = engine.compute_fair_prices(snapshot)
        for outcome in snapshot.outcome_order:
            mark[(snapshot.condition_id, outcome)] = fair[outcome]

    total_notional = 0.0
    open_positions = 0
    for condition_id, by_outcome in engine.get_open_inventory().items():
        for outcome, state in by_outcome.items():
            px = mark.get((condition_id, outcome), state.avg_price)
            total_notional += state.notional(px)
            open_positions += 1

    return {
        "open_positions": float(open_positions),
        "marked_notional": total_notional,
        "realized_pnl": engine.get_realized_pnl(),
    }


async def fetch_resolved_conditions(
    client: httpx.AsyncClient,
    condition_ids: list[str],
    *,
    now_ts: float,
    winner_min_price: float,
    loser_max_price: float,
    allow_ended_open: bool,
    enddate_grace_seconds: float,
    fetch_chunk_size: int,
) -> dict[str, ResolvedCondition]:
    resolved: dict[str, ResolvedCondition] = {}
    if not condition_ids:
        return resolved

    unique_ids = [cid for cid in dict.fromkeys(condition_ids) if cid]
    chunk_size = max(1, fetch_chunk_size)
    for idx in range(0, len(unique_ids), chunk_size):
        chunk = unique_ids[idx: idx + chunk_size]
        try:
            params = [("condition_ids", cid) for cid in chunk]
            response = await client.get(
                GAMMA_API,
                params=params,
            )
            response.raise_for_status()
            rows = response.json()
        except Exception as exc:
            logger.warning(
                "resolution_fetch_error",
                chunk_size=len(chunk),
                error=repr(exc),
            )
            continue

        if not isinstance(rows, list):
            continue

        for raw in rows:
            if not isinstance(raw, dict):
                continue
            parsed = _parse_resolved_binary_market(
                raw,
                winner_min_price=winner_min_price,
                loser_max_price=loser_max_price,
                now_ts=now_ts,
                allow_ended_open=allow_ended_open,
                enddate_grace_seconds=enddate_grace_seconds,
            )
            if parsed is not None:
                resolved[parsed.condition_id] = parsed

    return resolved


def settle_resolved_inventory(
    engine: TwoSidedInventoryEngine,
    resolved_conditions: dict[str, ResolvedCondition],
    *,
    manager: Optional[TradeManager] = None,
    paper_recorder: Optional[TwoSidedPaperRecorder] = None,
    now_ts: float,
) -> int:
    if not resolved_conditions:
        return 0

    settled = 0
    open_inventory = engine.get_open_inventory()
    for condition_id, by_outcome in open_inventory.items():
        resolved = resolved_conditions.get(condition_id)
        if resolved is None:
            continue

        for outcome in list(by_outcome.keys()):
            settlement_price = _lookup_outcome_price(resolved.outcome_prices, outcome)
            if settlement_price is None:
                logger.debug(
                    "settlement_outcome_missing",
                    condition_id=condition_id,
                    outcome=outcome,
                    resolved_outcomes=list(resolved.outcome_prices.keys()),
                )
                continue

            fill = engine.settle_position(
                condition_id=condition_id,
                outcome=outcome,
                settlement_price=settlement_price,
                timestamp=now_ts,
            )
            if fill.shares <= 0:
                continue
            settled += 1

            logger.info(
                "paper_position_settled",
                condition_id=condition_id,
                outcome=outcome,
                settlement_price=settlement_price,
                shares=fill.shares,
                realized_pnl=fill.realized_pnl_delta,
            )

            intent = TradeIntent(
                condition_id=condition_id,
                title=resolved.title,
                outcome=outcome,
                token_id="",
                side="SELL",
                price=settlement_price,
                size_usd=fill.shares * settlement_price,
                edge_pct=0.0,
                reason="settlement_closed_market",
                timestamp=now_ts,
            )

            if manager is not None and manager.recorder is not None:
                extra = _build_extra_state(
                    intent=intent, fill=fill, snapshot=None,
                )
                try:
                    manager.recorder.record_settle(
                        intent=_to_exec_intent(intent),
                        fill=_to_exec_fill(fill),
                        fair_prices={outcome: settlement_price},
                        extra_state=extra,
                    )
                except Exception as exc:
                    logger.warning(
                        "paper_db_persist_failed",
                        condition_id=condition_id,
                        outcome=outcome,
                        side="SELL",
                        reason="settlement_closed_market",
                        error=repr(exc),
                    )
            elif paper_recorder is not None:
                try:
                    paper_recorder.persist_fill(
                        intent=intent,
                        fill=fill,
                        snapshot=None,
                        fair_prices={outcome: settlement_price},
                        execution_mode="paper_settlement",
                    )
                except Exception as exc:
                    logger.warning(
                        "paper_db_persist_failed",
                        condition_id=condition_id,
                        outcome=outcome,
                        side="SELL",
                        reason="settlement_closed_market",
                        error=repr(exc),
                    )

    return settled


def paper_merge_binary_pairs(
    engine: TwoSidedInventoryEngine,
    snapshots_by_condition: dict[str, MarketSnapshot],
    *,
    manager: Optional[TradeManager] = None,
    paper_recorder: Optional[TwoSidedPaperRecorder] = None,
    now_ts: float,
    min_edge_pct: float,
    max_pair_notional_usd: float,
) -> int:
    """Paper-only merge for binary complete sets.

    When both outcomes are held, merge equal shares and realize pair value at $1.
    We model this as two synthetic SELL fills at 0.5 (one per leg), which preserves:
        pnl_pair = 1 - (avg_yes + avg_no)
    """
    merged_pairs = 0
    open_inventory = engine.get_open_inventory()

    for condition_id, by_outcome in open_inventory.items():
        snapshot = snapshots_by_condition.get(condition_id)
        if snapshot is None or len(snapshot.outcome_order) != 2:
            continue

        out_a, out_b = snapshot.outcome_order
        state_a = by_outcome.get(out_a)
        state_b = by_outcome.get(out_b)
        if state_a is None or state_b is None:
            continue
        if state_a.shares <= 0 or state_b.shares <= 0:
            continue

        pair_cost = state_a.avg_price + state_b.avg_price
        merge_edge = 1.0 - pair_cost
        if merge_edge < min_edge_pct:
            continue

        merge_shares = min(state_a.shares, state_b.shares)
        if max_pair_notional_usd > 0:
            # One complete set (Yes+No) settles to $1.
            merge_shares = min(merge_shares, max_pair_notional_usd)
        if merge_shares <= 1e-9:
            continue

        leg_size_usd = merge_shares * DEFAULT_PAIR_MERGE_PRICE
        realized_total = 0.0
        fair_prices = {out_a: DEFAULT_PAIR_MERGE_PRICE, out_b: DEFAULT_PAIR_MERGE_PRICE}
        for outcome in (out_a, out_b):
            quote = snapshot.outcomes.get(outcome)
            intent = TradeIntent(
                condition_id=condition_id,
                title=snapshot.title,
                outcome=outcome,
                token_id=quote.token_id if quote else "",
                side="SELL",
                price=DEFAULT_PAIR_MERGE_PRICE,
                size_usd=leg_size_usd,
                edge_pct=merge_edge,
                reason="pair_merge",
                timestamp=now_ts,
            )
            fill = engine.apply_fill(intent)
            if fill.shares <= 0:
                continue
            realized_total += fill.realized_pnl_delta

            if manager is not None and manager.recorder is not None:
                extra = _build_extra_state(
                    intent=intent, fill=fill, snapshot=snapshot,
                )
                try:
                    manager.recorder.record_fill(
                        intent=_to_exec_intent(intent),
                        fill=_to_exec_fill(fill),
                        fair_prices=fair_prices,
                        execution_mode="paper_merge",
                        extra_state=extra,
                    )
                except Exception as exc:
                    logger.warning(
                        "paper_db_persist_failed",
                        condition_id=condition_id,
                        outcome=outcome,
                        side="SELL",
                        reason="pair_merge",
                        error=repr(exc),
                    )
            elif paper_recorder is not None:
                try:
                    paper_recorder.persist_fill(
                        intent=intent,
                        fill=fill,
                        snapshot=snapshot,
                        fair_prices=fair_prices,
                        execution_mode="paper_merge",
                    )
                except Exception as exc:
                    logger.warning(
                        "paper_db_persist_failed",
                        condition_id=condition_id,
                        outcome=outcome,
                        side="SELL",
                        reason="pair_merge",
                        error=repr(exc),
                    )

        merged_pairs += 1
        logger.info(
            "paper_pair_merged",
            condition_id=condition_id,
            title=snapshot.title,
            merge_shares=merge_shares,
            pair_cost=pair_cost,
            merge_edge=merge_edge,
            realized_pnl=realized_total,
        )

    return merged_pairs


def select_intents_for_execution(intents: list[TradeIntent], max_orders_per_cycle: int) -> list[TradeIntent]:
    """Pick intents while keeping pair bundles together.

    If a pair bundle has two legs, executing only one leg creates inventory drift.
    This selector keeps same-condition pair intents atomic, even when cap is small.
    """
    cap = max(0, int(max_orders_per_cycle))
    if cap <= 0 or not intents:
        return []

    pair_groups: dict[tuple[str, str, str], list[int]] = {}
    for idx, intent in enumerate(intents):
        reason = str(intent.reason or "")
        if reason in PAIR_BUNDLE_REASONS:
            key = (intent.condition_id, intent.side, reason)
            pair_groups.setdefault(key, []).append(idx)

    selected: list[TradeIntent] = []
    used: set[int] = set()

    for idx, intent in enumerate(intents):
        if idx in used:
            continue

        reason = str(intent.reason or "")
        group_idxs = [idx]
        if reason in PAIR_BUNDLE_REASONS:
            key = (intent.condition_id, intent.side, reason)
            group_idxs = sorted(pair_groups.get(key, [idx]), key=lambda i: intents[i].outcome)

        needed = len(group_idxs)
        remaining = cap - len(selected)
        allow_pair_overflow = len(selected) == 0 and needed > remaining and needed <= 2
        if needed <= remaining or allow_pair_overflow:
            for gi in group_idxs:
                if gi in used:
                    continue
                selected.append(intents[gi])
                used.add(gi)
            if len(selected) >= cap and not allow_pair_overflow:
                break
            continue

        if len(selected) >= cap:
            break

    return selected


def signal_key(intent: TradeIntent) -> tuple[str, str, str, int]:
    return (
        intent.condition_id,
        intent.outcome,
        intent.side,
        int(round(intent.price * 10000)),
    )


def should_execute_response(response: dict[str, Any]) -> bool:
    status = str(response.get("status", "")).upper()
    if status == "ERROR":
        return False
    return bool(status)


def print_intents(
    intents: list[TradeIntent],
    fair_cache: dict[str, dict[str, float]],
    top_n: int,
) -> None:
    print(f"\n{'#':>3}  {'Edge':>7}  {'Side':>4}  {'Size$':>8}  {'Px':>6}  {'Outcome':>12}  Title")
    print("-" * 120)
    for idx, intent in enumerate(intents[:top_n], start=1):
        fair = fair_cache.get(intent.condition_id, {}).get(intent.outcome)
        edge = intent.edge_pct
        fair_s = f"{fair:.3f}" if fair is not None else "n/a"
        print(
            f"{idx:>3}  {edge:>7.2%}  {intent.side:>4}  {intent.size_usd:>8.2f}  "
            f"{intent.price:>6.3f}  {intent.outcome[:12]:>12}  "
            f"{intent.title[:50]} (fair={fair_s}, {intent.reason})"
        )


async def run_cycle(
    client: httpx.AsyncClient,
    engine: TwoSidedInventoryEngine,
    executor: Optional[PolymarketExecutor],
    fair_runtime: Optional[ExternalFairRuntime],
    manager: Optional[TradeManager],
    args: argparse.Namespace,
    signal_memory: dict[tuple[str, str, str, int], float],
    pending_maker_orders: Optional[list[PendingMakerOrder]] = None,
) -> None:
    crypto_symbols = _parse_csv_values(getattr(args, "crypto_symbols", ""))
    if crypto_symbols:
        raw_markets = await fetch_crypto_markets(
            client=client,
            symbols=crypto_symbols,
        )
    else:
        raw_markets = await fetch_markets(
            client=client,
            limit=args.limit,
            min_liquidity=args.min_liquidity,
            min_volume_24h=args.min_volume_24h,
            sports_only=not args.include_nonsports,
            max_days_to_end=args.max_days_to_end,
            event_prefixes=_parse_csv_values(args.event_prefixes),
            entry_require_ended=args.entry_require_ended,
            entry_min_seconds_since_end=args.entry_min_seconds_since_end,
        )
    snapshots = await build_snapshots(
        client=client,
        markets=raw_markets,
        max_concurrency=args.max_book_concurrency,
    )
    raw_market_by_condition = {
        str(raw.get("conditionId", "")): raw
        for raw in raw_markets
        if raw.get("conditionId")
    }

    now = time.time()
    snapshots_by_condition = {s.condition_id: s for s in snapshots}

    # -- Maker mode: check pending limit orders for simulated fills --
    maker_fills = 0
    if pending_maker_orders is not None and pending_maker_orders:
        still_pending: list[PendingMakerOrder] = []
        for pending in pending_maker_orders:
            snapshot = snapshots_by_condition.get(pending.intent.condition_id)
            if snapshot is None:
                logger.debug("maker_order_expired_no_market", condition_id=pending.intent.condition_id)
                continue
            age = now - pending.placed_at
            if age > args.max_hold_seconds:
                logger.debug(
                    "maker_order_expired_age",
                    condition_id=pending.intent.condition_id,
                    outcome=pending.intent.outcome,
                    age_seconds=round(age, 1),
                )
                continue

            quote = snapshot.outcomes.get(pending.intent.outcome)
            filled = False
            if quote is not None:
                if pending.intent.side == "BUY" and quote.ask is not None:
                    filled = quote.ask <= pending.intent.price
                elif pending.intent.side == "SELL" and quote.bid is not None:
                    filled = quote.bid >= pending.intent.price

            if filled:
                fill = engine.apply_fill(pending.intent)
                maker_fills += 1
                logger.info(
                    "maker_order_filled",
                    condition_id=pending.intent.condition_id,
                    outcome=pending.intent.outcome,
                    side=pending.intent.side,
                    price=pending.intent.price,
                    age_seconds=round(age, 1),
                )
                if manager is not None and manager.recorder is not None and args.persist_paper:
                    fair_prices = engine.compute_fair_prices(snapshot)
                    extra = _build_extra_state(
                        intent=pending.intent, fill=fill, snapshot=snapshot,
                    )
                    try:
                        manager.recorder.record_fill(
                            intent=_to_exec_intent(pending.intent),
                            fill=_to_exec_fill(fill),
                            fair_prices=fair_prices,
                            execution_mode="maker_paper",
                            extra_state=extra,
                        )
                    except Exception as exc:
                        logger.warning(
                            "paper_db_persist_failed",
                            condition_id=pending.intent.condition_id,
                            outcome=pending.intent.outcome,
                            error=repr(exc),
                        )
            else:
                still_pending.append(pending)
        pending_maker_orders.clear()
        pending_maker_orders.extend(still_pending)

    settled = 0
    merged = 0
    if args.settle_resolved and args.paper_fill:
        open_inventory = engine.get_open_inventory()
        if open_inventory:
            resolved_conditions = await fetch_resolved_conditions(
                client=client,
                condition_ids=list(open_inventory.keys()),
                now_ts=now,
                winner_min_price=args.settlement_winner_min_price,
                loser_max_price=args.settlement_loser_max_price,
                allow_ended_open=args.settlement_allow_ended_open,
                enddate_grace_seconds=args.settlement_enddate_grace_seconds,
                fetch_chunk_size=args.settlement_fetch_chunk,
            )
            settled = settle_resolved_inventory(
                engine=engine,
                resolved_conditions=resolved_conditions,
                manager=manager if args.persist_paper else None,
                now_ts=now,
            )
    if args.paper_fill and args.pair_merge:
        merged = paper_merge_binary_pairs(
            engine=engine,
            snapshots_by_condition=snapshots_by_condition,
            manager=manager if args.persist_paper else None,
            now_ts=now,
            min_edge_pct=args.pair_merge_min_edge,
            max_pair_notional_usd=args.max_order,
        )

    if fair_runtime is not None:
        await fair_runtime.refresh_if_needed(client=client, raw_markets=raw_markets, now_ts=now)

    all_intents: list[TradeIntent] = []
    fair_cache: dict[str, dict[str, float]] = {}

    for snapshot in snapshots:
        market_fair = engine.compute_fair_prices(snapshot)
        fair = market_fair
        if fair_runtime is not None:
            external = fair_runtime.fair_for_snapshot(snapshot=snapshot, market_fair=market_fair)
            if external is not None:
                fair = external
        if args.timing_gamma_proxy and fair_runtime is None:
            proxy_fair = _build_gamma_timing_fair(
                snapshot=snapshot,
                raw_market=raw_market_by_condition.get(snapshot.condition_id),
                now_ts=now,
                min_prob=args.timing_gamma_proxy_min_prob,
                min_gap=args.timing_gamma_proxy_min_gap,
                require_ended=args.timing_gamma_proxy_require_ended,
            )
            if proxy_fair is not None:
                fair = proxy_fair
        fair_cache[snapshot.condition_id] = fair
        intents = engine.evaluate_market(snapshot, fair_prices=fair, now_ts=now)
        for intent in intents:
            key = signal_key(intent)
            last_ts = signal_memory.get(key, 0.0)
            if args.signal_cooldown > 0 and now - last_ts < args.signal_cooldown:
                continue
            signal_memory[key] = now
            all_intents.append(intent)

    all_intents.sort(key=lambda x: x.edge_pct * x.size_usd, reverse=True)

    pending_count = len(pending_maker_orders) if pending_maker_orders is not None else 0
    print(
        f"\n[{time.strftime('%H:%M:%S')}] markets={len(raw_markets)} "
        f"snapshots={len(snapshots)} intents={len(all_intents)} settled={settled} merged={merged}"
        + (f" maker_fills={maker_fills} pending={pending_count}" if args.maker_mode else "")
    )
    if fair_runtime is not None:
        stats = fair_runtime.stats
        print(
            f"External fair: events={stats.odds_events}, "
            f"matched={stats.matched_conditions}, applied={stats.applied_conditions}, "
            f"credits_last={stats.credits_last_call}, remaining={stats.credits_remaining}"
        )

    if not all_intents:
        inv = inventory_mark_summary(engine, snapshots, fair_cache=fair_cache)
        print(
            f"Inventory: open={int(inv['open_positions'])}, "
            f"marked=${inv['marked_notional']:,.2f}, realized=${inv['realized_pnl']:,.2f}"
        )
        return

    print_intents(all_intents, fair_cache=fair_cache, top_n=args.top)

    executed = 0
    failures = 0

    intents_to_execute = select_intents_for_execution(all_intents, args.max_orders_per_cycle)
    for intent in intents_to_execute:
        if args.autopilot and executor is not None:
            if intent.reason == "pair_merge":
                logger.warning(
                    "pair_merge_skipped_in_autopilot",
                    condition_id=intent.condition_id,
                    outcome=intent.outcome,
                    reason=intent.reason,
                )
                continue
            response = await executor.place_order(
                token_id=intent.token_id,
                side=intent.side,
                size=intent.size_usd,
                price=intent.price,
                outcome=intent.outcome,
            )
            ok = should_execute_response(response)
            status = response.get("status", "UNKNOWN")
            if ok:
                fill = engine.apply_fill(intent)
                executed += 1
                logger.info(
                    "order_executed",
                    condition_id=intent.condition_id,
                    outcome=intent.outcome,
                    side=intent.side,
                    size=intent.size_usd,
                    price=intent.price,
                    status=status,
                )
                if manager is not None and manager.recorder is not None:
                    snapshot = snapshots_by_condition.get(intent.condition_id)
                    extra = _build_extra_state(
                        intent=intent, fill=fill, snapshot=snapshot,
                    )
                    try:
                        manager.recorder.record_fill(
                            intent=_to_exec_intent(intent),
                            fill=_to_exec_fill(fill),
                            fair_prices=fair_cache.get(intent.condition_id, {}),
                            execution_mode="autopilot",
                            extra_state=extra,
                        )
                    except Exception as exc:
                        logger.warning(
                            "paper_db_persist_failed",
                            condition_id=intent.condition_id,
                            outcome=intent.outcome,
                            side=intent.side,
                            error=repr(exc),
                        )
            else:
                failures += 1
                logger.warning(
                    "order_failed",
                    condition_id=intent.condition_id,
                    outcome=intent.outcome,
                    side=intent.side,
                    status=status,
                    response=response,
                )
        elif args.paper_fill:
            if args.maker_mode and pending_maker_orders is not None:
                pending_maker_orders.append(PendingMakerOrder(intent=intent, placed_at=now))
                executed += 1
                logger.info(
                    "maker_order_queued",
                    condition_id=intent.condition_id,
                    outcome=intent.outcome,
                    side=intent.side,
                    price=intent.price,
                    pending_total=len(pending_maker_orders),
                )
            else:
                fill = engine.apply_fill(intent)
                executed += 1
                if manager is not None and manager.recorder is not None:
                    snapshot = snapshots_by_condition.get(intent.condition_id)
                    extra = _build_extra_state(
                        intent=intent, fill=fill, snapshot=snapshot,
                    )
                    try:
                        manager.recorder.record_fill(
                            intent=_to_exec_intent(intent),
                            fill=_to_exec_fill(fill),
                            fair_prices=fair_cache.get(intent.condition_id, {}),
                            execution_mode="paper",
                            extra_state=extra,
                        )
                    except Exception as exc:
                        logger.warning(
                            "paper_db_persist_failed",
                            condition_id=intent.condition_id,
                            outcome=intent.outcome,
                            side=intent.side,
                            error=repr(exc),
                        )

    inv = inventory_mark_summary(engine, snapshots, fair_cache=fair_cache)
    pending_count = len(pending_maker_orders) if pending_maker_orders is not None else 0
    print(
        f"Executed this cycle: {executed} (failures={failures}, settled={settled}, merged={merged}"
        + (f", maker_fills={maker_fills}, pending={pending_count}" if args.maker_mode else "")
        + f") | Inventory open={int(inv['open_positions'])} marked=${inv['marked_notional']:,.2f} "
        f"realized=${inv['realized_pnl']:,.2f}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Two-sided inventory arbitrage runner")
    parser.add_argument("mode", choices=["scan", "watch"], default="scan", nargs="?")
    parser.add_argument("--limit", type=int, default=250, help="Max number of active markets to scan.")
    parser.add_argument("--top", type=int, default=20, help="Number of top intents to print.")
    parser.add_argument(
        "--interval",
        type=float,
        default=settings.TWO_SIDED_SCAN_INTERVAL,
        help="Watch mode polling interval (seconds).",
    )
    parser.add_argument(
        "--min-liquidity",
        type=float,
        default=settings.TWO_SIDED_MIN_LIQUIDITY,
        help="Minimum market liquidity.",
    )
    parser.add_argument(
        "--min-volume-24h",
        type=float,
        default=settings.TWO_SIDED_MIN_VOLUME_24H,
        help="Minimum 24h volume.",
    )
    parser.add_argument(
        "--max-days-to-end",
        type=float,
        default=settings.TWO_SIDED_MAX_DAYS_TO_END,
        help="Keep markets ending within N days (0 disables the filter).",
    )
    parser.add_argument("--include-nonsports", action="store_true", help="Include non-sports markets.")
    parser.add_argument(
        "--crypto-symbols",
        type=str,
        default="",
        help="Comma-separated crypto symbols (e.g. BTCUSDT,ETHUSDT). When set, "
        "discovers 15-min binary markets via /events instead of /markets.",
    )
    parser.add_argument(
        "--event-prefixes",
        type=str,
        default="",
        help="Optional comma-separated event slug prefixes filter (e.g. epl,cs2,lal).",
    )
    parser.add_argument(
        "--entry-require-ended",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only consider markets whose endDate has passed.",
    )
    parser.add_argument(
        "--entry-min-seconds-since-end",
        type=float,
        default=0.0,
        help="When --entry-require-ended is enabled, minimum seconds since market endDate.",
    )
    parser.add_argument(
        "--max-book-concurrency",
        type=int,
        default=settings.TWO_SIDED_MAX_BOOK_CONCURRENCY,
        help="Concurrent orderbook requests.",
    )
    parser.add_argument(
        "--signal-cooldown",
        type=float,
        default=settings.TWO_SIDED_SIGNAL_COOLDOWN_SECONDS,
        help="Cooldown per identical signal key.",
    )
    parser.add_argument(
        "--max-orders-per-cycle",
        type=int,
        default=settings.TWO_SIDED_MAX_ORDERS_PER_CYCLE,
        help="Cap orders/fills per cycle.",
    )
    parser.add_argument(
        "--external-fair",
        action=argparse.BooleanOptionalAction,
        default=bool(settings.ODDS_API_KEY),
        help="Use Odds API + matcher to build external fair prices.",
    )
    parser.add_argument(
        "--odds-sports",
        type=str,
        default=settings.ODDS_API_SPORTS,
        help="Comma-separated Odds API sport keys (e.g. upcoming,soccer_epl).",
    )
    parser.add_argument(
        "--odds-regions",
        type=str,
        default=settings.ODDS_API_REGIONS,
        help="Odds API regions parameter.",
    )
    parser.add_argument(
        "--odds-markets",
        type=str,
        default=settings.ODDS_API_MARKETS,
        help="Odds API markets parameter.",
    )
    parser.add_argument(
        "--odds-refresh-seconds",
        type=float,
        default=settings.ODDS_API_MIN_REFRESH_SECONDS,
        help="Minimum seconds between Odds API refreshes.",
    )
    parser.add_argument(
        "--odds-min-confidence",
        type=float,
        default=settings.ODDS_MATCH_MIN_CONFIDENCE,
        help="Minimum event match confidence for external fair.",
    )
    parser.add_argument(
        "--odds-shared-cache",
        action=argparse.BooleanOptionalAction,
        default=settings.ODDS_SHARED_CACHE_ENABLED,
        help="Share Odds API snapshots through DB cache across daemon instances.",
    )
    parser.add_argument(
        "--odds-shared-cache-ttl-seconds",
        type=float,
        default=settings.ODDS_SHARED_CACHE_TTL_SECONDS,
        help="TTL for shared Odds API cache (0 = use --odds-refresh-seconds).",
    )
    parser.add_argument(
        "--fair-blend",
        type=float,
        default=settings.TWO_SIDED_EXTERNAL_FAIR_BLEND,
        help="Blend weight for external fair vs market fair (1.0=external only).",
    )
    parser.add_argument(
        "--timing-gamma-proxy",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use Gamma outcomePrices as winner-proxy fair on strong post-event dislocations.",
    )
    parser.add_argument(
        "--timing-gamma-proxy-min-prob",
        type=float,
        default=0.80,
        help="Minimum top outcome probability in Gamma outcomePrices to activate timing proxy.",
    )
    parser.add_argument(
        "--timing-gamma-proxy-min-gap",
        type=float,
        default=0.25,
        help="Minimum probability gap between outcomes to activate timing proxy.",
    )
    parser.add_argument(
        "--timing-gamma-proxy-require-ended",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require endDate to be passed before using timing gamma proxy fair.",
    )

    # Engine parameters
    parser.add_argument(
        "--min-edge",
        type=float,
        default=settings.TWO_SIDED_MIN_EDGE_PCT,
        help="Minimum net edge to open.",
    )
    parser.add_argument(
        "--exit-edge",
        type=float,
        default=settings.TWO_SIDED_EXIT_EDGE_PCT,
        help="Edge threshold to close.",
    )
    parser.add_argument(
        "--min-order",
        type=float,
        default=settings.TWO_SIDED_MIN_ORDER_USD,
        help="Minimum order notional in USD.",
    )
    parser.add_argument(
        "--max-order",
        type=float,
        default=settings.TWO_SIDED_MAX_ORDER_USD,
        help="Maximum order notional in USD.",
    )
    parser.add_argument(
        "--max-outcome-inv",
        type=float,
        default=settings.TWO_SIDED_MAX_OUTCOME_INVENTORY_USD,
        help="Max inventory per outcome (USD).",
    )
    parser.add_argument(
        "--max-market-net",
        type=float,
        default=settings.TWO_SIDED_MAX_MARKET_NET_USD,
        help="Max directional net per market (USD).",
    )
    parser.add_argument(
        "--inventory-skew",
        type=float,
        default=settings.TWO_SIDED_INVENTORY_SKEW_PCT,
        help="Inventory skew penalty.",
    )
    parser.add_argument(
        "--max-hold-seconds",
        type=float,
        default=settings.TWO_SIDED_MAX_HOLD_SECONDS,
        help="Max hold age before stale exit.",
    )
    parser.add_argument(
        "--buy-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disable SELL intents from over-fair/max-hold/inventory logic and pair exits; keep BUY intents.",
    )
    parser.add_argument(
        "--allow-pair-exit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow pair_arb_exit SELL intents when bid_yes + bid_no is rich enough.",
    )
    parser.add_argument(
        "--pair-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only generate pair-level intents (pair_arb_entry/pair_arb_exit), no single-leg under_fair entries.",
    )
    parser.add_argument(
        "--maker-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Market-making mode: place limit orders at bid instead of crossing the spread. "
        "Paper fills are queued and simulated via orderbook crossing.",
    )
    parser.add_argument(
        "--pair-merge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Paper-only: merge binary complete sets (Yes+No) into $1 when pair edge is positive enough.",
    )
    parser.add_argument(
        "--pair-merge-min-edge",
        type=float,
        default=DEFAULT_PAIR_MERGE_MIN_EDGE,
        help="Minimum pair merge edge (1 - avg_yes - avg_no) to trigger paper merge.",
    )
    parser.add_argument(
        "--settle-resolved",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Settle open paper inventory when market has decisive 0/1 outcome prices (closed or ended-open).",
    )
    parser.add_argument(
        "--settlement-winner-min-price",
        type=float,
        default=DEFAULT_SETTLEMENT_WINNER_MIN_PRICE,
        help="Minimum winner price to treat closed market as resolved.",
    )
    parser.add_argument(
        "--settlement-loser-max-price",
        type=float,
        default=DEFAULT_SETTLEMENT_LOSER_MAX_PRICE,
        help="Maximum loser price to treat market as resolved.",
    )
    parser.add_argument(
        "--settlement-allow-ended-open",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow settlement when market endDate is passed even if Gamma still reports closed=false.",
    )
    parser.add_argument(
        "--settlement-enddate-grace-seconds",
        type=float,
        default=DEFAULT_SETTLEMENT_ENDDATE_GRACE_SECONDS,
        help="Grace delay after endDate before settling ended-open markets.",
    )
    parser.add_argument(
        "--settlement-fetch-chunk",
        type=int,
        default=DEFAULT_SETTLEMENT_FETCH_CHUNK,
        help="Condition-id batch size for settlement resolution lookup.",
    )

    # Execution mode
    parser.add_argument("--autopilot", action="store_true", help="Place live orders via CLOB executor.")
    parser.add_argument(
        "--paper-fill",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply intents as local paper fills.",
    )
    parser.add_argument(
        "--persist-paper",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Persist executed fills to existing paper trading DB tables.",
    )
    parser.add_argument(
        "--resume-paper",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Replay persisted two-sided fills at startup to restore inventory state.",
    )
    parser.add_argument(
        "--strategy-tag",
        type=str,
        default="default",
        help="Experiment tag for this run (used for DB grouping and replay isolation).",
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default=settings.DATABASE_URL,
        help="Override database URL for paper persistence/reporting.",
    )
    return parser


def build_executor_if_needed(autopilot: bool) -> Optional[PolymarketExecutor]:
    if not autopilot:
        return None
    return PolymarketExecutor.from_settings()


async def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    strategy_tag = args.strategy_tag.strip() or "default"
    run_id = f"{strategy_tag}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    engine = TwoSidedInventoryEngine(
        min_edge_pct=args.min_edge,
        exit_edge_pct=args.exit_edge,
        min_order_usd=args.min_order,
        max_order_usd=args.max_order,
        max_outcome_inventory_usd=args.max_outcome_inv,
        max_market_net_usd=args.max_market_net,
        inventory_skew_pct=args.inventory_skew,
        max_hold_seconds=args.max_hold_seconds,
        fee_bps=settings.POLYMARKET_FEE_BPS,
        enable_sells=not args.buy_only,
        allow_pair_exit=(args.allow_pair_exit and not args.buy_only),
        allow_single_leg_entries=not args.pair_only,
        maker_mode=args.maker_mode,
    )
    executor = build_executor_if_needed(args.autopilot)
    fair_runtime: Optional[ExternalFairRuntime] = None
    if args.external_fair:
        if not settings.ODDS_API_KEY:
            raise RuntimeError("External fair requires ODDS_API_KEY in .env")
        shared_cache_ttl = (
            args.odds_shared_cache_ttl_seconds
            if args.odds_shared_cache_ttl_seconds > 0
            else args.odds_refresh_seconds
        )
        fair_runtime = ExternalFairRuntime(
            api_key=settings.ODDS_API_KEY,
            base_url=settings.ODDS_API_BASE_URL,
            sports=_parse_csv_values(args.odds_sports),
            regions=args.odds_regions,
            markets=args.odds_markets,
            min_refresh_seconds=args.odds_refresh_seconds,
            min_match_confidence=args.odds_min_confidence,
            blend=args.fair_blend,
            shared_cache_db_url=args.db_url if args.odds_shared_cache else "",
            shared_cache_ttl_seconds=shared_cache_ttl,
        )
    signal_memory: dict[tuple[str, str, str, int], float] = {}
    pending_maker_orders: list[PendingMakerOrder] = [] if args.maker_mode else []

    # TradeManager replaces TwoSidedPaperRecorder for DB + Telegram.
    # TwoSidedPaperRecorder is still used only for replay_into_engine().
    manager: Optional[TradeManager] = None
    if args.persist_paper:
        manager = TradeManager(
            strategy=strategy_tag,
            paper=not args.autopilot,
            db_url=args.db_url,
            event_type=TWO_SIDED_EVENT_TYPE,
            run_id=run_id,
            notify_bids=False,
            notify_fills=args.autopilot,
            notify_closes=args.autopilot,
        )
        # Replay uses TwoSidedPaperRecorder (reads DB, feeds engine state)
        if args.resume_paper:
            replay_recorder = TwoSidedPaperRecorder(
                args.db_url,
                strategy_tag=strategy_tag,
                run_id=run_id,
                min_edge_pct=args.min_edge,
                exit_edge_pct=args.exit_edge,
            )
            replay_recorder.bootstrap()
            restored = replay_recorder.replay_into_engine(engine)
            logger.info(
                "paper_inventory_restored",
                strategy_tag=strategy_tag,
                fills=restored,
                open_conditions=len(engine.get_open_inventory()),
                realized_pnl=engine.get_realized_pnl(),
                mode="autopilot" if args.autopilot else ("paper" if args.paper_fill else "scan_only"),
            )
    logger.info(
        "runner_configuration",
        strategy_tag=strategy_tag,
        run_id=run_id,
        db_url=args.db_url,
        min_edge_pct=args.min_edge,
        exit_edge_pct=args.exit_edge,
    )

    timeout = httpx.Timeout(20.0, connect=10.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            if args.mode == "scan":
                await run_cycle(client, engine, executor, fair_runtime, manager, args, signal_memory, pending_maker_orders)
                return

            while True:
                try:
                    await run_cycle(client, engine, executor, fair_runtime, manager, args, signal_memory, pending_maker_orders)
                except Exception as exc:
                    logger.error("watch_cycle_error", error=str(exc))
                await asyncio.sleep(args.interval)
    finally:
        if manager is not None:
            await manager.close()


if __name__ == "__main__":
    asyncio.run(main())
