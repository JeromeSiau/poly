#!/usr/bin/env python3
"""Run two-sided inventory strategy on Polymarket.

Strategy intent:
- exploit temporary mispricing between two outcomes of the same condition
- trade both sides over time (inventory-aware)
- prioritize fast exits when outcomes become over-fair or positions age out

Default mode is paper execution (fills applied locally, no live order).
Enable `--autopilot` to place real orders through Polymarket CLOB.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx
import structlog
from sqlalchemy import select

# Add project root to path for imports
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

from config.settings import settings
from src.arb.polymarket_executor import PolymarketExecutor
from src.arb.two_sided_inventory import (
    FillResult,
    MarketSnapshot,
    OutcomeQuote,
    TradeIntent,
    TwoSidedInventoryEngine,
)
from src.db.database import get_sync_session, init_db
from src.db.models import LiveObservation, PaperTrade
from src.feeds.odds_api import OddsApiClient, OddsApiSnapshot
from src.matching.bookmaker_matcher import (
    BookmakerEvent,
    BookmakerMarket,
    BookmakerMatcher,
    BookmakerMarketMatch,
)

logger = structlog.get_logger()

GAMMA_API = "https://gamma-api.polymarket.com/markets"
CLOB_API = "https://clob.polymarket.com"
TWO_SIDED_EVENT_TYPE = "two_sided_inventory"

SPORT_HINT_PATTERNS = (
    r"\bmap\s+\d+\b",
    r"\bset\s+\d+\b",
    r"\bcounter-?strike\b",
    r"\bcs2\b",
    r"\bnba\b",
    r"\bnfl\b",
    r"\bnhl\b",
    r"\bmlb\b",
    r"\bpremier league\b",
    r"\bserie a\b",
    r"\bla liga\b",
    r"\btennis\b",
    r"\bgrand slam\b",
    r"\bchampions league\b",
    r"\beuropa league\b",
    r"\bundesliga\b",
    r"\bsuper lig\b",
)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def parse_json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
            return decoded if isinstance(decoded, list) else []
        except json.JSONDecodeError:
            return []
    return []


def _looks_sports(question: str, outcomes: list[str]) -> bool:
    q = question.lower()
    if " vs " in q or " vs. " in q:
        return True

    if any(re.search(pattern, q) for pattern in SPORT_HINT_PATTERNS):
        return True

    if re.search(r"will\s+.+\s+win\s+on\s+\d{4}-\d{2}-\d{2}", q):
        return True

    if " o/u " in q or "over/under" in q:
        return True

    return False


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _ensure_sync_db_url(database_url: str) -> str:
    if not database_url:
        return "sqlite:///data/arb.db"
    if "://" not in database_url:
        return database_url
    scheme, suffix = database_url.split("://", 1)
    if "+" in scheme:
        scheme = scheme.split("+", 1)[0]
    return f"{scheme}://{suffix}"


def _parse_datetime(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str) and value:
        raw = value.strip()
        if raw.endswith("Z"):
            raw = raw.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(raw)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def _first_event_slug(raw: dict[str, Any]) -> str:
    events = raw.get("events", [])
    if isinstance(events, str):
        events = parse_json_list(events)
    if isinstance(events, list) and events:
        slug = events[0].get("slug")
        if isinstance(slug, str):
            return slug
    return str(raw.get("slug", ""))


def _parse_csv_values(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _find_market_by_id(event: BookmakerEvent, market_id: str) -> Optional[BookmakerMarket]:
    for market in event.markets:
        if market.market_id == market_id:
            return market
    return None


def _resolve_probability(odds_map: dict[str, float], mapping_key: str) -> Optional[float]:
    if not mapping_key:
        return None
    if mapping_key.startswith("1-"):
        base_key = mapping_key[2:]
        base = odds_map.get(base_key)
        if base is None:
            return None
        return _clamp(1.0 - float(base), 0.001, 0.999)
    value = odds_map.get(mapping_key)
    if value is None:
        return None
    return _clamp(float(value), 0.001, 0.999)


@dataclass(slots=True)
class ExternalFairStats:
    odds_events: int = 0
    matched_conditions: int = 0
    applied_conditions: int = 0
    credits_remaining: Optional[int] = None
    credits_used: Optional[int] = None
    credits_last_call: Optional[int] = None


class ExternalFairRuntime:
    """Odds API + matcher runtime cache for low-credit operation."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        sports: list[str],
        regions: str,
        markets: str,
        min_refresh_seconds: float,
        min_match_confidence: float,
        blend: float,
    ) -> None:
        self._sports = sports or ["upcoming"]
        self._regions = regions
        self._markets = markets
        self._min_refresh_seconds = max(1.0, min_refresh_seconds)
        self._blend = _clamp(blend, 0.0, 1.0)

        self._odds = OddsApiClient(api_key=api_key, base_url=base_url)
        self._matcher = BookmakerMatcher(min_event_confidence=min_match_confidence)

        self._last_refresh_ts: float = 0.0
        self._events_by_id: dict[str, BookmakerEvent] = {}
        self._matches_by_condition: dict[str, BookmakerMarketMatch] = {}
        self._stats = ExternalFairStats()

    async def refresh_if_needed(
        self,
        client: httpx.AsyncClient,
        raw_markets: list[dict[str, Any]],
        now_ts: float,
        force: bool = False,
    ) -> None:
        needs_refresh = force or (now_ts - self._last_refresh_ts >= self._min_refresh_seconds)
        if not needs_refresh and self._events_by_id:
            return

        snapshot: OddsApiSnapshot = await self._odds.fetch_events(
            client=client,
            sports=self._sports,
            regions=self._regions,
            markets=self._markets,
        )
        self._last_refresh_ts = now_ts
        self._events_by_id = {event.event_id: event for event in snapshot.events}

        matches = self._matcher.match_all(raw_markets, snapshot.events)
        self._matches_by_condition = {match.condition_id: match for match in matches}
        self._stats = ExternalFairStats(
            odds_events=len(snapshot.events),
            matched_conditions=len(matches),
            applied_conditions=0,
            credits_remaining=snapshot.usage.remaining,
            credits_used=snapshot.usage.used,
            credits_last_call=snapshot.usage.last,
        )

        logger.info(
            "external_fair_refreshed",
            sports=self._sports,
            odds_events=len(snapshot.events),
            matched_conditions=len(matches),
            credits_remaining=snapshot.usage.remaining,
            credits_used=snapshot.usage.used,
            credits_last_call=snapshot.usage.last,
        )

    def fair_for_snapshot(
        self,
        snapshot: MarketSnapshot,
        market_fair: dict[str, float],
    ) -> Optional[dict[str, float]]:
        match = self._matches_by_condition.get(snapshot.condition_id)
        if match is None:
            return None

        event = self._events_by_id.get(match.bookmaker_event_id)
        if event is None:
            return None
        market = _find_market_by_id(event, match.bookmaker_market_id)
        if market is None:
            return None

        external: dict[str, float] = {}
        for outcome in snapshot.outcome_order:
            mapping_key = match.outcome_map.get(outcome, "")
            probability = _resolve_probability(market.outcomes, mapping_key)
            if probability is None:
                return None
            external[outcome] = probability

        # Keep binary outcomes normalized.
        total = sum(external.values())
        if total > 0:
            external = {k: _clamp(v / total, 0.001, 0.999) for k, v in external.items()}

        if self._blend < 1.0:
            external = {
                outcome: _clamp(
                    self._blend * external[outcome] + (1.0 - self._blend) * market_fair[outcome],
                    0.001,
                    0.999,
                )
                for outcome in snapshot.outcome_order
            }

        self._stats.applied_conditions += 1
        return external

    @property
    def stats(self) -> ExternalFairStats:
        return self._stats


class TwoSidedPaperRecorder:
    """Persist and replay two-sided paper fills using existing dashboard tables."""

    def __init__(self, database_url: str) -> None:
        self._database_url = _ensure_sync_db_url(database_url)

    def bootstrap(self) -> None:
        init_db(self._database_url)

    def replay_into_engine(self, engine: TwoSidedInventoryEngine) -> int:
        stmt = (
            select(PaperTrade, LiveObservation)
            .join(LiveObservation, LiveObservation.id == PaperTrade.observation_id)
            .where(LiveObservation.event_type == TWO_SIDED_EVENT_TYPE)
            .order_by(PaperTrade.created_at.asc(), PaperTrade.id.asc())
        )
        session = get_sync_session(self._database_url)
        try:
            rows = session.execute(stmt).all()
        finally:
            session.close()

        restored = 0
        for trade, observation in rows:
            game_state = observation.game_state if isinstance(observation.game_state, dict) else {}
            condition_id = str(game_state.get("condition_id") or observation.match_id or "")
            outcome = str(game_state.get("outcome") or "")
            token_id = str(game_state.get("token_id") or "")
            title = str(game_state.get("title") or condition_id or "restored")
            side = str(trade.side or game_state.get("side") or "").upper()
            price = _to_float(
                trade.simulated_fill_price if trade.simulated_fill_price is not None else trade.entry_price,
                default=0.0,
            )
            size_usd = _to_float(trade.size, default=0.0)
            if not condition_id or not outcome or side not in {"BUY", "SELL"}:
                continue
            if price <= 0 or size_usd <= 0:
                continue

            timestamp = trade.created_at or observation.timestamp
            if timestamp is None:
                ts = time.time()
            else:
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                ts = timestamp.timestamp()

            restored_intent = TradeIntent(
                condition_id=condition_id,
                title=title,
                outcome=outcome,
                token_id=token_id,
                side=side,
                price=price,
                size_usd=size_usd,
                edge_pct=_to_float(trade.edge_theoretical, default=0.0),
                reason=str(game_state.get("reason") or "restore"),
                timestamp=ts,
            )
            fill = engine.apply_fill(restored_intent)
            if fill.shares > 0:
                restored += 1
        return restored

    def persist_fill(
        self,
        *,
        intent: TradeIntent,
        fill: FillResult,
        snapshot: Optional[MarketSnapshot],
        fair_prices: dict[str, float],
        execution_mode: str,
    ) -> None:
        if fill.shares <= 0:
            return

        fair_price = _to_float(fair_prices.get(intent.outcome), default=intent.price)
        quote = snapshot.outcomes.get(intent.outcome) if snapshot else None
        observation_ts = datetime.fromtimestamp(intent.timestamp, tz=timezone.utc)

        game_state = {
            "strategy": "two_sided_inventory",
            "condition_id": intent.condition_id,
            "title": intent.title,
            "slug": snapshot.slug if snapshot else "",
            "outcome": intent.outcome,
            "token_id": intent.token_id,
            "side": intent.side,
            "reason": intent.reason,
            "mode": execution_mode,
            "fair_price": fair_price,
            "edge_theoretical": intent.edge_pct,
            "fill_price": fill.fill_price,
            "shares": fill.shares,
            "size_usd": intent.size_usd,
            "inventory_avg_price": fill.avg_price,
            "inventory_remaining_shares": fill.remaining_shares,
            "liquidity": snapshot.liquidity if snapshot else None,
            "volume_24h": snapshot.volume_24h if snapshot else None,
            "market_bid": quote.bid if quote else None,
            "market_ask": quote.ask if quote else None,
        }

        edge_realized: Optional[float] = None
        pnl: Optional[float] = None
        exit_price: Optional[float] = None
        if intent.side == "SELL":
            pnl = fill.realized_pnl_delta
            edge_realized = pnl / intent.size_usd if intent.size_usd > 0 else 0.0
            exit_price = fill.fill_price

        observation = LiveObservation(
            timestamp=observation_ts,
            match_id=intent.condition_id,
            event_type=TWO_SIDED_EVENT_TYPE,
            game_state=game_state,
            model_prediction=fair_price,
            polymarket_price=fill.fill_price,
        )

        trade = PaperTrade(
            observation_id=0,  # set after flush
            side=intent.side,
            entry_price=intent.price,
            simulated_fill_price=fill.fill_price,
            size=intent.size_usd,
            edge_theoretical=intent.edge_pct,
            edge_realized=edge_realized,
            exit_price=exit_price,
            pnl=pnl,
            created_at=observation_ts,
        )

        session = get_sync_session(self._database_url)
        try:
            session.add(observation)
            session.flush()
            trade.observation_id = int(observation.id)
            session.add(trade)
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


async def fetch_markets(
    client: httpx.AsyncClient,
    limit: int,
    min_liquidity: float,
    min_volume_24h: float,
    sports_only: bool,
    max_days_to_end: float,
) -> list[dict[str, Any]]:
    all_markets: list[dict[str, Any]] = []
    offset = 0
    batch = 100

    while len(all_markets) < limit:
        response = await client.get(
            GAMMA_API,
            params={
                "limit": min(batch, limit - len(all_markets)),
                "offset": offset,
                "active": "true",
                "closed": "false",
            },
        )
        response.raise_for_status()
        rows = response.json()
        if not isinstance(rows, list) or not rows:
            break

        for raw in rows:
            outcomes = parse_json_list(raw.get("outcomes", []))
            clob_ids = parse_json_list(raw.get("clobTokenIds", []))
            question = str(raw.get("question", ""))
            liquidity = _to_float(raw.get("liquidityNum"))
            volume_24h = _to_float(raw.get("volume24hr"))

            if len(outcomes) != 2 or len(clob_ids) < 2:
                continue
            if not question:
                continue
            if liquidity < min_liquidity or volume_24h < min_volume_24h:
                continue
            if sports_only and not _looks_sports(question, [str(o) for o in outcomes]):
                continue
            if max_days_to_end > 0:
                end_dt = _parse_datetime(raw.get("endDate"))
                if end_dt is None:
                    continue
                delta_days = (end_dt - datetime.now(timezone.utc)).total_seconds() / 86400.0
                if delta_days < -1.0 or delta_days > max_days_to_end:
                    continue

            all_markets.append(raw)
            if len(all_markets) >= limit:
                break

        if len(rows) < batch:
            break
        offset += len(rows)

    return all_markets


def _parse_orderbook_level(level: Any) -> tuple[Optional[float], Optional[float]]:
    if isinstance(level, dict):
        return _to_float(level.get("price"), default=-1.0), _to_float(level.get("size"), default=0.0)
    if isinstance(level, list) and len(level) >= 2:
        return _to_float(level[0], default=-1.0), _to_float(level[1], default=0.0)
    return None, None


def _best_orderbook_level(
    levels: list[Any],
    side: str,
) -> tuple[Optional[float], Optional[float]]:
    """Extract top-of-book robustly.

    CLOB `/book` arrays are not guaranteed in the order this runner expects,
    so we explicitly pick max(bid) and min(ask) by price.
    """
    best_price: Optional[float] = None
    best_size: Optional[float] = None

    for level in levels:
        price, size = _parse_orderbook_level(level)
        if price is None or size is None:
            continue
        if not (0 < price < 1) or size <= 0:
            continue

        if best_price is None:
            best_price = price
            best_size = size
            continue

        if side == "bid" and price > best_price:
            best_price = price
            best_size = size
        elif side == "ask" and price < best_price:
            best_price = price
            best_size = size

    return best_price, best_size


async def fetch_book(
    client: httpx.AsyncClient,
    token_id: str,
    semaphore: asyncio.Semaphore,
) -> dict[str, Optional[float]]:
    async with semaphore:
        try:
            response = await client.get(f"{CLOB_API}/book", params={"token_id": token_id})
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            logger.debug(
                "book_fetch_error",
                token_id=token_id,
                error=repr(exc),
                error_type=type(exc).__name__,
            )
            return {"bid": None, "ask": None, "bid_size": None, "ask_size": None}

    bids = payload.get("bids", []) if isinstance(payload, dict) else []
    asks = payload.get("asks", []) if isinstance(payload, dict) else []

    best_bid, bid_size = _best_orderbook_level(bids, side="bid")
    best_ask, ask_size = _best_orderbook_level(asks, side="ask")

    return {
        "bid": best_bid,
        "ask": best_ask,
        "bid_size": bid_size,
        "ask_size": ask_size,
    }


async def build_snapshots(
    client: httpx.AsyncClient,
    markets: list[dict[str, Any]],
    max_concurrency: int,
) -> list[MarketSnapshot]:
    token_ids: set[str] = set()
    for raw in markets:
        for token in parse_json_list(raw.get("clobTokenIds", []))[:2]:
            token_ids.add(str(token))

    semaphore = asyncio.Semaphore(max_concurrency)
    book_tasks = {
        token_id: asyncio.create_task(fetch_book(client, token_id, semaphore))
        for token_id in token_ids
    }
    books = {token_id: await task for token_id, task in book_tasks.items()}

    now_ts = time.time()
    snapshots: list[MarketSnapshot] = []

    for raw in markets:
        outcomes = [str(o) for o in parse_json_list(raw.get("outcomes", []))]
        clob_ids = [str(t) for t in parse_json_list(raw.get("clobTokenIds", []))]
        if len(outcomes) != 2 or len(clob_ids) < 2:
            continue

        outcome_quotes: dict[str, OutcomeQuote] = {}
        for idx, outcome in enumerate(outcomes[:2]):
            token_id = clob_ids[idx]
            book = books.get(token_id, {})
            outcome_quotes[outcome] = OutcomeQuote(
                outcome=outcome,
                token_id=token_id,
                bid=book.get("bid"),
                ask=book.get("ask"),
                bid_size=book.get("bid_size"),
                ask_size=book.get("ask_size"),
            )

        # Need at least one tradable side.
        if not any(q.bid is not None or q.ask is not None for q in outcome_quotes.values()):
            continue

        snapshots.append(
            MarketSnapshot(
                condition_id=str(raw.get("conditionId", "")),
                title=str(raw.get("question", "")),
                slug=_first_event_slug(raw),
                outcome_order=outcomes[:2],
                outcomes=outcome_quotes,
                timestamp=now_ts,
                liquidity=_to_float(raw.get("liquidityNum")),
                volume_24h=_to_float(raw.get("volume24hr")),
            )
        )

    return snapshots


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
    paper_recorder: Optional[TwoSidedPaperRecorder],
    args: argparse.Namespace,
    signal_memory: dict[tuple[str, str, str, int], float],
) -> None:
    raw_markets = await fetch_markets(
        client=client,
        limit=args.limit,
        min_liquidity=args.min_liquidity,
        min_volume_24h=args.min_volume_24h,
        sports_only=not args.include_nonsports,
        max_days_to_end=args.max_days_to_end,
    )
    snapshots = await build_snapshots(
        client=client,
        markets=raw_markets,
        max_concurrency=args.max_book_concurrency,
    )

    now = time.time()
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

    print(
        f"\n[{time.strftime('%H:%M:%S')}] markets={len(raw_markets)} "
        f"snapshots={len(snapshots)} intents={len(all_intents)}"
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
    snapshots_by_condition = {s.condition_id: s for s in snapshots}

    executed = 0
    failures = 0

    for intent in all_intents[: args.max_orders_per_cycle]:
        if args.autopilot and executor is not None:
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
                if paper_recorder is not None:
                    try:
                        paper_recorder.persist_fill(
                            intent=intent,
                            fill=fill,
                            snapshot=snapshots_by_condition.get(intent.condition_id),
                            fair_prices=fair_cache.get(intent.condition_id, {}),
                            execution_mode="autopilot",
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
            fill = engine.apply_fill(intent)
            executed += 1
            if paper_recorder is not None:
                try:
                    paper_recorder.persist_fill(
                        intent=intent,
                        fill=fill,
                        snapshot=snapshots_by_condition.get(intent.condition_id),
                        fair_prices=fair_cache.get(intent.condition_id, {}),
                        execution_mode="paper",
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
    print(
        f"Executed this cycle: {executed} (failures={failures}) | "
        f"Inventory open={int(inv['open_positions'])} marked=${inv['marked_notional']:,.2f} "
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
        "--fair-blend",
        type=float,
        default=settings.TWO_SIDED_EXTERNAL_FAIR_BLEND,
        help="Blend weight for external fair vs market fair (1.0=external only).",
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
    return parser


def build_executor_if_needed(autopilot: bool) -> Optional[PolymarketExecutor]:
    if not autopilot:
        return None

    if not settings.POLYMARKET_PRIVATE_KEY or not settings.POLYMARKET_WALLET_ADDRESS:
        raise RuntimeError("Autopilot requires POLYMARKET_PRIVATE_KEY and POLYMARKET_WALLET_ADDRESS.")

    return PolymarketExecutor(
        host=settings.POLYMARKET_CLOB_HTTP,
        chain_id=settings.POLYMARKET_CHAIN_ID,
        private_key=settings.POLYMARKET_PRIVATE_KEY,
        funder=settings.POLYMARKET_WALLET_ADDRESS,
        api_key=settings.POLYMARKET_API_KEY or None,
        api_secret=settings.POLYMARKET_API_SECRET or None,
        api_passphrase=settings.POLYMARKET_API_PASSPHRASE or None,
    )


async def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

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
    )
    executor = build_executor_if_needed(args.autopilot)
    fair_runtime: Optional[ExternalFairRuntime] = None
    if args.external_fair:
        if not settings.ODDS_API_KEY:
            raise RuntimeError("External fair requires ODDS_API_KEY in .env")
        fair_runtime = ExternalFairRuntime(
            api_key=settings.ODDS_API_KEY,
            base_url=settings.ODDS_API_BASE_URL,
            sports=_parse_csv_values(args.odds_sports),
            regions=args.odds_regions,
            markets=args.odds_markets,
            min_refresh_seconds=args.odds_refresh_seconds,
            min_match_confidence=args.odds_min_confidence,
            blend=args.fair_blend,
        )
    signal_memory: dict[tuple[str, str, str, int], float] = {}
    paper_recorder: Optional[TwoSidedPaperRecorder] = None
    if args.persist_paper:
        paper_recorder = TwoSidedPaperRecorder(settings.DATABASE_URL)
        paper_recorder.bootstrap()
        if args.resume_paper and args.paper_fill and not args.autopilot:
            restored = paper_recorder.replay_into_engine(engine)
            if restored:
                logger.info(
                    "paper_inventory_restored",
                    fills=restored,
                    realized_pnl=engine.get_realized_pnl(),
                )

    timeout = httpx.Timeout(20.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        if args.mode == "scan":
            await run_cycle(client, engine, executor, fair_runtime, paper_recorder, args, signal_memory)
            return

        while True:
            try:
                await run_cycle(client, engine, executor, fair_runtime, paper_recorder, args, signal_memory)
            except Exception as exc:
                logger.error("watch_cycle_error", error=str(exc))
            await asyncio.sleep(args.interval)


if __name__ == "__main__":
    asyncio.run(main())
