"""Gamma Markets API helpers for Polymarket.

Fetches active markets, orderbooks and builds ``MarketSnapshot`` objects
used by two-sided inventory and related strategies.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
import structlog

from src.utils.parsing import (
    _to_float,
    _first_event_slug,
    _parse_datetime,
    parse_json_list,
)

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Shared data structures (originally in two_sided_inventory)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class OutcomeQuote:
    """Best quote levels for one outcome."""

    outcome: str
    token_id: str
    bid: Optional[float]
    ask: Optional[float]
    bid_size: Optional[float] = None  # shares
    ask_size: Optional[float] = None  # shares

    @property
    def mid(self) -> Optional[float]:
        if self.bid is None and self.ask is None:
            return None
        if self.bid is None:
            return self.ask
        if self.ask is None:
            return self.bid
        return (self.bid + self.ask) / 2.0


@dataclass(slots=True)
class MarketSnapshot:
    """Current market state required for decisioning."""

    condition_id: str
    title: str
    outcome_order: list[str]
    outcomes: dict[str, OutcomeQuote]
    timestamp: float
    liquidity: float = 0.0
    volume_24h: float = 0.0
    slug: str = ""


GAMMA_API = "https://gamma-api.polymarket.com/markets"
CLOB_API = "https://clob.polymarket.com"
DEFAULT_BOOK_404_COOLDOWN_SECONDS = 600.0

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

_BOOK_404_SUPPRESS_UNTIL: dict[str, float] = {}


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

    CLOB ``/book`` arrays are not guaranteed in the order this runner expects,
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


async def fetch_markets(
    client: httpx.AsyncClient,
    limit: int,
    min_liquidity: float,
    min_volume_24h: float,
    sports_only: bool,
    max_days_to_end: float,
    event_prefixes: list[str],
    entry_require_ended: bool,
    entry_min_seconds_since_end: float,
) -> list[dict[str, Any]]:
    all_markets: list[dict[str, Any]] = []
    offset = 0
    batch = 100
    prefixes = [p.strip().lower() for p in event_prefixes if p.strip()]

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
            event_slug = _first_event_slug(raw).strip().lower()

            if len(outcomes) != 2 or len(clob_ids) < 2:
                continue
            if not question:
                continue
            if liquidity < min_liquidity or volume_24h < min_volume_24h:
                continue
            if prefixes:
                if not event_slug:
                    continue
                if not any(event_slug == p or event_slug.startswith(f"{p}-") for p in prefixes):
                    continue
            if sports_only and not _looks_sports(question, [str(o) for o in outcomes]):
                continue
            end_dt: Optional[datetime] = None
            if max_days_to_end > 0 or entry_require_ended:
                end_dt = _parse_datetime(raw.get("endDate"))
                if end_dt is None:
                    continue

            if max_days_to_end > 0 and end_dt is not None:
                delta_days = (end_dt - datetime.now(timezone.utc)).total_seconds() / 86400.0
                if delta_days < -1.0 or delta_days > max_days_to_end:
                    continue

            if entry_require_ended and end_dt is not None:
                seconds_since_end = (datetime.now(timezone.utc) - end_dt).total_seconds()
                if seconds_since_end < entry_min_seconds_since_end:
                    continue

            all_markets.append(raw)
            if len(all_markets) >= limit:
                break

        if len(rows) < batch:
            break
        offset += len(rows)

    return all_markets


async def fetch_book(
    client: httpx.AsyncClient,
    token_id: str,
    semaphore: asyncio.Semaphore,
) -> dict[str, Optional[float]]:
    now_ts = time.time()
    suppress_until = _BOOK_404_SUPPRESS_UNTIL.get(token_id)
    if suppress_until is not None:
        if now_ts < suppress_until:
            return {"bid": None, "ask": None, "bid_size": None, "ask_size": None}
        _BOOK_404_SUPPRESS_UNTIL.pop(token_id, None)

    async with semaphore:
        try:
            response = await client.get(f"{CLOB_API}/book", params={"token_id": token_id})
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            if status_code == 404:
                previous = _BOOK_404_SUPPRESS_UNTIL.get(token_id, 0.0)
                suppress_until_next = time.time() + DEFAULT_BOOK_404_COOLDOWN_SECONDS
                _BOOK_404_SUPPRESS_UNTIL[token_id] = max(previous, suppress_until_next)
                if previous <= now_ts:
                    logger.info(
                        "book_fetch_404_suppressed",
                        token_id=token_id,
                        suppress_seconds=DEFAULT_BOOK_404_COOLDOWN_SECONDS,
                    )
            else:
                logger.debug(
                    "book_fetch_error",
                    token_id=token_id,
                    error=repr(exc),
                    error_type=type(exc).__name__,
                )
            return {"bid": None, "ask": None, "bid_size": None, "ask_size": None}
        except Exception as exc:
            logger.debug(
                "book_fetch_error",
                token_id=token_id,
                error=repr(exc),
                error_type=type(exc).__name__,
            )
            return {"bid": None, "ask": None, "bid_size": None, "ask_size": None}

    _BOOK_404_SUPPRESS_UNTIL.pop(token_id, None)

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
