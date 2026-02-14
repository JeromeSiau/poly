"""Polymarket Market Scanner for Last-Penny Sniper.

Polls the Gamma REST API to discover active binary markets with one outcome
trading near certainty (ask >= threshold). Returns targets for the SniperEngine.

The scanner is the "eyes" — it finds markets. The engine does the trading.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
import structlog

from src.utils.parsing import parse_json_list

logger = structlog.get_logger()

GAMMA_API = "https://gamma-api.polymarket.com/markets"
CLOB_API = "https://clob.polymarket.com"


@dataclass(slots=True)
class SniperTarget:
    """A market outcome that's near certainty — potential sniper entry."""

    condition_id: str
    token_id: str
    outcome: str
    question: str
    slug: str
    ask_price: float
    ask_size: float
    end_date: str
    category: str
    has_fee: bool
    fee_pct: float


def classify_slug(slug: str, question: str) -> tuple[str, bool, float]:
    """Classify market category and fee structure.

    Returns (category, has_fee, fee_rate).
    Fee formula: fee_per_share = price * (1 - price) * rate
    """
    slug_lower = slug.lower() if slug else ""
    q_lower = question.lower() if question else ""

    if "updown-15m" in slug_lower:
        return "crypto_15min", True, 0.0625
    if "updown" in slug_lower:
        return "crypto_hourly", False, 0.0
    if "price-of" in slug_lower:
        return "crypto_bracket", False, 0.0

    # Sports with fees: NCAAB, Serie A
    if "ncaab" in q_lower or "ncaa" in q_lower:
        return "sports", True, 0.0175
    if "serie a" in q_lower or "serie-a" in slug_lower:
        return "sports", True, 0.0175

    # Sports without fees
    sport_hints = [
        "nba", "nfl", "nhl", "mlb", "premier league", "tennis",
        "champions league", " vs ", "football", "basketball",
        "la liga", "bundesliga",
    ]
    if any(h in q_lower for h in sport_hints):
        return "sports", False, 0.0

    return "other", False, 0.0


class MarketScanner:
    """Polls Gamma API for active markets with high-priced outcomes.

    Workflow:
    1. Fetch all active binary markets (paginated)
    2. Pre-filter by outcomePrices from listing (soft threshold)
    3. For candidates, fetch CLOB orderbook for precise ask prices
    4. Return targets where ask >= min_price
    """

    def __init__(
        self,
        *,
        min_price: float = 0.99,
        scan_interval: float = 15.0,
        max_markets: int = 2000,
        book_concurrency: int = 20,
        fee_ok_above: float = 0.99,
        max_end_hours: float = 1.0,
    ) -> None:
        self.min_price = min_price
        self.scan_interval = scan_interval
        self.max_markets = max_markets
        self.book_concurrency = book_concurrency
        self.fee_ok_above = fee_ok_above
        self.max_end_hours = max_end_hours

        self._targets: list[SniperTarget] = []
        self._last_scan_ts: float = 0.0
        self._seen_conditions: set[str] = set()

    @property
    def targets(self) -> list[SniperTarget]:
        return list(self._targets)

    def mark_traded(self, condition_id: str) -> None:
        """Mark a market as already traded (skip in future scans)."""
        self._seen_conditions.add(condition_id)

    async def scan(self, client: httpx.AsyncClient) -> list[SniperTarget]:
        """Run one full scan cycle. Returns actionable targets."""
        t0 = time.time()

        candidates = await self._fetch_candidates(client)

        if not candidates:
            self._targets = []
            return []

        targets = await self._fetch_books(client, candidates)

        self._targets = targets
        self._last_scan_ts = time.time()

        elapsed = time.time() - t0
        logger.info(
            "scanner_complete",
            candidates=len(candidates),
            targets=len(targets),
            elapsed_s=round(elapsed, 1),
        )

        return targets

    async def _fetch_candidates(
        self, client: httpx.AsyncClient
    ) -> list[dict[str, Any]]:
        """Fetch active binary markets with high outcome prices."""
        candidates: list[dict[str, Any]] = []
        offset = 0
        batch_size = 100
        soft_threshold = self.min_price - 0.05

        while len(candidates) < self.max_markets:
            try:
                resp = await client.get(
                    GAMMA_API,
                    params={
                        "limit": batch_size,
                        "offset": offset,
                        "active": "true",
                        "closed": "false",
                    },
                    timeout=15.0,
                )
                resp.raise_for_status()
                rows = resp.json()
            except Exception as exc:
                logger.warning("scanner_fetch_error", offset=offset, error=str(exc))
                break

            if not isinstance(rows, list) or not rows:
                break

            for raw in rows:
                cid = str(raw.get("conditionId", ""))
                if not cid or cid in self._seen_conditions:
                    continue

                outcomes = parse_json_list(raw.get("outcomes", []))
                tokens = parse_json_list(raw.get("clobTokenIds", []))
                prices_raw = parse_json_list(raw.get("outcomePrices", []))

                if len(outcomes) != 2 or len(tokens) < 2 or len(prices_raw) < 2:
                    continue

                try:
                    prices = [float(p) for p in prices_raw[:2]]
                except (ValueError, TypeError):
                    continue

                if max(prices) < soft_threshold:
                    continue

                question = str(raw.get("question", ""))

                # Extract slug from events or market-level slug
                slug = ""
                events = raw.get("events", [])
                if events and isinstance(events, list):
                    slug = str(events[0].get("slug", ""))
                if not slug:
                    slug = str(raw.get("slug", ""))

                cat, has_fee, fee_rate = classify_slug(slug, question)

                # Fee markets only accepted if price > fee_ok_above
                if has_fee and max(prices) < self.fee_ok_above:
                    continue

                # Filter by time to resolution — category-specific
                # Crypto: price is the signal, just exclude long-dated (24h)
                # Sports: tight filter to ensure in-play near end (1h)
                end_date_str = str(raw.get("endDate", ""))
                cat_max_hours = self._max_hours_for_category(cat)
                if not self._ends_within(end_date_str, cat_max_hours):
                    continue

                raw["_category"] = cat
                raw["_has_fee"] = has_fee
                raw["_fee_rate"] = fee_rate
                raw["_slug"] = slug
                candidates.append(raw)

            if len(rows) < batch_size:
                break
            offset += len(rows)

        return candidates

    async def _fetch_books(
        self,
        client: httpx.AsyncClient,
        candidates: list[dict[str, Any]],
    ) -> list[SniperTarget]:
        """Fetch CLOB orderbooks and filter by actual ask price."""
        semaphore = asyncio.Semaphore(self.book_concurrency)
        targets: list[SniperTarget] = []

        async def check_outcome(
            raw: dict[str, Any], idx: int,
        ) -> Optional[SniperTarget]:
            tokens = [str(t) for t in parse_json_list(raw.get("clobTokenIds", []))]
            outcomes = [str(o) for o in parse_json_list(raw.get("outcomes", []))]
            if idx >= len(tokens) or idx >= len(outcomes):
                return None

            token_id = tokens[idx]
            outcome = outcomes[idx]
            cid = str(raw.get("conditionId", ""))
            question = str(raw.get("question", ""))
            end_date = str(raw.get("endDate", ""))
            cat = raw["_category"]
            has_fee = raw["_has_fee"]
            fee_rate = raw["_fee_rate"]
            slug = raw["_slug"]

            async with semaphore:
                try:
                    resp = await client.get(
                        f"{CLOB_API}/book",
                        params={"token_id": token_id},
                        timeout=10.0,
                    )
                    resp.raise_for_status()
                    book = resp.json()
                except Exception:
                    return None

            asks = book.get("asks", [])
            if not asks:
                return None

            # Find best (lowest) ask
            best_ask: Optional[float] = None
            best_ask_size = 0.0
            for level in asks:
                try:
                    price = float(level.get("price", 0))
                    size = float(level.get("size", 0))
                except (ValueError, TypeError):
                    continue
                if 0 < price < 1 and size > 0:
                    if best_ask is None or price < best_ask:
                        best_ask = price
                        best_ask_size = size

            if best_ask is None or best_ask < self.min_price:
                return None

            fee_pct = best_ask * (1 - best_ask) * fee_rate if has_fee else 0.0

            return SniperTarget(
                condition_id=cid,
                token_id=token_id,
                outcome=outcome,
                question=question,
                slug=slug,
                ask_price=best_ask,
                ask_size=best_ask_size,
                end_date=end_date,
                category=cat,
                has_fee=has_fee,
                fee_pct=fee_pct,
            )

        # Check both outcomes for each candidate
        tasks = []
        for raw in candidates:
            tasks.append(check_outcome(raw, 0))
            tasks.append(check_outcome(raw, 1))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in results:
            if isinstance(r, SniperTarget):
                targets.append(r)

        # Sort by price descending (highest certainty first)
        targets.sort(key=lambda t: t.ask_price, reverse=True)

        return targets

    def _max_hours_for_category(self, category: str) -> float:
        """Category-specific time-to-resolution limits.

        Crypto: price at 0.99+ IS the timing signal. Just exclude long-dated.
        Sports: tight filter ensures we only see in-play games near the end.
        """
        if category in ("crypto_15min", "crypto_hourly"):
            return 1.0
        if category == "crypto_bracket":
            return 24.0
        if category == "sports":
            return self.max_end_hours  # default 1h — in-play near end
        # politics, weather, other
        return self.max_end_hours

    def _ends_within(self, end_date_str: str, max_hours: float) -> bool:
        """Return True if the market resolves within max_hours."""
        if not end_date_str:
            return False
        try:
            # Gamma returns ISO 8601: "2026-02-14T21:00:00Z"
            end_dt = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            hours_left = (end_dt - now).total_seconds() / 3600
            return 0 < hours_left <= max_hours
        except (ValueError, TypeError):
            return False
