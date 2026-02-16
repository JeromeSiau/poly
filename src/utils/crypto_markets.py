"""Crypto market discovery utilities for Polymarket binary markets (5m / 15m).

Extracted from ``scripts/run_two_sided_inventory.py`` so that multiple
crypto strategy scripts can share market-discovery logic without importing
the entire two-sided runner.
"""

from __future__ import annotations

import time
from typing import Any

import httpx
import structlog

from src.utils.parsing import parse_json_list

logger = structlog.get_logger()

GAMMA_EVENTS_API = "https://gamma-api.polymarket.com/events"

CRYPTO_SYMBOL_TO_SLUG: dict[str, str] = {
    "BTCUSDT": "btc",
    "ETHUSDT": "eth",
    "SOLUSDT": "sol",
    "XRPUSDT": "xrp",
}

# Chainlink symbol ↔ Polymarket slug prefix.
CHAINLINK_TO_SLUG: dict[str, str] = {
    "btc/usd": "btc",
    "eth/usd": "eth",
    "sol/usd": "sol",
    "xrp/usd": "xrp",
}
SLUG_TO_CHAINLINK: dict[str, str] = {v: k for k, v in CHAINLINK_TO_SLUG.items()}

DURATION_TO_SLUG_SUFFIX: dict[int, str] = {300: "5m", 900: "15m"}


async def fetch_crypto_markets(
    client: httpx.AsyncClient,
    symbols: list[str],
    slots_ahead: int = 2,
    slot_duration_sec: int = 900,
) -> list[dict[str, Any]]:
    """Discover active crypto binary markets via Gamma /events API.

    For each symbol, constructs slugs for current and upcoming slots
    and fetches the corresponding event/market data.  Returns a list of market
    dicts in the same shape as :func:`fetch_markets` for seamless integration.

    Args:
        slot_duration_sec: Slot duration in seconds (300 for 5m, 900 for 15m).
    """
    slug_suffix = DURATION_TO_SLUG_SUFFIX.get(slot_duration_sec, f"{slot_duration_sec // 60}m")
    now = time.time()
    current_slot = int(now // slot_duration_sec) * slot_duration_sec
    all_markets: list[dict[str, Any]] = []
    seen_conditions: set[str] = set()

    for symbol in symbols:
        slug_prefix = (
            CRYPTO_SYMBOL_TO_SLUG.get(symbol.upper())
            or CHAINLINK_TO_SLUG.get(symbol.lower())
        )
        if not slug_prefix:
            continue

        for offset in range(slots_ahead):
            slot_ts = current_slot + offset * slot_duration_sec
            slug = f"{slug_prefix}-updown-{slug_suffix}-{slot_ts}"

            try:
                resp = await client.get(
                    GAMMA_EVENTS_API,
                    params={"slug": slug},
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                if resp.status_code != 200:
                    continue
                events = resp.json()
                if not events:
                    continue
            except Exception as exc:
                logger.warning("crypto_event_fetch_error", slug=slug, error=str(exc))
                continue

            for event in events:
                for mkt in event.get("markets", []):
                    cid = mkt.get("conditionId", "")
                    if not cid or cid in seen_conditions:
                        continue

                    outcomes = parse_json_list(mkt.get("outcomes", []))
                    clob_ids = parse_json_list(mkt.get("clobTokenIds", []))
                    if len(outcomes) != 2 or len(clob_ids) < 2:
                        continue

                    # Inject event slug so _first_event_slug can find it
                    if "events" not in mkt:
                        mkt["events"] = [{"slug": slug}]

                    seen_conditions.add(cid)
                    all_markets.append(mkt)

    logger.info("crypto_markets_discovered", count=len(all_markets), symbols=symbols,
                duration=slug_suffix)
    return all_markets
