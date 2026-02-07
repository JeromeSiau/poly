"""Runner for contrarian NO bet strategy (NeverYES / DidiTrading).

Fetches active Polymarket markets via Gamma API (free, no key),
scores them for optimism bias, and displays top NO bet candidates.

Usage:
    python scripts/run_no_bet.py [--limit 200] [--top 20]
"""

import argparse
import asyncio
import json
from datetime import datetime, timezone

import httpx
import structlog

from config.settings import settings
from src.arb.no_bet_scanner import NoBetScanner

logger = structlog.get_logger()

GAMMA_API = "https://gamma-api.polymarket.com/markets"


async def fetch_markets(limit: int = 200) -> list[dict]:
    """Fetch active markets from Polymarket Gamma API."""
    all_markets = []
    offset = 0
    batch = 100

    async with httpx.AsyncClient(timeout=30.0) as client:
        while len(all_markets) < limit:
            resp = await client.get(
                GAMMA_API,
                params={
                    "limit": min(batch, limit - len(all_markets)),
                    "offset": offset,
                    "active": "true",
                    "closed": "false",
                },
            )
            resp.raise_for_status()
            markets = resp.json()
            if not markets:
                break
            all_markets.extend(markets)
            offset += len(markets)

    logger.info("markets_fetched", count=len(all_markets))
    return all_markets


def normalize_market(raw: dict) -> dict:
    """Convert Gamma API market to scanner format."""
    prices = json.loads(raw.get("outcomePrices", "[0.5, 0.5]"))
    outcomes = json.loads(raw.get("outcomes", '["Yes", "No"]'))

    tokens = []
    clob_ids = json.loads(raw.get("clobTokenIds", "[]"))
    for i, outcome in enumerate(outcomes):
        tokens.append({
            "outcome": outcome,
            "price": float(prices[i]) if i < len(prices) else 0.5,
            "token_id": clob_ids[i] if i < len(clob_ids) else "",
        })

    return {
        "condition_id": raw.get("conditionId", ""),
        "title": raw.get("question", ""),
        "tokens": tokens,
        "volume_24h": raw.get("volume24hr", 0) or 0,
        "liquidity": raw.get("liquidityNum", 0) or 0,
        "end_date": raw.get("endDate", ""),
        "slug": raw.get("slug", ""),
    }


def days_until(end_date_str: str) -> int:
    """Calculate days until resolution."""
    try:
        end = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
        delta = end - datetime.now(timezone.utc)
        return max(0, delta.days)
    except (ValueError, TypeError):
        return 365


async def main(limit: int, top_n: int) -> None:
    scanner = NoBetScanner(
        min_yes_price=settings.NO_BET_MIN_YES_PRICE,
        max_yes_price=settings.NO_BET_MAX_YES_PRICE,
        min_liquidity=settings.NO_BET_MIN_LIQUIDITY,
        min_volume_24h=settings.NO_BET_MIN_VOLUME_24H,
    )

    print(f"\nFetching up to {limit} active markets from Polymarket...\n")
    raw_markets = await fetch_markets(limit)
    markets = [normalize_market(m) for m in raw_markets]

    # Filter to sweet spot
    candidates = scanner.filter_candidates(markets)
    print(f"Found {len(candidates)} markets with YES in [{scanner.min_yes_price:.0%}, {scanner.max_yes_price:.0%}]\n")

    # Score and rank
    scored = []
    for m in candidates:
        yes_price = next(
            (t["price"] for t in m["tokens"] if t["outcome"] == "Yes"), 0.5
        )
        no_price = next(
            (t["price"] for t in m["tokens"] if t["outcome"] == "No"), 0.5
        )
        end_days = days_until(m["end_date"])

        score = scanner.score_market(
            title=m["title"],
            yes_price=yes_price,
            volume_24h=m["volume_24h"],
            end_date_days=end_days,
        )
        scored.append((score, m, yes_price, no_price, end_days))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Display top N
    print(f"{'#':>3}  {'Score':>5}  {'YES':>5}  {'NO':>5}  {'Days':>5}  {'Vol24h':>10}  Title")
    print("-" * 100)

    for i, (score, m, yes_price, no_price, end_days) in enumerate(scored[:top_n]):
        title = m["title"][:55]
        vol = m["volume_24h"]
        print(
            f"{i+1:>3}  {score:>5.2f}  {yes_price:>5.2f}  {no_price:>5.2f}  "
            f"{end_days:>5d}  {vol:>10,.0f}  {title}"
        )

    if not scored:
        print("No candidates found in the optimism-bias sweet spot.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Contrarian NO Bet Scanner")
    parser.add_argument("--limit", type=int, default=500, help="Max markets to fetch")
    parser.add_argument("--top", type=int, default=20, help="Top N results to show")
    args = parser.parse_args()

    asyncio.run(main(args.limit, args.top))
