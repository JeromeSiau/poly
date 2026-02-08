"""Runner for contrarian NO bet strategy (NeverYES / DidiTrading).

Two modes:
  scan   — One-shot scan of active markets, ranked by score
  watch  — Continuous polling, alerts on NEW FDV/launch markets

Usage:
    ./run scripts/run_no_bet.py scan  [--limit 500] [--top 20]
    ./run scripts/run_no_bet.py watch [--interval 300]
"""

import argparse
import asyncio
import json
from datetime import datetime, timezone

import httpx
import structlog
from dotenv import load_dotenv

load_dotenv()

from config.settings import settings
from src.arb.no_bet_scanner import NoBetScanner

logger = structlog.get_logger()

GAMMA_API = "https://gamma-api.polymarket.com/markets"

# FDV/launch keywords — the core NeverYES edge
FDV_KEYWORDS = [
    "fdv", "fully diluted", "market cap", "one day after launch",
    "tge", "launch above", "launch price",
]


async def fetch_markets(limit: int = 500, active: bool = True, closed: bool = False) -> list[dict]:
    """Fetch markets from Polymarket Gamma API."""
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
                    "active": str(active).lower(),
                    "closed": str(closed).lower(),
                },
            )
            resp.raise_for_status()
            markets = resp.json()
            if not markets:
                break
            all_markets.extend(markets)
            offset += len(markets)

    return all_markets


def normalize_market(raw: dict) -> dict:
    """Convert Gamma API market to scanner format."""
    prices = json.loads(raw.get("outcomePrices", "[0.5, 0.5]"))
    outcomes = json.loads(raw.get("outcomes", '["Yes", "No"]'))
    clob_ids = json.loads(raw.get("clobTokenIds", "[]"))

    tokens = []
    for i, outcome in enumerate(outcomes):
        tokens.append({
            "outcome": outcome,
            "price": float(prices[i]) if i < len(prices) else 0.5,
            "token_id": clob_ids[i] if i < len(clob_ids) else "",
        })

    events = raw.get("events", [])
    if isinstance(events, str):
        events = json.loads(events)
    event_slug = events[0].get("slug", raw.get("slug", "")) if events else raw.get("slug", "")

    return {
        "condition_id": raw.get("conditionId", ""),
        "title": raw.get("question", ""),
        "tokens": tokens,
        "volume_24h": raw.get("volume24hr", 0) or 0,
        "liquidity": raw.get("liquidityNum", 0) or 0,
        "end_date": raw.get("endDate", ""),
        "created_at": raw.get("createdAt", ""),
        "event_slug": event_slug,
        "description": raw.get("description", ""),
    }


def days_until(date_str: str) -> int:
    try:
        d = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        delta = d - datetime.now(timezone.utc)
        return max(0, delta.days)
    except (ValueError, TypeError):
        return 365


def days_since(date_str: str) -> int:
    try:
        d = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        delta = datetime.now(timezone.utc) - d
        return max(0, delta.days)
    except (ValueError, TypeError):
        return 999


def is_fdv_market(title: str) -> bool:
    t = title.lower()
    return any(kw in t for kw in FDV_KEYWORDS)


def market_flags(m: dict) -> str:
    """Return warning flags for a market."""
    flags = []
    desc = m.get("description", "")
    if "50-50" in desc or "50/50" in desc:
        flags.append("5050")
    if is_fdv_market(m["title"]):
        flags.append("FDV")
    age = days_since(m.get("created_at", ""))
    if age <= 7:
        flags.append("NEW!")
    return " ".join(flags)


# ── SCAN mode ────────────────────────────────────────────────────────

async def cmd_scan(limit: int, top_n: int) -> None:
    scanner = NoBetScanner(
        min_yes_price=settings.NO_BET_MIN_YES_PRICE,
        max_yes_price=settings.NO_BET_MAX_YES_PRICE,
        min_liquidity=settings.NO_BET_MIN_LIQUIDITY,
        min_volume_24h=settings.NO_BET_MIN_VOLUME_24H,
    )

    print(f"\nFetching up to {limit} active markets...\n")
    raw_markets = await fetch_markets(limit)
    markets = [normalize_market(m) for m in raw_markets]

    candidates = scanner.filter_candidates(markets)
    print(f"Found {len(candidates)} markets with YES in [{scanner.min_yes_price:.0%}, {scanner.max_yes_price:.0%}]\n")

    scored = []
    for m in candidates:
        yes_price = next((t["price"] for t in m["tokens"] if t["outcome"] == "Yes"), 0.5)
        no_price = next((t["price"] for t in m["tokens"] if t["outcome"] == "No"), 0.5)
        end_days = days_until(m["end_date"])
        age_days = days_since(m.get("created_at", ""))

        score = scanner.score_market(
            title=m["title"],
            yes_price=yes_price,
            volume_24h=m["volume_24h"],
            end_date_days=end_days,
        )
        # Boost fresh markets
        if age_days <= 3:
            score += 0.20
        elif age_days <= 7:
            score += 0.10

        scored.append((score, m, yes_price, no_price, end_days, age_days))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Table
    print(f"{'#':>3}  {'Score':>5}  {'YES':>5}  {'NO':>5}  {'Days':>5}  {'Age':>4}  {'Vol24h':>10}  {'Flags':>10}  Title")
    print("-" * 120)

    for i, (score, m, yes_price, no_price, end_days, age_days) in enumerate(scored[:top_n]):
        title = m["title"][:45]
        vol = m["volume_24h"]
        flags = market_flags(m)
        print(
            f"{i+1:>3}  {score:>5.2f}  {yes_price:>5.2f}  {no_price:>5.2f}  "
            f"{end_days:>5d}  {age_days:>3}d  {vol:>10,.0f}  {flags:>10}  {title}"
        )

    # Details
    print(f"\n{'='*80}")
    for i, (score, m, yes_price, no_price, end_days, age_days) in enumerate(scored[:top_n]):
        desc = m.get("description", "")
        flags = market_flags(m)
        url = f"https://polymarket.com/event/{m.get('event_slug', '')}"

        print(f"\n  #{i+1} {m['title']}")
        print(f"  URL: {url}")
        print(f"  Flags: {flags if flags else 'none'}")
        if desc:
            lines = [l.strip() for l in desc.strip().split("\n") if l.strip()]
            for line in lines[:4]:
                print(f"    {line[:95]}")
            if len(lines) > 4:
                print(f"    (...)")
        print()

    if not scored:
        print("No candidates found.")


# ── WATCH mode ───────────────────────────────────────────────────────

async def cmd_watch(interval: int) -> None:
    """Poll for new FDV/launch markets and alert when found."""
    seen_ids: set[str] = set()

    print(f"\n{'='*60}")
    print(f"  FDV WATCHDOG — Polling every {interval}s")
    print(f"  Looking for new FDV/launch markets with YES > 0.30")
    print(f"{'='*60}\n")

    # Initial load
    raw_markets = await fetch_markets(500)
    for m in raw_markets:
        seen_ids.add(m.get("conditionId", ""))
    print(f"Loaded {len(seen_ids)} existing markets. Watching for new ones...\n")

    cycle = 0
    while True:
        cycle += 1
        await asyncio.sleep(interval)
        now = datetime.now(timezone.utc).strftime("%H:%M:%S")

        try:
            raw_markets = await fetch_markets(500)
        except Exception as e:
            print(f"[{now}] Fetch error: {e}")
            continue

        new_fdv = []
        for raw in raw_markets:
            cid = raw.get("conditionId", "")
            if cid in seen_ids:
                continue
            seen_ids.add(cid)

            m = normalize_market(raw)
            if not is_fdv_market(m["title"]):
                continue

            yes_price = next((t["price"] for t in m["tokens"] if t["outcome"] == "Yes"), 0.5)
            if yes_price < 0.30:
                continue  # NO already expensive, too late

            new_fdv.append((m, yes_price))

        if new_fdv:
            print(f"\n{'!'*60}")
            print(f"  [{now}] {len(new_fdv)} NEW FDV MARKET(S) DETECTED!")
            print(f"{'!'*60}")
            for m, yp in new_fdv:
                no_price = 1 - yp
                url = f"https://polymarket.com/event/{m.get('event_slug', '')}"
                print(f"\n  {m['title']}")
                print(f"  YES: {yp:.2f}  NO: {no_price:.2f}  Liq: ${m['liquidity']:,.0f}")
                print(f"  URL: {url}")
                desc = m.get("description", "")
                if desc:
                    lines = [l.strip() for l in desc.strip().split("\n") if l.strip()]
                    for line in lines[:3]:
                        print(f"    {line[:90]}")
            print()
        else:
            total_active = len(raw_markets)
            print(f"[{now}] Cycle {cycle} — {total_active} markets checked, no new FDV markets")


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NO Bet Scanner (NeverYES Strategy)")
    sub = parser.add_subparsers(dest="command", help="Mode")

    scan_p = sub.add_parser("scan", help="One-shot scan of active markets")
    scan_p.add_argument("--limit", type=int, default=500, help="Max markets to fetch")
    scan_p.add_argument("--top", type=int, default=20, help="Top N results to show")

    watch_p = sub.add_parser("watch", help="Watch for new FDV/launch markets")
    watch_p.add_argument("--interval", type=int, default=300, help="Poll interval in seconds")

    args = parser.parse_args()

    if args.command == "watch":
        asyncio.run(cmd_watch(args.interval))
    else:
        asyncio.run(cmd_scan(args.limit if hasattr(args, "limit") else 500,
                             args.top if hasattr(args, "top") else 20))
