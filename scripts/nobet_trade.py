"""Interactive NO bet trader — scan, pick, and execute.

Scans Polymarket for optimism-biased markets, shows candidates one by one,
lets you confirm before placing each NO buy order via the CLOB API.

Usage:
    PYTHONPATH=. python scripts/nobet_trade.py [--limit 500] [--top 15] [--size 10]
"""

import argparse
import asyncio
import json
import math
from datetime import datetime, timezone

import httpx
import structlog
from dotenv import load_dotenv

load_dotenv()

from config.settings import settings
from src.arb.no_bet_scanner import NoBetScanner

logger = structlog.get_logger()

GAMMA_API = "https://gamma-api.polymarket.com/markets"


# ── Polymarket CLOB client ──────────────────────────────────────────

def make_clob_client():
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import ApiCreds

    return ClobClient(
        host="https://clob.polymarket.com",
        key=settings.POLYMARKET_PRIVATE_KEY,
        chain_id=settings.POLYMARKET_CHAIN_ID,
        creds=ApiCreds(
            api_key=settings.POLYMARKET_API_KEY,
            api_secret=settings.POLYMARKET_API_SECRET,
            api_passphrase=settings.POLYMARKET_API_PASSPHRASE,
        ),
    )


# ── Market fetching ─────────────────────────────────────────────────

async def fetch_markets(limit: int = 500) -> list[dict]:
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

    return all_markets


def normalize_market(raw: dict) -> dict:
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

    # Extract event slug for correct URL
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
        "event_slug": event_slug,
        "description": raw.get("description", ""),
    }


def days_until(end_date_str: str) -> int:
    try:
        end = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
        delta = end - datetime.now(timezone.utc)
        return max(0, delta.days)
    except (ValueError, TypeError):
        return 365


# ── Interactive trading ─────────────────────────────────────────────

def get_real_no_price(clob, no_token_id: str) -> dict:
    """Get real bid/ask from CLOB for the NO token."""
    try:
        buy_price = clob.get_price(no_token_id, "BUY")
        sell_price = clob.get_price(no_token_id, "SELL")
        spread = clob.get_spread(no_token_id)
        neg_risk = clob.get_neg_risk(no_token_id)
        tick_size = clob.get_tick_size(no_token_id)
        return {
            "buy": float(buy_price.get("price", 0)),
            "sell": float(sell_price.get("price", 0)),
            "spread": float(spread.get("spread", 0)),
            "neg_risk": neg_risk,
            "tick_size": tick_size,
        }
    except Exception as e:
        return {"buy": 0, "sell": 0, "spread": 0, "neg_risk": False, "tick_size": "0.01", "error": str(e)}


def place_no_order(clob, token_id: str, price: float, size_usd: float, neg_risk: bool, tick_size: str):
    """Place a BUY order for NO shares."""
    from py_clob_client.clob_types import OrderArgs, PartialCreateOrderOptions, OrderType

    shares = math.floor((size_usd / price) * 100) / 100  # round down to 2 decimals

    order_args = OrderArgs(
        token_id=token_id,
        price=price,
        size=shares,
        side="BUY",
        fee_rate_bps=int(settings.POLYMARKET_FEE_BPS),
    )

    options = PartialCreateOrderOptions(
        tick_size=tick_size,
        neg_risk=neg_risk,
    )

    response = clob.create_and_post_order(order_args, options)
    return response


async def main(limit: int, top_n: int, default_size: float) -> None:
    scanner = NoBetScanner(
        min_yes_price=settings.NO_BET_MIN_YES_PRICE,
        max_yes_price=settings.NO_BET_MAX_YES_PRICE,
        min_liquidity=settings.NO_BET_MIN_LIQUIDITY,
        min_volume_24h=settings.NO_BET_MIN_VOLUME_24H,
    )

    print(f"\n{'='*70}")
    print("  NO BET TRADER — NeverYES Strategy")
    print(f"{'='*70}")
    print(f"\nFetching up to {limit} active markets...\n")

    raw_markets = await fetch_markets(limit)
    markets = [normalize_market(m) for m in raw_markets]
    candidates = scanner.filter_candidates(markets)
    print(f"Found {len(candidates)} markets with YES in [{scanner.min_yes_price:.0%}, {scanner.max_yes_price:.0%}]\n")

    # Score and rank
    scored = []
    for m in candidates:
        yes_price = next((t["price"] for t in m["tokens"] if t["outcome"] == "Yes"), 0.5)
        no_price = next((t["price"] for t in m["tokens"] if t["outcome"] == "No"), 0.5)
        no_token_id = next((t["token_id"] for t in m["tokens"] if t["outcome"] == "No"), "")
        end_days = days_until(m["end_date"])

        score = scanner.score_market(
            title=m["title"],
            yes_price=yes_price,
            volume_24h=m["volume_24h"],
            end_date_days=end_days,
        )
        scored.append({
            "score": score,
            "market": m,
            "yes_price": yes_price,
            "no_price": no_price,
            "no_token_id": no_token_id,
            "end_days": end_days,
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    scored = scored[:top_n]

    if not scored:
        print("No candidates found.")
        return

    # Connect CLOB
    print("Connecting to Polymarket CLOB...\n")
    clob = make_clob_client()

    orders_placed = []

    for i, entry in enumerate(scored):
        m = entry["market"]
        title = m["title"]
        no_token = entry["no_token_id"]

        if not no_token:
            continue

        # Get real CLOB price
        real = get_real_no_price(clob, no_token)
        if real.get("error") or real["buy"] == 0:
            print(f"\n[{i+1}/{len(scored)}] {title[:60]}")
            print(f"  ⚠ Could not get CLOB price: {real.get('error', 'no price')}")
            print(f"  SKIP\n")
            continue

        no_buy_price = real["buy"]
        no_sell_price = real["sell"]
        spread = real["spread"]
        shares_for_default = math.floor((default_size / no_buy_price) * 100) / 100

        # Check for 50-50 trap in resolution rules
        desc = m.get("description", "")
        has_5050 = "50-50" in desc or "50/50" in desc
        url = f"https://polymarket.com/event/{m['event_slug']}"

        print(f"\n{'─'*70}")
        print(f"  [{i+1}/{len(scored)}] {title}")
        print(f"{'─'*70}")
        print(f"  Score:      {entry['score']:.2f}")
        print(f"  Gamma:      YES {entry['yes_price']:.2f}  /  NO {entry['no_price']:.2f}")
        print(f"  CLOB NO:    Buy {no_buy_price:.2f}  /  Sell {no_sell_price:.2f}  (spread {spread:.2f})")
        print(f"  Volume 24h: ${m['volume_24h']:,.0f}   Liquidity: ${m['liquidity']:,.0f}")
        print(f"  Resolves:   {entry['end_days']} days")
        print(f"  URL:        {url}")
        if has_5050:
            print(f"  *** WARNING: Resolution rules contain 50-50 clause! ***")
        print(f"")
        # Show resolution rules (truncated)
        if desc:
            print(f"  Resolution rules:")
            for line in desc.strip().split("\n")[:6]:
                line = line.strip()
                if line:
                    print(f"    {line[:90]}")
            if desc.count("\n") > 6:
                print(f"    (...)")
        print(f"")
        print(f"  ${default_size:.0f} → {shares_for_default:.1f} NO shares @ ${no_buy_price:.2f}")
        print(f"  If NO wins: +${shares_for_default * (1 - no_buy_price):.2f} profit")
        print(f"  If YES wins: -${default_size:.2f} loss")
        if has_5050:
            loss_5050 = shares_for_default * (no_buy_price - 0.50)
            print(f"  If 50-50:   -${loss_5050:.2f} loss (spread trap!)")

        choice = input(f"\n  [y] Buy ${default_size:.0f} / [amount] Custom $ / [s] Skip / [q] Quit → ").strip().lower()

        if choice == "q":
            print("\nDone.")
            break
        elif choice == "s" or choice == "":
            print("  → Skipped")
            continue
        elif choice == "y":
            size_usd = default_size
        else:
            try:
                size_usd = float(choice)
            except ValueError:
                print("  → Skipped (invalid input)")
                continue

        # Confirm
        shares = math.floor((size_usd / no_buy_price) * 100) / 100
        print(f"\n  CONFIRM: BUY {shares:.1f} NO shares @ ${no_buy_price:.2f} = ${size_usd:.2f}")
        confirm = input("  Type 'yes' to confirm → ").strip().lower()

        if confirm != "yes":
            print("  → Cancelled")
            continue

        # Place order
        print("  Placing order...")
        try:
            result = place_no_order(
                clob,
                token_id=no_token,
                price=no_buy_price,
                size_usd=size_usd,
                neg_risk=real["neg_risk"],
                tick_size=real["tick_size"],
            )
            print(f"  → Order result: {result}")
            orders_placed.append({
                "title": title,
                "size_usd": size_usd,
                "price": no_buy_price,
                "shares": shares,
                "result": result,
            })
        except Exception as e:
            print(f"  → ORDER FAILED: {e}")

    # Summary
    if orders_placed:
        print(f"\n{'='*70}")
        print(f"  SUMMARY — {len(orders_placed)} orders placed")
        print(f"{'='*70}")
        total = 0
        for o in orders_placed:
            status = o["result"].get("status", "?") if isinstance(o["result"], dict) else "?"
            print(f"  ${o['size_usd']:.0f} → {o['shares']:.1f} NO @ {o['price']:.2f}  [{status}]  {o['title'][:50]}")
            total += o["size_usd"]
        print(f"\n  Total deployed: ${total:.2f}")
    else:
        print("\nNo orders placed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive NO Bet Trader")
    parser.add_argument("--limit", type=int, default=500, help="Max markets to fetch")
    parser.add_argument("--top", type=int, default=15, help="Top N candidates to review")
    parser.add_argument("--size", type=float, default=10.0, help="Default order size in $")
    args = parser.parse_args()

    asyncio.run(main(args.limit, args.top, args.size))
