#!/usr/bin/env python3
"""Win rate report from Polymarket on-chain wallet activity.

Fetches actual TRADE/REDEEM/MERGE activity from the Polymarket Data API,
groups by market, and computes win rate on resolved positions.

Usage:
    ./run scripts/winrate.py                     # default: 24h
    ./run scripts/winrate.py --hours 17          # last 17h
    ./run scripts/winrate.py --wallet 0x...      # explicit wallet
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from config.settings import settings

DATA_API_ACTIVITY = "https://data-api.polymarket.com/activity"
CLOB_API_MARKET = "https://clob.polymarket.com/markets"


# ---------------------------------------------------------------------------
# Polymarket CLOB API — check market resolution
# ---------------------------------------------------------------------------

def check_resolution(condition_id: str, bought_outcome: str) -> str | None:
    """Query CLOB API to check if market resolved. Returns 'WIN', 'LOSS', or None."""
    try:
        resp = httpx.get(f"{CLOB_API_MARKET}/{condition_id}", timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json()
        if not data.get("closed"):
            return None
        tokens = data.get("tokens", [])
        for t in tokens:
            if t.get("outcome") == bought_outcome:
                return "WIN" if t.get("winner") else "LOSS"
        # If bought outcome not found in tokens, it's a loss
        return "LOSS" if tokens else None
    except Exception:
        return None


def resolve_open_markets(markets: list[dict[str, Any]]) -> None:
    """Resolve OPEN markets by checking CLOB API for winner status."""
    open_markets = [m for m in markets if m["status"] == "OPEN" and m["condition_id"]]
    if not open_markets:
        return

    print(f"Checking resolution for {len(open_markets)} unresolved markets...")
    for m in open_markets:
        if not m["outcome"]:
            continue
        result = check_resolution(m["condition_id"], m["outcome"])
        if result == "WIN":
            m["status"] = "WIN"
            m["pnl"] = m["cost"] / _avg_price_to_payout(m["avg_entry"]) - m["cost"]
        elif result == "LOSS":
            m["status"] = "LOSS"
            m["pnl"] = -m["cost"]


def _avg_price_to_payout(avg_price: float) -> float:
    """Convert avg entry price to payout ratio (e.g. 0.75 -> shares cost 0.75, pay 1.0)."""
    return avg_price if avg_price > 0 else 1.0


# ---------------------------------------------------------------------------
# Polymarket Data API fetcher
# ---------------------------------------------------------------------------

def fetch_activity(wallet: str, hours: float) -> list[dict[str, Any]]:
    cutoff_ts = int(
        (datetime.now(timezone.utc) - timedelta(hours=hours)).timestamp()
    )
    rows: list[dict[str, Any]] = []

    with httpx.Client(timeout=30) as client:
        for page in range(20):
            offset = page * 500
            try:
                resp = client.get(
                    DATA_API_ACTIVITY,
                    params={"user": wallet, "limit": 500, "offset": offset},
                )
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                if exc.response is not None and exc.response.status_code == 400:
                    break
                raise
            payload = resp.json()
            if not isinstance(payload, list) or not payload:
                break

            stop = False
            for raw in payload:
                ts = _parse_ts(raw.get("timestamp"))
                if ts <= 0:
                    continue
                if ts < cutoff_ts:
                    stop = True
                    continue
                rows.append(_normalize(raw, ts))

            if stop or len(payload) < 500:
                break

    # Deduplicate
    seen: set[tuple] = set()
    deduped: list[dict[str, Any]] = []
    for r in sorted(rows, key=lambda x: x["ts"]):
        key = (r["tx_hash"], r["condition_id"], r["type"], r["ts"],
               r["outcome"], r["usdc"], r["price"])
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    return deduped


def _parse_ts(v: Any) -> int:
    if isinstance(v, (int, float)):
        return int(v)
    if isinstance(v, str):
        try:
            return int(v)
        except ValueError:
            pass
        for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"):
            try:
                return int(datetime.strptime(v, fmt).replace(tzinfo=timezone.utc).timestamp())
            except ValueError:
                continue
    return 0


def _sfloat(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _normalize(raw: dict, ts: int) -> dict[str, Any]:
    rtype = (str(raw.get("type") or "")).upper()
    price = _sfloat(raw.get("price"))
    usdc = _sfloat(raw.get("usdcSize") or raw.get("size"))
    return {
        "ts": ts,
        "dt": datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "type": rtype,
        "side": (str(raw.get("side") or "")).upper(),
        "condition_id": str(raw.get("conditionId") or ""),
        "title": str(raw.get("title") or ""),
        "slug": str(raw.get("slug") or ""),
        "event_slug": str(raw.get("eventSlug") or ""),
        "outcome": str(raw.get("outcome") or ""),
        "price": price,
        "usdc": usdc,
        "tx_hash": str(raw.get("transactionHash") or ""),
    }


# ---------------------------------------------------------------------------
# Group by market and compute PnL
# ---------------------------------------------------------------------------

def analyse(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group activity by condition_id, compute per-market PnL."""
    by_cid: dict[str, dict] = defaultdict(lambda: {
        "buys": [], "sells": [], "redeems": [], "merges": [],
        "title": "", "slug": "", "outcome": "",
    })

    for r in rows:
        cid = r["condition_id"]
        if not by_cid[cid]["title"]:
            by_cid[cid]["title"] = r["title"]
            by_cid[cid]["slug"] = r.get("event_slug") or r.get("slug", "")
        if r["outcome"]:
            by_cid[cid]["outcome"] = r["outcome"]

        if r["type"] == "TRADE":
            if r["side"] == "BUY":
                by_cid[cid]["buys"].append(r)
            else:
                by_cid[cid]["sells"].append(r)
        elif r["type"] == "REDEEM":
            by_cid[cid]["redeems"].append(r)
        elif r["type"] == "MERGE":
            by_cid[cid]["merges"].append(r)

    markets: list[dict[str, Any]] = []
    for cid, data in by_cid.items():
        cost = sum(b["usdc"] for b in data["buys"])
        sell_proceeds = sum(s["usdc"] for s in data["sells"])
        redeem_proceeds = sum(r["usdc"] for r in data["redeems"])
        has_redeem = len(data["redeems"]) > 0
        has_merge = len(data["merges"]) > 0

        if has_redeem:
            pnl = redeem_proceeds - cost + sell_proceeds
            status = "WIN" if pnl > 0 else "LOSS"
        elif has_merge:
            # Merge = early exit, treat as resolved
            pnl = sell_proceeds - cost
            status = "WIN" if pnl > 0 else "LOSS"
        elif sell_proceeds > 0 and not data["buys"]:
            # Sell-only (short position closed)
            pnl = sell_proceeds
            status = "WIN" if pnl > 0 else "LOSS"
        else:
            pnl = 0.0
            status = "OPEN"

        first_ts = min(
            (b["ts"] for b in data["buys"]),
            default=min((r["ts"] for r in data["redeems"]), default=0),
        )

        avg_entry = (
            sum(b["price"] for b in data["buys"]) / len(data["buys"])
            if data["buys"] else 0
        )

        markets.append({
            "condition_id": cid,
            "title": data["title"],
            "slug": data["slug"],
            "outcome": data["outcome"],
            "status": status,
            "cost": cost,
            "redeem": redeem_proceeds,
            "sell_proceeds": sell_proceeds,
            "pnl": pnl,
            "n_fills": len(data["buys"]) + len(data["sells"]),
            "avg_entry": avg_entry,
            "first_ts": first_ts,
        })

    markets.sort(key=lambda m: m["first_ts"])
    return markets


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(markets: list[dict[str, Any]], hours: float) -> None:
    resolved = [m for m in markets if m["status"] in ("WIN", "LOSS")]
    still_open = [m for m in markets if m["status"] == "OPEN"]

    wins = [m for m in resolved if m["status"] == "WIN"]
    losses = [m for m in resolved if m["status"] == "LOSS"]

    total_pnl = sum(m["pnl"] for m in resolved)
    total_cost = sum(m["cost"] for m in markets)
    win_pnl = sum(m["pnl"] for m in wins)
    loss_pnl = sum(m["pnl"] for m in losses)

    print()
    print("=" * 70)
    print(f"  Polymarket Win Rate — last {hours}h")
    print("=" * 70)
    print()
    print(f"  Total markets:    {len(markets)}")
    print(f"  Resolved:         {len(resolved)}")
    print(f"  Still open:       {len(still_open)}")
    print()
    print(f"  Wins:             {len(wins)}")
    print(f"  Losses:           {len(losses)}")
    if resolved:
        print(f"  Win rate:         {len(wins)/len(resolved)*100:.1f}%")
    print()
    print(f"  Total PnL:        ${total_pnl:+.2f}")
    print(f"  Total invested:   ${total_cost:.2f}")
    if total_cost > 0:
        print(f"  ROI:              {total_pnl/total_cost*100:+.1f}%")
    if wins:
        print(f"  Avg win:          ${win_pnl/len(wins):+.2f}")
    if losses:
        print(f"  Avg loss:         ${loss_pnl/len(losses):+.2f}")
    if loss_pnl < 0:
        print(f"  Profit factor:    {abs(win_pnl/loss_pnl):.2f}")

    # Open positions
    if still_open:
        open_cost = sum(m["cost"] for m in still_open)
        print()
        print(f"  Open exposure:    ${open_cost:.2f} across {len(still_open)} markets")

    # Per-market detail for resolved
    if resolved:
        print()
        print("-" * 70)
        print("  Resolved markets:")
        print("-" * 70)
        for m in sorted(resolved, key=lambda x: x["first_ts"]):
            ts = datetime.fromtimestamp(m["first_ts"], timezone.utc).strftime("%m-%d %H:%M")
            tag = "W" if m["status"] == "WIN" else "L"
            title = (m["title"] or m["slug"] or "?")[:45]
            print(f"  [{tag}] ${m['pnl']:+6.2f}  @{m['avg_entry']:.2f}  "
                  f"${m['cost']:6.2f}  {ts}  {title}")

    # Top wins / losses
    if len(resolved) > 5:
        print()
        print("-" * 70)
        top_wins = sorted(resolved, key=lambda x: x["pnl"], reverse=True)[:5]
        print("  Top 5 wins:")
        for m in top_wins:
            if m["pnl"] <= 0:
                break
            title = (m["title"] or m["slug"] or "?")[:45]
            print(f"    ${m['pnl']:+.2f}  {title}")

        top_losses = sorted(resolved, key=lambda x: x["pnl"])[:5]
        print("  Top 5 losses:")
        for m in top_losses:
            if m["pnl"] >= 0:
                break
            title = (m["title"] or m["slug"] or "?")[:45]
            print(f"    ${m['pnl']:+.2f}  {title}")

    print()
    print("=" * 70)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Win rate from Polymarket wallet activity"
    )
    parser.add_argument("--hours", type=float, default=24.0,
                        help="Lookback window in hours (default: 24)")
    parser.add_argument("--wallet", type=str, default="",
                        help="Wallet address (default: from .env)")
    args = parser.parse_args()

    wallet = args.wallet or settings.POLYMARKET_WALLET_ADDRESS
    if not wallet:
        print("ERROR: no wallet — set POLYMARKET_WALLET_ADDRESS or use --wallet",
              file=sys.stderr)
        sys.exit(1)

    print(f"Wallet: {wallet}")
    print(f"Window: {args.hours}h")
    print("Fetching activity from Polymarket...")

    rows = fetch_activity(wallet, args.hours)
    if not rows:
        print(f"No activity found in the last {args.hours}h")
        sys.exit(0)

    print(f"Found {len(rows)} activity records")

    markets = analyse(rows)
    resolve_open_markets(markets)
    print_report(markets, args.hours)


if __name__ == "__main__":
    main()
