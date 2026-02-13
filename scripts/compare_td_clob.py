#!/usr/bin/env python3
"""Compare crypto TD maker internal tracking vs Polymarket CLOB activity.

Fetches actual wallet activity from the Polymarket Data API (TRADE/REDEEM/MERGE)
and compares it against the prod trades API (localhost:8788).

Usage:
    ./run compare_td_clob.py                         # default: 24h
    ./run compare_td_clob.py --hours 72              # last 72 hours
    ./run compare_td_clob.py --wallet 0x...          # explicit wallet
    ./run compare_td_clob.py --api http://host:8788  # custom API base URL
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from config.settings import settings

# ---------------------------------------------------------------------------
# CLOB activity fetcher
# ---------------------------------------------------------------------------

DATA_API_ACTIVITY = "https://data-api.polymarket.com/activity"


def fetch_wallet_activity(
    wallet: str, hours: float, *, timeout: float = 30.0,
) -> list[dict[str, Any]]:
    """Fetch wallet activity from Polymarket Data API."""
    cutoff_ts = int(
        (datetime.now(timezone.utc) - timedelta(hours=hours)).timestamp()
    )
    rows: list[dict[str, Any]] = []

    with httpx.Client(timeout=timeout) as client:
        for page in range(10):
            offset = page * 500
            if offset > 3000:
                break
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
            for row in payload:
                ts = _parse_ts(row.get("timestamp"))
                if ts <= 0:
                    continue
                if ts < cutoff_ts:
                    stop = True
                    continue
                rows.append(_normalize_row(row, ts))

            if stop or len(payload) < 500:
                break

    # Deduplicate
    seen: set[tuple] = set()
    deduped: list[dict[str, Any]] = []
    for r in sorted(rows, key=lambda x: x["timestamp"]):
        key = (r["tx_hash"], r["condition_id"], r["type"], r["timestamp"],
               r["outcome"], r["usdc_size"], r["price"])
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


def _normalize_row(raw: dict, ts: int) -> dict[str, Any]:
    rtype = (str(raw.get("type") or "")).upper()
    if rtype not in ("TRADE", "REDEEM", "MERGE"):
        rtype = "UNKNOWN"
    side = (str(raw.get("side") or "")).upper()
    if rtype == "REDEEM":
        side = "REDEEM"
    elif rtype == "MERGE":
        side = "MERGE"
    price = _sfloat(raw.get("price"))
    usdc = _sfloat(raw.get("usdcSize") or raw.get("size"))
    shares = (usdc / price) if (rtype == "TRADE" and price > 0) else _sfloat(raw.get("shares"))
    return {
        "timestamp": ts,
        "dt": datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "type": rtype,
        "side": side,
        "condition_id": str(raw.get("conditionId") or ""),
        "title": str(raw.get("title") or ""),
        "slug": str(raw.get("slug") or ""),
        "event_slug": str(raw.get("eventSlug") or ""),
        "outcome": str(raw.get("outcome") or ""),
        "price": price,
        "usdc_size": usdc,
        "shares": shares,
        "tx_hash": str(raw.get("transactionHash") or ""),
    }


def _sfloat(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# Prod API reader
# ---------------------------------------------------------------------------

def fetch_api_trades(
    api_base: str, hours: float, *, timeout: float = 15.0,
) -> list[dict[str, Any]]:
    """Fetch trades from the prod trades API (localhost:8788)."""
    with httpx.Client(timeout=timeout) as client:
        resp = client.get(
            f"{api_base}/trades",
            params={
                "event_type": "crypto_td_maker",
                "hours": min(hours, 720),
                "limit": 2000,
            },
        )
        resp.raise_for_status()
        data = resp.json()
    return data.get("trades", []), data


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------

def compare(
    clob_rows: list[dict[str, Any]],
    api_trades: list[dict[str, Any]],
    api_summary: dict[str, Any],
) -> None:
    """Print comparison between CLOB and internal API records."""

    # -- CLOB: group by condition_id --
    clob_by_cid: dict[str, dict] = defaultdict(
        lambda: {"buys": [], "redeems": [], "merges": [], "slug": ""}
    )
    for r in clob_rows:
        slug = r.get("event_slug") or r.get("slug", "")
        if "updown-15m" not in slug:
            continue
        cid = r["condition_id"]
        clob_by_cid[cid]["slug"] = slug
        if r["type"] == "TRADE":
            clob_by_cid[cid]["buys"].append(r)
        elif r["type"] == "REDEEM":
            clob_by_cid[cid]["redeems"].append(r)
        elif r["type"] == "MERGE":
            clob_by_cid[cid]["merges"].append(r)

    # -- Build slug→condition_id mapping from CLOB data --
    slug_to_cid: dict[str, str] = {}
    for cid, data in clob_by_cid.items():
        slug = data.get("slug", "")
        if slug:
            slug_to_cid[slug] = cid

    # -- API: group by condition_id (matched via slug) --
    api_by_cid: dict[str, list[dict]] = defaultdict(list)
    for t in api_trades:
        title = t.get("title") or t.get("slug") or ""
        # Match API title (slug) to CLOB condition_id
        cid = slug_to_cid.get(title, title)
        if cid:
            api_by_cid[cid].append(t)

    # -- Header --
    all_cids = sorted(
        set(clob_by_cid) | set(api_by_cid),
        key=lambda c: min(
            (b["timestamp"] for b in clob_by_cid.get(c, {}).get("buys", [])),
            default=0,
        ),
    )

    print("=" * 100)
    print(f"{'COMPARISON: CLOB (reality) vs API (dashboard)':^100}")
    print("=" * 100)
    print()

    total_clob_usd = 0.0
    total_api_usd = 0.0
    total_clob_pnl = 0.0
    total_api_pnl = 0.0
    issues: list[str] = []

    for cid in all_cids:
        clob = clob_by_cid.get(cid, {"buys": [], "redeems": [], "merges": [], "slug": ""})
        api = api_by_cid.get(cid, [])
        slug = clob.get("slug", "") or (api[0].get("title", "?") if api else "?")

        # CLOB stats
        clob_buys = clob["buys"]
        clob_fills = len(clob_buys)
        clob_usd = sum(b["usdc_size"] for b in clob_buys)
        clob_outcomes = set(b["outcome"] for b in clob_buys)
        clob_redeems = sum(r["usdc_size"] for r in clob["redeems"])
        clob_has_redeem = len(clob["redeems"]) > 0
        clob_has_merge = len(clob["merges"]) > 0

        if clob_has_redeem:
            clob_pnl = clob_redeems - clob_usd
            clob_status = f"WIN ${clob_pnl:+.2f}" if clob_pnl > 0 else f"LOSS ${clob_pnl:+.2f}"
        elif clob_has_merge:
            clob_pnl = -clob_usd
            clob_status = f"LOSS ${clob_pnl:+.2f}"
        else:
            clob_pnl = 0.0
            clob_status = "PENDING"

        # API stats
        api_fills = [t for t in api if t.get("side") == "BUY"]
        api_settles = [t for t in api if t.get("pnl") is not None]
        api_usd = sum(_sfloat(t.get("size")) for t in api_fills)
        api_pnl = sum(_sfloat(t.get("pnl")) for t in api_settles)
        api_wins = sum(1 for t in api_settles if _sfloat(t.get("pnl")) > 0)
        api_losses = sum(1 for t in api_settles if _sfloat(t.get("pnl")) <= 0)
        api_open = sum(1 for t in api if t.get("is_open") is True)

        total_clob_usd += clob_usd
        total_api_usd += api_usd
        if clob_status != "PENDING":
            total_clob_pnl += clob_pnl
        total_api_pnl += api_pnl

        # Detect issues
        mkt_issues: list[str] = []
        if clob_fills > 0 and not api:
            mkt_issues.append("NO_API_RECORD")
        if clob_fills > 0 and abs(clob_usd - api_usd) > 1.0:
            mkt_issues.append(f"SIZE_MISMATCH(clob=${clob_usd:.0f} api=${api_usd:.0f})")
        if clob_status != "PENDING" and api_settles:
            if (clob_pnl > 0) != (api_pnl > 0):
                mkt_issues.append(f"PNL_DIRECTION(clob={clob_pnl:+.2f} api={api_pnl:+.2f})")
            elif abs(clob_pnl - api_pnl) > 1.0:
                mkt_issues.append(f"PNL_AMOUNT(clob={clob_pnl:+.2f} api={api_pnl:+.2f})")
        if len(clob_outcomes) > 1:
            mkt_issues.append(f"BOTH_SIDES({','.join(clob_outcomes)})")

        # Print
        slug_short = slug[-40:]
        status_tag = " ** ISSUE **" if mkt_issues else ""
        print(f"--- {slug_short}{status_tag} ---")
        print(
            f"  CLOB:  {clob_fills:2d} fills, ${clob_usd:7.2f}, "
            f"outcomes={clob_outcomes or '?'}, {clob_status}"
        )
        api_status = (
            f"{api_wins}W/{api_losses}L pnl=${api_pnl:+.2f}"
            if api_settles else f"{api_open} open" if api_open else "none"
        )
        print(
            f"  API:   {len(api_fills):2d} fills, ${api_usd:7.2f}, {api_status}"
        )
        if mkt_issues:
            for iss in mkt_issues:
                print(f"  !! {iss}")
                issues.append(f"{slug_short}: {iss}")
        print()

    # -- Summary --
    print("=" * 100)
    print(f"{'SUMMARY':^100}")
    print("=" * 100)
    print(f"  Markets (CLOB):        {len(clob_by_cid)}")
    print(f"  Markets (API):         {len(api_by_cid)}")
    print(f"  Total invested (CLOB): ${total_clob_usd:.2f}")
    print(f"  Total invested (API):  ${total_api_usd:.2f}")
    delta_usd = total_clob_usd - total_api_usd
    if abs(delta_usd) > 1:
        print(f"  ** EXPOSURE GAP:       ${delta_usd:+.2f} untracked **")
    print()
    print(f"  Dashboard says:        {api_summary.get('wins', '?')}W / {api_summary.get('losses', '?')}L "
          f"({api_summary.get('winrate', '?')}% WR) "
          f"pnl=${api_summary.get('total_pnl', 0):+.2f}")
    print(f"  CLOB reality (resolved): ${total_clob_pnl:+.2f}")
    if abs(total_clob_pnl - _sfloat(api_summary.get("total_pnl"))) > 0.5:
        print(
            f"  ** PNL GAP:            "
            f"${total_clob_pnl - _sfloat(api_summary.get('total_pnl')):+.2f} **"
        )
    print()

    if issues:
        print(f"ISSUES FOUND: {len(issues)}")
        for i, iss in enumerate(issues, 1):
            print(f"  {i}. {iss}")
    else:
        print("No issues found — CLOB and API are in sync.")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare TD maker: Polymarket CLOB vs prod API"
    )
    parser.add_argument("--hours", type=float, default=24)
    parser.add_argument("--wallet", type=str, default="")
    parser.add_argument(
        "--api", type=str, default="http://localhost:8788",
        help="Prod trades API base URL (default: http://localhost:8788)",
    )
    args = parser.parse_args()

    wallet = args.wallet or settings.POLYMARKET_WALLET_ADDRESS
    if not wallet:
        print("ERROR: no wallet — set POLYMARKET_WALLET_ADDRESS or --wallet")
        sys.exit(1)

    api_base = args.api.rstrip("/")
    hours = args.hours

    print(f"Wallet:  {wallet}")
    print(f"API:     {api_base}")
    print(f"Window:  {hours}h")
    print()

    # 1. Check API health
    print("Checking API...")
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.get(f"{api_base}/health")
            resp.raise_for_status()
        print("  API is up")
    except Exception as exc:
        print(f"  API is DOWN: {exc}")
        print("  Check the autossh tunnel: ps aux | grep autossh")
        sys.exit(1)
    print()

    # 2. Fetch CLOB activity
    print("Fetching CLOB activity...")
    clob_rows = fetch_wallet_activity(wallet, hours)
    td_count = sum(1 for r in clob_rows
                   if "updown-15m" in (r.get("event_slug") or r.get("slug", "")))
    print(f"  {len(clob_rows)} total rows, {td_count} TD maker trades")
    print()

    # 3. Fetch API trades
    print("Fetching API trades...")
    api_trades, api_summary = fetch_api_trades(api_base, hours)
    print(
        f"  {len(api_trades)} trades, "
        f"{api_summary.get('wins', 0)}W/{api_summary.get('losses', 0)}L, "
        f"pnl=${api_summary.get('total_pnl', 0):+.2f}"
    )
    print()

    # 4. Compare
    compare(clob_rows, api_trades, api_summary)


if __name__ == "__main__":
    main()
