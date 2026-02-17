"""Polymarket on-chain win rate calculation.

Fetches wallet activity from the Polymarket Data API, groups by market,
and resolves open positions via the CLOB API to compute accurate win/loss.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

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
        return "LOSS" if tokens else None
    except Exception:
        return None


def resolve_open_markets(markets: list[dict[str, Any]]) -> None:
    """Resolve OPEN markets by checking CLOB API for winner status."""
    for m in markets:
        if m["status"] != "OPEN" or not m["condition_id"] or not m["outcome"]:
            continue
        result = check_resolution(m["condition_id"], m["outcome"])
        if result == "WIN":
            m["status"] = "WIN"
            avg = m["avg_entry"]
            m["pnl"] = m["cost"] / avg - m["cost"] if avg > 0 else 0
        elif result == "LOSS":
            m["status"] = "LOSS"
            m["pnl"] = -m["cost"]


# ---------------------------------------------------------------------------
# Polymarket Data API fetcher
# ---------------------------------------------------------------------------

def fetch_activity(
    wallet: str,
    hours: float,
    start_ts: int | None = None,
    end_ts: int | None = None,
) -> list[dict[str, Any]]:
    cutoff_ts = start_ts if start_ts is not None else int(
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
                if end_ts is not None and ts > end_ts:
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
            pnl = sell_proceeds - cost
            status = "WIN" if pnl > 0 else "LOSS"
        elif sell_proceeds > 0 and not data["buys"]:
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
