#!/usr/bin/env python3
"""Analyze Polymarket user timing patterns from public trades.

This script reverse-engineers execution behavior for a public profile/wallet:
- trade flow and notional
- round-trip hold times (BUY -> SELL FIFO matching)
- realized timing quality (markout at +5/+15/+60 min after BUY entries)
- open-lot status vs current price

Usage:
    uv run python scripts/analyze_user_timing.py --user RN1
    uv run python scripts/analyze_user_timing.py --user 0xabc... --sample-days 14
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import httpx
import structlog

logger = structlog.get_logger()

GAMMA_API = "https://gamma-api.polymarket.com"
DATA_API = "https://data-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"


@dataclass(slots=True)
class Trade:
    timestamp: int
    side: str
    asset: str
    condition_id: str
    outcome: str
    title: str
    slug: str
    tx_hash: str
    size: float
    price: float

    @property
    def notional(self) -> float:
        return self.size * self.price


@dataclass(slots=True)
class RoundTrip:
    asset: str
    condition_id: str
    outcome: str
    title: str
    entry_ts: int
    exit_ts: int
    size: float
    entry_price: float
    exit_price: float

    @property
    def hold_seconds(self) -> int:
        return max(0, self.exit_ts - self.entry_ts)

    @property
    def pnl(self) -> float:
        return (self.exit_price - self.entry_price) * self.size

    @property
    def entry_notional(self) -> float:
        return self.entry_price * self.size

    @property
    def roi(self) -> float:
        return self.pnl / self.entry_notional if self.entry_notional > 0 else 0.0


@dataclass(slots=True)
class OpenLot:
    asset: str
    condition_id: str
    outcome: str
    title: str
    entry_ts: int
    size: float
    entry_price: float

    @property
    def entry_notional(self) -> float:
        return self.entry_price * self.size


def parse_horizons(raw: str) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise ValueError("Horizons must be positive minutes.")
        values.append(value)
    if not values:
        raise ValueError("At least one markout horizon is required.")
    return sorted(set(values))


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def is_wallet(value: str) -> bool:
    return value.startswith("0x") and len(value) == 42


def resolve_wallet(client: httpx.Client, user: str) -> tuple[str, str]:
    """Resolve a username (or wallet) to a wallet address."""
    if is_wallet(user):
        return user.lower(), user

    response = client.get(
        f"{GAMMA_API}/public-search",
        params={
            "q": user,
            "search_profiles": "true",
            "search_tags": "false",
            "limit_per_type": 20,
        },
    )
    response.raise_for_status()
    payload = response.json()
    profiles = payload.get("profiles", []) if isinstance(payload, dict) else []
    if not isinstance(profiles, list) or not profiles:
        raise RuntimeError(f"Could not resolve username '{user}' to a wallet.")

    target = user.lower()
    chosen = None
    for profile in profiles:
        name = str(profile.get("name", "")).lower()
        if name == target:
            chosen = profile
            break
    if chosen is None:
        chosen = profiles[0]

    wallet = str(chosen.get("proxyWallet", "")).lower()
    name = str(chosen.get("name", user))
    if not is_wallet(wallet):
        raise RuntimeError(f"Profile '{name}' did not return a valid wallet.")
    return wallet, name


def fetch_trades(
    client: httpx.Client,
    wallet: str,
    page_size: int,
    max_pages: int | None,
) -> list[Trade]:
    """Fetch paginated trade history for a wallet."""
    trades: list[Trade] = []
    offset = 0
    pages = 0

    while True:
        response = client.get(
            f"{DATA_API}/trades",
            params={
                "user": wallet,
                "limit": page_size,
                "offset": offset,
            },
        )
        if response.status_code == 400 and offset > 0:
            # Data API can return 400 when offset reaches the end.
            break
        response.raise_for_status()
        batch = response.json()
        if not isinstance(batch, list) or not batch:
            break

        for raw in batch:
            trade = Trade(
                timestamp=_to_int(raw.get("timestamp")),
                side=str(raw.get("side", "")).upper(),
                asset=str(raw.get("asset", "")),
                condition_id=str(raw.get("conditionId", "")),
                outcome=str(raw.get("outcome", "")),
                title=str(raw.get("title", "")),
                slug=str(raw.get("slug", "")),
                tx_hash=str(raw.get("transactionHash", "")),
                size=_to_float(raw.get("size")),
                price=_to_float(raw.get("price")),
            )
            if trade.asset and trade.size > 0 and 0 < trade.price < 1.01:
                trades.append(trade)

        pages += 1
        if len(batch) < page_size:
            break
        if max_pages is not None and pages >= max_pages:
            break
        offset += len(batch)

    trades.sort(key=lambda t: t.timestamp)
    logger.info("trades_fetched", count=len(trades), pages=pages, wallet=wallet)
    return trades


def chunked(items: list[str], size: int) -> Iterable[list[str]]:
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


def fetch_market_statuses(
    client: httpx.Client,
    condition_ids: list[str],
) -> dict[str, dict[str, bool]]:
    """Fetch market status metadata from Gamma API by condition IDs."""
    statuses: dict[str, dict[str, bool]] = {}
    unique_ids = sorted(set(cid for cid in condition_ids if cid))
    if not unique_ids:
        return statuses

    for batch in chunked(unique_ids, 100):
        try:
            response = client.get(
                f"{GAMMA_API}/markets",
                params={"condition_ids": ",".join(batch)},
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            logger.debug("market_status_fetch_error", error=str(exc))
            continue

        if not isinstance(payload, list):
            continue

        for row in payload:
            cid = str(row.get("conditionId", ""))
            if not cid:
                continue
            statuses[cid] = {
                "active": bool(row.get("active", False)),
                "closed": bool(row.get("closed", False)),
                "accepting_orders": bool(row.get("acceptingOrders", False)),
            }

    return statuses


def build_roundtrips(trades: Iterable[Trade]) -> tuple[list[RoundTrip], list[OpenLot]]:
    """Create FIFO round-trips from BUY/SELL flows for each asset token."""
    inventory: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    roundtrips: list[RoundTrip] = []

    for trade in trades:
        if trade.side == "BUY":
            inventory[trade.asset].append(
                {
                    "remaining": trade.size,
                    "entry_ts": trade.timestamp,
                    "entry_price": trade.price,
                    "asset": trade.asset,
                    "condition_id": trade.condition_id,
                    "outcome": trade.outcome,
                    "title": trade.title,
                }
            )
            continue

        if trade.side != "SELL":
            continue

        qty_left = trade.size
        lots = inventory[trade.asset]
        while qty_left > 1e-12 and lots:
            lot = lots[0]
            matched = min(qty_left, float(lot["remaining"]))
            roundtrips.append(
                RoundTrip(
                    asset=trade.asset,
                    condition_id=str(lot["condition_id"]),
                    outcome=str(lot["outcome"]),
                    title=str(lot["title"]),
                    entry_ts=int(lot["entry_ts"]),
                    exit_ts=trade.timestamp,
                    size=matched,
                    entry_price=float(lot["entry_price"]),
                    exit_price=trade.price,
                )
            )
            lot["remaining"] = float(lot["remaining"]) - matched
            qty_left -= matched
            if lot["remaining"] <= 1e-12:
                lots.popleft()

    open_lots: list[OpenLot] = []
    for asset, lots in inventory.items():
        for lot in lots:
            remaining = float(lot["remaining"])
            if remaining <= 1e-12:
                continue
            open_lots.append(
                OpenLot(
                    asset=asset,
                    condition_id=str(lot["condition_id"]),
                    outcome=str(lot["outcome"]),
                    title=str(lot["title"]),
                    entry_ts=int(lot["entry_ts"]),
                    size=remaining,
                    entry_price=float(lot["entry_price"]),
                )
            )

    open_lots.sort(key=lambda l: l.entry_ts)
    return roundtrips, open_lots


def fetch_price_history(
    client: httpx.Client,
    asset: str,
    start_ts: int,
    end_ts: int,
    fidelity: int = 1,
) -> list[tuple[int, float]]:
    """Fetch CLOB price history for one outcome token."""
    if start_ts >= end_ts:
        return []
    try:
        response = client.get(
            f"{CLOB_API}/prices-history",
            params={
                "market": asset,
                "startTs": start_ts,
                "endTs": end_ts,
                "fidelity": fidelity,
            },
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        logger.debug("price_history_error", asset=asset, error=str(exc))
        return []

    history = payload.get("history", []) if isinstance(payload, dict) else []
    points: list[tuple[int, float]] = []
    for row in history:
        ts = _to_int(row.get("t"))
        price = _to_float(row.get("p"), default=-1.0)
        if ts > 0 and 0 <= price <= 1:
            points.append((ts, price))
    points.sort(key=lambda x: x[0])
    return points


def price_near(
    points: list[tuple[int, float]],
    target_ts: int,
    tolerance_seconds: int,
) -> float | None:
    """Find closest price around target timestamp within tolerance."""
    if not points:
        return None
    ts_only = [p[0] for p in points]
    idx = bisect.bisect_left(ts_only, target_ts)

    best: tuple[int, float] | None = None
    candidates: list[tuple[int, float]] = []
    if idx < len(points):
        candidates.append(points[idx])
    if idx - 1 >= 0:
        candidates.append(points[idx - 1])

    for ts, price in candidates:
        delta = abs(ts - target_ts)
        if delta <= tolerance_seconds:
            if best is None or delta < abs(best[0] - target_ts):
                best = (ts, price)

    return best[1] if best is not None else None


def summarize_roundtrip_holds(roundtrips: list[RoundTrip]) -> dict[str, int]:
    buckets = {
        "<5m": 0,
        "5m-30m": 0,
        "30m-2h": 0,
        "2h-24h": 0,
        ">24h": 0,
    }
    for r in roundtrips:
        seconds = r.hold_seconds
        if seconds < 5 * 60:
            buckets["<5m"] += 1
        elif seconds < 30 * 60:
            buckets["5m-30m"] += 1
        elif seconds < 2 * 3600:
            buckets["30m-2h"] += 1
        elif seconds < 24 * 3600:
            buckets["2h-24h"] += 1
        else:
            buckets[">24h"] += 1
    return buckets


def summarize_condition_coverage(open_lots: list[OpenLot]) -> dict[str, Any]:
    grouped: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"title": "", "outcomes": set(), "notional": 0.0}
    )
    for lot in open_lots:
        row = grouped[lot.condition_id]
        row["title"] = lot.title
        row["outcomes"].add(lot.outcome)
        row["notional"] += lot.entry_notional

    if not grouped:
        return {}

    total_conditions = len(grouped)
    two_sided = [row for row in grouped.values() if len(row["outcomes"]) >= 2]
    total_notional = sum(row["notional"] for row in grouped.values())
    two_sided_notional = sum(row["notional"] for row in two_sided)

    top_two_sided = sorted(two_sided, key=lambda r: r["notional"], reverse=True)[:10]

    return {
        "total_conditions": total_conditions,
        "two_sided_conditions": len(two_sided),
        "two_sided_condition_ratio": len(two_sided) / total_conditions,
        "total_notional": total_notional,
        "two_sided_notional": two_sided_notional,
        "two_sided_notional_ratio": (
            two_sided_notional / total_notional if total_notional > 0 else 0.0
        ),
        "top_two_sided_by_notional": [
            {
                "title": row["title"],
                "outcomes": sorted(row["outcomes"]),
                "notional": row["notional"],
            }
            for row in top_two_sided
        ],
    }


def compute_markouts(
    client: httpx.Client,
    trades: list[Trade],
    horizons_min: list[int],
    sample_days: int,
    min_notional: float,
    fidelity: int,
    max_assets: int,
) -> tuple[dict[int, dict[str, float]], int]:
    """Compute post-entry markout for recent BUY entries."""
    now_ts = int(datetime.now(UTC).timestamp())
    cutoff_ts = now_ts - sample_days * 86400

    sampled = [
        t for t in trades
        if t.side == "BUY"
        and t.timestamp >= cutoff_ts
        and t.notional >= min_notional
    ]
    if not sampled:
        return {}, 0

    max_horizon = max(horizons_min) * 60
    grouped: dict[str, list[Trade]] = defaultdict(list)
    for t in sampled:
        grouped[t.asset].append(t)

    # Focus on assets with the highest sampled notional.
    ranked_assets = sorted(
        grouped.keys(),
        key=lambda asset: sum(t.notional for t in grouped[asset]),
        reverse=True,
    )
    selected_assets = ranked_assets[:max_assets] if max_assets > 0 else ranked_assets

    # Gather all markouts by horizon
    values: dict[int, list[tuple[float, float]]] = {h: [] for h in horizons_min}
    # tuple: (delta_price, entry_notional_weight)

    for asset in selected_assets:
        asset_trades = grouped[asset]
        asset_trades.sort(key=lambda t: t.timestamp)
        start_ts = asset_trades[0].timestamp - 900
        end_ts = asset_trades[-1].timestamp + max_horizon + 900
        points = fetch_price_history(
            client,
            asset=asset,
            start_ts=start_ts,
            end_ts=end_ts,
            fidelity=fidelity,
        )
        if len(points) < 2:
            continue

        tolerance = max(180, fidelity * 60 * 2)
        for trade in asset_trades:
            for horizon in horizons_min:
                target_ts = trade.timestamp + horizon * 60
                future_price = price_near(points, target_ts, tolerance_seconds=tolerance)
                if future_price is None:
                    continue
                delta = future_price - trade.price
                values[horizon].append((delta, trade.notional))

    summary: dict[int, dict[str, float]] = {}
    for horizon in horizons_min:
        rows = values[horizon]
        if not rows:
            continue
        deltas = [d for d, _ in rows]
        weights = [w for _, w in rows]
        total_weight = sum(weights)
        weighted_avg = (
            sum(d * w for d, w in rows) / total_weight if total_weight > 0 else 0.0
        )
        positive_ratio = sum(1 for d in deltas if d > 0) / len(deltas)
        summary[horizon] = {
            "samples": float(len(rows)),
            "avg_delta": sum(deltas) / len(deltas),
            "weighted_avg_delta": weighted_avg,
            "positive_ratio": positive_ratio,
        }

    return summary, len(sampled)


def mark_open_lots(
    client: httpx.Client,
    open_lots: list[OpenLot],
    max_assets: int,
    active_only: bool,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Mark open lots to latest price from CLOB history."""
    if not open_lots:
        return {}, []

    market_status = fetch_market_statuses(
        client, [lot.condition_id for lot in open_lots]
    )
    filtered_lots = open_lots
    if active_only and market_status:
        filtered_lots = [
            lot
            for lot in open_lots
            if lot.condition_id in market_status
            and market_status[lot.condition_id]["active"]
            and not market_status[lot.condition_id]["closed"]
        ]

    if not filtered_lots:
        return {
            "input_open_lots": float(len(open_lots)),
            "filtered_open_lots": 0.0,
            "active_only": 1.0 if active_only else 0.0,
        }, []

    # Focus on largest exposures first for faster/better signal.
    open_sorted = sorted(filtered_lots, key=lambda l: l.entry_notional, reverse=True)
    selected_assets: list[str] = []
    seen_assets: set[str] = set()
    for lot in open_sorted:
        if lot.asset in seen_assets:
            continue
        selected_assets.append(lot.asset)
        seen_assets.add(lot.asset)
        if len(selected_assets) >= max_assets:
            break

    now_ts = int(datetime.now(UTC).timestamp())
    current_prices: dict[str, float] = {}
    for asset in selected_assets:
        points = fetch_price_history(
            client,
            asset=asset,
            start_ts=now_ts - 3600 * 48,
            end_ts=now_ts,
            fidelity=5,
        )
        if points:
            current_prices[asset] = points[-1][1]

    rows: list[dict[str, Any]] = []
    now_ts = int(datetime.now(UTC).timestamp())
    for lot in filtered_lots:
        current = current_prices.get(lot.asset)
        if current is None:
            continue
        pnl = (current - lot.entry_price) * lot.size
        roi = pnl / lot.entry_notional if lot.entry_notional > 0 else 0.0
        age_hours = (now_ts - lot.entry_ts) / 3600.0
        rows.append(
            {
                "asset": lot.asset,
                "condition_id": lot.condition_id,
                "title": lot.title,
                "outcome": lot.outcome,
                "entry_ts": lot.entry_ts,
                "entry_price": lot.entry_price,
                "current_price": current,
                "size": lot.size,
                "entry_notional": lot.entry_notional,
                "pnl_now": pnl,
                "roi_now": roi,
                "age_hours": age_hours,
            }
        )

    if not rows:
        return {}, []

    total_notional = sum(r["entry_notional"] for r in rows)
    weighted_roi = (
        sum(r["roi_now"] * r["entry_notional"] for r in rows) / total_notional
        if total_notional > 0
        else 0.0
    )
    age_windows = [
        ("<=2h", 0.0, 2.0),
        ("2-6h", 2.0, 6.0),
        ("6-12h", 6.0, 12.0),
        ("12-24h", 12.0, 24.0),
        (">24h", 24.0, float("inf")),
    ]
    age_summary: dict[str, dict[str, float]] = {}
    for label, lo, hi in age_windows:
        bucket = [r for r in rows if lo <= r["age_hours"] < hi]
        if not bucket:
            continue
        bucket_notional = sum(r["entry_notional"] for r in bucket)
        bucket_wroi = (
            sum(r["roi_now"] * r["entry_notional"] for r in bucket) / bucket_notional
            if bucket_notional > 0
            else 0.0
        )
        age_summary[label] = {
            "count": float(len(bucket)),
            "notional": bucket_notional,
            "weighted_roi": bucket_wroi,
        }

    summary = {
        "input_open_lots": float(len(open_lots)),
        "filtered_open_lots": float(len(filtered_lots)),
        "active_only": 1.0 if active_only else 0.0,
        "marked_open_lots": float(len(rows)),
        "total_entry_notional": total_notional,
        "total_unrealized_pnl": sum(r["pnl_now"] for r in rows),
        "weighted_roi_now": weighted_roi,
        "negative_ratio": sum(1 for r in rows if r["roi_now"] < 0) / len(rows),
        "age_buckets": age_summary,
    }
    return summary, rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def ts_to_iso(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=UTC).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze timing patterns for a Polymarket user.")
    parser.add_argument("--user", required=True, help="Username (e.g. RN1) or wallet address.")
    parser.add_argument("--page-size", type=int, default=500, help="Trades page size.")
    parser.add_argument("--max-pages", type=int, default=None, help="Optional page cap for faster runs.")
    parser.add_argument("--sample-days", type=int, default=30, help="Window for markout analysis.")
    parser.add_argument("--min-notional", type=float, default=25.0, help="Min buy notional for markout samples.")
    parser.add_argument(
        "--horizons",
        type=str,
        default="5,15,60",
        help="Comma-separated markout horizons in minutes.",
    )
    parser.add_argument("--fidelity", type=int, default=1, help="Price history fidelity (minutes).")
    parser.add_argument(
        "--max-markout-assets",
        type=int,
        default=250,
        help="Max number of assets for markout analysis (highest sampled notional first).",
    )
    parser.add_argument(
        "--include-resolved-open",
        action="store_true",
        help="Include resolved/closed markets in open-lot marking.",
    )
    parser.add_argument("--max-open-assets", type=int, default=300, help="Max assets for open-lot marking.")
    parser.add_argument("--output-dir", type=str, default="reports", help="Directory for output artifacts.")
    args = parser.parse_args()

    horizons = parse_horizons(args.horizons)

    timeout = httpx.Timeout(30.0, connect=15.0)
    with httpx.Client(timeout=timeout) as client:
        wallet, resolved_name = resolve_wallet(client, args.user)
        trades = fetch_trades(
            client=client,
            wallet=wallet,
            page_size=args.page_size,
            max_pages=args.max_pages,
        )
        if not trades:
            raise RuntimeError("No trades found for this user.")

        roundtrips, open_lots = build_roundtrips(trades)
        markout_summary, sampled_entries = compute_markouts(
            client=client,
            trades=trades,
            horizons_min=horizons,
            sample_days=args.sample_days,
            min_notional=args.min_notional,
            fidelity=args.fidelity,
            max_assets=args.max_markout_assets,
        )
        open_summary, marked_open_rows = mark_open_lots(
            client=client,
            open_lots=open_lots,
            max_assets=args.max_open_assets,
            active_only=not args.include_resolved_open,
        )
        condition_coverage = summarize_condition_coverage(open_lots)

    buy_count = sum(1 for t in trades if t.side == "BUY")
    sell_count = sum(1 for t in trades if t.side == "SELL")
    total_notional = sum(t.notional for t in trades)

    hold_minutes = [r.hold_seconds / 60.0 for r in roundtrips]
    realized_pnl = sum(r.pnl for r in roundtrips)
    realized_notional = sum(r.entry_notional for r in roundtrips)
    win_rate = (
        sum(1 for r in roundtrips if r.pnl > 0) / len(roundtrips)
        if roundtrips
        else 0.0
    )

    summary: dict[str, Any] = {
        "user_input": args.user,
        "resolved_name": resolved_name,
        "wallet": wallet,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "trade_window": {
            "start": ts_to_iso(trades[0].timestamp),
            "end": ts_to_iso(trades[-1].timestamp),
        },
        "trade_stats": {
            "trades": len(trades),
            "buy_count": buy_count,
            "sell_count": sell_count,
            "total_notional": total_notional,
            "avg_trade_notional": total_notional / len(trades),
        },
        "roundtrip_stats": {
            "closed_roundtrips": len(roundtrips),
            "realized_pnl": realized_pnl,
            "realized_roi": (realized_pnl / realized_notional) if realized_notional > 0 else 0.0,
            "win_rate": win_rate,
            "median_hold_min": (
                sorted(hold_minutes)[len(hold_minutes) // 2] if hold_minutes else 0.0
            ),
            "mean_hold_min": (sum(hold_minutes) / len(hold_minutes)) if hold_minutes else 0.0,
            "hold_buckets": summarize_roundtrip_holds(roundtrips),
        },
        "open_lots": {
            "count": len(open_lots),
            "total_entry_notional": sum(l.entry_notional for l in open_lots),
            "marked_summary": open_summary,
            "condition_coverage": condition_coverage,
        },
        "markout": {
            "sampled_recent_buy_entries": sampled_entries,
            "horizons_min": horizons,
            "sample_days": args.sample_days,
            "min_notional": args.min_notional,
            "max_markout_assets": args.max_markout_assets,
            "summary": markout_summary,
        },
    }

    output_dir = Path(args.output_dir) / "timing_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = resolved_name.lower().replace(" ", "_")

    summary_path = output_dir / f"{slug}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    roundtrip_rows = [
        {
            "asset": r.asset,
            "condition_id": r.condition_id,
            "outcome": r.outcome,
            "title": r.title,
            "entry_ts": r.entry_ts,
            "exit_ts": r.exit_ts,
            "hold_seconds": r.hold_seconds,
            "size": r.size,
            "entry_price": r.entry_price,
            "exit_price": r.exit_price,
            "pnl": r.pnl,
            "roi": r.roi,
        }
        for r in roundtrips
    ]
    open_rows = [
        {
            "asset": lot.asset,
            "condition_id": lot.condition_id,
            "outcome": lot.outcome,
            "title": lot.title,
            "entry_ts": lot.entry_ts,
            "size": lot.size,
            "entry_price": lot.entry_price,
            "entry_notional": lot.entry_notional,
        }
        for lot in open_lots
    ]
    write_csv(output_dir / f"{slug}_roundtrips.csv", roundtrip_rows)
    write_csv(output_dir / f"{slug}_open_lots.csv", open_rows)
    write_csv(output_dir / f"{slug}_open_lots_marked.csv", marked_open_rows)

    # Console report
    print("=" * 72)
    print(f"TIMING ANALYSIS — {resolved_name} ({wallet})")
    print("=" * 72)
    print(f"Trades: {len(trades):,} (BUY {buy_count:,} / SELL {sell_count:,})")
    print(f"Trade window: {summary['trade_window']['start']} -> {summary['trade_window']['end']}")
    print(f"Total traded notional: ${total_notional:,.2f}")
    print()
    print("Round-trips (FIFO matched):")
    print(f"  Closed: {len(roundtrips):,}")
    print(f"  Realized PnL: ${realized_pnl:,.2f}")
    print(f"  Realized ROI: {summary['roundtrip_stats']['realized_roi']:.2%}")
    print(f"  Win rate: {win_rate:.2%}")
    print(f"  Median hold: {summary['roundtrip_stats']['median_hold_min']:.1f} min")
    print(f"  Mean hold: {summary['roundtrip_stats']['mean_hold_min']:.1f} min")
    print(f"  Hold buckets: {summary['roundtrip_stats']['hold_buckets']}")
    print()
    print("Recent BUY markout:")
    print(
        f"  Sampled entries: {sampled_entries:,} (last {args.sample_days}d, min notional ${args.min_notional:.2f})"
    )
    if markout_summary:
        for horizon in horizons:
            row = markout_summary.get(horizon)
            if not row:
                continue
            print(
                f"  +{horizon:>3}m | n={int(row['samples']):>5} | "
                f"avg={row['avg_delta']:+.4f} | wavg={row['weighted_avg_delta']:+.4f} | "
                f"positive={row['positive_ratio']:.1%}"
            )
    else:
        print("  Not enough data for markout.")
    print()
    print("Open lots:")
    print(f"  Open lots count: {len(open_lots):,}")
    print(f"  Total entry notional: ${summary['open_lots']['total_entry_notional']:,.2f}")
    coverage = summary["open_lots"].get("condition_coverage", {})
    if coverage:
        print(
            "  Two-sided conditions: "
            f"{coverage['two_sided_conditions']}/{coverage['total_conditions']} "
            f"({coverage['two_sided_condition_ratio']:.1%})"
        )
        print(
            "  Two-sided notional share: "
            f"{coverage['two_sided_notional_ratio']:.1%}"
        )
    if open_summary:
        active_flag = bool(int(open_summary.get("active_only", 0)))
        print(
            f"  Filtered lots ({'active only' if active_flag else 'including resolved'}): "
            f"{int(open_summary.get('filtered_open_lots', 0))}"
        )
        if "marked_open_lots" in open_summary:
            print(f"  Marked lots: {int(open_summary['marked_open_lots'])}")
            print(f"  Weighted ROI now: {open_summary['weighted_roi_now']:.2%}")
            print(f"  Unrealized PnL now: ${open_summary['total_unrealized_pnl']:,.2f}")
            print(f"  Negative ratio: {open_summary['negative_ratio']:.1%}")
            age_buckets = open_summary.get("age_buckets", {})
            if age_buckets:
                print("  Weighted ROI by age:")
                for label, row in age_buckets.items():
                    print(
                        f"    {label:>6} | n={int(row['count']):>3} | "
                        f"notional=${row['notional']:,.2f} | wROI={row['weighted_roi']:+.2%}"
                    )
        else:
            print("  No markable lots after filtering.")
    else:
        print("  Could not mark open lots to current prices.")
    print()
    print(f"Summary JSON: {summary_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
