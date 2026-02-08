#!/usr/bin/env python3
"""Snapshot RN1 transactions and compare local entry alignment.

This script does two things in one run:
1) stores RN1 transactions to disk (JSON + CSV)
2) checks if local strategy entries matched RN1 entry locations
   at condition/outcome/timing level
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.rn1_comparison import (  # noqa: E402
    DEFAULT_RN1_WALLET,
    ActivityEvent,
    build_rn1_transaction_report,
    load_local_two_sided_events,
)

GAMMA_API = "https://gamma-api.polymarket.com/markets"

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Store RN1 transactions and compare whether local entries matched them."
    )
    parser.add_argument("--db", type=str, default="data/arb.db", help="SQLite path or SQLAlchemy URL.")
    parser.add_argument("--hours", type=float, default=12.0, help="Analysis window in hours.")
    parser.add_argument(
        "--strategy-tag",
        type=str,
        default=None,
        help="Optional local strategy_tag filter.",
    )
    parser.add_argument("--rn1-wallet", type=str, default=DEFAULT_RN1_WALLET, help="Benchmark wallet.")
    parser.add_argument("--page-limit", type=int, default=500, help="RN1 activity rows per page.")
    parser.add_argument("--max-pages", type=int, default=20, help="Maximum RN1 pages to fetch.")
    parser.add_argument(
        "--transaction-limit",
        type=int,
        default=20000,
        help="Maximum transaction rows fetched from RN1 report endpoint.",
    )
    parser.add_argument(
        "--time-tolerance-minutes",
        type=float,
        default=10.0,
        help="Timing window to consider RN1/local entries as same location.",
    )
    parser.add_argument(
        "--event-prefixes",
        type=str,
        default="epl,cs2,lal,nba,fl1,sea,por,tur,cbb",
        help="Comma-separated event slug/league prefixes for in-focus universe estimation.",
    )
    parser.add_argument("--screen-min-liquidity", type=float, default=100.0, help="Static universe filter.")
    parser.add_argument("--screen-min-volume-24h", type=float, default=20.0, help="Static universe filter.")
    parser.add_argument(
        "--screen-max-days-to-end",
        type=float,
        default=3.0,
        help="Static universe filter (0 to disable).",
    )
    parser.add_argument(
        "--screen-include-nonsports",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Static universe filter: include non-sports.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="reports/rn1",
        help="Directory where RN1 snapshots and alignment reports are saved.",
    )
    return parser


def _csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _safe_name(raw: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_") or "all"


def _parse_prefixes(raw: str) -> list[str]:
    return [part.strip().lower() for part in raw.split(",") if part.strip()]


def _parse_json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
            return decoded if isinstance(decoded, list) else []
        except json.JSONDecodeError:
            return []
    return []


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_datetime(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str) and value:
        raw = value.strip()
        if raw.endswith("Z"):
            raw = raw.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(raw)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


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

    for outcome in outcomes:
        if str(outcome).strip().lower() in {"yes", "no", "over", "under"}:
            continue
        if len(str(outcome).strip()) > 2:
            return True

    return False


def _in_focus_universe(tx: dict[str, Any], prefixes: list[str]) -> bool:
    if not prefixes:
        return True
    league = str(tx.get("league_prefix") or "").strip().lower()
    slug = str(tx.get("event_slug") or "").strip().lower()
    for pref in prefixes:
        if league == pref:
            return True
        if slug == pref or slug.startswith(f"{pref}-"):
            return True
    return False


def _fetch_market_map(condition_ids: list[str], *, timeout_seconds: float = 30.0) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    unique_ids = [cid for cid in dict.fromkeys(condition_ids) if cid]
    if not unique_ids:
        return out

    with httpx.Client(timeout=timeout_seconds) as client:
        chunk_size = 40
        for idx in range(0, len(unique_ids), chunk_size):
            chunk = unique_ids[idx : idx + chunk_size]
            try:
                response = client.get(GAMMA_API, params=[("condition_ids", cid) for cid in chunk])
                response.raise_for_status()
                rows = response.json()
            except Exception:
                continue

            if not isinstance(rows, list):
                continue

            for raw in rows:
                if not isinstance(raw, dict):
                    continue
                cid = str(raw.get("conditionId") or "")
                if cid:
                    out[cid] = raw

    return out


def _static_screen_fail_reasons(
    tx: dict[str, Any],
    market: Optional[dict[str, Any]],
    *,
    prefixes: list[str],
    min_liquidity: float,
    min_volume_24h: float,
    max_days_to_end: float,
    include_nonsports: bool,
) -> list[str]:
    reasons: list[str] = []
    if market is None:
        return ["market_not_found"]

    liquidity = _safe_float(market.get("liquidityNum"), default=-1.0)
    volume_24h = _safe_float(market.get("volume24hr"), default=-1.0)
    outcomes = [str(o) for o in _parse_json_list(market.get("outcomes", []))]
    clob_ids = [str(t) for t in _parse_json_list(market.get("clobTokenIds", []))]
    question = str(market.get("question") or "")
    if len(outcomes) != 2 or len(clob_ids) < 2:
        reasons.append("not_binary_two_leg")
    if not question:
        reasons.append("missing_question")

    if min_liquidity > 0 and liquidity < min_liquidity:
        reasons.append("liquidity_below_min")
    if min_volume_24h > 0 and volume_24h < min_volume_24h:
        reasons.append("volume24h_below_min")

    if not include_nonsports and not _looks_sports(question, outcomes):
        reasons.append("filtered_nonsports")

    if prefixes and not _in_focus_universe(tx, prefixes):
        reasons.append("prefix_filtered")

    if max_days_to_end > 0:
        end_dt = _parse_datetime(market.get("endDate"))
        tx_ts = int(tx.get("timestamp") or 0)
        tx_dt = datetime.fromtimestamp(tx_ts, tz=timezone.utc) if tx_ts > 0 else None
        if end_dt is None or tx_dt is None:
            reasons.append("missing_enddate")
        else:
            delta_days = (end_dt - tx_dt).total_seconds() / 86400.0
            if delta_days < -1.0 or delta_days > max_days_to_end:
                reasons.append("enddate_outside_window")

    return reasons


def _nearest_gap_seconds(ts: int, series: Optional[list[int]]) -> Optional[int]:
    if not series:
        return None
    return min(abs(ts - value) for value in series)


def _status_priority(status: str) -> int:
    # Lower means "better match" first in sorted outputs.
    order = {
        "matched_outcome_timing": 0,
        "matched_condition_timing_only": 1,
        "matched_outcome_different_timing": 2,
        "matched_condition_different_outcome": 3,
        "not_entered_condition": 4,
        "out_of_focus_universe": 5,
    }
    return order.get(status, 99)


def _build_alignment_rows(
    rn1_buy_rows: list[dict[str, Any]],
    local_buy_events: Iterable[ActivityEvent],
    *,
    time_tolerance_seconds: int,
    focus_prefixes: list[str],
    market_map: dict[str, dict[str, Any]],
    screen_min_liquidity: float,
    screen_min_volume_24h: float,
    screen_max_days_to_end: float,
    screen_include_nonsports: bool,
) -> list[dict[str, Any]]:
    by_condition: dict[str, list[int]] = defaultdict(list)
    by_condition_outcome: dict[tuple[str, str], list[int]] = defaultdict(list)

    for item in local_buy_events:
        ts = int(item.timestamp)
        cid = item.condition_id
        outcome = item.outcome
        by_condition[cid].append(ts)
        by_condition_outcome[(cid, outcome)].append(ts)

    for series in by_condition.values():
        series.sort()
    for series in by_condition_outcome.values():
        series.sort()

    out: list[dict[str, Any]] = []
    for tx in rn1_buy_rows:
        ts = int(tx.get("timestamp") or 0)
        cid = str(tx.get("condition_id") or "")
        outcome = str(tx.get("outcome") or "")
        same_outcome_times = by_condition_outcome.get((cid, outcome))
        same_condition_times = by_condition.get(cid)

        nearest_outcome_gap = _nearest_gap_seconds(ts, same_outcome_times)
        nearest_condition_gap = _nearest_gap_seconds(ts, same_condition_times)

        same_outcome_near = (
            nearest_outcome_gap is not None and nearest_outcome_gap <= time_tolerance_seconds
        )
        same_condition_near = (
            nearest_condition_gap is not None and nearest_condition_gap <= time_tolerance_seconds
        )
        same_outcome_any = same_outcome_times is not None
        same_condition_any = same_condition_times is not None
        in_focus = _in_focus_universe(tx, focus_prefixes)

        if not in_focus:
            status = "out_of_focus_universe"
        elif same_outcome_near:
            status = "matched_outcome_timing"
        elif same_condition_near:
            status = "matched_condition_timing_only"
        elif same_outcome_any:
            status = "matched_outcome_different_timing"
        elif same_condition_any:
            status = "matched_condition_different_outcome"
        else:
            status = "not_entered_condition"

        static_fail = _static_screen_fail_reasons(
            tx,
            market_map.get(cid),
            prefixes=focus_prefixes,
            min_liquidity=screen_min_liquidity,
            min_volume_24h=screen_min_volume_24h,
            max_days_to_end=screen_max_days_to_end,
            include_nonsports=screen_include_nonsports,
        )
        static_pass = len(static_fail) == 0

        usdc = float(tx.get("usdc_size") or 0.0)
        out.append(
            {
                "timestamp": ts,
                "datetime_utc": tx.get("datetime_utc"),
                "condition_id": cid,
                "outcome": outcome,
                "title": tx.get("title"),
                "event_slug": tx.get("event_slug"),
                "league_prefix": tx.get("league_prefix"),
                "market_type": tx.get("market_type"),
                "price": tx.get("price"),
                "usdc_size": usdc,
                "in_focus_universe": in_focus,
                "local_same_condition_any": same_condition_any,
                "local_same_outcome_any": same_outcome_any,
                "local_same_condition_within_tolerance": same_condition_near,
                "local_same_outcome_within_tolerance": same_outcome_near,
                "nearest_condition_gap_seconds": nearest_condition_gap,
                "nearest_outcome_gap_seconds": nearest_outcome_gap,
                "alignment_status": status,
                "static_screen_pass": static_pass,
                "static_screen_fail_reasons": "|".join(static_fail),
            }
        )
    return out


def _summarize_alignment(
    rows: list[dict[str, Any]],
    *,
    focus_prefixes: list[str],
    tolerance_seconds: int,
) -> dict[str, Any]:
    if not rows:
        return {
            "rn1_buy_trades": 0,
            "focus_prefixes": focus_prefixes,
            "time_tolerance_seconds": tolerance_seconds,
            "status_counts": {},
            "status_usdc": {},
            "coverage": {},
            "top_unmatched_conditions": [],
        }

    status_counts: Counter[str] = Counter()
    status_usdc: defaultdict[str, float] = defaultdict(float)
    condition_buckets: dict[str, dict[str, Any]] = {}

    for row in rows:
        status = str(row["alignment_status"])
        status_counts[status] += 1
        status_usdc[status] += float(row.get("usdc_size") or 0.0)

        if status == "matched_outcome_timing":
            continue

        cid = str(row.get("condition_id") or "")
        bucket = condition_buckets.setdefault(
            cid,
            {
                "condition_id": cid,
                "title": row.get("title"),
                "event_slug": row.get("event_slug"),
                "league_prefix": row.get("league_prefix"),
                "market_type": row.get("market_type"),
                "rn1_buy_trades": 0,
                "rn1_buy_usdc": 0.0,
                "status_breakdown": Counter(),
            },
        )
        bucket["rn1_buy_trades"] += 1
        bucket["rn1_buy_usdc"] += float(row.get("usdc_size") or 0.0)
        bucket["status_breakdown"][status] += 1

    total = float(len(rows))
    total_usdc = sum(float(row.get("usdc_size") or 0.0) for row in rows)
    matched_outcome = float(status_counts.get("matched_outcome_timing", 0))
    matched_condition = matched_outcome + float(status_counts.get("matched_condition_timing_only", 0))

    top_unmatched = sorted(
        (
            {
                **bucket,
                "rn1_buy_usdc": round(float(bucket["rn1_buy_usdc"]), 6),
                "status_breakdown": dict(bucket["status_breakdown"]),
            }
            for bucket in condition_buckets.values()
        ),
        key=lambda item: float(item["rn1_buy_usdc"]),
        reverse=True,
    )[:50]

    static_pass_count = sum(1 for row in rows if row.get("static_screen_pass"))
    static_pass_usdc = sum(float(row.get("usdc_size") or 0.0) for row in rows if row.get("static_screen_pass"))
    static_fail_reason_counter: Counter[str] = Counter()
    for row in rows:
        reasons_raw = str(row.get("static_screen_fail_reasons") or "").strip()
        if not reasons_raw:
            continue
        for token in reasons_raw.split("|"):
            if token:
                static_fail_reason_counter[token] += 1

    return {
        "rn1_buy_trades": int(total),
        "focus_prefixes": focus_prefixes,
        "time_tolerance_seconds": tolerance_seconds,
        "status_counts": dict(status_counts),
        "status_usdc": {key: round(value, 6) for key, value in status_usdc.items()},
        "static_screen": {
            "pass_count": static_pass_count,
            "pass_ratio": (static_pass_count / total) if total > 0 else 0.0,
            "pass_usdc": round(static_pass_usdc, 6),
            "pass_usdc_ratio": (static_pass_usdc / total_usdc) if total_usdc > 0 else 0.0,
            "fail_reason_counts": dict(static_fail_reason_counter),
        },
        "coverage": {
            "matched_outcome_timing_ratio": (matched_outcome / total) if total > 0 else 0.0,
            "matched_condition_timing_ratio": (matched_condition / total) if total > 0 else 0.0,
            "matched_outcome_timing_usdc_ratio": (
                status_usdc.get("matched_outcome_timing", 0.0) / total_usdc
            )
            if total_usdc > 0
            else 0.0,
        },
        "top_unmatched_conditions": top_unmatched,
    }


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts_tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    strategy_label = _safe_name(args.strategy_tag or "all")
    focus_prefixes = _parse_prefixes(args.event_prefixes)
    tolerance_seconds = max(1, int(round(args.time_tolerance_minutes * 60.0)))

    rn1_report = build_rn1_transaction_report(
        window_hours=args.hours,
        rn1_wallet=args.rn1_wallet,
        page_limit=args.page_limit,
        max_pages=args.max_pages,
        include_transactions=True,
        transaction_limit=args.transaction_limit,
        top_conditions=500,
    )
    rn1_transactions = rn1_report.get("transactions", [])
    rn1_buy_rows = [
        row for row in rn1_transactions if row.get("type") == "TRADE" and row.get("side") == "BUY"
    ]

    local_events = load_local_two_sided_events(
        db_url=args.db,
        window_hours=args.hours,
        strategy_tag=args.strategy_tag,
    )
    local_buy_events = [row for row in local_events if row.side == "BUY"]
    market_map = _fetch_market_map([str(row.get("condition_id") or "") for row in rn1_buy_rows])

    alignment_rows = _build_alignment_rows(
        rn1_buy_rows,
        local_buy_events,
        time_tolerance_seconds=tolerance_seconds,
        focus_prefixes=focus_prefixes,
        market_map=market_map,
        screen_min_liquidity=args.screen_min_liquidity,
        screen_min_volume_24h=args.screen_min_volume_24h,
        screen_max_days_to_end=args.screen_max_days_to_end,
        screen_include_nonsports=args.screen_include_nonsports,
    )
    alignment_summary = _summarize_alignment(
        alignment_rows,
        focus_prefixes=focus_prefixes,
        tolerance_seconds=tolerance_seconds,
    )

    tx_json_path = out_dir / f"rn1_transactions_{args.hours:g}h_{ts_tag}.json"
    tx_csv_path = out_dir / f"rn1_transactions_{args.hours:g}h_{ts_tag}.csv"
    align_json_path = out_dir / f"rn1_vs_local_alignment_{strategy_label}_{args.hours:g}h_{ts_tag}.json"
    align_csv_path = out_dir / f"rn1_vs_local_alignment_{strategy_label}_{args.hours:g}h_{ts_tag}.csv"

    tx_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window_hours": args.hours,
        "rn1_wallet": args.rn1_wallet,
        "summary": rn1_report.get("summary", {}),
        "distribution": rn1_report.get("distribution", {}),
        "method_inference": rn1_report.get("method_inference", {}),
        "transactions": rn1_transactions,
    }
    tx_json_path.write_text(json.dumps(tx_payload, indent=2), encoding="utf-8")
    _csv_write(tx_csv_path, rn1_transactions)

    align_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window_hours": args.hours,
        "strategy_tag": args.strategy_tag,
        "rn1_wallet": args.rn1_wallet,
        "db_url": args.db,
        "alignment_summary": alignment_summary,
        "alignment_rows": sorted(
            alignment_rows,
            key=lambda item: (_status_priority(str(item["alignment_status"])), -float(item["usdc_size"])),
        ),
    }
    align_json_path.write_text(json.dumps(align_payload, indent=2), encoding="utf-8")
    _csv_write(align_csv_path, alignment_rows)

    print("=" * 96)
    print(
        f"RN1 snapshot saved: trades={len(rn1_transactions)} (BUY trades={len(rn1_buy_rows)}) "
        f"| local buys={len(local_buy_events)} | strategy_tag={args.strategy_tag or 'ALL'}"
    )
    print("=" * 96)
    print(f"RN1 JSON: {tx_json_path}")
    print(f"RN1 CSV : {tx_csv_path}")
    print(f"ALIGN JSON: {align_json_path}")
    print(f"ALIGN CSV : {align_csv_path}")
    print("-" * 96)

    summary = alignment_summary
    print(f"Status counts: {summary.get('status_counts', {})}")
    print(f"Status usdc  : {summary.get('status_usdc', {})}")
    print(f"Static screen: {summary.get('static_screen', {})}")
    print(f"Coverage     : {summary.get('coverage', {})}")

    top_miss = summary.get("top_unmatched_conditions", [])[:10]
    if top_miss:
        print("\nTop unmatched conditions (by RN1 buy USDC):")
        for idx, row in enumerate(top_miss, start=1):
            print(
                f"{idx:>2}. ${float(row.get('rn1_buy_usdc') or 0):>10,.2f} "
                f"trades={int(row.get('rn1_buy_trades') or 0):>4} "
                f"{str(row.get('league_prefix') or ''):<5} {str(row.get('title') or '')[:72]}"
            )


if __name__ == "__main__":
    main()
