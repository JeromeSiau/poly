#!/usr/bin/env python3
"""Deep RN1 transaction analyzer.

Usage:
    uv run python scripts/analyze_rn1_transactions.py --hours 6
    uv run python scripts/analyze_rn1_transactions.py --hours 24 --include-transactions --transaction-limit 5000
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.analysis.rn1_comparison import DEFAULT_RN1_WALLET, build_rn1_transaction_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze RN1 transactions condition by condition.")
    parser.add_argument("--hours", type=float, default=6.0, help="Analysis window in hours.")
    parser.add_argument("--rn1-wallet", type=str, default=DEFAULT_RN1_WALLET, help="Benchmark wallet.")
    parser.add_argument("--page-limit", type=int, default=500, help="Rows fetched per page from activity API.")
    parser.add_argument("--max-pages", type=int, default=7, help="Maximum pages fetched from activity API.")
    parser.add_argument(
        "--include-transactions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include per-transaction rows in report payload.",
    )
    parser.add_argument("--transaction-limit", type=int, default=5000, help="Max transaction rows in output.")
    parser.add_argument("--top-conditions", type=int, default=100, help="Top condition summaries to keep.")
    parser.add_argument("--json-out", type=str, default=None, help="Optional JSON output path.")
    parser.add_argument("--transactions-csv", type=str, default=None, help="Optional transactions CSV output.")
    parser.add_argument("--conditions-csv", type=str, default=None, help="Optional conditions CSV output.")
    parser.add_argument(
        "--write-default-files",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write JSON/CSV files to reports/rn1 when explicit output paths are not provided.",
    )
    return parser


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = build_parser().parse_args()
    report = build_rn1_transaction_report(
        window_hours=args.hours,
        rn1_wallet=args.rn1_wallet,
        page_limit=args.page_limit,
        max_pages=args.max_pages,
        include_transactions=args.include_transactions,
        transaction_limit=args.transaction_limit,
        top_conditions=args.top_conditions,
    )

    summary = report.get("summary", {})
    dist = report.get("distribution", {})
    print("=" * 90)
    print(
        f"RN1 deep analysis | window={report.get('window_hours')}h "
        f"| events={summary.get('events_total', 0)} trades={summary.get('trade_count', 0)} "
        f"merges={summary.get('merge_count', 0)} conditions={summary.get('unique_conditions', 0)}"
    )
    print("=" * 90)
    print(
        f"Cadence/min window={summary.get('events_per_minute_window', 0):.3f} "
        f"active={summary.get('events_per_minute_active', 0):.3f} "
        f"max/min={summary.get('max_events_per_minute', 0)}"
    )
    print(
        f"Buy share={summary.get('buy_share_of_trades', 0)*100:.1f}% "
        f"multi-outcome ratio={summary.get('multi_outcome_ratio', 0)*100:.1f}% "
        f"median trade=${summary.get('trade_usdc_median', 0):,.2f}"
    )
    print(f"Top leagues: {dist.get('league_prefix_top', {})}")
    print(f"Market types: {dist.get('market_type_counts', {})}")

    conditions = report.get("conditions", [])
    if conditions:
        print("\nTop conditions (locked_pnl_est):")
        for idx, item in enumerate(conditions[:10], start=1):
            print(
                f"{idx:>2}. {item.get('locked_pnl_est', 0):>10.2f} | trades={item.get('trade_count', 0):>4} "
                f"merge={item.get('merge_count', 0):>3} | pair_cost={item.get('pair_cost_est')} | "
                f"{str(item.get('title') or '')[:80]}"
            )

    now_tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base_dir = Path("reports/rn1")
    default_json = base_dir / f"rn1_deep_report_{now_tag}.json"
    default_tx_csv = base_dir / f"rn1_transactions_{now_tag}.csv"
    default_cond_csv = base_dir / f"rn1_conditions_{now_tag}.csv"

    json_path = Path(args.json_out) if args.json_out else (default_json if args.write_default_files else None)
    tx_csv_path = (
        Path(args.transactions_csv)
        if args.transactions_csv
        else (default_tx_csv if args.write_default_files and args.include_transactions else None)
    )
    cond_csv_path = (
        Path(args.conditions_csv)
        if args.conditions_csv
        else (default_cond_csv if args.write_default_files else None)
    )

    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nJSON saved: {json_path}")
    if cond_csv_path is not None:
        _write_csv(cond_csv_path, conditions)
        print(f"Conditions CSV saved: {cond_csv_path}")
    if tx_csv_path is not None:
        _write_csv(tx_csv_path, report.get("transactions", []))
        print(f"Transactions CSV saved: {tx_csv_path}")


if __name__ == "__main__":
    main()

