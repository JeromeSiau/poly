#!/usr/bin/env python3
"""Compare RN1 traded conditions with local traded conditions.

Usage:
    uv run python scripts/compare_rn1_conditions_vs_local.py --hours 6
    uv run python scripts/compare_rn1_conditions_vs_local.py --hours 24 --strategy-tag edge_1p5_0p3
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from src.analysis.rn1_comparison import DEFAULT_RN1_WALLET, build_rn1_vs_local_condition_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare RN1 conditions versus local condition universe.")
    parser.add_argument("--db", type=str, default="", help="Database URL (default: from settings).")
    parser.add_argument("--hours", type=float, default=6.0, help="Analysis window in hours.")
    parser.add_argument("--strategy-tag", type=str, default=None, help="Optional local strategy tag filter.")
    parser.add_argument("--rn1-wallet", type=str, default=DEFAULT_RN1_WALLET, help="Benchmark wallet.")
    parser.add_argument("--page-limit", type=int, default=500, help="Rows fetched per page from RN1 activity API.")
    parser.add_argument("--max-pages", type=int, default=7, help="Maximum pages fetched from RN1 activity API.")
    parser.add_argument("--top-conditions", type=int, default=100, help="Rows kept per top section.")
    parser.add_argument("--json-out", type=str, default=None, help="Optional JSON output path.")
    parser.add_argument("--overlap-csv", type=str, default=None, help="Optional overlap CSV output path.")
    parser.add_argument("--rn1-only-csv", type=str, default=None, help="Optional RN1-only CSV output path.")
    return parser


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = build_parser().parse_args()
    report = build_rn1_vs_local_condition_report(
        db_url=args.db,
        window_hours=args.hours,
        strategy_tag=args.strategy_tag,
        rn1_wallet=args.rn1_wallet,
        page_limit=args.page_limit,
        max_pages=args.max_pages,
        top_conditions=args.top_conditions,
    )

    summary = report.get("summary", {})
    print("=" * 90)
    print(
        f"RN1 vs local conditions | window={report.get('window_hours')}h "
        f"| overlap={summary.get('overlap_conditions', 0)}/{summary.get('rn1_conditions', 0)} "
        f"({summary.get('overlap_ratio_vs_rn1', 0)*100:.1f}%)"
    )
    print("=" * 90)
    print("Recommendations:")
    for idx, reco in enumerate(report.get("recommendations", []), start=1):
        print(f"{idx}. {reco}")

    print("\nTop RN1-only conditions (you are missing):")
    for idx, row in enumerate(report.get("rn1_only_top", [])[:15], start=1):
        print(
            f"{idx:>2}. buy_usdc=${float(row.get('rn1_buy_usdc') or 0):>10,.2f} "
            f"trades={int(row.get('rn1_trade_count') or 0):>4} "
            f"merge={int(row.get('rn1_merge_count') or 0):>3} "
            f"{str(row.get('title') or '')[:80]}"
        )

    print("\nTop overlap conditions:")
    for idx, row in enumerate(report.get("overlap_top", [])[:15], start=1):
        print(
            f"{idx:>2}. rn1_buy=${float(row.get('rn1_buy_usdc') or 0):>10,.2f} "
            f"local_buy=${float(row.get('local_buy_usdc') or 0):>8,.2f} "
            f"activity_ratio={float(row.get('activity_ratio_local_vs_rn1') or 0):>5.2f} "
            f"{str(row.get('title') or '')[:75]}"
        )

    if args.json_out:
        p = Path(args.json_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nJSON saved: {p}")
    if args.overlap_csv:
        p = Path(args.overlap_csv)
        _write_csv(p, report.get("overlap_top", []))
        print(f"Overlap CSV saved: {p}")
    if args.rn1_only_csv:
        p = Path(args.rn1_only_csv)
        _write_csv(p, report.get("rn1_only_top", []))
        print(f"RN1-only CSV saved: {p}")


if __name__ == "__main__":
    main()

