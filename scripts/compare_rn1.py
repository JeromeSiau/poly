#!/usr/bin/env python3
"""CLI report: compare local two-sided behavior vs RN1 public activity.

Usage:
    uv run python scripts/compare_rn1.py --hours 6
    uv run python scripts/compare_rn1.py --strategy-tag edge_1p5_0p3 --hours 24
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.analysis.rn1_comparison import DEFAULT_RN1_WALLET, build_comparison_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare local two-sided behavior against RN1.")
    parser.add_argument("--db", type=str, default="", help="Database URL (default: from settings).")
    parser.add_argument("--hours", type=float, default=6.0, help="Comparison window in hours.")
    parser.add_argument("--strategy-tag", type=str, default=None, help="Optional local strategy_tag filter.")
    parser.add_argument("--rn1-wallet", type=str, default=DEFAULT_RN1_WALLET, help="Benchmark wallet address.")
    parser.add_argument("--page-limit", type=int, default=500, help="Rows fetched per page for RN1 activity.")
    parser.add_argument("--max-pages", type=int, default=7, help="Maximum pages fetched for RN1 activity.")
    parser.add_argument("--json-out", type=str, default=None, help="Optional path to save JSON report.")
    return parser


def _fmt_pct(value: float) -> str:
    return f"{value * 100.0:.1f}%"


def _fmt_ratio(value: float) -> str:
    return f"{value:.2f}x"


def print_human_summary(report: dict) -> None:
    local = report.get("local", {})
    rn1 = report.get("rn1", {})
    gaps = report.get("gaps", {})

    print("=" * 84)
    print(
        f"RN1 comparison | window={report.get('window_hours')}h "
        f"| strategy_tag={report.get('filters', {}).get('strategy_tag') or 'ALL'}"
    )
    print("=" * 84)
    print(
        f"Cadence/min (window): local={local.get('cadence_per_minute_window', 0):.3f} "
        f"rn1={rn1.get('cadence_per_minute_window', 0):.3f} "
        f"ratio={_fmt_ratio(float(gaps.get('cadence_window_ratio_local_vs_rn1', 0)))}"
    )
    print(
        f"Cadence/min (active): local={local.get('cadence_per_minute_active', 0):.3f} "
        f"rn1={rn1.get('cadence_per_minute_active', 0):.3f} "
        f"ratio={_fmt_ratio(float(gaps.get('cadence_active_ratio_local_vs_rn1', 0)))}"
    )
    print(
        f"BUY share: local={_fmt_pct(float(local.get('buy_share_of_trades', 0)))} "
        f"rn1={_fmt_pct(float(rn1.get('buy_share_of_trades', 0)))} "
        f"gap={_fmt_pct(float(gaps.get('buy_share_gap', 0)))}"
    )
    print(
        f"Multi-outcome ratio: local={_fmt_pct(float(local.get('multi_outcome_ratio', 0)))} "
        f"rn1={_fmt_pct(float(rn1.get('multi_outcome_ratio', 0)))} "
        f"gap={_fmt_pct(float(gaps.get('multi_outcome_ratio_gap', 0)))}"
    )
    print(
        f"Median ticket: local=${float(local.get('size_median_usd', 0)):,.2f} "
        f"rn1=${float(rn1.get('size_median_usd', 0)):,.2f} "
        f"ratio={_fmt_ratio(float(gaps.get('size_median_ratio_local_vs_rn1', 0)))}"
    )
    print(
        f"Unique conditions: local={int(local.get('unique_conditions', 0))} "
        f"rn1={int(rn1.get('unique_conditions', 0))} "
        f"ratio={_fmt_ratio(float(gaps.get('unique_conditions_ratio_local_vs_rn1', 0)))}"
    )
    print("\nRecommendations:")
    for idx, reco in enumerate(report.get("recommendations", []), start=1):
        print(f"{idx}. {reco}")


def main() -> None:
    args = build_parser().parse_args()
    report = build_comparison_report(
        db_url=args.db,
        window_hours=args.hours,
        strategy_tag=args.strategy_tag,
        rn1_wallet=args.rn1_wallet,
        page_limit=args.page_limit,
        max_pages=args.max_pages,
    )
    print_human_summary(report)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved JSON: {out}")
    else:
        print("\nJSON:")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
