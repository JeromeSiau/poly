#!/usr/bin/env python3
"""Test a trained TD model on random slot scenarios.

Loads the most recent td_model joblib, pulls test data from MySQL
(same pipeline as train_td_model.py), picks N random resolved slots,
and displays the model's prediction vs. market vs. actual outcome.

Usage:
    ./run scripts/test_slot_scenarios.py
    ./run scripts/test_slot_scenarios.py --n 20 --symbol BTC
    ./run scripts/test_slot_scenarios.py --model data/models/td_model_20260215.joblib
    ./run scripts/test_slot_scenarios.py --interactive
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Import feature engineering from training script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_td_model import (
    FEATURE_COLS,
    compute_trend_features,
    deduplicate_per_slot,
    engineer_features,
    load_data,
    temporal_split,
)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def find_latest_model(model_dir: str = "data/models") -> str:
    """Return path to the most recent td_model_*.joblib file."""
    pattern = str(Path(model_dir) / "td_model_*.joblib")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No model files found matching {pattern}")
        sys.exit(1)
    return files[-1]


def load_model(path: str) -> dict:
    """Load model payload from joblib."""
    payload = joblib.load(path)
    required_keys = {"model", "feature_cols"}
    if not required_keys.issubset(payload.keys()):
        print(f"Model file missing keys: {required_keys - payload.keys()}")
        sys.exit(1)
    return payload


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
def fmt_pct(val: float, signed: bool = False) -> str:
    """Format a float as percentage string."""
    if signed:
        return f"{val:+.1%}"
    return f"{val:.1%}"


def fmt_float(val: float, decimals: int = 3) -> str:
    return f"{val:.{decimals}f}"


def print_scenario(idx: int, total: int, row: pd.Series,
                   p_up: float, model_correct: bool) -> None:
    """Print a single scenario block."""
    # Header
    slot_dt = datetime.fromtimestamp(row["slot_ts"], tz=timezone.utc)
    slot_str = slot_dt.strftime("%Y-%m-%d %H:%M UTC")
    resolved_up = bool(row["resolved_up"])
    result_str = "UP" if resolved_up else "DOWN"

    print(f"\n{'':=<60}")
    print(f"  Scenario {idx}/{total}  "
          f"{row['symbol']} | {slot_str} | "
          f"{row['minutes_into_slot']:.1f} min into slot")
    print(f"{'':=<60}")

    # Key features table
    features = [
        ("bid_up",            fmt_float(row["bid_up"])),
        ("ask_up",            fmt_float(row["ask_up"])),
        ("dir_move_pct",      f"{row['dir_move_pct']:+.2f}%"),
        ("abs_move_pct",      f"{row['abs_move_pct']:.2f}%"),
        ("spread_up",         fmt_float(row["spread_up"])),
        ("minutes_into_slot", fmt_float(row["minutes_into_slot"], 1)),
        ("bid_imbalance",     f"{row['bid_imbalance']:+.3f}"),
        ("move_velocity",     fmt_float(row["move_velocity"])),
        ("book_pressure",     f"{row['book_pressure']:+.3f}"),
    ]

    print()
    print(f"  {'Feature':<22} {'Value':>10}")
    print(f"  {'-' * 22} {'-' * 10}")
    for name, val in features:
        print(f"  {name:<22} {val:>10}")

    # Model prediction vs market
    market_prob = row["bid_up"]
    edge = p_up - market_prob
    model_predicted_up = p_up >= 0.5

    print()
    print(f"  Model P(UP):          {fmt_pct(p_up)}")
    print(f"  Market bid_up:        {fmt_pct(market_prob)}")
    print(f"  Edge (model - mkt):   {fmt_pct(edge, signed=True)}")
    print(f"  Actual result:        {result_str}")

    correct_str = "YES" if model_correct else "NO"
    pred_dir = "UP" if model_predicted_up else "DOWN"
    print(f"  Model correct:        {correct_str}"
          f" (predicted {pred_dir}, was {result_str})")


def print_summary(results: list[dict]) -> None:
    """Print aggregate stats across all scenarios."""
    n = len(results)
    correct = sum(1 for r in results if r["correct"])
    accuracy = correct / n if n > 0 else 0.0

    correct_results = [r for r in results if r["correct"]]
    wrong_results = [r for r in results if not r["correct"]]

    avg_edge_correct = (np.mean([r["edge"] for r in correct_results])
                        if correct_results else 0.0)
    avg_edge_wrong = (np.mean([r["edge"] for r in wrong_results])
                      if wrong_results else 0.0)
    avg_conf_correct = (np.mean([r["confidence"] for r in correct_results])
                        if correct_results else 0.0)
    avg_conf_wrong = (np.mean([r["confidence"] for r in wrong_results])
                      if wrong_results else 0.0)

    print(f"\n{'=' * 60}")
    print(f"  Summary ({n} scenarios)")
    print(f"{'=' * 60}")
    print(f"  Model accuracy:          {accuracy:.1%} ({correct}/{n})")
    print(f"  Avg edge (correct):      {avg_edge_correct:+.1%}")
    print(f"  Avg edge (wrong):        {avg_edge_wrong:+.1%}")
    print(f"  Avg confidence (correct): {avg_conf_correct:.1%}")
    print(f"  Avg confidence (wrong):   {avg_conf_wrong:.1%}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def run(args: argparse.Namespace) -> None:
    # Resolve model path
    model_path = args.model or find_latest_model()
    payload = load_model(model_path)
    model = payload["model"]
    feature_cols = payload["feature_cols"]

    print("=" * 60)
    print("  Slot Scenario Tester")
    print("=" * 60)
    print(f"  Model:     {model_path}")
    if "trained_at" in payload:
        print(f"  Trained:   {payload['trained_at']}")

    # Load data
    df = await load_data(args.db_url)

    if df.empty:
        print("\nNo resolved slot data found. Check your database connection.")
        sys.exit(1)

    # Deduplicate: one snapshot per slot (latest in window)
    df = deduplicate_per_slot(df)

    # Feature engineering
    df = engineer_features(df)

    # Drop rows with NaN in feature columns
    df = df.dropna(subset=feature_cols)

    if len(df) < 10:
        print(f"\nOnly {len(df)} usable slots found. Need more data.")
        sys.exit(1)

    # Temporal split: match training's temporal_split
    _, test_df = temporal_split(df, val_days=args.val_days)

    # Optional symbol filter
    if args.symbol:
        test_df = test_df[
            test_df["symbol"].str.contains(args.symbol, case=False)
        ]
        if test_df.empty:
            print(f"\nNo test slots found for symbol '{args.symbol}'.")
            sys.exit(1)

    print(f"  Scenarios: {args.n}")
    print(f"  Test set:  {len(test_df):,} slots (last {args.val_days} days)")

    # Pick N random scenarios
    n = min(args.n, len(test_df))
    sample = test_df.sample(n=n, random_state=None)

    # Run predictions
    X_sample = sample[feature_cols]
    probas = model.predict_proba(X_sample)[:, 1]  # P(UP)

    results: list[dict] = []

    for i, ((_, row), p_up) in enumerate(zip(sample.iterrows(), probas), 1):
        resolved_up = bool(row["resolved_up"])
        model_predicted_up = p_up >= 0.5
        correct = model_predicted_up == resolved_up
        edge = p_up - row["bid_up"]
        confidence = p_up if model_predicted_up else (1.0 - p_up)

        results.append({
            "correct": correct,
            "edge": edge,
            "confidence": confidence,
            "p_up": p_up,
        })

        print_scenario(i, n, row, p_up, correct)

        if args.interactive and i < n:
            try:
                input("\n  Press Enter for next scenario...")
            except (EOFError, KeyboardInterrupt):
                print("\n\n  Interrupted. Showing summary for scenarios so far.")
                break

    # Summary
    print_summary(results)


def build_parser() -> argparse.ArgumentParser:
    from config.settings import settings
    p = argparse.ArgumentParser(
        description="Test TD model predictions on random slot scenarios")
    p.add_argument("--model", type=str, default=None,
                   help="Path to model joblib (default: most recent in data/models/)")
    p.add_argument("--n", type=int, default=10,
                   help="Number of scenarios to display (default: 10)")
    p.add_argument("--interactive", action="store_true",
                   help="Press Enter between scenarios")
    p.add_argument("--symbol", type=str, default=None,
                   help="Filter by symbol (e.g. BTC, ETH)")
    p.add_argument("--val-days", type=int, default=3,
                   help="Days of data held out for test set (must match training)")
    p.add_argument("--db-url", type=str,
                   default=settings.DATABASE_URL,
                   help="Database connection string")
    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(run(args))
