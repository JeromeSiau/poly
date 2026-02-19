#!/usr/bin/env python3
"""Backtest ML-driven TD maker strategy on local slot data.

Simulates:
  - Entry:    ML entry model decides whether to enter each slot
  - Sizing:   order size scaled by model confidence
  - Exit:     ML exit model monitors open positions, triggers early sell
  - Baseline: enter every slot with bid_up in [target_bid, max_bid], hold to resolution

Usage:
    ./run scripts/backtest_ml_strategy.py
    ./run scripts/backtest_ml_strategy.py --entry-threshold 0.60
    ./run scripts/backtest_ml_strategy.py --exit-threshold 0.30
    ./run scripts/backtest_ml_strategy.py --no-exit-model
    ./run scripts/backtest_ml_strategy.py --no-sizing
    ./run scripts/backtest_ml_strategy.py --symbol BTC
"""

from __future__ import annotations

import argparse
import asyncio
import sys

import pandas as pd

from src.ml.data import load_all_snapshots, temporal_split
from src.ml.features import (
    EXIT_FEATURE_COLS,
    FEATURE_COLS,
    compute_trend_features,
    engineer_features,
)
from src.ml.model import find_latest_model, load_model


def _size_scale(p_win: float) -> float:
    """Match run_crypto_td_maker.py: scale ∈ [0.2, 2.0]."""
    return max(0.2, min(2.0, (p_win - 0.5) / 0.25))


def simulate_slot(
    slot_df: pd.DataFrame,
    entry_model,
    exit_model,
    entry_features: list[str],
    exit_features: list[str],
    entry_threshold: float,
    exit_threshold: float,
    target_bid: float,
    max_bid: float,
    base_size: float,
    use_sizing: bool,
    use_exit_model: bool,
) -> dict | None:
    """Simulate one slot. Returns result dict or None if no entry."""
    slot_df = slot_df.sort_values("captured_at")

    # Entry snapshot: latest in [4, 10] min window
    window = slot_df[
        (slot_df["minutes_into_slot"] >= 4.0) &
        (slot_df["minutes_into_slot"] <= 10.0)
    ]
    if window.empty:
        return None

    entry_row = window.iloc[-1]
    bid = entry_row["bid_up"]

    if not (target_bid <= bid <= max_bid):
        return None

    # Entry model
    X_entry = entry_row[entry_features].to_frame().T.astype(float)
    if X_entry.isnull().any().any():
        return None

    p_win = float(entry_model.predict_proba(X_entry)[0, 1])
    if p_win < entry_threshold:
        return None

    # Sizing
    entry_price = bid
    size_usd = base_size * (_size_scale(p_win) if use_sizing else 1.0)
    shares = size_usd / entry_price
    resolved_up = bool(entry_row["resolved_up"])

    # Post-entry snapshots
    post_entry = slot_df[slot_df["captured_at"] > entry_row["captured_at"]].copy()

    # Exit model simulation
    exit_price = None
    bid_max = entry_price
    fill_minutes = float(entry_row["minutes_into_slot"])

    if use_exit_model and exit_model is not None and not post_entry.empty:
        for _, snap in post_entry.iterrows():
            current_bid = float(snap["bid_up"])
            bid_max = max(bid_max, current_bid)

            snap_dict = snap.to_dict()
            snap_dict["entry_price"] = entry_price
            snap_dict["bid_max"] = bid_max
            snap_dict["bid_drop"] = bid_max - current_bid
            snap_dict["bid_drop_pct"] = (bid_max - current_bid) / max(bid_max, 0.01)
            snap_dict["minutes_remaining"] = max(15.0 - snap["minutes_into_slot"], 0.0)
            snap_dict["minutes_held"] = float(snap["minutes_into_slot"]) - fill_minutes
            snap_dict["pnl_unrealized"] = current_bid - entry_price

            X_exit = pd.DataFrame([snap_dict])[exit_features].astype(float)
            if X_exit.isnull().any().any():
                continue

            p_exit = float(exit_model.predict_proba(X_exit)[0, 1])
            if p_exit < exit_threshold:
                exit_price = current_bid
                break

    # PnL
    if exit_price is not None:
        pnl = shares * (exit_price - entry_price)
        outcome = "stopped"
    elif resolved_up:
        pnl = shares * (1.0 - entry_price)
        outcome = "won"
    else:
        pnl = -size_usd
        outcome = "lost"

    return {
        "symbol": entry_row["symbol"],
        "slot_ts": int(entry_row["slot_ts"]),
        "entry_price": round(entry_price, 3),
        "p_win": round(p_win, 3),
        "size_usd": round(size_usd, 2),
        "resolved_up": resolved_up,
        "outcome": outcome,
        "pnl": round(pnl, 4),
        "exit_price": round(exit_price, 3) if exit_price is not None else None,
    }


def print_summary(results: list[dict], label: str) -> None:
    if not results:
        print(f"\n  {label}: no trades")
        return

    df = pd.DataFrame(results)
    n = len(df)
    wins = (df["outcome"] == "won").sum() if "outcome" in df else df["won"].sum()
    stopped = (df["outcome"] == "stopped").sum() if "outcome" in df else 0
    lost = (df["outcome"] == "lost").sum() if "outcome" in df else (n - wins)
    total_pnl = df["pnl"].sum()
    total_size = df["size_usd"].sum()
    roi = total_pnl / total_size if total_size > 0 else 0.0
    winrate = wins / n

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Trades:      {n:,}")
    if stopped:
        print(f"  Outcomes:    won={wins} ({wins/n:.1%})  "
              f"stopped={stopped} ({stopped/n:.1%})  "
              f"lost={lost} ({lost/n:.1%})")
    else:
        print(f"  Win rate:    {winrate:.1%} ({wins}/{n})")
    print(f"  Total PnL:   ${total_pnl:+.2f}")
    print(f"  ROI:         {roi:+.2%}  (on ${total_size:.0f} deployed)")

    if "p_win" in df.columns:
        print(f"  Avg p_win:   {df['p_win'].mean():.1%}")
    if "size_usd" in df.columns:
        print(f"  Avg size:    ${df['size_usd'].mean():.2f}")

    print(f"\n  Per symbol:")
    for sym, g in df.groupby("symbol"):
        g_wins = (g["outcome"] == "won").sum() if "outcome" in g else g["won"].sum()
        g_pnl = g["pnl"].sum()
        g_n = len(g)
        print(f"    {sym:4s}  n={g_n:4d}  win={g_wins/g_n:.1%}  pnl=${g_pnl:+.2f}")


async def run(args: argparse.Namespace) -> None:
    # --- Load models ----------------------------------------------------------
    entry_path = args.model or find_latest_model(prefix="td_model")
    entry_payload = load_model(entry_path)
    entry_model = entry_payload["model"]
    entry_features = entry_payload["feature_cols"]

    exit_model = None
    exit_features: list[str] = EXIT_FEATURE_COLS
    exit_path = "(disabled)"
    if not args.no_exit_model:
        exit_path = args.exit_model or find_latest_model(prefix="td_exit_model")
        exit_payload = load_model(exit_path)
        exit_model = exit_payload["model"]
        exit_features = exit_payload["feature_cols"]

    print("=" * 60)
    print("  ML Strategy Backtest")
    print("=" * 60)
    print(f"  Entry model:     {entry_path}")
    print(f"  Exit model:      {exit_path}")
    print(f"  Entry threshold: p_win >= {args.entry_threshold}")
    print(f"  Exit threshold:  p_win < {args.exit_threshold}")
    print(f"  Bid range:       [{args.target_bid}, {args.max_bid}]")
    print(f"  Base size:       ${args.base_size}")
    print(f"  Sizing:          {'model-scaled' if not args.no_sizing else 'flat'}")
    print(f"  Val days:        {args.val_days}")

    # --- Load data ------------------------------------------------------------
    df = await load_all_snapshots(args.db_url)
    if df.empty:
        print("No data available.")
        sys.exit(1)

    df = compute_trend_features(df)
    df = engineer_features(df)

    # Temporal split: use val set only (unseen during training)
    entry_window = df[
        (df["minutes_into_slot"] >= 4.0) & (df["minutes_into_slot"] <= 10.0)
    ]
    entry_dedup = (
        entry_window.sort_values("minutes_into_slot", ascending=False)
        .groupby(["symbol", "slot_ts"])
        .first()
        .reset_index()
    )
    _, test_entry = temporal_split(entry_dedup, val_days=args.val_days)
    test_keys = set(zip(test_entry["symbol"], test_entry["slot_ts"]))

    # Optional symbol filter
    if args.symbol:
        test_keys = {(s, t) for s, t in test_keys
                     if args.symbol.upper() in s.upper()}

    print(f"\n  Test slots available: {len(test_keys):,}")

    # --- Simulate -------------------------------------------------------------
    ml_results: list[dict] = []
    baseline_results: list[dict] = []

    for (sym, slot_ts), slot_df in df.groupby(["symbol", "slot_ts"]):
        if (sym, slot_ts) not in test_keys:
            continue

        # ML strategy
        result = simulate_slot(
            slot_df,
            entry_model, exit_model,
            entry_features, exit_features,
            args.entry_threshold, args.exit_threshold,
            args.target_bid, args.max_bid,
            args.base_size,
            not args.no_sizing,
            not args.no_exit_model,
        )
        if result:
            ml_results.append(result)

        # Baseline: enter everything in range, no model, flat size, hold to end
        bw = slot_df[
            (slot_df["minutes_into_slot"] >= 4.0) &
            (slot_df["minutes_into_slot"] <= 10.0)
        ]
        if not bw.empty:
            brow = bw.sort_values("minutes_into_slot").iloc[-1]
            b_bid = float(brow["bid_up"])
            if args.target_bid <= b_bid <= args.max_bid:
                b_won = bool(brow["resolved_up"])
                b_shares = args.base_size / b_bid
                b_pnl = b_shares * (1.0 - b_bid) if b_won else -args.base_size
                baseline_results.append({
                    "symbol": sym,
                    "entry_price": b_bid,
                    "size_usd": args.base_size,
                    "won": b_won,
                    "outcome": "won" if b_won else "lost",
                    "pnl": round(b_pnl, 4),
                })

    # --- Results --------------------------------------------------------------
    n_tested = len(test_keys)
    n_entered = len(ml_results)
    n_baseline = len(baseline_results)
    print(f"  ML entered:  {n_entered:,} / {n_tested:,} slots "
          f"({n_entered/n_tested:.1%} selectivity)")
    print(f"  Baseline:    {n_baseline:,} / {n_tested:,} slots in range")

    print_summary(ml_results, "ML Strategy (entry + sizing + exit)")
    print_summary(baseline_results, "Baseline (no model, flat size, hold to end)")

    # Delta
    if ml_results and baseline_results:
        ml_roi = sum(r["pnl"] for r in ml_results) / sum(r["size_usd"] for r in ml_results)
        bl_roi = sum(r["pnl"] for r in baseline_results) / sum(r["size_usd"] for r in baseline_results)
        print(f"\n  ML ROI advantage: {(ml_roi - bl_roi):+.2%} over baseline")


def build_parser() -> argparse.ArgumentParser:
    from config.settings import settings
    p = argparse.ArgumentParser(description="Backtest ML-driven TD maker strategy")
    p.add_argument("--model", type=str, default=None,
                   help="Entry model path (default: latest td_model_*.joblib)")
    p.add_argument("--exit-model", type=str, default=None,
                   help="Exit model path (default: latest td_exit_model_*.joblib)")
    p.add_argument("--entry-threshold", type=float, default=0.55,
                   help="Min p_win to enter (default: 0.55)")
    p.add_argument("--exit-threshold", type=float, default=0.35,
                   help="Exit when p_win drops below this (default: 0.35)")
    p.add_argument("--target-bid", type=float, default=0.75,
                   help="Min bid to consider (default: 0.75)")
    p.add_argument("--max-bid", type=float, default=0.85,
                   help="Max bid to consider (default: 0.85)")
    p.add_argument("--base-size", type=float, default=10.0,
                   help="Base order size in USD (default: 10)")
    p.add_argument("--no-sizing", action="store_true",
                   help="Flat sizing (ignore model confidence for size)")
    p.add_argument("--no-exit-model", action="store_true",
                   help="Disable exit model (hold all positions to resolution)")
    p.add_argument("--symbol", type=str, default=None,
                   help="Filter by symbol (e.g. BTC, ETH)")
    p.add_argument("--val-days", type=int, default=3,
                   help="Test window in days (must match training, default: 3)")
    p.add_argument("--db-url", type=str, default=settings.DATABASE_URL)
    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(run(args))
