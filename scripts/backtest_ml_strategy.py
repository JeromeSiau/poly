#!/usr/bin/env python3
"""Backtest ML-driven TD maker strategy on local slot data.

Simulates:
  - Entry:    ML entry model decides whether to enter each slot
  - Sizing:   order size scaled by model confidence
  - Exit:     ML exit model monitors open positions, triggers early sell
  - Baseline: enter every slot with bid_up in [target_bid, max_bid], hold to resolution

Entry modes (four options):
  Default:       classifier, enter when p_win >= entry_threshold (e.g. 0.55)
  --edge-mode:   classifier, enter when p_win - bid_up >= entry_threshold (e.g. 0.03)
  --edge-model:  regressor,  enter when predicted_edge >= entry_threshold (e.g. 0.03)
  --dynamic:     ML decides timing + price. Scans all snapshots from minute 1,
                 enters at first where p_win - bid_up >= min_edge. No fixed window.

Usage:
    ./run scripts/backtest_ml_strategy.py
    ./run scripts/backtest_ml_strategy.py --edge-mode --entry-threshold 0.03
    ./run scripts/backtest_ml_strategy.py --edge-model data/models/td_edge_model_20260219.joblib --entry-threshold 0.03
    ./run scripts/backtest_ml_strategy.py --dynamic --min-edge 0.03
    ./run scripts/backtest_ml_strategy.py --dynamic --min-edge 0.05 --no-exit-model
    ./run scripts/backtest_ml_strategy.py --no-exit-model
    ./run scripts/backtest_ml_strategy.py --no-sizing
    ./run scripts/backtest_ml_strategy.py --symbol BTC
"""

from __future__ import annotations

import argparse
import asyncio
import random
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


def _edge_size_scale(edge: float) -> float:
    """Scale by edge magnitude: scale=1 at 5% edge, scale=2 at 10%."""
    return max(0.2, min(2.0, edge / 0.05))


def _decaying_edge(min_edge: float, minutes: float) -> float:
    """Require more edge early in the slot (patience), less edge later (urgency).

    Linear decay from 3× min_edge at minute 1 down to min_edge at minute 12.
    Example with min_edge=0.03:
      minute 1  → 0.09  (only enter on exceptional signals)
      minute 5  → 0.065
      minute 8  → 0.046
      minute 10 → 0.035
      minute 12 → 0.03  (last chance, accept base edge)
    """
    t = max(1.0, min(12.0, minutes))
    multiplier = 3.0 - (2.0 * (t - 1.0) / 11.0)  # 3.0 at min 1 → 1.0 at min 12
    return min_edge * multiplier


def _fill_probability(minutes: float, spread: float) -> float:
    """Estimate maker fill probability based on time and spread.

    Early in the slot the orderbook is thin → few takers → low fill prob.
    Tight spread = active market = more likely someone hits our bid.

    Returns probability ∈ [0.1, 0.95].
    """
    # Time factor: ramps from 0.15 at min 1 to 0.95 at min 12
    t = max(1.0, min(12.0, minutes))
    time_factor = 0.15 + (0.80 * (t - 1.0) / 11.0)

    # Spread factor: tight spread = active market
    if spread <= 0.03:
        spread_factor = 1.0
    elif spread <= 0.06:
        spread_factor = 0.8
    elif spread <= 0.10:
        spread_factor = 0.6
    else:
        spread_factor = 0.35

    return min(0.95, time_factor * spread_factor)


def _entry_signal(
    entry_row: pd.Series,
    entry_model,
    entry_features: list[str],
    bid: float,
    entry_threshold: float,
    edge_mode: bool,
    is_regressor: bool,
) -> tuple[float | None, float, float]:
    """Evaluate entry. Returns (None, 0, 0) to skip, or (score, p_win, edge).

    score   = the value compared to threshold (p_win, p_win-bid, or predicted_edge)
    p_win   = probability of win (approximate for regressor: bid + predicted_edge)
    edge    = model's edge over market (p_win - bid)
    """
    X = entry_row[entry_features].to_frame().T.astype(float)
    if X.isnull().any().any():
        return None, 0.0, 0.0

    if is_regressor:
        predicted_edge = float(entry_model.predict(X)[0])
        if predicted_edge < entry_threshold:
            return None, 0.0, 0.0
        p_win_approx = min(1.0, max(0.0, bid + predicted_edge))
        return predicted_edge, p_win_approx, predicted_edge

    p_win = float(entry_model.predict_proba(X)[0, 1])
    edge = p_win - bid
    score = edge if edge_mode else p_win
    if score < entry_threshold:
        return None, 0.0, 0.0
    return score, p_win, edge


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
    edge_mode: bool,
    is_regressor: bool,
    min_entry_minutes: float = 10.0,
    max_entry_minutes: float = 14.0,
) -> dict | None:
    """Simulate one slot. Returns result dict or None if no entry."""
    slot_df = slot_df.sort_values("captured_at")

    # Entry snapshot: latest in window (default: 10-14 min, matching live config)
    window = slot_df[
        (slot_df["minutes_into_slot"] >= min_entry_minutes) &
        (slot_df["minutes_into_slot"] <= max_entry_minutes)
    ]
    if window.empty:
        return None

    entry_row = window.iloc[-1]
    bid = float(entry_row["bid_up"])

    if not (target_bid <= bid <= max_bid):
        return None

    score, p_win, edge = _entry_signal(
        entry_row, entry_model, entry_features, bid,
        entry_threshold, edge_mode, is_regressor,
    )
    if score is None:
        return None

    # Sizing
    entry_price = bid
    if use_sizing:
        size_usd = base_size * (_edge_size_scale(edge) if (edge_mode or is_regressor)
                                else _size_scale(p_win))
    else:
        size_usd = base_size
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
        "edge": round(edge, 3),
        "score": round(score, 3),
        "size_usd": round(size_usd, 2),
        "resolved_up": resolved_up,
        "outcome": outcome,
        "pnl": round(pnl, 4),
        "exit_price": round(exit_price, 3) if exit_price is not None else None,
    }


def simulate_slot_dynamic(
    slot_df: pd.DataFrame,
    entry_model,
    exit_model,
    entry_features: list[str],
    exit_features: list[str],
    min_edge: float,
    exit_threshold: float,
    base_size: float,
    use_sizing: bool,
    use_exit_model: bool,
    is_regressor: bool,
    use_fill_model: bool = True,
    use_decay: bool = True,
) -> dict | None:
    """ML-driven slot: model decides timing + price.

    Scans every snapshot chronologically. At each one:
      - Computes p_win (or predicted_edge for regressor)
      - Required edge decays over time (patience early, urgency late)
      - Fill probability estimated from time + spread (maker realism)
      - Enters at first snapshot that passes both edge and fill checks
    """
    slot_df = slot_df.sort_values("captured_at")
    resolved_up = bool(slot_df.iloc[0]["resolved_up"])

    # Scan for entry
    entry_idx = None
    p_win = 0.0
    edge = 0.0

    for idx, row in slot_df.iterrows():
        mins = float(row["minutes_into_slot"])
        if mins < 1.0:
            continue
        if mins > 12.0:
            break

        bid = float(row["bid_up"])
        X = row[entry_features].to_frame().T.astype(float)
        if X.isnull().any().any():
            continue

        # Decaying threshold: require more edge early, less late
        required_edge = _decaying_edge(min_edge, mins) if use_decay else min_edge

        if is_regressor:
            pred_edge = float(entry_model.predict(X)[0])
            if pred_edge < required_edge:
                continue
            pw = min(1.0, max(0.0, bid + pred_edge))
            e = pred_edge
        else:
            pw = float(entry_model.predict_proba(X)[0, 1])
            e = pw - bid
            if e < required_edge:
                continue

        # Fill probability: would our maker order actually get filled?
        if use_fill_model:
            spread = float(row["spread_up"]) if "spread_up" in row.index else 0.05
            fill_prob = _fill_probability(mins, spread)
            if random.random() > fill_prob:
                continue  # Order placed but not filled, try next snapshot

        p_win = pw
        edge = e
        entry_idx = idx
        break

    if entry_idx is None:
        return None

    entry_row = slot_df.loc[entry_idx]
    entry_price = float(entry_row["bid_up"])
    fill_minutes = float(entry_row["minutes_into_slot"])

    # Sizing: Kelly-scaled on edge
    if use_sizing:
        size_usd = base_size * _edge_size_scale(edge)
    else:
        size_usd = base_size
    shares = size_usd / entry_price

    # Post-entry: exit model
    post_entry = slot_df[slot_df["captured_at"] > entry_row["captured_at"]].copy()
    exit_price = None
    bid_max = entry_price

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
        "entry_min": round(fill_minutes, 1),
        "entry_price": round(entry_price, 3),
        "p_win": round(p_win, 3),
        "edge": round(edge, 3),
        "score": round(edge, 3),
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
    wins = (df["outcome"] == "won").sum()
    stopped = (df["outcome"] == "stopped").sum()
    lost = (df["outcome"] == "lost").sum()
    total_pnl = df["pnl"].sum()
    total_size = df["size_usd"].sum()
    roi = total_pnl / total_size if total_size > 0 else 0.0

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Trades:      {n:,}")
    if stopped:
        print(f"  Outcomes:    won={wins} ({wins/n:.1%})  "
              f"stopped={stopped} ({stopped/n:.1%})  "
              f"lost={lost} ({lost/n:.1%})")
    else:
        print(f"  Win rate:    {wins/n:.1%} ({wins}/{n})")
    print(f"  Total PnL:   ${total_pnl:+.2f}")
    print(f"  ROI:         {roi:+.2%}  (on ${total_size:.0f} deployed)")

    if "edge" in df.columns:
        print(f"  Avg edge:    {df['edge'].mean():+.1%}  "
              f"(correct: {df.loc[df['outcome']=='won','edge'].mean():+.1%}  "
              f"wrong: {df.loc[df['outcome']=='lost','edge'].mean():+.1%})")
    if "size_usd" in df.columns:
        print(f"  Avg size:    ${df['size_usd'].mean():.2f}")
    if "entry_min" in df.columns:
        print(f"  Avg entry:   minute {df['entry_min'].mean():.1f}  "
              f"(range {df['entry_min'].min():.0f}-{df['entry_min'].max():.0f})")

    print(f"\n  Per symbol:")
    for sym, g in df.groupby("symbol"):
        g_wins = (g["outcome"] == "won").sum()
        g_pnl = g["pnl"].sum()
        g_n = len(g)
        print(f"    {sym:4s}  n={g_n:4d}  win={g_wins/g_n:.1%}  pnl=${g_pnl:+.2f}")


def print_baseline(results: list[dict], label: str) -> None:
    if not results:
        print(f"\n  {label}: no trades")
        return
    df = pd.DataFrame(results)
    n = len(df)
    wins = df["won"].sum()
    total_pnl = df["pnl"].sum()
    total_size = df["size_usd"].sum()
    roi = total_pnl / total_size if total_size > 0 else 0.0
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Trades:      {n:,}")
    print(f"  Win rate:    {wins/n:.1%} ({wins}/{n})")
    print(f"  Total PnL:   ${total_pnl:+.2f}")
    print(f"  ROI:         {roi:+.2%}  (on ${total_size:.0f} deployed)")
    print(f"\n  Per symbol:")
    for sym, g in df.groupby("symbol"):
        g_pnl = g["pnl"].sum()
        g_n = len(g)
        g_wins = g["won"].sum()
        print(f"    {sym:4s}  n={g_n:4d}  win={g_wins/g_n:.1%}  pnl=${g_pnl:+.2f}")


async def run(args: argparse.Namespace) -> None:
    random.seed(42)  # Reproducible fill simulation
    dynamic = args.dynamic
    is_regressor = bool(args.edge_model)
    edge_mode = args.edge_mode or is_regressor or dynamic

    # --- Load entry model -----------------------------------------------------
    if is_regressor:
        entry_path = args.edge_model
        entry_payload = load_model(entry_path)
        entry_model = entry_payload["model"]
        entry_features = entry_payload["feature_cols"]
    else:
        entry_path = args.model or find_latest_model(prefix="td_model")
        entry_payload = load_model(entry_path)
        entry_model = entry_payload["model"]
        entry_features = entry_payload["feature_cols"]

    # --- Load exit model ------------------------------------------------------
    exit_model = None
    exit_features: list[str] = EXIT_FEATURE_COLS
    exit_path = "(disabled)"
    if not args.no_exit_model:
        exit_path = args.exit_model or find_latest_model(prefix="td_exit_model")
        exit_payload = load_model(exit_path)
        exit_model = exit_payload["model"]
        exit_features = exit_payload["feature_cols"]

    # --- Print config ---------------------------------------------------------
    if dynamic:
        decay_str = "decaying" if not args.no_decay else "flat"
        fill_str = "+fill_model" if not args.no_fill_model else ""
        mode_label = f"dynamic ({decay_str}{fill_str})" + ("-regressor" if is_regressor else "")
        threshold_label = (f"p_win - bid >= {args.min_edge}"
                           f" ({'×3→×1 decay' if not args.no_decay else 'flat'},"
                           f" {'fill prob simulated' if not args.no_fill_model else 'instant fill'})")
    else:
        mode_label = ("edge-regressor" if is_regressor
                      else "edge-classifier" if edge_mode
                      else "classifier")
        threshold_label = (f"predicted_edge >= {args.entry_threshold}" if is_regressor
                           else f"p_win - bid >= {args.entry_threshold}" if edge_mode
                           else f"p_win >= {args.entry_threshold}")

    print("=" * 60)
    print("  ML Strategy Backtest")
    print("=" * 60)
    print(f"  Entry model:     {entry_path}  [{mode_label}]")
    print(f"  Exit model:      {exit_path}")
    print(f"  Entry criterion: {threshold_label}")
    print(f"  Exit threshold:  p_win < {args.exit_threshold}")
    if not dynamic:
        print(f"  Bid range:       [{args.target_bid}, {args.max_bid}]")
        print(f"  Entry window:    [{args.min_entry_minutes}, {args.max_entry_minutes}] min")
    print(f"  Base size:       ${args.base_size}")
    print(f"  Sizing:          {'edge-scaled' if (edge_mode and not args.no_sizing) else 'model-scaled' if not args.no_sizing else 'flat'}")
    print(f"  Slots:           {'ALL (in-sample warning)' if args.all_slots else f'val set ({args.val_days} days)'}")

    # --- Load data ------------------------------------------------------------
    df = await load_all_snapshots(args.db_url)
    if df.empty:
        print("No data available.")
        sys.exit(1)

    df = compute_trend_features(df)
    df = engineer_features(df)

    # --- Temporal split -------------------------------------------------------
    # For dynamic mode, use ALL snapshots for test key selection (no window filter)
    if dynamic:
        dedup = (
            df.sort_values("minutes_into_slot", ascending=False)
            .groupby(["symbol", "slot_ts"])
            .first()
            .reset_index()
        )
    else:
        entry_window = df[
            (df["minutes_into_slot"] >= args.min_entry_minutes) &
            (df["minutes_into_slot"] <= args.max_entry_minutes)
        ]
        dedup = (
            entry_window.sort_values("minutes_into_slot", ascending=False)
            .groupby(["symbol", "slot_ts"])
            .first()
            .reset_index()
        )

    if args.all_slots:
        test_keys = set(zip(dedup["symbol"], dedup["slot_ts"]))
    else:
        _, test_split = temporal_split(dedup, val_days=args.val_days)
        test_keys = set(zip(test_split["symbol"], test_split["slot_ts"]))

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
        if dynamic:
            result = simulate_slot_dynamic(
                slot_df,
                entry_model, exit_model,
                entry_features, exit_features,
                args.min_edge, args.exit_threshold,
                args.base_size,
                not args.no_sizing,
                not args.no_exit_model,
                is_regressor,
                use_fill_model=not args.no_fill_model,
                use_decay=not args.no_decay,
            )
        else:
            result = simulate_slot(
                slot_df,
                entry_model, exit_model,
                entry_features, exit_features,
                args.entry_threshold, args.exit_threshold,
                args.target_bid, args.max_bid,
                args.base_size,
                not args.no_sizing,
                not args.no_exit_model,
                edge_mode, is_regressor,
                args.min_entry_minutes, args.max_entry_minutes,
            )
        if result:
            ml_results.append(result)

        # Baseline: enter everything in range, flat size, hold to resolution
        bw = slot_df[
            (slot_df["minutes_into_slot"] >= args.min_entry_minutes) &
            (slot_df["minutes_into_slot"] <= args.max_entry_minutes)
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
                    "pnl": round(b_pnl, 4),
                })

    # --- Results --------------------------------------------------------------
    n_tested = len(test_keys)
    n_entered = len(ml_results)
    n_baseline = len(baseline_results)
    print(f"  ML entered:  {n_entered:,} / {n_tested:,} slots "
          f"({n_entered/n_tested:.1%} selectivity)")
    print(f"  Baseline:    {n_baseline:,} / {n_tested:,} slots in range")

    print_summary(ml_results, f"ML Strategy [{mode_label}]")
    print_baseline(baseline_results, "Baseline (no model, flat size, hold to end)")

    if ml_results and baseline_results:
        ml_roi = sum(r["pnl"] for r in ml_results) / sum(r["size_usd"] for r in ml_results)
        bl_roi = sum(r["pnl"] for r in baseline_results) / sum(r["size_usd"] for r in baseline_results)
        print(f"\n  ML ROI advantage: {(ml_roi - bl_roi):+.2%} over baseline")


def build_parser() -> argparse.ArgumentParser:
    from config.settings import settings
    p = argparse.ArgumentParser(description="Backtest ML-driven TD maker strategy")
    # Model selection
    p.add_argument("--model", type=str, default=None,
                   help="Classifier entry model (default: latest td_model_*.joblib)")
    p.add_argument("--edge-model", type=str, default=None,
                   help="Regressor edge model (td_edge_model_*.joblib). Overrides --model.")
    p.add_argument("--exit-model", type=str, default=None,
                   help="Exit model path (default: latest td_exit_model_*.joblib)")
    # Entry modes (mutually exclusive concept, not enforced)
    p.add_argument("--edge-mode", action="store_true",
                   help="Use p_win - bid_up >= threshold instead of p_win >= threshold")
    p.add_argument("--dynamic", action="store_true",
                   help="ML decides timing + price: scan all snapshots, enter at first edge signal")
    p.add_argument("--min-edge", type=float, default=0.03,
                   help="Minimum edge for --dynamic mode (default: 0.03)")
    p.add_argument("--no-fill-model", action="store_true",
                   help="Disable fill probability simulation (assume instant fill)")
    p.add_argument("--no-decay", action="store_true",
                   help="Disable decaying edge threshold (use flat min_edge at all times)")
    # Thresholds
    p.add_argument("--entry-threshold", type=float, default=0.55,
                   help="Entry threshold for non-dynamic modes (p_win, edge, or predicted_edge)")
    p.add_argument("--exit-threshold", type=float, default=0.35,
                   help="Exit when p_win drops below this (default: 0.35)")
    # Market params (defaults match live: MIN_ENTRY_MINUTES=10, no max)
    p.add_argument("--target-bid", type=float, default=0.75)
    p.add_argument("--max-bid", type=float, default=0.85)
    p.add_argument("--min-entry-minutes", type=float, default=10.0,
                   help="Earliest entry minute (default: 10, matching live)")
    p.add_argument("--max-entry-minutes", type=float, default=14.0,
                   help="Latest entry minute (default: 14)")
    p.add_argument("--base-size", type=float, default=10.0)
    p.add_argument("--no-sizing", action="store_true")
    p.add_argument("--no-exit-model", action="store_true")
    p.add_argument("--symbol", type=str, default=None)
    p.add_argument("--val-days", type=int, default=3)
    p.add_argument("--all-slots", action="store_true",
                   help="Test on ALL slots (includes training set — results will be optimistic)")
    p.add_argument("--db-url", type=str, default=settings.DATABASE_URL)
    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(run(args))
