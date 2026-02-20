#!/usr/bin/env python3
"""Train XGBoost edge regression model for TD maker entry decisions.

Instead of predicting direction (resolved_up ∈ {0,1}), this model predicts
the RESIDUAL: how much the actual outcome deviates from the market price.

    target = resolved_up - bid_up

    Examples:
      bid_up=0.80, resolves UP   → target = +0.20  (market underpriced)
      bid_up=0.80, resolves DOWN → target = -0.80  (market overpriced)
      bid_up=0.75, resolves UP   → target = +0.25  (more edge than at 0.80)

A positive predicted_edge means the model thinks the market is underpriced.
Entry criterion: predicted_edge >= threshold (e.g. 0.03 = 3% edge over market).

This forces the model to learn signals that BEAT the market, not just predict
direction (which the market price already partially encodes).

Usage:
    ./run scripts/train_td_edge_model.py
    ./run scripts/train_td_edge_model.py --val-days 3
    ./run scripts/train_td_edge_model.py --db-url mysql+aiomysql://...

Output: data/models/td_edge_model_YYYYMMDD.joblib
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import structlog
from xgboost import XGBRegressor

from src.ml.data import deduplicate_per_slot, load_snapshots, temporal_split
from src.ml.features import FEATURE_COLS, engineer_features

logger = structlog.get_logger()


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> XGBRegressor:
    model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model: XGBRegressor, X: pd.DataFrame, y: pd.Series,
             bid_up: pd.Series, label: str) -> dict:
    y_pred = model.predict(X)

    mae = float(np.mean(np.abs(y_pred - y)))
    corr = float(np.corrcoef(y, y_pred)[0, 1]) if len(y) > 1 else 0.0

    # Directional accuracy: does the sign of predicted_edge match actual?
    dir_acc = float(np.mean(np.sign(y_pred) == np.sign(y)))

    # Precision at positive predicted edge: when model says "enter", is actual edge positive?
    pos_mask = y_pred > 0
    pos_precision = float(np.mean(y[pos_mask] > 0)) if pos_mask.sum() > 0 else 0.0
    n_pos = int(pos_mask.sum())

    # Baseline: always predict 0 (market is correctly priced)
    baseline_mae = float(np.mean(np.abs(y)))

    print(f"\n=== {label} (n={len(y):,}) ===")
    print(f"  MAE:              {mae:.4f}  (baseline: {baseline_mae:.4f})")
    print(f"  Correlation:      {corr:.4f}")
    print(f"  Directional acc:  {dir_acc:.1%}")
    print(f"  Positive entries: {n_pos} ({n_pos/len(y):.1%} of slots)")
    print(f"  Precision (pos):  {pos_precision:.1%}  "
          f"(when model predicts edge > 0, actual edge > 0 {pos_precision:.1%} of the time)")

    # Edge distribution by predicted bucket
    print(f"\n  Predicted edge buckets:")
    bins = [-1, -0.05, 0, 0.02, 0.05, 0.10, 1]
    labels_b = ["< -5%", "-5% to 0%", "0% to 2%", "2% to 5%", "5% to 10%", "> 10%"]
    for lo, hi, lbl in zip(bins[:-1], bins[1:], labels_b):
        mask = (y_pred > lo) & (y_pred <= hi)
        if mask.sum() == 0:
            continue
        actual_edge = float(np.mean(y[mask]))
        win_rate = float(np.mean((y[mask] + bid_up[mask]) >= 1.0))
        print(f"    pred {lbl:12s}: n={mask.sum():4d}  "
              f"actual_edge={actual_edge:+.3f}  win_rate={win_rate:.1%}")

    return {"mae": mae, "corr": corr, "dir_acc": dir_acc, "pos_precision": pos_precision}


def save_model(model: XGBRegressor, output_dir: str) -> str:
    import joblib

    os.makedirs(output_dir, exist_ok=True)
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = os.path.join(output_dir, f"td_edge_model_{date_str}.joblib")

    payload = {
        "model": model,
        "feature_cols": FEATURE_COLS,
        "model_type": "edge_regressor",
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    joblib.dump(payload, path)
    logger.info("edge_model_saved", path=path)
    return path


async def run(args: argparse.Namespace) -> None:
    df = await load_snapshots(args.db_url)
    if df.empty:
        print("No data available.")
        sys.exit(1)

    df = deduplicate_per_slot(df)
    df = engineer_features(df)
    df = df.dropna(subset=FEATURE_COLS)

    n_slots = len(df)
    print(f"Dataset: {n_slots} slots")
    print(f"Symbols: {sorted(df['symbol'].unique().tolist())}")

    # Target: residual over market price
    df["edge_target"] = df["resolved_up"].astype(float) - df["bid_up"]

    print(f"\nEdge target stats:")
    print(f"  Mean:   {df['edge_target'].mean():+.4f}")
    print(f"  Std:    {df['edge_target'].std():.4f}")
    print(f"  % positive (market underpriced): {(df['edge_target'] > 0).mean():.1%}")
    print(f"  % in [0.75, 0.85] bid range: "
          f"{((df['bid_up'] >= 0.75) & (df['bid_up'] <= 0.85)).mean():.1%}")

    train_df, val_df = temporal_split(df, val_days=args.val_days)
    if len(train_df) < 50:
        print(f"Training set too small ({len(train_df)} rows). Need more data.")
        sys.exit(1)
    if len(val_df) < 20:
        print(f"Validation set too small ({len(val_df)} rows). Need more data.")
        sys.exit(1)

    X_train = train_df[FEATURE_COLS]
    y_train = train_df["edge_target"]
    X_val = val_df[FEATURE_COLS]
    y_val = val_df["edge_target"]

    print(f"\nSplit: train={len(train_df)}, val={len(val_df)}")

    model = train_model(X_train, y_train)

    evaluate(model, X_train, y_train, train_df["bid_up"], "Train")
    val_metrics = evaluate(model, X_val, y_val, val_df["bid_up"], "Validation")

    # Feature importance
    if hasattr(model, "feature_importances_"):
        print("\n=== Feature Importance (gain) ===")
        importances = sorted(
            zip(FEATURE_COLS, model.feature_importances_),
            key=lambda x: x[1], reverse=True,
        )
        for name, imp in importances[:10]:
            print(f"  {name:25s} {imp:.4f}")

    path = save_model(model, args.output_dir)
    print(f"\nEdge model saved to: {path}")
    print(f"Val MAE={val_metrics['mae']:.4f}, "
          f"Corr={val_metrics['corr']:.4f}, "
          f"DirAcc={val_metrics['dir_acc']:.1%}, "
          f"PosPrecision={val_metrics['pos_precision']:.1%}")
    print(f"\nTo backtest with this model:")
    print(f"  ./run scripts/backtest_ml_strategy.py --edge-model {path} --entry-threshold 0.03")


def build_parser() -> argparse.ArgumentParser:
    from config.settings import settings
    p = argparse.ArgumentParser(description="Train TD maker edge regression model")
    p.add_argument("--db-url", type=str, default=settings.DATABASE_URL)
    p.add_argument("--val-days", type=int, default=3)
    p.add_argument("--output-dir", type=str, default="data/models")
    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(run(args))
