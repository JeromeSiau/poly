#!/usr/bin/env python3
"""Train XGBoost exit model for TD maker positions.

Decides whether to hold or exit an open position based on current market
state, position history (bid_max, drawdown), and time remaining.

Uses the same slot_snapshots + slot_resolutions data as the entry model
but looks at ALL snapshots after a simulated fill point.

Usage:
    ./run scripts/train_td_exit_model.py --db-url mysql+aiomysql://...
    ./run scripts/train_td_exit_model.py  # uses DATABASE_URL from .env

Output: data/models/td_exit_model_YYYYMMDD.joblib
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime, timezone

import structlog
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

from src.ml.data import load_all_snapshots, temporal_split
from src.ml.evaluate import evaluate
from src.ml.features import (
    EXIT_FEATURE_COLS,
    compute_exit_features,
    compute_trend_features,
    engineer_features,
)

logger = structlog.get_logger()


def train_model(X_train, y_train, X_val, y_val) -> CalibratedClassifierCV:
    """Train XGBoost + isotonic calibration for exit decisions."""
    import pandas as pd

    X_all = pd.concat([X_train, X_val], ignore_index=True)
    y_all = pd.concat([y_train, y_val], ignore_index=True)

    base = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric="logloss",
        random_state=42,
    )

    calibrated = CalibratedClassifierCV(
        estimator=base,
        method="isotonic",
        cv=3,
    )
    calibrated.fit(X_all, y_all)

    return calibrated


def save_model(model: CalibratedClassifierCV, output_dir: str) -> str:
    """Save exit model + feature list to joblib."""
    import joblib

    os.makedirs(output_dir, exist_ok=True)
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = os.path.join(output_dir, f"td_exit_model_{date_str}.joblib")

    payload = {
        "model": model,
        "feature_cols": EXIT_FEATURE_COLS,
        "model_type": "exit",
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    joblib.dump(payload, path)
    logger.info("exit_model_saved", path=path)
    return path


async def run(args: argparse.Namespace) -> None:
    # Load all snapshots
    df = await load_all_snapshots(args.db_url)
    if df.empty:
        print("No data available.")
        sys.exit(1)

    n_slots = df.groupby(["symbol", "slot_ts"]).ngroups
    print(f"Raw data: {len(df)} snapshots across {n_slots} slots")

    # Compute trend features first (needs all snapshots).
    df = compute_trend_features(df)

    # Engineer base features.
    df = engineer_features(df)

    # Compute exit-specific features (simulates fill + running bid_max).
    df = compute_exit_features(df, entry_price=args.entry_price)
    if df.empty or len(df) < 100:
        print(f"Only {len(df)} post-fill snapshots — need at least 100. Collect more data.")
        sys.exit(1)

    n_slots_with_fill = df.groupby(["symbol", "slot_ts"]).ngroups
    print(f"Post-fill snapshots: {len(df)} across {n_slots_with_fill} slots "
          f"(entry >= {args.entry_price})")

    # Drop NaN rows.
    before = len(df)
    df = df.dropna(subset=EXIT_FEATURE_COLS)
    if len(df) < before:
        logger.info("dropped_nan", dropped=before - len(df))

    # Target
    y = df["resolved_up"].astype(int)
    X = df[EXIT_FEATURE_COLS]

    print(f"\nDataset: {len(df)} rows, {y.mean():.1%} resolved Up")
    print(f"  bid_drop stats: mean={df['bid_drop'].mean():.4f}, "
          f"max={df['bid_drop'].max():.4f}, "
          f"pct>0.05={(df['bid_drop'] > 0.05).mean():.1%}")

    # Temporal split
    train_df, val_df = temporal_split(df, val_days=args.val_days)
    if len(val_df) < 20:
        print("Validation set too small — need more data.")
        sys.exit(1)

    X_train, y_train = train_df[EXIT_FEATURE_COLS], train_df["resolved_up"].astype(int)
    X_val, y_val = val_df[EXIT_FEATURE_COLS], val_df["resolved_up"].astype(int)

    # Train
    model = train_model(X_train, y_train, X_val, y_val)

    # Evaluate
    evaluate(model, X_train, y_train, "Train")
    metrics = evaluate(model, X_val, y_val, "Validation")

    # Feature importance
    base_model = model.calibrated_classifiers_[0].estimator
    if hasattr(base_model, "feature_importances_"):
        print("\n=== Feature Importance (gain) ===")
        importances = sorted(
            zip(EXIT_FEATURE_COLS, base_model.feature_importances_),
            key=lambda x: x[1], reverse=True,
        )
        for name, imp in importances:
            print(f"  {name:25s} {imp:.4f}")

    # Save
    path = save_model(model, args.output_dir)
    print(f"\nExit model saved to: {path}")
    print(f"Val AUC={metrics['auc']:.4f}, Brier={metrics['brier']:.4f}")


def build_parser() -> argparse.ArgumentParser:
    from config.settings import settings
    p = argparse.ArgumentParser(description="Train TD maker EXIT model")
    p.add_argument("--db-url", type=str,
                   default=settings.DATABASE_URL,
                   help="Database connection string")
    p.add_argument("--entry-price", type=float, default=0.75,
                   help="Simulated entry price for fill detection")
    p.add_argument("--val-days", type=int, default=3,
                   help="Days of data held out for validation")
    p.add_argument("--output-dir", type=str, default="data/models",
                   help="Directory to save model")
    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(run(args))
