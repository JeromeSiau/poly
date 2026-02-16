#!/usr/bin/env python3
"""Train XGBoost model for TD maker entry decisions.

Queries MySQL slot data (snapshots + resolutions), engineers features,
trains a calibrated binary classifier, and saves to joblib.

Usage:
    ./run scripts/train_td_model.py --db-url mysql+aiomysql://...
    ./run scripts/train_td_model.py  # uses DATABASE_URL from .env

Output: data/models/td_model_YYYYMMDD.joblib
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

from src.ml.data import deduplicate_per_slot, load_snapshots, temporal_split
from src.ml.evaluate import evaluate
from src.ml.features import FEATURE_COLS, engineer_features

logger = structlog.get_logger()


def train_model(X_train, y_train, X_val, y_val) -> CalibratedClassifierCV:
    """Train XGBoost + isotonic calibration."""
    import pandas as pd

    X_all = pd.concat([X_train, X_val], ignore_index=True)
    y_all = pd.concat([y_train, y_val], ignore_index=True)

    base = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
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
    """Save model + feature list to joblib."""
    import joblib

    os.makedirs(output_dir, exist_ok=True)
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = os.path.join(output_dir, f"td_model_{date_str}.joblib")

    payload = {
        "model": model,
        "feature_cols": FEATURE_COLS,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    joblib.dump(payload, path)
    logger.info("model_saved", path=path)
    return path


async def run(args: argparse.Namespace) -> None:
    # Load
    df = await load_snapshots(args.db_url, args.min_minutes, args.max_minutes)
    if len(df) < 100:
        print(f"Only {len(df)} rows — need at least 100 resolved slots. Collect more data.")
        sys.exit(1)

    # Deduplicate: one snapshot per slot
    df = deduplicate_per_slot(df)
    logger.info("after_dedup", rows=len(df))

    # Feature engineering
    df = engineer_features(df)

    # Drop rows with NaN in features
    before = len(df)
    df = df.dropna(subset=FEATURE_COLS)
    if len(df) < before:
        logger.info("dropped_nan", dropped=before - len(df))

    # Target
    y = df["resolved_up"].astype(int)
    X = df[FEATURE_COLS]

    print(f"\nDataset: {len(df)} slots, {y.mean():.1%} resolved Up")
    print(f"Symbols: {sorted(df['symbol'].unique().tolist())}")

    # Temporal split
    train_df, val_df = temporal_split(df, val_days=args.val_days)
    if len(val_df) < 20:
        print("Validation set too small — need more data.")
        sys.exit(1)

    X_train, y_train = train_df[FEATURE_COLS], train_df["resolved_up"].astype(int)
    X_val, y_val = val_df[FEATURE_COLS], val_df["resolved_up"].astype(int)

    # Train
    model = train_model(X_train, y_train, X_val, y_val)

    # Evaluate
    evaluate(model, X_train, y_train, "Train")
    metrics = evaluate(model, X_val, y_val, "Validation")

    # Feature importance (from base estimator inside calibrated wrapper)
    base_model = model.calibrated_classifiers_[0].estimator
    if hasattr(base_model, "feature_importances_"):
        print("\n=== Feature Importance (gain) ===")
        importances = sorted(
            zip(FEATURE_COLS, base_model.feature_importances_),
            key=lambda x: x[1], reverse=True,
        )
        for name, imp in importances:
            print(f"  {name:25s} {imp:.4f}")

    # Save
    path = save_model(model, args.output_dir)
    print(f"\nModel saved to: {path}")
    print(f"Val AUC={metrics['auc']:.4f}, Brier={metrics['brier']:.4f}")


def build_parser() -> argparse.ArgumentParser:
    from config.settings import settings
    p = argparse.ArgumentParser(description="Train TD maker ML model")
    p.add_argument("--db-url", type=str,
                   default=settings.DATABASE_URL,
                   help="Database connection string")
    p.add_argument("--min-minutes", type=float, default=4.0,
                   help="Min minutes into slot for snapshot selection")
    p.add_argument("--max-minutes", type=float, default=10.0,
                   help="Max minutes into slot for snapshot selection")
    p.add_argument("--val-days", type=int, default=3,
                   help="Days of data held out for validation")
    p.add_argument("--output-dir", type=str, default="data/models",
                   help="Directory to save model")
    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(run(args))
