#!/usr/bin/env python3
"""Train XGBoost model for TD maker entry decisions.

Queries MySQL slot data (snapshots + resolutions), engineers features,
trains a calibrated binary classifier, and saves to joblib.

Usage:
    ./run scripts/train_td_model.py --mysql-url mysql+aiomysql://...
    ./run scripts/train_td_model.py  # uses MYSQL_URL from .env

Output: data/models/td_model_YYYYMMDD.joblib
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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    brier_score_loss,
    classification_report,
    log_loss,
    roc_auc_score,
)
from xgboost import XGBClassifier

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Feature columns expected by the model (order matters for inference)
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    # Book prices
    "bid_up", "ask_up", "bid_down", "ask_down",
    # Spreads
    "spread_up", "spread_down",
    # Book sizes
    "bid_size_up", "ask_size_up", "bid_size_down", "ask_size_down",
    # Chainlink move
    "dir_move_pct", "abs_move_pct",
    # Timing
    "minutes_into_slot",
    "hour_utc", "day_of_week",
    # Derived
    "spread_ratio",
    "bid_imbalance",
    "fav_bid",
    "move_velocity",
    "book_pressure",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features to raw snapshot data."""
    df = df.copy()

    # spread_ratio: how skewed are spreads between Up and Down
    df["spread_ratio"] = df["spread_up"] / df["spread_down"].clip(lower=0.001)

    # bid_imbalance: which side the market favours
    df["bid_imbalance"] = df["bid_up"] - df["bid_down"]

    # fav_bid: highest bid (the favourite side's price)
    df["fav_bid"] = df[["bid_up", "bid_down"]].max(axis=1)

    # move_velocity: how fast price moved per minute
    df["move_velocity"] = df["dir_move_pct"] / df["minutes_into_slot"].clip(lower=0.5)

    # book_pressure: relative size advantage of Up vs Down bids
    total_bid_sz = (df["bid_size_up"] + df["bid_size_down"]).clip(lower=0.01)
    df["book_pressure"] = (df["bid_size_up"] - df["bid_size_down"]) / total_bid_sz

    return df


async def load_data(mysql_url: str, min_minutes: float = 4.0,
                    max_minutes: float = 10.0) -> pd.DataFrame:
    """Load snapshots joined with resolutions from MySQL."""
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(mysql_url, echo=False)
    query = f"""
        SELECT
            s.symbol, s.slot_ts, s.minutes_into_slot,
            s.bid_up, s.ask_up, s.bid_down, s.ask_down,
            s.bid_size_up, s.ask_size_up, s.bid_size_down, s.ask_size_down,
            s.spread_up, s.spread_down,
            s.chainlink_price, s.dir_move_pct, s.abs_move_pct,
            s.hour_utc, s.day_of_week,
            r.resolved_up
        FROM slot_snapshots s
        JOIN slot_resolutions r ON s.symbol = r.symbol AND s.slot_ts = r.slot_ts
        WHERE r.resolved_up IS NOT NULL
          AND s.minutes_into_slot BETWEEN {min_minutes} AND {max_minutes}
        ORDER BY s.slot_ts, s.symbol, s.minutes_into_slot
    """

    async with engine.connect() as conn:
        result = await conn.execute(query)  # type: ignore[arg-type]
        rows = result.fetchall()
        cols = result.keys()

    await engine.dispose()

    df = pd.DataFrame(rows, columns=list(cols))
    logger.info("loaded_data", rows=len(df),
                symbols=sorted(df["symbol"].unique().tolist()) if len(df) > 0 else [])
    return df


def deduplicate_per_slot(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one snapshot per (symbol, slot_ts) — the latest within the window."""
    return (
        df.sort_values("minutes_into_slot", ascending=False)
        .groupby(["symbol", "slot_ts"])
        .first()
        .reset_index()
    )


def temporal_split(df: pd.DataFrame, val_days: int = 3
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by time: last val_days of slot_ts data go to validation."""
    max_ts = df["slot_ts"].max()
    cutoff = max_ts - val_days * 86400
    train = df[df["slot_ts"] <= cutoff].copy()
    val = df[df["slot_ts"] > cutoff].copy()
    logger.info("temporal_split",
                train_rows=len(train), val_rows=len(val),
                cutoff_ts=cutoff)
    return train, val


def train_model(X_train: pd.DataFrame, y_train: pd.Series,
                X_val: pd.DataFrame, y_val: pd.Series) -> CalibratedClassifierCV:
    """Train XGBoost + isotonic calibration."""
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
        use_label_encoder=False,
    )

    base.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Calibrate on validation set
    calibrated = CalibratedClassifierCV(
        estimator=base,
        method="isotonic",
        cv="prefit",
    )
    calibrated.fit(X_val, y_val)

    return calibrated


def evaluate(model: CalibratedClassifierCV, X: pd.DataFrame,
             y: pd.Series, label: str) -> dict:
    """Print and return evaluation metrics."""
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= 0.5).astype(int)

    auc = roc_auc_score(y, proba)
    brier = brier_score_loss(y, proba)
    ll = log_loss(y, proba)

    print(f"\n=== {label} ===")
    print(f"  AUC:   {auc:.4f}")
    print(f"  Brier: {brier:.4f}")
    print(f"  LogLoss: {ll:.4f}")
    print(classification_report(y, preds, target_names=["Down", "Up"]))

    # Calibration by decile
    print("  Calibration (deciles):")
    df_cal = pd.DataFrame({"proba": proba, "actual": y})
    df_cal["bin"] = pd.qcut(df_cal["proba"], q=10, duplicates="drop")
    cal_table = df_cal.groupby("bin", observed=True).agg(
        mean_pred=("proba", "mean"),
        mean_actual=("actual", "mean"),
        count=("actual", "count"),
    )
    for _, row in cal_table.iterrows():
        delta = row["mean_actual"] - row["mean_pred"]
        print(f"    pred={row['mean_pred']:.3f}  actual={row['mean_actual']:.3f}"
              f"  delta={delta:+.3f}  n={int(row['count'])}")

    return {"auc": auc, "brier": brier, "log_loss": ll}


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
    df = await load_data(args.mysql_url, args.min_minutes, args.max_minutes)
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

    # Feature importance (from base estimator)
    base_model = model.estimators_[0]  # type: ignore[attr-defined]
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
    p.add_argument("--mysql-url", type=str,
                   default=settings.MYSQL_URL,
                   help="MySQL connection string")
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
