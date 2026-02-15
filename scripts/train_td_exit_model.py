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
# Feature columns — superset of entry model + position-specific features
# ---------------------------------------------------------------------------
EXIT_FEATURE_COLS = [
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
    # Derived (same as entry model)
    "spread_ratio",
    "bid_imbalance",
    "fav_bid",
    "move_velocity",
    "book_pressure",
    # Trend
    "bid_trend_30s",
    "bid_trend_2m",
    "ask_trend_30s",
    "spread_trend",
    # Exit-specific: position context
    "entry_price",
    "bid_max",           # highest bid since fill
    "bid_drop",          # bid_max - current bid
    "bid_drop_pct",      # bid_drop / bid_max
    "minutes_remaining",
    "minutes_held",
    "pnl_unrealized",    # bid_up - entry_price (per share)
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features to raw snapshot data."""
    df = df.copy()
    df["spread_ratio"] = df["spread_up"] / df["spread_down"].clip(lower=0.001)
    df["bid_imbalance"] = df["bid_up"] - df["bid_down"]
    df["fav_bid"] = df[["bid_up", "bid_down"]].max(axis=1)
    df["move_velocity"] = df["dir_move_pct"] / df["minutes_into_slot"].clip(lower=0.5)
    total_bid_sz = (df["bid_size_up"] + df["bid_size_down"]).clip(lower=0.01)
    df["book_pressure"] = (df["bid_size_up"] - df["bid_size_down"]) / total_bid_sz
    return df


def compute_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute trend features from consecutive snapshots within each slot."""
    df = df.copy()
    df.sort_values(["symbol", "slot_ts", "captured_at"], inplace=True)
    group = df.groupby(["symbol", "slot_ts"])
    df["bid_trend_30s"] = df["bid_up"] - group["bid_up"].shift(1)
    df["bid_trend_2m"] = df["bid_up"] - group["bid_up"].shift(4)
    df["ask_trend_30s"] = df["ask_up"] - group["ask_up"].shift(1)
    df["spread_trend"] = df["spread_up"] - group["spread_up"].shift(2)
    return df


def compute_exit_features(df: pd.DataFrame, entry_price: float = 0.75) -> pd.DataFrame:
    """Compute position-specific features for exit model.

    For each slot, identifies the simulated fill point (first snapshot
    where bid_up >= entry_price) and computes running features for all
    subsequent snapshots.
    """
    df = df.copy()
    df.sort_values(["symbol", "slot_ts", "captured_at"], inplace=True)

    results = []
    for (sym, slot_ts), group in df.groupby(["symbol", "slot_ts"]):
        rows = group.reset_index(drop=True)
        resolved_up = rows["resolved_up"].iloc[0]

        # Find fill point: first snapshot where bid_up >= entry_price.
        fill_idx = None
        for i, row in rows.iterrows():
            if row["bid_up"] >= entry_price:
                fill_idx = i
                break

        if fill_idx is None:
            continue  # no fill in this slot

        fill_price = rows.loc[fill_idx, "bid_up"]
        fill_minutes = rows.loc[fill_idx, "minutes_into_slot"]

        # Take all snapshots after fill (including fill itself for context).
        post_fill = rows.loc[fill_idx:].copy()
        if len(post_fill) < 2:
            continue

        # Running bid_max since fill.
        post_fill["bid_max"] = post_fill["bid_up"].cummax()
        post_fill["entry_price"] = fill_price
        post_fill["bid_drop"] = post_fill["bid_max"] - post_fill["bid_up"]
        post_fill["bid_drop_pct"] = post_fill["bid_drop"] / post_fill["bid_max"].clip(lower=0.01)
        post_fill["minutes_remaining"] = 15.0 - post_fill["minutes_into_slot"]
        post_fill["minutes_held"] = post_fill["minutes_into_slot"] - fill_minutes
        post_fill["pnl_unrealized"] = post_fill["bid_up"] - fill_price

        # Skip the fill snapshot itself (no exit decision at fill time).
        post_fill = post_fill.iloc[1:]
        if len(post_fill) == 0:
            continue

        results.append(post_fill)

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)


async def load_data(db_url: str) -> pd.DataFrame:
    """Load ALL snapshots for resolved slots."""
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(db_url, echo=False)
    query = """
        SELECT
            s.symbol, s.slot_ts, s.minutes_into_slot, s.captured_at,
            s.bid_up, s.ask_up, s.bid_down, s.ask_down,
            s.bid_size_up, s.ask_size_up, s.bid_size_down, s.ask_size_down,
            s.spread_up, s.spread_down,
            s.chainlink_price, s.dir_move_pct, s.abs_move_pct,
            s.hour_utc, s.day_of_week,
            r.resolved_up
        FROM slot_snapshots s
        JOIN slot_resolutions r ON s.symbol = r.symbol AND s.slot_ts = r.slot_ts
        WHERE r.resolved_up IS NOT NULL
        ORDER BY s.symbol, s.slot_ts, s.captured_at
    """

    async with engine.connect() as conn:
        result = await conn.execute(query)  # type: ignore[arg-type]
        rows = result.fetchall()
        cols = result.keys()

    await engine.dispose()

    df = pd.DataFrame(rows, columns=list(cols))
    logger.info("loaded_raw_snapshots", rows=len(df),
                symbols=sorted(df["symbol"].unique().tolist()) if len(df) > 0 else [])
    return df


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
    """Train XGBoost + isotonic calibration for exit decisions."""
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
        use_label_encoder=False,
    )

    base.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

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

    # Exit-specific: show P(win) vs bid_drop to validate the stop-loss signal
    if "bid_drop" in X.columns:
        print("\n  P(win) by bid_drop bucket:")
        df_check = pd.DataFrame({
            "proba": proba, "actual": y, "bid_drop": X["bid_drop"].values,
        })
        df_check["drop_bin"] = pd.cut(
            df_check["bid_drop"],
            bins=[0, 0.02, 0.05, 0.10, 0.15, 0.25, 1.0],
        )
        for bucket, g in df_check.groupby("drop_bin", observed=True):
            if len(g) > 5:
                print(f"    drop {bucket}: n={len(g):>5}  "
                      f"pred_p={g['proba'].mean():.3f}  "
                      f"actual_p={g['actual'].mean():.3f}")

    return {"auc": auc, "brier": brier, "log_loss": ll}


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
    df = await load_data(args.db_url)
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
    base_model = model.estimators_[0]  # type: ignore[attr-defined]
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
