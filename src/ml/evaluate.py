"""Evaluation metrics for TD maker ML models."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import (
    brier_score_loss,
    classification_report,
    log_loss,
    roc_auc_score,
)


def evaluate(model, X: pd.DataFrame, y: pd.Series, label: str) -> dict:
    """Print and return evaluation metrics (AUC, Brier, LogLoss, calibration)."""
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
