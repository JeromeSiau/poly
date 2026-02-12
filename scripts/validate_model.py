#!/usr/bin/env python3
# scripts/validate_model.py
"""Validate ML model calibration.

Usage:
    uv run python scripts/validate_model.py \
        --model models/lol_impact.pkl \
        --data data/lol_training.csv \
        --output reports/validation.html
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import structlog

from src.ml.train import ImpactModel
from src.ml.validation.report import ValidationReport

logger = structlog.get_logger()


def main():
    parser = argparse.ArgumentParser(description="Validate model calibration")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--data", type=str, required=True, help="Path to validation data CSV")
    parser.add_argument("--output", type=str, default="reports/validation.html", help="Output HTML path")

    args = parser.parse_args()

    model_path = Path(args.model)
    data_path = Path(args.data)
    output_path = Path(args.output)

    if not model_path.exists():
        logger.error("model_not_found", path=str(model_path))
        sys.exit(1)

    if not data_path.exists():
        logger.error("data_not_found", path=str(data_path))
        sys.exit(1)

    # Load model and data
    logger.info("loading_model", path=str(model_path))
    model = ImpactModel.load(model_path)

    logger.info("loading_data", path=str(data_path))
    df = pd.read_csv(data_path)

    # Generate predictions
    X = df.drop(columns=["label"])
    y_true = df["label"].values
    y_pred = model.predict_proba(X)

    # Prepare validation DataFrame
    val_df = df.copy()
    val_df["y_true"] = y_true
    val_df["y_pred"] = y_pred

    # Rename columns to match expected format
    if "game_time_minutes" not in val_df.columns and "game_time" in val_df.columns:
        val_df["game_time_minutes"] = val_df["game_time"]

    # Generate report
    logger.info("generating_report")
    report = ValidationReport(val_df)
    summary = report.get_summary()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.generate_html(output_path)

    # Print summary
    print("\n" + "=" * 50)
    print("MODEL VALIDATION REPORT")
    print("=" * 50)
    print(f"Brier Score:    {summary['brier_score']:.4f} [{summary['brier_ci_lower']:.4f} - {summary['brier_ci_upper']:.4f}]")
    print(f"ECE:            {summary['ece']:.4f}")
    print(f"H-L p-value:    {summary['hl_pvalue']:.4f}")
    print(f"Samples:        {summary['n_samples']}")
    print("-" * 50)
    print(f"Recommendation: {summary['recommendation']}")
    print("=" * 50)
    print(f"\nFull report saved to: {output_path}")


if __name__ == "__main__":
    main()
