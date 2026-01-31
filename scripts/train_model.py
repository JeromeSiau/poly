#!/usr/bin/env python3
# scripts/train_model.py
"""Train ML model from collected data.

Usage:
    uv run python scripts/train_model.py --input data/lol_training.csv --output models/lol_impact.pkl
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog

from src.ml.train import train_model

logger = structlog.get_logger()


def main():
    parser = argparse.ArgumentParser(
        description="Train impact prediction model"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input CSV with training data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/impact_model.pkl",
        help="Output model path",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2,
        help="Validation split fraction",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        logger.error("input_not_found", path=str(input_path))
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = train_model(
        data_path=input_path,
        output_path=output_path,
        validation_split=args.validation_split,
    )

    logger.info(
        "training_complete",
        metrics=metrics,
        model_path=str(output_path),
    )

    print(f"\nTraining complete!")
    print(f"  Train AUC: {metrics.get('train_auc', 'N/A'):.4f}")
    print(f"  Val AUC:   {metrics.get('val_auc', 'N/A'):.4f}")
    print(f"  Model saved to: {output_path}")


if __name__ == "__main__":
    main()
