#!/usr/bin/env python3
# scripts/train_prematch_model.py
"""Train pre-match prediction model.

Uses historical team stats to predict match outcomes before they start.

Usage:
    uv run python scripts/train_prematch_model.py \
        --input data/prematch_training.csv \
        --output models/prematch_model.pkl
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
import xgboost as xgb
import structlog

logger = structlog.get_logger()

# Features to use for training (excludes metadata columns)
FEATURE_COLS = [
    # Team 1 stats
    'team1_winrate',
    'team1_recent_winrate',
    'team1_blue_winrate',
    'team1_games',
    'team1_fb_rate',
    'team1_ft_rate',
    'team1_fd_rate',
    'team1_avg_gd15',

    # Team 2 stats
    'team2_winrate',
    'team2_recent_winrate',
    'team2_red_winrate',
    'team2_games',
    'team2_fb_rate',
    'team2_ft_rate',
    'team2_fd_rate',
    'team2_avg_gd15',

    # Relative features
    'winrate_diff',
    'recent_form_diff',
    'fb_rate_diff',
    'gd15_diff',

    # Head-to-head
    'h2h_games',
    'h2h_winrate',

    # Context
    'league_tier',
    'is_playoffs',
]


class PrematchModel:
    """XGBoost model for pre-match win probability prediction."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.05,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.model = None
        self.feature_names = FEATURE_COLS

    def train(self, X: pd.DataFrame, y: pd.Series, validation_split: float = 0.2) -> dict:
        """Train the model with validation."""
        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            objective="binary:logistic",
            eval_metric="auc",
            random_state=42,
            use_label_encoder=False,
        )

        metrics = {}

        if validation_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=42
            )

            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            # Train metrics
            train_probs = self.model.predict_proba(X_train)[:, 1]
            metrics['train_auc'] = roc_auc_score(y_train, train_probs)
            metrics['train_brier'] = brier_score_loss(y_train, train_probs)

            # Validation metrics
            val_probs = self.model.predict_proba(X_val)[:, 1]
            metrics['val_auc'] = roc_auc_score(y_val, val_probs)
            metrics['val_brier'] = brier_score_loss(y_val, val_probs)
            metrics['val_logloss'] = log_loss(y_val, val_probs)

            # Cross-validation
            cv_scores = cross_val_score(
                xgb.XGBClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    objective="binary:logistic",
                    random_state=42,
                    use_label_encoder=False,
                ),
                X, y, cv=5, scoring='roc_auc'
            )
            metrics['cv_auc_mean'] = cv_scores.mean()
            metrics['cv_auc_std'] = cv_scores.std()

        else:
            self.model.fit(X, y, verbose=False)
            probs = self.model.predict_proba(X)[:, 1]
            metrics['train_auc'] = roc_auc_score(y, probs)

        return metrics

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict win probability."""
        if self.model is None:
            raise RuntimeError("Model not trained")
        return self.model.predict_proba(X)[:, 1]

    def predict_single(self, features: dict) -> float:
        """Predict for a single match."""
        df = pd.DataFrame([features])
        # Ensure all expected columns exist
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[self.feature_names]
        return float(self.predict_proba(df)[0])

    def get_feature_importance(self) -> dict:
        """Get feature importance scores."""
        if self.model is None:
            raise RuntimeError("Model not trained")
        importance = self.model.feature_importances_
        return dict(sorted(
            zip(self.feature_names, importance),
            key=lambda x: x[1],
            reverse=True
        ))

    def save(self, path: Path) -> None:
        """Save model to disk."""
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names,
            'params': {
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'learning_rate': self.learning_rate,
            },
        }, path)
        logger.info("model_saved", path=str(path))

    @classmethod
    def load(cls, path: Path) -> "PrematchModel":
        """Load model from disk."""
        data = joblib.load(path)
        instance = cls(**data['params'])
        instance.model = data['model']
        instance.feature_names = data['feature_names']
        return instance


def main():
    parser = argparse.ArgumentParser(description="Train pre-match prediction model")
    parser.add_argument("--input", type=str, required=True, help="Input CSV with features")
    parser.add_argument("--output", type=str, default="models/prematch_model.pkl", help="Output model path")
    parser.add_argument("--validation-split", type=float, default=0.2, help="Validation split")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    # Load data
    logger.info("loading_data", path=str(input_path))
    df = pd.read_csv(input_path)

    # Prepare features
    X = df[FEATURE_COLS].copy()
    y = df['label']

    # Handle missing values
    X = X.fillna(0)

    logger.info("training_model", samples=len(df), features=len(FEATURE_COLS))

    # Train
    model = PrematchModel()
    metrics = model.train(X, y, validation_split=args.validation_split)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)

    # Feature importance
    importance = model.get_feature_importance()

    # Print results
    print(f"\n{'='*60}")
    print("PRE-MATCH MODEL TRAINED")
    print(f"{'='*60}")
    print(f"Samples:        {len(df):,}")
    print(f"Features:       {len(FEATURE_COLS)}")
    print(f"-" * 60)
    print(f"Train AUC:      {metrics['train_auc']:.4f}")
    print(f"Val AUC:        {metrics['val_auc']:.4f}")
    print(f"Val Brier:      {metrics['val_brier']:.4f}")
    print(f"CV AUC:         {metrics['cv_auc_mean']:.4f} (+/- {metrics['cv_auc_std']:.4f})")
    print(f"-" * 60)
    print("Top 10 Important Features:")
    for i, (feat, imp) in enumerate(list(importance.items())[:10]):
        print(f"  {i+1}. {feat}: {imp:.4f}")
    print(f"{'='*60}")
    print(f"Model saved to: {output_path}")
    print(f"\nNext step:")
    print(f"  uv run python scripts/predict_match.py --team1 'T1' --team2 'G2'")


if __name__ == "__main__":
    main()
