# src/ml/train.py
"""XGBoost model training for impact prediction."""

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss
import xgboost as xgb

import structlog

logger = structlog.get_logger()


class ImpactModel:
    """XGBoost model for predicting win probability after events."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.1,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_names: list[str] = []

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_split: float = 0.0,
    ) -> dict:
        """Train the model."""
        self.feature_names = list(X.columns)

        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            objective="binary:logistic",
            eval_metric="auc",
            random_state=42,
        )

        metrics = {}

        if validation_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=42
            )

            self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            train_probs = self.model.predict_proba(X_train)[:, 1]
            val_probs = self.model.predict_proba(X_val)[:, 1]

            metrics["train_auc"] = roc_auc_score(y_train, train_probs)
            metrics["val_auc"] = roc_auc_score(y_val, val_probs)
            metrics["train_brier"] = brier_score_loss(y_train, train_probs)
            metrics["val_brier"] = brier_score_loss(y_val, val_probs)

            logger.info(
                "training_complete",
                train_auc=metrics["train_auc"],
                val_auc=metrics["val_auc"],
            )
        else:
            self.model.fit(X, y, verbose=False)
            probs = self.model.predict_proba(X)[:, 1]
            metrics["train_auc"] = roc_auc_score(y, probs)

        return metrics

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict win probability for events."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        return self.model.predict_proba(X)[:, 1]

    def predict_single(self, features: dict) -> float:
        """Predict win probability for a single event."""
        df = pd.DataFrame([features])
        df = df[self.feature_names]
        return float(self.predict_proba(df)[0])

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance scores."""
        if self.model is None:
            raise RuntimeError("Model not trained.")
        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance))

    def save(self, path: Path) -> None:
        """Save model to disk."""
        if self.model is None:
            raise RuntimeError("Model not trained.")
        joblib.dump(
            {
                "model": self.model,
                "feature_names": self.feature_names,
                "params": {
                    "n_estimators": self.n_estimators,
                    "max_depth": self.max_depth,
                    "learning_rate": self.learning_rate,
                },
            },
            path,
        )
        logger.info("model_saved", path=str(path))

    @classmethod
    def load(cls, path: Path) -> "ImpactModel":
        """Load model from disk."""
        data = joblib.load(path)
        instance = cls(**data["params"])
        instance.model = data["model"]
        instance.feature_names = data["feature_names"]
        logger.info("model_loaded", path=str(path))
        return instance


def train_model(
    data_path: Path,
    output_path: Path,
    game: str = "lol",
    validation_split: float = 0.2,
) -> dict:
    """Train model from CSV data file."""
    logger.info("loading_data", path=str(data_path))
    df = pd.read_csv(data_path)

    X = df.drop(columns=["label"])
    y = df["label"]

    logger.info("training_model", samples=len(df), features=len(X.columns))
    model = ImpactModel()
    metrics = model.train(X, y, validation_split=validation_split)

    model.save(output_path)

    return metrics
