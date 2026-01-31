# tests/ml/test_train.py
"""Tests for model training."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from src.ml.train import ImpactModel, train_model


class TestImpactModel:
    """Tests for ImpactModel."""

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n = 100

        data = {
            "game_time_minutes": np.random.uniform(5, 40, n),
            "gold_diff": np.random.uniform(-10000, 10000, n),
            "gold_diff_normalized": np.random.uniform(-500, 500, n),
            "kill_diff": np.random.randint(-10, 10, n),
            "kill_diff_normalized": np.random.uniform(-1, 1, n),
            "tower_diff": np.random.randint(-5, 5, n),
            "dragon_diff": np.random.randint(-3, 3, n),
            "baron_diff": np.random.randint(-2, 2, n),
            "is_ahead": np.random.randint(0, 2, n),
            "is_late_game": np.random.randint(0, 2, n),
            "event_kill": np.random.randint(0, 2, n),
            "event_baron_kill": np.random.randint(0, 2, n),
            "event_dragon_kill": np.random.randint(0, 2, n),
        }

        # Label correlates with gold_diff
        data["label"] = (data["gold_diff"] > 0).astype(int)

        return pd.DataFrame(data)

    def test_model_trains_without_error(self, sample_data):
        """Model trains on sample data."""
        model = ImpactModel()
        X = sample_data.drop(columns=["label"])
        y = sample_data["label"]

        model.train(X, y)

        assert model.model is not None

    def test_model_predicts_probabilities(self, sample_data):
        """Model returns probabilities between 0 and 1."""
        model = ImpactModel()
        X = sample_data.drop(columns=["label"])
        y = sample_data["label"]

        model.train(X, y)
        probs = model.predict_proba(X)

        assert len(probs) == len(X)
        assert all(0 <= p <= 1 for p in probs)

    def test_model_save_load(self, sample_data):
        """Model can be saved and loaded."""
        model = ImpactModel()
        X = sample_data.drop(columns=["label"])
        y = sample_data["label"]
        model.train(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pkl"
            model.save(path)

            loaded = ImpactModel.load(path)
            probs_original = model.predict_proba(X)
            probs_loaded = loaded.predict_proba(X)

            np.testing.assert_array_almost_equal(probs_original, probs_loaded)

    def test_model_has_feature_importance(self, sample_data):
        """Model exposes feature importance."""
        model = ImpactModel()
        X = sample_data.drop(columns=["label"])
        y = sample_data["label"]
        model.train(X, y)

        importance = model.get_feature_importance()

        assert len(importance) == len(X.columns)
        assert "gold_diff" in importance

    def test_train_with_validation_split(self, sample_data):
        """Training with validation returns metrics."""
        model = ImpactModel()
        X = sample_data.drop(columns=["label"])
        y = sample_data["label"]

        metrics = model.train(X, y, validation_split=0.2)

        assert "train_auc" in metrics
        assert "val_auc" in metrics
        assert metrics["val_auc"] > 0.5

    def test_train_model_from_csv(self, sample_data):
        """train_model loads CSV and trains model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "data.csv"
            model_path = Path(tmpdir) / "model.pkl"

            sample_data.to_csv(data_path, index=False)

            metrics = train_model(data_path, model_path)

            assert model_path.exists()
            assert "train_auc" in metrics
            assert "val_auc" in metrics
