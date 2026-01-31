# tests/unit/test_event_detector_ml.py
"""Tests for ML-based event detection."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile
import pandas as pd
import numpy as np

from src.realtime.event_detector import EventDetector, SignificantEvent
from src.feeds.base import FeedEvent
from src.ml.train import ImpactModel


class TestEventDetectorML:
    """Tests for ML-integrated EventDetector."""

    @pytest.fixture
    def trained_model_path(self):
        """Create a trained model for testing."""
        np.random.seed(42)
        n = 50
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
            "event_kill": np.zeros(n),
            "event_tower_destroyed": np.zeros(n),
            "event_dragon_kill": np.zeros(n),
            "event_baron_kill": np.zeros(n),
            "event_elder_kill": np.zeros(n),
            "event_inhibitor_destroyed": np.zeros(n),
            "event_ace": np.zeros(n),
        }
        for i in range(n):
            event_col = np.random.choice([
                "event_kill", "event_baron_kill", "event_dragon_kill"
            ])
            data[event_col][i] = 1

        data["label"] = (np.array(data["gold_diff"]) > 0).astype(int)
        df = pd.DataFrame(data)

        X = df.drop(columns=["label"])
        y = df["label"]

        model = ImpactModel()
        model.train(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_model.pkl"
            model.save(path)
            yield path

    @pytest.fixture
    def detector_with_model(self, trained_model_path):
        """EventDetector with ML model loaded."""
        return EventDetector(model_path=trained_model_path)

    @pytest.fixture
    def detector_no_model(self):
        """EventDetector without ML model (static fallback)."""
        return EventDetector()

    @pytest.fixture
    def sample_event(self):
        """Sample LoL event with game state."""
        return FeedEvent(
            source="pandascore",
            event_type="baron_kill",
            game="lol",
            data={
                "team": "T1",
                "game_time_minutes": 25,
                "gold_diff": 5000,
                "kill_diff": 4,
                "tower_diff": 2,
                "dragon_diff": 1,
                "baron_diff": 1,
            },
            timestamp=1234567890.0,
            match_id="12345",
        )

    def test_detector_loads_model(self, trained_model_path):
        """Detector loads model from path."""
        detector = EventDetector(model_path=trained_model_path)
        assert detector.model is not None

    def test_detector_without_model_uses_static(self, detector_no_model):
        """Detector without model uses static weights."""
        assert detector_no_model.model is None

    def test_estimate_price_impact_with_model(
        self, detector_with_model, sample_event
    ):
        """estimate_price_impact uses ML model when available."""
        classified = detector_with_model.classify(sample_event)
        new_price = detector_with_model.estimate_price_impact(
            classified, current_price=0.5
        )
        assert 0.01 <= new_price <= 0.99

    def test_estimate_price_impact_without_model(
        self, detector_no_model, sample_event
    ):
        """estimate_price_impact uses static weights when no model."""
        classified = detector_no_model.classify(sample_event)
        new_price = detector_no_model.estimate_price_impact(
            classified, current_price=0.5
        )
        assert 0.01 <= new_price <= 0.99

    def test_ml_prediction_uses_game_state(self, detector_with_model):
        """ML prediction considers full game state, not just event type."""
        event_behind = FeedEvent(
            source="pandascore",
            event_type="baron_kill",
            game="lol",
            data={
                "team": "T1",
                "game_time_minutes": 25,
                "gold_diff": -5000,
                "kill_diff": -4,
                "tower_diff": -2,
                "dragon_diff": -1,
                "baron_diff": 0,
            },
            timestamp=1234567890.0,
            match_id="12345",
        )

        event_ahead = FeedEvent(
            source="pandascore",
            event_type="baron_kill",
            game="lol",
            data={
                "team": "T1",
                "game_time_minutes": 25,
                "gold_diff": 5000,
                "kill_diff": 4,
                "tower_diff": 2,
                "dragon_diff": 1,
                "baron_diff": 1,
            },
            timestamp=1234567890.0,
            match_id="12345",
        )

        classified_behind = detector_with_model.classify(event_behind)
        classified_ahead = detector_with_model.classify(event_ahead)

        price_behind = detector_with_model.estimate_price_impact(
            classified_behind, 0.5
        )
        price_ahead = detector_with_model.estimate_price_impact(
            classified_ahead, 0.5
        )

        # Baron when ahead should give higher win prob than baron when behind
        assert price_ahead > price_behind
