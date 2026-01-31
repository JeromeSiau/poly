# tests/integration/test_ml_pipeline.py
"""End-to-end test for ML pipeline."""

import pytest
import tempfile
from pathlib import Path

import pandas as pd
import numpy as np

from src.ml.features import FeatureExtractor
from src.ml.train import ImpactModel
from src.ml.data_collector import EventData
from src.realtime.event_detector import EventDetector
from src.feeds.base import FeedEvent


class TestMLPipeline:
    """End-to-end ML pipeline tests."""

    @pytest.fixture
    def synthetic_events(self):
        """Create synthetic events that mimic real game patterns."""
        events = []
        np.random.seed(42)

        # Simulate 20 games
        for game_num in range(20):
            winner = "TeamA" if np.random.random() > 0.5 else "TeamB"

            game_time = 0
            gold_diff = 0
            kill_diff = 0
            tower_diff = 0
            dragon_diff = 0
            baron_diff = 0

            for _ in range(30):
                game_time += np.random.uniform(0.5, 2)
                if game_time > 40:
                    break

                is_winner_event = (
                    (winner == "TeamA" and np.random.random() > 0.4) or
                    (winner == "TeamB" and np.random.random() <= 0.4)
                )
                team = "TeamA" if is_winner_event else "TeamB"

                if is_winner_event:
                    gold_diff += np.random.randint(100, 500)
                    kill_diff += 1 if np.random.random() > 0.7 else 0
                else:
                    gold_diff -= np.random.randint(100, 500)
                    kill_diff -= 1 if np.random.random() > 0.7 else 0

                event_type = np.random.choice([
                    "kill", "kill", "kill",
                    "tower_destroyed",
                    "dragon_kill",
                    "baron_kill",
                ])

                if event_type == "tower_destroyed":
                    tower_diff += 1 if is_winner_event else -1
                elif event_type == "dragon_kill":
                    dragon_diff += 1 if is_winner_event else -1
                elif event_type == "baron_kill":
                    baron_diff += 1 if is_winner_event else -1

                events.append(EventData(
                    event_type=event_type,
                    timestamp=game_time * 60,
                    team=team,
                    game_time_minutes=game_time,
                    gold_diff=gold_diff,
                    kill_diff=kill_diff,
                    tower_diff=tower_diff,
                    dragon_diff=dragon_diff,
                    baron_diff=baron_diff,
                    winner=winner,
                ))

        return events

    def test_full_pipeline(self, synthetic_events):
        """Test: extract features -> train -> predict."""
        # 1. Extract features
        extractor = FeatureExtractor(game="lol")
        df = extractor.extract_batch(synthetic_events)

        assert len(df) > 100
        assert "label" in df.columns

        # 2. Train model
        X = df.drop(columns=["label"])
        y = df["label"]

        model = ImpactModel(n_estimators=50)
        metrics = model.train(X, y, validation_split=0.2)

        assert metrics["val_auc"] > 0.55

        # 3. Save and load model
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pkl"
            model.save(model_path)

            # 4. Integrate with EventDetector
            detector = EventDetector(model_path=model_path)

            # 5. Make prediction
            event = FeedEvent(
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

            classified = detector.classify(event)
            new_price = detector.estimate_price_impact(classified, 0.5)

            assert 0.01 <= new_price <= 0.99

    def test_model_improves_with_more_data(self, synthetic_events):
        """More training data should improve model performance."""
        extractor = FeatureExtractor(game="lol")
        df = extractor.extract_batch(synthetic_events)

        X = df.drop(columns=["label"])
        y = df["label"]

        # Train with 20% of data
        small_X = X.iloc[:len(X)//5]
        small_y = y.iloc[:len(y)//5]

        small_model = ImpactModel(n_estimators=50)
        small_metrics = small_model.train(small_X, small_y, validation_split=0.2)

        # Train with all data
        full_model = ImpactModel(n_estimators=50)
        full_metrics = full_model.train(X, y, validation_split=0.2)

        assert full_metrics["val_auc"] >= small_metrics["val_auc"] * 0.9
