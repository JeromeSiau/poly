# tests/ml/test_features.py
"""Tests for feature extraction pipeline."""

import pytest
import pandas as pd

from src.ml.features import FeatureExtractor, extract_features_from_events
from src.ml.data_collector import EventData


class TestFeatureExtractor:
    """Tests for FeatureExtractor."""

    @pytest.fixture
    def extractor(self):
        return FeatureExtractor(game="lol")

    @pytest.fixture
    def sample_event(self):
        return EventData(
            event_type="baron_kill",
            timestamp=1200,
            team="T1",
            game_time_minutes=20.0,
            gold_diff=5000,
            kill_diff=4,
            tower_diff=2,
            dragon_diff=1,
            baron_diff=1,
            winner="T1",
        )

    def test_extract_single_event_returns_dict(self, extractor, sample_event):
        """extract_single returns feature dict."""
        features = extractor.extract_single(sample_event)

        assert isinstance(features, dict)
        assert "game_time_minutes" in features
        assert "gold_diff_normalized" in features
        assert "event_baron_kill" in features

    def test_gold_diff_is_normalized(self, extractor, sample_event):
        """Gold diff is normalized by game time."""
        features = extractor.extract_single(sample_event)

        # 5000 gold diff at 20 min = 250 gold/min
        assert features["gold_diff_normalized"] == pytest.approx(250.0, rel=0.01)

    def test_event_type_is_one_hot(self, extractor, sample_event):
        """Event type is one-hot encoded."""
        features = extractor.extract_single(sample_event)

        assert features["event_baron_kill"] == 1
        assert features["event_kill"] == 0
        assert features["event_dragon_kill"] == 0

    def test_label_is_binary(self, extractor, sample_event):
        """Label is 1 if event team won, 0 otherwise."""
        features = extractor.extract_single(sample_event)

        # T1 did the event, T1 won -> label = 1
        assert features["label"] == 1

    def test_label_zero_when_event_team_lost(self, extractor):
        """Label is 0 when event team lost."""
        event = EventData(
            event_type="baron_kill",
            timestamp=1200,
            team="T1",
            game_time_minutes=20.0,
            gold_diff=5000,
            kill_diff=4,
            tower_diff=2,
            dragon_diff=1,
            baron_diff=1,
            winner="Gen.G",  # T1 lost
        )
        features = extractor.extract_single(event)

        assert features["label"] == 0

    def test_extract_batch_returns_dataframe(self, extractor, sample_event):
        """extract_batch returns pandas DataFrame."""
        events = [sample_event, sample_event]
        df = extractor.extract_batch(events)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_feature_columns_are_consistent(self, extractor):
        """All events produce same columns."""
        event1 = EventData(
            event_type="kill",
            timestamp=300,
            team="T1",
            game_time_minutes=5.0,
            gold_diff=500,
            kill_diff=1,
            tower_diff=0,
            dragon_diff=0,
            baron_diff=0,
            winner="T1",
        )
        event2 = EventData(
            event_type="baron_kill",
            timestamp=1800,
            team="Gen.G",
            game_time_minutes=30.0,
            gold_diff=-2000,
            kill_diff=-3,
            tower_diff=-2,
            dragon_diff=0,
            baron_diff=1,
            winner="Gen.G",
        )

        f1 = extractor.extract_single(event1)
        f2 = extractor.extract_single(event2)

        assert set(f1.keys()) == set(f2.keys())
