# src/ml/features.py
"""Feature extraction for ML impact prediction.

Converts raw event data into feature vectors for XGBoost.
Features are team-agnostic to prevent overfitting to team names.
"""

from typing import Optional

import pandas as pd

from src.ml.data_collector import EventData


# Event types to one-hot encode (per game)
LOL_EVENT_TYPES = [
    "kill",
    "tower_destroyed",
    "dragon_kill",
    "baron_kill",
    "elder_kill",
    "inhibitor_destroyed",
    "ace",
]

CSGO_EVENT_TYPES = [
    "kill",
    "round_end",
    "bomb_planted",
    "bomb_defused",
    "ace",
    "clutch",
]

DOTA2_EVENT_TYPES = [
    "kill",
    "tower_destroyed",
    "roshan_kill",
    "barracks_destroyed",
    "aegis_pickup",
    "team_wipe",
]


class FeatureExtractor:
    """Extracts ML features from game events."""

    def __init__(self, game: str):
        """Initialize extractor for a specific game."""
        self.game = game.lower()

        if self.game == "lol":
            self.event_types = LOL_EVENT_TYPES
        elif self.game == "csgo":
            self.event_types = CSGO_EVENT_TYPES
        elif self.game == "dota2":
            self.event_types = DOTA2_EVENT_TYPES
        else:
            self.event_types = LOL_EVENT_TYPES

    def extract_single(self, event: EventData) -> dict:
        """Extract features from a single event."""
        features = {}

        # Time features
        features["game_time_minutes"] = event.game_time_minutes

        # Normalized state features
        if event.game_time_minutes > 0:
            features["gold_diff_normalized"] = event.gold_diff / event.game_time_minutes
            features["kill_diff_normalized"] = event.kill_diff / event.game_time_minutes
        else:
            features["gold_diff_normalized"] = 0.0
            features["kill_diff_normalized"] = 0.0

        # Raw diff features
        features["gold_diff"] = event.gold_diff
        features["kill_diff"] = event.kill_diff
        features["tower_diff"] = event.tower_diff
        features["dragon_diff"] = event.dragon_diff
        features["baron_diff"] = event.baron_diff

        # Derived features
        features["is_ahead"] = 1 if event.gold_diff > 0 else 0
        features["is_late_game"] = 1 if event.game_time_minutes > 25 else 0

        # One-hot encode event type
        for et in self.event_types:
            features[f"event_{et}"] = 1 if event.event_type == et else 0

        # Label
        features["label"] = 1 if event.team == event.winner else 0

        return features

    def extract_batch(self, events: list[EventData]) -> pd.DataFrame:
        """Extract features from multiple events."""
        rows = [self.extract_single(e) for e in events]
        return pd.DataFrame(rows)

    def get_feature_columns(self) -> list[str]:
        """Get list of feature column names (excluding label)."""
        dummy = EventData(
            event_type="kill",
            timestamp=0,
            team="A",
            game_time_minutes=1.0,
            gold_diff=0,
            kill_diff=0,
            tower_diff=0,
            dragon_diff=0,
            baron_diff=0,
            winner="A",
        )
        features = self.extract_single(dummy)
        return [k for k in features.keys() if k != "label"]


def extract_features_from_events(events: list[EventData], game: str) -> pd.DataFrame:
    """Convenience function to extract features."""
    extractor = FeatureExtractor(game)
    return extractor.extract_batch(events)
