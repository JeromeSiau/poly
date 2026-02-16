"""Model loading utilities for TD maker ML models."""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import joblib


def find_latest_model(model_dir: str = "data/models",
                      prefix: str = "td_model") -> str:
    """Return path to the most recent model joblib matching prefix."""
    pattern = str(Path(model_dir) / f"{prefix}_*.joblib")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No model files found matching {pattern}")
        sys.exit(1)
    return files[-1]


def load_model(path: str) -> dict:
    """Load model payload from joblib.

    Returns dict with keys: model, feature_cols, trained_at (optional).
    """
    payload = joblib.load(path)
    required_keys = {"model", "feature_cols"}
    if not required_keys.issubset(payload.keys()):
        print(f"Model file missing keys: {required_keys - payload.keys()}")
        sys.exit(1)
    return payload
