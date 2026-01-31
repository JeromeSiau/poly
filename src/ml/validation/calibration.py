# src/ml/validation/calibration.py
"""Calibration analysis tools for model validation.

This module provides functions for analyzing model calibration:
- Brier score decomposition into reliability, resolution, and uncertainty
- Expected Calibration Error (ECE)
- Reliability diagram data generation
- CalibrationAnalyzer for context-aware analysis
"""

import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike


def brier_score_decomposition(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    n_bins: int = 10,
) -> dict[str, float]:
    """Decompose Brier score into reliability, resolution, and uncertainty.

    The Brier score can be decomposed as:
        Brier = Reliability - Resolution + Uncertainty

    Where:
    - Reliability (calibration error): measures how well predicted probabilities
      match observed frequencies within bins
    - Resolution: measures how much predictions deviate from the overall mean
    - Uncertainty: base rate uncertainty (p_bar * (1 - p_bar))

    Args:
        y_true: Binary ground truth labels (0 or 1)
        y_pred: Predicted probabilities in [0, 1]
        n_bins: Number of bins for grouping predictions

    Returns:
        Dictionary with brier_score, reliability, resolution, uncertainty
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    n = len(y_true)
    if n == 0:
        return {
            "brier_score": 0.0,
            "reliability": 0.0,
            "resolution": 0.0,
            "uncertainty": 0.0,
        }

    # Direct Brier score calculation
    brier_score = float(np.mean((y_true - y_pred) ** 2))

    # Overall base rate
    p_bar = np.mean(y_true)
    uncertainty = float(p_bar * (1 - p_bar))

    # Bin predictions
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred, bin_edges[1:-1])  # Assigns to bins 0 to n_bins-1

    reliability = 0.0
    resolution = 0.0

    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        n_k = np.sum(mask)

        if n_k == 0:
            continue

        # Mean predicted probability in bin
        p_k = np.mean(y_pred[mask])
        # Observed frequency in bin
        o_k = np.mean(y_true[mask])

        # Reliability: weighted squared difference between observed and predicted
        reliability += (n_k / n) * (o_k - p_k) ** 2

        # Resolution: weighted squared difference between bin frequency and overall
        resolution += (n_k / n) * (o_k - p_bar) ** 2

    return {
        "brier_score": brier_score,
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": uncertainty,
    }


def expected_calibration_error(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    n_bins: int = 10,
) -> float:
    """Calculate Expected Calibration Error (ECE).

    ECE measures the weighted average absolute difference between
    accuracy and confidence in each bin:

        ECE = sum_k (n_k / N) * |accuracy_k - confidence_k|

    Args:
        y_true: Binary ground truth labels (0 or 1)
        y_pred: Predicted probabilities in [0, 1]
        n_bins: Number of bins for grouping predictions

    Returns:
        ECE value in [0, 1], where 0 is perfectly calibrated
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    n = len(y_true)
    if n == 0:
        return 0.0

    # Bin predictions
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred, bin_edges[1:-1])

    ece = 0.0

    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        n_k = np.sum(mask)

        if n_k == 0:
            continue

        # Accuracy in bin (fraction of positives)
        accuracy_k = np.mean(y_true[mask])
        # Confidence in bin (mean predicted probability)
        confidence_k = np.mean(y_pred[mask])

        # Weighted absolute difference
        ece += (n_k / n) * abs(accuracy_k - confidence_k)

    return float(ece)


def reliability_diagram_data(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    n_bins: int = 10,
) -> dict[str, list[float]]:
    """Generate data for a reliability diagram.

    A reliability diagram plots the observed frequency (accuracy) against
    the predicted probability (confidence) for each bin. A perfectly
    calibrated model will have all points on the diagonal.

    Args:
        y_true: Binary ground truth labels (0 or 1)
        y_pred: Predicted probabilities in [0, 1]
        n_bins: Number of bins for grouping predictions

    Returns:
        Dictionary with:
        - bin_centers: Center of each bin
        - true_fractions: Observed frequency in each bin (NaN if empty)
        - counts: Number of samples in each bin
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Bin edges and centers
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_indices = np.digitize(y_pred, bin_edges[1:-1])

    true_fractions = []
    counts = []

    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        n_k = int(np.sum(mask))
        counts.append(n_k)

        if n_k == 0:
            true_fractions.append(float("nan"))
        else:
            true_fractions.append(float(np.mean(y_true[mask])))

    return {
        "bin_centers": bin_centers.tolist(),
        "true_fractions": true_fractions,
        "counts": counts,
    }


class CalibrationAnalyzer:
    """Analyze model calibration with context-aware breakdowns.

    Provides overall calibration metrics and breakdowns by game time,
    gold difference, and event type.

    Attributes:
        GAME_TIME_BINS: Time ranges for early/mid/late game
        GOLD_DIFF_BINS: Gold difference ranges for behind/even/ahead
    """

    GAME_TIME_BINS: dict[str, tuple[float, float]] = {
        "early": (0, 15),
        "mid": (15, 25),
        "late": (25, math.inf),
    }

    GOLD_DIFF_BINS: dict[str, tuple[float, float]] = {
        "behind": (-math.inf, -2000),
        "even": (-2000, 2000),
        "ahead": (2000, math.inf),
    }

    def __init__(
        self,
        y_true: ArrayLike,
        y_pred: ArrayLike,
        game_time_minutes: ArrayLike | None = None,
        gold_diff: ArrayLike | None = None,
        event_type: ArrayLike | None = None,
        n_bins: int = 10,
    ) -> None:
        """Initialize the analyzer with predictions and context.

        Args:
            y_true: Binary ground truth labels (0 or 1)
            y_pred: Predicted probabilities in [0, 1]
            game_time_minutes: Game time for each sample (optional)
            gold_diff: Gold difference for each sample (optional)
            event_type: Event type for each sample (optional)
            n_bins: Number of bins for calibration analysis
        """
        self.y_true = np.asarray(y_true, dtype=float)
        self.y_pred = np.asarray(y_pred, dtype=float)
        self.game_time_minutes = (
            np.asarray(game_time_minutes, dtype=float)
            if game_time_minutes is not None
            else None
        )
        self.gold_diff = (
            np.asarray(gold_diff, dtype=float) if gold_diff is not None else None
        )
        self.event_type = np.asarray(event_type) if event_type is not None else None
        self.n_bins = n_bins

    def analyze_overall(self) -> dict[str, Any]:
        """Compute overall calibration metrics.

        Returns:
            Dictionary with:
            - brier_score: Overall Brier score
            - ece: Expected Calibration Error
            - reliability: Calibration component of Brier
            - resolution: Resolution component of Brier
            - uncertainty: Uncertainty component of Brier
            - n_samples: Number of samples
        """
        decomposition = brier_score_decomposition(
            self.y_true, self.y_pred, self.n_bins
        )
        ece = expected_calibration_error(self.y_true, self.y_pred, self.n_bins)

        return {
            "brier_score": decomposition["brier_score"],
            "ece": ece,
            "reliability": decomposition["reliability"],
            "resolution": decomposition["resolution"],
            "uncertainty": decomposition["uncertainty"],
            "n_samples": len(self.y_true),
        }

    def _analyze_subset(self, mask: np.ndarray) -> dict[str, Any]:
        """Analyze calibration for a subset of samples.

        Args:
            mask: Boolean mask selecting samples to analyze

        Returns:
            Calibration metrics for the subset
        """
        if np.sum(mask) == 0:
            return {
                "brier_score": float("nan"),
                "ece": float("nan"),
                "reliability": float("nan"),
                "resolution": float("nan"),
                "uncertainty": float("nan"),
                "n_samples": 0,
            }

        y_true_subset = self.y_true[mask]
        y_pred_subset = self.y_pred[mask]

        decomposition = brier_score_decomposition(
            y_true_subset, y_pred_subset, self.n_bins
        )
        ece = expected_calibration_error(y_true_subset, y_pred_subset, self.n_bins)

        return {
            "brier_score": decomposition["brier_score"],
            "ece": ece,
            "reliability": decomposition["reliability"],
            "resolution": decomposition["resolution"],
            "uncertainty": decomposition["uncertainty"],
            "n_samples": int(np.sum(mask)),
        }

    def analyze_by_context(self) -> dict[str, dict[str, Any]]:
        """Analyze calibration broken down by context.

        Returns:
            Dictionary with:
            - by_game_time: Metrics for early/mid/late game
            - by_gold_diff: Metrics for behind/even/ahead
            - by_event_type: Metrics for each event type
        """
        result: dict[str, dict[str, Any]] = {
            "by_game_time": {},
            "by_gold_diff": {},
            "by_event_type": {},
        }

        # Analyze by game time
        if self.game_time_minutes is not None:
            for name, (low, high) in self.GAME_TIME_BINS.items():
                mask = (self.game_time_minutes >= low) & (self.game_time_minutes < high)
                if np.sum(mask) > 0:
                    result["by_game_time"][name] = self._analyze_subset(mask)

        # Analyze by gold diff
        if self.gold_diff is not None:
            for name, (low, high) in self.GOLD_DIFF_BINS.items():
                mask = (self.gold_diff >= low) & (self.gold_diff < high)
                if np.sum(mask) > 0:
                    result["by_gold_diff"][name] = self._analyze_subset(mask)

        # Analyze by event type
        if self.event_type is not None:
            unique_events = np.unique(self.event_type)
            for event in unique_events:
                mask = self.event_type == event
                if np.sum(mask) > 0:
                    result["by_event_type"][str(event)] = self._analyze_subset(mask)

        return result
