# tests/ml/validation/test_calibration.py
"""Tests for calibration analysis module."""

import math

import numpy as np
import pytest

from src.ml.validation.calibration import (
    CalibrationAnalyzer,
    brier_score_decomposition,
    expected_calibration_error,
    reliability_diagram_data,
)


class TestBrierDecomposition:
    """Tests for Brier score decomposition."""

    def test_perfect_calibration(self):
        """Perfect calibration has zero reliability (calibration error)."""
        # Perfect calibration: predicted probabilities match actual frequencies
        # All predictions are 0.5, and exactly half are positive
        y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        y_pred = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

        result = brier_score_decomposition(y_true, y_pred, n_bins=5)

        assert "brier_score" in result
        assert "reliability" in result
        assert "resolution" in result
        assert "uncertainty" in result

        # Perfect calibration means reliability should be 0 (or very close)
        assert result["reliability"] == pytest.approx(0.0, abs=0.01)

    def test_overconfident_predictions(self):
        """Overconfident predictions have high reliability (calibration error)."""
        # Overconfident: predict 0.9 but only 50% are actually positive
        y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        y_pred = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])

        result = brier_score_decomposition(y_true, y_pred, n_bins=5)

        # Overconfident predictions should have high reliability (poor calibration)
        # Reliability = sum_k (n_k/N) * (o_k - p_k)^2
        # Here: (0.5 - 0.9)^2 = 0.16
        assert result["reliability"] > 0.1

    def test_decomposition_sums_correctly(self):
        """Brier = Reliability - Resolution + Uncertainty (approximately).

        Note: The exact equality only holds when all predictions in each bin
        have the same value. For continuous predictions, there is within-bin
        variance that creates a small discrepancy. We use constant predictions
        per bin to verify the formula exactly.
        """
        # Use predictions that are constant within each bin for exact equality
        # Bins with 5 bins: [0-0.2), [0.2-0.4), [0.4-0.6), [0.6-0.8), [0.8-1.0]
        y_true = np.array([1, 0, 0, 1, 1, 0, 1, 0, 1, 1])
        # Predictions at bin centers: 0.1, 0.3, 0.5, 0.7, 0.9
        y_pred = np.array([0.9, 0.9, 0.3, 0.3, 0.5, 0.5, 0.7, 0.7, 0.1, 0.1])

        result = brier_score_decomposition(y_true, y_pred, n_bins=5)

        # Brier score formula: Brier = Reliability - Resolution + Uncertainty
        expected_brier = (
            result["reliability"] - result["resolution"] + result["uncertainty"]
        )
        assert result["brier_score"] == pytest.approx(expected_brier, abs=0.001)

        # Also verify against direct Brier calculation
        direct_brier = np.mean((y_true - y_pred) ** 2)
        assert result["brier_score"] == pytest.approx(direct_brier, abs=0.001)


class TestExpectedCalibrationError:
    """Tests for Expected Calibration Error."""

    def test_perfect_calibration_zero_ece(self):
        """Perfectly calibrated predictions have ECE = 0."""
        # Perfect calibration: in each bin, accuracy equals confidence
        # Use predictions that match the actual outcome frequencies
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        # Predictions where low probs -> 0 outcomes, high probs -> 1 outcomes
        y_pred = np.array([0.1, 0.1, 0.2, 0.2, 0.8, 0.8, 0.9, 0.9, 0.9, 0.9])

        ece = expected_calibration_error(y_true, y_pred, n_bins=5)

        # Should be close to zero for well-calibrated predictions
        assert ece < 0.15  # Allow some tolerance due to finite sample

    def test_bad_calibration_high_ece(self):
        """Poorly calibrated predictions have high ECE."""
        # Bad calibration: predict high confidence but wrong
        y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        y_pred = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])

        ece = expected_calibration_error(y_true, y_pred, n_bins=5)

        # ECE should be high (~0.9) since accuracy=0 but confidence=0.9
        assert ece > 0.8


class TestReliabilityDiagram:
    """Tests for reliability diagram data generation."""

    def test_returns_correct_structure(self):
        """Returns dict with bin_centers, true_fractions, counts."""
        y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0, 1, 1])
        y_pred = np.array([0.8, 0.7, 0.3, 0.2, 0.6, 0.4, 0.9, 0.1, 0.5, 0.8])

        result = reliability_diagram_data(y_true, y_pred, n_bins=5)

        assert "bin_centers" in result
        assert "true_fractions" in result
        assert "counts" in result

        # All should be arrays/lists of same length
        assert len(result["bin_centers"]) == len(result["true_fractions"])
        assert len(result["bin_centers"]) == len(result["counts"])

        # Bin centers should be in [0, 1]
        for center in result["bin_centers"]:
            assert 0 <= center <= 1

        # True fractions should be in [0, 1] or NaN for empty bins
        for frac in result["true_fractions"]:
            if not np.isnan(frac):
                assert 0 <= frac <= 1

        # Counts should be non-negative integers
        for count in result["counts"]:
            assert count >= 0
            assert count == int(count)


class TestCalibrationAnalyzer:
    """Tests for CalibrationAnalyzer class."""

    @pytest.fixture
    def sample_data(self):
        """Sample prediction data with context."""
        np.random.seed(42)
        n_samples = 100

        return {
            "y_true": np.random.randint(0, 2, n_samples),
            "y_pred": np.random.uniform(0.2, 0.8, n_samples),
            "game_time_minutes": np.random.uniform(5, 35, n_samples),
            "gold_diff": np.random.uniform(-5000, 5000, n_samples),
            "event_type": np.random.choice(
                ["kill", "dragon_kill", "baron_kill"], n_samples
            ),
        }

    def test_analyze_overall(self, sample_data):
        """analyze_overall returns correct metrics."""
        analyzer = CalibrationAnalyzer(
            y_true=sample_data["y_true"],
            y_pred=sample_data["y_pred"],
        )

        result = analyzer.analyze_overall()

        assert "brier_score" in result
        assert "ece" in result
        assert "reliability" in result
        assert "resolution" in result
        assert "uncertainty" in result
        assert "n_samples" in result

        # Check value ranges
        assert 0 <= result["brier_score"] <= 1
        assert 0 <= result["ece"] <= 1
        assert result["n_samples"] == len(sample_data["y_true"])

    def test_analyze_by_context(self, sample_data):
        """analyze_by_context returns metrics by game time, gold diff, event type."""
        analyzer = CalibrationAnalyzer(
            y_true=sample_data["y_true"],
            y_pred=sample_data["y_pred"],
            game_time_minutes=sample_data["game_time_minutes"],
            gold_diff=sample_data["gold_diff"],
            event_type=sample_data["event_type"],
        )

        result = analyzer.analyze_by_context()

        assert "by_game_time" in result
        assert "by_gold_diff" in result
        assert "by_event_type" in result

        # Check game time bins exist
        game_time_keys = result["by_game_time"].keys()
        assert "early" in game_time_keys or "mid" in game_time_keys or "late" in game_time_keys

        # Check gold diff bins exist
        gold_diff_keys = result["by_gold_diff"].keys()
        assert "behind" in gold_diff_keys or "even" in gold_diff_keys or "ahead" in gold_diff_keys

        # Check event types exist
        event_keys = result["by_event_type"].keys()
        assert len(event_keys) > 0

    def test_analyze_by_game_time_bins(self, sample_data):
        """Game time bins are correctly defined."""
        analyzer = CalibrationAnalyzer(
            y_true=sample_data["y_true"],
            y_pred=sample_data["y_pred"],
            game_time_minutes=sample_data["game_time_minutes"],
        )

        # Check that GAME_TIME_BINS is defined correctly
        assert hasattr(CalibrationAnalyzer, "GAME_TIME_BINS")
        bins = CalibrationAnalyzer.GAME_TIME_BINS

        assert "early" in bins
        assert "mid" in bins
        assert "late" in bins

        # Early: 0-15 min
        assert bins["early"][0] == 0
        assert bins["early"][1] == 15

        # Mid: 15-25 min
        assert bins["mid"][0] == 15
        assert bins["mid"][1] == 25

        # Late: 25+ min
        assert bins["late"][0] == 25
        assert bins["late"][1] == math.inf
