# tests/ml/validation/test_statistical.py
"""Tests for statistical calibration tests."""

import pytest
import numpy as np

from src.ml.validation.statistical_tests import (
    hosmer_lemeshow_test,
    bootstrap_brier_ci,
)


class TestHosmerLemeshow:
    def test_perfect_calibration_high_pvalue(self):
        """Well-calibrated predictions should have high p-value."""
        np.random.seed(42)
        # Generate well-calibrated predictions
        y_pred = np.random.uniform(0.3, 0.7, 200)
        y_true = (np.random.random(200) < y_pred).astype(int)

        stat, pvalue = hosmer_lemeshow_test(y_true, y_pred)

        assert pvalue > 0.05  # Fail to reject calibration

    def test_bad_calibration_low_pvalue(self):
        """Miscalibrated predictions should have low p-value."""
        np.random.seed(42)
        y_true = np.array([1] * 30 + [0] * 70)  # 30% win rate
        y_pred = np.array([0.9] * 100)  # Always predicts 90%

        stat, pvalue = hosmer_lemeshow_test(y_true, y_pred)

        assert pvalue < 0.05  # Reject calibration

    def test_returns_statistic_and_pvalue(self):
        """Function returns both test statistic and p-value."""
        y_true = np.random.randint(0, 2, 100)
        y_pred = np.random.uniform(0, 1, 100)

        result = hosmer_lemeshow_test(y_true, y_pred)

        assert len(result) == 2
        assert result[0] >= 0  # Chi-square statistic is non-negative


class TestBootstrapCI:
    def test_returns_confidence_interval(self):
        """bootstrap_brier_ci returns lower and upper bounds."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_pred = np.random.uniform(0, 1, 100)

        ci = bootstrap_brier_ci(y_true, y_pred, n_bootstrap=100)

        assert "lower" in ci
        assert "upper" in ci
        assert "mean" in ci
        assert ci["lower"] <= ci["mean"] <= ci["upper"]

    def test_narrow_ci_with_consistent_predictions(self):
        """Consistent predictions should have narrow CI."""
        np.random.seed(42)
        y_pred = np.array([0.5] * 200)
        y_true = np.random.randint(0, 2, 200)

        ci = bootstrap_brier_ci(y_true, y_pred, n_bootstrap=500)

        ci_width = ci["upper"] - ci["lower"]
        assert ci_width < 0.1  # Narrow interval
