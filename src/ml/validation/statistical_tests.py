# src/ml/validation/statistical_tests.py
"""Statistical tests for calibration assessment."""

import numpy as np
from scipy import stats


def hosmer_lemeshow_test(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, float]:
    """Hosmer-Lemeshow goodness-of-fit test for calibration.

    Tests null hypothesis that the model is well-calibrated.
    Low p-value = reject calibration hypothesis = bad calibration.

    Args:
        y_true: Binary outcomes
        y_pred: Predicted probabilities
        n_bins: Number of bins (typically 10)

    Returns:
        Tuple of (chi-square statistic, p-value)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Bin by predicted probability
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred, bin_edges[1:-1])

    chi_square = 0.0
    df = 0

    for i in range(n_bins):
        mask = bin_indices == i
        n_k = np.sum(mask)

        if n_k == 0:
            continue

        # Observed and expected positives
        o_k = np.sum(y_true[mask])
        e_k = np.sum(y_pred[mask])

        # Avoid division by zero
        if e_k > 0 and e_k < n_k:
            chi_square += (o_k - e_k) ** 2 / (e_k * (1 - e_k / n_k))
            df += 1

    # Degrees of freedom = n_bins - 2 (for HL test)
    df = max(df - 2, 1)
    pvalue = 1 - stats.chi2.cdf(chi_square, df)

    return chi_square, pvalue


def bootstrap_brier_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Bootstrap confidence interval for Brier score.

    Args:
        y_true: Binary outcomes
        y_pred: Predicted probabilities
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (default 0.95)

    Returns:
        Dict with lower, upper, mean Brier scores
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)

    brier_scores = []
    rng = np.random.default_rng(42)

    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        brier = np.mean((y_pred[indices] - y_true[indices]) ** 2)
        brier_scores.append(brier)

    brier_scores = np.array(brier_scores)
    alpha = 1 - confidence

    return {
        "lower": np.percentile(brier_scores, 100 * alpha / 2),
        "upper": np.percentile(brier_scores, 100 * (1 - alpha / 2)),
        "mean": np.mean(brier_scores),
    }
