# Model Validation & Paper Trading — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a complete validation suite and paper trading system to prove model calibration and measure realized edge before going live.

**Architecture:** (1) Calibration module with Brier decomposition and reliability diagrams, (2) Paper trading engine with live Polymarket price capture at T+0/30/60/120s, (3) Streamlit dashboard for monitoring.

**Tech Stack:** Python 3.11+, XGBoost, Streamlit, Plotly, scipy, SQLite, Telegram

---

## Project Structure (Extended)

```
poly/
├── src/
│   ├── ml/
│   │   ├── validation/
│   │   │   ├── __init__.py
│   │   │   ├── calibration.py       # NEW: Brier decomposition, ECE
│   │   │   ├── statistical_tests.py # NEW: Hosmer-Lemeshow, bootstrap
│   │   │   └── report.py            # NEW: HTML report generation
│   │   └── ...
│   ├── paper_trading/
│   │   ├── __init__.py              # NEW
│   │   ├── engine.py                # NEW: Main orchestrator
│   │   ├── market_observer.py       # NEW: Polymarket price capture
│   │   ├── position_manager.py      # NEW: Kelly sizing, risk limits
│   │   ├── execution_sim.py         # NEW: Slippage simulation
│   │   ├── metrics.py               # NEW: P&L calculations
│   │   └── dashboard.py             # NEW: Streamlit app
│   └── db/
│       └── models.py                # MODIFY: Add LiveObservation, PaperTrade
├── scripts/
│   ├── validate_model.py            # NEW
│   ├── paper_trade.py               # NEW
│   └── performance_report.py        # NEW
└── tests/
    ├── ml/
    │   └── validation/
    │       ├── test_calibration.py  # NEW
    │       └── test_statistical.py  # NEW
    └── paper_trading/
        ├── test_position_manager.py # NEW
        ├── test_execution_sim.py    # NEW
        └── test_metrics.py          # NEW
```

---

## Task 1: Add Dependencies

**Files:**
- Modify: `requirements.txt`

**Step 1: Add new dependencies**

Add to requirements.txt:

```
# Validation & Dashboard
streamlit>=1.30.0
plotly>=5.18.0
scipy>=1.12.0
jinja2>=3.1.0
```

**Step 2: Install dependencies**

Run: `uv pip install -r requirements.txt`
Expected: Successfully installed packages

**Step 3: Commit**

```bash
git add requirements.txt
git commit -m "deps: add streamlit, plotly, scipy for validation dashboard"
```

---

## Task 2: Database Models for Paper Trading

**Files:**
- Modify: `src/db/models.py`
- Test: `tests/db/test_paper_trading_models.py`

**Step 1: Write the failing test**

```python
# tests/db/test_paper_trading_models.py
"""Tests for paper trading database models."""

import pytest
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.models import Base, LiveObservation, PaperTrade


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


class TestLiveObservation:
    def test_create_observation(self, db_session):
        obs = LiveObservation(
            match_id="pandascore_12345",
            event_type="baron_kill",
            game_state={"gold_diff": 5000, "game_time": 25},
            model_prediction=0.72,
            polymarket_price=0.58,
        )
        db_session.add(obs)
        db_session.commit()

        assert obs.id is not None
        assert obs.model_prediction == 0.72
        assert obs.edge_theoretical == pytest.approx(0.14, rel=0.01)

    def test_observation_with_followup_prices(self, db_session):
        obs = LiveObservation(
            match_id="pandascore_12345",
            event_type="dragon_kill",
            game_state={"gold_diff": 2000},
            model_prediction=0.65,
            polymarket_price=0.55,
            polymarket_price_30s=0.58,
            polymarket_price_60s=0.62,
            polymarket_price_120s=0.64,
        )
        db_session.add(obs)
        db_session.commit()

        assert obs.polymarket_price_120s == 0.64


class TestPaperTrade:
    def test_create_paper_trade(self, db_session):
        obs = LiveObservation(
            match_id="pandascore_12345",
            event_type="baron_kill",
            game_state={},
            model_prediction=0.72,
            polymarket_price=0.58,
        )
        db_session.add(obs)
        db_session.commit()

        trade = PaperTrade(
            observation_id=obs.id,
            side="BUY",
            entry_price=0.58,
            simulated_fill_price=0.585,
            size=50.0,
            edge_theoretical=0.14,
        )
        db_session.add(trade)
        db_session.commit()

        assert trade.id is not None
        assert trade.size == 50.0

    def test_paper_trade_pnl_calculation(self, db_session):
        trade = PaperTrade(
            observation_id=1,
            side="BUY",
            entry_price=0.58,
            simulated_fill_price=0.585,
            size=50.0,
            edge_theoretical=0.14,
            exit_price=0.68,
            pnl=8.12,  # (0.68 - 0.585) * 50 * (1/0.585)
        )
        db_session.add(trade)
        db_session.commit()

        assert trade.pnl == 8.12
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/db/test_paper_trading_models.py -v`
Expected: FAIL with "cannot import name 'LiveObservation'"

**Step 3: Write the implementation**

Add to `src/db/models.py` after the existing models:

```python
class LiveObservation(Base):
    """Live observations for paper trading."""

    __tablename__ = "live_observations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    match_id = Column(String(255), nullable=False, index=True)
    event_type = Column(String(100), nullable=False)
    game_state = Column(JSON, nullable=False)

    # Model prediction
    model_prediction = Column(Float, nullable=False)

    # Market prices at different times
    polymarket_price = Column(Float, nullable=True)
    polymarket_price_30s = Column(Float, nullable=True)
    polymarket_price_60s = Column(Float, nullable=True)
    polymarket_price_120s = Column(Float, nullable=True)

    # Result
    actual_winner = Column(String(255), nullable=True)
    latency_ms = Column(Integer, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    @property
    def edge_theoretical(self) -> float:
        """Calculate theoretical edge."""
        if self.polymarket_price is None:
            return 0.0
        return self.model_prediction - self.polymarket_price

    def __repr__(self) -> str:
        return f"<LiveObservation(id={self.id}, match={self.match_id}, pred={self.model_prediction:.2f})>"


class PaperTrade(Base):
    """Simulated trades for paper trading."""

    __tablename__ = "paper_trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    observation_id = Column(Integer, nullable=False, index=True)

    side = Column(String(10), nullable=False)  # BUY or SELL
    entry_price = Column(Float, nullable=False)
    simulated_fill_price = Column(Float, nullable=False)
    size = Column(Float, nullable=False)

    edge_theoretical = Column(Float, nullable=False)
    edge_realized = Column(Float, nullable=True)
    exit_price = Column(Float, nullable=True)
    pnl = Column(Float, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<PaperTrade(id={self.id}, side={self.side}, size={self.size}, pnl={self.pnl})>"
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/db/test_paper_trading_models.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/db/models.py tests/db/test_paper_trading_models.py
git commit -m "feat: add LiveObservation and PaperTrade models"
```

---

## Task 3: Calibration Module — Brier Decomposition

**Files:**
- Create: `src/ml/validation/__init__.py`
- Create: `src/ml/validation/calibration.py`
- Test: `tests/ml/validation/test_calibration.py`

**Step 1: Write the failing test**

```python
# tests/ml/validation/test_calibration.py
"""Tests for calibration analysis."""

import pytest
import numpy as np
import pandas as pd

from src.ml.validation.calibration import (
    CalibrationAnalyzer,
    brier_score_decomposition,
    expected_calibration_error,
    reliability_diagram_data,
)


class TestBrierDecomposition:
    def test_perfect_calibration(self):
        """Perfectly calibrated predictions should have 0 reliability."""
        # 100 predictions of 0.7, 70% actually win
        y_true = np.array([1] * 70 + [0] * 30)
        y_pred = np.array([0.7] * 100)

        decomp = brier_score_decomposition(y_true, y_pred)

        assert decomp["reliability"] == pytest.approx(0.0, abs=0.01)
        assert decomp["brier_score"] == pytest.approx(0.21, abs=0.01)

    def test_overconfident_predictions(self):
        """Overconfident predictions should have high reliability error."""
        # Predicts 0.9 but only 60% win
        y_true = np.array([1] * 60 + [0] * 40)
        y_pred = np.array([0.9] * 100)

        decomp = brier_score_decomposition(y_true, y_pred)

        assert decomp["reliability"] > 0.05  # High miscalibration

    def test_decomposition_sums_correctly(self):
        """Brier = Reliability - Resolution + Uncertainty."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 200)
        y_pred = np.random.uniform(0.3, 0.7, 200)

        decomp = brier_score_decomposition(y_true, y_pred)

        expected_brier = (
            decomp["reliability"]
            - decomp["resolution"]
            + decomp["uncertainty"]
        )
        assert decomp["brier_score"] == pytest.approx(expected_brier, abs=0.01)


class TestExpectedCalibrationError:
    def test_perfect_calibration_zero_ece(self):
        """Perfect calibration should have ECE near 0."""
        y_true = np.array([1] * 70 + [0] * 30)
        y_pred = np.array([0.7] * 100)

        ece = expected_calibration_error(y_true, y_pred, n_bins=10)

        assert ece == pytest.approx(0.0, abs=0.01)

    def test_bad_calibration_high_ece(self):
        """Bad calibration should have high ECE."""
        y_true = np.array([1] * 30 + [0] * 70)  # 30% win rate
        y_pred = np.array([0.8] * 100)  # Predicts 80%

        ece = expected_calibration_error(y_true, y_pred, n_bins=10)

        assert ece > 0.4  # Very miscalibrated


class TestReliabilityDiagram:
    def test_returns_correct_structure(self):
        """reliability_diagram_data returns bins with counts."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_pred = np.random.uniform(0, 1, 100)

        data = reliability_diagram_data(y_true, y_pred, n_bins=10)

        assert "bin_centers" in data
        assert "true_fractions" in data
        assert "counts" in data
        assert len(data["bin_centers"]) == 10


class TestCalibrationAnalyzer:
    @pytest.fixture
    def sample_predictions(self):
        """Sample predictions DataFrame."""
        np.random.seed(42)
        n = 500
        return pd.DataFrame({
            "y_true": np.random.randint(0, 2, n),
            "y_pred": np.clip(np.random.normal(0.5, 0.2, n), 0.01, 0.99),
            "game_time_minutes": np.random.uniform(5, 40, n),
            "gold_diff": np.random.uniform(-10000, 10000, n),
            "event_type": np.random.choice(
                ["kill", "baron_kill", "dragon_kill"], n
            ),
        })

    def test_analyze_overall(self, sample_predictions):
        """Analyzer computes overall metrics."""
        analyzer = CalibrationAnalyzer(sample_predictions)
        results = analyzer.analyze_overall()

        assert "brier_score" in results
        assert "ece" in results
        assert "reliability" in results
        assert 0 <= results["brier_score"] <= 1

    def test_analyze_by_context(self, sample_predictions):
        """Analyzer computes metrics by game context."""
        analyzer = CalibrationAnalyzer(sample_predictions)
        results = analyzer.analyze_by_context()

        assert "by_game_time" in results
        assert "by_gold_diff" in results
        assert "by_event_type" in results

    def test_analyze_by_game_time_bins(self, sample_predictions):
        """Game time analysis uses correct bins."""
        analyzer = CalibrationAnalyzer(sample_predictions)
        results = analyzer.analyze_by_context()

        game_time_results = results["by_game_time"]
        assert "early" in game_time_results  # 0-15 min
        assert "mid" in game_time_results    # 15-25 min
        assert "late" in game_time_results   # 25+ min
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ml/validation/test_calibration.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Create package init**

```python
# src/ml/validation/__init__.py
"""Model validation tools."""

from .calibration import (
    CalibrationAnalyzer,
    brier_score_decomposition,
    expected_calibration_error,
    reliability_diagram_data,
)

__all__ = [
    "CalibrationAnalyzer",
    "brier_score_decomposition",
    "expected_calibration_error",
    "reliability_diagram_data",
]
```

**Step 4: Write the implementation**

```python
# src/ml/validation/calibration.py
"""Calibration analysis for probability predictions.

Implements Brier score decomposition, Expected Calibration Error (ECE),
and reliability diagram generation.
"""

from typing import Any

import numpy as np
import pandas as pd


def brier_score_decomposition(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> dict[str, float]:
    """Decompose Brier score into reliability, resolution, uncertainty.

    Brier = Reliability - Resolution + Uncertainty

    - Reliability: How close predictions are to observed frequencies (lower is better)
    - Resolution: How much predictions differ from base rate (higher is better)
    - Uncertainty: Inherent uncertainty in the data (fixed)

    Args:
        y_true: Binary outcomes (0 or 1)
        y_pred: Predicted probabilities
        n_bins: Number of bins for grouping predictions

    Returns:
        Dict with brier_score, reliability, resolution, uncertainty
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Overall Brier score
    brier = np.mean((y_pred - y_true) ** 2)

    # Base rate (climatological probability)
    base_rate = np.mean(y_true)
    uncertainty = base_rate * (1 - base_rate)

    # Bin predictions
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred, bin_edges[1:-1])

    reliability = 0.0
    resolution = 0.0

    for i in range(n_bins):
        mask = bin_indices == i
        if not np.any(mask):
            continue

        n_k = np.sum(mask)
        o_k = np.mean(y_true[mask])  # Observed frequency in bin
        f_k = np.mean(y_pred[mask])  # Mean prediction in bin

        reliability += n_k * (f_k - o_k) ** 2
        resolution += n_k * (o_k - base_rate) ** 2

    n = len(y_true)
    reliability /= n
    resolution /= n

    return {
        "brier_score": brier,
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
    }


def expected_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Calculate Expected Calibration Error (ECE).

    ECE = sum_k (n_k / N) * |accuracy_k - confidence_k|

    Lower is better. 0 = perfectly calibrated.

    Args:
        y_true: Binary outcomes
        y_pred: Predicted probabilities
        n_bins: Number of bins

    Returns:
        ECE value between 0 and 1
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred, bin_edges[1:-1])

    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        mask = bin_indices == i
        if not np.any(mask):
            continue

        n_k = np.sum(mask)
        accuracy_k = np.mean(y_true[mask])
        confidence_k = np.mean(y_pred[mask])

        ece += (n_k / n) * np.abs(accuracy_k - confidence_k)

    return ece


def reliability_diagram_data(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> dict[str, np.ndarray]:
    """Generate data for reliability diagram.

    Args:
        y_true: Binary outcomes
        y_pred: Predicted probabilities
        n_bins: Number of bins

    Returns:
        Dict with bin_centers, true_fractions, counts
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_indices = np.digitize(y_pred, bin_edges[1:-1])

    true_fractions = np.zeros(n_bins)
    counts = np.zeros(n_bins)

    for i in range(n_bins):
        mask = bin_indices == i
        counts[i] = np.sum(mask)
        if counts[i] > 0:
            true_fractions[i] = np.mean(y_true[mask])

    return {
        "bin_centers": bin_centers,
        "true_fractions": true_fractions,
        "counts": counts,
    }


class CalibrationAnalyzer:
    """Analyze model calibration across different contexts."""

    GAME_TIME_BINS = {
        "early": (0, 15),
        "mid": (15, 25),
        "late": (25, float("inf")),
    }

    GOLD_DIFF_BINS = {
        "behind": (float("-inf"), -2000),
        "even": (-2000, 2000),
        "ahead": (2000, float("inf")),
    }

    def __init__(self, df: pd.DataFrame):
        """Initialize with predictions DataFrame.

        Required columns: y_true, y_pred
        Optional columns: game_time_minutes, gold_diff, event_type
        """
        self.df = df.copy()

    def analyze_overall(self) -> dict[str, float]:
        """Compute overall calibration metrics."""
        y_true = self.df["y_true"].values
        y_pred = self.df["y_pred"].values

        decomp = brier_score_decomposition(y_true, y_pred)
        ece = expected_calibration_error(y_true, y_pred)

        return {
            "brier_score": decomp["brier_score"],
            "reliability": decomp["reliability"],
            "resolution": decomp["resolution"],
            "uncertainty": decomp["uncertainty"],
            "ece": ece,
            "n_samples": len(y_true),
        }

    def analyze_by_context(self) -> dict[str, dict[str, Any]]:
        """Compute calibration metrics by game context."""
        results = {}

        # By game time
        if "game_time_minutes" in self.df.columns:
            results["by_game_time"] = self._analyze_by_bins(
                "game_time_minutes", self.GAME_TIME_BINS
            )

        # By gold diff
        if "gold_diff" in self.df.columns:
            results["by_gold_diff"] = self._analyze_by_bins(
                "gold_diff", self.GOLD_DIFF_BINS
            )

        # By event type
        if "event_type" in self.df.columns:
            results["by_event_type"] = {}
            for event_type in self.df["event_type"].unique():
                mask = self.df["event_type"] == event_type
                subset = self.df[mask]
                if len(subset) >= 10:
                    results["by_event_type"][event_type] = self._compute_metrics(
                        subset["y_true"].values,
                        subset["y_pred"].values,
                    )

        return results

    def _analyze_by_bins(
        self,
        column: str,
        bins: dict[str, tuple[float, float]],
    ) -> dict[str, dict[str, float]]:
        """Analyze by custom bins."""
        results = {}
        for name, (low, high) in bins.items():
            mask = (self.df[column] >= low) & (self.df[column] < high)
            subset = self.df[mask]
            if len(subset) >= 10:
                results[name] = self._compute_metrics(
                    subset["y_true"].values,
                    subset["y_pred"].values,
                )
        return results

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[str, float]:
        """Compute metrics for a subset."""
        decomp = brier_score_decomposition(y_true, y_pred)
        return {
            "brier_score": decomp["brier_score"],
            "ece": expected_calibration_error(y_true, y_pred),
            "n_samples": len(y_true),
        }
```

**Step 5: Create test directory**

```bash
mkdir -p tests/ml/validation
touch tests/ml/validation/__init__.py
```

**Step 6: Run test to verify it passes**

Run: `uv run pytest tests/ml/validation/test_calibration.py -v`
Expected: PASS

**Step 7: Commit**

```bash
git add src/ml/validation/ tests/ml/validation/
git commit -m "feat: add calibration analysis with Brier decomposition and ECE"
```

---

## Task 4: Statistical Tests

**Files:**
- Create: `src/ml/validation/statistical_tests.py`
- Test: `tests/ml/validation/test_statistical.py`

**Step 1: Write the failing test**

```python
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
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ml/validation/test_statistical.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write the implementation**

```python
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
```

**Step 4: Update __init__.py**

Add to `src/ml/validation/__init__.py`:

```python
from .statistical_tests import hosmer_lemeshow_test, bootstrap_brier_ci

__all__ = [
    # ... existing exports
    "hosmer_lemeshow_test",
    "bootstrap_brier_ci",
]
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/ml/validation/test_statistical.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/ml/validation/statistical_tests.py src/ml/validation/__init__.py tests/ml/validation/test_statistical.py
git commit -m "feat: add Hosmer-Lemeshow test and bootstrap CI for calibration"
```

---

## Task 5: Report Generation

**Files:**
- Create: `src/ml/validation/report.py`
- Create: `templates/validation_report.html`
- Test: `tests/ml/validation/test_report.py`

**Step 1: Write the failing test**

```python
# tests/ml/validation/test_report.py
"""Tests for report generation."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile

from src.ml.validation.report import ValidationReport


class TestValidationReport:
    @pytest.fixture
    def sample_df(self):
        np.random.seed(42)
        n = 200
        return pd.DataFrame({
            "y_true": np.random.randint(0, 2, n),
            "y_pred": np.clip(np.random.normal(0.5, 0.2, n), 0.01, 0.99),
            "game_time_minutes": np.random.uniform(5, 40, n),
            "gold_diff": np.random.uniform(-10000, 10000, n),
            "event_type": np.random.choice(["kill", "baron_kill"], n),
        })

    def test_generate_html_report(self, sample_df):
        """Report generates valid HTML."""
        report = ValidationReport(sample_df)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"
            report.generate_html(output_path)

            assert output_path.exists()
            content = output_path.read_text()
            assert "<html>" in content
            assert "Brier Score" in content

    def test_report_includes_plots(self, sample_df):
        """Report includes reliability diagram."""
        report = ValidationReport(sample_df)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"
            report.generate_html(output_path)

            content = output_path.read_text()
            # Plotly embeds charts as divs
            assert "plotly" in content.lower() or "svg" in content.lower()

    def test_report_has_recommendation(self, sample_df):
        """Report includes go/no-go recommendation."""
        report = ValidationReport(sample_df)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"
            report.generate_html(output_path)

            content = output_path.read_text()
            assert "GO" in content or "NO-GO" in content

    def test_get_summary_dict(self, sample_df):
        """get_summary returns metrics dict."""
        report = ValidationReport(sample_df)
        summary = report.get_summary()

        assert "brier_score" in summary
        assert "ece" in summary
        assert "recommendation" in summary
        assert summary["recommendation"] in ["GO", "NO-GO"]
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ml/validation/test_report.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write the implementation**

```python
# src/ml/validation/report.py
"""Generate validation reports with plots and recommendations."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from jinja2 import Template

from .calibration import (
    CalibrationAnalyzer,
    reliability_diagram_data,
    expected_calibration_error,
)
from .statistical_tests import hosmer_lemeshow_test, bootstrap_brier_ci


# Thresholds for go/no-go
BRIER_THRESHOLD = 0.25
ECE_THRESHOLD = 0.08


class ValidationReport:
    """Generate calibration validation report."""

    def __init__(self, df: pd.DataFrame):
        """Initialize with predictions DataFrame."""
        self.df = df
        self.analyzer = CalibrationAnalyzer(df)

    def get_summary(self) -> dict[str, Any]:
        """Get summary metrics."""
        overall = self.analyzer.analyze_overall()
        ci = bootstrap_brier_ci(
            self.df["y_true"].values,
            self.df["y_pred"].values,
        )
        hl_stat, hl_pvalue = hosmer_lemeshow_test(
            self.df["y_true"].values,
            self.df["y_pred"].values,
        )

        # Determine recommendation
        is_calibrated = (
            overall["brier_score"] < BRIER_THRESHOLD
            and overall["ece"] < ECE_THRESHOLD
            and hl_pvalue > 0.05
        )

        return {
            "brier_score": overall["brier_score"],
            "brier_ci_lower": ci["lower"],
            "brier_ci_upper": ci["upper"],
            "ece": overall["ece"],
            "reliability": overall["reliability"],
            "resolution": overall["resolution"],
            "hl_pvalue": hl_pvalue,
            "n_samples": overall["n_samples"],
            "recommendation": "GO" if is_calibrated else "NO-GO",
        }

    def _create_reliability_plot(self) -> str:
        """Create reliability diagram as HTML."""
        data = reliability_diagram_data(
            self.df["y_true"].values,
            self.df["y_pred"].values,
        )

        fig = go.Figure()

        # Perfect calibration line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect calibration",
            line=dict(dash="dash", color="gray"),
        ))

        # Actual calibration
        fig.add_trace(go.Scatter(
            x=data["bin_centers"],
            y=data["true_fractions"],
            mode="lines+markers",
            name="Model calibration",
            marker=dict(size=10),
            line=dict(color="blue"),
        ))

        fig.update_layout(
            title="Reliability Diagram",
            xaxis_title="Predicted Probability",
            yaxis_title="Observed Frequency",
            width=600,
            height=500,
        )

        return fig.to_html(include_plotlyjs="cdn", full_html=False)

    def _create_context_heatmap(self) -> str:
        """Create calibration heatmap by context."""
        context = self.analyzer.analyze_by_context()

        if "by_game_time" not in context or "by_gold_diff" not in context:
            return "<p>Insufficient context data for heatmap</p>"

        # Build heatmap data
        game_times = ["early", "mid", "late"]
        gold_diffs = ["behind", "even", "ahead"]

        z = []
        for gt in game_times:
            row = []
            for gd in gold_diffs:
                # Filter and compute ECE for this combination
                gt_range = CalibrationAnalyzer.GAME_TIME_BINS[gt]
                gd_range = CalibrationAnalyzer.GOLD_DIFF_BINS[gd]

                mask = (
                    (self.df["game_time_minutes"] >= gt_range[0])
                    & (self.df["game_time_minutes"] < gt_range[1])
                    & (self.df["gold_diff"] >= gd_range[0])
                    & (self.df["gold_diff"] < gd_range[1])
                )
                subset = self.df[mask]

                if len(subset) >= 10:
                    ece = expected_calibration_error(
                        subset["y_true"].values,
                        subset["y_pred"].values,
                    )
                else:
                    ece = np.nan
                row.append(ece)
            z.append(row)

        fig = go.Figure(data=go.Heatmap(
            z=z,
            x=gold_diffs,
            y=game_times,
            colorscale="RdYlGn_r",  # Red = bad, green = good
            zmin=0,
            zmax=0.2,
            text=[[f"{v:.3f}" if not np.isnan(v) else "N/A" for v in row] for row in z],
            texttemplate="%{text}",
            colorbar=dict(title="ECE"),
        ))

        fig.update_layout(
            title="ECE by Game Context",
            xaxis_title="Gold Difference",
            yaxis_title="Game Time",
            width=500,
            height=400,
        )

        return fig.to_html(include_plotlyjs=False, full_html=False)

    def generate_html(self, output_path: Path) -> None:
        """Generate full HTML report."""
        summary = self.get_summary()
        reliability_plot = self._create_reliability_plot()
        context_heatmap = self._create_context_heatmap()

        template = Template(HTML_TEMPLATE)
        html = template.render(
            summary=summary,
            reliability_plot=reliability_plot,
            context_heatmap=context_heatmap,
        )

        output_path.write_text(html)


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Model Validation Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .metric { display: inline-block; padding: 20px; margin: 10px; background: #f5f5f5; border-radius: 8px; }
        .metric-value { font-size: 32px; font-weight: bold; }
        .metric-label { color: #666; }
        .go { color: #28a745; }
        .no-go { color: #dc3545; }
        .plots { display: flex; flex-wrap: wrap; gap: 20px; }
        h1 { border-bottom: 2px solid #333; padding-bottom: 10px; }
    </style>
</head>
<body>
    <h1>Model Validation Report</h1>

    <h2>Recommendation: <span class="{{ 'go' if summary.recommendation == 'GO' else 'no-go' }}">
        {{ summary.recommendation }}
    </span></h2>

    <h3>Key Metrics</h3>
    <div class="metrics">
        <div class="metric">
            <div class="metric-value">{{ "%.4f"|format(summary.brier_score) }}</div>
            <div class="metric-label">Brier Score</div>
            <div class="metric-label">[{{ "%.4f"|format(summary.brier_ci_lower) }} - {{ "%.4f"|format(summary.brier_ci_upper) }}]</div>
        </div>
        <div class="metric">
            <div class="metric-value">{{ "%.4f"|format(summary.ece) }}</div>
            <div class="metric-label">ECE</div>
        </div>
        <div class="metric">
            <div class="metric-value">{{ "%.4f"|format(summary.hl_pvalue) }}</div>
            <div class="metric-label">H-L p-value</div>
        </div>
        <div class="metric">
            <div class="metric-value">{{ summary.n_samples }}</div>
            <div class="metric-label">Samples</div>
        </div>
    </div>

    <h3>Calibration Analysis</h3>
    <div class="plots">
        <div>{{ reliability_plot | safe }}</div>
        <div>{{ context_heatmap | safe }}</div>
    </div>

    <h3>Thresholds</h3>
    <ul>
        <li>Brier Score: < 0.25 ✓</li>
        <li>ECE: < 0.08 ✓</li>
        <li>Hosmer-Lemeshow p-value: > 0.05 (fail to reject calibration)</li>
    </ul>
</body>
</html>
"""
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ml/validation/test_report.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/ml/validation/report.py tests/ml/validation/test_report.py
git commit -m "feat: add HTML validation report with reliability diagram"
```

---

## Task 6: Validate Model Script

**Files:**
- Create: `scripts/validate_model.py`

**Step 1: Write the script**

```python
#!/usr/bin/env python3
# scripts/validate_model.py
"""Validate ML model calibration.

Usage:
    uv run python scripts/validate_model.py \\
        --model models/lol_impact.pkl \\
        --data data/lol_training.csv \\
        --output reports/validation.html
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import structlog

from src.ml.train import ImpactModel
from src.ml.validation.report import ValidationReport

logger = structlog.get_logger()


def main():
    parser = argparse.ArgumentParser(description="Validate model calibration")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--data", type=str, required=True, help="Path to validation data CSV")
    parser.add_argument("--output", type=str, default="reports/validation.html", help="Output HTML path")

    args = parser.parse_args()

    model_path = Path(args.model)
    data_path = Path(args.data)
    output_path = Path(args.output)

    if not model_path.exists():
        logger.error("model_not_found", path=str(model_path))
        sys.exit(1)

    if not data_path.exists():
        logger.error("data_not_found", path=str(data_path))
        sys.exit(1)

    # Load model and data
    logger.info("loading_model", path=str(model_path))
    model = ImpactModel.load(model_path)

    logger.info("loading_data", path=str(data_path))
    df = pd.read_csv(data_path)

    # Generate predictions
    X = df.drop(columns=["label"])
    y_true = df["label"].values
    y_pred = model.predict_proba(X)

    # Prepare validation DataFrame
    val_df = df.copy()
    val_df["y_true"] = y_true
    val_df["y_pred"] = y_pred

    # Rename columns to match expected format
    if "game_time_minutes" not in val_df.columns and "game_time" in val_df.columns:
        val_df["game_time_minutes"] = val_df["game_time"]

    # Generate report
    logger.info("generating_report")
    report = ValidationReport(val_df)
    summary = report.get_summary()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.generate_html(output_path)

    # Print summary
    print("\n" + "=" * 50)
    print("MODEL VALIDATION REPORT")
    print("=" * 50)
    print(f"Brier Score:    {summary['brier_score']:.4f} [{summary['brier_ci_lower']:.4f} - {summary['brier_ci_upper']:.4f}]")
    print(f"ECE:            {summary['ece']:.4f}")
    print(f"H-L p-value:    {summary['hl_pvalue']:.4f}")
    print(f"Samples:        {summary['n_samples']}")
    print("-" * 50)
    print(f"Recommendation: {summary['recommendation']}")
    print("=" * 50)
    print(f"\nFull report saved to: {output_path}")


if __name__ == "__main__":
    main()
```

**Step 2: Make executable**

```bash
chmod +x scripts/validate_model.py
```

**Step 3: Commit**

```bash
git add scripts/validate_model.py
git commit -m "feat: add validate_model.py CLI script"
```

---

## Task 7: Position Manager

**Files:**
- Create: `src/paper_trading/__init__.py`
- Create: `src/paper_trading/position_manager.py`
- Test: `tests/paper_trading/test_position_manager.py`

**Step 1: Write the failing test**

```python
# tests/paper_trading/test_position_manager.py
"""Tests for position manager."""

import pytest

from src.paper_trading.position_manager import PositionManager, kelly_fraction


class TestKellyFraction:
    def test_positive_edge_positive_fraction(self):
        """Positive edge should give positive Kelly fraction."""
        # Our prob: 60%, market: 50% → edge = 10%
        fraction = kelly_fraction(our_prob=0.60, market_price=0.50)
        assert fraction > 0
        assert fraction < 1

    def test_no_edge_zero_fraction(self):
        """No edge should give zero fraction."""
        fraction = kelly_fraction(our_prob=0.50, market_price=0.50)
        assert fraction == pytest.approx(0.0, abs=0.01)

    def test_negative_edge_zero_fraction(self):
        """Negative edge should return 0 (don't bet)."""
        fraction = kelly_fraction(our_prob=0.40, market_price=0.50)
        assert fraction == 0

    def test_capped_at_max(self):
        """Kelly fraction is capped at 25%."""
        # Huge edge
        fraction = kelly_fraction(our_prob=0.99, market_price=0.10)
        assert fraction <= 0.25


class TestPositionManager:
    @pytest.fixture
    def manager(self):
        return PositionManager(
            capital=10000,
            max_position_pct=0.25,
            min_edge=0.05,
            max_daily_loss_pct=0.10,
        )

    def test_calculate_position_size(self, manager):
        """Calculate position size based on Kelly."""
        size = manager.calculate_position_size(
            our_prob=0.70,
            market_price=0.55,
        )
        assert size > 0
        assert size <= 2500  # Max 25% of 10k

    def test_position_size_zero_below_min_edge(self, manager):
        """No position if edge below minimum."""
        size = manager.calculate_position_size(
            our_prob=0.52,  # 2% edge, below 5% min
            market_price=0.50,
        )
        assert size == 0

    def test_position_respects_daily_loss_limit(self, manager):
        """Position size reduced when approaching daily loss limit."""
        manager.record_loss(800)  # Already lost 8%

        size = manager.calculate_position_size(
            our_prob=0.80,
            market_price=0.50,
        )
        # Should be capped at remaining 2% = $200
        assert size <= 200

    def test_can_trade_checks_limits(self, manager):
        """can_trade returns False when at limit."""
        manager.record_loss(1000)  # Hit 10% daily limit

        assert manager.can_trade() is False

    def test_record_pnl_updates_state(self, manager):
        """Recording P&L updates daily totals."""
        manager.record_win(100)
        manager.record_loss(50)

        assert manager.daily_pnl == 50
        assert manager.capital == 10050
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/paper_trading/test_position_manager.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Create package and implementation**

```python
# src/paper_trading/__init__.py
"""Paper trading system."""

from .position_manager import PositionManager, kelly_fraction

__all__ = ["PositionManager", "kelly_fraction"]
```

```python
# src/paper_trading/position_manager.py
"""Position sizing and risk management for paper trading."""

from dataclasses import dataclass, field
from datetime import date


def kelly_fraction(
    our_prob: float,
    market_price: float,
    max_fraction: float = 0.25,
) -> float:
    """Calculate Kelly criterion fraction.

    Kelly: f* = (p * b - q) / b
    where p = our prob, q = 1-p, b = decimal odds - 1

    Args:
        our_prob: Our predicted probability of winning
        market_price: Current market price (implied prob)
        max_fraction: Maximum fraction to bet (default 25%)

    Returns:
        Optimal fraction of bankroll to bet (0 to max_fraction)
    """
    if our_prob <= market_price:
        return 0.0

    p = our_prob
    q = 1 - p
    b = (1 / market_price) - 1  # Decimal odds - 1

    if b <= 0:
        return 0.0

    kelly = (p * b - q) / b

    return max(0.0, min(kelly, max_fraction))


@dataclass
class PositionManager:
    """Manage positions and risk for paper trading."""

    capital: float
    max_position_pct: float = 0.25
    min_edge: float = 0.05
    max_daily_loss_pct: float = 0.10

    # State
    daily_pnl: float = field(default=0.0, init=False)
    daily_trades: int = field(default=0, init=False)
    current_date: date = field(default_factory=date.today, init=False)

    def calculate_position_size(
        self,
        our_prob: float,
        market_price: float,
    ) -> float:
        """Calculate position size for a trade.

        Args:
            our_prob: Our predicted probability
            market_price: Current market price

        Returns:
            Dollar amount to bet (0 if shouldn't trade)
        """
        self._check_new_day()

        edge = our_prob - market_price
        if edge < self.min_edge:
            return 0.0

        if not self.can_trade():
            return 0.0

        # Kelly fraction
        fraction = kelly_fraction(our_prob, market_price, self.max_position_pct)

        # Base position size
        size = self.capital * fraction

        # Cap by remaining daily loss allowance
        remaining_loss_allowance = (
            self.capital * self.max_daily_loss_pct + self.daily_pnl
        )
        if remaining_loss_allowance <= 0:
            return 0.0

        # Size can't exceed what we can afford to lose
        size = min(size, remaining_loss_allowance)

        return max(0.0, size)

    def can_trade(self) -> bool:
        """Check if we can place more trades today."""
        self._check_new_day()

        # Check daily loss limit
        if self.daily_pnl <= -self.capital * self.max_daily_loss_pct:
            return False

        return True

    def record_win(self, amount: float) -> None:
        """Record a winning trade."""
        self._check_new_day()
        self.daily_pnl += amount
        self.capital += amount
        self.daily_trades += 1

    def record_loss(self, amount: float) -> None:
        """Record a losing trade."""
        self._check_new_day()
        self.daily_pnl -= amount
        self.capital -= amount
        self.daily_trades += 1

    def _check_new_day(self) -> None:
        """Reset daily counters if new day."""
        today = date.today()
        if today != self.current_date:
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.current_date = today
```

**Step 4: Create test directory**

```bash
mkdir -p tests/paper_trading
touch tests/paper_trading/__init__.py
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/paper_trading/test_position_manager.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/paper_trading/ tests/paper_trading/
git commit -m "feat: add position manager with Kelly sizing and risk limits"
```

---

## Task 8: Execution Simulator

**Files:**
- Create: `src/paper_trading/execution_sim.py`
- Test: `tests/paper_trading/test_execution_sim.py`

**Step 1: Write the failing test**

```python
# tests/paper_trading/test_execution_sim.py
"""Tests for execution simulation."""

import pytest

from src.paper_trading.execution_sim import ExecutionSimulator


class TestExecutionSimulator:
    @pytest.fixture
    def simulator(self):
        return ExecutionSimulator(
            default_depth=10000,  # $10k default depth
            max_slippage=0.05,
        )

    def test_small_order_minimal_slippage(self, simulator):
        """Small orders should have minimal slippage."""
        fill_price = simulator.simulate_fill(
            target_price=0.50,
            size=100,
            side="BUY",
        )

        # Slippage should be tiny
        slippage = (fill_price - 0.50) / 0.50
        assert slippage < 0.01

    def test_large_order_more_slippage(self, simulator):
        """Large orders relative to depth should have more slippage."""
        fill_price = simulator.simulate_fill(
            target_price=0.50,
            size=5000,  # 50% of depth
            side="BUY",
        )

        slippage = (fill_price - 0.50) / 0.50
        assert slippage > 0.01

    def test_buy_slippage_increases_price(self, simulator):
        """Buying should fill at higher price."""
        fill_price = simulator.simulate_fill(
            target_price=0.50,
            size=1000,
            side="BUY",
        )

        assert fill_price >= 0.50

    def test_sell_slippage_decreases_price(self, simulator):
        """Selling should fill at lower price."""
        fill_price = simulator.simulate_fill(
            target_price=0.50,
            size=1000,
            side="SELL",
        )

        assert fill_price <= 0.50

    def test_max_slippage_cap(self, simulator):
        """Slippage should be capped."""
        fill_price = simulator.simulate_fill(
            target_price=0.50,
            size=100000,  # Huge order
            side="BUY",
        )

        slippage = (fill_price - 0.50) / 0.50
        assert slippage <= 0.05

    def test_custom_depth(self, simulator):
        """Can specify custom orderbook depth."""
        fill_price = simulator.simulate_fill(
            target_price=0.50,
            size=1000,
            side="BUY",
            orderbook_depth=50000,  # Deeper book
        )

        slippage = (fill_price - 0.50) / 0.50
        assert slippage < 0.005  # Very small
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/paper_trading/test_execution_sim.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write the implementation**

```python
# src/paper_trading/execution_sim.py
"""Simulate trade execution with slippage."""

from dataclasses import dataclass


@dataclass
class ExecutionSimulator:
    """Simulate realistic trade execution with slippage.

    Models slippage as proportional to size / depth.
    """

    default_depth: float = 10000  # Default orderbook depth in dollars
    max_slippage: float = 0.05   # Max 5% slippage

    def simulate_fill(
        self,
        target_price: float,
        size: float,
        side: str,
        orderbook_depth: float | None = None,
    ) -> float:
        """Simulate order fill with slippage.

        Args:
            target_price: Target fill price (best bid/ask)
            size: Order size in dollars
            side: "BUY" or "SELL"
            orderbook_depth: Available liquidity (uses default if None)

        Returns:
            Simulated fill price including slippage
        """
        depth = orderbook_depth or self.default_depth

        # Slippage proportional to size / depth
        # Using square root for more realistic impact
        impact_ratio = (size / depth) ** 0.5
        slippage_pct = min(impact_ratio * 0.02, self.max_slippage)

        if side.upper() == "BUY":
            fill_price = target_price * (1 + slippage_pct)
        else:
            fill_price = target_price * (1 - slippage_pct)

        # Clamp to valid price range
        return max(0.01, min(0.99, fill_price))

    def estimate_slippage_pct(
        self,
        size: float,
        orderbook_depth: float | None = None,
    ) -> float:
        """Estimate slippage percentage for an order.

        Args:
            size: Order size in dollars
            orderbook_depth: Available liquidity

        Returns:
            Estimated slippage as decimal (e.g., 0.01 = 1%)
        """
        depth = orderbook_depth or self.default_depth
        impact_ratio = (size / depth) ** 0.5
        return min(impact_ratio * 0.02, self.max_slippage)
```

**Step 4: Update __init__.py**

Add to `src/paper_trading/__init__.py`:

```python
from .execution_sim import ExecutionSimulator

__all__ = ["PositionManager", "kelly_fraction", "ExecutionSimulator"]
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/paper_trading/test_execution_sim.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/paper_trading/execution_sim.py src/paper_trading/__init__.py tests/paper_trading/test_execution_sim.py
git commit -m "feat: add execution simulator with slippage modeling"
```

---

## Task 9: Metrics Calculator

**Files:**
- Create: `src/paper_trading/metrics.py`
- Test: `tests/paper_trading/test_metrics.py`

**Step 1: Write the failing test**

```python
# tests/paper_trading/test_metrics.py
"""Tests for paper trading metrics."""

import pytest
from datetime import datetime

from src.paper_trading.metrics import PaperTradingMetrics, TradeRecord


class TestPaperTradingMetrics:
    @pytest.fixture
    def sample_trades(self):
        return [
            TradeRecord(
                timestamp=datetime(2026, 1, 1, 10, 0),
                edge_theoretical=0.10,
                edge_realized=0.08,
                pnl=40.0,
                size=500,
            ),
            TradeRecord(
                timestamp=datetime(2026, 1, 1, 11, 0),
                edge_theoretical=0.12,
                edge_realized=-0.02,
                pnl=-10.0,
                size=500,
            ),
            TradeRecord(
                timestamp=datetime(2026, 1, 1, 12, 0),
                edge_theoretical=0.08,
                edge_realized=0.06,
                pnl=30.0,
                size=500,
            ),
        ]

    def test_total_pnl(self, sample_trades):
        metrics = PaperTradingMetrics(sample_trades)
        assert metrics.total_pnl == 60.0

    def test_win_rate(self, sample_trades):
        metrics = PaperTradingMetrics(sample_trades)
        assert metrics.win_rate == pytest.approx(2 / 3, rel=0.01)

    def test_avg_edge_theoretical(self, sample_trades):
        metrics = PaperTradingMetrics(sample_trades)
        assert metrics.avg_edge_theoretical == pytest.approx(0.10, rel=0.01)

    def test_avg_edge_realized(self, sample_trades):
        metrics = PaperTradingMetrics(sample_trades)
        expected = (0.08 + (-0.02) + 0.06) / 3
        assert metrics.avg_edge_realized == pytest.approx(expected, rel=0.01)

    def test_sharpe_ratio(self, sample_trades):
        metrics = PaperTradingMetrics(sample_trades)
        # Should be positive given overall profit
        assert metrics.sharpe_ratio > 0

    def test_max_drawdown(self, sample_trades):
        metrics = PaperTradingMetrics(sample_trades)
        # Max drawdown happened at trade 2: from 40 to 30
        assert metrics.max_drawdown == 10.0

    def test_empty_trades_handle_gracefully(self):
        metrics = PaperTradingMetrics([])
        assert metrics.total_pnl == 0
        assert metrics.win_rate == 0
        assert metrics.sharpe_ratio == 0

    def test_as_dict(self, sample_trades):
        metrics = PaperTradingMetrics(sample_trades)
        d = metrics.as_dict()

        assert "total_pnl" in d
        assert "win_rate" in d
        assert "n_trades" in d
        assert d["n_trades"] == 3
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/paper_trading/test_metrics.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write the implementation**

```python
# src/paper_trading/metrics.py
"""Metrics calculation for paper trading."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np


@dataclass
class TradeRecord:
    """Record of a single paper trade."""

    timestamp: datetime
    edge_theoretical: float
    edge_realized: float
    pnl: float
    size: float


class PaperTradingMetrics:
    """Calculate performance metrics from paper trades."""

    def __init__(self, trades: list[TradeRecord]):
        """Initialize with list of trades."""
        self.trades = trades

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def total_pnl(self) -> float:
        if not self.trades:
            return 0.0
        return sum(t.pnl for t in self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t.pnl > 0)
        return wins / len(self.trades)

    @property
    def avg_edge_theoretical(self) -> float:
        if not self.trades:
            return 0.0
        return np.mean([t.edge_theoretical for t in self.trades])

    @property
    def avg_edge_realized(self) -> float:
        if not self.trades:
            return 0.0
        return np.mean([t.edge_realized for t in self.trades])

    @property
    def sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio (simplified, no risk-free rate)."""
        if len(self.trades) < 2:
            return 0.0

        returns = [t.pnl / t.size for t in self.trades if t.size > 0]
        if not returns or np.std(returns) == 0:
            return 0.0

        return np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized

    @property
    def max_drawdown(self) -> float:
        """Calculate maximum drawdown in absolute terms."""
        if not self.trades:
            return 0.0

        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0

        for t in self.trades:
            cumulative += t.pnl
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd

        return max_dd

    @property
    def profit_factor(self) -> float:
        """Gross profit / gross loss."""
        if not self.trades:
            return 0.0

        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))

        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0

        return gross_profit / gross_loss

    def as_dict(self) -> dict[str, Any]:
        """Return all metrics as dictionary."""
        return {
            "n_trades": self.n_trades,
            "total_pnl": self.total_pnl,
            "win_rate": self.win_rate,
            "avg_edge_theoretical": self.avg_edge_theoretical,
            "avg_edge_realized": self.avg_edge_realized,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "profit_factor": self.profit_factor,
        }
```

**Step 4: Update __init__.py**

Add to `src/paper_trading/__init__.py`:

```python
from .metrics import PaperTradingMetrics, TradeRecord

__all__ = [
    "PositionManager",
    "kelly_fraction",
    "ExecutionSimulator",
    "PaperTradingMetrics",
    "TradeRecord",
]
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/paper_trading/test_metrics.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/paper_trading/metrics.py src/paper_trading/__init__.py tests/paper_trading/test_metrics.py
git commit -m "feat: add paper trading metrics calculator"
```

---

## Task 10: Market Observer

**Files:**
- Create: `src/paper_trading/market_observer.py`
- Test: `tests/paper_trading/test_market_observer.py`

**Step 1: Write the failing test**

```python
# tests/paper_trading/test_market_observer.py
"""Tests for market observer."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from src.paper_trading.market_observer import MarketObserver, PriceCapture


class TestMarketObserver:
    @pytest.fixture
    def observer(self):
        return MarketObserver()

    @pytest.mark.asyncio
    async def test_capture_price_returns_price_capture(self, observer):
        """capture_price returns PriceCapture object."""
        with patch.object(observer, "_fetch_price", return_value=0.58):
            capture = await observer.capture_price("market_123", "TeamA")

        assert isinstance(capture, PriceCapture)
        assert capture.market_id == "market_123"
        assert capture.price == 0.58

    @pytest.mark.asyncio
    async def test_schedule_followups_captures_at_intervals(self, observer):
        """schedule_followups captures prices at 30s, 60s, 120s."""
        prices = [0.58, 0.62, 0.65, 0.68]
        call_count = 0

        async def mock_fetch(*args):
            nonlocal call_count
            price = prices[call_count]
            call_count += 1
            return price

        with patch.object(observer, "_fetch_price", side_effect=mock_fetch):
            # Use very short intervals for testing
            captures = await observer.capture_with_followups(
                "market_123",
                "TeamA",
                intervals=[0.01, 0.02, 0.03],  # 10ms, 20ms, 30ms for testing
            )

        assert len(captures) == 4
        assert captures[0].price == 0.58
        assert captures[1].price == 0.62
        assert captures[2].price == 0.65
        assert captures[3].price == 0.68

    @pytest.mark.asyncio
    async def test_handles_fetch_failure(self, observer):
        """Gracefully handles API failures."""
        with patch.object(observer, "_fetch_price", side_effect=Exception("API error")):
            capture = await observer.capture_price("market_123", "TeamA")

        assert capture.price is None
        assert capture.error is not None


class TestPriceCapture:
    def test_price_capture_has_timestamp(self):
        capture = PriceCapture(
            market_id="123",
            outcome="TeamA",
            price=0.55,
            timestamp=datetime.utcnow(),
        )
        assert capture.timestamp is not None

    def test_price_capture_optional_fields(self):
        capture = PriceCapture(
            market_id="123",
            outcome="TeamA",
            price=None,
            timestamp=datetime.utcnow(),
            error="Fetch failed",
        )
        assert capture.error == "Fetch failed"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/paper_trading/test_market_observer.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write the implementation**

```python
# src/paper_trading/market_observer.py
"""Observe and capture Polymarket prices."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import structlog

logger = structlog.get_logger()


@dataclass
class PriceCapture:
    """A captured price at a point in time."""

    market_id: str
    outcome: str
    price: Optional[float]
    timestamp: datetime
    error: Optional[str] = None


@dataclass
class MarketObserver:
    """Observe Polymarket prices and capture at intervals."""

    # Polymarket feed will be injected
    polymarket_feed: Optional[object] = None

    # Default follow-up intervals (seconds)
    default_intervals: list[float] = field(
        default_factory=lambda: [30.0, 60.0, 120.0]
    )

    async def capture_price(
        self,
        market_id: str,
        outcome: str,
    ) -> PriceCapture:
        """Capture current price for a market outcome.

        Args:
            market_id: Polymarket market ID
            outcome: Outcome to capture (e.g., team name)

        Returns:
            PriceCapture with current price
        """
        try:
            price = await self._fetch_price(market_id, outcome)
            return PriceCapture(
                market_id=market_id,
                outcome=outcome,
                price=price,
                timestamp=datetime.utcnow(),
            )
        except Exception as e:
            logger.error("price_capture_failed", error=str(e), market_id=market_id)
            return PriceCapture(
                market_id=market_id,
                outcome=outcome,
                price=None,
                timestamp=datetime.utcnow(),
                error=str(e),
            )

    async def capture_with_followups(
        self,
        market_id: str,
        outcome: str,
        intervals: Optional[list[float]] = None,
    ) -> list[PriceCapture]:
        """Capture price now and at follow-up intervals.

        Args:
            market_id: Polymarket market ID
            outcome: Outcome to capture
            intervals: Follow-up intervals in seconds (default: 30, 60, 120)

        Returns:
            List of PriceCaptures (T+0, T+30s, T+60s, T+120s)
        """
        intervals = intervals or self.default_intervals
        captures = []

        # Capture T+0
        capture = await self.capture_price(market_id, outcome)
        captures.append(capture)

        # Schedule follow-ups
        for interval in intervals:
            await asyncio.sleep(interval)
            capture = await self.capture_price(market_id, outcome)
            captures.append(capture)

        return captures

    async def _fetch_price(
        self,
        market_id: str,
        outcome: str,
    ) -> float:
        """Fetch current price from Polymarket.

        Override this method or inject polymarket_feed.
        """
        if self.polymarket_feed is None:
            # Return dummy price for testing
            logger.warning("no_polymarket_feed", market_id=market_id)
            return 0.50

        # Use the injected feed
        return await self.polymarket_feed.get_price(market_id, outcome)
```

**Step 4: Update __init__.py**

Add to `src/paper_trading/__init__.py`:

```python
from .market_observer import MarketObserver, PriceCapture

__all__ = [
    "PositionManager",
    "kelly_fraction",
    "ExecutionSimulator",
    "PaperTradingMetrics",
    "TradeRecord",
    "MarketObserver",
    "PriceCapture",
]
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/paper_trading/test_market_observer.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/paper_trading/market_observer.py src/paper_trading/__init__.py tests/paper_trading/test_market_observer.py
git commit -m "feat: add market observer for Polymarket price capture"
```

---

## Task 11: Paper Trading Engine

**Files:**
- Create: `src/paper_trading/engine.py`
- Test: `tests/paper_trading/test_engine.py`

**Step 1: Write the failing test**

```python
# tests/paper_trading/test_engine.py
"""Tests for paper trading engine."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from src.paper_trading.engine import PaperTradingEngine
from src.feeds.base import FeedEvent


class TestPaperTradingEngine:
    @pytest.fixture
    def mock_model(self):
        model = MagicMock()
        model.predict_single.return_value = 0.72
        return model

    @pytest.fixture
    def engine(self, mock_model):
        return PaperTradingEngine(
            model=mock_model,
            capital=10000,
            min_edge=0.05,
        )

    @pytest.fixture
    def sample_event(self):
        return FeedEvent(
            source="pandascore",
            event_type="baron_kill",
            game="lol",
            data={
                "team": "T1",
                "game_time_minutes": 25,
                "gold_diff": 5000,
                "kill_diff": 3,
                "tower_diff": 2,
                "dragon_diff": 1,
                "baron_diff": 1,
            },
            timestamp=datetime.utcnow().timestamp(),
            match_id="12345",
        )

    @pytest.mark.asyncio
    async def test_process_event_creates_observation(self, engine, sample_event):
        """Processing an event creates a LiveObservation."""
        # Mock market price fetch
        with patch.object(engine.market_observer, "capture_price") as mock_capture:
            mock_capture.return_value = MagicMock(price=0.58)

            result = await engine.process_event(sample_event, market_id="market_123")

        assert result is not None
        assert result["model_prediction"] == 0.72
        assert result["market_price"] == 0.58
        assert result["edge"] == pytest.approx(0.14, rel=0.01)

    @pytest.mark.asyncio
    async def test_process_event_creates_trade_if_edge(self, engine, sample_event):
        """Creates paper trade if edge exceeds minimum."""
        with patch.object(engine.market_observer, "capture_price") as mock_capture:
            mock_capture.return_value = MagicMock(price=0.58)

            result = await engine.process_event(sample_event, market_id="market_123")

        assert result["trade"] is not None
        assert result["trade"]["size"] > 0

    @pytest.mark.asyncio
    async def test_no_trade_if_edge_too_small(self, engine, sample_event):
        """No trade if edge below minimum."""
        engine.model.predict_single.return_value = 0.60  # 2% edge

        with patch.object(engine.market_observer, "capture_price") as mock_capture:
            mock_capture.return_value = MagicMock(price=0.58)

            result = await engine.process_event(sample_event, market_id="market_123")

        assert result["trade"] is None

    @pytest.mark.asyncio
    async def test_schedules_followup_captures(self, engine, sample_event):
        """Engine schedules follow-up price captures."""
        with patch.object(engine.market_observer, "capture_price") as mock_capture:
            mock_capture.return_value = MagicMock(price=0.58)

            with patch.object(
                engine, "_schedule_followups", new_callable=AsyncMock
            ) as mock_schedule:
                await engine.process_event(sample_event, market_id="market_123")

                mock_schedule.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/paper_trading/test_engine.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write the implementation**

```python
# src/paper_trading/engine.py
"""Main paper trading engine orchestrating all components."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import structlog

from src.feeds.base import FeedEvent
from src.ml.train import ImpactModel
from .position_manager import PositionManager
from .execution_sim import ExecutionSimulator
from .market_observer import MarketObserver

logger = structlog.get_logger()


@dataclass
class PaperTradingEngine:
    """Orchestrates paper trading: events → predictions → trades."""

    model: ImpactModel
    capital: float = 10000
    min_edge: float = 0.05

    # Components (created in __post_init__)
    position_manager: PositionManager = field(init=False)
    execution_sim: ExecutionSimulator = field(init=False)
    market_observer: MarketObserver = field(init=False)

    # State
    observations: list[dict] = field(default_factory=list)
    trades: list[dict] = field(default_factory=list)
    pending_followups: dict[str, asyncio.Task] = field(default_factory=dict)

    def __post_init__(self):
        self.position_manager = PositionManager(
            capital=self.capital,
            min_edge=self.min_edge,
        )
        self.execution_sim = ExecutionSimulator()
        self.market_observer = MarketObserver()

    async def process_event(
        self,
        event: FeedEvent,
        market_id: str,
        outcome: Optional[str] = None,
    ) -> dict[str, Any]:
        """Process a game event and potentially create a paper trade.

        Args:
            event: Game event from PandaScore
            market_id: Polymarket market ID
            outcome: Outcome to bet on (defaults to event team)

        Returns:
            Dict with observation and trade details
        """
        outcome = outcome or event.data.get("team", "YES")

        # Get model prediction
        features = self._extract_features(event)
        prediction = self.model.predict_single(features)

        # Get market price
        price_capture = await self.market_observer.capture_price(market_id, outcome)
        market_price = price_capture.price

        if market_price is None:
            logger.warning("no_market_price", market_id=market_id)
            return {"error": "Could not fetch market price"}

        edge = prediction - market_price

        # Create observation
        observation = {
            "timestamp": datetime.utcnow(),
            "match_id": event.match_id,
            "event_type": event.event_type,
            "game_state": event.data,
            "model_prediction": prediction,
            "market_price": market_price,
            "edge": edge,
        }
        self.observations.append(observation)

        # Create trade if edge sufficient
        trade = None
        if edge >= self.min_edge:
            trade = await self._create_trade(
                observation=observation,
                market_id=market_id,
                outcome=outcome,
                prediction=prediction,
                market_price=market_price,
            )

        # Schedule follow-up captures
        await self._schedule_followups(market_id, outcome, len(self.observations) - 1)

        logger.info(
            "event_processed",
            event_type=event.event_type,
            prediction=prediction,
            market_price=market_price,
            edge=edge,
            trade_size=trade["size"] if trade else 0,
        )

        return {
            "observation": observation,
            "model_prediction": prediction,
            "market_price": market_price,
            "edge": edge,
            "trade": trade,
        }

    async def _create_trade(
        self,
        observation: dict,
        market_id: str,
        outcome: str,
        prediction: float,
        market_price: float,
    ) -> dict[str, Any]:
        """Create a paper trade."""
        size = self.position_manager.calculate_position_size(prediction, market_price)

        if size == 0:
            return None

        fill_price = self.execution_sim.simulate_fill(
            target_price=market_price,
            size=size,
            side="BUY",
        )

        trade = {
            "timestamp": datetime.utcnow(),
            "market_id": market_id,
            "outcome": outcome,
            "side": "BUY",
            "entry_price": market_price,
            "fill_price": fill_price,
            "size": size,
            "edge_theoretical": prediction - market_price,
        }
        self.trades.append(trade)

        return trade

    async def _schedule_followups(
        self,
        market_id: str,
        outcome: str,
        observation_idx: int,
    ) -> None:
        """Schedule follow-up price captures."""
        # This runs in background - don't await
        task = asyncio.create_task(
            self._capture_followups(market_id, outcome, observation_idx)
        )
        self.pending_followups[f"{market_id}_{observation_idx}"] = task

    async def _capture_followups(
        self,
        market_id: str,
        outcome: str,
        observation_idx: int,
    ) -> None:
        """Capture prices at T+30s, T+60s, T+120s."""
        intervals = [30.0, 60.0, 120.0]
        keys = ["price_30s", "price_60s", "price_120s"]

        for interval, key in zip(intervals, keys):
            await asyncio.sleep(interval)
            capture = await self.market_observer.capture_price(market_id, outcome)
            if capture.price is not None:
                self.observations[observation_idx][key] = capture.price

    def _extract_features(self, event: FeedEvent) -> dict:
        """Extract model features from event."""
        data = event.data
        game_time = data.get("game_time_minutes", 15)

        features = {
            "game_time_minutes": game_time,
            "gold_diff": data.get("gold_diff", 0),
            "kill_diff": data.get("kill_diff", 0),
            "tower_diff": data.get("tower_diff", 0),
            "dragon_diff": data.get("dragon_diff", 0),
            "baron_diff": data.get("baron_diff", 0),
        }

        # Normalized features
        if game_time > 0:
            features["gold_diff_normalized"] = features["gold_diff"] / game_time
            features["kill_diff_normalized"] = features["kill_diff"] / game_time
        else:
            features["gold_diff_normalized"] = 0.0
            features["kill_diff_normalized"] = 0.0

        features["is_ahead"] = 1 if features["gold_diff"] > 0 else 0
        features["is_late_game"] = 1 if game_time > 25 else 0

        # One-hot event type
        event_type = event.event_type.lower()
        for et in [
            "kill", "tower_destroyed", "dragon_kill", "baron_kill",
            "elder_kill", "inhibitor_destroyed", "ace"
        ]:
            features[f"event_{et}"] = 1 if event_type == et else 0

        return features
```

**Step 4: Update __init__.py**

Add to `src/paper_trading/__init__.py`:

```python
from .engine import PaperTradingEngine

__all__ = [
    # ... existing
    "PaperTradingEngine",
]
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/paper_trading/test_engine.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/paper_trading/engine.py src/paper_trading/__init__.py tests/paper_trading/test_engine.py
git commit -m "feat: add paper trading engine orchestrator"
```

---

## Task 12: Paper Trade CLI Script

**Files:**
- Create: `scripts/paper_trade.py`

**Step 1: Write the script**

```python
#!/usr/bin/env python3
# scripts/paper_trade.py
"""Run paper trading simulation.

Usage:
    uv run python scripts/paper_trade.py \\
        --model models/lol_impact.pkl \\
        --game lol \\
        --capital 10000 \\
        --telegram
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog

from config.settings import settings
from src.ml.train import ImpactModel
from src.paper_trading.engine import PaperTradingEngine
from src.feeds.pandascore import PandaScoreFeed

logger = structlog.get_logger()


async def run_paper_trading(
    model_path: Path,
    game: str,
    capital: float,
    use_telegram: bool,
) -> None:
    """Run paper trading loop."""
    # Load model
    logger.info("loading_model", path=str(model_path))
    model = ImpactModel.load(model_path)

    # Initialize engine
    engine = PaperTradingEngine(
        model=model,
        capital=capital,
        min_edge=settings.MIN_EDGE_PCT,
    )

    # Initialize PandaScore feed
    feed = PandaScoreFeed()
    await feed.connect()

    logger.info(
        "paper_trading_started",
        game=game,
        capital=capital,
    )

    print(f"\n{'='*50}")
    print("PAPER TRADING STARTED")
    print(f"{'='*50}")
    print(f"Game: {game}")
    print(f"Capital: ${capital:,.2f}")
    print(f"Min Edge: {settings.MIN_EDGE_PCT:.1%}")
    print(f"{'='*50}\n")

    try:
        # Get live matches
        while True:
            matches = await feed.get_live_matches(game)

            if not matches:
                logger.info("no_live_matches", game=game)
                await asyncio.sleep(60)
                continue

            for match in matches:
                logger.info(
                    "monitoring_match",
                    match_id=match.get("id"),
                    name=match.get("name"),
                )

                # Subscribe to match events
                await feed.subscribe(game, str(match["id"]))

            # Wait for events
            await asyncio.sleep(settings.CROSSMARKET_SCAN_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        logger.info("paper_trading_stopped")
    finally:
        await feed.disconnect()

        # Print summary
        print(f"\n{'='*50}")
        print("PAPER TRADING SUMMARY")
        print(f"{'='*50}")
        print(f"Observations: {len(engine.observations)}")
        print(f"Trades: {len(engine.trades)}")

        if engine.trades:
            total_pnl = sum(t.get("pnl", 0) for t in engine.trades)
            print(f"Total P&L: ${total_pnl:,.2f}")

        print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(description="Run paper trading")
    parser.add_argument(
        "--model",
        type=str,
        default="models/impact_model.pkl",
        help="Path to trained model",
    )
    parser.add_argument(
        "--game",
        type=str,
        default="lol",
        choices=["lol", "csgo", "dota2"],
        help="Game to monitor",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=10000,
        help="Starting capital",
    )
    parser.add_argument(
        "--telegram",
        action="store_true",
        help="Enable Telegram alerts",
    )

    args = parser.parse_args()

    asyncio.run(
        run_paper_trading(
            model_path=Path(args.model),
            game=args.game,
            capital=args.capital,
            use_telegram=args.telegram,
        )
    )


if __name__ == "__main__":
    main()
```

**Step 2: Make executable**

```bash
chmod +x scripts/paper_trade.py
```

**Step 3: Commit**

```bash
git add scripts/paper_trade.py
git commit -m "feat: add paper_trade.py CLI script"
```

---

## Task 13: Streamlit Dashboard

**Files:**
- Create: `src/paper_trading/dashboard.py`

**Step 1: Write the implementation**

```python
# src/paper_trading/dashboard.py
"""Streamlit dashboard for paper trading monitoring."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime, timedelta

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.db.models import LiveObservation, PaperTrade
from src.ml.validation.calibration import reliability_diagram_data
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def load_data(db_path: str = "data/arb.db"):
    """Load data from SQLite database."""
    engine = create_engine(f"sqlite:///{db_path}")
    Session = sessionmaker(bind=engine)
    session = Session()

    observations = session.query(LiveObservation).all()
    trades = session.query(PaperTrade).all()

    session.close()
    return observations, trades


def main():
    st.set_page_config(
        page_title="Paper Trading Dashboard",
        page_icon="📊",
        layout="wide",
    )

    st.title("📊 Paper Trading Dashboard")

    # Sidebar
    st.sidebar.header("Settings")
    db_path = st.sidebar.text_input("Database Path", "data/arb.db")
    refresh = st.sidebar.button("Refresh Data")

    # Load data
    try:
        observations, trades = load_data(db_path)
    except Exception as e:
        st.error(f"Could not load data: {e}")
        st.info("Make sure the database exists and paper trading has run.")

        # Show demo data
        st.subheader("Demo Mode (No Data)")
        observations = []
        trades = []

    # Overview metrics
    st.header("Overview")

    col1, col2, col3, col4 = st.columns(4)

    total_pnl = sum(t.pnl or 0 for t in trades)
    win_rate = (
        sum(1 for t in trades if (t.pnl or 0) > 0) / len(trades)
        if trades else 0
    )
    avg_edge_theoretical = (
        sum(t.edge_theoretical for t in trades) / len(trades)
        if trades else 0
    )
    avg_edge_realized = (
        sum(t.edge_realized or 0 for t in trades) / len(trades)
        if trades else 0
    )

    col1.metric("Total P&L", f"${total_pnl:,.2f}")
    col2.metric("Trades", len(trades))
    col3.metric("Win Rate", f"{win_rate:.1%}")
    col4.metric(
        "Avg Edge",
        f"{avg_edge_realized:.1%}",
        delta=f"{avg_edge_realized - avg_edge_theoretical:.1%} vs theoretical",
    )

    # P&L Chart
    st.header("P&L Over Time")

    if trades:
        pnl_df = pd.DataFrame([
            {"timestamp": t.created_at, "pnl": t.pnl or 0}
            for t in trades
        ])
        pnl_df = pnl_df.sort_values("timestamp")
        pnl_df["cumulative_pnl"] = pnl_df["pnl"].cumsum()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pnl_df["timestamp"],
            y=pnl_df["cumulative_pnl"],
            mode="lines+markers",
            name="Cumulative P&L",
        ))
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Cumulative P&L ($)",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No trades yet")

    # Calibration
    st.header("Model Calibration")

    if observations:
        obs_df = pd.DataFrame([
            {
                "y_pred": o.model_prediction,
                "y_true": 1 if o.actual_winner else 0,
            }
            for o in observations
            if o.actual_winner is not None
        ])

        if len(obs_df) > 10:
            data = reliability_diagram_data(
                obs_df["y_true"].values,
                obs_df["y_pred"].values,
            )

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode="lines",
                line=dict(dash="dash", color="gray"),
                name="Perfect",
            ))
            fig.add_trace(go.Scatter(
                x=data["bin_centers"],
                y=data["true_fractions"],
                mode="lines+markers",
                name="Model",
            ))
            fig.update_layout(
                title="Reliability Diagram",
                xaxis_title="Predicted Probability",
                yaxis_title="Observed Frequency",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need more resolved observations for calibration plot")
    else:
        st.info("No observations yet")

    # Edge Analysis
    st.header("Edge Analysis")

    if trades:
        col1, col2 = st.columns(2)

        with col1:
            fig = go.Figure()
            edge_theoretical = [t.edge_theoretical for t in trades]
            edge_realized = [t.edge_realized or 0 for t in trades]

            fig.add_trace(go.Scatter(
                x=edge_theoretical,
                y=edge_realized,
                mode="markers",
                marker=dict(size=10),
            ))
            fig.add_trace(go.Scatter(
                x=[0, max(edge_theoretical) if edge_theoretical else 0.2],
                y=[0, max(edge_theoretical) if edge_theoretical else 0.2],
                mode="lines",
                line=dict(dash="dash", color="gray"),
                name="1:1 Line",
            ))
            fig.update_layout(
                title="Theoretical vs Realized Edge",
                xaxis_title="Theoretical Edge",
                yaxis_title="Realized Edge",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Edge by event type
            event_edges = {}
            for t, o in zip(trades, observations):
                if hasattr(o, "event_type"):
                    et = o.event_type
                    if et not in event_edges:
                        event_edges[et] = []
                    event_edges[et].append(t.edge_realized or 0)

            if event_edges:
                fig = go.Figure(data=[
                    go.Bar(
                        x=list(event_edges.keys()),
                        y=[sum(v) / len(v) for v in event_edges.values()],
                    )
                ])
                fig.update_layout(
                    title="Avg Realized Edge by Event Type",
                    xaxis_title="Event Type",
                    yaxis_title="Avg Realized Edge",
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)

    # Recent Trades
    st.header("Recent Trades")

    if trades:
        recent = sorted(trades, key=lambda t: t.created_at, reverse=True)[:20]
        trade_df = pd.DataFrame([
            {
                "Time": t.created_at.strftime("%Y-%m-%d %H:%M"),
                "Side": t.side,
                "Entry": f"{t.entry_price:.2%}",
                "Fill": f"{t.simulated_fill_price:.2%}",
                "Size": f"${t.size:.2f}",
                "Edge": f"{t.edge_theoretical:.1%}",
                "P&L": f"${t.pnl or 0:.2f}",
            }
            for t in recent
        ])
        st.dataframe(trade_df, use_container_width=True)
    else:
        st.info("No trades yet")


if __name__ == "__main__":
    main()
```

**Step 2: Create run script**

```bash
# To run the dashboard:
# streamlit run src/paper_trading/dashboard.py
```

**Step 3: Commit**

```bash
git add src/paper_trading/dashboard.py
git commit -m "feat: add Streamlit dashboard for paper trading monitoring"
```

---

## Task 14: Telegram Alerts

**Files:**
- Create: `src/paper_trading/alerts.py`

**Step 1: Write the implementation**

```python
# src/paper_trading/alerts.py
"""Telegram alerts for paper trading opportunities."""

from dataclasses import dataclass
from typing import Optional

import structlog
from telegram import Bot
from telegram.constants import ParseMode

from config.settings import settings

logger = structlog.get_logger()


@dataclass
class TelegramAlerter:
    """Send alerts to Telegram."""

    bot_token: str = ""
    chat_id: str = ""

    def __post_init__(self):
        self.bot_token = self.bot_token or settings.TELEGRAM_BOT_TOKEN
        self.chat_id = self.chat_id or settings.TELEGRAM_CHAT_ID
        self._bot: Optional[Bot] = None

    async def _get_bot(self) -> Bot:
        """Lazy initialize bot."""
        if self._bot is None:
            self._bot = Bot(token=self.bot_token)
        return self._bot

    async def send_opportunity_alert(
        self,
        match_name: str,
        event_type: str,
        team: str,
        game_time: float,
        gold_diff: int,
        model_prediction: float,
        market_price: float,
        edge: float,
        trade_size: float,
    ) -> bool:
        """Send opportunity detected alert.

        Returns:
            True if sent successfully
        """
        message = f"""
🎯 *OPPORTUNITY DETECTED*

*Match:* {match_name}
*Event:* {event_type} by {team}
*Game:* {game_time:.0f}min | Gold: {gold_diff:+,}

📊 *Model:* {model_prediction:.0%} → *Market:* {market_price:.0%}
💰 *Edge:* +{edge:.1%} | Confidence: {"HIGH" if edge > 0.10 else "MEDIUM"}

📝 *Simulated trade:*
   BUY {team} @ {market_price:.1%}
   Size: ${trade_size:.2f}

⏱️ Will update at T+30s, T+60s, T+120s
"""

        return await self._send(message)

    async def send_followup_alert(
        self,
        match_name: str,
        team: str,
        initial_price: float,
        current_price: float,
        time_elapsed: int,
        edge_captured: float,
        running_pnl: float,
    ) -> bool:
        """Send follow-up price update alert."""
        direction = "📈" if current_price > initial_price else "📉"

        message = f"""
{direction} *UPDATE: {match_name}*

T+{time_elapsed}s: Market moved {initial_price:.0%} → {current_price:.0%}
Edge captured: {edge_captured:+.1%} {"✓" if edge_captured > 0 else "✗"}
Running P&L: ${running_pnl:+.2f}
"""

        return await self._send(message)

    async def send_match_result(
        self,
        match_name: str,
        winner: str,
        our_bet: str,
        pnl: float,
        total_pnl: float,
    ) -> bool:
        """Send match result alert."""
        won = winner == our_bet
        emoji = "🎉" if won else "💔"

        message = f"""
{emoji} *MATCH RESULT: {match_name}*

Winner: {winner}
Our bet: {our_bet}
Result: {"WIN" if won else "LOSS"}

P&L: ${pnl:+.2f}
Session Total: ${total_pnl:+.2f}
"""

        return await self._send(message)

    async def _send(self, message: str) -> bool:
        """Send message to Telegram."""
        if not self.bot_token or not self.chat_id:
            logger.warning("telegram_not_configured")
            return False

        try:
            bot = await self._get_bot()
            await bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=ParseMode.MARKDOWN,
            )
            return True
        except Exception as e:
            logger.error("telegram_send_failed", error=str(e))
            return False
```

**Step 2: Update __init__.py**

Add to `src/paper_trading/__init__.py`:

```python
from .alerts import TelegramAlerter

__all__ = [
    # ... existing
    "TelegramAlerter",
]
```

**Step 3: Commit**

```bash
git add src/paper_trading/alerts.py src/paper_trading/__init__.py
git commit -m "feat: add Telegram alerts for paper trading"
```

---

## Task 15: Performance Report Script

**Files:**
- Create: `scripts/performance_report.py`

**Step 1: Write the script**

```python
#!/usr/bin/env python3
# scripts/performance_report.py
"""Generate paper trading performance report.

Usage:
    uv run python scripts/performance_report.py \\
        --since 2026-01-31 \\
        --output reports/performance.html
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.models import Base, LiveObservation, PaperTrade
from src.paper_trading.metrics import PaperTradingMetrics, TradeRecord


def main():
    parser = argparse.ArgumentParser(description="Generate performance report")
    parser.add_argument(
        "--since",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="data/arb.db",
        help="Database path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/performance.html",
        help="Output HTML path",
    )

    args = parser.parse_args()

    # Load data
    engine = create_engine(f"sqlite:///{args.db}")
    Session = sessionmaker(bind=engine)
    session = Session()

    query = session.query(PaperTrade)
    if args.since:
        since_date = datetime.strptime(args.since, "%Y-%m-%d")
        query = query.filter(PaperTrade.created_at >= since_date)

    trades_db = query.all()
    session.close()

    if not trades_db:
        print("No trades found")
        return

    # Convert to TradeRecords
    trades = [
        TradeRecord(
            timestamp=t.created_at,
            edge_theoretical=t.edge_theoretical,
            edge_realized=t.edge_realized or 0,
            pnl=t.pnl or 0,
            size=t.size,
        )
        for t in trades_db
    ]

    # Calculate metrics
    metrics = PaperTradingMetrics(trades)
    summary = metrics.as_dict()

    # Print summary
    print("\n" + "=" * 50)
    print("PAPER TRADING PERFORMANCE")
    print("=" * 50)
    print(f"Period: {args.since or 'All time'}")
    print(f"Trades: {summary['n_trades']}")
    print("-" * 50)
    print(f"Total P&L:      ${summary['total_pnl']:,.2f}")
    print(f"Win Rate:       {summary['win_rate']:.1%}")
    print(f"Sharpe Ratio:   {summary['sharpe_ratio']:.2f}")
    print(f"Max Drawdown:   ${summary['max_drawdown']:,.2f}")
    print(f"Profit Factor:  {summary['profit_factor']:.2f}")
    print("-" * 50)
    print(f"Avg Edge (theoretical): {summary['avg_edge_theoretical']:.2%}")
    print(f"Avg Edge (realized):    {summary['avg_edge_realized']:.2%}")
    print("=" * 50)

    # Generate HTML report (simple version)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Paper Trading Performance</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .metric {{ display: inline-block; padding: 20px; margin: 10px; background: #f5f5f5; }}
        .metric-value {{ font-size: 32px; font-weight: bold; }}
        .positive {{ color: #28a745; }}
        .negative {{ color: #dc3545; }}
    </style>
</head>
<body>
    <h1>Paper Trading Performance</h1>
    <p>Period: {args.since or 'All time'}</p>

    <div class="metrics">
        <div class="metric">
            <div class="metric-value {'positive' if summary['total_pnl'] > 0 else 'negative'}">
                ${summary['total_pnl']:,.2f}
            </div>
            <div>Total P&L</div>
        </div>
        <div class="metric">
            <div class="metric-value">{summary['n_trades']}</div>
            <div>Trades</div>
        </div>
        <div class="metric">
            <div class="metric-value">{summary['win_rate']:.1%}</div>
            <div>Win Rate</div>
        </div>
        <div class="metric">
            <div class="metric-value">{summary['sharpe_ratio']:.2f}</div>
            <div>Sharpe Ratio</div>
        </div>
    </div>

    <h2>Edge Analysis</h2>
    <ul>
        <li>Avg Theoretical Edge: {summary['avg_edge_theoretical']:.2%}</li>
        <li>Avg Realized Edge: {summary['avg_edge_realized']:.2%}</li>
        <li>Edge Capture: {(summary['avg_edge_realized'] / summary['avg_edge_theoretical'] * 100) if summary['avg_edge_theoretical'] > 0 else 0:.1f}%</li>
    </ul>

    <h2>Risk Metrics</h2>
    <ul>
        <li>Max Drawdown: ${summary['max_drawdown']:,.2f}</li>
        <li>Profit Factor: {summary['profit_factor']:.2f}</li>
    </ul>
</body>
</html>
"""

    output_path.write_text(html)
    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
```

**Step 2: Make executable**

```bash
chmod +x scripts/performance_report.py
```

**Step 3: Commit**

```bash
git add scripts/performance_report.py
git commit -m "feat: add performance_report.py CLI script"
```

---

## Summary

After completing all tasks, you will have:

1. **Validation Module** (`src/ml/validation/`)
   - Brier score decomposition
   - Expected Calibration Error (ECE)
   - Hosmer-Lemeshow statistical test
   - Bootstrap confidence intervals
   - HTML report generation with reliability diagrams

2. **Paper Trading System** (`src/paper_trading/`)
   - Position manager with Kelly sizing
   - Execution simulator with slippage
   - Market observer for price capture
   - Main engine orchestrating everything
   - Metrics calculator
   - Streamlit dashboard
   - Telegram alerts

3. **CLI Scripts**
   - `validate_model.py` — Run full validation suite
   - `paper_trade.py` — Start paper trading
   - `performance_report.py` — Generate performance report

4. **Database Models**
   - `LiveObservation` — Store observations with follow-up prices
   - `PaperTrade` — Store simulated trades

**Workflow:**

```bash
# 1. Validate model
uv run python scripts/validate_model.py --model models/lol_impact.pkl --data data/lol_training.csv

# 2. Start paper trading (runs in background)
uv run python scripts/paper_trade.py --model models/lol_impact.pkl --game lol --telegram

# 3. Monitor via dashboard
streamlit run src/paper_trading/dashboard.py

# 4. Generate performance report
uv run python scripts/performance_report.py --since 2026-01-31
```
