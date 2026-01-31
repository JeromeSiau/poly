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
