# src/ml/validation/report.py
"""Validation report generation with interactive Plotly charts."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .calibration import (
    CalibrationAnalyzer,
    brier_score_decomposition,
    expected_calibration_error,
    reliability_diagram_data,
)
from .statistical_tests import bootstrap_brier_ci, hosmer_lemeshow_test

# GO/NO-GO thresholds
BRIER_THRESHOLD = 0.25
ECE_THRESHOLD = 0.08


class ValidationReport:
    """Generate HTML validation reports with calibration metrics and charts.

    Combines calibration analysis and statistical tests into a comprehensive
    HTML report with interactive Plotly visualizations.

    Attributes:
        BRIER_THRESHOLD: Maximum acceptable Brier score for GO recommendation
        ECE_THRESHOLD: Maximum acceptable ECE for GO recommendation
    """

    BRIER_THRESHOLD = BRIER_THRESHOLD
    ECE_THRESHOLD = ECE_THRESHOLD

    def __init__(self, df: pd.DataFrame) -> None:
        """Initialize with prediction data.

        Args:
            df: DataFrame with required columns:
                - y_true: Binary ground truth labels (0 or 1)
                - y_pred: Predicted probabilities in [0, 1]
                Optional context columns:
                - game_time_minutes: Game time for each sample
                - gold_diff: Gold difference for each sample
                - event_type: Event type for each sample
        """
        self.df = df
        self.y_true = df["y_true"].values
        self.y_pred = df["y_pred"].values

        # Optional context columns
        self.game_time_minutes = (
            df["game_time_minutes"].values if "game_time_minutes" in df.columns else None
        )
        self.gold_diff = df["gold_diff"].values if "gold_diff" in df.columns else None
        self.event_type = df["event_type"].values if "event_type" in df.columns else None

        # Create calibration analyzer
        self.analyzer = CalibrationAnalyzer(
            y_true=self.y_true,
            y_pred=self.y_pred,
            game_time_minutes=self.game_time_minutes,
            gold_diff=self.gold_diff,
            event_type=self.event_type,
        )

    def get_summary(self) -> dict[str, Any]:
        """Get summary metrics dictionary.

        Returns:
            Dictionary containing:
            - brier_score: Overall Brier score
            - brier_ci_lower: Bootstrap CI lower bound
            - brier_ci_upper: Bootstrap CI upper bound
            - ece: Expected Calibration Error
            - reliability: Reliability component of Brier decomposition
            - resolution: Resolution component of Brier decomposition
            - hl_pvalue: Hosmer-Lemeshow test p-value
            - n_samples: Number of samples
            - recommendation: "GO" or "NO-GO"
        """
        # Get calibration metrics
        overall = self.analyzer.analyze_overall()

        # Get bootstrap CI
        brier_ci = bootstrap_brier_ci(self.y_true, self.y_pred)

        # Get Hosmer-Lemeshow test
        _, hl_pvalue = hosmer_lemeshow_test(self.y_true, self.y_pred)

        # Determine recommendation
        brier_ok = overall["brier_score"] <= self.BRIER_THRESHOLD
        ece_ok = overall["ece"] <= self.ECE_THRESHOLD
        recommendation = "GO" if (brier_ok and ece_ok) else "NO-GO"

        return {
            "brier_score": overall["brier_score"],
            "brier_ci_lower": brier_ci["lower"],
            "brier_ci_upper": brier_ci["upper"],
            "ece": overall["ece"],
            "reliability": overall["reliability"],
            "resolution": overall["resolution"],
            "hl_pvalue": hl_pvalue,
            "n_samples": overall["n_samples"],
            "recommendation": recommendation,
        }

    def _create_reliability_plot(self) -> str:
        """Create reliability diagram as Plotly HTML.

        Returns:
            HTML string containing the Plotly chart
        """
        # Get reliability diagram data
        diagram_data = reliability_diagram_data(self.y_true, self.y_pred)

        # Filter out NaN values for plotting
        bin_centers = np.array(diagram_data["bin_centers"])
        true_fractions = np.array(diagram_data["true_fractions"])
        counts = np.array(diagram_data["counts"])

        # Create valid mask
        valid_mask = ~np.isnan(true_fractions)
        valid_centers = bin_centers[valid_mask]
        valid_fractions = true_fractions[valid_mask]
        valid_counts = counts[valid_mask]

        # Create figure
        fig = go.Figure()

        # Add perfect calibration line
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Perfect Calibration",
                line={"dash": "dash", "color": "gray"},
            )
        )

        # Add reliability curve with markers sized by count
        marker_sizes = np.clip(valid_counts / np.max(valid_counts) * 30, 8, 30)
        fig.add_trace(
            go.Scatter(
                x=valid_centers,
                y=valid_fractions,
                mode="markers+lines",
                name="Model Calibration",
                marker={
                    "size": marker_sizes,
                    "color": "blue",
                    "opacity": 0.7,
                },
                line={"color": "blue"},
                text=[f"n={c}" for c in valid_counts],
                hovertemplate=(
                    "Predicted: %{x:.2f}<br>"
                    "Observed: %{y:.2f}<br>"
                    "%{text}<extra></extra>"
                ),
            )
        )

        fig.update_layout(
            title="Reliability Diagram",
            xaxis_title="Mean Predicted Probability",
            yaxis_title="Observed Frequency",
            xaxis={"range": [0, 1]},
            yaxis={"range": [0, 1]},
            showlegend=True,
            width=600,
            height=500,
        )

        return fig.to_html(full_html=False, include_plotlyjs="cdn")

    def _create_context_heatmap(self) -> str:
        """Create ECE heatmap by game time and gold diff.

        Returns:
            HTML string containing the Plotly heatmap chart
        """
        if self.game_time_minutes is None or self.gold_diff is None:
            return "<p>Context data not available for heatmap</p>"

        # Define bins for game time and gold diff
        time_bins = [(0, 15, "Early"), (15, 25, "Mid"), (25, 60, "Late")]
        gold_bins = [
            (-np.inf, -5000, "Far Behind"),
            (-5000, -2000, "Behind"),
            (-2000, 2000, "Even"),
            (2000, 5000, "Ahead"),
            (5000, np.inf, "Far Ahead"),
        ]

        # Calculate ECE for each cell
        ece_matrix = []
        count_matrix = []

        for time_low, time_high, _ in time_bins:
            ece_row = []
            count_row = []
            for gold_low, gold_high, _ in gold_bins:
                mask = (
                    (self.game_time_minutes >= time_low)
                    & (self.game_time_minutes < time_high)
                    & (self.gold_diff >= gold_low)
                    & (self.gold_diff < gold_high)
                )
                n_samples = np.sum(mask)
                count_row.append(n_samples)

                if n_samples >= 10:  # Minimum samples for reliable ECE
                    ece = expected_calibration_error(
                        self.y_true[mask], self.y_pred[mask]
                    )
                    ece_row.append(ece)
                else:
                    ece_row.append(np.nan)

            ece_matrix.append(ece_row)
            count_matrix.append(count_row)

        ece_matrix = np.array(ece_matrix)
        count_matrix = np.array(count_matrix)

        # Create heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=ece_matrix,
                x=[label for _, _, label in gold_bins],
                y=[label for _, _, label in time_bins],
                colorscale="RdYlGn_r",  # Red = bad, Green = good
                zmin=0,
                zmax=0.15,
                text=[[f"n={c}" for c in row] for row in count_matrix],
                hovertemplate=(
                    "Game Time: %{y}<br>"
                    "Gold Diff: %{x}<br>"
                    "ECE: %{z:.3f}<br>"
                    "%{text}<extra></extra>"
                ),
            )
        )

        fig.update_layout(
            title="ECE by Game Context",
            xaxis_title="Gold Difference",
            yaxis_title="Game Phase",
            width=700,
            height=400,
        )

        return fig.to_html(full_html=False, include_plotlyjs=False)

    def generate_html(self, output_path: Path) -> None:
        """Generate full HTML validation report.

        Args:
            output_path: Path to write the HTML report
        """
        summary = self.get_summary()

        # Determine recommendation styling
        if summary["recommendation"] == "GO":
            rec_color = "#28a745"  # Green
            rec_bg = "#d4edda"
        else:
            rec_color = "#dc3545"  # Red
            rec_bg = "#f8d7da"

        # Create plots
        reliability_plot = self._create_reliability_plot()
        context_heatmap = self._create_context_heatmap()

        # Build HTML
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Model Validation Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
        }}
        .recommendation {{
            display: inline-block;
            padding: 15px 40px;
            font-size: 28px;
            font-weight: bold;
            color: {rec_color};
            background: {rec_bg};
            border: 3px solid {rec_color};
            border-radius: 10px;
            margin: 20px 0;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }}
        .metric-label {{
            font-size: 14px;
            color: #666;
            margin-top: 5px;
        }}
        .metric-threshold {{
            font-size: 12px;
            color: #999;
        }}
        .metric-pass {{
            color: #28a745;
        }}
        .metric-fail {{
            color: #dc3545;
        }}
        .section {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 20px 0;
        }}
        .section h2 {{
            margin-top: 0;
            color: #333;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }}
        .thresholds {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin-top: 20px;
        }}
        .thresholds h3 {{
            margin-top: 0;
        }}
        .charts-container {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            justify-content: center;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Model Validation Report</h1>
        <div class="recommendation">{summary["recommendation"]}</div>
    </div>

    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-value {'metric-pass' if summary['brier_score'] <= self.BRIER_THRESHOLD else 'metric-fail'}">
                {summary["brier_score"]:.4f}
            </div>
            <div class="metric-label">Brier Score</div>
            <div class="metric-threshold">
                CI: [{summary["brier_ci_lower"]:.4f}, {summary["brier_ci_upper"]:.4f}]
            </div>
            <div class="metric-threshold">Threshold: &le; {self.BRIER_THRESHOLD}</div>
        </div>

        <div class="metric-card">
            <div class="metric-value {'metric-pass' if summary['ece'] <= self.ECE_THRESHOLD else 'metric-fail'}">
                {summary["ece"]:.4f}
            </div>
            <div class="metric-label">ECE</div>
            <div class="metric-threshold">Threshold: &le; {self.ECE_THRESHOLD}</div>
        </div>

        <div class="metric-card">
            <div class="metric-value">
                {summary["hl_pvalue"]:.4f}
            </div>
            <div class="metric-label">H-L p-value</div>
            <div class="metric-threshold">High = good calibration</div>
        </div>

        <div class="metric-card">
            <div class="metric-value">
                {summary["n_samples"]:,}
            </div>
            <div class="metric-label">Samples</div>
        </div>
    </div>

    <div class="section">
        <h2>Brier Score Decomposition</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{summary["reliability"]:.4f}</div>
                <div class="metric-label">Reliability</div>
                <div class="metric-threshold">Lower = better calibration</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{summary["resolution"]:.4f}</div>
                <div class="metric-label">Resolution</div>
                <div class="metric-threshold">Higher = better discrimination</div>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>Calibration Charts</h2>
        <div class="charts-container">
            {reliability_plot}
        </div>
    </div>

    <div class="section">
        <h2>Context Analysis</h2>
        <div class="charts-container">
            {context_heatmap}
        </div>
    </div>

    <div class="section thresholds">
        <h3>GO/NO-GO Thresholds</h3>
        <p>The model receives a <strong>GO</strong> recommendation if:</p>
        <ul>
            <li>Brier Score &le; {self.BRIER_THRESHOLD} (current: {summary["brier_score"]:.4f})
                <strong class="{'metric-pass' if summary['brier_score'] <= self.BRIER_THRESHOLD else 'metric-fail'}">
                    {'PASS' if summary['brier_score'] <= self.BRIER_THRESHOLD else 'FAIL'}
                </strong>
            </li>
            <li>ECE &le; {self.ECE_THRESHOLD} (current: {summary["ece"]:.4f})
                <strong class="{'metric-pass' if summary['ece'] <= self.ECE_THRESHOLD else 'metric-fail'}">
                    {'PASS' if summary['ece'] <= self.ECE_THRESHOLD else 'FAIL'}
                </strong>
            </li>
        </ul>
        <p><strong>Note:</strong> The Hosmer-Lemeshow p-value provides additional context.
           A low p-value (&lt; 0.05) suggests potential calibration issues, but is not used
           directly in the GO/NO-GO decision.</p>
    </div>

    <footer style="text-align: center; color: #666; margin-top: 30px; padding: 20px;">
        Generated by ML Validation Pipeline
    </footer>
</body>
</html>
"""

        # Write to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html)
