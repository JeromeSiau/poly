"""Streamlit dashboard for paper trading monitoring."""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.models import LiveObservation, PaperTrade
from src.ml.validation.calibration import reliability_diagram_data
from src.paper_trading.metrics import PaperTradingMetrics, TradeRecord


def load_data(db_path: str = "data/arb.db"):
    """Load data from SQLite database.

    Args:
        db_path: Path to SQLite database file.

    Returns:
        Tuple of (observations, trades) lists.
    """
    engine = create_engine(f"sqlite:///{db_path}")
    Session = sessionmaker(bind=engine)
    session = Session()

    observations = session.query(LiveObservation).all()
    trades = session.query(PaperTrade).all()

    session.close()
    return observations, trades


def create_pnl_chart(trades: list[PaperTrade]) -> go.Figure:
    """Create cumulative P&L chart.

    Args:
        trades: List of PaperTrade objects.

    Returns:
        Plotly Figure with P&L chart.
    """
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
        line=dict(color="#2ecc71", width=2),
        marker=dict(size=6),
    ))
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Cumulative P&L ($)",
        height=400,
        template="plotly_dark",
        margin=dict(l=50, r=50, t=30, b=50),
    )
    return fig


def create_reliability_diagram(observations: list[LiveObservation]) -> go.Figure:
    """Create model calibration reliability diagram.

    Args:
        observations: List of LiveObservation objects with outcomes.

    Returns:
        Plotly Figure with reliability diagram.
    """
    # Filter observations that have actual outcomes
    resolved = [
        obs for obs in observations
        if obs.actual_winner is not None and obs.model_prediction is not None
    ]

    if not resolved:
        fig = go.Figure()
        fig.add_annotation(
            text="No resolved observations yet",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16),
        )
        fig.update_layout(
            height=400,
            template="plotly_dark",
        )
        return fig

    # Extract predictions and outcomes
    # Assuming model predicts probability of team1 winning, actual_winner is team name
    y_pred = [obs.model_prediction for obs in resolved]
    # Binary outcome: 1 if prediction > 0.5 and team won, else complex logic
    # Simplified: treat as binary where high prediction = positive outcome expected
    y_true = [
        1.0 if obs.model_prediction > 0.5 else 0.0
        for obs in resolved
    ]
    # Better approach: if we have actual market movement data
    # For now, use polymarket price at resolution vs entry
    y_true = []
    for obs in resolved:
        # If final price moved in predicted direction, count as "correct"
        if obs.polymarket_price_120s is not None and obs.polymarket_price is not None:
            moved_up = obs.polymarket_price_120s > obs.polymarket_price
            predicted_up = obs.model_prediction > obs.polymarket_price
            y_true.append(1.0 if moved_up == predicted_up else 0.0)
        else:
            y_true.append(0.5)  # Unknown

    # Get reliability diagram data
    diagram_data = reliability_diagram_data(y_true, y_pred, n_bins=10)

    fig = go.Figure()

    # Perfect calibration line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Perfect Calibration",
        line=dict(dash="dash", color="gray"),
    ))

    # Actual calibration
    fig.add_trace(go.Scatter(
        x=diagram_data["bin_centers"],
        y=diagram_data["true_fractions"],
        mode="lines+markers",
        name="Model Calibration",
        line=dict(color="#3498db", width=2),
        marker=dict(size=10),
        text=[f"n={c}" for c in diagram_data["counts"]],
        hovertemplate="Predicted: %{x:.2f}<br>Actual: %{y:.2f}<br>%{text}",
    ))

    fig.update_layout(
        xaxis_title="Predicted Probability",
        yaxis_title="Observed Frequency",
        height=400,
        template="plotly_dark",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        margin=dict(l=50, r=50, t=30, b=50),
    )
    return fig


def create_edge_analysis_chart(trades: list[PaperTrade]) -> go.Figure:
    """Create edge analysis chart (theoretical vs realized).

    Args:
        trades: List of PaperTrade objects.

    Returns:
        Plotly Figure comparing theoretical and realized edge.
    """
    if not trades:
        fig = go.Figure()
        fig.add_annotation(
            text="No trades yet",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16),
        )
        fig.update_layout(height=400, template="plotly_dark")
        return fig

    # Filter trades with realized edge
    resolved_trades = [t for t in trades if t.edge_realized is not None]

    if not resolved_trades:
        fig = go.Figure()
        fig.add_annotation(
            text="No resolved trades yet",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16),
        )
        fig.update_layout(height=400, template="plotly_dark")
        return fig

    fig = go.Figure()

    # Scatter plot: theoretical vs realized edge
    theoretical = [t.edge_theoretical * 100 for t in resolved_trades]
    realized = [(t.edge_realized or 0) * 100 for t in resolved_trades]

    fig.add_trace(go.Scatter(
        x=theoretical,
        y=realized,
        mode="markers",
        name="Trades",
        marker=dict(
            size=10,
            color=[t.pnl or 0 for t in resolved_trades],
            colorscale="RdYlGn",
            showscale=True,
            colorbar=dict(title="P&L ($)"),
        ),
        hovertemplate="Theoretical: %{x:.1f}%<br>Realized: %{y:.1f}%",
    ))

    # Perfect correlation line
    max_edge = max(max(theoretical), max(realized)) if theoretical else 10
    min_edge = min(min(theoretical), min(realized)) if theoretical else -10
    fig.add_trace(go.Scatter(
        x=[min_edge, max_edge],
        y=[min_edge, max_edge],
        mode="lines",
        name="Perfect",
        line=dict(dash="dash", color="gray"),
    ))

    fig.update_layout(
        xaxis_title="Theoretical Edge (%)",
        yaxis_title="Realized Edge (%)",
        height=400,
        template="plotly_dark",
        margin=dict(l=50, r=50, t=30, b=50),
    )
    return fig


def calculate_metrics(trades: list[PaperTrade]) -> dict:
    """Calculate trading metrics using PaperTradingMetrics.

    Args:
        trades: List of PaperTrade objects.

    Returns:
        Dictionary of metrics.
    """
    if not trades:
        return {
            "n_trades": 0,
            "total_pnl": 0.0,
            "win_rate": 0.0,
            "avg_edge_theoretical": 0.0,
            "avg_edge_realized": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "profit_factor": 0.0,
        }

    # Convert PaperTrade to TradeRecord
    trade_records = [
        TradeRecord(
            timestamp=t.created_at,
            edge_theoretical=t.edge_theoretical,
            edge_realized=t.edge_realized or 0.0,
            pnl=t.pnl or 0.0,
            size=t.size,
        )
        for t in trades
    ]

    metrics = PaperTradingMetrics(trade_records)
    return metrics.as_dict()


def main():
    """Main dashboard entry point."""
    st.set_page_config(
        page_title="Paper Trading Dashboard",
        page_icon="chart_with_upwards_trend",
        layout="wide",
    )

    st.title("Paper Trading Dashboard")

    # Sidebar
    st.sidebar.header("Settings")
    db_path = st.sidebar.text_input("Database Path", "data/arb.db")
    refresh = st.sidebar.button("Refresh Data")

    # Load data
    try:
        observations, trades = load_data(db_path)
        st.sidebar.success(f"Loaded {len(observations)} observations, {len(trades)} trades")
    except Exception as e:
        st.error(f"Could not load data: {e}")
        st.info("Make sure the database exists and paper trading has run.")
        st.code(f"Expected database at: {db_path}")
        observations = []
        trades = []

    # Calculate metrics
    metrics = calculate_metrics(trades)

    # Overview metrics
    st.header("Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total P&L",
        f"${metrics['total_pnl']:,.2f}",
        delta=f"{metrics['total_pnl']:+,.2f}" if metrics['total_pnl'] != 0 else None,
    )
    col2.metric("Trades", metrics["n_trades"])
    col3.metric("Win Rate", f"{metrics['win_rate']:.1%}")
    col4.metric("Avg Edge (Realized)", f"{metrics['avg_edge_realized']:.1%}")

    # Second row of metrics
    col5, col6, col7, col8 = st.columns(4)

    col5.metric("Avg Edge (Theoretical)", f"{metrics['avg_edge_theoretical']:.1%}")
    col6.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
    col7.metric("Max Drawdown", f"${metrics['max_drawdown']:,.2f}")
    col8.metric(
        "Profit Factor",
        f"{metrics['profit_factor']:.2f}" if metrics['profit_factor'] != float('inf') else "N/A",
    )

    st.divider()

    # Charts row
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("P&L Over Time")
        if trades:
            fig_pnl = create_pnl_chart(trades)
            st.plotly_chart(fig_pnl, use_container_width=True)
        else:
            st.info("No trades yet")

    with col_right:
        st.subheader("Edge Analysis")
        fig_edge = create_edge_analysis_chart(trades)
        st.plotly_chart(fig_edge, use_container_width=True)

    st.divider()

    # Calibration
    st.subheader("Model Calibration")
    fig_calibration = create_reliability_diagram(observations)
    st.plotly_chart(fig_calibration, use_container_width=True)

    st.divider()

    # Recent Trades table
    st.subheader("Recent Trades")

    if trades:
        recent = sorted(trades, key=lambda t: t.created_at, reverse=True)[:20]
        trade_df = pd.DataFrame([
            {
                "Time": t.created_at.strftime("%Y-%m-%d %H:%M") if t.created_at else "N/A",
                "Side": t.side,
                "Entry Price": f"{t.entry_price:.2%}" if t.entry_price else "N/A",
                "Fill Price": f"{t.simulated_fill_price:.2%}" if t.simulated_fill_price else "N/A",
                "Size": f"${t.size:.2f}" if t.size else "N/A",
                "Edge (Theo)": f"{t.edge_theoretical:.1%}" if t.edge_theoretical else "N/A",
                "Edge (Real)": f"{t.edge_realized:.1%}" if t.edge_realized else "Pending",
                "P&L": f"${t.pnl:.2f}" if t.pnl is not None else "Pending",
            }
            for t in recent
        ])
        st.dataframe(trade_df, use_container_width=True, hide_index=True)
    else:
        st.info("No trades yet")

    # Recent Observations
    st.subheader("Recent Observations")

    if observations:
        recent_obs = sorted(observations, key=lambda o: o.created_at, reverse=True)[:20]
        obs_df = pd.DataFrame([
            {
                "Time": o.timestamp.strftime("%Y-%m-%d %H:%M") if o.timestamp else "N/A",
                "Match": o.match_id[:20] + "..." if len(o.match_id) > 20 else o.match_id,
                "Event": o.event_type,
                "Model Pred": f"{o.model_prediction:.2%}" if o.model_prediction else "N/A",
                "Market Price": f"{o.polymarket_price:.2%}" if o.polymarket_price else "N/A",
                "Edge": f"{o.edge_theoretical:.1%}" if o.polymarket_price else "N/A",
                "Latency": f"{o.latency_ms}ms" if o.latency_ms else "N/A",
            }
            for o in recent_obs
        ])
        st.dataframe(obs_df, use_container_width=True, hide_index=True)
    else:
        st.info("No observations yet")

    # Footer
    st.divider()
    st.caption(
        "Paper Trading Dashboard | "
        f"Data from: {db_path} | "
        "Run with: streamlit run src/paper_trading/dashboard.py"
    )


if __name__ == "__main__":
    main()
