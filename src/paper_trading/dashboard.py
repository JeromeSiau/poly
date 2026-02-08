"""Streamlit dashboard for paper trading monitoring."""

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

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

TWO_SIDED_EVENT_TYPE = "two_sided_inventory"
CONSERVATIVE_BID_MAX_AGE_MINUTES = 20.0


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


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _observation_game_state(observation: LiveObservation) -> dict[str, Any]:
    if isinstance(observation.game_state, dict):
        return observation.game_state
    return {}


def _format_sell_reason(reason: str) -> str:
    labels = {
        "ready_to_sell": "ready to sell",
        "below_min_order": "blocked: below min order",
        "missing_bid_mark": "blocked: missing bid",
        "missing_fair_mark": "blocked: missing fair",
        "edge_below_exit": "hold: edge below exit",
        "hold_time_not_reached": "hold: max-hold not reached",
        "waiting_rebalance": "hold: waiting rebalance",
    }
    return labels.get(reason, reason)


def extract_two_sided_trade_rows(
    observations: list[LiveObservation],
    trades: list[PaperTrade],
) -> list[dict[str, Any]]:
    """Build normalized rows for two-sided experiment analysis."""
    obs_by_id = {obs.id: obs for obs in observations}
    rows: list[dict[str, Any]] = []

    for trade in trades:
        obs = obs_by_id.get(trade.observation_id)
        if obs is None or obs.event_type != TWO_SIDED_EVENT_TYPE:
            continue

        state = _observation_game_state(obs)
        strategy_tag = str(state.get("strategy_tag") or "default")
        condition_id = str(state.get("condition_id") or obs.match_id or "")
        if not condition_id:
            continue

        fill_price = _safe_float(
            trade.simulated_fill_price if trade.simulated_fill_price is not None else trade.entry_price,
            default=0.0,
        )
        size_usd = _safe_float(trade.size, default=0.0)
        shares = _safe_float(state.get("shares"), default=0.0)
        if shares <= 0 and fill_price > 0:
            shares = size_usd / fill_price

        side = str(trade.side or state.get("side") or "").upper()
        if side not in {"BUY", "SELL"}:
            continue
        signed_shares = shares if side == "BUY" else -shares

        rows.append(
            {
                "observation_id": obs.id,
                "trade_id": trade.id,
                "timestamp": trade.created_at or obs.timestamp,
                "strategy_tag": strategy_tag,
                "condition_id": condition_id,
                "title": str(state.get("title") or condition_id),
                "outcome": str(state.get("outcome") or ""),
                "side": side,
                "shares": shares,
                "signed_shares": signed_shares,
                "size_usd": size_usd,
                "fill_price": fill_price,
                "fair_price": _safe_float(state.get("fair_price"), default=0.0),
                "market_bid": _safe_float(state.get("market_bid"), default=0.0),
                "market_ask": _safe_float(state.get("market_ask"), default=0.0),
                "exit_edge_pct": _safe_float(state.get("exit_edge_pct"), default=0.0),
                "min_order_usd": _safe_float(state.get("min_order_usd"), default=0.0),
                "max_hold_seconds": _safe_float(state.get("max_hold_seconds"), default=0.0),
                "max_outcome_inventory_usd": _safe_float(state.get("max_outcome_inventory_usd"), default=0.0),
                "fee_pct": _safe_float(state.get("fee_pct"), default=0.0),
                "edge_theoretical": _safe_float(trade.edge_theoretical),
                "edge_realized": _safe_float(trade.edge_realized),
                "pnl": _safe_float(trade.pnl),
            }
        )
    return rows


def available_two_sided_tags(rows: list[dict[str, Any]]) -> list[str]:
    tags = sorted({str(row.get("strategy_tag", "")).strip() for row in rows if row.get("strategy_tag")})
    return [tag for tag in tags if tag]


def filter_scope_by_strategy_tag(
    observations: list[LiveObservation],
    trades: list[PaperTrade],
    strategy_tag: str,
) -> tuple[list[LiveObservation], list[PaperTrade]]:
    if strategy_tag == "All":
        return observations, trades

    obs_ids: set[int] = set()
    filtered_observations: list[LiveObservation] = []
    for obs in observations:
        if obs.event_type != TWO_SIDED_EVENT_TYPE:
            continue
        state = _observation_game_state(obs)
        tag = str(state.get("strategy_tag") or "default")
        if tag == strategy_tag:
            filtered_observations.append(obs)
            if obs.id is not None:
                obs_ids.add(int(obs.id))

    filtered_trades = [trade for trade in trades if int(trade.observation_id) in obs_ids]
    return filtered_observations, filtered_trades


def summarize_two_sided_pairs(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Aggregate two-sided rows by strategy tag and condition."""
    if not rows:
        return pd.DataFrame()

    buckets: dict[tuple[str, str, str], dict[str, float]] = {}
    for row in rows:
        key = (
            str(row.get("strategy_tag") or "default"),
            str(row.get("condition_id") or ""),
            str(row.get("title") or ""),
        )
        cur = buckets.get(key)
        if cur is None:
            cur = {
                "trades": 0.0,
                "sells": 0.0,
                "win_sells": 0.0,
                "realized_pnl": 0.0,
                "gross_notional": 0.0,
                "net_shares": 0.0,
                "sum_edge_theoretical": 0.0,
                "sum_edge_realized_sells": 0.0,
            }
            buckets[key] = cur

        side = str(row.get("side") or "")
        pnl = _safe_float(row.get("pnl"))
        cur["trades"] += 1
        cur["realized_pnl"] += pnl
        cur["gross_notional"] += _safe_float(row.get("size_usd"))
        cur["net_shares"] += _safe_float(row.get("signed_shares"))
        cur["sum_edge_theoretical"] += _safe_float(row.get("edge_theoretical"))
        if side == "SELL":
            cur["sells"] += 1
            cur["sum_edge_realized_sells"] += _safe_float(row.get("edge_realized"))
            if pnl > 0:
                cur["win_sells"] += 1

    summary_rows: list[dict[str, Any]] = []
    for (tag, condition_id, title), cur in buckets.items():
        sells = int(cur["sells"])
        trades = int(cur["trades"])
        win_rate = (cur["win_sells"] / sells) if sells > 0 else 0.0
        avg_edge_theoretical = (cur["sum_edge_theoretical"] / trades) if trades > 0 else 0.0
        avg_edge_realized_sells = (cur["sum_edge_realized_sells"] / sells) if sells > 0 else 0.0
        summary_rows.append(
            {
                "strategy_tag": tag,
                "condition_id": condition_id,
                "title": title,
                "trades": trades,
                "sells": sells,
                "win_rate_sells": win_rate,
                "realized_pnl": cur["realized_pnl"],
                "gross_notional": cur["gross_notional"],
                "net_shares": cur["net_shares"],
                "avg_edge_theoretical": avg_edge_theoretical,
                "avg_edge_realized_sells": avg_edge_realized_sells,
            }
        )

    return pd.DataFrame(summary_rows).sort_values(
        ["strategy_tag", "realized_pnl"],
        ascending=[True, False],
    )


def build_two_sided_open_inventory(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Estimate open inventory and unrealized P&L from stored two-sided rows."""
    if not rows:
        return pd.DataFrame()

    ordered = sorted(
        rows,
        key=lambda row: (
            row.get("timestamp") or datetime.min,
            int(row.get("trade_id") or 0),
        ),
    )
    state: dict[tuple[str, str, str, str], dict[str, Any]] = {}

    for row in ordered:
        key = (
            str(row.get("strategy_tag") or "default"),
            str(row.get("condition_id") or ""),
            str(row.get("title") or ""),
            str(row.get("outcome") or ""),
        )
        if not key[1] or not key[3]:
            continue

        cur = state.get(key)
        if cur is None:
            cur = {
                "shares": 0.0,
                "avg_entry_price": 0.0,
                "mark_price": 0.0,
                "mark_timestamp": None,
                "latest_fair_price": 0.0,
                "latest_bid_price": 0.0,
                "latest_bid_timestamp": None,
                "exit_edge_pct": 0.0,
                "min_order_usd": 0.0,
                "max_hold_seconds": 0.0,
                "max_outcome_inventory_usd": 0.0,
                "fee_pct": 0.0,
                "first_buy_timestamp": None,
                "latest_timestamp": None,
            }
            state[key] = cur

        shares = _safe_float(row.get("shares"), default=0.0)
        fill_price = _safe_float(row.get("fill_price"), default=0.0)
        side = str(row.get("side") or "")
        ts = row.get("timestamp")

        mark_price = _safe_float(row.get("market_bid"), default=0.0)
        fair_price = _safe_float(row.get("fair_price"), default=0.0)
        if mark_price <= 0:
            mark_price = fair_price
        if mark_price <= 0:
            mark_price = fill_price
        if mark_price > 0:
            cur["mark_price"] = mark_price
            cur["mark_timestamp"] = ts
        if fair_price > 0:
            cur["latest_fair_price"] = fair_price
        bid_price = _safe_float(row.get("market_bid"), default=0.0)
        if bid_price > 0:
            cur["latest_bid_price"] = bid_price
            cur["latest_bid_timestamp"] = ts
        exit_edge_pct = _safe_float(row.get("exit_edge_pct"), default=0.0)
        if exit_edge_pct > 0:
            cur["exit_edge_pct"] = exit_edge_pct
        min_order_usd = _safe_float(row.get("min_order_usd"), default=0.0)
        if min_order_usd > 0:
            cur["min_order_usd"] = min_order_usd
        max_hold_seconds = _safe_float(row.get("max_hold_seconds"), default=0.0)
        if max_hold_seconds > 0:
            cur["max_hold_seconds"] = max_hold_seconds
        max_outcome_inventory_usd = _safe_float(row.get("max_outcome_inventory_usd"), default=0.0)
        if max_outcome_inventory_usd > 0:
            cur["max_outcome_inventory_usd"] = max_outcome_inventory_usd
        fee_pct = _safe_float(row.get("fee_pct"), default=0.0)
        if fee_pct > 0:
            cur["fee_pct"] = fee_pct
        if isinstance(ts, datetime):
            cur["latest_timestamp"] = ts

        if shares <= 0 or fill_price <= 0:
            continue

        if side == "BUY":
            prev_shares = _safe_float(cur["shares"], default=0.0)
            total = prev_shares + shares
            if total > 0:
                if prev_shares > 0:
                    cur["avg_entry_price"] = (
                        _safe_float(cur["avg_entry_price"], default=0.0) * prev_shares
                        + fill_price * shares
                    ) / total
                else:
                    cur["avg_entry_price"] = fill_price
                cur["shares"] = total
            if isinstance(ts, datetime) and cur["first_buy_timestamp"] is None:
                cur["first_buy_timestamp"] = ts
        elif side == "SELL":
            prev_shares = _safe_float(cur["shares"], default=0.0)
            sold = min(prev_shares, shares)
            remaining = prev_shares - sold
            if remaining <= 1e-9:
                cur["shares"] = 0.0
                cur["avg_entry_price"] = 0.0
            else:
                cur["shares"] = remaining

    now = datetime.now()
    rows_out: list[dict[str, Any]] = []
    for (tag, condition_id, title, outcome), cur in state.items():
        open_shares = _safe_float(cur.get("shares"), default=0.0)
        if open_shares <= 1e-9:
            continue
        avg_entry = _safe_float(cur.get("avg_entry_price"), default=0.0)
        mark = _safe_float(cur.get("mark_price"), default=0.0)
        unrealized = open_shares * (mark - avg_entry)
        mark_ts = cur.get("mark_timestamp")
        mark_age_minutes: Optional[float] = None
        if isinstance(mark_ts, datetime):
            ts = mark_ts
            if ts.tzinfo is not None:
                ts = ts.replace(tzinfo=None)
            mark_age_minutes = max(0.0, (now - ts).total_seconds() / 60.0)

        fair = _safe_float(cur.get("latest_fair_price"), default=0.0)
        bid = _safe_float(cur.get("latest_bid_price"), default=0.0)
        bid_ts = cur.get("latest_bid_timestamp")
        bid_age_minutes: Optional[float] = None
        if isinstance(bid_ts, datetime):
            ts = bid_ts
            if ts.tzinfo is not None:
                ts = ts.replace(tzinfo=None)
            bid_age_minutes = max(0.0, (now - ts).total_seconds() / 60.0)
        fee_pct = _safe_float(cur.get("fee_pct"), default=0.0)
        exit_edge = _safe_float(cur.get("exit_edge_pct"), default=0.0)
        min_order = _safe_float(cur.get("min_order_usd"), default=0.0)
        max_hold = _safe_float(cur.get("max_hold_seconds"), default=0.0)
        max_outcome_inv = _safe_float(cur.get("max_outcome_inventory_usd"), default=0.0)
        first_buy_ts = cur.get("first_buy_timestamp")
        hold_age_seconds: Optional[float] = None
        if isinstance(first_buy_ts, datetime):
            ts = first_buy_ts
            if ts.tzinfo is not None:
                ts = ts.replace(tzinfo=None)
            hold_age_seconds = max(0.0, (now - ts).total_seconds())

        edge_sell_est: Optional[float] = None
        if bid > 0 and fair > 0:
            edge_sell_est = bid - fair - fee_pct
        inv_value = open_shares * fair if fair > 0 else 0.0
        open_notional = open_shares * bid if bid > 0 else 0.0
        fresh_bid = bool(
            bid > 0
            and bid_age_minutes is not None
            and bid_age_minutes <= CONSERVATIVE_BID_MAX_AGE_MINUTES
        )
        conservative_mark = bid if fresh_bid else 0.0
        conservative_mark_reason = "fresh_bid" if fresh_bid else ("stale_or_missing_bid_zero")
        unrealized_conservative = open_shares * (conservative_mark - avg_entry)
        open_notional_conservative = open_shares * conservative_mark
        stale_exit = bool(
            max_hold > 0
            and hold_age_seconds is not None
            and hold_age_seconds >= max_hold
            and bid > 0
            and bid >= (avg_entry + fee_pct)
        )
        risk_exit = bool(max_outcome_inv > 0 and inv_value > max_outcome_inv)
        edge_ready = bool(edge_sell_est is not None and edge_sell_est >= exit_edge)
        sell_signal = edge_ready or stale_exit or risk_exit
        sell_trigger_parts: list[str] = []
        if edge_ready:
            sell_trigger_parts.append("edge")
        if stale_exit:
            sell_trigger_parts.append("max_hold")
        if risk_exit:
            sell_trigger_parts.append("max_inventory")
        sell_trigger = "+".join(sell_trigger_parts) if sell_trigger_parts else "none"

        if sell_signal and min_order > 0 and open_notional > 0 and open_notional < min_order:
            sell_block_reason = "below_min_order"
        elif sell_signal:
            sell_block_reason = "ready_to_sell"
        elif bid <= 0:
            sell_block_reason = "missing_bid_mark"
        elif fair <= 0:
            sell_block_reason = "missing_fair_mark"
        elif edge_sell_est is not None and edge_sell_est < exit_edge:
            sell_block_reason = "edge_below_exit"
        elif max_hold > 0 and hold_age_seconds is not None and hold_age_seconds < max_hold:
            sell_block_reason = "hold_time_not_reached"
        else:
            sell_block_reason = "waiting_rebalance"

        rows_out.append(
            {
                "strategy_tag": tag,
                "condition_id": condition_id,
                "title": title,
                "outcome": outcome,
                "open_shares": open_shares,
                "avg_entry_price": avg_entry,
                "mark_price": mark,
                "unrealized_pnl": unrealized,
                "unrealized_pnl_mark": unrealized,
                "conservative_mark_price": conservative_mark,
                "conservative_mark_reason": conservative_mark_reason,
                "unrealized_conservative": unrealized_conservative,
                "mark_timestamp": mark_ts,
                "mark_age_minutes": mark_age_minutes,
                "bid_age_minutes": bid_age_minutes,
                "fair_price": fair,
                "edge_sell_est": edge_sell_est,
                "exit_edge_pct": exit_edge,
                "open_notional": open_notional,
                "open_notional_conservative": open_notional_conservative,
                "hold_age_seconds": hold_age_seconds,
                "sell_signal": sell_signal,
                "sell_trigger": sell_trigger,
                "sell_block_reason": sell_block_reason,
            }
        )

    if not rows_out:
        return pd.DataFrame()
    return pd.DataFrame(rows_out).sort_values(["strategy_tag", "unrealized_conservative"], ascending=[True, True])


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

    two_sided_rows = extract_two_sided_trade_rows(observations, trades)
    two_sided_tags = available_two_sided_tags(two_sided_rows)
    selected_tag = st.sidebar.selectbox(
        "Strategy Tag",
        ["All"] + two_sided_tags,
        help="Filter dashboard to one two-sided experiment tag.",
    )

    observations_view, trades_view = filter_scope_by_strategy_tag(
        observations=observations,
        trades=trades,
        strategy_tag=selected_tag,
    )
    two_sided_rows_view = [
        row for row in two_sided_rows
        if selected_tag == "All" or row.get("strategy_tag") == selected_tag
    ]
    two_sided_summary_df = summarize_two_sided_pairs(two_sided_rows_view)
    two_sided_open_inventory_df = build_two_sided_open_inventory(two_sided_rows_view)
    if not two_sided_summary_df.empty:
        if not two_sided_open_inventory_df.empty:
            pair_unrealized = (
                two_sided_open_inventory_df
                .groupby(["strategy_tag", "condition_id", "title"], as_index=False)
                .agg(
                    unrealized_pnl_mark=("unrealized_pnl_mark", "sum"),
                    unrealized_conservative=("unrealized_conservative", "sum"),
                    open_outcomes=("outcome", "count"),
                    max_mark_age_minutes=("mark_age_minutes", "max"),
                )
            )
            two_sided_summary_df = two_sided_summary_df.merge(
                pair_unrealized,
                on=["strategy_tag", "condition_id", "title"],
                how="left",
            )
        if "unrealized_pnl_mark" not in two_sided_summary_df.columns:
            two_sided_summary_df["unrealized_pnl_mark"] = 0.0
        if "unrealized_conservative" not in two_sided_summary_df.columns:
            two_sided_summary_df["unrealized_conservative"] = 0.0
        if "open_outcomes" not in two_sided_summary_df.columns:
            two_sided_summary_df["open_outcomes"] = 0
        if "max_mark_age_minutes" not in two_sided_summary_df.columns:
            two_sided_summary_df["max_mark_age_minutes"] = 0.0
        two_sided_summary_df["unrealized_pnl_mark"] = two_sided_summary_df["unrealized_pnl_mark"].fillna(0.0)
        two_sided_summary_df["unrealized_conservative"] = two_sided_summary_df["unrealized_conservative"].fillna(0.0)
        # Keep `unrealized_pnl` as the trusted conservative series for exports/tables.
        two_sided_summary_df["unrealized_pnl"] = two_sided_summary_df["unrealized_conservative"]
        two_sided_summary_df["open_outcomes"] = two_sided_summary_df["open_outcomes"].fillna(0).astype(int)
        two_sided_summary_df["max_mark_age_minutes"] = two_sided_summary_df["max_mark_age_minutes"].fillna(0.0)
        two_sided_summary_df["total_pnl_mark"] = (
            two_sided_summary_df["realized_pnl"] + two_sided_summary_df["unrealized_pnl_mark"]
        )
        two_sided_summary_df["total_pnl"] = (
            two_sided_summary_df["realized_pnl"] + two_sided_summary_df["unrealized_pnl"]
        )

    # Calculate metrics
    metrics = calculate_metrics(trades_view)

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

    if not two_sided_summary_df.empty:
        st.caption("Two-sided risk view (current tag filter)")
        ts_realized = float(two_sided_summary_df["realized_pnl"].sum())
        ts_unrealized_mark = float(two_sided_summary_df["unrealized_pnl_mark"].sum())
        ts_unrealized = float(two_sided_summary_df["unrealized_pnl"].sum())
        ts_total = float(two_sided_summary_df["total_pnl"].sum())
        risk_cols = st.columns(4)
        risk_cols[0].metric("Two-Sided Realized", f"${ts_realized:,.2f}")
        risk_cols[1].metric("Unrealized (Conservative)", f"${ts_unrealized:,.2f}")
        risk_cols[2].metric("Total (Conservative)", f"${ts_total:,.2f}")
        risk_cols[3].metric("Unrealized (Mark)", f"${ts_unrealized_mark:,.2f}")

    st.divider()

    # Charts row
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("P&L Over Time")
        if trades_view:
            fig_pnl = create_pnl_chart(trades_view)
            st.plotly_chart(fig_pnl, use_container_width=True)
        else:
            st.info("No trades yet")

    with col_right:
        st.subheader("Edge Analysis")
        fig_edge = create_edge_analysis_chart(trades_view)
        st.plotly_chart(fig_edge, use_container_width=True)

    st.divider()

    # Calibration
    st.subheader("Model Calibration")
    fig_calibration = create_reliability_diagram(observations_view)
    st.plotly_chart(fig_calibration, use_container_width=True)

    st.divider()

    st.subheader("Two-Sided P&L By Pair")
    if two_sided_summary_df.empty:
        st.info("No two-sided rows for current filter.")
    else:
        col_a, col_b, col_c, col_d, col_e, col_f = st.columns(6)
        col_a.metric("Pairs", int(two_sided_summary_df.shape[0]))
        col_b.metric("Two-Sided Trades", int(len(two_sided_rows_view)))
        col_c.metric("Realized P&L", f"${two_sided_summary_df['realized_pnl'].sum():,.2f}")
        col_d.metric("Unrealized (Conservative)", f"${two_sided_summary_df['unrealized_pnl'].sum():,.2f}")
        open_pairs = int((two_sided_summary_df["open_outcomes"] > 0).sum())
        col_e.metric("Total P&L", f"${two_sided_summary_df['total_pnl'].sum():,.2f}")
        col_f.metric("Open Pairs", open_pairs)

        st.caption("Best pairs (conservative total P&L = realized + conservative unrealized)")
        top_pairs = two_sided_summary_df.nlargest(20, "total_pnl").copy()
        top_pairs["win_rate_sells"] = top_pairs["win_rate_sells"].map(lambda x: f"{x:.1%}")
        top_pairs["avg_edge_theoretical"] = top_pairs["avg_edge_theoretical"].map(lambda x: f"{x:.2%}")
        top_pairs["avg_edge_realized_sells"] = top_pairs["avg_edge_realized_sells"].map(lambda x: f"{x:.2%}")
        top_pairs["max_mark_age_minutes"] = top_pairs["max_mark_age_minutes"].map(lambda x: f"{x:.1f}")
        st.dataframe(top_pairs, use_container_width=True, hide_index=True)

        st.caption("Worst pairs (conservative total P&L = realized + conservative unrealized)")
        worst_pairs = two_sided_summary_df.nsmallest(20, "total_pnl").copy()
        worst_pairs["win_rate_sells"] = worst_pairs["win_rate_sells"].map(lambda x: f"{x:.1%}")
        worst_pairs["avg_edge_theoretical"] = worst_pairs["avg_edge_theoretical"].map(lambda x: f"{x:.2%}")
        worst_pairs["avg_edge_realized_sells"] = worst_pairs["avg_edge_realized_sells"].map(lambda x: f"{x:.2%}")
        worst_pairs["max_mark_age_minutes"] = worst_pairs["max_mark_age_minutes"].map(lambda x: f"{x:.1f}")
        st.dataframe(worst_pairs, use_container_width=True, hide_index=True)

        st.caption("Open inventory by outcome (conservative vs mark unrealized)")
        if two_sided_open_inventory_df.empty:
            st.info("No open two-sided inventory.")
        else:
            inv_view = two_sided_open_inventory_df.copy()
            inv_view = inv_view.sort_values("unrealized_conservative")
            inv_view = inv_view[
                [
                    "strategy_tag",
                    "condition_id",
                    "title",
                    "outcome",
                    "open_shares",
                    "avg_entry_price",
                    "conservative_mark_price",
                    "conservative_mark_reason",
                    "unrealized_conservative",
                    "mark_price",
                    "unrealized_pnl_mark",
                    "fair_price",
                    "edge_sell_est",
                    "exit_edge_pct",
                    "sell_signal",
                    "sell_trigger",
                    "sell_block_reason",
                    "open_notional",
                    "open_notional_conservative",
                    "hold_age_seconds",
                    "bid_age_minutes",
                    "mark_age_minutes",
                ]
            ]
            inv_view["avg_entry_price"] = inv_view["avg_entry_price"].map(lambda x: f"{x:.3f}")
            inv_view["conservative_mark_price"] = inv_view["conservative_mark_price"].map(lambda x: f"{x:.3f}")
            inv_view["mark_price"] = inv_view["mark_price"].map(lambda x: f"{x:.3f}")
            inv_view["fair_price"] = inv_view["fair_price"].map(lambda x: f"{x:.3f}")
            inv_view["edge_sell_est"] = inv_view["edge_sell_est"].map(
                lambda x: "n/a" if x is None else f"{float(x):.2%}"
            )
            inv_view["exit_edge_pct"] = inv_view["exit_edge_pct"].map(lambda x: f"{float(x):.2%}")
            inv_view["open_shares"] = inv_view["open_shares"].map(lambda x: f"{x:.2f}")
            inv_view["sell_signal"] = inv_view["sell_signal"].map(lambda x: "yes" if bool(x) else "no")
            inv_view["sell_block_reason"] = inv_view["sell_block_reason"].map(_format_sell_reason)
            inv_view["open_notional"] = inv_view["open_notional"].map(lambda x: f"${float(x):,.2f}")
            inv_view["open_notional_conservative"] = inv_view["open_notional_conservative"].map(
                lambda x: f"${float(x):,.2f}"
            )
            inv_view["hold_age_seconds"] = inv_view["hold_age_seconds"].map(
                lambda x: "n/a" if x is None else f"{float(x) / 60.0:.1f}m"
            )
            inv_view["unrealized_conservative"] = inv_view["unrealized_conservative"].map(lambda x: f"${x:,.2f}")
            inv_view["unrealized_pnl_mark"] = inv_view["unrealized_pnl_mark"].map(lambda x: f"${x:,.2f}")
            inv_view["bid_age_minutes"] = inv_view["bid_age_minutes"].map(
                lambda x: "n/a" if x is None else f"{float(x):.1f}"
            )
            inv_view["mark_age_minutes"] = inv_view["mark_age_minutes"].map(
                lambda x: "n/a" if x is None else f"{float(x):.1f}"
            )
            st.dataframe(inv_view, use_container_width=True, hide_index=True)

    st.divider()

    # Recent Trades table
    st.subheader("Recent Trades")

    if trades_view:
        recent = sorted(trades_view, key=lambda t: t.created_at, reverse=True)[:20]
        obs_by_id = {obs.id: obs for obs in observations}
        trade_df = pd.DataFrame([
            {
                "Time": t.created_at.strftime("%Y-%m-%d %H:%M") if t.created_at else "N/A",
                "Tag": str(_observation_game_state(obs_by_id.get(t.observation_id)).get("strategy_tag") or "")
                if obs_by_id.get(t.observation_id) is not None else "",
                "Condition": str(_observation_game_state(obs_by_id.get(t.observation_id)).get("condition_id") or "")
                if obs_by_id.get(t.observation_id) is not None else "",
                "Outcome": str(_observation_game_state(obs_by_id.get(t.observation_id)).get("outcome") or "")
                if obs_by_id.get(t.observation_id) is not None else "",
                "Side": t.side,
                "Entry Price": f"{t.entry_price:.2%}" if t.entry_price else "N/A",
                "Fill Price": f"{t.simulated_fill_price:.2%}" if t.simulated_fill_price else "N/A",
                "Book Bid": (
                    f"{_safe_float(_observation_game_state(obs_by_id.get(t.observation_id)).get('market_bid')):.2%}"
                    if obs_by_id.get(t.observation_id) is not None
                    and _safe_float(_observation_game_state(obs_by_id.get(t.observation_id)).get("market_bid")) > 0
                    else "N/A"
                ),
                "Book Ask": (
                    f"{_safe_float(_observation_game_state(obs_by_id.get(t.observation_id)).get('market_ask')):.2%}"
                    if obs_by_id.get(t.observation_id) is not None
                    and _safe_float(_observation_game_state(obs_by_id.get(t.observation_id)).get("market_ask")) > 0
                    else "N/A"
                ),
                "Size": f"${t.size:.2f}" if t.size else "N/A",
                "Reason": str(_observation_game_state(obs_by_id.get(t.observation_id)).get("reason") or "")
                if obs_by_id.get(t.observation_id) is not None else "",
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

    if observations_view:
        recent_obs = sorted(observations_view, key=lambda o: o.created_at, reverse=True)[:20]
        obs_df = pd.DataFrame([
            {
                "Time": o.timestamp.strftime("%Y-%m-%d %H:%M") if o.timestamp else "N/A",
                "Match": o.match_id[:20] + "..." if len(o.match_id) > 20 else o.match_id,
                "Tag": str(_observation_game_state(o).get("strategy_tag") or ""),
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
