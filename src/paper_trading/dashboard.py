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

from src.db.models import FearPosition, LiveObservation, PaperTrade
from src.ml.validation.calibration import reliability_diagram_data
from src.paper_trading.metrics import PaperTradingMetrics, TradeRecord

TWO_SIDED_EVENT_TYPE = "two_sided_inventory"
SNIPER_EVENT_TYPE = "sniper_sports"
CRYPTO_MINUTE_EVENT_TYPE = "crypto_minute"
WEATHER_ORACLE_EVENT_TYPE = "weather_oracle"
CRYPTO_MAKER_EVENT_TYPE = "crypto_maker"
TD_MAKER_EVENT_TYPE = "crypto_td_maker"
CONSERVATIVE_BID_MAX_AGE_MINUTES = 20.0

_LIVE_MODES = {"live", "live_fill", "autopilot", "settlement"}


def _is_live_mode(observation: LiveObservation) -> bool:
    """Return True if the observation was recorded in live (non-paper) mode."""
    gs = observation.game_state if isinstance(observation.game_state, dict) else {}
    mode = str(gs.get("mode", "paper")).lower()
    return mode in _LIVE_MODES


def filter_by_execution_mode(
    observations: list[LiveObservation],
    trades: list[PaperTrade],
    mode_filter: str,
) -> tuple[list[LiveObservation], list[PaperTrade]]:
    """Filter observations and trades by execution mode (Paper / Live / All)."""
    if mode_filter == "All":
        return observations, trades
    keep_live = mode_filter == "Live"
    filtered_obs = [o for o in observations if _is_live_mode(o) == keep_live]
    obs_ids = {int(o.id) for o in filtered_obs if o.id is not None}
    filtered_trades = [t for t in trades if int(t.observation_id) in obs_ids]
    return filtered_obs, filtered_trades


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


def load_fear_positions(db_path: str = "data/arb.db") -> list:
    """Load fear selling positions from SQLite database.

    Args:
        db_path: Path to SQLite database file.

    Returns:
        List of FearPosition objects.
    """
    engine = create_engine(f"sqlite:///{db_path}")
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        positions = session.query(FearPosition).all()
    except Exception:
        positions = []

    session.close()
    return positions


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
    event_types: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Build normalized rows for two-sided experiment analysis."""
    if event_types is None:
        event_types = {TWO_SIDED_EVENT_TYPE}
    obs_by_id = {obs.id: obs for obs in observations}
    rows: list[dict[str, Any]] = []

    for trade in trades:
        obs = obs_by_id.get(trade.observation_id)
        if obs is None or obs.event_type not in event_types:
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
    event_types: set[str] | None = None,
) -> tuple[list[LiveObservation], list[PaperTrade]]:
    if event_types is None:
        event_types = {TWO_SIDED_EVENT_TYPE}
    if strategy_tag == "All":
        filtered_obs = [o for o in observations if o.event_type in event_types]
        obs_ids = {int(o.id) for o in filtered_obs if o.id is not None}
        filtered_trades = [t for t in trades if int(t.observation_id) in obs_ids]
        return filtered_obs, filtered_trades

    obs_ids: set[int] = set()
    filtered_observations: list[LiveObservation] = []
    for obs in observations:
        if obs.event_type not in event_types:
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


def summarize_two_sided_by_tag(
    rows: list[dict[str, Any]],
    open_inventory_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate strategy performance by experiment tag."""
    if not rows:
        return pd.DataFrame()

    buckets: dict[str, dict[str, float]] = {}
    for row in rows:
        tag = str(row.get("strategy_tag") or "default")
        cur = buckets.get(tag)
        if cur is None:
            cur = {
                "trades": 0.0,
                "sells": 0.0,
                "win_sells": 0.0,
                "realized_pnl": 0.0,
                "gross_notional": 0.0,
                "sum_edge_theoretical": 0.0,
                "sum_edge_realized_sells": 0.0,
            }
            buckets[tag] = cur

        side = str(row.get("side") or "").upper()
        pnl = _safe_float(row.get("pnl"))
        cur["trades"] += 1
        cur["realized_pnl"] += pnl
        cur["gross_notional"] += _safe_float(row.get("size_usd"))
        cur["sum_edge_theoretical"] += _safe_float(row.get("edge_theoretical"))
        if side == "SELL":
            cur["sells"] += 1
            cur["sum_edge_realized_sells"] += _safe_float(row.get("edge_realized"))
            if pnl > 0:
                cur["win_sells"] += 1

    summary_rows: list[dict[str, Any]] = []
    for tag, cur in buckets.items():
        trades = int(cur["trades"])
        sells = int(cur["sells"])
        summary_rows.append(
            {
                "strategy_tag": tag,
                "trades": trades,
                "sells": sells,
                "win_rate_sells": (cur["win_sells"] / sells) if sells > 0 else 0.0,
                "realized_pnl": cur["realized_pnl"],
                "gross_notional": cur["gross_notional"],
                "avg_edge_theoretical": (cur["sum_edge_theoretical"] / trades) if trades > 0 else 0.0,
                "avg_edge_realized_sells": (cur["sum_edge_realized_sells"] / sells) if sells > 0 else 0.0,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    if summary_df.empty:
        return summary_df

    if not open_inventory_df.empty:
        inv = (
            open_inventory_df
            .groupby("strategy_tag", as_index=False)
            .agg(
                open_outcomes=("outcome", "count"),
                open_notional=("open_notional", "sum"),
                unrealized_conservative=("unrealized_conservative", "sum"),
                unrealized_pnl_mark=("unrealized_pnl_mark", "sum"),
                ready_to_sell=("sell_block_reason", lambda s: int((s == "ready_to_sell").sum())),
                blocked_missing_bid=("sell_block_reason", lambda s: int((s == "missing_bid_mark").sum())),
                max_mark_age_minutes=("mark_age_minutes", "max"),
            )
        )
        summary_df = summary_df.merge(inv, on="strategy_tag", how="left")
    else:
        summary_df["open_outcomes"] = 0
        summary_df["open_notional"] = 0.0
        summary_df["unrealized_conservative"] = 0.0
        summary_df["unrealized_pnl_mark"] = 0.0
        summary_df["ready_to_sell"] = 0
        summary_df["blocked_missing_bid"] = 0
        summary_df["max_mark_age_minutes"] = 0.0

    for col, default in [
        ("open_outcomes", 0),
        ("open_notional", 0.0),
        ("unrealized_conservative", 0.0),
        ("unrealized_pnl_mark", 0.0),
        ("ready_to_sell", 0),
        ("blocked_missing_bid", 0),
        ("max_mark_age_minutes", 0.0),
    ]:
        if col in summary_df.columns:
            summary_df[col] = summary_df[col].fillna(default)

    summary_df["open_outcomes"] = summary_df["open_outcomes"].astype(int)
    summary_df["ready_to_sell"] = summary_df["ready_to_sell"].astype(int)
    summary_df["blocked_missing_bid"] = summary_df["blocked_missing_bid"].astype(int)
    summary_df["total_pnl"] = summary_df["realized_pnl"] + summary_df["unrealized_conservative"]
    summary_df["total_pnl_mark"] = summary_df["realized_pnl"] + summary_df["unrealized_pnl_mark"]

    return summary_df.sort_values("total_pnl", ascending=False)


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


def extract_crypto_minute_rows(
    observations: list[LiveObservation],
    trades: list[PaperTrade],
) -> list[dict[str, Any]]:
    """Build rows for crypto minute strategy analysis."""
    obs_by_id = {obs.id: obs for obs in observations}
    rows: list[dict[str, Any]] = []

    for trade in trades:
        obs = obs_by_id.get(trade.observation_id)
        if obs is None or obs.event_type != CRYPTO_MINUTE_EVENT_TYPE:
            continue
        gs = _observation_game_state(obs)
        rows.append({
            "timestamp": trade.created_at or obs.timestamp,
            "strategy_tag": gs.get("strategy_tag", ""),
            "sub_strategy": gs.get("sub_strategy", ""),
            "symbol": gs.get("symbol", ""),
            "side": gs.get("outcome", ""),
            "slug": gs.get("slug", ""),
            "entry_price": _safe_float(trade.entry_price),
            "exit_price": _safe_float(trade.exit_price),
            "size_usd": _safe_float(trade.size),
            "pnl": _safe_float(trade.pnl),
            "edge_theoretical": _safe_float(trade.edge_theoretical),
            "gap_pct": _safe_float(gs.get("gap_pct")),
            "gap_bucket": gs.get("gap_bucket", ""),
            "time_bucket": gs.get("time_bucket", ""),
            "time_remaining_s": _safe_float(gs.get("time_remaining_s")),
            "spot_at_entry": _safe_float(gs.get("spot_at_entry")),
            "spot_at_resolution": _safe_float(gs.get("spot_at_resolution")),
            "won": _safe_float(trade.exit_price) >= 0.5,
        })
    return rows


def extract_weather_oracle_rows(
    observations: list[LiveObservation],
    trades: list[PaperTrade],
) -> list[dict[str, Any]]:
    """Build rows for weather oracle strategy analysis."""
    obs_by_id = {obs.id: obs for obs in observations}
    rows: list[dict[str, Any]] = []

    for trade in trades:
        obs = obs_by_id.get(trade.observation_id)
        if obs is None or obs.event_type != WEATHER_ORACLE_EVENT_TYPE:
            continue
        gs = _observation_game_state(obs)
        rows.append({
            "timestamp": trade.created_at or obs.timestamp,
            "city": gs.get("city", ""),
            "target_date": gs.get("target_date", ""),
            "outcome": gs.get("outcome", ""),
            "entry_price": _safe_float(trade.entry_price),
            "exit_price": _safe_float(trade.exit_price),
            "size_usd": _safe_float(trade.size),
            "pnl": _safe_float(trade.pnl),
            "confidence": _safe_float(gs.get("confidence")),
            "forecast_temp_max": _safe_float(gs.get("forecast_temp_max")),
            "forecast_temp_min": _safe_float(gs.get("forecast_temp_min")),
            "reason": gs.get("reason", ""),
            "slug": gs.get("slug", ""),
            "won": _safe_float(trade.exit_price) >= 0.5,
        })
    return rows


def _render_sniper_tab(
    observations: list[LiveObservation],
    trades: list[PaperTrade],
) -> None:
    """Render the Sniper Sports tab content."""
    # Accept both old (two_sided_inventory with sniper tag) and new (sniper_sports) event types
    sniper_event_types = {SNIPER_EVENT_TYPE, TWO_SIDED_EVENT_TYPE}
    all_rows = extract_two_sided_trade_rows(observations, trades, event_types=sniper_event_types)
    # Keep only sniper-tagged rows
    sniper_rows = [r for r in all_rows if "sniper" in str(r.get("strategy_tag", "")).lower()]

    if not sniper_rows:
        st.info("No sniper trades found.")
        return

    sniper_tags = sorted({r["strategy_tag"] for r in sniper_rows})
    selected = st.selectbox("Sniper Tag", ["All"] + sniper_tags, key="sniper_tag_select")
    rows_view = sniper_rows if selected == "All" else [r for r in sniper_rows if r["strategy_tag"] == selected]

    # Summary metrics
    pnls = [r["pnl"] for r in rows_view if r.get("pnl")]
    total_pnl = sum(pnls)
    buys = [r for r in rows_view if r["side"] == "BUY"]
    sells = [r for r in rows_view if r["side"] == "SELL"]
    sell_wins = sum(1 for r in sells if r["pnl"] > 0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Trades", len(rows_view))
    c2.metric("Buys / Sells", f"{len(buys)} / {len(sells)}")
    c3.metric("Sell Win Rate", f"{sell_wins / len(sells):.1%}" if sells else "N/A")
    c4.metric("Realized P&L", f"${total_pnl:,.2f}")

    # Open inventory
    open_inv_df = build_two_sided_open_inventory(rows_view)
    if not open_inv_df.empty:
        unrealized = float(open_inv_df["unrealized_pnl_mark"].sum())
        open_count = len(open_inv_df)
        c5, c6, c7, _ = st.columns(4)
        c5.metric("Open Positions", open_count)
        c6.metric("Unrealized P&L", f"${unrealized:,.2f}")
        c7.metric("Total P&L", f"${total_pnl + unrealized:,.2f}")

    st.divider()

    # Pair summary
    summary_df = summarize_two_sided_pairs(rows_view)
    if not summary_df.empty:
        if not open_inv_df.empty:
            pair_unr = (
                open_inv_df
                .groupby(["strategy_tag", "condition_id", "title"], as_index=False)
                .agg(unrealized_pnl_mark=("unrealized_pnl_mark", "sum"), open_outcomes=("outcome", "count"))
            )
            summary_df = summary_df.merge(pair_unr, on=["strategy_tag", "condition_id", "title"], how="left")
        for col, default in [("unrealized_pnl_mark", 0.0), ("open_outcomes", 0)]:
            if col not in summary_df.columns:
                summary_df[col] = default
            summary_df[col] = summary_df[col].fillna(default)
        summary_df["total_pnl"] = summary_df["realized_pnl"] + summary_df["unrealized_pnl_mark"]

        st.subheader("By Market")
        show_df = summary_df.sort_values("total_pnl", ascending=False).copy()
        show_df["win_rate_sells"] = show_df["win_rate_sells"].map(lambda x: f"{x:.1%}")
        show_df["avg_edge_theoretical"] = show_df["avg_edge_theoretical"].map(lambda x: f"{x:.2%}")
        st.dataframe(show_df, use_container_width=True, hide_index=True)

    # Open inventory detail
    if not open_inv_df.empty:
        st.subheader("Open Inventory")
        inv = open_inv_df.copy()
        inv = inv[["strategy_tag", "condition_id", "title", "outcome", "open_shares",
                    "avg_entry_price", "mark_price", "unrealized_pnl_mark",
                    "sell_signal", "sell_block_reason", "hold_age_seconds"]]
        inv["avg_entry_price"] = inv["avg_entry_price"].map(lambda x: f"{x:.3f}")
        inv["mark_price"] = inv["mark_price"].map(lambda x: f"{x:.3f}")
        inv["open_shares"] = inv["open_shares"].map(lambda x: f"{x:.2f}")
        inv["sell_signal"] = inv["sell_signal"].map(lambda x: "yes" if bool(x) else "no")
        inv["sell_block_reason"] = inv["sell_block_reason"].map(_format_sell_reason)
        inv["hold_age_seconds"] = inv["hold_age_seconds"].map(
            lambda x: "n/a" if x is None else f"{float(x) / 60.0:.1f}m"
        )
        st.dataframe(inv, use_container_width=True, hide_index=True)

    # Recent trades
    st.subheader("Recent Sniper Trades")
    recent = sorted(rows_view, key=lambda r: r.get("timestamp") or datetime.min, reverse=True)[:30]
    if recent:
        df = pd.DataFrame([{
            "Time": r["timestamp"].strftime("%Y-%m-%d %H:%M") if isinstance(r.get("timestamp"), datetime) else "N/A",
            "Tag": r.get("strategy_tag", ""),
            "Title": str(r.get("title", ""))[:50],
            "Outcome": r.get("outcome", ""),
            "Side": r.get("side", ""),
            "Price": f"{r['fill_price']:.3f}",
            "Size": f"${r['size_usd']:.2f}",
            "Edge": f"{r['edge_theoretical']:.2%}",
            "P&L": f"${r['pnl']:.2f}" if r.get("pnl") else "",
        } for r in recent])
        st.dataframe(df, use_container_width=True, hide_index=True)


def _render_crypto_minute_tab(
    observations: list[LiveObservation],
    trades: list[PaperTrade],
) -> None:
    """Render the Crypto Minute tab content."""
    rows = extract_crypto_minute_rows(observations, trades)

    if not rows:
        st.info("No crypto minute trades found.")
        return

    # Sub-strategy filter
    strategies = sorted({r["sub_strategy"] for r in rows if r["sub_strategy"]})
    selected = st.selectbox("Sub-Strategy", ["All"] + strategies, key="cm_substrategy_select")
    if selected != "All":
        rows = [r for r in rows if r["sub_strategy"] == selected]

    # Summary metrics
    pnls = [r["pnl"] for r in rows]
    total_pnl = sum(pnls)
    wins = sum(1 for p in pnls if p > 0)
    losses = sum(1 for p in pnls if p <= 0)
    avg_gap = sum(r["gap_pct"] for r in rows) / len(rows) if rows else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Trades", len(rows))
    c2.metric("Win Rate", f"{wins / len(rows):.1%}" if rows else "N/A")
    c3.metric("Total P&L", f"${total_pnl:,.2f}")
    c4.metric("Avg Gap %", f"{avg_gap:.2%}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Wins / Losses", f"{wins} / {losses}")
    avg_win = sum(p for p in pnls if p > 0) / wins if wins else 0
    avg_loss = sum(p for p in pnls if p <= 0) / losses if losses else 0
    c6.metric("Avg Win", f"${avg_win:,.2f}")
    c7.metric("Avg Loss", f"${avg_loss:,.2f}")
    c8.metric("Profit Factor", f"{abs(sum(p for p in pnls if p > 0) / sum(p for p in pnls if p < 0)):.2f}" if any(p < 0 for p in pnls) else "N/A")

    st.divider()

    # P&L by sub-strategy
    if len(strategies) > 1 and selected == "All":
        st.subheader("By Sub-Strategy")
        all_rows = extract_crypto_minute_rows(
            [o for o in observations if o.event_type == CRYPTO_MINUTE_EVENT_TYPE],
            trades,
        )
        strat_data: list[dict[str, Any]] = []
        for strat in strategies:
            s_rows = [r for r in all_rows if r["sub_strategy"] == strat]
            s_pnls = [r["pnl"] for r in s_rows]
            s_wins = sum(1 for p in s_pnls if p > 0)
            strat_data.append({
                "Strategy": strat,
                "Trades": len(s_rows),
                "Win Rate": f"{s_wins / len(s_rows):.1%}" if s_rows else "N/A",
                "Total P&L": f"${sum(s_pnls):,.2f}",
                "Avg P&L": f"${sum(s_pnls) / len(s_rows):,.4f}" if s_rows else "N/A",
            })
        st.dataframe(pd.DataFrame(strat_data), use_container_width=True, hide_index=True)
        st.divider()

    # P&L by symbol
    st.subheader("By Symbol")
    symbols = sorted({r["symbol"] for r in rows if r["symbol"]})
    sym_data: list[dict[str, Any]] = []
    for sym in symbols:
        s_rows = [r for r in rows if r["symbol"] == sym]
        s_pnls = [r["pnl"] for r in s_rows]
        s_wins = sum(1 for p in s_pnls if p > 0)
        sym_data.append({
            "Symbol": sym,
            "Trades": len(s_rows),
            "Win Rate": f"{s_wins / len(s_rows):.1%}" if s_rows else "N/A",
            "Total P&L": f"${sum(s_pnls):,.2f}",
            "Avg Gap": f"{sum(r['gap_pct'] for r in s_rows) / len(s_rows):.2%}" if s_rows else "N/A",
        })
    if sym_data:
        st.dataframe(pd.DataFrame(sym_data), use_container_width=True, hide_index=True)

    st.divider()

    # P&L over time chart
    st.subheader("Cumulative P&L")
    sorted_rows = sorted(rows, key=lambda r: r.get("timestamp") or datetime.min)
    if sorted_rows:
        cm_df = pd.DataFrame(sorted_rows)
        cm_df = cm_df.sort_values("timestamp")
        cm_df["cumulative_pnl"] = cm_df["pnl"].cumsum()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cm_df["timestamp"], y=cm_df["cumulative_pnl"],
            mode="lines+markers", name="Cumulative P&L",
            line=dict(color="#2ecc71", width=2), marker=dict(size=5),
        ))
        fig.update_layout(
            xaxis_title="Time", yaxis_title="Cumulative P&L ($)",
            height=350, template="plotly_dark",
            margin=dict(l=50, r=50, t=30, b=50),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Recent trades table
    st.subheader("Recent Trades")
    recent = sorted(rows, key=lambda r: r.get("timestamp") or datetime.min, reverse=True)[:50]
    if recent:
        df = pd.DataFrame([{
            "Time": r["timestamp"].strftime("%m-%d %H:%M") if isinstance(r.get("timestamp"), datetime) else "N/A",
            "Strategy": r["sub_strategy"],
            "Symbol": r["symbol"],
            "Side": r["side"],
            "Entry": f"{r['entry_price']:.3f}",
            "Exit": f"{r['exit_price']:.3f}",
            "Won": "Y" if r["won"] else "N",
            "Size": f"${r['size_usd']:.2f}",
            "P&L": f"${r['pnl']:.4f}",
            "Gap %": f"{r['gap_pct']:.2%}",
            "Gap Bucket": r["gap_bucket"],
            "Time Bucket": r["time_bucket"],
            "Time Left": f"{r['time_remaining_s']:.0f}s",
            "Spot Entry": f"${r['spot_at_entry']:,.0f}" if r["spot_at_entry"] else "",
            "Spot Resolve": f"${r['spot_at_resolution']:,.0f}" if r["spot_at_resolution"] else "",
        } for r in recent])
        st.dataframe(df, use_container_width=True, hide_index=True)


def _render_weather_oracle_tab(
    observations: list[LiveObservation],
    trades: list[PaperTrade],
) -> None:
    """Render the Weather Oracle tab content."""
    rows = extract_weather_oracle_rows(observations, trades)

    if not rows:
        st.info("No weather oracle trades found. Run: python scripts/run_weather_oracle.py watch")
        return

    pnls = [r["pnl"] for r in rows]
    total_pnl = sum(pnls)
    wins = sum(1 for p in pnls if p > 0)
    losses = sum(1 for p in pnls if p <= 0)
    total_invested = sum(r["size_usd"] for r in rows)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Trades", len(rows))
    c2.metric("Win Rate", f"{wins / len(rows):.1%}" if rows else "N/A")
    c3.metric("Total P&L", f"${total_pnl:,.2f}")
    c4.metric("ROI", f"{total_pnl / total_invested:.0%}" if total_invested else "N/A")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Wins / Losses", f"{wins} / {losses}")
    c6.metric("Total Invested", f"${total_invested:,.2f}")
    avg_win = sum(p for p in pnls if p > 0) / wins if wins else 0
    c7.metric("Avg Win", f"${avg_win:,.2f}")
    avg_conf = sum(r["confidence"] for r in rows) / len(rows) if rows else 0
    c8.metric("Avg Confidence", f"{avg_conf:.0%}")

    st.divider()

    st.subheader("By City")
    cities = sorted({r["city"] for r in rows if r["city"]})
    city_data: list[dict[str, Any]] = []
    for city in cities:
        c_rows = [r for r in rows if r["city"] == city]
        c_pnls = [r["pnl"] for r in c_rows]
        c_wins = sum(1 for p in c_pnls if p > 0)
        city_data.append({
            "City": city,
            "Trades": len(c_rows),
            "Win Rate": f"{c_wins / len(c_rows):.1%}" if c_rows else "N/A",
            "Total P&L": f"${sum(c_pnls):,.2f}",
            "Invested": f"${sum(r['size_usd'] for r in c_rows):,.2f}",
        })
    if city_data:
        st.dataframe(pd.DataFrame(city_data), use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("Cumulative P&L")
    sorted_rows = sorted(rows, key=lambda r: r.get("timestamp") or datetime.min)
    if sorted_rows:
        wo_df = pd.DataFrame(sorted_rows)
        wo_df = wo_df.sort_values("timestamp")
        wo_df["cumulative_pnl"] = wo_df["pnl"].cumsum()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=wo_df["timestamp"], y=wo_df["cumulative_pnl"],
            mode="lines+markers", name="Cumulative P&L",
            line=dict(color="#f39c12", width=2), marker=dict(size=5),
        ))
        fig.update_layout(
            xaxis_title="Time", yaxis_title="Cumulative P&L ($)",
            height=350, template="plotly_dark",
            margin=dict(l=50, r=50, t=30, b=50),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Recent Trades")
    recent = sorted(rows, key=lambda r: r.get("timestamp") or datetime.min, reverse=True)[:50]
    if recent:
        df = pd.DataFrame([{
            "Time": r["timestamp"].strftime("%m-%d %H:%M") if isinstance(r.get("timestamp"), datetime) else "N/A",
            "City": r["city"],
            "Date": r["target_date"],
            "Outcome": r["outcome"],
            "Entry": f"{r['entry_price']:.3f}",
            "Exit": f"{r['exit_price']:.3f}",
            "Won": "Y" if r["won"] else "N",
            "Size": f"${r['size_usd']:.2f}",
            "P&L": f"${r['pnl']:.2f}",
            "Conf": f"{r['confidence']:.0%}",
            "Forecast High": f"{r['forecast_temp_max']:.0f}°",
        } for r in recent])
        st.dataframe(df, use_container_width=True, hide_index=True)


def _render_crypto_maker_tab(
    observations: list[LiveObservation],
    trades: list[PaperTrade],
) -> None:
    """Render the Crypto Maker tab content."""
    rows = extract_two_sided_trade_rows(
        observations, trades, event_types={CRYPTO_MAKER_EVENT_TYPE}
    )

    if not rows:
        st.info("No crypto maker trades found. Run: python scripts/run_crypto_maker.py --paper")
        return

    tags = available_two_sided_tags(rows)
    selected_tag = st.selectbox("Strategy Tag", ["All"] + tags, key="cm_maker_tag_select")
    rows_view = rows if selected_tag == "All" else [r for r in rows if r["strategy_tag"] == selected_tag]

    # Summary metrics
    pnls = [r["pnl"] for r in rows_view if r.get("pnl")]
    total_pnl = sum(pnls)
    buys = [r for r in rows_view if r["side"] == "BUY"]
    sells = [r for r in rows_view if r["side"] == "SELL"]
    sell_wins = sum(1 for r in sells if r["pnl"] > 0)
    total_notional = sum(r["size_usd"] for r in rows_view)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Trades", len(rows_view))
    c2.metric("Buys / Sells", f"{len(buys)} / {len(sells)}")
    c3.metric("Sell Win Rate", f"{sell_wins / len(sells):.1%}" if sells else "N/A")
    c4.metric("Realized P&L", f"${total_pnl:,.2f}")

    # Open inventory
    open_inv_df = build_two_sided_open_inventory(rows_view)
    if not open_inv_df.empty:
        unrealized = float(open_inv_df["unrealized_pnl_mark"].sum())
        open_not = float(open_inv_df["open_notional"].sum())
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Open Positions", len(open_inv_df))
        c6.metric("Open Notional", f"${open_not:,.2f}")
        c7.metric("Unrealized P&L", f"${unrealized:,.2f}")
        c8.metric("Total P&L", f"${total_pnl + unrealized:,.2f}")

    st.divider()

    # P&L over time
    st.subheader("Cumulative P&L")
    sorted_rows = sorted(rows_view, key=lambda r: r.get("timestamp") or datetime.min)
    if sorted_rows:
        mk_df = pd.DataFrame(sorted_rows).sort_values("timestamp")
        mk_df["cumulative_pnl"] = mk_df["pnl"].cumsum()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=mk_df["timestamp"], y=mk_df["cumulative_pnl"],
            mode="lines+markers", name="Cumulative P&L",
            line=dict(color="#9b59b6", width=2), marker=dict(size=5),
        ))
        fig.update_layout(
            xaxis_title="Time", yaxis_title="Cumulative P&L ($)",
            height=350, template="plotly_dark",
            margin=dict(l=50, r=50, t=30, b=50),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # By-pair summary
    summary_df = summarize_two_sided_pairs(rows_view)
    if not summary_df.empty:
        if not open_inv_df.empty:
            pair_unr = (
                open_inv_df
                .groupby(["strategy_tag", "condition_id", "title"], as_index=False)
                .agg(unrealized_pnl_mark=("unrealized_pnl_mark", "sum"), open_outcomes=("outcome", "count"))
            )
            summary_df = summary_df.merge(pair_unr, on=["strategy_tag", "condition_id", "title"], how="left")
        for col, default in [("unrealized_pnl_mark", 0.0), ("open_outcomes", 0)]:
            if col not in summary_df.columns:
                summary_df[col] = default
            summary_df[col] = summary_df[col].fillna(default)
        summary_df["total_pnl"] = summary_df["realized_pnl"] + summary_df["unrealized_pnl_mark"]

        st.subheader("By Market")
        show_df = summary_df.sort_values("total_pnl", ascending=False).copy()
        show_df["win_rate_sells"] = show_df["win_rate_sells"].map(lambda x: f"{x:.1%}")
        show_df["avg_edge_theoretical"] = show_df["avg_edge_theoretical"].map(lambda x: f"{x:.2%}")
        st.dataframe(show_df, use_container_width=True, hide_index=True)

    # Open inventory detail
    if not open_inv_df.empty:
        st.subheader("Open Inventory")
        inv = open_inv_df.copy()
        inv = inv[["strategy_tag", "condition_id", "title", "outcome", "open_shares",
                    "avg_entry_price", "mark_price", "unrealized_pnl_mark",
                    "sell_signal", "sell_block_reason", "hold_age_seconds"]]
        inv["avg_entry_price"] = inv["avg_entry_price"].map(lambda x: f"{x:.3f}")
        inv["mark_price"] = inv["mark_price"].map(lambda x: f"{x:.3f}")
        inv["open_shares"] = inv["open_shares"].map(lambda x: f"{x:.2f}")
        inv["sell_signal"] = inv["sell_signal"].map(lambda x: "yes" if bool(x) else "no")
        inv["sell_block_reason"] = inv["sell_block_reason"].map(_format_sell_reason)
        inv["hold_age_seconds"] = inv["hold_age_seconds"].map(
            lambda x: "n/a" if x is None else f"{float(x) / 60.0:.1f}m"
        )
        st.dataframe(inv, use_container_width=True, hide_index=True)

    st.divider()

    # Recent trades
    st.subheader("Recent Trades")
    recent = sorted(rows_view, key=lambda r: r.get("timestamp") or datetime.min, reverse=True)[:50]
    if recent:
        df = pd.DataFrame([{
            "Time": r["timestamp"].strftime("%m-%d %H:%M") if isinstance(r.get("timestamp"), datetime) else "N/A",
            "Tag": r.get("strategy_tag", ""),
            "Title": str(r.get("title", ""))[:50],
            "Outcome": r.get("outcome", ""),
            "Side": r.get("side", ""),
            "Price": f"{r['fill_price']:.3f}",
            "Size": f"${r['size_usd']:.2f}",
            "Edge": f"{r['edge_theoretical']:.2%}",
            "P&L": f"${r['pnl']:.2f}" if r.get("pnl") else "",
        } for r in recent])
        st.dataframe(df, use_container_width=True, hide_index=True)


def _render_td_maker_tab(
    observations: list[LiveObservation],
    trades: list[PaperTrade],
) -> None:
    """Render the TD Maker (passive time-decay) tab content."""
    rows = extract_two_sided_trade_rows(
        observations, trades, event_types={TD_MAKER_EVENT_TYPE}
    )

    if not rows:
        st.info("No TD maker trades found. Run: ./run_crypto_td_maker.sh")
        return

    tags = available_two_sided_tags(rows)
    selected_tag = st.selectbox("Strategy Tag", ["All"] + tags, key="td_maker_tag_select")
    rows_view = rows if selected_tag == "All" else [r for r in rows if r["strategy_tag"] == selected_tag]

    # Separate entry (BUY) and settlement (SELL) rows.
    buys = [r for r in rows_view if r["side"] == "BUY"]
    sells = [r for r in rows_view if r["side"] == "SELL"]
    sell_wins = sum(1 for r in sells if (r.get("pnl") or 0) > 0)
    pnls = [r["pnl"] for r in rows_view if r.get("pnl")]
    total_pnl = sum(pnls)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fills (entries)", len(buys))
    c2.metric("Settled", len(sells))
    c3.metric("Win Rate", f"{sell_wins / len(sells):.1%}" if sells else "N/A")
    c4.metric("Realized P&L", f"${total_pnl:,.2f}")

    # Open positions (entries without matching settlement).
    settled_cids = {r.get("condition_id") for r in sells}
    open_entries = [r for r in buys if r.get("condition_id") not in settled_cids]
    if open_entries:
        open_exposure = sum(r["size_usd"] for r in open_entries)
        c5, c6 = st.columns(2)
        c5.metric("Open Positions", len(open_entries))
        c6.metric("Open Exposure", f"${open_exposure:,.2f}")

    st.divider()

    # Cumulative P&L
    st.subheader("Cumulative P&L")
    sorted_rows = sorted(rows_view, key=lambda r: r.get("timestamp") or datetime.min)
    if sorted_rows:
        mk_df = pd.DataFrame(sorted_rows).sort_values("timestamp")
        mk_df["cumulative_pnl"] = mk_df["pnl"].cumsum()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=mk_df["timestamp"], y=mk_df["cumulative_pnl"],
            mode="lines+markers", name="Cumulative P&L",
            line=dict(color="#2ecc71", width=2), marker=dict(size=5),
        ))
        fig.update_layout(
            xaxis_title="Time", yaxis_title="Cumulative P&L ($)",
            height=350, template="plotly_dark",
            margin=dict(l=50, r=50, t=30, b=50),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Win/loss breakdown by entry price bucket
    if sells:
        st.subheader("Performance by Entry Price")
        sell_data = []
        for r in sells:
            entry = r.get("fill_price", 0)
            bucket = f"{int(entry * 20) * 5}c"  # 5c buckets
            sell_data.append({"bucket": bucket, "won": (r.get("pnl") or 0) > 0, "pnl": r.get("pnl", 0)})
        sell_df = pd.DataFrame(sell_data)
        if not sell_df.empty:
            bucket_stats = sell_df.groupby("bucket").agg(
                n=("won", "count"),
                wins=("won", "sum"),
                total_pnl=("pnl", "sum"),
            ).reset_index()
            bucket_stats["win_rate"] = (bucket_stats["wins"] / bucket_stats["n"]).map(lambda x: f"{x:.1%}")
            bucket_stats["total_pnl"] = bucket_stats["total_pnl"].map(lambda x: f"${x:,.2f}")
            st.dataframe(bucket_stats, use_container_width=True, hide_index=True)

    st.divider()

    # Recent trades
    st.subheader("Recent Trades")
    recent = sorted(rows_view, key=lambda r: r.get("timestamp") or datetime.min, reverse=True)[:50]
    if recent:
        df = pd.DataFrame([{
            "Time": r["timestamp"].strftime("%m-%d %H:%M") if isinstance(r.get("timestamp"), datetime) else "N/A",
            "Tag": r.get("strategy_tag", ""),
            "Title": str(r.get("title", ""))[:50],
            "Outcome": r.get("outcome", ""),
            "Side": r.get("side", ""),
            "Price": f"{r['fill_price']:.3f}",
            "Size": f"${r['size_usd']:.2f}",
            "P&L": f"${r['pnl']:.2f}" if r.get("pnl") else "",
        } for r in recent])
        st.dataframe(df, use_container_width=True, hide_index=True)


def _render_fear_selling_tab(positions: list) -> None:
    """Render the Fear Selling tab content."""
    if not positions:
        st.info("No fear selling positions found.")
        return

    # Build dataframe from FearPosition objects
    rows: list[dict[str, Any]] = []
    for p in positions:
        rows.append({
            "condition_id": p.condition_id,
            "token_id": p.token_id,
            "title": p.title,
            "cluster": p.cluster,
            "side": p.side,
            "entry_price": _safe_float(p.entry_price),
            "size_usd": _safe_float(p.size_usd),
            "shares": _safe_float(p.shares),
            "fear_score": _safe_float(p.fear_score),
            "yes_price_at_entry": _safe_float(p.yes_price_at_entry),
            "exit_price": _safe_float(p.exit_price) if p.exit_price is not None else None,
            "realized_pnl": _safe_float(p.realized_pnl) if p.realized_pnl is not None else 0.0,
            "unrealized_pnl": _safe_float(p.unrealized_pnl) if p.unrealized_pnl is not None else 0.0,
            "is_open": bool(p.is_open),
            "entry_trigger": p.entry_trigger or "",
            "opened_at": p.opened_at,
            "closed_at": p.closed_at,
        })

    df = pd.DataFrame(rows)
    open_df = df[df["is_open"]].copy()
    closed_df = df[~df["is_open"]].copy()

    # Summary metrics
    total_realized = closed_df["realized_pnl"].sum() if not closed_df.empty else 0.0
    total_unrealized = open_df["unrealized_pnl"].sum() if not open_df.empty else 0.0
    total_exposure = open_df["size_usd"].sum() if not open_df.empty else 0.0
    avg_fear = df["fear_score"].mean() if not df.empty else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Positions", len(df))
    c2.metric("Open / Closed", f"{len(open_df)} / {len(closed_df)}")
    c3.metric("Realized P&L", f"${total_realized:,.2f}")
    c4.metric("Unrealized P&L", f"${total_unrealized:,.2f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Total P&L", f"${total_realized + total_unrealized:,.2f}")
    c6.metric("Open Exposure", f"${total_exposure:,.2f}")
    c7.metric("Avg Fear Score", f"{avg_fear:.2f}")
    win_count = len(closed_df[closed_df["realized_pnl"] > 0]) if not closed_df.empty else 0
    c8.metric("Win Rate (Closed)", f"{win_count / len(closed_df):.1%}" if not closed_df.empty else "N/A")

    st.divider()

    # P&L by cluster (bar chart)
    st.subheader("P&L by Cluster")
    cluster_pnl = df.groupby("cluster", as_index=False).agg(
        realized_pnl=("realized_pnl", "sum"),
        unrealized_pnl=("unrealized_pnl", "sum"),
        positions=("condition_id", "count"),
    )
    cluster_pnl["total_pnl"] = cluster_pnl["realized_pnl"] + cluster_pnl["unrealized_pnl"]
    cluster_pnl = cluster_pnl.sort_values("total_pnl", ascending=False)

    if not cluster_pnl.empty:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=cluster_pnl["cluster"],
            y=cluster_pnl["realized_pnl"],
            name="Realized P&L",
            marker_color="#2ecc71",
        ))
        fig.add_trace(go.Bar(
            x=cluster_pnl["cluster"],
            y=cluster_pnl["unrealized_pnl"],
            name="Unrealized P&L",
            marker_color="#3498db",
        ))
        fig.update_layout(
            barmode="group",
            xaxis_title="Cluster",
            yaxis_title="P&L ($)",
            height=400,
            template="plotly_dark",
            margin=dict(l=50, r=50, t=30, b=50),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Cluster exposure summary
    st.subheader("Cluster Exposure Summary")
    if not open_df.empty:
        cluster_exposure = open_df.groupby("cluster", as_index=False).agg(
            open_positions=("condition_id", "count"),
            total_exposure=("size_usd", "sum"),
            total_shares=("shares", "sum"),
            avg_entry_price=("entry_price", "mean"),
            avg_fear_score=("fear_score", "mean"),
            unrealized_pnl=("unrealized_pnl", "sum"),
        )
        cluster_exposure = cluster_exposure.sort_values("total_exposure", ascending=False)
        show_exposure = cluster_exposure.copy()
        show_exposure["avg_entry_price"] = show_exposure["avg_entry_price"].map(lambda x: f"{x:.3f}")
        show_exposure["avg_fear_score"] = show_exposure["avg_fear_score"].map(lambda x: f"{x:.2f}")
        show_exposure["total_exposure"] = show_exposure["total_exposure"].map(lambda x: f"${x:,.2f}")
        show_exposure["total_shares"] = show_exposure["total_shares"].map(lambda x: f"{x:.2f}")
        show_exposure["unrealized_pnl"] = show_exposure["unrealized_pnl"].map(lambda x: f"${x:,.2f}")
        st.dataframe(show_exposure, use_container_width=True, hide_index=True)
    else:
        st.info("No open positions.")

    st.divider()

    # Fear score distribution
    st.subheader("Fear Score Distribution")
    if not df.empty:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=df["fear_score"],
            nbinsx=20,
            name="Fear Score",
            marker_color="#e74c3c",
        ))
        fig_hist.update_layout(
            xaxis_title="Fear Score",
            yaxis_title="Count",
            height=350,
            template="plotly_dark",
            margin=dict(l=50, r=50, t=30, b=50),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    st.divider()

    # Active fear positions table
    st.subheader("Active Positions")
    if not open_df.empty:
        active = open_df.sort_values("opened_at", ascending=False).copy()
        show_active = pd.DataFrame([{
            "Opened": r["opened_at"].strftime("%m-%d %H:%M") if isinstance(r.get("opened_at"), datetime) else "N/A",
            "Title": str(r["title"])[:60],
            "Cluster": r["cluster"],
            "Side": r["side"],
            "Entry Price": f"{r['entry_price']:.3f}",
            "Size": f"${r['size_usd']:.2f}",
            "Shares": f"{r['shares']:.2f}",
            "Fear Score": f"{r['fear_score']:.2f}",
            "Yes Price": f"{r['yes_price_at_entry']:.3f}",
            "Unrealized": f"${r['unrealized_pnl']:.2f}",
            "Trigger": r["entry_trigger"],
        } for _, r in active.iterrows()])
        st.dataframe(show_active, use_container_width=True, hide_index=True)
    else:
        st.info("No active positions.")

    # Closed positions table
    st.subheader("Closed Positions")
    if not closed_df.empty:
        closed = closed_df.sort_values("closed_at", ascending=False).head(50).copy()
        show_closed = pd.DataFrame([{
            "Opened": r["opened_at"].strftime("%m-%d %H:%M") if isinstance(r.get("opened_at"), datetime) else "N/A",
            "Closed": r["closed_at"].strftime("%m-%d %H:%M") if isinstance(r.get("closed_at"), datetime) else "N/A",
            "Title": str(r["title"])[:60],
            "Cluster": r["cluster"],
            "Side": r["side"],
            "Entry": f"{r['entry_price']:.3f}",
            "Exit": f"{r['exit_price']:.3f}" if r["exit_price"] is not None else "N/A",
            "Size": f"${r['size_usd']:.2f}",
            "P&L": f"${r['realized_pnl']:.2f}",
            "Fear Score": f"{r['fear_score']:.2f}",
            "Trigger": r["entry_trigger"],
        } for _, r in closed.iterrows()])
        st.dataframe(show_closed, use_container_width=True, hide_index=True)
    else:
        st.info("No closed positions yet.")


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
    mode_filter = st.sidebar.radio("Execution Mode", ["All", "Paper", "Live"], horizontal=True)
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

    # Apply global execution-mode filter
    observations, trades = filter_by_execution_mode(observations, trades, mode_filter)

    # Load fear positions
    try:
        fear_positions = load_fear_positions(db_path)
    except Exception:
        fear_positions = []

    # --- Tabs ---
    tab_td_maker, tab_maker, tab_weather, tab_sniper, tab_crypto, tab_fear, tab_two_sided = st.tabs(
        ["TD Maker", "Crypto Maker", "Weather Oracle", "Sniper Sports", "Crypto Minute", "Fear Selling", "Two-Sided"]
    )

    # ===== TWO-SIDED TAB =====
    with tab_two_sided:
        # Exclude sniper rows from two-sided view
        two_sided_rows_all = extract_two_sided_trade_rows(observations, trades)
        two_sided_rows = [r for r in two_sided_rows_all if "sniper" not in str(r.get("strategy_tag", "")).lower()]
        two_sided_tags = available_two_sided_tags(two_sided_rows)
        selected_tag = st.selectbox(
            "Strategy Tag",
            ["All"] + two_sided_tags,
            help="Filter to one two-sided experiment tag.",
            key="ts_tag_select",
        )

        observations_view, trades_view = filter_scope_by_strategy_tag(
            observations=observations,
            trades=trades,
            strategy_tag=selected_tag,
        )
        # Also exclude sniper from the filtered views
        if selected_tag == "All":
            sniper_obs_ids: set[int] = set()
            for obs in observations_view:
                gs = _observation_game_state(obs)
                if "sniper" in str(gs.get("strategy_tag", "")).lower():
                    if obs.id is not None:
                        sniper_obs_ids.add(int(obs.id))
            observations_view = [o for o in observations_view if o.id not in sniper_obs_ids]
            trades_view = [t for t in trades_view if int(t.observation_id) not in sniper_obs_ids]

        two_sided_rows_view = [
            row for row in two_sided_rows
            if selected_tag == "All" or row.get("strategy_tag") == selected_tag
        ]
        two_sided_open_inventory_all_df = build_two_sided_open_inventory(two_sided_rows)
        two_sided_summary_df = summarize_two_sided_pairs(two_sided_rows_view)
        two_sided_open_inventory_df = build_two_sided_open_inventory(two_sided_rows_view)
        two_sided_tag_summary_df = summarize_two_sided_by_tag(
            two_sided_rows if selected_tag == "All" else two_sided_rows_view,
            two_sided_open_inventory_all_df if selected_tag == "All" else two_sided_open_inventory_df,
        )
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
            two_sided_summary_df["unrealized_pnl"] = two_sided_summary_df["unrealized_conservative"]
            two_sided_summary_df["open_outcomes"] = two_sided_summary_df["open_outcomes"].fillna(0).astype(int)
            two_sided_summary_df["max_mark_age_minutes"] = two_sided_summary_df["max_mark_age_minutes"].fillna(0.0)
            two_sided_summary_df["total_pnl_mark"] = (
                two_sided_summary_df["realized_pnl"] + two_sided_summary_df["unrealized_pnl_mark"]
            )
            two_sided_summary_df["total_pnl"] = (
                two_sided_summary_df["realized_pnl"] + two_sided_summary_df["unrealized_pnl"]
            )

        metrics = calculate_metrics(trades_view)

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
                        "strategy_tag", "condition_id", "title", "outcome",
                        "open_shares", "avg_entry_price",
                        "conservative_mark_price", "conservative_mark_reason",
                        "unrealized_conservative", "mark_price", "unrealized_pnl_mark",
                        "fair_price", "edge_sell_est", "exit_edge_pct",
                        "sell_signal", "sell_trigger", "sell_block_reason",
                        "open_notional", "open_notional_conservative",
                        "hold_age_seconds", "bid_age_minutes", "mark_age_minutes",
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

        st.subheader("Two-Sided Strategy Comparison")
        if two_sided_tag_summary_df.empty:
            st.info("No two-sided strategy data for current filter.")
        else:
            compare_cols = st.columns(4)
            compare_cols[0].metric("Tags", int(two_sided_tag_summary_df["strategy_tag"].nunique()))
            compare_cols[1].metric("Total Trades", int(two_sided_tag_summary_df["trades"].sum()))
            compare_cols[2].metric("Realized P&L", f"${two_sided_tag_summary_df['realized_pnl'].sum():,.2f}")
            compare_cols[3].metric("Total (Conservative)", f"${two_sided_tag_summary_df['total_pnl'].sum():,.2f}")

            by_tag_view = two_sided_tag_summary_df.copy()
            if "max_mark_age_minutes" in by_tag_view.columns:
                by_tag_view["max_mark_age_minutes"] = by_tag_view["max_mark_age_minutes"].fillna(0.0)
            by_tag_view = by_tag_view[
                [
                    "strategy_tag", "trades", "sells", "win_rate_sells",
                    "realized_pnl", "unrealized_conservative", "total_pnl",
                    "unrealized_pnl_mark", "total_pnl_mark",
                    "open_outcomes", "ready_to_sell", "blocked_missing_bid",
                    "max_mark_age_minutes",
                    "avg_edge_theoretical", "avg_edge_realized_sells", "gross_notional",
                ]
            ]
            by_tag_view["win_rate_sells"] = by_tag_view["win_rate_sells"].map(lambda x: f"{float(x):.1%}")
            by_tag_view["avg_edge_theoretical"] = by_tag_view["avg_edge_theoretical"].map(lambda x: f"{float(x):.2%}")
            by_tag_view["avg_edge_realized_sells"] = by_tag_view["avg_edge_realized_sells"].map(
                lambda x: f"{float(x):.2%}"
            )
            by_tag_view["max_mark_age_minutes"] = by_tag_view["max_mark_age_minutes"].map(lambda x: f"{float(x):.1f}")
            for col in [
                "realized_pnl", "unrealized_conservative", "total_pnl",
                "unrealized_pnl_mark", "total_pnl_mark", "gross_notional",
            ]:
                by_tag_view[col] = by_tag_view[col].map(lambda x: f"${float(x):,.2f}")
            st.dataframe(by_tag_view, use_container_width=True, hide_index=True)

        st.divider()

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

    # ===== SNIPER TAB =====
    with tab_sniper:
        _render_sniper_tab(observations, trades)

    # ===== CRYPTO MINUTE TAB =====
    with tab_crypto:
        _render_crypto_minute_tab(observations, trades)

    # ===== WEATHER ORACLE TAB =====
    with tab_weather:
        _render_weather_oracle_tab(observations, trades)

    # ===== TD MAKER TAB =====
    with tab_td_maker:
        _render_td_maker_tab(observations, trades)

    # ===== CRYPTO MAKER TAB =====
    with tab_maker:
        _render_crypto_maker_tab(observations, trades)

    # ===== FEAR SELLING TAB =====
    with tab_fear:
        _render_fear_selling_tab(fear_positions)

    # Footer
    st.divider()
    st.caption(
        "Paper Trading Dashboard | "
        f"Data from: {db_path} | "
        "Run with: streamlit run src/paper_trading/dashboard.py"
    )


if __name__ == "__main__":
    main()
