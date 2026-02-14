"""Real-time portfolio dashboard with Live/Paper toggle.

All data comes from the REST API at localhost:8788.
No direct DB access — pure API consumer.
"""

from datetime import datetime, timezone

import httpx
import pandas as pd
import streamlit as st

API_BASE = "http://localhost:8788"

LOOKBACK_MAP = {"1h": 1, "4h": 4, "12h": 12, "24h": 24, "48h": 48, "7d": 168}


def _api(endpoint: str, params: dict | None = None) -> dict:
    """Call the trades API and return JSON. Returns {} on any error."""
    try:
        resp = httpx.get(
            API_BASE + endpoint, params=params or {}, timeout=10
        )
        return resp.json()
    except Exception as exc:
        st.warning(f"API error on {endpoint}: {exc}")
        return {}


def _humanize_age(ts_str: str) -> str:
    """Convert an ISO timestamp string to a human-readable age like '2h 15m'."""
    try:
        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - ts
        total_minutes = int(delta.total_seconds() / 60)
        if total_minutes < 1:
            return "<1m"
        hours, minutes = divmod(total_minutes, 60)
        days, hours = divmod(hours, 24)
        if days > 0:
            return f"{days}d {hours}h"
        if hours > 0:
            return f"{hours}h {minutes}m"
        return f"{minutes}m"
    except Exception:
        return "?"


# ---------------------------------------------------------------------------
# Page config & sidebar
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Poly Dashboard", layout="wide")
st.title("Portfolio Dashboard")

mode = st.sidebar.radio("Mode", ["Live", "Paper"], index=1, key="mode")
mode_param = mode.lower()

lookback_label = st.sidebar.selectbox(
    "Lookback", list(LOOKBACK_MAP.keys()), index=3, key="lookback"
)
hours = LOOKBACK_MAP[lookback_label]

tags_data = _api("/tags", {"hours": hours})
available_tags = sorted(tags_data.get("strategy_tags", {}).keys())
selected_tags = st.sidebar.multiselect(
    "Strategies", available_tags, key="strategies"
)


# ---------------------------------------------------------------------------
# Section 1: KPIs
# ---------------------------------------------------------------------------

@st.fragment(run_every="15s")
def kpi_section():
    m = st.session_state.get("mode", "Paper").lower()
    h = LOOKBACK_MAP[st.session_state.get("lookback", "24h")]

    balance_data = _api("/balance", {"mode": m})
    winrate_data = _api("/winrate", {"mode": m, "hours": h})

    balance = balance_data.get("balance", 0.0)
    pnl = winrate_data.get("total_pnl", 0.0)
    winrate = winrate_data.get("winrate", 0.0)
    profit_factor = winrate_data.get("profit_factor")
    wins = winrate_data.get("wins", 0)
    losses = winrate_data.get("losses", 0)
    total_trades = wins + losses

    trades_data = _api("/trades", {"mode": m, "is_open": "true", "hours": 720})
    open_count = trades_data.get("still_open", 0)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Balance", f"${balance:,.2f}")
    c2.metric("PnL (period)", f"${pnl:,.2f}", delta=f"${pnl:,.2f}")
    c3.metric("Win Rate", f"{winrate:.1f}%")
    c4.metric("Profit Factor", f"{profit_factor:.2f}" if profit_factor else "N/A")
    c5.metric("Trades", total_trades)
    c6.metric("Open", open_count)


kpi_section()

st.divider()


# ---------------------------------------------------------------------------
# Section 2: Open Positions
# ---------------------------------------------------------------------------

@st.fragment(run_every="15s")
def open_positions_section():
    m = st.session_state.get("mode", "Paper").lower()

    data = _api("/trades", {"mode": m, "is_open": "true", "hours": 720})
    trades = data.get("trades", [])

    st.header("Open Positions")

    if not trades:
        st.info("No open positions")
        return

    df = pd.DataFrame(trades)

    grouped = df.groupby("strategy_tag", sort=True)
    for tag, group in grouped:
        st.subheader(str(tag))
        display = pd.DataFrame({
            "Title": group["title"],
            "Outcome": group["outcome"],
            "Entry": group["entry_price"].apply(
                lambda x: f"{x:.2f}" if x is not None else ""
            ),
            "Size ($)": group["size"].apply(
                lambda x: f"${x:.2f}" if x is not None else ""
            ),
            "Age": group["timestamp"].apply(_humanize_age),
        })
        st.dataframe(display, use_container_width=True, hide_index=True)


open_positions_section()

st.divider()


# ---------------------------------------------------------------------------
# Section 3: Recent Trades
# ---------------------------------------------------------------------------

@st.fragment(run_every="15s")
def recent_trades_section():
    m = st.session_state.get("mode", "Paper").lower()
    h = LOOKBACK_MAP[st.session_state.get("lookback", "24h")]
    strats = st.session_state.get("strategies", [])

    data = _api("/trades", {"mode": m, "hours": h, "limit": 100})
    trades = data.get("trades", [])

    st.header("Recent Trades")

    if not trades:
        st.info("No trades in this period")
        return

    df = pd.DataFrame(trades)

    # Apply strategy filter locally if set
    if strats:
        df = df[df["strategy_tag"].isin(strats)]

    if df.empty:
        st.info("No trades matching selected strategies")
        return

    def _status(row):
        if row.get("is_open"):
            return "OPEN"
        if row.get("pnl") is not None and row["pnl"] > 0:
            return "WIN"
        return "LOSS"

    def _format_time(ts_str):
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            return ts.strftime("%H:%M")
        except Exception:
            return ""

    def _format_price(val):
        if val is None:
            return ""
        return f"{val:.2f}"

    def _format_pnl(val):
        if val is None:
            return ""
        return f"${val:+.2f}"

    display = pd.DataFrame({
        "Time": df["timestamp"].apply(_format_time),
        "Strategy": df["strategy_tag"],
        "Market": df["title"],
        "Outcome": df["outcome"],
        "Entry": df["entry_price"].apply(_format_price),
        "Exit": df["exit_price"].apply(_format_price),
        "PnL ($)": df["pnl"].apply(_format_pnl),
        "Status": df.apply(_status, axis=1),
    })

    def _color_pnl(val):
        if not val or val == "":
            return ""
        try:
            num = float(val.replace("$", "").replace("+", ""))
            if num > 0:
                return "color: green"
            if num < 0:
                return "color: red"
        except ValueError:
            pass
        return ""

    styled = display.style.map(_color_pnl, subset=["PnL ($)"])
    st.dataframe(styled, use_container_width=True, hide_index=True)


recent_trades_section()
