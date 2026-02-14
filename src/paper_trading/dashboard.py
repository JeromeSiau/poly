"""Real-time portfolio dashboard with Live/Paper toggle.

All data comes from the REST API at localhost:8788.
No direct DB access — pure API consumer.
"""

from datetime import datetime, timezone
import time

import httpx
import pandas as pd
import streamlit as st

API_BASE = "http://localhost:8788"

LOOKBACK_MAP = {"1h": 1, "4h": 4, "12h": 12, "24h": 24, "48h": 48, "7d": 168}

# 15-min markets older than this (seconds) are considered stale / resolved
_STALE_THRESHOLD_S = 30 * 60  # 30 minutes


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


def _is_stale_position(trade: dict) -> bool:
    """Return True if this open position is stale (15-min market already resolved)."""
    match_id = trade.get("match_id", "")
    # CryptoTD match_ids contain a unix timestamp: e.g. "btc-updown-15m-1771082100"
    parts = str(match_id).split("-")
    for part in reversed(parts):
        if part.isdigit() and len(part) >= 10:
            market_ts = int(part)
            # Market resolves at ts + 15min (900s)
            if time.time() > market_ts + _STALE_THRESHOLD_S:
                return True
            break
    # Fallback: if the trade timestamp is older than 30 min, consider stale
    ts_str = trade.get("timestamp", "")
    try:
        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age_s = (datetime.now(timezone.utc) - ts).total_seconds()
        if age_s > _STALE_THRESHOLD_S:
            return True
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# Page config & custom styling
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Poly Dashboard", layout="wide")

st.markdown("""
<style>
/* Dark modern feel */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #0f3460;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
}
[data-testid="stMetricLabel"] {
    color: #8892b0 !important;
    font-size: 0.85rem !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
[data-testid="stMetricValue"] {
    font-size: 1.6rem !important;
    font-weight: 700 !important;
}

/* Section headers */
h2 {
    color: #ccd6f6 !important;
    border-bottom: 2px solid #0f3460;
    padding-bottom: 8px;
    margin-top: 1rem !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0a0a1a;
}

/* Dataframe styling */
[data-testid="stDataFrame"] {
    border-radius: 8px;
    overflow: hidden;
}

/* Divider */
hr {
    border-color: #1a1a3e !important;
    margin: 1rem 0 !important;
}

/* Chart container */
[data-testid="stVegaLiteChart"] {
    background: #0d1117;
    border-radius: 12px;
    padding: 10px;
    border: 1px solid #1a1a3e;
}

/* Tabs styling */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 8px;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 8px 20px;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("## :chart_with_upward_trend: Portfolio Dashboard")

mode = st.sidebar.radio("Mode", ["Live", "Paper"], index=0, key="mode")

lookback_label = st.sidebar.selectbox(
    "Lookback", list(LOOKBACK_MAP.keys()), index=3, key="lookback"
)
hours = LOOKBACK_MAP[lookback_label]

tags_data = _api("/tags", {"hours": hours})
available_tags = sorted(tags_data.get("strategy_tags", {}).keys())
st.sidebar.multiselect("Strategies", available_tags, key="strategies")


# ---------------------------------------------------------------------------
# Section 1: KPIs
# ---------------------------------------------------------------------------

@st.fragment(run_every="15s")
def kpi_section():
    m = st.session_state.get("mode", "Live").lower()
    h = LOOKBACK_MAP[st.session_state.get("lookback", "24h")]

    balance_data = _api("/balance", {"mode": m})
    winrate_data = _api("/winrate", {"mode": m, "hours": h})

    bal = balance_data.get("balance", 0.0)
    pnl = winrate_data.get("total_pnl", 0.0)
    wr = winrate_data.get("winrate", 0.0)
    pf = winrate_data.get("profit_factor")
    wins = winrate_data.get("wins", 0)
    losses = winrate_data.get("losses", 0)
    roi = winrate_data.get("roi_pct", 0.0)

    # Count truly active open positions (not stale)
    open_data = _api("/trades", {"mode": m, "is_open": "true", "hours": 2})
    active = [t for t in open_data.get("trades", []) if not _is_stale_position(t)]

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Balance", f"${bal:,.2f}")
    pnl_delta = f"{pnl:+.2f}" if pnl != 0 else None
    c2.metric("PnL", f"${pnl:,.2f}", delta=pnl_delta)
    c3.metric("Win Rate", f"{wr:.1f}%", delta=f"{wins}W / {losses}L")
    c4.metric("Profit Factor", f"{pf:.2f}" if pf else "N/A")
    c5.metric("ROI", f"{roi:+.1f}%")
    c6.metric("Active", len(active))


kpi_section()

st.divider()


# ---------------------------------------------------------------------------
# Section 2: PnL Chart + Open Positions (tabs)
# ---------------------------------------------------------------------------

@st.fragment(run_every="15s")
def charts_section():
    m = st.session_state.get("mode", "Live").lower()
    h = LOOKBACK_MAP[st.session_state.get("lookback", "24h")]
    strats = st.session_state.get("strategies", [])

    tab_chart, tab_open = st.tabs(["PnL Chart", "Open Positions"])

    # -- PnL Chart tab --
    with tab_chart:
        winrate_data = _api("/winrate", {"mode": m, "hours": h})
        markets = winrate_data.get("markets", [])

        if not markets:
            st.info("No resolved trades in this period")
        else:
            # Build cumulative PnL dataframe sorted by time
            rows = []
            for mk in markets:
                ts_val = mk.get("timestamp")
                if ts_val is None:
                    continue
                if isinstance(ts_val, (int, float)):
                    dt = datetime.fromtimestamp(ts_val, tz=timezone.utc)
                else:
                    dt = datetime.fromisoformat(
                        str(ts_val).replace("Z", "+00:00")
                    )
                rows.append({
                    "time": dt,
                    "pnl": mk["pnl"],
                    "title": mk.get("title", ""),
                    "status": mk.get("status", ""),
                    "strategy": mk.get("title", "").split(" - ")[0][:20] if mk.get("title") else "",
                })

            if rows:
                df = pd.DataFrame(rows).sort_values("time")

                # Strategy filter
                if strats:
                    # Fuzzy match on title since winrate doesn't return strategy_tag
                    pass  # winrate endpoint doesn't have strategy_tag, skip filter

                df["cumulative_pnl"] = df["pnl"].cumsum()

                # Cumulative PnL line chart
                chart_df = df[["time", "cumulative_pnl"]].copy()
                chart_df.columns = ["Time", "Cumulative PnL ($)"]

                st.line_chart(
                    chart_df,
                    x="Time",
                    y="Cumulative PnL ($)",
                    color="#00d4aa",
                )

                # Per-trade PnL bar chart
                bar_df = df[["time", "pnl"]].copy()
                bar_df.columns = ["Time", "PnL ($)"]
                st.bar_chart(
                    bar_df,
                    x="Time",
                    y="PnL ($)",
                    color="#5e60ce",
                )
            else:
                st.info("No timestamped trade data available")

    # -- Open Positions tab --
    with tab_open:
        open_data = _api("/trades", {"mode": m, "is_open": "true", "hours": 2})
        trades = [t for t in open_data.get("trades", []) if not _is_stale_position(t)]

        if not trades:
            st.info("No active positions")
        else:
            df = pd.DataFrame(trades)

            for tag, group in df.groupby("strategy_tag", sort=True):
                st.markdown(f"**{tag}**")
                display = pd.DataFrame({
                    "Market": group["title"],
                    "Side": group["outcome"],
                    "Entry": group["entry_price"].apply(
                        lambda x: f"{x:.2f}" if x is not None else ""
                    ),
                    "Size": group["size"].apply(
                        lambda x: f"${x:.2f}" if x is not None else ""
                    ),
                    "Age": group["timestamp"].apply(_humanize_age),
                })
                st.dataframe(display, use_container_width=True, hide_index=True)


charts_section()

st.divider()


# ---------------------------------------------------------------------------
# Section 3: Recent Trades
# ---------------------------------------------------------------------------

@st.fragment(run_every="15s")
def recent_trades_section():
    m = st.session_state.get("mode", "Live").lower()
    h = LOOKBACK_MAP[st.session_state.get("lookback", "24h")]
    strats = st.session_state.get("strategies", [])

    data = _api("/trades", {"mode": m, "hours": h, "is_open": "false", "limit": 200})
    trades = data.get("trades", [])

    st.markdown("## Recent Trades")

    if not trades:
        st.info("No trades in this period")
        return

    df = pd.DataFrame(trades)

    if strats:
        df = df[df["strategy_tag"].isin(strats)]

    if df.empty:
        st.info("No trades matching selected strategies")
        return

    def _fmt_time(ts_str):
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            return ts.strftime("%H:%M")
        except Exception:
            return ""

    def _fmt_price(val):
        return f"{val:.2f}" if val is not None else ""

    def _fmt_pnl(val):
        return f"${val:+.2f}" if val is not None else ""

    def _status(row):
        if row.get("pnl") is not None and row["pnl"] > 0:
            return "WIN"
        if row.get("pnl") is not None and row["pnl"] <= 0:
            return "LOSS"
        return "PENDING"

    display = pd.DataFrame({
        "Time": df["timestamp"].apply(_fmt_time),
        "Strategy": df["strategy_tag"],
        "Market": df["title"],
        "Side": df["outcome"],
        "Entry": df["entry_price"].apply(_fmt_price),
        "Exit": df["exit_price"].apply(_fmt_price),
        "PnL": df["pnl"].apply(_fmt_pnl),
        "Result": df.apply(_status, axis=1),
    })

    def _color_pnl(val):
        if not val or val == "":
            return ""
        try:
            num = float(val.replace("$", "").replace("+", ""))
            if num > 0:
                return "color: #00d4aa; font-weight: 600"
            if num < 0:
                return "color: #ff6b6b; font-weight: 600"
        except ValueError:
            pass
        return ""

    def _color_result(val):
        if val == "WIN":
            return "color: #00d4aa; font-weight: 600"
        if val == "LOSS":
            return "color: #ff6b6b; font-weight: 600"
        return "color: #8892b0"

    styled = display.style.map(_color_pnl, subset=["PnL"]).map(
        _color_result, subset=["Result"]
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Summary row
    pnls = [t["pnl"] for t in trades if t.get("pnl") is not None]
    if pnls:
        w = sum(1 for p in pnls if p > 0)
        l = sum(1 for p in pnls if p <= 0)
        total = sum(pnls)
        st.caption(
            f"Showing {len(display)} trades | "
            f"**{w}W {l}L** | "
            f"Total PnL: **${total:+.2f}**"
        )


recent_trades_section()
