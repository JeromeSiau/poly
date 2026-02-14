"""Real-time portfolio dashboard with Live/Paper toggle.

All data comes from the REST API at localhost:8788.
No direct DB access — pure API consumer.
"""

from datetime import datetime, timezone

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

API_BASE = "http://localhost:8788"

LOOKBACK_MAP = {"1h": 1, "4h": 4, "12h": 12, "24h": 24, "48h": 48, "7d": 168}

# -- Colors --
C_BG = "#0b0e17"
C_CARD = "#111827"
C_CARD_BORDER = "#1e2a3a"
C_GREEN = "#34d399"
C_RED = "#f87171"
C_MUTED = "#64748b"
C_TEXT = "#e2e8f0"
C_ACCENT = "#38bdf8"
C_GRID = "rgba(148,163,184,0.06)"


def _api(endpoint: str, params: dict | None = None) -> dict:
    try:
        resp = httpx.get(API_BASE + endpoint, params=params or {}, timeout=10)
        return resp.json()
    except Exception as exc:
        st.warning(f"API error on {endpoint}: {exc}")
        return {}


def _humanize_age(ts_str: str) -> str:
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


def _style_pnl(val):
    """Color PnL values green/red."""
    if not val:
        return ""
    try:
        n = float(val.replace("$", "").replace("+", ""))
        if n > 0:
            return f"color: {C_GREEN}; font-weight: 600"
        if n < 0:
            return f"color: {C_RED}; font-weight: 600"
    except ValueError:
        pass
    return ""


def _style_result(val):
    """Color WIN/LOSS labels."""
    if val == "WIN":
        return f"color: {C_GREEN}"
    if val == "LOSS":
        return f"color: {C_RED}"
    return f"color: {C_MUTED}"


def _plotly_layout(**overrides) -> dict:
    """Base Plotly layout — clean dark terminal aesthetic."""
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono, SF Mono, Menlo, monospace", color=C_TEXT, size=11),
        margin=dict(l=0, r=0, t=28, b=0),
        xaxis=dict(
            gridcolor=C_GRID, zerolinecolor=C_GRID,
            tickfont=dict(color=C_MUTED, size=10),
        ),
        yaxis=dict(
            gridcolor=C_GRID, zerolinecolor=C_GRID,
            tickfont=dict(color=C_MUTED, size=10),
            tickprefix="$", side="right",
        ),
        hoverlabel=dict(
            bgcolor="#1e293b", bordercolor="#334155",
            font=dict(color=C_TEXT, family="JetBrains Mono, monospace", size=12),
        ),
        dragmode=False,
        hovermode="x unified",
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Poly", page_icon="P", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=DM+Sans:wght@400;500;600;700&display=swap');

/* Root */
.stApp {
    background: #0b0e17;
    font-family: 'DM Sans', sans-serif;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header {visibility: hidden;}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #070a12;
    border-right: 1px solid #1e2a3a;
}
section[data-testid="stSidebar"] .stRadio > label,
section[data-testid="stSidebar"] .stSelectbox > label,
section[data-testid="stSidebar"] .stMultiSelect > label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #64748b;
}

/* KPI cards — equal height */
[data-testid="stMetric"] {
    background: #111827;
    border: 1px solid #1e2a3a;
    border-radius: 8px;
    padding: 14px 18px 12px;
    height: 110px;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    justify-content: center;
    overflow: hidden;
}
/* Force equal-width columns */
[data-testid="stHorizontalBlock"] > [data-testid="column"] {
    flex: 1 1 0% !important;
    width: 0 !important;
    min-width: 0 !important;
}
[data-testid="stHorizontalBlock"] {
    gap: 12px !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.65rem !important;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #64748b !important;
}
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.5rem !important;
    font-weight: 600 !important;
    color: #e2e8f0 !important;
}
[data-testid="stMetricDelta"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem !important;
}

/* Divider */
hr {
    border-color: #1e2a3a !important;
    margin: 0.5rem 0 !important;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    border: 1px solid #1e2a3a;
    border-radius: 6px;
    overflow: hidden;
}
[data-testid="stDataFrame"] * {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
}

/* Plotly chart containers */
[data-testid="stPlotlyChart"] {
    border: 1px solid #1e2a3a;
    border-radius: 8px;
    overflow: hidden;
    background: #0f1219;
}

/* Section labels */
.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #475569;
    margin: 0;
    padding: 8px 0 4px;
}

/* Header bar */
.dash-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 0 0 12px;
    border-bottom: 1px solid #1e2a3a;
    margin-bottom: 16px;
}
.dash-header .logo {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 1.1rem;
    color: #38bdf8;
    background: rgba(56, 189, 248, 0.1);
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-radius: 6px;
    padding: 4px 10px;
    letter-spacing: 2px;
}
.dash-header .title {
    font-family: 'DM Sans', sans-serif;
    font-weight: 400;
    font-size: 0.9rem;
    color: #94a3b8;
}
.dash-header .live-dot {
    width: 6px; height: 6px;
    background: #34d399;
    border-radius: 50%;
    animation: pulse-dot 2s ease-in-out infinite;
    margin-left: auto;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(52,211,153,0.4); }
    50% { opacity: 0.7; box-shadow: 0 0 0 6px rgba(52,211,153,0); }
}

/* Trade row highlight on hover */
[data-testid="stDataFrame"] tbody tr:hover {
    background: rgba(56, 189, 248, 0.04) !important;
}

/* Captions */
.stCaption {
    font-family: 'JetBrains Mono', monospace !important;
    color: #475569 !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

mode = st.sidebar.radio("Mode", ["Live", "Paper"], index=0, key="mode")
lookback_label = st.sidebar.selectbox(
    "Lookback", list(LOOKBACK_MAP.keys()), index=3, key="lookback"
)
hours = LOOKBACK_MAP[lookback_label]
tags_data = _api("/tags", {"hours": hours})
available_tags = sorted(tags_data.get("strategy_tags", {}).keys())
st.sidebar.multiselect("Strategies", available_tags, key="strategies")

mode_lower = st.session_state.get("mode", "Live").lower()
live_dot = '<div class="live-dot"></div>' if mode_lower == "live" else ""
st.markdown(
    f'<div class="dash-header">'
    f'<span class="logo">POLY</span>'
    f'<span class="title">Portfolio Dashboard</span>'
    f'{live_dot}'
    f'</div>',
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# KPIs
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

    pos_data = _api("/positions", {"mode": m})
    n_positions = pos_data.get("count", 0)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Balance", f"${bal:,.2f}")
    c2.metric("PnL", f"${pnl:+,.2f}", delta=f"{pnl:+.2f}" if pnl else None)
    c3.metric("Win Rate", f"{wr:.1f}%", delta=f"{wins}W {losses}L")
    c4.metric("Profit Factor", f"{pf:.2f}" if pf else "--")
    c5.metric("ROI", f"{roi:+.1f}%")
    c6.metric("Positions", str(n_positions))


kpi_section()

st.markdown('<div style="height: 8px"></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cumulative PnL
# ---------------------------------------------------------------------------

@st.fragment(run_every="15s")
def pnl_chart_section():
    m = st.session_state.get("mode", "Live").lower()
    h = LOOKBACK_MAP[st.session_state.get("lookback", "24h")]

    st.markdown('<p class="section-label">Cumulative PnL</p>', unsafe_allow_html=True)

    winrate_data = _api("/winrate", {"mode": m, "hours": h})
    markets = winrate_data.get("markets", [])

    if not markets:
        st.caption("No resolved trades yet")
        return

    rows = []
    for mk in markets:
        ts_val = mk.get("timestamp")
        if ts_val is None:
            continue
        if isinstance(ts_val, (int, float)):
            dt = datetime.fromtimestamp(ts_val, tz=timezone.utc)
        else:
            dt = datetime.fromisoformat(str(ts_val).replace("Z", "+00:00"))
        rows.append({"time": dt, "pnl": mk["pnl"], "status": mk.get("status", "")})

    if not rows:
        return

    df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
    df["cum_pnl"] = df["pnl"].cumsum()

    final_pnl = df["cum_pnl"].iloc[-1]
    color = C_GREEN if final_pnl >= 0 else C_RED
    fill = "rgba(52,211,153,0.08)" if final_pnl >= 0 else "rgba(248,113,113,0.08)"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["cum_pnl"],
        mode="lines",
        line=dict(color=color, width=1.5),
        fill="tozeroy", fillcolor=fill,
        hovertemplate="%{x|%H:%M}<br>$%{y:+.2f}<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color="#334155", width=0.5, dash="dot"))
    fig.update_layout(**_plotly_layout(height=300))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


pnl_chart_section()


# ---------------------------------------------------------------------------
# Hourly PnL
# ---------------------------------------------------------------------------

@st.fragment(run_every="15s")
def hourly_pnl_section():
    m = st.session_state.get("mode", "Live").lower()
    h = LOOKBACK_MAP[st.session_state.get("lookback", "24h")]

    st.markdown('<p class="section-label">PnL by Hour</p>', unsafe_allow_html=True)

    winrate_data = _api("/winrate", {"mode": m, "hours": h})
    markets = winrate_data.get("markets", [])

    if not markets:
        st.caption("No data")
        return

    rows = []
    for mk in markets:
        ts_val = mk.get("timestamp")
        if ts_val is None:
            continue
        if isinstance(ts_val, (int, float)):
            dt = datetime.fromtimestamp(ts_val, tz=timezone.utc)
        else:
            dt = datetime.fromisoformat(str(ts_val).replace("Z", "+00:00"))
        rows.append({"time": dt, "pnl": mk["pnl"]})

    if not rows:
        return

    df = pd.DataFrame(rows)
    df["hour"] = df["time"].dt.floor("h")
    hourly = df.groupby("hour").agg(pnl=("pnl", "sum"), trades=("pnl", "count")).reset_index()

    colors = [C_GREEN if p >= 0 else C_RED for p in hourly["pnl"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=hourly["hour"], y=hourly["pnl"],
        marker_color=colors, marker_line_width=0,
        opacity=0.85,
        hovertemplate="%{x|%H:%M}<br>$%{y:+.2f}<br>%{customdata} trades<extra></extra>",
        customdata=hourly["trades"],
    ))
    fig.add_hline(y=0, line=dict(color="#334155", width=0.5, dash="dot"))
    fig.update_layout(**_plotly_layout(height=220))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


hourly_pnl_section()


# ---------------------------------------------------------------------------
# Open Positions
# ---------------------------------------------------------------------------

@st.fragment(run_every="15s")
def open_positions_section():
    m = st.session_state.get("mode", "Live").lower()

    st.markdown('<p class="section-label">Open Positions</p>', unsafe_allow_html=True)

    pos_data = _api("/positions", {"mode": m})
    items = pos_data.get("positions", [])

    if not items:
        st.caption("No open positions")
        return

    df = pd.DataFrame(items)

    display = pd.DataFrame({
        "Market": df["title"],
        "Side": df["outcome"],
        "Shares": df["size"].apply(lambda x: f"{x:.1f}"),
        "Entry": df["avg_price"].apply(lambda x: f"{x:.2f}" if x else ""),
        "Price": df["cur_price"].apply(lambda x: f"{x:.2f}" if x else ""),
        "Value": df["value"].apply(lambda x: f"${x:.2f}" if x else ""),
        "PnL": df["pnl"].apply(lambda x: f"${x:+.2f}" if x is not None else ""),
    })

    styled = display.style.map(_style_pnl, subset=["PnL"])
    st.dataframe(styled, use_container_width=True, hide_index=True)


open_positions_section()


# ---------------------------------------------------------------------------
# Recent Trades
# ---------------------------------------------------------------------------

@st.fragment(run_every="15s")
def recent_trades_section():
    m = st.session_state.get("mode", "Live").lower()
    h = LOOKBACK_MAP[st.session_state.get("lookback", "24h")]

    st.markdown('<p class="section-label">Recent Trades</p>', unsafe_allow_html=True)

    if m == "live":
        # Live: use on-chain data from /winrate
        winrate_data = _api("/winrate", {"mode": "live", "hours": h})
        markets = winrate_data.get("markets", [])
        if not markets:
            st.caption("No trades in this period")
            return
        rows = []
        for mk in markets:
            ts_val = mk.get("timestamp")
            if isinstance(ts_val, (int, float)):
                dt = datetime.fromtimestamp(ts_val, tz=timezone.utc)
            else:
                continue
            rows.append({
                "Time": dt.strftime("%H:%M"),
                "Market": mk.get("title", ""),
                "Side": mk.get("outcome", ""),
                "Entry": f"{mk['avg_entry']:.2f}" if mk.get("avg_entry") else "",
                "Cost": f"${mk['cost']:.2f}" if mk.get("cost") else "",
                "PnL": f"${mk['pnl']:+.2f}",
                "Result": mk.get("status", ""),
            })
        display = pd.DataFrame(rows)
        styled = display.style.map(_style_pnl, subset=["PnL"]).map(
            _style_result, subset=["Result"]
        )
        st.dataframe(styled, use_container_width=True, hide_index=True, height=400)
        pnls = [mk["pnl"] for mk in markets]
        w = sum(1 for p in pnls if p > 0)
        l = sum(1 for p in pnls if p <= 0)
        st.caption(f"{len(markets)} trades  |  {w}W {l}L  |  ${sum(pnls):+.2f}")
    else:
        # Paper: use internal DB
        data = _api("/trades", {"mode": "paper", "hours": h, "is_open": "false", "limit": 200})
        trades = data.get("trades", [])
        if not trades:
            st.caption("No trades in this period")
            return
        df = pd.DataFrame(trades)
        if df.empty:
            return

        def _fmt_time(ts_str):
            try:
                return datetime.fromisoformat(ts_str.replace("Z", "+00:00")).strftime("%H:%M")
            except Exception:
                return ""

        display = pd.DataFrame({
            "Time": df["timestamp"].apply(_fmt_time),
            "Strategy": df["strategy_tag"],
            "Market": df["title"],
            "Side": df["outcome"],
            "Entry": df["entry_price"].apply(lambda x: f"{x:.2f}" if x is not None else ""),
            "Exit": df["exit_price"].apply(lambda x: f"{x:.2f}" if x is not None else ""),
            "PnL": df["pnl"].apply(lambda x: f"${x:+.2f}" if x is not None else ""),
            "Result": df.apply(
                lambda r: "WIN" if r.get("pnl") and r["pnl"] > 0
                else ("LOSS" if r.get("pnl") is not None else "--"), axis=1
            ),
        })
        styled = display.style.map(_style_pnl, subset=["PnL"]).map(
            _style_result, subset=["Result"]
        )
        st.dataframe(styled, use_container_width=True, hide_index=True, height=400)
        pnls = [t["pnl"] for t in trades if t.get("pnl") is not None]
        if pnls:
            w = sum(1 for p in pnls if p > 0)
            l = sum(1 for p in pnls if p <= 0)
            st.caption(f"{len(display)} trades  |  {w}W {l}L  |  ${sum(pnls):+.2f}")


recent_trades_section()
