"""Real-time portfolio dashboard with Live/Paper toggle.

All data comes from the REST API at localhost:8788.
No direct DB access — pure API consumer.
"""

import re
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
    """Color WIN/LOSS/OPEN labels."""
    if val == "WIN":
        return f"color: {C_GREEN}"
    if val == "LOSS":
        return f"color: {C_RED}"
    if val == "OPEN":
        return f"color: {C_ACCENT}"
    return f"color: {C_MUTED}"


def _plotly_layout(**overrides) -> dict:
    """Base Plotly layout — clean dark terminal aesthetic."""
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono, SF Mono, Menlo, monospace", color=C_TEXT, size=11),
        margin=dict(l=0, r=0, t=36, b=0),
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


def _parse_slot_ts(title: str) -> int | None:
    """Extract slot unix timestamp from slug like 'btc-updown-15m-1771079400'."""
    if not title:
        return None
    m = re.search(r"-(\d{10})$", title)
    return int(m.group(1)) if m else None


def _parse_asset(title: str) -> str | None:
    """Extract asset from slug like 'btc-updown-15m-1771079400' -> 'BTC'."""
    if not title:
        return None
    parts = title.split("-")
    if len(parts) >= 2:
        return parts[0].upper()
    return None


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

/* Tabs — single bottom border only */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 1px solid #1e2a3a !important;
}
.stTabs [data-baseweb="tab-highlight"] {
    display: none !important;
}
.stTabs [data-baseweb="tab-border"] {
    display: none !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #64748b;
    padding: 8px 24px;
    border: none !important;
    background: transparent;
}
.stTabs [aria-selected="true"] {
    color: #38bdf8 !important;
    border-bottom: 2px solid #38bdf8 !important;
    background: transparent !important;
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
    margin-bottom: 4px;
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
# Header & Sidebar
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


# ═══════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════

tab_live, tab_analysis, tab_slot = st.tabs(["Live", "Strategy Analysis", "Slot ML"])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: LIVE
# ═══════════════════════════════════════════════════════════════════════════

with tab_live:

    # -- KPIs --
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

    # -- Cumulative PnL --
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

    # -- Hourly PnL --
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

    # -- Open Positions --
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

    # -- Recent Trades --
    @st.fragment(run_every="15s")
    def recent_trades_section():
        m = st.session_state.get("mode", "Live").lower()
        h = LOOKBACK_MAP[st.session_state.get("lookback", "24h")]

        st.markdown('<p class="section-label">Recent Trades</p>', unsafe_allow_html=True)

        if m == "live":
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
                    "_sort": dt,
                    "Time": dt.strftime("%H:%M"),
                    "Market": mk.get("title", ""),
                    "Side": mk.get("outcome", ""),
                    "Entry": f"{mk['avg_entry']:.2f}" if mk.get("avg_entry") else "",
                    "Cost": f"${mk['cost']:.2f}" if mk.get("cost") else "",
                    "PnL": f"${mk['pnl']:+.2f}",
                    "Result": mk.get("status", ""),
                })
            display = pd.DataFrame(rows).sort_values("_sort", ascending=False).drop(columns=["_sort"])
            styled = display.style.map(_style_pnl, subset=["PnL"]).map(
                _style_result, subset=["Result"]
            )
            st.dataframe(styled, use_container_width=True, hide_index=True, height=400)
            pnls = [mk["pnl"] for mk in markets]
            w = sum(1 for p in pnls if p > 0)
            l = sum(1 for p in pnls if p <= 0)
            st.caption(f"{len(markets)} trades  |  {w}W {l}L  |  ${sum(pnls):+.2f}")
        else:
            data = _api("/trades", {"mode": "paper", "hours": h, "limit": 200})
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
                "Entry": df["entry_price"].apply(lambda x: f"{x:.3f}" if x is not None else ""),
                "Exit": df["exit_price"].apply(lambda x: f"{x:.3f}" if x is not None else ""),
                "PnL": df["pnl"].apply(lambda x: f"${x:+.2f}" if x is not None else ""),
                "Result": df.apply(
                    lambda r: "WIN" if r.get("pnl") and r["pnl"] > 0
                    else ("LOSS" if r.get("pnl") is not None and r["pnl"] <= 0
                          else ("OPEN" if r.get("is_open") else "--")), axis=1
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


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: STRATEGY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

with tab_analysis:

    @st.fragment(run_every="30s")
    def analysis_tab_content():
        m = st.session_state.get("mode", "Live").lower()
        h = LOOKBACK_MAP[st.session_state.get("lookback", "24h")]
        selected_tags = st.session_state.get("strategies", [])

        data = _api("/trades", {"mode": m, "hours": h, "limit": 2000})
        trades = data.get("trades", [])

        # Apply strategy tag filter
        if selected_tags:
            trades = [t for t in trades if t.get("strategy_tag") in selected_tags]

        resolved = [t for t in trades if t.get("pnl") is not None]

        if not resolved:
            st.caption("No resolved trades in this period")
            return

        # ---------------------------------------------------------------
        # 1. Win Rate by Move % and Timing (always shown)
        # ---------------------------------------------------------------

        has_move = [t for t in resolved if t.get("dir_move_pct") is not None]
        has_timing = [t for t in resolved if t.get("minutes_into_slot") is not None]

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown(
                '<p class="section-label">Win Rate by Underlying Move %</p>',
                unsafe_allow_html=True,
            )
            if has_move:
                df_m = pd.DataFrame(has_move)
                bins = [-float("inf"), 0.0, 0.1, 0.2, 0.5, 1.0, float("inf")]
                labels = ["<0%", "0~0.1%", "0.1~0.2%", "0.2~0.5%", "0.5~1%", ">1%"]
                df_m["bucket"] = pd.cut(df_m["dir_move_pct"], bins=bins, labels=labels)
                grouped = df_m.groupby("bucket", observed=False).agg(
                    wins=("pnl", lambda x: (x > 0).sum()),
                    total=("pnl", "count"),
                ).reset_index()
                grouped["wr"] = (grouped["wins"] / grouped["total"].replace(0, float("nan")) * 100).round(1)
                grouped["bucket"] = grouped["bucket"].astype(str)

                colors = [C_GREEN if w >= 70 else (C_ACCENT if w >= 50 else C_RED) if pd.notna(w) else C_GRID for w in grouped["wr"]]

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=grouped["bucket"], y=grouped["wr"],
                    marker_color=colors, marker_line_width=0,
                    opacity=0.85,
                    text=grouped.apply(
                        lambda r: f"{r['wr']:.0f}% ({int(r['total'])})" if pd.notna(r['wr']) else "", axis=1
                    ),
                    textposition="outside",
                    cliponaxis=False,
                    textfont=dict(size=10),
                    hovertemplate="%{x}<br>Win Rate: %{y:.1f}%<br>%{customdata[0]}W / %{customdata[1]} total<extra></extra>",
                    customdata=grouped[["wins", "total"]].values,
                ))
                layout_mv = _plotly_layout(height=260)
                layout_mv["yaxis"]["tickprefix"] = ""
                layout_mv["yaxis"]["ticksuffix"] = "%"
                layout_mv["yaxis"]["range"] = [0, min(grouped["wr"].max(skipna=True) + 20, 110) if grouped["wr"].notna().any() else 110]
                layout_mv["xaxis"]["type"] = "category"
                fig.update_layout(**layout_mv)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            else:
                st.caption("Waiting for trades with move data (needs deploy)")

        with col_right:
            st.markdown(
                '<p class="section-label">Win Rate by Entry Timing (min)</p>',
                unsafe_allow_html=True,
            )
            if has_timing:
                df_t = pd.DataFrame(has_timing)
                bins_t = [0, 2, 4, 6, 8, 10, 12, 15]
                labels_t = ["0-2", "2-4", "4-6", "6-8", "8-10", "10-12", "12-15"]
                df_t["bucket"] = pd.cut(df_t["minutes_into_slot"], bins=bins_t, labels=labels_t, include_lowest=True)
                grouped_t = df_t.groupby("bucket", observed=False).agg(
                    wins=("pnl", lambda x: (x > 0).sum()),
                    total=("pnl", "count"),
                ).reset_index()
                grouped_t["wr"] = (grouped_t["wins"] / grouped_t["total"].replace(0, float("nan")) * 100).round(1)
                grouped_t["bucket"] = grouped_t["bucket"].astype(str)

                colors_t = [C_GREEN if w >= 70 else (C_ACCENT if w >= 50 else C_RED) if pd.notna(w) else C_GRID for w in grouped_t["wr"]]

                fig_t = go.Figure()
                fig_t.add_trace(go.Bar(
                    x=grouped_t["bucket"], y=grouped_t["wr"],
                    marker_color=colors_t, marker_line_width=0,
                    opacity=0.85,
                    text=grouped_t.apply(
                        lambda r: f"{r['wr']:.0f}% ({int(r['total'])})" if pd.notna(r['wr']) else "", axis=1
                    ),
                    textposition="outside",
                    cliponaxis=False,
                    textfont=dict(size=10),
                    hovertemplate="%{x} min<br>Win Rate: %{y:.1f}%<br>%{customdata[0]}W / %{customdata[1]} total<extra></extra>",
                    customdata=grouped_t[["wins", "total"]].values,
                ))
                layout_t = _plotly_layout(height=260)
                layout_t["yaxis"]["tickprefix"] = ""
                layout_t["yaxis"]["ticksuffix"] = "%"
                layout_t["yaxis"]["range"] = [0, min(grouped_t["wr"].max(skipna=True) + 20, 110) if grouped_t["wr"].notna().any() else 110]
                layout_t["xaxis"]["title"] = dict(text="minutes into slot", font=dict(size=10, color=C_MUTED))
                layout_t["xaxis"]["type"] = "category"
                fig_t.update_layout(**layout_t)
                st.plotly_chart(fig_t, use_container_width=True, config={"displayModeBar": False})
            else:
                st.caption("Waiting for trades with timing data (needs deploy)")

        st.markdown('<div style="height: 16px"></div>', unsafe_allow_html=True)

        # ---------------------------------------------------------------
        # 2. Market Correlation — do markets win/lose together per slot?
        # ---------------------------------------------------------------

        st.markdown('<p class="section-label">Market Correlation per Slot</p>', unsafe_allow_html=True)

        slot_trades: dict[int, list] = {}
        for t in resolved:
            title = t.get("title", "")
            slot_ts = _parse_slot_ts(title)
            if slot_ts is None:
                continue
            slot_trades.setdefault(slot_ts, []).append(t)

        if slot_trades:
            slot_rows = []
            for slot_ts, st_trades in sorted(slot_trades.items()):
                n_markets = len(st_trades)
                n_wins = sum(1 for t in st_trades if t["pnl"] > 0)
                n_losses = n_markets - n_wins
                total_pnl = sum(t["pnl"] for t in st_trades)
                assets = sorted(set(_parse_asset(t.get("title", "")) or "?" for t in st_trades))
                slot_rows.append({
                    "slot_ts": slot_ts,
                    "n_markets": n_markets,
                    "n_wins": n_wins,
                    "n_losses": n_losses,
                    "total_pnl": total_pnl,
                    "pattern": f"{n_wins}W {n_losses}L",
                    "assets": ", ".join(assets),
                })

            df_slots = pd.DataFrame(slot_rows)
            multi = df_slots[df_slots["n_markets"] >= 2]

            if not multi.empty:
                col_corr1, col_corr2 = st.columns(2)

                with col_corr1:
                    st.markdown(
                        '<p style="font-family: JetBrains Mono, monospace; font-size: 0.7rem; '
                        'color: #64748b; letter-spacing: 1px; text-transform: uppercase; margin: 0 0 4px;">'
                        'Slot Outcome Patterns</p>',
                        unsafe_allow_html=True,
                    )

                    pattern_counts = multi["pattern"].value_counts().reset_index()
                    pattern_counts.columns = ["pattern", "count"]
                    pattern_counts = pattern_counts.sort_values("pattern")

                    def _pattern_color(p):
                        if "0L" in p:
                            return C_GREEN
                        if "0W" in p:
                            return C_RED
                        return C_ACCENT

                    colors_p = [_pattern_color(p) for p in pattern_counts["pattern"]]
                    total_slots = pattern_counts["count"].sum()

                    fig_p = go.Figure()
                    fig_p.add_trace(go.Bar(
                        x=pattern_counts["pattern"], y=pattern_counts["count"],
                        marker_color=colors_p, marker_line_width=0,
                        opacity=0.85,
                        text=pattern_counts.apply(
                            lambda r: f"{int(r['count'])} ({r['count']/total_slots*100:.0f}%)", axis=1
                        ),
                        textposition="outside",
                        cliponaxis=False,
                        textfont=dict(size=10),
                        hovertemplate="%{x}<br>%{y} slots<extra></extra>",
                    ))
                    layout_p = _plotly_layout(height=260)
                    layout_p["yaxis"]["tickprefix"] = ""
                    layout_p["xaxis"]["title"] = dict(text="outcome pattern", font=dict(size=10, color=C_MUTED))
                    fig_p.update_layout(**layout_p)
                    st.plotly_chart(fig_p, use_container_width=True, config={"displayModeBar": False})

                with col_corr2:
                    st.markdown(
                        '<p style="font-family: JetBrains Mono, monospace; font-size: 0.7rem; '
                        'color: #64748b; letter-spacing: 1px; text-transform: uppercase; margin: 0 0 4px;">'
                        'Avg Slot PnL by Pattern</p>',
                        unsafe_allow_html=True,
                    )

                    pnl_by_pattern = multi.groupby("pattern").agg(
                        avg_pnl=("total_pnl", "mean"),
                        total_pnl=("total_pnl", "sum"),
                        count=("total_pnl", "count"),
                    ).reset_index().sort_values("pattern")

                    colors_pnl = [C_GREEN if p >= 0 else C_RED for p in pnl_by_pattern["avg_pnl"]]

                    fig_pnl = go.Figure()
                    fig_pnl.add_trace(go.Bar(
                        x=pnl_by_pattern["pattern"], y=pnl_by_pattern["avg_pnl"],
                        marker_color=colors_pnl, marker_line_width=0,
                        opacity=0.85,
                        text=pnl_by_pattern.apply(
                            lambda r: f"${r['avg_pnl']:+.2f}", axis=1
                        ),
                        textposition="outside",
                        cliponaxis=False,
                        textfont=dict(size=10),
                        hovertemplate="%{x}<br>Avg: $%{y:+.2f}<br>Total: $%{customdata[0]:+.2f}<br>%{customdata[1]} slots<extra></extra>",
                        customdata=pnl_by_pattern[["total_pnl", "count"]].values,
                    ))
                    fig_pnl.add_hline(y=0, line=dict(color="#334155", width=0.5, dash="dot"))
                    layout_pnl = _plotly_layout(height=260)
                    layout_pnl["xaxis"]["title"] = dict(text="outcome pattern", font=dict(size=10, color=C_MUTED))
                    fig_pnl.update_layout(**layout_pnl)
                    st.plotly_chart(fig_pnl, use_container_width=True, config={"displayModeBar": False})

                all_win = multi[multi["n_losses"] == 0]
                all_loss = multi[multi["n_wins"] == 0]
                mixed = multi[(multi["n_wins"] > 0) & (multi["n_losses"] > 0)]
                st.caption(
                    f"{len(multi)} slots with 2+ markets  |  "
                    f"All win: {len(all_win)} ({len(all_win)/len(multi)*100:.0f}%)  |  "
                    f"All loss: {len(all_loss)} ({len(all_loss)/len(multi)*100:.0f}%)  |  "
                    f"Mixed: {len(mixed)} ({len(mixed)/len(multi)*100:.0f}%)"
                )

            else:
                st.caption("No slots with multiple markets found")

    analysis_tab_content()


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3: SLOT ML
# ═══════════════════════════════════════════════════════════════════════════

_TIMING_ORDER = ["0-2", "2-4", "4-6", "6-8", "8-10", "10-12"]
_MOVE_ORDER = ["< -0.2", "-0.2/-0.1", "-0.1/0", "0/0.1", "0.1/0.2", "> 0.2"]
_DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

with tab_slot:

    @st.fragment(run_every="60s")
    def slot_ml_content():
        h = max(LOOKBACK_MAP[st.session_state.get("lookback", "24h")], 168)

        sym_choice = st.radio(
            "Symbol", ["All", "BTC", "ETH", "SOL", "XRP"],
            horizontal=True, key="slot_symbol",
        )

        params: dict = {"hours": h}
        if sym_choice != "All":
            params["symbol"] = sym_choice

        data = _api("/slots", params)

        if data.get("error"):
            st.warning(f"Slot data: {data['error']}")
            return

        total = data.get("total_slots", 0)
        if total == 0:
            st.caption("No slot data yet. Start the collector: `bin/run_slot_collector.sh`")
            return

        # -- KPIs --
        resolved = data.get("resolved", 0)
        unresolved = data.get("unresolved", 0)
        snap_count = data.get("snapshot_count", 0)
        last_ts = data.get("last_ts")

        import time as _time
        if last_ts:
            age_min = int((_time.time() - last_ts) / 60)
            age_str = f"{age_min}m ago" if age_min < 60 else f"{age_min // 60}h {age_min % 60}m ago"
        else:
            age_str = "--"

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Slots", f"{total:,}", delta=f"{resolved} resolved")
        c2.metric("In Progress", str(unresolved))
        c3.metric("Snapshots", f"{snap_count:,}")
        c4.metric("Data Age", age_str)

        st.markdown('<div style="height: 8px"></div>', unsafe_allow_html=True)

        # -- Heatmap: timing x move -> WR --
        heatmap = data.get("heatmap", [])
        if heatmap:
            st.markdown(
                '<p class="section-label">Win Rate Heatmap: Entry Timing x Price Move</p>',
                unsafe_allow_html=True,
            )

            lookup = {(r["timing"], r["move"]): r for r in heatmap}
            z = []
            text_arr = []
            for move in _MOVE_ORDER:
                row_z = []
                row_t = []
                for timing in _TIMING_ORDER:
                    r = lookup.get((timing, move))
                    if r and r["total"] >= 3:
                        row_z.append(r["wr"])
                        row_t.append(f"{r['wr']:.0f}%\n({r['total']})")
                    else:
                        row_z.append(None)
                        row_t.append("")
                z.append(row_z)
                text_arr.append(row_t)

            fig_hm = go.Figure(data=go.Heatmap(
                z=z,
                x=_TIMING_ORDER,
                y=_MOVE_ORDER,
                text=text_arr,
                texttemplate="%{text}",
                textfont=dict(size=11, color=C_TEXT),
                colorscale=[
                    [0, "#fca5a5"], [0.35, "#dc2626"],
                    [0.5, "#1e293b"],
                    [0.65, "#16a34a"], [1, "#22c55e"],
                ],
                zmid=50, zmin=25, zmax=75,
                colorbar=dict(
                    title=dict(text="WR%", font=dict(size=10, color=C_MUTED)),
                    ticksuffix="%",
                    tickfont=dict(color=C_MUTED, size=10),
                    bgcolor="rgba(0,0,0,0)",
                ),
                hovertemplate="Timing: %{x} min<br>Move: %{y}%<br>WR: %{z:.1f}%<extra></extra>",
                xgap=2, ygap=2,
            ))
            layout_hm = _plotly_layout(height=340)
            layout_hm["yaxis"]["tickprefix"] = ""
            layout_hm["yaxis"]["side"] = "left"
            layout_hm["xaxis"]["type"] = "category"
            layout_hm["yaxis"]["type"] = "category"
            layout_hm["margin"] = dict(l=80, r=60, t=36, b=40)
            layout_hm["xaxis"]["title"] = dict(text="minutes into slot", font=dict(size=10, color=C_MUTED))
            layout_hm["yaxis"]["title"] = dict(text="dir. move %", font=dict(size=10, color=C_MUTED))
            fig_hm.update_layout(**layout_hm)
            st.plotly_chart(fig_hm, use_container_width=True, config={"displayModeBar": False})

        st.markdown('<div style="height: 12px"></div>', unsafe_allow_html=True)

        # -- Bottom row: calibration + per-symbol --
        col_left, col_right = st.columns(2)

        with col_left:
            cal = data.get("calibration", [])
            if cal:
                st.markdown(
                    '<p class="section-label">Market Calibration (Bid Up vs Actual P(Up))</p>',
                    unsafe_allow_html=True,
                )
                fig_cal = go.Figure()
                # Perfect calibration line
                fig_cal.add_trace(go.Scatter(
                    x=[10, 95], y=[10, 95],
                    mode="lines", line=dict(color=C_MUTED, dash="dot", width=1),
                    showlegend=False, hoverinfo="skip",
                ))
                # Actual
                fig_cal.add_trace(go.Scatter(
                    x=[r["avg_bid"] * 100 for r in cal],
                    y=[r["wr"] for r in cal],
                    mode="lines+markers",
                    line=dict(color=C_ACCENT, width=2),
                    marker=dict(
                        size=[max(6, min(r["total"] / 5, 16)) for r in cal],
                        color=C_ACCENT,
                    ),
                    showlegend=False,
                    customdata=[[r["total"]] for r in cal],
                    hovertemplate=(
                        "Bid: %{x:.0f}%<br>Actual: %{y:.1f}%<br>"
                        "n=%{customdata[0]}<extra></extra>"
                    ),
                ))
                layout_cal = _plotly_layout(height=300)
                layout_cal["xaxis"]["title"] = dict(
                    text="market implied P(Up) %", font=dict(size=10, color=C_MUTED),
                )
                layout_cal["yaxis"]["title"] = dict(
                    text="actual win rate %", font=dict(size=10, color=C_MUTED),
                )
                layout_cal["yaxis"]["tickprefix"] = ""
                layout_cal["yaxis"]["ticksuffix"] = "%"
                layout_cal["xaxis"]["ticksuffix"] = "%"
                fig_cal.update_layout(**layout_cal)
                st.plotly_chart(fig_cal, use_container_width=True, config={"displayModeBar": False})
            else:
                st.caption("Calibration data requires snapshots at 4-10 min")

        with col_right:
            by_symbol = data.get("by_symbol", [])
            if by_symbol:
                st.markdown(
                    '<p class="section-label">Win Rate by Symbol</p>',
                    unsafe_allow_html=True,
                )
                colors_s = [C_GREEN if s["wr"] >= 50 else C_RED for s in by_symbol]
                fig_s = go.Figure()
                fig_s.add_trace(go.Bar(
                    x=[s["symbol"] for s in by_symbol],
                    y=[s["wr"] for s in by_symbol],
                    marker_color=colors_s, marker_line_width=0, opacity=0.85,
                    text=[f"{s['wr']:.1f}% ({s['total']})" for s in by_symbol],
                    textposition="outside", cliponaxis=False, textfont=dict(size=10),
                    customdata=[[s["wins"], s["total"]] for s in by_symbol],
                    hovertemplate=(
                        "%{x}<br>WR: %{y:.1f}%<br>"
                        "%{customdata[0]}W / %{customdata[1]}<extra></extra>"
                    ),
                ))
                fig_s.add_hline(y=50, line=dict(color=C_MUTED, width=0.5, dash="dot"))
                layout_s = _plotly_layout(height=300)
                layout_s["yaxis"]["tickprefix"] = ""
                layout_s["yaxis"]["ticksuffix"] = "%"
                layout_s["yaxis"]["range"] = [0, 100]
                fig_s.update_layout(**layout_s)
                st.plotly_chart(fig_s, use_container_width=True, config={"displayModeBar": False})
            else:
                st.caption("No resolved slots yet")

        st.markdown('<div style="height: 12px"></div>', unsafe_allow_html=True)

        # -- Bottom: hour + day patterns --
        by_hour = data.get("by_hour", [])
        by_day = data.get("by_day", [])

        if by_hour or by_day:
            col_h, col_d = st.columns(2)

            with col_h:
                if by_hour:
                    st.markdown(
                        '<p class="section-label">Win Rate by Hour (UTC)</p>',
                        unsafe_allow_html=True,
                    )
                    colors_hr = [C_GREEN if h_d["wr"] >= 50 else C_RED for h_d in by_hour]
                    fig_hr = go.Figure()
                    fig_hr.add_trace(go.Bar(
                        x=[f"{h_d['hour']:02d}" for h_d in by_hour],
                        y=[h_d["wr"] for h_d in by_hour],
                        marker_color=colors_hr, marker_line_width=0, opacity=0.85,
                        text=[f"{h_d['wr']:.0f}%" for h_d in by_hour],
                        textposition="outside", cliponaxis=False, textfont=dict(size=9),
                        customdata=[[h_d["wins"], h_d["total"]] for h_d in by_hour],
                        hovertemplate=(
                            "%{x}:00 UTC<br>WR: %{y:.1f}%<br>"
                            "%{customdata[0]}W / %{customdata[1]}<extra></extra>"
                        ),
                    ))
                    fig_hr.add_hline(y=50, line=dict(color=C_MUTED, width=0.5, dash="dot"))
                    layout_hr = _plotly_layout(height=240)
                    layout_hr["yaxis"]["tickprefix"] = ""
                    layout_hr["yaxis"]["ticksuffix"] = "%"
                    max_hr = max(h_d["wr"] for h_d in by_hour) if by_hour else 50
                    layout_hr["yaxis"]["range"] = [0, min(max_hr + 15, 100)]
                    layout_hr["xaxis"]["type"] = "category"
                    fig_hr.update_layout(**layout_hr)
                    st.plotly_chart(fig_hr, use_container_width=True, config={"displayModeBar": False})

            with col_d:
                if by_day:
                    st.markdown(
                        '<p class="section-label">Win Rate by Day of Week</p>',
                        unsafe_allow_html=True,
                    )
                    colors_dy = [C_GREEN if d["wr"] >= 50 else C_RED for d in by_day]
                    fig_dy = go.Figure()
                    fig_dy.add_trace(go.Bar(
                        x=[_DAY_NAMES[d["day"]] if d["day"] < 7 else str(d["day"]) for d in by_day],
                        y=[d["wr"] for d in by_day],
                        marker_color=colors_dy, marker_line_width=0, opacity=0.85,
                        text=[f"{d['wr']:.0f}% ({d['total']})" for d in by_day],
                        textposition="outside", cliponaxis=False, textfont=dict(size=9),
                        customdata=[[d["wins"], d["total"]] for d in by_day],
                        hovertemplate=(
                            "%{x}<br>WR: %{y:.1f}%<br>"
                            "%{customdata[0]}W / %{customdata[1]}<extra></extra>"
                        ),
                    ))
                    fig_dy.add_hline(y=50, line=dict(color=C_MUTED, width=0.5, dash="dot"))
                    layout_dy = _plotly_layout(height=240)
                    layout_dy["yaxis"]["tickprefix"] = ""
                    layout_dy["yaxis"]["ticksuffix"] = "%"
                    max_dy = max(d["wr"] for d in by_day) if by_day else 50
                    layout_dy["yaxis"]["range"] = [0, min(max_dy + 15, 100)]
                    layout_dy["xaxis"]["type"] = "category"
                    fig_dy.update_layout(**layout_dy)
                    st.plotly_chart(fig_dy, use_container_width=True, config={"displayModeBar": False})

        # Summary caption
        if by_symbol:
            total_wins = sum(s["wins"] for s in by_symbol)
            total_res = sum(s["total"] for s in by_symbol)
            overall_wr = round(total_wins / total_res * 100, 1) if total_res else 0
            st.caption(
                f"{total_res} resolved slots  |  "
                f"Overall WR: {overall_wr}%  |  "
                f"Lookback: {h}h  |  "
                f"Auto-refresh 60s"
            )

    slot_ml_content()
