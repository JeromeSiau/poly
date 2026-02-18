"""Shared constants, helpers and CSS for the dashboard."""

import re
from datetime import datetime, timezone

import httpx
import streamlit as st

API_BASE = "http://localhost:8788"

LOOKBACK_MAP = {"1h": 1, "4h": 4, "12h": 12, "24h": 24, "48h": 48, "7d": 168}

PERIOD_PRESETS = {
    "1h": 1,
    "4h": 4,
    "8h": 8,
    "12h": 12,
    "24h": 24,
    "48h": 48,
    "7d": 168,
    "14d": 336,
    "30d": 720,
    "All": None,
    "Custom": -1,
}

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


def period_params() -> dict:
    """Build API params dict from the current period session state."""
    preset_hours = PERIOD_PRESETS.get(st.session_state.get("period", "24h"), 24)
    if preset_hours == -1:
        # Custom date range
        dates = st.session_state.get("period_dates")
        if isinstance(dates, (list, tuple)) and len(dates) == 2:
            d_start, d_end = dates
            s = int(datetime.combine(d_start, datetime.min.time(), tzinfo=timezone.utc).timestamp())
            e = int(datetime.combine(d_end, datetime.max.time(), tzinfo=timezone.utc).timestamp())
            return {"hours": 720, "start_ts": s, "end_ts": e}
        return {"hours": 24}
    elif preset_hours is None:
        return {"hours": 720}
    else:
        return {"hours": preset_hours}


def api(endpoint: str, params: dict | None = None) -> dict:
    try:
        resp = httpx.get(API_BASE + endpoint, params=params or {}, timeout=10)
        return resp.json()
    except Exception as exc:
        st.warning(f"API error on {endpoint}: {exc}")
        return {}


def humanize_age(ts_str: str) -> str:
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


def style_pnl(val):
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


def style_result(val):
    """Color WIN/LOSS/OPEN labels."""
    if val == "WIN":
        return f"color: {C_GREEN}"
    if val == "LOSS":
        return f"color: {C_RED}"
    if val == "OPEN":
        return f"color: {C_ACCENT}"
    return f"color: {C_MUTED}"


def plotly_layout(**overrides) -> dict:
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


def parse_slot_ts(title: str) -> int | None:
    """Extract slot unix timestamp from slug like 'btc-updown-15m-1771079400'."""
    if not title:
        return None
    m = re.search(r"-(\d{10})$", title)
    return int(m.group(1)) if m else None


def parse_asset(title: str) -> str | None:
    """Extract asset from slug like 'btc-updown-15m-1771079400' -> 'BTC'."""
    if not title:
        return None
    parts = title.split("-")
    if len(parts) >= 2:
        return parts[0].upper()
    return None


DASHBOARD_CSS = """
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
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(52,211,153,0.4); }
    50% { opacity: 0.7; box-shadow: 0 0 0 6px rgba(52,211,153,0); }
}

/* Nav mode links — style st.radio as inline links */
div[data-testid="stHorizontalBlock"]:has(> div .nav-radio) {
    margin-left: auto;
}
.nav-radio [role="radiogroup"] {
    gap: 0 !important;
    flex-direction: row !important;
}
.nav-radio [role="radiogroup"] label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #64748b !important;
    padding: 4px 16px !important;
    border-bottom: 2px solid transparent;
    cursor: pointer;
    background: transparent !important;
}
.nav-radio [role="radiogroup"] label[data-checked="true"],
.nav-radio [role="radiogroup"] label:has(input:checked) {
    color: #38bdf8 !important;
    border-bottom-color: #38bdf8;
}
.nav-radio [role="radiogroup"] label p {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    color: inherit !important;
}
/* Hide the radio circle */
.nav-radio [role="radiogroup"] label > div:first-child {
    display: none !important;
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
"""
