"""Dashboard entrypoint — navigation, shared sidebar, header."""

import streamlit as st

from datetime import date, timedelta

from src.dashboard._shared import DASHBOARD_CSS, PERIOD_PRESETS, api

# -- Page config (must be first st call) --
st.set_page_config(page_title="Poly", page_icon="P", layout="wide")
st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)

# -- Pages --
portfolio_page = st.Page(
    "pages/portfolio.py", title="Portfolio", url_path="live", default=True,
)
ml_page = st.Page(
    "pages/slot_analysis.py", title="Slot Analysis", url_path="ml",
)
pg = st.navigation([portfolio_page, ml_page], position="hidden")

# -- Determine current nav mode --
# On ML page, force nav_mode to "ML". On portfolio page, default to "Live".
is_ml = pg == ml_page
if is_ml:
    st.session_state["nav_mode"] = "ML"
elif "nav_mode" not in st.session_state or st.session_state["nav_mode"] == "ML":
    # Landing on portfolio page (or first visit): check query params
    qp_mode = st.query_params.get("mode", "Live")
    st.session_state["nav_mode"] = qp_mode if qp_mode in ("Live", "Paper") else "Live"

nav_mode = st.session_state["nav_mode"]

# -- Sidebar --
if not is_ml:
    st.sidebar.radio("Period", list(PERIOD_PRESETS.keys()), index=0, key="period", horizontal=True)
    preset_days = PERIOD_PRESETS.get(st.session_state.get("period", "24h"), 1)
    if preset_days == -1:
        today = date.today()
        st.sidebar.date_input(
            "Date range",
            value=(today - timedelta(days=14), today),
            max_value=today,
            key="period_dates",
        )

    from src.dashboard._shared import period_params
    pp = period_params()
    tags_data = api("/tags", pp)
    available_tags = sorted(tags_data.get("strategy_tags", {}).keys())
    st.sidebar.multiselect("Strategies", available_tags, key="strategies")

# -- Header --
hdr_left, hdr_right = st.columns([3, 2])
with hdr_left:
    live_dot = '<div class="live-dot"></div>' if nav_mode == "Live" else ""
    st.markdown(
        f'<div class="dash-header">'
        f'<span class="logo">POLY</span>'
        f'<span class="title">Portfolio Dashboard</span>'
        f'{live_dot}'
        f'</div>',
        unsafe_allow_html=True,
    )
with hdr_right:
    st.markdown('<div class="nav-radio">', unsafe_allow_html=True)
    selected = st.radio(
        "nav", ["Live", "ML", "Paper"],
        index=["Live", "ML", "Paper"].index(nav_mode),
        key="nav_radio",
        horizontal=True,
        label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Route between pages based on radio selection
    if selected == "ML" and not is_ml:
        st.switch_page(ml_page)
    elif selected in ("Live", "Paper") and is_ml:
        st.session_state["nav_mode"] = selected
        st.switch_page(portfolio_page)
    elif selected in ("Live", "Paper") and selected != nav_mode:
        st.session_state["nav_mode"] = selected
        st.query_params["mode"] = selected
        st.rerun()

# Sync query params for bookmarkability
if not is_ml:
    st.query_params["mode"] = nav_mode

# -- Run the selected page --
pg.run()
