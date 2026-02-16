"""Slot analysis page — ML mode."""

import time as _time
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard._shared import (
    C_ACCENT,
    C_GREEN,
    C_GRID,
    C_MUTED,
    C_RED,
    C_TEXT,
    api,
    plotly_layout,
)

_TIMING_15M = ["0-2", "2-4", "4-6", "6-8", "8-10", "10-12"]
_TIMING_5M = ["0-1", "1-2", "2-3", "3-4", "4-5"]
_MOVE_ORDER = ["< -0.2", "-0.2/-0.1", "-0.1/0", "0/0.1", "0.1/0.2", "> 0.2"]
_DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


_PERIOD_PRESETS = {
    "7d": 7,
    "14d": 14,
    "30d": 30,
    "All": None,
    "Custom": -1,
}


@st.fragment(run_every="60s")
def slot_ml_content():
    # -- Filters row 1: duration + symbol --
    filter_left, filter_right = st.columns(2)
    with filter_left:
        dur_choice = st.radio(
            "Duration", ["All", "15m", "5m"],
            horizontal=True, key="slot_duration",
        )
    with filter_right:
        sym_options = ["All", "BTC", "ETH", "SOL", "XRP"] if dur_choice != "5m" else ["BTC"]
        sym_choice = st.radio(
            "Symbol", sym_options,
            horizontal=True, key="slot_symbol",
        )

    # -- Filters row 2: period --
    period_col, date_col = st.columns([1, 2])
    with period_col:
        period_choice = st.radio(
            "Period", list(_PERIOD_PRESETS.keys()),
            horizontal=True, key="slot_period",
        )
    preset_days = _PERIOD_PRESETS[period_choice]

    start_ts = None
    end_ts = None
    h = 2160  # max fallback

    if preset_days == -1:
        # Custom date range
        with date_col:
            today = date.today()
            dates = st.date_input(
                "Date range",
                value=(today - timedelta(days=14), today),
                max_value=today,
                key="slot_dates",
            )
            if isinstance(dates, (list, tuple)) and len(dates) == 2:
                d_start, d_end = dates
                start_ts = int(datetime.combine(d_start, datetime.min.time(), tzinfo=timezone.utc).timestamp())
                end_ts = int(datetime.combine(d_end, datetime.max.time(), tzinfo=timezone.utc).timestamp())
    elif preset_days is not None:
        h = preset_days * 24
    else:
        # "All" — use max lookback
        h = 2160

    duration = dur_choice if dur_choice != "All" else None
    timing_order = _TIMING_5M if duration == "5m" else _TIMING_15M

    params: dict = {"hours": h}
    if start_ts is not None:
        params["start_ts"] = start_ts
    if end_ts is not None:
        params["end_ts"] = end_ts
    if sym_choice != "All":
        params["symbol"] = sym_choice
    if duration:
        params["duration"] = duration

    data = api("/slots", params)

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
            for timing in timing_order:
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
            x=timing_order,
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
        layout_hm = plotly_layout(height=340)
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
            fig_cal.add_trace(go.Scatter(
                x=[10, 95], y=[10, 95],
                mode="lines", line=dict(color=C_MUTED, dash="dot", width=1),
                showlegend=False, hoverinfo="skip",
            ))
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
            layout_cal = plotly_layout(height=300)
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
            layout_s = plotly_layout(height=300)
            layout_s["yaxis"]["tickprefix"] = ""
            layout_s["yaxis"]["ticksuffix"] = "%"
            layout_s["yaxis"]["range"] = [0, 100]
            fig_s.update_layout(**layout_s)
            st.plotly_chart(fig_s, use_container_width=True, config={"displayModeBar": False})
        else:
            st.caption("No resolved slots yet")

    st.markdown('<div style="height: 12px"></div>', unsafe_allow_html=True)

    # -- Stop-loss analysis --
    sl_params: dict = {"hours": h, "peak": 0.75}
    if start_ts is not None:
        sl_params["start_ts"] = start_ts
    if end_ts is not None:
        sl_params["end_ts"] = end_ts
    if sym_choice != "All":
        sl_params["symbol"] = sym_choice
    if duration:
        sl_params["duration"] = duration
    sl_data = api("/slots/stoploss", sl_params)

    if sl_data.get("thresholds") and sl_data.get("total_peaked", 0) > 0:
        st.markdown(
            '<p class="section-label">Stop-Loss Threshold Sweep (bid dip after peak &ge; 0.75)</p>',
            unsafe_allow_html=True,
        )

        sl_items = sl_data["thresholds"]
        total_peaked = sl_data["total_peaked"]

        thresholds = [f"{s['threshold']:.2f}" for s in sl_items]
        true_saves = [s.get("true_saves", 0) for s in sl_items]
        false_exits = [s.get("false_exits", 0) for s in sl_items]
        precision = [s.get("precision", 0) for s in sl_items]

        fig_sl = go.Figure()
        fig_sl.add_trace(go.Bar(
            x=thresholds, y=true_saves,
            name="True Saves",
            marker_color=C_GREEN, opacity=0.7,
            yaxis="y",
            hovertemplate="Threshold: %{x}<br>True saves: %{y}<extra></extra>",
        ))
        fig_sl.add_trace(go.Bar(
            x=thresholds, y=false_exits,
            name="False Exits",
            marker_color=C_RED, opacity=0.7,
            yaxis="y",
            hovertemplate="Threshold: %{x}<br>False exits: %{y}<extra></extra>",
        ))
        fig_sl.add_trace(go.Scatter(
            x=thresholds, y=precision,
            name="Precision %",
            mode="lines+markers",
            line=dict(color=C_ACCENT, width=2),
            marker=dict(size=6, color=C_ACCENT),
            yaxis="y2",
            hovertemplate="Threshold: %{x}<br>Precision: %{y:.0f}%<extra></extra>",
        ))

        layout_sl = plotly_layout(height=300)
        layout_sl.update(
            xaxis=dict(
                title=dict(text="dip threshold", font=dict(size=10, color=C_MUTED)),
                type="category",
                gridcolor=C_GRID,
                tickfont=dict(color=C_MUTED, size=10),
            ),
            yaxis=dict(
                title=dict(text="slots", font=dict(size=10, color=C_MUTED)),
                side="left",
                gridcolor=C_GRID,
                tickfont=dict(color=C_MUTED, size=10),
            ),
            yaxis2=dict(
                title=dict(text="precision %", font=dict(size=10, color=C_ACCENT)),
                side="right",
                overlaying="y",
                gridcolor="rgba(0,0,0,0)",
                tickfont=dict(color=C_ACCENT, size=10),
                ticksuffix="%",
                range=[0, 100],
            ),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                font=dict(color=C_MUTED, size=10),
            ),
            barmode="stack",
        )
        fig_sl.update_layout(**layout_sl)
        st.plotly_chart(fig_sl, use_container_width=True, config={"displayModeBar": False})

        sl_df = pd.DataFrame(sl_items)
        sl_df = sl_df[sl_df["triggered"] > 0]
        if not sl_df.empty:
            sl_df["threshold"] = sl_df["threshold"].apply(lambda x: f"{x:.2f}")
            if "precision" not in sl_df.columns:
                sl_df["precision"] = sl_df.apply(
                    lambda r: round(r["true_saves"] / r["triggered"] * 100, 1) if r["triggered"] else 0, axis=1
                )
            sl_df["precision"] = sl_df["precision"].apply(lambda x: f"{x:.0f}%")
            st.dataframe(
                sl_df[["threshold", "triggered", "true_saves", "false_exits", "precision"]].rename(
                    columns={
                        "threshold": "Threshold",
                        "triggered": "Triggered",
                        "true_saves": "True Saves",
                        "false_exits": "False Exits",
                        "precision": "Precision",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
        st.caption(
            f"{total_peaked} slots peaked 0.75+ ({sl_items[0]['wins']}W / {sl_items[0]['losses']}L)  |  "
            f"Precision = true saves / triggered"
        )

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
                layout_hr = plotly_layout(height=240)
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
                layout_dy = plotly_layout(height=240)
                layout_dy["yaxis"]["tickprefix"] = ""
                layout_dy["yaxis"]["ticksuffix"] = "%"
                max_dy = max(d["wr"] for d in by_day) if by_day else 50
                layout_dy["yaxis"]["range"] = [0, min(max_dy + 15, 100)]
                layout_dy["xaxis"]["type"] = "category"
                fig_dy.update_layout(**layout_dy)
                st.plotly_chart(fig_dy, use_container_width=True, config={"displayModeBar": False})

    # Summary caption
    by_symbol = data.get("by_symbol", [])
    if by_symbol:
        total_wins = sum(s["wins"] for s in by_symbol)
        total_res = sum(s["total"] for s in by_symbol)
        overall_wr = round(total_wins / total_res * 100, 1) if total_res else 0
        if start_ts is not None and end_ts is not None:
            period_label = f"{dates[0].strftime('%b %d')} - {dates[1].strftime('%b %d')}"
        elif preset_days is not None and preset_days > 0:
            period_label = f"{preset_days}d"
        else:
            period_label = "all"
        st.caption(
            f"{total_res} resolved slots  |  "
            f"Overall WR: {overall_wr}%  |  "
            f"Period: {period_label}  |  "
            f"Auto-refresh 60s"
        )


slot_ml_content()
