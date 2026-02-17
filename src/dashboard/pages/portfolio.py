"""Portfolio page — Live + Paper mode with internal toggle."""

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
    PERIOD_PRESETS,
    api,
    parse_asset,
    parse_slot_ts,
    period_params,
    plotly_layout,
    style_pnl,
    style_result,
)

# -- Period filter (inline, same position as ML page) --
period_col, date_col = st.columns([1, 2])
with period_col:
    st.radio("Period", list(PERIOD_PRESETS.keys()), index=0, horizontal=True, key="period")
preset_days = PERIOD_PRESETS.get(st.session_state.get("period", "24h"), 1)
if preset_days == -1:
    with date_col:
        today = date.today()
        st.date_input(
            "Date range",
            value=(today - timedelta(days=14), today),
            max_value=today,
            key="period_dates",
        )

# ---------------------------------------------------------------------------
# Tab 1: Live
# ---------------------------------------------------------------------------

tab_live, tab_analysis = st.tabs(["Live", "Strategy Analysis"])

with tab_live:

    @st.fragment(run_every="15s")
    def kpi_section():
        m = "live" if st.session_state.get("nav_mode", "Live") == "Live" else "paper"
        pp = period_params()

        balance_data = api("/balance", {"mode": m})
        winrate_data = api("/winrate", {"mode": m, **pp})

        bal = balance_data.get("balance", 0.0)
        pnl = winrate_data.get("total_pnl", 0.0)
        wr = winrate_data.get("winrate", 0.0)
        pf = winrate_data.get("profit_factor")
        wins = winrate_data.get("wins", 0)
        losses = winrate_data.get("losses", 0)
        roi = winrate_data.get("roi_pct", 0.0)

        pos_data = api("/positions", {"mode": m})
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

    # -- Cumulative PnL + Balance curves --
    @st.fragment(run_every="15s")
    def pnl_chart_section():
        m = "live" if st.session_state.get("nav_mode", "Live") == "Live" else "paper"
        pp = period_params()

        balance_data = api("/balance", {"mode": m})
        winrate_data = api("/winrate", {"mode": m, **pp})
        markets = winrate_data.get("markets", [])
        current_bal = balance_data.get("balance", 0.0)
        total_pnl = winrate_data.get("total_pnl", 0.0)

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
        start_bal = current_bal - total_pnl
        df["balance"] = start_bal + df["cum_pnl"]

        col_pnl, col_bal = st.columns(2)

        # -- Cumulative PnL --
        with col_pnl:
            st.markdown('<p class="section-label">Cumulative PnL</p>', unsafe_allow_html=True)
            final_pnl = df["cum_pnl"].iloc[-1]
            pnl_up = final_pnl >= 0
            pnl_color = C_GREEN if pnl_up else C_RED
            pnl_fill = "rgba(52,211,153,0.08)" if pnl_up else "rgba(248,113,113,0.08)"

            fig_pnl = go.Figure()
            fig_pnl.add_trace(go.Scatter(
                x=df["time"], y=df["cum_pnl"],
                mode="lines",
                line=dict(color=pnl_color, width=1.5),
                fill="tozeroy", fillcolor=pnl_fill,
                hovertemplate="%{x|%H:%M}<br>$%{y:+,.2f}<extra></extra>",
            ))
            fig_pnl.add_hline(y=0, line=dict(color="#334155", width=0.5, dash="dot"))
            fig_pnl.update_layout(**plotly_layout(height=280))
            st.plotly_chart(fig_pnl, use_container_width=True, config={"displayModeBar": False})

        # -- Balance --
        with col_bal:
            st.markdown('<p class="section-label">Balance</p>', unsafe_allow_html=True)
            final_bal = df["balance"].iloc[-1]
            bal_up = final_bal >= start_bal
            bal_color = C_GREEN if bal_up else C_RED
            bal_fill = "rgba(52,211,153,0.08)" if bal_up else "rgba(248,113,113,0.08)"

            fig_bal = go.Figure()
            fig_bal.add_trace(go.Scatter(
                x=df["time"], y=df["balance"],
                mode="lines",
                line=dict(color=bal_color, width=1.5),
                fill="tozeroy", fillcolor=bal_fill,
                hovertemplate="%{x|%H:%M}<br>$%{y:,.2f}<extra></extra>",
            ))
            fig_bal.add_hline(y=start_bal, line=dict(color="#334155", width=0.5, dash="dot"))
            fig_bal.update_layout(**plotly_layout(height=280))
            st.plotly_chart(fig_bal, use_container_width=True, config={"displayModeBar": False})

    pnl_chart_section()

    # -- Hourly PnL --
    @st.fragment(run_every="15s")
    def hourly_pnl_section():
        m = "live" if st.session_state.get("nav_mode", "Live") == "Live" else "paper"
        pp = period_params()

        st.markdown('<p class="section-label">PnL by Hour</p>', unsafe_allow_html=True)

        winrate_data = api("/winrate", {"mode": m, **pp})
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
        fig.update_layout(**plotly_layout(height=220))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    hourly_pnl_section()

    # -- Open Positions --
    @st.fragment(run_every="15s")
    def open_positions_section():
        m = "live" if st.session_state.get("nav_mode", "Live") == "Live" else "paper"

        st.markdown('<p class="section-label">Open Positions</p>', unsafe_allow_html=True)

        pos_data = api("/positions", {"mode": m})
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

        styled = display.style.map(style_pnl, subset=["PnL"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

    open_positions_section()

    # -- Recent Trades --
    @st.fragment(run_every="15s")
    def recent_trades_section():
        m = "live" if st.session_state.get("nav_mode", "Live") == "Live" else "paper"
        pp = period_params()

        st.markdown('<p class="section-label">Recent Trades</p>', unsafe_allow_html=True)

        if m == "live":
            winrate_data = api("/winrate", {"mode": "live", **pp})
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
            styled = display.style.map(style_pnl, subset=["PnL"]).map(
                style_result, subset=["Result"]
            )
            st.dataframe(styled, use_container_width=True, hide_index=True, height=400)
            pnls = [mk["pnl"] for mk in markets]
            w = sum(1 for p in pnls if p > 0)
            l = sum(1 for p in pnls if p <= 0)
            st.caption(f"{len(markets)} trades  |  {w}W {l}L  |  ${sum(pnls):+.2f}")
        else:
            data = api("/trades", {"mode": "paper", **pp, "limit": 200})
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
            styled = display.style.map(style_pnl, subset=["PnL"]).map(
                style_result, subset=["Result"]
            )
            st.dataframe(styled, use_container_width=True, hide_index=True, height=400)
            pnls = [t["pnl"] for t in trades if t.get("pnl") is not None]
            if pnls:
                w = sum(1 for p in pnls if p > 0)
                l = sum(1 for p in pnls if p <= 0)
                st.caption(f"{len(display)} trades  |  {w}W {l}L  |  ${sum(pnls):+.2f}")

    recent_trades_section()


# ---------------------------------------------------------------------------
# Tab 2: Strategy Analysis
# ---------------------------------------------------------------------------

with tab_analysis:

    @st.fragment(run_every="30s")
    def analysis_tab_content():
        m = "live" if st.session_state.get("nav_mode", "Live") == "Live" else "paper"
        pp = period_params()
        selected_tags = st.session_state.get("strategies", [])

        data = api("/trades", {"mode": m, **pp, "limit": 2000})
        trades = data.get("trades", [])

        if selected_tags:
            trades = [t for t in trades if t.get("strategy_tag") in selected_tags]

        resolved = [t for t in trades if t.get("pnl") is not None]

        if not resolved:
            st.caption("No resolved trades in this period")
            return

        # -- Win Rate by Move % and Timing --
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
                layout_mv = plotly_layout(height=260)
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
                layout_t = plotly_layout(height=260)
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

        # -- Market Correlation per Slot --
        st.markdown('<p class="section-label">Market Correlation per Slot</p>', unsafe_allow_html=True)

        slot_trades: dict[int, list] = {}
        for t in resolved:
            title = t.get("title", "")
            slot_ts = parse_slot_ts(title)
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
                assets = sorted(set(parse_asset(t.get("title", "")) or "?" for t in st_trades))
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
                    layout_p = plotly_layout(height=260)
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
                    layout_pnl = plotly_layout(height=260)
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
