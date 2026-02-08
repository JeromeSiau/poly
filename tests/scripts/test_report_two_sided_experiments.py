"""Tests for two-sided experiment reporting helpers."""

from datetime import datetime

import pytest

from scripts.report_two_sided_experiments import (
    TradeRow,
    _extract_trade_row,
    summarize_pairs,
)
from src.db.models import LiveObservation, PaperTrade


def test_extract_trade_row_reads_strategy_tag_and_shares() -> None:
    observation = LiveObservation(
        timestamp=datetime(2026, 2, 8, 10, 0, 0),
        match_id="cond-1",
        event_type="two_sided_inventory",
        game_state={
            "strategy_tag": "edge_1p2_0p2",
            "run_id": "edge_1p2_0p2-20260208",
            "condition_id": "cond-1",
            "title": "Will Team A win?",
            "outcome": "Yes",
            "shares": 200.0,
        },
        model_prediction=0.55,
        polymarket_price=0.50,
    )
    trade = PaperTrade(
        observation_id=1,
        side="BUY",
        entry_price=0.50,
        simulated_fill_price=0.50,
        size=100.0,
        edge_theoretical=0.02,
        edge_realized=None,
        pnl=None,
        created_at=datetime(2026, 2, 8, 10, 0, 0),
    )

    row = _extract_trade_row(trade, observation)
    assert row is not None
    assert row.strategy_tag == "edge_1p2_0p2"
    assert row.run_id == "edge_1p2_0p2-20260208"
    assert row.shares == pytest.approx(200.0, rel=1e-9)


def test_summarize_pairs_aggregates_pnl_and_inventory() -> None:
    rows = [
        TradeRow(
            strategy_tag="edge_1p5_0p3",
            run_id="run-a",
            condition_id="cond-1",
            title="Will Team A win?",
            outcome="Yes",
            side="BUY",
            shares=200.0,
            size_usd=100.0,
            edge_theoretical=0.02,
            edge_realized=0.0,
            pnl=0.0,
            created_at=datetime(2026, 2, 8, 10, 0, 0),
        ),
        TradeRow(
            strategy_tag="edge_1p5_0p3",
            run_id="run-a",
            condition_id="cond-1",
            title="Will Team A win?",
            outcome="Yes",
            side="SELL",
            shares=80.0,
            size_usd=48.0,
            edge_theoretical=0.01,
            edge_realized=0.15,
            pnl=12.0,
            created_at=datetime(2026, 2, 8, 10, 5, 0),
        ),
    ]

    summaries = summarize_pairs(rows)
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.realized_pnl == pytest.approx(12.0, rel=1e-9)
    assert summary.trades == 2
    assert summary.sells == 1
    assert summary.win_rate == pytest.approx(1.0, rel=1e-9)
    assert summary.net_shares == pytest.approx(120.0, rel=1e-9)
