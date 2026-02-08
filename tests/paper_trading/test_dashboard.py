"""Tests for dashboard helpers used by two-sided reporting."""

from datetime import datetime

import pytest

from src.db.models import LiveObservation, PaperTrade
from src.paper_trading.dashboard import (
    available_two_sided_tags,
    extract_two_sided_trade_rows,
    summarize_two_sided_pairs,
)


def test_extract_two_sided_trade_rows_and_tags() -> None:
    observations = [
        LiveObservation(
            id=1,
            timestamp=datetime(2026, 2, 8, 10, 0, 0),
            match_id="cond-1",
            event_type="two_sided_inventory",
            game_state={
                "strategy_tag": "edge_1p5_0p3",
                "condition_id": "cond-1",
                "title": "Will Team A win?",
                "outcome": "Yes",
                "shares": 200.0,
            },
            model_prediction=0.55,
            polymarket_price=0.50,
        ),
        LiveObservation(
            id=2,
            timestamp=datetime(2026, 2, 8, 10, 5, 0),
            match_id="other",
            event_type="paper_trade",
            game_state={},
            model_prediction=0.50,
            polymarket_price=0.50,
        ),
    ]
    trades = [
        PaperTrade(
            id=10,
            observation_id=1,
            side="BUY",
            entry_price=0.50,
            simulated_fill_price=0.50,
            size=100.0,
            edge_theoretical=0.02,
            edge_realized=None,
            pnl=None,
            created_at=datetime(2026, 2, 8, 10, 0, 0),
        ),
        PaperTrade(
            id=11,
            observation_id=2,
            side="BUY",
            entry_price=0.40,
            simulated_fill_price=0.40,
            size=80.0,
            edge_theoretical=0.01,
            edge_realized=None,
            pnl=None,
            created_at=datetime(2026, 2, 8, 10, 5, 0),
        ),
    ]

    rows = extract_two_sided_trade_rows(observations, trades)
    assert len(rows) == 1
    assert rows[0]["strategy_tag"] == "edge_1p5_0p3"
    assert rows[0]["shares"] == pytest.approx(200.0, rel=1e-9)

    tags = available_two_sided_tags(rows)
    assert tags == ["edge_1p5_0p3"]


def test_summarize_two_sided_pairs_aggregates_metrics() -> None:
    rows = [
        {
            "strategy_tag": "edge_1p2_0p2",
            "condition_id": "cond-1",
            "title": "Will Team A win?",
            "side": "BUY",
            "pnl": 0.0,
            "size_usd": 100.0,
            "signed_shares": 200.0,
            "edge_theoretical": 0.02,
            "edge_realized": 0.0,
        },
        {
            "strategy_tag": "edge_1p2_0p2",
            "condition_id": "cond-1",
            "title": "Will Team A win?",
            "side": "SELL",
            "pnl": 12.0,
            "size_usd": 48.0,
            "signed_shares": -80.0,
            "edge_theoretical": 0.01,
            "edge_realized": 0.15,
        },
    ]

    summary = summarize_two_sided_pairs(rows)
    assert summary.shape[0] == 1
    item = summary.iloc[0]
    assert item["strategy_tag"] == "edge_1p2_0p2"
    assert item["trades"] == 2
    assert item["sells"] == 1
    assert item["realized_pnl"] == pytest.approx(12.0, rel=1e-9)
    assert item["net_shares"] == pytest.approx(120.0, rel=1e-9)
    assert item["win_rate_sells"] == pytest.approx(1.0, rel=1e-9)
