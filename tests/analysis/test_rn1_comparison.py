import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.analysis.rn1_comparison import (
    ActivityEvent,
    build_gaps,
    build_recommendations,
    summarize_behavior,
)


def test_summarize_behavior_computes_core_metrics() -> None:
    events = [
        ActivityEvent(
            timestamp=1000,
            condition_id="cond-1",
            outcome="Yes",
            side="BUY",
            size_usd=10.0,
            reason="under_fair",
            strategy_tag="edge_a",
        ),
        ActivityEvent(
            timestamp=1020,
            condition_id="cond-1",
            outcome="No",
            side="BUY",
            size_usd=14.0,
            reason="pair_arb_entry",
            strategy_tag="edge_a",
        ),
        ActivityEvent(
            timestamp=1030,
            condition_id="cond-1",
            outcome="Yes",
            side="SELL",
            size_usd=12.0,
            reason="over_fair",
            strategy_tag="edge_a",
        ),
        ActivityEvent(
            timestamp=1060,
            condition_id="cond-2",
            outcome="Yes",
            side="MERGE",
            size_usd=20.0,
            reason="merge",
            strategy_tag="RN1",
        ),
    ]

    summary = summarize_behavior(events, window_hours=1.0)
    assert summary["events_total"] == 4
    assert summary["events_trade"] == 3
    assert summary["events_merge"] == 1
    assert summary["buy_count"] == 2
    assert summary["sell_count"] == 1
    assert summary["buy_share_of_trades"] == 2 / 3
    assert summary["merge_share_of_events"] == 0.25
    assert summary["unique_conditions"] == 2
    assert summary["multi_outcome_conditions"] == 1
    assert summary["multi_outcome_ratio"] == 0.5
    assert summary["size_median_usd"] == 13.0
    assert summary["max_events_per_minute"] >= 3


def test_build_gaps_and_recommendations_flag_main_gaps() -> None:
    local = {
        "cadence_per_minute_window": 0.02,
        "cadence_per_minute_active": 0.05,
        "size_median_usd": 50.0,
        "size_mean_usd": 60.0,
        "multi_outcome_ratio": 0.20,
        "buy_share_of_trades": 0.90,
        "merge_share_of_events": 0.00,
        "unique_conditions": 8,
    }
    rn1 = {
        "cadence_per_minute_window": 0.20,
        "cadence_per_minute_active": 0.80,
        "size_median_usd": 10.0,
        "size_mean_usd": 15.0,
        "multi_outcome_ratio": 0.60,
        "buy_share_of_trades": 0.55,
        "merge_share_of_events": 0.10,
        "unique_conditions": 40,
    }

    gaps = build_gaps(local, rn1)
    assert gaps["cadence_window_ratio_local_vs_rn1"] == pytest.approx(0.1, rel=1e-9)
    assert gaps["multi_outcome_ratio_gap"] == pytest.approx(-0.4, rel=1e-9)
    assert gaps["buy_share_gap"] == pytest.approx(0.35, rel=1e-9)
    assert gaps["size_median_ratio_local_vs_rn1"] == pytest.approx(5.0, rel=1e-9)

    recos = build_recommendations(local, rn1, gaps)
    joined = " ".join(recos).lower()
    assert "cadence" in joined
    assert "deux-cotes" in joined
    assert "sorties" in joined
    assert "ticket" in joined
