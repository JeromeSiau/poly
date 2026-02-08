import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.analysis.rn1_comparison import (
    ActivityEvent,
    build_rn1_transaction_report,
    build_rn1_vs_local_condition_report,
    build_gaps,
    build_recommendations,
    fetch_rn1_activity,
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


def test_fetch_rn1_activity_stops_gracefully_on_400(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Resp:
        def __init__(self, payload, status_code: int = 200):
            self._payload = payload
            self.status_code = status_code
            self.request = httpx.Request("GET", "https://data-api.polymarket.com/activity")

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise httpx.HTTPStatusError(
                    f"{self.status_code}",
                    request=self.request,
                    response=httpx.Response(self.status_code, request=self.request),
                )

        def json(self):
            return self._payload

    class _Client:
        def __init__(self, *args, **kwargs):
            self.calls = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url, params=None):
            self.calls += 1
            if self.calls == 1:
                return _Resp(
                    [
                        {
                            "timestamp": 1770570981,
                            "conditionId": "cond-1",
                            "type": "TRADE",
                            "usdcSize": 12.0,
                            "side": "BUY",
                            "outcome": "Yes",
                        }
                    ],
                    status_code=200,
                )
            return _Resp([], status_code=400)

    monkeypatch.setattr("src.analysis.rn1_comparison.httpx.Client", _Client)

    rows = fetch_rn1_activity(
        wallet="0xabc",
        window_hours=48.0,
        page_limit=500,
        max_pages=8,
    )
    assert len(rows) == 1
    assert rows[0].condition_id == "cond-1"


def test_fetch_rn1_activity_treats_missing_type_as_trade(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Resp:
        def __init__(self, payload, status_code: int = 200):
            self._payload = payload
            self.status_code = status_code
            self.request = httpx.Request("GET", "https://data-api.polymarket.com/activity")

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise httpx.HTTPStatusError(
                    f"{self.status_code}",
                    request=self.request,
                    response=httpx.Response(self.status_code, request=self.request),
                )

        def json(self):
            return self._payload

    class _Client:
        def __init__(self, *args, **kwargs):
            self.calls = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url, params=None):
            self.calls += 1
            if self.calls == 1:
                return _Resp(
                    [
                        {
                            "timestamp": 1770570981,
                            "conditionId": "cond-2",
                            # no `type` field on purpose
                            "side": "BUY",
                            "price": 0.4,
                            "usdcSize": 8.0,
                            "outcome": "Yes",
                            "title": "Will Team B win on 2026-02-08?",
                            "eventSlug": "epl-a-b-2026-02-08",
                            "transactionHash": "0xabc",
                        }
                    ],
                    status_code=200,
                )
            return _Resp([], status_code=200)

    monkeypatch.setattr("src.analysis.rn1_comparison.httpx.Client", _Client)

    rows = fetch_rn1_activity(
        wallet="0xabc",
        window_hours=48.0,
        page_limit=500,
        max_pages=2,
    )
    assert len(rows) == 1
    assert rows[0].reason == "trade"
    assert rows[0].side == "BUY"


def test_build_rn1_transaction_report_enriches_conditions(monkeypatch: pytest.MonkeyPatch) -> None:
    sample_rows = [
        {
            "timestamp": 1000,
            "datetime_utc": "2026-02-08T10:00:00+00:00",
            "type": "TRADE",
            "side": "BUY",
            "condition_id": "cond-a",
            "title": "Will Team A win on 2026-02-08?",
            "slug": "a-b",
            "event_slug": "epl-a-b-2026-02-08",
            "league_prefix": "epl",
            "market_type": "winner",
            "outcome": "Yes",
            "price": 0.45,
            "price_bucket": "0.40-0.60",
            "usdc_size": 9.0,
            "shares": 20.0,
            "tx_hash": "0xtx1",
        },
        {
            "timestamp": 1010,
            "datetime_utc": "2026-02-08T10:00:10+00:00",
            "type": "TRADE",
            "side": "BUY",
            "condition_id": "cond-a",
            "title": "Will Team A win on 2026-02-08?",
            "slug": "a-b",
            "event_slug": "epl-a-b-2026-02-08",
            "league_prefix": "epl",
            "market_type": "winner",
            "outcome": "No",
            "price": 0.48,
            "price_bucket": "0.40-0.60",
            "usdc_size": 9.6,
            "shares": 20.0,
            "tx_hash": "0xtx2",
        },
        {
            "timestamp": 1060,
            "datetime_utc": "2026-02-08T10:01:00+00:00",
            "type": "MERGE",
            "side": "MERGE",
            "condition_id": "cond-a",
            "title": "Will Team A win on 2026-02-08?",
            "slug": "a-b",
            "event_slug": "epl-a-b-2026-02-08",
            "league_prefix": "epl",
            "market_type": "winner",
            "outcome": "",
            "price": 0.0,
            "price_bucket": "n/a",
            "usdc_size": 20.0,
            "shares": 0.0,
            "tx_hash": "0xtx3",
        },
    ]

    monkeypatch.setattr(
        "src.analysis.rn1_comparison.fetch_rn1_raw_activity",
        lambda **kwargs: sample_rows,
    )

    report = build_rn1_transaction_report(
        window_hours=6.0,
        include_transactions=True,
        transaction_limit=100,
        top_conditions=10,
    )
    summary = report["summary"]
    assert summary["events_total"] == 3
    assert summary["trade_count"] == 2
    assert summary["merge_count"] == 1
    assert summary["multi_outcome_conditions"] == 1

    cond = report["conditions"][0]
    assert cond["condition_id"] == "cond-a"
    assert cond["is_multi_outcome"] is True
    assert cond["pair_cost_est"] == pytest.approx(0.93, rel=1e-9)
    assert cond["locked_edge_est"] == pytest.approx(0.07, rel=1e-9)

    tx_rows = report["transactions"]
    assert len(tx_rows) == 3
    assert tx_rows[-1]["type"] == "MERGE"
    assert tx_rows[-1]["seconds_since_last_trade_same_condition"] == 50


def test_build_rn1_vs_local_condition_report_computes_overlap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.analysis.rn1_comparison.build_rn1_transaction_report",
        lambda **kwargs: {
            "conditions": [
                {
                    "condition_id": "cond-a",
                    "title": "A vs B",
                    "event_slug": "epl-a-b-2026",
                    "league_prefix": "epl",
                    "market_type": "winner",
                    "events": 20,
                    "trade_count": 18,
                    "merge_count": 2,
                    "buy_usdc": 180.0,
                    "locked_edge_est": 0.05,
                    "locked_pnl_est": 9.0,
                },
                {
                    "condition_id": "cond-b",
                    "title": "C vs D",
                    "event_slug": "fl1-c-d-2026",
                    "league_prefix": "fl1",
                    "market_type": "draw",
                    "events": 30,
                    "trade_count": 30,
                    "merge_count": 0,
                    "buy_usdc": 300.0,
                    "locked_edge_est": 0.0,
                    "locked_pnl_est": 0.0,
                },
            ]
        },
    )
    monkeypatch.setattr(
        "src.analysis.rn1_comparison.aggregate_local_conditions",
        lambda **kwargs: [
            {
                "condition_id": "cond-a",
                "events": 10,
                "buy_count": 6,
                "sell_count": 4,
                "buy_usdc": 60.0,
                "sell_usdc": 40.0,
                "total_usdc": 100.0,
                "reasons": {"under_fair": 6, "over_fair": 4},
                "strategy_tags": {"edge_1p5_0p3": 10},
            },
            {
                "condition_id": "cond-local-only",
                "events": 5,
                "buy_count": 3,
                "sell_count": 2,
                "buy_usdc": 20.0,
                "sell_usdc": 10.0,
                "total_usdc": 30.0,
                "reasons": {"under_fair": 3},
                "strategy_tags": {"edge_1p5_0p3": 5},
            },
        ],
    )

    report = build_rn1_vs_local_condition_report(
        db_url="data/arb.db",
        window_hours=6.0,
        strategy_tag="edge_1p5_0p3",
        top_conditions=20,
    )
    summary = report["summary"]
    assert summary["rn1_conditions"] == 2
    assert summary["local_conditions"] == 2
    assert summary["overlap_conditions"] == 1
    assert summary["rn1_only_conditions"] == 1
    assert summary["local_only_conditions"] == 1
    assert summary["overlap_ratio_vs_rn1"] == pytest.approx(0.5, rel=1e-9)

    overlap = report["overlap_top"][0]
    assert overlap["condition_id"] == "cond-a"
    assert overlap["activity_ratio_local_vs_rn1"] == pytest.approx(0.5, rel=1e-9)
    assert overlap["buy_usdc_ratio_local_vs_rn1"] == pytest.approx(60.0 / 180.0, rel=1e-9)

    rn1_only = report["rn1_only_top"][0]
    assert rn1_only["condition_id"] == "cond-b"
