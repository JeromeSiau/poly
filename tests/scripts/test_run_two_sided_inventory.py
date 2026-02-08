"""Tests for helper logic in run_two_sided_inventory script."""

import time

import httpx
import pytest
from sqlalchemy import func, select

from scripts.run_two_sided_inventory import (
    ExternalFairRuntime,
    TwoSidedPaperRecorder,
    _best_orderbook_level,
    _resolve_probability,
)
from src.arb.two_sided_inventory import MarketSnapshot, OutcomeQuote, TradeIntent, TwoSidedInventoryEngine
from src.db.database import get_sync_session
from src.db.models import LiveObservation, PaperTrade
from src.feeds.odds_api import OddsApiSnapshot, OddsApiUsage


def test_best_orderbook_level_uses_real_top_of_book() -> None:
    bids = [
        {"price": "0.001", "size": "13629.85"},
        {"price": "0.02", "size": "75"},
        {"price": "0.032", "size": "29.93"},
    ]
    asks = [
        {"price": "0.999", "size": "2674.44"},
        {"price": "0.048", "size": "21.82"},
        {"price": "0.04", "size": "7.81"},
    ]

    bid_px, bid_sz = _best_orderbook_level(bids, side="bid")
    ask_px, ask_sz = _best_orderbook_level(asks, side="ask")

    assert bid_px == pytest.approx(0.032, abs=1e-9)
    assert bid_sz == pytest.approx(29.93, abs=1e-9)
    assert ask_px == pytest.approx(0.04, abs=1e-9)
    assert ask_sz == pytest.approx(7.81, abs=1e-9)


def test_best_orderbook_level_ignores_invalid_rows() -> None:
    mixed = [
        {"price": "-1", "size": "10"},
        {"price": "0.5", "size": "0"},
        [0.61, 14],
        {"price": "0.62", "size": "20"},
        {"oops": "bad"},
    ]

    bid_px, bid_sz = _best_orderbook_level(mixed, side="bid")
    ask_px, ask_sz = _best_orderbook_level(mixed, side="ask")

    assert bid_px == pytest.approx(0.62, abs=1e-9)
    assert bid_sz == pytest.approx(20.0, abs=1e-9)
    assert ask_px == pytest.approx(0.61, abs=1e-9)
    assert ask_sz == pytest.approx(14.0, abs=1e-9)


def test_resolve_probability_supports_inverse_mapping() -> None:
    odds = {"home": 0.61, "away": 0.39}
    assert _resolve_probability(odds, "home") == pytest.approx(0.61, abs=1e-9)
    assert _resolve_probability(odds, "1-home") == pytest.approx(0.39, abs=1e-9)


def test_paper_recorder_persists_and_replays_inventory(tmp_path) -> None:
    db_url = f"sqlite+aiosqlite:///{tmp_path / 'two_sided_test.db'}"
    recorder = TwoSidedPaperRecorder(
        db_url,
        strategy_tag="edge_1p5_0p3",
        run_id="edge_1p5_0p3-test",
        min_edge_pct=0.015,
        exit_edge_pct=0.003,
    )
    recorder.bootstrap()

    engine = TwoSidedInventoryEngine()
    snapshot = MarketSnapshot(
        condition_id="cond-xyz",
        title="Will Team A win?",
        outcome_order=["Yes", "No"],
        timestamp=time.time(),
        outcomes={
            "Yes": OutcomeQuote(
                outcome="Yes",
                token_id="tok-yes",
                bid=0.59,
                ask=0.60,
                bid_size=500.0,
                ask_size=500.0,
            ),
            "No": OutcomeQuote(
                outcome="No",
                token_id="tok-no",
                bid=0.40,
                ask=0.41,
                bid_size=500.0,
                ask_size=500.0,
            ),
        },
        liquidity=3000.0,
        volume_24h=1500.0,
    )

    buy_intent = TradeIntent(
        condition_id="cond-xyz",
        title="Will Team A win?",
        outcome="Yes",
        token_id="tok-yes",
        side="BUY",
        price=0.50,
        size_usd=100.0,
        edge_pct=0.02,
        reason="under_fair",
        timestamp=time.time(),
    )
    buy_fill = engine.apply_fill(buy_intent)
    recorder.persist_fill(
        intent=buy_intent,
        fill=buy_fill,
        snapshot=snapshot,
        fair_prices={"Yes": 0.53, "No": 0.47},
        execution_mode="paper",
    )

    sell_intent = TradeIntent(
        condition_id="cond-xyz",
        title="Will Team A win?",
        outcome="Yes",
        token_id="tok-yes",
        side="SELL",
        price=0.60,
        size_usd=60.0,
        edge_pct=0.01,
        reason="over_fair",
        timestamp=time.time() + 10.0,
    )
    sell_fill = engine.apply_fill(sell_intent)
    recorder.persist_fill(
        intent=sell_intent,
        fill=sell_fill,
        snapshot=snapshot,
        fair_prices={"Yes": 0.53, "No": 0.47},
        execution_mode="paper",
    )

    session = get_sync_session("sqlite:///" + str(tmp_path / "two_sided_test.db"))
    try:
        obs_count = session.execute(select(func.count()).select_from(LiveObservation)).scalar_one()
        trades = session.execute(select(PaperTrade).order_by(PaperTrade.id.asc())).scalars().all()
    finally:
        session.close()

    assert obs_count == 2
    assert len(trades) == 2
    assert trades[0].pnl is None
    assert trades[1].pnl == pytest.approx(sell_fill.realized_pnl_delta, rel=1e-9)
    assert trades[1].edge_realized == pytest.approx(sell_fill.realized_pnl_delta / sell_intent.size_usd, rel=1e-9)

    replay_engine = TwoSidedInventoryEngine()
    restored = recorder.replay_into_engine(replay_engine)

    assert restored == 2
    state = replay_engine.get_state("cond-xyz", "Yes")
    assert state.shares == pytest.approx(100.0, rel=1e-9)
    assert replay_engine.get_realized_pnl() == pytest.approx(sell_fill.realized_pnl_delta, rel=1e-9)


@pytest.mark.asyncio
async def test_external_fair_shared_db_cache_avoids_duplicate_api_calls(tmp_path) -> None:
    db_url = f"sqlite:///{tmp_path / 'shared_odds_cache.db'}"
    call_count = {"n": 0}

    async def _fake_fetch_events(**kwargs):
        call_count["n"] += 1
        return OddsApiSnapshot(
            events=[],
            usage=OddsApiUsage(remaining=499, used=1, last=1),
        )

    runtime_a = ExternalFairRuntime(
        api_key="test-key",
        base_url="https://api.the-odds-api.com/v4",
        sports=["upcoming"],
        regions="eu",
        markets="h2h",
        min_refresh_seconds=30.0,
        min_match_confidence=0.7,
        blend=1.0,
        shared_cache_db_url=db_url,
        shared_cache_ttl_seconds=3600.0,
    )
    runtime_a._odds.fetch_events = _fake_fetch_events  # type: ignore[method-assign]

    runtime_b = ExternalFairRuntime(
        api_key="test-key",
        base_url="https://api.the-odds-api.com/v4",
        sports=["upcoming"],
        regions="eu",
        markets="h2h",
        min_refresh_seconds=30.0,
        min_match_confidence=0.7,
        blend=1.0,
        shared_cache_db_url=db_url,
        shared_cache_ttl_seconds=3600.0,
    )
    runtime_b._odds.fetch_events = _fake_fetch_events  # type: ignore[method-assign]

    async with httpx.AsyncClient() as client:
        await runtime_a.refresh_if_needed(client=client, raw_markets=[], now_ts=1000.0, force=True)
        await runtime_b.refresh_if_needed(client=client, raw_markets=[], now_ts=1010.0, force=True)

    assert call_count["n"] == 1
    assert runtime_b.stats.credits_last_call == 1
