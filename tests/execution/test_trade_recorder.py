"""Tests for TradeRecorder — generic DB persistence layer."""

import pytest

from src.execution.trade_recorder import TradeRecorder
from src.execution.models import TradeIntent, FillResult


@pytest.fixture
def recorder(tmp_path):
    """Create a TradeRecorder with a fresh test DB, resetting globals after."""
    from src.db.database import reset_engines

    # Reset before to ensure clean state
    reset_engines()

    db_url = f"sqlite:///{tmp_path}/test.db"
    rec = TradeRecorder(
        db_url=db_url,
        strategy_tag="test_strat",
        event_type="test_event",
        run_id="run_001",
    )
    rec.bootstrap()
    yield rec

    # Reset after to avoid leaking into other tests
    reset_engines()


@pytest.fixture
def buy_intent():
    return TradeIntent(
        condition_id="cid_1",
        token_id="tok_1",
        outcome="Up",
        side="BUY",
        price=0.80,
        size_usd=10.0,
        reason="test_entry",
        title="BTC test",
        edge_pct=0.02,
        timestamp=1700000000.0,
    )


@pytest.fixture
def sell_intent():
    return TradeIntent(
        condition_id="cid_1",
        token_id="tok_1",
        outcome="Up",
        side="SELL",
        price=1.0,
        size_usd=12.5,
        reason="settlement",
        title="BTC test",
        edge_pct=0.0,
        timestamp=1700001000.0,
    )


def test_record_fill_creates_observation_and_trade(recorder, buy_intent):
    fill = FillResult(filled=True, shares=12.5, avg_price=0.80)
    obs_id = recorder.record_fill(
        intent=buy_intent,
        fill=fill,
        fair_prices={"Up": 0.82},
        execution_mode="paper",
    )
    assert obs_id > 0


def test_record_fill_skips_zero_shares(recorder, buy_intent):
    fill = FillResult(filled=True, shares=0.0, avg_price=0.80)
    obs_id = recorder.record_fill(
        intent=buy_intent,
        fill=fill,
        fair_prices={"Up": 0.82},
        execution_mode="paper",
    )
    assert obs_id == 0


def test_record_settle_closes_buy_records(recorder, buy_intent, sell_intent):
    fill_buy = FillResult(filled=True, shares=12.5, avg_price=0.80)
    recorder.record_fill(
        intent=buy_intent,
        fill=fill_buy,
        fair_prices={"Up": 0.82},
        execution_mode="paper",
    )

    fill_sell = FillResult(filled=True, shares=12.5, avg_price=1.0, pnl_delta=2.50)
    obs_id = recorder.record_settle(
        intent=sell_intent,
        fill=fill_sell,
        fair_prices={"Up": 1.0},
    )
    assert obs_id > 0

    # Verify the BUY record was closed
    from src.db.models import PaperTrade
    from src.db.database import get_sync_session

    session = get_sync_session(recorder._db_url)
    try:
        open_buys = (
            session.query(PaperTrade)
            .filter_by(side="BUY", is_open=True)
            .count()
        )
        assert open_buys == 0
    finally:
        session.close()


def test_game_state_contains_strategy_info(recorder, buy_intent):
    fill = FillResult(filled=True, shares=12.5, avg_price=0.80)
    obs_id = recorder.record_fill(
        intent=buy_intent,
        fill=fill,
        fair_prices={"Up": 0.82},
        execution_mode="paper",
    )

    from src.db.models import LiveObservation
    from src.db.database import get_sync_session

    session = get_sync_session(recorder._db_url)
    try:
        obs = session.query(LiveObservation).get(obs_id)
        gs = obs.game_state
        assert gs["strategy_tag"] == "test_strat"
        assert gs["run_id"] == "run_001"
        assert gs["condition_id"] == "cid_1"
        assert gs["outcome"] == "Up"
        assert gs["side"] == "BUY"
        assert gs["mode"] == "paper"
    finally:
        session.close()
