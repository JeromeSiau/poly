"""Tests for helper logic in run_two_sided_inventory script."""

import asyncio
import time

import httpx
import pytest
from sqlalchemy import func, select

import scripts.run_two_sided_inventory as two_sided_runner
from scripts.run_two_sided_inventory import (
    ExternalFairRuntime,
    ResolvedCondition,
    TwoSidedPaperRecorder,
    _parse_resolved_binary_market,
    _best_orderbook_level,
    _resolve_probability,
    fetch_book,
    fetch_resolved_conditions,
    paper_merge_binary_pairs,
    select_intents_for_execution,
    settle_resolved_inventory,
)
from src.arb.two_sided_inventory import MarketSnapshot, OutcomeQuote, TradeIntent, TwoSidedInventoryEngine
from src.db.database import get_sync_session, reset_engines
from src.db.models import LiveObservation, PaperTrade
from src.feeds.odds_api import OddsApiSnapshot, OddsApiUsage


@pytest.fixture(autouse=True)
def _reset_global_db_engines():
    reset_engines()
    yield
    reset_engines()


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


def test_parse_resolved_binary_market_requires_closed_and_decisive() -> None:
    raw = {
        "conditionId": "cond-1",
        "question": "Will Team A win?",
        "closed": True,
        "endDate": "2026-02-08T12:00:00Z",
        "outcomes": '["Yes","No"]',
        "outcomePrices": '["0.999","0.001"]',
    }
    parsed = _parse_resolved_binary_market(
        raw,
        now_ts=0.0,
        winner_min_price=0.985,
        loser_max_price=0.015,
        allow_ended_open=True,
        enddate_grace_seconds=300.0,
    )
    assert parsed is not None
    assert parsed.condition_id == "cond-1"
    assert parsed.outcome_prices["Yes"] == pytest.approx(0.999, rel=1e-9)

    not_decisive = {
        **raw,
        "outcomePrices": '["0.61","0.39"]',
    }
    assert _parse_resolved_binary_market(
        not_decisive,
        now_ts=0.0,
        winner_min_price=0.985,
        loser_max_price=0.015,
        allow_ended_open=True,
        enddate_grace_seconds=300.0,
    ) is None

    not_closed = {**raw, "closed": False, "endDate": "2126-02-08T12:00:00Z"}
    assert _parse_resolved_binary_market(
        not_closed,
        now_ts=0.0,
        winner_min_price=0.985,
        loser_max_price=0.015,
        allow_ended_open=True,
        enddate_grace_seconds=300.0,
    ) is None


def test_parse_resolved_binary_market_accepts_ended_open_when_enabled() -> None:
    raw = {
        "conditionId": "cond-2",
        "question": "Will Team B win?",
        "closed": False,
        "endDate": "2026-02-08T12:00:00Z",
        "outcomes": '["Yes","No"]',
        "outcomePrices": '["0.0005","0.9995"]',
    }

    parsed = _parse_resolved_binary_market(
        raw,
        now_ts=1_770_854_400.0,  # 2026-02-08T13:20:00Z
        winner_min_price=0.985,
        loser_max_price=0.015,
        allow_ended_open=True,
        enddate_grace_seconds=300.0,
    )
    assert parsed is not None
    assert parsed.condition_id == "cond-2"

    blocked = _parse_resolved_binary_market(
        raw,
        now_ts=1_770_854_400.0,
        winner_min_price=0.985,
        loser_max_price=0.015,
        allow_ended_open=False,
        enddate_grace_seconds=300.0,
    )
    assert blocked is None


def test_select_intents_for_execution_keeps_pair_bundle_when_cap_is_one() -> None:
    now = time.time()
    intents = [
        TradeIntent(
            condition_id="cond-1",
            title="Will Team A win?",
            outcome="Yes",
            token_id="tok-yes",
            side="BUY",
            price=0.48,
            size_usd=48.0,
            edge_pct=0.02,
            reason="pair_arb_entry",
            timestamp=now,
        ),
        TradeIntent(
            condition_id="cond-1",
            title="Will Team A win?",
            outcome="No",
            token_id="tok-no",
            side="BUY",
            price=0.49,
            size_usd=49.0,
            edge_pct=0.02,
            reason="pair_arb_entry",
            timestamp=now,
        ),
        TradeIntent(
            condition_id="cond-2",
            title="Will Team B win?",
            outcome="Yes",
            token_id="tok-yes-2",
            side="BUY",
            price=0.40,
            size_usd=40.0,
            edge_pct=0.03,
            reason="under_fair",
            timestamp=now,
        ),
    ]

    selected = select_intents_for_execution(intents, max_orders_per_cycle=1)
    pair_legs = [i for i in selected if i.reason == "pair_arb_entry"]
    assert len(pair_legs) == 2
    assert {i.outcome for i in pair_legs} == {"Yes", "No"}


@pytest.mark.asyncio
async def test_fetch_resolved_conditions_uses_repeated_condition_ids_params() -> None:
    class _DummyResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return self._payload

    class _DummyClient:
        def __init__(self) -> None:
            self.calls = []

        async def get(self, url, params=None):
            self.calls.append((url, params))
            return _DummyResponse(
                [
                    {
                        "conditionId": "cond-a",
                        "question": "Will Team A win?",
                        "closed": True,
                        "endDate": "2026-02-08T12:00:00Z",
                        "outcomes": '["Yes","No"]',
                        "outcomePrices": '["0.999","0.001"]',
                    }
                ]
            )

    client = _DummyClient()
    resolved = await fetch_resolved_conditions(
        client=client,  # type: ignore[arg-type]
        condition_ids=["cond-a", "cond-b"],
        now_ts=1_770_854_400.0,
        winner_min_price=0.985,
        loser_max_price=0.015,
        allow_ended_open=True,
        enddate_grace_seconds=300.0,
        fetch_chunk_size=40,
    )

    assert "cond-a" in resolved
    assert len(client.calls) == 1
    _, params = client.calls[0]
    assert params == [("condition_ids", "cond-a"), ("condition_ids", "cond-b")]


@pytest.mark.asyncio
async def test_fetch_book_404_is_temporarily_suppressed(monkeypatch) -> None:
    class _Dummy404Response:
        def __init__(self, token: str) -> None:
            self._request = httpx.Request("GET", f"https://clob.polymarket.com/book?token_id={token}")
            self._response = httpx.Response(404, request=self._request)

        def raise_for_status(self) -> None:
            raise httpx.HTTPStatusError("404", request=self._request, response=self._response)

        def json(self):
            return {}

    class _DummyClient:
        def __init__(self) -> None:
            self.calls = 0

        async def get(self, url, params=None):
            self.calls += 1
            token = str((params or {}).get("token_id") or "unknown")
            return _Dummy404Response(token)

    token_id = "missing-token-123"
    now = [1_000.0]
    monkeypatch.setattr(two_sided_runner.time, "time", lambda: now[0])
    two_sided_runner._BOOK_404_SUPPRESS_UNTIL.clear()

    client = _DummyClient()
    semaphore = asyncio.Semaphore(1)

    first = await fetch_book(client, token_id, semaphore)  # triggers remote call + suppression
    second = await fetch_book(client, token_id, semaphore)  # should be skipped during suppression window

    assert client.calls == 1
    assert first["bid"] is None
    assert second["ask"] is None

    now[0] += two_sided_runner.DEFAULT_BOOK_404_COOLDOWN_SECONDS + 1.0
    await fetch_book(client, token_id, semaphore)
    assert client.calls == 2


def test_paper_merge_binary_pairs_persists_and_replays(tmp_path) -> None:
    db_path = tmp_path / "two_sided_merge_test.db"
    if db_path.exists():
        db_path.unlink()
    db_url = f"sqlite+aiosqlite:///{db_path}"
    recorder = TwoSidedPaperRecorder(
        db_url,
        strategy_tag="edge_merge_case",
        run_id="edge_merge_case-test",
        min_edge_pct=0.015,
        exit_edge_pct=0.003,
    )
    recorder.bootstrap()

    engine = TwoSidedInventoryEngine()
    snapshot = MarketSnapshot(
        condition_id="cond-merge",
        title="Will Team A win?",
        outcome_order=["Yes", "No"],
        timestamp=time.time(),
        outcomes={
            "Yes": OutcomeQuote(
                outcome="Yes",
                token_id="tok-yes",
                bid=0.51,
                ask=0.52,
                bid_size=500.0,
                ask_size=500.0,
            ),
            "No": OutcomeQuote(
                outcome="No",
                token_id="tok-no",
                bid=0.47,
                ask=0.48,
                bid_size=500.0,
                ask_size=500.0,
            ),
        },
        liquidity=3000.0,
        volume_24h=1500.0,
    )

    buy_yes = TradeIntent(
        condition_id="cond-merge",
        title="Will Team A win?",
        outcome="Yes",
        token_id="tok-yes",
        side="BUY",
        price=0.40,
        size_usd=40.0,  # 100 shares
        edge_pct=0.02,
        reason="under_fair",
        timestamp=time.time(),
    )
    buy_no = TradeIntent(
        condition_id="cond-merge",
        title="Will Team A win?",
        outcome="No",
        token_id="tok-no",
        side="BUY",
        price=0.45,
        size_usd=45.0,  # 100 shares
        edge_pct=0.02,
        reason="under_fair",
        timestamp=time.time(),
    )
    fill_yes = engine.apply_fill(buy_yes)
    fill_no = engine.apply_fill(buy_no)
    recorder.persist_fill(
        intent=buy_yes,
        fill=fill_yes,
        snapshot=snapshot,
        fair_prices={"Yes": 0.50, "No": 0.50},
        execution_mode="paper",
    )
    recorder.persist_fill(
        intent=buy_no,
        fill=fill_no,
        snapshot=snapshot,
        fair_prices={"Yes": 0.50, "No": 0.50},
        execution_mode="paper",
    )

    merged = paper_merge_binary_pairs(
        engine=engine,
        snapshots_by_condition={"cond-merge": snapshot},
        paper_recorder=recorder,
        now_ts=time.time() + 10.0,
        min_edge_pct=0.01,
        max_pair_notional_usd=1000.0,
    )
    assert merged == 1
    assert engine.get_state("cond-merge", "Yes").shares == pytest.approx(0.0, abs=1e-9)
    assert engine.get_state("cond-merge", "No").shares == pytest.approx(0.0, abs=1e-9)
    assert engine.get_realized_pnl() == pytest.approx(15.0, rel=1e-9)

    replay_engine = TwoSidedInventoryEngine()
    restored = recorder.replay_into_engine(replay_engine)
    assert restored == 4
    assert replay_engine.get_state("cond-merge", "Yes").shares == pytest.approx(0.0, abs=1e-9)
    assert replay_engine.get_state("cond-merge", "No").shares == pytest.approx(0.0, abs=1e-9)
    assert replay_engine.get_realized_pnl() == pytest.approx(15.0, rel=1e-9)


def test_paper_recorder_persists_and_replays_inventory(tmp_path) -> None:
    db_path = tmp_path / "two_sided_test.db"
    if db_path.exists():
        db_path.unlink()
    db_url = f"sqlite+aiosqlite:///{db_path}"
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

    session = get_sync_session("sqlite:///" + str(db_path))
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


def test_settle_resolved_inventory_persists_and_replays(tmp_path) -> None:
    db_path = tmp_path / "two_sided_settlement_test.db"
    if db_path.exists():
        db_path.unlink()
    db_url = f"sqlite+aiosqlite:///{db_path}"
    recorder = TwoSidedPaperRecorder(
        db_url,
        strategy_tag="edge_settlement_case",
        run_id="edge_settlement_case-test",
        min_edge_pct=0.015,
        exit_edge_pct=0.003,
    )
    recorder.bootstrap()

    engine = TwoSidedInventoryEngine()
    buy_intent = TradeIntent(
        condition_id="cond-resolved",
        title="Will Team A win?",
        outcome="Yes",
        token_id="tok-yes",
        side="BUY",
        price=0.60,
        size_usd=120.0,  # 200 shares
        edge_pct=0.02,
        reason="under_fair",
        timestamp=time.time(),
    )
    buy_fill = engine.apply_fill(buy_intent)
    recorder.persist_fill(
        intent=buy_intent,
        fill=buy_fill,
        snapshot=None,
        fair_prices={"Yes": 0.62},
        execution_mode="paper",
    )

    settled = settle_resolved_inventory(
        engine=engine,
        resolved_conditions={
            "cond-resolved": ResolvedCondition(
                condition_id="cond-resolved",
                title="Will Team A win?",
                outcome_prices={"Yes": 0.0, "No": 1.0},
            )
        },
        paper_recorder=recorder,
        now_ts=time.time() + 10.0,
    )
    assert settled == 1
    assert engine.get_state("cond-resolved", "Yes").shares == pytest.approx(0.0, abs=1e-9)
    assert engine.get_realized_pnl() == pytest.approx(-120.0, rel=1e-9)

    replay_engine = TwoSidedInventoryEngine()
    restored = recorder.replay_into_engine(replay_engine)
    assert restored == 2
    assert replay_engine.get_state("cond-resolved", "Yes").shares == pytest.approx(0.0, abs=1e-9)
    assert replay_engine.get_realized_pnl() == pytest.approx(-120.0, rel=1e-9)
