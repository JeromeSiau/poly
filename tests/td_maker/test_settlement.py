import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.td_maker.state import MarketState, MarketRegistry, OpenPosition
from src.td_maker.settlement import SettlementManager


def make_market_expired():
    m = MarketState(
        condition_id="c1", slug="btc-up-15m-1000", symbol="btc/usd",
        slot_ts=int(time.time()) - 1500,  # expired
        token_ids={"Up": "tok_up"},
        ref_price=95000.0, chainlink_symbol="btc/usd",
    )
    m.position = OpenPosition(
        condition_id="c1", outcome="Up", token_id="tok_up",
        entry_price=0.80, size_usd=10.0, shares=12.5,
        filled_at=time.time() - 900,
    )
    m.last_bids["Up"] = 0.95
    return m


def make_sm():
    return SettlementManager(
        registry=MarketRegistry(),
        executor=AsyncMock(),
        trade_manager=AsyncMock(),
        shadow=MagicMock(),
        guard=MagicMock(),
        db=MagicMock(),
        config=MagicMock(strategy_tag="test", paper_mode=False),
        order_mgr=AsyncMock(),
    )


def test_force_from_last_bid_win():
    sm = make_sm()
    m = make_market_expired()
    m.last_bids["Up"] = 0.97
    result = sm._force_from_last_bid(m)
    assert result == "win"


def test_force_from_last_bid_loss():
    sm = make_sm()
    m = make_market_expired()
    m.last_bids["Up"] = 0.03
    result = sm._force_from_last_bid(m)
    assert result == "loss"


def test_force_from_last_bid_ambiguous_is_conservative():
    sm = make_sm()
    m = make_market_expired()
    m.last_bids["Up"] = 0.50  # ambiguous
    result = sm._force_from_last_bid(m)
    assert result == "loss"  # conservative


@pytest.mark.asyncio
async def test_settle_defers_when_no_resolution():
    sm = make_sm()
    m = make_market_expired()
    sm.registry.register(m)

    with patch.object(sm, "_query_resolution", new=AsyncMock(return_value=None)):
        await sm._settle(m)

    assert m.awaiting_settlement is True
    assert m.settlement_deferred_until is not None
    assert sm.registry.get("c1") is not None  # not removed yet


@pytest.mark.asyncio
async def test_settle_resolves_win():
    sm = make_sm()
    m = make_market_expired()
    sm.registry.register(m)

    with patch.object(sm, "_query_resolution", new=AsyncMock(return_value="win")):
        await sm._settle(m)

    sm.trade_manager.record_settle_direct.assert_called_once()
    assert sm.registry.get("c1") is None  # removed after settlement
