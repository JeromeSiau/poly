from __future__ import annotations
import time
import pytest
from collections import deque
from src.td_maker.state import MarketState, PassiveOrder, OpenPosition, MarketRegistry


def make_market(cid="cid1") -> MarketState:
    return MarketState(
        condition_id=cid,
        slug=f"btc-up-15m-1000",
        symbol="btc/usd",
        slot_ts=1000,
        token_ids={"Up": "tok_up", "Down": "tok_dn"},
        ref_price=95000.0,
        chainlink_symbol="btc/usd",
    )


def make_order(order_id="o1", outcome="Up", price=0.80, cid="cid1") -> PassiveOrder:
    return PassiveOrder(
        order_id=order_id, condition_id=cid, outcome=outcome,
        token_id="tok_up", price=price, size_usd=10.0, placed_at=time.time()
    )


# --- MarketState tests ---

def test_add_order_ok():
    m = make_market()
    o = make_order()
    assert m.add_order(o) is True
    assert "o1" in m.active_orders


def test_add_order_duplicate_id_rejected():
    m = make_market()
    o = make_order()
    m.add_order(o)
    assert m.add_order(o) is False


def test_is_placeholder():
    assert MarketState.is_placeholder("_placing_abc123") is True
    assert MarketState.is_placeholder("real-order-id") is False


def test_replace_order_id():
    m = make_market()
    o = make_order(order_id="_placing_abc")
    m.add_order(o)
    m.replace_order_id("_placing_abc", "real-id-123")
    assert "real-id-123" in m.active_orders
    assert "_placing_abc" not in m.active_orders


def test_move_to_pending_cancel():
    m = make_market()
    o = make_order()
    m.add_order(o)
    result = m.move_to_pending_cancel("o1")
    assert result is not None
    assert "o1" not in m.active_orders
    assert "o1" in m.pending_cancels


def test_move_to_pending_cancel_unknown_returns_none():
    m = make_market()
    assert m.move_to_pending_cancel("nonexistent") is None


def test_record_fill_creates_position():
    m = make_market()
    o = make_order(order_id="o1", price=0.80)
    m.add_order(o)
    ok = m.record_fill("o1", shares=12.5)
    assert ok is True
    assert m.position is not None
    assert m.fill_count == 1
    assert abs(m.position.shares - 12.5) < 0.001


def test_record_fill_scale_in():
    m = make_market()
    o1 = make_order(order_id="o1", price=0.80)
    o2 = make_order(order_id="o2", price=0.78)
    m.add_order(o1)
    m.add_order(o2)
    m.record_fill("o1", shares=12.5)
    m.record_fill("o2", shares=12.82)
    assert m.fill_count == 2
    assert m.position.shares > 12.5


def test_record_fill_unknown_order_rejected():
    m = make_market()
    assert m.record_fill("ghost", shares=10.0) is False


# --- MarketRegistry tests ---

def test_registry_register_and_get():
    reg = MarketRegistry()
    m = make_market("c1")
    reg.register(m)
    assert reg.get("c1") is m


def test_registry_remove():
    reg = MarketRegistry()
    reg.register(make_market("c1"))
    reg.remove("c1")
    assert reg.get("c1") is None


def test_registry_markets_with_positions():
    reg = MarketRegistry()
    m1 = make_market("c1")
    m2 = make_market("c2")
    reg.register(m1)
    reg.register(m2)
    o = make_order(cid="c1")
    m1.add_order(o)
    m1.record_fill("o1", shares=12.5)
    assert m1 in reg.markets_with_positions()
    assert m2 not in reg.markets_with_positions()


def test_registry_expired_markets():
    reg = MarketRegistry()
    m = make_market()
    m.slot_ts = int(time.time()) - 1500  # 15m + 5m grace ago
    reg.register(m)
    expired = reg.expired_markets(time.time())
    assert m in expired


def test_registry_total_exposure():
    reg = MarketRegistry()
    m1 = make_market("c1")
    m2 = make_market("c2")
    reg.register(m1)
    reg.register(m2)
    o1 = make_order(order_id="o1", cid="c1")
    m1.add_order(o1)
    m1.record_fill("o1", shares=12.5)
    assert reg.total_exposure() == pytest.approx(10.0)
