import time
import pytest
from unittest.mock import AsyncMock, MagicMock
from src.td_maker.state import MarketState, MarketRegistry, PassiveOrder
from src.td_maker.fill_detector import FillDetector


def make_market():
    m = MarketState(
        condition_id="c1", slug="btc-up-15m-1000", symbol="btc/usd",
        slot_ts=int(time.time()) - 300,
        token_ids={"Up": "tok_up"},
        ref_price=95000.0, chainlink_symbol="btc/usd",
    )
    return m


def make_order(order_id="o1"):
    return PassiveOrder(
        order_id=order_id, condition_id="c1", outcome="Up",
        token_id="tok_up", price=0.80, size_usd=10.0, placed_at=time.time()
    )


def make_fd(registry=None):
    if registry is None:
        registry = MarketRegistry()
    return FillDetector(
        registry=registry,
        order_mgr=AsyncMock(),
        poly_feed=MagicMock(),
        user_feed=MagicMock(),
        trade_manager=AsyncMock(),
        shadow=MagicMock(),
        db=MagicMock(),
        config=MagicMock(paper_mode=True),
        executor=AsyncMock(),
    )


def test_dedup_prevents_double_process():
    fd = make_fd()
    key = fd._fill_key("c1", "o1", "t1")
    fd._processed_fills[key] = time.time()
    assert fd._is_duplicate(key) is True


def test_purge_removes_old_entries():
    fd = make_fd()
    old_key = fd._fill_key("c1", "o1", "t1")
    fd._processed_fills[old_key] = time.time() - 2000  # older than 30 min
    fresh_key = fd._fill_key("c1", "o2", "t2")
    fd._processed_fills[fresh_key] = time.time()
    fd._purge_old_fills()
    assert old_key not in fd._processed_fills
    assert fresh_key in fd._processed_fills


def test_paper_fill_ask_crossed():
    fd = make_fd()
    m = make_market()
    o = make_order()
    m.add_order(o)
    # ask is 0.79 (below our bid of 0.80) → fill condition met
    fd.poly_feed.get_best_levels.return_value = (0.79, 100, 0.79, 50)
    assert fd._paper_fill_triggered(m, o) is True


def test_paper_fill_not_triggered_when_ask_above():
    fd = make_fd()
    m = make_market()
    o = make_order()
    m.add_order(o)
    # ask is 0.82 (above our bid of 0.80) → no fill
    fd.poly_feed.get_best_levels.return_value = (0.80, 100, 0.82, 50)
    assert fd._paper_fill_triggered(m, o) is False


def test_match_order_exact_id():
    fd = make_fd()
    m = make_market()
    o = make_order("real-id-123")
    m.add_order(o)

    class Msg:
        order_id = "real-id-123"
        condition_id = "c1"
        outcome = "Up"
        maker_order_id = "real-id-123"

    result = fd._match_order(m, Msg())
    assert result is o
