import time
import pytest
from unittest.mock import MagicMock
from src.td_maker.filters import EntryFilters, FilterResult
from src.td_maker.state import MarketState


def make_config(**kwargs):
    cfg = MagicMock()
    cfg.target_bid = 0.75
    cfg.max_bid = 0.85
    cfg.min_move_pct = 0.0
    cfg.max_move_pct = 0.0
    cfg.min_entry_minutes = 0.0
    cfg.max_entry_minutes = 0.0
    cfg.entry_fair_margin = 0.0
    cfg.hybrid_skip_below = 0.55
    cfg.hybrid_taker_above = 0.72
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


def make_market(slot_ts=None):
    return MarketState(
        condition_id="c1", slug="btc-up-15m-1000", symbol="btc/usd",
        slot_ts=slot_ts or (int(time.time()) - 300),
        token_ids={"Up": "t1", "Down": "t2"},
        ref_price=95000.0, chainlink_symbol="btc/usd",
    )


def test_bid_below_range_skips():
    chainlink = MagicMock()
    f = EntryFilters(chainlink, make_config())
    result = f.should_bid(make_market(), outcome="Up", bid=0.70, ask=0.71)
    assert result.is_skip
    assert result.reason == "bid_out_of_range"


def test_bid_above_range_skips():
    chainlink = MagicMock()
    f = EntryFilters(chainlink, make_config())
    result = f.should_bid(make_market(), outcome="Up", bid=0.90, ask=0.91)
    assert result.is_skip
    assert result.reason == "bid_out_of_range"


def test_bid_in_range_no_filters_returns_maker():
    chainlink = MagicMock()
    f = EntryFilters(chainlink, make_config())
    result = f.should_bid(make_market(), outcome="Up", bid=0.80, ask=0.81)
    assert result.action == "maker"
    assert result.price == pytest.approx(0.80)


def test_max_entry_time_blocks_late():
    chainlink = MagicMock()
    # slot started 14 minutes ago, max_entry_minutes=10
    market = make_market(slot_ts=int(time.time()) - 840)
    f = EntryFilters(chainlink, make_config(max_entry_minutes=10.0))
    result = f.should_bid(market, outcome="Up", bid=0.80, ask=0.81)
    assert result.is_skip
    assert result.reason == "time_gate"


def test_min_entry_time_blocks_early():
    chainlink = MagicMock()
    # slot started 1 minute ago, min_entry_minutes=3
    market = make_market(slot_ts=int(time.time()) - 60)
    f = EntryFilters(chainlink, make_config(min_entry_minutes=3.0))
    result = f.should_bid(market, outcome="Up", bid=0.80, ask=0.81)
    assert result.is_skip
    assert result.reason == "time_gate"


def test_filter_result_is_skip_property():
    r = FilterResult(action="skip", reason="test", price=0.0, outcome="Up")
    assert r.is_skip is True
    r2 = FilterResult(action="maker", reason="", price=0.80, outcome="Up")
    assert r2.is_skip is False
