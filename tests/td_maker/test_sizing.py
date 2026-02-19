import pytest
from unittest.mock import MagicMock
from src.td_maker.sizing import compute_rung_prices, Sizing
from src.td_maker.state import MarketRegistry, MarketState, PassiveOrder, OpenPosition
import time


def test_compute_rung_prices_single():
    prices = compute_rung_prices(0.75, 0.85, 1)
    assert len(prices) == 1
    assert prices[0] == pytest.approx(0.80, abs=0.01)


def test_compute_rung_prices_three():
    prices = compute_rung_prices(0.75, 0.85, 3)
    assert len(prices) == 3
    assert prices[0] == pytest.approx(0.75, abs=0.01)
    assert prices[-1] == pytest.approx(0.85, abs=0.01)


def test_compute_rung_prices_dedup():
    # Very narrow range — rungs would snap to same cent
    prices = compute_rung_prices(0.80, 0.81, 5)
    assert len(prices) == len(set(prices))


def test_sizing_available_budget():
    config = MagicMock()
    config.max_exposure = 100.0
    config.target_bid = 0.75
    config.max_bid = 0.85
    config.ladder_rungs = 1
    reg = MarketRegistry()
    sizing = Sizing(config)
    assert sizing.available_budget(reg) == pytest.approx(100.0)


def test_sizing_build_order_returns_none_when_no_budget():
    config = MagicMock()
    config.target_bid = 0.75
    config.max_bid = 0.85
    config.ladder_rungs = 1
    config.order_size = 10.0
    config.max_exposure = 5.0

    m = MarketState(condition_id="c1", slug="s", symbol="btc/usd", slot_ts=1,
                    token_ids={"Up": "t"}, ref_price=1.0, chainlink_symbol="btc/usd")
    reg = MarketRegistry()
    reg.register(m)

    sizing = Sizing(config)

    class FakeResult:
        action = "maker"
        price = 0.80
        outcome = "Up"

    order = sizing.build_order(m, FakeResult(), budget=3.0)
    assert order is None  # budget < order_size
