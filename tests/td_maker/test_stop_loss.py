import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.td_maker.state import MarketState, MarketRegistry, OpenPosition
from src.td_maker.stop_loss import StopLossManager


def make_config(**kwargs):
    cfg = MagicMock()
    cfg.stoploss_peak = 0.90
    cfg.stoploss_exit = 0.82
    cfg.stoploss_fair_margin = 0.10
    cfg.exit_threshold = 0.35
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


def make_market_with_position(bid_max=0.0) -> MarketState:
    m = MarketState(
        condition_id="c1", slug="btc-up-15m-1000", symbol="btc/usd",
        slot_ts=int(time.time()) - 300,
        token_ids={"Up": "tok_up"},
        ref_price=95000.0, chainlink_symbol="btc/usd",
    )
    m.position = OpenPosition(
        condition_id="c1", outcome="Up", token_id="tok_up",
        entry_price=0.80, size_usd=10.0, shares=12.5,
        filled_at=time.time() - 60,
    )
    m.bid_max = bid_max
    return m


def test_no_trigger_when_below_peak():
    cfg = make_config(stoploss_peak=0.90, stoploss_exit=0.82)
    sl = StopLossManager(
        registry=MagicMock(), order_mgr=MagicMock(), executor=AsyncMock(),
        chainlink_feed=MagicMock(), trade_manager=MagicMock(),
        shadow=MagicMock(), db=MagicMock(), config=cfg)
    m = make_market_with_position(bid_max=0.85)  # hasn't reached peak
    assert sl._check_rule_based(m, current_bid=0.78) is False


def test_triggers_after_peak_and_crash():
    cfg = make_config(stoploss_peak=0.90, stoploss_exit=0.82)
    chainlink = MagicMock()
    chainlink.get_price.return_value = None
    sl = StopLossManager(
        registry=MagicMock(), order_mgr=MagicMock(), executor=AsyncMock(),
        chainlink_feed=chainlink, trade_manager=MagicMock(),
        shadow=MagicMock(), db=MagicMock(), config=cfg)
    m = make_market_with_position(bid_max=0.92)  # hit peak
    assert sl._check_rule_based(m, current_bid=0.78) is True  # crashed below exit


def test_no_trigger_when_bid_above_exit():
    cfg = make_config(stoploss_peak=0.90, stoploss_exit=0.82)
    chainlink = MagicMock()
    chainlink.get_price.return_value = None
    sl = StopLossManager(
        registry=MagicMock(), order_mgr=MagicMock(), executor=AsyncMock(),
        chainlink_feed=chainlink, trade_manager=MagicMock(),
        shadow=MagicMock(), db=MagicMock(), config=cfg)
    m = make_market_with_position(bid_max=0.92)  # hit peak
    assert sl._check_rule_based(m, current_bid=0.85) is False  # still above exit


def test_empty_book_sets_awaiting_settlement():
    cfg = make_config()
    sl = StopLossManager(
        registry=MagicMock(), order_mgr=MagicMock(), executor=AsyncMock(),
        chainlink_feed=MagicMock(), trade_manager=MagicMock(),
        shadow=MagicMock(), db=MagicMock(), config=cfg)
    m = make_market_with_position()
    sl._handle_empty_book(m)
    assert m.awaiting_settlement is True


@pytest.mark.asyncio
async def test_execute_aborts_if_bid_recovered():
    cfg = make_config(stoploss_exit=0.82)
    poly_feed = MagicMock()
    poly_feed.get_best_levels.return_value = (0.88, 100.0, 0.89, 50.0)
    sl = StopLossManager(
        registry=MagicMock(), order_mgr=MagicMock(), executor=AsyncMock(),
        chainlink_feed=MagicMock(), trade_manager=AsyncMock(),
        shadow=MagicMock(), db=MagicMock(), config=cfg,
        poly_feed=poly_feed)
    m = make_market_with_position()
    await sl._execute(m)
    # Executor should NOT have been called since bid recovered
    sl.executor.sell_fok.assert_not_called()
