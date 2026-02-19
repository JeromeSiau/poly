import asyncio
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.td_maker.engine import TDMakerEngine
from src.td_maker.state import MarketRegistry


def make_engine():
    config = MagicMock()
    config.maker_interval = 0.01
    config.discovery_interval = 999
    config.paper_mode = True

    guard = MagicMock()
    guard.is_trading_allowed.return_value = True
    guard.heartbeat = MagicMock()

    poly_feed = MagicMock()
    poly_feed.book_updated = AsyncMock()
    poly_feed.last_message_at = time.time()

    return TDMakerEngine(
        registry=MarketRegistry(),
        discovery=AsyncMock(),
        bidding=AsyncMock(),
        order_mgr=AsyncMock(),
        fill_detector=AsyncMock(),
        stop_loss=AsyncMock(),
        settlement=AsyncMock(),
        status=MagicMock(),
        guard=guard,
        poly_feed=poly_feed,
        user_feed=AsyncMock(),
        chainlink_feed=MagicMock(),
        config=config,
    )


@pytest.mark.asyncio
async def test_tick_calls_stoploss_before_circuit_breaker():
    engine = make_engine()
    call_order = []

    async def record_stoploss(*a, **kw):
        call_order.append("stop_loss")
    async def record_bidding(*a, **kw):
        call_order.append("bidding")

    engine.stop_loss.check_all = record_stoploss
    engine.bidding.scan_and_place = record_bidding
    engine.fill_detector.check_paper_fills = AsyncMock()
    engine.fill_detector.periodic_reconcile = AsyncMock()
    engine.order_mgr.expire_stale_cancels = MagicMock()

    await engine._tick()

    assert call_order.index("stop_loss") < call_order.index("bidding")


@pytest.mark.asyncio
async def test_tick_skips_bidding_when_cb_tripped():
    engine = make_engine()
    engine.guard.is_trading_allowed.return_value = False
    engine.fill_detector.check_paper_fills = AsyncMock()
    engine.fill_detector.periodic_reconcile = AsyncMock()
    engine.order_mgr.expire_stale_cancels = MagicMock()
    engine.stop_loss.check_all = AsyncMock()

    await engine._tick()

    engine.bidding.scan_and_place.assert_not_called()
