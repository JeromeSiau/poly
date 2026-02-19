import time
import pytest
from unittest.mock import AsyncMock, MagicMock
from src.td_maker.state import MarketState, MarketRegistry, PassiveOrder
from src.td_maker.filters import FilterResult
from src.td_maker.bidding import BiddingEngine


def make_market(cid="c1"):
    return MarketState(
        condition_id=cid, slug="btc-up-15m-1000", symbol="btc/usd",
        slot_ts=int(time.time()) - 300,
        token_ids={"Up": "tok_up"},  # single outcome — bidding places one order
        ref_price=95000.0, chainlink_symbol="btc/usd",
    )


@pytest.mark.asyncio
async def test_scan_places_order_when_filters_pass():
    registry = MarketRegistry()
    m = make_market()
    registry.register(m)

    filters = MagicMock()
    filters.should_bid.return_value = FilterResult(
        action="maker", reason="", price=0.80, outcome="Up")

    sizing = MagicMock()
    sizing.available_budget.return_value = 100.0
    sizing.build_order.return_value = PassiveOrder(
        order_id="_placing_abc", condition_id="c1", outcome="Up",
        token_id="tok_up", price=0.80, size_usd=10.0, placed_at=time.time())

    order_mgr = AsyncMock()
    order_mgr.place_order = AsyncMock(return_value="real-id-123")
    poly_feed = MagicMock()
    poly_feed.get_best_levels.return_value = (0.80, 100, 0.81, 50)

    engine = BiddingEngine(
        registry=registry, filters=filters, order_mgr=order_mgr,
        sizing=sizing, config=MagicMock(ladder_rungs=1),
        poly_feed=poly_feed)

    await engine.scan_and_place(registry)

    order_mgr.place_order.assert_called_once()


@pytest.mark.asyncio
async def test_scan_skips_awaiting_settlement():
    registry = MarketRegistry()
    m = make_market()
    m.awaiting_settlement = True
    registry.register(m)

    filters = MagicMock()
    order_mgr = AsyncMock()

    engine = BiddingEngine(
        registry=registry, filters=filters, order_mgr=order_mgr,
        sizing=MagicMock(), config=MagicMock(ladder_rungs=1),
        poly_feed=MagicMock())

    await engine.scan_and_place(registry)
    order_mgr.place_order.assert_not_called()
