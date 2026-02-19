import asyncio
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.td_maker.state import MarketState, MarketRegistry, PassiveOrder
from src.td_maker.order_manager import OrderManager


def make_market():
    return MarketState(
        condition_id="c1", slug="btc-up-15m-1000", symbol="btc/usd",
        slot_ts=int(time.time()) - 300,
        token_ids={"Up": "tok_up", "Down": "tok_dn"},
        ref_price=95000.0, chainlink_symbol="btc/usd",
    )


def make_order(order_id="_placing_abc"):
    return PassiveOrder(
        order_id=order_id, condition_id="c1", outcome="Up",
        token_id="tok_up", price=0.80, size_usd=10.0, placed_at=time.time()
    )


def make_om(executor=None):
    if executor is None:
        executor = AsyncMock()
        executor.place_order = AsyncMock(return_value="real-order-id-123")
    reg = MarketRegistry()
    return OrderManager(
        executor=executor,
        registry=reg,
        db=MagicMock(),
        trade_manager=AsyncMock(),
        config=MagicMock(paper_mode=False),
    ), reg


@pytest.mark.asyncio
async def test_place_order_replaces_placeholder():
    om, reg = make_om()
    m = make_market()
    reg.register(m)
    order = make_order()

    real_id = await om.place_order(m, order)
    assert real_id == "real-order-id-123"
    assert "real-order-id-123" in m.active_orders
    assert "_placing_abc" not in m.active_orders


@pytest.mark.asyncio
async def test_place_order_timeout_cleans_placeholder():
    executor = AsyncMock()
    executor.place_order = AsyncMock(side_effect=asyncio.TimeoutError())
    om, reg = make_om(executor)
    m = make_market()
    reg.register(m)
    order = make_order()

    with patch.object(om, "_check_ghost_order", new=AsyncMock()):
        result = await om.place_order(m, order)

    assert result is None
    assert "_placing_abc" not in m.active_orders


@pytest.mark.asyncio
async def test_cancel_order_moves_to_pending():
    executor = AsyncMock()
    executor.cancel_order = AsyncMock(return_value=True)
    om, reg = make_om(executor)
    m = make_market()
    reg.register(m)
    o = make_order("real-id-456")
    m.add_order(o)

    await om.cancel_order(m, "real-id-456")

    assert "real-id-456" not in m.active_orders
    assert "real-id-456" in m.pending_cancels


@pytest.mark.asyncio
async def test_cancel_other_side_cancels_down_orders():
    executor = AsyncMock()
    executor.cancel_order = AsyncMock(return_value=True)
    om, reg = make_om(executor)
    m = make_market()
    reg.register(m)

    o_up = make_order("up-order")
    o_dn = PassiveOrder(order_id="dn-order", condition_id="c1", outcome="Down",
                        token_id="tok_dn", price=0.20, size_usd=10.0,
                        placed_at=time.time())
    m.add_order(o_up)
    m.add_order(o_dn)

    await om.cancel_other_side(m, filled_outcome="Up")
    assert "dn-order" not in m.active_orders
