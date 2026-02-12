# tests/execution/test_trade_manager.py
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.execution.trade_manager import TradeManager
from src.execution.models import TradeIntent, PendingOrder, FillResult, OrderResult


@pytest.fixture
def intent():
    return TradeIntent(
        condition_id="cid_1", token_id="tok_1", outcome="Up",
        side="BUY", price=0.80, size_usd=10.0, reason="test",
        title="BTC-test",
    )


@pytest.fixture
def paper_manager(tmp_path):
    """Paper mode manager with real recorder, mocked Telegram."""
    from src.db.database import reset_engines
    reset_engines()

    db_url = f"sqlite:///{tmp_path}/test.db"
    mgr = TradeManager(
        executor=None,
        strategy="TestStrat",
        paper=True,
        db_url=db_url,
        event_type="test",
    )
    mgr._alerter = MagicMock()
    mgr._alerter.send_custom_alert = AsyncMock(return_value=True)

    yield mgr

    reset_engines()


# --- place() ---

@pytest.mark.asyncio
async def test_place_paper_returns_pending_order(paper_manager, intent):
    result = await paper_manager.place(intent)
    assert isinstance(result, PendingOrder)
    assert result.order_id.startswith("paper_")
    assert result.intent is intent


@pytest.mark.asyncio
async def test_place_paper_increments_counter(paper_manager, intent):
    r1 = await paper_manager.place(intent)
    r2 = await paper_manager.place(intent)
    assert r1.order_id != r2.order_id


@pytest.mark.asyncio
async def test_place_paper_sends_telegram(paper_manager, intent):
    await paper_manager.place(intent)
    paper_manager._alerter.send_custom_alert.assert_called_once()
    msg = paper_manager._alerter.send_custom_alert.call_args[0][0]
    assert "BID" in msg
    assert "Up" in msg


@pytest.mark.asyncio
async def test_place_paper_notify_bids_false(paper_manager, intent):
    paper_manager.notify_bids = False
    await paper_manager.place(intent)
    paper_manager._alerter.send_custom_alert.assert_not_called()


@pytest.mark.asyncio
async def test_place_live_calls_executor(tmp_path, intent):
    from src.db.database import reset_engines
    reset_engines()

    executor = AsyncMock()
    executor.place_order = AsyncMock(return_value={"orderID": "live_1", "status": "PLACED"})
    db_url = f"sqlite:///{tmp_path}/test_live.db"
    mgr = TradeManager(
        executor=executor,
        strategy="TestStrat",
        paper=False,
        db_url=db_url,
        event_type="test",
    )
    mgr._alerter = MagicMock()
    mgr._alerter.send_custom_alert = AsyncMock(return_value=True)
    result = await mgr.place(intent)
    assert isinstance(result, PendingOrder)
    assert result.order_id == "live_1"
    executor.place_order.assert_called_once()

    reset_engines()


# --- check_paper_fills() ---

@pytest.mark.asyncio
async def test_check_paper_fills_buy(paper_manager, intent):
    await paper_manager.place(intent)  # BUY Up @ 0.80

    def mock_levels(cid, outcome):
        return (0.79, 100, 0.80, 100)  # ask == our price -> fill

    fills = paper_manager.check_paper_fills(mock_levels)
    assert len(fills) == 1
    assert fills[0].filled is True
    assert abs(fills[0].shares - 12.5) < 0.01


@pytest.mark.asyncio
async def test_check_paper_fills_no_fill(paper_manager, intent):
    await paper_manager.place(intent)  # BUY Up @ 0.80

    def mock_levels(cid, outcome):
        return (0.79, 100, 0.81, 100)  # ask > our price -> no fill

    fills = paper_manager.check_paper_fills(mock_levels)
    assert len(fills) == 0


# --- settle() ---

@pytest.mark.asyncio
async def test_settle_win(paper_manager, intent):
    await paper_manager.place(intent)

    def mock_levels(cid, outcome):
        return (0.79, 100, 0.80, 100)

    paper_manager.check_paper_fills(mock_levels)
    pnl = await paper_manager.settle("cid_1", "Up", 1.0, won=True)
    assert pnl > 0
    stats = paper_manager.get_stats()
    assert stats["wins"] == 1
    assert stats["losses"] == 0


@pytest.mark.asyncio
async def test_settle_loss(paper_manager, intent):
    await paper_manager.place(intent)

    def mock_levels(cid, outcome):
        return (0.79, 100, 0.80, 100)

    paper_manager.check_paper_fills(mock_levels)
    pnl = await paper_manager.settle("cid_1", "Up", 0.0, won=False)
    assert pnl < 0
    stats = paper_manager.get_stats()
    assert stats["wins"] == 0
    assert stats["losses"] == 1


@pytest.mark.asyncio
async def test_settle_sends_telegram(paper_manager, intent):
    await paper_manager.place(intent)

    def mock_levels(cid, outcome):
        return (0.79, 100, 0.80, 100)

    paper_manager.check_paper_fills(mock_levels)
    paper_manager._alerter.send_custom_alert.reset_mock()
    await paper_manager.settle("cid_1", "Up", 1.0, won=True)
    paper_manager._alerter.send_custom_alert.assert_called_once()
    msg = paper_manager._alerter.send_custom_alert.call_args[0][0]
    assert "WIN" in msg


# --- cancel() ---

@pytest.mark.asyncio
async def test_cancel_paper(paper_manager, intent):
    order = await paper_manager.place(intent)
    ok = await paper_manager.cancel(order.order_id)
    assert ok is True
    assert len(paper_manager.get_pending_orders()) == 0


# --- multi-outcome positions ---

@pytest.mark.asyncio
async def test_two_outcomes_same_condition(paper_manager):
    """Two orders on same condition_id but different outcomes don't collide."""
    up_intent = TradeIntent(
        condition_id="cid_1", token_id="tok_up", outcome="Up",
        side="BUY", price=0.60, size_usd=6.0, reason="test", title="BTC-test",
    )
    down_intent = TradeIntent(
        condition_id="cid_1", token_id="tok_down", outcome="Down",
        side="BUY", price=0.40, size_usd=4.0, reason="test", title="BTC-test",
    )
    await paper_manager.place(up_intent)
    await paper_manager.place(down_intent)

    def mock_levels(cid, outcome):
        if outcome == "Up":
            return (0.59, 100, 0.60, 100)  # fills
        return (0.39, 100, 0.40, 100)  # fills

    fills = paper_manager.check_paper_fills(mock_levels)
    assert len(fills) == 2

    # Settle each independently
    pnl_up = await paper_manager.settle("cid_1", "Up", 1.0, won=True)
    pnl_down = await paper_manager.settle("cid_1", "Down", 0.0, won=False)
    assert pnl_up > 0  # 10 * (1.0 - 0.60) = 4.0
    assert pnl_down < 0  # 10 * (0.0 - 0.40) = -4.0


# --- settle() partial loss ---

@pytest.mark.asyncio
async def test_settle_partial_loss(paper_manager, intent):
    """settle() with partial loss uses actual settlement price, not full loss."""
    await paper_manager.place(intent)  # BUY Up @ 0.80, size 10

    def mock_levels(cid, outcome):
        return (0.79, 100, 0.80, 100)

    paper_manager.check_paper_fills(mock_levels)
    # Settles at 0.50 — partial loss
    pnl = await paper_manager.settle("cid_1", "Up", 0.50, won=False)
    expected = 12.5 * (0.50 - 0.80)  # -3.75, not -10.0
    assert abs(pnl - expected) < 0.01


# --- get_stats() ---

@pytest.mark.asyncio
async def test_get_stats_initial(paper_manager):
    stats = paper_manager.get_stats()
    assert stats["wins"] == 0
    assert stats["losses"] == 0
    assert stats["total_pnl"] == 0.0
    assert stats["pending_orders"] == 0
