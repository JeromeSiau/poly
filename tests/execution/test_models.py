import time
from src.execution.models import TradeIntent, PendingOrder, FillResult, OrderResult


def test_trade_intent_shares():
    intent = TradeIntent(
        condition_id="abc", token_id="t1", outcome="Up",
        side="BUY", price=0.80, size_usd=10.0, reason="test",
    )
    assert abs(intent.shares - 12.5) < 0.01


def test_trade_intent_shares_zero_price():
    intent = TradeIntent(
        condition_id="abc", token_id="t1", outcome="Up",
        side="BUY", price=0.0, size_usd=10.0, reason="test",
    )
    assert intent.shares == 0.0


def test_trade_intent_default_timestamp():
    before = time.time()
    intent = TradeIntent(
        condition_id="abc", token_id="t1", outcome="Up",
        side="BUY", price=0.50, size_usd=5.0, reason="test",
    )
    after = time.time()
    assert before <= intent.timestamp <= after


def test_pending_order():
    intent = TradeIntent(
        condition_id="abc", token_id="t1", outcome="Up",
        side="BUY", price=0.80, size_usd=10.0, reason="test",
    )
    order = PendingOrder(order_id="paper_1", intent=intent, placed_at=1000.0)
    assert order.order_id == "paper_1"
    assert order.intent.price == 0.80


def test_fill_result_defaults():
    fill = FillResult(filled=True, shares=12.5, avg_price=0.80)
    assert fill.pnl_delta == 0.0


def test_order_result_defaults():
    result = OrderResult(order_id="abc123")
    assert result.status == "placed"
    assert result.filled is False
    assert result.error is None
