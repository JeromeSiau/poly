# tests/execution/test_executor.py
import pytest
from src.execution.executor import ExecutorProtocol, adapt_polymarket_response
from src.execution.models import OrderResult


def test_adapt_polymarket_success():
    raw = {"orderID": "abc123", "status": "PLACED"}
    result = adapt_polymarket_response(raw)
    assert result.order_id == "abc123"
    assert result.status == "placed"
    assert result.error is None


def test_adapt_polymarket_error():
    raw = {"status": "ERROR", "message": "insufficient funds"}
    result = adapt_polymarket_response(raw)
    assert result.status == "error"
    assert result.error == "insufficient funds"
    assert result.order_id == ""


def test_adapt_polymarket_fallback_id():
    raw = {"id": "xyz", "status": "OK"}
    result = adapt_polymarket_response(raw)
    assert result.order_id == "xyz"


def test_protocol_is_runtime_checkable():
    class FakeExecutor:
        async def place_order(self, token_id, side, size, price, outcome="", order_type="GTC"):
            return OrderResult(order_id="fake")
        async def cancel_order(self, order_id):
            return True
        async def get_balance(self):
            return 100.0

    assert isinstance(FakeExecutor(), ExecutorProtocol)
