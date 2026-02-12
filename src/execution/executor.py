"""Executor protocol and adapters."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from src.execution.models import OrderResult


@runtime_checkable
class ExecutorProtocol(Protocol):
    """Interface that all exchange executors must satisfy."""

    async def place_order(
        self,
        token_id: str,
        side: str,
        size: float,
        price: float,
        outcome: str = "",
        order_type: str = "GTC",
    ) -> dict[str, Any] | OrderResult: ...

    async def cancel_order(self, order_id: str) -> dict[str, Any] | bool: ...


def adapt_polymarket_response(raw: dict[str, Any]) -> OrderResult:
    """Convert PolymarketExecutor dict response to OrderResult."""
    if raw.get("status") == "ERROR":
        return OrderResult(
            order_id="",
            filled=False,
            status="error",
            error=raw.get("message", "unknown error"),
        )
    order_id = raw.get("orderID") or raw.get("id") or ""
    return OrderResult(
        order_id=str(order_id),
        filled=False,
        status="placed",
    )
