"""Shared data structures for trade execution."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass(slots=True)
class TradeIntent:
    """What the strategy wants to do."""

    condition_id: str
    token_id: str
    outcome: str
    side: str  # "BUY" or "SELL"
    price: float
    size_usd: float
    reason: str
    title: str = ""
    edge_pct: float = 0.0
    timestamp: float = field(default_factory=time.time)

    @property
    def shares(self) -> float:
        return self.size_usd / self.price if self.price > 0 else 0.0


@dataclass(slots=True)
class PendingOrder:
    """An order placed but not yet filled."""

    order_id: str
    intent: TradeIntent
    placed_at: float


@dataclass(slots=True)
class FillResult:
    """Result of an order fill."""

    filled: bool
    shares: float
    avg_price: float
    pnl_delta: float = 0.0


@dataclass(slots=True)
class OrderResult:
    """Result from an executor's place_order call."""

    order_id: str
    filled: bool = False
    status: str = "placed"  # "placed", "filled", "error"
    error: Optional[str] = None
