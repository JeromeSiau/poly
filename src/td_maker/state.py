# src/td_maker/state.py
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass(slots=True)
class PassiveOrder:
    order_id: str
    condition_id: str
    outcome: str
    token_id: str
    price: float
    size_usd: float
    placed_at: float
    cancelled_at: float = 0.0


@dataclass(slots=True)
class OpenPosition:
    condition_id: str
    outcome: str
    token_id: str
    entry_price: float
    size_usd: float
    shares: float
    filled_at: float
    order_legs: list[tuple[str, float]] = field(default_factory=list)


@dataclass
class MarketState:
    # Identity
    condition_id: str
    slug: str
    symbol: str
    slot_ts: int
    token_ids: dict[str, str]

    # Chainlink reference
    ref_price: float
    chainlink_symbol: str

    # Slot duration in seconds — configurable for 5-min vs 15-min markets
    slot_duration: int = 15 * 60

    # Orders
    active_orders: dict[str, PassiveOrder] = field(default_factory=dict)
    pending_cancels: dict[str, PassiveOrder] = field(default_factory=dict)

    # Position
    position: Optional[OpenPosition] = None
    bid_max: float = 0.0
    bid_below_exit_since: Optional[float] = None

    # Ladder
    fill_count: int = 0
    rungs_placed: set[tuple[str, int]] = field(default_factory=set)

    # Book
    book_history: deque = field(default_factory=lambda: deque(maxlen=20))
    last_bids: dict[str, float] = field(default_factory=dict)

    # Lifecycle
    discovered_at: float = field(default_factory=time.time)
    awaiting_settlement: bool = False
    settlement_deferred_until: Optional[float] = None

    # ── Mutating helpers ────────────────────────────────────────────

    def add_order(self, order: PassiveOrder) -> bool:
        """Add order. Returns False if order_id already tracked."""
        if order.order_id in self.active_orders:
            return False
        self.active_orders[order.order_id] = order
        return True

    def replace_order_id(self, old_id: str, new_id: str) -> None:
        """Swap placeholder id for real exchange id."""
        if old_id not in self.active_orders:
            return
        order = self.active_orders.pop(old_id)
        order.order_id = new_id
        self.active_orders[new_id] = order
        # Also update order_legs if present
        if self.position:
            legs = self.position.order_legs
            for i, (oid, sz) in enumerate(legs):
                if oid == old_id:
                    legs[i] = (new_id, sz)

    def move_to_pending_cancel(self, order_id: str) -> Optional[PassiveOrder]:
        """Move order from active to pending_cancels. Returns None if unknown."""
        order = self.active_orders.pop(order_id, None)
        if order is None:
            return None
        order.cancelled_at = time.time()
        self.pending_cancels[order_id] = order
        return order

    def record_fill(self, order_id: str, shares: float) -> bool:
        """
        Record a fill. Scale-in for ladder: accumulates into existing position.
        Returns False if order_id not found in active_orders or pending_cancels.
        """
        order = self.active_orders.pop(order_id, None) or \
                self.pending_cancels.pop(order_id, None)
        if order is None:
            return False

        if self.position is None:
            self.position = OpenPosition(
                condition_id=self.condition_id,
                outcome=order.outcome,
                token_id=order.token_id,
                entry_price=order.price,
                size_usd=order.size_usd,
                shares=shares,
                filled_at=time.time(),
                order_legs=[(order_id, order.size_usd)],
            )
        else:
            # Scale-in: weighted average entry price
            pos = self.position
            total_size = pos.size_usd + order.size_usd
            pos.entry_price = (
                pos.entry_price * pos.size_usd + order.price * order.size_usd
            ) / total_size
            pos.size_usd = total_size
            pos.shares += shares
            pos.order_legs.append((order_id, order.size_usd))

        self.fill_count += 1
        return True

    def slot_end_ts(self) -> float:
        """Absolute timestamp when this slot expires."""
        return self.slot_ts + self.slot_duration

    @staticmethod
    def is_placeholder(order_id: str) -> bool:
        return order_id.startswith("_placing_")

    @classmethod
    def orphan(cls, row, slot_duration: int = 15 * 60) -> "MarketState":
        """Create a minimal MarketState from a DB row (for startup recovery)."""
        return cls(
            condition_id=row.condition_id,
            slug="",
            symbol="",
            slot_ts=0,
            token_ids={row.outcome: row.token_id},
            ref_price=0.0,
            chainlink_symbol="",
            slot_duration=slot_duration,
        )


class MarketRegistry:
    """Central store of all active MarketState objects."""

    def __init__(self) -> None:
        self._markets: dict[str, MarketState] = {}

    def get(self, cid: str) -> Optional[MarketState]:
        return self._markets.get(cid)

    def register(self, market: MarketState) -> None:
        self._markets[market.condition_id] = market

    def remove(self, cid: str) -> None:
        self._markets.pop(cid, None)

    def __len__(self) -> int:
        return len(self._markets)

    def active_markets(self) -> list[MarketState]:
        return list(self._markets.values())

    def markets_with_positions(self) -> list[MarketState]:
        return [m for m in self._markets.values() if m.position is not None]

    def markets_with_orders(self) -> list[MarketState]:
        return [m for m in self._markets.values()
                if m.active_orders or m.pending_cancels]

    def expired_markets(self, now: float) -> list[MarketState]:
        grace = 5 * 60
        return [
            m for m in self._markets.values()
            if m.slot_ts > 0 and (m.slot_end_ts() + grace) < now
        ]

    def total_exposure(self) -> float:
        return sum(
            m.position.size_usd for m in self._markets.values()
            if m.position is not None
        )

    def total_pending(self) -> float:
        return sum(
            o.size_usd
            for m in self._markets.values()
            for o in m.active_orders.values()
            if not MarketState.is_placeholder(o.order_id)
        )

    def all_orders(self) -> list[PassiveOrder]:
        return [
            o for m in self._markets.values()
            for o in list(m.active_orders.values()) + list(m.pending_cancels.values())
        ]
