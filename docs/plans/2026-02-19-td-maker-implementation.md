# TD Maker Refactoring Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Decompose the 2,582-line `scripts/run_crypto_td_maker.py` monolith into 13 focused, testable modules under `src/td_maker/` with unified `MarketState`, resilience patterns, and all existing features preserved.

**Architecture:** Each component owns a single responsibility and receives dependencies via constructor injection. Shared state passes exclusively through `MarketRegistry` holding `MarketState` objects — replacing 12+ parallel dicts. See `docs/plans/2026-02-18-td-maker-refactor-design.md` for full design.

**Tech Stack:** Python 3.13, asyncio, structlog, httpx, SQLAlchemy async, pytest-asyncio, existing feeds/executor/guard/shadow unchanged.

**Worktree:** `.worktrees/refactor-td-maker` on branch `refactor/td-maker`

**Run tests from project root:** `pytest tests/td_maker/ -v`

---

## Task 1: Package scaffold + state.py

**Files:**
- Create: `src/td_maker/__init__.py`
- Create: `src/td_maker/state.py`
- Create: `tests/td_maker/__init__.py`
- Create: `tests/td_maker/test_state.py`

**Step 1: Write failing tests**

```python
# tests/td_maker/test_state.py
from __future__ import annotations
import time
from collections import deque
from src.td_maker.state import MarketState, PassiveOrder, OpenPosition, MarketRegistry


def make_market(cid="cid1") -> MarketState:
    return MarketState(
        condition_id=cid,
        slug=f"btc-up-15m-1000",
        symbol="btc/usd",
        slot_ts=1000,
        token_ids={"Up": "tok_up", "Down": "tok_dn"},
        ref_price=95000.0,
        chainlink_symbol="btc/usd",
    )


def make_order(order_id="o1", outcome="Up", price=0.80, cid="cid1") -> PassiveOrder:
    return PassiveOrder(
        order_id=order_id, condition_id=cid, outcome=outcome,
        token_id="tok_up", price=price, size_usd=10.0, placed_at=time.time()
    )


# --- MarketState tests ---

def test_add_order_ok():
    m = make_market()
    o = make_order()
    assert m.add_order(o) is True
    assert "o1" in m.active_orders


def test_add_order_duplicate_id_rejected():
    m = make_market()
    o = make_order()
    m.add_order(o)
    assert m.add_order(o) is False


def test_is_placeholder():
    assert MarketState.is_placeholder("_placing_abc123") is True
    assert MarketState.is_placeholder("real-order-id") is False


def test_replace_order_id():
    m = make_market()
    o = make_order(order_id="_placing_abc")
    m.add_order(o)
    m.replace_order_id("_placing_abc", "real-id-123")
    assert "real-id-123" in m.active_orders
    assert "_placing_abc" not in m.active_orders


def test_move_to_pending_cancel():
    m = make_market()
    o = make_order()
    m.add_order(o)
    result = m.move_to_pending_cancel("o1")
    assert result is not None
    assert "o1" not in m.active_orders
    assert "o1" in m.pending_cancels


def test_move_to_pending_cancel_unknown_returns_none():
    m = make_market()
    assert m.move_to_pending_cancel("nonexistent") is None


def test_record_fill_creates_position():
    m = make_market()
    o = make_order(order_id="o1", price=0.80)
    m.add_order(o)
    ok = m.record_fill("o1", shares=12.5)
    assert ok is True
    assert m.position is not None
    assert m.fill_count == 1
    assert abs(m.position.shares - 12.5) < 0.001


def test_record_fill_scale_in():
    m = make_market()
    o1 = make_order(order_id="o1", price=0.80)
    o2 = make_order(order_id="o2", price=0.78)
    m.add_order(o1)
    m.add_order(o2)
    m.record_fill("o1", shares=12.5)
    m.record_fill("o2", shares=12.82)
    assert m.fill_count == 2
    assert m.position.shares > 12.5


def test_record_fill_unknown_order_rejected():
    m = make_market()
    assert m.record_fill("ghost", shares=10.0) is False


# --- MarketRegistry tests ---

def test_registry_register_and_get():
    reg = MarketRegistry()
    m = make_market("c1")
    reg.register(m)
    assert reg.get("c1") is m


def test_registry_remove():
    reg = MarketRegistry()
    reg.register(make_market("c1"))
    reg.remove("c1")
    assert reg.get("c1") is None


def test_registry_markets_with_positions():
    reg = MarketRegistry()
    m1 = make_market("c1")
    m2 = make_market("c2")
    reg.register(m1)
    reg.register(m2)
    o = make_order(cid="c1")
    m1.add_order(o)
    m1.record_fill("o1", shares=12.5)
    assert m1 in reg.markets_with_positions()
    assert m2 not in reg.markets_with_positions()


def test_registry_expired_markets():
    reg = MarketRegistry()
    m = make_market()
    m.slot_ts = int(time.time()) - 1500  # 15m + 5m grace ago
    reg.register(m)
    expired = reg.expired_markets(time.time())
    assert m in expired


def test_registry_total_exposure():
    reg = MarketRegistry()
    m1 = make_market("c1")
    m2 = make_market("c2")
    reg.register(m1)
    reg.register(m2)
    o1 = make_order(order_id="o1", cid="c1")
    m1.add_order(o1)
    m1.record_fill("o1", shares=12.5)
    assert reg.total_exposure() == pytest.approx(10.0)
```

**Step 2: Run to confirm failure**

```bash
cd /Users/jerome/Projets/web/python/poly
pytest tests/td_maker/test_state.py -v 2>&1 | head -20
```
Expected: `ModuleNotFoundError: No module named 'src.td_maker'`

**Step 3: Implement `src/td_maker/__init__.py`**

```python
# src/td_maker/__init__.py
"""TD Maker strategy — refactored modular version."""
```

**Step 4: Implement `src/td_maker/state.py`**

```python
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

    @staticmethod
    def is_placeholder(order_id: str) -> bool:
        return order_id.startswith("_placing_")

    @classmethod
    def orphan(cls, row) -> "MarketState":
        """Create a minimal MarketState from a DB row (for startup recovery)."""
        return cls(
            condition_id=row.condition_id,
            slug="",
            symbol="",
            slot_ts=0,
            token_ids={row.outcome: row.token_id},
            ref_price=0.0,
            chainlink_symbol="",
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
        slot_duration = 15 * 60
        return [
            m for m in self._markets.values()
            if m.slot_ts > 0 and (m.slot_ts + slot_duration + grace) < now
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
```

**Step 5: Run tests**

```bash
pytest tests/td_maker/test_state.py -v
```
Expected: all pass (add `import pytest` to test file if needed)

**Step 6: Commit**

```bash
git add src/td_maker/ tests/td_maker/
git commit -m "feat(td_maker): add MarketState and MarketRegistry foundation"
```

---

## Task 2: resilience.py — clob_retry + FeedMonitor

**Files:**
- Create: `src/td_maker/resilience.py`
- Create: `tests/td_maker/test_resilience.py`

**Step 1: Write failing tests**

```python
# tests/td_maker/test_resilience.py
from __future__ import annotations
import asyncio
import time
import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock
from src.td_maker.resilience import clob_retry, FeedMonitor


@pytest.mark.asyncio
async def test_clob_retry_success_first_attempt():
    calls = 0
    async def op():
        nonlocal calls
        calls += 1
        return "ok"
    result = await clob_retry(op, operation="test")
    assert result == "ok"
    assert calls == 1


@pytest.mark.asyncio
async def test_clob_retry_retries_on_timeout():
    calls = 0
    async def op():
        nonlocal calls
        calls += 1
        if calls < 3:
            raise httpx.TimeoutException("timeout")
        return "ok"
    result = await clob_retry(op, base_delay=0.001, operation="test")
    assert result == "ok"
    assert calls == 3


@pytest.mark.asyncio
async def test_clob_retry_raises_after_max_attempts():
    async def op():
        raise httpx.TimeoutException("timeout")
    with pytest.raises(httpx.TimeoutException):
        await clob_retry(op, max_attempts=2, base_delay=0.001, operation="test")


def test_feed_monitor_not_stale_initially():
    feed = MagicMock()
    feed.last_message_at = time.time()
    monitor = FeedMonitor(feed, stale_threshold=30, name="test")
    assert monitor.is_stale() is False


def test_feed_monitor_stale_after_threshold():
    feed = MagicMock()
    feed.last_message_at = time.time() - 60
    monitor = FeedMonitor(feed, stale_threshold=30, name="test")
    assert monitor.is_stale() is True
```

**Step 2: Run to confirm failure**

```bash
pytest tests/td_maker/test_resilience.py -v 2>&1 | head -10
```

**Step 3: Implement**

```python
# src/td_maker/resilience.py
from __future__ import annotations

import asyncio
import time
from typing import Callable, Awaitable, TypeVar

import httpx
import structlog

logger = structlog.get_logger()
T = TypeVar("T")

RETRYABLE = (httpx.TimeoutException, httpx.HTTPStatusError, httpx.ConnectError)


async def clob_retry(
    coro_factory: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    operation: str = "",
) -> T:
    """Retry an async CLOB/REST call with exponential backoff."""
    last_exc: Exception = RuntimeError("no attempts")
    for attempt in range(max_attempts):
        try:
            return await coro_factory()
        except RETRYABLE as e:
            last_exc = e
            if attempt == max_attempts - 1:
                logger.error("clob_failed", op=operation, error=str(e),
                             attempts=attempt + 1)
                raise
            delay = base_delay * (2 ** attempt)
            logger.warning("clob_retry", op=operation, attempt=attempt + 1,
                           delay=delay, error=str(e))
            await asyncio.sleep(delay)
    raise last_exc


class FeedMonitor:
    """Detects silent WSS disconnections via last_message_at timestamp."""

    def __init__(self, feed, *, stale_threshold: float = 30.0, name: str = ""):
        self.feed = feed
        self.stale_threshold = stale_threshold
        self.name = name

    def is_stale(self) -> bool:
        return (time.time() - self.feed.last_message_at) > self.stale_threshold

    async def ensure_connected(self) -> bool:
        """Reconnect if stale. Returns True if reconnect was triggered."""
        if self.is_stale():
            logger.warning("feed_stale_reconnecting", feed=self.name)
            await self.feed.reconnect()
            return True
        return False
```

**Step 4: Run tests**

```bash
pytest tests/td_maker/test_resilience.py -v
```

**Step 5: Commit**

```bash
git add src/td_maker/resilience.py tests/td_maker/test_resilience.py
git commit -m "feat(td_maker): add clob_retry and FeedMonitor resilience utilities"
```

---

## Task 3: sizing.py

**Files:**
- Create: `src/td_maker/sizing.py`
- Create: `tests/td_maker/test_sizing.py`

**Step 1: Write failing tests**

```python
# tests/td_maker/test_sizing.py
import pytest
from unittest.mock import MagicMock
from src.td_maker.sizing import compute_rung_prices, Sizing
from src.td_maker.state import MarketRegistry, MarketState, PassiveOrder, OpenPosition
import time


def test_compute_rung_prices_single():
    prices = compute_rung_prices(0.75, 0.85, 1)
    assert len(prices) == 1
    assert prices[0] == pytest.approx(0.80, abs=0.01)


def test_compute_rung_prices_three():
    prices = compute_rung_prices(0.75, 0.85, 3)
    assert len(prices) == 3
    assert prices[0] == pytest.approx(0.75, abs=0.01)
    assert prices[-1] == pytest.approx(0.85, abs=0.01)


def test_compute_rung_prices_dedup():
    # Very narrow range — rungs would snap to same cent
    prices = compute_rung_prices(0.80, 0.81, 5)
    assert len(prices) == len(set(prices))


def test_sizing_available_budget():
    config = MagicMock()
    config.max_exposure = 100.0
    reg = MarketRegistry()
    sizing = Sizing(config)
    assert sizing.available_budget(reg) == pytest.approx(100.0)


def test_sizing_build_order_returns_none_when_no_budget():
    config = MagicMock()
    config.target_bid = 0.75
    config.max_bid = 0.85
    config.ladder_rungs = 1
    config.order_size = 10.0
    config.max_exposure = 5.0

    from src.td_maker.state import MarketState
    m = MarketState(condition_id="c1", slug="s", symbol="btc/usd", slot_ts=1,
                    token_ids={"Up":"t"}, ref_price=1.0, chainlink_symbol="btc/usd")
    reg = MarketRegistry()
    reg.register(m)

    sizing = Sizing(config)
    from dataclasses import dataclass

    class FakeResult:
        action = "maker"
        price = 0.80
    order = sizing.build_order(m, FakeResult(), budget=3.0)
    assert order is None  # budget < order_size
```

**Step 2: Run to confirm failure**

```bash
pytest tests/td_maker/test_sizing.py -v 2>&1 | head -10
```

**Step 3: Implement**

```python
# src/td_maker/sizing.py
from __future__ import annotations
import time
import uuid
from typing import Optional, Any
from src.td_maker.state import MarketState, MarketRegistry, PassiveOrder


def compute_rung_prices(lo: float, hi: float, n_rungs: int) -> list[float]:
    """Evenly-spaced rung prices in [lo, hi], snapped to 1c, deduplicated."""
    if n_rungs <= 1:
        return [round((lo + hi) / 2, 2)]
    raw = [lo + i * (hi - lo) / (n_rungs - 1) for i in range(n_rungs)]
    seen: set[float] = set()
    result: list[float] = []
    for p in raw:
        r = round(p, 2)
        if r not in seen:
            seen.add(r)
            result.append(r)
    return result


def model_size_scale(model: Any, p_win: float) -> float:
    """Map model P(win) → position size multiplier [0.2, 2.0]."""
    lo, hi = 0.55, 0.90
    clamped = max(lo, min(hi, p_win))
    t = (clamped - lo) / (hi - lo)
    return round(0.2 + t * (2.0 - 0.2), 3)


class Sizing:
    def __init__(self, config, model: Optional[Any] = None) -> None:
        self.config = config
        self.model = model
        self.rung_prices = compute_rung_prices(
            config.target_bid, config.max_bid, config.ladder_rungs
        )

    def available_budget(self, registry: MarketRegistry) -> float:
        return (
            self.config.max_exposure
            - registry.total_exposure()
            - registry.total_pending()
        )

    def build_order(
        self,
        market: MarketState,
        filter_result: Any,
        budget: float,
    ) -> Optional[PassiveOrder]:
        rung_idx = market.fill_count
        if rung_idx >= len(self.rung_prices):
            return None
        price = self.rung_prices[rung_idx]
        size = self._compute_size()
        if size > budget:
            return None
        token_id = market.token_ids.get(market.position.outcome
                                        if market.position else filter_result.outcome
                                        if hasattr(filter_result, "outcome")
                                        else next(iter(market.token_ids)), "")
        return PassiveOrder(
            order_id=f"_placing_{uuid.uuid4().hex[:8]}",
            condition_id=market.condition_id,
            outcome=getattr(filter_result, "outcome", "Up"),
            token_id=token_id,
            price=price,
            size_usd=size,
            placed_at=time.time(),
        )

    def _compute_size(self) -> float:
        return self.config.order_size
```

**Step 4: Run tests**

```bash
pytest tests/td_maker/test_sizing.py -v
```

**Step 5: Commit**

```bash
git add src/td_maker/sizing.py tests/td_maker/test_sizing.py
git commit -m "feat(td_maker): add Sizing with rung price computation"
```

---

## Task 4: filters.py — EntryFilters

**Files:**
- Create: `src/td_maker/filters.py`
- Create: `tests/td_maker/test_filters.py`

**Step 1: Write failing tests**

```python
# tests/td_maker/test_filters.py
import time
import pytest
from unittest.mock import MagicMock
from src.td_maker.filters import EntryFilters, FilterResult
from src.td_maker.state import MarketState


def make_config(**kwargs):
    cfg = MagicMock()
    cfg.target_bid = 0.75
    cfg.max_bid = 0.85
    cfg.min_move_pct = 0.0
    cfg.max_move_pct = 0.0
    cfg.min_entry_minutes = 0.0
    cfg.max_entry_minutes = 0.0
    cfg.entry_fair_margin = 0.0
    cfg.hybrid_skip_below = 0.55
    cfg.hybrid_taker_above = 0.72
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


def make_market(slot_ts=None):
    return MarketState(
        condition_id="c1", slug="btc-up-15m-1000", symbol="btc/usd",
        slot_ts=slot_ts or (int(time.time()) - 300),
        token_ids={"Up": "t1", "Down": "t2"},
        ref_price=95000.0, chainlink_symbol="btc/usd",
    )


def test_bid_below_range_skips():
    chainlink = MagicMock()
    f = EntryFilters(chainlink, make_config())
    result = f.should_bid(make_market(), outcome="Up", bid=0.70, ask=0.71)
    assert result.is_skip
    assert result.reason == "bid_out_of_range"


def test_bid_above_range_skips():
    chainlink = MagicMock()
    f = EntryFilters(chainlink, make_config())
    result = f.should_bid(make_market(), outcome="Up", bid=0.90, ask=0.91)
    assert result.is_skip
    assert result.reason == "bid_out_of_range"


def test_bid_in_range_no_filters_returns_maker():
    chainlink = MagicMock()
    f = EntryFilters(chainlink, make_config())
    result = f.should_bid(make_market(), outcome="Up", bid=0.80, ask=0.81)
    assert result.action == "maker"
    assert result.price == pytest.approx(0.80)


def test_max_entry_time_blocks_late():
    chainlink = MagicMock()
    # slot started 14 minutes ago, max_entry_minutes=10
    market = make_market(slot_ts=int(time.time()) - 840)
    f = EntryFilters(chainlink, make_config(max_entry_minutes=10.0))
    result = f.should_bid(market, outcome="Up", bid=0.80, ask=0.81)
    assert result.is_skip
    assert result.reason == "time_gate"


def test_min_entry_time_blocks_early():
    chainlink = MagicMock()
    # slot started 1 minute ago, min_entry_minutes=3
    market = make_market(slot_ts=int(time.time()) - 60)
    f = EntryFilters(chainlink, make_config(min_entry_minutes=3.0))
    result = f.should_bid(market, outcome="Up", bid=0.80, ask=0.81)
    assert result.is_skip
    assert result.reason == "time_gate"


def test_filter_result_is_skip_property():
    r = FilterResult(action="skip", reason="test", price=0.0, outcome="Up")
    assert r.is_skip is True
    r2 = FilterResult(action="maker", reason="", price=0.80, outcome="Up")
    assert r2.is_skip is False
```

**Step 2: Run to confirm failure**

```bash
pytest tests/td_maker/test_filters.py -v 2>&1 | head -10
```

**Step 3: Implement**

```python
# src/td_maker/filters.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Any

import structlog

logger = structlog.get_logger()


@dataclass(slots=True)
class FilterResult:
    action: str   # "skip" | "maker" | "taker"
    reason: str
    price: float
    outcome: str

    @property
    def is_skip(self) -> bool:
        return self.action == "skip"

    @classmethod
    def skip(cls, reason: str, outcome: str = "") -> "FilterResult":
        return cls(action="skip", reason=reason, price=0.0, outcome=outcome)

    @classmethod
    def maker(cls, price: float, outcome: str) -> "FilterResult":
        return cls(action="maker", reason="", price=price, outcome=outcome)

    @classmethod
    def taker(cls, price: float, outcome: str) -> "FilterResult":
        return cls(action="taker", reason="", price=price, outcome=outcome)


class EntryFilters:
    def __init__(self, chainlink_feed: Any, config: Any,
                 model: Optional[Any] = None) -> None:
        self.chainlink = chainlink_feed
        self.config = config
        self.model = model

    def should_bid(
        self,
        market: Any,
        *,
        outcome: str,
        bid: float,
        ask: float,
    ) -> FilterResult:
        """Returns FilterResult(action, reason, price, outcome)."""
        cfg = self.config

        if not (cfg.target_bid <= bid <= cfg.max_bid):
            return FilterResult.skip("bid_out_of_range", outcome)

        if not self._check_entry_time(market):
            return FilterResult.skip("time_gate", outcome)

        if cfg.min_move_pct > 0 and not self._check_min_move(market, outcome):
            return FilterResult.skip("insufficient_move", outcome)

        if cfg.max_move_pct > 0 and not self._check_max_move(market, outcome):
            return FilterResult.skip("excessive_move", outcome)

        if cfg.entry_fair_margin > 0 and not self._check_fair_value(
                market, outcome, bid):
            return FilterResult.skip("above_fair_value", outcome)

        if self.model:
            return self._check_model(market, outcome, bid, ask)

        return FilterResult.maker(bid, outcome)

    def _check_entry_time(self, market: Any) -> bool:
        cfg = self.config
        if cfg.min_entry_minutes <= 0 and cfg.max_entry_minutes <= 0:
            return True
        elapsed = (time.time() - market.slot_ts) / 60.0
        if cfg.min_entry_minutes > 0 and elapsed < cfg.min_entry_minutes:
            return False
        if cfg.max_entry_minutes > 0 and elapsed > cfg.max_entry_minutes:
            return False
        return True

    def _check_min_move(self, market: Any, outcome: str) -> bool:
        current = self.chainlink.get_price(market.chainlink_symbol)
        if current is None or market.ref_price <= 0:
            return True
        move_pct = (current - market.ref_price) / market.ref_price * 100.0
        if outcome == "Down":
            move_pct = -move_pct
        return move_pct >= self.config.min_move_pct

    def _check_max_move(self, market: Any, outcome: str) -> bool:
        current = self.chainlink.get_price(market.chainlink_symbol)
        if current is None or market.ref_price <= 0:
            return True
        move_pct = abs((current - market.ref_price) / market.ref_price * 100.0)
        return move_pct <= self.config.max_move_pct

    def _check_fair_value(self, market: Any, outcome: str, bid: float) -> bool:
        current = self.chainlink.get_price(market.chainlink_symbol)
        if current is None or market.ref_price <= 0:
            return True
        from src.utils.fair_value import estimate_fair_value
        slot_remaining = max(0, (market.slot_ts + 900) - time.time())
        fv = estimate_fair_value(
            current, market.ref_price, outcome, slot_remaining)
        return bid <= fv + self.config.entry_fair_margin

    def _check_model(self, market: Any, outcome: str,
                     bid: float, ask: float) -> FilterResult:
        # Delegate to existing model logic — placeholder for ML integration
        return FilterResult.maker(bid, outcome)
```

**Step 4: Run tests**

```bash
pytest tests/td_maker/test_filters.py -v
```

**Step 5: Commit**

```bash
git add src/td_maker/filters.py tests/td_maker/test_filters.py
git commit -m "feat(td_maker): add EntryFilters with FilterResult"
```

---

## Task 5: stop_loss.py — StopLossManager

**Files:**
- Create: `src/td_maker/stop_loss.py`
- Create: `tests/td_maker/test_stop_loss.py`

**Step 1: Write failing tests**

```python
# tests/td_maker/test_stop_loss.py
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
```

**Step 2: Run to confirm failure**

```bash
pytest tests/td_maker/test_stop_loss.py -v 2>&1 | head -15
```

**Step 3: Implement**

```python
# src/td_maker/stop_loss.py
from __future__ import annotations

import time
from typing import Optional, Any

import structlog

from src.td_maker.state import MarketState, MarketRegistry
from src.td_maker.resilience import clob_retry

logger = structlog.get_logger()


class StopLossManager:

    def __init__(
        self, *, registry: MarketRegistry, order_mgr: Any,
        executor: Any, chainlink_feed: Any, trade_manager: Any,
        shadow: Any, db: Any, config: Any,
        exit_model: Optional[Any] = None,
        poly_feed: Optional[Any] = None,
    ) -> None:
        self.registry = registry
        self.order_mgr = order_mgr
        self.executor = executor
        self.chainlink = chainlink_feed
        self.trade_manager = trade_manager
        self.shadow = shadow
        self.db = db
        self.config = config
        self.exit_model = exit_model
        self.poly_feed = poly_feed
        self._consecutive_failures: dict[str, int] = {}

    async def check_all(self, registry: MarketRegistry) -> None:
        """Check all positions. MUST run even when feed is stale."""
        for market in registry.markets_with_positions():
            await self._check_one(market)

    async def _check_one(self, market: MarketState) -> None:
        cfg = self.config
        if cfg.stoploss_peak <= 0 and self.exit_model is None:
            return

        bid = self._get_current_bid(market)
        if bid is None:
            self._handle_empty_book(market)
            return

        # Track bid max
        if bid > market.bid_max:
            market.bid_max = bid

        if self.exit_model:
            triggered = self._check_ml_exit(market, bid)
        else:
            triggered = self._check_rule_based(market, bid)

        if triggered:
            await self._execute(market)

    def _check_rule_based(self, market: MarketState, current_bid: float) -> bool:
        cfg = self.config
        if market.bid_max < cfg.stoploss_peak:
            return False
        if current_bid > cfg.stoploss_exit:
            market.bid_below_exit_since = None
            return False

        # Fair value override: suppress trigger when Chainlink says price is OK
        fair = self._estimate_fair_value(market)
        if fair is not None and fair > cfg.stoploss_exit + cfg.stoploss_fair_margin:
            now = time.time()
            if market.bid_below_exit_since is None:
                market.bid_below_exit_since = now
            if now - market.bid_below_exit_since < 10.0:
                return False

        return True

    def _check_ml_exit(self, market: MarketState, bid: float) -> bool:
        # Placeholder: integrate with exit model
        return False

    def _handle_empty_book(self, market: MarketState) -> None:
        """Empty book = market resolved/expired. Don't sell, wait for settlement."""
        market.awaiting_settlement = True
        logger.info("stoploss_book_empty_awaiting_settlement",
                    cid=market.condition_id)

    async def _execute(self, market: MarketState) -> None:
        """Sell with 3-level escalation."""
        # Re-check bid before selling
        bid = self._get_current_bid(market)
        if bid is not None and bid > self.config.stoploss_exit:
            logger.info("stoploss_aborted_recovered",
                        cid=market.condition_id, bid=bid)
            return

        pos = market.position
        try:
            await clob_retry(
                lambda: self.executor.cancel_and_sell(
                    token_id=pos.token_id,
                    shares=pos.shares,
                    price=0.01,
                    force_taker=True),
                max_attempts=3,
                base_delay=1.0,
                operation="stop_loss")

            # Success: cleanup
            pnl = (1.0 - pos.entry_price) * pos.shares - pos.size_usd
            await self.trade_manager.record_settle_direct(
                condition_id=market.condition_id,
                outcome=pos.outcome,
                entry_price=pos.entry_price,
                exit_price=0.01,
                size_usd=pos.size_usd,
                pnl=pnl,
                context=f"STOP-LOSS | peak {market.bid_max:.2f} -> {bid or 0:.2f}",
            )
            self.shadow.settle(market.condition_id, won=False)
            market.position = None
            market.bid_max = 0.0
            self._consecutive_failures.pop(market.condition_id, None)
            logger.info("stoploss_executed", cid=market.condition_id, pnl=pnl)

        except Exception as e:
            count = self._consecutive_failures.get(market.condition_id, 0) + 1
            self._consecutive_failures[market.condition_id] = count
            logger.error("stoploss_failed", cid=market.condition_id,
                         attempt=count, error=str(e))
            if count >= 3:
                await self.trade_manager.notify_critical(
                    f"STOP-LOSS FAILED x{count} — INTERVENTION REQUISE\n"
                    f"Market: {market.slug}\n"
                    f"Position: {pos.size_usd:.2f} USD @ {pos.entry_price:.2f}")
            # Never cleanup position on failure — retry next tick

    def _get_current_bid(self, market: MarketState) -> Optional[float]:
        if self.poly_feed is None:
            return None
        if market.position is None:
            return None
        outcome = market.position.outcome
        try:
            bid, _, _, _ = self.poly_feed.get_best_levels(
                market.condition_id, outcome)
            if bid and bid > 0:
                market.last_bids[outcome] = bid
                return bid
        except Exception:
            pass
        return None

    def _estimate_fair_value(self, market: MarketState) -> Optional[float]:
        current = self.chainlink.get_price(market.chainlink_symbol)
        if current is None or market.ref_price <= 0 or market.position is None:
            return None
        from src.utils.fair_value import estimate_fair_value
        slot_remaining = max(0, (market.slot_ts + 900) - time.time())
        return estimate_fair_value(
            current, market.ref_price, market.position.outcome, slot_remaining)
```

**Step 4: Run tests**

```bash
pytest tests/td_maker/test_stop_loss.py -v
```

**Step 5: Commit**

```bash
git add src/td_maker/stop_loss.py tests/td_maker/test_stop_loss.py
git commit -m "feat(td_maker): add StopLossManager with 3-level escalation"
```

---

## Task 6: fill_detector.py — FillDetector

**Files:**
- Create: `src/td_maker/fill_detector.py`
- Create: `tests/td_maker/test_fill_detector.py`

**Step 1: Write failing tests**

```python
# tests/td_maker/test_fill_detector.py
import time
import pytest
from unittest.mock import AsyncMock, MagicMock
from src.td_maker.state import MarketState, MarketRegistry, PassiveOrder
from src.td_maker.fill_detector import FillDetector


def make_market():
    m = MarketState(
        condition_id="c1", slug="btc-up-15m-1000", symbol="btc/usd",
        slot_ts=int(time.time()) - 300,
        token_ids={"Up": "tok_up"},
        ref_price=95000.0, chainlink_symbol="btc/usd",
    )
    return m


def make_order(order_id="o1"):
    return PassiveOrder(
        order_id=order_id, condition_id="c1", outcome="Up",
        token_id="tok_up", price=0.80, size_usd=10.0, placed_at=time.time()
    )


def make_fd(registry=None):
    if registry is None:
        registry = MarketRegistry()
    return FillDetector(
        registry=registry,
        order_mgr=AsyncMock(),
        poly_feed=MagicMock(),
        user_feed=MagicMock(),
        trade_manager=AsyncMock(),
        shadow=MagicMock(),
        db=MagicMock(),
        config=MagicMock(paper_mode=True),
        executor=AsyncMock(),
    )


def test_dedup_prevents_double_process():
    fd = make_fd()
    key = fd._fill_key("c1", "o1", "t1")
    fd._processed_fills[key] = time.time()
    assert fd._is_duplicate(key) is True


def test_purge_removes_old_entries():
    fd = make_fd()
    old_key = fd._fill_key("c1", "o1", "t1")
    fd._processed_fills[old_key] = time.time() - 2000  # older than 30 min
    fresh_key = fd._fill_key("c1", "o2", "t2")
    fd._processed_fills[fresh_key] = time.time()
    fd._purge_old_fills()
    assert old_key not in fd._processed_fills
    assert fresh_key in fd._processed_fills


def test_paper_fill_ask_crossed():
    fd = make_fd()
    m = make_market()
    o = make_order()
    m.add_order(o)
    # ask is 0.79 (below our bid of 0.80) → fill condition met
    fd.poly_feed.get_best_levels.return_value = (0.79, 100, 0.79, 50)
    assert fd._paper_fill_triggered(m, o) is True


def test_paper_fill_not_triggered_when_ask_above():
    fd = make_fd()
    m = make_market()
    o = make_order()
    m.add_order(o)
    # ask is 0.82 (above our bid of 0.80) → no fill
    fd.poly_feed.get_best_levels.return_value = (0.80, 100, 0.82, 50)
    assert fd._paper_fill_triggered(m, o) is False


def test_match_order_exact_id():
    fd = make_fd()
    m = make_market()
    o = make_order("real-id-123")
    m.add_order(o)

    class Msg:
        order_id = "real-id-123"
        condition_id = "c1"
        outcome = "Up"
        maker_order_id = "real-id-123"

    result = fd._match_order(m, Msg())
    assert result is o
```

**Step 2: Run to confirm failure**

```bash
pytest tests/td_maker/test_fill_detector.py -v 2>&1 | head -15
```

**Step 3: Implement**

```python
# src/td_maker/fill_detector.py
from __future__ import annotations

import time
from typing import Optional, Any

import structlog

from src.td_maker.state import MarketState, MarketRegistry, PassiveOrder
from src.td_maker.resilience import clob_retry

logger = structlog.get_logger()

_FILL_DEDUP_TTL = 1800  # 30 minutes


class FillDetector:

    def __init__(
        self, *, registry: MarketRegistry, order_mgr: Any,
        poly_feed: Any, user_feed: Any, trade_manager: Any,
        shadow: Any, db: Any, config: Any, executor: Any,
    ) -> None:
        self.registry = registry
        self.order_mgr = order_mgr
        self.poly_feed = poly_feed
        self.user_feed = user_feed
        self.trade_manager = trade_manager
        self.shadow = shadow
        self.db = db
        self.config = config
        self.executor = executor
        self._processed_fills: dict[str, float] = {}
        self._last_reconcile: float = 0.0

    def _fill_key(self, cid: str, order_id: str, trade_id: str) -> str:
        return f"{cid}:{order_id}:{trade_id}"

    def _is_duplicate(self, key: str) -> bool:
        return key in self._processed_fills

    def _purge_old_fills(self) -> None:
        cutoff = time.time() - _FILL_DEDUP_TTL
        self._processed_fills = {
            k: v for k, v in self._processed_fills.items() if v > cutoff
        }

    # ── WSS Real-time (live) ────────────────────────────────────────

    async def listen(self, registry: MarketRegistry) -> None:
        """Infinite loop draining User WS fills."""
        async for msg in self.user_feed:
            if getattr(msg, "type", None) == "fill":
                market = registry.get(msg.condition_id)
                if not market:
                    continue  # already settled, ignore
                key = self._fill_key(
                    msg.condition_id,
                    getattr(msg, "order_id", ""),
                    getattr(msg, "trade_id", ""))
                if self._is_duplicate(key):
                    continue
                order = self._match_order(market, msg)
                if order:
                    await self._process_fill(market, order, shares=msg.shares)
                    self._processed_fills[key] = time.time()
            elif getattr(msg, "type", None) == "reconnected":
                await self.reconcile()

    def _match_order(self, market: MarketState, msg: Any) -> Optional[PassiveOrder]:
        """4-priority matching: exact → placeholder → pending-cancel → broad."""
        maker_oid = getattr(msg, "maker_order_id", None)

        # 1. Exact maker_order_id
        if maker_oid and maker_oid in market.active_orders:
            return market.active_orders[maker_oid]

        # 2. Placeholder by cid
        for oid, o in market.active_orders.items():
            if MarketState.is_placeholder(oid):
                return o

        # 3. Pending cancel
        if maker_oid and maker_oid in market.pending_cancels:
            return market.pending_cancels[maker_oid]

        # 4. Broad cid match (only if no maker_order_id)
        if not maker_oid:
            if market.active_orders:
                return next(iter(market.active_orders.values()))

        return None

    # ── Paper fills ──────────────────────────────────────────────────

    async def check_paper_fills(self, registry: MarketRegistry) -> None:
        for market in registry.markets_with_orders():
            for order in list(market.active_orders.values()):
                if MarketState.is_placeholder(order.order_id):
                    continue
                if self._paper_fill_triggered(market, order):
                    shares = round(order.size_usd / order.price, 4)
                    key = self._fill_key(market.condition_id, order.order_id,
                                        "paper")
                    if not self._is_duplicate(key):
                        await self._process_fill(market, order, shares=shares,
                                                  paper=True)
                        self._processed_fills[key] = time.time()

    def _paper_fill_triggered(self, market: MarketState,
                               order: PassiveOrder) -> bool:
        try:
            bid, bid_sz, ask, ask_sz = self.poly_feed.get_best_levels(
                market.condition_id, order.outcome)
        except Exception:
            return False
        if ask is None:
            return False
        # Condition 1: ask crossed down to our bid
        if ask <= order.price:
            return True
        # Condition 2: bid dropped below our price
        if bid is not None and bid < order.price:
            return True
        return False

    # ── CLOB Reconciliation ──────────────────────────────────────────

    async def reconcile(self) -> None:
        """Poll CLOB for all active + pending-cancel orders."""
        for market in self.registry.markets_with_orders():
            for oid, order in list(market.active_orders.items()):
                if MarketState.is_placeholder(oid):
                    continue
                await self._check_clob_order(market, oid, order)
            for oid, order in list(market.pending_cancels.items()):
                await self._check_clob_order(market, oid, order)

    async def periodic_reconcile(self) -> None:
        if time.time() - self._last_reconcile < 60:
            return
        self._last_reconcile = time.time()
        self._purge_old_fills()
        await self.reconcile()

    async def _check_clob_order(self, market: MarketState,
                                 oid: str, order: PassiveOrder) -> None:
        try:
            result = await clob_retry(
                lambda: self.executor.get_order(oid),
                operation="reconcile_order")
            status = getattr(result, "status", result) if result else None
            if status in ("MATCHED", "FILLED"):
                shares = getattr(result, "size_matched", None) or \
                         round(order.size_usd / order.price, 4)
                key = self._fill_key(market.condition_id, oid, "recon")
                if not self._is_duplicate(key):
                    await self._process_fill(market, order, shares=shares)
                    self._processed_fills[key] = time.time()
        except Exception as e:
            logger.warning("reconcile_failed", oid=oid, error=str(e))

    # ── Shared fill processing ───────────────────────────────────────

    async def _process_fill(
        self, market: MarketState, order: PassiveOrder,
        shares: float, paper: bool = False,
    ) -> None:
        """Central fill handler — shared by WS, paper, reconciliation."""
        ok = market.record_fill(order.order_id, shares)
        if not ok:
            # Try from pending_cancels
            ok = market.record_fill(order.order_id, shares)
        if not ok:
            logger.warning("fill_rejected_state_inconsistency",
                           cid=market.condition_id, oid=order.order_id)
            return

        # Cancel other side on first fill
        if market.fill_count == 1:
            await self.order_mgr.cancel_other_side(market, filled_outcome=order.outcome)

        # Shadow taker entry
        try:
            bid, _, ask, _ = self.poly_feed.get_best_levels(
                market.condition_id, order.outcome)
            if ask:
                self.shadow.record(
                    market.condition_id, order.outcome, ask, order.size_usd)
        except Exception:
            pass

        # Telegram + DB
        await self.trade_manager.record_fill_direct(
            condition_id=market.condition_id,
            outcome=order.outcome,
            price=order.price,
            size_usd=order.size_usd,
            shares=shares,
            paper=paper,
        )
        self.db.fire(self.db.mark_filled(order.order_id, shares=shares))

        logger.info("fill_processed", cid=market.condition_id,
                    outcome=order.outcome, price=order.price,
                    shares=shares, paper=paper)
```

**Step 4: Run tests**

```bash
pytest tests/td_maker/test_fill_detector.py -v
```

**Step 5: Commit**

```bash
git add src/td_maker/fill_detector.py tests/td_maker/test_fill_detector.py
git commit -m "feat(td_maker): add FillDetector with dedup and 3 fill sources"
```

---

## Task 7: settlement.py — SettlementManager

**Files:**
- Create: `src/td_maker/settlement.py`
- Create: `tests/td_maker/test_settlement.py`

**Step 1: Write failing tests**

```python
# tests/td_maker/test_settlement.py
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.td_maker.state import MarketState, MarketRegistry, OpenPosition
from src.td_maker.settlement import SettlementManager


def make_market_expired():
    m = MarketState(
        condition_id="c1", slug="btc-up-15m-1000", symbol="btc/usd",
        slot_ts=int(time.time()) - 1500,  # expired
        token_ids={"Up": "tok_up"},
        ref_price=95000.0, chainlink_symbol="btc/usd",
    )
    m.position = OpenPosition(
        condition_id="c1", outcome="Up", token_id="tok_up",
        entry_price=0.80, size_usd=10.0, shares=12.5,
        filled_at=time.time() - 900,
    )
    m.last_bids["Up"] = 0.95
    return m


def make_sm():
    return SettlementManager(
        registry=MarketRegistry(),
        executor=AsyncMock(),
        trade_manager=AsyncMock(),
        shadow=MagicMock(),
        guard=MagicMock(),
        db=MagicMock(),
        config=MagicMock(strategy_tag="test", paper_mode=False),
        order_mgr=AsyncMock(),
    )


def test_force_from_last_bid_win():
    sm = make_sm()
    m = make_market_expired()
    m.last_bids["Up"] = 0.97
    result = sm._force_from_last_bid(m)
    assert result == "win"


def test_force_from_last_bid_loss():
    sm = make_sm()
    m = make_market_expired()
    m.last_bids["Up"] = 0.03
    result = sm._force_from_last_bid(m)
    assert result == "loss"


def test_force_from_last_bid_ambiguous_is_conservative():
    sm = make_sm()
    m = make_market_expired()
    m.last_bids["Up"] = 0.50  # ambiguous
    result = sm._force_from_last_bid(m)
    assert result == "loss"  # conservative


@pytest.mark.asyncio
async def test_settle_defers_when_no_resolution():
    sm = make_sm()
    m = make_market_expired()
    sm.registry.register(m)

    with patch.object(sm, "_query_resolution", new=AsyncMock(return_value=None)):
        await sm._settle(m)

    assert m.awaiting_settlement is True
    assert m.settlement_deferred_until is not None
    assert sm.registry.get("c1") is not None  # not removed yet


@pytest.mark.asyncio
async def test_settle_resolves_win():
    sm = make_sm()
    m = make_market_expired()
    sm.registry.register(m)

    with patch.object(sm, "_query_resolution", new=AsyncMock(return_value="win")):
        await sm._settle(m)

    sm.trade_manager.record_settle_direct.assert_called_once()
    assert sm.registry.get("c1") is None  # removed after settlement
```

**Step 2: Run to confirm failure**

```bash
pytest tests/td_maker/test_settlement.py -v 2>&1 | head -15
```

**Step 3: Implement**

```python
# src/td_maker/settlement.py
from __future__ import annotations

import time
from typing import Optional, Any

import httpx
import structlog

from src.td_maker.state import MarketState, MarketRegistry
from src.td_maker.resilience import clob_retry

logger = structlog.get_logger()

_SETTLE_DEFER_MAX = 3600  # 1 hour


class SettlementManager:

    def __init__(
        self, *, registry: MarketRegistry, executor: Any,
        trade_manager: Any, shadow: Any, guard: Any,
        db: Any, config: Any, order_mgr: Any,
    ) -> None:
        self.registry = registry
        self.executor = executor
        self.trade_manager = trade_manager
        self.shadow = shadow
        self.guard = guard
        self.db = db
        self.config = config
        self.order_mgr = order_mgr

    # ── DB Startup Recovery ─────────────────────────────────────────

    async def load_db_state(self, registry: MarketRegistry) -> None:
        """Restore pending/filled orders and positions from DB on startup."""
        from src.db.td_orders import load_orders
        rows = await load_orders(
            db_url=self.config.db_url,
            platform="polymarket",
            strategy_tag=self.config.strategy_tag)
        now = time.time()
        restored_orders = 0
        restored_positions = 0
        for row in rows:
            if row.placed_at and (now - row.placed_at) > 1800:
                from src.db.td_orders import delete_order
                await delete_order(db_url=self.config.db_url,
                                   order_id=row.order_id)
                continue
            market = registry.get(row.condition_id)
            if not market:
                market = MarketState.orphan(row)
                registry.register(market)
            if row.status == "pending":
                from src.td_maker.state import PassiveOrder
                o = PassiveOrder(
                    order_id=row.order_id,
                    condition_id=row.condition_id,
                    outcome=row.outcome,
                    token_id=row.token_id,
                    price=row.price,
                    size_usd=row.size_usd,
                    placed_at=row.placed_at or now,
                )
                market.add_order(o)
                restored_orders += 1
            elif row.status == "filled" and market.position is None:
                from src.td_maker.state import OpenPosition
                market.position = OpenPosition(
                    condition_id=row.condition_id,
                    outcome=row.outcome,
                    token_id=row.token_id,
                    entry_price=row.price,
                    size_usd=row.size_usd,
                    shares=row.shares or round(row.size_usd / row.price, 4),
                    filled_at=row.filled_at or now,
                )
                restored_positions += 1
        logger.info("db_state_loaded", orders=restored_orders,
                    positions=restored_positions)

    # ── Pruning ─────────────────────────────────────────────────────

    async def prune_expired(self, registry: MarketRegistry) -> None:
        """Settle and clean up markets past slot_end + grace."""
        for market in registry.expired_markets(time.time()):
            for oid in list(market.active_orders):
                await self.order_mgr.cancel_order(market, oid)
            if market.position:
                await self._settle(market)
            elif not market.awaiting_settlement:
                registry.remove(market.condition_id)

    # ── Settlement ──────────────────────────────────────────────────

    async def _settle(self, market: MarketState) -> None:
        resolution = await self._query_resolution(market)

        if resolution is None:
            if market.settlement_deferred_until is None:
                market.settlement_deferred_until = time.time() + _SETTLE_DEFER_MAX
                market.awaiting_settlement = True
                logger.info("settlement_deferred", cid=market.condition_id)
                return
            if time.time() < market.settlement_deferred_until:
                return
            resolution = self._force_from_last_bid(market)
            logger.warning("settlement_forced", cid=market.condition_id,
                           resolution=resolution)

        pos = market.position
        if pos is None:
            self.registry.remove(market.condition_id)
            return

        pnl = (pos.shares - pos.size_usd) if resolution == "win" else -pos.size_usd
        exit_price = 1.0 if resolution == "win" else 0.0

        if not self.config.paper_mode:
            await self.trade_manager.record_settle_direct(
                condition_id=market.condition_id,
                outcome=pos.outcome,
                entry_price=pos.entry_price,
                exit_price=exit_price,
                size_usd=pos.size_usd,
                pnl=pnl,
            )

        self.guard.record_result(pnl)
        self.shadow.settle(market.condition_id, won=(resolution == "win"))
        self.db.fire(self.db.mark_settled(pos, pnl))

        logger.info("settled", cid=market.condition_id, resolution=resolution,
                    pnl=round(pnl, 4))
        self.registry.remove(market.condition_id)

    async def _query_resolution(self, market: MarketState) -> Optional[str]:
        """Try Gamma API, then CLOB API. Returns 'win'|'loss'|None."""
        # 1. Gamma API via slug
        try:
            res = await clob_retry(
                lambda: self._query_gamma(market.slug),
                max_attempts=2, base_delay=0.5,
                operation="resolve_gamma")
            if res:
                return res
        except Exception:
            pass

        # 2. CLOB API via condition_id
        try:
            res = await clob_retry(
                lambda: self._query_clob(market.condition_id),
                max_attempts=2, base_delay=0.5,
                operation="resolve_clob")
            if res:
                return res
        except Exception:
            pass

        return None

    async def _query_gamma(self, slug: str) -> Optional[str]:
        from src.utils.parsing import _first_event_slug
        url = f"https://gamma-api.polymarket.com/events?slug={_first_event_slug(slug)}"
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url)
            r.raise_for_status()
            data = r.json()
        if not data:
            return None
        event = data[0] if isinstance(data, list) else data
        markets = event.get("markets", [])
        for m in markets:
            if m.get("closed") and m.get("outcomePrices"):
                prices = m["outcomePrices"]
                if isinstance(prices, str):
                    import json
                    prices = json.loads(prices)
                if prices and float(prices[0]) >= 0.99:
                    return "win"
                elif prices and float(prices[1]) >= 0.99:
                    return "loss"
        return None

    async def _query_clob(self, condition_id: str) -> Optional[str]:
        url = f"https://clob.polymarket.com/markets/{condition_id}"
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url)
            r.raise_for_status()
            data = r.json()
        tokens = data.get("tokens", [])
        for token in tokens:
            price = float(token.get("price", 0))
            if price >= 0.99:
                outcome = token.get("outcome", "").lower()
                return "win" if "yes" in outcome or "up" in outcome else "loss"
        return None

    def _force_from_last_bid(self, market: MarketState) -> str:
        if market.position is None:
            return "loss"
        outcome = market.position.outcome
        last_bid = market.last_bids.get(outcome, 0.5)
        if last_bid >= 0.9:
            return "win"
        elif last_bid <= 0.1:
            return "loss"
        else:
            logger.error("settlement_ambiguous_bid", bid=last_bid,
                         cid=market.condition_id)
            return "loss"  # conservative
```

**Step 4: Run tests**

```bash
pytest tests/td_maker/test_settlement.py -v
```

**Step 5: Commit**

```bash
git add src/td_maker/settlement.py tests/td_maker/test_settlement.py
git commit -m "feat(td_maker): add SettlementManager with 3-level resolution"
```

---

## Task 8: order_manager.py — OrderManager

**Files:**
- Create: `src/td_maker/order_manager.py`
- Create: `tests/td_maker/test_order_manager.py`

**Step 1: Write failing tests**

```python
# tests/td_maker/test_order_manager.py
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
```

**Step 2: Run to confirm failure**

```bash
pytest tests/td_maker/test_order_manager.py -v 2>&1 | head -15
```

**Step 3: Implement**

```python
# src/td_maker/order_manager.py
from __future__ import annotations

import asyncio
import time
from typing import Optional, Any

import structlog

from src.td_maker.state import MarketState, MarketRegistry, PassiveOrder
from src.td_maker.resilience import clob_retry

logger = structlog.get_logger()

_PLACEHOLDER_TIMEOUT = 15.0
_PENDING_CANCEL_TTL = 30.0


class OrderManager:

    def __init__(
        self, *, executor: Any, registry: MarketRegistry,
        db: Any, trade_manager: Any, config: Any,
    ) -> None:
        self.executor = executor
        self.registry = registry
        self.db = db
        self.trade_manager = trade_manager
        self.config = config

    async def place_order(
        self, market: MarketState, order: PassiveOrder
    ) -> Optional[str]:
        """Place order with placeholder safety and 15s timeout."""
        # Pre-register placeholder
        market.add_order(order)
        placeholder_id = order.order_id

        try:
            real_id = await asyncio.wait_for(
                clob_retry(
                    lambda: self.executor.place_order(
                        token_id=order.token_id,
                        price=order.price,
                        size_usd=order.size_usd,
                    ),
                    operation="place_order"),
                timeout=_PLACEHOLDER_TIMEOUT)
        except (asyncio.TimeoutError, Exception) as e:
            market.active_orders.pop(placeholder_id, None)
            await self._check_ghost_order(market, order)
            logger.error("place_order_failed",
                         cid=market.condition_id, error=str(e))
            return None

        market.replace_order_id(placeholder_id, real_id)
        self.db.fire(self.db.save_order(order_id=real_id, market=market,
                                        order=order))
        logger.info("order_placed", cid=market.condition_id,
                    outcome=order.outcome, price=order.price,
                    size_usd=order.size_usd, order_id=real_id)
        return real_id

    async def cancel_order(self, market: MarketState, order_id: str) -> None:
        """Cancel order — moves to pending_cancels, calls CLOB."""
        order = market.move_to_pending_cancel(order_id)
        if order is None:
            return
        try:
            await clob_retry(
                lambda: self.executor.cancel_order(order_id),
                operation="cancel_order")
        except Exception as e:
            logger.warning("cancel_order_failed", oid=order_id, error=str(e))

    async def cancel_other_side(
        self, market: MarketState, *, filled_outcome: str
    ) -> None:
        """Cancel all orders on the opposite outcome."""
        to_cancel = [
            oid for oid, o in list(market.active_orders.items())
            if o.outcome != filled_outcome
        ]
        for oid in to_cancel:
            await self.cancel_order(market, oid)

    async def cancel_batch(self, pairs: list[tuple[MarketState, str]]) -> None:
        for market, oid in pairs:
            await self.cancel_order(market, oid)

    async def place_batch(
        self, pairs: list[tuple[MarketState, PassiveOrder]]
    ) -> None:
        for market, order in pairs:
            await self.place_order(market, order)

    def expire_stale_cancels(self) -> None:
        """Clean pending_cancels older than 30s — schedule CLOB check."""
        now = time.time()
        for market in self.registry.active_markets():
            stale = [
                oid for oid, o in list(market.pending_cancels.items())
                if o.cancelled_at and (now - o.cancelled_at) > _PENDING_CANCEL_TTL
            ]
            for oid in stale:
                market.pending_cancels.pop(oid, None)
                self.db.fire(self.db.delete_order(oid))

    async def cancel_orphaned_orders(self) -> None:
        """Startup: cancel every open CLOB order from previous runs."""
        try:
            await clob_retry(
                lambda: self.executor.cancel_all(),
                operation="cancel_orphans")
            logger.info("orphaned_orders_cancelled")
        except Exception as e:
            logger.error("cancel_orphans_failed", error=str(e))

    async def _check_ghost_order(
        self, market: MarketState, order: PassiveOrder
    ) -> None:
        """After placement timeout: verify order doesn't exist on CLOB."""
        try:
            result = await self.executor.get_open_orders(
                token_id=order.token_id, price=order.price)
            if result:
                real_id = getattr(result[0], "id", None)
                if real_id:
                    logger.warning("ghost_order_detected",
                                   cid=market.condition_id, real_id=real_id)
                    order.order_id = real_id
                    market.add_order(order)
        except Exception as e:
            logger.warning("ghost_check_failed",
                           cid=market.condition_id, error=str(e))
```

**Step 4: Run tests**

```bash
pytest tests/td_maker/test_order_manager.py -v
```

**Step 5: Commit**

```bash
git add src/td_maker/order_manager.py tests/td_maker/test_order_manager.py
git commit -m "feat(td_maker): add OrderManager with placeholder timeout and ghost order detection"
```

---

## Task 9: discovery.py + bidding.py

**Files:**
- Create: `src/td_maker/discovery.py`
- Create: `src/td_maker/bidding.py`
- Create: `tests/td_maker/test_discovery.py`
- Create: `tests/td_maker/test_bidding.py`

**Step 1: Write failing tests**

```python
# tests/td_maker/test_discovery.py
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.td_maker.state import MarketRegistry
from src.td_maker.discovery import MarketDiscovery


@pytest.mark.asyncio
async def test_discover_registers_new_markets():
    poly_feed = MagicMock()
    poly_feed.subscribe_batch = AsyncMock()
    chainlink = MagicMock()
    chainlink.get_price.return_value = 95000.0

    config = MagicMock()
    config.symbols = ["BTCUSDT"]

    discovery = MarketDiscovery(poly_feed, chainlink, config)
    registry = MarketRegistry()

    fake_market = MagicMock()
    fake_market.condition_id = "cid1"
    fake_market.slug = "btc-up-15m-1771079400"
    fake_market.token_ids = {"Up": "tok_up", "Down": "tok_dn"}

    with patch("src.td_maker.discovery.fetch_crypto_markets",
               new=AsyncMock(return_value=[fake_market])):
        await discovery.discover(registry)

    assert registry.get("cid1") is not None


@pytest.mark.asyncio
async def test_discover_skips_already_known():
    poly_feed = MagicMock()
    poly_feed.subscribe_batch = AsyncMock()
    chainlink = MagicMock()
    chainlink.get_price.return_value = 95000.0

    config = MagicMock()
    config.symbols = ["BTCUSDT"]

    discovery = MarketDiscovery(poly_feed, chainlink, config)
    registry = MarketRegistry()

    fake_market = MagicMock()
    fake_market.condition_id = "cid1"
    fake_market.slug = "btc-up-15m-1771079400"
    fake_market.token_ids = {"Up": "tok_up"}

    with patch("src.td_maker.discovery.fetch_crypto_markets",
               new=AsyncMock(return_value=[fake_market])):
        await discovery.discover(registry)
        initial_market = registry.get("cid1")
        await discovery.discover(registry)

    assert registry.get("cid1") is initial_market  # same object, not re-registered
```

```python
# tests/td_maker/test_bidding.py
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
        token_ids={"Up": "tok_up", "Down": "tok_dn"},
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
```

**Step 2: Run to confirm failures**

```bash
pytest tests/td_maker/test_discovery.py tests/td_maker/test_bidding.py -v 2>&1 | head -20
```

**Step 3: Implement discovery.py**

```python
# src/td_maker/discovery.py
from __future__ import annotations
import time
from typing import Any
import structlog
from src.td_maker.state import MarketState, MarketRegistry
from src.td_maker.resilience import clob_retry
from src.utils.crypto_markets import fetch_crypto_markets, SLUG_TO_CHAINLINK

logger = structlog.get_logger()


def parse_slug_info(slug: str) -> tuple[str, int]:
    """Parse 'btc-up-15m-1771079400' -> ('btc/usd', 1771079400)."""
    parts = slug.split("-")
    try:
        slot_ts = int(parts[-1])
    except (ValueError, IndexError):
        slot_ts = 0
    symbol_key = parts[0] if parts else ""
    chainlink_symbol = SLUG_TO_CHAINLINK.get(symbol_key, f"{symbol_key}/usd")
    return chainlink_symbol, slot_ts


class MarketDiscovery:

    def __init__(self, poly_feed: Any, chainlink_feed: Any, config: Any) -> None:
        self.poly_feed = poly_feed
        self.chainlink = chainlink_feed
        self.config = config

    async def discover(self, registry: MarketRegistry) -> None:
        try:
            raw_markets = await clob_retry(
                lambda: fetch_crypto_markets(self.config.symbols),
                operation="discover_markets")
        except Exception as e:
            logger.error("discovery_failed", error=str(e))
            return

        new_cids: list[str] = []
        for m in raw_markets:
            if registry.get(m.condition_id):
                continue
            chainlink_symbol, slot_ts = parse_slug_info(m.slug)
            ref_price = self.chainlink.get_price(chainlink_symbol) or 0.0
            state = MarketState(
                condition_id=m.condition_id,
                slug=m.slug,
                symbol=chainlink_symbol,
                slot_ts=slot_ts,
                token_ids=getattr(m, "token_ids", {}),
                ref_price=ref_price,
                chainlink_symbol=chainlink_symbol,
            )
            registry.register(state)
            new_cids.append(m.condition_id)

        if new_cids:
            await self.poly_feed.subscribe_batch(new_cids)
            logger.info("markets_discovered", count=len(new_cids))
```

**Step 4: Implement bidding.py**

```python
# src/td_maker/bidding.py
from __future__ import annotations
from typing import Any
import structlog
from src.td_maker.state import MarketState, MarketRegistry, PassiveOrder

logger = structlog.get_logger()


class BiddingEngine:

    def __init__(
        self, *, registry: MarketRegistry, filters: Any, order_mgr: Any,
        sizing: Any, config: Any, poly_feed: Any,
    ) -> None:
        self.registry = registry
        self.filters = filters
        self.order_mgr = order_mgr
        self.sizing = sizing
        self.config = config
        self.poly_feed = poly_feed

    async def scan_and_place(self, registry: MarketRegistry) -> None:
        budget = self.sizing.available_budget(registry)

        for market in registry.active_markets():
            if market.awaiting_settlement:
                continue
            if market.fill_count >= self.config.ladder_rungs:
                continue

            for outcome in market.token_ids:
                await self._check_outcome(market, outcome, budget)

    async def _check_outcome(
        self, market: MarketState, outcome: str, budget: float
    ) -> None:
        try:
            bid, bid_sz, ask, ask_sz = self.poly_feed.get_best_levels(
                market.condition_id, outcome)
        except Exception:
            return

        if bid is None:
            return

        # Track last known bid for settlement fallback
        market.last_bids[outcome] = bid

        result = self.filters.should_bid(market, outcome=outcome,
                                          bid=bid, ask=ask)
        if result.is_skip:
            return

        # Check rung dedup
        price_cents = int(result.price * 100)
        rung_key = (outcome, price_cents)
        if rung_key in market.rungs_placed:
            return

        order = self.sizing.build_order(market, result, budget)
        if order is None:
            return

        market.rungs_placed.add(rung_key)
        placed_id = await self.order_mgr.place_order(market, order)
        if placed_id is None:
            market.rungs_placed.discard(rung_key)
```

**Step 5: Run tests**

```bash
pytest tests/td_maker/test_discovery.py tests/td_maker/test_bidding.py -v
```

**Step 6: Commit**

```bash
git add src/td_maker/discovery.py src/td_maker/bidding.py \
        tests/td_maker/test_discovery.py tests/td_maker/test_bidding.py
git commit -m "feat(td_maker): add MarketDiscovery and BiddingEngine"
```

---

## Task 10: engine.py + status.py — Orchestrator

**Files:**
- Create: `src/td_maker/engine.py`
- Create: `src/td_maker/status.py`
- Create: `tests/td_maker/test_engine.py`

**Step 1: Write failing tests**

```python
# tests/td_maker/test_engine.py
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
```

**Step 2: Run to confirm failure**

```bash
pytest tests/td_maker/test_engine.py -v 2>&1 | head -15
```

**Step 3: Implement status.py**

```python
# src/td_maker/status.py
from __future__ import annotations
import time
from datetime import datetime, timezone
from typing import Any
from src.td_maker.state import MarketRegistry


class StatusLine:

    def __init__(self, registry: MarketRegistry, guard: Any,
                 shadow: Any, config: Any) -> None:
        self.registry = registry
        self.guard = guard
        self.shadow = shadow
        self.config = config
        self._last_print: float = 0.0
        self._interval: float = 30.0

    def print_if_due(self) -> None:
        if time.time() - self._last_print < self._interval:
            return
        self._last_print = time.time()
        print(self._format(), flush=True)

    def _format(self) -> str:
        r = self.registry
        now = datetime.now(timezone.utc).strftime("%H:%M:%S")
        stats = self.guard.stats if hasattr(self.guard, "stats") else {}
        wins = stats.get("wins", 0)
        losses = stats.get("losses", 0)
        pnl = stats.get("pnl", 0.0)
        exp = r.total_exposure()
        n_orders = sum(
            len(m.active_orders) for m in r.active_markets())
        n_pos = len(r.markets_with_positions())
        shadow_line = self.shadow.status_line() if hasattr(
            self.shadow, "status_line") else ""
        return (
            f"[{now}] mkts={len(r)} orders={n_orders} pos={n_pos} "
            f"{wins}W-{losses}L pnl=${pnl:+.2f} exp=${exp:.0f}"
            + (f" | {shadow_line}" if shadow_line else "")
        )
```

**Step 4: Implement engine.py**

```python
# src/td_maker/engine.py
from __future__ import annotations
import asyncio
import time
from typing import Any
import structlog
from src.td_maker.state import MarketRegistry

logger = structlog.get_logger()


class TDMakerEngine:
    """Orchestrator — zero business logic, wires components and runs loops."""

    def __init__(
        self, *,
        registry: MarketRegistry,
        discovery: Any,
        bidding: Any,
        order_mgr: Any,
        fill_detector: Any,
        stop_loss: Any,
        settlement: Any,
        status: Any,
        guard: Any,
        poly_feed: Any,
        user_feed: Any,
        chainlink_feed: Any,
        config: Any,
    ) -> None:
        self.registry = registry
        self.discovery = discovery
        self.bidding = bidding
        self.order_mgr = order_mgr
        self.fill_detector = fill_detector
        self.stop_loss = stop_loss
        self.settlement = settlement
        self.status = status
        self.guard = guard
        self.poly_feed = poly_feed
        self.user_feed = user_feed
        self.chainlink_feed = chainlink_feed
        self.config = config

    async def run(self) -> None:
        await self._startup()
        await asyncio.gather(
            self._discovery_loop(),
            self._maker_loop(),
            self._fill_listener(),
        )

    async def _startup(self) -> None:
        logger.info("engine_startup")
        await self.settlement.load_db_state(self.registry)
        await self.poly_feed.connect()
        await self.user_feed.connect()
        await self.chainlink_feed.connect()
        await self.order_mgr.cancel_orphaned_orders()
        await self.fill_detector.reconcile()
        logger.info("engine_ready")

    async def _discovery_loop(self) -> None:
        while True:
            await self.discovery.discover(self.registry)
            await self.settlement.prune_expired(self.registry)
            await asyncio.sleep(self.config.discovery_interval)

    async def _maker_loop(self) -> None:
        while True:
            try:
                await asyncio.wait_for(
                    self.poly_feed.book_updated.wait(),
                    timeout=self.config.maker_interval)
            except asyncio.TimeoutError:
                pass
            self.guard.heartbeat()
            await self._tick()

    async def _tick(self) -> None:
        """Core tick. Order is intentional — stop-loss before CB gate."""
        # 1. Cleanup stale cancels
        self.order_mgr.expire_stale_cancels()

        # 2. Stop-loss MUST run before CB gate (detect crash even when stale)
        await self.stop_loss.check_all(self.registry)

        # 3. Circuit breaker gate
        if not self.guard.is_trading_allowed():
            self.status.print_if_due()
            return

        # 4. Paper fill simulation
        if self.config.paper_mode:
            await self.fill_detector.check_paper_fills(self.registry)

        # 5. Periodic CLOB reconciliation
        await self.fill_detector.periodic_reconcile()

        # 6. New orders
        await self.bidding.scan_and_place(self.registry)

        # 7. Status line
        self.status.print_if_due()

    async def _fill_listener(self) -> None:
        await self.fill_detector.listen(self.registry)
```

**Step 5: Run tests**

```bash
pytest tests/td_maker/test_engine.py -v
```

**Step 6: Commit**

```bash
git add src/td_maker/engine.py src/td_maker/status.py \
        tests/td_maker/test_engine.py
git commit -m "feat(td_maker): add TDMakerEngine orchestrator and StatusLine"
```

---

## Task 11: Wire up scripts/run_crypto_td_maker.py

**Files:**
- Modify: `scripts/run_crypto_td_maker.py` — replace body with new wiring (~100 lines)
- Keep: All CLI args identical (no breaking changes)

**Step 1: Verify all args are accounted for**

Before modifying, verify CLI args match by running:
```bash
python scripts/run_crypto_td_maker.py --help 2>&1 | grep "\-\-" | wc -l
```

Note the count — the new script must have the same count.

**Step 2: Write the new script**

Replace the entire content of `scripts/run_crypto_td_maker.py` with:

```python
#!/usr/bin/env python3
"""Passive time-decay maker for Polymarket 15-min crypto markets.

Entry point — wires dependencies, starts TDMakerEngine.
See src/td_maker/ for business logic.
"""
from __future__ import annotations

import argparse
import asyncio

import structlog

from src.utils.logging import configure_logging

configure_logging()

try:
    import uvloop
except ImportError:
    uvloop = None

from config.settings import settings
from src.arb.polymarket_executor import PolymarketExecutor
from src.execution import TradeManager
from src.feeds.chainlink import ChainlinkFeed
from src.feeds.polymarket import PolymarketFeed, PolymarketUserFeed
from src.risk.guard import RiskGuard
from src.shadow.taker_shadow import TakerShadow

from src.td_maker.state import MarketRegistry
from src.td_maker.engine import TDMakerEngine
from src.td_maker.discovery import MarketDiscovery
from src.td_maker.filters import EntryFilters
from src.td_maker.sizing import Sizing, compute_rung_prices
from src.td_maker.order_manager import OrderManager
from src.td_maker.fill_detector import FillDetector
from src.td_maker.stop_loss import StopLossManager
from src.td_maker.settlement import SettlementManager
from src.td_maker.bidding import BiddingEngine
from src.td_maker.status import StatusLine

logger = structlog.get_logger()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="TD Maker strategy")

    # Mode
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--paper", dest="paper_mode", action="store_true",
                      default=True)
    mode.add_argument("--live", dest="paper_mode", action="store_false")
    mode.add_argument("--autopilot", dest="paper_mode", action="store_false")

    # Market
    p.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT")

    # Sizing
    p.add_argument("--wallet", type=float, default=0.0)
    p.add_argument("--order-size", type=float, default=0.0, dest="order_size")
    p.add_argument("--max-exposure", type=float, default=0.0,
                   dest="max_exposure")

    # Ladder
    p.add_argument("--target-bid", type=float, default=0.75,
                   dest="target_bid")
    p.add_argument("--max-bid", type=float, default=0.85, dest="max_bid")
    p.add_argument("--ladder-rungs", type=int, default=1,
                   dest="ladder_rungs")

    # Timing
    p.add_argument("--discovery-interval", type=float, default=60.0,
                   dest="discovery_interval")
    p.add_argument("--maker-interval", type=float, default=0.5,
                   dest="maker_interval")

    # Filters
    p.add_argument("--min-move-pct", type=float, default=0.0,
                   dest="min_move_pct")
    p.add_argument("--max-move-pct", type=float, default=0.0,
                   dest="max_move_pct")
    p.add_argument("--min-entry-minutes", type=float, default=0.0,
                   dest="min_entry_minutes")
    p.add_argument("--max-entry-minutes", type=float, default=0.0,
                   dest="max_entry_minutes")
    p.add_argument("--entry-fair-margin", type=float, default=0.0,
                   dest="entry_fair_margin")

    # ML entry
    p.add_argument("--model-path", default="", dest="model_path")
    p.add_argument("--hybrid-skip-below", type=float, default=0.55,
                   dest="hybrid_skip_below")
    p.add_argument("--hybrid-taker-above", type=float, default=0.72,
                   dest="hybrid_taker_above")

    # Stop-loss
    p.add_argument("--stoploss-peak", type=float, default=0.0,
                   dest="stoploss_peak")
    p.add_argument("--stoploss-exit", type=float, default=0.0,
                   dest="stoploss_exit")
    p.add_argument("--stoploss-fair-margin", type=float, default=0.10,
                   dest="stoploss_fair_margin")

    # ML exit
    p.add_argument("--exit-model-path", default="", dest="exit_model_path")
    p.add_argument("--exit-threshold", type=float, default=0.35,
                   dest="exit_threshold")

    # Circuit breaker
    p.add_argument("--cb-max-losses", type=int, default=5,
                   dest="cb_max_losses")
    p.add_argument("--cb-max-drawdown", type=float, default=-50.0,
                   dest="cb_max_drawdown")
    p.add_argument("--cb-stale-seconds", type=float, default=30.0,
                   dest="cb_stale_seconds")
    p.add_argument("--cb-stale-cancel", type=float, default=120.0,
                   dest="cb_stale_cancel")
    p.add_argument("--cb-stale-exit", type=float, default=300.0,
                   dest="cb_stale_exit")
    p.add_argument("--cb-daily-limit", type=float, default=-200.0,
                   dest="cb_daily_limit")

    # Misc
    p.add_argument("--strategy-tag", default="crypto_td_maker",
                   dest="strategy_tag")
    p.add_argument("--db-url", default="", dest="db_url")

    return p


class Config:
    """Flat config object built from parsed args + derived values."""

    def __init__(self, args: argparse.Namespace) -> None:
        # Copy all args
        for k, v in vars(args).items():
            setattr(self, k, v)
        # Normalize symbols
        self.symbols = [s.strip() for s in args.symbols.split(",")]
        # db_url fallback
        if not self.db_url:
            from config.settings import settings
            self.db_url = settings.DATABASE_URL or ""


async def _auto_detect_wallet(executor: PolymarketExecutor) -> float:
    try:
        balance = await executor.get_wallet_balance()
        return float(balance)
    except Exception:
        return 0.0


async def main_async() -> None:
    args = build_parser().parse_args()
    config = Config(args)

    # Infrastructure
    executor = PolymarketExecutor(settings) if not config.paper_mode else None
    poly_feed = PolymarketFeed(settings)
    user_feed = PolymarketUserFeed(settings) if not config.paper_mode else None
    chainlink_feed = ChainlinkFeed(settings)
    shadow = TakerShadow()

    # Auto-sizing
    if config.wallet <= 0 and not config.paper_mode and executor:
        config.wallet = await _auto_detect_wallet(executor)
        logger.info("wallet_auto_detected", balance=config.wallet)
    if config.order_size <= 0:
        config.order_size = max(1.0, config.wallet * 0.025)
    if config.max_exposure <= 0:
        config.max_exposure = max(config.order_size, config.wallet * 0.50)

    logger.info("sizing_configured",
                order_size=config.order_size,
                max_exposure=config.max_exposure)

    # Risk guard
    guard = RiskGuard(
        max_consecutive_losses=config.cb_max_losses,
        max_drawdown=config.cb_max_drawdown,
        daily_loss_limit=config.cb_daily_limit,
        stale_threshold=config.cb_stale_seconds,
        poly_feed=poly_feed,
        db_url=config.db_url,
    )

    # Trade manager
    manager = TradeManager(
        executor=executor,
        guard=guard,
        settings=settings,
        strategy_tag=config.strategy_tag,
        paper_mode=config.paper_mode,
    )

    # DB wrapper (lazy fire-and-forget)
    class DBFire:
        def __init__(self, db_url):
            self.db_url = db_url

        def fire(self, coro):
            asyncio.create_task(self._run(coro))

        async def _run(self, coro):
            try:
                await coro
            except Exception as e:
                logger.error("db_fire_error", error=str(e))

        def save_order(self, *, order_id, market, order):
            from src.db.td_orders import save_order
            return save_order(
                db_url=self.db_url, platform="polymarket",
                strategy_tag=config.strategy_tag,
                order_id=order_id, condition_id=market.condition_id,
                token_id=order.token_id, outcome=order.outcome,
                price=order.price, size_usd=order.size_usd,
                placed_at=order.placed_at)

        def mark_filled(self, order_id, *, shares):
            from src.db.td_orders import mark_filled
            return mark_filled(db_url=self.db_url,
                               order_id=order_id, shares=shares)

        def mark_settled(self, position, pnl):
            from src.db.td_orders import mark_settled
            return mark_settled(db_url=self.db_url,
                                order_id=position.order_legs[0][0]
                                if position.order_legs else "",
                                pnl=pnl)

        def delete_order(self, order_id):
            from src.db.td_orders import delete_order
            return delete_order(db_url=self.db_url, order_id=order_id)

    db = DBFire(config.db_url)

    # Components
    registry = MarketRegistry()
    filters = EntryFilters(chainlink_feed, config)
    sizing = Sizing(config)
    order_mgr = OrderManager(executor=executor or MagicExecutor(),
                             registry=registry, db=db,
                             trade_manager=manager, config=config)
    fill_detector = FillDetector(
        registry=registry, order_mgr=order_mgr,
        poly_feed=poly_feed, user_feed=user_feed or AsyncUserFeed(),
        trade_manager=manager, shadow=shadow,
        db=db, config=config, executor=executor or MagicExecutor())
    stop_loss = StopLossManager(
        registry=registry, order_mgr=order_mgr,
        executor=executor or MagicExecutor(),
        chainlink_feed=chainlink_feed, trade_manager=manager,
        shadow=shadow, db=db, config=config, poly_feed=poly_feed)
    settlement = SettlementManager(
        registry=registry, executor=executor or MagicExecutor(),
        trade_manager=manager, shadow=shadow, guard=guard,
        db=db, config=config, order_mgr=order_mgr)
    discovery = MarketDiscovery(poly_feed, chainlink_feed, config)
    bidding = BiddingEngine(registry=registry, filters=filters,
                            order_mgr=order_mgr, sizing=sizing,
                            config=config, poly_feed=poly_feed)
    status = StatusLine(registry, guard, shadow, config)

    engine = TDMakerEngine(
        registry=registry, discovery=discovery, bidding=bidding,
        order_mgr=order_mgr, fill_detector=fill_detector,
        stop_loss=stop_loss, settlement=settlement, status=status,
        guard=guard, poly_feed=poly_feed,
        user_feed=user_feed or AsyncUserFeed(),
        chainlink_feed=chainlink_feed, config=config)

    print(f"TD Maker starting — paper={config.paper_mode} "
          f"size=${config.order_size:.0f} max_exp=${config.max_exposure:.0f} "
          f"rungs={config.ladder_rungs} [{config.target_bid}-{config.max_bid}]")

    await engine.run()


class MagicExecutor:
    """Paper mode stub — raises on any real CLOB call."""
    async def place_order(self, **kw): raise RuntimeError("paper mode")
    async def cancel_order(self, *a): pass
    async def cancel_all(self): pass
    async def get_order(self, *a): return None
    async def get_open_orders(self, **kw): return []
    async def sell_fok(self, **kw): raise RuntimeError("paper mode")
    async def cancel_and_sell(self, **kw): raise RuntimeError("paper mode")


class AsyncUserFeed:
    """Paper mode stub for user feed."""
    async def connect(self): pass
    async def __aiter__(self):
        while True:
            await asyncio.sleep(3600)
            yield


def main() -> None:
    if uvloop:
        uvloop.install()
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
```

**Step 3: Run paper mode smoke test**

```bash
timeout 5 ./run scripts/run_crypto_td_maker.py --paper 2>&1 | head -20
```
Expected: starts up, prints banner, connects feeds (may fail on network — that's OK, just check no import errors)

**Step 4: Run all td_maker tests**

```bash
pytest tests/td_maker/ -v
```
Expected: all pass

**Step 5: Commit**

```bash
git add scripts/run_crypto_td_maker.py
git commit -m "feat(td_maker): wire up new modular engine in entry script"
```

---

## Task 12: Run full test suite + fix any regressions

**Step 1: Run full test suite**

```bash
pytest tests/ -v --ignore=tests/api -q 2>&1 | tail -20
```
Expected: same baseline as before (445 pass, 1 fail for weather_oracle)

**Step 2: Fix any new failures**

If tests unrelated to td_maker fail, investigate. Do NOT touch existing tests outside `tests/td_maker/`.

**Step 3: Run paper mode 30s integration test**

```bash
timeout 30 ./run scripts/run_crypto_td_maker.py --paper \
  --order-size 5 --max-exposure 50 2>&1
```
Expected: discovers markets, subscribes to WS, status line prints after 30s.

**Step 4: Final commit**

```bash
git add -A
git commit -m "test(td_maker): full test suite passing, integration smoke test OK"
```

---

## Summary

| Task | Files | Tests |
|------|-------|-------|
| 1 | state.py | 13 tests |
| 2 | resilience.py | 5 tests |
| 3 | sizing.py | 5 tests |
| 4 | filters.py | 6 tests |
| 5 | stop_loss.py | 5 tests |
| 6 | fill_detector.py | 5 tests |
| 7 | settlement.py | 4 tests |
| 8 | order_manager.py | 4 tests |
| 9 | discovery.py + bidding.py | 4 tests |
| 10 | engine.py + status.py | 2 tests |
| 11 | scripts/run_crypto_td_maker.py | smoke test |
| 12 | full suite | regression check |

After completion, use `superpowers:finishing-a-development-branch` to merge.
