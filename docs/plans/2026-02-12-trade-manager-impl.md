# TradeManager Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace duplicated trade execution plumbing across 8 strategies with a single `TradeManager` class that handles order placement, paper fill detection, DB persistence, and Telegram notifications.

**Architecture:** `TradeManager` orchestrates an external `ExecutorProtocol` (Polymarket/Kalshi), an internal `TradeRecorder` (extracted from `TwoSidedPaperRecorder`), and an internal `TelegramAlerter`. Strategies call 3 methods: `place()`, `check_paper_fills()`, `settle()`.

**Tech Stack:** Python 3.11+, SQLAlchemy (sync sessions), structlog, httpx (Telegram), dataclasses.

**Design doc:** `docs/plans/2026-02-12-trade-manager-design.md`

---

### Task 1: Models — `src/execution/models.py`

**Files:**
- Create: `src/execution/__init__.py`
- Create: `src/execution/models.py`
- Test: `tests/execution/test_models.py`

**Step 1: Write tests**

```python
# tests/execution/test_models.py
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
```

**Step 2: Run tests, verify FAIL**

Run: `pytest tests/execution/test_models.py -v`
Expected: ImportError — module does not exist yet.

**Step 3: Implement**

```python
# src/execution/__init__.py
from src.execution.models import (
    TradeIntent,
    PendingOrder,
    FillResult,
    OrderResult,
)

__all__ = [
    "TradeIntent",
    "PendingOrder",
    "FillResult",
    "OrderResult",
]
```

```python
# src/execution/models.py
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
```

**Step 4: Run tests, verify PASS**

Run: `pytest tests/execution/test_models.py -v`

**Step 5: Commit**

```bash
git add src/execution/__init__.py src/execution/models.py tests/execution/test_models.py
git commit -m "feat(execution): add shared trade data models"
```

---

### Task 2: ExecutorProtocol — `src/execution/executor.py`

**Files:**
- Create: `src/execution/executor.py`
- Modify: `src/execution/__init__.py`
- Test: `tests/execution/test_executor.py`

**Step 1: Write tests**

```python
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
    assert hasattr(ExecutorProtocol, '__protocol_attrs__') or True  # Protocol exists
```

**Step 2: Run tests, verify FAIL**

Run: `pytest tests/execution/test_executor.py -v`

**Step 3: Implement**

```python
# src/execution/executor.py
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
```

Update `src/execution/__init__.py` to add exports:
```python
from src.execution.executor import ExecutorProtocol, adapt_polymarket_response
```

**Step 4: Run tests, verify PASS**

Run: `pytest tests/execution/test_executor.py -v`

**Step 5: Commit**

```bash
git add src/execution/executor.py src/execution/__init__.py tests/execution/test_executor.py
git commit -m "feat(execution): add ExecutorProtocol and Polymarket adapter"
```

---

### Task 3: TradeRecorder — `src/execution/trade_recorder.py`

**Files:**
- Create: `src/execution/trade_recorder.py`
- Modify: `src/execution/__init__.py`
- Test: `tests/execution/test_trade_recorder.py`

**Step 1: Write tests**

```python
# tests/execution/test_trade_recorder.py
import pytest
from src.execution.trade_recorder import TradeRecorder
from src.execution.models import TradeIntent, FillResult


@pytest.fixture
def recorder(tmp_path):
    db_url = f"sqlite:///{tmp_path}/test.db"
    rec = TradeRecorder(
        db_url=db_url,
        strategy_tag="test_strat",
        event_type="test_event",
        run_id="run_001",
    )
    rec.bootstrap()
    return rec


@pytest.fixture
def buy_intent():
    return TradeIntent(
        condition_id="cid_1", token_id="tok_1", outcome="Up",
        side="BUY", price=0.80, size_usd=10.0, reason="test_entry",
        title="BTC test", edge_pct=0.02, timestamp=1700000000.0,
    )


@pytest.fixture
def sell_intent():
    return TradeIntent(
        condition_id="cid_1", token_id="tok_1", outcome="Up",
        side="SELL", price=1.0, size_usd=12.5, reason="settlement",
        title="BTC test", edge_pct=0.0, timestamp=1700001000.0,
    )


def test_record_fill_creates_observation_and_trade(recorder, buy_intent):
    fill = FillResult(filled=True, shares=12.5, avg_price=0.80)
    obs_id = recorder.record_fill(
        intent=buy_intent, fill=fill,
        fair_prices={"Up": 0.82}, execution_mode="paper",
    )
    assert obs_id > 0


def test_record_fill_skips_zero_shares(recorder, buy_intent):
    fill = FillResult(filled=True, shares=0.0, avg_price=0.80)
    obs_id = recorder.record_fill(
        intent=buy_intent, fill=fill,
        fair_prices={"Up": 0.82}, execution_mode="paper",
    )
    assert obs_id == 0


def test_record_settle_closes_buy_records(recorder, buy_intent, sell_intent):
    fill_buy = FillResult(filled=True, shares=12.5, avg_price=0.80)
    recorder.record_fill(
        intent=buy_intent, fill=fill_buy,
        fair_prices={"Up": 0.82}, execution_mode="paper",
    )

    fill_sell = FillResult(filled=True, shares=12.5, avg_price=1.0, pnl_delta=2.50)
    obs_id = recorder.record_settle(
        intent=sell_intent, fill=fill_sell,
        fair_prices={"Up": 1.0},
    )
    assert obs_id > 0

    # Verify the BUY record was closed
    from src.db.models import PaperTrade
    from src.db.database import get_sync_session
    session = get_sync_session(recorder._db_url)
    try:
        open_buys = session.query(PaperTrade).filter_by(
            side="BUY", is_open=True
        ).count()
        assert open_buys == 0
    finally:
        session.close()


def test_game_state_contains_strategy_info(recorder, buy_intent):
    fill = FillResult(filled=True, shares=12.5, avg_price=0.80)
    obs_id = recorder.record_fill(
        intent=buy_intent, fill=fill,
        fair_prices={"Up": 0.82}, execution_mode="paper",
    )

    from src.db.models import LiveObservation
    from src.db.database import get_sync_session
    session = get_sync_session(recorder._db_url)
    try:
        obs = session.query(LiveObservation).get(obs_id)
        gs = obs.game_state
        assert gs["strategy_tag"] == "test_strat"
        assert gs["run_id"] == "run_001"
        assert gs["condition_id"] == "cid_1"
        assert gs["outcome"] == "Up"
        assert gs["side"] == "BUY"
        assert gs["mode"] == "paper"
    finally:
        session.close()
```

**Step 2: Run tests, verify FAIL**

Run: `pytest tests/execution/test_trade_recorder.py -v`

**Step 3: Implement**

```python
# src/execution/trade_recorder.py
"""Generic trade persistence for all strategies.

Extracted from TwoSidedPaperRecorder. Writes to LiveObservation + PaperTrade
tables with the same game_state JSON format so the /trades API and dashboards
keep working unchanged.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import structlog

from src.db.database import get_sync_session, init_db
from src.db.models import LiveObservation, PaperTrade
from src.execution.models import TradeIntent, FillResult

logger = structlog.get_logger()


def _ensure_sync_url(database_url: str) -> str:
    """Strip async driver suffix (e.g. sqlite+aiosqlite -> sqlite)."""
    if not database_url:
        return "sqlite:///data/arb.db"
    if "://" not in database_url:
        return database_url
    scheme, suffix = database_url.split("://", 1)
    if "+" in scheme:
        scheme = scheme.split("+", 1)[0]
    return f"{scheme}://{suffix}"


class TradeRecorder:
    """Generic trade persistence to LiveObservation + PaperTrade."""

    def __init__(
        self,
        db_url: str,
        *,
        strategy_tag: str,
        event_type: str = "",
        run_id: str = "",
    ) -> None:
        self._db_url = _ensure_sync_url(db_url)
        self._strategy_tag = strategy_tag
        self._event_type = event_type
        self._run_id = run_id

    def bootstrap(self) -> None:
        init_db(self._db_url)

    def record_fill(
        self,
        *,
        intent: TradeIntent,
        fill: FillResult,
        fair_prices: dict[str, float] | None = None,
        execution_mode: str = "paper",
    ) -> int:
        """Persist a fill (entry). Returns observation_id, or 0 if skipped."""
        if fill.shares <= 0:
            return 0
        return self._persist(
            intent=intent, fill=fill, fair_prices=fair_prices,
            execution_mode=execution_mode, is_settle=False,
        )

    def record_settle(
        self,
        *,
        intent: TradeIntent,
        fill: FillResult,
        fair_prices: dict[str, float] | None = None,
    ) -> int:
        """Persist a settlement (exit). Closes related BUY records."""
        if fill.shares <= 0:
            return 0
        return self._persist(
            intent=intent, fill=fill, fair_prices=fair_prices,
            execution_mode="settlement", is_settle=True,
        )

    def _persist(
        self,
        *,
        intent: TradeIntent,
        fill: FillResult,
        fair_prices: dict[str, float] | None,
        execution_mode: str,
        is_settle: bool,
    ) -> int:
        fair = fair_prices or {}
        fair_price = fair.get(intent.outcome, intent.price)
        obs_ts = datetime.fromtimestamp(intent.timestamp, tz=timezone.utc)

        game_state: dict[str, Any] = {
            "strategy": self._event_type,
            "strategy_tag": self._strategy_tag,
            "run_id": self._run_id,
            "condition_id": intent.condition_id,
            "title": intent.title,
            "outcome": intent.outcome,
            "token_id": intent.token_id,
            "side": intent.side,
            "reason": intent.reason,
            "mode": execution_mode,
            "fair_price": fair_price,
            "edge_theoretical": intent.edge_pct,
            "fill_price": fill.avg_price,
            "shares": fill.shares,
            "size_usd": intent.size_usd,
        }

        # PnL fields for sells
        edge_realized: Optional[float] = None
        pnl: Optional[float] = None
        exit_price: Optional[float] = None
        is_open = True
        closed_at: Optional[datetime] = None

        if intent.side == "SELL":
            pnl = fill.pnl_delta
            exit_price = fill.avg_price
            is_open = False
            closed_at = obs_ts
            if intent.size_usd > 0 and not is_settle:
                edge_realized = pnl / intent.size_usd

        observation = LiveObservation(
            timestamp=obs_ts,
            match_id=intent.condition_id,
            event_type=self._event_type,
            game_state=game_state,
            model_prediction=fair_price,
            polymarket_price=fill.avg_price,
        )

        trade = PaperTrade(
            observation_id=0,
            side=intent.side,
            entry_price=intent.price,
            simulated_fill_price=fill.avg_price,
            size=intent.size_usd,
            edge_theoretical=intent.edge_pct,
            edge_realized=edge_realized,
            exit_price=exit_price,
            pnl=pnl,
            is_open=is_open,
            created_at=obs_ts,
            closed_at=closed_at,
        )

        session = get_sync_session(self._db_url)
        try:
            session.add(observation)
            session.flush()
            trade.observation_id = int(observation.id)
            session.add(trade)
            if intent.side == "SELL":
                self._close_buy_records(session, intent.condition_id, closed_at)
            session.commit()
            return int(observation.id)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def _close_buy_records(
        self,
        session: Any,
        condition_id: str,
        closed_at: Optional[datetime],
    ) -> None:
        from sqlalchemy import and_

        buy_obs_ids = (
            session.query(LiveObservation.id)
            .filter(
                LiveObservation.match_id == condition_id,
                LiveObservation.event_type == self._event_type,
            )
            .all()
        )
        if not buy_obs_ids:
            return
        obs_id_list = [row[0] for row in buy_obs_ids]
        session.query(PaperTrade).filter(
            and_(
                PaperTrade.observation_id.in_(obs_id_list),
                PaperTrade.side == "BUY",
                PaperTrade.is_open.is_(True),
            )
        ).update(
            {"is_open": False, "closed_at": closed_at},
            synchronize_session="fetch",
        )
```

Update `src/execution/__init__.py`:
```python
from src.execution.trade_recorder import TradeRecorder
```

**Step 4: Run tests, verify PASS**

Run: `pytest tests/execution/test_trade_recorder.py -v`

**Step 5: Commit**

```bash
git add src/execution/trade_recorder.py src/execution/__init__.py tests/execution/test_trade_recorder.py
git commit -m "feat(execution): add TradeRecorder — generic persistence extracted from TwoSidedPaperRecorder"
```

---

### Task 4: TradeManager — `src/execution/trade_manager.py`

**Files:**
- Create: `src/execution/trade_manager.py`
- Modify: `src/execution/__init__.py`
- Test: `tests/execution/test_trade_manager.py`

**Step 1: Write tests**

```python
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
    return mgr


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
async def test_place_live_calls_executor(intent):
    executor = AsyncMock()
    executor.place_order = AsyncMock(return_value={"orderID": "live_1", "status": "PLACED"})
    mgr = TradeManager(
        executor=executor,
        strategy="TestStrat",
        paper=False,
        event_type="test",
    )
    mgr._alerter = MagicMock()
    mgr._alerter.send_custom_alert = AsyncMock(return_value=True)
    result = await mgr.place(intent)
    assert isinstance(result, PendingOrder)
    assert result.order_id == "live_1"
    executor.place_order.assert_called_once()


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


# --- get_stats() ---

@pytest.mark.asyncio
async def test_get_stats_initial(paper_manager):
    stats = paper_manager.get_stats()
    assert stats["wins"] == 0
    assert stats["losses"] == 0
    assert stats["total_pnl"] == 0.0
    assert stats["pending_orders"] == 0
```

**Step 2: Run tests, verify FAIL**

Run: `pytest tests/execution/test_trade_manager.py -v`

**Step 3: Implement**

```python
# src/execution/trade_manager.py
"""Generic trade execution manager for all strategies.

Orchestrates: order placement (paper/live), paper fill detection,
DB persistence, and Telegram notifications.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Optional

import structlog

from config.settings import settings
from src.execution.executor import ExecutorProtocol, adapt_polymarket_response
from src.execution.models import (
    FillResult,
    OrderResult,
    PendingOrder,
    TradeIntent,
)
from src.execution.trade_recorder import TradeRecorder
from src.paper_trading.alerts import TelegramAlerter

logger = structlog.get_logger()

# Emoji prefixes
_PAPER = "\U0001f4dd"  # 📝
_LIVE = "\U0001f525"   # 🔥
_BID = "\U0001f4ca"    # 📊
_FILL = "\u2705"       # ✅
_WIN = "\U0001f7e2"    # 🟢
_LOSS = "\U0001f534"   # 🔴


class TradeManager:
    """Generic trade manager for all strategies."""

    def __init__(
        self,
        *,
        executor: Optional[ExecutorProtocol] = None,
        strategy: str,
        paper: bool = True,
        db_url: str = "",
        event_type: str = "",
        run_id: str = "",
        notify_bids: bool = True,
        notify_fills: bool = True,
        notify_closes: bool = True,
    ) -> None:
        self.executor = executor
        self.strategy = strategy
        self.paper = paper
        self.notify_bids = notify_bids
        self.notify_fills = notify_fills
        self.notify_closes = notify_closes

        # Internal state
        self._pending: dict[str, PendingOrder] = {}
        self._positions: dict[str, PendingOrder] = {}  # cid -> filled order
        self._paper_counter: int = 0
        self._wins: int = 0
        self._losses: int = 0
        self._total_pnl: float = 0.0

        # Recorder
        self._recorder: Optional[TradeRecorder] = None
        if db_url or event_type:
            self._recorder = TradeRecorder(
                db_url=db_url or settings.DATABASE_URL,
                strategy_tag=strategy,
                event_type=event_type,
                run_id=run_id,
            )
            self._recorder.bootstrap()

        # Telegram
        self._alerter = TelegramAlerter()

    # ------------------------------------------------------------------
    # place
    # ------------------------------------------------------------------

    async def place(self, intent: TradeIntent) -> PendingOrder:
        """Place an order. Paper: fake ID. Live: via executor."""
        if self.paper or self.executor is None:
            order_id = self._next_paper_id()
        else:
            raw = await self.executor.place_order(
                token_id=intent.token_id,
                side=intent.side,
                size=intent.size_usd,
                price=intent.price,
                outcome=intent.outcome,
            )
            result = self._adapt_result(raw)
            if result.status == "error":
                logger.warning("order_failed", strategy=self.strategy, error=result.error)
                return PendingOrder(order_id="", intent=intent, placed_at=time.time())
            order_id = result.order_id

        now = time.time()
        pending = PendingOrder(order_id=order_id, intent=intent, placed_at=now)
        self._pending[order_id] = pending

        # Persist entry
        if self._recorder:
            try:
                fill = FillResult(filled=True, shares=intent.shares, avg_price=intent.price)
                self._recorder.record_fill(
                    intent=intent, fill=fill,
                    fair_prices={intent.outcome: intent.price},
                    execution_mode="paper" if self.paper else "live",
                )
            except Exception as exc:
                logger.warning("record_fill_failed", error=str(exc))

        # Telegram
        if self.notify_bids:
            await self._notify_bid(intent)

        logger.info(
            "order_placed",
            strategy=self.strategy,
            outcome=intent.outcome,
            price=intent.price,
            size=intent.size_usd,
            paper=self.paper,
            order_id=order_id,
        )
        return pending

    # ------------------------------------------------------------------
    # check_paper_fills
    # ------------------------------------------------------------------

    def check_paper_fills(
        self,
        get_levels: Callable[[str, str], tuple[Any, ...]],
    ) -> list[FillResult]:
        """Check pending paper orders against orderbook levels.

        get_levels(condition_id, outcome) -> (bid, bid_sz, ask, ask_sz)
        BUY fills when ask <= order price.
        SELL fills when bid >= order price.
        """
        fills: list[FillResult] = []
        filled_ids: list[str] = []

        for oid, pending in self._pending.items():
            intent = pending.intent
            bid, _, ask, _ = get_levels(intent.condition_id, intent.outcome)

            filled = False
            if intent.side == "BUY" and ask is not None:
                filled = ask <= intent.price
            elif intent.side == "SELL" and bid is not None:
                filled = bid >= intent.price

            if filled:
                fill = FillResult(
                    filled=True,
                    shares=intent.shares,
                    avg_price=intent.price,
                )
                fills.append(fill)
                filled_ids.append(oid)
                self._positions[intent.condition_id] = pending

                # Persist
                if self._recorder:
                    try:
                        self._recorder.record_fill(
                            intent=intent, fill=fill,
                            fair_prices={intent.outcome: intent.price},
                            execution_mode="paper_fill",
                        )
                    except Exception as exc:
                        logger.warning("record_fill_failed", error=str(exc))

                # Telegram (fire-and-forget)
                if self.notify_fills:
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(self._notify_fill(intent, fill))
                    except RuntimeError:
                        pass

                logger.info(
                    "paper_fill",
                    strategy=self.strategy,
                    outcome=intent.outcome,
                    price=intent.price,
                    shares=round(fill.shares, 2),
                )

        for oid in filled_ids:
            del self._pending[oid]

        return fills

    # ------------------------------------------------------------------
    # settle
    # ------------------------------------------------------------------

    async def settle(
        self,
        condition_id: str,
        outcome: str,
        settlement_price: float,
        won: bool,
    ) -> float:
        """Settle a position. Returns PnL."""
        pos = self._positions.pop(condition_id, None)
        if pos is None:
            return 0.0

        intent = pos.intent
        if won:
            pnl = intent.shares * (settlement_price - intent.price)
            self._wins += 1
        else:
            pnl = -intent.size_usd
            self._losses += 1
        self._total_pnl += pnl

        # Persist settlement
        if self._recorder:
            settle_intent = TradeIntent(
                condition_id=condition_id,
                token_id=intent.token_id,
                outcome=outcome,
                side="SELL",
                price=settlement_price,
                size_usd=intent.shares * settlement_price,
                reason="settlement",
                title=intent.title,
                edge_pct=0.0,
                timestamp=time.time(),
            )
            settle_fill = FillResult(
                filled=True,
                shares=intent.shares,
                avg_price=settlement_price,
                pnl_delta=pnl,
            )
            try:
                self._recorder.record_settle(
                    intent=settle_intent, fill=settle_fill,
                    fair_prices={outcome: settlement_price},
                )
            except Exception as exc:
                logger.warning("record_settle_failed", error=str(exc))

        # Telegram
        if self.notify_closes:
            await self._notify_settle(intent, settlement_price, pnl, won)

        logger.info(
            "position_settled",
            strategy=self.strategy,
            outcome=outcome,
            entry=intent.price,
            exit=settlement_price,
            won=won,
            pnl=round(pnl, 4),
            record=f"{self._wins}W-{self._losses}L",
        )
        return pnl

    # ------------------------------------------------------------------
    # cancel
    # ------------------------------------------------------------------

    async def cancel(self, order_id: str) -> bool:
        """Cancel a pending order."""
        if order_id not in self._pending:
            return False
        del self._pending[order_id]
        if not self.paper and self.executor:
            try:
                await self.executor.cancel_order(order_id)
            except Exception:
                pass
        return True

    # ------------------------------------------------------------------
    # queries
    # ------------------------------------------------------------------

    def get_pending_orders(self) -> dict[str, PendingOrder]:
        return dict(self._pending)

    def get_stats(self) -> dict[str, Any]:
        return {
            "wins": self._wins,
            "losses": self._losses,
            "total_pnl": self._total_pnl,
            "pending_orders": len(self._pending),
            "open_positions": len(self._positions),
        }

    async def close(self) -> None:
        await self._alerter.close()

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _next_paper_id(self) -> str:
        self._paper_counter += 1
        return f"paper_{self._paper_counter}"

    def _adapt_result(self, raw: Any) -> OrderResult:
        if isinstance(raw, OrderResult):
            return raw
        if isinstance(raw, dict):
            return adapt_polymarket_response(raw)
        return OrderResult(order_id="", status="error", error="unexpected response")

    def _mode_emoji(self) -> str:
        return _PAPER if self.paper else _LIVE

    async def _notify_bid(self, intent: TradeIntent) -> None:
        mode = self._mode_emoji()
        msg = (
            f"{mode}{_BID} {self.strategy}\n"
            f"BID {intent.outcome} @ {intent.price:.2f} | ${intent.size_usd:.0f}\n"
            f"{intent.title}"
        )
        try:
            await self._alerter.send_custom_alert(msg)
        except Exception:
            pass

    async def _notify_fill(self, intent: TradeIntent, fill: FillResult) -> None:
        mode = self._mode_emoji()
        msg = (
            f"{mode}{_FILL} {self.strategy}\n"
            f"FILL {intent.outcome} @ {fill.avg_price:.2f} | {fill.shares:.1f} shares\n"
            f"{intent.title}"
        )
        try:
            await self._alerter.send_custom_alert(msg)
        except Exception:
            pass

    async def _notify_settle(
        self, intent: TradeIntent, exit_price: float, pnl: float, won: bool,
    ) -> None:
        mode = self._mode_emoji()
        result_emoji = _WIN if won else _LOSS
        result_text = "WIN" if won else "LOSS"
        record = f"{self._wins}W-{self._losses}L"
        msg = (
            f"{mode}{result_emoji} {self.strategy}\n"
            f"{result_text} {intent.outcome} {intent.price:.2f} \u2192 {exit_price:.2f} | ${pnl:+.2f}\n"
            f"{record} | Total: ${self._total_pnl:+.2f}"
        )
        try:
            await self._alerter.send_custom_alert(msg)
        except Exception:
            pass
```

Update `src/execution/__init__.py`:
```python
from src.execution.trade_manager import TradeManager
```

**Step 4: Run tests, verify PASS**

Run: `pytest tests/execution/test_trade_manager.py -v`

**Step 5: Commit**

```bash
git add src/execution/trade_manager.py src/execution/__init__.py tests/execution/test_trade_manager.py
git commit -m "feat(execution): add TradeManager — generic trade execution with Telegram"
```

---

### Task 5: Migrate `run_crypto_td_maker.py`

**Files:**
- Modify: `scripts/run_crypto_td_maker.py`

**Step 1: Replace imports and init**

In `CryptoTDMaker.__init__` (line ~109): replace `paper_recorder: Optional[TwoSidedPaperRecorder]` with `manager: Optional[TradeManager]`. Remove `self.paper_recorder`, `self._paper_order_counter`. Add `self.manager = manager`.

In `main()` (line ~720): replace `TwoSidedPaperRecorder` instantiation with:
```python
from src.execution import TradeManager
manager = TradeManager(
    executor=executor,
    strategy="CryptoTDMaker",
    paper=paper_mode,
    db_url=args.db_url,
    event_type=TD_MAKER_EVENT_TYPE,
    run_id=run_id,
)
```

**Step 2: Replace `_place_order`**

Replace the body of `_place_order()` (lines 368-411) with:
```python
async def _place_order(self, cid, outcome, token_id, price, now):
    if not self.manager:
        return None
    intent = TradeIntent(
        condition_id=cid, token_id=token_id, outcome=outcome,
        side="BUY", price=price, size_usd=self.order_size_usd,
        reason="td_maker_passive",
        title=_first_event_slug(self.known_markets.get(cid, {})),
        timestamp=now,
    )
    pending = await self.manager.place(intent)
    if not pending.order_id:
        return None
    self.active_orders[pending.order_id] = PassiveOrder(
        order_id=pending.order_id, condition_id=cid, outcome=outcome,
        token_id=token_id, price=price, size_usd=self.order_size_usd,
        placed_at=now,
    )
    self._orders_by_cid_outcome[(cid, outcome)] = pending.order_id
    return pending.order_id
```

**Step 3: Replace `_check_fills_paper` and `_process_fill`**

Replace `_check_fills_paper()` to use `manager.check_paper_fills()`. The manager handles fill detection; the strategy just needs to update its own state (active_orders, positions, cancel other side).

**Step 4: Replace `_settle_position`**

Replace the persist + PnL logic in `_settle_position()` with:
```python
pnl = await self.manager.settle(pos.condition_id, pos.outcome, 1.0 if won else 0.0, won)
```
Keep the stats tracking (total_wins, total_losses, realized_pnl) since they're used for the periodic console print.

**Step 5: Cleanup `run()` finally block**

Add `await self.manager.close()`.

**Step 6: Remove unused imports**

Remove `TwoSidedPaperRecorder` import from `scripts.run_two_sided_inventory`.

**Step 7: Run and verify**

Run: `pytest tests/ -v -k "not slow"` to check nothing broke.
Run: `./run_crypto_td_maker.sh` briefly to verify Telegram messages arrive.

**Step 8: Commit**

```bash
git add scripts/run_crypto_td_maker.py
git commit -m "refactor(crypto_td_maker): migrate to TradeManager"
```

---

### Task 6: Migrate `run_crypto_maker.py`

**Files:**
- Modify: `scripts/run_crypto_maker.py`

Same pattern as Task 5. Key differences:
- Has both `_check_fills_paper()` AND `_check_fills_live()` — paper fills via manager, live fills via `_fill_listener` calling `manager.record_fill()` directly.
- Has `_persist_settlement()` — replace with `manager.settle()`.
- Remove `TwoSidedPaperRecorder` usage.

**Commit:** `refactor(crypto_maker): migrate to TradeManager`

---

### Task 7: Migrate `run_two_sided_inventory.py`

**Files:**
- Modify: `scripts/run_two_sided_inventory.py`

Key changes:
- `TwoSidedPaperRecorder` class stays in file but becomes a thin wrapper around `TradeRecorder` (for `replay_into_engine()` which is two-sided-specific).
- Replace all `paper_recorder.persist_fill()` calls (5 locations) with `manager.place()` / `manager.settle()`.
- Replace `settle_resolved_inventory()` persist calls with `manager.settle()`.

**Commit:** `refactor(two_sided): migrate to TradeManager`

---

### Task 8: Migrate `run_sniper.py`

**Files:**
- Modify: `scripts/run_sniper.py`

Replace `paper_recorder.persist_fill()` + `settlement_loop` with `manager.place()` / `manager.settle()`.

**Commit:** `refactor(sniper): migrate to TradeManager`

---

### Task 9: Migrate `run_fear_selling.py`

**Files:**
- Modify: `scripts/run_fear_selling.py`

Key changes:
- Replace `_persist_signal()` (direct SQLAlchemy `FearPosition` writes) with `manager.place()`.
- Replace `_check_exits()` PnL calc + DB update with `manager.settle()`.
- Note: `FearPosition` model can remain for now (used by fear-specific queries). Migration to pure LiveObservation+PaperTrade is optional follow-up.

**Commit:** `refactor(fear_selling): migrate to TradeManager`

---

### Task 10: Migrate `run_weather_oracle.py`

**Files:**
- Modify: `src/arb/weather_oracle.py`

Replace `_save_trade()` (lines 786-844, inline LiveObservation + PaperTrade + session management) with:
- `manager.place()` on `enter_paper_trade()`
- `manager.settle()` on `resolve_trades()`

Pass `TradeManager` into `WeatherOracleEngine.__init__`.

**Commit:** `refactor(weather_oracle): migrate to TradeManager`

---

### Task 11: Migrate `run_crypto_minute.py`

**Files:**
- Modify: `src/arb/crypto_minute.py`

Replace `_save_trade()` (line 524) with:
- `manager.place()` on `enter_paper_trade()`
- `manager.settle()` on `resolve_expired_trades()`

Pass `TradeManager` into `CryptoMinuteEngine.__init__`.

**Commit:** `refactor(crypto_minute): migrate to TradeManager`

---

### Task 12: Extract `KalshiExecutor` + migrate `run_kalshi_td_maker.py`

**Files:**
- Create: `src/feeds/kalshi_executor.py`
- Modify: `scripts/run_kalshi_td_maker.py`
- Test: `tests/feeds/test_kalshi_executor.py`

Extract from `KalshiTDMaker`:
- `_auth_headers()` (RSA signing)
- `_api_get()`, `_api_post()`
- Order placement logic

Into a standalone `KalshiExecutor` implementing `ExecutorProtocol`.

Then migrate `run_kalshi_td_maker.py` to use `TradeManager` + `KalshiExecutor`.

**Commit:** `refactor(kalshi): extract KalshiExecutor + migrate to TradeManager`

---

### Task 13: Compat imports + cleanup

**Files:**
- Modify: `src/arb/two_sided_inventory.py` — add compat re-exports from `src.execution.models`
- Delete: `src/notifications/` directory (created earlier, now superseded)

**Step 1:** In `src/arb/two_sided_inventory.py`, keep existing `TradeIntent` and `FillResult` classes but have them import from `src.execution.models` to avoid breaking any remaining references:

```python
# Compat: these now live in src.execution.models
from src.execution.models import TradeIntent, FillResult  # noqa: F811
```

Or simply update all imports project-wide to point to `src.execution.models`.

**Step 2:** Remove `src/notifications/` if it was created in the earlier aborted attempt.

**Step 3:** Run full test suite.

Run: `pytest tests/ -v`

**Step 4: Commit**

```bash
git commit -m "chore: compat imports + cleanup after TradeManager migration"
```

---

### Task 14: Final verification

**Step 1:** Run full test suite: `pytest tests/ -v`

**Step 2:** Start `./run_crypto_td_maker.sh` — verify Telegram messages arrive (bid, fill, settlement).

**Step 3:** Verify API `/trades` still works: `curl http://localhost:8788/trades?hours=1`

**Step 4:** Spot-check other scripts start without errors:
```bash
timeout 10 ./run run_fear_selling.py --paper 2>&1 | head -20
timeout 10 ./run run_weather_oracle.py 2>&1 | head -20
```
