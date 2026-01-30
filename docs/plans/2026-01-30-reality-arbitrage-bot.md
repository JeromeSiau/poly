# Reality Arbitrage Bot — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a bot that exploits broadcast lag on Polymarket sports/esports markets by accessing real-time game data before it's reflected in market prices.

**Architecture:** Three-layer system: (1) Real-time data feeds from game APIs (PandaScore for esports, SportsDataIO for traditional sports), (2) Polymarket WebSocket connection for live order book, (3) Arbitrage engine that detects when real-world events haven't been priced in yet and executes trades within the 15-40 second window.

**Tech Stack:** Python 3.11+, asyncio, websockets, PandaScore API, SportsDataIO API, Polymarket CLOB API, SQLite, Telegram for alerts

---

## Phase Overview

| Phase | Focus | Priority | Estimated Edge |
|-------|-------|----------|----------------|
| **Phase 1** | Esports (LoL, Dota 2, CS:GO) | HIGH | 30-40s advantage |
| **Phase 2** | Traditional Sports | MEDIUM | 15-40s advantage |
| **Phase 3** | Cross-platform Arb (existing design) | LOW | Price discrepancy |

---

## Project Structure (Extended)

```
poly/
├── config/
│   ├── settings.py
│   └── settings.example.py
├── src/
│   ├── feeds/
│   │   ├── base.py                 # Abstract base feed
│   │   ├── polymarket.py           # Polymarket WebSocket
│   │   ├── betfair.py              # (Existing design)
│   │   ├── pandascore.py           # NEW: Esports real-time
│   │   └── sportsdataio.py         # NEW: Sports real-time
│   ├── realtime/                   # NEW MODULE
│   │   ├── __init__.py
│   │   ├── event_detector.py       # Detects game events
│   │   ├── market_mapper.py        # Maps events to markets
│   │   └── lag_calculator.py       # Estimates broadcast delay
│   ├── matching/
│   │   ├── matcher.py
│   │   └── llm_verifier.py
│   ├── arb/
│   │   ├── detector.py
│   │   ├── calculator.py
│   │   ├── executor.py
│   │   └── reality_arb.py          # NEW: Broadcast lag arb
│   ├── risk/
│   │   ├── manager.py
│   │   └── monitor.py
│   ├── bot/
│   │   ├── telegram.py
│   │   └── handlers.py
│   ├── dashboard/
│   │   ├── app.py
│   │   └── templates/
│   └── db/
│       ├── models.py
│       └── database.py
├── tests/
│   ├── feeds/
│   │   ├── test_pandascore.py
│   │   ├── test_sportsdataio.py
│   │   └── test_polymarket.py
│   ├── realtime/
│   │   ├── test_event_detector.py
│   │   ├── test_market_mapper.py
│   │   └── test_lag_calculator.py
│   └── arb/
│       └── test_reality_arb.py
├── scripts/
│   ├── run_bot.py
│   ├── run_reality_arb.py          # NEW
│   └── backfill_events.py
├── data/
│   └── arb.db
├── requirements.txt
└── README.md
```

---

## Task 1: Project Setup & Dependencies

**Files:**
- Create: `requirements.txt`
- Create: `config/settings.example.py`
- Create: `config/settings.py` (gitignored)

**Step 1: Create requirements.txt**

```txt
# Core
python-dotenv==1.0.0
pydantic==2.5.0
pydantic-settings==2.1.0

# Async
aiohttp==3.9.1
websockets==12.0
asyncio==3.4.3

# APIs
py-clob-client==0.1.0  # Polymarket

# Database
sqlalchemy==2.0.23
aiosqlite==0.19.0

# Telegram
python-telegram-bot==20.7

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-cov==4.1.0
respx==0.20.2  # Mock HTTP

# Utils
structlog==23.2.0
```

**Step 2: Create config/settings.example.py**

```python
"""Configuration template - copy to settings.py and fill in values."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # === Polymarket ===
    POLYMARKET_API_KEY: str = ""
    POLYMARKET_API_SECRET: str = ""
    POLYMARKET_WALLET_ADDRESS: str = ""
    POLYMARKET_PRIVATE_KEY: str = ""

    # WebSocket endpoints
    POLYMARKET_CLOB_HTTP: str = "https://clob.polymarket.com"
    POLYMARKET_CLOB_WS: str = "wss://ws-subscriptions-clob.polymarket.com/ws/"
    POLYMARKET_RTDS_WS: str = "wss://ws-live-data.polymarket.com"

    # === PandaScore (Esports) ===
    PANDASCORE_API_KEY: str = ""
    PANDASCORE_BASE_URL: str = "https://api.pandascore.co"

    # === SportsDataIO ===
    SPORTSDATAIO_API_KEY: str = ""
    SPORTSDATAIO_BASE_URL: str = "https://api.sportsdata.io/v3"

    # === Telegram ===
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""

    # === Risk Parameters ===
    MAX_POSITION_PCT: float = 0.10  # 10% of capital
    DAILY_LOSS_LIMIT_PCT: float = 0.05  # 5% daily loss cap
    MIN_EDGE_PCT: float = 0.02  # 2% minimum edge
    ANOMALY_THRESHOLD_PCT: float = 0.15  # 15% edge = suspicious
    MIN_BROADCAST_LAG_SECONDS: float = 5.0  # Min lag to trade
    MAX_BROADCAST_LAG_SECONDS: float = 60.0  # Max lag (stale data?)

    # === Execution ===
    ORDER_TIMEOUT_SECONDS: float = 5.0
    MAX_TRADES_PER_HOUR: int = 20
    AUTOPILOT_MODE: bool = False

    # === Database ===
    DATABASE_URL: str = "sqlite+aiosqlite:///data/arb.db"

    class Config:
        env_file = ".env"


settings = Settings()
```

**Step 3: Run tests (placeholder)**

```bash
echo "No tests yet - setup complete"
```

**Step 4: Commit**

```bash
git add requirements.txt config/settings.example.py .gitignore
git commit -m "feat: project setup with dependencies and config template"
```

---

## Task 2: Database Models

**Files:**
- Create: `src/db/__init__.py`
- Create: `src/db/models.py`
- Create: `src/db/database.py`
- Create: `tests/db/test_models.py`

**Step 1: Write the failing test**

```python
# tests/db/test_models.py
import pytest
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.models import Base, GameEvent, Market, Trade, Position


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_game_event_creation(db_session):
    event = GameEvent(
        external_id="pandascore_12345",
        source="pandascore",
        game="lol",
        event_type="kill",
        team="T1",
        player="Faker",
        timestamp=datetime.utcnow(),
        raw_data={"kills": 5, "deaths": 1}
    )
    db_session.add(event)
    db_session.commit()

    assert event.id is not None
    assert event.game == "lol"
    assert event.event_type == "kill"


def test_market_creation(db_session):
    market = Market(
        polymarket_id="0x123abc",
        title="T1 vs Gen.G - Winner",
        game="lol",
        event_name="LCK Spring 2026",
        outcomes=["T1", "Gen.G"],
        current_prices={"T1": 0.65, "Gen.G": 0.35}
    )
    db_session.add(market)
    db_session.commit()

    assert market.id is not None
    assert market.outcomes == ["T1", "Gen.G"]


def test_trade_creation(db_session):
    trade = Trade(
        market_id="0x123abc",
        side="BUY",
        outcome="T1",
        price=0.55,
        size=100.0,
        edge_pct=0.08,
        trigger_event="kill",
        status="FILLED",
        execution_time_ms=120
    )
    db_session.add(trade)
    db_session.commit()

    assert trade.id is not None
    assert trade.edge_pct == 0.08
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/db/test_models.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'src.db.models'"

**Step 3: Write the implementation**

```python
# src/db/__init__.py
from .models import Base, GameEvent, Market, Trade, Position
from .database import get_session, init_db

__all__ = ["Base", "GameEvent", "Market", "Trade", "Position", "get_session", "init_db"]
```

```python
# src/db/models.py
from datetime import datetime
from typing import Optional
from sqlalchemy import String, Float, Integer, DateTime, JSON, Enum as SQLEnum
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
import enum


class Base(DeclarativeBase):
    pass


class TradeStatus(enum.Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class GameEvent(Base):
    """Real-time game events from data feeds."""
    __tablename__ = "game_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    external_id: Mapped[str] = mapped_column(String(100), unique=True)
    source: Mapped[str] = mapped_column(String(50))  # pandascore, sportsdataio
    game: Mapped[str] = mapped_column(String(50))  # lol, dota2, csgo, nba, nfl
    event_type: Mapped[str] = mapped_column(String(50))  # kill, tower, goal, score
    team: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    player: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    raw_data: Mapped[dict] = mapped_column(JSON, default=dict)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Market(Base):
    """Polymarket markets we're tracking."""
    __tablename__ = "markets"

    id: Mapped[int] = mapped_column(primary_key=True)
    polymarket_id: Mapped[str] = mapped_column(String(100), unique=True)
    title: Mapped[str] = mapped_column(String(500))
    game: Mapped[str] = mapped_column(String(50))
    event_name: Mapped[str] = mapped_column(String(200))
    outcomes: Mapped[list] = mapped_column(JSON)
    current_prices: Mapped[dict] = mapped_column(JSON, default=dict)

    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Trade(Base):
    """Executed trades."""
    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(primary_key=True)
    market_id: Mapped[str] = mapped_column(String(100))
    side: Mapped[str] = mapped_column(String(10))  # BUY, SELL
    outcome: Mapped[str] = mapped_column(String(100))
    price: Mapped[float] = mapped_column(Float)
    size: Mapped[float] = mapped_column(Float)
    edge_pct: Mapped[float] = mapped_column(Float)
    trigger_event: Mapped[str] = mapped_column(String(100))
    status: Mapped[str] = mapped_column(String(20), default="PENDING")
    execution_time_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    polymarket_order_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Position(Base):
    """Open and closed positions."""
    __tablename__ = "positions"

    id: Mapped[int] = mapped_column(primary_key=True)
    market_id: Mapped[str] = mapped_column(String(100))
    outcome: Mapped[str] = mapped_column(String(100))
    entry_price: Mapped[float] = mapped_column(Float)
    current_price: Mapped[float] = mapped_column(Float)
    size: Mapped[float] = mapped_column(Float)
    realized_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    unrealized_pnl: Mapped[float] = mapped_column(Float, default=0.0)

    is_open: Mapped[bool] = mapped_column(default=True)
    opened_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    closed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
```

```python
# src/db/database.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from contextlib import asynccontextmanager

from config.settings import settings
from .models import Base


engine = create_async_engine(settings.DATABASE_URL, echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db():
    """Create all tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@asynccontextmanager
async def get_session():
    """Get an async database session."""
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/db/test_models.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/db/ tests/db/
git commit -m "feat: database models for game events, markets, trades, positions"
```

---

## Task 3: Abstract Base Feed

**Files:**
- Create: `src/feeds/__init__.py`
- Create: `src/feeds/base.py`
- Create: `tests/feeds/test_base.py`

**Step 1: Write the failing test**

```python
# tests/feeds/test_base.py
import pytest
from abc import ABC
from src.feeds.base import BaseFeed, FeedEvent


def test_feed_event_creation():
    event = FeedEvent(
        source="test",
        event_type="kill",
        game="lol",
        data={"player": "Faker", "kills": 5},
        timestamp=1234567890.0
    )
    assert event.source == "test"
    assert event.data["player"] == "Faker"


def test_base_feed_is_abstract():
    with pytest.raises(TypeError):
        BaseFeed()  # Cannot instantiate abstract class


class MockFeed(BaseFeed):
    async def connect(self):
        self._connected = True

    async def disconnect(self):
        self._connected = False

    async def subscribe(self, game: str, match_id: str):
        pass


@pytest.mark.asyncio
async def test_mock_feed_connects():
    feed = MockFeed()
    await feed.connect()
    assert feed._connected is True
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/feeds/test_base.py -v
```

Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write the implementation**

```python
# src/feeds/__init__.py
from .base import BaseFeed, FeedEvent

__all__ = ["BaseFeed", "FeedEvent"]
```

```python
# src/feeds/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from datetime import datetime
import asyncio


@dataclass
class FeedEvent:
    """Standardized event from any data feed."""
    source: str  # pandascore, sportsdataio, polymarket
    event_type: str  # kill, tower, goal, price_update, etc.
    game: str  # lol, dota2, csgo, nba, nfl
    data: dict[str, Any]
    timestamp: float  # Unix timestamp
    match_id: Optional[str] = None

    @property
    def age_seconds(self) -> float:
        """How old is this event?"""
        return datetime.utcnow().timestamp() - self.timestamp


class BaseFeed(ABC):
    """Abstract base class for all data feeds."""

    def __init__(self):
        self._connected: bool = False
        self._callbacks: list[Callable[[FeedEvent], None]] = []
        self._subscriptions: set[tuple[str, str]] = set()  # (game, match_id)

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the data source."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the data source."""
        pass

    @abstractmethod
    async def subscribe(self, game: str, match_id: str) -> None:
        """Subscribe to events for a specific match."""
        pass

    def on_event(self, callback: Callable[[FeedEvent], None]) -> None:
        """Register a callback for incoming events."""
        self._callbacks.append(callback)

    async def _emit(self, event: FeedEvent) -> None:
        """Emit an event to all registered callbacks."""
        for callback in self._callbacks:
            if asyncio.iscoroutinefunction(callback):
                await callback(event)
            else:
                callback(event)

    @property
    def is_connected(self) -> bool:
        return self._connected
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/feeds/test_base.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/feeds/ tests/feeds/
git commit -m "feat: abstract base feed class with event system"
```

---

## Task 4: PandaScore Esports Feed (HIGH PRIORITY)

**Files:**
- Create: `src/feeds/pandascore.py`
- Create: `tests/feeds/test_pandascore.py`

**Step 1: Write the failing test**

```python
# tests/feeds/test_pandascore.py
import pytest
import respx
from httpx import Response
from datetime import datetime

from src.feeds.pandascore import PandaScoreFeed, PandaScoreEvent


@pytest.fixture
def mock_api():
    with respx.mock:
        yield respx


class TestPandaScoreEventParsing:
    def test_parse_lol_kill_event(self):
        raw_data = {
            "type": "kill",
            "timestamp": 1234567890,
            "payload": {
                "killer": {"name": "Faker", "team": "T1"},
                "victim": {"name": "Chovy", "team": "Gen.G"},
                "assists": [{"name": "Keria"}]
            }
        }

        event = PandaScoreEvent.from_raw("lol", "match_123", raw_data)

        assert event.event_type == "kill"
        assert event.game == "lol"
        assert event.data["killer"] == "Faker"
        assert event.data["killer_team"] == "T1"

    def test_parse_lol_tower_event(self):
        raw_data = {
            "type": "tower_destroyed",
            "timestamp": 1234567891,
            "payload": {
                "team": "T1",
                "tower": "top_outer"
            }
        }

        event = PandaScoreEvent.from_raw("lol", "match_123", raw_data)

        assert event.event_type == "tower_destroyed"
        assert event.data["team"] == "T1"

    def test_parse_csgo_round_event(self):
        raw_data = {
            "type": "round_end",
            "timestamp": 1234567892,
            "payload": {
                "winner": "Navi",
                "score": {"Navi": 10, "FaZe": 8}
            }
        }

        event = PandaScoreEvent.from_raw("csgo", "match_456", raw_data)

        assert event.event_type == "round_end"
        assert event.data["winner"] == "Navi"


class TestPandaScoreFeed:
    @pytest.mark.asyncio
    async def test_fetch_live_matches(self, mock_api):
        mock_api.get("https://api.pandascore.co/lol/matches/running").mock(
            return_value=Response(200, json=[
                {
                    "id": 123,
                    "name": "T1 vs Gen.G",
                    "league": {"name": "LCK"},
                    "status": "running"
                }
            ])
        )

        feed = PandaScoreFeed(api_key="test_key")
        matches = await feed.get_live_matches("lol")

        assert len(matches) == 1
        assert matches[0]["name"] == "T1 vs Gen.G"

    @pytest.mark.asyncio
    async def test_latency_measurement(self, mock_api):
        mock_api.get("https://api.pandascore.co/lol/matches/123").mock(
            return_value=Response(200, json={"id": 123, "status": "running"})
        )

        feed = PandaScoreFeed(api_key="test_key")
        latency_ms = await feed.measure_latency()

        assert latency_ms > 0
        assert latency_ms < 5000  # Should be under 5 seconds
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/feeds/test_pandascore.py -v
```

Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write the implementation**

```python
# src/feeds/pandascore.py
"""PandaScore API feed for esports real-time data.

PandaScore provides ~300ms latency from stream for LoL, Dota 2, CS:GO.
This gives us 30-40 seconds advantage over Twitch/YouTube viewers.

Docs: https://developers.pandascore.co/
"""

import asyncio
import time
from typing import Optional
from dataclasses import dataclass
import aiohttp
import structlog

from .base import BaseFeed, FeedEvent
from config.settings import settings


logger = structlog.get_logger()


@dataclass
class PandaScoreEvent(FeedEvent):
    """Event parsed from PandaScore API."""

    @classmethod
    def from_raw(cls, game: str, match_id: str, raw: dict) -> "PandaScoreEvent":
        """Parse raw PandaScore event into standardized format."""
        event_type = raw.get("type", "unknown")
        timestamp = raw.get("timestamp", time.time())
        payload = raw.get("payload", {})

        # Normalize data based on event type
        data = {}

        if event_type == "kill":
            killer = payload.get("killer", {})
            victim = payload.get("victim", {})
            data = {
                "killer": killer.get("name"),
                "killer_team": killer.get("team"),
                "victim": victim.get("name"),
                "victim_team": victim.get("team"),
                "assists": [a.get("name") for a in payload.get("assists", [])]
            }
        elif event_type == "tower_destroyed":
            data = {
                "team": payload.get("team"),
                "tower": payload.get("tower")
            }
        elif event_type == "dragon_kill":
            data = {
                "team": payload.get("team"),
                "dragon_type": payload.get("dragon_type")
            }
        elif event_type == "baron_kill":
            data = {
                "team": payload.get("team")
            }
        elif event_type == "round_end":
            data = {
                "winner": payload.get("winner"),
                "score": payload.get("score", {})
            }
        elif event_type == "map_end":
            data = {
                "winner": payload.get("winner"),
                "final_score": payload.get("score", {})
            }
        else:
            data = payload

        return cls(
            source="pandascore",
            event_type=event_type,
            game=game,
            data=data,
            timestamp=timestamp,
            match_id=match_id
        )


class PandaScoreFeed(BaseFeed):
    """Real-time esports data feed from PandaScore.

    Supported games: lol, dota2, csgo, valorant
    Latency: ~300ms from actual game events
    """

    SUPPORTED_GAMES = ["lol", "dota2", "csgo", "valorant"]
    BASE_URL = "https://api.pandascore.co"

    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        self.api_key = api_key or settings.PANDASCORE_API_KEY
        self._session: Optional[aiohttp.ClientSession] = None
        self._poll_task: Optional[asyncio.Task] = None
        self._poll_interval: float = 0.5  # 500ms polling

    async def connect(self) -> None:
        """Initialize HTTP session."""
        if self._session is None:
            self._session = aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
        self._connected = True
        logger.info("pandascore_connected")

    async def disconnect(self) -> None:
        """Close HTTP session and stop polling."""
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass

        if self._session:
            await self._session.close()
            self._session = None

        self._connected = False
        logger.info("pandascore_disconnected")

    async def subscribe(self, game: str, match_id: str) -> None:
        """Start polling for events on a specific match."""
        if game not in self.SUPPORTED_GAMES:
            raise ValueError(f"Unsupported game: {game}. Supported: {self.SUPPORTED_GAMES}")

        self._subscriptions.add((game, match_id))
        logger.info("pandascore_subscribed", game=game, match_id=match_id)

        # Start polling if not already running
        if self._poll_task is None or self._poll_task.done():
            self._poll_task = asyncio.create_task(self._poll_loop())

    async def get_live_matches(self, game: str) -> list[dict]:
        """Fetch currently running matches for a game."""
        if not self._session:
            await self.connect()

        url = f"{self.BASE_URL}/{game}/matches/running"
        async with self._session.get(url) as resp:
            if resp.status == 200:
                return await resp.json()
            else:
                logger.error("pandascore_api_error", status=resp.status, url=url)
                return []

    async def get_match_events(self, game: str, match_id: str, since: Optional[float] = None) -> list[dict]:
        """Fetch events for a specific match."""
        if not self._session:
            await self.connect()

        url = f"{self.BASE_URL}/{game}/matches/{match_id}/frames"
        params = {}
        if since:
            params["since"] = int(since)

        async with self._session.get(url, params=params) as resp:
            if resp.status == 200:
                return await resp.json()
            else:
                logger.error("pandascore_api_error", status=resp.status, url=url)
                return []

    async def measure_latency(self) -> float:
        """Measure API latency in milliseconds."""
        if not self._session:
            await self.connect()

        url = f"{self.BASE_URL}/lol/matches/running"
        start = time.perf_counter()
        async with self._session.get(url) as resp:
            await resp.read()
        end = time.perf_counter()

        latency_ms = (end - start) * 1000
        logger.info("pandascore_latency", latency_ms=latency_ms)
        return latency_ms

    async def _poll_loop(self) -> None:
        """Continuously poll for new events."""
        last_event_times: dict[tuple[str, str], float] = {}

        while self._subscriptions:
            for game, match_id in list(self._subscriptions):
                try:
                    since = last_event_times.get((game, match_id))
                    events = await self.get_match_events(game, match_id, since)

                    for raw_event in events:
                        event = PandaScoreEvent.from_raw(game, match_id, raw_event)
                        await self._emit(event)
                        last_event_times[(game, match_id)] = event.timestamp

                except Exception as e:
                    logger.error("pandascore_poll_error", error=str(e), game=game, match_id=match_id)

            await asyncio.sleep(self._poll_interval)
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/feeds/test_pandascore.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/feeds/pandascore.py tests/feeds/test_pandascore.py
git commit -m "feat: PandaScore esports feed with LoL, Dota2, CS:GO support"
```

---

## Task 5: Polymarket WebSocket Feed

**Files:**
- Create: `src/feeds/polymarket.py`
- Create: `tests/feeds/test_polymarket.py`

**Step 1: Write the failing test**

```python
# tests/feeds/test_polymarket.py
import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch

from src.feeds.polymarket import PolymarketFeed, OrderBookUpdate


class TestOrderBookUpdate:
    def test_parse_price_update(self):
        raw = {
            "type": "price_change",
            "market_id": "0x123abc",
            "outcome": "YES",
            "price": 0.65,
            "timestamp": 1234567890
        }

        update = OrderBookUpdate.from_raw(raw)

        assert update.market_id == "0x123abc"
        assert update.outcome == "YES"
        assert update.price == 0.65

    def test_parse_trade_update(self):
        raw = {
            "type": "trade",
            "market_id": "0x123abc",
            "outcome": "YES",
            "price": 0.66,
            "size": 100.0,
            "side": "BUY",
            "timestamp": 1234567891
        }

        update = OrderBookUpdate.from_raw(raw)

        assert update.event_type == "trade"
        assert update.data["size"] == 100.0
        assert update.data["side"] == "BUY"


class TestPolymarketFeed:
    def test_initialization(self):
        feed = PolymarketFeed()
        assert feed.is_connected is False
        assert feed._local_orderbook == {}

    @pytest.mark.asyncio
    async def test_subscribe_market(self):
        feed = PolymarketFeed()
        feed._ws = AsyncMock()
        feed._connected = True

        await feed.subscribe_market("0x123abc")

        feed._ws.send.assert_called_once()
        call_args = json.loads(feed._ws.send.call_args[0][0])
        assert call_args["type"] == "subscribe"
        assert "0x123abc" in call_args["markets"]

    def test_get_best_price(self):
        feed = PolymarketFeed()
        feed._local_orderbook = {
            "0x123abc": {
                "YES": {"bids": [(0.64, 100), (0.63, 200)], "asks": [(0.66, 150)]},
                "NO": {"bids": [(0.34, 100)], "asks": [(0.36, 150)]}
            }
        }

        best_bid, best_ask = feed.get_best_prices("0x123abc", "YES")

        assert best_bid == 0.64
        assert best_ask == 0.66

    def test_calculate_implied_probability(self):
        feed = PolymarketFeed()

        # Mid price of 0.65 = 65% implied probability
        prob = feed.calculate_implied_probability(0.64, 0.66)

        assert prob == 0.65
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/feeds/test_polymarket.py -v
```

Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write the implementation**

```python
# src/feeds/polymarket.py
"""Polymarket WebSocket feed for real-time order book updates.

Endpoints:
- CLOB WS: wss://ws-subscriptions-clob.polymarket.com/ws/
- RTDS (Real-Time Data Stream): wss://ws-live-data.polymarket.com

Latency: ~100ms (much faster than REST API's 1s)
"""

import asyncio
import json
import time
from typing import Optional
from dataclasses import dataclass, field
import websockets
import structlog

from .base import BaseFeed, FeedEvent
from config.settings import settings


logger = structlog.get_logger()


@dataclass
class OrderBookUpdate(FeedEvent):
    """Parsed Polymarket order book update."""

    market_id: str = ""
    outcome: str = ""
    price: float = 0.0

    @classmethod
    def from_raw(cls, raw: dict) -> "OrderBookUpdate":
        """Parse raw WebSocket message."""
        event_type = raw.get("type", "unknown")
        market_id = raw.get("market_id", raw.get("asset_id", ""))
        outcome = raw.get("outcome", "")
        price = raw.get("price", 0.0)
        timestamp = raw.get("timestamp", time.time())

        data = {
            "price": price,
            "size": raw.get("size"),
            "side": raw.get("side"),
        }

        return cls(
            source="polymarket",
            event_type=event_type,
            game="prediction",
            data=data,
            timestamp=timestamp,
            market_id=market_id,
            outcome=outcome,
            price=price
        )


class PolymarketFeed(BaseFeed):
    """Real-time Polymarket order book feed via WebSocket.

    Maintains a local copy of the order book for fast price lookups.
    """

    def __init__(self):
        super().__init__()
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._local_orderbook: dict[str, dict] = {}  # market_id -> {outcome -> {bids, asks}}
        self._subscribed_markets: set[str] = set()
        self._keepalive_task: Optional[asyncio.Task] = None
        self._receive_task: Optional[asyncio.Task] = None

    async def connect(self) -> None:
        """Connect to Polymarket WebSocket."""
        try:
            self._ws = await websockets.connect(
                settings.POLYMARKET_CLOB_WS,
                ping_interval=10,
                ping_timeout=5
            )
            self._connected = True

            # Start background tasks
            self._keepalive_task = asyncio.create_task(self._keepalive_loop())
            self._receive_task = asyncio.create_task(self._receive_loop())

            logger.info("polymarket_ws_connected")
        except Exception as e:
            logger.error("polymarket_ws_connect_failed", error=str(e))
            raise

    async def disconnect(self) -> None:
        """Disconnect from WebSocket."""
        if self._keepalive_task:
            self._keepalive_task.cancel()
        if self._receive_task:
            self._receive_task.cancel()

        if self._ws:
            await self._ws.close()
            self._ws = None

        self._connected = False
        logger.info("polymarket_ws_disconnected")

    async def subscribe(self, game: str, match_id: str) -> None:
        """Subscribe interface (routes to subscribe_market)."""
        await self.subscribe_market(match_id)

    async def subscribe_market(self, market_id: str) -> None:
        """Subscribe to order book updates for a market."""
        if not self._ws:
            raise RuntimeError("Not connected. Call connect() first.")

        msg = json.dumps({
            "type": "subscribe",
            "markets": [market_id]
        })
        await self._ws.send(msg)
        self._subscribed_markets.add(market_id)

        # Initialize local orderbook
        if market_id not in self._local_orderbook:
            self._local_orderbook[market_id] = {
                "YES": {"bids": [], "asks": []},
                "NO": {"bids": [], "asks": []}
            }

        logger.info("polymarket_subscribed", market_id=market_id)

    async def unsubscribe_market(self, market_id: str) -> None:
        """Unsubscribe from a market."""
        if self._ws and market_id in self._subscribed_markets:
            msg = json.dumps({
                "type": "unsubscribe",
                "markets": [market_id]
            })
            await self._ws.send(msg)
            self._subscribed_markets.discard(market_id)
            logger.info("polymarket_unsubscribed", market_id=market_id)

    def get_best_prices(self, market_id: str, outcome: str) -> tuple[float, float]:
        """Get best bid and ask for an outcome.

        Returns: (best_bid, best_ask)
        """
        if market_id not in self._local_orderbook:
            return (0.0, 0.0)

        book = self._local_orderbook[market_id].get(outcome, {})
        bids = book.get("bids", [])
        asks = book.get("asks", [])

        best_bid = bids[0][0] if bids else 0.0
        best_ask = asks[0][0] if asks else 0.0

        return (best_bid, best_ask)

    def calculate_implied_probability(self, best_bid: float, best_ask: float) -> float:
        """Calculate implied probability from bid/ask spread."""
        if best_bid == 0 and best_ask == 0:
            return 0.5
        return (best_bid + best_ask) / 2

    def get_market_prices(self, market_id: str) -> dict[str, float]:
        """Get current prices for all outcomes in a market."""
        prices = {}
        for outcome in ["YES", "NO"]:
            bid, ask = self.get_best_prices(market_id, outcome)
            prices[outcome] = self.calculate_implied_probability(bid, ask)
        return prices

    async def _keepalive_loop(self) -> None:
        """Send periodic pings to keep connection alive."""
        while self._connected:
            try:
                if self._ws:
                    await self._ws.ping()
                await asyncio.sleep(10)  # Ping every 10 seconds
            except Exception as e:
                logger.error("polymarket_keepalive_error", error=str(e))
                break

    async def _receive_loop(self) -> None:
        """Receive and process WebSocket messages."""
        while self._connected and self._ws:
            try:
                message = await self._ws.recv()
                data = json.loads(message)

                # Update local orderbook
                self._update_orderbook(data)

                # Emit event
                update = OrderBookUpdate.from_raw(data)
                await self._emit(update)

            except websockets.ConnectionClosed:
                logger.warning("polymarket_ws_closed")
                self._connected = False
                break
            except Exception as e:
                logger.error("polymarket_receive_error", error=str(e))

    def _update_orderbook(self, data: dict) -> None:
        """Update local order book from WebSocket message."""
        msg_type = data.get("type")
        market_id = data.get("market_id", data.get("asset_id"))

        if not market_id or market_id not in self._local_orderbook:
            return

        if msg_type == "book_snapshot":
            # Full orderbook snapshot
            for outcome in ["YES", "NO"]:
                if outcome in data:
                    self._local_orderbook[market_id][outcome] = {
                        "bids": sorted(data[outcome].get("bids", []), reverse=True),
                        "asks": sorted(data[outcome].get("asks", []))
                    }

        elif msg_type == "book_delta":
            # Incremental update
            outcome = data.get("outcome")
            if outcome and outcome in self._local_orderbook[market_id]:
                book = self._local_orderbook[market_id][outcome]

                # Apply delta (simplified - real implementation would be more complex)
                if "bid" in data:
                    price, size = data["bid"]["price"], data["bid"]["size"]
                    book["bids"] = self._apply_delta(book["bids"], price, size, reverse=True)
                if "ask" in data:
                    price, size = data["ask"]["price"], data["ask"]["size"]
                    book["asks"] = self._apply_delta(book["asks"], price, size, reverse=False)

    def _apply_delta(self, levels: list, price: float, size: float, reverse: bool) -> list:
        """Apply a price level update to the orderbook."""
        # Remove existing level at this price
        levels = [(p, s) for p, s in levels if p != price]

        # Add new level if size > 0
        if size > 0:
            levels.append((price, size))

        # Re-sort
        return sorted(levels, reverse=reverse)
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/feeds/test_polymarket.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/feeds/polymarket.py tests/feeds/test_polymarket.py
git commit -m "feat: Polymarket WebSocket feed with local orderbook"
```

---

## Task 6: Event Detector (Reality Arbitrage Core)

**Files:**
- Create: `src/realtime/__init__.py`
- Create: `src/realtime/event_detector.py`
- Create: `tests/realtime/test_event_detector.py`

**Step 1: Write the failing test**

```python
# tests/realtime/test_event_detector.py
import pytest
from datetime import datetime

from src.feeds.base import FeedEvent
from src.realtime.event_detector import EventDetector, SignificantEvent


class TestSignificantEventClassification:
    @pytest.fixture
    def detector(self):
        return EventDetector()

    def test_lol_kill_is_significant_late_game(self, detector):
        """Kill at 35 minutes is significant (late game)."""
        event = FeedEvent(
            source="pandascore",
            event_type="kill",
            game="lol",
            data={
                "killer": "Faker",
                "killer_team": "T1",
                "game_time_minutes": 35
            },
            timestamp=datetime.utcnow().timestamp()
        )

        result = detector.classify(event)

        assert result is not None
        assert result.is_significant is True
        assert result.impact_score >= 0.5

    def test_lol_kill_not_significant_early_game(self, detector):
        """Single kill at 5 minutes is not very significant."""
        event = FeedEvent(
            source="pandascore",
            event_type="kill",
            game="lol",
            data={
                "killer": "Faker",
                "killer_team": "T1",
                "game_time_minutes": 5
            },
            timestamp=datetime.utcnow().timestamp()
        )

        result = detector.classify(event)

        assert result.impact_score < 0.3

    def test_lol_baron_is_highly_significant(self, detector):
        """Baron kill is always highly significant."""
        event = FeedEvent(
            source="pandascore",
            event_type="baron_kill",
            game="lol",
            data={"team": "T1"},
            timestamp=datetime.utcnow().timestamp()
        )

        result = detector.classify(event)

        assert result.is_significant is True
        assert result.impact_score >= 0.8

    def test_csgo_round_end_calculates_momentum(self, detector):
        """Round end with score change affects momentum."""
        event = FeedEvent(
            source="pandascore",
            event_type="round_end",
            game="csgo",
            data={
                "winner": "Navi",
                "score": {"Navi": 12, "FaZe": 8}
            },
            timestamp=datetime.utcnow().timestamp()
        )

        result = detector.classify(event)

        assert result is not None
        assert result.favored_team == "Navi"
        assert result.impact_score > 0.5


class TestEventDetectorPriceImpact:
    @pytest.fixture
    def detector(self):
        return EventDetector()

    def test_estimate_price_impact_baron(self, detector):
        """Baron should swing price by ~10-15%."""
        event = SignificantEvent(
            original_event=None,
            is_significant=True,
            impact_score=0.85,
            favored_team="T1",
            event_description="Baron kill by T1"
        )

        current_price = 0.50  # 50/50 game
        estimated_new_price = detector.estimate_price_impact(event, current_price)

        # Baron should push price up by ~10-15%
        assert estimated_new_price >= 0.60
        assert estimated_new_price <= 0.70

    def test_estimate_price_impact_late_kill(self, detector):
        """Late game kill should have smaller impact."""
        event = SignificantEvent(
            original_event=None,
            is_significant=True,
            impact_score=0.55,
            favored_team="T1",
            event_description="Kill by Faker"
        )

        current_price = 0.60
        estimated_new_price = detector.estimate_price_impact(event, current_price)

        # Should increase slightly
        assert estimated_new_price > 0.60
        assert estimated_new_price < 0.70
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/realtime/test_event_detector.py -v
```

Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write the implementation**

```python
# src/realtime/__init__.py
from .event_detector import EventDetector, SignificantEvent

__all__ = ["EventDetector", "SignificantEvent"]
```

```python
# src/realtime/event_detector.py
"""Detects significant game events that should trigger trades.

This is the core of the Reality Arbitrage strategy: identify events
that will move the market before the market reacts.

Event impact is scored 0-1 where:
- 0.0-0.3: Minor event (single kill early game, etc.)
- 0.3-0.6: Moderate event (objective, multi-kill)
- 0.6-0.8: Major event (baron, ace, significant score change)
- 0.8-1.0: Game-changing event (base race, match point)
"""

from dataclasses import dataclass
from typing import Optional
import structlog

from src.feeds.base import FeedEvent


logger = structlog.get_logger()


@dataclass
class SignificantEvent:
    """A game event classified as significant for trading."""
    original_event: Optional[FeedEvent]
    is_significant: bool
    impact_score: float  # 0.0 to 1.0
    favored_team: Optional[str]
    event_description: str

    @property
    def should_trade(self) -> bool:
        """Whether this event warrants a trade attempt."""
        return self.is_significant and self.impact_score >= 0.4


class EventDetector:
    """Classifies game events by their market impact potential."""

    # Event importance weights by game
    LOL_WEIGHTS = {
        "kill": 0.15,
        "tower_destroyed": 0.25,
        "dragon_kill": 0.30,
        "rift_herald": 0.20,
        "baron_kill": 0.85,
        "elder_dragon": 0.90,
        "inhibitor": 0.40,
        "ace": 0.60,
        "nexus_turret": 0.70,
    }

    CSGO_WEIGHTS = {
        "kill": 0.10,
        "round_end": 0.35,
        "ace": 0.50,
        "clutch": 0.45,
        "bomb_planted": 0.15,
        "bomb_defused": 0.35,
    }

    DOTA2_WEIGHTS = {
        "kill": 0.12,
        "tower_destroyed": 0.20,
        "roshan_kill": 0.75,
        "barracks": 0.50,
        "ancient_damage": 0.80,
        "mega_creeps": 0.85,
    }

    def __init__(self):
        self._game_state: dict[str, dict] = {}  # match_id -> state

    def classify(self, event: FeedEvent) -> SignificantEvent:
        """Classify an event's significance for trading."""
        if event.game == "lol":
            return self._classify_lol(event)
        elif event.game == "csgo":
            return self._classify_csgo(event)
        elif event.game == "dota2":
            return self._classify_dota2(event)
        else:
            return SignificantEvent(
                original_event=event,
                is_significant=False,
                impact_score=0.0,
                favored_team=None,
                event_description="Unknown game type"
            )

    def _classify_lol(self, event: FeedEvent) -> SignificantEvent:
        """Classify League of Legends events."""
        event_type = event.event_type
        data = event.data
        base_weight = self.LOL_WEIGHTS.get(event_type, 0.1)

        # Time multiplier: events matter more late game
        game_time = data.get("game_time_minutes", 15)
        time_multiplier = min(1.0, 0.5 + (game_time / 60))  # Scales 0.5 to 1.0

        # Special cases
        if event_type == "kill":
            # Multi-kills are more significant
            if "multi_kill" in data:
                base_weight *= (1 + data["multi_kill"] * 0.3)
            # Shutdown bounties
            if data.get("is_shutdown"):
                base_weight *= 1.5

        elif event_type == "baron_kill":
            # Baron is always huge
            base_weight = 0.85
            time_multiplier = 1.0

        elif event_type == "elder_dragon":
            base_weight = 0.90
            time_multiplier = 1.0

        impact_score = min(1.0, base_weight * time_multiplier)
        favored_team = data.get("killer_team") or data.get("team")

        return SignificantEvent(
            original_event=event,
            is_significant=impact_score >= 0.3,
            impact_score=impact_score,
            favored_team=favored_team,
            event_description=f"{event_type} by {favored_team}"
        )

    def _classify_csgo(self, event: FeedEvent) -> SignificantEvent:
        """Classify CS:GO events."""
        event_type = event.event_type
        data = event.data
        base_weight = self.CSGO_WEIGHTS.get(event_type, 0.1)

        favored_team = None

        if event_type == "round_end":
            winner = data.get("winner")
            score = data.get("score", {})

            # Calculate score differential
            if score and winner:
                winner_score = score.get(winner, 0)
                other_scores = [v for k, v in score.items() if k != winner]
                loser_score = max(other_scores) if other_scores else 0

                diff = winner_score - loser_score

                # Match point scenarios
                if winner_score >= 15:
                    base_weight = 0.80  # Match point
                elif winner_score >= 12 and diff >= 4:
                    base_weight = 0.60  # Commanding lead
                elif diff >= 5:
                    base_weight = 0.50  # Strong lead

                favored_team = winner

        elif event_type == "ace":
            base_weight = 0.50
            favored_team = data.get("player_team")

        impact_score = min(1.0, base_weight)

        return SignificantEvent(
            original_event=event,
            is_significant=impact_score >= 0.3,
            impact_score=impact_score,
            favored_team=favored_team,
            event_description=f"{event_type}: {data}"
        )

    def _classify_dota2(self, event: FeedEvent) -> SignificantEvent:
        """Classify Dota 2 events."""
        event_type = event.event_type
        data = event.data
        base_weight = self.DOTA2_WEIGHTS.get(event_type, 0.1)

        favored_team = data.get("team")

        # Roshan gives Aegis - very important
        if event_type == "roshan_kill":
            base_weight = 0.75

        # Barracks down = mega creeps potential
        elif event_type == "barracks":
            barracks_count = data.get("barracks_destroyed", 1)
            if barracks_count >= 2:
                base_weight = 0.70  # Multiple lanes down

        impact_score = min(1.0, base_weight)

        return SignificantEvent(
            original_event=event,
            is_significant=impact_score >= 0.3,
            impact_score=impact_score,
            favored_team=favored_team,
            event_description=f"{event_type} by {favored_team}"
        )

    def estimate_price_impact(
        self,
        event: SignificantEvent,
        current_price: float
    ) -> float:
        """Estimate what the new price should be after this event.

        Args:
            event: The significant event
            current_price: Current market price for favored team (0-1)

        Returns:
            Estimated new price (0-1)
        """
        if not event.is_significant:
            return current_price

        # Impact score translates to price swing
        # 0.4 impact = ~5% swing
        # 0.6 impact = ~10% swing
        # 0.8 impact = ~15% swing
        # 1.0 impact = ~20% swing

        max_swing = 0.20
        swing = event.impact_score * max_swing

        # Apply swing in direction of favored team
        new_price = current_price + swing

        # Clamp to valid range
        return max(0.01, min(0.99, new_price))
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/realtime/test_event_detector.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/realtime/ tests/realtime/
git commit -m "feat: event detector for classifying significant game events"
```

---

## Task 7: Market Mapper (Event to Polymarket Market)

**Files:**
- Create: `src/realtime/market_mapper.py`
- Create: `tests/realtime/test_market_mapper.py`

**Step 1: Write the failing test**

```python
# tests/realtime/test_market_mapper.py
import pytest
from src.realtime.market_mapper import MarketMapper, MarketMapping


class TestMarketMapper:
    @pytest.fixture
    def mapper(self):
        mapper = MarketMapper()
        # Pre-populate with test mappings
        mapper.add_mapping(
            game="lol",
            event_identifier="LCK_T1_vs_GenG_2026",
            polymarket_id="0x123abc",
            outcomes={"T1": "YES", "Gen.G": "NO"}
        )
        return mapper

    def test_find_market_for_event(self, mapper):
        mapping = mapper.find_market(
            game="lol",
            teams=["T1", "Gen.G"],
            league="LCK"
        )

        assert mapping is not None
        assert mapping.polymarket_id == "0x123abc"
        assert mapping.get_outcome_for_team("T1") == "YES"

    def test_find_market_returns_none_for_unknown(self, mapper):
        mapping = mapper.find_market(
            game="lol",
            teams=["Cloud9", "100T"],
            league="LCS"
        )

        assert mapping is None

    def test_outcome_mapping_correct(self, mapper):
        mapping = mapper.find_market(game="lol", teams=["T1", "Gen.G"], league="LCK")

        # When T1 does something good, we buy YES
        assert mapping.get_outcome_for_team("T1") == "YES"
        # When Gen.G does something good, we buy NO (or sell YES)
        assert mapping.get_outcome_for_team("Gen.G") == "NO"


class TestMarketMappingCreation:
    def test_create_mapping_from_polymarket_market(self):
        polymarket_data = {
            "id": "0x456def",
            "question": "Will T1 beat Gen.G in LCK Spring Finals?",
            "outcomes": ["Yes", "No"],
            "tags": ["esports", "lol", "lck"]
        }

        mapping = MarketMapping.from_polymarket(polymarket_data, team_a="T1", team_b="Gen.G")

        assert mapping.polymarket_id == "0x456def"
        assert mapping.get_outcome_for_team("T1") == "Yes"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/realtime/test_market_mapper.py -v
```

Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write the implementation**

```python
# src/realtime/market_mapper.py
"""Maps real-time game events to Polymarket markets.

The mapper maintains a registry of:
- Known matches and their corresponding Polymarket market IDs
- Team name normalization (T1, SKT, SK Telecom -> T1)
- Outcome mapping (which team = YES, which = NO)
"""

from dataclasses import dataclass, field
from typing import Optional
import re
import structlog


logger = structlog.get_logger()


@dataclass
class MarketMapping:
    """Mapping between a game match and a Polymarket market."""
    polymarket_id: str
    game: str
    event_identifier: str  # e.g., "LCK_T1_vs_GenG_2026"
    team_to_outcome: dict[str, str]  # {"T1": "YES", "Gen.G": "NO"}

    def get_outcome_for_team(self, team: str) -> Optional[str]:
        """Get the Polymarket outcome for a team."""
        # Try exact match first
        if team in self.team_to_outcome:
            return self.team_to_outcome[team]

        # Try normalized match
        normalized = self._normalize_team_name(team)
        for known_team, outcome in self.team_to_outcome.items():
            if self._normalize_team_name(known_team) == normalized:
                return outcome

        return None

    @staticmethod
    def _normalize_team_name(name: str) -> str:
        """Normalize team name for matching."""
        # Remove common suffixes/prefixes
        name = name.lower().strip()
        name = re.sub(r'\s*(esports?|gaming|team)\s*', '', name)
        name = re.sub(r'[^a-z0-9]', '', name)
        return name

    @classmethod
    def from_polymarket(
        cls,
        market_data: dict,
        team_a: str,
        team_b: str
    ) -> "MarketMapping":
        """Create mapping from Polymarket market data.

        Assumes binary market where YES = team_a wins.
        """
        outcomes = market_data.get("outcomes", ["Yes", "No"])

        return cls(
            polymarket_id=market_data["id"],
            game=cls._detect_game(market_data),
            event_identifier=f"{team_a}_vs_{team_b}",
            team_to_outcome={
                team_a: outcomes[0],  # YES
                team_b: outcomes[1]   # NO
            }
        )

    @staticmethod
    def _detect_game(market_data: dict) -> str:
        """Detect game type from market tags or title."""
        tags = market_data.get("tags", [])
        title = market_data.get("question", "").lower()

        if "lol" in tags or "league" in title:
            return "lol"
        elif "dota" in tags or "dota" in title:
            return "dota2"
        elif "csgo" in tags or "counter-strike" in title or "cs2" in title:
            return "csgo"
        elif "valorant" in tags or "valorant" in title:
            return "valorant"

        return "unknown"


class MarketMapper:
    """Registry of game matches to Polymarket markets."""

    # Common team name aliases
    TEAM_ALIASES = {
        # LoL
        "skt": "T1",
        "sk telecom": "T1",
        "skt t1": "T1",
        "gen.g": "Gen.G",
        "geng": "Gen.G",
        "samsung": "Gen.G",
        "damwon": "DK",
        "dwg": "DK",
        "dk": "DK",
        # CS:GO
        "natus vincere": "Navi",
        "navi": "Navi",
        "faze clan": "FaZe",
        "faze": "FaZe",
        "g2 esports": "G2",
        "g2": "G2",
    }

    def __init__(self):
        self._mappings: dict[str, MarketMapping] = {}  # event_identifier -> mapping
        self._polymarket_index: dict[str, MarketMapping] = {}  # polymarket_id -> mapping

    def add_mapping(
        self,
        game: str,
        event_identifier: str,
        polymarket_id: str,
        outcomes: dict[str, str]
    ) -> MarketMapping:
        """Add a new mapping."""
        mapping = MarketMapping(
            polymarket_id=polymarket_id,
            game=game,
            event_identifier=event_identifier,
            team_to_outcome=outcomes
        )

        self._mappings[event_identifier] = mapping
        self._polymarket_index[polymarket_id] = mapping

        logger.info("market_mapping_added",
                   event=event_identifier,
                   market=polymarket_id)

        return mapping

    def find_market(
        self,
        game: str,
        teams: list[str],
        league: Optional[str] = None,
        match_id: Optional[str] = None
    ) -> Optional[MarketMapping]:
        """Find a Polymarket market for a match.

        Args:
            game: Game type (lol, dota2, csgo)
            teams: List of team names
            league: Optional league name (LCK, LCS, etc.)
            match_id: Optional specific match ID

        Returns:
            MarketMapping if found, None otherwise
        """
        # Normalize team names
        normalized_teams = [self._normalize_team(t) for t in teams]

        # Try exact match with match_id
        if match_id and match_id in self._mappings:
            return self._mappings[match_id]

        # Search by teams
        for event_id, mapping in self._mappings.items():
            if mapping.game != game:
                continue

            # Check if both teams are in this mapping
            mapping_teams = set(self._normalize_team(t) for t in mapping.team_to_outcome.keys())
            if set(normalized_teams) == mapping_teams:
                return mapping

        return None

    def find_by_polymarket_id(self, polymarket_id: str) -> Optional[MarketMapping]:
        """Find mapping by Polymarket market ID."""
        return self._polymarket_index.get(polymarket_id)

    def _normalize_team(self, team: str) -> str:
        """Normalize team name using aliases."""
        lower = team.lower().strip()
        return self.TEAM_ALIASES.get(lower, team)

    def get_all_mappings(self) -> list[MarketMapping]:
        """Get all registered mappings."""
        return list(self._mappings.values())
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/realtime/test_market_mapper.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/realtime/market_mapper.py tests/realtime/test_market_mapper.py
git commit -m "feat: market mapper to link game events to Polymarket markets"
```

---

## Task 8: Reality Arbitrage Engine

**Files:**
- Create: `src/arb/reality_arb.py`
- Create: `tests/arb/test_reality_arb.py`

**Step 1: Write the failing test**

```python
# tests/arb/test_reality_arb.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

from src.arb.reality_arb import RealityArbEngine, ArbOpportunity
from src.realtime.event_detector import SignificantEvent
from src.realtime.market_mapper import MarketMapping


class TestArbOpportunityDetection:
    @pytest.fixture
    def engine(self):
        engine = RealityArbEngine()
        engine.polymarket_feed = MagicMock()
        engine.event_detector = MagicMock()
        engine.market_mapper = MagicMock()
        return engine

    def test_detect_opportunity_from_baron_kill(self, engine):
        """Baron kill should create arb opportunity."""
        # Setup: market currently at 50/50
        engine.polymarket_feed.get_best_prices.return_value = (0.49, 0.51)

        # Event: T1 kills Baron
        event = SignificantEvent(
            original_event=None,
            is_significant=True,
            impact_score=0.85,
            favored_team="T1",
            event_description="Baron kill by T1"
        )

        mapping = MarketMapping(
            polymarket_id="0x123",
            game="lol",
            event_identifier="T1_vs_GenG",
            team_to_outcome={"T1": "YES", "Gen.G": "NO"}
        )

        opportunity = engine.evaluate_opportunity(event, mapping)

        assert opportunity is not None
        assert opportunity.edge_pct >= 0.10  # At least 10% edge
        assert opportunity.side == "BUY"
        assert opportunity.outcome == "YES"

    def test_no_opportunity_if_market_already_moved(self, engine):
        """If market already reflects event, no opportunity."""
        # Market already at 75% for T1
        engine.polymarket_feed.get_best_prices.return_value = (0.74, 0.76)

        event = SignificantEvent(
            original_event=None,
            is_significant=True,
            impact_score=0.60,
            favored_team="T1",
            event_description="Tower destroyed"
        )

        mapping = MarketMapping(
            polymarket_id="0x123",
            game="lol",
            event_identifier="T1_vs_GenG",
            team_to_outcome={"T1": "YES", "Gen.G": "NO"}
        )

        opportunity = engine.evaluate_opportunity(event, mapping)

        # Edge should be minimal or negative
        assert opportunity is None or opportunity.edge_pct < 0.02

    def test_calculates_position_size_correctly(self, engine):
        """Position size respects risk limits."""
        engine.capital = 10000
        engine.max_position_pct = 0.10

        opportunity = ArbOpportunity(
            market_id="0x123",
            side="BUY",
            outcome="YES",
            current_price=0.50,
            estimated_fair_price=0.65,
            edge_pct=0.15,
            trigger_event="baron_kill",
            timestamp=datetime.utcnow().timestamp()
        )

        size = engine.calculate_position_size(opportunity)

        assert size <= 1000  # Max 10% of $10k
        assert size > 0


class TestArbExecution:
    @pytest.fixture
    def engine(self):
        engine = RealityArbEngine()
        engine.executor = AsyncMock()
        return engine

    @pytest.mark.asyncio
    async def test_execute_opportunity(self, engine):
        """Executes trade via executor."""
        engine.executor.place_order.return_value = {
            "order_id": "order_123",
            "status": "FILLED",
            "fill_price": 0.51
        }

        opportunity = ArbOpportunity(
            market_id="0x123",
            side="BUY",
            outcome="YES",
            current_price=0.50,
            estimated_fair_price=0.65,
            edge_pct=0.15,
            trigger_event="baron_kill",
            timestamp=datetime.utcnow().timestamp()
        )

        result = await engine.execute(opportunity, size=100)

        assert result["status"] == "FILLED"
        engine.executor.place_order.assert_called_once()
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/arb/test_reality_arb.py -v
```

Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write the implementation**

```python
# src/arb/reality_arb.py
"""Reality Arbitrage Engine - Core trading logic.

This engine:
1. Receives significant game events from EventDetector
2. Looks up corresponding Polymarket market via MarketMapper
3. Compares estimated fair price vs current market price
4. If edge > threshold, generates trading opportunity
5. Executes trade if in autopilot mode, or alerts for manual approval
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import asyncio
import structlog

from src.feeds.polymarket import PolymarketFeed
from src.realtime.event_detector import EventDetector, SignificantEvent
from src.realtime.market_mapper import MarketMapper, MarketMapping
from config.settings import settings


logger = structlog.get_logger()


@dataclass
class ArbOpportunity:
    """A detected arbitrage opportunity."""
    market_id: str
    side: str  # BUY or SELL
    outcome: str  # YES or NO
    current_price: float
    estimated_fair_price: float
    edge_pct: float
    trigger_event: str
    timestamp: float

    @property
    def is_valid(self) -> bool:
        """Check if opportunity meets minimum criteria."""
        return (
            self.edge_pct >= settings.MIN_EDGE_PCT and
            self.edge_pct <= settings.ANOMALY_THRESHOLD_PCT
        )

    def to_dict(self) -> dict:
        return {
            "market_id": self.market_id,
            "side": self.side,
            "outcome": self.outcome,
            "current_price": self.current_price,
            "estimated_fair_price": self.estimated_fair_price,
            "edge_pct": self.edge_pct,
            "trigger_event": self.trigger_event,
            "timestamp": self.timestamp
        }


class RealityArbEngine:
    """Main engine for Reality Arbitrage strategy."""

    def __init__(
        self,
        polymarket_feed: Optional[PolymarketFeed] = None,
        event_detector: Optional[EventDetector] = None,
        market_mapper: Optional[MarketMapper] = None
    ):
        self.polymarket_feed = polymarket_feed or PolymarketFeed()
        self.event_detector = event_detector or EventDetector()
        self.market_mapper = market_mapper or MarketMapper()
        self.executor = None  # Set externally

        # Risk parameters
        self.capital = 10000  # Default $10k
        self.max_position_pct = settings.MAX_POSITION_PCT
        self.min_edge_pct = settings.MIN_EDGE_PCT
        self.anomaly_threshold = settings.ANOMALY_THRESHOLD_PCT

        # State
        self._running = False
        self._pending_opportunities: list[ArbOpportunity] = []

    def evaluate_opportunity(
        self,
        event: SignificantEvent,
        mapping: MarketMapping
    ) -> Optional[ArbOpportunity]:
        """Evaluate if an event creates a trading opportunity.

        Args:
            event: Significant game event
            mapping: Market mapping for this match

        Returns:
            ArbOpportunity if edge exists, None otherwise
        """
        if not event.is_significant or not event.favored_team:
            return None

        # Get current market price
        outcome = mapping.get_outcome_for_team(event.favored_team)
        if not outcome:
            logger.warning("no_outcome_mapping",
                         team=event.favored_team,
                         mapping=mapping.event_identifier)
            return None

        best_bid, best_ask = self.polymarket_feed.get_best_prices(
            mapping.polymarket_id,
            outcome
        )

        if best_bid == 0 and best_ask == 0:
            logger.warning("no_market_prices", market_id=mapping.polymarket_id)
            return None

        current_price = (best_bid + best_ask) / 2

        # Estimate fair price after event
        estimated_fair = self.event_detector.estimate_price_impact(
            event,
            current_price
        )

        # Calculate edge
        if estimated_fair > current_price:
            # We should BUY - fair price is higher than current
            edge_pct = (estimated_fair - best_ask) / best_ask if best_ask > 0 else 0
            side = "BUY"
            entry_price = best_ask
        else:
            # We should SELL - fair price is lower than current
            edge_pct = (best_bid - estimated_fair) / estimated_fair if estimated_fair > 0 else 0
            side = "SELL"
            entry_price = best_bid

        # Check if edge is sufficient
        if edge_pct < self.min_edge_pct:
            logger.debug("insufficient_edge",
                        edge_pct=edge_pct,
                        min_required=self.min_edge_pct)
            return None

        # Check for anomaly (likely stale data)
        if edge_pct > self.anomaly_threshold:
            logger.warning("anomaly_detected",
                         edge_pct=edge_pct,
                         threshold=self.anomaly_threshold,
                         event=event.event_description)
            return None

        opportunity = ArbOpportunity(
            market_id=mapping.polymarket_id,
            side=side,
            outcome=outcome,
            current_price=entry_price,
            estimated_fair_price=estimated_fair,
            edge_pct=edge_pct,
            trigger_event=event.event_description,
            timestamp=datetime.utcnow().timestamp()
        )

        logger.info("opportunity_detected",
                   market=mapping.polymarket_id,
                   side=side,
                   outcome=outcome,
                   edge_pct=f"{edge_pct:.2%}",
                   event=event.event_description)

        return opportunity

    def calculate_position_size(self, opportunity: ArbOpportunity) -> float:
        """Calculate position size respecting risk limits.

        Uses a simplified Kelly criterion:
        size = edge * capital * kelly_fraction

        Capped at max_position_pct of capital.
        """
        # Kelly fraction (conservative 0.25x Kelly)
        kelly_fraction = 0.25

        # Base size from edge
        edge_size = opportunity.edge_pct * self.capital * kelly_fraction

        # Apply maximum position limit
        max_size = self.capital * self.max_position_pct

        size = min(edge_size, max_size)

        logger.info("position_size_calculated",
                   edge_size=edge_size,
                   max_size=max_size,
                   final_size=size)

        return size

    async def execute(self, opportunity: ArbOpportunity, size: float) -> dict:
        """Execute a trade for an opportunity.

        Args:
            opportunity: The arb opportunity
            size: Position size in USD

        Returns:
            Execution result dict
        """
        if not self.executor:
            raise RuntimeError("No executor configured")

        logger.info("executing_trade",
                   market=opportunity.market_id,
                   side=opportunity.side,
                   outcome=opportunity.outcome,
                   size=size)

        result = await self.executor.place_order(
            market_id=opportunity.market_id,
            side=opportunity.side,
            outcome=opportunity.outcome,
            price=opportunity.current_price,
            size=size
        )

        return result

    async def process_event(self, event: SignificantEvent, mapping: MarketMapping) -> Optional[ArbOpportunity]:
        """Process a single event end-to-end.

        Args:
            event: Game event to process
            mapping: Market mapping

        Returns:
            Opportunity if one was found and valid
        """
        opportunity = self.evaluate_opportunity(event, mapping)

        if opportunity and opportunity.is_valid:
            self._pending_opportunities.append(opportunity)
            return opportunity

        return None

    def get_pending_opportunities(self) -> list[ArbOpportunity]:
        """Get all pending (unapproved) opportunities."""
        return list(self._pending_opportunities)

    def clear_opportunity(self, market_id: str) -> None:
        """Remove an opportunity from pending list."""
        self._pending_opportunities = [
            o for o in self._pending_opportunities
            if o.market_id != market_id
        ]
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/arb/test_reality_arb.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/arb/reality_arb.py tests/arb/test_reality_arb.py
git commit -m "feat: reality arbitrage engine with opportunity detection and sizing"
```

---

## Task 9: Telegram Bot Integration

**Files:**
- Modify: `src/bot/telegram.py` (from existing design)
- Create: `src/bot/reality_handlers.py`
- Create: `tests/bot/test_reality_handlers.py`

**Step 1: Write the failing test**

```python
# tests/bot/test_reality_handlers.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from src.bot.reality_handlers import RealityArbHandler
from src.arb.reality_arb import ArbOpportunity


class TestAlertFormatting:
    @pytest.fixture
    def handler(self):
        return RealityArbHandler(bot=MagicMock(), engine=MagicMock())

    def test_format_opportunity_alert(self, handler):
        opportunity = ArbOpportunity(
            market_id="0x123abc",
            side="BUY",
            outcome="YES",
            current_price=0.52,
            estimated_fair_price=0.65,
            edge_pct=0.12,
            trigger_event="Baron kill by T1",
            timestamp=datetime.utcnow().timestamp()
        )

        alert = handler.format_alert(opportunity, market_title="T1 vs Gen.G Winner")

        assert "REALITY ARB" in alert
        assert "T1 vs Gen.G" in alert
        assert "12%" in alert or "12.0%" in alert
        assert "Baron kill" in alert
        assert "BUY YES" in alert

    def test_format_includes_expiry(self, handler):
        opportunity = ArbOpportunity(
            market_id="0x123",
            side="BUY",
            outcome="YES",
            current_price=0.50,
            estimated_fair_price=0.60,
            edge_pct=0.10,
            trigger_event="Dragon kill",
            timestamp=datetime.utcnow().timestamp()
        )

        alert = handler.format_alert(opportunity, market_title="Test", expiry_seconds=30)

        assert "30s" in alert or "30 sec" in alert.lower()


class TestApprovalFlow:
    @pytest.fixture
    def handler(self):
        handler = RealityArbHandler(bot=AsyncMock(), engine=MagicMock())
        handler.engine.execute = AsyncMock(return_value={"status": "FILLED"})
        return handler

    @pytest.mark.asyncio
    async def test_approve_executes_trade(self, handler):
        opportunity = ArbOpportunity(
            market_id="0x123",
            side="BUY",
            outcome="YES",
            current_price=0.50,
            estimated_fair_price=0.60,
            edge_pct=0.10,
            trigger_event="test",
            timestamp=datetime.utcnow().timestamp()
        )
        handler._pending = {"0x123": (opportunity, 100)}

        result = await handler.handle_approve("0x123")

        assert result["executed"] is True
        handler.engine.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_skip_removes_opportunity(self, handler):
        opportunity = ArbOpportunity(
            market_id="0x123",
            side="BUY",
            outcome="YES",
            current_price=0.50,
            estimated_fair_price=0.60,
            edge_pct=0.10,
            trigger_event="test",
            timestamp=datetime.utcnow().timestamp()
        )
        handler._pending = {"0x123": (opportunity, 100)}

        result = await handler.handle_skip("0x123")

        assert result["skipped"] is True
        assert "0x123" not in handler._pending
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/bot/test_reality_handlers.py -v
```

Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write the implementation**

```python
# src/bot/reality_handlers.py
"""Telegram bot handlers for Reality Arbitrage alerts."""

from datetime import datetime
from typing import Optional
import asyncio
import structlog
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

from src.arb.reality_arb import RealityArbEngine, ArbOpportunity


logger = structlog.get_logger()


class RealityArbHandler:
    """Handles Telegram interactions for Reality Arb opportunities."""

    ALERT_EXPIRY_SECONDS = 30  # Opportunities expire quickly!

    def __init__(self, bot, engine: RealityArbEngine):
        self.bot = bot
        self.engine = engine
        self._pending: dict[str, tuple[ArbOpportunity, float]] = {}  # market_id -> (opp, size)
        self._expiry_tasks: dict[str, asyncio.Task] = {}

    def format_alert(
        self,
        opportunity: ArbOpportunity,
        market_title: str,
        expiry_seconds: int = ALERT_EXPIRY_SECONDS
    ) -> str:
        """Format opportunity as Telegram alert message."""

        edge_display = f"{opportunity.edge_pct:.1%}"
        price_display = f"{opportunity.current_price:.2f}"
        fair_display = f"{opportunity.estimated_fair_price:.2f}"

        alert = f"""⚡ REALITY ARB DETECTED

📊 Market: {market_title}
🎯 Trigger: {opportunity.trigger_event}

┌─────────────────────────────┐
│ Current Price:  {price_display:>10} │
│ Est. Fair:      {fair_display:>10} │
│ Edge:           {edge_display:>10} │
└─────────────────────────────┘

Action: {opportunity.side} {opportunity.outcome} @ {price_display}

⏱️ Expires in {expiry_seconds}s - ACT FAST!
"""
        return alert

    def get_approval_keyboard(self, market_id: str) -> InlineKeyboardMarkup:
        """Create inline keyboard for approve/skip."""
        keyboard = [
            [
                InlineKeyboardButton("✅ APPROVE", callback_data=f"arb_approve:{market_id}"),
                InlineKeyboardButton("❌ SKIP", callback_data=f"arb_skip:{market_id}")
            ]
        ]
        return InlineKeyboardMarkup(keyboard)

    async def send_alert(
        self,
        chat_id: str,
        opportunity: ArbOpportunity,
        market_title: str,
        size: float
    ) -> None:
        """Send opportunity alert to Telegram."""
        # Store pending opportunity
        self._pending[opportunity.market_id] = (opportunity, size)

        # Format and send message
        message = self.format_alert(opportunity, market_title)
        keyboard = self.get_approval_keyboard(opportunity.market_id)

        sent_message = await self.bot.send_message(
            chat_id=chat_id,
            text=message,
            reply_markup=keyboard,
            parse_mode="HTML"
        )

        # Start expiry timer
        self._expiry_tasks[opportunity.market_id] = asyncio.create_task(
            self._expire_opportunity(
                opportunity.market_id,
                chat_id,
                sent_message.message_id
            )
        )

        logger.info("alert_sent",
                   market_id=opportunity.market_id,
                   edge=opportunity.edge_pct)

    async def _expire_opportunity(
        self,
        market_id: str,
        chat_id: str,
        message_id: int
    ) -> None:
        """Expire an opportunity after timeout."""
        await asyncio.sleep(self.ALERT_EXPIRY_SECONDS)

        if market_id in self._pending:
            del self._pending[market_id]

            # Update message to show expired
            try:
                await self.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=message_id,
                    text="⏰ Opportunity expired (not acted on in time)"
                )
            except Exception as e:
                logger.error("failed_to_update_expired", error=str(e))

            logger.info("opportunity_expired", market_id=market_id)

    async def handle_approve(self, market_id: str) -> dict:
        """Handle approval of an opportunity."""
        if market_id not in self._pending:
            return {"executed": False, "error": "Opportunity not found or expired"}

        opportunity, size = self._pending[market_id]

        # Cancel expiry timer
        if market_id in self._expiry_tasks:
            self._expiry_tasks[market_id].cancel()
            del self._expiry_tasks[market_id]

        # Execute trade
        try:
            result = await self.engine.execute(opportunity, size)
            del self._pending[market_id]

            logger.info("trade_executed",
                       market_id=market_id,
                       result=result)

            return {"executed": True, "result": result}

        except Exception as e:
            logger.error("execution_failed", market_id=market_id, error=str(e))
            return {"executed": False, "error": str(e)}

    async def handle_skip(self, market_id: str) -> dict:
        """Handle skip of an opportunity."""
        if market_id not in self._pending:
            return {"skipped": False, "error": "Opportunity not found"}

        # Cancel expiry timer
        if market_id in self._expiry_tasks:
            self._expiry_tasks[market_id].cancel()
            del self._expiry_tasks[market_id]

        del self._pending[market_id]

        logger.info("opportunity_skipped", market_id=market_id)
        return {"skipped": True}

    async def callback_handler(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle inline keyboard callbacks."""
        query = update.callback_query
        await query.answer()

        data = query.data
        if data.startswith("arb_approve:"):
            market_id = data.split(":")[1]
            result = await self.handle_approve(market_id)

            if result["executed"]:
                await query.edit_message_text(
                    f"✅ Trade executed!\n\n{result.get('result', {})}"
                )
            else:
                await query.edit_message_text(
                    f"❌ Execution failed: {result.get('error')}"
                )

        elif data.startswith("arb_skip:"):
            market_id = data.split(":")[1]
            result = await self.handle_skip(market_id)

            await query.edit_message_text("⏭️ Opportunity skipped")
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/bot/test_reality_handlers.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/bot/reality_handlers.py tests/bot/test_reality_handlers.py
git commit -m "feat: telegram handlers for reality arb alerts with approve/skip flow"
```

---

## Task 10: Main Runner Script

**Files:**
- Create: `scripts/run_reality_arb.py`

**Step 1: Write the runner script**

```python
# scripts/run_reality_arb.py
"""
Reality Arbitrage Bot - Main Entry Point

Usage:
    python scripts/run_reality_arb.py [--game lol|dota2|csgo] [--autopilot]

This script:
1. Connects to PandaScore for live esports events
2. Connects to Polymarket WebSocket for order book
3. Monitors for significant game events
4. Detects arbitrage opportunities from broadcast lag
5. Sends Telegram alerts (or auto-executes in autopilot mode)
"""

import asyncio
import argparse
import signal
import sys
from typing import Optional
import structlog

# Setup logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.dev.ConsoleRenderer()
    ]
)
logger = structlog.get_logger()

from config.settings import settings
from src.feeds.pandascore import PandaScoreFeed
from src.feeds.polymarket import PolymarketFeed
from src.realtime.event_detector import EventDetector
from src.realtime.market_mapper import MarketMapper
from src.arb.reality_arb import RealityArbEngine
from src.db.database import init_db


class RealityArbBot:
    """Main bot orchestrator."""

    def __init__(self, game: str = "lol", autopilot: bool = False):
        self.game = game
        self.autopilot = autopilot

        # Components
        self.pandascore = PandaScoreFeed()
        self.polymarket = PolymarketFeed()
        self.detector = EventDetector()
        self.mapper = MarketMapper()
        self.engine = RealityArbEngine(
            polymarket_feed=self.polymarket,
            event_detector=self.detector,
            market_mapper=self.mapper
        )

        # Telegram (optional)
        self.telegram_handler: Optional[any] = None

        self._running = False

    async def start(self) -> None:
        """Start the bot."""
        logger.info("starting_reality_arb_bot", game=self.game, autopilot=self.autopilot)

        # Initialize database
        await init_db()

        # Connect to feeds
        await self.pandascore.connect()
        await self.polymarket.connect()

        # Register event handler
        self.pandascore.on_event(self._on_game_event)

        # Discover and subscribe to live matches
        await self._subscribe_live_matches()

        self._running = True
        logger.info("bot_started")

        # Keep running
        while self._running:
            # Periodically refresh live matches
            await asyncio.sleep(60)
            await self._subscribe_live_matches()

    async def stop(self) -> None:
        """Stop the bot gracefully."""
        logger.info("stopping_bot")
        self._running = False

        await self.pandascore.disconnect()
        await self.polymarket.disconnect()

        logger.info("bot_stopped")

    async def _subscribe_live_matches(self) -> None:
        """Find and subscribe to currently live matches."""
        matches = await self.pandascore.get_live_matches(self.game)

        for match in matches:
            match_id = str(match["id"])

            # Check if we have a Polymarket mapping for this match
            teams = self._extract_teams(match)
            league = match.get("league", {}).get("name")

            mapping = self.mapper.find_market(
                game=self.game,
                teams=teams,
                league=league
            )

            if mapping:
                await self.pandascore.subscribe(self.game, match_id)
                await self.polymarket.subscribe_market(mapping.polymarket_id)

                logger.info("subscribed_to_match",
                           match=match.get("name"),
                           market=mapping.polymarket_id)

    def _extract_teams(self, match: dict) -> list[str]:
        """Extract team names from match data."""
        opponents = match.get("opponents", [])
        return [opp.get("opponent", {}).get("name", "Unknown") for opp in opponents]

    async def _on_game_event(self, event) -> None:
        """Handle incoming game event."""
        # Classify event
        significant = self.detector.classify(event)

        if not significant.should_trade:
            return

        logger.info("significant_event_detected",
                   event=significant.event_description,
                   impact=significant.impact_score)

        # Find market mapping
        # (In production, we'd get match context from the event)
        mapping = self.mapper.find_by_polymarket_id(event.match_id)
        if not mapping:
            return

        # Evaluate opportunity
        opportunity = await self.engine.process_event(significant, mapping)

        if opportunity:
            size = self.engine.calculate_position_size(opportunity)

            if self.autopilot:
                # Auto-execute
                result = await self.engine.execute(opportunity, size)
                logger.info("auto_executed", result=result)
            else:
                # Send Telegram alert
                if self.telegram_handler:
                    await self.telegram_handler.send_alert(
                        chat_id=settings.TELEGRAM_CHAT_ID,
                        opportunity=opportunity,
                        market_title=mapping.event_identifier,
                        size=size
                    )


async def main():
    parser = argparse.ArgumentParser(description="Reality Arbitrage Bot")
    parser.add_argument("--game", choices=["lol", "dota2", "csgo"], default="lol",
                       help="Game to monitor (default: lol)")
    parser.add_argument("--autopilot", action="store_true",
                       help="Enable autopilot mode (auto-execute trades)")

    args = parser.parse_args()

    bot = RealityArbBot(game=args.game, autopilot=args.autopilot)

    # Handle shutdown signals
    loop = asyncio.get_event_loop()

    def signal_handler():
        logger.info("shutdown_signal_received")
        asyncio.create_task(bot.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    try:
        await bot.start()
    except KeyboardInterrupt:
        await bot.stop()


if __name__ == "__main__":
    asyncio.run(main())
```

**Step 2: Commit**

```bash
git add scripts/run_reality_arb.py
git commit -m "feat: main runner script for reality arbitrage bot"
```

---

## Task 11: Integration Test with Mock Data

**Files:**
- Create: `tests/integration/test_reality_arb_flow.py`

**Step 1: Write the integration test**

```python
# tests/integration/test_reality_arb_flow.py
"""Integration test for full Reality Arb flow with mock data."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from src.feeds.base import FeedEvent
from src.feeds.pandascore import PandaScoreFeed, PandaScoreEvent
from src.feeds.polymarket import PolymarketFeed
from src.realtime.event_detector import EventDetector
from src.realtime.market_mapper import MarketMapper
from src.arb.reality_arb import RealityArbEngine


class TestFullArbFlow:
    """Test complete flow from game event to trade opportunity."""

    @pytest.fixture
    def setup_components(self):
        """Setup all components with mocks."""
        # Mock Polymarket feed
        polymarket = MagicMock(spec=PolymarketFeed)
        polymarket.get_best_prices.return_value = (0.48, 0.52)  # 50% market

        # Real detector and mapper
        detector = EventDetector()
        mapper = MarketMapper()

        # Add test mapping
        mapper.add_mapping(
            game="lol",
            event_identifier="T1_vs_GenG_LCK_Finals",
            polymarket_id="0xtest123",
            outcomes={"T1": "YES", "Gen.G": "NO"}
        )

        # Engine
        engine = RealityArbEngine(
            polymarket_feed=polymarket,
            event_detector=detector,
            market_mapper=mapper
        )
        engine.capital = 10000

        return {
            "polymarket": polymarket,
            "detector": detector,
            "mapper": mapper,
            "engine": engine
        }

    @pytest.mark.asyncio
    async def test_baron_kill_creates_opportunity(self, setup_components):
        """Baron kill should create high-edge opportunity."""
        engine = setup_components["engine"]
        mapper = setup_components["mapper"]

        # Simulate Baron kill event
        event = FeedEvent(
            source="pandascore",
            event_type="baron_kill",
            game="lol",
            data={"team": "T1"},
            timestamp=datetime.utcnow().timestamp(),
            match_id="match_123"
        )

        # Classify event
        significant = setup_components["detector"].classify(event)

        assert significant.is_significant
        assert significant.impact_score >= 0.8
        assert significant.favored_team == "T1"

        # Find market and evaluate opportunity
        mapping = mapper.find_market(game="lol", teams=["T1", "Gen.G"])
        opportunity = engine.evaluate_opportunity(significant, mapping)

        assert opportunity is not None
        assert opportunity.side == "BUY"
        assert opportunity.outcome == "YES"
        assert opportunity.edge_pct >= 0.10

    @pytest.mark.asyncio
    async def test_csgo_match_point_opportunity(self, setup_components):
        """CS:GO match point should create opportunity."""
        engine = setup_components["engine"]
        mapper = setup_components["mapper"]
        detector = setup_components["detector"]

        # Add CS:GO mapping
        mapper.add_mapping(
            game="csgo",
            event_identifier="Navi_vs_FaZe_Major",
            polymarket_id="0xcsgo456",
            outcomes={"Navi": "YES", "FaZe": "NO"}
        )

        # Simulate match point round win
        event = FeedEvent(
            source="pandascore",
            event_type="round_end",
            game="csgo",
            data={"winner": "Navi", "score": {"Navi": 15, "FaZe": 10}},
            timestamp=datetime.utcnow().timestamp()
        )

        significant = detector.classify(event)

        assert significant.is_significant
        assert significant.impact_score >= 0.6  # Match point is important

        mapping = mapper.find_market(game="csgo", teams=["Navi", "FaZe"])
        opportunity = engine.evaluate_opportunity(significant, mapping)

        assert opportunity is not None

    @pytest.mark.asyncio
    async def test_no_opportunity_when_market_fair(self, setup_components):
        """No opportunity if market already reflects event."""
        engine = setup_components["engine"]
        mapper = setup_components["mapper"]
        polymarket = setup_components["polymarket"]

        # Market already at 75% (fair after baron)
        polymarket.get_best_prices.return_value = (0.74, 0.76)

        event = FeedEvent(
            source="pandascore",
            event_type="baron_kill",
            game="lol",
            data={"team": "T1"},
            timestamp=datetime.utcnow().timestamp()
        )

        significant = setup_components["detector"].classify(event)
        mapping = mapper.find_market(game="lol", teams=["T1", "Gen.G"])
        opportunity = engine.evaluate_opportunity(significant, mapping)

        # Should be None or have very low edge
        assert opportunity is None or opportunity.edge_pct < 0.02

    @pytest.mark.asyncio
    async def test_position_sizing_respects_limits(self, setup_components):
        """Position size should respect risk limits."""
        engine = setup_components["engine"]
        engine.capital = 10000
        engine.max_position_pct = 0.10

        from src.arb.reality_arb import ArbOpportunity

        opportunity = ArbOpportunity(
            market_id="0x123",
            side="BUY",
            outcome="YES",
            current_price=0.50,
            estimated_fair_price=0.70,
            edge_pct=0.40,  # Very high edge
            trigger_event="test",
            timestamp=datetime.utcnow().timestamp()
        )

        size = engine.calculate_position_size(opportunity)

        # Even with 40% edge, should be capped at 10% of capital
        assert size <= 1000
        assert size > 0
```

**Step 2: Run integration test**

```bash
pytest tests/integration/test_reality_arb_flow.py -v
```

Expected: PASS

**Step 3: Commit**

```bash
git add tests/integration/
git commit -m "test: integration tests for reality arb flow"
```

---

## Phase 2: Traditional Sports (Future Tasks)

### Task 12: SportsDataIO Feed (Lower Priority)

**Files to create:**
- `src/feeds/sportsdataio.py`
- `tests/feeds/test_sportsdataio.py`

**Note:** Similar structure to PandaScore but for NBA, NFL, etc.

### Task 13: Sport-Specific Event Detectors

**Files to modify:**
- `src/realtime/event_detector.py` - Add NBA, NFL weights

### Task 14: Cross-Platform Arbitrage (Existing Design)

Integrate with existing `betfair.py` design for price-discrepancy arbitrage.

---

## Deployment Checklist

- [ ] Set up VPS close to Polymarket servers (NYC/EU)
- [ ] Configure environment variables
- [ ] Set up Telegram bot via BotFather
- [ ] Get PandaScore API key
- [ ] Fund Polymarket wallet (Polygon USDC)
- [ ] Run in paper-trading mode first
- [ ] Monitor latency metrics
- [ ] Gradually enable autopilot after validation

---

## Risk Reminders

1. **Start with esports** - Better edge, lower competition
2. **Paper trade first** - Validate signals before real money
3. **Monitor latency** - Edge disappears if API is slow
4. **Small positions** - 10% max per trade
5. **Daily loss limit** - Auto-halt if losing
6. **Anomaly detection** - >15% edge = likely stale data

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy and configure settings
cp config/settings.example.py config/settings.py
# Edit settings.py with your API keys

# 3. Initialize database
python -c "import asyncio; from src.db.database import init_db; asyncio.run(init_db())"

# 4. Run tests
pytest -v

# 5. Start bot (manual approval mode)
python scripts/run_reality_arb.py --game lol

# 6. Start bot (autopilot mode - use with caution!)
python scripts/run_reality_arb.py --game lol --autopilot
```
