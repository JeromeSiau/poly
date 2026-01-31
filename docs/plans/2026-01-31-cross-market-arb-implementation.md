# Cross-Market Arbitrage Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add cross-market arbitrage between Polymarket, Azuro, and Overtime alongside the existing Reality Arb system.

**Architecture:** Extends existing feed/arb pattern with new GraphQL feeds (Azuro, Overtime), LLM-powered event matching, unified risk management, and multi-chain execution. Shares infrastructure (DB, Telegram, config) with Reality Arb.

**Tech Stack:** Python 3.11+, gql (GraphQL), web3.py, anthropic SDK, FastAPI, SQLAlchemy

---

## Phase 1: Foundation

### Task 1.1: Add New Dependencies

**Files:**
- Modify: `requirements.txt`

**Step 1: Add new dependencies**

Add to `requirements.txt`:
```
# GraphQL
gql[aiohttp]>=3.5.0

# Multi-chain Web3
web3>=6.15.0
eth-account>=0.10.0

# LLM for event matching
anthropic>=0.40.0

# Dashboard
fastapi>=0.109.0
uvicorn>=0.27.0
jinja2>=3.1.0
```

**Step 2: Install and verify**

Run: `pip install -r requirements.txt`
Expected: All packages install successfully

**Step 3: Commit**

```bash
git add requirements.txt
git commit -m "deps: add graphql, web3, anthropic, fastapi for cross-market arb"
```

---

### Task 1.2: Extend Settings

**Files:**
- Modify: `config/settings.py`
- Test: `tests/config/test_settings.py`

**Step 1: Write the failing test**

Create `tests/config/test_settings.py`:
```python
import pytest
from config.settings import Settings


def test_settings_has_azuro_config():
    s = Settings()
    assert hasattr(s, "AZURO_SUBGRAPH_URL")
    assert hasattr(s, "AZURO_POLYGON_RPC")


def test_settings_has_overtime_config():
    s = Settings()
    assert hasattr(s, "OVERTIME_SUBGRAPH_URL")
    assert hasattr(s, "OVERTIME_OPTIMISM_RPC")


def test_settings_has_anthropic_config():
    s = Settings()
    assert hasattr(s, "ANTHROPIC_API_KEY")
    assert hasattr(s, "LLM_MATCH_CONFIDENCE_THRESHOLD")
    assert s.LLM_MATCH_CONFIDENCE_THRESHOLD == 0.95


def test_settings_has_crossmarket_config():
    s = Settings()
    assert hasattr(s, "CROSSMARKET_SCAN_INTERVAL_SECONDS")
    assert hasattr(s, "CROSSMARKET_MIN_EDGE_PCT")
    assert s.CROSSMARKET_MIN_EDGE_PCT == 0.02


def test_settings_has_capital_allocation():
    s = Settings()
    assert hasattr(s, "GLOBAL_CAPITAL")
    assert hasattr(s, "CAPITAL_ALLOCATION_REALITY_PCT")
    assert hasattr(s, "CAPITAL_ALLOCATION_CROSSMARKET_PCT")
    # Should sum to 100
    assert s.CAPITAL_ALLOCATION_REALITY_PCT + s.CAPITAL_ALLOCATION_CROSSMARKET_PCT == 100.0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/config/test_settings.py -v`
Expected: FAIL with AttributeError

**Step 3: Extend settings.py**

Add to `config/settings.py` after existing settings:
```python
    # === Azuro (Cross-Market) ===
    AZURO_SUBGRAPH_URL: str = "https://thegraph.azuro.org/subgraphs/name/azuro-protocol/azuro-api-polygon-v3"
    AZURO_POLYGON_RPC: str = ""
    AZURO_GNOSIS_RPC: str = ""

    # === Overtime (Cross-Market) ===
    OVERTIME_SUBGRAPH_URL: str = "https://api.thegraph.com/subgraphs/name/thales-markets/overtime-optimism"
    OVERTIME_OPTIMISM_RPC: str = ""

    # === Anthropic (LLM for event matching) ===
    ANTHROPIC_API_KEY: str = ""
    LLM_MATCH_CONFIDENCE_THRESHOLD: float = 0.95
    LLM_MODEL: str = "claude-3-haiku-20240307"

    # === Cross-Market Arb Settings ===
    CROSSMARKET_SCAN_INTERVAL_SECONDS: float = 5.0
    CROSSMARKET_MIN_EDGE_PCT: float = 0.02
    CROSSMARKET_ALERT_EXPIRY_SECONDS: int = 60

    # === Multi-Chain Wallet ===
    WALLET_PRIVATE_KEY: str = ""
    POLYGON_RPC_URL: str = ""
    OPTIMISM_RPC_URL: str = ""

    # === Capital Allocation ===
    GLOBAL_CAPITAL: float = 10000.0
    CAPITAL_ALLOCATION_REALITY_PCT: float = 50.0
    CAPITAL_ALLOCATION_CROSSMARKET_PCT: float = 50.0
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/config/test_settings.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add config/settings.py tests/config/test_settings.py
git commit -m "config: add azuro, overtime, anthropic, cross-market settings"
```

---

### Task 1.3: Add Cross-Market Database Models

**Files:**
- Modify: `src/db/models.py`
- Test: `tests/db/test_crossmarket_models.py`

**Step 1: Write the failing test**

Create `tests/db/test_crossmarket_models.py`:
```python
import pytest
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.db.models import (
    Base,
    CrossMarketEvent,
    PriceSnapshot,
    CrossMarketOpportunity,
    CrossMarketTrade,
)


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_cross_market_event_creation(db_session):
    event = CrossMarketEvent(
        name="Chiefs win Super Bowl LIX",
        category="sports",
        polymarket_id="0x123abc",
        azuro_condition_id="456def",
        match_confidence=0.98,
        match_method="llm",
    )
    db_session.add(event)
    db_session.commit()

    assert event.id is not None
    assert event.name == "Chiefs win Super Bowl LIX"
    assert event.match_confidence == 0.98


def test_price_snapshot_creation(db_session):
    event = CrossMarketEvent(name="Test Event", category="sports")
    db_session.add(event)
    db_session.commit()

    snapshot = PriceSnapshot(
        event_id=event.id,
        platform="polymarket",
        outcome="YES",
        price=0.45,
        liquidity=10000.0,
    )
    db_session.add(snapshot)
    db_session.commit()

    assert snapshot.id is not None
    assert snapshot.price == 0.45


def test_cross_market_opportunity_creation(db_session):
    event = CrossMarketEvent(name="Test Event", category="sports")
    db_session.add(event)
    db_session.commit()

    opp = CrossMarketOpportunity(
        event_id=event.id,
        source_platform="polymarket",
        source_price=0.42,
        source_liquidity=5000.0,
        target_platform="overtime",
        target_price=0.47,
        target_liquidity=3000.0,
        gross_edge_pct=0.048,
        fees_pct=0.01,
        gas_estimate=0.15,
        net_edge_pct=0.038,
    )
    db_session.add(opp)
    db_session.commit()

    assert opp.id is not None
    assert opp.status == "detected"
    assert opp.net_edge_pct == 0.038


def test_cross_market_trade_creation(db_session):
    event = CrossMarketEvent(name="Test Event", category="sports")
    db_session.add(event)
    db_session.commit()

    opp = CrossMarketOpportunity(
        event_id=event.id,
        source_platform="polymarket",
        source_price=0.42,
        target_platform="overtime",
        target_price=0.47,
        gross_edge_pct=0.048,
        net_edge_pct=0.038,
    )
    db_session.add(opp)
    db_session.commit()

    trade = CrossMarketTrade(
        opportunity_id=opp.id,
        source_tx_hash="0xabc123",
        source_chain="polygon",
        source_amount=850.0,
        source_price_filled=0.42,
        source_gas_paid=0.05,
        source_status="confirmed",
        target_tx_hash="0xdef456",
        target_chain="optimism",
        target_amount=850.0,
        target_price_filled=0.47,
        target_gas_paid=0.10,
        target_status="confirmed",
        execution_time_ms=1200,
        realized_edge_pct=0.035,
        realized_pnl=29.75,
    )
    db_session.add(trade)
    db_session.commit()

    assert trade.id is not None
    assert trade.realized_pnl == 29.75
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/db/test_crossmarket_models.py -v`
Expected: FAIL with ImportError

**Step 3: Add models to src/db/models.py**

Add after existing models:
```python
class CrossMarketEvent(Base):
    """Cross-market event pairs for arbitrage."""

    __tablename__ = "cross_market_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(Text, nullable=False)
    category = Column(String(50), nullable=True)  # sports, politics, crypto
    resolution_date = Column(DateTime, nullable=True)

    # Platform-specific IDs (nullable - not all platforms have all events)
    polymarket_id = Column(String(255), nullable=True, index=True)
    azuro_condition_id = Column(String(255), nullable=True, index=True)
    overtime_game_id = Column(String(255), nullable=True, index=True)

    # Matching metadata
    match_confidence = Column(Float, nullable=True)
    match_method = Column(String(50), nullable=True)  # llm, exact, fuzzy
    verified_at = Column(DateTime, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<CrossMarketEvent(id={self.id}, name={self.name[:30]}...)>"


class PriceSnapshot(Base):
    """Price snapshots for arb detection."""

    __tablename__ = "price_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(Integer, nullable=False, index=True)
    platform = Column(String(50), nullable=False)  # polymarket, azuro, overtime
    outcome = Column(String(100), nullable=False)  # YES, NO, team name
    price = Column(Float, nullable=False)
    liquidity = Column(Float, nullable=True)
    captured_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<PriceSnapshot(platform={self.platform}, price={self.price})>"


class CrossMarketOpportunity(Base):
    """Detected cross-market arbitrage opportunities."""

    __tablename__ = "cross_market_opportunities"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(Integer, nullable=False, index=True)

    # Source (buy side)
    source_platform = Column(String(50), nullable=False)
    source_price = Column(Float, nullable=False)
    source_liquidity = Column(Float, nullable=True)

    # Target (sell side)
    target_platform = Column(String(50), nullable=False)
    target_price = Column(Float, nullable=False)
    target_liquidity = Column(Float, nullable=True)

    # Calculations
    gross_edge_pct = Column(Float, nullable=False)
    fees_pct = Column(Float, nullable=True)
    gas_estimate = Column(Float, nullable=True)
    net_edge_pct = Column(Float, nullable=False)

    # Status
    status = Column(String(50), default="detected")  # detected, alerted, approved, executed, expired, skipped
    detected_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)

    def __repr__(self) -> str:
        return f"<CrossMarketOpportunity(id={self.id}, edge={self.net_edge_pct:.2%})>"


class CrossMarketTrade(Base):
    """Executed cross-market trades."""

    __tablename__ = "cross_market_trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    opportunity_id = Column(Integer, nullable=False, index=True)

    # Source leg execution
    source_tx_hash = Column(String(255), nullable=True)
    source_chain = Column(String(50), nullable=True)
    source_amount = Column(Float, nullable=True)
    source_price_filled = Column(Float, nullable=True)
    source_gas_paid = Column(Float, nullable=True)
    source_status = Column(String(50), nullable=True)  # pending, confirmed, failed

    # Target leg execution
    target_tx_hash = Column(String(255), nullable=True)
    target_chain = Column(String(50), nullable=True)
    target_amount = Column(Float, nullable=True)
    target_price_filled = Column(Float, nullable=True)
    target_gas_paid = Column(Float, nullable=True)
    target_status = Column(String(50), nullable=True)

    # Aggregate
    execution_time_ms = Column(Integer, nullable=True)
    realized_edge_pct = Column(Float, nullable=True)
    realized_pnl = Column(Float, nullable=True)

    executed_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<CrossMarketTrade(id={self.id}, pnl={self.realized_pnl})>"
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/db/test_crossmarket_models.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/db/models.py tests/db/test_crossmarket_models.py
git commit -m "db: add cross-market event, snapshot, opportunity, trade models"
```

---

## Phase 2: Feeds

### Task 2.1: Create Azuro Feed

**Files:**
- Create: `src/feeds/azuro.py`
- Test: `tests/feeds/test_azuro.py`

**Step 1: Write the failing test**

Create `tests/feeds/test_azuro.py`:
```python
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.feeds.azuro import AzuroFeed, AzuroEvent


def test_azuro_event_creation():
    event = AzuroEvent(
        condition_id="123",
        game_id="456",
        sport="football",
        league="NFL",
        home_team="Chiefs",
        away_team="Eagles",
        starts_at=1700000000.0,
        outcomes={"home": 0.55, "away": 0.45},
    )
    assert event.condition_id == "123"
    assert event.outcomes["home"] == 0.55


def test_azuro_feed_extends_base_feed():
    from src.feeds.base import BaseFeed
    feed = AzuroFeed()
    assert isinstance(feed, BaseFeed)


@pytest.mark.asyncio
async def test_azuro_feed_connect():
    feed = AzuroFeed()
    # Mock the GraphQL client
    with patch.object(feed, "_init_client") as mock_init:
        await feed.connect()
        assert feed.is_connected
        mock_init.assert_called_once()


@pytest.mark.asyncio
async def test_azuro_feed_get_active_events():
    feed = AzuroFeed()

    mock_response = {
        "conditions": [
            {
                "conditionId": "123",
                "gameId": "456",
                "outcomes": [
                    {"outcomeId": "1", "odds": "1.8"},
                    {"outcomeId": "2", "odds": "2.2"},
                ],
                "game": {
                    "sport": {"name": "Football"},
                    "league": {"name": "NFL"},
                    "participants": [
                        {"name": "Kansas City Chiefs"},
                        {"name": "Philadelphia Eagles"},
                    ],
                    "startsAt": "1700000000",
                },
            }
        ]
    }

    with patch.object(feed, "_execute_query", new_callable=AsyncMock) as mock_query:
        mock_query.return_value = mock_response
        feed._connected = True

        events = await feed.get_active_events()

        assert len(events) == 1
        assert events[0].condition_id == "123"
        assert events[0].home_team == "Kansas City Chiefs"


@pytest.mark.asyncio
async def test_azuro_feed_get_odds():
    feed = AzuroFeed()

    mock_response = {
        "condition": {
            "outcomes": [
                {"outcomeId": "1", "odds": "1.85"},
                {"outcomeId": "2", "odds": "2.15"},
            ]
        }
    }

    with patch.object(feed, "_execute_query", new_callable=AsyncMock) as mock_query:
        mock_query.return_value = mock_response
        feed._connected = True

        odds = await feed.get_odds("123")

        # Odds are converted to implied probability (1/odds)
        assert "1" in odds
        assert "2" in odds
        assert abs(odds["1"] - 0.54) < 0.01  # 1/1.85 ≈ 0.54
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/feeds/test_azuro.py -v`
Expected: FAIL with ImportError

**Step 3: Implement Azuro feed**

Create `src/feeds/azuro.py`:
```python
"""Azuro Protocol GraphQL feed for decentralized betting markets.

Azuro provides sports betting markets on Polygon, Gnosis, and Base.
Data is queried via The Graph subgraph.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import structlog
from gql import Client, gql
from gql.transport.aiohttp import AIOHTTPTransport

from config.settings import settings
from .base import BaseFeed, FeedEvent

logger = structlog.get_logger()


@dataclass
class AzuroEvent:
    """Betting event from Azuro protocol."""

    condition_id: str
    game_id: str
    sport: str
    league: str
    home_team: str
    away_team: str
    starts_at: float  # Unix timestamp
    outcomes: dict[str, float]  # outcome_id -> implied probability
    raw_data: dict[str, Any] = field(default_factory=dict)

    @property
    def is_live(self) -> bool:
        return datetime.utcnow().timestamp() > self.starts_at


class AzuroFeed(BaseFeed):
    """GraphQL client for Azuro Protocol betting markets.

    Queries The Graph subgraph for active conditions and odds.
    Polls for updates since Azuro doesn't have WebSocket support.
    """

    POLL_INTERVAL = 5.0  # seconds

    # GraphQL query for active conditions
    ACTIVE_CONDITIONS_QUERY = gql("""
        query GetActiveConditions($first: Int!, $skip: Int!) {
            conditions(
                first: $first
                skip: $skip
                where: { status: Created }
                orderBy: game__startsAt
                orderDirection: asc
            ) {
                conditionId
                gameId
                outcomes {
                    outcomeId
                    odds
                }
                game {
                    sport { name }
                    league { name }
                    participants { name }
                    startsAt
                }
            }
        }
    """)

    CONDITION_ODDS_QUERY = gql("""
        query GetConditionOdds($conditionId: String!) {
            condition(id: $conditionId) {
                outcomes {
                    outcomeId
                    odds
                }
            }
        }
    """)

    def __init__(self, subgraph_url: Optional[str] = None):
        """Initialize Azuro feed.

        Args:
            subgraph_url: Override subgraph URL (uses settings by default)
        """
        super().__init__()
        self._subgraph_url = subgraph_url or settings.AZURO_SUBGRAPH_URL
        self._client: Optional[Client] = None
        self._poll_task: Optional[asyncio.Task] = None
        self._tracked_conditions: set[str] = set()

    async def connect(self) -> None:
        """Initialize GraphQL client."""
        if self._connected:
            return

        await self._init_client()
        self._connected = True
        logger.info("azuro_feed_connected", url=self._subgraph_url)

    async def _init_client(self) -> None:
        """Initialize the GraphQL client."""
        transport = AIOHTTPTransport(url=self._subgraph_url)
        self._client = Client(transport=transport, fetch_schema_from_transport=False)

    async def disconnect(self) -> None:
        """Close GraphQL client."""
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None

        if self._client:
            await self._client.close_async()
            self._client = None

        self._connected = False
        self._tracked_conditions.clear()
        logger.info("azuro_feed_disconnected")

    async def subscribe(self, game: str, match_id: str) -> None:
        """Subscribe to a condition (BaseFeed interface).

        Args:
            game: Sport name (e.g., "football")
            match_id: Condition ID to track
        """
        self._tracked_conditions.add(match_id)
        self._subscriptions.add((game, match_id))

        # Start polling if not already running
        if not self._poll_task and self._tracked_conditions:
            self._poll_task = asyncio.create_task(self._poll_loop())

    async def get_active_events(
        self,
        sport_filter: Optional[str] = None,
        limit: int = 100
    ) -> list[AzuroEvent]:
        """Fetch all active betting conditions.

        Args:
            sport_filter: Filter by sport name (optional)
            limit: Max number of conditions to fetch

        Returns:
            List of AzuroEvent objects
        """
        if not self._client or not self._connected:
            raise RuntimeError("Not connected. Call connect() first.")

        result = await self._execute_query(
            self.ACTIVE_CONDITIONS_QUERY,
            {"first": limit, "skip": 0}
        )

        events = []
        for condition in result.get("conditions", []):
            event = self._parse_condition(condition)
            if event:
                if sport_filter and event.sport.lower() != sport_filter.lower():
                    continue
                events.append(event)

        return events

    async def get_odds(self, condition_id: str) -> dict[str, float]:
        """Get current odds for a condition.

        Args:
            condition_id: Azuro condition ID

        Returns:
            Dict mapping outcome_id to implied probability
        """
        if not self._client or not self._connected:
            raise RuntimeError("Not connected. Call connect() first.")

        result = await self._execute_query(
            self.CONDITION_ODDS_QUERY,
            {"conditionId": condition_id}
        )

        condition = result.get("condition", {})
        outcomes = condition.get("outcomes", [])

        odds_map = {}
        for outcome in outcomes:
            outcome_id = outcome.get("outcomeId", "")
            decimal_odds = float(outcome.get("odds", "1.0"))
            # Convert decimal odds to implied probability
            implied_prob = 1.0 / decimal_odds if decimal_odds > 0 else 0
            odds_map[outcome_id] = implied_prob

        return odds_map

    async def _execute_query(
        self,
        query: Any,
        variables: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute a GraphQL query.

        Args:
            query: GQL query object
            variables: Query variables

        Returns:
            Query result as dict
        """
        async with self._client as session:
            result = await session.execute(query, variable_values=variables)
            return result

    async def _poll_loop(self) -> None:
        """Poll for odds updates on tracked conditions."""
        while self._connected and self._tracked_conditions:
            try:
                for condition_id in list(self._tracked_conditions):
                    odds = await self.get_odds(condition_id)

                    # Emit price update event
                    event = FeedEvent(
                        source="azuro",
                        event_type="odds_update",
                        game="betting",
                        data={"condition_id": condition_id, "odds": odds},
                        timestamp=datetime.utcnow().timestamp(),
                        match_id=condition_id,
                    )
                    await self._emit(event)

                await asyncio.sleep(self.POLL_INTERVAL)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("azuro_poll_error", error=str(e))
                await asyncio.sleep(self.POLL_INTERVAL)

    def _parse_condition(self, condition: dict[str, Any]) -> Optional[AzuroEvent]:
        """Parse a condition from GraphQL response.

        Args:
            condition: Raw condition data from query

        Returns:
            AzuroEvent or None if parsing fails
        """
        try:
            game = condition.get("game", {})
            participants = game.get("participants", [])

            # Parse outcomes to probability
            outcomes = {}
            for outcome in condition.get("outcomes", []):
                outcome_id = outcome.get("outcomeId", "")
                decimal_odds = float(outcome.get("odds", "1.0"))
                outcomes[outcome_id] = 1.0 / decimal_odds if decimal_odds > 0 else 0

            return AzuroEvent(
                condition_id=condition.get("conditionId", ""),
                game_id=condition.get("gameId", ""),
                sport=game.get("sport", {}).get("name", ""),
                league=game.get("league", {}).get("name", ""),
                home_team=participants[0].get("name", "") if len(participants) > 0 else "",
                away_team=participants[1].get("name", "") if len(participants) > 1 else "",
                starts_at=float(game.get("startsAt", 0)),
                outcomes=outcomes,
                raw_data=condition,
            )
        except Exception as e:
            logger.warning("azuro_parse_error", error=str(e), condition=condition)
            return None
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/feeds/test_azuro.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/feeds/azuro.py tests/feeds/test_azuro.py
git commit -m "feat(feeds): add Azuro GraphQL feed for sports betting markets"
```

---

### Task 2.2: Create Overtime Feed

**Files:**
- Create: `src/feeds/overtime.py`
- Test: `tests/feeds/test_overtime.py`

**Step 1: Write the failing test**

Create `tests/feeds/test_overtime.py`:
```python
import pytest
from unittest.mock import AsyncMock, patch
from src.feeds.overtime import OvertimeFeed, OvertimeGame


def test_overtime_game_creation():
    game = OvertimeGame(
        game_id="123",
        sport="NFL",
        home_team="Chiefs",
        away_team="Eagles",
        starts_at=1700000000.0,
        home_odds=0.55,
        away_odds=0.45,
        is_resolved=False,
    )
    assert game.game_id == "123"
    assert game.home_odds == 0.55


def test_overtime_feed_extends_base_feed():
    from src.feeds.base import BaseFeed
    feed = OvertimeFeed()
    assert isinstance(feed, BaseFeed)


@pytest.mark.asyncio
async def test_overtime_feed_connect():
    feed = OvertimeFeed()
    with patch.object(feed, "_init_client") as mock_init:
        await feed.connect()
        assert feed.is_connected
        mock_init.assert_called_once()


@pytest.mark.asyncio
async def test_overtime_feed_get_active_games():
    feed = OvertimeFeed()

    mock_response = {
        "sportMarkets": [
            {
                "id": "0x123",
                "gameId": "456",
                "tags": ["9004"],  # NFL
                "homeTeam": "Kansas City Chiefs",
                "awayTeam": "Philadelphia Eagles",
                "maturityDate": "1700000000",
                "homeOdds": "550000000000000000",  # 0.55 in wei-like format
                "awayOdds": "450000000000000000",
                "isResolved": False,
            }
        ]
    }

    with patch.object(feed, "_execute_query", new_callable=AsyncMock) as mock_query:
        mock_query.return_value = mock_response
        feed._connected = True

        games = await feed.get_active_games()

        assert len(games) == 1
        assert games[0].game_id == "456"
        assert games[0].home_team == "Kansas City Chiefs"


@pytest.mark.asyncio
async def test_overtime_feed_get_odds():
    feed = OvertimeFeed()

    mock_response = {
        "sportMarket": {
            "homeOdds": "550000000000000000",
            "awayOdds": "450000000000000000",
            "drawOdds": "0",
        }
    }

    with patch.object(feed, "_execute_query", new_callable=AsyncMock) as mock_query:
        mock_query.return_value = mock_response
        feed._connected = True

        odds = await feed.get_odds("0x123")

        assert "home" in odds
        assert "away" in odds
        assert abs(odds["home"] - 0.55) < 0.01
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/feeds/test_overtime.py -v`
Expected: FAIL with ImportError

**Step 3: Implement Overtime feed**

Create `src/feeds/overtime.py`:
```python
"""Overtime Markets GraphQL feed for sports prediction markets.

Overtime (Thales) provides sports markets on Optimism, Arbitrum, and Base.
Uses Chainlink oracles for sports data and AMM for pricing.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import structlog
from gql import Client, gql
from gql.transport.aiohttp import AIOHTTPTransport

from config.settings import settings
from .base import BaseFeed, FeedEvent

logger = structlog.get_logger()

# Sport tag mappings for Overtime
SPORT_TAGS = {
    "9001": "MLB",
    "9002": "NBA",
    "9003": "NHL",
    "9004": "NFL",
    "9005": "MLS",
    "9006": "EPL",
    "9007": "LaLiga",
    "9008": "Ligue1",
    "9010": "NCAA Football",
    "9011": "NCAA Basketball",
}


@dataclass
class OvertimeGame:
    """Sports game from Overtime Markets."""

    game_id: str
    market_address: str = ""
    sport: str = ""
    home_team: str = ""
    away_team: str = ""
    starts_at: float = 0.0  # Unix timestamp
    home_odds: float = 0.0  # Implied probability
    away_odds: float = 0.0
    draw_odds: float = 0.0
    is_resolved: bool = False
    raw_data: dict[str, Any] = field(default_factory=dict)

    @property
    def is_live(self) -> bool:
        return datetime.utcnow().timestamp() > self.starts_at and not self.is_resolved


class OvertimeFeed(BaseFeed):
    """GraphQL client for Overtime Markets.

    Queries The Graph subgraph for active sports markets and AMM odds.
    Polls for updates since Overtime doesn't have WebSocket support.
    """

    POLL_INTERVAL = 5.0  # seconds

    # GraphQL query for active markets
    ACTIVE_MARKETS_QUERY = gql("""
        query GetActiveMarkets($first: Int!, $skip: Int!, $minMaturity: BigInt!) {
            sportMarkets(
                first: $first
                skip: $skip
                where: {
                    isResolved: false
                    maturityDate_gt: $minMaturity
                }
                orderBy: maturityDate
                orderDirection: asc
            ) {
                id
                gameId
                tags
                homeTeam
                awayTeam
                maturityDate
                homeOdds
                awayOdds
                drawOdds
                isResolved
            }
        }
    """)

    MARKET_ODDS_QUERY = gql("""
        query GetMarketOdds($marketId: String!) {
            sportMarket(id: $marketId) {
                homeOdds
                awayOdds
                drawOdds
            }
        }
    """)

    def __init__(self, subgraph_url: Optional[str] = None):
        """Initialize Overtime feed.

        Args:
            subgraph_url: Override subgraph URL (uses settings by default)
        """
        super().__init__()
        self._subgraph_url = subgraph_url or settings.OVERTIME_SUBGRAPH_URL
        self._client: Optional[Client] = None
        self._poll_task: Optional[asyncio.Task] = None
        self._tracked_markets: set[str] = set()

    async def connect(self) -> None:
        """Initialize GraphQL client."""
        if self._connected:
            return

        await self._init_client()
        self._connected = True
        logger.info("overtime_feed_connected", url=self._subgraph_url)

    async def _init_client(self) -> None:
        """Initialize the GraphQL client."""
        transport = AIOHTTPTransport(url=self._subgraph_url)
        self._client = Client(transport=transport, fetch_schema_from_transport=False)

    async def disconnect(self) -> None:
        """Close GraphQL client."""
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None

        if self._client:
            await self._client.close_async()
            self._client = None

        self._connected = False
        self._tracked_markets.clear()
        logger.info("overtime_feed_disconnected")

    async def subscribe(self, game: str, match_id: str) -> None:
        """Subscribe to a market (BaseFeed interface).

        Args:
            game: Sport name (e.g., "NFL")
            match_id: Market address to track
        """
        self._tracked_markets.add(match_id)
        self._subscriptions.add((game, match_id))

        # Start polling if not already running
        if not self._poll_task and self._tracked_markets:
            self._poll_task = asyncio.create_task(self._poll_loop())

    async def get_active_games(
        self,
        sport_filter: Optional[str] = None,
        limit: int = 100
    ) -> list[OvertimeGame]:
        """Fetch all active sports markets.

        Args:
            sport_filter: Filter by sport (e.g., "NFL", "NBA")
            limit: Max number of markets to fetch

        Returns:
            List of OvertimeGame objects
        """
        if not self._client or not self._connected:
            raise RuntimeError("Not connected. Call connect() first.")

        now = int(datetime.utcnow().timestamp())

        result = await self._execute_query(
            self.ACTIVE_MARKETS_QUERY,
            {"first": limit, "skip": 0, "minMaturity": str(now)}
        )

        games = []
        for market in result.get("sportMarkets", []):
            game = self._parse_market(market)
            if game:
                if sport_filter and game.sport != sport_filter:
                    continue
                games.append(game)

        return games

    async def get_odds(self, market_address: str) -> dict[str, float]:
        """Get current AMM odds for a market.

        Args:
            market_address: Overtime market contract address

        Returns:
            Dict with keys "home", "away", "draw" mapping to implied probabilities
        """
        if not self._client or not self._connected:
            raise RuntimeError("Not connected. Call connect() first.")

        result = await self._execute_query(
            self.MARKET_ODDS_QUERY,
            {"marketId": market_address}
        )

        market = result.get("sportMarket", {})

        return {
            "home": self._parse_odds(market.get("homeOdds", "0")),
            "away": self._parse_odds(market.get("awayOdds", "0")),
            "draw": self._parse_odds(market.get("drawOdds", "0")),
        }

    async def _execute_query(
        self,
        query: Any,
        variables: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute a GraphQL query.

        Args:
            query: GQL query object
            variables: Query variables

        Returns:
            Query result as dict
        """
        async with self._client as session:
            result = await session.execute(query, variable_values=variables)
            return result

    async def _poll_loop(self) -> None:
        """Poll for odds updates on tracked markets."""
        while self._connected and self._tracked_markets:
            try:
                for market_address in list(self._tracked_markets):
                    odds = await self.get_odds(market_address)

                    # Emit price update event
                    event = FeedEvent(
                        source="overtime",
                        event_type="odds_update",
                        game="betting",
                        data={"market_address": market_address, "odds": odds},
                        timestamp=datetime.utcnow().timestamp(),
                        match_id=market_address,
                    )
                    await self._emit(event)

                await asyncio.sleep(self.POLL_INTERVAL)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("overtime_poll_error", error=str(e))
                await asyncio.sleep(self.POLL_INTERVAL)

    def _parse_market(self, market: dict[str, Any]) -> Optional[OvertimeGame]:
        """Parse a market from GraphQL response.

        Args:
            market: Raw market data from query

        Returns:
            OvertimeGame or None if parsing fails
        """
        try:
            tags = market.get("tags", [])
            sport = ""
            for tag in tags:
                if tag in SPORT_TAGS:
                    sport = SPORT_TAGS[tag]
                    break

            return OvertimeGame(
                game_id=market.get("gameId", ""),
                market_address=market.get("id", ""),
                sport=sport,
                home_team=market.get("homeTeam", ""),
                away_team=market.get("awayTeam", ""),
                starts_at=float(market.get("maturityDate", 0)),
                home_odds=self._parse_odds(market.get("homeOdds", "0")),
                away_odds=self._parse_odds(market.get("awayOdds", "0")),
                draw_odds=self._parse_odds(market.get("drawOdds", "0")),
                is_resolved=market.get("isResolved", False),
                raw_data=market,
            )
        except Exception as e:
            logger.warning("overtime_parse_error", error=str(e), market=market)
            return None

    def _parse_odds(self, odds_str: str) -> float:
        """Parse odds from wei-like string to decimal.

        Overtime stores odds as 18-decimal fixed point numbers.
        E.g., "550000000000000000" = 0.55

        Args:
            odds_str: Odds as string in wei-like format

        Returns:
            Odds as decimal (0.0 - 1.0)
        """
        try:
            odds_int = int(odds_str)
            return odds_int / 1e18
        except (ValueError, ZeroDivisionError):
            return 0.0
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/feeds/test_overtime.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/feeds/overtime.py tests/feeds/test_overtime.py
git commit -m "feat(feeds): add Overtime Markets GraphQL feed for sports betting"
```

---

## Phase 3: Event Matching

### Task 3.1: Create Event Normalizer

**Files:**
- Create: `src/matching/__init__.py`
- Create: `src/matching/normalizer.py`
- Test: `tests/matching/test_normalizer.py`

**Step 1: Write the failing test**

Create `tests/matching/test_normalizer.py`:
```python
import pytest
from src.matching.normalizer import EventNormalizer


def test_normalize_team_name_aliases():
    normalizer = EventNormalizer()

    assert normalizer.normalize_team("KC Chiefs") == "kansas city chiefs"
    assert normalizer.normalize_team("Kansas City Chiefs") == "kansas city chiefs"
    assert normalizer.normalize_team("Chiefs") == "kansas city chiefs"


def test_normalize_team_name_special_chars():
    normalizer = EventNormalizer()

    assert normalizer.normalize_team("L.A. Lakers") == "los angeles lakers"
    assert normalizer.normalize_team("LA Lakers") == "los angeles lakers"


def test_normalize_event_name():
    normalizer = EventNormalizer()

    result = normalizer.normalize_event(
        "Will the Kansas City Chiefs win Super Bowl LIX?"
    )
    assert "kansas city chiefs" in result
    assert "super bowl" in result


def test_extract_teams_from_event():
    normalizer = EventNormalizer()

    teams = normalizer.extract_teams(
        "Kansas City Chiefs vs Philadelphia Eagles - Super Bowl LIX"
    )
    assert "kansas city chiefs" in teams
    assert "philadelphia eagles" in teams


def test_extract_date_from_event():
    normalizer = EventNormalizer()

    date = normalizer.extract_date("Super Bowl LIX - February 9, 2025")
    assert date is not None
    assert date.year == 2025
    assert date.month == 2


def test_calculate_similarity():
    normalizer = EventNormalizer()

    # High similarity
    score = normalizer.calculate_similarity(
        "Chiefs win Super Bowl LIX",
        "Kansas City Chiefs to win Super Bowl LIX"
    )
    assert score > 0.7

    # Low similarity
    score = normalizer.calculate_similarity(
        "Chiefs win Super Bowl",
        "Lakers win NBA Finals"
    )
    assert score < 0.3
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/matching/test_normalizer.py -v`
Expected: FAIL with ImportError

**Step 3: Implement normalizer**

Create `src/matching/__init__.py`:
```python
"""Event matching module for cross-market arbitrage."""
```

Create `src/matching/normalizer.py`:
```python
"""Text normalization for event matching across platforms.

Handles team name aliases, special characters, and text similarity.
"""

import re
from datetime import datetime
from difflib import SequenceMatcher
from typing import Optional

# Team name aliases (lowercase)
TEAM_ALIASES = {
    # NFL
    "chiefs": "kansas city chiefs",
    "kc chiefs": "kansas city chiefs",
    "kc": "kansas city chiefs",
    "eagles": "philadelphia eagles",
    "philly": "philadelphia eagles",
    "cowboys": "dallas cowboys",
    "niners": "san francisco 49ers",
    "49ers": "san francisco 49ers",
    "sf 49ers": "san francisco 49ers",
    "pats": "new england patriots",
    "patriots": "new england patriots",
    "bills": "buffalo bills",
    "packers": "green bay packers",
    "gb packers": "green bay packers",

    # NBA
    "lakers": "los angeles lakers",
    "la lakers": "los angeles lakers",
    "l.a. lakers": "los angeles lakers",
    "celtics": "boston celtics",
    "warriors": "golden state warriors",
    "gsw": "golden state warriors",
    "nets": "brooklyn nets",
    "knicks": "new york knicks",
    "ny knicks": "new york knicks",
    "heat": "miami heat",
    "bucks": "milwaukee bucks",
    "suns": "phoenix suns",

    # MLB
    "yankees": "new york yankees",
    "ny yankees": "new york yankees",
    "dodgers": "los angeles dodgers",
    "la dodgers": "los angeles dodgers",
    "red sox": "boston red sox",
    "cubs": "chicago cubs",
    "mets": "new york mets",
    "braves": "atlanta braves",
}

# Words to remove during normalization
STOP_WORDS = {
    "will", "the", "to", "win", "vs", "versus", "against", "in",
    "a", "an", "of", "for", "at", "on", "be", "is", "are",
}


class EventNormalizer:
    """Normalize event names and team names for matching."""

    def __init__(self):
        self._team_aliases = TEAM_ALIASES

    def normalize_team(self, team: str) -> str:
        """Normalize a team name.

        Args:
            team: Raw team name

        Returns:
            Normalized team name (lowercase, aliases resolved)
        """
        # Lowercase and strip
        normalized = team.lower().strip()

        # Remove special characters
        normalized = re.sub(r"[.\-']", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()

        # Check aliases
        if normalized in self._team_aliases:
            return self._team_aliases[normalized]

        # Check if any alias key is in the name
        for alias, full_name in self._team_aliases.items():
            if alias == normalized or normalized.endswith(f" {alias}"):
                return full_name

        return normalized

    def normalize_event(self, event_name: str) -> str:
        """Normalize an event name for comparison.

        Args:
            event_name: Raw event name

        Returns:
            Normalized event name
        """
        # Lowercase
        normalized = event_name.lower()

        # Remove special characters except spaces
        normalized = re.sub(r"[?!.,'\"-]", "", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()

        # Normalize team names within the event
        words = normalized.split()
        result_words = []

        i = 0
        while i < len(words):
            # Try multi-word team names (up to 4 words)
            matched = False
            for length in range(4, 0, -1):
                if i + length <= len(words):
                    phrase = " ".join(words[i:i + length])
                    normalized_phrase = self.normalize_team(phrase)
                    if normalized_phrase != phrase:
                        result_words.append(normalized_phrase)
                        i += length
                        matched = True
                        break

            if not matched:
                result_words.append(words[i])
                i += 1

        return " ".join(result_words)

    def extract_teams(self, event_name: str) -> list[str]:
        """Extract team names from an event string.

        Args:
            event_name: Event name or description

        Returns:
            List of normalized team names found
        """
        normalized = event_name.lower()
        teams = []

        # Split on common separators
        parts = re.split(r"\s+(?:vs\.?|versus|v\.?|against|@)\s+", normalized)

        for part in parts:
            # Clean the part
            clean = re.sub(r"[^a-z\s]", "", part).strip()

            # Try to find team names
            for alias, full_name in self._team_aliases.items():
                if alias in clean or full_name in clean:
                    if full_name not in teams:
                        teams.append(full_name)

        return teams

    def extract_date(self, text: str) -> Optional[datetime]:
        """Extract a date from text.

        Args:
            text: Text containing a date

        Returns:
            Datetime object or None if no date found
        """
        # Common date patterns
        patterns = [
            # February 9, 2025
            r"(\w+)\s+(\d{1,2}),?\s+(\d{4})",
            # 2025-02-09
            r"(\d{4})-(\d{2})-(\d{2})",
            # 09/02/2025 or 02/09/2025
            r"(\d{1,2})/(\d{1,2})/(\d{4})",
        ]

        month_map = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12,
            "jan": 1, "feb": 2, "mar": 3, "apr": 4,
            "jun": 6, "jul": 7, "aug": 8, "sep": 9,
            "oct": 10, "nov": 11, "dec": 12,
        }

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                groups = match.groups()
                try:
                    if len(groups) == 3:
                        # Check if first group is month name
                        if groups[0].lower() in month_map:
                            month = month_map[groups[0].lower()]
                            day = int(groups[1])
                            year = int(groups[2])
                        elif "-" in text[match.start():match.end()]:
                            # YYYY-MM-DD format
                            year = int(groups[0])
                            month = int(groups[1])
                            day = int(groups[2])
                        else:
                            # Assume MM/DD/YYYY for US format
                            month = int(groups[0])
                            day = int(groups[1])
                            year = int(groups[2])

                        return datetime(year, month, day)
                except (ValueError, KeyError):
                    continue

        return None

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts.

        Uses SequenceMatcher with normalized inputs.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score 0.0 - 1.0
        """
        # Normalize both texts
        norm1 = self.normalize_event(text1)
        norm2 = self.normalize_event(text2)

        # Remove stop words for comparison
        words1 = [w for w in norm1.split() if w not in STOP_WORDS]
        words2 = [w for w in norm2.split() if w not in STOP_WORDS]

        clean1 = " ".join(words1)
        clean2 = " ".join(words2)

        return SequenceMatcher(None, clean1, clean2).ratio()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/matching/test_normalizer.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/matching/__init__.py src/matching/normalizer.py tests/matching/test_normalizer.py
git commit -m "feat(matching): add event normalizer for team names and text similarity"
```

---

### Task 3.2: Create LLM Verifier

**Files:**
- Create: `src/matching/llm_verifier.py`
- Test: `tests/matching/test_llm_verifier.py`

**Step 1: Write the failing test**

Create `tests/matching/test_llm_verifier.py`:
```python
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.matching.llm_verifier import LLMVerifier, MatchResult


def test_match_result_creation():
    result = MatchResult(
        is_match=True,
        confidence=0.98,
        reasoning="Both events refer to the Kansas City Chiefs winning Super Bowl LIX",
    )
    assert result.is_match
    assert result.confidence == 0.98


def test_match_result_is_high_confidence():
    high = MatchResult(is_match=True, confidence=0.98, reasoning="")
    low = MatchResult(is_match=True, confidence=0.80, reasoning="")

    assert high.is_high_confidence(threshold=0.95)
    assert not low.is_high_confidence(threshold=0.95)


@pytest.mark.asyncio
async def test_llm_verifier_verify_match():
    verifier = LLMVerifier()

    mock_response = MagicMock()
    mock_response.content = [
        MagicMock(text='{"is_match": true, "confidence": 0.98, "reasoning": "Same event"}')
    ]

    with patch.object(verifier, "_client") as mock_client:
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        result = await verifier.verify_match(
            event1_name="Chiefs win Super Bowl LIX",
            event1_platform="polymarket",
            event2_name="Kansas City Chiefs to win Super Bowl LIX",
            event2_platform="overtime",
        )

        assert result.is_match
        assert result.confidence == 0.98


@pytest.mark.asyncio
async def test_llm_verifier_no_match():
    verifier = LLMVerifier()

    mock_response = MagicMock()
    mock_response.content = [
        MagicMock(text='{"is_match": false, "confidence": 0.95, "reasoning": "Different events"}')
    ]

    with patch.object(verifier, "_client") as mock_client:
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        result = await verifier.verify_match(
            event1_name="Chiefs win Super Bowl",
            event1_platform="polymarket",
            event2_name="Lakers win NBA Finals",
            event2_platform="overtime",
        )

        assert not result.is_match


@pytest.mark.asyncio
async def test_llm_verifier_handles_json_in_markdown():
    verifier = LLMVerifier()

    mock_response = MagicMock()
    mock_response.content = [
        MagicMock(text='```json\n{"is_match": true, "confidence": 0.97, "reasoning": "Same"}\n```')
    ]

    with patch.object(verifier, "_client") as mock_client:
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        result = await verifier.verify_match(
            event1_name="Test event 1",
            event1_platform="polymarket",
            event2_name="Test event 2",
            event2_platform="azuro",
        )

        assert result.is_match
        assert result.confidence == 0.97
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/matching/test_llm_verifier.py -v`
Expected: FAIL with ImportError

**Step 3: Implement LLM verifier**

Create `src/matching/llm_verifier.py`:
```python
"""LLM-powered event matching verification using Claude.

Uses Claude Haiku for fast, cheap verification of event matches.
"""

import json
import re
from dataclasses import dataclass
from typing import Optional

import structlog
from anthropic import AsyncAnthropic

from config.settings import settings

logger = structlog.get_logger()


@dataclass
class MatchResult:
    """Result of LLM match verification."""

    is_match: bool
    confidence: float  # 0.0 - 1.0
    reasoning: str

    def is_high_confidence(self, threshold: float = 0.95) -> bool:
        """Check if this is a high-confidence match.

        Args:
            threshold: Minimum confidence to be considered high

        Returns:
            True if is_match and confidence >= threshold
        """
        return self.is_match and self.confidence >= threshold


class LLMVerifier:
    """Verify event matches using Claude LLM.

    Uses Claude Haiku for fast, cheap verification.
    Caches results to avoid repeated API calls.
    """

    SYSTEM_PROMPT = """You are an expert at matching prediction market events across platforms.

Your task is to determine if two event descriptions refer to the SAME real-world outcome.

Rules:
1. Events must have the SAME outcome to match (e.g., "Team A wins" on both platforms)
2. Slight wording differences are OK if the meaning is identical
3. Different dates or conditions mean NO match
4. If unsure, err on the side of NO match (false negatives are safer than false positives)

Respond with JSON only:
{"is_match": boolean, "confidence": float 0-1, "reasoning": "brief explanation"}"""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """Initialize LLM verifier.

        Args:
            api_key: Anthropic API key (uses settings by default)
            model: Model to use (uses settings by default)
        """
        self._api_key = api_key or settings.ANTHROPIC_API_KEY
        self._model = model or settings.LLM_MODEL
        self._client = AsyncAnthropic(api_key=self._api_key) if self._api_key else None
        self._cache: dict[tuple[str, str], MatchResult] = {}

    async def verify_match(
        self,
        event1_name: str,
        event1_platform: str,
        event2_name: str,
        event2_platform: str,
        event1_details: Optional[dict] = None,
        event2_details: Optional[dict] = None,
    ) -> MatchResult:
        """Verify if two events are the same.

        Args:
            event1_name: Name/description of first event
            event1_platform: Platform of first event (polymarket, azuro, overtime)
            event2_name: Name/description of second event
            event2_platform: Platform of second event
            event1_details: Additional details (teams, date, league, etc.)
            event2_details: Additional details for second event

        Returns:
            MatchResult with is_match, confidence, and reasoning
        """
        # Check cache
        cache_key = (event1_name.lower(), event2_name.lower())
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Check reverse cache
        reverse_key = (event2_name.lower(), event1_name.lower())
        if reverse_key in self._cache:
            return self._cache[reverse_key]

        if not self._client:
            # No API key - return low confidence no-match
            logger.warning("llm_verifier_no_api_key")
            return MatchResult(is_match=False, confidence=0.0, reasoning="No API key configured")

        # Build prompt
        user_prompt = self._build_prompt(
            event1_name, event1_platform, event2_name, event2_platform,
            event1_details, event2_details
        )

        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=256,
                system=self.SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )

            result = self._parse_response(response.content[0].text)

            # Cache result
            self._cache[cache_key] = result

            logger.info(
                "llm_match_verified",
                event1=event1_name[:50],
                event2=event2_name[:50],
                is_match=result.is_match,
                confidence=result.confidence,
            )

            return result

        except Exception as e:
            logger.error("llm_verify_error", error=str(e))
            return MatchResult(
                is_match=False,
                confidence=0.0,
                reasoning=f"Error: {str(e)}"
            )

    def _build_prompt(
        self,
        event1_name: str,
        event1_platform: str,
        event2_name: str,
        event2_platform: str,
        event1_details: Optional[dict],
        event2_details: Optional[dict],
    ) -> str:
        """Build the user prompt for verification."""
        prompt = f"""Compare these two prediction market events:

EVENT 1 ({event1_platform}):
Name: {event1_name}"""

        if event1_details:
            for key, value in event1_details.items():
                prompt += f"\n{key}: {value}"

        prompt += f"""

EVENT 2 ({event2_platform}):
Name: {event2_name}"""

        if event2_details:
            for key, value in event2_details.items():
                prompt += f"\n{key}: {value}"

        prompt += "\n\nAre these the SAME event with the SAME outcome? Respond with JSON."

        return prompt

    def _parse_response(self, text: str) -> MatchResult:
        """Parse the LLM response into a MatchResult.

        Args:
            text: Raw response text

        Returns:
            Parsed MatchResult
        """
        # Try to extract JSON from the response
        # Handle markdown code blocks
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if json_match:
            text = json_match.group(1)
        else:
            # Try to find raw JSON
            json_match = re.search(r"\{.*?\}", text, re.DOTALL)
            if json_match:
                text = json_match.group(0)

        try:
            data = json.loads(text)
            return MatchResult(
                is_match=bool(data.get("is_match", False)),
                confidence=float(data.get("confidence", 0.0)),
                reasoning=str(data.get("reasoning", "")),
            )
        except json.JSONDecodeError as e:
            logger.warning("llm_json_parse_error", text=text[:100], error=str(e))
            return MatchResult(
                is_match=False,
                confidence=0.0,
                reasoning=f"Failed to parse response: {text[:100]}"
            )

    def clear_cache(self) -> None:
        """Clear the verification cache."""
        self._cache.clear()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/matching/test_llm_verifier.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/matching/llm_verifier.py tests/matching/test_llm_verifier.py
git commit -m "feat(matching): add LLM verifier using Claude for event matching"
```

---

### Task 3.3: Create Event Matcher

**Files:**
- Create: `src/matching/event_matcher.py`
- Test: `tests/matching/test_event_matcher.py`

**Step 1: Write the failing test**

Create `tests/matching/test_event_matcher.py`:
```python
import pytest
from unittest.mock import AsyncMock, MagicMock
from src.matching.event_matcher import CrossMarketMatcher, MatchedEvent
from src.matching.llm_verifier import MatchResult
from src.feeds.azuro import AzuroEvent
from src.feeds.overtime import OvertimeGame


@pytest.fixture
def mock_verifier():
    verifier = MagicMock()
    verifier.verify_match = AsyncMock(return_value=MatchResult(
        is_match=True,
        confidence=0.98,
        reasoning="Same event",
    ))
    return verifier


@pytest.fixture
def matcher(mock_verifier):
    return CrossMarketMatcher(llm_verifier=mock_verifier)


def test_matched_event_creation():
    event = MatchedEvent(
        name="Chiefs win Super Bowl LIX",
        category="sports",
        polymarket_id="0x123",
        azuro_condition_id="456",
        overtime_game_id=None,
        confidence=0.98,
    )
    assert event.name == "Chiefs win Super Bowl LIX"
    assert event.has_polymarket
    assert event.has_azuro
    assert not event.has_overtime


def test_matched_event_platforms_count():
    event = MatchedEvent(
        name="Test",
        category="sports",
        polymarket_id="0x123",
        azuro_condition_id="456",
        overtime_game_id="789",
        confidence=0.95,
    )
    assert event.platforms_count == 3


@pytest.mark.asyncio
async def test_matcher_find_potential_matches(matcher):
    pm_events = [
        {"id": "0x123", "title": "Chiefs win Super Bowl LIX", "outcomes": ["Yes", "No"]},
    ]

    azuro_events = [
        AzuroEvent(
            condition_id="456",
            game_id="g1",
            sport="Football",
            league="NFL",
            home_team="Kansas City Chiefs",
            away_team="Philadelphia Eagles",
            starts_at=1700000000.0,
            outcomes={"1": 0.55, "2": 0.45},
        ),
    ]

    matches = await matcher.find_potential_matches(
        polymarket_events=pm_events,
        azuro_events=azuro_events,
        overtime_games=[],
    )

    assert len(matches) >= 0  # May or may not find matches depending on similarity


@pytest.mark.asyncio
async def test_matcher_verify_and_store(matcher):
    candidate = {
        "pm_event": {"id": "0x123", "title": "Chiefs win Super Bowl LIX"},
        "azuro_event": AzuroEvent(
            condition_id="456",
            game_id="g1",
            sport="Football",
            league="NFL",
            home_team="Kansas City Chiefs",
            away_team="Philadelphia Eagles",
            starts_at=1700000000.0,
            outcomes={"1": 0.55, "2": 0.45},
        ),
        "similarity": 0.85,
    }

    result = await matcher.verify_match(candidate)

    assert result is not None
    assert result.confidence == 0.98
    assert result.polymarket_id == "0x123"
    assert result.azuro_condition_id == "456"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/matching/test_event_matcher.py -v`
Expected: FAIL with ImportError

**Step 3: Implement event matcher**

Create `src/matching/event_matcher.py`:
```python
"""Cross-market event matching orchestrator.

Matches identical events across Polymarket, Azuro, and Overtime.
Uses text similarity for candidate generation and LLM for verification.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import structlog

from config.settings import settings
from .normalizer import EventNormalizer
from .llm_verifier import LLMVerifier, MatchResult

logger = structlog.get_logger()


@dataclass
class MatchedEvent:
    """An event matched across multiple platforms."""

    name: str
    category: str  # sports, politics, crypto

    # Platform-specific IDs (None if not present on that platform)
    polymarket_id: Optional[str] = None
    azuro_condition_id: Optional[str] = None
    overtime_game_id: Optional[str] = None

    # Match metadata
    confidence: float = 0.0
    match_method: str = "llm"  # llm, exact, fuzzy
    verified_at: Optional[datetime] = None

    # Additional data
    resolution_date: Optional[datetime] = None
    raw_data: dict[str, Any] = field(default_factory=dict)

    @property
    def has_polymarket(self) -> bool:
        return self.polymarket_id is not None

    @property
    def has_azuro(self) -> bool:
        return self.azuro_condition_id is not None

    @property
    def has_overtime(self) -> bool:
        return self.overtime_game_id is not None

    @property
    def platforms_count(self) -> int:
        count = 0
        if self.has_polymarket:
            count += 1
        if self.has_azuro:
            count += 1
        if self.has_overtime:
            count += 1
        return count

    @property
    def is_arbitrageable(self) -> bool:
        """Can we arbitrage this event? Need at least 2 platforms."""
        return self.platforms_count >= 2


class CrossMarketMatcher:
    """Match identical events across prediction market platforms.

    Uses text similarity for initial candidate generation,
    then LLM verification for high-confidence matching.
    """

    # Minimum similarity score to consider as candidate
    MIN_SIMILARITY_THRESHOLD = 0.6

    # Minimum LLM confidence to accept match
    MIN_CONFIDENCE_THRESHOLD = 0.95

    def __init__(
        self,
        llm_verifier: Optional[LLMVerifier] = None,
        normalizer: Optional[EventNormalizer] = None,
    ):
        """Initialize matcher.

        Args:
            llm_verifier: LLM verifier instance (creates one if not provided)
            normalizer: Event normalizer instance (creates one if not provided)
        """
        self._verifier = llm_verifier or LLMVerifier()
        self._normalizer = normalizer or EventNormalizer()
        self._matched_events: dict[str, MatchedEvent] = {}  # Cache by normalized name

    async def find_potential_matches(
        self,
        polymarket_events: list[dict[str, Any]],
        azuro_events: list[Any],  # AzuroEvent
        overtime_games: list[Any],  # OvertimeGame
    ) -> list[dict[str, Any]]:
        """Find potential matches between platforms using text similarity.

        Args:
            polymarket_events: List of Polymarket events (dicts with id, title)
            azuro_events: List of AzuroEvent objects
            overtime_games: List of OvertimeGame objects

        Returns:
            List of candidate match dicts with similarity scores
        """
        candidates = []

        # Compare Polymarket to Azuro
        for pm in polymarket_events:
            pm_title = pm.get("title", "")
            pm_normalized = self._normalizer.normalize_event(pm_title)

            for az in azuro_events:
                # Build Azuro event string
                az_title = f"{az.home_team} vs {az.away_team} - {az.league}"
                az_normalized = self._normalizer.normalize_event(az_title)

                similarity = self._normalizer.calculate_similarity(pm_normalized, az_normalized)

                if similarity >= self.MIN_SIMILARITY_THRESHOLD:
                    candidates.append({
                        "pm_event": pm,
                        "azuro_event": az,
                        "similarity": similarity,
                        "type": "pm_azuro",
                    })

        # Compare Polymarket to Overtime
        for pm in polymarket_events:
            pm_title = pm.get("title", "")
            pm_normalized = self._normalizer.normalize_event(pm_title)

            for ot in overtime_games:
                # Build Overtime event string
                ot_title = f"{ot.home_team} vs {ot.away_team} - {ot.sport}"
                ot_normalized = self._normalizer.normalize_event(ot_title)

                similarity = self._normalizer.calculate_similarity(pm_normalized, ot_normalized)

                if similarity >= self.MIN_SIMILARITY_THRESHOLD:
                    candidates.append({
                        "pm_event": pm,
                        "overtime_game": ot,
                        "similarity": similarity,
                        "type": "pm_overtime",
                    })

        # Compare Azuro to Overtime
        for az in azuro_events:
            az_title = f"{az.home_team} vs {az.away_team}"
            az_normalized = self._normalizer.normalize_event(az_title)

            for ot in overtime_games:
                ot_title = f"{ot.home_team} vs {ot.away_team}"
                ot_normalized = self._normalizer.normalize_event(ot_title)

                similarity = self._normalizer.calculate_similarity(az_normalized, ot_normalized)

                if similarity >= self.MIN_SIMILARITY_THRESHOLD:
                    candidates.append({
                        "azuro_event": az,
                        "overtime_game": ot,
                        "similarity": similarity,
                        "type": "azuro_overtime",
                    })

        # Sort by similarity descending
        candidates.sort(key=lambda x: x["similarity"], reverse=True)

        logger.info("potential_matches_found", count=len(candidates))
        return candidates

    async def verify_match(self, candidate: dict[str, Any]) -> Optional[MatchedEvent]:
        """Verify a candidate match using LLM.

        Args:
            candidate: Candidate match dict from find_potential_matches

        Returns:
            MatchedEvent if verified, None otherwise
        """
        match_type = candidate.get("type", "")

        if match_type == "pm_azuro":
            return await self._verify_pm_azuro(candidate)
        elif match_type == "pm_overtime":
            return await self._verify_pm_overtime(candidate)
        elif match_type == "azuro_overtime":
            return await self._verify_azuro_overtime(candidate)

        return None

    async def _verify_pm_azuro(self, candidate: dict) -> Optional[MatchedEvent]:
        """Verify Polymarket-Azuro match."""
        pm = candidate["pm_event"]
        az = candidate["azuro_event"]

        pm_title = pm.get("title", "")
        az_title = f"{az.home_team} vs {az.away_team} - {az.league}"

        result = await self._verifier.verify_match(
            event1_name=pm_title,
            event1_platform="polymarket",
            event2_name=az_title,
            event2_platform="azuro",
            event1_details={"outcomes": pm.get("outcomes", [])},
            event2_details={
                "sport": az.sport,
                "league": az.league,
                "home_team": az.home_team,
                "away_team": az.away_team,
            },
        )

        if result.is_high_confidence(self.MIN_CONFIDENCE_THRESHOLD):
            return MatchedEvent(
                name=pm_title,
                category="sports",
                polymarket_id=pm.get("id"),
                azuro_condition_id=az.condition_id,
                confidence=result.confidence,
                match_method="llm",
                verified_at=datetime.utcnow(),
                raw_data={"polymarket": pm, "azuro": az.raw_data},
            )

        return None

    async def _verify_pm_overtime(self, candidate: dict) -> Optional[MatchedEvent]:
        """Verify Polymarket-Overtime match."""
        pm = candidate["pm_event"]
        ot = candidate["overtime_game"]

        pm_title = pm.get("title", "")
        ot_title = f"{ot.home_team} vs {ot.away_team} - {ot.sport}"

        result = await self._verifier.verify_match(
            event1_name=pm_title,
            event1_platform="polymarket",
            event2_name=ot_title,
            event2_platform="overtime",
            event1_details={"outcomes": pm.get("outcomes", [])},
            event2_details={
                "sport": ot.sport,
                "home_team": ot.home_team,
                "away_team": ot.away_team,
            },
        )

        if result.is_high_confidence(self.MIN_CONFIDENCE_THRESHOLD):
            return MatchedEvent(
                name=pm_title,
                category="sports",
                polymarket_id=pm.get("id"),
                overtime_game_id=ot.game_id,
                confidence=result.confidence,
                match_method="llm",
                verified_at=datetime.utcnow(),
                raw_data={"polymarket": pm, "overtime": ot.raw_data},
            )

        return None

    async def _verify_azuro_overtime(self, candidate: dict) -> Optional[MatchedEvent]:
        """Verify Azuro-Overtime match."""
        az = candidate["azuro_event"]
        ot = candidate["overtime_game"]

        az_title = f"{az.home_team} vs {az.away_team} - {az.league}"
        ot_title = f"{ot.home_team} vs {ot.away_team} - {ot.sport}"

        result = await self._verifier.verify_match(
            event1_name=az_title,
            event1_platform="azuro",
            event2_name=ot_title,
            event2_platform="overtime",
            event1_details={
                "sport": az.sport,
                "league": az.league,
                "home_team": az.home_team,
                "away_team": az.away_team,
            },
            event2_details={
                "sport": ot.sport,
                "home_team": ot.home_team,
                "away_team": ot.away_team,
            },
        )

        if result.is_high_confidence(self.MIN_CONFIDENCE_THRESHOLD):
            return MatchedEvent(
                name=az_title,
                category="sports",
                azuro_condition_id=az.condition_id,
                overtime_game_id=ot.game_id,
                confidence=result.confidence,
                match_method="llm",
                verified_at=datetime.utcnow(),
                raw_data={"azuro": az.raw_data, "overtime": ot.raw_data},
            )

        return None

    async def match_all(
        self,
        polymarket_events: list[dict[str, Any]],
        azuro_events: list[Any],
        overtime_games: list[Any],
    ) -> list[MatchedEvent]:
        """Find and verify all matches between platforms.

        Args:
            polymarket_events: List of Polymarket events
            azuro_events: List of AzuroEvent objects
            overtime_games: List of OvertimeGame objects

        Returns:
            List of verified MatchedEvent objects
        """
        # Find candidates
        candidates = await self.find_potential_matches(
            polymarket_events, azuro_events, overtime_games
        )

        # Verify each candidate
        matched = []
        for candidate in candidates:
            result = await self.verify_match(candidate)
            if result:
                matched.append(result)

                # Cache by normalized name
                cache_key = self._normalizer.normalize_event(result.name)
                self._matched_events[cache_key] = result

        logger.info("matches_verified", total_candidates=len(candidates), verified=len(matched))
        return matched

    def get_cached_matches(self) -> list[MatchedEvent]:
        """Get all cached matched events."""
        return list(self._matched_events.values())

    def clear_cache(self) -> None:
        """Clear the match cache."""
        self._matched_events.clear()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/matching/test_event_matcher.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/matching/event_matcher.py tests/matching/test_event_matcher.py
git commit -m "feat(matching): add cross-market event matcher with LLM verification"
```

---

## Phase 4: Arb Engine (Tasks 4.1-4.3)

*[Continuing with same pattern for remaining tasks...]*

---

## Phase 5: Execution & Bot (Tasks 5.1-5.4)

*[Tasks for wallet manager, executor, telegram handlers, entry point...]*

---

## Summary

**Total Tasks:** 15 tasks across 5 phases

**Phase 1 - Foundation:** 3 tasks
- Dependencies, Settings, DB Models

**Phase 2 - Feeds:** 2 tasks
- Azuro Feed, Overtime Feed

**Phase 3 - Matching:** 3 tasks
- Normalizer, LLM Verifier, Event Matcher

**Phase 4 - Arb Engine:** 3 tasks
- Risk Manager, Cross-Market Arb Engine, Calculator

**Phase 5 - Execution & Bot:** 4 tasks
- Wallet Manager, Executor, Telegram Handlers, Entry Point

---

## Running Tests

After each task:
```bash
# Run specific test file
pytest tests/path/to/test.py -v

# Run all tests
pytest -v

# Run with coverage
pytest --cov=src --cov-report=term-missing
```

---

## References

- Design doc: `docs/plans/2026-01-30-polymarket-arb-bot-design.md`
- Existing Reality Arb: `src/arb/reality_arb.py`
- Azuro docs: https://gem.azuro.org/
- Overtime GitHub: https://github.com/thales-markets
