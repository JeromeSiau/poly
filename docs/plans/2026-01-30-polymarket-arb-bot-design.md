# Cross-Market Arbitrage Bot — Design Document

**Date:** 2026-01-30
**Status:** Approved
**Region:** EU/France (crypto-only markets)
**Integration:** Extends existing Reality Arb infrastructure

---

## Overview

Cross-market arbitrage module that spots price discrepancies between crypto prediction markets (Polymarket ↔ Azuro ↔ Overtime), executes trades automatically, and shares risk management with the existing Reality Arb system.

**This is a new strategy module, not a replacement.** It runs alongside the existing Reality Arb (esports broadcast lag) system.

---

## Strategy Comparison

| Aspect | Reality Arb (existant) | Cross-Market Arb (nouveau) |
|--------|------------------------|---------------------------|
| **Edge type** | Information (broadcast lag ~300ms) | Price (same event, different prices) |
| **Data sources** | PandaScore → Polymarket | Polymarket ↔ Azuro ↔ Overtime |
| **Event types** | Esports only (LoL, CS:GO, Dota2) | Sports, politics, crypto, all |
| **Trigger** | Game event detected | Price discrepancy detected |
| **Timing** | Reactive (event → immediate trade) | Continuous (price scanning loop) |
| **Risk profile** | Higher edge, shorter window | Lower edge, longer window |

---

## Markets

All crypto-native, no geo-restrictions:

| Platform | Chain | Focus | API | Status |
|----------|-------|-------|-----|--------|
| **Polymarket** | Polygon | Politics, crypto, events | REST + WebSocket | ✅ Existant |
| **Azuro** | Polygon, Gnosis, Base | Sports, esports | GraphQL (The Graph) | 🆕 À implémenter |
| **Overtime** | Optimism, Arbitrum, Base | Sports (NFL, NBA, etc.) | GraphQL, Chainlink | 🆕 À implémenter |

**Arb pairs:**
- Polymarket ↔ Azuro (sports + political overlap)
- Polymarket ↔ Overtime (sports events)
- Azuro ↔ Overtime (pure sports arb)

---

## Integration Architecture

### What's Reused (from Reality Arb)

```
src/
├── feeds/
│   ├── base.py              ✅ REUSE - BaseFeed, FeedEvent abstractions
│   └── polymarket.py        ✅ REUSE - WebSocket feed, orderbook
│
├── db/
│   ├── database.py          ✅ REUSE - Async SQLite/Postgres
│   └── models.py            🔄 EXTEND - Add new tables
│
├── bot/
│   └── telegram.py          ✅ REUSE - Bot setup (si extrait)
│
└── config/
    └── settings.py          🔄 EXTEND - Add Azuro/Overtime config
```

### What's New (Cross-Market Arb)

```
src/
├── feeds/
│   ├── azuro.py             🆕 NEW - Azuro GraphQL feed
│   └── overtime.py          🆕 NEW - Overtime GraphQL feed
│
├── matching/                🆕 NEW MODULE
│   ├── event_matcher.py     # Orchestrates cross-market matching
│   ├── normalizer.py        # Team/event name normalization
│   └── llm_verifier.py      # Claude API for match verification
│
├── arb/
│   └── cross_market_arb.py  🆕 NEW - Price arb engine
│
├── risk/                    🆕 NEW MODULE (shared)
│   ├── manager.py           # Unified risk limits
│   └── exposure.py          # Cross-chain exposure tracking
│
└── bot/
    └── crossmarket_handlers.py  🆕 NEW - Telegram handlers
```

### Entry Points (Independent)

```
scripts/
├── run_reality_arb.py       ✅ EXISTING - Esports arb
├── run_crossmarket_arb.py   🆕 NEW - Price arb
└── run_all.py               🆕 NEW - Both strategies (optional)
```

---

## Updated Project Structure

```
poly/
├── config/
│   ├── settings.py              # 🔄 Extended with new configs
│   ├── chains.py                # 🆕 Chain configs (RPCs, contracts)
│   └── settings.example.py
│
├── src/
│   ├── feeds/
│   │   ├── base.py              # ✅ Existing - abstract interface
│   │   ├── polymarket.py        # ✅ Existing - WS orderbook
│   │   ├── pandascore.py        # ✅ Existing - esports events
│   │   ├── azuro.py             # 🆕 Azuro GraphQL client
│   │   └── overtime.py          # 🆕 Overtime GraphQL client
│   │
│   ├── matching/                # 🆕 New module
│   │   ├── __init__.py
│   │   ├── event_matcher.py     # Cross-market event matching
│   │   ├── normalizer.py        # Text normalization
│   │   └── llm_verifier.py      # Claude API verification
│   │
│   ├── realtime/
│   │   ├── event_detector.py    # ✅ Existing - esports classifier
│   │   └── market_mapper.py     # ✅ Existing - esports→PM mapping
│   │
│   ├── ml/
│   │   ├── train.py             # ✅ Existing
│   │   ├── features.py          # ✅ Existing
│   │   └── data_collector.py    # ✅ Existing
│   │
│   ├── arb/
│   │   ├── reality_arb.py       # ✅ Existing - broadcast lag arb
│   │   └── cross_market_arb.py  # 🆕 Price discrepancy arb
│   │
│   ├── risk/                    # 🆕 New module (shared)
│   │   ├── __init__.py
│   │   ├── manager.py           # Unified risk management
│   │   └── exposure.py          # Cross-chain position tracking
│   │
│   ├── wallet/                  # 🆕 New module
│   │   ├── __init__.py
│   │   ├── manager.py           # Multi-chain wallet ops
│   │   ├── gas.py               # Gas price monitoring
│   │   └── balances.py          # Cross-chain balances
│   │
│   ├── bot/
│   │   ├── reality_handlers.py      # ✅ Existing
│   │   └── crossmarket_handlers.py  # 🆕 New handlers
│   │
│   ├── dashboard/               # 🆕 New module
│   │   ├── __init__.py
│   │   ├── app.py               # FastAPI dashboard
│   │   └── templates/
│   │
│   └── db/
│       ├── models.py            # 🔄 Extended with new tables
│       └── database.py          # ✅ Existing
│
├── scripts/
│   ├── run_reality_arb.py       # ✅ Existing
│   ├── run_crossmarket_arb.py   # 🆕 New entry point
│   ├── run_all.py               # 🆕 Both strategies
│   └── sync_events.py           # 🆕 Initial cross-market sync
│
├── tests/
│   ├── ... existing tests ...
│   ├── feeds/
│   │   ├── test_azuro.py        # 🆕
│   │   └── test_overtime.py     # 🆕
│   ├── matching/
│   │   ├── test_event_matcher.py    # 🆕
│   │   └── test_llm_verifier.py     # 🆕
│   ├── arb/
│   │   └── test_cross_market_arb.py # 🆕
│   └── integration/
│       └── test_crossmarket_flow.py # 🆕
│
├── data/
│   └── arb.db                   # SQLite (extended schema)
│
├── .env.example                 # 🔄 Extended
├── requirements.txt             # 🔄 Extended
└── README.md
```

---

## Database Schema Extensions

### New Tables (alongside existing)

```sql
-- Existing tables (unchanged):
-- game_events, markets, trades, positions

-- 🆕 Cross-market event pairs
cross_market_events (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,                    -- Normalized event name
    category TEXT,                         -- sports, politics, crypto
    resolution_date TIMESTAMP,

    -- Platform-specific IDs (nullable)
    polymarket_id TEXT,
    azuro_condition_id TEXT,
    overtime_game_id TEXT,

    -- Matching metadata
    match_confidence REAL,                 -- 0.0 - 1.0
    match_method TEXT,                     -- 'llm', 'exact', 'fuzzy'
    verified_at TIMESTAMP,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)

-- 🆕 Price snapshots for arb detection
price_snapshots (
    id INTEGER PRIMARY KEY,
    event_id INTEGER REFERENCES cross_market_events(id),
    platform TEXT NOT NULL,                -- 'polymarket', 'azuro', 'overtime'
    outcome TEXT NOT NULL,                 -- 'yes', 'no', team name
    price REAL NOT NULL,                   -- 0.0 - 1.0
    liquidity REAL,                        -- Available liquidity
    captured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)

-- 🆕 Cross-market opportunities
cross_market_opportunities (
    id INTEGER PRIMARY KEY,
    event_id INTEGER REFERENCES cross_market_events(id),

    -- Source (buy side)
    source_platform TEXT NOT NULL,
    source_price REAL NOT NULL,
    source_liquidity REAL,

    -- Target (sell side)
    target_platform TEXT NOT NULL,
    target_price REAL NOT NULL,
    target_liquidity REAL,

    -- Calculations
    gross_edge_pct REAL NOT NULL,
    fees_pct REAL,
    gas_estimate REAL,
    net_edge_pct REAL NOT NULL,

    -- Status
    status TEXT DEFAULT 'detected',        -- detected, alerted, approved, executed, expired, skipped
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP
)

-- 🆕 Cross-market trades (extends existing trades concept)
cross_market_trades (
    id INTEGER PRIMARY KEY,
    opportunity_id INTEGER REFERENCES cross_market_opportunities(id),

    -- Execution details per leg
    source_tx_hash TEXT,
    source_chain TEXT,
    source_amount REAL,
    source_price_filled REAL,
    source_gas_paid REAL,
    source_status TEXT,                    -- pending, confirmed, failed

    target_tx_hash TEXT,
    target_chain TEXT,
    target_amount REAL,
    target_price_filled REAL,
    target_gas_paid REAL,
    target_status TEXT,

    -- Aggregate
    execution_time_ms INTEGER,
    realized_edge_pct REAL,
    realized_pnl REAL,

    executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)

-- 🆕 Unified daily stats (both strategies)
unified_daily_stats (
    id INTEGER PRIMARY KEY,
    date DATE UNIQUE NOT NULL,

    -- Reality Arb stats
    reality_opportunities INTEGER DEFAULT 0,
    reality_trades INTEGER DEFAULT 0,
    reality_pnl REAL DEFAULT 0,

    -- Cross-Market Arb stats
    crossmarket_opportunities INTEGER DEFAULT 0,
    crossmarket_trades INTEGER DEFAULT 0,
    crossmarket_pnl REAL DEFAULT 0,

    -- Combined
    total_pnl REAL DEFAULT 0,
    total_fees REAL DEFAULT 0,
    total_gas REAL DEFAULT 0,
    net_pnl REAL DEFAULT 0
)
```

---

## Config Extensions

### settings.py additions

```python
# ============================================
# 🆕 AZURO CONFIG
# ============================================
AZURO_SUBGRAPH_URL: str = "https://thegraph.azuro.org/subgraphs/name/azuro-protocol/azuro-api-polygon-v3"
AZURO_POLYGON_RPC: str = ""  # From env
AZURO_GNOSIS_RPC: str = ""   # From env (optional)

# ============================================
# 🆕 OVERTIME CONFIG
# ============================================
OVERTIME_SUBGRAPH_URL: str = "https://api.thegraph.com/subgraphs/name/thales-markets/overtime-optimism"
OVERTIME_OPTIMISM_RPC: str = ""  # From env

# ============================================
# 🆕 ANTHROPIC (for event matching)
# ============================================
ANTHROPIC_API_KEY: str = ""
LLM_MATCH_CONFIDENCE_THRESHOLD: float = 0.95
LLM_MODEL: str = "claude-3-haiku-20240307"

# ============================================
# 🆕 CROSS-MARKET ARB SETTINGS
# ============================================
CROSSMARKET_SCAN_INTERVAL_SECONDS: float = 5.0
CROSSMARKET_MIN_EDGE_PCT: float = 2.0
CROSSMARKET_ALERT_EXPIRY_SECONDS: int = 60

# ============================================
# 🆕 MULTI-CHAIN WALLET
# ============================================
WALLET_PRIVATE_KEY: str = ""  # From env, NEVER commit
POLYGON_RPC_URL: str = ""
OPTIMISM_RPC_URL: str = ""

# ============================================
# 🔄 RISK (shared, extended)
# ============================================
# Existing: MAX_POSITION_PCT, DAILY_LOSS_LIMIT_PCT, etc.
# 🆕 Add:
GLOBAL_CAPITAL: float = 10000.0  # Total across both strategies
CAPITAL_ALLOCATION_REALITY_PCT: float = 50.0   # 50% to reality arb
CAPITAL_ALLOCATION_CROSSMARKET_PCT: float = 50.0  # 50% to cross-market
```

---

## Component Details

### 1. Azuro Feed (`src/feeds/azuro.py`)

```python
class AzuroFeed(BaseFeed):
    """
    GraphQL client for Azuro Protocol.
    Queries The Graph subgraph for active conditions and odds.
    """

    async def connect(self):
        """Initialize GraphQL client."""

    async def get_active_events(self, game_filter: str = None) -> List[AzuroEvent]:
        """Fetch all active betting conditions."""

    async def get_odds(self, condition_id: str) -> Dict[str, float]:
        """Get current odds for a condition."""

    async def subscribe_odds_updates(self, condition_ids: List[str]):
        """Poll for odds changes (Azuro doesn't have WS)."""
```

### 2. Overtime Feed (`src/feeds/overtime.py`)

```python
class OvertimeFeed(BaseFeed):
    """
    GraphQL client for Overtime Markets (Thales).
    Uses Chainlink for sports data, AMM for pricing.
    """

    async def connect(self):
        """Initialize GraphQL client."""

    async def get_active_games(self, sport: str = None) -> List[OvertimeGame]:
        """Fetch active sports markets."""

    async def get_odds(self, game_id: str) -> Dict[str, float]:
        """Get current AMM odds."""

    async def subscribe_odds_updates(self, game_ids: List[str]):
        """Poll for price changes."""
```

### 3. Event Matcher (`src/matching/event_matcher.py`)

```python
class CrossMarketMatcher:
    """
    Matches identical events across Polymarket, Azuro, and Overtime.
    Uses LLM verification for high-confidence matching.
    """

    def __init__(self, llm_verifier: LLMVerifier, db: Database):
        self.verifier = llm_verifier
        self.db = db
        self.cache: Dict[str, CrossMarketEvent] = {}

    async def find_matches(self) -> List[CrossMarketEvent]:
        """
        Scan all platforms, find matching events.
        1. Fetch active events from all platforms
        2. Normalize names (teams, dates, leagues)
        3. Group by category + approximate match
        4. LLM verify high-confidence matches
        5. Cache and return
        """

    async def refresh_matches(self):
        """Periodic refresh of event matches."""
```

### 4. Cross-Market Arb Engine (`src/arb/cross_market_arb.py`)

```python
@dataclass
class CrossMarketOpportunity:
    event: CrossMarketEvent
    source_platform: str
    source_price: float
    source_liquidity: float
    target_platform: str
    target_price: float
    target_liquidity: float
    gross_edge_pct: float
    net_edge_pct: float  # After fees + gas

    @property
    def is_valid(self) -> bool:
        return self.net_edge_pct >= settings.CROSSMARKET_MIN_EDGE_PCT


class CrossMarketArbEngine:
    """
    Detects and executes cross-market arbitrage.
    """

    def __init__(
        self,
        polymarket_feed: PolymarketFeed,
        azuro_feed: AzuroFeed,
        overtime_feed: OvertimeFeed,
        matcher: CrossMarketMatcher,
        risk_manager: RiskManager,
        executor: CrossMarketExecutor,
    ):
        ...

    async def scan_opportunities(self) -> List[CrossMarketOpportunity]:
        """
        Main scanning loop.
        1. Get matched events
        2. Fetch current prices from all platforms
        3. Calculate spreads
        4. Filter by min edge threshold
        5. Return valid opportunities
        """

    async def evaluate_opportunity(
        self,
        opp: CrossMarketOpportunity
    ) -> Optional[CrossMarketOpportunity]:
        """
        Pre-flight checks before execution.
        - Verify prices still valid
        - Check liquidity sufficient
        - Verify risk limits not exceeded
        - Calculate final position size
        """

    async def execute(self, opp: CrossMarketOpportunity) -> TradeResult:
        """
        Execute both legs of the arb.
        - Parallel transaction submission
        - Handle partial fills
        - Log results
        """
```

### 5. Unified Risk Manager (`src/risk/manager.py`)

```python
class UnifiedRiskManager:
    """
    Shared risk management for both Reality Arb and Cross-Market Arb.
    Enforces global limits across strategies.
    """

    def __init__(self, db: Database):
        self.db = db
        self.daily_pnl: float = 0.0
        self.open_exposure: Dict[str, float] = {}  # By platform

    def get_available_capital(self, strategy: str) -> float:
        """
        Returns capital available for a strategy.
        Accounts for:
        - Allocation split (50/50 default)
        - Current open positions
        - Daily loss limit remaining
        """

    def check_position_limit(self, size: float, strategy: str) -> bool:
        """Check if position size is within limits."""

    def check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit hit."""

    def record_trade(self, trade: Trade):
        """Update exposure and P&L tracking."""

    async def get_cross_chain_exposure(self) -> Dict[str, float]:
        """Get exposure by chain (Polygon, Optimism, etc.)."""
```

---

## Execution Flow

### Cross-Market Arb Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                 CROSS-MARKET ARB FLOW                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 1. EVENT MATCHING (runs periodically)                   │   │
│  │    - Fetch events from PM, Azuro, Overtime              │   │
│  │    - Normalize names                                    │   │
│  │    - LLM verify matches (cache results)                 │   │
│  └─────────────────────────┬───────────────────────────────┘   │
│                            ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 2. PRICE SCANNING (every 5 seconds)                     │   │
│  │    - For each matched event:                            │   │
│  │      - Get prices from all platforms                    │   │
│  │      - Calculate spreads                                │   │
│  │      - If spread > 2% → opportunity detected            │   │
│  └─────────────────────────┬───────────────────────────────┘   │
│                            ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 3. RISK CHECK                                           │   │
│  │    - Check unified risk manager                         │   │
│  │    - Verify capital available                           │   │
│  │    - Check daily loss limit                             │   │
│  │    - Calculate position size                            │   │
│  └─────────────────────────┬───────────────────────────────┘   │
│                            ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 4. ALERT / EXECUTE                                      │   │
│  │    Phase 1: Send Telegram alert, wait for approval      │   │
│  │    Phase 2: Auto-execute if AUTOPILOT_MODE=True         │   │
│  └─────────────────────────┬───────────────────────────────┘   │
│                            ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 5. EXECUTION (if approved)                              │   │
│  │    - Pre-flight: recheck prices                         │   │
│  │    - Build transactions for both chains                 │   │
│  │    - Submit in parallel                                 │   │
│  │    - Monitor confirmations                              │   │
│  │    - Handle partial fills                               │   │
│  │    - Log results                                        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Running Both Strategies

```
┌─────────────────────────────────────────────────────────────────┐
│                    DUAL STRATEGY MODE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────┐    ┌──────────────────────┐          │
│  │   REALITY ARB        │    │   CROSS-MARKET ARB   │          │
│  │   (Esports)          │    │   (Price Arb)        │          │
│  │                      │    │                      │          │
│  │   PandaScore Feed    │    │   Azuro + Overtime   │          │
│  │        ↓             │    │        ↓             │          │
│  │   Event Detector     │    │   Event Matcher      │          │
│  │        ↓             │    │        ↓             │          │
│  │   Reality Arb Engine │    │   CrossMarket Engine │          │
│  └──────────┬───────────┘    └──────────┬───────────┘          │
│             │                           │                       │
│             └─────────────┬─────────────┘                       │
│                           ▼                                     │
│             ┌─────────────────────────────┐                     │
│             │   UNIFIED RISK MANAGER      │                     │
│             │   - Global capital limits   │                     │
│             │   - Daily loss tracking     │                     │
│             │   - Cross-chain exposure    │                     │
│             └─────────────┬───────────────┘                     │
│                           ▼                                     │
│             ┌─────────────────────────────┐                     │
│             │   SHARED TELEGRAM BOT       │                     │
│             │   - Reality alerts          │                     │
│             │   - CrossMarket alerts      │                     │
│             │   - Unified /status         │                     │
│             └─────────────────────────────┘                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Telegram Bot Extensions

### New Commands

| Command | Scope | Description |
|---------|-------|-------------|
| `/status` | Both | Combined P&L, positions, balances |
| `/status reality` | Reality | Reality arb only stats |
| `/status crossmarket` | CrossMarket | Cross-market only stats |
| `/positions` | Both | All open positions |
| `/gas` | CrossMarket | Gas prices on all chains |
| `/matches` | CrossMarket | Show matched events |
| `/pause reality` | Reality | Pause reality arb |
| `/pause crossmarket` | CrossMarket | Pause cross-market arb |
| `/pause all` | Both | Pause everything |

### Alert Format (Cross-Market)

```
🎯 CROSS-MARKET ARB

Event: Chiefs win Super Bowl LIX
Match confidence: 98%

┌─────────────┬────────────┬────────────┐
│             │ Polymarket │  Overtime  │
├─────────────┼────────────┼────────────┤
│ YES price   │    42¢     │    47¢     │
│ Liquidity   │   $12,400  │   $8,200   │
│ Chain       │  Polygon   │  Optimism  │
└─────────────┴────────────┴────────────┘

Action: Buy YES @ Polymarket, Sell YES @ Overtime
Position: $850 (10% of cross-market capital)
Est. fees: $8.50 | Est. gas: $0.15
Net edge: 4.6%
Expected profit: ~$39

⏱️ Expires in 60s

[✅ APPROVE]  [❌ SKIP]
```

---

## Dependencies (additions to requirements.txt)

```
# 🆕 GraphQL
gql[aiohttp]>=3.5.0

# 🆕 Multi-chain Web3
web3>=6.0.0
eth-account>=0.10.0

# 🆕 LLM for event matching
anthropic>=0.18.0

# 🆕 Dashboard
fastapi>=0.109.0
uvicorn>=0.27.0
jinja2>=3.1.0
```

---

## Implementation Phases

### Phase 1: Foundation (Week 1)
- [ ] Add config extensions (settings.py, chains.py)
- [ ] Implement Azuro feed (GraphQL client)
- [ ] Implement Overtime feed (GraphQL client)
- [ ] Add new DB models (migrations)
- [ ] Write feed tests

### Phase 2: Matching (Week 2)
- [ ] Implement normalizer (team names, dates)
- [ ] Implement LLM verifier (Claude API)
- [ ] Implement event matcher
- [ ] Cache matched events
- [ ] Write matching tests

### Phase 3: Arb Engine (Week 3)
- [ ] Implement cross-market arb engine
- [ ] Implement opportunity detection
- [ ] Integrate with unified risk manager
- [ ] Write arb tests

### Phase 4: Execution (Week 4)
- [ ] Implement wallet manager (multi-chain)
- [ ] Implement parallel transaction execution
- [ ] Handle partial fills
- [ ] Telegram handlers for cross-market
- [ ] Integration tests

### Phase 5: Polish (Week 5)
- [ ] Dashboard (FastAPI)
- [ ] Combined entry point (run_all.py)
- [ ] Documentation
- [ ] Phase 1 testing (semi-auto)

---

## Risk Considerations

### Capital Allocation
- Default: 50% Reality Arb, 50% Cross-Market
- Configurable in settings
- Unified daily loss limit applies to combined P&L

### Chain Risk
- Funds split across Polygon + Optimism
- Bridge delays if rebalancing needed
- Gas spike protection (max gas price setting)

### Execution Risk
- Cross-market arb has longer execution window than reality arb
- Prices can move during multi-chain execution
- Partial fill handling is critical

---

## Setup Checklist

### Existing (from Reality Arb)
- [x] Polymarket wallet + API keys
- [x] PandaScore API key
- [x] Telegram bot

### New (for Cross-Market)
- [ ] Fund wallet on Optimism (ETH for gas, USDC for trading)
- [ ] Anthropic API key
- [ ] Alchemy/Infura RPC endpoints (Polygon + Optimism)

---

## References

- Polymarket docs: https://docs.polymarket.com/
- Azuro docs: https://gem.azuro.org/
- Overtime GitHub: https://github.com/thales-markets
- Azuro SDK: https://github.com/Azuro-protocol/sdk
- Existing codebase: `src/` (Reality Arb implementation)
