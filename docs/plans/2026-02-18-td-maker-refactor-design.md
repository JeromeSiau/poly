# TD Maker Refactoring Design

## Context

`scripts/run_crypto_td_maker.py` is a 2,582-line monolith with ~40 methods in a single class and 12+ parallel state dictionaries. Every cleanup must touch all dicts, making it fragile. This refactoring decomposes it into focused, testable classes with a unified state model and comprehensive resilience patterns.

**Approach:** Big bang rewrite in a git worktree. Current bot stays running in prod untouched.

## File Structure

```
src/td_maker/
  __init__.py
  engine.py           ~250 lines  Orchestrator — wires components, runs async loops
  state.py            ~200 lines  MarketState, PassiveOrder, OpenPosition, MarketRegistry
  discovery.py        ~150 lines  MarketDiscovery — REST fetch, WS subscribe, Chainlink ref
  bidding.py          ~200 lines  BiddingEngine — scans markets, produces bid/skip/taker intents
  order_manager.py    ~350 lines  OrderManager — placement w/ placeholder, cancel, reconcile orphans
  fill_detector.py    ~350 lines  FillDetector — WSS listener, paper fills, CLOB reconciliation
  stop_loss.py        ~250 lines  StopLossManager — rule-based, ML exit, fair value override
  settlement.py       ~300 lines  SettlementManager — resolution APIs, PnL, DB startup recovery
  filters.py          ~200 lines  EntryFilters — Chainlink moves, fair value, time gates, ML model
  sizing.py           ~100 lines  Rung prices, order sizing, exposure budget
  status.py           ~80 lines   StatusLine formatting + shadow stats
  resilience.py       ~100 lines  clob_retry, FeedMonitor, shared utilities

scripts/run_crypto_td_maker.py  ~100 lines  CLI argparse + dependency wiring + engine.run()
```

Total: ~2,330 lines across 13 files (vs 2,582 in 1 file). Similar LOC but separated by responsibility.

## Core Data Model (state.py)

### PassiveOrder
```python
@dataclass
class PassiveOrder:
    order_id: str
    condition_id: str
    outcome: str
    token_id: str
    price: float
    size_usd: float
    placed_at: float
    cancelled_at: float | None = None
```

### OpenPosition
```python
@dataclass
class OpenPosition:
    condition_id: str
    outcome: str
    token_id: str
    entry_price: float
    size_usd: float
    shares: float
    filled_at: float
    order_legs: list[tuple[str, float]]  # (order_id, size_usd) for PnL fan-out
```

### MarketState
Single dataclass per market. Replaces all 12+ parallel dicts.

```python
@dataclass
class MarketState:
    # Identity
    condition_id: str
    slug: str
    symbol: str               # "btc/usd"
    slot_ts: int              # slot start timestamp
    token_ids: dict[str, str] # outcome -> token_id

    # Chainlink reference
    ref_price: float
    chainlink_symbol: str

    # Orders
    active_orders: dict[str, PassiveOrder]    # order_id -> order
    pending_cancels: dict[str, PassiveOrder]

    # Position (None if no fill)
    position: OpenPosition | None
    bid_max: float                   # highest bid since fill
    bid_below_exit_since: float | None

    # Ladder
    fill_count: int
    rungs_placed: set[tuple[str, int]]  # (outcome, price_cents)

    # Book
    book_history: deque               # ring buffer for ML features
    last_bids: dict[str, float]       # outcome -> last known bid

    # Lifecycle
    discovered_at: float
    awaiting_settlement: bool
    settlement_deferred_until: float | None
```

Defensive state transition methods:
- `add_order(order) -> bool` — refuses if outcome already positioned beyond rungs
- `record_fill(order_id, shares) -> bool` — refuses duplicate/over-limit fills, re-checks exposure via registry
- `move_to_pending_cancel(order_id) -> PassiveOrder | None`
- `replace_order_id(old, new)` — placeholder swap
- `is_placeholder(order_id) -> bool` — static check for `_placing_` prefix

### MarketRegistry
```python
class MarketRegistry:
    def get(cid) -> MarketState | None
    def register(market) -> None
    def remove(cid) -> None              # one-line cleanup, replaces 12-dict deletion
    def active_markets() -> list[MarketState]
    def markets_with_positions() -> list[MarketState]
    def markets_with_orders() -> list[MarketState]
    def expired_markets(now) -> list[MarketState]
    def total_exposure() -> float
    def total_pending() -> float
```

## Component Details

### TDMakerEngine (engine.py)
Zero business logic. Wires components and orchestrates 3 async loops:
- `_discovery_loop()` — periodic market discovery + prune expired
- `_maker_loop()` — event-driven on WS book update; calls `_tick()`
- `_fill_listener()` — drains WSS User channel fills

`_tick()` order is intentional and critical:
1. `order_mgr.expire_stale_cancels()` — cleanup first
2. `stop_loss.check_all()` — BEFORE circuit breaker (must detect crashes even when stale)
3. Circuit breaker gate — skip new orders if tripped
4. `fill_detector.check_paper_fills()` — paper mode
5. `fill_detector.periodic_reconcile()` — catch missed fills
6. `bidding.scan_and_place()` — new orders
7. `status.print_if_due()` — status line

Startup sequence:
1. `settlement.load_db_state()` — restore from DB
2. Connect all feeds (poly, user, chainlink)
3. `order_mgr.cancel_orphaned_orders()` — clean CLOB
4. `fill_detector.reconcile()` — catch fills missed while down

### MarketDiscovery (discovery.py)
- Fetches new 15-min crypto markets via REST
- Registers WS subscriptions in batch
- Captures Chainlink reference prices at discovery time
- Single responsibility: find markets and register them

### EntryFilters (filters.py)
Returns `FilterResult(action, reason, price)` where action = skip/maker/taker.

Filter chain:
1. Bid in range [target_bid, max_bid]
2. Min directional Chainlink move
3. Max directional Chainlink move (volatility cap)
4. Entry time gate (min/max minutes into slot)
5. Fair value check (price vs Chainlink-derived fair value)
6. ML model (optional, replaces rule-based filters when present)

Each filter is a pure method, independently testable.

### BiddingEngine (bidding.py)
- Scans all active markets per tick
- Skips markets: awaiting settlement, all rungs filled, no budget
- Produces cancel/placement intents (doesn't touch CLOB directly)
- Delegates execution to OrderManager

### OrderManager (order_manager.py)
- `place_order()` — placeholder pre-registration + 15s timeout + ghost order check
- `cancel_order()` — moves to pending_cancels, calls CLOB
- `cancel_batch()` / `place_batch()` — batch execution from BiddingEngine
- `expire_stale_cancels()` — cleanup pending_cancels > 30s with CLOB verification
- `cancel_orphaned_orders()` — startup: cancel everything on CLOB
- `_check_ghost_order()` — after placement timeout, checks if order exists on CLOB anyway

### FillDetector (fill_detector.py)
Three fill sources converge to single `_process_fill()`:
1. **WSS User channel** (live) — real-time, 4-priority order matching
2. **Paper simulation** — ask crossed / bid-through / time-at-bid with tight spread
3. **CLOB reconciliation** — periodic polling, catches missed fills

Deduplication via `_processed_fills` dict (fill_key -> timestamp), purged every 30 min.

On fill:
- `market.record_fill()` — validates and updates state
- Cancel other side (first fill only)
- Record shadow taker entry at ask
- Persist to DB
- Telegram notification

### StopLossManager (stop_loss.py)
Two modes (mutually exclusive):
- **Rule-based**: bid_max >= peak AND current_bid <= exit, with Chainlink fair value override (10s grace)
- **ML exit model**: sell when P(win) < exit_threshold

Empty book = market resolved/expired. Sets `awaiting_settlement`, does NOT attempt to sell.

Execution with 3-level defense:
1. Re-check bid before selling (may have recovered)
2. 3x retry with exponential backoff via `clob_retry`
3. After 3 consecutive tick failures: Telegram critical alert for manual intervention

Never cleans up position on failure — retries every tick.

### SettlementManager (settlement.py)
- `load_db_state()` — startup recovery, creates orphan MarketState for DB rows without discovered markets
- `prune_expired()` — settles markets past slot_end + 5 min grace
- 3-level resolution: Gamma API -> CLOB API -> fallback last bid (conservatively counts ambiguous as loss)
- Deferred settlement up to 1 hour if resolution unavailable
- Force settlement after 1h timeout

## Resilience Patterns (resilience.py)

### 1. clob_retry
Exponential backoff for all CLOB/REST calls. 3 attempts, logs each retry, raises on final failure.

### 2. FeedMonitor
Detects silent WSS disconnections via `last_message_at` timestamp. Forces reconnect when stale.

### 3. Fill idempotence
`_processed_fills` dict keyed by `{cid}:{order_id}:{trade_id}`. Purged at 30 min intervals.

### 4. Placeholder timeout
15s timeout on order placement API calls. On timeout: cleanup placeholder, then `_check_ghost_order()` to verify if order was actually created on CLOB despite local timeout.

### 5. Stop-loss escalation
Track consecutive failures per market. After 3 failed ticks: critical Telegram alert. Position never silently dropped.

### 6. Concurrent fill exposure check
`market.record_fill()` re-checks total exposure via registry before accepting. Prevents two simultaneous fills from exceeding max_exposure.

### 7. Startup reconciliation
Cross-reference CLOB open orders with DB state. Recreate positions that exist on-chain but are missing from DB (crash between fill detection and DB persist).

### 8. Reconnect replay safety
Ignore book updates for already-settled markets. Fill dedup handles replayed fill messages.

## Features Preserved (complete list)

- Ladder (multi-rung sequential DCA with scale-in)
- Stop-loss (rule-based peak->crash + ML exit model + fair value override)
- Taker shadow tracking (maker vs taker comparison)
- ML entry model (hybrid skip/maker/taker based on P(win))
- ML exit model (sell when P(win) < threshold)
- Chainlink filters (min/max move, fair value entry gate)
- Time gates (min/max entry minutes into slot)
- Paper trading mode (3-condition fill simulation)
- Circuit breaker integration (RiskGuard)
- Stale escalation (cancel all -> exit process)
- Placeholder order system (race condition safety)
- 4-priority fill matching (exact -> placeholder -> pending cancel -> broad)
- Periodic CLOB reconciliation (~60s)
- DB persistence + startup recovery
- Telegram notifications (fill, settle, stop-loss, critical alerts)
- Status line (30s interval)
- Wallet auto-sizing (live mode balance detection)
- Exposure budget enforcement
- All CLI arguments preserved

## Migration Plan

1. Create git worktree on branch `refactor/td-maker`
2. Implement `state.py` first (foundation for everything)
3. Implement each component with unit tests
4. Integration test: run refactored version in paper mode alongside current prod
5. Validate identical behavior over 24h paper session
6. Switch prod to new version

## Testing Strategy

Each component gets unit tests:
- `state.py` — MarketState transitions, MarketRegistry queries, edge cases
- `filters.py` — each filter independently, FilterResult logic
- `fill_detector.py` — dedup, 4-priority matching, paper fill conditions
- `stop_loss.py` — rule-based triggers, fair value override timing, empty book handling
- `settlement.py` — resolution fallback chain, deferred settlement, forced settlement
- `order_manager.py` — placeholder lifecycle, timeout handling, ghost order detection
- `sizing.py` — rung price computation, budget calculation

Integration tests:
- Full engine tick with mocked feeds
- Startup recovery from various DB states
- WSS disconnect + reconnect + reconciliation flow
