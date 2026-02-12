# TradeManager — Generic Trade Execution Layer

## Problem

Every strategy script reimplements the same trade flow:
- Paper/live order placement branching (4 copies)
- Paper fill detection `ask <= price` (3 copies)
- `persist_fill()` with try/except (14 call sites across 4 files)
- Settlement PnL calculation (3 copies)
- Paper order ID counter (2 copies)

`TwoSidedPaperRecorder` is named after one strategy but used by 5 others.
No Telegram notifications except Reality Arb and Cross-Market Arb.
Kalshi has no executor — API code is inline in the script.

## Solution

A single `TradeManager` class that every strategy uses for:
1. **Order placement** (paper or live)
2. **Paper fill detection**
3. **DB persistence** (LiveObservation + PaperTrade)
4. **Telegram notifications** (bid, fill, settlement)

Strategies keep their decision logic (when to trade, what edge, when to settle).
TradeManager handles everything that happens after the decision.

## Architecture

```
Strategy (decision)           TradeManager (execution)
  ┌──────────────┐           ┌───────────────────────┐
  │ CryptoTDMaker│           │                       │
  │ FearSelling  │ place()   │  ExecutorProtocol     │──▶ Polymarket / Kalshi API
  │ TwoSided     │ fill()    │  TradeRecorder        │──▶ SQLite (LiveObs+PaperTrade)
  │ CryptoMaker  │ settle()  │  TelegramAlerter      │──▶ Telegram
  │ Sniper       │ cancel()  │                       │
  │ WeatherOracle│           └───────────────────────┘
  │ CryptoMinute │
  │ KalshiTDMaker│
  └──────────────┘
```

## New Files

### `src/execution/__init__.py`

Exports: `TradeManager`, `TradeIntent`, `PendingOrder`, `FillResult`, `OrderResult`, `ExecutorProtocol`.

### `src/execution/models.py`

Data structures moved from `src/arb/two_sided_inventory.py`:

```python
@dataclass
class TradeIntent:
    condition_id: str
    token_id: str
    outcome: str
    side: str            # "BUY" / "SELL"
    price: float
    size_usd: float
    reason: str          # "td_maker_passive", "fear_no_sell", etc.
    title: str = ""      # human-readable market name (for Telegram)
    timestamp: float = 0.0

@dataclass
class PendingOrder:
    order_id: str
    intent: TradeIntent
    placed_at: float

@dataclass
class FillResult:
    filled: bool
    shares: float
    avg_price: float
    pnl_delta: float = 0.0  # 0 at entry, computed at settle

@dataclass
class OrderResult:
    order_id: str
    filled: bool = False
    status: str = "placed"   # "placed", "filled", "error"
    error: str | None = None
```

Compat imports in `src/arb/two_sided_inventory.py` to avoid breaking existing code during migration.

### `src/execution/executor.py`

```python
class ExecutorProtocol(Protocol):
    async def place_order(self, *, token_id: str, side: str, size: float,
                          price: float, outcome: str,
                          order_type: str = "GTC") -> OrderResult: ...
    async def cancel_order(self, order_id: str) -> bool: ...
```

`PolymarketExecutor` already implements this de facto — add a thin adapter for the return type (`dict` → `OrderResult`).

Extract `KalshiExecutor` from `run_kalshi_td_maker.py` inline code (RSA auth, `_api_get`, `_api_post`, order placement) into `src/feeds/kalshi_executor.py`.

### `src/execution/trade_recorder.py`

Generic persistence extracted from `TwoSidedPaperRecorder`:

```python
class TradeRecorder:
    def __init__(self, db_url: str, *, strategy_tag: str,
                 event_type: str = "", run_id: str = ""):
        # SQLAlchemy engine + sessionmaker + bootstrap

    def record_fill(self, intent: TradeIntent, fill: FillResult,
                    fair_prices: dict[str, float] | None = None,
                    execution_mode: str = "paper") -> int:
        # LiveObservation + PaperTrade, returns observation_id

    def record_settle(self, intent: TradeIntent, fill: FillResult,
                      fair_prices: dict[str, float] | None = None) -> int:
        # Persist settlement + close related BUY records

    def bootstrap(self):
        # Create tables if needed
```

Removes two-sided-specific coupling (min_edge_pct, exit_edge_pct, replay_into_engine).
Same `game_state` JSON format — API `/trades` and dashboards keep working.

### `src/execution/trade_manager.py`

```python
class TradeManager:
    def __init__(self, *,
        executor: ExecutorProtocol | None,  # None = paper mode
        strategy: str,                       # "CryptoTDMaker", etc.
        paper: bool = True,
        db_url: str = "",
        event_type: str = "",
        notify_bids: bool = True,
        notify_fills: bool = True,
        notify_closes: bool = True,
    ):
        # Creates internally: TradeRecorder + TelegramAlerter
        # Manages paper_order_counter

    async def place(self, intent: TradeIntent) -> PendingOrder | FillResult:
        """Place order. Paper: fake ID. Live: executor.
        Persists entry + Telegram '📝📊 BID placed'"""

    def check_paper_fills(self,
        get_levels: Callable[[str, str], tuple]
    ) -> list[FillResult]:
        """Check pending paper orders against orderbook.
        BUY: fill if ask <= price. SELL: fill if bid >= price.
        Persists + Telegram '📝✅ FILLED' for each."""

    async def settle(self, condition_id: str, outcome: str,
                     settlement_price: float, won: bool) -> float:
        """Record close. Compute PnL.
        Persists + Telegram '🔥🟢 WIN' or '🔥🔴 LOSS'.
        Returns pnl."""

    async def cancel(self, order_id: str) -> bool:
        """Cancel pending order."""

    def get_pending_orders(self) -> dict[str, PendingOrder]: ...
    def get_stats(self) -> dict: ...
    async def close(self): ...
```

### `src/feeds/kalshi_executor.py`

Extracted from `run_kalshi_td_maker.py`:
- RSA auth header generation
- `_api_get()`, `_api_post()`
- `place_order()` → `OrderResult`
- `cancel_order()`

### `tests/execution/test_trade_manager.py`

Unit tests with mocked executor, recorder, and alerter.

## Telegram Message Format

```
📝📊 CryptoTDMaker              📝 = paper, 🔥 = live
BID Up @ 0.78 | $10
BTC-0212-1430

📝✅ CryptoTDMaker
FILL Up @ 0.78 | 12.8 shares
BTC-0212-1430

🔥🟢 CryptoTDMaker
WIN Up 0.78 → 1.00 | +$2.82
5W-2L | Total: +$12.40

🔥🔴 FearSelling
LOSS No 0.92 → 0.00 | -$5.00
3W-1L | Total: +$8.20
```

## Migration Plan (Big Bang)

All 8 scripts migrated at once:

| Script | What changes |
|---|---|
| `run_crypto_td_maker.py` | Replace `_place_order` + `_check_fills_paper` + `_process_fill` + `_settle_position` + `TwoSidedPaperRecorder` with `manager.place()` / `.check_paper_fills()` / `.settle()` |
| `run_crypto_maker.py` | Same as above + `_fill_listener` calls `manager.record_fill()` |
| `run_two_sided_inventory.py` | Replace `TwoSidedPaperRecorder` + 5 persist sites with manager calls |
| `run_sniper.py` | Replace `paper_recorder.persist_fill()` + `settlement_loop` |
| `run_fear_selling.py` | Replace `_persist_signal()` direct SQLAlchemy + `_check_exits()` |
| `run_kalshi_td_maker.py` | Extract `KalshiExecutor` + use manager |
| `run_weather_oracle.py` | Replace `_save_trade()` inline persistence (L810-844) |
| `run_crypto_minute.py` | Replace `_save_trade()` in `CryptoMinuteEngine` (L524) |
| `run_no_bet.py` | No migration (scanner only, no trades) |

## Typical Strategy Code After Migration

```python
# In main()
manager = TradeManager(
    executor=executor,       # or None for paper
    strategy="CryptoTDMaker",
    paper=paper_mode,
    db_url=args.db_url,
    event_type="crypto_td_maker",
)

# In the trading loop
intent = TradeIntent(condition_id=cid, token_id=token_id,
                     outcome="Up", side="BUY", price=0.78,
                     size_usd=10.0, reason="td_maker_passive",
                     title="BTC-0212-1430")
order = await manager.place(intent)

# In the tick loop
fills = manager.check_paper_fills(polymarket.get_best_levels)

# When market resolves
pnl = await manager.settle(cid, "Up", settlement_price=1.0, won=True)

# Cleanup
await manager.close()
```

3 lines replace ~40 lines of duplicated plumbing per strategy.
