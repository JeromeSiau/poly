# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Poly is a cryptocurrency arbitrage trading system for prediction markets with multiple strategies:
- **Reality Arbitrage**: Exploits ~300ms broadcast lag in esports events (LoL, CS:GO, Dota2) on Polymarket
- **Cross-Market Arbitrage**: Detects price discrepancies across Polymarket, Azuro, and Overtime
- **Two-Sided Inventory**: Micro-inefficiency capture with dynamic wallet sizing
- **Fear Selling**: Tail risk premium capture with LLM-based market classification
- **Crypto Minute**: 15-min crypto volatility plays
- **Crypto Arb**: CEX vs Polymarket price mismatches
- **Weather Oracle**: Forecast-based weather market discrepancies
- **No-Be Scanner**: Hype bias detection

All strategies are coordinated by `UnifiedRiskManager` with fractional Kelly sizing, per-strategy capital allocation, and a global daily loss halt.

## Development Commands

```bash
# Run all tests
pytest

# Run specific test suite
pytest tests/db/
pytest tests/arb/

# Run with coverage
pytest -v --cov=src

# Run specific test by name
pytest -k test_models

# Initialize database
python -c "from src.db.database import init_db_async; import asyncio; asyncio.run(init_db_async())"

# Train ML model
python scripts/train_model.py

# Paper trading simulation
python scripts/paper_trade.py --model models/impact_model.pkl --game lol --capital 10000

# Run strategies (all default to --paper mode)
python scripts/run_reality_arb.py --game lol [--autopilot]
python scripts/run_crossmarket_arb.py [--autopilot] [--scan-interval 5]
./run_two_sided.sh [min_edge] [exit_edge] [strategy_tag] [paper|live] [wallet_usd]
./run_crypto_minute.sh
./run_fear_selling.sh
./run_weather_oracle.sh

# Validation and reporting
python scripts/validate_model.py
python scripts/performance_report.py
```

Use `./run <script.py> [args]` as a generic launcher — it handles PYTHONPATH and venv activation.

## Architecture

```
UnifiedRiskManager (global capital, daily loss limits)
         │
    ┌────┴────────────────┬──────────────────┐
    ▼                     ▼                  ▼
RealityArbEngine    PaperTradingEngine   CrossMarketArbEngine
    │                     │                  │
    ▼                     ▼                  ▼
EventDetector       MarketObserver      EventMatcher (LLM)
    │                     │                  │
    ▼                     ▼                  ▼
PandaScore Feed    Polymarket Feed    Azuro + Overtime Feeds
```

**Key module responsibilities:**
- `src/arb/`: Trading engines for all strategies, position management with Kelly sizing
- `src/feeds/`: WebSocket/REST/GraphQL clients for market data (PandaScore, Polymarket, Azuro, Overtime, Binance)
- `src/matching/`: Cross-market event matching with LLM verification via Claude API (0.95 confidence threshold)
- `src/ml/`: XGBoost impact prediction model training and validation
- `src/paper_trading/`: Simulation engine with slippage modeling and Streamlit dashboard
- `src/realtime/`: Game event detection and market mapping for esports
- `src/risk/`: Unified risk management across strategies (1/4 Kelly, daily loss halt)
- `src/db/`: SQLAlchemy async ORM with 10+ core tables
- `src/api/`: REST APIs for trade monitoring and benchmarking
- `src/screening/`: LLM-based market screener
- `src/bot/`: Telegram notification handlers

**Key conventions:**
- All prices are 0.0–1.0 (probability space); binary markets: YES + NO = 1.0
- Scripts default to `--paper` (simulated fills); `--autopilot` required for real orders
- Strategy tags are mandatory for two-sided experiments (used for filtering in API and dashboards)
- All feeds inherit from `BaseFeed` ABC with standardized `FeedEvent`

## Trades API (Production Stats)

The trades API runs on the production server and provides access to live trading stats. Source: `src/api/trades_api.py`.

**Base URL:** `http://localhost:8788` (on the production server)

### Endpoints

**`GET /health`** — Health check. Returns `{"ok": true}`.

**`GET /trades`** — Trades with observations, newest first, plus performance summary (wins, losses, winrate, total PnL).
- `tag` (string, optional): filter by strategy_tag (substring match)
- `event_type` (string, optional): filter by event_type (exact match)
- `hours` (float, default=24, range 0.1–720): lookback window
- `limit` (int, default=200, range 1–2000): max rows

**`GET /tags`** — Distinct strategy_tags and event_types in the lookback window.
- `hours` (float, default=24, range 0.1–720): lookback window

There is also an RN1 comparison API on port 8787 (`src/api/rn1_compare_api.py`) for benchmarking against reference wallets.

## Production Logs

Logs are on the production server and can be accessed via:

```bash
ssh ploi@5.135.136.96
# Logs directory:
cd /home/ploi/orb.lvlup-dev.com/logs
```

## Configuration

Settings are managed via Pydantic in `config/settings.py`, reading from `.env`:

```bash
# Copy template and configure
cp .env.example .env
```

Required API keys: `POLYMARKET_*`, `PANDASCORE_API_KEY`, `ANTHROPIC_API_KEY`, `TELEGRAM_BOT_TOKEN`

Key risk parameters (configurable via env):
- `MAX_POSITION_PCT`: 10% of capital per trade
- `DAILY_LOSS_LIMIT_PCT`: 5% daily loss cap
- `MIN_EDGE_PCT`: 2% minimum edge to trade

## Database

Default: SQLite at `data/arb.db` (async via aiosqlite)
Supports PostgreSQL via `DATABASE_URL` env variable

## Historical Dataset (Backtesting)

A full Polymarket + Kalshi historical dataset is available at `../prediction-market-analysis/data/` (cloned from [Jon-Becker/prediction-market-analysis](https://github.com/Jon-Becker/prediction-market-analysis)).

**Location:** `/Users/jerome/Projets/web/python/prediction-market-analysis/`

**Contents (~50 GB):**
- `data/polymarket/trades/*.parquet` — 691M on-chain CTF OrderFilled trades (45 GB), columns: block_number, maker, taker, maker_asset_id, taker_asset_id, maker_amount, taker_amount, fee
- `data/polymarket/markets/*.parquet` — 408K markets with resolution data (question, slug, outcome_prices, clob_token_ids, volume, end_date, closed)
- `data/polymarket/blocks/*.parquet` — block_number to timestamp mapping
- `data/polymarket/legacy_trades/*.parquet` — pre-2023 FPMM trades
- `data/kalshi/trades/*.parquet` — Kalshi trade history (3.3 GB)
- `data/kalshi/markets/*.parquet` — Kalshi market metadata (570 MB)

**Queryable via DuckDB** directly on parquet files — no loading into memory. Example:
```python
import duckdb
con = duckdb.connect()
df = con.execute("SELECT * FROM '../prediction-market-analysis/data/polymarket/markets/*.parquet' WHERE closed = true LIMIT 10").df()
```

**Key data for backtesting:**
- Fear selling: 7.4K fear-keyword markets resolved, token-level trade VWAP available
- Crypto minute: 41K resolved BTC/ETH 15-min updown markets (Oct 2025–Feb 2026, $1.8B volume)
- Calibration: longshots (5-15c) overpriced by +0.7% (crypto) to +6.7% (macro); break-even NO win rate at 93c entry is ~93%
- Maker-taker: makers outperform takers by ~2.3pp since mid-2024 (Kalshi data)

**Existing backtests:**
- `scripts/backtest_fear_selling.py` — market-level resolution backtest with Kelly sizing, category filtering, and parameter sweep.
- `scripts/backtest_crypto_minute.py` — time_decay & long_vol backtest with DuckDB VWAP, per-1c calibration, and threshold sweep. Key finding: time_decay +1% edge at 0.80-0.94, long_vol negative EV everywhere.

## Testing Patterns

- Tests use in-memory SQLite databases
- Async tests use `@pytest.mark.asyncio`
- HTTP mocking with `respx`
- Test files mirror source structure in `tests/`
