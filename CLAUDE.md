# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Poly is a cryptocurrency arbitrage trading system for prediction markets. Active strategies:

- **Crypto TD Maker**: Passive time-decay maker on Polymarket 15-min crypto markets (BTC/ETH up/down). Buys favourite side at 0.75-0.85, holds to resolution. 76-81% win rate, 0 maker fees.
- **Kalshi TD Maker**: Same time-decay approach on Kalshi hourly crypto bracket markets. 93-96% win rate.
- **Crypto Two-Sided**: Micro-inefficiency capture on crypto up/down markets with dynamic wallet sizing.
- **Fear Selling**: Tail risk premium capture with LLM-based market classification.
- **Weather Oracle**: Forecast-based weather market discrepancies (lottery YES/NO bets).
- **Crypto Minute**: 15-min crypto volatility plays (time_decay only, long_vol disabled).
- **Auto Redeem**: Utility — automatically redeems resolved Polymarket positions via Builder Relayer (gas-free).

Legacy strategies (code exists but not actively running):
- **Reality Arbitrage**: Esports broadcast lag exploitation (`src/realtime/`, `src/matching/`)
- **Cross-Market Arbitrage**: Price discrepancies across Polymarket, Azuro, Overtime
- **Crypto Arb**: CEX vs Polymarket price mismatches
- **No-Bet Scanner**: Hype bias detection

## Setup

Python 3.13+. Install dependencies:

```bash
uv sync
cp .env.example .env  # then fill in API keys
```

No linting or formatting tooling is configured (no ruff, black, mypy).

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

# Run active strategies (all default to --paper mode)
bin/run_crypto_td_maker.sh          # crypto TD maker (paper)
bin/run_crypto_two_sided.sh         # crypto two-sided
bin/run_fear_selling.sh             # fear selling
bin/run_weather_oracle.sh           # weather oracle
bin/run_crypto_minute.sh            # crypto minute
./run scripts/run_auto_redeem.py --loop   # auto redeem (continuous)
bin/run_kalshi_td_maker.sh          # Kalshi TD maker
```

Use `./run <script.py> [args]` as a generic launcher — it handles PYTHONPATH and venv activation.

## Architecture

```
RiskGuard (circuit breaker, consecutive losses, daily loss limit)
     │
     ├── TradeManager (order execution, Telegram alerts, DB persistence)
     │        │
     │   PolymarketExecutor / KalshiExecutor
     │
     ├── Strategy Engines
     │    ├── CryptoTDMaker (scripts/run_crypto_td_maker.py)
     │    ├── CryptoTwoSidedEngine (src/arb/crypto_two_sided.py)
     │    ├── FearSellingEngine (src/arb/fear_engine.py)
     │    ├── WeatherOracleEngine (src/arb/weather_oracle.py)
     │    ├── CryptoMinuteEngine (src/arb/crypto_minute.py)
     │    └── KalshiTDMaker (scripts/run_kalshi_td_maker.py)
     │
     └── Feeds
          ├── PolymarketFeed / PolymarketUserFeed (WebSocket)
          ├── KalshiExecutor (REST + WebSocket)
          └── BinanceFeed (REST)
```

**Key module responsibilities:**
- `src/arb/`: Trading engines for all strategies, position management with Kelly sizing
- `src/execution/`: TradeManager (order lifecycle, Telegram alerts), PolymarketExecutor, PolymarketRedeemer
- `src/feeds/`: WebSocket/REST clients for market data (Polymarket, Kalshi, Binance). All inherit from `BaseFeed` ABC with standardized `FeedEvent`
- `src/risk/`: `RiskGuard` — circuit breaker, consecutive loss tracking, daily loss halt (1/4 Kelly)
- `src/db/`: SQLAlchemy async ORM (models, td_orders for maker strategies)
- `src/api/`: REST APIs for trade monitoring and benchmarking
- `src/paper_trading/`: Streamlit dashboard and metrics
- `src/screening/`: LLM-based market screener
- `src/bot/`: Telegram notification handlers

Legacy modules (not actively used):
- `src/matching/`: Cross-market event matching with LLM verification
- `src/realtime/`: Game event detection for esports
- `src/ml/`: XGBoost impact prediction model

**Key conventions:**
- All prices are 0.0-1.0 (probability space); binary markets: YES + NO = 1.0
- Scripts default to `--paper` (simulated fills); `--autopilot` or `--live` required for real orders
- Strategy tags are mandatory for two-sided experiments (used for filtering in API and dashboards)
- Logging via `structlog` — call `configure_logging()` from `src/utils/logging.py` at script startup
- All strategy shell scripts source `bin/_common.sh` which sets PYTHONPATH, proxy from `.env`, and circuit breaker defaults: `CB_MAX_LOSSES=5`, `CB_MAX_DRAWDOWN=-50`, `CB_STALE_SECONDS=30`, `CB_DAILY_LIMIT=-200` (override via env vars)

## Trades API (Production Stats)

The trades API runs on the production server and provides access to live trading stats. Source: `src/api/trades_api.py`.

**Base URL:** `http://localhost:8788` (on local server, it works with ssh tunnel)

### Data sources

The API exposes two distinct data sources with different reliability:

1. **Internal DB** (`/trades`, `/tags`) — reads from our local SQLite (`data/arb.db`). Only contains trades our bots recorded. Can miss trades if the bot crashed, didn't log properly, or if settlement wasn't tracked. Win/loss is based on `PaperTrade.pnl` which depends on our own settlement logic.

2. **Polymarket on-chain** (`/winrate`) — fetches actual wallet activity directly from `https://data-api.polymarket.com/activity` (TRADE/REDEEM/MERGE events). More reliable because it reflects what actually happened on-chain. For markets without a REDEEM (losses = shares worth $0), it cross-checks resolution via the CLOB API (`https://clob.polymarket.com/markets/{condition_id}`). Core logic in `src/api/winrate.py`.

**Use `/winrate` for accurate P&L reporting. Use `/trades` for strategy-level debugging (observations, edge, game_state).**

### Endpoints

**`GET /health`** — Health check. Returns `{"ok": true}`.

**`GET /trades`** — Trades from internal DB with observations, newest first, plus performance summary. Source: internal DB.
- `tag` (string, optional): filter by strategy_tag (substring match)
- `event_type` (string, optional): filter by event_type (exact match)
- `hours` (float, default=24, range 0.1-720): lookback window
- `limit` (int, default=200, range 1-2000): max rows

**`GET /winrate`** — Win rate and PnL from on-chain Polymarket wallet activity. Source: Polymarket Data API + CLOB API.
- `hours` (float, default=24, range 0.1-720): lookback window
- `wallet` (string, optional): wallet address (default: from settings)
- Returns: winrate, total_pnl, roi_pct, profit_factor, avg_win, avg_loss, per-market breakdown

**`GET /tags`** — Distinct strategy_tags and event_types in the lookback window. Source: internal DB.
- `hours` (float, default=24, range 0.1-720): lookback window

CLI: `./run scripts/winrate.py --hours 17` for terminal output of the same on-chain data.

## Deployment (Smart Restart)

Deploys are triggered by Ploi on `git push`. The deploy script runs `bin/smart_restart.py` which only restarts daemons whose code actually changed (via `git diff` + dependency mapping).

**IMPORTANT:** When adding a new strategy, script, or daemon, you MUST update the `RULES` mapping in `bin/smart_restart.py` so that file changes trigger the correct daemon restart. Also update the rules if you rename or move source files that are already mapped.

The dependency map works as follows:
- **Shared modules** (`src/execution/`, `src/risk/`, `src/utils/`, `config/`) -> restart ALL daemons
- **Strategy-specific modules** (e.g. `src/arb/fear_*`) -> restart only that strategy's daemon
- **Shared components** (e.g. `src/feeds/polymarket`) -> restart daemons that use them

Daemons are discovered automatically from supervisor configs in `/etc/supervisor/conf.d/worker-*.conf` and matched by command substring.

## Production Logs

Logs are on the production server and can be accessed via:

```bash
ssh ploi@94.130.218.197
# Logs directory:
cd /home/ploi/orb.lvlup-dev.com/logs
```

## Configuration

Settings are managed via Pydantic in `config/settings.py`, reading from `.env`:

```bash
# Copy template and configure
cp .env.example .env
```

Required API keys: `POLYMARKET_*`, `ANTHROPIC_API_KEY`, `TELEGRAM_BOT_TOKEN`
For Kalshi: `KALSHI_API_KEY_ID`, `KALSHI_PRIVATE_KEY_PATH` (RSA PEM)

Key risk parameters (configurable via env):
- `MAX_POSITION_PCT`: 10% of capital per trade
- `DAILY_LOSS_LIMIT_PCT`: 5% daily loss cap
- `MIN_EDGE_PCT`: 2% minimum edge to trade

## Database

Production: MySQL via `DATABASE_URL=mysql+aiomysql://user:pass@host/db`
Development/tests: SQLite via `DATABASE_URL=sqlite+aiosqlite:///data/arb.db` (default)

All tables (trades, risk_state, slot_snapshots, slot_resolutions) live in a single database. RiskGuard uses SQLAlchemy ORM (no raw SQL). No Alembic — schema migration is manual via `_migrate_add_columns()` in `src/db/database.py` (adds missing columns on init). Models split across `src/db/models.py` (legacy), `src/db/td_orders.py` (maker orders), `src/db/slot_models.py` (slot snapshots/resolutions).

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
- Crypto minute: 41K resolved BTC/ETH 15-min updown markets (Oct 2025-Feb 2026, $1.8B volume)
- Calibration: longshots (5-15c) overpriced by +0.7% (crypto) to +6.7% (macro); break-even NO win rate at 93c entry is ~93%
- Maker-taker: makers outperform takers by ~2.3pp since mid-2024 (Kalshi data)

**Existing backtests:**
- `scripts/backtest_fear_selling.py` — market-level resolution backtest with Kelly sizing, category filtering, and parameter sweep.
- `scripts/backtest_crypto_minute.py` — time_decay & long_vol backtest with DuckDB VWAP, per-1c calibration, and threshold sweep. Key finding: time_decay +1% edge at 0.80-0.94, long_vol negative EV everywhere.
- `scripts/backtest_crypto_td_maker.py` — TD maker backtest with VWAP fill simulation.
- `scripts/backtest_crypto_two_sided.py` — two-sided arbitrage backtest.

## Testing Patterns

- Tests use in-memory SQLite databases
- Async tests use `@pytest.mark.asyncio`
- HTTP mocking with `respx`
- Test files mirror source structure in `tests/`
