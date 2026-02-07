# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Poly is a cryptocurrency arbitrage trading system for prediction markets with two strategies:
- **Reality Arbitrage**: Exploits ~300ms broadcast lag in esports events (LoL, CS:GO, Dota2) on Polymarket
- **Cross-Market Arbitrage**: Detects price discrepancies across Polymarket, Azuro, and Overtime

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

# Reality arbitrage (esports)
python scripts/run_reality_arb.py --game lol [--autopilot]

# Cross-market arbitrage
python scripts/run_crossmarket_arb.py [--autopilot] [--scan-interval 5]

# Validation and reporting
python scripts/validate_model.py
python scripts/performance_report.py
```

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
- `src/arb/`: Trading engines for both strategies, position management with Kelly sizing
- `src/feeds/`: WebSocket/REST/GraphQL clients for market data (PandaScore, Polymarket, Azuro, Overtime)
- `src/matching/`: Cross-market event matching with LLM verification via Claude API
- `src/ml/`: XGBoost impact prediction model training and validation
- `src/paper_trading/`: Simulation engine with slippage modeling and Streamlit dashboard
- `src/realtime/`: Game event detection and market mapping for esports
- `src/risk/`: Unified risk management across strategies
- `src/db/`: SQLAlchemy async ORM with 10 core tables

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

## Testing Patterns

- Tests use in-memory SQLite databases
- Async tests use `@pytest.mark.asyncio`
- HTTP mocking with `respx`
- Test files mirror source structure in `tests/`
