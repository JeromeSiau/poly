# Polymarket Arbitrage Bot — Design Document

**Date:** 2026-01-30
**Status:** Approved
**Region:** EU/France (crypto-only markets)

## Overview

Arbitrage bot that spots price discrepancies between crypto-native prediction markets, executes trades automatically (after human approval in Phase 1), and enforces strict risk limits.

## Markets

All crypto-native, no geo-restrictions:

| Platform | Chain | Focus | API |
|----------|-------|-------|-----|
| **Polymarket** | Polygon | Politics, crypto, events | REST + WebSocket, Python SDK |
| **Azuro** | Polygon, Gnosis, Base | Sports, esports | GraphQL (The Graph), TypeScript SDK |
| **Overtime Markets** | Optimism, Arbitrum, Base | Sports (NFL, NBA, MLB, etc.) | GraphQL, Chainlink oracles |

**Why these three:**
- No KYC, no geo-blocks, fully on-chain
- Different user bases = price discrepancies
- Azuro has $250M+ volume, Overtime has $200M+ volume
- All EVM-compatible = similar tooling

**Arb pairs:**
- Polymarket ↔ Azuro (political events, some sports overlap)
- Polymarket ↔ Overtime (sports events)
- Azuro ↔ Overtime (sports arbitrage)

## Wallet Setup

Since all markets are on-chain, you need:

| Chain | Wallet | Gas Token | Betting Token |
|-------|--------|-----------|---------------|
| **Polygon** | MetaMask/EOA | MATIC | USDC (Polymarket), USDT (Azuro) |
| **Optimism** | Same wallet | ETH | USDC (Overtime) |
| **Gnosis** | Same wallet | xDAI | xDAI (Azuro) |
| **Base** | Same wallet | ETH | WETH (Azuro) |

**Recommendation:** Start with Polygon + Optimism only (Polymarket ↔ Overtime). Add Gnosis/Base later.

**Bridge consideration:** You'll need to bridge funds between chains. Use official bridges or LayerZero/Stargate for fast transfers.

## Strategy

**Pure arbitrage only:** Same event priced differently across platforms.

Example:
- "Chiefs win Super Bowl" at 45¢ on Polymarket, 50¢ on Overtime
- Buy low on Polymarket, sell high on Overtime
- Gross spread: 5¢ per share = ~11% edge before fees

No synthetic arbs or correlation plays — too much model risk for small capital.

## Risk Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Max position size | 10% of capital | ~$1K per trade, allows diversification |
| Daily loss limit | 5% of capital | Auto-halt if down ~$500 — something is broken |
| Minimum edge | 2% after fees | Below this, slippage + gas eats profit |
| Max trades/hour (Phase 2) | 10 | Prevents runaway execution |
| Anomaly threshold | 15% edge | If edge > 15%, alert instead of execute (likely stale data) |

**Fee structure to account for:**
- Polymarket: ~1% trading fee
- Azuro: ~3-5% margin built into odds
- Overtime: ~2-3% margin
- Gas fees: ~$0.01-0.10 per tx (Polygon/L2s are cheap)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    CRYPTO ARB BOT                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Polymarket  │  │    Azuro     │  │   Overtime   │          │
│  │   (Polygon)  │  │  (Polygon/   │  │  (Optimism)  │          │
│  │              │  │   Gnosis)    │  │              │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                 │                   │
│         ▼                 ▼                 ▼                   │
│  ┌─────────────────────────────────────────────────────┐       │
│  │              PRICE FEED COLLECTOR                    │       │
│  │  - Polymarket: REST API polling (5s)                │       │
│  │  - Azuro: GraphQL subgraph queries                  │       │
│  │  - Overtime: GraphQL + Chainlink price feeds        │       │
│  └─────────────────────────┬───────────────────────────┘       │
│                            ▼                                    │
│  ┌─────────────────────────────────────────────────────┐       │
│  │              EVENT MATCHER                           │       │
│  │  - LLM-verified (Claude Haiku, 95% confidence)      │       │
│  │  - Sports: team names + date + league               │       │
│  │  - Politics: candidate + election + date            │       │
│  └─────────────────────────┬───────────────────────────┘       │
│                            ▼                                    │
│  ┌─────────────────────────────────────────────────────┐       │
│  │              ARB DETECTOR                            │       │
│  │  - Calculate spread after all fees + gas            │       │
│  │  - Edge > 2%? → Trigger alert                       │       │
│  └─────────────────────────┬───────────────────────────┘       │
│                            ▼                                    │
│  ┌─────────────────────────────────────────────────────┐       │
│  │              RISK MANAGER                            │       │
│  │  - Position limits, daily loss tracking             │       │
│  │  - Cross-chain exposure monitoring                  │       │
│  └─────────────────────────┬───────────────────────────┘       │
│                            ▼                                    │
│  ┌────────────────────┐    ┌────────────────────────┐          │
│  │   TELEGRAM BOT     │    │   EXECUTION ENGINE     │          │
│  │ Alert + Approve    │───▶│ - Sign transactions    │          │
│  │                    │    │ - Submit to chains     │          │
│  └────────────────────┘    │ - Monitor confirmations│          │
│                            └────────────────────────┘          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Event Matching

**Fully automated via LLM:**

1. Extract structured data:
   - Sports: team names, league, date, match type
   - Politics: candidate/party, election name, date
2. Normalize across platforms (different naming conventions)
3. Send both event descriptions to Claude Haiku
4. Only match if confidence ≥ 95%
5. Cache approved matches in SQLite

**Example matching:**
```
Polymarket: "Will the Kansas City Chiefs win Super Bowl LIX?"
Overtime:   "NFL - Super Bowl LIX - Kansas City Chiefs"
→ LLM confidence: 98% → MATCH
```

**Safety:** If confidence < 95%, skip entirely.

## Execution Engine

**On-chain execution requires:**
1. Wallet with private key (secured via env variable)
2. Sufficient gas on each chain
3. Approved token spending for each protocol

**Flow:**
```
┌─────────────────────────────────────────────────────────────────┐
│                     EXECUTION FLOW                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. PRE-FLIGHT CHECKS                                          │
│     - Recheck prices via API (still profitable?)               │
│     - Check wallet balances on both chains                     │
│     - Verify gas prices acceptable                             │
│     - Confirm token approvals                                  │
│                                                                 │
│  2. PARALLEL TRANSACTION SUBMISSION                            │
│     - Build both transactions                                  │
│     - Sign with wallet                                         │
│     - Submit via asyncio.gather()                              │
│     - Timeout: 30 seconds for confirmation                     │
│                                                                 │
│  3. RESULT HANDLING                                            │
│     ✓ Both confirmed → Log profit, update positions            │
│     ✗ One failed → ALERT! Unhedged exposure                    │
│     ✗ Both failed → No harm, log and retry later               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Partial execution handling:**
- If only one side confirms, immediately alert via Telegram
- Show current exposure and options (hold, manual close, retry)
- Log for analysis

**Position sizing:**
```python
position_size = min(
    capital * 0.10,                    # 10% max rule
    source_liquidity * 0.5,            # Don't take >50% of book
    target_liquidity * 0.5,
    remaining_daily_limit,             # Stay under 5% daily loss cap
    max_gas_efficient_size             # Don't let gas eat >10% of profit
)
```

## Telegram Bot

**Alert format:**
```
🎯 ARB OPPORTUNITY

Event: Chiefs win Super Bowl LIX
Edge: 4.8% after fees

┌─────────────┬────────────┬────────────┐
│             │ Polymarket │  Overtime  │
├─────────────┼────────────┼────────────┤
│ YES price   │    42¢     │    47¢     │
│ Liquidity   │   $12,400  │   $8,200   │
│ Chain       │  Polygon   │  Optimism  │
└─────────────┴────────────┴────────────┘

Action: Buy YES @ Polymarket, Sell YES @ Overtime
Position: $850 (10% of capital)
Est. gas: $0.15
Expected profit: ~$40

⏱️ Expires in 60s

[✅ APPROVE]  [❌ SKIP]
```

**Commands:**
- `/status` — Positions, P&L, balances per chain
- `/pause` / `/resume` — Stop/start scanning
- `/kill` — Emergency stop all
- `/positions` — List open positions by market
- `/history` — Last 10 trades with P&L
- `/gas` — Current gas prices on all chains
- `/bridge` — Show cross-chain balance suggestions

## Data Storage

**SQLite database:**

```sql
-- Matched event pairs across platforms
events (
    id, name, category, resolution_date,
    polymarket_id, azuro_id, overtime_id,
    match_confidence, created_at
)

-- Detected arbitrage opportunities
opportunities (
    id, event_id, detected_at,
    source_platform, source_price,
    target_platform, target_price,
    edge_pct, status
)

-- Executed trades (one per side)
trades (
    id, opportunity_id, platform, chain,
    tx_hash, amount, price, gas_paid,
    status, confirmed_at
)

-- Aggregated positions per event
positions (
    id, event_id,
    polymarket_shares, azuro_shares, overtime_shares,
    entry_edge, current_value,
    status, realized_pnl
)

-- Daily performance tracking
daily_stats (
    date, opportunities_found, trades_executed,
    gross_pnl, fees_paid, gas_paid, net_pnl,
    win_rate, avg_edge
)
```

## Web Dashboard

Flask/FastAPI at `http://server:8080`:

- Real-time P&L chart
- Open positions by platform/chain
- Opportunity hit rate
- Gas spending tracker
- Cross-chain balance overview
- Event match quality metrics

## Phases

### Phase 1: Semi-Auto (Weeks 1-4)
- Bot detects opportunities, sends Telegram alert
- You approve/skip via inline buttons
- 60-second expiry window
- Goal: Validate signal quality, test execution reliability

### Phase 2: Full Autopilot (Week 5+)
- Config change: `AUTOPILOT_MODE = True`
- Auto-executes without approval
- Additional safety rails:
  - Max trades/hour: 10
  - Anomaly detection (edge > 15% = sus)
  - Hourly heartbeat messages
  - Auto-pause if 3 consecutive failures

## Project Structure

```
poly/
├── config/
│   ├── settings.py              # Risk limits, thresholds
│   ├── chains.py                # Chain configs (RPC URLs, contract addresses)
│   └── settings.example.py
│
├── src/
│   ├── feeds/
│   │   ├── base.py              # Abstract feed interface
│   │   ├── polymarket.py        # Polymarket REST/WS client
│   │   ├── azuro.py             # Azuro GraphQL client
│   │   └── overtime.py          # Overtime GraphQL client
│   │
│   ├── matching/
│   │   ├── matcher.py           # Event matching orchestrator
│   │   ├── normalizer.py        # Text normalization (teams, dates)
│   │   └── llm_verifier.py      # Claude API verification
│   │
│   ├── arb/
│   │   ├── detector.py          # Arb opportunity detection
│   │   ├── calculator.py        # Edge calculation with fees + gas
│   │   └── executor.py          # Multi-chain execution engine
│   │
│   ├── wallet/
│   │   ├── manager.py           # Wallet management, signing
│   │   ├── gas.py               # Gas price monitoring
│   │   └── balances.py          # Cross-chain balance tracking
│   │
│   ├── risk/
│   │   ├── manager.py           # Position limits, daily loss
│   │   └── monitor.py           # Real-time exposure tracking
│   │
│   ├── bot/
│   │   ├── telegram.py          # Telegram bot setup
│   │   └── handlers.py          # Command + button handlers
│   │
│   ├── dashboard/
│   │   ├── app.py               # Flask/FastAPI app
│   │   └── templates/
│   │
│   └── db/
│       ├── models.py            # SQLAlchemy models
│       └── database.py          # DB helpers
│
├── scripts/
│   ├── run_bot.py               # Main entry point
│   ├── sync_events.py           # Initial event sync
│   └── check_balances.py        # Utility: show all balances
│
├── tests/
│
├── data/
│   └── arb.db
│
├── .env.example                 # Template for secrets
├── requirements.txt
└── README.md
```

## Tech Stack

- **Language:** Python 3.11+
- **Web3:** web3.py, eth-account
- **APIs:**
  - Polymarket: `polymarket-apis` (PyPI)
  - Azuro: GraphQL via `gql` library
  - Overtime: GraphQL via `gql` library
- **LLM:** Claude Haiku (Anthropic API)
- **Database:** SQLite + SQLAlchemy
- **Telegram:** `python-telegram-bot`
- **Dashboard:** FastAPI + Jinja2
- **Deployment:** OVH dedicated server

## Dependencies

```
# requirements.txt
web3>=6.0.0
eth-account>=0.9.0
polymarket-apis>=0.1.0
gql[requests]>=3.4.0
anthropic>=0.18.0
python-telegram-bot>=20.0
fastapi>=0.109.0
uvicorn>=0.27.0
sqlalchemy>=2.0.0
httpx>=0.26.0
python-dotenv>=1.0.0
```

## Security Notes

- **Private key:** Store in `.env`, never commit
- **RPC URLs:** Use private RPCs (Alchemy/Infura) to avoid rate limits
- **Telegram bot token:** Store in `.env`
- **Anthropic API key:** Store in `.env`

## Setup Checklist

1. [ ] Create wallets (one EOA works for all EVM chains)
2. [ ] Fund wallet with:
   - MATIC on Polygon (~$10 for gas)
   - ETH on Optimism (~$10 for gas)
   - USDC on Polygon (~$500+ for trading)
   - USDC on Optimism (~$500+ for trading)
3. [ ] Get Anthropic API key
4. [ ] Create Telegram bot via @BotFather
5. [ ] Set up private RPC endpoints (Alchemy free tier works)
6. [ ] Clone repo, create `.env`, install deps
7. [ ] Run initial event sync
8. [ ] Start bot in Phase 1 mode

## References

- Polymarket docs: https://docs.polymarket.com/
- Azuro docs: https://gem.azuro.org/
- Overtime GitHub: https://github.com/thales-markets
- Azuro SDK: https://github.com/Azuro-protocol/sdk
