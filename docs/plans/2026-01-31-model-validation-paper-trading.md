# Model Validation & Paper Trading Design

> **Goal:** Validate the ML prediction model's calibration and test the trading strategy in simulation before going live.

## Overview

Two parallel workstreams:
1. **Model Validation** — Prove the model is well-calibrated (70% predicted = 70% actual wins)
2. **Paper Trading** — Simulate live trading to measure realized edge vs theoretical edge

---

## Architecture

```
src/
├── ml/
│   └── validation/
│       ├── __init__.py
│       ├── calibration.py       # Calibration plots, Brier decomposition
│       ├── backtester.py        # Historical simulation with P&L
│       ├── statistical_tests.py # Hosmer-Lemeshow, bootstrap CI
│       └── report.py            # HTML/markdown report generation
│
├── paper_trading/
│   ├── __init__.py
│   ├── engine.py                # Main orchestrator
│   ├── market_observer.py       # Capture Polymarket prices at T+0/30/60/120s
│   ├── position_manager.py      # Virtual portfolio, Kelly sizing, risk limits
│   ├── execution_sim.py         # Slippage simulation, orderbook depth
│   ├── metrics.py               # P&L, win rate, edge calculations
│   └── dashboard.py             # Streamlit dashboard
│
├── db/
│   └── models.py                # Add LiveObservation table
│
scripts/
├── validate_model.py            # Run validation suite
├── paper_trade.py               # Start paper trading
├── performance_report.py        # Generate performance report
└── recalibrate.py               # Retrain with live data
```

---

## Part 1: Model Validation

### Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Brier Score** | Mean squared error of probabilities | < 0.20 |
| **ECE** | Expected Calibration Error | < 0.05 |
| **Reliability** | Brier decomposition — how close to diagonal | Low |
| **Resolution** | Brier decomposition — spread of predictions | High |

### Calibration Analysis

**By probability bucket (10% bins):**
- For each bin [0.6-0.7], count predictions and actual win rate
- Plot reliability diagram (predicted vs actual)
- Calculate ECE = weighted average of |predicted - actual|

**By game context:**
- Game time: early (0-15min), mid (15-25min), late (25+min)
- Gold diff: behind (<-2k), even (-2k to +2k), ahead (>+2k)
- Event type: baron, dragon, tower, kill, ace

**Statistical tests:**
- Hosmer-Lemeshow test for calibration
- Bootstrap confidence intervals on Brier score

### Backtesting

**Simulated market approach:**
Since we don't have historical Polymarket prices, we create a "naive market" that:
- Uses a simple model: `price = sigmoid(gold_diff / 5000 + kill_diff / 10)`
- Reacts to events with 30-60s lag
- Our edge = how much better our model is vs this naive market

**Metrics:**
- Simulated P&L over N historical matches
- Sharpe ratio
- Max drawdown
- Win rate by event type

### Output

HTML report with:
- Reliability diagram
- Calibration heatmap (game_time x gold_diff)
- Brier score breakdown
- Go/No-Go recommendation

---

## Part 2: Paper Trading

### Data Collection

```sql
CREATE TABLE live_observations (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME NOT NULL,
    match_id VARCHAR(100) NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    game_state JSON NOT NULL,           -- gold_diff, kills, towers, game_time
    model_prediction FLOAT NOT NULL,    -- P(win) from our model
    polymarket_price FLOAT,             -- Price at T+0
    polymarket_price_30s FLOAT,         -- Price at T+30s
    polymarket_price_60s FLOAT,         -- Price at T+60s
    polymarket_price_120s FLOAT,        -- Price at T+120s
    actual_winner VARCHAR(100),         -- Filled at match end
    latency_ms INTEGER,                 -- Event detection latency
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE paper_trades (
    id INTEGER PRIMARY KEY,
    observation_id INTEGER REFERENCES live_observations(id),
    side VARCHAR(10) NOT NULL,          -- BUY or SELL
    entry_price FLOAT NOT NULL,
    simulated_fill_price FLOAT NOT NULL, -- With slippage
    size FLOAT NOT NULL,
    edge_theoretical FLOAT NOT NULL,
    edge_realized FLOAT,                -- Filled when market moves
    pnl FLOAT,                          -- Filled at match end
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### Live Flow

```
1. PandaScore WebSocket → Event received (baron_kill, 25min, +5k gold)
                              │
                              ▼
2. Model prediction ────────► P(win) = 72%
                              │
                              ▼
3. Polymarket price ────────► Current = 58%
                              │
                              ▼
4. Edge calculation ────────► Edge = 14% (72% - 58%)
                              │
                              ▼
5. Position sizing ─────────► Kelly = 12% of bankroll = $50
                              │
                              ▼
6. Execution sim ───────────► Slippage 0.5% → Fill at 58.5%
                              │
                              ▼
7. Schedule follow-ups ─────► Capture prices at T+30s, T+60s, T+120s
                              │
                              ▼
8. Log & Alert ─────────────► SQLite + Telegram notification
```

### Position Sizing (Kelly Criterion)

```python
def kelly_fraction(edge: float, odds: float) -> float:
    """
    edge: our predicted probability - market probability
    odds: decimal odds (1/market_price for YES)

    Kelly: f* = (p * b - q) / b
    where p = our prob, q = 1-p, b = odds-1
    """
    p = market_price + edge
    q = 1 - p
    b = (1 / market_price) - 1
    kelly = (p * b - q) / b
    return max(0, min(kelly, 0.25))  # Cap at 25%
```

### Risk Management

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Max position size | 25% of bankroll | Kelly cap |
| Min edge to trade | 5% | Below this, not worth the risk |
| Max daily loss | 10% of bankroll | Stop trading for the day |
| Max concurrent positions | 3 | Concentration risk |

### Execution Simulation

**Slippage model:**
```python
def estimate_slippage(size: float, orderbook_depth: float) -> float:
    """
    Estimate slippage based on trade size vs available liquidity.
    Polymarket typically has $5k-$50k depth on esports.
    """
    impact = size / orderbook_depth
    return min(impact * 0.02, 0.05)  # Max 5% slippage
```

---

## Part 3: Alerts & Dashboard

### Telegram Alerts

```
🎯 OPPORTUNITY DETECTED
Match: T1 vs Gen.G (LCK)
Event: baron_kill by T1
Game: 25:32 | Gold: +5.2k | Kills: +3

📊 Model: 72% → Market: 58%
💰 Edge: +14% | Confidence: HIGH

📝 Simulated trade:
   BUY T1 @ 0.585 (0.5% slippage)
   Size: $50 (Kelly: 12%)

⏱️ Will update at T+30s, T+60s, T+120s
```

**Follow-up alert:**
```
📈 UPDATE: T1 vs Gen.G
T+60s: Market moved 58% → 68%
Edge captured: +10% ✓
Running P&L: +$43
```

### Streamlit Dashboard

**Pages:**

1. **Overview**
   - Portfolio value chart
   - Key metrics: total P&L, win rate, avg edge
   - Recent trades table

2. **Calibration**
   - Reliability diagram
   - ECE by context heatmap
   - Brier score trend over time

3. **Trade Analysis**
   - Edge theoretical vs realized scatter
   - P&L by event type
   - Latency histogram

4. **Live Monitor**
   - Current active matches
   - Pending observations (waiting for T+30/60/120)
   - Real-time event feed

---

## Scripts

### validate_model.py

```bash
uv run python scripts/validate_model.py \
  --model models/lol_impact.pkl \
  --data data/lol_training.csv \
  --output reports/validation_2026-01-31.html
```

Outputs:
- Brier score: 0.18 ✓
- ECE: 0.04 ✓
- Calibration plot saved
- Recommendation: GO

### paper_trade.py

```bash
uv run python scripts/paper_trade.py \
  --model models/lol_impact.pkl \
  --game lol \
  --capital 10000 \
  --min-edge 0.05 \
  --telegram
```

Runs continuously, connects to PandaScore + Polymarket, logs everything.

### performance_report.py

```bash
uv run python scripts/performance_report.py \
  --since 2026-01-31 \
  --output reports/performance.html
```

Generates performance report from paper trading data.

### recalibrate.py

```bash
uv run python scripts/recalibrate.py \
  --model models/lol_impact.pkl \
  --live-data data/paper_trading.db \
  --output models/lol_impact_v2.pkl
```

Retrains model incorporating live observations with real market prices.

---

## Implementation Order

1. **Database schema** — Add LiveObservation and PaperTrade tables
2. **Calibration module** — calibration.py, statistical_tests.py, report.py
3. **validate_model.py script** — Can run immediately on existing data
4. **Market observer** — Polymarket price capture with scheduled follow-ups
5. **Paper trading engine** — Orchestrator connecting everything
6. **Telegram alerts** — Real-time notifications
7. **Streamlit dashboard** — Visualization and monitoring
8. **Backtester** — Historical simulation with naive market

---

## Success Criteria

**Before going live with real money:**

| Metric | Target |
|--------|--------|
| Brier score | < 0.20 |
| ECE | < 0.05 |
| Paper trading edge realized | > 3% average |
| Paper trading win rate | > 55% |
| Sample size | > 50 trades |
| Max drawdown | < 15% |

---

## Dependencies

```
# Add to requirements.txt
streamlit>=1.30.0
plotly>=5.18.0
scipy>=1.12.0  # For statistical tests
```

---

## Notes

- **PandaScore trial**: Check if they offer a trial API key for live data
- **Polymarket WebSocket**: Need to handle reconnection and rate limits
- **Timezone**: All timestamps in UTC
- **Logging**: Use structlog, same as rest of project
