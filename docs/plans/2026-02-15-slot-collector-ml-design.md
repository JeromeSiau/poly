# Slot Collector + ML Prediction Pipeline

Design for passive market data collection and ML-based entry prediction for the crypto TD maker strategy.

## Goal

Replace manual entry filters (min_move_pct, max_move_pct, min_entry_minutes) with an XGBoost model that predicts P(win) for each slot and adjusts sizing by confidence.

## Decision: Go/No-Go + Sizing

The model decides:
1. Whether to enter a slot (P(win) > threshold)
2. How much to bet (order_size scaled by confidence)

Pricing (target_bid, max_bid, ladder rungs) stays manual.

## Data Collection

### Storage: MySQL

Existing MySQL on production server. Async driver: `aiomysql`.

### Schema

```sql
slot_snapshots (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    symbol          VARCHAR(10) NOT NULL,     -- BTC, ETH, SOL, XRP
    slot_ts         INT NOT NULL,             -- epoch start of 15-min slot
    captured_at     DOUBLE NOT NULL,
    minutes_into_slot FLOAT NOT NULL,

    -- Polymarket book
    bid_up          FLOAT, ask_up       FLOAT,
    bid_down        FLOAT, ask_down     FLOAT,
    bid_size_up     FLOAT, ask_size_up  FLOAT,
    bid_size_down   FLOAT, ask_size_down FLOAT,
    spread_up       FLOAT, spread_down  FLOAT,

    -- Chainlink
    chainlink_price DOUBLE,
    dir_move_pct    FLOAT,
    abs_move_pct    FLOAT,

    -- Context
    hour_utc        TINYINT,
    day_of_week     TINYINT,            -- 0=Monday

    INDEX idx_symbol_slot (symbol, slot_ts),
    INDEX idx_slot_ts (slot_ts)
)

slot_resolutions (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    symbol          VARCHAR(10) NOT NULL,
    slot_ts         INT NOT NULL,
    condition_id    VARCHAR(66),

    resolved_up     BOOLEAN,            -- NULL = not yet resolved
    prev_resolved_up BOOLEAN,
    resolved_at     DOUBLE,

    UNIQUE INDEX idx_symbol_slot (symbol, slot_ts)
)
```

### Collector Architecture

Separate daemon: `scripts/run_slot_collector.py` with its own supervisor config.

Reuses existing `PolymarketFeed` and `ChainlinkFeed` from `src/feeds/`.

Three async loops:
- **Discovery (60s):** Query Gamma API for active 15-min crypto markets, INSERT into slot_resolutions
- **Snapshot (30s):** For each active slot, capture book + chainlink state, INSERT into slot_snapshots
- **Resolution (60s):** Check expired slots via Gamma/CLOB API, UPDATE resolved_up

Volume: ~8,600 rows/day for 4 symbols (4 symbols x 96 slots x ~30 snapshots).

### Robustness

Inspired by TD maker's battle-tested patterns:

- **WS reconnection:** BaseFeed handles auto-reconnect. Stale detection at 30s forces reconnect.
- **Missing snapshots:** Tolerated. ML feature `snapshot_count` per slot lets the model discount sparse data.
- **Delayed resolution:** Retry for 60 min (same as TD maker _settle_position). After that, resolved_up=NULL, ML ignores.
- **MySQL unavailable:** Async write queue (max 1000). Flush on reconnect. Drop oldest on overflow.
- **Restart:** On startup, resume active slots and retry unresolved slot_resolutions.

## ML Pipeline

### Training (offline, `scripts/train_td_model.py`)

Run manually or weekly cron once enough data exists.

1. Query MySQL: JOIN snapshots at decision-time minutes (5m, 7m, 10m) with resolutions
2. Feature engineering: spread_ratio, move_velocity, derived features
3. Train XGBoost binary classifier (target: resolved_up)
4. Temporal train/val split (last 3 days = validation, no random split)
5. Calibration via Platt scaling or isotonic regression
6. Save model to `data/models/td_model_YYYYMMDD.joblib`

### Inference (online, in TD maker)

New CLI arg `--model-path`. At each entry decision:

```python
if self.model:
    features = self._build_features(cid, outcome)
    p_win = self.model.predict_proba(features)
    if p_win < self.min_confidence:
        continue
    # Sizing: order_size * (p_win - 0.5) / 0.5
```

Without a model, manual filters (min_move, max_move, min_entry) remain as fallback. No breaking change.

## Roadmap

### Phase 1: Collector (now)
- `src/db/slot_models.py` — SQLAlchemy models for MySQL
- `scripts/run_slot_collector.py` — daemon
- `bin/run_slot_collector.sh` + supervisor config
- Update `bin/smart_restart.py`

### Phase 2: Dashboard (after ~1 week of data)
- API endpoint `/slots` to query snapshots + resolutions
- Streamlit page: heatmap WR by (timing x move), calibration curves

### Phase 3: ML model (after 2-3 weeks of data)
- `scripts/train_td_model.py` — feature engineering + XGBoost + calibration
- `--model-path` in TD maker — inference + sizing
- Fallback on manual filters if no model

### Dependencies
- `aiomysql` (new)
- `xgboost`, `joblib` (already in project via `src/ml/`)
