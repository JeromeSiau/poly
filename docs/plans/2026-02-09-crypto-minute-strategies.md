# Crypto 15-Minute Market Strategies

## Overview

Two complementary strategies on Polymarket 15-minute crypto binary markets (BTC, ETH):

- **Time Decay**: Buy the expensive side (>88c) when the gap between spot price and threshold is large enough that the current state should hold.
- **Long Vol**: Buy the cheap side (<15c) when the gap is small enough that a reversal is realistic.

Both strategies are symmetric (work for Up or Down).

## Data Sources

- **Polymarket Gamma API**: Discover active 15-min markets, get outcome prices, resolve trades
- **Binance public API**: GET `/api/v3/ticker/price` (no auth) for BTC/ETH spot price every 3s

## Entry Criteria

Common:
- Time remaining: 2-5 minutes (120-300s)
- Market is active and accepting orders

Time Decay:
- Expensive side > 88c
- Gap spot vs threshold > 0.3%

Long Vol:
- Cheap side < 15c
- Gap spot vs threshold < 0.5%

## Paper Trading

- Fixed $10 per trade, $1000 capital per strategy
- Trades stored in `data/crypto_minute_paper.jsonl`
- Tags: strategy, symbol, gap_bucket, time_bucket
- Resolution: poll market at end_time, compute PnL

## Files

- `src/arb/crypto_minute.py` — engine
- `scripts/run_crypto_minute.py` — runner
- `config/settings.py` — parameters (CRYPTO_MINUTE_* prefix)
- `tests/arb/test_crypto_minute.py` — tests
