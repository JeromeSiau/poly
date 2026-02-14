# Last-Penny Sniper

Buy quasi-certain outcomes at 0.95-0.999 across all fee-free Polymarket markets, hold to resolution.

## Inspiration

Sharky6999 ($597K profit, 22K trades, 99.3% win rate). No external feeds needed: the Polymarket orderbook price IS the signal. If a market trades at 0.999, the outcome is effectively decided.

## Architecture

```
MarketScanner (poll REST every 10-30s)
    |
    +-- Discover active markets with best_ask >= threshold
    |   Filter: no fees, or fees OK if price > 0.99
    |
    +-- Feed targets to SniperEngine
            |
            +-- WebSocket on top targets (live orderbook)
            |
            +-- When ask available at price >= threshold:
                    |
                    +-- Sizing (price-based, constant risk per trade)
                    +-- BUY taker via PolymarketExecutor
                    +-- Hold to resolution -> redeem
```

## Market Selection

Zero-fee markets (all eligible):
- Crypto hourly up/down (BTC, ETH, SOL, XRP)
- Crypto brackets ("BTC above $90K?", "ETH between $2000-$2100?")
- Sports: football (excl. Serie A), tennis, basketball (excl. NCAAB), etc.
- Politics, weather, everything else

Fee markets accepted if price > 0.99:
- 15-min crypto (fee = 0.006% at 0.999)
- NCAAB, Serie A (fee = 0.002% at 0.999)

Fee formula: `fee = p * (1-p) * r` where r = 0.0625 for crypto, 0.0175 for sports.

## Sizing

Constant risk per trade regardless of entry price:

```python
profit_per_share = 1.0 - ask_price
loss_per_share = ask_price
implied_winrate = ask_price

# Kelly: f = (p*b - q) / b, fractional (1/4), capped
risk_per_trade = capital * risk_pct  # e.g. 1% of capital
max_shares = risk_per_trade / loss_per_share
order_size_usd = max_shares * ask_price
```

Example with $500 capital, 1% risk ($5 real risk per trade):

| Entry Price | Gain/share | Max shares | Order size |
|-------------|-----------|------------|------------|
| 0.999       | $0.001    | 5.005      | $5.00      |
| 0.99        | $0.01     | 5.05       | $5.00      |
| 0.97        | $0.03     | 5.15       | $5.00      |
| 0.95        | $0.05     | 5.26       | $5.00      |

## Position Management

- No stop-loss, no early exit (hold to resolution like Sharky)
- Track total exposure (sum of all pending positions)
- Circuit breaker: halt on consecutive losses or daily loss limit
- Auto-redeem resolved positions

## Backtest

Using `../prediction-market-analysis/data/` parquet files via DuckDB.

1. Load resolved markets (markets/*.parquet, closed=true)
2. Load on-chain trades (trades/*.parquet) for those markets
3. Find trades executed at price >= threshold (= available asks that got filled)
4. Simulate taking those trades, apply resolution
5. Parameter sweep: min_price (0.90-0.999), sizing, with/without fee markets

Limitation: backtest shows upper bound. We see executed trades but not how long the ask was available. Latency matters.

Output: win rate, PnL, Sharpe, drawdown, breakdown by price bucket and market type.

## Streamlit Dashboard

New tab in existing `src/paper_trading/dashboard.py`:
- Cumulative PnL, win rate, trade count
- Current exposure (capital locked awaiting resolution)
- Trades by category (crypto hourly, brackets, sports)
- Entry price distribution histogram
- Open positions table (market, outcome, entry price, size, time to resolution)

## File Structure

```
scripts/
  run_sniper.py              # Main async loop
  backtest_sniper.py         # DuckDB backtest on parquets

src/arb/
  sniper_engine.py           # SniperEngine: scanner + decision logic

src/feeds/
  polymarket_scanner.py      # MarketScanner: REST poll + targeted ws

src/paper_trading/
  dashboard.py               # +1 sniper tab (edit existing)
```

Reuses: PolymarketExecutor, TradeManager, RiskGuard, PolymarketFeed.
