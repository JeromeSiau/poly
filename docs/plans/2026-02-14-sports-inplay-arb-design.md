# Sports In-Play Arbitrage — Design Document

**Date:** 2026-02-14
**Status:** Design validated, feed solution TBD
**Author:** Jerome + Claude

## 1. Marcus177 Analysis (Source of Insight)

Trader Marcus177 (wallet `0x71cd52a9bf9121cf8376ba13999468f5d659912d`, alias Total-Treaty) on Polymarket.

### Profile Stats

| Metric | Value |
|--------|-------|
| Total volume | $2,502,504 |
| PnL | +$193,062 |
| Markets traded | 2,363 |
| Period | Aug 2024 — Feb 2026 (~18 months) |
| Current portfolio | $792 |

### On-Chain Activity Analysis (3,500 most recent activities)

- **3,235 trades** across **877 unique markets**
- **265 redeems** (winning positions held to resolution)
- Period covered by data: Sep 2025 — Feb 2026

### Market Categories

**98% sports** (mostly European football/soccer):
- EPL, UCL, UEL, Serie A, La Liga, Bundesliga, Ligue 1, Eredivisie, Copa del Rey
- Some NHL, NBA on the side
- Non-sports: <1% volume (a few geopolitics bets)

### Two-Part Strategy

#### Part A — In-Play Scalping (Primary PnL Driver)

- **268 round-trip markets**, 233 completed in <30 minutes
- **89.2% win rate** on round-trips (239/268 profitable)
- Median hold time: **6 minutes**
- Buy price: median **4c** (71% of buys at 0-10c)
- Sell price: median **30c**

Sample 1-minute flips:
```
buy@0.38 → sell@0.89 | +$573 | Charlton Athletic win
buy@0.07 → sell@0.45 | +$548 | Stade Rennais win
buy@0.03 → sell@0.66 | +$205 | Real Madrid win
buy@0.01 → sell@0.68 | +$177 | San Diego vs San Jose draw
```

He detects in-play events (goals, red cards) before Polymarket prices react, exploiting the latency delta between traditional bookmakers and the Polymarket CLOB.

#### Part B — Longshot Buy-and-Hold

- **199 winning hold-to-resolution positions**
- Total PnL: **+$58,215 on $5,145 cost (1132% ROI)**
- Buy at 1-4c (extreme longshots: spreads, underdogs)
- Payoff 50-100x when they hit, losses are negligible ($2-50 per position)

Top wins:
```
buy@1c   → redeemed $10,852 | Napoli -1.5 spread
buy@1c   → redeemed  $6,734 | Como -0.5 spread
buy@1.5c → redeemed  $6,510 | Tottenham win
buy@1.5c → redeemed  $5,555 | Genoa win
buy@1.1c → redeemed  $4,000 | Latvia win
```

### Trading Patterns

- **Hours (UTC):** Peak at 19:00-21:00 (European evening football), secondary at 00:00-04:00 (US sports)
- **P&L profile:** 469 profitable markets vs 405 losing. Avg win $298, avg loss $24. Extreme asymmetry.
- **Price profile:** 71% of buys at 0-10c. He systematically buys cheap and either flips on events or holds pennies to resolution.

### Key Insight

The fundamental edge is the **latency delta** between traditional bookmaker odds (which react in sub-seconds to in-play events) and Polymarket CLOB prices (which take 1-2 minutes to adjust due to lower liquidity and fewer active market makers on sports).

---

## 2. Strategy: Sports In-Play Arbitrage

### Principle

Exploit the reaction delay of Polymarket prices versus traditional bookmakers during live sports events (goals, red cards, match completions). When a goal is scored, bookmaker odds adjust in sub-seconds but Polymarket takes 30s-2min to react — this is the arbitrage window.

### Two Operating Modes

#### Mode: Scalp

When an in-play event is detected:
1. External feed signals a change (goal, odds shift)
2. Calculate new fair value post-event from bookmaker odds
3. Compare to Polymarket CLOB prices (which haven't moved yet)
4. If edge > threshold → immediate buy on Polymarket
5. Exit in 1-5 min when market catches up, or hold if edge persists

#### Mode: Longshot

On pre-match and in-play markets:
1. Identify heavily underpriced outcomes on Polymarket vs bookmakers (draws at 3c when bookmakers imply 8c)
2. Buy small positions (penny bets, $5-50)
3. Hold to resolution — asymmetric payoff 10-100x

### Target Markets

European football: EPL, UCL, Serie A, La Liga, Bundesliga, Ligue 1. These are the most liquid sports markets on Polymarket with the highest pricing inefficiency.

### Target Metrics (Based on Marcus177)

- 85%+ win rate on scalps
- Position sizing <$500/trade (scalp), <$50/trade (longshot)
- Daily PnL target: $500-2000 on active matchdays

---

## 3. Architecture

### Reusable Existing Infrastructure (~70%)

| Component | Path | Role |
|-----------|------|------|
| `BookmakerMatcher` | `src/matching/bookmaker_matcher.py` | Match Polymarket markets ↔ bookmaker events/outcomes |
| `EventNormalizer` | `src/matching/normalizer.py` | Team name aliases, text similarity |
| `BaseFeed` | `src/feeds/base.py` | Abstract feed interface |
| `PolymarketFeed` | `src/feeds/polymarket.py` | Real-time CLOB prices via WebSocket |
| `ScoreTracker` | `src/feeds/odds_api.py` | Detect score changes between polls |
| `SniperRouter` | `src/arb/sniper_router.py` | Route score changes / spikes to buy actions |
| `EventConditionMapper` | `src/analysis/event_condition_mapper.py` | Group Polymarket markets by event slug |
| `OddsApiClient` | `src/feeds/odds_api.py` | Fetch bookmaker odds + scores |
| `RiskGuard` | `src/risk/` | Circuit breaker, consecutive losses, daily loss |
| `TradeManager` | `src/execution/` | Order lifecycle, Telegram alerts, DB |
| `PolymarketExecutor` | `src/execution/` | Order execution on Polymarket CLOB |

### New Components (3 total)

```
┌──────────────────────┐
│  Fast Feed (TBD)     │  Betfair / OpticOdds / Pinnacle / Score API
│  implements BaseFeed │  → FeedEvent(source="xxx", type="odds_update"|"score_change")
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  SportsInPlayEngine  │  New strategy engine
│                      │  - Compares fair_value (bookmaker) vs poly_price (CLOB)
│                      │  - Calculates edge, filters by threshold
│                      │  - Dispatches scalp vs longshot by profile
└──────────┬───────────┘
           │
           ▼
   [RiskGuard → TradeManager → PolymarketExecutor]  (existing)
```

### Data Flow

```
Fast Feed ──→ ScoreTracker ──→ ScoreChange event
                                     │
OddsApiClient (pre-match odds) ──────┤
                                     ▼
                          SportsInPlayEngine
                           ├── fair_value = bookmaker post-event probability
                           ├── poly_price = PolymarketFeed.get_best_prices()
                           ├── edge = fair_value - poly_price
                           │
                           ├── edge > 5%? → Scalp order
                           ├── outcome < 10c, edge > 50%? → Longshot order
                           └── else → skip
                                     │
                                     ▼
                          TradeManager.execute()
```

---

## 4. Feed Options (Decision Deferred)

The feed is the critical unsolved piece. Options ranked by fitness:

| Option | Latency | Cost | Complexity | Constraint |
|--------|---------|------|------------|------------|
| **OpticOdds** | <1s (WebSocket push) | $99-499/mo | Low — clean API | No geo restriction |
| **Pinnacle API** | ~5s (polling) | Free (account needed) | Medium — polling + auth | Check France availability |
| **The Odds API (boost)** | ~30s (polling) | Already paid | None — already integrated | Probably too slow for scalp |
| **Score feed** (API-Football, SofaScore) | 5-15s | $0-50/mo | Medium | Scores only, no odds |
| **Betfair Streaming** | <1s (WebSocket) | Free | Medium | Not available in France |

### Hybrid Approach (Possible)

A fast score feed (API-Football at ~10s) for event detection, combined with pre-computed fair values from The Odds API (pre-match odds + post-goal adjustment model). Less precise than live odds, but functional without expensive dependencies.

### Key Constraint

Our detection must beat the Polymarket CLOB reaction time. Based on Marcus177's data, the window is 1-2 minutes. Even a 15-30s feed may suffice if execution is fast — this must be validated empirically.

---

## 5. Edge Detection and Sizing

### Edge Calculation (Scalp Mode)

```
fair_value = bookmaker post-event implied probability (devigged)
poly_price = best ask on Polymarket CLOB
edge = fair_value - poly_price
```

Example: Liverpool leads 1-0 at 70'. Bookmakers give Liverpool win at 85% implied. On Polymarket, "Will Liverpool win?" is still at 0.72. Edge = 13c = 18%.

### Entry Thresholds

- **Scalp:** edge > 5% minimum (conservative start, adjustable)
- **Longshot:** edge > 50% on outcomes < 10c (structural mispricing on penny outcomes)

### Position Sizing

- **Scalp:** fractional Kelly, capped at $200-500 per trade. CLOB liquidity is the real limiter.
- **Longshot:** fixed $5-20 per position. Lottery ticket — diversification over concentration.

### Exit Rules (Scalp)

- **Target:** when poly_price catches up to fair_value (edge < 1%)
- **Stop:** if market moves against us (event re-evaluated — e.g. goal annulled by VAR)
- **Timeout:** forced exit after 5 min if edge doesn't close

### Risk Management

- Max 2-3 simultaneous scalp positions
- Separate daily loss limit from other strategies (via strategy tag in RiskGuard)
- Kill switch on 3 consecutive scalp losses (signal that our latency is no longer competitive)

---

## 6. Validation Plan (Before Writing Code)

### Phase 0 — Empirical Measurement (No Code)

1. **Measure CLOB latency:** During 2-3 football evenings (EPL, UCL), manually observe how long Polymarket prices take to react after a goal. If <30s → strategy is dead with most feeds. If 1-2 min → viable.

2. **Measure pre-match mispricing:** Compare Odds API fair values vs Polymarket prices on 50-100 football markets. Quantify average edge. Can be automated with existing `BookmakerMatcher` + `OddsApiClient`.

3. **Backtest on historical data:** Cross-reference on-chain trades on football markets from `../prediction-market-analysis/data/polymarket/` with resolution outcomes. Estimate retrospective edge. Pattern similar to `backtest_crypto_td_maker.py`.

### Phase 1 — Paper Trading

- Connect feed (whichever) + engine in `--paper` mode
- Log every signal with: detection timestamp, fair_value, poly_price, edge, and Polymarket price at +1/+2/+5 min
- After 1-2 weeks: analyze whether signals would have been profitable

### Go/No-Go Criteria

- **Kill:** average edge < 3% OR reaction window < 15s
- **Proceed:** average edge > 5% AND window > 45s AND paper PnL positive

---

## 7. Smart Restart Integration

When implemented, add to `bin/smart_restart.py` RULES:
- `src/arb/sports_inplay.py` → restart sports-inplay daemon
- `src/feeds/fast_feed.py` (whatever the feed) → restart sports-inplay daemon
- Shared modules already covered by existing rules

---

## Appendix: Marcus177 Raw Data Reference

- Wallet: `0x71cd52a9bf9121cf8376ba13999468f5d659912d`
- Profile: https://polymarket.com/@Marcus177
- API endpoint: `https://data-api.polymarket.com/activity?user=0x71cd52a9bf9121cf8376ba13999468f5d659912d&limit=500&offset=0`
- Total on-chain activities retrieved: 3,500 (API limit, actual count higher)
- Analysis date: 2026-02-14
