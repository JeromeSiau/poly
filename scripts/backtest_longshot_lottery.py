#!/usr/bin/env python3
"""Backtest longshot lottery strategy on historical Polymarket data.

STRATEGY (inspired by planktonXD — $150 → $104K):
    Buy cheap contracts (1-5c) across all market categories.
    Unlike hold-to-resolution (which is -100% EV at these prices),
    EXIT EARLY when price moves (1c → 10-50c).

    The key insight: you don't need the event to happen.
    You just need the PRICE to move. Cheap contracts are call options
    on narrative shifts.

BACKTEST METHODOLOGY:
    For each resolved market, use on-chain 1c price buckets with block ranges
    to detect if a cheap token's price rose AFTER entry (exit opportunity).
    Three simulation modes:
    - hold_to_resolution: baseline (buy cheap, hold to end)
    - early_exit: sell when price rises above exit threshold
    - combined: exit early if possible, otherwise hold to resolution

DATASET: 380K+ resolved markets across all categories

Usage:
    python scripts/backtest_longshot_lottery.py --diagnostic
    python scripts/backtest_longshot_lottery.py --mode combined
    python scripts/backtest_longshot_lottery.py --entry-max 0.05 --exit-min 0.10
    python scripts/backtest_longshot_lottery.py --sweep
    python scripts/backtest_longshot_lottery.py --category crypto_bracket
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from dataclasses import dataclass, field

import duckdb
import pandas as pd


DEFAULT_DATA_DIR = "../prediction-market-analysis/data/polymarket"


# ---------------------------------------------------------------------------
# Market classification (extended for all categories)
# ---------------------------------------------------------------------------

def classify_market(slug: str) -> str:
    """Classify a market by slug pattern — extended beyond crypto."""
    s = slug.lower() if slug else ""

    # Crypto categories
    if "updown-15m" in s:
        return "crypto_15min"
    if "updown" in s or "up-or-down" in s:
        return "crypto_hourly"
    if "price-of" in s:
        return "crypto_bracket"
    crypto_coins = ["bitcoin", "btc", "ethereum", "eth", "solana", "sol", "xrp", "doge", "bnb"]
    if any(c in s for c in crypto_coins) and ("reach" in s or "above" in s or "below" in s):
        return "crypto_weekly"

    # Sports
    sport_hints = [
        "nba", "nfl", "nhl", "mlb", "premier-league", "tennis", "champions-league",
        "la-liga", "bundesliga", "serie-a", "ncaab", "ncaa", "ufc", "boxing",
        "pga", "golf", "f1", "formula", "nascar", "cricket", "rugby",
        "world-cup", "super-bowl", "playoff", "grand-slam",
    ]
    if any(h in s for h in sport_hints):
        return "sports"

    # Politics / elections
    politics_hints = [
        "president", "election", "governor", "senate", "congress", "trump",
        "biden", "democrat", "republican", "approval-rating", "poll", "vote",
        "party", "primary", "cabinet", "impeach", "legislation",
    ]
    if any(h in s for h in politics_hints):
        return "politics"

    # Weather
    weather_hints = [
        "temperature", "weather", "hurricane", "tornado", "snow", "rain",
        "heat", "cold", "climate", "storm", "flood", "wildfire",
    ]
    if any(h in s for h in weather_hints):
        return "weather"

    # Entertainment / pop culture
    entertainment_hints = [
        "oscar", "grammy", "emmy", "box-office", "movie", "album", "song",
        "spotify", "youtube", "tiktok", "subscriber", "follower", "viewer",
        "stream", "award", "celebrity",
    ]
    if any(h in s for h in entertainment_hints):
        return "entertainment"

    # Macro / economics
    macro_hints = [
        "fed", "interest-rate", "inflation", "cpi", "gdp", "unemployment",
        "fomc", "treasury", "rate-cut", "rate-hike", "jobs", "payroll",
    ]
    if any(h in s for h in macro_hints):
        return "macro"

    return "other"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LotteryMarket:
    """A resolved market with winning/losing token IDs."""
    market_id: str
    slug: str
    category: str
    end_date: str
    volume: float
    winning_token: str
    losing_token: str


@dataclass
class TokenAnalysis:
    """Per-token analysis: entry price, exit opportunity, resolution."""
    token_id: str
    market_id: str
    slug: str
    category: str
    is_winner: bool
    entry_vwap: float        # VWAP in entry zone
    entry_volume_usd: float  # USD volume in entry zone
    first_entry_block: int   # earliest trade in entry zone
    # Exit opportunity
    has_exit: bool           # is there a higher price bucket after entry?
    exit_price: float        # lowest qualifying exit bucket price (conservative)
    exit_vwap: float         # VWAP of exit bucket
    max_price_after: float   # highest bucket traded after entry


@dataclass
class TradeResult:
    """A simulated trade."""
    token_id: str
    market_id: str
    slug: str
    category: str
    mode: str
    entry_price: float
    exit_price: float
    is_winner: bool
    has_exit: bool
    size_usd: float
    pnl_usd: float
    capital_after: float


@dataclass
class StrategyStats:
    """Aggregated backtest results."""
    mode: str
    initial_capital: float
    final_capital: float = 0.0
    peak_capital: float = 0.0
    max_drawdown_pct: float = 0.0
    total_pnl: float = 0.0
    win_count: int = 0
    loss_count: int = 0
    trades: list[TradeResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Phase 1: Load resolved markets
# ---------------------------------------------------------------------------

def load_markets(
    con: duckdb.DuckDBPyConnection,
    markets_dir: str,
    category_filter: str | None = None,
) -> list[LotteryMarket]:
    """Load ALL resolved binary markets with known winner/loser."""
    print("Loading resolved markets...")
    t0 = time.time()

    df = con.execute(f"""
        SELECT id, slug, outcome_prices, clob_token_ids, volume, end_date
        FROM '{markets_dir}/*.parquet'
        WHERE closed = true
          AND outcome_prices IS NOT NULL
          AND clob_token_ids IS NOT NULL
    """).df()

    print(f"  {len(df):,} closed markets loaded in {time.time()-t0:.1f}s")

    markets: list[LotteryMarket] = []
    for _, row in df.iterrows():
        try:
            prices = json.loads(row["outcome_prices"])
            tokens = json.loads(row["clob_token_ids"])
            if not prices or not tokens or len(prices) != 2 or len(tokens) != 2:
                continue

            p0, p1 = float(prices[0]), float(prices[1])

            # Must be fully resolved
            if p0 > 0.9 and p1 < 0.1:
                winning_token, losing_token = tokens[0], tokens[1]
            elif p0 < 0.1 and p1 > 0.9:
                winning_token, losing_token = tokens[1], tokens[0]
            else:
                continue

            cat = classify_market(row["slug"])
            if category_filter and cat != category_filter:
                continue

            markets.append(LotteryMarket(
                market_id=row["id"],
                slug=row["slug"],
                category=cat,
                end_date=str(row.get("end_date", "")),
                volume=float(row["volume"]) if row["volume"] else 0,
                winning_token=winning_token,
                losing_token=losing_token,
            ))
        except (json.JSONDecodeError, ValueError, IndexError):
            continue

    print(f"  {len(markets):,} fully resolved binary markets")
    cats = Counter(m.category for m in markets)
    for cat, count in cats.most_common():
        print(f"    {cat}: {count:,}")

    return markets


# ---------------------------------------------------------------------------
# Phase 2: DuckDB trade scan — 1c buckets with block ranges
# ---------------------------------------------------------------------------

def compute_trade_buckets(
    con: duckdb.DuckDBPyConnection,
    trades_dir: str,
    markets: list[LotteryMarket],
) -> pd.DataFrame:
    """Scan all trades and compute per-token 1c-bucket aggregates WITH block ranges.

    Returns DataFrame with columns:
        token_id, price_bucket, sum_usd, sum_usd_price, trade_count,
        min_block, max_block
    """
    if not markets:
        return pd.DataFrame()

    # Register ALL token IDs (both sides)
    token_rows = []
    for m in markets:
        token_rows.append((m.winning_token,))
        token_rows.append((m.losing_token,))

    # Deduplicate
    token_rows = list(set(token_rows))

    con.execute("CREATE OR REPLACE TABLE lottery_tokens (token_id VARCHAR)")
    con.executemany("INSERT INTO lottery_tokens VALUES (?)", token_rows)

    print(f"\nComputing trade buckets for {len(token_rows):,} tokens...")
    print("  Scanning 45GB of trades (this takes a few minutes)...")
    t0 = time.time()

    buckets_df = con.execute(f"""
        WITH priced_trades AS (
            SELECT
                CASE WHEN t.maker_asset_id = '0'
                     THEN t.taker_asset_id
                     ELSE t.maker_asset_id
                END AS token_id,
                CASE WHEN t.maker_asset_id = '0'
                     THEN 1.0 * t.maker_amount / t.taker_amount
                     ELSE 1.0 * t.taker_amount / t.maker_amount
                END AS price,
                CASE WHEN t.maker_asset_id = '0'
                     THEN t.maker_amount / 1e6
                     ELSE t.taker_amount / 1e6
                END AS usdc_amount,
                t.block_number
            FROM '{trades_dir}/*.parquet' t
            INNER JOIN lottery_tokens ct
                ON ct.token_id = (
                    CASE WHEN t.maker_asset_id = '0'
                         THEN t.taker_asset_id
                         ELSE t.maker_asset_id
                    END
                )
            WHERE t.taker_amount > 0 AND t.maker_amount > 0
        )
        SELECT
            token_id,
            FLOOR(price * 100) / 100.0 AS price_bucket,
            SUM(usdc_amount) AS sum_usd,
            SUM(usdc_amount * price) AS sum_usd_price,
            COUNT(*) AS trade_count,
            MIN(block_number) AS min_block,
            MAX(block_number) AS max_block
        FROM priced_trades
        WHERE price > 0 AND price < 1
        GROUP BY token_id, price_bucket
    """).df()

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s — {len(buckets_df):,} bucket rows")
    return buckets_df


# ---------------------------------------------------------------------------
# Phase 3: Exit opportunity analysis
# ---------------------------------------------------------------------------

def build_token_lookup(
    markets: list[LotteryMarket],
    buckets_df: pd.DataFrame,
) -> tuple[
    dict[str, list[tuple[float, float, float, int, int, int]]],
    dict[str, tuple[LotteryMarket, bool]],
]:
    """Pre-build lookups from buckets_df (expensive — do once).

    Returns:
        token_buckets: token_id -> list of (bucket, sum_usd, sum_usd_price, count, min_block, max_block)
        token_market: token_id -> (market, is_winner)
    """
    print("  Building token lookup from buckets...")
    t0 = time.time()

    token_buckets: dict[str, list[tuple[float, float, float, int, int, int]]] = {}
    for _, row in buckets_df.iterrows():
        tid = row["token_id"]
        token_buckets.setdefault(tid, []).append((
            float(row["price_bucket"]),
            float(row["sum_usd"]),
            float(row["sum_usd_price"]),
            int(row["trade_count"]),
            int(row["min_block"]),
            int(row["max_block"]),
        ))

    token_market: dict[str, tuple[LotteryMarket, bool]] = {}
    for m in markets:
        token_market[m.winning_token] = (m, True)
        token_market[m.losing_token] = (m, False)

    print(f"  Lookup built in {time.time()-t0:.1f}s — {len(token_buckets):,} tokens with data")
    return token_buckets, token_market


def analyze_tokens(
    token_buckets: dict[str, list[tuple[float, float, float, int, int, int]]],
    token_market: dict[str, tuple[LotteryMarket, bool]],
    entry_min: float,
    entry_max: float,
    exit_min: float,
    min_volume: float,
) -> list[TokenAnalysis]:
    """For each token with cheap trades, determine entry VWAP and exit opportunities."""
    results: list[TokenAnalysis] = []

    for token_id, (market, is_winner) in token_market.items():
        buckets = token_buckets.get(token_id, [])
        if not buckets:
            continue

        # Entry buckets: price in [entry_min, entry_max]
        entry_buckets = [b for b in buckets if entry_min <= b[0] <= entry_max]
        if not entry_buckets:
            continue

        # Entry VWAP and volume
        entry_usd = sum(b[1] for b in entry_buckets)
        entry_usd_price = sum(b[2] for b in entry_buckets)
        if entry_usd < min_volume:
            continue

        entry_vwap = entry_usd_price / entry_usd
        first_entry_block = min(b[4] for b in entry_buckets)  # min of min_blocks

        # Exit opportunity: any bucket >= exit_min with min_block > first_entry_block
        # This ensures the higher price was reached AFTER the cheap entry existed
        exit_buckets = [
            b for b in buckets
            if b[0] >= exit_min and b[4] > first_entry_block  # min_block > first_entry_block
        ]

        has_exit = len(exit_buckets) > 0
        exit_price = 0.0
        exit_vwap = 0.0
        max_price_after = 0.0

        if has_exit:
            # Sort by bucket price — use lowest qualifying exit bucket (conservative)
            exit_buckets.sort(key=lambda b: b[0])
            lowest_exit = exit_buckets[0]
            exit_price = lowest_exit[0]
            exit_vwap = lowest_exit[2] / lowest_exit[1] if lowest_exit[1] > 0 else lowest_exit[0]
            max_price_after = max(b[0] for b in exit_buckets)

        results.append(TokenAnalysis(
            token_id=token_id,
            market_id=market.market_id,
            slug=market.slug,
            category=market.category,
            is_winner=is_winner,
            entry_vwap=entry_vwap,
            entry_volume_usd=entry_usd,
            first_entry_block=first_entry_block,
            has_exit=has_exit,
            exit_price=exit_price,
            exit_vwap=exit_vwap,
            max_price_after=max_price_after,
        ))

    return results


# ---------------------------------------------------------------------------
# Phase 3b: Diagnostic report (no simulation)
# ---------------------------------------------------------------------------

def print_diagnostic(tokens: list[TokenAnalysis], entry_min: float, entry_max: float,
                     exit_min: float) -> None:
    """Print exit opportunity rates and distributions — no portfolio simulation."""
    n = len(tokens)
    if n == 0:
        print("\n  No tokens found with trades in entry zone.")
        return

    winners = [t for t in tokens if t.is_winner]
    losers = [t for t in tokens if not t.is_winner]
    with_exit = [t for t in tokens if t.has_exit]
    winners_with_exit = [t for t in winners if t.has_exit]
    losers_with_exit = [t for t in losers if t.has_exit]

    print(f"\n{'='*70}")
    print(f"LONGSHOT DIAGNOSTIC — entry [{entry_min:.2f}, {entry_max:.2f}], exit >= {exit_min:.2f}")
    print(f"{'='*70}")

    print(f"\n  Tokens with cheap trades:    {n:,}")
    print(f"  → Resolved YES (winners):    {len(winners):,} ({len(winners)/n*100:.1f}%)")
    print(f"  → Resolved NO  (losers):     {len(losers):,} ({len(losers)/n*100:.1f}%)")
    print(f"\n  Exit opportunity (>= {exit_min:.2f}):   {len(with_exit):,} ({len(with_exit)/n*100:.1f}%)")
    print(f"  → Winners with exit:         {len(winners_with_exit):,} "
          f"({len(winners_with_exit)/len(winners)*100:.1f}% of winners)" if winners else "")
    print(f"  → Losers with exit:          {len(losers_with_exit):,} "
          f"({len(losers_with_exit)/len(losers)*100:.1f}% of losers)" if losers else "")

    # Entry VWAP distribution
    vwaps = [t.entry_vwap for t in tokens]
    print(f"\n  Entry VWAP — min: {min(vwaps):.4f}, median: {sorted(vwaps)[len(vwaps)//2]:.4f}, "
          f"max: {max(vwaps):.4f}")

    # Exit price distribution for those with exit
    if with_exit:
        exit_prices = [t.exit_price for t in with_exit]
        max_prices = [t.max_price_after for t in with_exit]
        print(f"  Exit price — min: {min(exit_prices):.2f}, median: {sorted(exit_prices)[len(exit_prices)//2]:.2f}, "
              f"max: {max(exit_prices):.2f}")
        print(f"  Max price after — min: {min(max_prices):.2f}, median: {sorted(max_prices)[len(max_prices)//2]:.2f}, "
              f"max: {max(max_prices):.2f}")

    # By category
    cats = sorted(set(t.category for t in tokens))
    print(f"\n  {'Category':<16} | {'Tokens':>7} | {'Win%':>5} | {'Exit%':>5} | {'WinExit%':>8} | {'LoseExit%':>9}")
    print(f"  {'-'*16}-+-{'-'*7}-+-{'-'*5}-+-{'-'*5}-+-{'-'*8}-+-{'-'*9}")
    for cat in cats:
        ct = [t for t in tokens if t.category == cat]
        c_winners = [t for t in ct if t.is_winner]
        c_losers = [t for t in ct if not t.is_winner]
        c_exit = [t for t in ct if t.has_exit]
        c_win_rate = len(c_winners) / len(ct) * 100 if ct else 0
        c_exit_rate = len(c_exit) / len(ct) * 100 if ct else 0
        c_w_exit = sum(1 for t in c_winners if t.has_exit) / len(c_winners) * 100 if c_winners else 0
        c_l_exit = sum(1 for t in c_losers if t.has_exit) / len(c_losers) * 100 if c_losers else 0
        print(f"  {cat:<16} | {len(ct):>7,} | {c_win_rate:>4.1f}% | {c_exit_rate:>4.1f}% | "
              f"{c_w_exit:>7.1f}% | {c_l_exit:>8.1f}%")

    # Exit opportunity by exit threshold
    # Use max_price_after to check varying thresholds (doesn't need re-scan)
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.80]
    print(f"\n  Exit opportunity rate by threshold (entry [{entry_min:.2f}, {entry_max:.2f}]):")
    print(f"  {'Threshold':>10} | {'HasExit':>7} | {'Rate':>5} | {'WinRate':>7}")
    print(f"  {'-'*10}-+-{'-'*7}-+-{'-'*5}-+-{'-'*7}")
    for thr in thresholds:
        has = [t for t in tokens if t.max_price_after >= thr]
        if not has:
            print(f"  {thr:>10.2f} | {0:>7} | {0:>4.1f}% |      --")
            continue
        h_wr = sum(1 for t in has if t.is_winner) / len(has) * 100
        print(f"  {thr:>10.2f} | {len(has):>7,} | {len(has)/n*100:>4.1f}% | {h_wr:>6.1f}%")

    # Top 20 highest-exit tokens
    top_exit = sorted(with_exit, key=lambda t: t.max_price_after, reverse=True)[:20]
    if top_exit:
        print(f"\n  Top 20 tokens by max price after entry:")
        print(f"  {'Slug':<45} | {'Entry':>5} | {'MaxP':>5} | {'Exit':>5} | {'Won':>3}")
        print(f"  {'-'*45}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}-+-{'-'*3}")
        for t in top_exit:
            slug_short = t.slug[:45]
            won = "YES" if t.is_winner else "no"
            print(f"  {slug_short:<45} | {t.entry_vwap:>.3f} | {t.max_price_after:>.3f} | "
                  f"{t.exit_price:>.3f} | {won:>3}")


# ---------------------------------------------------------------------------
# Phase 4: Portfolio simulation
# ---------------------------------------------------------------------------

def simulate(
    tokens: list[TokenAnalysis],
    capital: float,
    size_usd: float,
    mode: str,
) -> StrategyStats:
    """Simulate longshot strategy across all qualifying tokens.

    Modes:
        hold_to_resolution: buy cheap, hold → PnL = (1/entry - 1)*size or -size
        early_exit: buy cheap, sell at exit_vwap if available → PnL = (exit/entry - 1)*size, else -size
        combined: exit early if possible, otherwise hold to resolution
    """
    # Sort by first_entry_block (chronological order)
    sorted_tokens = sorted(tokens, key=lambda t: t.first_entry_block)

    stats = StrategyStats(
        mode=mode,
        initial_capital=capital,
        peak_capital=capital,
    )
    current_capital = capital

    for t in sorted_tokens:
        if current_capital < size_usd:
            break

        actual_size = min(size_usd, current_capital)
        entry = t.entry_vwap

        if entry <= 0:
            continue

        if mode == "hold_to_resolution":
            if t.is_winner:
                pnl = actual_size * (1.0 / entry - 1.0)
                won = True
            else:
                pnl = -actual_size
                won = False
            exit_p = 1.0 if t.is_winner else 0.0

        elif mode == "early_exit":
            if t.has_exit:
                # Sell at exit VWAP
                pnl = actual_size * (t.exit_vwap / entry - 1.0)
                won = pnl > 0
                exit_p = t.exit_vwap
            else:
                # No exit opportunity → total loss
                pnl = -actual_size
                won = False
                exit_p = 0.0

        elif mode == "combined":
            if t.has_exit:
                # Exit early
                pnl = actual_size * (t.exit_vwap / entry - 1.0)
                won = pnl > 0
                exit_p = t.exit_vwap
            elif t.is_winner:
                # Hold to resolution — winner
                pnl = actual_size * (1.0 / entry - 1.0)
                won = True
                exit_p = 1.0
            else:
                # Hold to resolution — loser
                pnl = -actual_size
                won = False
                exit_p = 0.0
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if won:
            stats.win_count += 1
        else:
            stats.loss_count += 1

        current_capital += pnl
        stats.peak_capital = max(stats.peak_capital, current_capital)
        dd = ((stats.peak_capital - current_capital) / stats.peak_capital
              if stats.peak_capital > 0 else 0)
        stats.max_drawdown_pct = max(stats.max_drawdown_pct, dd)

        stats.trades.append(TradeResult(
            token_id=t.token_id,
            market_id=t.market_id,
            slug=t.slug,
            category=t.category,
            mode=mode,
            entry_price=entry,
            exit_price=exit_p,
            is_winner=t.is_winner,
            has_exit=t.has_exit,
            size_usd=actual_size,
            pnl_usd=pnl,
            capital_after=current_capital,
        ))

    stats.final_capital = current_capital
    stats.total_pnl = current_capital - capital
    return stats


# ---------------------------------------------------------------------------
# Phase 5: Reporting
# ---------------------------------------------------------------------------

def compute_sharpe(trades: list[TradeResult]) -> float:
    """Annualized Sharpe from per-trade returns."""
    if len(trades) < 2:
        return 0.0
    returns = [t.pnl_usd / t.size_usd if t.size_usd > 0 else 0 for t in trades]
    mean_r = sum(returns) / len(returns)
    var_r = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
    std_r = math.sqrt(var_r) if var_r > 0 else 0
    if std_r == 0:
        return 0.0
    trades_per_year = max(len(trades), 100)
    return (mean_r / std_r) * math.sqrt(trades_per_year)


def print_report(stats: StrategyStats, entry_min: float, entry_max: float,
                 exit_min: float) -> None:
    """Print portfolio simulation results."""
    n = len(stats.trades)
    if n == 0:
        print(f"\n  {stats.mode.upper()}: No trades executed.")
        return

    win_rate = stats.win_count / n * 100
    ret_pct = stats.total_pnl / stats.initial_capital * 100
    avg_entry = sum(t.entry_price for t in stats.trades) / n
    avg_pnl = stats.total_pnl / n

    early_exits = sum(1 for t in stats.trades if t.has_exit)
    held = n - early_exits if stats.mode != "hold_to_resolution" else n

    print(f"\n  {stats.mode.upper()} — entry [{entry_min:.2f}, {entry_max:.2f}], exit >= {exit_min:.2f}")
    print(f"  {'—' * 55}")
    print(f"  Trades:          {n:,}")
    print(f"  Wins / Losses:   {stats.win_count:,} / {stats.loss_count:,}")
    print(f"  Win rate:        {win_rate:.1f}%")
    if stats.mode != "hold_to_resolution":
        print(f"  Early exits:     {early_exits:,} ({early_exits/n*100:.1f}%)")
        print(f"  Held to resolve: {held:,}")
    print(f"  Avg entry price: {avg_entry:.4f}")
    print(f"  Avg PnL/trade:   ${avg_pnl:+,.4f}")
    print(f"  Total PnL:       ${stats.total_pnl:+,.2f}")
    print(f"  Return:          {ret_pct:+.1f}%")
    print(f"  Final capital:   ${stats.final_capital:,.2f}")
    print(f"  Max drawdown:    {stats.max_drawdown_pct*100:.1f}%")
    print(f"  Sharpe (ann.):   {compute_sharpe(stats.trades):.2f}")

    # By category
    cats = sorted(set(t.category for t in stats.trades))
    print(f"\n  {'Category':<16} | {'Trades':>7} | {'Win%':>5} | {'PnL':>12} | {'Avg Entry':>9} | {'Exit%':>5}")
    print(f"  {'-'*16}-+-{'-'*7}-+-{'-'*5}-+-{'-'*12}-+-{'-'*9}-+-{'-'*5}")
    for cat in cats:
        ct = [t for t in stats.trades if t.category == cat]
        if not ct:
            continue
        c_wins = sum(1 for t in ct if t.pnl_usd > 0)
        c_wr = c_wins / len(ct) * 100
        c_pnl = sum(t.pnl_usd for t in ct)
        c_avg = sum(t.entry_price for t in ct) / len(ct)
        c_exits = sum(1 for t in ct if t.has_exit)
        c_exit_pct = c_exits / len(ct) * 100
        print(f"  {cat:<16} | {len(ct):>7,} | {c_wr:>4.1f}% | ${c_pnl:>+11,.2f} | {c_avg:>9.4f} | {c_exit_pct:>4.1f}%")

    # By 1c entry price zone
    zones: list[tuple[str, float, float]] = []
    lo = entry_min
    while lo < entry_max - 0.001:
        hi = round(lo + 0.01, 2)
        label = f"{lo:.2f}-{hi:.2f}"
        zones.append((label, lo, hi))
        lo = hi

    print(f"\n  {'Entry Zone':<10} | {'Trades':>7} | {'Win%':>5} | {'PnL':>12} | {'Exit%':>5}")
    print(f"  {'-'*10}-+-{'-'*7}-+-{'-'*5}-+-{'-'*12}-+-{'-'*5}")
    for label, lo, hi in zones:
        zt = [t for t in stats.trades if lo <= t.entry_price < hi]
        if not zt:
            continue
        z_wins = sum(1 for t in zt if t.pnl_usd > 0)
        z_wr = z_wins / len(zt) * 100
        z_pnl = sum(t.pnl_usd for t in zt)
        z_exits = sum(1 for t in zt if t.has_exit)
        z_exit_pct = z_exits / len(zt) * 100
        print(f"  {label:<10} | {len(zt):>7,} | {z_wr:>4.1f}% | ${z_pnl:>+11,.2f} | {z_exit_pct:>4.1f}%")

    # Top 10 wins and top 10 losses
    sorted_by_pnl = sorted(stats.trades, key=lambda t: t.pnl_usd, reverse=True)
    top_wins = sorted_by_pnl[:10]
    top_losses = sorted_by_pnl[-10:]

    if top_wins and top_wins[0].pnl_usd > 0:
        print(f"\n  Top 10 wins:")
        print(f"  {'Slug':<40} | {'Entry':>5} | {'Exit':>5} | {'PnL':>10} | {'Won':>3}")
        print(f"  {'-'*40}-+-{'-'*5}-+-{'-'*5}-+-{'-'*10}-+-{'-'*3}")
        for t in top_wins:
            if t.pnl_usd <= 0:
                break
            slug_short = t.slug[:40]
            won = "YES" if t.is_winner else "no"
            print(f"  {slug_short:<40} | {t.entry_price:>.3f} | {t.exit_price:>.3f} | "
                  f"${t.pnl_usd:>+9,.2f} | {won:>3}")

    if top_losses and top_losses[-1].pnl_usd < 0:
        print(f"\n  Top 10 losses:")
        for t in reversed(top_losses):
            if t.pnl_usd >= 0:
                continue
            slug_short = t.slug[:40]
            won = "YES" if t.is_winner else "no"
            print(f"  {slug_short:<40} | {t.entry_price:>.3f} | {t.exit_price:>.3f} | "
                  f"${t.pnl_usd:>+9,.2f} | {won:>3}")


# ---------------------------------------------------------------------------
# Phase 5b: Parameter sweep
# ---------------------------------------------------------------------------

def run_sweep(
    markets: list[LotteryMarket],
    token_buckets: dict[str, list[tuple[float, float, float, int, int, int]]],
    token_market: dict[str, tuple[LotteryMarket, bool]],
    capital: float,
    size_usd: float,
    min_volume: float,
) -> None:
    """Sweep entry_max × exit_min grid, per mode."""
    print()
    print("=" * 90)
    print("PARAMETER SWEEP")
    print("=" * 90)

    entry_maxes = [0.02, 0.03, 0.05, 0.08, 0.10, 0.15]
    exit_mins = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]

    for mode in ["hold_to_resolution", "early_exit", "combined"]:
        print(f"\n  MODE: {mode.upper()}")
        print(f"  {'':>10}", end="")
        for em in exit_mins:
            print(f" | exit≥{em:.2f}", end="")
        print()
        print(f"  {'-'*10}", end="")
        for _ in exit_mins:
            print(f"-+-{'-'*9}", end="")
        print()

        for entry_max in entry_maxes:
            print(f"  ent≤{entry_max:.2f}  ", end="")
            for exit_min in exit_mins:
                if exit_min <= entry_max:
                    print(f" | {'--':>9}", end="")
                    continue

                tokens = analyze_tokens(token_buckets, token_market, 0.01, entry_max,
                                        exit_min, min_volume)
                if not tokens:
                    print(f" | {'--':>9}", end="")
                    continue

                s = simulate(tokens, capital, size_usd, mode)
                n = len(s.trades)
                if n == 0:
                    print(f" | {'--':>9}", end="")
                else:
                    ret = s.total_pnl / capital * 100
                    print(f" | {ret:>+8.1f}%", end="")
            print()

    # Detailed sweep for best mode (combined)
    print(f"\n  COMBINED MODE — Detailed (trades / win% / PnL)")
    header = f"  {'':>10}"
    for em in exit_mins:
        header += f" | {'exit≥'+f'{em:.2f}':>16}"
    print(header)

    for entry_max in entry_maxes:
        print(f"  ent≤{entry_max:.2f}  ", end="")
        for exit_min in exit_mins:
            if exit_min <= entry_max:
                print(f" | {'--':>16}", end="")
                continue

            tokens = analyze_tokens(token_buckets, token_market, 0.01, entry_max,
                                    exit_min, min_volume)
            s = simulate(tokens, capital, size_usd, "combined")
            n = len(s.trades)
            if n == 0:
                print(f" | {'--':>16}", end="")
            else:
                wr = s.win_count / n * 100
                print(f" | {n:>5} {wr:>4.0f}% ${s.total_pnl:>+7,.0f}", end="")
        print()

    # Per-category best configs
    print(f"\n  BEST CONFIG PER CATEGORY (combined mode, entry≤0.05, exit≥0.10):")
    cats = sorted(set(m.category for m in markets))
    print(f"  {'Category':<16} | {'Tokens':>7} | {'Trades':>6} | {'Win%':>5} | {'PnL':>12} | {'Return':>8} | {'Exit%':>5}")
    print(f"  {'-'*16}-+-{'-'*7}-+-{'-'*6}-+-{'-'*5}-+-{'-'*12}-+-{'-'*8}-+-{'-'*5}")

    for cat in cats:
        cat_token_market = {tid: (m, w) for tid, (m, w) in token_market.items()
                            if m.category == cat}
        tokens = analyze_tokens(token_buckets, cat_token_market, 0.01, 0.05, 0.10, min_volume)
        if not tokens:
            continue
        s = simulate(tokens, capital, size_usd, "combined")
        n = len(s.trades)
        if n == 0:
            continue
        wr = s.win_count / n * 100
        ret = s.total_pnl / capital * 100
        exits = sum(1 for t in s.trades if t.has_exit)
        exit_pct = exits / n * 100
        print(f"  {cat:<16} | {len(tokens):>7,} | {n:>6,} | {wr:>4.1f}% | "
              f"${s.total_pnl:>+11,.2f} | {ret:>+7.1f}% | {exit_pct:>4.1f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Backtest longshot lottery strategy on Polymarket history")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                        help="Path to polymarket data directory")
    parser.add_argument("--entry-min", type=float, default=0.01,
                        help="Min entry price (default: 0.01)")
    parser.add_argument("--entry-max", type=float, default=0.05,
                        help="Max entry price (default: 0.05)")
    parser.add_argument("--exit-min", type=float, default=0.10,
                        help="Min exit price threshold (default: 0.10)")
    parser.add_argument("--capital", type=float, default=1_000,
                        help="Starting capital (USD)")
    parser.add_argument("--size", type=float, default=5,
                        help="Position size per trade (USD)")
    parser.add_argument("--min-volume", type=float, default=5.0,
                        help="Min USD volume in entry zone for a token to qualify")
    parser.add_argument("--category", type=str, default=None,
                        help="Filter to category: crypto_bracket, crypto_hourly, sports, politics, etc.")
    parser.add_argument("--mode", default="combined",
                        choices=["hold_to_resolution", "early_exit", "combined"],
                        help="Simulation mode (default: combined)")
    parser.add_argument("--sweep", action="store_true",
                        help="Run parameter sweep over entry/exit grid")
    parser.add_argument("--diagnostic", action="store_true",
                        help="Print exit opportunity analysis without simulation")
    args = parser.parse_args()

    data_dir = args.data_dir
    markets_dir = f"{data_dir}/markets"
    trades_dir = f"{data_dir}/trades"

    con = duckdb.connect()

    # Phase 1: Load markets
    markets = load_markets(con, markets_dir, args.category)
    if not markets:
        print("No resolved markets found.")
        sys.exit(0)

    # Phase 2: Compute trade buckets (single expensive scan)
    buckets_df = compute_trade_buckets(con, trades_dir, markets)
    if buckets_df.empty:
        print("No trade data found.")
        sys.exit(0)

    # Build lookup once (expensive with 14M rows)
    token_buckets, token_market = build_token_lookup(markets, buckets_df)

    # Phase 3: Analyze tokens
    tokens = analyze_tokens(token_buckets, token_market, args.entry_min, args.entry_max,
                            args.exit_min, args.min_volume)
    print(f"\n  {len(tokens):,} tokens with trades in [{args.entry_min:.2f}, {args.entry_max:.2f}]")
    print(f"  → {sum(1 for t in tokens if t.has_exit):,} have exit opportunity >= {args.exit_min:.2f}")
    print(f"  → {sum(1 for t in tokens if t.is_winner):,} resolved YES (winners)")

    if args.diagnostic:
        print_diagnostic(tokens, args.entry_min, args.entry_max, args.exit_min)
        con.close()
        return

    if args.sweep:
        # Run all 3 modes in the sweep
        run_sweep(markets, token_buckets, token_market, args.capital, args.size, args.min_volume)
        con.close()
        return

    # Phase 4: Simulate
    print()
    print("=" * 70)
    print("LONGSHOT LOTTERY BACKTEST")
    print("=" * 70)
    print(f"  Capital: ${args.capital:,.0f}  |  Size: ${args.size:.0f}/trade  "
          f"|  Category: {args.category or 'ALL'}")

    if args.mode == "combined":
        # Show all 3 for comparison
        for mode in ["hold_to_resolution", "early_exit", "combined"]:
            stats = simulate(tokens, args.capital, args.size, mode)
            print_report(stats, args.entry_min, args.entry_max, args.exit_min)
    else:
        stats = simulate(tokens, args.capital, args.size, args.mode)
        print_report(stats, args.entry_min, args.entry_max, args.exit_min)

    print()
    print("=" * 70)

    con.close()


if __name__ == "__main__":
    main()
