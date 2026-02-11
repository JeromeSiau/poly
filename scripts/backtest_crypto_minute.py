#!/usr/bin/env python3
"""Backtest crypto-minute (15-min binary) strategies on historical Polymarket data.

Two strategies tested:
- Time Decay: Buy expensive side (>88c) — bet that current state holds to expiry.
- Long Vol: Buy cheap side (<15c) — bet on reversal before expiry.

Uses DuckDB to scan 45GB of trade parquet files. For each resolved BTC/ETH
15-min updown market, computes VWAP in the strategy's entry zone, then
simulates a flat-size portfolio.

Dataset: 41K resolved markets (Oct 2025 – Feb 2026), ~50/50 resolution split.

FINDINGS (41,339 resolved markets, Oct 2025 – Jan 2026):
  TIME DECAY IS PROFITABLE:
  - +50% return on $10K, 37,700 trades, 98.2% win rate, Sharpe 18.9
  - Edge: +1.3% to +1.9% above break-even across thresholds
  - Best at lower thresholds (0.82-0.85): +74% return, +1.9% edge
  - Very consistent: all 4 months profitable, max drawdown only 1.0%
  - ETH slightly better than BTC (98.3% vs 97.9% win rate)
  - PnL scales linearly with position size (no ruin risk)
  - The 0.95-1.00 entry zone is the profit engine (99.0% win, +1.9% edge)

  LONG VOL IS DEEPLY NEGATIVE:
  - -100% across all thresholds and position sizes
  - 5.7% win rate at avg entry 10.5c (break-even ~10.5%)
  - Cheap side almost never reverses in 15 min
  - Not viable even at extreme lows (2-5c: 0% win rate on 94 trades)

  CAVEATS:
  - VWAP uses all trades in price zone, not time-filtered to 2-5min window
  - No spread/slippage modeling (entry at VWAP, not ask price)
  - No gap_pct filter (live strategy uses spot-vs-threshold gap)
  - 4 months of data only — seasonal effects unknown

Usage:
    python scripts/backtest_crypto_minute.py
    python scripts/backtest_crypto_minute.py --symbol BTC
    python scripts/backtest_crypto_minute.py --strategy long_vol --lv-threshold 0.12
    python scripts/backtest_crypto_minute.py --sweep
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import pandas as pd


# ---------------------------------------------------------------------------
# Data loading: resolved crypto-minute markets
# ---------------------------------------------------------------------------

@dataclass
class CryptoMinuteMarket:
    """A resolved 15-min crypto binary market."""
    market_id: str
    slug: str
    symbol: str          # "BTC" or "ETH"
    up_token_id: str
    down_token_id: str
    up_won: bool
    end_date: str
    volume: float
    # Filled after VWAP computation — per-token, per-zone:
    up_vwap_hi: float = 0.0    # VWAP of Up in expensive zone (>td_thresh)
    up_vol_hi: float = 0.0
    down_vwap_hi: float = 0.0  # VWAP of Down in expensive zone
    down_vol_hi: float = 0.0
    up_vwap_lo: float = 0.0    # VWAP of Up in cheap zone (<lv_thresh)
    up_vol_lo: float = 0.0
    down_vwap_lo: float = 0.0  # VWAP of Down in cheap zone
    down_vol_lo: float = 0.0


def load_markets(con: duckdb.DuckDBPyConnection, markets_dir: str,
                 symbol_filter: str | None) -> list[CryptoMinuteMarket]:
    """Load resolved 15-min crypto binary markets."""
    print("Loading resolved crypto-minute markets...")
    t0 = time.time()

    slug_filter = ""
    if symbol_filter:
        prefix = symbol_filter.lower()
        slug_filter = f"AND slug LIKE '{prefix}%'"

    df = con.execute(f"""
        SELECT id, slug, outcome_prices, clob_token_ids, volume, end_date
        FROM '{markets_dir}/*.parquet'
        WHERE slug LIKE '%updown-15m%' AND closed = true
        {slug_filter}
    """).df()

    print(f"  {len(df):,} resolved markets loaded in {time.time()-t0:.1f}s")

    markets: list[CryptoMinuteMarket] = []
    for _, row in df.iterrows():
        try:
            prices = json.loads(row["outcome_prices"]) if row["outcome_prices"] else None
            if not prices or len(prices) != 2:
                continue
            p0, p1 = float(prices[0]), float(prices[1])

            # Must be fully resolved (one outcome = 1, other = 0)
            if p0 > 0.99 and p1 < 0.01:
                up_won = True
            elif p0 < 0.01 and p1 > 0.99:
                up_won = False
            else:
                continue

            token_ids = json.loads(row["clob_token_ids"]) if row["clob_token_ids"] else None
            if not token_ids or len(token_ids) != 2:
                continue

            slug = row["slug"]
            symbol = "BTC" if slug.startswith("btc") else "ETH"

            markets.append(CryptoMinuteMarket(
                market_id=str(row["id"]),
                slug=slug,
                symbol=symbol,
                up_token_id=token_ids[0],
                down_token_id=token_ids[1],
                up_won=up_won,
                end_date=str(row.get("end_date", "")),
                volume=float(row.get("volume", 0) or 0),
            ))
        except (json.JSONDecodeError, ValueError, TypeError, IndexError):
            continue

    print(f"  {len(markets):,} markets parsed "
          f"(Up won: {sum(1 for m in markets if m.up_won):,}, "
          f"Down won: {sum(1 for m in markets if not m.up_won):,})")
    return markets


# ---------------------------------------------------------------------------
# DuckDB trade scan: per-token price buckets (single pass)
# ---------------------------------------------------------------------------

def compute_trade_buckets(
    con: duckdb.DuckDBPyConnection,
    trades_dir: str,
    markets: list[CryptoMinuteMarket],
) -> pd.DataFrame:
    """Scan all trades and compute per-token 5c-bucket aggregates.

    Returns DataFrame with columns:
        token_id, price_bucket, sum_usd, sum_usd_price, trade_count
    """
    if not markets:
        return pd.DataFrame()

    # Register all token IDs
    token_rows = []
    for m in markets:
        token_rows.append((m.up_token_id,))
        token_rows.append((m.down_token_id,))

    con.execute("CREATE OR REPLACE TABLE cm_tokens (token_id VARCHAR)")
    con.executemany("INSERT INTO cm_tokens VALUES (?)", token_rows)

    print(f"Computing trade buckets for {len(token_rows):,} tokens...")
    print("  Scanning trades (this may take a few minutes)...")
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
                END AS usdc_amount
            FROM '{trades_dir}/*.parquet' t
            INNER JOIN cm_tokens ct
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
            FLOOR(price * 20) / 20.0 AS price_bucket,
            SUM(usdc_amount) AS sum_usd,
            SUM(usdc_amount * price) AS sum_usd_price,
            COUNT(*) AS trade_count
        FROM priced_trades
        WHERE price > 0 AND price < 1
        GROUP BY token_id, price_bucket
    """).df()

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s — {len(buckets_df):,} bucket rows")
    return buckets_df


def assign_vwap_from_buckets(
    markets: list[CryptoMinuteMarket],
    buckets_df: pd.DataFrame,
    td_threshold: float,
    lv_threshold: float,
) -> None:
    """Compute VWAP per token in TD and LV zones from pre-computed buckets."""
    if buckets_df.empty:
        return

    # Build lookup: token_id -> list of (bucket, sum_usd, sum_usd_price, count)
    token_buckets: dict[str, list[tuple[float, float, float, int]]] = {}
    for _, row in buckets_df.iterrows():
        tid = row["token_id"]
        token_buckets.setdefault(tid, []).append((
            float(row["price_bucket"]),
            float(row["sum_usd"]),
            float(row["sum_usd_price"]),
            int(row["trade_count"]),
        ))

    for m in markets:
        for token_id, prefix in [(m.up_token_id, "up"), (m.down_token_id, "down")]:
            buckets = token_buckets.get(token_id, [])

            # Expensive zone (>= td_threshold)
            hi_usd = sum(b[1] for b in buckets if b[0] >= td_threshold)
            hi_usd_price = sum(b[2] for b in buckets if b[0] >= td_threshold)
            hi_vwap = hi_usd_price / hi_usd if hi_usd > 0 else 0.0

            # Cheap zone (<= lv_threshold)
            lv_usd = sum(b[1] for b in buckets if b[0] <= lv_threshold)
            lv_usd_price = sum(b[2] for b in buckets if b[0] <= lv_threshold)
            lv_vwap = lv_usd_price / lv_usd if lv_usd > 0 else 0.0

            if prefix == "up":
                m.up_vwap_hi, m.up_vol_hi = hi_vwap, hi_usd
                m.up_vwap_lo, m.up_vol_lo = lv_vwap, lv_usd
            else:
                m.down_vwap_hi, m.down_vol_hi = hi_vwap, hi_usd
                m.down_vwap_lo, m.down_vol_lo = lv_vwap, lv_usd


# ---------------------------------------------------------------------------
# Portfolio simulation
# ---------------------------------------------------------------------------

@dataclass
class TradeResult:
    market_id: str
    slug: str
    symbol: str
    strategy: str     # "time_decay" or "long_vol"
    side: str         # "Up" or "Down"
    entry_price: float
    won: bool
    size_usd: float
    pnl_usd: float
    capital_after: float


@dataclass
class StrategyStats:
    strategy: str
    trades: list[TradeResult] = field(default_factory=list)
    initial_capital: float = 0.0
    final_capital: float = 0.0
    peak_capital: float = 0.0
    max_drawdown_pct: float = 0.0
    total_pnl: float = 0.0
    win_count: int = 0
    loss_count: int = 0


def simulate_strategy(
    markets: list[CryptoMinuteMarket],
    strategy: str,
    capital: float,
    size_usd: float,
    td_threshold: float,
    lv_threshold: float,
    min_volume: float,
) -> StrategyStats:
    """Simulate a single strategy across all markets chronologically."""
    sorted_markets = sorted(markets, key=lambda m: m.end_date)

    stats = StrategyStats(
        strategy=strategy,
        initial_capital=capital,
        peak_capital=capital,
    )
    current_capital = capital

    for m in sorted_markets:
        if current_capital < size_usd:
            break

        # Determine entry for this strategy
        entry_price = 0.0
        side = ""
        won = False

        if strategy == "time_decay":
            # Pick the MORE expensive token (higher volume in hi zone)
            up_ok = m.up_vwap_hi >= td_threshold and m.up_vol_hi >= min_volume
            dn_ok = m.down_vwap_hi >= td_threshold and m.down_vol_hi >= min_volume
            if up_ok and dn_ok:
                # Both sides have trades in expensive zone (market flipped)
                # Pick the one with more volume — that's the dominant side
                if m.up_vol_hi >= m.down_vol_hi:
                    entry_price, side, won = m.up_vwap_hi, "Up", m.up_won
                else:
                    entry_price, side, won = m.down_vwap_hi, "Down", not m.up_won
            elif up_ok:
                entry_price, side, won = m.up_vwap_hi, "Up", m.up_won
            elif dn_ok:
                entry_price, side, won = m.down_vwap_hi, "Down", not m.up_won
        else:  # long_vol
            # Pick the cheaper token (higher volume in lo zone)
            up_ok = m.up_vwap_lo > 0 and m.up_vwap_lo <= lv_threshold and m.up_vol_lo >= min_volume
            dn_ok = m.down_vwap_lo > 0 and m.down_vwap_lo <= lv_threshold and m.down_vol_lo >= min_volume
            if up_ok and dn_ok:
                if m.up_vol_lo >= m.down_vol_lo:
                    entry_price, side, won = m.up_vwap_lo, "Up", m.up_won
                else:
                    entry_price, side, won = m.down_vwap_lo, "Down", not m.up_won
            elif up_ok:
                entry_price, side, won = m.up_vwap_lo, "Up", m.up_won
            elif dn_ok:
                entry_price, side, won = m.down_vwap_lo, "Down", not m.up_won

        if entry_price <= 0 or not side:
            continue

        # PnL: buy tokens at entry_price, they resolve to 1 or 0
        actual_size = min(size_usd, current_capital)
        if won:
            pnl = actual_size * (1.0 / entry_price - 1.0)
            stats.win_count += 1
        else:
            pnl = -actual_size
            stats.loss_count += 1

        current_capital += pnl
        stats.peak_capital = max(stats.peak_capital, current_capital)

        dd = ((stats.peak_capital - current_capital) / stats.peak_capital
              if stats.peak_capital > 0 else 0)
        stats.max_drawdown_pct = max(stats.max_drawdown_pct, dd)

        stats.trades.append(TradeResult(
            market_id=m.market_id,
            slug=m.slug,
            symbol=m.symbol,
            strategy=strategy,
            side=side,
            entry_price=entry_price,
            won=won,
            size_usd=actual_size,
            pnl_usd=pnl,
            capital_after=current_capital,
        ))

    stats.final_capital = current_capital
    stats.total_pnl = current_capital - capital
    return stats


# ---------------------------------------------------------------------------
# Reporting
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
    # ~192 markets/day × 365 ≈ 70K/year, but we only enter a fraction
    trades_per_year = max(len(trades), 100)
    return (mean_r / std_r) * math.sqrt(trades_per_year)


def print_report(stats: StrategyStats, args: argparse.Namespace):
    """Print results for one strategy."""
    n = len(stats.trades)
    if n == 0:
        print(f"\n  {stats.strategy}: No trades executed.")
        return

    win_rate = stats.win_count / n * 100
    ret_pct = stats.total_pnl / stats.initial_capital * 100
    avg_entry = sum(t.entry_price for t in stats.trades) / n
    avg_pnl = stats.total_pnl / n

    print(f"\n  {stats.strategy.upper()}")
    print(f"  {'—' * 50}")
    print(f"  Trades:          {n:,}")
    print(f"  Wins / Losses:   {stats.win_count} / {stats.loss_count}")
    print(f"  Win rate:        {win_rate:.1f}%")
    print(f"  Avg entry price: {avg_entry:.4f}")
    print(f"  Avg PnL/trade:   ${avg_pnl:+,.2f}")
    print(f"  Total PnL:       ${stats.total_pnl:+,.2f}")
    print(f"  Return:          {ret_pct:+.1f}%")
    print(f"  Final capital:   ${stats.final_capital:,.2f}")
    print(f"  Max drawdown:    {stats.max_drawdown_pct*100:.1f}%")
    print(f"  Sharpe (ann.):   {compute_sharpe(stats.trades):.2f}")

    # Breakdown by symbol
    print(f"\n  {'Symbol':<6} | {'Trades':>6} | {'Win%':>5} | {'PnL':>10} | {'Avg Entry':>9}")
    print(f"  {'-'*6}-+-{'-'*6}-+-{'-'*5}-+-{'-'*10}-+-{'-'*9}")
    for sym in ["BTC", "ETH"]:
        st = [t for t in stats.trades if t.symbol == sym]
        if not st:
            continue
        s_wins = sum(1 for t in st if t.won)
        s_wr = s_wins / len(st) * 100
        s_pnl = sum(t.pnl_usd for t in st)
        s_avg = sum(t.entry_price for t in st) / len(st)
        print(f"  {sym:<6} | {len(st):>6} | {s_wr:>4.1f}% | ${s_pnl:>+9,.2f} | {s_avg:>9.4f}")

    # Breakdown by entry price zone
    if stats.strategy == "time_decay":
        zones = [
            ("0.85-0.88", 0.85, 0.88),
            ("0.88-0.90", 0.88, 0.90),
            ("0.90-0.92", 0.90, 0.92),
            ("0.92-0.95", 0.92, 0.95),
            ("0.95-1.00", 0.95, 1.00),
        ]
    else:
        zones = [
            ("0.02-0.05", 0.02, 0.05),
            ("0.05-0.08", 0.05, 0.08),
            ("0.08-0.10", 0.08, 0.10),
            ("0.10-0.12", 0.10, 0.12),
            ("0.12-0.15", 0.12, 0.15),
        ]

    print(f"\n  {'Price Zone':<10} | {'Trades':>6} | {'Win%':>5} | {'Break-even':>10} | {'Edge':>6}")
    print(f"  {'-'*10}-+-{'-'*6}-+-{'-'*5}-+-{'-'*10}-+-{'-'*6}")
    for label, lo, hi in zones:
        zt = [t for t in stats.trades if lo <= t.entry_price < hi]
        if not zt:
            continue
        z_wins = sum(1 for t in zt if t.won)
        z_wr = z_wins / len(zt) * 100
        z_avg_entry = sum(t.entry_price for t in zt) / len(zt)
        be = z_avg_entry * 100  # break-even win rate = entry price
        edge = z_wr - be
        print(f"  {label:<10} | {len(zt):>6} | {z_wr:>4.1f}% | {be:>9.1f}% | {edge:>+5.1f}%")

    # Monthly breakdown
    months: dict[str, list[TradeResult]] = {}
    for t in stats.trades:
        month = t.slug.split("-")  # e.g. btc-updown-15m-1760028300
        # Extract month from end_date if available
        if hasattr(t, "slug"):
            # Use the market's end_date from the trade result
            pass
        # Group by YYYY-MM from slug timestamp
        try:
            ts = int(t.slug.split("-")[-1])
            from datetime import datetime, timezone
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            key = dt.strftime("%Y-%m")
        except (ValueError, IndexError):
            key = "unknown"
        months.setdefault(key, []).append(t)

    if len(months) > 1:
        print(f"\n  {'Month':<8} | {'Trades':>6} | {'Win%':>5} | {'PnL':>10}")
        print(f"  {'-'*8}-+-{'-'*6}-+-{'-'*5}-+-{'-'*10}")
        for month in sorted(months.keys()):
            mt = months[month]
            m_wins = sum(1 for t in mt if t.won)
            m_wr = m_wins / len(mt) * 100
            m_pnl = sum(t.pnl_usd for t in mt)
            print(f"  {month:<8} | {len(mt):>6} | {m_wr:>4.1f}% | ${m_pnl:>+9,.2f}")


def run_sweep(markets: list[CryptoMinuteMarket], buckets_df: pd.DataFrame,
              capital: float, size_usd: float, min_volume: float):
    """Parameter sensitivity sweep."""
    print()
    print("=" * 70)
    print("PARAMETER SENSITIVITY SWEEP")
    print("=" * 70)

    # Time Decay sweep
    td_thresholds = [0.82, 0.85, 0.88, 0.90, 0.92, 0.95]
    print(f"\n  TIME DECAY — varying entry threshold")
    print(f"  {'Threshold':>9} | {'Trades':>6} | {'Win%':>5} | {'PnL':>12} | {'Return':>8} | {'Edge':>6} | {'Sharpe':>6}")
    print(f"  {'-'*9}-+-{'-'*6}-+-{'-'*5}-+-{'-'*12}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}")

    for td in td_thresholds:
        assign_vwap_from_buckets(markets, buckets_df, td, 0.15)
        s = simulate_strategy(markets, "time_decay", capital, size_usd, td, 0.15, min_volume)
        n = len(s.trades)
        if n == 0:
            continue
        wr = s.win_count / n * 100
        avg_e = sum(t.entry_price for t in s.trades) / n
        be = avg_e * 100
        edge = wr - be
        ret = s.total_pnl / capital * 100
        sh = compute_sharpe(s.trades)
        tag = " <--" if td == 0.88 else ""
        print(f"  {td:>9.2f} | {n:>6} | {wr:>4.1f}% | ${s.total_pnl:>+11,.2f} | {ret:>+7.1f}% | {edge:>+5.1f}% | {sh:>6.2f}{tag}")

    # Long Vol sweep
    lv_thresholds = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]
    print(f"\n  LONG VOL — varying entry threshold")
    print(f"  {'Threshold':>9} | {'Trades':>6} | {'Win%':>5} | {'PnL':>12} | {'Return':>8} | {'Edge':>6} | {'Sharpe':>6}")
    print(f"  {'-'*9}-+-{'-'*6}-+-{'-'*5}-+-{'-'*12}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}")

    for lv in lv_thresholds:
        assign_vwap_from_buckets(markets, buckets_df, 0.88, lv)
        s = simulate_strategy(markets, "long_vol", capital, size_usd, 0.88, lv, min_volume)
        n = len(s.trades)
        if n == 0:
            continue
        wr = s.win_count / n * 100
        avg_e = sum(t.entry_price for t in s.trades) / n
        be = avg_e * 100
        edge = wr - be
        ret = s.total_pnl / capital * 100
        sh = compute_sharpe(s.trades)
        tag = " <--" if lv == 0.15 else ""
        print(f"  {lv:>9.2f} | {n:>6} | {wr:>4.1f}% | ${s.total_pnl:>+11,.2f} | {ret:>+7.1f}% | {edge:>+5.1f}% | {sh:>6.2f}{tag}")

    # Position size sweep
    sizes = [5, 10, 20, 50]
    print(f"\n  POSITION SIZE — time_decay @ 0.88 / long_vol @ 0.15")
    print(f"  {'Size':>6} | {'TD Trades':>9} | {'TD PnL':>10} | {'TD Ret%':>7} | {'LV Trades':>9} | {'LV PnL':>10} | {'LV Ret%':>7}")
    print(f"  {'-'*6}-+-{'-'*9}-+-{'-'*10}-+-{'-'*7}-+-{'-'*9}-+-{'-'*10}-+-{'-'*7}")

    assign_vwap_from_buckets(markets, buckets_df, 0.88, 0.15)
    for sz in sizes:
        td_s = simulate_strategy(markets, "time_decay", capital, sz, 0.88, 0.15, min_volume)
        lv_s = simulate_strategy(markets, "long_vol", capital, sz, 0.88, 0.15, min_volume)
        td_ret = td_s.total_pnl / capital * 100
        lv_ret = lv_s.total_pnl / capital * 100
        tag = " <--" if sz == 10 else ""
        print(f"  ${sz:>4} | {len(td_s.trades):>9} | ${td_s.total_pnl:>+9,.2f} | {td_ret:>+6.1f}% | {len(lv_s.trades):>9} | ${lv_s.total_pnl:>+9,.2f} | {lv_ret:>+6.1f}%{tag}")

    print()
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Backtest crypto-minute strategies on Polymarket history")
    parser.add_argument("--data-dir",
                        default="../prediction-market-analysis/data/polymarket",
                        help="Path to polymarket data directory")
    parser.add_argument("--capital", type=float, default=1_000)
    parser.add_argument("--size", type=float, default=10,
                        help="Position size per trade (USD)")
    parser.add_argument("--td-threshold", type=float, default=0.88,
                        help="Time Decay: min entry price for expensive side")
    parser.add_argument("--lv-threshold", type=float, default=0.15,
                        help="Long Vol: max entry price for cheap side")
    parser.add_argument("--min-volume", type=float, default=0.5,
                        help="Min USDC volume in zone for a tradeable market")
    parser.add_argument("--symbol", default=None, choices=["BTC", "ETH"],
                        help="Filter to BTC or ETH only")
    parser.add_argument("--strategy", default=None,
                        choices=["time_decay", "long_vol"],
                        help="Run only one strategy")
    parser.add_argument("--sweep", action="store_true",
                        help="Run parameter sensitivity sweep")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    markets_dir = str(data_dir / "markets")
    trades_dir = str(data_dir / "trades")

    if not (data_dir / "markets").exists():
        print(f"Error: markets directory not found at {markets_dir}")
        sys.exit(1)
    if not (data_dir / "trades").exists():
        print(f"Error: trades directory not found at {trades_dir}")
        sys.exit(1)

    con = duckdb.connect()

    # Step 1: Load markets
    markets = load_markets(con, markets_dir, args.symbol)
    if not markets:
        print("No crypto-minute markets found.")
        sys.exit(0)

    # Step 2: Compute trade buckets (single scan)
    buckets_df = compute_trade_buckets(con, trades_dir, markets)

    # Step 3: Assign VWAP from buckets
    assign_vwap_from_buckets(markets, buckets_df, args.td_threshold, args.lv_threshold)

    # Count tradeable markets
    td_count = sum(1 for m in markets
                   if m.up_vwap_hi >= args.td_threshold or m.down_vwap_hi >= args.td_threshold)
    lv_count = sum(1 for m in markets
                   if (m.up_vwap_lo > 0 and m.up_vwap_lo <= args.lv_threshold) or
                      (m.down_vwap_lo > 0 and m.down_vwap_lo <= args.lv_threshold))
    print(f"  Tradeable: {td_count:,} for time_decay, {lv_count:,} for long_vol")

    # Step 4: Simulate
    print()
    print("=" * 70)
    print("CRYPTO MINUTE BACKTEST RESULTS")
    print("=" * 70)
    print(f"  Capital: ${args.capital:,.0f}  |  Size: ${args.size:.0f}/trade  "
          f"|  Symbol: {args.symbol or 'ALL'}")
    print(f"  TD threshold: {args.td_threshold}  |  LV threshold: {args.lv_threshold}")

    strategies = [args.strategy] if args.strategy else ["time_decay", "long_vol"]

    for strategy in strategies:
        stats = simulate_strategy(
            markets, strategy, args.capital, args.size,
            args.td_threshold, args.lv_threshold, args.min_volume,
        )
        print_report(stats, args)

    print()
    print("=" * 70)

    # Step 5: Sweep
    if args.sweep:
        run_sweep(markets, buckets_df, args.capital, args.size, args.min_volume)

    con.close()


if __name__ == "__main__":
    main()
