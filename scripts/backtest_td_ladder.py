#!/usr/bin/env python3
"""Backtest price ladder for crypto TD maker on historical Polymarket data.

Tests whether placing N orders at evenly-spaced price levels within
[lo, hi] (a "ladder") outperforms a single order at the midpoint.

Each rung is an independent GTC maker BUY. When one fills, the others
stay active (scale-in). Total market PnL = sum of individual rung PnLs.

Uses 1c price buckets from 45GB of historical trade data via DuckDB.
A rung "fills" if its 1c bucket has sufficient trade volume.

Dataset: 41K resolved BTC/ETH 15-min updown markets (Oct 2025 - Feb 2026)

Usage:
    python scripts/backtest_td_ladder.py                     # sweep 1-5 rungs
    python scripts/backtest_td_ladder.py --max-rungs 8       # wider sweep
    python scripts/backtest_td_ladder.py --rungs 3           # single config
    python scripts/backtest_td_ladder.py --lo 0.78 --hi 0.85 # narrower range
    python scripts/backtest_td_ladder.py --symbol BTC        # BTC only
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd


# ---------------------------------------------------------------------------
# Data loading (duplicated from backtest_crypto_td_maker.py for standalone use)
# ---------------------------------------------------------------------------

@dataclass
class Market:
    """A resolved 15-min crypto binary market."""
    market_id: str
    slug: str
    symbol: str          # "BTC" or "ETH"
    up_token_id: str
    down_token_id: str
    up_won: bool
    end_date: str
    volume: float


def load_markets(con: duckdb.DuckDBPyConnection, markets_dir: str,
                 symbol_filter: str | None) -> list[Market]:
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

    markets: list[Market] = []
    for _, row in df.iterrows():
        try:
            prices = json.loads(row["outcome_prices"]) if row["outcome_prices"] else None
            if not prices or len(prices) != 2:
                continue
            p0, p1 = float(prices[0]), float(prices[1])

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

            markets.append(Market(
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


def compute_trade_buckets(
    con: duckdb.DuckDBPyConnection,
    trades_dir: str,
    markets: list[Market],
) -> pd.DataFrame:
    """Scan all trades and compute per-token 1c-bucket aggregates."""
    if not markets:
        return pd.DataFrame()

    token_rows = []
    for m in markets:
        token_rows.append((m.up_token_id,))
        token_rows.append((m.down_token_id,))

    con.execute("CREATE OR REPLACE TABLE tdl_tokens (token_id VARCHAR)")
    con.executemany("INSERT INTO tdl_tokens VALUES (?)", token_rows)

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
            INNER JOIN tdl_tokens ct
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
            COUNT(*) AS trade_count
        FROM priced_trades
        WHERE price > 0 AND price < 1
        GROUP BY token_id, price_bucket
    """).df()

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s -- {len(buckets_df):,} bucket rows")
    return buckets_df


# ---------------------------------------------------------------------------
# Bucket lookup
# ---------------------------------------------------------------------------

BucketInfo = tuple[float, float, int]  # (sum_usd, vwap, trade_count)


def build_bucket_lookup(
    buckets_df: pd.DataFrame,
) -> dict[str, dict[float, BucketInfo]]:
    """Build fast lookup: token_id -> {price_bucket -> (sum_usd, vwap, count)}."""
    lookup: dict[str, dict[float, BucketInfo]] = {}
    for _, row in buckets_df.iterrows():
        tid = row["token_id"]
        bucket = round(float(row["price_bucket"]), 2)
        sum_usd = float(row["sum_usd"])
        vwap = float(row["sum_usd_price"]) / sum_usd if sum_usd > 0 else bucket
        count = int(row["trade_count"])
        lookup.setdefault(tid, {})[bucket] = (sum_usd, vwap, count)
    return lookup


# ---------------------------------------------------------------------------
# Ladder logic
# ---------------------------------------------------------------------------

def compute_rung_prices(lo: float, hi: float, n_rungs: int) -> list[float]:
    """Compute evenly-spaced rung prices within [lo, hi], snapped to 1c."""
    if n_rungs == 1:
        return [round((lo + hi) / 2, 2)]
    raw = [lo + i * (hi - lo) / (n_rungs - 1) for i in range(n_rungs)]
    # Snap to 1c and deduplicate while preserving order
    seen: set[float] = set()
    result: list[float] = []
    for p in raw:
        rounded = round(p, 2)
        if rounded not in seen:
            seen.add(rounded)
            result.append(rounded)
    return result


def check_fills_for_token(
    token_id: str,
    rung_prices: list[float],
    lookup: dict[str, dict[float, BucketInfo]],
    min_volume: float,
) -> list[tuple[int, float, float]]:
    """Check which rungs fill for a given token.

    Returns list of (rung_index, rung_price, entry_vwap) for filled rungs.
    """
    buckets = lookup.get(token_id, {})
    fills: list[tuple[int, float, float]] = []
    for i, price in enumerate(rung_prices):
        bucket = round(price, 2)
        info = buckets.get(bucket)
        if info and info[0] >= min_volume:
            fills.append((i, price, info[1]))  # (index, rung_price, vwap)
    return fills


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

@dataclass
class RungFill:
    rung_index: int
    rung_price: float
    entry_price: float  # bucket VWAP
    size_usd: float
    won: bool
    pnl_usd: float


@dataclass
class LadderTrade:
    market_id: str
    slug: str
    symbol: str
    side: str           # "Up" or "Down"
    won: bool
    fills: list[RungFill]
    total_size_usd: float
    total_pnl_usd: float
    capital_after: float


@dataclass
class LadderStats:
    num_rungs: int
    rung_prices: list[float]
    trades: list[LadderTrade] = field(default_factory=list)
    initial_capital: float = 0.0
    final_capital: float = 0.0
    peak_capital: float = 0.0
    max_drawdown_pct: float = 0.0
    total_pnl: float = 0.0
    total_fills: int = 0
    win_count: int = 0    # markets won
    loss_count: int = 0   # markets lost


def simulate_ladder(
    markets: list[Market],
    capital: float,
    size_per_rung: float,
    lo: float,
    hi: float,
    n_rungs: int,
    min_volume: float,
    lookup: dict[str, dict[float, BucketInfo]],
) -> LadderStats:
    """Simulate ladder strategy across all markets chronologically."""
    rung_prices = compute_rung_prices(lo, hi, n_rungs)
    sorted_markets = sorted(markets, key=lambda m: m.end_date)

    stats = LadderStats(
        num_rungs=n_rungs,
        rung_prices=rung_prices,
        initial_capital=capital,
        peak_capital=capital,
    )
    current_capital = capital

    for m in sorted_markets:
        # Budget: need at least 1 rung
        if current_capital < size_per_rung:
            break

        # Check fills for both outcomes
        up_fills = check_fills_for_token(m.up_token_id, rung_prices, lookup, min_volume)
        dn_fills = check_fills_for_token(m.down_token_id, rung_prices, lookup, min_volume)

        if not up_fills and not dn_fills:
            continue

        # Pick outcome with most fills (tie: more total volume)
        if len(up_fills) > len(dn_fills):
            chosen_fills, side, won = up_fills, "Up", m.up_won
        elif len(dn_fills) > len(up_fills):
            chosen_fills, side, won = dn_fills, "Down", not m.up_won
        else:
            # Same number of fills — pick by volume
            up_vol = sum(lookup.get(m.up_token_id, {}).get(round(p, 2), (0, 0, 0))[0]
                         for _, p, _ in up_fills)
            dn_vol = sum(lookup.get(m.down_token_id, {}).get(round(p, 2), (0, 0, 0))[0]
                         for _, p, _ in dn_fills)
            if up_vol >= dn_vol:
                chosen_fills, side, won = up_fills, "Up", m.up_won
            else:
                chosen_fills, side, won = dn_fills, "Down", not m.up_won

        # Cap fills by budget
        max_affordable = int(current_capital // size_per_rung)
        chosen_fills = chosen_fills[:max_affordable]
        if not chosen_fills:
            continue

        # Compute PnL per rung
        rung_results: list[RungFill] = []
        total_pnl = 0.0
        total_size = 0.0

        for rung_idx, rung_price, entry_vwap in chosen_fills:
            if won:
                pnl = size_per_rung * (1.0 / entry_vwap - 1.0)
            else:
                pnl = -size_per_rung
            rung_results.append(RungFill(
                rung_index=rung_idx,
                rung_price=rung_price,
                entry_price=entry_vwap,
                size_usd=size_per_rung,
                won=won,
                pnl_usd=pnl,
            ))
            total_pnl += pnl
            total_size += size_per_rung

        current_capital += total_pnl
        stats.peak_capital = max(stats.peak_capital, current_capital)
        dd = ((stats.peak_capital - current_capital) / stats.peak_capital
              if stats.peak_capital > 0 else 0)
        stats.max_drawdown_pct = max(stats.max_drawdown_pct, dd)

        if won:
            stats.win_count += 1
        else:
            stats.loss_count += 1
        stats.total_fills += len(rung_results)

        stats.trades.append(LadderTrade(
            market_id=m.market_id,
            slug=m.slug,
            symbol=m.symbol,
            side=side,
            won=won,
            fills=rung_results,
            total_size_usd=total_size,
            total_pnl_usd=total_pnl,
            capital_after=current_capital,
        ))

    stats.final_capital = current_capital
    stats.total_pnl = current_capital - capital
    return stats


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def compute_sharpe(trades: list[LadderTrade]) -> float:
    """Annualized Sharpe from per-market returns."""
    if len(trades) < 2:
        return 0.0
    returns = [t.total_pnl_usd / t.total_size_usd if t.total_size_usd > 0 else 0
               for t in trades]
    mean_r = sum(returns) / len(returns)
    var_r = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
    std_r = math.sqrt(var_r) if var_r > 0 else 0
    if std_r == 0:
        return 0.0
    trades_per_year = max(len(trades), 100)
    return (mean_r / std_r) * math.sqrt(trades_per_year)


def print_sweep_table(all_stats: list[LadderStats], capital: float) -> None:
    """Print side-by-side comparison of rung configurations."""
    print(f"\n  {'Rungs':>5} | {'Rung Prices':<24} | {'Markets':>7} | {'Fills':>6} | "
          f"{'Avg Fill':>8} | {'Win%':>5} | {'PnL':>12} | {'Return':>8} | {'MaxDD':>6} | {'Sharpe':>6}")
    print(f"  {'-'*5}-+-{'-'*24}-+-{'-'*7}-+-{'-'*6}-+-"
          f"{'-'*8}-+-{'-'*5}-+-{'-'*12}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}")

    for s in all_stats:
        n = len(s.trades)
        if n == 0:
            prices_str = ",".join(f"{p:.2f}" for p in s.rung_prices)
            print(f"  {s.num_rungs:>5} | {prices_str:<24} |       0 |      0 |"
                  f"       -- |    -- |           -- |       -- |     -- |     --")
            continue
        wr = s.win_count / n * 100
        avg_fills = s.total_fills / n
        ret = s.total_pnl / capital * 100
        dd = s.max_drawdown_pct * 100
        sh = compute_sharpe(s.trades)
        prices_str = ",".join(f"{p:.2f}" for p in s.rung_prices)
        if len(prices_str) > 24:
            prices_str = prices_str[:21] + "..."
        print(f"  {s.num_rungs:>5} | {prices_str:<24} | {n:>7,} | {s.total_fills:>6,} | "
              f"{avg_fills:>8.2f} | {wr:>4.1f}% | ${s.total_pnl:>+11,.2f} | "
              f"{ret:>+7.1f}% | {dd:>5.1f}% | {sh:>6.2f}")


def print_per_rung_analysis(stats: LadderStats) -> None:
    """Show per-rung fill rate, win rate, and PnL."""
    n_markets = len(stats.trades)
    if n_markets == 0:
        return

    # Aggregate per rung index
    rung_data: dict[int, dict] = {}
    for i, price in enumerate(stats.rung_prices):
        rung_data[i] = {"price": price, "fills": 0, "wins": 0, "pnl": 0.0,
                        "entry_sum": 0.0}

    for trade in stats.trades:
        for fill in trade.fills:
            rd = rung_data[fill.rung_index]
            rd["fills"] += 1
            rd["entry_sum"] += fill.entry_price
            rd["pnl"] += fill.pnl_usd
            if fill.won:
                rd["wins"] += 1

    print(f"\n  PER-RUNG ANALYSIS ({stats.num_rungs} rungs, {n_markets:,} markets)")
    print(f"  {'Rung':>4} | {'Price':>5} | {'Fills':>6} | {'Fill%':>5} | "
          f"{'Avg Entry':>9} | {'Win%':>5} | {'PnL':>10}")
    print(f"  {'-'*4}-+-{'-'*5}-+-{'-'*6}-+-{'-'*5}-+-"
          f"{'-'*9}-+-{'-'*5}-+-{'-'*10}")

    for i in sorted(rung_data.keys()):
        rd = rung_data[i]
        fills = rd["fills"]
        if fills == 0:
            print(f"  {i:>4} | {rd['price']:>5.2f} |      0 |  0.0% |"
                  f"        -- |    -- |         --")
            continue
        fill_pct = fills / n_markets * 100
        avg_entry = rd["entry_sum"] / fills
        wr = rd["wins"] / fills * 100
        print(f"  {i:>4} | {rd['price']:>5.2f} | {fills:>6,} | {fill_pct:>4.1f}% | "
              f"{avg_entry:>9.4f} | {wr:>4.1f}% | ${rd['pnl']:>+9,.2f}")


def print_fill_distribution(stats: LadderStats) -> None:
    """Show distribution of fills per market."""
    if not stats.trades:
        return

    fill_counts = Counter(len(t.fills) for t in stats.trades)

    print(f"\n  FILL DISTRIBUTION ({stats.num_rungs} rungs)")
    print(f"  {'Fills':>5} | {'Markets':>7} | {'Pct':>5} | {'Avg PnL/Mkt':>11} | {'Win%':>5}")
    print(f"  {'-'*5}-+-{'-'*7}-+-{'-'*5}-+-{'-'*11}-+-{'-'*5}")

    n_total = len(stats.trades)
    for k in sorted(fill_counts.keys()):
        trades_k = [t for t in stats.trades if len(t.fills) == k]
        n_k = len(trades_k)
        pct = n_k / n_total * 100
        avg_pnl = sum(t.total_pnl_usd for t in trades_k) / n_k
        wins_k = sum(1 for t in trades_k if t.won)
        wr = wins_k / n_k * 100
        print(f"  {k:>5} | {n_k:>7,} | {pct:>4.1f}% | ${avg_pnl:>+10,.4f} | {wr:>4.1f}%")


def print_monthly_breakdown(stats: LadderStats) -> None:
    """Monthly PnL breakdown."""
    if not stats.trades:
        return

    months: dict[str, list[LadderTrade]] = {}
    for t in stats.trades:
        try:
            ts = int(t.slug.split("-")[-1])
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            key = dt.strftime("%Y-%m")
        except (ValueError, IndexError):
            key = "unknown"
        months.setdefault(key, []).append(t)

    if len(months) <= 1:
        return

    print(f"\n  {'Month':<8} | {'Markets':>7} | {'Fills':>6} | {'Win%':>5} | {'PnL':>10}")
    print(f"  {'-'*8}-+-{'-'*7}-+-{'-'*6}-+-{'-'*5}-+-{'-'*10}")
    for month in sorted(months.keys()):
        mt = months[month]
        m_wins = sum(1 for t in mt if t.won)
        m_wr = m_wins / len(mt) * 100
        m_pnl = sum(t.total_pnl_usd for t in mt)
        m_fills = sum(len(t.fills) for t in mt)
        print(f"  {month:<8} | {len(mt):>7,} | {m_fills:>6,} | {m_wr:>4.1f}% | ${m_pnl:>+9,.2f}")


def print_symbol_breakdown(stats: LadderStats) -> None:
    """Breakdown by BTC vs ETH."""
    if not stats.trades:
        return

    print(f"\n  {'Symbol':<6} | {'Markets':>7} | {'Fills':>6} | {'Win%':>5} | {'PnL':>10}")
    print(f"  {'-'*6}-+-{'-'*7}-+-{'-'*6}-+-{'-'*5}-+-{'-'*10}")
    for sym in ["BTC", "ETH"]:
        st = [t for t in stats.trades if t.symbol == sym]
        if not st:
            continue
        s_wins = sum(1 for t in st if t.won)
        s_wr = s_wins / len(st) * 100
        s_pnl = sum(t.total_pnl_usd for t in st)
        s_fills = sum(len(t.fills) for t in st)
        print(f"  {sym:<6} | {len(st):>7,} | {s_fills:>6,} | {s_wr:>4.1f}% | ${s_pnl:>+9,.2f}")


def print_detailed_report(stats: LadderStats, capital: float) -> None:
    """Full report for a single ladder configuration."""
    n = len(stats.trades)
    if n == 0:
        print("\n  No trades executed.")
        return

    wr = stats.win_count / n * 100
    ret = stats.total_pnl / capital * 100
    avg_fills = stats.total_fills / n
    avg_pnl = stats.total_pnl / n
    prices_str = ", ".join(f"{p:.2f}" for p in stats.rung_prices)

    print(f"\n  LADDER: {stats.num_rungs} rungs at [{prices_str}]")
    print(f"  {'=' * 55}")
    print(f"  Markets entered:  {n:,}")
    print(f"  Total fills:      {stats.total_fills:,} ({avg_fills:.2f} avg/market)")
    print(f"  Wins / Losses:    {stats.win_count:,} / {stats.loss_count:,}")
    print(f"  Win rate:         {wr:.1f}%")
    print(f"  Avg PnL/market:   ${avg_pnl:+,.4f}")
    print(f"  Total PnL:        ${stats.total_pnl:+,.2f}")
    print(f"  Return:           {ret:+.1f}%")
    print(f"  Final capital:    ${stats.final_capital:,.2f}")
    print(f"  Max drawdown:     {stats.max_drawdown_pct*100:.1f}%")
    print(f"  Sharpe (ann.):    {compute_sharpe(stats.trades):.2f}")

    print_symbol_breakdown(stats)
    print_per_rung_analysis(stats)
    print_fill_distribution(stats)
    print_monthly_breakdown(stats)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Backtest price ladder for crypto TD maker on Polymarket history")
    parser.add_argument("--lo", type=float, default=0.75,
                        help="Lower bound of ladder range (default: 0.75)")
    parser.add_argument("--hi", type=float, default=0.85,
                        help="Upper bound of ladder range (default: 0.85)")
    parser.add_argument("--max-rungs", type=int, default=5,
                        help="Maximum rungs in sweep (default: 5)")
    parser.add_argument("--rungs", type=int, default=None,
                        help="Run only this rung count (skip sweep)")
    parser.add_argument("--size", type=float, default=5,
                        help="USD per rung (default: 5)")
    parser.add_argument("--capital", type=float, default=1_000,
                        help="Starting capital (default: 1000)")
    parser.add_argument("--symbol", default=None, choices=["BTC", "ETH"],
                        help="Filter to BTC or ETH only")
    parser.add_argument("--min-volume", type=float, default=1.0,
                        help="Min USD volume in 1c bucket for fill (default: 1.0)")
    parser.add_argument("--data-dir",
                        default="../prediction-market-analysis/data/polymarket",
                        help="Path to polymarket data directory")
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

    # Step 2: Compute 1c trade buckets (single scan — expensive)
    buckets_df = compute_trade_buckets(con, trades_dir, markets)

    # Step 3: Build fast lookup
    lookup = build_bucket_lookup(buckets_df)

    print()
    print("=" * 80)
    print("CRYPTO TD MAKER — LADDER BACKTEST")
    print("=" * 80)
    print(f"  Capital: ${args.capital:,.0f}  |  Size: ${args.size:.0f}/rung  "
          f"|  Range: [{args.lo}-{args.hi}]  |  Symbol: {args.symbol or 'ALL'}")

    if args.rungs is not None:
        # Single config mode
        stats = simulate_ladder(
            markets, args.capital, args.size, args.lo, args.hi,
            args.rungs, args.min_volume, lookup,
        )
        print_detailed_report(stats, args.capital)
    else:
        # Sweep mode (default)
        all_stats: list[LadderStats] = []
        for n in range(1, args.max_rungs + 1):
            stats = simulate_ladder(
                markets, args.capital, args.size, args.lo, args.hi,
                n, args.min_volume, lookup,
            )
            all_stats.append(stats)

        print_sweep_table(all_stats, args.capital)

        # Find best config by PnL
        best = max(all_stats, key=lambda s: s.total_pnl)
        print(f"\n  Best: {best.num_rungs} rungs (${best.total_pnl:+,.2f} PnL)")

        # Detailed report for best config
        print_per_rung_analysis(best)
        print_fill_distribution(best)
        print_monthly_breakdown(best)

    print()
    print("=" * 80)
    con.close()


if __name__ == "__main__":
    main()
