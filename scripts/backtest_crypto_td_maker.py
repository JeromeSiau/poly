#!/usr/bin/env python3
"""Backtest crypto TD maker strategy on historical Polymarket data.

STRATEGY (from run_crypto_td_maker.py):
    On Polymarket 15-min crypto binary markets (BTC/ETH up/down), place
    passive maker bids when the best bid enters [target_bid, max_bid]
    (default [0.75, 0.85]). Hold filled position to market resolution.
    Maker orders pay 0 fees -- the edge is the gap between actual win
    rate and the implied probability at entry price.

BACKTEST METHODOLOGY:
    Without orderbook history, we approximate maker fills using trade-level
    VWAP in the bid range as the entry price proxy. For each resolved market,
    if either outcome (Up or Down) has sufficient trade volume in [target_bid,
    max_bid], we simulate a fill at the VWAP. If both qualify, we pick the
    outcome with more volume (higher fill probability).

    This is CONSERVATIVE for a maker because:
    - Maker fills at their exact limit price (best case = target_bid)
    - VWAP includes trades above the limit price
    - Real edge may be higher than reported

DATASET: 41K resolved BTC/ETH 15-min updown markets (Oct 2025 - Feb 2026)

FINDINGS:
    (to be filled after first run)

Usage:
    python scripts/backtest_crypto_td_maker.py
    python scripts/backtest_crypto_td_maker.py --symbol BTC
    python scripts/backtest_crypto_td_maker.py --target-bid 0.78 --max-bid 0.88
    python scripts/backtest_crypto_td_maker.py --sweep
    python scripts/backtest_crypto_td_maker.py --no-chart
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd


# ---------------------------------------------------------------------------
# Data loading: resolved crypto-minute markets
# ---------------------------------------------------------------------------

@dataclass
class TDMakerMarket:
    """A resolved 15-min crypto binary market for TD maker backtest."""
    market_id: str
    slug: str
    symbol: str          # "BTC" or "ETH"
    up_token_id: str
    down_token_id: str
    up_won: bool
    end_date: str
    volume: float
    # Filled after VWAP computation -- per-outcome in bid range:
    up_vwap: float = 0.0       # VWAP of Up trades in [target_bid, max_bid]
    up_vol_usd: float = 0.0    # USD volume of Up trades in range
    down_vwap: float = 0.0
    down_vol_usd: float = 0.0


def load_markets(con: duckdb.DuckDBPyConnection, markets_dir: str,
                 symbol_filter: str | None) -> list[TDMakerMarket]:
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

    markets: list[TDMakerMarket] = []
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

            markets.append(TDMakerMarket(
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
# DuckDB trade scan: per-token 1c price buckets (single pass)
# ---------------------------------------------------------------------------

def compute_trade_buckets(
    con: duckdb.DuckDBPyConnection,
    trades_dir: str,
    markets: list[TDMakerMarket],
) -> pd.DataFrame:
    """Scan all trades and compute per-token 1c-bucket aggregates.

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

    con.execute("CREATE OR REPLACE TABLE tdm_tokens (token_id VARCHAR)")
    con.executemany("INSERT INTO tdm_tokens VALUES (?)", token_rows)

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
            INNER JOIN tdm_tokens ct
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


def assign_vwap_from_buckets(
    markets: list[TDMakerMarket],
    buckets_df: pd.DataFrame,
    target_bid: float,
    max_bid: float,
) -> None:
    """Compute VWAP per token in [target_bid, max_bid] from pre-computed buckets."""
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

            # Filter to bid range [target_bid, max_bid]
            relevant = [b for b in buckets if target_bid <= b[0] <= max_bid]
            total_usd = sum(b[1] for b in relevant)
            total_usd_price = sum(b[2] for b in relevant)
            vwap = total_usd_price / total_usd if total_usd > 0 else 0.0

            if prefix == "up":
                m.up_vwap, m.up_vol_usd = vwap, total_usd
            else:
                m.down_vwap, m.down_vol_usd = vwap, total_usd


# ---------------------------------------------------------------------------
# Portfolio simulation
# ---------------------------------------------------------------------------

@dataclass
class TradeResult:
    market_id: str
    slug: str
    symbol: str
    side: str         # "Up" or "Down"
    entry_price: float
    won: bool
    size_usd: float
    pnl_usd: float
    capital_after: float


@dataclass
class StrategyStats:
    trades: list[TradeResult] = field(default_factory=list)
    initial_capital: float = 0.0
    final_capital: float = 0.0
    peak_capital: float = 0.0
    max_drawdown_pct: float = 0.0
    total_pnl: float = 0.0
    win_count: int = 0
    loss_count: int = 0


def simulate(
    markets: list[TDMakerMarket],
    capital: float,
    size_usd: float,
    target_bid: float,
    max_bid: float,
    min_volume: float,
) -> StrategyStats:
    """Simulate TD maker strategy across all markets chronologically."""
    sorted_markets = sorted(markets, key=lambda m: m.end_date)

    stats = StrategyStats(
        initial_capital=capital,
        peak_capital=capital,
    )
    current_capital = capital

    for m in sorted_markets:
        if current_capital < size_usd:
            break

        # Check each outcome for tradeability
        up_ok = (m.up_vwap >= target_bid
                 and m.up_vwap <= max_bid
                 and m.up_vol_usd >= min_volume)
        dn_ok = (m.down_vwap >= target_bid
                 and m.down_vwap <= max_bid
                 and m.down_vol_usd >= min_volume)

        # Entry rule: pick one outcome per market
        entry_price = 0.0
        side = ""
        won = False

        if up_ok and dn_ok:
            # Both outcomes tradeable -- pick the one with more volume
            if m.up_vol_usd >= m.down_vol_usd:
                entry_price, side, won = m.up_vwap, "Up", m.up_won
            else:
                entry_price, side, won = m.down_vwap, "Down", not m.up_won
        elif up_ok:
            entry_price, side, won = m.up_vwap, "Up", m.up_won
        elif dn_ok:
            entry_price, side, won = m.down_vwap, "Down", not m.up_won
        else:
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
    trades_per_year = max(len(trades), 100)
    return (mean_r / std_r) * math.sqrt(trades_per_year)


def print_report(stats: StrategyStats, args: argparse.Namespace) -> None:
    """Print results."""
    n = len(stats.trades)
    if n == 0:
        print("\n  No trades executed.")
        return

    win_rate = stats.win_count / n * 100
    ret_pct = stats.total_pnl / stats.initial_capital * 100
    avg_entry = sum(t.entry_price for t in stats.trades) / n
    avg_pnl = stats.total_pnl / n
    be_wr = avg_entry * 100  # break-even win rate = entry price
    edge = win_rate - be_wr

    print(f"\n  TD MAKER [{args.target_bid}-{args.max_bid}]")
    print(f"  {'—' * 50}")
    print(f"  Trades:          {n:,}")
    print(f"  Wins / Losses:   {stats.win_count:,} / {stats.loss_count:,}")
    print(f"  Win rate:        {win_rate:.1f}%")
    print(f"  Avg entry price: {avg_entry:.4f}")
    print(f"  Break-even WR:   {be_wr:.1f}%")
    print(f"  Edge:            {edge:+.1f}%")
    print(f"  Avg PnL/trade:   ${avg_pnl:+,.4f}")
    print(f"  Total PnL:       ${stats.total_pnl:+,.2f}")
    print(f"  Return:          {ret_pct:+.1f}%")
    print(f"  Final capital:   ${stats.final_capital:,.2f}")
    print(f"  Max drawdown:    {stats.max_drawdown_pct*100:.1f}%")
    print(f"  Sharpe (ann.):   {compute_sharpe(stats.trades):.2f}")

    # Breakdown by symbol
    print(f"\n  {'Symbol':<6} | {'Trades':>6} | {'Win%':>5} | {'PnL':>10} | {'Avg Entry':>9} | {'Edge':>6}")
    print(f"  {'-'*6}-+-{'-'*6}-+-{'-'*5}-+-{'-'*10}-+-{'-'*9}-+-{'-'*6}")
    for sym in ["BTC", "ETH"]:
        st = [t for t in stats.trades if t.symbol == sym]
        if not st:
            continue
        s_wins = sum(1 for t in st if t.won)
        s_wr = s_wins / len(st) * 100
        s_pnl = sum(t.pnl_usd for t in st)
        s_avg = sum(t.entry_price for t in st) / len(st)
        s_edge = s_wr - s_avg * 100
        print(f"  {sym:<6} | {len(st):>6} | {s_wr:>4.1f}% | ${s_pnl:>+9,.2f} | {s_avg:>9.4f} | {s_edge:>+5.1f}%")

    # Breakdown by 1c entry price zone
    zones = []
    lo = args.target_bid
    while lo < args.max_bid - 0.001:
        hi = round(lo + 0.01, 2)
        label = f"{lo:.2f}-{hi:.2f}"
        zones.append((label, lo, hi))
        lo = hi

    print(f"\n  {'Price Zone':<10} | {'Trades':>6} | {'Win%':>5} | {'Break-even':>10} | {'Edge':>6}")
    print(f"  {'-'*10}-+-{'-'*6}-+-{'-'*5}-+-{'-'*10}-+-{'-'*6}")
    for label, lo, hi in zones:
        zt = [t for t in stats.trades if lo <= t.entry_price < hi]
        if not zt:
            continue
        z_wins = sum(1 for t in zt if t.won)
        z_wr = z_wins / len(zt) * 100
        z_avg_entry = sum(t.entry_price for t in zt) / len(zt)
        z_be = z_avg_entry * 100
        z_edge = z_wr - z_be
        print(f"  {label:<10} | {len(zt):>6} | {z_wr:>4.1f}% | {z_be:>9.1f}% | {z_edge:>+5.1f}%")

    # Monthly breakdown
    months: dict[str, list[TradeResult]] = {}
    for t in stats.trades:
        try:
            ts = int(t.slug.split("-")[-1])
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


def plot_capital_curve(stats: StrategyStats, args: argparse.Namespace) -> None:
    """Optional matplotlib capital curve chart."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available -- skipping chart)")
        return

    capital_series = [stats.initial_capital] + [t.capital_after for t in stats.trades]
    plt.figure(figsize=(12, 5))
    plt.plot(capital_series, linewidth=0.8)
    plt.title(f"TD Maker Capital Curve [{args.target_bid}-{args.max_bid}]")
    plt.xlabel("Trade #")
    plt.ylabel("Capital ($)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = "td_maker_capital_curve.png"
    plt.savefig(out_path, dpi=150)
    print(f"  Capital curve saved to {out_path}")


# ---------------------------------------------------------------------------
# Parameter sweep
# ---------------------------------------------------------------------------

def run_sweep(
    markets: list[TDMakerMarket],
    buckets_df: pd.DataFrame,
    capital: float,
    size_usd: float,
    min_volume: float,
) -> None:
    """Parameter sensitivity sweep."""
    print()
    print("=" * 70)
    print("PARAMETER SENSITIVITY SWEEP")
    print("=" * 70)

    header = f"  {'Param':>9} | {'Trades':>6} | {'Win%':>5} | {'PnL':>12} | {'Return':>8} | {'Edge':>6} | {'Sharpe':>6}"
    divider = f"  {'-'*9}-+-{'-'*6}-+-{'-'*5}-+-{'-'*12}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}"

    def _row(label: str, s: StrategyStats, mark: bool = False) -> str:
        n = len(s.trades)
        if n == 0:
            return f"  {label:>9} |      0 |    -- |           -- |       -- |     -- |     --"
        wr = s.win_count / n * 100
        avg_e = sum(t.entry_price for t in s.trades) / n
        edge = wr - avg_e * 100
        ret = s.total_pnl / capital * 100
        sh = compute_sharpe(s.trades)
        tag = " <--" if mark else ""
        return f"  {label:>9} | {n:>6} | {wr:>4.1f}% | ${s.total_pnl:>+11,.2f} | {ret:>+7.1f}% | {edge:>+5.1f}% | {sh:>6.2f}{tag}"

    # 1. Vary target_bid (lower bound), max_bid fixed at 0.85
    target_bids = [0.70, 0.73, 0.75, 0.78, 0.80, 0.82]
    print(f"\n  VARY TARGET BID (max_bid=0.85)")
    print(header)
    print(divider)
    for tb in target_bids:
        assign_vwap_from_buckets(markets, buckets_df, tb, 0.85)
        s = simulate(markets, capital, size_usd, tb, 0.85, min_volume)
        print(_row(f"{tb:.2f}", s, mark=(tb == 0.75)))

    # 2. Vary max_bid (upper bound), target_bid fixed at 0.75
    max_bids = [0.80, 0.82, 0.85, 0.88, 0.90]
    print(f"\n  VARY MAX BID (target_bid=0.75)")
    print(header)
    print(divider)
    for mb in max_bids:
        assign_vwap_from_buckets(markets, buckets_df, 0.75, mb)
        s = simulate(markets, capital, size_usd, 0.75, mb, min_volume)
        print(_row(f"{mb:.2f}", s, mark=(mb == 0.85)))

    # 3. 2D grid: target_bid x max_bid (compact PnL table)
    t_range = [0.70, 0.75, 0.78, 0.80]
    m_range = [0.82, 0.85, 0.88, 0.90]
    print(f"\n  2D GRID — PnL (rows=target_bid, cols=max_bid)")
    col_header = "           " + "".join(f"  mb={mb:.2f}" for mb in m_range)
    print(col_header)
    for tb in t_range:
        cells = []
        for mb in m_range:
            if tb >= mb:
                cells.append("       --")
                continue
            assign_vwap_from_buckets(markets, buckets_df, tb, mb)
            s = simulate(markets, capital, size_usd, tb, mb, min_volume)
            cells.append(f"  ${s.total_pnl:>+7,.0f}")
        print(f"  tb={tb:.2f}" + "".join(cells))

    # 4. Position size sensitivity at default bid range
    sizes = [5, 10, 20, 50]
    print(f"\n  POSITION SIZE (target=0.75, max=0.85)")
    print(f"  {'Size':>6} | {'Trades':>6} | {'PnL':>12} | {'Return':>8} | {'MaxDD':>6}")
    print(f"  {'-'*6}-+-{'-'*6}-+-{'-'*12}-+-{'-'*8}-+-{'-'*6}")
    assign_vwap_from_buckets(markets, buckets_df, 0.75, 0.85)
    for sz in sizes:
        s = simulate(markets, capital, sz, 0.75, 0.85, min_volume)
        n = len(s.trades)
        ret = s.total_pnl / capital * 100
        dd = s.max_drawdown_pct * 100
        tag = " <--" if sz == 10 else ""
        print(f"  ${sz:>4} | {n:>6} | ${s.total_pnl:>+11,.2f} | {ret:>+7.1f}% | {dd:>5.1f}%{tag}")

    print()
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Backtest crypto TD maker strategy on Polymarket history")
    parser.add_argument("--data-dir",
                        default="../prediction-market-analysis/data/polymarket",
                        help="Path to polymarket data directory")
    parser.add_argument("--target-bid", type=float, default=0.75,
                        help="Min bid price to enter (default: 0.75)")
    parser.add_argument("--max-bid", type=float, default=0.85,
                        help="Max bid price to enter (default: 0.85)")
    parser.add_argument("--size-usd", type=float, default=10,
                        help="Position size per trade (USD)")
    parser.add_argument("--capital", type=float, default=1_000,
                        help="Starting capital (USD)")
    parser.add_argument("--min-volume", type=float, default=5.0,
                        help="Min USD volume in bid range for entry")
    parser.add_argument("--symbol", default=None, choices=["BTC", "ETH"],
                        help="Filter to BTC or ETH only")
    parser.add_argument("--sweep", action="store_true",
                        help="Run parameter sensitivity sweep")
    parser.add_argument("--no-chart", action="store_true",
                        help="Skip matplotlib capital curve chart")
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

    # Step 2: Compute trade buckets (single scan -- expensive)
    buckets_df = compute_trade_buckets(con, trades_dir, markets)

    # Step 3: Assign VWAP from buckets
    assign_vwap_from_buckets(markets, buckets_df, args.target_bid, args.max_bid)

    # Count tradeable markets
    tradeable = sum(1 for m in markets
                    if (m.up_vwap >= args.target_bid and m.up_vol_usd >= args.min_volume) or
                       (m.down_vwap >= args.target_bid and m.down_vol_usd >= args.min_volume))
    print(f"  Tradeable: {tradeable:,} markets with volume in [{args.target_bid}, {args.max_bid}]")

    # Step 4: Simulate
    print()
    print("=" * 70)
    print("CRYPTO TD MAKER BACKTEST RESULTS")
    print("=" * 70)
    print(f"  Capital: ${args.capital:,.0f}  |  Size: ${args.size_usd:.0f}/trade  "
          f"|  Symbol: {args.symbol or 'ALL'}")
    print(f"  Bid range: [{args.target_bid}, {args.max_bid}]  "
          f"|  Min volume: ${args.min_volume:.0f}")

    stats = simulate(markets, args.capital, args.size_usd,
                     args.target_bid, args.max_bid, args.min_volume)
    print_report(stats, args)

    print()
    print("=" * 70)

    # Step 5: Sweep
    if args.sweep:
        run_sweep(markets, buckets_df, args.capital, args.size_usd, args.min_volume)

    # Step 6: Capital curve
    if not args.no_chart and stats.trades:
        plot_capital_curve(stats, args)

    con.close()


if __name__ == "__main__":
    main()
