#!/usr/bin/env python3
"""Backtest crypto two-sided arb on historical Polymarket data.

Strategy: Buy equal shares of both Up and Down at market open when
ask_up + ask_down < 1.0 - fees, locking in guaranteed profit regardless
of which side wins.

Uses DuckDB to scan 45GB of trade parquet. For each resolved BTC/ETH
updown market, reconstructs the best available prices in the first N
seconds after market open using block-number interpolation for
timestamp estimation.

METHODOLOGY:
  1. Parse slug timestamp → market open time (slot boundary)
  2. Interpolate block numbers from sampled blocks table
  3. Single-pass trade scan filtered by token_id → temp table
  4. Filter to entry window (first N blocks after market open)
  5. Compute best ask (min fill price) and VWAP per side
  6. Edge = 1 - ask_up - ask_down - 2 * fee_rate
  7. If edge > min_edge: PnL = budget * edge / (ask_up + ask_down)

LIMITATIONS:
  - No orderbook snapshots — uses executed fill prices as proxy for asks
  - Block-time interpolation has ±5s accuracy (Polygon ~2s blocks)
  - Cannot model execution risk (partial fills, orphan positions)
  - Assumes both sides fill at best observed prices (optimistic)

Usage:
    python scripts/backtest_crypto_two_sided.py
    python scripts/backtest_crypto_two_sided.py --timeframe 300 --symbol BTC
    python scripts/backtest_crypto_two_sided.py --window 15 --fee-bps 200
    python scripts/backtest_crypto_two_sided.py --sweep
    python scripts/backtest_crypto_two_sided.py --pricing vwap
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import sys
import time as time_mod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import duckdb


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TwoSidedMarket:
    """A resolved crypto updown market for backtesting."""

    market_id: str
    slug: str
    symbol: str  # "BTC" or "ETH"
    timeframe: int  # 300 or 900
    up_token_id: str
    down_token_id: str
    up_won: bool
    open_epoch: float  # slot start time (from slug)
    volume: float
    # Filled after trade scan:
    best_ask_up: float = 0.0
    best_ask_down: float = 0.0
    vwap_up: float = 0.0
    vwap_down: float = 0.0
    volume_up: float = 0.0
    volume_down: float = 0.0
    trades_up: int = 0
    trades_down: int = 0


@dataclass
class TradeResult:
    slug: str
    symbol: str
    edge: float
    ask_up: float
    ask_down: float
    size_usd: float
    pnl_usd: float
    capital_after: float


@dataclass
class BacktestStats:
    pricing: str  # "best_ask" or "vwap"
    trades: list[TradeResult] = field(default_factory=list)
    initial_capital: float = 0.0
    final_capital: float = 0.0
    peak_capital: float = 0.0
    max_drawdown_pct: float = 0.0
    total_pnl: float = 0.0
    markets_with_edge: int = 0
    markets_scanned: int = 0


# ---------------------------------------------------------------------------
# Market loading
# ---------------------------------------------------------------------------


def load_markets(
    con: duckdb.DuckDBPyConnection,
    markets_dir: str,
    timeframe: int,
    symbol_filter: str | None,
) -> list[TwoSidedMarket]:
    """Load resolved crypto updown markets from parquet."""
    tf_min = timeframe // 60
    slug_pattern = f"%updown-{tf_min}m%"

    sym_filter = ""
    if symbol_filter:
        sym_filter = f"AND slug LIKE '{symbol_filter.lower()}%'"

    print(f"Loading resolved {tf_min}-min crypto updown markets...")
    t0 = time_mod.time()

    df = con.execute(f"""
        SELECT id, slug, outcome_prices, clob_token_ids, volume
        FROM '{markets_dir}/*.parquet'
        WHERE slug LIKE '{slug_pattern}' AND closed = true
        AND (slug LIKE 'btc%' OR slug LIKE 'eth%')
        {sym_filter}
    """).df()

    print(f"  {len(df):,} rows loaded in {time_mod.time()-t0:.1f}s")

    markets: list[TwoSidedMarket] = []
    for _, row in df.iterrows():
        try:
            prices = (
                json.loads(row["outcome_prices"])
                if row["outcome_prices"]
                else None
            )
            if not prices or len(prices) != 2:
                continue
            p0, p1 = float(prices[0]), float(prices[1])

            if p0 > 0.99 and p1 < 0.01:
                up_won = True
            elif p0 < 0.01 and p1 > 0.99:
                up_won = False
            else:
                continue

            token_ids = (
                json.loads(row["clob_token_ids"])
                if row["clob_token_ids"]
                else None
            )
            if not token_ids or len(token_ids) != 2:
                continue

            slug = row["slug"]
            # Parse open_epoch from slug: btc-updown-15m-1760028300
            parts = slug.split("-")
            slot_ts = int(parts[-1])

            symbol = "BTC" if slug.startswith("btc") else "ETH"

            markets.append(
                TwoSidedMarket(
                    market_id=str(row["id"]),
                    slug=slug,
                    symbol=symbol,
                    timeframe=timeframe,
                    up_token_id=token_ids[0],
                    down_token_id=token_ids[1],
                    up_won=up_won,
                    open_epoch=float(slot_ts),
                    volume=float(row.get("volume", 0) or 0),
                )
            )
        except (json.JSONDecodeError, ValueError, TypeError, IndexError):
            continue

    print(
        f"  {len(markets):,} markets parsed "
        f"(BTC: {sum(1 for m in markets if m.symbol == 'BTC'):,}, "
        f"ETH: {sum(1 for m in markets if m.symbol == 'ETH'):,})"
    )
    print(
        f"  Resolution: Up won {sum(1 for m in markets if m.up_won):,}, "
        f"Down won {sum(1 for m in markets if not m.up_won):,}"
    )
    if markets:
        min_ts = min(m.open_epoch for m in markets)
        max_ts = max(m.open_epoch for m in markets)
        dt_min = datetime.fromtimestamp(min_ts, tz=timezone.utc)
        dt_max = datetime.fromtimestamp(max_ts, tz=timezone.utc)
        print(f"  Period: {dt_min:%Y-%m-%d} to {dt_max:%Y-%m-%d}")
    return markets


# ---------------------------------------------------------------------------
# Block-time interpolation
# ---------------------------------------------------------------------------


def build_block_interpolator(
    con: duckdb.DuckDBPyConnection,
    blocks_dir: str,
    sample_interval: int = 500,
) -> tuple[list[int], list[float], float]:
    """Build sampled block-time index for epoch<->block conversion.

    Returns (block_numbers, epochs, sec_per_block).
    """
    print("Building block-time index...")
    t0 = time_mod.time()

    df = con.execute(f"""
        SELECT block_number, EPOCH(timestamp::TIMESTAMPTZ) as epoch
        FROM '{blocks_dir}/*.parquet'
        WHERE block_number % {sample_interval} = 0
        ORDER BY block_number
    """).df()

    block_nums = df["block_number"].tolist()
    epochs = df["epoch"].tolist()

    if len(block_nums) < 2:
        raise ValueError("Not enough blocks for interpolation")

    avg_spb = (epochs[-1] - epochs[0]) / (block_nums[-1] - block_nums[0])

    print(
        f"  {len(block_nums):,} samples in {time_mod.time()-t0:.1f}s "
        f"(avg {avg_spb:.2f}s/block)"
    )
    return block_nums, epochs, avg_spb


def _epoch_to_block(
    epoch: float, block_nums: list[int], epochs: list[float]
) -> int:
    """Convert epoch timestamp to estimated block number."""
    idx = bisect.bisect_right(epochs, epoch)
    if idx == 0:
        spb = (epochs[1] - epochs[0]) / max(block_nums[1] - block_nums[0], 1)
        return int(block_nums[0] + (epoch - epochs[0]) / spb)
    if idx >= len(epochs):
        spb = (epochs[-1] - epochs[-2]) / max(
            block_nums[-1] - block_nums[-2], 1
        )
        return int(block_nums[-1] + (epoch - epochs[-1]) / spb)
    # Interpolate between two nearest samples
    t0, t1 = epochs[idx - 1], epochs[idx]
    b0, b1 = block_nums[idx - 1], block_nums[idx]
    frac = (epoch - t0) / (t1 - t0) if t1 != t0 else 0
    return int(b0 + frac * (b1 - b0))


# ---------------------------------------------------------------------------
# Trade scanning
# ---------------------------------------------------------------------------


def scan_opening_trades(
    con: duckdb.DuckDBPyConnection,
    trades_dir: str,
    markets: list[TwoSidedMarket],
    block_nums: list[int],
    block_epochs: list[float],
    sec_per_block: float,
    entry_window_s: int,
) -> None:
    """Scan trades and compute per-token opening window stats.

    Modifies markets in-place with best_ask and vwap data.
    """
    if not markets:
        return

    # Build token -> market mapping and block ranges
    token_to_market: dict[str, tuple[int, str]] = {}
    token_rows: list[tuple[str,]] = []
    block_rows: list[tuple[str, int, int]] = []

    window_blocks = int(entry_window_s / sec_per_block) + 10  # margin

    for i, m in enumerate(markets):
        open_block = _epoch_to_block(m.open_epoch, block_nums, block_epochs)
        max_block = open_block + window_blocks

        token_rows.append((m.up_token_id,))
        token_rows.append((m.down_token_id,))
        block_rows.append((m.up_token_id, open_block, max_block))
        block_rows.append((m.down_token_id, open_block, max_block))
        token_to_market[m.up_token_id] = (i, "up")
        token_to_market[m.down_token_id] = (i, "down")

    # Register token IDs (for scan filter)
    con.execute("CREATE OR REPLACE TABLE bt_token_ids (token_id VARCHAR)")
    con.executemany("INSERT INTO bt_token_ids VALUES (?)", token_rows)

    # Register block ranges (for window filter)
    con.execute(
        "CREATE OR REPLACE TABLE bt_token_blocks "
        "(token_id VARCHAR, min_block BIGINT, max_block BIGINT)"
    )
    con.executemany(
        "INSERT INTO bt_token_blocks VALUES (?, ?, ?)", block_rows
    )

    print(
        f"Scanning trades for {len(token_rows):,} tokens "
        f"(window: {entry_window_s}s ~ {window_blocks} blocks)..."
    )
    print(
        "  Phase 1: extracting matching trades "
        "(this may take several minutes)..."
    )
    t0 = time_mod.time()

    # Phase 1: Single scan — filter by token_id, extract all matching trades
    con.execute(f"""
        CREATE OR REPLACE TABLE bt_raw_trades AS
        SELECT
            CASE WHEN t.maker_asset_id = '0' THEN t.taker_asset_id
                 ELSE t.maker_asset_id END AS token_id,
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
        INNER JOIN bt_token_ids ct ON ct.token_id = (
            CASE WHEN t.maker_asset_id = '0' THEN t.taker_asset_id
                 ELSE t.maker_asset_id END
        )
        WHERE t.taker_amount > 0 AND t.maker_amount > 0
    """)

    raw_count = con.execute("SELECT COUNT(*) FROM bt_raw_trades").fetchone()[0]
    print(
        f"  Phase 1 done in {time_mod.time()-t0:.1f}s "
        f"— {raw_count:,} trades extracted"
    )

    # Phase 2: Filter by block range and aggregate
    print("  Phase 2: filtering to entry windows and aggregating...")
    t1 = time_mod.time()

    agg_df = con.execute("""
        SELECT
            r.token_id,
            MIN(r.price) AS best_ask,
            SUM(r.usdc_amount * r.price)
                / NULLIF(SUM(r.usdc_amount), 0) AS vwap,
            SUM(r.usdc_amount) AS total_volume,
            COUNT(*) AS trade_count
        FROM bt_raw_trades r
        INNER JOIN bt_token_blocks b ON r.token_id = b.token_id
        WHERE r.block_number >= b.min_block
          AND r.block_number <= b.max_block
          AND r.price > 0.01 AND r.price < 0.99
        GROUP BY r.token_id
    """).df()

    print(
        f"  Phase 2 done in {time_mod.time()-t1:.1f}s "
        f"— {len(agg_df):,} tokens with trades in window"
    )

    # Assign to markets
    for _, row in agg_df.iterrows():
        token_id = row["token_id"]
        if token_id not in token_to_market:
            continue
        idx, side = token_to_market[token_id]
        m = markets[idx]

        best = float(row["best_ask"]) if row["best_ask"] is not None else 0.0
        vwap = float(row["vwap"]) if row["vwap"] is not None else 0.0
        vol = float(row["total_volume"]) if row["total_volume"] else 0.0
        cnt = int(row["trade_count"]) if row["trade_count"] else 0

        if side == "up":
            m.best_ask_up = best
            m.vwap_up = vwap
            m.volume_up = vol
            m.trades_up = cnt
        else:
            m.best_ask_down = best
            m.vwap_down = vwap
            m.volume_down = vol
            m.trades_down = cnt

    # Cleanup
    con.execute("DROP TABLE IF EXISTS bt_raw_trades")
    con.execute("DROP TABLE IF EXISTS bt_token_ids")
    con.execute("DROP TABLE IF EXISTS bt_token_blocks")


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def _compute_edge(
    ask_up: float, ask_down: float, fee_rate: float
) -> float:
    """Structural edge: guaranteed profit per share pair."""
    return 1.0 - ask_up - ask_down - 2 * fee_rate


def simulate(
    markets: list[TwoSidedMarket],
    pricing: str,
    min_edge: float,
    fee_rate: float,
    budget: float,
    capital: float,
    min_volume: float,
) -> BacktestStats:
    """Simulate two-sided arb across all markets chronologically."""
    stats = BacktestStats(
        pricing=pricing,
        initial_capital=capital,
        peak_capital=capital,
    )
    current = capital

    sorted_markets = sorted(markets, key=lambda m: m.open_epoch)

    for m in sorted_markets:
        stats.markets_scanned += 1

        if current < budget:
            break

        # Get prices based on pricing mode
        if pricing == "best_ask":
            ask_up = m.best_ask_up
            ask_down = m.best_ask_down
        else:  # vwap
            ask_up = m.vwap_up
            ask_down = m.vwap_down

        # Need both sides tradeable
        if ask_up <= 0 or ask_down <= 0:
            continue

        # Volume filter
        if m.volume_up < min_volume or m.volume_down < min_volume:
            continue

        edge = _compute_edge(ask_up, ask_down, fee_rate)
        if edge < min_edge:
            continue

        stats.markets_with_edge += 1

        # PnL: buy equal shares of both sides
        # shares = budget / (ask_up + ask_down)
        # pnl = shares * edge (guaranteed regardless of outcome)
        actual_budget = min(budget, current)
        cost_per_pair = ask_up + ask_down
        pnl = actual_budget * edge / cost_per_pair

        current += pnl
        stats.peak_capital = max(stats.peak_capital, current)

        dd = (
            (stats.peak_capital - current) / stats.peak_capital
            if stats.peak_capital > 0
            else 0
        )
        stats.max_drawdown_pct = max(stats.max_drawdown_pct, dd)

        stats.trades.append(
            TradeResult(
                slug=m.slug,
                symbol=m.symbol,
                edge=edge,
                ask_up=ask_up,
                ask_down=ask_down,
                size_usd=actual_budget,
                pnl_usd=pnl,
                capital_after=current,
            )
        )

    stats.final_capital = current
    stats.total_pnl = current - capital
    return stats


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _compute_sharpe(trades: list[TradeResult]) -> float:
    """Annualized Sharpe from per-trade returns."""
    if len(trades) < 2:
        return 0.0
    returns = [
        t.pnl_usd / t.size_usd if t.size_usd > 0 else 0 for t in trades
    ]
    mean_r = sum(returns) / len(returns)
    var_r = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
    std_r = math.sqrt(var_r) if var_r > 0 else 0
    if std_r == 0:
        return 0.0
    trades_per_year = max(len(trades), 100)
    return (mean_r / std_r) * math.sqrt(trades_per_year)


def _slug_to_month(slug: str) -> str:
    """Extract YYYY-MM from slug timestamp."""
    try:
        ts = int(slug.split("-")[-1])
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        return dt.strftime("%Y-%m")
    except (ValueError, IndexError):
        return "unknown"


def print_edge_distribution(
    markets: list[TwoSidedMarket], fee_rate: float, pricing: str
) -> None:
    """Show distribution of edges across all markets."""
    edges = []
    for m in markets:
        ask_up = m.best_ask_up if pricing == "best_ask" else m.vwap_up
        ask_down = m.best_ask_down if pricing == "best_ask" else m.vwap_down
        if ask_up > 0 and ask_down > 0:
            edges.append(_compute_edge(ask_up, ask_down, fee_rate))

    if not edges:
        print("\n  No markets with tradeable quotes on both sides.")
        return

    print(
        f"\n  EDGE DISTRIBUTION "
        f"({len(edges):,} markets with both sides quoted)"
    )
    zones = [
        ("< -5%", -999, -0.05),
        ("-5% to -2%", -0.05, -0.02),
        ("-2% to 0%", -0.02, 0.0),
        ("0% to 1%", 0.0, 0.01),
        ("1% to 2%", 0.01, 0.02),
        ("2% to 5%", 0.02, 0.05),
        ("5% to 10%", 0.05, 0.10),
        ("> 10%", 0.10, 999),
    ]

    print(
        f"  {'Edge Zone':<14} | {'Markets':>7} | {'Pct':>5} | {'Avg Edge':>9}"
    )
    print(f"  {'-'*14}-+-{'-'*7}-+-{'-'*5}-+-{'-'*9}")

    for label, lo, hi in zones:
        in_zone = [e for e in edges if lo <= e < hi]
        if not in_zone:
            continue
        pct = len(in_zone) / len(edges) * 100
        avg = sum(in_zone) / len(in_zone) * 100
        print(
            f"  {label:<14} | {len(in_zone):>7,} | {pct:>4.1f}% | "
            f"{avg:>+8.2f}%"
        )

    positive = [e for e in edges if e > 0]
    print(
        f"\n  Markets with positive edge: {len(positive):,} / {len(edges):,} "
        f"({len(positive)/len(edges)*100:.1f}%)"
    )
    if positive:
        print(f"  Avg positive edge: {sum(positive)/len(positive)*100:+.2f}%")


def print_report(stats: BacktestStats) -> None:
    """Print simulation results."""
    n = len(stats.trades)
    if n == 0:
        print(f"\n  {stats.pricing.upper()}: No trades executed.")
        return

    ret_pct = stats.total_pnl / stats.initial_capital * 100
    avg_edge = sum(t.edge for t in stats.trades) / n * 100
    avg_pnl = stats.total_pnl / n

    print(f"\n  SIMULATION ({stats.pricing.upper()} pricing)")
    print(f"  {'—' * 55}")
    print(
        f"  Trades:          {n:,} "
        f"(from {stats.markets_scanned:,} markets scanned)"
    )
    print(f"  Markets w/ edge: {stats.markets_with_edge:,}")
    print(f"  Win rate:        100.0% (structural arb)")
    print(f"  Avg edge:        {avg_edge:+.2f}%")
    print(f"  Avg PnL/trade:   ${avg_pnl:+,.2f}")
    print(f"  Total PnL:       ${stats.total_pnl:+,.2f}")
    print(f"  Return:          {ret_pct:+.1f}%")
    print(f"  Final capital:   ${stats.final_capital:,.2f}")
    print(f"  Max drawdown:    {stats.max_drawdown_pct*100:.1f}%")
    print(f"  Sharpe (ann.):   {_compute_sharpe(stats.trades):.2f}")

    # By symbol
    print(
        f"\n  {'Symbol':<6} | {'Trades':>6} | {'Avg Edge':>9} | "
        f"{'PnL':>12}"
    )
    print(f"  {'-'*6}-+-{'-'*6}-+-{'-'*9}-+-{'-'*12}")
    for sym in ["BTC", "ETH"]:
        st = [t for t in stats.trades if t.symbol == sym]
        if not st:
            continue
        s_edge = sum(t.edge for t in st) / len(st) * 100
        s_pnl = sum(t.pnl_usd for t in st)
        print(
            f"  {sym:<6} | {len(st):>6} | {s_edge:>+8.2f}% | "
            f"${s_pnl:>+11,.2f}"
        )

    # By month
    months: dict[str, list[TradeResult]] = {}
    for t in stats.trades:
        months.setdefault(_slug_to_month(t.slug), []).append(t)

    if len(months) > 1:
        print(
            f"\n  {'Month':<8} | {'Trades':>6} | {'Avg Edge':>9} | "
            f"{'PnL':>12}"
        )
        print(f"  {'-'*8}-+-{'-'*6}-+-{'-'*9}-+-{'-'*12}")
        for month in sorted(months.keys()):
            mt = months[month]
            m_edge = sum(t.edge for t in mt) / len(mt) * 100
            m_pnl = sum(t.pnl_usd for t in mt)
            print(
                f"  {month:<8} | {len(mt):>6} | {m_edge:>+8.2f}% | "
                f"${m_pnl:>+11,.2f}"
            )

    # By edge zone
    print(
        f"\n  {'Edge Zone':<12} | {'Trades':>6} | {'Avg PnL':>10} | "
        f"{'Total PnL':>12}"
    )
    print(f"  {'-'*12}-+-{'-'*6}-+-{'-'*10}-+-{'-'*12}")
    edge_zones = [
        ("0-1%", 0.0, 0.01),
        ("1-2%", 0.01, 0.02),
        ("2-5%", 0.02, 0.05),
        ("5-10%", 0.05, 0.10),
        (">10%", 0.10, 999),
    ]
    for label, lo, hi in edge_zones:
        zt = [t for t in stats.trades if lo <= t.edge < hi]
        if not zt:
            continue
        z_avg_pnl = sum(t.pnl_usd for t in zt) / len(zt)
        z_total_pnl = sum(t.pnl_usd for t in zt)
        print(
            f"  {label:<12} | {len(zt):>6} | "
            f"${z_avg_pnl:>+9,.2f} | ${z_total_pnl:>+11,.2f}"
        )


def run_sweep(
    markets: list[TwoSidedMarket],
    fee_rate: float,
    capital: float,
    budget: float,
    min_volume: float,
) -> None:
    """Parameter sensitivity sweep."""
    print()
    print("=" * 70)
    print("PARAMETER SENSITIVITY SWEEP")
    print("=" * 70)

    # Min edge sweep
    print(f"\n  MIN EDGE — varying minimum edge threshold (best_ask pricing)")
    thresholds = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.10]
    print(
        f"  {'Min Edge':>9} | {'Trades':>6} | {'Avg Edge':>9} | "
        f"{'PnL':>12} | {'Return':>8} | {'Sharpe':>6}"
    )
    print(
        f"  {'-'*9}-+-{'-'*6}-+-{'-'*9}-+-{'-'*12}-+-{'-'*8}-+-{'-'*6}"
    )

    for me in thresholds:
        s = simulate(
            markets, "best_ask", me, fee_rate, budget, capital, min_volume
        )
        n = len(s.trades)
        if n == 0:
            continue
        avg_e = sum(t.edge for t in s.trades) / n * 100
        ret = s.total_pnl / capital * 100
        sh = _compute_sharpe(s.trades)
        tag = " <--" if me == 0.01 else ""
        print(
            f"  {me*100:>8.1f}% | {n:>6} | {avg_e:>+8.2f}% | "
            f"${s.total_pnl:>+11,.2f} | {ret:>+7.1f}% | {sh:>6.2f}{tag}"
        )

    # Fee rate sweep
    print(f"\n  FEE RATE — varying taker fee (best_ask, min_edge=1%)")
    fees = [0.0, 0.005, 0.01, 0.015, 0.02, 0.03]
    print(
        f"  {'Fee BPS':>8} | {'Trades':>6} | {'Avg Edge':>9} | "
        f"{'PnL':>12} | {'Return':>8}"
    )
    print(f"  {'-'*8}-+-{'-'*6}-+-{'-'*9}-+-{'-'*12}-+-{'-'*8}")

    for fr in fees:
        s = simulate(
            markets, "best_ask", 0.01, fr, budget, capital, min_volume
        )
        n = len(s.trades)
        if n == 0:
            print(f"  {fr*10000:>7.0f} |      0 |           |              |")
            continue
        avg_e = sum(t.edge for t in s.trades) / n * 100
        ret = s.total_pnl / capital * 100
        tag = " <--" if fr == fee_rate else ""
        print(
            f"  {fr*10000:>7.0f} | {n:>6} | {avg_e:>+8.2f}% | "
            f"${s.total_pnl:>+11,.2f} | {ret:>+7.1f}%{tag}"
        )

    # Pricing comparison
    print(
        f"\n  PRICING MODE COMPARISON "
        f"(min_edge=1%, fee={fee_rate*10000:.0f}bps)"
    )
    for pricing in ["best_ask", "vwap"]:
        s = simulate(
            markets, pricing, 0.01, fee_rate, budget, capital, min_volume
        )
        n = len(s.trades)
        if n == 0:
            print(f"  {pricing:<10}: no trades")
            continue
        avg_e = sum(t.edge for t in s.trades) / n * 100
        ret = s.total_pnl / capital * 100
        print(
            f"  {pricing:<10}: {n:>6} trades, avg edge {avg_e:+.2f}%, "
            f"PnL ${s.total_pnl:+,.2f}, return {ret:+.1f}%"
        )

    # Budget sweep
    print(f"\n  BUDGET — varying per-market budget (best_ask, min_edge=1%)")
    budgets = [50, 100, 200, 500]
    print(
        f"  {'Budget':>8} | {'Trades':>6} | {'PnL':>12} | "
        f"{'Return':>8} | {'Max DD':>7}"
    )
    print(f"  {'-'*8}-+-{'-'*6}-+-{'-'*12}-+-{'-'*8}-+-{'-'*7}")

    for b in budgets:
        s = simulate(
            markets, "best_ask", 0.01, fee_rate, b, capital, min_volume
        )
        n = len(s.trades)
        if n == 0:
            continue
        ret = s.total_pnl / capital * 100
        dd = s.max_drawdown_pct * 100
        tag = " <--" if b == budget else ""
        print(
            f"  ${b:>6} | {n:>6} | ${s.total_pnl:>+11,.2f} | "
            f"{ret:>+7.1f}% | {dd:>6.1f}%{tag}"
        )

    print()
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Backtest crypto two-sided arb on Polymarket history"
    )
    parser.add_argument(
        "--data-dir",
        default="../prediction-market-analysis/data/polymarket",
        help="Path to polymarket data directory",
    )
    parser.add_argument(
        "--timeframe",
        type=int,
        default=900,
        choices=[300, 900],
        help="Market timeframe in seconds (300=5min, 900=15min)",
    )
    parser.add_argument(
        "--symbol",
        default=None,
        choices=["BTC", "ETH"],
        help="Filter to BTC or ETH only",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=30,
        help="Entry window in seconds after market open",
    )
    parser.add_argument(
        "--fee-bps",
        type=int,
        default=100,
        help="Taker fee in basis points (default: 100 = 1%%)",
    )
    parser.add_argument(
        "--min-edge",
        type=float,
        default=0.01,
        help="Minimum edge to trade (default: 0.01 = 1%%)",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=200,
        help="Budget per market in USD",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=2000,
        help="Starting capital in USD",
    )
    parser.add_argument(
        "--min-volume",
        type=float,
        default=1.0,
        help="Min USD volume per side in window to consider tradeable",
    )
    parser.add_argument(
        "--pricing",
        default="best_ask",
        choices=["best_ask", "vwap"],
        help="Price estimation method",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run parameter sensitivity sweep",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    markets_dir = str(data_dir / "markets")
    trades_dir = str(data_dir / "trades")
    blocks_dir = str(data_dir / "blocks")

    for d, name in [
        (data_dir / "markets", "markets"),
        (data_dir / "trades", "trades"),
        (data_dir / "blocks", "blocks"),
    ]:
        if not d.exists():
            print(f"Error: {name} directory not found at {d}")
            sys.exit(1)

    con = duckdb.connect()
    fee_rate = args.fee_bps / 10_000

    # Step 1: Load markets
    markets = load_markets(con, markets_dir, args.timeframe, args.symbol)
    if not markets:
        print("No markets found.")
        sys.exit(0)

    # Step 2: Build block-time interpolator
    block_nums, block_epochs, sec_per_block = build_block_interpolator(
        con, blocks_dir
    )

    # Step 3: Scan trades for opening window prices
    scan_opening_trades(
        con,
        trades_dir,
        markets,
        block_nums,
        block_epochs,
        sec_per_block,
        args.window,
    )

    # Summary
    both_sides = sum(
        1 for m in markets if m.best_ask_up > 0 and m.best_ask_down > 0
    )
    print(
        f"  Markets with both sides quoted: "
        f"{both_sides:,} / {len(markets):,}"
    )

    # Step 4: Report
    print()
    print("=" * 70)
    tf_min = args.timeframe // 60
    print(f"CRYPTO TWO-SIDED BACKTEST — {tf_min}-MIN MARKETS")
    print("=" * 70)
    print(
        f"  Capital: ${args.capital:,.0f}  |  Budget: ${args.budget:.0f}/market"
        f"  |  Symbol: {args.symbol or 'ALL'}"
    )
    print(
        f"  Entry window: {args.window}s  |  Fee: {args.fee_bps}bps"
        f"  |  Min edge: {args.min_edge*100:.1f}%"
    )

    print_edge_distribution(markets, fee_rate, args.pricing)

    stats = simulate(
        markets,
        args.pricing,
        args.min_edge,
        fee_rate,
        args.budget,
        args.capital,
        args.min_volume,
    )
    print_report(stats)

    print()
    print("=" * 70)

    # Step 5: Sweep
    if args.sweep:
        run_sweep(markets, fee_rate, args.capital, args.budget, args.min_volume)

    con.close()


if __name__ == "__main__":
    main()
