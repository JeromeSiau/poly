#!/usr/bin/env python3
"""Calibrate TD maker sizing based on historical fill conditions.

For each resolved crypto 15-min market, looks at individual trades on the
favorite-side token in our bid range, joins with block timestamps to compute
minutes remaining at fill time, and builds a WR calibration table.

The key question: "given a maker fill at price P with M minutes remaining,
what is the historical win rate?"  This directly feeds Kelly sizing.

Dataset: 41K+ resolved BTC/ETH/SOL/XRP 15-min updown markets (Oct 2025-Feb 2026)
Trades: ~45GB parquet joined with block timestamps for per-trade timing.

Usage:
    ./run scripts/calibrate_td_sizing.py
    ./run scripts/calibrate_td_sizing.py --bid-lo 0.73 --bid-hi 0.85
    ./run scripts/calibrate_td_sizing.py --symbol BTC
    ./run scripts/calibrate_td_sizing.py --base-size 10 --max-size 20
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import duckdb
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "prediction-market-analysis" / "data" / "polymarket"
MARKETS_DIR = DATA_DIR / "markets"
TRADES_DIR = DATA_DIR / "trades"
BLOCKS_DIR = DATA_DIR / "blocks"


def build_market_tokens(con: duckdb.DuckDBPyConnection, symbol_filter: str | None = None) -> int:
    """Parse markets in pure SQL — creates mkt_tokens table in DuckDB.

    Each row: (token_id VARCHAR, is_up BOOL, up_won BOOL, end_ts BIGINT, symbol VARCHAR)
    Returns the number of markets parsed.
    """
    t0 = time.time()
    slug_filter = f"AND slug LIKE '{symbol_filter.lower()}%'" if symbol_filter else ""

    # Parse JSON inside DuckDB — no Python loop needed.
    con.execute(f"""
        CREATE OR REPLACE TABLE mkt_tokens AS
        WITH parsed AS (
            SELECT
                slug,
                outcome_prices::JSON AS op,
                clob_token_ids::JSON AS ct,
                CAST(SPLIT_PART(slug, '-', -1) AS BIGINT) AS end_ts
            FROM '{MARKETS_DIR}/*.parquet'
            WHERE slug LIKE '%updown-15m%'
              AND closed = true
              {slug_filter}
        ),
        resolved AS (
            SELECT
                slug, end_ts,
                CAST(ct->>0 AS VARCHAR) AS up_token,
                CAST(ct->>1 AS VARCHAR) AS down_token,
                CAST(op->>0 AS DOUBLE) AS p0,
                CAST(op->>1 AS DOUBLE) AS p1,
                CASE
                    WHEN slug LIKE 'btc%' THEN 'BTC'
                    WHEN slug LIKE 'eth%' THEN 'ETH'
                    WHEN slug LIKE 'sol%' THEN 'SOL'
                    WHEN slug LIKE 'xrp%' THEN 'XRP'
                END AS symbol
            FROM parsed
            WHERE CAST(op->>0 AS DOUBLE) IS NOT NULL
              AND CAST(op->>1 AS DOUBLE) IS NOT NULL
              AND CAST(ct->>0 AS VARCHAR) IS NOT NULL
              AND CAST(ct->>1 AS VARCHAR) IS NOT NULL
        ),
        fully_resolved AS (
            SELECT *,
                CASE WHEN p0 > 0.99 AND p1 < 0.01 THEN TRUE
                     WHEN p0 < 0.01 AND p1 > 0.99 THEN FALSE
                END AS up_won
            FROM resolved
            WHERE (p0 > 0.99 AND p1 < 0.01) OR (p0 < 0.01 AND p1 > 0.99)
        )
        -- Unpivot: one row per token (up + down)
        SELECT up_token AS token_id, TRUE AS is_up, up_won, end_ts, symbol
        FROM fully_resolved
        WHERE symbol IS NOT NULL
        UNION ALL
        SELECT down_token AS token_id, FALSE AS is_up, up_won, end_ts, symbol
        FROM fully_resolved
        WHERE symbol IS NOT NULL
    """)

    count = con.execute("SELECT COUNT(*) FROM mkt_tokens").fetchone()[0]
    n_markets = count // 2
    elapsed = time.time() - t0
    print(f"Loaded {n_markets:,} resolved markets ({count:,} tokens) in {elapsed:.1f}s")

    # Quick sanity check
    sample = con.execute("SELECT token_id, symbol, end_ts FROM mkt_tokens LIMIT 3").df()
    print(f"  Sample tokens: {sample.to_dict('records')}")

    return n_markets


def build_calibration(
    con: duckdb.DuckDBPyConnection,
    bid_lo: float,
    bid_hi: float,
) -> pd.DataFrame:
    """Join trades with blocks and market info to compute per-fill WR table."""

    print(f"Scanning trades in [{bid_lo:.2f}, {bid_hi:.2f}] and joining with blocks...")
    print("  (this may take several minutes on 45GB of trades)")
    t0 = time.time()

    # Step 1: Get all matching trades with prices (no blocks yet)
    con.execute(f"""
        CREATE OR REPLACE TABLE matched_trades AS
        SELECT
            t.block_number,
            mt.end_ts,
            mt.is_up,
            mt.up_won,
            mt.symbol,
            CASE WHEN t.maker_asset_id = '0'
                 THEN 1.0 * t.maker_amount / t.taker_amount
                 ELSE 1.0 * t.taker_amount / t.maker_amount
            END AS price,
            CASE WHEN t.maker_asset_id = '0'
                 THEN t.maker_amount / 1e6
                 ELSE t.taker_amount / 1e6
            END AS usdc_vol
        FROM '{TRADES_DIR}/*.parquet' t
        INNER JOIN mkt_tokens mt
            ON mt.token_id = (
                CASE WHEN t.maker_asset_id = '0'
                     THEN t.taker_asset_id
                     ELSE t.maker_asset_id
                END
            )
        WHERE t.taker_amount > 0 AND t.maker_amount > 0
    """)

    n_all = con.execute("SELECT COUNT(*) FROM matched_trades").fetchone()[0]
    print(f"  Step 1: {n_all:,} trades matched to tokens in {time.time()-t0:.1f}s")

    # Step 2: Filter to price range
    n_range = con.execute(f"""
        SELECT COUNT(*) FROM matched_trades
        WHERE price >= {bid_lo} AND price <= {bid_hi}
    """).fetchone()[0]
    print(f"  Step 2: {n_range:,} trades in price range [{bid_lo}, {bid_hi}]")

    # Step 3: Join with blocks for timestamps and compute minutes_left
    t1 = time.time()
    result = con.execute(f"""
        SELECT
            CAST(FLOOR((mt.end_ts - EPOCH(CAST(b.timestamp AS TIMESTAMP))) / 60.0) AS INTEGER)
                AS minutes_left,
            FLOOR(mt.price * 100) / 100.0 AS price_1c,
            mt.symbol,
            COUNT(*) AS n_trades,
            SUM(mt.usdc_vol) AS volume_usd,
            SUM(CASE
                WHEN (mt.is_up AND mt.up_won) OR (NOT mt.is_up AND NOT mt.up_won) THEN 1
                ELSE 0
            END) AS wins,
            ROUND(
                SUM(CASE
                    WHEN (mt.is_up AND mt.up_won) OR (NOT mt.is_up AND NOT mt.up_won) THEN 1
                    ELSE 0
                END)::DOUBLE / COUNT(*), 4
            ) AS win_rate
        FROM matched_trades mt
        INNER JOIN '{BLOCKS_DIR}/*.parquet' b
            ON b.block_number = mt.block_number
        WHERE mt.price >= {bid_lo} AND mt.price <= {bid_hi}
          AND (mt.end_ts - EPOCH(CAST(b.timestamp AS TIMESTAMP))) >= 0
          AND (mt.end_ts - EPOCH(CAST(b.timestamp AS TIMESTAMP))) <= 900
        GROUP BY minutes_left, price_1c, mt.symbol
        ORDER BY minutes_left, price_1c, mt.symbol
    """).df()

    elapsed = time.time() - t0
    print(f"  Step 3: blocks join + aggregation in {time.time()-t1:.1f}s")
    print(f"  Total: {elapsed:.1f}s — {len(result):,} calibration rows, "
          f"{result['n_trades'].sum():,} trades in final output")
    return result


def kelly_size(p_win: float, entry_price: float, base_size: float, max_size: float) -> float:
    """Kelly criterion size for a binary bet at entry_price."""
    if p_win <= entry_price:
        return 0.0
    b = (1.0 / entry_price) - 1.0  # payout ratio (win/loss)
    q = 1.0 - p_win
    kelly_frac = (p_win * b - q) / b
    # Use fractional Kelly (1/4) for safety, scale to base_size
    size = base_size * (1.0 + kelly_frac * 4.0)
    return min(max(size * 0.5, 0.0), max_size)  # floor at 50% of base


def print_calibration_table(cal: pd.DataFrame, bid_lo: float, bid_hi: float,
                            base_size: float, max_size: float) -> None:
    """Pretty-print the calibration table."""

    # Aggregate across symbols for the main table
    agg = cal.groupby(["minutes_left", "price_1c"]).agg(
        n_trades=("n_trades", "sum"),
        wins=("wins", "sum"),
        volume_usd=("volume_usd", "sum"),
    ).reset_index()
    agg["wr"] = agg["wins"] / agg["n_trades"]

    # --- Table 1: WR by minutes_left (all prices) ---
    print("\n" + "=" * 70)
    print("WR by MINUTES REMAINING (all prices in range)")
    print("=" * 70)
    by_min = cal.groupby("minutes_left").agg(
        n_trades=("n_trades", "sum"),
        wins=("wins", "sum"),
    ).reset_index()
    by_min["wr"] = by_min["wins"] / by_min["n_trades"]
    by_min = by_min[by_min["minutes_left"] <= 15].sort_values("minutes_left")

    print(f"{'Min':>4} | {'Trades':>8} | {'Wins':>8} | {'WR':>6} | {'Edge vs 0.75':>12} | {'Kelly $':>8}")
    print("-" * 60)
    for _, r in by_min.iterrows():
        wr = r["wr"]
        edge = wr - 0.75
        ks = kelly_size(wr, 0.75, base_size, max_size)
        flag = " ***" if wr < 0.73 else " +++" if wr > 0.80 else ""
        print(f"{int(r['minutes_left']):4d} | {int(r['n_trades']):8,} | {int(r['wins']):8,} | "
              f"{wr:5.1%} | {edge:+11.1%} | ${ks:7.2f}{flag}")

    # --- Table 2: WR by price_1c (all minutes) ---
    print("\n" + "=" * 70)
    print("WR by PRICE (all minutes)")
    print("=" * 70)
    by_price = cal.groupby("price_1c").agg(
        n_trades=("n_trades", "sum"),
        wins=("wins", "sum"),
    ).reset_index()
    by_price["wr"] = by_price["wins"] / by_price["n_trades"]
    by_price = by_price.sort_values("price_1c")

    print(f"{'Price':>6} | {'Trades':>8} | {'WR':>6} | {'BE WR':>6} | {'Edge':>6} | {'Kelly $':>8}")
    print("-" * 56)
    for _, r in by_price.iterrows():
        p = r["price_1c"]
        wr = r["wr"]
        be = p  # break-even WR = entry price for binary bet
        edge = wr - be
        ks = kelly_size(wr, p, base_size, max_size)
        flag = " ***" if edge < -0.01 else " +++" if edge > 0.03 else ""
        print(f"{p:6.2f} | {int(r['n_trades']):8,} | {wr:5.1%} | {be:5.1%} | {edge:+5.1%} | ${ks:7.2f}{flag}")

    # --- Table 3: Cross-table (minutes × price) for the entry window ---
    print("\n" + "=" * 70)
    print("WR CROSS-TABLE: minutes_left × price (entry window 0-5 min)")
    print("  (? = less than 100 trades, - = no data)")
    print("=" * 70)
    cross = agg[(agg["minutes_left"] >= 0) & (agg["minutes_left"] <= 5)]
    pivot = cross.pivot_table(
        index="minutes_left", columns="price_1c", values="wr", aggfunc="first",
    )
    pivot_n = cross.pivot_table(
        index="minutes_left", columns="price_1c", values="n_trades", aggfunc="first",
    )

    prices = sorted(pivot.columns)
    header = f"{'Min':>4} |" + "".join(f" {p:5.2f}  " for p in prices)
    print(header)
    print("-" * len(header))
    for m in sorted(pivot.index):
        cells = []
        for p in prices:
            wr = pivot.loc[m, p] if p in pivot.columns and pd.notna(pivot.loc[m, p]) else None
            n = int(pivot_n.loc[m, p]) if p in pivot_n.columns and pd.notna(pivot_n.loc[m, p]) else 0
            if wr is not None and n >= 100:
                cells.append(f" {wr:4.0%}   ")
            elif wr is not None and n >= 20:
                cells.append(f" {wr:4.0%}?  ")  # low sample
            elif wr is not None:
                cells.append(f" ({wr:3.0%})  ")  # very low
            else:
                cells.append("    -   ")
        print(f"{int(m):4d} |" + "".join(cells))

    # --- Table 4: Suggested sizing ---
    print("\n" + "=" * 70)
    print(f"SUGGESTED SIZING (base=${base_size:.0f}, max=${max_size:.0f}, 1/4 Kelly)")
    print("  Only cells with 100+ trades")
    print("=" * 70)
    cross_sz = agg[(agg["minutes_left"] >= 0) & (agg["minutes_left"] <= 5) & (agg["n_trades"] >= 100)]
    cross_sz = cross_sz.sort_values(["minutes_left", "price_1c"])

    print(f"{'Min':>4} | {'Price':>6} | {'Trades':>7} | {'WR':>6} | {'Edge':>6} | {'Size $':>7}")
    print("-" * 50)
    for _, r in cross_sz.iterrows():
        m = int(r["minutes_left"])
        p = r["price_1c"]
        wr = r["wr"]
        edge = wr - p
        ks = kelly_size(wr, p, base_size, max_size)
        action = "SKIP" if ks < 1.0 else f"${ks:.1f}"
        print(f"{m:4d} | {p:6.2f} | {int(r['n_trades']):7,} | {wr:5.1%} | {edge:+5.1%} | {action:>7}")

    # --- Table 5: Per-symbol summary ---
    print("\n" + "=" * 70)
    print("PER-SYMBOL SUMMARY (0-5 min remaining)")
    print("=" * 70)
    sym = cal[cal["minutes_left"] <= 5].groupby("symbol").agg(
        n_trades=("n_trades", "sum"),
        wins=("wins", "sum"),
    ).reset_index()
    sym["wr"] = sym["wins"] / sym["n_trades"]

    for _, r in sym.iterrows():
        wr = r["wr"]
        edge = wr - 0.75
        print(f"  {r['symbol']:4s}: {int(r['n_trades']):7,} trades, "
              f"WR={wr:.1%}, edge vs 0.75={edge:+.1%}")


def main():
    parser = argparse.ArgumentParser(description="Calibrate TD maker sizing")
    parser.add_argument("--bid-lo", type=float, default=0.70,
                        help="Lower bound of bid range (default: 0.70)")
    parser.add_argument("--bid-hi", type=float, default=0.90,
                        help="Upper bound of bid range (default: 0.90)")
    parser.add_argument("--symbol", type=str, default=None,
                        help="Filter by symbol (BTC, ETH, SOL, XRP)")
    parser.add_argument("--base-size", type=float, default=10.0,
                        help="Base order size in USD (default: 10)")
    parser.add_argument("--max-size", type=float, default=20.0,
                        help="Maximum order size in USD (default: 20)")
    args = parser.parse_args()

    con = duckdb.connect()

    # 1. Load markets — all SQL, no Python parsing
    n_markets = build_market_tokens(con, symbol_filter=args.symbol)
    if n_markets == 0:
        print("No markets found!")
        return

    # 2. Build calibration table
    cal = build_calibration(con, args.bid_lo, args.bid_hi)
    if cal.empty:
        print("No trades found in range!")
        return

    # 3. Print results
    print_calibration_table(cal, args.bid_lo, args.bid_hi, args.base_size, args.max_size)

    # 4. Save raw data
    out_path = Path("data/td_calibration.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cal.to_csv(out_path, index=False)
    print(f"\nRaw calibration data saved to {out_path}")


if __name__ == "__main__":
    main()
