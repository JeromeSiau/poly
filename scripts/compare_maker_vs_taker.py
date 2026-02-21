#!/usr/bin/env python3
"""Compare maker vs taker approach for crypto TD strategy.

Uses historical Polymarket data to answer:
  - Maker: bid in [0.75, 0.85], 0 fees, but fewer fills
  - Taker: buy at ask (VWAP shifted +1-2c for spread), pays taker fees, but fills on every opportunity

Key metrics: fill count, win rate, PnL, edge after fees.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import pandas as pd


# ---------------------------------------------------------------------------
# Taker fee formula (from src/shadow/taker_shadow.py)
# ---------------------------------------------------------------------------

def crypto_taker_fee(price: float) -> float:
    """Per-share taker fee: price * 0.25 * (price * (1-price))^2"""
    return price * 0.25 * (price * (1.0 - price)) ** 2


# ---------------------------------------------------------------------------
# Data loading (reused from backtest_crypto_td_maker.py)
# ---------------------------------------------------------------------------

@dataclass
class Market:
    slug: str
    symbol: str
    up_token_id: str
    down_token_id: str
    up_won: bool
    end_date: str
    volume: float
    # Per-outcome VWAP in different price ranges:
    up_vwap: dict = field(default_factory=dict)    # range_key -> vwap
    up_vol: dict = field(default_factory=dict)      # range_key -> usd_vol
    down_vwap: dict = field(default_factory=dict)
    down_vol: dict = field(default_factory=dict)


def load_markets(con: duckdb.DuckDBPyConnection, markets_dir: str) -> list[Market]:
    print("Loading resolved crypto-minute markets...")
    t0 = time.time()
    df = con.execute(f"""
        SELECT id, slug, outcome_prices, clob_token_ids, volume, end_date
        FROM '{markets_dir}/*.parquet'
        WHERE slug LIKE '%updown-15m%' AND closed = true
    """).df()
    print(f"  {len(df):,} rows loaded in {time.time()-t0:.1f}s")

    markets = []
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
            if slug.startswith("btc"):
                sym = "BTC"
            elif slug.startswith("eth"):
                sym = "ETH"
            elif slug.startswith("sol"):
                sym = "SOL"
            elif slug.startswith("xrp"):
                sym = "XRP"
            else:
                continue
            markets.append(Market(
                slug=slug, symbol=sym,
                up_token_id=token_ids[0], down_token_id=token_ids[1],
                up_won=up_won, end_date=str(row.get("end_date", "")),
                volume=float(row.get("volume", 0) or 0),
            ))
        except (json.JSONDecodeError, ValueError, TypeError, IndexError):
            continue
    print(f"  {len(markets):,} markets parsed")
    return markets


def compute_trade_buckets(con, trades_dir, markets):
    """Single-pass DuckDB scan for 1c price buckets."""
    token_rows = []
    for m in markets:
        token_rows.append((m.up_token_id,))
        token_rows.append((m.down_token_id,))

    con.execute("CREATE OR REPLACE TABLE cmp_tokens (token_id VARCHAR)")
    con.executemany("INSERT INTO cmp_tokens VALUES (?)", token_rows)

    print(f"Scanning trades for {len(token_rows):,} tokens...")
    t0 = time.time()
    buckets_df = con.execute(f"""
        WITH priced_trades AS (
            SELECT
                CASE WHEN t.maker_asset_id = '0'
                     THEN t.taker_asset_id ELSE t.maker_asset_id
                END AS token_id,
                CASE WHEN t.maker_asset_id = '0'
                     THEN 1.0 * t.maker_amount / t.taker_amount
                     ELSE 1.0 * t.taker_amount / t.maker_amount
                END AS price,
                CASE WHEN t.maker_asset_id = '0'
                     THEN t.maker_amount / 1e6 ELSE t.taker_amount / 1e6
                END AS usdc_amount
            FROM '{trades_dir}/*.parquet' t
            INNER JOIN cmp_tokens ct ON ct.token_id = (
                CASE WHEN t.maker_asset_id = '0'
                     THEN t.taker_asset_id ELSE t.maker_asset_id END
            )
            WHERE t.taker_amount > 0 AND t.maker_amount > 0
        )
        SELECT token_id, FLOOR(price * 100) / 100.0 AS price_bucket,
               SUM(usdc_amount) AS sum_usd,
               SUM(usdc_amount * price) AS sum_usd_price,
               COUNT(*) AS trade_count
        FROM priced_trades
        WHERE price > 0 AND price < 1
        GROUP BY token_id, price_bucket
    """).df()
    print(f"  Done in {time.time()-t0:.1f}s — {len(buckets_df):,} bucket rows")
    return buckets_df


def assign_vwap(markets, buckets_df, lo, hi, range_key):
    """Compute VWAP per token in [lo, hi] and store under range_key."""
    token_buckets: dict[str, list] = {}
    for _, row in buckets_df.iterrows():
        tid = row["token_id"]
        token_buckets.setdefault(tid, []).append((
            float(row["price_bucket"]), float(row["sum_usd"]),
            float(row["sum_usd_price"]), int(row["trade_count"]),
        ))

    for m in markets:
        for token_id, prefix in [(m.up_token_id, "up"), (m.down_token_id, "down")]:
            buckets = token_buckets.get(token_id, [])
            relevant = [b for b in buckets if lo <= b[0] <= hi]
            total_usd = sum(b[1] for b in relevant)
            total_usd_price = sum(b[2] for b in relevant)
            vwap = total_usd_price / total_usd if total_usd > 0 else 0.0
            if prefix == "up":
                m.up_vwap[range_key] = vwap
                m.up_vol[range_key] = total_usd
            else:
                m.down_vwap[range_key] = vwap
                m.down_vol[range_key] = total_usd


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

@dataclass
class SimResult:
    label: str
    trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    total_fees: float = 0.0
    entry_prices: list = field(default_factory=list)
    pnls: list = field(default_factory=list)


def simulate_scenario(
    markets: list[Market],
    range_key: str,
    lo: float, hi: float,
    min_volume: float,
    fee_fn=None,  # None = 0 fees (maker), callable = taker
    label: str = "",
    size_usd: float = 10.0,
) -> SimResult:
    result = SimResult(label=label)
    sorted_markets = sorted(markets, key=lambda m: m.end_date)

    for m in sorted_markets:
        # Check each outcome
        up_vwap = m.up_vwap.get(range_key, 0)
        up_vol = m.up_vol.get(range_key, 0)
        dn_vwap = m.down_vwap.get(range_key, 0)
        dn_vol = m.down_vol.get(range_key, 0)

        up_ok = lo <= up_vwap <= hi and up_vol >= min_volume
        dn_ok = lo <= dn_vwap <= hi and dn_vol >= min_volume

        if not up_ok and not dn_ok:
            continue

        # Pick side with more volume
        if up_ok and dn_ok:
            if up_vol >= dn_vol:
                entry, won = up_vwap, m.up_won
            else:
                entry, won = dn_vwap, not m.up_won
        elif up_ok:
            entry, won = up_vwap, m.up_won
        else:
            entry, won = dn_vwap, not m.up_won

        # Apply taker fees
        fee_per_share = fee_fn(entry) if fee_fn else 0.0
        effective_price = entry + fee_per_share

        if effective_price >= 1.0:
            continue  # Can't buy at >= $1

        shares = size_usd / effective_price

        if won:
            pnl = shares * 1.0 - size_usd  # shares resolve to $1 each
            result.wins += 1
        else:
            pnl = -size_usd
            result.losses += 1

        result.trades += 1
        result.total_pnl += pnl
        result.total_fees += fee_per_share * shares
        result.entry_prices.append(entry)
        result.pnls.append(pnl)

    return result


def print_result(r: SimResult):
    if r.trades == 0:
        print(f"  {r.label:<35} | {'no trades':>50}")
        return
    wr = r.wins / r.trades * 100
    avg_entry = sum(r.entry_prices) / len(r.entry_prices)
    be_wr = avg_entry * 100
    edge = wr - be_wr
    avg_pnl = r.total_pnl / r.trades

    # Sharpe
    if len(r.pnls) > 1:
        mean_r = sum(r.pnls) / len(r.pnls)
        var_r = sum((p - mean_r)**2 for p in r.pnls) / (len(r.pnls) - 1)
        sharpe = (mean_r / math.sqrt(var_r)) * math.sqrt(len(r.pnls)) if var_r > 0 else 0
    else:
        sharpe = 0

    print(f"  {r.label:<35} | {r.trades:>6} trades | WR {wr:>5.1f}% | "
          f"avg entry {avg_entry:.3f} | BE {be_wr:.1f}% | edge {edge:>+5.1f}% | "
          f"PnL ${r.total_pnl:>+10,.2f} | fees ${r.total_fees:>8,.2f} | "
          f"avg ${avg_pnl:>+.3f}/trade | Sharpe {sharpe:.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    data_dir = Path("../prediction-market-analysis/data/polymarket")
    markets_dir = str(data_dir / "markets")
    trades_dir = str(data_dir / "trades")

    con = duckdb.connect()
    markets = load_markets(con, markets_dir)
    buckets_df = compute_trade_buckets(con, trades_dir, markets)

    # Define price ranges to test
    ranges = {
        # Maker ranges (bidding at these levels)
        "maker_75_85": (0.75, 0.85),
        "maker_75_90": (0.75, 0.90),
        "maker_80_90": (0.80, 0.90),
        # Taker ranges (buying at ask — shifted up by ~1-2c spread)
        "taker_76_86": (0.76, 0.86),
        "taker_76_90": (0.76, 0.90),
        "taker_78_88": (0.78, 0.88),
        "taker_80_90": (0.80, 0.90),
        "taker_75_90": (0.75, 0.90),
        "taker_75_95": (0.75, 0.95),
    }

    print("\nComputing VWAP for all ranges...")
    for key, (lo, hi) in ranges.items():
        assign_vwap(markets, buckets_df, lo, hi, key)

    print()
    print("=" * 160)
    print("MAKER vs TAKER COMPARISON — Crypto TD Strategy")
    print("=" * 160)
    print(f"  Markets: {len(markets):,} | Size: $10/trade | Min volume: $5")
    print()

    # --- Maker scenarios (0 fees) ---
    print("  MAKER (0 fees, GTC post_only)")
    print("  " + "-" * 155)
    for key in ["maker_75_85", "maker_75_90", "maker_80_90"]:
        lo, hi = ranges[key]
        r = simulate_scenario(markets, key, lo, hi, 5.0, fee_fn=None,
                              label=f"Maker [{lo:.2f}-{hi:.2f}]")
        print_result(r)

    print()

    # --- Taker scenarios (with fees) ---
    print("  TAKER (with taker fees, FOK at ask)")
    print("  " + "-" * 155)
    for key in ["taker_76_86", "taker_76_90", "taker_78_88", "taker_80_90",
                "taker_75_90", "taker_75_95"]:
        lo, hi = ranges[key]
        r = simulate_scenario(markets, key, lo, hi, 5.0,
                              fee_fn=crypto_taker_fee,
                              label=f"Taker [{lo:.2f}-{hi:.2f}] +fees")
        print_result(r)

    print()

    # --- Same range comparison (apples to apples) ---
    print("  DIRECT COMPARISON — Same range [0.75-0.90], maker vs taker fees")
    print("  " + "-" * 155)
    # Maker at [0.75, 0.90]
    r_maker = simulate_scenario(markets, "maker_75_90", 0.75, 0.90, 5.0,
                                fee_fn=None, label="Maker [0.75-0.90] 0 fees")
    print_result(r_maker)
    # Taker at same range with fees
    r_taker = simulate_scenario(markets, "taker_75_90", 0.75, 0.90, 5.0,
                                fee_fn=crypto_taker_fee,
                                label="Taker [0.75-0.90] +fees")
    print_result(r_taker)

    print()

    # --- Fee sensitivity: what if taker fees were lower? ---
    print("  FEE SENSITIVITY — Taker [0.75-0.90] with various fee levels")
    print("  " + "-" * 155)
    for fee_mult, fee_label in [(0.0, "0% (hypothetical)"),
                                 (0.25, "25% of current"),
                                 (0.50, "50% of current"),
                                 (0.75, "75% of current"),
                                 (1.0, "100% (current)")]:
        fee_fn = (lambda p, m=fee_mult: crypto_taker_fee(p) * m) if fee_mult > 0 else None
        r = simulate_scenario(markets, "taker_75_90", 0.75, 0.90, 5.0,
                              fee_fn=fee_fn,
                              label=f"Taker fees={fee_label}")
        print_result(r)

    print()

    # --- Volume analysis: how many markets have activity in each range ---
    print("  FILL RATE ANALYSIS — Markets with >=$5 volume in range")
    print("  " + "-" * 80)
    for key in sorted(ranges.keys()):
        lo, hi = ranges[key]
        count = 0
        for m in markets:
            up_ok = lo <= m.up_vwap.get(key, 0) <= hi and m.up_vol.get(key, 0) >= 5
            dn_ok = lo <= m.down_vwap.get(key, 0) <= hi and m.down_vol.get(key, 0) >= 5
            if up_ok or dn_ok:
                count += 1
        pct = count / len(markets) * 100
        print(f"  {key:<20} [{lo:.2f}-{hi:.2f}]: {count:>6,} / {len(markets):,} ({pct:.1f}%)")

    print()

    # --- Per price-cent breakdown: maker vs taker edge ---
    print("  PER-CENT BREAKDOWN — Edge by entry price (maker vs taker)")
    print(f"  {'Cent':>5} | {'Maker Trades':>12} | {'Maker WR':>8} | {'Maker Edge':>10} | "
          f"{'Taker Trades':>12} | {'Taker WR':>8} | {'Taker Edge':>10} | {'Fee @cent':>9}")
    print(f"  {'-'*5}-+-{'-'*12}-+-{'-'*8}-+-{'-'*10}-+-{'-'*12}-+-{'-'*8}-+-{'-'*10}-+-{'-'*9}")

    r_m = simulate_scenario(markets, "maker_75_90", 0.75, 0.90, 5.0,
                            fee_fn=None, label="maker")
    r_t = simulate_scenario(markets, "taker_75_90", 0.75, 0.90, 5.0,
                            fee_fn=crypto_taker_fee, label="taker")

    for cent in range(75, 91):
        lo_c = cent / 100.0
        hi_c = (cent + 1) / 100.0

        # Maker trades in this cent
        m_idx = [i for i, p in enumerate(r_m.entry_prices) if lo_c <= p < hi_c]
        m_n = len(m_idx)
        m_wins = sum(1 for i in m_idx if r_m.pnls[i] > 0) if m_n else 0
        m_wr = m_wins / m_n * 100 if m_n else 0
        m_edge = m_wr - (lo_c + 0.005) * 100 if m_n else 0

        # Taker trades in this cent
        t_idx = [i for i, p in enumerate(r_t.entry_prices) if lo_c <= p < hi_c]
        t_n = len(t_idx)
        t_wins = sum(1 for i in t_idx if r_t.pnls[i] > 0) if t_n else 0
        t_wr = t_wins / t_n * 100 if t_n else 0
        fee_at_cent = crypto_taker_fee(lo_c + 0.005)
        effective = lo_c + 0.005 + fee_at_cent
        t_edge = t_wr - effective * 100 if t_n else 0

        if m_n > 0 or t_n > 0:
            print(f"  {lo_c:.2f}  | {m_n:>12,} | {m_wr:>7.1f}% | {m_edge:>+9.1f}% | "
                  f"{t_n:>12,} | {t_wr:>7.1f}% | {t_edge:>+9.1f}% | {fee_at_cent:.4f}")

    print()
    print("=" * 160)
    con.close()


if __name__ == "__main__":
    main()
