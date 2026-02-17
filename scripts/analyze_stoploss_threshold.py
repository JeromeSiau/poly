#!/usr/bin/env python3
"""Analyze stoploss threshold effectiveness for crypto TD maker strategy.

QUESTION: When the trade price reaches >= 0.75 (strong favourite) and then
drops to <= some threshold, how often does the position ultimately WIN vs LOSE?

This tells us whether setting a stoploss_exit would save money (selling losers
early) or cost money (selling winners prematurely).

METHODOLOGY:
    For each resolved 15-min crypto binary market:
    1. Get all on-chain trades for each token, ordered by block number (time).
    2. Track the price trajectory using individual trade prices.
    3. Find tokens where:
       a) Some trade occurred at price in [0.75, 0.85] (our entry zone)
       b) A LATER trade occurred at price <= threshold (price crashed)
    4. Check if the token ultimately resolved to 1 (win) or 0 (loss).
    5. Compare EV of holding through the crash vs selling at the threshold.

    We use trade prices as a proxy for bid levels. While not perfect (trades
    happen at various prices), a trade at 0.40 means someone was willing to
    buy at that level, so the bid was at least 0.40.

DATASET: 41K+ resolved BTC/ETH/SOL/XRP 15-min updown markets (Oct 2025 - Feb 2026)

Usage:
    ./run scripts/analyze_stoploss_threshold.py
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import duckdb


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MarketToken:
    """A single token (one side) of a resolved binary market."""
    market_id: str
    slug: str
    symbol: str
    token_id: str
    side: str          # "Up" or "Down"
    resolved_to_1: bool  # Did this token resolve to 1 (win)?


# ---------------------------------------------------------------------------
# Load resolved markets
# ---------------------------------------------------------------------------

def load_market_tokens(
    con: duckdb.DuckDBPyConnection, markets_dir: str
) -> list[MarketToken]:
    """Load all resolved 15-min crypto binary market tokens."""
    print("Loading resolved crypto 15-min markets...")
    t0 = time.time()

    df = con.execute(f"""
        SELECT id, slug, outcome_prices, clob_token_ids
        FROM '{markets_dir}/*.parquet'
        WHERE slug LIKE '%updown-15m%' AND closed = true
    """).df()

    print(f"  {len(df):,} resolved markets loaded in {time.time()-t0:.1f}s")

    tokens: list[MarketToken] = []
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
                symbol = "BTC"
            elif slug.startswith("eth"):
                symbol = "ETH"
            elif slug.startswith("sol"):
                symbol = "SOL"
            elif slug.startswith("xrp"):
                symbol = "XRP"
            else:
                continue

            tokens.append(MarketToken(
                market_id=str(row["id"]), slug=slug, symbol=symbol,
                token_id=token_ids[0], side="Up", resolved_to_1=up_won,
            ))
            tokens.append(MarketToken(
                market_id=str(row["id"]), slug=slug, symbol=symbol,
                token_id=token_ids[1], side="Down", resolved_to_1=not up_won,
            ))
        except (json.JSONDecodeError, ValueError, TypeError, IndexError):
            continue

    n_markets = len(tokens) // 2
    print(f"  {n_markets:,} markets -> {len(tokens):,} tokens")
    return tokens


# ---------------------------------------------------------------------------
# Single-scan analysis: entry zone trades + post-entry trough
# ---------------------------------------------------------------------------

def scan_trades(
    con: duckdb.DuckDBPyConnection,
    trades_dir: str,
    tokens: list[MarketToken],
    entry_lo: float,
    entry_hi: float,
) -> tuple[dict, dict]:
    """Single scan of trade data to get:
    1. All tokens with trades in entry zone (for baseline win rate)
    2. For those tokens, the minimum price after the first entry-zone trade

    Returns:
        (baseline_tokens, trajectory_data)
        baseline_tokens: {token_id: entry_vwap}
        trajectory_data: {token_id: (entry_vwap, min_price_after_entry)}
    """
    if not tokens:
        return {}, {}

    # Register token IDs
    token_rows = [(t.token_id,) for t in tokens]
    con.execute("CREATE OR REPLACE TABLE sl_tokens (token_id VARCHAR)")
    con.executemany("INSERT INTO sl_tokens VALUES (?)", token_rows)
    print(f"  Registered {len(token_rows):,} token IDs in DuckDB")

    print(f"\nScanning trades for entry range [{entry_lo}, {entry_hi}]...")
    print("  Scanning ~45GB of trade data -- this may take several minutes...")
    t0 = time.time()

    # Single DuckDB query that computes:
    # - Which tokens had trades in entry zone [entry_lo, entry_hi]
    # - VWAP in entry zone
    # - Minimum trade price AFTER the first entry-zone trade
    result_df = con.execute(f"""
        WITH priced_trades AS (
            SELECT
                CASE WHEN t.maker_asset_id = '0'
                     THEN t.taker_asset_id
                     ELSE t.maker_asset_id
                END AS token_id,
                t.block_number,
                CASE WHEN t.maker_asset_id = '0'
                     THEN 1.0 * t.maker_amount / t.taker_amount
                     ELSE 1.0 * t.taker_amount / t.maker_amount
                END AS price,
                CASE WHEN t.maker_asset_id = '0'
                     THEN t.maker_amount / 1e6
                     ELSE t.taker_amount / 1e6
                END AS usdc_amount
            FROM '{trades_dir}/*.parquet' t
            INNER JOIN sl_tokens st
                ON st.token_id = (
                    CASE WHEN t.maker_asset_id = '0'
                         THEN t.taker_asset_id
                         ELSE t.maker_asset_id
                    END
                )
            WHERE t.taker_amount > 0 AND t.maker_amount > 0
        ),
        valid_trades AS (
            SELECT token_id, block_number, price, usdc_amount
            FROM priced_trades
            WHERE price > 0 AND price < 1
        ),
        -- Tokens with trades in the entry zone
        entry_tokens AS (
            SELECT
                token_id,
                MIN(block_number) AS first_entry_block,
                SUM(usdc_amount * price) / SUM(usdc_amount) AS entry_vwap,
                SUM(usdc_amount) AS entry_volume_usd
            FROM valid_trades
            WHERE price >= {entry_lo} AND price <= {entry_hi}
            GROUP BY token_id
            HAVING SUM(usdc_amount) >= 1.0
        ),
        -- Find min price of ALL trades (not just entry zone) AFTER the first entry block
        post_entry AS (
            SELECT
                et.token_id,
                et.entry_vwap,
                MIN(vt.price) AS min_price_after_entry
            FROM entry_tokens et
            JOIN valid_trades vt ON vt.token_id = et.token_id
            WHERE vt.block_number > et.first_entry_block
            GROUP BY et.token_id, et.entry_vwap
        )
        SELECT
            et.token_id,
            et.entry_vwap,
            pe.min_price_after_entry
        FROM entry_tokens et
        LEFT JOIN post_entry pe ON pe.token_id = et.token_id
    """).df()

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s -- {len(result_df):,} tokens had entry-zone trades")

    baseline: dict[str, float] = {}
    trajectory: dict[str, tuple[float, float]] = {}

    for _, row in result_df.iterrows():
        tid = row["token_id"]
        entry_vwap = float(row["entry_vwap"])
        baseline[tid] = entry_vwap

        min_after = row["min_price_after_entry"]
        if min_after is not None:
            trajectory[tid] = (entry_vwap, float(min_after))

    return baseline, trajectory


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze(
    tokens: list[MarketToken],
    baseline: dict[str, float],
    trajectory: dict[str, tuple[float, float]],
    thresholds: list[float],
    entry_lo: float,
    entry_hi: float,
    size_usd: float,
) -> None:
    """Analyze stoploss effectiveness and print results."""
    token_map = {t.token_id: t for t in tokens}

    # Baseline stats
    total_entry = 0
    total_wins = 0
    total_losses = 0
    for tid in baseline:
        token = token_map.get(tid)
        if not token:
            continue
        total_entry += 1
        if token.resolved_to_1:
            total_wins += 1
        else:
            total_losses += 1

    # Per-threshold stats
    results: dict[float, dict] = {}
    for threshold in thresholds:
        wins = 0
        losses = 0
        entry_prices_win: list[float] = []
        entry_prices_lose: list[float] = []

        for tid, (entry_vwap, min_after) in trajectory.items():
            if min_after > threshold:
                continue
            token = token_map.get(tid)
            if not token:
                continue
            if token.resolved_to_1:
                wins += 1
                entry_prices_win.append(entry_vwap)
            else:
                losses += 1
                entry_prices_lose.append(entry_vwap)

        results[threshold] = {
            "total_dropped": wins + losses,
            "wins": wins,
            "losses": losses,
            "entry_prices_win": entry_prices_win,
            "entry_prices_lose": entry_prices_lose,
        }

    # Print results
    print()
    print("=" * 95)
    print("STOPLOSS THRESHOLD ANALYSIS -- Crypto TD Maker (15-min updown markets)")
    print("=" * 95)
    print()
    print(f"  Entry zone: [{entry_lo}, {entry_hi}]  (TD maker bid range)")
    print(f"  Position size: ${size_usd:.0f}")
    print()
    print(f"  BASELINE (all tokens with trades in entry zone, >= $1 volume):")
    if total_entry > 0:
        base_wr = total_wins / total_entry * 100
        avg_entry = (entry_lo + entry_hi) / 2
        ev_hold_base = (total_wins / total_entry) * (1/avg_entry - 1) * size_usd - \
                       (total_losses / total_entry) * size_usd
        print(f"    Total tokens: {total_entry:,}")
        print(f"    Wins: {total_wins:,}  |  Losses: {total_losses:,}  |  Win rate: {base_wr:.1f}%")
        print(f"    EV per trade (hold to resolution): ${ev_hold_base:+.4f} (approx at avg entry)")
    print()
    print("  QUESTION: When the price drops from entry zone to <= threshold,")
    print("  should we SELL (stoploss) or HOLD to resolution?")
    print()

    # Main table
    header = (
        f"  {'Threshold':>9} | {'Dropped':>8} | "
        f"{'Still Won':>9} | {'Lost':>8} | "
        f"{'Win%':>6} | "
        f"{'EV Hold':>10} | {'EV Sell':>10} | "
        f"{'Diff':>10} | "
        f"{'Verdict':>12}"
    )
    divider = (
        f"  {'-'*9}-+-{'-'*8}-+-"
        f"{'-'*9}-+-{'-'*8}-+-"
        f"{'-'*6}-+-"
        f"{'-'*10}-+-{'-'*10}-+-"
        f"{'-'*10}-+-"
        f"{'-'*12}"
    )
    print(header)
    print(divider)

    for threshold in sorted(results.keys()):
        r = results[threshold]
        total = r["total_dropped"]
        wins = r["wins"]
        losses = r["losses"]

        if total == 0:
            print(f"  {threshold:>9.2f} | {'0':>8} | "
                  f"{'--':>9} | {'--':>8} | {'--':>6} | "
                  f"{'--':>10} | {'--':>10} | {'--':>10} | {'--':>12}")
            continue

        win_rate = wins / total * 100

        all_entries = r["entry_prices_win"] + r["entry_prices_lose"]
        avg_entry = sum(all_entries) / len(all_entries)

        wr = wins / total

        # EV HOLD: PnL from original entry, holding to resolution
        ev_hold = wr * (1.0 / avg_entry - 1.0) * size_usd + (1 - wr) * (-size_usd)

        # EV SELL: sell at threshold price
        # We hold (size/entry) tokens. Selling at threshold:
        # revenue = threshold * (size/entry), PnL = threshold/entry * size - size
        ev_sell = (threshold / avg_entry - 1.0) * size_usd

        diff = ev_hold - ev_sell
        verdict = "HOLD better" if diff > 0 else "SELL better"

        print(
            f"  {threshold:>9.2f} | "
            f"{total:>8,} | "
            f"{wins:>9,} | "
            f"{losses:>8,} | "
            f"{win_rate:>5.1f}% | "
            f"${ev_hold:>+9.4f} | "
            f"${ev_sell:>+9.4f} | "
            f"${diff:>+9.4f} | "
            f"{verdict:>12}"
        )

    # Separator
    print()
    print("-" * 95)
    print()

    # Simplified decision rule
    print("  SIMPLIFIED DECISION RULE:")
    print()
    print("  From the moment the price drops to the threshold, the decision is simple:")
    print("  - HOLD if you expect the token to win more than 'threshold' fraction of the time")
    print("  - SELL if you expect it to win less than 'threshold' fraction of the time")
    print()
    print("  Because: EV(hold) = win_rate * (1/entry), EV(sell) = threshold * (1/entry)")
    print("  Hold > Sell <==> win_rate > threshold")
    print()

    for threshold in sorted(results.keys()):
        r = results[threshold]
        total = r["total_dropped"]
        wins = r["wins"]
        if total > 0:
            wr = wins / total
            print(f"    threshold={threshold:.2f}: win_rate={wr:.3f} "
                  f"{'>' if wr > threshold else '<'} threshold={threshold:.2f}  "
                  f"--> {'HOLD' if wr > threshold else 'SELL'}")

    # Loss capture analysis
    print()
    print("-" * 95)
    print()
    print("  LOSS CAPTURE ANALYSIS:")
    print("  How many actual losses does each stoploss catch? And at what false-positive cost?")
    print()
    print(f"  {'Threshold':>9} | {'Losses caught':>14} | {'% of all losses':>15} | "
          f"{'False stops':>12} | {'False stop %':>12}")
    print(f"  {'-'*9}-+-{'-'*14}-+-{'-'*15}-+-{'-'*12}-+-{'-'*12}")
    for threshold in sorted(results.keys()):
        r = results[threshold]
        total = r["total_dropped"]
        losses = r["losses"]
        wins = r["wins"]
        pct_losses = losses / total_losses * 100 if total_losses > 0 else 0
        pct_false = wins / total * 100 if total > 0 else 0
        print(
            f"  {threshold:>9.2f} | "
            f"{losses:>14,} | "
            f"{pct_losses:>14.1f}% | "
            f"{wins:>12,} | "
            f"{pct_false:>11.1f}%"
        )

    print()
    print("  'Losses caught': losses where the price dropped to threshold before resolution")
    print("  '% of all losses': what fraction of total losses this stoploss catches")
    print("  'False stops': winners incorrectly sold (token dropped then recovered)")
    print("  'False stop %': what fraction of stoploss triggers are false positives")

    # Net impact analysis
    print()
    print("-" * 95)
    print()
    print("  NET PORTFOLIO IMPACT (per $10 position):")
    print("  If we had applied the stoploss to ALL positions, what is the net effect?")
    print()
    print(f"  {'Threshold':>9} | {'Saved on losses':>15} | {'Lost on false':>14} | "
          f"{'Net impact':>11} | {'Per trade':>10}")
    print(f"  {'-'*9}-+-{'-'*15}-+-{'-'*14}-+-{'-'*11}-+-{'-'*10}")

    for threshold in sorted(results.keys()):
        r = results[threshold]
        wins = r["wins"]
        losses = r["losses"]
        total = r["total_dropped"]
        if total == 0:
            continue

        all_entries = r["entry_prices_win"] + r["entry_prices_lose"]
        avg_entry = sum(all_entries) / len(all_entries)

        # Money SAVED on true losses: instead of losing full $size, we lose (1 - threshold/entry)*size
        # saved per true loss = size - (1 - threshold/entry)*size = threshold/entry * size
        # Actually: without stoploss, loss = -size. With stoploss, loss = (threshold/entry - 1)*size
        # saved = (-size) - (threshold/entry - 1)*size = -(1 - threshold/entry)*size ... wait
        # Without stoploss on a loss: PnL = -size (token goes to 0)
        # With stoploss on a loss: PnL = (threshold/entry - 1)*size (sell at threshold, still a loss but smaller)
        # Improvement = with - without = (threshold/entry - 1)*size - (-size) = threshold/entry * size
        improvement_per_loss = threshold / avg_entry * size_usd
        total_saved = improvement_per_loss * losses

        # Money LOST on false stops (winners incorrectly sold):
        # Without stoploss on a win: PnL = (1/entry - 1)*size
        # With stoploss on a win: PnL = (threshold/entry - 1)*size
        # Harm = with - without = (threshold/entry - 1)*size - (1/entry - 1)*size = (threshold - 1)/entry * size
        harm_per_false = (1.0 / avg_entry - threshold / avg_entry) * size_usd
        total_lost = harm_per_false * wins

        net = total_saved - total_lost
        per_trade = net / total_entry if total_entry > 0 else 0

        print(
            f"  {threshold:>9.2f} | "
            f"${total_saved:>14,.2f} | "
            f"-${total_lost:>13,.2f} | "
            f"${net:>+10,.2f} | "
            f"${per_trade:>+9.4f}"
        )

    print()
    print("  'Saved on losses': money saved by selling losers early instead of losing 100%")
    print("  'Lost on false': profit foregone by selling winners prematurely")
    print("  'Net impact': total portfolio impact across all tokens")
    print("  'Per trade': net impact spread across all entry-zone trades (not just dropped ones)")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    data_dir = Path("../prediction-market-analysis/data/polymarket")
    markets_dir = str(data_dir / "markets")
    trades_dir = str(data_dir / "trades")

    if not (data_dir / "markets").exists():
        print(f"Error: markets directory not found at {markets_dir}")
        sys.exit(1)
    if not (data_dir / "trades").exists():
        print(f"Error: trades directory not found at {trades_dir}")
        sys.exit(1)

    con = duckdb.connect()

    # Step 1: Load all market tokens
    tokens = load_market_tokens(con, markets_dir)
    if not tokens:
        print("No crypto-minute markets found.")
        sys.exit(0)

    # Parameters
    entry_lo = 0.75
    entry_hi = 0.85
    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    size_usd = 10.0

    # Step 2: Single scan of all trades
    baseline, trajectory = scan_trades(con, trades_dir, tokens, entry_lo, entry_hi)

    # Step 3: Analyze and print
    analyze(tokens, baseline, trajectory, thresholds, entry_lo, entry_hi, size_usd)

    con.close()


if __name__ == "__main__":
    main()
