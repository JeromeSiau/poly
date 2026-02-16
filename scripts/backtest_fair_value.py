#!/usr/bin/env python3
"""Backtest fair value guards on historical crypto TD maker data.

Measures the impact of:
1. Entry guard: block trades where entry_price > fair_value + margin
2. Stop loss override: hold position when fair_value > stoploss_exit + margin
   despite bid dropping below stoploss_exit

Uses Binance 1-min klines to compute the underlying price at each point
during the 15-min slot, then applies the fair value table.

Usage:
    ./run scripts/backtest_fair_value.py
    ./run scripts/backtest_fair_value.py --symbol BTC
    ./run scripts/backtest_fair_value.py --entry-minutes 5
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import httpx

from src.utils.fair_value import estimate_fair_value, _MOVE_EDGES

# ---------------------------------------------------------------------------
# Binance kline loading
# ---------------------------------------------------------------------------

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
KLINE_LIMIT = 1000

# Map Polymarket symbol prefix to Binance symbol
SYMBOL_MAP = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT", "XRP": "XRPUSDT"}

@dataclass
class Kline:
    open_time_ms: int
    close: float


def fetch_klines(symbol: str, start_ms: int, end_ms: int) -> list[Kline]:
    """Fetch 1-min klines from Binance REST API."""
    klines: list[Kline] = []
    cursor = start_ms
    with httpx.Client(timeout=30) as client:
        while cursor < end_ms:
            resp = client.get(BINANCE_KLINES_URL, params={
                "symbol": symbol,
                "interval": "1m",
                "startTime": cursor,
                "endTime": end_ms,
                "limit": KLINE_LIMIT,
            })
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            for k in batch:
                klines.append(Kline(open_time_ms=int(k[0]), close=float(k[4])))
            cursor = int(batch[-1][0]) + 60_000
            time.sleep(0.15)
    return klines


def build_kline_index(klines: list[Kline]) -> dict[int, float]:
    """Build minute-level index: open_time_ms -> close price."""
    return {k.open_time_ms: k.close for k in klines}


# ---------------------------------------------------------------------------
# Market loading (reused from backtest_crypto_td_maker)
# ---------------------------------------------------------------------------

@dataclass
class Market:
    market_id: str
    slug: str
    symbol: str       # "BTC", "ETH", etc.
    up_token_id: str
    down_token_id: str
    up_won: bool
    slot_ts: int      # slot start unix timestamp
    # VWAP in bid range (filled later)
    up_vwap: float = 0.0
    up_vol_usd: float = 0.0
    down_vwap: float = 0.0
    down_vol_usd: float = 0.0


def parse_slot_ts(slug: str) -> int | None:
    """Extract slot start timestamp from slug like 'btc-updown-15m-1771079400'."""
    parts = slug.split("-")
    if len(parts) >= 4 and parts[-1].isdigit():
        return int(parts[-1])
    return None


def load_markets(con: duckdb.DuckDBPyConnection, markets_dir: str,
                 symbol_filter: str | None) -> list[Market]:
    """Load resolved 15-min crypto binary markets with slot timestamps."""
    slug_filter = ""
    if symbol_filter:
        slug_filter = f"AND slug LIKE '{symbol_filter.lower()}%'"

    df = con.execute(f"""
        SELECT id, slug, outcome_prices, clob_token_ids, volume
        FROM '{markets_dir}/*.parquet'
        WHERE slug LIKE '%updown-15m%' AND closed = true
        {slug_filter}
    """).df()

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
            slot_ts = parse_slot_ts(slug)
            if slot_ts is None:
                continue

            sym = slug.split("-")[0].upper()
            if sym not in SYMBOL_MAP:
                continue

            markets.append(Market(
                market_id=str(row["id"]),
                slug=slug,
                symbol=sym,
                up_token_id=token_ids[0],
                down_token_id=token_ids[1],
                up_won=up_won,
                slot_ts=slot_ts,
            ))
        except (json.JSONDecodeError, ValueError, TypeError, IndexError):
            continue

    return markets


def compute_vwap(con: duckdb.DuckDBPyConnection, trades_dir: str,
                 markets: list[Market], target_bid: float, max_bid: float) -> None:
    """Compute per-token VWAP in bid range (same as existing backtest)."""
    token_rows = []
    for m in markets:
        token_rows.append((m.up_token_id,))
        token_rows.append((m.down_token_id,))

    con.execute("CREATE OR REPLACE TABLE fv_tokens (token_id VARCHAR)")
    con.executemany("INSERT INTO fv_tokens VALUES (?)", token_rows)

    print(f"  Scanning trades for VWAP in [{target_bid}, {max_bid}]...")
    t0 = time.time()

    buckets_df = con.execute(f"""
        WITH priced_trades AS (
            SELECT
                CASE WHEN t.maker_asset_id = '0'
                     THEN t.taker_asset_id ELSE t.maker_asset_id END AS token_id,
                CASE WHEN t.maker_asset_id = '0'
                     THEN 1.0 * t.maker_amount / t.taker_amount
                     ELSE 1.0 * t.taker_amount / t.maker_amount END AS price,
                CASE WHEN t.maker_asset_id = '0'
                     THEN t.maker_amount / 1e6 ELSE t.taker_amount / 1e6 END AS usdc_amount
            FROM '{trades_dir}/*.parquet' t
            INNER JOIN fv_tokens ct ON ct.token_id = (
                CASE WHEN t.maker_asset_id = '0'
                     THEN t.taker_asset_id ELSE t.maker_asset_id END)
            WHERE t.taker_amount > 0 AND t.maker_amount > 0
        )
        SELECT token_id,
               SUM(usdc_amount) AS sum_usd,
               SUM(usdc_amount * price) AS sum_usd_price
        FROM priced_trades
        WHERE price >= {target_bid} AND price <= {max_bid}
        GROUP BY token_id
    """).df()

    print(f"  Done in {time.time()-t0:.1f}s")

    lookup = {}
    for _, row in buckets_df.iterrows():
        tid = row["token_id"]
        usd = float(row["sum_usd"])
        usd_price = float(row["sum_usd_price"])
        lookup[tid] = (usd_price / usd if usd > 0 else 0.0, usd)

    for m in markets:
        v, vol = lookup.get(m.up_token_id, (0.0, 0.0))
        m.up_vwap, m.up_vol_usd = v, vol
        v, vol = lookup.get(m.down_token_id, (0.0, 0.0))
        m.down_vwap, m.down_vol_usd = v, vol


# ---------------------------------------------------------------------------
# Fair value analysis
# ---------------------------------------------------------------------------

@dataclass
class TradeAnalysis:
    slug: str
    symbol: str
    side: str           # "Up" or "Down"
    entry_price: float
    won: bool
    pnl: float
    fair_value: float | None  # at entry time
    overpay: float | None     # entry_price - fair_value


def analyze_trades(
    markets: list[Market],
    kline_indexes: dict[str, dict[int, float]],  # symbol -> {open_time_ms -> close}
    target_bid: float,
    max_bid: float,
    min_volume: float,
    entry_minutes: float,
    size_usd: float,
) -> list[TradeAnalysis]:
    """For each tradeable market, compute fair value at entry time."""
    results: list[TradeAnalysis] = []

    for m in markets:
        # Check tradeability
        up_ok = target_bid <= m.up_vwap <= max_bid and m.up_vol_usd >= min_volume
        dn_ok = target_bid <= m.down_vwap <= max_bid and m.down_vol_usd >= min_volume

        if not up_ok and not dn_ok:
            continue

        # Pick side (same logic as existing backtest)
        if up_ok and dn_ok:
            if m.up_vol_usd >= m.down_vol_usd:
                entry_price, side, won = m.up_vwap, "Up", m.up_won
            else:
                entry_price, side, won = m.down_vwap, "Down", not m.up_won
        elif up_ok:
            entry_price, side, won = m.up_vwap, "Up", m.up_won
        else:
            entry_price, side, won = m.down_vwap, "Down", not m.up_won

        pnl = size_usd * (1.0 / entry_price - 1.0) if won else -size_usd

        # Compute fair value at entry_minutes into the slot
        kline_idx = kline_indexes.get(m.symbol)
        fair_value = None
        overpay = None
        if kline_idx:
            ref_ms = m.slot_ts * 1000
            entry_ms = ref_ms + int(entry_minutes * 60 * 1000)
            # Snap to nearest minute
            entry_ms = (entry_ms // 60_000) * 60_000

            ref_price = kline_idx.get(ref_ms)
            entry_price_underlying = kline_idx.get(entry_ms)

            if ref_price and entry_price_underlying:
                dir_move = (entry_price_underlying - ref_price) / ref_price * 100
                if side == "Down":
                    dir_move = -dir_move
                minutes_remaining = 15.0 - entry_minutes
                fair_value = estimate_fair_value(dir_move, minutes_remaining)
                overpay = entry_price - fair_value

        results.append(TradeAnalysis(
            slug=m.slug, symbol=m.symbol, side=side,
            entry_price=entry_price, won=won, pnl=pnl,
            fair_value=fair_value, overpay=overpay,
        ))

    return results


# ---------------------------------------------------------------------------
# Stop loss analysis
# ---------------------------------------------------------------------------

@dataclass
class StopLossCase:
    slug: str
    symbol: str
    side: str
    entry_price: float
    won: bool
    min_fair_value: float   # lowest fair value during slot (after entry)
    max_fair_value: float   # highest fair value during slot
    would_stoploss: bool    # did fair value ever drop below stoploss_exit?
    override_correct: bool  # if we overrode, was it the right call?


def analyze_stop_losses(
    markets: list[Market],
    kline_indexes: dict[str, dict[int, float]],
    target_bid: float,
    max_bid: float,
    min_volume: float,
    entry_minutes: float,
    stoploss_exit: float,
    size_usd: float,
) -> list[StopLossCase]:
    """Analyze which markets had mid-slot fair value dips.

    Simulates: if fair value drops below stoploss_exit during the slot,
    would stopping out have been correct (market lost) or a false trigger
    (market won)?
    """
    results: list[StopLossCase] = []

    for m in markets:
        up_ok = target_bid <= m.up_vwap <= max_bid and m.up_vol_usd >= min_volume
        dn_ok = target_bid <= m.down_vwap <= max_bid and m.down_vol_usd >= min_volume
        if not up_ok and not dn_ok:
            continue

        if up_ok and dn_ok:
            if m.up_vol_usd >= m.down_vol_usd:
                entry_price, side, won = m.up_vwap, "Up", m.up_won
            else:
                entry_price, side, won = m.down_vwap, "Down", not m.up_won
        elif up_ok:
            entry_price, side, won = m.up_vwap, "Up", m.up_won
        else:
            entry_price, side, won = m.down_vwap, "Down", not m.up_won

        kline_idx = kline_indexes.get(m.symbol)
        if not kline_idx:
            continue

        ref_ms = m.slot_ts * 1000
        ref_price = kline_idx.get(ref_ms)
        if not ref_price:
            continue

        # Compute fair value at each minute after entry
        entry_min_idx = max(1, int(entry_minutes))
        fair_values: list[float] = []
        for t in range(entry_min_idx, 15):
            t_ms = ref_ms + t * 60_000
            price_at_t = kline_idx.get(t_ms)
            if price_at_t is None:
                continue
            dir_move = (price_at_t - ref_price) / ref_price * 100
            if side == "Down":
                dir_move = -dir_move
            fv = estimate_fair_value(dir_move, 15.0 - t)
            fair_values.append(fv)

        if not fair_values:
            continue

        min_fv = min(fair_values)
        max_fv = max(fair_values)
        would_stop = min_fv < stoploss_exit

        results.append(StopLossCase(
            slug=m.slug, symbol=m.symbol, side=side,
            entry_price=entry_price, won=won,
            min_fair_value=min_fv, max_fair_value=max_fv,
            would_stoploss=would_stop,
            override_correct=won if would_stop else True,
        ))

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report_entry_sweep(trades: list[TradeAnalysis], size_usd: float) -> None:
    """Sweep entry_fair_margin and report impact."""
    # Only trades with fair value data
    with_fv = [t for t in trades if t.fair_value is not None]
    total = len(with_fv)
    if total == 0:
        print("  No trades with fair value data.")
        return

    baseline_wins = sum(1 for t in with_fv if t.won)
    baseline_pnl = sum(t.pnl for t in with_fv)
    baseline_wr = baseline_wins / total * 100

    print(f"\n  ENTRY FAIR VALUE GUARD — PARAMETER SWEEP")
    print(f"  Baseline: {total:,} trades, {baseline_wr:.1f}% WR, ${baseline_pnl:+,.2f} PnL")
    print()

    margins = [0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    header = (f"  {'Margin':>6} | {'Trades':>6} | {'Blocked':>7} | "
              f"{'Win%':>5} | {'PnL':>12} | {'PnL Δ':>10} | "
              f"{'Blk Wins':>8} | {'Blk Loss':>8} | {'Net Save':>10}")
    divider = (f"  {'-'*6}-+-{'-'*6}-+-{'-'*7}-+-"
               f"{'-'*5}-+-{'-'*12}-+-{'-'*10}-+-"
               f"{'-'*8}-+-{'-'*8}-+-{'-'*10}")
    print(header)
    print(divider)

    for margin in margins:
        if margin == 0:
            # No filter
            kept = with_fv
            blocked = []
        else:
            kept = [t for t in with_fv if t.overpay is not None and t.overpay <= margin]
            blocked = [t for t in with_fv if t.overpay is not None and t.overpay > margin]

        n_kept = len(kept)
        n_blocked = len(blocked)
        kept_wins = sum(1 for t in kept if t.won)
        kept_pnl = sum(t.pnl for t in kept)
        wr = kept_wins / n_kept * 100 if n_kept > 0 else 0

        blocked_wins = sum(1 for t in blocked if t.won)
        blocked_losses = n_blocked - blocked_wins
        # Net save: blocked losses * size_usd - blocked wins * avg_win_pnl
        blocked_win_pnl = sum(t.pnl for t in blocked if t.won)
        blocked_loss_pnl = sum(t.pnl for t in blocked if not t.won)
        net_save = -blocked_loss_pnl - blocked_win_pnl  # positive = net benefit

        pnl_delta = kept_pnl - baseline_pnl
        mark = " <--" if margin == 0.25 else ""
        print(f"  {margin:>6.2f} | {n_kept:>6} | {n_blocked:>7} | "
              f"{wr:>4.1f}% | ${kept_pnl:>+11,.2f} | ${pnl_delta:>+9,.2f} | "
              f"{blocked_wins:>8} | {blocked_losses:>8} | ${net_save:>+9,.2f}{mark}")

    # Show distribution of overpay values
    overpays = sorted([t.overpay for t in with_fv if t.overpay is not None])
    if overpays:
        print(f"\n  Overpay distribution (entry_price - fair_value):")
        pcts = [10, 25, 50, 75, 90, 95, 99]
        vals = []
        for p in pcts:
            idx = min(int(len(overpays) * p / 100), len(overpays) - 1)
            vals.append(f"P{p}={overpays[idx]:.3f}")
        print(f"  {', '.join(vals)}")

        # By outcome
        win_overpays = sorted([t.overpay for t in with_fv if t.overpay is not None and t.won])
        loss_overpays = sorted([t.overpay for t in with_fv if t.overpay is not None and not t.won])
        if win_overpays and loss_overpays:
            w_avg = sum(win_overpays) / len(win_overpays)
            l_avg = sum(loss_overpays) / len(loss_overpays)
            print(f"  Avg overpay — wins: {w_avg:.3f}, losses: {l_avg:.3f}")


def report_stop_loss(cases: list[StopLossCase], stoploss_exit: float) -> None:
    """Report stop loss analysis."""
    if not cases:
        print("  No stop loss cases.")
        return

    total = len(cases)
    dips = [c for c in cases if c.would_stoploss]
    n_dips = len(dips)

    print(f"\n  STOP LOSS FAIR VALUE ANALYSIS (exit threshold: {stoploss_exit})")
    print(f"  Total trades: {total:,}")
    print(f"  Trades where fair value dipped below {stoploss_exit}: {n_dips:,} ({n_dips/total*100:.1f}%)")

    if not dips:
        print("  No fair value dips detected — stop loss rarely relevant at this threshold.")
        return

    dip_wins = sum(1 for c in dips if c.won)
    dip_losses = n_dips - dip_wins

    print(f"  Of those {n_dips} dips:")
    print(f"    Won (false dip, should have held):  {dip_wins:>6} ({dip_wins/n_dips*100:.1f}%)")
    print(f"    Lost (real dip, correct to exit):   {dip_losses:>6} ({dip_losses/n_dips*100:.1f}%)")

    # Win rate by min fair value bucket — "at what fair value is it really dead?"
    # This answers: if mid-slot fair value drops to X, what's the chance it still wins?
    print(f"\n  WIN RATE BY MIN FAIR VALUE (during slot, after entry)")
    print(f"  Shows: when fair value temporarily dips to X, does it recover?")
    buckets = [(0.00, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, 0.40),
               (0.40, 0.50), (0.50, 0.60), (0.60, 0.70), (0.70, 0.80),
               (0.80, 0.90), (0.90, 1.00)]
    header = (f"  {'Min FV':>10} | {'Trades':>6} | {'Wins':>5} | "
              f"{'Win%':>5} | {'Cum Exit PnL':>12} | {'Cum Hold PnL':>12}")
    divider = (f"  {'-'*10}-+-{'-'*6}-+-{'-'*5}-+-"
               f"{'-'*5}-+-{'-'*12}-+-{'-'*12}")
    print(header)
    print(divider)

    for lo, hi in buckets:
        in_bucket = [c for c in cases if lo <= c.min_fair_value < hi]
        if not in_bucket:
            continue
        n = len(in_bucket)
        wins = sum(1 for c in in_bucket if c.won)
        wr = wins / n * 100
        # PnL if you exit at min_fair_value (approximate: sell at min_fv)
        # vs PnL if you hold to resolution
        exit_pnl = sum(
            (c.min_fair_value - c.entry_price) * 10  # size_usd=10, sell at min_fv
            for c in in_bucket
        )
        hold_pnl = sum(
            10 * (1.0 / c.entry_price - 1.0) if c.won else -10
            for c in in_bucket
        )
        mark = " <--" if lo <= stoploss_exit < hi else ""
        print(f"  [{lo:.2f},{hi:.2f}) | {n:>6} | {wins:>5} | "
              f"{wr:>4.1f}% | ${exit_pnl:>+11,.2f} | ${hold_pnl:>+11,.2f}{mark}")

    # Cumulative: "exit all trades where min FV < threshold" vs "hold all"
    print(f"\n  STOP LOSS THRESHOLD SWEEP — EXIT when min FV drops below threshold")
    print(f"  Compares: hold everything vs exit when fair value dips below X")
    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    header = (f"  {'Threshold':>9} | {'Exits':>6} | {'False%':>6} | "
              f"{'Hold PnL':>12} | {'Exit PnL':>12} | {'Savings':>12}")
    divider = (f"  {'-'*9}-+-{'-'*6}-+-{'-'*6}-+-"
               f"{'-'*12}-+-{'-'*12}-+-{'-'*12}")
    print(header)
    print(divider)

    total_hold_pnl = sum(
        10 * (1.0 / c.entry_price - 1.0) if c.won else -10
        for c in cases
    )

    for thresh in thresholds:
        exits = [c for c in cases if c.min_fair_value < thresh]
        holds = [c for c in cases if c.min_fair_value >= thresh]
        n_exits = len(exits)
        if n_exits == 0:
            continue
        false_exits = sum(1 for c in exits if c.won)
        false_pct = false_exits / n_exits * 100

        # PnL if we exit at min_fair_value for triggered trades, hold the rest
        exit_pnl = sum(
            (c.min_fair_value - c.entry_price) * 10
            for c in exits
        )
        hold_rest_pnl = sum(
            10 * (1.0 / c.entry_price - 1.0) if c.won else -10
            for c in holds
        )
        combined_pnl = exit_pnl + hold_rest_pnl
        savings = combined_pnl - total_hold_pnl
        mark = " <--" if thresh == stoploss_exit else ""
        print(f"  {thresh:>9.2f} | {n_exits:>6} | {false_pct:>5.1f}% | "
              f"${total_hold_pnl:>+11,.2f} | ${combined_pnl:>+11,.2f} | ${savings:>+11,.2f}{mark}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Backtest fair value guards for crypto TD maker")
    parser.add_argument("--data-dir",
                        default="../prediction-market-analysis/data/polymarket",
                        help="Path to polymarket data directory")
    parser.add_argument("--target-bid", type=float, default=0.75)
    parser.add_argument("--max-bid", type=float, default=0.85)
    parser.add_argument("--size-usd", type=float, default=10)
    parser.add_argument("--min-volume", type=float, default=5.0)
    parser.add_argument("--entry-minutes", type=float, default=5.0,
                        help="Assumed minutes into slot at entry time (default: 5)")
    parser.add_argument("--stoploss-exit", type=float, default=0.35,
                        help="Stop loss exit threshold for analysis")
    parser.add_argument("--symbol", default=None, choices=["BTC", "ETH", "SOL", "XRP"])
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    markets_dir = str(data_dir / "markets")
    trades_dir = str(data_dir / "trades")

    for d in ["markets", "trades"]:
        if not (data_dir / d).exists():
            print(f"Error: {d} directory not found at {data_dir / d}")
            sys.exit(1)

    con = duckdb.connect()

    # Step 1: Load markets
    print("Loading markets...")
    markets = load_markets(con, markets_dir, args.symbol)
    print(f"  {len(markets):,} resolved markets")
    if not markets:
        sys.exit(0)

    # Step 2: Compute VWAPs
    compute_vwap(con, trades_dir, markets, args.target_bid, args.max_bid)

    # Step 3: Load Binance klines
    symbols_needed = sorted({m.symbol for m in markets})
    min_ts = min(m.slot_ts for m in markets)
    max_ts = max(m.slot_ts for m in markets) + 900  # +15 min

    kline_indexes: dict[str, dict[int, float]] = {}
    for sym in symbols_needed:
        binance_sym = SYMBOL_MAP.get(sym)
        if not binance_sym:
            continue
        print(f"  Fetching {binance_sym} klines ({datetime.fromtimestamp(min_ts, tz=timezone.utc).strftime('%Y-%m-%d')} "
              f"to {datetime.fromtimestamp(max_ts, tz=timezone.utc).strftime('%Y-%m-%d')})...")
        klines = fetch_klines(binance_sym, min_ts * 1000, max_ts * 1000)
        kline_indexes[sym] = build_kline_index(klines)
        print(f"    {len(klines):,} klines loaded")

    # Step 4: Analyze
    print()
    print("=" * 70)
    print("FAIR VALUE GUARD BACKTEST")
    print("=" * 70)
    print(f"  Markets: {len(markets):,}  |  Entry at: {args.entry_minutes:.0f} min into slot")
    print(f"  Bid range: [{args.target_bid}, {args.max_bid}]  |  Symbol: {args.symbol or 'ALL'}")

    trades = analyze_trades(
        markets, kline_indexes, args.target_bid, args.max_bid,
        args.min_volume, args.entry_minutes, args.size_usd,
    )
    print(f"  Tradeable: {len(trades):,}  |  With fair value: {sum(1 for t in trades if t.fair_value is not None):,}")

    report_entry_sweep(trades, args.size_usd)

    # Stop loss analysis
    stop_cases = analyze_stop_losses(
        markets, kline_indexes, args.target_bid, args.max_bid,
        args.min_volume, args.entry_minutes, args.stoploss_exit, args.size_usd,
    )
    report_stop_loss(stop_cases, args.stoploss_exit)

    print()
    print("=" * 70)
    con.close()


if __name__ == "__main__":
    main()
