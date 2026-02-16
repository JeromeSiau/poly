#!/usr/bin/env python3
"""Backtest last-penny sniper on historical Polymarket data.

STRATEGY (inspired by Sharky6999):
    Buy any outcome trading at 0.95+ (market consensus = quasi-certain).
    Hold to resolution. Profit = (1 - entry_price) per share on wins.
    No stop-loss, no early exit.

    The Polymarket price IS the signal — no external feeds needed.
    If something trades at 0.99, the market has decided the outcome.

BACKTEST METHODOLOGY:
    For each resolved market, find all on-chain trades at price >= threshold.
    Each trade is a potential sniper fill. If the traded token is the winner,
    PnL = (1 - price) * shares. If loser, PnL = -price * shares.
    This naturally captures the win/loss ratio without predicting outcomes.

DATASET: 380K+ resolved markets (crypto updown, brackets, hourly, etc.)

Usage:
    python scripts/backtest_sniper.py
    python scripts/backtest_sniper.py --min-price 0.99
    python scripts/backtest_sniper.py --sweep
    python scripts/backtest_sniper.py --category crypto_15min
    python scripts/backtest_sniper.py --category crypto_hourly
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


MARKETS_DIR = "../prediction-market-analysis/data/polymarket/markets"
TRADES_DIR = "../prediction-market-analysis/data/polymarket/trades"
BLOCKS_DIR = "../prediction-market-analysis/data/polymarket/blocks"


# ---------------------------------------------------------------------------
# Market classification
# ---------------------------------------------------------------------------

def classify_market(slug: str) -> str:
    """Classify a market by slug pattern."""
    if "updown-15m" in slug:
        return "crypto_15min"
    elif "updown" in slug:
        return "crypto_hourly"
    elif "price-of" in slug:
        return "crypto_bracket"
    elif "reach" in slug and any(c in slug for c in ["bitcoin", "ethereum", "solana", "xrp"]):
        return "crypto_weekly"
    else:
        return "other"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SniperMarket:
    """A resolved market with its winning/losing token IDs."""
    market_id: str
    slug: str
    category: str
    end_date: str
    end_ts: int          # unix timestamp of resolution
    volume: float
    winning_token: str
    losing_token: str


@dataclass
class SniperTrade:
    """A simulated sniper trade."""
    market_id: str
    slug: str
    category: str
    entry_price: float
    size_usd: float
    won: bool
    pnl_usd: float
    capital_after: float


@dataclass
class SniperStats:
    """Aggregated backtest results."""
    initial_capital: float
    final_capital: float = 0.0
    peak_capital: float = 0.0
    max_drawdown_pct: float = 0.0
    total_pnl: float = 0.0
    win_count: int = 0
    loss_count: int = 0
    trades: list[SniperTrade] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_markets(
    con: duckdb.DuckDBPyConnection,
    categories: list[str] | None = None,
) -> list[SniperMarket]:
    """Load resolved markets with known winning/losing tokens."""
    print("Loading resolved markets...")
    t0 = time.time()

    df = con.execute(f"""
        SELECT id, slug, outcome_prices, clob_token_ids, volume, end_date
        FROM '{MARKETS_DIR}/*.parquet'
        WHERE closed = true
          AND outcome_prices IS NOT NULL
          AND clob_token_ids IS NOT NULL
    """).df()

    print(f"  {len(df):,} closed markets loaded in {time.time()-t0:.1f}s")

    markets: list[SniperMarket] = []
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
            if categories and cat not in categories:
                continue

            # Parse end_date to unix timestamp
            end_date_val = row["end_date"]
            try:
                if hasattr(end_date_val, "timestamp"):
                    end_ts = int(end_date_val.timestamp())
                else:
                    end_ts = int(pd.Timestamp(str(end_date_val)).timestamp())
            except (ValueError, TypeError):
                end_ts = 0

            markets.append(SniperMarket(
                market_id=row["id"],
                slug=row["slug"],
                category=cat,
                end_date=str(row["end_date"]),
                end_ts=end_ts,
                volume=float(row["volume"]) if row["volume"] else 0,
                winning_token=winning_token,
                losing_token=losing_token,
            ))
        except (json.JSONDecodeError, ValueError, IndexError):
            continue

    print(f"  {len(markets):,} fully resolved markets after filtering")

    cats = Counter(m.category for m in markets)
    for cat, count in cats.most_common():
        print(f"    {cat}: {count:,}")

    return markets


# ---------------------------------------------------------------------------
# Trade scanning
# ---------------------------------------------------------------------------

def scan_sniper_trades(
    con: duckdb.DuckDBPyConnection,
    markets: list[SniperMarket],
    min_price: float,
    *,
    winners_only: bool = False,
    last_minutes: float = 0,
) -> pd.DataFrame:
    """Find all on-chain trades at price >= min_price for our markets.

    Args:
        winners_only: If True, only scan winning tokens (oracle mode / upper bound).
        last_minutes: If > 0, only include trades within N minutes before resolution.

    Returns DataFrame with columns:
        token_id, price, usdc_amount, is_winner, market_id
    """
    token_rows = []
    for m in markets:
        token_rows.append((m.winning_token, m.market_id, True, m.end_ts))
        if not winners_only:
            token_rows.append((m.losing_token, m.market_id, False, m.end_ts))

    con.execute("CREATE OR REPLACE TABLE sniper_tokens "
                "(token_id VARCHAR, market_id VARCHAR, is_winner BOOLEAN, end_ts BIGINT)")
    con.executemany("INSERT INTO sniper_tokens VALUES (?, ?, ?, ?)", token_rows)

    timing_label = f", last {last_minutes:.0f}min" if last_minutes > 0 else ""
    print(f"\nScanning trades at price >= {min_price:.3f}{timing_label} "
          f"for {len(token_rows):,} tokens...")
    t0 = time.time()

    # Time filter: join with blocks table to get trade timestamps
    blocks_join = ""
    time_filter = ""
    if last_minutes > 0:
        blocks_join = f"INNER JOIN '{BLOCKS_DIR}/*.parquet' b ON b.block_number = t.block_number"
        time_filter = f"AND EPOCH(CAST(b.timestamp AS TIMESTAMP)) >= (st2.end_ts - {int(last_minutes * 60)})"

    df = con.execute(f"""
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
            FROM '{TRADES_DIR}/*.parquet' t
            {blocks_join}
            INNER JOIN sniper_tokens st2
                ON st2.token_id = (
                    CASE WHEN t.maker_asset_id = '0'
                         THEN t.taker_asset_id
                         ELSE t.maker_asset_id
                    END
                )
            WHERE t.taker_amount > 0 AND t.maker_amount > 0
            {time_filter}
        )
        SELECT
            pt.token_id,
            pt.price,
            pt.usdc_amount,
            st.is_winner,
            st.market_id
        FROM priced_trades pt
        INNER JOIN sniper_tokens st ON st.token_id = pt.token_id
        WHERE pt.price >= {min_price} AND pt.price < 1.0
    """).df()

    elapsed = time.time() - t0
    winners = df[df["is_winner"] == True]  # noqa: E712
    losers = df[df["is_winner"] == False]  # noqa: E712
    print(f"  Done in {elapsed:.1f}s — {len(df):,} trades found")
    print(f"  Winners: {len(winners):,} trades, ${winners['usdc_amount'].sum():,.0f}")
    print(f"  Losers:  {len(losers):,} trades, ${losers['usdc_amount'].sum():,.0f}")

    return df


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate(
    markets: list[SniperMarket],
    trades_df: pd.DataFrame,
    capital: float,
    risk_pct: float,
    max_per_market_pct: float,
    fee_rate: float = 0.25,
    fee_exponent: int = 2,
) -> SniperStats:
    """Simulate sniper strategy on historical trades.

    For each market (chronological), take all available high-price trades.
    The sniper doesn't know which side wins — it buys anything at high prices.
    Winners profit, losers lose.

    Sizing: constant risk per trade = capital * risk_pct.
    """
    market_map = {m.market_id: m for m in markets}
    fee_categories = {"crypto_15min"}

    sorted_market_ids = [
        m.market_id for m in sorted(markets, key=lambda m: m.end_date)
    ]

    if trades_df.empty:
        return SniperStats(initial_capital=capital, final_capital=capital,
                           peak_capital=capital)

    # Pre-aggregate: per market+side -> avg price, total volume
    market_trades = trades_df.groupby(["market_id", "is_winner"]).agg(
        avg_price=("price", "mean"),
        total_usd=("usdc_amount", "sum"),
        trade_count=("price", "count"),
    ).reset_index()

    mt_lookup: dict[str, dict[bool, tuple[float, float]]] = {}
    for _, row in market_trades.iterrows():
        mid = row["market_id"]
        is_w = bool(row["is_winner"])
        mt_lookup.setdefault(mid, {})[is_w] = (row["avg_price"], row["total_usd"])

    stats = SniperStats(initial_capital=capital, peak_capital=capital)
    current_capital = capital

    for mid in sorted_market_ids:
        m = market_map[mid]
        if mid not in mt_lookup:
            continue

        for is_winner, (avg_price, total_usd) in mt_lookup[mid].items():
            # Fee calculation
            fee_pct = 0.0
            if m.category in fee_categories:
                fee_pct = avg_price * fee_rate * (avg_price * (1 - avg_price)) ** fee_exponent

            # Sizing: constant risk per trade
            max_risk = current_capital * risk_pct
            max_market = current_capital * max_per_market_pct
            loss_per_share = avg_price + fee_pct
            if loss_per_share <= 0:
                continue

            max_shares = max_risk / loss_per_share
            order_usd = min(max_shares * avg_price, max_market, total_usd)

            if order_usd < 1.0 or current_capital < order_usd:
                continue

            shares = order_usd / avg_price
            fee_total = shares * fee_pct

            if is_winner:
                pnl = shares * (1.0 - avg_price) - fee_total
                stats.win_count += 1
            else:
                pnl = -(shares * avg_price) - fee_total
                stats.loss_count += 1

            current_capital += pnl
            stats.peak_capital = max(stats.peak_capital, current_capital)
            dd = ((stats.peak_capital - current_capital) / stats.peak_capital
                  if stats.peak_capital > 0 else 0)
            stats.max_drawdown_pct = max(stats.max_drawdown_pct, dd)

            stats.trades.append(SniperTrade(
                market_id=m.market_id,
                slug=m.slug,
                category=m.category,
                entry_price=avg_price,
                size_usd=order_usd,
                won=is_winner,
                pnl_usd=pnl,
                capital_after=current_capital,
            ))

    stats.final_capital = current_capital
    stats.total_pnl = current_capital - capital
    return stats


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def compute_sharpe(trades: list[SniperTrade]) -> float:
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


def print_report(stats: SniperStats, min_price: float) -> None:
    """Print backtest results."""
    n = len(stats.trades)
    if n == 0:
        print("\n  No trades executed.")
        return

    win_rate = stats.win_count / n * 100
    ret_pct = stats.total_pnl / stats.initial_capital * 100
    avg_entry = sum(t.entry_price for t in stats.trades) / n
    avg_pnl = stats.total_pnl / n

    print(f"\n  LAST-PENNY SNIPER [min_price={min_price}]")
    print(f"  {'—' * 55}")
    print(f"  Trades:          {n:,}")
    print(f"  Wins / Losses:   {stats.win_count:,} / {stats.loss_count:,}")
    print(f"  Win rate:        {win_rate:.2f}%")
    print(f"  Avg entry price: {avg_entry:.4f}")
    print(f"  Break-even WR:   {avg_entry * 100:.2f}%")
    print(f"  Edge:            {win_rate - avg_entry * 100:+.2f}%")
    print(f"  Avg PnL/trade:   ${avg_pnl:+,.4f}")
    print(f"  Total PnL:       ${stats.total_pnl:+,.2f}")
    print(f"  Return:          {ret_pct:+.1f}%")
    print(f"  Final capital:   ${stats.final_capital:,.2f}")
    print(f"  Max drawdown:    {stats.max_drawdown_pct*100:.1f}%")
    print(f"  Sharpe (ann.):   {compute_sharpe(stats.trades):.2f}")

    # By category
    cats = sorted({t.category for t in stats.trades})
    print(f"\n  {'Category':<16} | {'Trades':>7} | {'Win%':>6} | {'PnL':>12} | {'Avg Entry':>9}")
    print(f"  {'-'*16}-+-{'-'*7}-+-{'-'*6}-+-{'-'*12}-+-{'-'*9}")
    for cat in cats:
        ct = [t for t in stats.trades if t.category == cat]
        if not ct:
            continue
        c_wins = sum(1 for t in ct if t.won)
        c_wr = c_wins / len(ct) * 100
        c_pnl = sum(t.pnl_usd for t in ct)
        c_avg = sum(t.entry_price for t in ct) / len(ct)
        print(f"  {cat:<16} | {len(ct):>7,} | {c_wr:>5.1f}% | ${c_pnl:>+11,.2f} | {c_avg:>9.4f}")

    # By price zone
    zones = [
        ("0.900-0.950", 0.90, 0.95),
        ("0.950-0.960", 0.95, 0.96),
        ("0.960-0.970", 0.96, 0.97),
        ("0.970-0.980", 0.97, 0.98),
        ("0.980-0.990", 0.98, 0.99),
        ("0.990-0.999", 0.99, 0.999),
        ("0.999+     ", 0.999, 1.01),
    ]
    print(f"\n  {'Price Zone':<12} | {'Trades':>7} | {'Win%':>6} | {'PnL':>12} | {'Avg Size':>9}")
    print(f"  {'-'*12}-+-{'-'*7}-+-{'-'*6}-+-{'-'*12}-+-{'-'*9}")
    for label, lo, hi in zones:
        zt = [t for t in stats.trades if lo <= t.entry_price < hi]
        if not zt:
            continue
        z_wins = sum(1 for t in zt if t.won)
        z_wr = z_wins / len(zt) * 100
        z_pnl = sum(t.pnl_usd for t in zt)
        z_avg_sz = sum(t.size_usd for t in zt) / len(zt)
        print(f"  {label:<12} | {len(zt):>7,} | {z_wr:>5.1f}% | ${z_pnl:>+11,.2f} | ${z_avg_sz:>8,.2f}")


def run_sweep(
    markets: list[SniperMarket],
    con: duckdb.DuckDBPyConnection,
    capital: float,
    risk_pct: float,
    max_per_market_pct: float,
    *,
    winners_only: bool = False,
    last_minutes: float = 0,
) -> None:
    """Parameter sweep over min_price thresholds."""
    labels = []
    if winners_only:
        labels.append("WINNERS ONLY")
    if last_minutes > 0:
        labels.append(f"last {last_minutes:.0f}min")
    mode_label = f" ({', '.join(labels)})" if labels else ""
    print("\n" + "=" * 70)
    print(f"PARAMETER SWEEP: min_price{mode_label}")
    print("=" * 70)

    header = f"  {'min_price':>9} | {'Trades':>7} | {'Win%':>6} | {'PnL':>12} | {'Return':>8} | {'Sharpe':>6} | {'MaxDD':>6}"
    print(header)
    print(f"  {'-'*9}-+-{'-'*7}-+-{'-'*6}-+-{'-'*12}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}")

    for mp in [0.90, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999]:
        trades_df = scan_sniper_trades(con, markets, mp, winners_only=winners_only,
                                       last_minutes=last_minutes)
        s = simulate(markets, trades_df, capital, risk_pct, max_per_market_pct)
        n = len(s.trades)
        if n == 0:
            print(f"  {mp:>9.3f} |       0 |     -- |           -- |       -- |     -- |     --")
            continue
        wr = s.win_count / n * 100
        ret = s.total_pnl / capital * 100
        sh = compute_sharpe(s.trades)
        dd = s.max_drawdown_pct * 100
        print(f"  {mp:>9.3f} | {n:>7,} | {wr:>5.1f}% | ${s.total_pnl:>+11,.2f} | {ret:>+7.1f}% | {sh:>6.2f} | {dd:>5.1f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Backtest last-penny sniper")
    parser.add_argument("--min-price", type=float, default=0.95,
                        help="Minimum entry price (default: 0.95)")
    parser.add_argument("--capital", type=float, default=500.0,
                        help="Starting capital in USD (default: 500)")
    parser.add_argument("--risk-pct", type=float, default=0.01,
                        help="Risk per trade as fraction of capital (default: 0.01)")
    parser.add_argument("--max-per-market", type=float, default=0.05,
                        help="Max USD per market as fraction of capital (default: 0.05)")
    parser.add_argument("--category", type=str, default=None,
                        help="Filter: crypto_15min, crypto_hourly, crypto_bracket, crypto_weekly, other")
    parser.add_argument("--sweep", action="store_true",
                        help="Run parameter sweep over min_price values")
    parser.add_argument("--winners-only", action="store_true",
                        help="Oracle mode: only buy winning tokens (upper bound)")
    parser.add_argument("--last-minutes", type=float, default=0,
                        help="Only trades within N minutes before resolution (0 = all)")
    args = parser.parse_args()

    con = duckdb.connect()

    categories = [args.category] if args.category else None
    markets = load_markets(con, categories)
    if not markets:
        print("No markets found.")
        sys.exit(1)

    if args.sweep:
        run_sweep(markets, con, args.capital, args.risk_pct, args.max_per_market,
                  winners_only=args.winners_only, last_minutes=args.last_minutes)
    else:
        trades_df = scan_sniper_trades(con, markets, args.min_price,
                                       winners_only=args.winners_only,
                                       last_minutes=args.last_minutes)
        stats = simulate(markets, trades_df, args.capital, args.risk_pct,
                         args.max_per_market)
        print_report(stats, args.min_price)

    con.close()


if __name__ == "__main__":
    main()
