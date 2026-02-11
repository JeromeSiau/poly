#!/usr/bin/env python3
"""Backtest fear-selling strategy on historical Polymarket data.

Uses DuckDB to query 691M trades in parquet files directly — no bulk loading.
For each resolved market matching fear keywords, computes the VWAP of NO-side
trades in the target price zone, then simulates a portfolio with Kelly sizing.

FINDINGS (691M trades, 383K resolved markets):
  - Keyword-only fear selling is NET NEGATIVE across all configurations.
  - At NO entry ~93c, break-even requires ~93% win rate. Best clusters
    reach 90-95% but after fees/variance it's not enough.
  - Only russia_ukraine cluster showed marginal profit (+$49 on $10K).
  - The "other" cluster (vague keyword matches) is a consistent loser.
  - Implication: the LLM classifier is the real edge, not the keywords.

Usage:
    python scripts/backtest_fear_selling.py
    python scripts/backtest_fear_selling.py --categories iran,russia_ukraine,pandemic
    python scripts/backtest_fear_selling.py --min-yes 0.02 --max-yes 0.10 --sweep
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import pandas as pd

# ---------------------------------------------------------------------------
# Fear keyword tiers (from src/arb/fear_scanner.py)
# ---------------------------------------------------------------------------

FEAR_KEYWORDS: dict[str, list[str]] = {
    "high": [
        "strike", "strikes", "invade", "invades", "invasion", "nuclear",
        "war", "wars", "warfare", "attack", "attacks",
        "bomb", "bombs", "bombing", "collapse",
        "fall", "falls", "regime change",
        "die", "dies", "killed", "coup", "coups",
        "assassinate", "assassination",
        "missile", "missiles", "drone strike",
        "terrorist", "terrorism", "pandemic", "outbreak",
    ],
    "medium": [
        "ceasefire", "resign", "impeach", "default", "defaults",
        "recession", "shutdown", "sanctions", "regime",
        "annex", "deploy", "mobilize",
        "embargo", "blockade", "martial law",
        "hurricane", "earthquake", "tsunami", "disaster",
        "virus", "epidemic",
    ],
    "low": [
        "tariff", "tariffs", "fired", "charged", "indicted", "arrested",
    ],
}

_TIER_WEIGHTS = {"high": 3, "medium": 2, "low": 1}

_KEYWORD_PATTERNS: dict[str, re.Pattern[str]] = {
    tier: re.compile(
        "|".join(r"\b" + re.escape(kw) + r"\b" for kw in keywords),
        re.IGNORECASE,
    )
    for tier, keywords in FEAR_KEYWORDS.items()
}

CLUSTER_PATTERNS: dict[str, list[str]] = {
    "iran": ["iran", "khamenei", "iranian", "tehran", "persian"],
    "russia_ukraine": ["russia", "ukraine", "kremlin", "putin", "zelensk"],
    "china_taiwan": ["china", "taiwan", "beijing", "strait"],
    "north_korea": ["north korea", "pyongyang", "kim jong"],
    "us_military": ["us strike", "us invade", "pentagon", "us troops"],
    "middle_east": ["gaza", "israel", "hamas", "hezbollah", "lebanon"],
    "climate": ["hurricane", "earthquake", "tsunami", "wildfire", "flood", "disaster"],
    "pandemic": ["pandemic", "outbreak", "epidemic", "bird flu", "virus", "h5n1"],
}

# Annualised base rates per cluster (from fear_scanner.py)
ANNUAL_RATES: dict[str, float] = {
    "iran": 0.10,
    "russia_ukraine": 0.15,
    "china_taiwan": 0.05,
    "north_korea": 0.03,
    "us_military": 0.08,
    "middle_east": 0.12,
    "climate": 0.20,
    "pandemic": 0.08,
    "other": 0.10,
}


def classify_market(question: str) -> tuple[str, float]:
    """Classify a market question into (cluster, fear_score).

    Returns ('', 0.0) if no fear keywords match.
    """
    score = 0.0
    for tier, pattern in _KEYWORD_PATTERNS.items():
        matches = pattern.findall(question)
        score += len(matches) * _TIER_WEIGHTS[tier]

    if score == 0:
        return "", 0.0

    # Normalise to 0-1 range (cap at 10 raw points)
    fear_score = min(score / 10.0, 1.0)

    # Detect cluster
    q_lower = question.lower()
    for cluster, keywords in CLUSTER_PATTERNS.items():
        if any(kw in q_lower for kw in keywords):
            return cluster, fear_score

    return "other", fear_score


# ---------------------------------------------------------------------------
# Data loading: resolved markets + trade VWAP via DuckDB
# ---------------------------------------------------------------------------

@dataclass
class FearMarket:
    """A resolved market that matched fear criteria."""
    market_id: str
    question: str
    cluster: str
    fear_score: float
    no_token_id: str
    no_won: bool
    end_date: str
    volume: float
    vwap_no: float = 0.0
    trade_volume_usd: float = 0.0
    trade_count: int = 0


def load_fear_markets(con: duckdb.DuckDBPyConnection, markets_dir: str,
                      min_yes: float, max_yes: float,
                      categories: list[str] | None,
                      min_fear_score: float) -> list[FearMarket]:
    """Load resolved markets, classify, and filter to fear candidates."""
    print("Loading resolved markets...")
    t0 = time.time()

    markets_df = con.execute(f"""
        SELECT id, question, slug, clob_token_ids, outcome_prices,
               volume, end_date
        FROM '{markets_dir}/*.parquet'
        WHERE closed = true
    """).df()

    print(f"  {len(markets_df):,} resolved markets loaded in {time.time()-t0:.1f}s")

    fear_markets: list[FearMarket] = []

    for _, row in markets_df.iterrows():
        try:
            prices = json.loads(row["outcome_prices"]) if row["outcome_prices"] else None
            if not prices or len(prices) != 2:
                continue
            p0, p1 = float(prices[0]), float(prices[1])

            # Determine resolution
            if p0 > 0.99 and p1 < 0.01:
                winning = 0
            elif p0 < 0.01 and p1 > 0.99:
                winning = 1
            else:
                continue

            token_ids = json.loads(row["clob_token_ids"]) if row["clob_token_ids"] else None
            if not token_ids or len(token_ids) != 2:
                continue

            # Classify
            question = row.get("question", "") or ""
            cluster, fear_score = classify_market(question)

            if fear_score < min_fear_score:
                continue

            if categories and cluster not in categories:
                continue

            # token_ids[0] = YES token, token_ids[1] = NO token
            no_token_id = token_ids[1]
            no_won = winning == 1  # outcome 1 = NO won

            fear_markets.append(FearMarket(
                market_id=row["id"],
                question=question,
                cluster=cluster,
                fear_score=fear_score,
                no_token_id=no_token_id,
                no_won=no_won,
                end_date=str(row.get("end_date", "")),
                volume=float(row.get("volume", 0) or 0),
            ))
        except (json.JSONDecodeError, ValueError, TypeError, IndexError):
            continue

    print(f"  {len(fear_markets):,} fear markets after classification")
    return fear_markets


def compute_vwap(con: duckdb.DuckDBPyConnection, trades_dir: str,
                 fear_markets: list[FearMarket],
                 min_yes: float, max_yes: float) -> list[FearMarket]:
    """Compute VWAP of NO token trades in the target price zone."""
    if not fear_markets:
        return []

    # NO price zone: if YES is [min_yes, max_yes], NO is [1-max_yes, 1-min_yes]
    min_no = 1.0 - max_yes
    max_no = 1.0 - min_yes

    # Register NO token IDs as a temp table
    token_data = [(m.no_token_id,) for m in fear_markets]
    con.execute("CREATE OR REPLACE TABLE fear_no_tokens (token_id VARCHAR)")
    con.executemany("INSERT INTO fear_no_tokens VALUES (?)", token_data)

    print(f"Computing VWAP for {len(fear_markets):,} NO tokens "
          f"(NO price zone: ${min_no:.2f}-${max_no:.2f})...")
    print("  Scanning trades (this may take a few minutes)...")
    t0 = time.time()

    # Query: compute VWAP per NO token in our price zone
    vwap_df = con.execute(f"""
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
            INNER JOIN fear_no_tokens fnt
                ON fnt.token_id = (
                    CASE WHEN t.maker_asset_id = '0'
                         THEN t.taker_asset_id
                         ELSE t.maker_asset_id
                    END
                )
            WHERE t.taker_amount > 0 AND t.maker_amount > 0
        )
        SELECT
            token_id,
            SUM(usdc_amount * price) / SUM(usdc_amount) AS vwap_no,
            SUM(usdc_amount) AS volume_usd,
            COUNT(*) AS trade_count
        FROM priced_trades
        WHERE price BETWEEN {min_no} AND {max_no}
        GROUP BY token_id
    """).df()

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s — {len(vwap_df):,} tokens with trades in zone")

    # Map back to fear markets
    vwap_map = {}
    for _, row in vwap_df.iterrows():
        vwap_map[row["token_id"]] = (
            float(row["vwap_no"]),
            float(row["volume_usd"]),
            int(row["trade_count"]),
        )

    result = []
    for m in fear_markets:
        if m.no_token_id in vwap_map:
            m.vwap_no, m.trade_volume_usd, m.trade_count = vwap_map[m.no_token_id]
            result.append(m)

    print(f"  {len(result):,} markets with tradeable NO in price zone")
    return result


# ---------------------------------------------------------------------------
# Portfolio simulation
# ---------------------------------------------------------------------------

@dataclass
class TradeResult:
    market_id: str
    question: str
    cluster: str
    entry_no: float
    no_won: bool
    size_usd: float
    pnl_usd: float
    capital_after: float


@dataclass
class PortfolioStats:
    trades: list[TradeResult] = field(default_factory=list)
    initial_capital: float = 0.0
    final_capital: float = 0.0
    peak_capital: float = 0.0
    max_drawdown_pct: float = 0.0
    total_pnl: float = 0.0
    win_count: int = 0
    loss_count: int = 0


def estimate_no_prob(cluster: str, no_price: float) -> float:
    """Estimate P(NO wins) using annualized base rates.

    Uses the cluster's annual rate of YES events to infer P(NO).
    """
    annual_yes_rate = ANNUAL_RATES.get(cluster, 0.10)
    # For a single market, P(YES) ~ annual_rate scaled down
    # Simplified: assume each market represents ~1/12 of the annual rate
    p_yes = min(annual_yes_rate / 12.0, 0.50)
    p_no = 1.0 - p_yes
    return p_no


def simulate(fear_markets: list[FearMarket], capital: float,
             kelly_fraction: float, max_position_pct: float) -> PortfolioStats:
    """Simulate portfolio walking through markets chronologically."""
    # Sort by end_date
    markets = sorted(fear_markets, key=lambda m: m.end_date)

    stats = PortfolioStats(initial_capital=capital, peak_capital=capital)
    current_capital = capital

    for m in markets:
        if current_capital <= 0:
            break

        no_price = m.vwap_no
        if no_price <= 0 or no_price >= 1:
            continue

        # Kelly sizing
        p_no = estimate_no_prob(m.cluster, no_price)
        b = (1.0 - no_price) / no_price  # payout ratio
        q = 1.0 - p_no
        kelly_f = (p_no * b - q) / b if b > 0 else 0
        kelly_f = max(kelly_f, 0)  # don't bet if negative edge

        if kelly_f <= 0:
            continue

        size = min(
            kelly_f * kelly_fraction * current_capital,
            max_position_pct * current_capital,
        )
        size = max(size, 0)

        if size < 1.0:  # minimum $1
            continue

        # Resolution
        if m.no_won:
            pnl = size * (1.0 / no_price - 1.0)
            stats.win_count += 1
        else:
            pnl = -size
            stats.loss_count += 1

        current_capital += pnl
        stats.peak_capital = max(stats.peak_capital, current_capital)

        dd = (stats.peak_capital - current_capital) / stats.peak_capital if stats.peak_capital > 0 else 0
        stats.max_drawdown_pct = max(stats.max_drawdown_pct, dd)

        stats.trades.append(TradeResult(
            market_id=m.market_id,
            question=m.question[:80],
            cluster=m.cluster,
            entry_no=no_price,
            no_won=m.no_won,
            size_usd=size,
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
    # Assume ~50 trades/year as annualization factor
    trades_per_year = max(len(trades), 50)
    return (mean_r / std_r) * math.sqrt(trades_per_year)


def print_report(stats: PortfolioStats, args: argparse.Namespace):
    """Print the backtest report."""
    n = len(stats.trades)
    win_rate = stats.win_count / n * 100 if n > 0 else 0
    ret_pct = stats.total_pnl / stats.initial_capital * 100

    print()
    print("=" * 70)
    print("FEAR SELLING BACKTEST RESULTS")
    print("=" * 70)
    print(f"  YES zone:        [{args.min_yes:.2f}, {args.max_yes:.2f}]")
    print(f"  Kelly fraction:  {args.kelly_fraction}")
    print(f"  Max position:    {args.max_position_pct*100:.0f}%")
    print(f"  Initial capital: ${stats.initial_capital:,.0f}")
    print()

    print("AGGREGATE")
    print(f"  Trades:          {n:,}")
    print(f"  Wins / Losses:   {stats.win_count} / {stats.loss_count}")
    print(f"  Win rate:        {win_rate:.1f}%")
    print(f"  Total PnL:       ${stats.total_pnl:+,.2f}")
    print(f"  Return:          {ret_pct:+.1f}%")
    print(f"  Final capital:   ${stats.final_capital:,.2f}")
    print(f"  Max drawdown:    {stats.max_drawdown_pct*100:.1f}%")
    print(f"  Sharpe (ann.):   {compute_sharpe(stats.trades):.2f}")
    print()

    # By cluster
    clusters: dict[str, list[TradeResult]] = {}
    for t in stats.trades:
        clusters.setdefault(t.cluster, []).append(t)

    print(f"  {'Cluster':<16} | {'Trades':>6} | {'Win%':>5} | {'Avg Entry':>9} | {'PnL':>10} | {'Sharpe':>6}")
    print(f"  {'-'*16}-+-{'-'*6}-+-{'-'*5}-+-{'-'*9}-+-{'-'*10}-+-{'-'*6}")

    for cluster in sorted(clusters.keys()):
        ct = clusters[cluster]
        c_wins = sum(1 for t in ct if t.no_won)
        c_wr = c_wins / len(ct) * 100 if ct else 0
        c_pnl = sum(t.pnl_usd for t in ct)
        c_avg_entry = sum(t.entry_no for t in ct) / len(ct) if ct else 0
        c_sharpe = compute_sharpe(ct)
        print(f"  {cluster:<16} | {len(ct):>6} | {c_wr:>4.1f}% | ${c_avg_entry:>7.4f} | ${c_pnl:>+9,.2f} | {c_sharpe:>6.2f}")

    print()

    # By YES price zone (YES = 1 - NO entry)
    zones = [
        ("0.05-0.15", 0.05, 0.15),
        ("0.15-0.25", 0.15, 0.25),
        ("0.25-0.35", 0.25, 0.35),
        ("0.35-0.50", 0.35, 0.50),
        ("0.50-0.65", 0.50, 0.65),
    ]
    print(f"  {'YES Zone':<12} | {'Trades':>6} | {'Win%':>5} | {'PnL/Trade':>10}")
    print(f"  {'-'*12}-+-{'-'*6}-+-{'-'*5}-+-{'-'*10}")
    for label, lo, hi in zones:
        zt = [t for t in stats.trades if lo <= (1 - t.entry_no) < hi]
        if not zt:
            continue
        z_wins = sum(1 for t in zt if t.no_won)
        z_wr = z_wins / len(zt) * 100
        z_pnl_avg = sum(t.pnl_usd for t in zt) / len(zt)
        print(f"  {label:<12} | {len(zt):>6} | {z_wr:>4.1f}% | ${z_pnl_avg:>+9,.2f}")

    print()
    print("=" * 70)


def run_sweep(fear_markets: list[FearMarket], capital: float, max_pos: float):
    """Parameter sensitivity sweep."""
    print()
    print("=" * 70)
    print("PARAMETER SENSITIVITY SWEEP")
    print("=" * 70)

    kellys = [0.125, 0.25, 0.5, 1.0]

    print(f"  {'Kelly':>8} | {'Trades':>6} | {'Win%':>5} | {'PnL':>12} | {'Return':>8} | {'MaxDD':>6} | {'Sharpe':>6}")
    print(f"  {'-'*8}-+-{'-'*6}-+-{'-'*5}-+-{'-'*12}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}")

    for kf in kellys:
        s = simulate(fear_markets, capital, kf, max_pos)
        n = len(s.trades)
        wr = s.win_count / n * 100 if n > 0 else 0
        ret = s.total_pnl / capital * 100
        sh = compute_sharpe(s.trades)
        tag = " <--" if kf == 0.25 else ""
        print(f"  {kf:>8.3f} | {n:>6} | {wr:>4.1f}% | ${s.total_pnl:>+11,.2f} | {ret:>+7.1f}% | {s.max_drawdown_pct*100:>5.1f}% | {sh:>6.2f}{tag}")

    print()
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Backtest fear-selling on Polymarket history")
    parser.add_argument("--data-dir", default="../prediction-market-analysis/data/polymarket",
                        help="Path to polymarket data directory")
    parser.add_argument("--capital", type=float, default=10_000)
    parser.add_argument("--min-yes", type=float, default=0.15)
    parser.add_argument("--max-yes", type=float, default=0.65)
    parser.add_argument("--kelly-fraction", type=float, default=0.25)
    parser.add_argument("--max-position-pct", type=float, default=0.10)
    parser.add_argument("--min-fear-score", type=float, default=0.1)
    parser.add_argument("--categories", default=None,
                        help="Comma-separated clusters to include (e.g. geopolitics,macro)")
    parser.add_argument("--sweep", action="store_true", help="Run parameter sensitivity sweep")
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

    categories = [c.strip() for c in args.categories.split(",")] if args.categories else None

    con = duckdb.connect()

    # Step 1: Load and classify fear markets
    fear_markets = load_fear_markets(
        con, markets_dir, args.min_yes, args.max_yes, categories, args.min_fear_score
    )

    if not fear_markets:
        print("No fear markets found. Try lowering --min-fear-score or broadening --categories.")
        sys.exit(0)

    # Step 2: Compute VWAP from historical trades
    fear_markets = compute_vwap(con, trades_dir, fear_markets, args.min_yes, args.max_yes)

    if not fear_markets:
        print("No trades found in the target price zone.")
        sys.exit(0)

    # Step 3: Simulate portfolio
    stats = simulate(fear_markets, args.capital, args.kelly_fraction, args.max_position_pct)

    # Step 4: Report
    print_report(stats, args)

    # Step 5: Sweep if requested
    if args.sweep:
        run_sweep(fear_markets, args.capital, args.max_position_pct)

    con.close()


if __name__ == "__main__":
    main()
