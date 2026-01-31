#!/usr/bin/env python3
# scripts/backtest_polymarket.py
"""Backtest our predictions against resolved Polymarket LoL markets.

Fetches historical resolved markets from Polymarket API,
compares our model's predictions to market odds and actual results.

Usage:
    uv run python scripts/backtest_polymarket.py --limit 50
    uv run python scripts/backtest_polymarket.py --save-csv backtest_results.csv
"""

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
import pandas as pd
import structlog

logger = structlog.get_logger()

# Polymarket API endpoints
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
LOL_SERIES_ID = "10311"


@dataclass
class MarketResult:
    """Result of a single market backtest."""
    market_id: str
    title: str
    date: str
    team1: str
    team2: str
    winner: str
    market_price_team1: float  # Pre-match odds for team1
    our_prediction: float | None  # Our model's prediction for team1
    actual_result: int  # 1 if team1 won, 0 otherwise
    volume: float

    @property
    def edge(self) -> float | None:
        """Calculate edge (our prediction - market price)."""
        if self.our_prediction is None:
            return None
        return self.our_prediction - self.market_price_team1

    @property
    def would_bet_team1(self) -> bool | None:
        """Would we bet on team1 (edge > 5%)?"""
        if self.edge is None:
            return None
        return self.edge > 0.05

    @property
    def would_bet_team2(self) -> bool | None:
        """Would we bet on team2 (edge < -5%)?"""
        if self.edge is None:
            return None
        return self.edge < -0.05

    @property
    def bet_correct(self) -> bool | None:
        """Was our bet (if any) correct?"""
        if self.would_bet_team1:
            return self.actual_result == 1
        elif self.would_bet_team2:
            return self.actual_result == 0
        return None  # No bet

    @property
    def profit(self) -> float | None:
        """Profit from a $1 bet (simplified)."""
        if self.would_bet_team1:
            if self.actual_result == 1:
                # Won: payout = 1/price - 1
                return (1 / self.market_price_team1) - 1
            else:
                return -1.0  # Lost bet
        elif self.would_bet_team2:
            team2_price = 1 - self.market_price_team1
            if self.actual_result == 0:
                return (1 / team2_price) - 1
            else:
                return -1.0
        return 0.0  # No bet


def fetch_resolved_markets(limit: int = 100) -> list[dict]:
    """Fetch resolved LoL markets from Polymarket Gamma API."""
    url = f"{GAMMA_API}/events"
    params = {
        "series_id": LOL_SERIES_ID,
        "closed": "true",
        "limit": limit,
        "order": "endDate",
        "ascending": "false",
    }

    logger.info("fetching_resolved_markets", url=url, limit=limit)

    with httpx.Client(timeout=30.0) as client:
        response = client.get(url, params=params)
        response.raise_for_status()
        events = response.json()

    logger.info("fetched_markets", count=len(events))
    return events


def extract_teams_from_title(title: str) -> tuple[str, str] | None:
    """Extract team names from market title.

    Examples:
        "LoL: T1 vs G2 Esports" -> ("T1", "G2 Esports")
        "LoL: Fnatic vs SK Gaming (BO3)" -> ("Fnatic", "SK Gaming")
    """
    # Remove "LoL: " prefix and match format indicators
    clean = re.sub(r'^LoL:\s*', '', title)
    clean = re.sub(r'\s*\(BO\d+\)\s*$', '', clean)
    clean = re.sub(r'\s*-\s*.*$', '', clean)  # Remove tournament info

    # Split by " vs "
    parts = clean.split(' vs ')
    if len(parts) != 2:
        return None

    team1 = parts[0].strip()
    team2 = parts[1].strip()

    return team1, team2


def get_price_history(token_id: str, start_ts: int, end_ts: int) -> list[dict]:
    """Get price history for a market token."""
    url = f"{CLOB_API}/prices-history"
    params = {
        "market": token_id,
        "startTs": start_ts,
        "endTs": end_ts,
        "fidelity": 60,  # 1 hour candles
    }

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            return data.get("history", [])
    except Exception as e:
        logger.warning("price_history_error", error=str(e))
        return []


def get_opening_price(market: dict) -> float | None:
    """Get the opening/pre-match price for a market."""
    if not market.get("markets"):
        return None

    market_data = market["markets"][0]

    # Get token IDs
    token_ids_str = market_data.get("clobTokenIds", "[]")
    try:
        token_ids = json.loads(token_ids_str)
    except json.JSONDecodeError:
        return None

    if not token_ids:
        return None

    # Get timestamps
    start_date = market.get("startDate") or market.get("creationDate")
    end_date = market.get("endDate")

    if not start_date or not end_date:
        return None

    try:
        start_dt = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
        start_ts = int(start_dt.timestamp())
        end_ts = int(end_dt.timestamp())
    except Exception:
        return None

    # Fetch price history for team1 token
    history = get_price_history(token_ids[0], start_ts, end_ts)

    if not history:
        return None

    # Get first price (opening price)
    # Skip any 0.5 prices at the very start (market just opened)
    for point in history[:5]:  # Check first 5 points
        price = point.get("p", 0.5)
        if price != 0.5:
            return price

    # Fall back to first price
    return history[0].get("p", 0.5)


def get_winner_from_market(market: dict) -> str | None:
    """Determine winner from resolved market."""
    if not market.get("markets"):
        return None

    market_data = market["markets"][0]

    # Check outcome prices - winner has price 1
    outcomes_str = market_data.get("outcomes", "[]")
    prices_str = market_data.get("outcomePrices", "[]")

    try:
        outcomes = json.loads(outcomes_str)
        prices = json.loads(prices_str)
    except json.JSONDecodeError:
        return None

    if len(outcomes) != 2 or len(prices) != 2:
        return None

    # Find which outcome has price "1" (winner)
    for i, price in enumerate(prices):
        if price == "1" or price == 1:
            return outcomes[i]

    return None


def load_team_stats(data_path: Path, team_name: str) -> dict | None:
    """Load team stats from training data for prediction."""
    from src.utils.team_matching import find_team_in_dataframe

    if not data_path.exists():
        return None

    df = pd.read_csv(data_path)

    # Use shared team matching utility
    as_team1, as_team2, is_team1_most_recent = find_team_in_dataframe(df, team_name)

    if len(as_team1) == 0 and len(as_team2) == 0:
        return None

    # Get most recent stats
    if len(as_team1) > 0 and (len(as_team2) == 0 or is_team1_most_recent):
        row = as_team1.iloc[-1]
        prefix = 'team1'
    else:
        row = as_team2.iloc[-1]
        prefix = 'team2'

    return {
        'winrate': row.get(f'{prefix}_winrate', 0.5),
        'recent_winrate': row.get(f'{prefix}_recent_winrate', 0.5),
        'blue_winrate': row.get(f'{prefix}_blue_winrate', 0.5),
        'red_winrate': row.get(f'{prefix}_red_winrate', 0.5),
        'games': row.get(f'{prefix}_games', 10),
        'fb_rate': row.get(f'{prefix}_fb_rate', 0.5),
        'ft_rate': row.get(f'{prefix}_ft_rate', 0.5),
        'fd_rate': row.get(f'{prefix}_fd_rate', 0.5),
        'avg_gd15': row.get(f'{prefix}_avg_gd15', 0),
    }


def predict_match(model, team1_stats: dict, team2_stats: dict) -> float:
    """Get model prediction for team1 winning."""
    features = {
        'team1_winrate': team1_stats['winrate'],
        'team1_recent_winrate': team1_stats['recent_winrate'],
        'team1_blue_winrate': team1_stats.get('blue_winrate', 0.5),
        'team1_games': team1_stats['games'],
        'team1_fb_rate': team1_stats['fb_rate'],
        'team1_ft_rate': team1_stats['ft_rate'],
        'team1_fd_rate': team1_stats['fd_rate'],
        'team1_avg_gd15': team1_stats['avg_gd15'],

        'team2_winrate': team2_stats['winrate'],
        'team2_recent_winrate': team2_stats['recent_winrate'],
        'team2_red_winrate': team2_stats.get('red_winrate', 0.5),
        'team2_games': team2_stats['games'],
        'team2_fb_rate': team2_stats['fb_rate'],
        'team2_ft_rate': team2_stats['ft_rate'],
        'team2_fd_rate': team2_stats['fd_rate'],
        'team2_avg_gd15': team2_stats['avg_gd15'],

        'winrate_diff': team1_stats['winrate'] - team2_stats['winrate'],
        'recent_form_diff': team1_stats['recent_winrate'] - team2_stats['recent_winrate'],
        'fb_rate_diff': team1_stats['fb_rate'] - team2_stats['fb_rate'],
        'gd15_diff': team1_stats['avg_gd15'] - team2_stats['avg_gd15'],

        'h2h_games': 0,
        'h2h_winrate': 0.5,
        'league_tier': 2,
        'is_playoffs': 0,
    }

    return model.predict_single(features)


def run_backtest(
    limit: int = 50,
    model_path: Path = Path("models/prematch_model.pkl"),
    data_path: Path = Path("data/prematch_training.csv"),
    min_volume: float = 0,
) -> list[MarketResult]:
    """Run backtest on resolved Polymarket markets."""

    # Load model if available
    model = None
    if model_path.exists():
        try:
            from scripts.train_prematch_model import PrematchModel
            model = PrematchModel.load(model_path)
            logger.info("model_loaded", path=str(model_path))
        except Exception as e:
            logger.warning("model_load_error", error=str(e))

    # Fetch resolved markets
    markets = fetch_resolved_markets(limit=limit)

    # Filter by volume if specified
    if min_volume > 0:
        markets = [m for m in markets if m.get("volume", 0) >= min_volume]
        logger.info("filtered_by_volume", count=len(markets), min_volume=min_volume)

    results = []

    for market in markets:
        title = market.get("title", "")

        # Extract teams
        teams = extract_teams_from_title(title)
        if not teams:
            logger.debug("skipping_market", title=title, reason="cannot_extract_teams")
            continue

        team1, team2 = teams

        # Get winner
        winner = get_winner_from_market(market)
        if not winner:
            logger.debug("skipping_market", title=title, reason="no_winner")
            continue

        # Determine actual result (1 if team1 won)
        actual_result = 1 if winner.lower() == team1.lower() else 0

        # Get opening price
        opening_price = get_opening_price(market)
        if opening_price is None:
            logger.debug("skipping_market", title=title, reason="no_price_history")
            continue

        # Get our prediction
        our_prediction = None
        if model and data_path.exists():
            team1_stats = load_team_stats(data_path, team1)
            team2_stats = load_team_stats(data_path, team2)

            if team1_stats and team2_stats:
                our_prediction = predict_match(model, team1_stats, team2_stats)

        # Get market metadata
        volume = market.get("volume", 0)
        date = market.get("endDate", "")[:10]
        market_id = market.get("id", "")

        result = MarketResult(
            market_id=market_id,
            title=title,
            date=date,
            team1=team1,
            team2=team2,
            winner=winner,
            market_price_team1=opening_price,
            our_prediction=our_prediction,
            actual_result=actual_result,
            volume=volume,
        )

        results.append(result)
        logger.info("processed_market",
                   title=title[:40],
                   market_price=f"{opening_price:.1%}",
                   our_pred=f"{our_prediction:.1%}" if our_prediction else "N/A",
                   winner=winner[:20])

        # Rate limiting
        time.sleep(0.2)

    return results


def print_backtest_summary(results: list[MarketResult]) -> None:
    """Print backtest summary statistics."""
    print(f"\n{'='*70}")
    print("BACKTEST RESULTS")
    print(f"{'='*70}")

    print(f"\nTotal markets analyzed: {len(results)}")

    # Markets with predictions
    with_pred = [r for r in results if r.our_prediction is not None]
    print(f"Markets with predictions: {len(with_pred)}")

    if not with_pred:
        print("\nNo predictions available. Train model first:")
        print("  uv run python scripts/train_prematch_model.py --input data/prematch_training.csv")
        return

    # Prediction accuracy (our model vs actual)
    correct_predictions = sum(
        1 for r in with_pred
        if (r.our_prediction > 0.5 and r.actual_result == 1) or
           (r.our_prediction < 0.5 and r.actual_result == 0)
    )
    print(f"\nModel accuracy: {correct_predictions}/{len(with_pred)} ({correct_predictions/len(with_pred):.1%})")

    # Market accuracy (market odds vs actual)
    market_correct = sum(
        1 for r in with_pred
        if (r.market_price_team1 > 0.5 and r.actual_result == 1) or
           (r.market_price_team1 < 0.5 and r.actual_result == 0)
    )
    print(f"Market accuracy: {market_correct}/{len(with_pred)} ({market_correct/len(with_pred):.1%})")

    # Bets analysis
    bets = [r for r in with_pred if r.would_bet_team1 or r.would_bet_team2]
    print(f"\n{'-'*70}")
    print(f"BETTING SIMULATION (>5% edge threshold)")
    print(f"{'-'*70}")
    print(f"Total bets placed: {len(bets)}")

    if bets:
        correct_bets = sum(1 for r in bets if r.bet_correct)
        total_profit = sum(r.profit or 0 for r in bets)

        print(f"Correct bets: {correct_bets}/{len(bets)} ({correct_bets/len(bets):.1%})")
        print(f"Total profit (per $1 per bet): ${total_profit:.2f}")
        print(f"ROI: {total_profit/len(bets)*100:.1f}%")

        # Show some example bets
        print(f"\n{'-'*70}")
        print("Sample bets:")
        for r in bets[:10]:
            bet_side = "Team1" if r.would_bet_team1 else "Team2"
            edge = r.edge if r.would_bet_team1 else -r.edge
            outcome = "WIN" if r.bet_correct else "LOSS"
            print(f"  {r.date} | {r.team1[:15]:15} vs {r.team2[:15]:15} | "
                  f"Bet: {bet_side} | Edge: {edge:+.1%} | {outcome}")

    # Edge distribution
    print(f"\n{'-'*70}")
    print("EDGE ANALYSIS")
    print(f"{'-'*70}")

    edges = [r.edge for r in with_pred if r.edge is not None]
    if edges:
        avg_abs_edge = sum(abs(e) for e in edges) / len(edges)
        print(f"Average absolute edge: {avg_abs_edge:.1%}")
        print(f"Markets with >5% edge: {sum(1 for e in edges if abs(e) > 0.05)}")
        print(f"Markets with >10% edge: {sum(1 for e in edges if abs(e) > 0.10)}")

    print(f"{'='*70}\n")


def save_results_csv(results: list[MarketResult], output_path: Path) -> None:
    """Save backtest results to CSV."""
    rows = []
    for r in results:
        rows.append({
            'market_id': r.market_id,
            'date': r.date,
            'team1': r.team1,
            'team2': r.team2,
            'winner': r.winner,
            'market_price_team1': r.market_price_team1,
            'our_prediction': r.our_prediction,
            'actual_result': r.actual_result,
            'edge': r.edge,
            'would_bet': r.would_bet_team1 or r.would_bet_team2,
            'bet_correct': r.bet_correct,
            'profit': r.profit,
            'volume': r.volume,
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Backtest predictions on Polymarket")
    parser.add_argument("--limit", type=int, default=50, help="Number of markets to analyze")
    parser.add_argument("--model", type=str, default="models/prematch_model.pkl", help="Model path")
    parser.add_argument("--data", type=str, default="data/prematch_training.csv", help="Training data path")
    parser.add_argument("--save-csv", type=str, default=None, help="Save results to CSV")
    parser.add_argument("--min-volume", type=float, default=0, help="Minimum market volume")

    args = parser.parse_args()

    results = run_backtest(
        limit=args.limit,
        model_path=Path(args.model),
        data_path=Path(args.data),
        min_volume=args.min_volume,
    )

    print_backtest_summary(results)

    if args.save_csv:
        save_results_csv(results, Path(args.save_csv))


if __name__ == "__main__":
    main()
