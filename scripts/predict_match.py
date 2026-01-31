#!/usr/bin/env python3
# scripts/predict_match.py
"""Predict match outcome and compare to Polymarket odds.

Usage:
    # Predict with manual stats
    uv run python scripts/predict_match.py --team1 "T1" --team2 "G2" --polymarket-odds 0.65

    # Predict using historical data lookup
    uv run python scripts/predict_match.py --team1 "T1" --team2 "G2" --lookup
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import structlog

logger = structlog.get_logger()


def load_team_stats(data_path: Path, team_name: str) -> dict | None:
    """Load latest stats for a team from the training data."""
    from src.utils.team_matching import find_team_in_dataframe

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
        'winrate': row[f'{prefix}_winrate'],
        'recent_winrate': row[f'{prefix}_recent_winrate'],
        'blue_winrate': row.get(f'{prefix}_blue_winrate', 0.5),
        'red_winrate': row.get(f'{prefix}_red_winrate', 0.5),
        'games': row[f'{prefix}_games'],
        'fb_rate': row[f'{prefix}_fb_rate'],
        'ft_rate': row[f'{prefix}_ft_rate'],
        'fd_rate': row[f'{prefix}_fd_rate'],
        'avg_gd15': row[f'{prefix}_avg_gd15'],
    }


def predict_match(
    model_path: Path,
    team1_stats: dict,
    team2_stats: dict,
    h2h_games: int = 0,
    h2h_winrate: float = 0.5,
    league_tier: int = 2,
    is_playoffs: bool = False,
) -> float:
    """Predict win probability for team1 (blue side)."""
    # Import here to avoid loading model if not needed
    from scripts.train_prematch_model import PrematchModel

    model = PrematchModel.load(model_path)

    features = {
        # Team 1 stats
        'team1_winrate': team1_stats['winrate'],
        'team1_recent_winrate': team1_stats['recent_winrate'],
        'team1_blue_winrate': team1_stats.get('blue_winrate', 0.5),
        'team1_games': team1_stats['games'],
        'team1_fb_rate': team1_stats['fb_rate'],
        'team1_ft_rate': team1_stats['ft_rate'],
        'team1_fd_rate': team1_stats['fd_rate'],
        'team1_avg_gd15': team1_stats['avg_gd15'],

        # Team 2 stats
        'team2_winrate': team2_stats['winrate'],
        'team2_recent_winrate': team2_stats['recent_winrate'],
        'team2_red_winrate': team2_stats.get('red_winrate', 0.5),
        'team2_games': team2_stats['games'],
        'team2_fb_rate': team2_stats['fb_rate'],
        'team2_ft_rate': team2_stats['ft_rate'],
        'team2_fd_rate': team2_stats['fd_rate'],
        'team2_avg_gd15': team2_stats['avg_gd15'],

        # Relative features
        'winrate_diff': team1_stats['winrate'] - team2_stats['winrate'],
        'recent_form_diff': team1_stats['recent_winrate'] - team2_stats['recent_winrate'],
        'fb_rate_diff': team1_stats['fb_rate'] - team2_stats['fb_rate'],
        'gd15_diff': team1_stats['avg_gd15'] - team2_stats['avg_gd15'],

        # Head-to-head
        'h2h_games': h2h_games,
        'h2h_winrate': h2h_winrate,

        # Context
        'league_tier': league_tier,
        'is_playoffs': 1 if is_playoffs else 0,
    }

    return model.predict_single(features)


def calculate_edge(our_prob: float, market_price: float) -> dict:
    """Calculate edge and expected value."""
    edge = our_prob - market_price

    # Kelly fraction
    if market_price > 0 and market_price < 1:
        odds = (1 - market_price) / market_price
        kelly = (our_prob * odds - (1 - our_prob)) / odds
        kelly = max(0, min(kelly, 0.25))  # Cap at 25%
    else:
        kelly = 0

    # Expected value per dollar
    ev = our_prob * (1 - market_price) - (1 - our_prob) * market_price

    return {
        'edge': edge,
        'kelly': kelly,
        'ev': ev,
    }


def main():
    parser = argparse.ArgumentParser(description="Predict match outcome")
    parser.add_argument("--team1", type=str, required=True, help="Team 1 name (Blue side)")
    parser.add_argument("--team2", type=str, required=True, help="Team 2 name (Red side)")
    parser.add_argument("--polymarket-odds", type=float, default=None, help="Polymarket odds for team1")
    parser.add_argument("--lookup", action="store_true", help="Look up team stats from training data")
    parser.add_argument("--model", type=str, default="models/prematch_model.pkl", help="Model path")
    parser.add_argument("--data", type=str, default="data/prematch_training.csv", help="Training data path")

    # Manual stat overrides
    parser.add_argument("--team1-winrate", type=float, default=None)
    parser.add_argument("--team1-recent", type=float, default=None)
    parser.add_argument("--team2-winrate", type=float, default=None)
    parser.add_argument("--team2-recent", type=float, default=None)

    args = parser.parse_args()

    model_path = Path(args.model)
    data_path = Path(args.data)

    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        print("Train the model first: uv run python scripts/train_prematch_model.py")
        sys.exit(1)

    # Get team stats
    if args.lookup:
        if not data_path.exists():
            print(f"Error: Data not found: {data_path}")
            sys.exit(1)

        team1_stats = load_team_stats(data_path, args.team1)
        team2_stats = load_team_stats(data_path, args.team2)

        if team1_stats is None:
            print(f"Error: Team not found in data: {args.team1}")
            print("Try a different spelling or use manual stats")
            sys.exit(1)

        if team2_stats is None:
            print(f"Error: Team not found in data: {args.team2}")
            print("Try a different spelling or use manual stats")
            sys.exit(1)
    else:
        # Use manual or default stats
        team1_stats = {
            'winrate': args.team1_winrate or 0.5,
            'recent_winrate': args.team1_recent or args.team1_winrate or 0.5,
            'blue_winrate': 0.5,
            'games': 20,
            'fb_rate': 0.5,
            'ft_rate': 0.5,
            'fd_rate': 0.5,
            'avg_gd15': 0,
        }
        team2_stats = {
            'winrate': args.team2_winrate or 0.5,
            'recent_winrate': args.team2_recent or args.team2_winrate or 0.5,
            'red_winrate': 0.5,
            'games': 20,
            'fb_rate': 0.5,
            'ft_rate': 0.5,
            'fd_rate': 0.5,
            'avg_gd15': 0,
        }

    # Make prediction
    prob = predict_match(model_path, team1_stats, team2_stats)

    # Print results
    print(f"\n{'='*60}")
    print(f"MATCH PREDICTION: {args.team1} vs {args.team2}")
    print(f"{'='*60}")

    print(f"\nTeam Stats (from {'lookup' if args.lookup else 'manual'}):")
    print(f"  {args.team1}:")
    print(f"    Win Rate:     {team1_stats['winrate']:.1%}")
    print(f"    Recent Form:  {team1_stats['recent_winrate']:.1%}")
    print(f"    Games:        {team1_stats['games']}")
    print(f"  {args.team2}:")
    print(f"    Win Rate:     {team2_stats['winrate']:.1%}")
    print(f"    Recent Form:  {team2_stats['recent_winrate']:.1%}")
    print(f"    Games:        {team2_stats['games']}")

    print(f"\n{'-'*60}")
    print(f"MODEL PREDICTION:")
    print(f"  P({args.team1} wins) = {prob:.1%}")
    print(f"  P({args.team2} wins) = {1-prob:.1%}")

    if args.polymarket_odds:
        market_price = args.polymarket_odds
        edge_info = calculate_edge(prob, market_price)

        print(f"\n{'-'*60}")
        print(f"POLYMARKET COMPARISON:")
        print(f"  Market Price:   {market_price:.1%}")
        print(f"  Our Estimate:   {prob:.1%}")
        print(f"  Edge:           {edge_info['edge']:+.1%}")
        print(f"  EV per $1:      ${edge_info['ev']:+.2f}")
        print(f"  Kelly Fraction: {edge_info['kelly']:.1%}")

        print(f"\n{'-'*60}")
        if edge_info['edge'] > 0.05:
            print(f"  RECOMMENDATION: BUY {args.team1} (Edge: {edge_info['edge']:.1%})")
        elif edge_info['edge'] < -0.05:
            print(f"  RECOMMENDATION: BUY {args.team2} (Edge: {-edge_info['edge']:.1%})")
        else:
            print(f"  RECOMMENDATION: NO TRADE (Edge too small)")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
