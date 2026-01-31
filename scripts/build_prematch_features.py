#!/usr/bin/env python3
# scripts/build_prematch_features.py
"""Build pre-match features from Oracle's Elixir historical data.

For each match, computes features BEFORE the match using only past data:
- Team win rates (overall, recent form, by side)
- Head-to-head record
- League tier
- Aggregate team stats (avg gold diff, first blood rate, etc.)

Usage:
    uv run python scripts/build_prematch_features.py \
        --input data/raw/2023_LoL*.csv data/raw/2024_LoL*.csv data/raw/2025_LoL*.csv \
        --output data/prematch_training.csv
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import structlog

logger = structlog.get_logger()

# League tiers (higher = stronger competition)
LEAGUE_TIERS = {
    # Tier 1: Major regions
    'LCK': 1, 'LPL': 1, 'LEC': 1, 'LCS': 1,
    # Tier 2: Minor regions / Academy
    'PCS': 2, 'VCS': 2, 'CBLOL': 2, 'LLA': 2, 'LJL': 2,
    'LCK CL': 2, 'LEC_2': 2,
    # Tier 3: Regional leagues
    'TCL': 3, 'LCO': 3, 'NACL': 3,
    # International
    'MSI': 1, 'WLDs': 1, 'Worlds': 1,
}


def load_all_data(input_paths: list[Path]) -> pd.DataFrame:
    """Load and combine all Oracle's Elixir CSV files."""
    all_dfs = []

    for path in input_paths:
        logger.info("loading_file", path=str(path))
        df = pd.read_csv(path, low_memory=False)
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)

    # Filter to team rows only
    if 'position' in combined.columns:
        combined = combined[combined['position'] == 'team'].copy()

    logger.info("data_loaded", total_rows=len(combined))
    return combined


def compute_team_stats(df: pd.DataFrame, team: str, before_date: str, lookback_games: int = 20) -> dict:
    """Compute team statistics using only games before the given date.

    Args:
        df: Full dataset
        team: Team name to compute stats for
        before_date: Only use games before this date (YYYY-MM-DD)
        lookback_games: Number of recent games for form calculation

    Returns:
        Dictionary of team statistics
    """
    # Filter to this team's games before the date
    team_games = df[
        (df['teamname'] == team) &
        (df['date'] < before_date)
    ].copy()

    if len(team_games) == 0:
        return None

    # Sort by date
    team_games = team_games.sort_values('date')

    # Overall stats
    total_games = len(team_games)
    wins = team_games['result'].sum()
    winrate = wins / total_games if total_games > 0 else 0.5

    # Recent form (last N games)
    recent = team_games.tail(lookback_games)
    recent_winrate = recent['result'].mean() if len(recent) > 0 else 0.5

    # Side-specific stats
    blue_games = team_games[team_games['side'] == 'Blue']
    red_games = team_games[team_games['side'] == 'Red']
    blue_winrate = blue_games['result'].mean() if len(blue_games) > 0 else 0.5
    red_winrate = red_games['result'].mean() if len(red_games) > 0 else 0.5

    # Aggregate game stats
    avg_gamelength = team_games['gamelength'].mean() / 60 if 'gamelength' in team_games else 30

    # First blood rate
    fb_rate = team_games['firstblood'].mean() if 'firstblood' in team_games else 0.5

    # First tower rate
    ft_rate = team_games['firsttower'].mean() if 'firsttower' in team_games else 0.5

    # First dragon rate
    fd_rate = team_games['firstdragon'].mean() if 'firstdragon' in team_games else 0.5

    # Average gold diff at 15
    avg_gd15 = team_games['golddiffat15'].mean() if 'golddiffat15' in team_games else 0

    # Average kills
    avg_kills = team_games['teamkills'].mean() if 'teamkills' in team_games else 10

    return {
        'games_played': total_games,
        'winrate': winrate,
        'recent_winrate': recent_winrate,
        'blue_winrate': blue_winrate,
        'red_winrate': red_winrate,
        'avg_gamelength': avg_gamelength,
        'first_blood_rate': fb_rate,
        'first_tower_rate': ft_rate,
        'first_dragon_rate': fd_rate,
        'avg_golddiff15': avg_gd15,
        'avg_kills': avg_kills,
    }


def compute_h2h_record(df: pd.DataFrame, team1: str, team2: str, before_date: str) -> dict:
    """Compute head-to-head record between two teams."""
    # Find games where these teams played each other
    # This is tricky because we need to match by gameid

    team1_games = df[(df['teamname'] == team1) & (df['date'] < before_date)]
    team2_games = df[(df['teamname'] == team2) & (df['date'] < before_date)]

    # Find common gameids
    common_games = set(team1_games['gameid']) & set(team2_games['gameid'])

    if len(common_games) == 0:
        return {'h2h_games': 0, 'h2h_winrate': 0.5}

    # Get team1's results in those games
    h2h_results = team1_games[team1_games['gameid'].isin(common_games)]

    return {
        'h2h_games': len(h2h_results),
        'h2h_winrate': h2h_results['result'].mean() if len(h2h_results) > 0 else 0.5,
    }


def get_league_tier(league: str) -> int:
    """Get tier for a league (1=highest, 4=lowest/unknown)."""
    if pd.isna(league):
        return 4

    # Check for exact match first
    if league in LEAGUE_TIERS:
        return LEAGUE_TIERS[league]

    # Check for partial match
    league_upper = league.upper()
    for key, tier in LEAGUE_TIERS.items():
        if key.upper() in league_upper:
            return tier

    return 4  # Unknown league


def build_prematch_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build pre-match features for all games in the dataset.

    For each game, we compute features using ONLY data from before that game.
    This prevents data leakage.
    """
    # Get unique games
    games = df.groupby('gameid').first().reset_index()
    games = games.sort_values('date')

    logger.info("processing_games", total_games=len(games))

    features_list = []
    processed = 0
    skipped = 0

    for _, game_row in games.iterrows():
        game_id = game_row['gameid']
        game_date = game_row['date']
        league = game_row.get('league', '')

        # Get both teams for this game
        game_teams = df[df['gameid'] == game_id]
        if len(game_teams) != 2:
            skipped += 1
            continue

        blue_row = game_teams[game_teams['side'] == 'Blue'].iloc[0] if len(game_teams[game_teams['side'] == 'Blue']) > 0 else None
        red_row = game_teams[game_teams['side'] == 'Red'].iloc[0] if len(game_teams[game_teams['side'] == 'Red']) > 0 else None

        if blue_row is None or red_row is None:
            skipped += 1
            continue

        team1 = blue_row['teamname']
        team2 = red_row['teamname']

        # Compute stats for each team (using only prior data)
        team1_stats = compute_team_stats(df, team1, game_date)
        team2_stats = compute_team_stats(df, team2, game_date)

        # Skip if no historical data
        if team1_stats is None or team2_stats is None:
            skipped += 1
            continue

        # Skip if teams have very few games (unreliable)
        if team1_stats['games_played'] < 3 or team2_stats['games_played'] < 3:
            skipped += 1
            continue

        # Head-to-head
        h2h = compute_h2h_record(df, team1, team2, game_date)

        # Build feature row
        features = {
            'game_id': game_id,
            'date': game_date,
            'league': league,
            'team1': team1,
            'team2': team2,

            # Team 1 (Blue side) stats
            'team1_winrate': team1_stats['winrate'],
            'team1_recent_winrate': team1_stats['recent_winrate'],
            'team1_blue_winrate': team1_stats['blue_winrate'],
            'team1_games': team1_stats['games_played'],
            'team1_fb_rate': team1_stats['first_blood_rate'],
            'team1_ft_rate': team1_stats['first_tower_rate'],
            'team1_fd_rate': team1_stats['first_dragon_rate'],
            'team1_avg_gd15': team1_stats['avg_golddiff15'],

            # Team 2 (Red side) stats
            'team2_winrate': team2_stats['winrate'],
            'team2_recent_winrate': team2_stats['recent_winrate'],
            'team2_red_winrate': team2_stats['red_winrate'],
            'team2_games': team2_stats['games_played'],
            'team2_fb_rate': team2_stats['first_blood_rate'],
            'team2_ft_rate': team2_stats['first_tower_rate'],
            'team2_fd_rate': team2_stats['first_dragon_rate'],
            'team2_avg_gd15': team2_stats['avg_golddiff15'],

            # Relative features
            'winrate_diff': team1_stats['winrate'] - team2_stats['winrate'],
            'recent_form_diff': team1_stats['recent_winrate'] - team2_stats['recent_winrate'],
            'fb_rate_diff': team1_stats['first_blood_rate'] - team2_stats['first_blood_rate'],
            'gd15_diff': team1_stats['avg_golddiff15'] - team2_stats['avg_golddiff15'],

            # Head-to-head
            'h2h_games': h2h['h2h_games'],
            'h2h_winrate': h2h['h2h_winrate'],

            # Context
            'league_tier': get_league_tier(league),
            'is_playoffs': 1 if game_row.get('playoffs', 0) == 1 else 0,

            # Label: did team1 (blue side) win?
            'label': int(blue_row['result']),
        }

        features_list.append(features)
        processed += 1

        if processed % 1000 == 0:
            logger.info("progress", processed=processed, skipped=skipped)

    logger.info("feature_building_complete", processed=processed, skipped=skipped)

    return pd.DataFrame(features_list)


def main():
    parser = argparse.ArgumentParser(description="Build pre-match features from Oracle's Elixir data")
    parser.add_argument(
        "--input",
        type=str,
        nargs='+',
        required=True,
        help="Path(s) to Oracle's Elixir CSV file(s)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/prematch_training.csv",
        help="Output path for training CSV",
    )

    args = parser.parse_args()

    input_paths = [Path(p) for p in args.input]
    output_path = Path(args.output)

    # Check files exist
    for p in input_paths:
        if not p.exists():
            print(f"Error: File not found: {p}")
            sys.exit(1)

    # Load data
    df = load_all_data(input_paths)

    # Build features
    features_df = build_prematch_features(df)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(output_path, index=False)

    # Summary
    print(f"\n{'='*60}")
    print("PRE-MATCH FEATURES BUILT")
    print(f"{'='*60}")
    print(f"Games processed: {len(features_df):,}")
    print(f"Features: {len([c for c in features_df.columns if c not in ['game_id', 'date', 'league', 'team1', 'team2', 'label']])}")
    print(f"Team 1 (Blue) win rate: {features_df['label'].mean():.1%}")
    print(f"Output: {output_path}")
    print(f"{'='*60}")
    print(f"\nNext step:")
    print(f"  uv run python scripts/train_prematch_model.py --input {output_path}")


if __name__ == "__main__":
    main()
