#!/usr/bin/env python3
# scripts/load_oracles_elixir.py
"""Load and transform Oracle's Elixir data for ML training.

Oracle's Elixir provides free historical LoL pro match data.
Download from: https://oracleselixir.com/tools/downloads

Usage:
    # Download the match data CSV from Oracle's Elixir, then:
    uv run python scripts/load_oracles_elixir.py \
        --input data/raw/2024_LoL_esports_match_data_from_OraclesElixir.csv \
        --output data/lol_training.csv

This script:
1. Loads Oracle's Elixir match data CSV
2. Transforms it into event-style training data
3. Generates features compatible with our ML pipeline
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import structlog

logger = structlog.get_logger()


def load_oracles_elixir(input_path: Path) -> pd.DataFrame:
    """Load raw Oracle's Elixir CSV data.

    Oracle's Elixir provides team-level data per game with columns like:
    - gameid, teamname, side (Blue/Red)
    - result (1=win, 0=loss)
    - gamelength (seconds)
    - kills, deaths, assists
    - towers, dragons, barons, elders
    - goldat10, goldat15, xpat10, xpat15, etc.
    - golddiffat10, golddiffat15, etc.
    """
    logger.info("loading_data", path=str(input_path))

    df = pd.read_csv(input_path)

    logger.info("data_loaded", rows=len(df), columns=list(df.columns)[:20])

    return df


def transform_to_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """Transform Oracle's Elixir data to our training format.

    We create synthetic "events" at different game times (10, 15, 20, 25 min)
    using the snapshot data Oracle's Elixir provides.

    Output format matches what FeatureExtractor expects:
    - game_time_minutes
    - gold_diff, kill_diff, tower_diff, dragon_diff, baron_diff
    - event_type (synthetic)
    - label (1 if team won, 0 if lost)
    """
    training_rows = []

    # Oracle's Elixir has 12 rows per game: 10 players + 2 team summary rows
    # Filter to team rows only (position == 'team')
    if 'position' in df.columns:
        df = df[df['position'] == 'team'].copy()
        logger.info("filtered_to_team_rows", rows=len(df))

    # Check available columns
    available_cols = set(df.columns)
    logger.info("available_columns", cols=list(available_cols)[:30])

    # Required columns mapping
    gameid_col = 'gameid'
    result_col = 'result'
    side_col = 'side'

    if gameid_col not in available_cols or result_col not in available_cols:
        logger.error("missing_required_columns",
                    available=list(available_cols)[:20])
        raise ValueError("Cannot find required columns (gameid, result)")

    # Time snapshots available in Oracle's Elixir
    time_points = []
    for t in [10, 15, 20, 25]:
        if f'golddiffat{t}' in available_cols:
            time_points.append(t)

    logger.info("time_points_found", times=time_points)

    # Group by game
    games = df.groupby(gameid_col)

    processed = 0
    skipped = 0

    for game_id, game_df in games:
        if len(game_df) != 2:
            skipped += 1
            continue

        # Get both teams (Blue and Red side)
        try:
            blue_rows = game_df[game_df[side_col] == 'Blue']
            red_rows = game_df[game_df[side_col] == 'Red']

            if len(blue_rows) != 1 or len(red_rows) != 1:
                skipped += 1
                continue

            blue_team = blue_rows.iloc[0]
            red_team = red_rows.iloc[0]
        except (IndexError, KeyError):
            skipped += 1
            continue

        # Get game length in minutes (gamelength is in seconds)
        game_length = blue_team.get('gamelength', 1800) / 60  # Convert to minutes

        # Create events at each time point
        for t in time_points:
            if t > game_length:
                continue  # Skip if game ended before this time

            # Get gold diff (from blue team perspective)
            # Oracle's Elixir has golddiffat10, golddiffat15, etc.
            gold_diff_col = f'golddiffat{t}'
            gold_diff = blue_team.get(gold_diff_col, 0)
            if pd.isna(gold_diff):
                gold_diff = 0
            gold_diff = float(gold_diff)

            # Get kill diff (killsat10 is blue team kills, opp_killsat10 is opponent kills)
            kills_col = f'killsat{t}'
            opp_kills_col = f'opp_killsat{t}'
            blue_kills = blue_team.get(kills_col, 0) or 0
            red_kills = blue_team.get(opp_kills_col, 0) or 0
            kill_diff = float(blue_kills) - float(red_kills)

            # End-of-game stats for objectives (scale by game progress)
            game_progress = t / game_length if game_length > 0 else 0.5

            # Dragons
            blue_dragons = blue_team.get('dragons', 0) or 0
            red_dragons = blue_team.get('opp_dragons', 0) or 0
            dragon_diff = int((float(blue_dragons) - float(red_dragons)) * game_progress)

            # Towers
            blue_towers = blue_team.get('towers', 0) or 0
            red_towers = blue_team.get('opp_towers', 0) or 0
            tower_diff = int((float(blue_towers) - float(red_towers)) * game_progress)

            # Barons
            blue_barons = blue_team.get('barons', 0) or 0
            red_barons = blue_team.get('opp_barons', 0) or 0
            baron_diff = int((float(blue_barons) - float(red_barons)) * game_progress)

            # Label: did blue team win?
            label = int(blue_team[result_col])

            # Create synthetic event type based on game state
            if baron_diff > 0:
                event_type = "baron_kill"
            elif dragon_diff > 0:
                event_type = "dragon_kill"
            elif tower_diff > 0:
                event_type = "tower_destroyed"
            elif kill_diff > 0:
                event_type = "kill"
            else:
                event_type = "state_snapshot"

            # Build row
            row = {
                'game_id': game_id,
                'game_time_minutes': float(t),
                'gold_diff': float(gold_diff),
                'kill_diff': float(kill_diff),
                'tower_diff': float(tower_diff),
                'dragon_diff': float(dragon_diff),
                'baron_diff': float(baron_diff),
                'event_type': event_type,
                'label': label,
                # Normalized features
                'gold_diff_normalized': float(gold_diff) / t if t > 0 else 0,
                'kill_diff_normalized': float(kill_diff) / t if t > 0 else 0,
            }

            training_rows.append(row)

        processed += 1

        if processed % 500 == 0:
            logger.info("progress", processed=processed)

    logger.info("transformation_complete",
               games_processed=processed,
               games_skipped=skipped,
               training_rows=len(training_rows))

    return pd.DataFrame(training_rows)


def add_event_type_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add one-hot encoded event type features."""
    event_types = ['kill', 'tower_destroyed', 'dragon_kill', 'baron_kill',
                   'elder_kill', 'inhibitor_destroyed', 'ace', 'state_snapshot']

    for et in event_types:
        df[f'event_{et}'] = (df['event_type'] == et).astype(int)

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Load Oracle's Elixir data for ML training"
    )
    parser.add_argument(
        "--input",
        type=str,
        nargs='+',
        required=True,
        help="Path(s) to Oracle's Elixir CSV file(s) - can specify multiple",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/lol_training.csv",
        help="Output path for training CSV",
    )

    args = parser.parse_args()

    input_paths = [Path(p) for p in args.input]
    output_path = Path(args.output)

    # Check all files exist
    for input_path in input_paths:
        if not input_path.exists():
            print(f"\nError: Input file not found: {input_path}")
            print("\nDownload data from: https://oracleselixir.com/tools/downloads")
            sys.exit(1)

    # Load and combine all files
    all_dfs = []
    for input_path in input_paths:
        df = load_oracles_elixir(input_path)
        all_dfs.append(df)
        logger.info("loaded_file", path=str(input_path), rows=len(df))

    raw_df = pd.concat(all_dfs, ignore_index=True)
    logger.info("combined_data", total_rows=len(raw_df), files=len(input_paths))
    training_df = transform_to_training_data(raw_df)

    # Add event type features
    training_df = add_event_type_features(training_df)

    # Remove non-feature columns before saving
    feature_cols = [
        'game_time_minutes',
        'gold_diff', 'gold_diff_normalized',
        'kill_diff', 'kill_diff_normalized',
        'tower_diff', 'dragon_diff', 'baron_diff',
        'event_kill', 'event_tower_destroyed', 'event_dragon_kill',
        'event_baron_kill', 'event_elder_kill', 'event_inhibitor_destroyed',
        'event_ace', 'event_state_snapshot',
        'label',
    ]

    # Keep only columns that exist
    output_cols = [c for c in feature_cols if c in training_df.columns]
    output_df = training_df[output_cols]

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    # Summary
    print(f"\n{'='*60}")
    print("ORACLE'S ELIXIR DATA LOADED")
    print(f"{'='*60}")
    print(f"Input:         {input_path}")
    print(f"Output:        {output_path}")
    print(f"Training rows: {len(output_df):,}")
    print(f"Features:      {len(output_cols) - 1}")  # -1 for label
    print(f"Win rate:      {output_df['label'].mean():.1%}")
    print(f"{'='*60}")
    print(f"\nNext steps:")
    print(f"  1. Train:    uv run python scripts/train_model.py --data {output_path}")
    print(f"  2. Validate: uv run python scripts/validate_model.py --model models/lol_impact.pkl --data {output_path}")


if __name__ == "__main__":
    main()
