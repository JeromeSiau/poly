#!/usr/bin/env python3
# scripts/update_model.py
"""Automated pipeline to update training data and retrain model.

Downloads latest Oracle's Elixir data and retrains the model.

Usage:
    # Full update (download + retrain)
    uv run python scripts/update_model.py

    # Just retrain with existing data
    uv run python scripts/update_model.py --skip-download

    # Download specific years
    uv run python scripts/update_model.py --years 2024 2025 2026

Requirements:
    pip install gdown  # For Google Drive downloads
"""

import argparse
import subprocess
import sys
from pathlib import Path

import structlog

logger = structlog.get_logger()

# Oracle's Elixir Google Drive folder
# https://drive.google.com/drive/folders/1gLSw0RLjBbtaNy0dgnGQDAZOHIgCe-HH
GDRIVE_FOLDER_ID = "1gLSw0RLjBbtaNy0dgnGQDAZOHIgCe-HH"

# Known file IDs (updated periodically - these may need manual updates)
# To find file IDs: open Google Drive, right-click file -> Get link -> extract ID
KNOWN_FILE_IDS = {
    # Format: year -> (file_id, filename)
    # These IDs may change when Oracle's Elixir updates files
    # If download fails, get new IDs from the Google Drive folder
}

# Alternative: Direct S3 URLs (may be rate-limited)
S3_URL_PATTERN = (
    "https://oracleselixir-downloadable-match-data.s3-us-west-2.amazonaws.com/"
    "{year}_LoL_esports_match_data_from_OraclesElixir.csv"
)


def check_gdown():
    """Check if gdown is installed."""
    try:
        import gdown
        return True
    except ImportError:
        return False


def download_from_gdrive_folder(output_dir: Path) -> list[Path]:
    """Download all CSV files from Google Drive folder."""
    try:
        import gdown
    except ImportError:
        print("\nError: gdown not installed")
        print("Install with: pip install gdown")
        print("\nAlternative: Download manually from:")
        print(f"  https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Download entire folder
    folder_url = f"https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}"

    logger.info("downloading_folder", url=folder_url, output=str(output_dir))
    print(f"\nDownloading Oracle's Elixir data from Google Drive...")
    print(f"Folder: {folder_url}")
    print(f"Output: {output_dir}")

    try:
        # gdown.download_folder downloads all files
        downloaded = gdown.download_folder(
            folder_url,
            output=str(output_dir),
            quiet=False,
            remaining_ok=True,  # Continue even if some files fail
        )

        if downloaded:
            logger.info("download_complete", files=len(downloaded))
            return [Path(f) for f in downloaded if f.endswith('.csv')]
        else:
            logger.warning("no_files_downloaded")
            return []

    except Exception as e:
        logger.error("download_failed", error=str(e))
        print(f"\nError downloading: {e}")
        print("\nTry manual download from:")
        print(f"  https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}")
        return []


def download_from_s3(years: list[int], output_dir: Path) -> list[Path]:
    """Try downloading from S3 (fallback)."""
    import httpx

    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []

    for year in years:
        url = S3_URL_PATTERN.format(year=year)
        output_path = output_dir / f"{year}_LoL_esports_match_data_from_OraclesElixir.csv"

        if output_path.exists():
            logger.info("file_exists", path=str(output_path))
            downloaded.append(output_path)
            continue

        logger.info("downloading", year=year, url=url)

        try:
            with httpx.Client(timeout=120.0, follow_redirects=True) as client:
                response = client.get(url)
                response.raise_for_status()
                output_path.write_bytes(response.content)
                size_mb = len(response.content) / (1024 * 1024)
                logger.info("downloaded", path=str(output_path), size_mb=f"{size_mb:.1f}")
                downloaded.append(output_path)
        except Exception as e:
            logger.warning("s3_download_failed", year=year, error=str(e))

    return downloaded


def find_existing_data(data_dir: Path, years: list[int]) -> list[Path]:
    """Find existing Oracle's Elixir CSV files."""
    files = []
    for year in years:
        pattern = f"*{year}*LoL*.csv"
        matches = list(data_dir.glob(pattern))
        if matches:
            files.extend(matches)
            logger.info("found_existing", year=year, files=[str(f) for f in matches])
    return files


def build_training_data(input_files: list[Path], output_path: Path) -> bool:
    """Build training data from Oracle's Elixir CSVs."""
    if not input_files:
        logger.error("no_input_files")
        return False

    # Import and run load_oracles_elixir
    from scripts.load_oracles_elixir import (
        load_oracles_elixir,
        transform_to_training_data,
        add_event_type_features,
    )
    import pandas as pd

    logger.info("building_training_data", files=len(input_files))

    all_dfs = []
    for input_path in input_files:
        try:
            df = load_oracles_elixir(input_path)
            all_dfs.append(df)
            logger.info("loaded", path=str(input_path), rows=len(df))
        except Exception as e:
            logger.warning("load_failed", path=str(input_path), error=str(e))

    if not all_dfs:
        logger.error("no_data_loaded")
        return False

    raw_df = pd.concat(all_dfs, ignore_index=True)
    training_df = transform_to_training_data(raw_df)
    training_df = add_event_type_features(training_df)

    # Keep only feature columns
    feature_cols = [
        'game_time_minutes', 'gold_diff', 'kill_diff', 'tower_diff',
        'dragon_diff', 'baron_diff', 'event_first_blood', 'event_tower_kill',
        'event_dragon_kill', 'event_baron_kill', 'event_inhibitor_kill',
        'event_champion_kill', 'event_state_update', 'label',
    ]
    available = [c for c in feature_cols if c in training_df.columns]
    training_df = training_df[available]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    training_df.to_csv(output_path, index=False)

    logger.info("training_data_saved", path=str(output_path), rows=len(training_df))
    return True


def build_prematch_data(input_files: list[Path], output_path: Path) -> bool:
    """Build prematch training data from Oracle's Elixir CSVs."""
    from scripts.build_prematch_features import main as build_prematch_main

    if not input_files:
        return False

    # Build using the existing script
    import sys
    orig_argv = sys.argv
    sys.argv = [
        'build_prematch_features.py',
        '--input', *[str(f) for f in input_files],
        '--output', str(output_path),
    ]

    try:
        build_prematch_main()
        return True
    except SystemExit:
        return output_path.exists()
    finally:
        sys.argv = orig_argv


def train_model(training_data: Path, model_output: Path) -> bool:
    """Train the prematch model."""
    from scripts.train_prematch_model import PrematchModel, FEATURE_COLS
    import pandas as pd

    if not training_data.exists():
        logger.error("training_data_not_found", path=str(training_data))
        return False

    df = pd.read_csv(training_data)
    X = df[FEATURE_COLS].copy().fillna(0)
    y = df['label']

    logger.info("training_model", samples=len(df), features=len(FEATURE_COLS))

    model = PrematchModel()
    metrics = model.train(X, y, validation_split=0.2)

    model_output.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_output)

    print(f"\n{'='*60}")
    print("MODEL TRAINED")
    print(f"{'='*60}")
    print(f"Samples:    {len(df):,}")
    print(f"Train AUC:  {metrics['train_auc']:.4f}")
    print(f"Val AUC:    {metrics['val_auc']:.4f}")
    print(f"CV AUC:     {metrics['cv_auc_mean']:.4f} (+/- {metrics['cv_auc_std']:.4f})")
    print(f"{'='*60}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Update model with latest data")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download, use existing data",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2024, 2025, 2026],
        help="Years to include",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Raw data directory",
    )
    parser.add_argument(
        "--use-s3",
        action="store_true",
        help="Try S3 instead of Google Drive",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    prematch_data = Path("data/prematch_training.csv")
    model_path = Path("models/prematch_model.pkl")

    print(f"\n{'='*60}")
    print("MODEL UPDATE PIPELINE")
    print(f"{'='*60}")
    print(f"Years: {args.years}")
    print(f"Data dir: {data_dir}")

    # Step 1: Get data files
    if args.skip_download:
        print("\n[1/3] Finding existing data...")
        input_files = find_existing_data(data_dir, args.years)
    elif args.use_s3:
        print("\n[1/3] Downloading from S3...")
        input_files = download_from_s3(args.years, data_dir)
    else:
        print("\n[1/3] Downloading from Google Drive...")
        if not check_gdown():
            print("gdown not installed, trying S3 fallback...")
            input_files = download_from_s3(args.years, data_dir)
        else:
            input_files = download_from_gdrive_folder(data_dir)

        # Filter to requested years
        input_files = [
            f for f in input_files
            if any(str(y) in f.name for y in args.years)
        ]

    if not input_files:
        # Try finding existing files
        input_files = find_existing_data(data_dir, args.years)

    if not input_files:
        print("\nError: No data files found!")
        print(f"\nDownload manually from:")
        print(f"  https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}")
        print(f"\nSave to: {data_dir}/")
        sys.exit(1)

    print(f"\nFound {len(input_files)} data files:")
    for f in input_files:
        print(f"  - {f.name}")

    # Step 2: Build training data
    print("\n[2/3] Building prematch training data...")
    if not build_prematch_data(input_files, prematch_data):
        print("Error building training data!")
        sys.exit(1)

    # Step 3: Train model
    print("\n[3/3] Training model...")
    if not train_model(prematch_data, model_path):
        print("Error training model!")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("UPDATE COMPLETE")
    print(f"{'='*60}")
    print(f"Training data: {prematch_data}")
    print(f"Model:         {model_path}")
    print(f"\nNext: Run backtest to validate")
    print(f"  uv run python scripts/backtest_polymarket.py --limit 100")


if __name__ == "__main__":
    main()
