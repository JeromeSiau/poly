#!/usr/bin/env python3
# scripts/download_oracles_elixir.py
"""Download Oracle's Elixir LoL data.

The direct S3 links may not work. Manual download options:

1. Oracle's Elixir (official):
   https://oracleselixir.com/tools/downloads
   → Click on the year you want → Download CSV

2. Kaggle (2024 data, requires free account):
   https://www.kaggle.com/datasets/barthetur/league-of-legends-2024-competitive-game-dataset

Usage (after manual download):
    uv run python scripts/load_oracles_elixir.py \\
        --input data/raw/2024_LoL_esports_match_data.csv \\
        --output data/lol_training.csv
"""

import argparse
import sys
from pathlib import Path

import httpx
import structlog

logger = structlog.get_logger()

# Oracle's Elixir S3 bucket pattern (discovered from GitHub repos using their data)
# Pattern: https://oracleselixir-downloadable-match-data.s3-us-west-2.amazonaws.com/{year}_LoL_esports_match_data_from_OraclesElixir.csv
ORACLES_ELIXIR_URL_PATTERN = (
    "https://oracleselixir-downloadable-match-data.s3-us-west-2.amazonaws.com/"
    "{year}_LoL_esports_match_data_from_OraclesElixir.csv"
)

# Alternative: Kaggle has 2024 data
KAGGLE_INFO = """
Alternative: Download from Kaggle (requires account):
https://www.kaggle.com/datasets/barthetur/league-of-legends-2024-competitive-game-dataset
"""


def download_data(year: int, output_dir: Path) -> Path:
    """Download Oracle's Elixir data for a specific year."""
    url = ORACLES_ELIXIR_URL_PATTERN.format(year=year)
    output_path = output_dir / f"{year}_LoL_esports_match_data_from_OraclesElixir.csv"

    logger.info("downloading", url=url)

    try:
        with httpx.Client(timeout=120.0, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()

            output_path.write_bytes(response.content)

            size_mb = len(response.content) / (1024 * 1024)
            logger.info("download_complete", path=str(output_path), size_mb=f"{size_mb:.1f}")

            return output_path

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 403:
            logger.error("access_denied",
                        msg="Oracle's Elixir may have changed their URL pattern")
            print(f"\nError: Cannot download from S3 bucket (403 Forbidden)")
            print(f"\nPlease download manually from: https://oracleselixir.com/tools/downloads")
            print(f"Save the file to: {output_path}")
            print(KAGGLE_INFO)
        elif e.response.status_code == 404:
            logger.error("not_found", year=year)
            print(f"\nError: Data for year {year} not found")
            print(f"Try a different year or download manually from:")
            print(f"  https://oracleselixir.com/tools/downloads")
        else:
            logger.error("download_failed", status=e.response.status_code)
            print(f"\nError downloading: {e}")
        sys.exit(1)

    except Exception as e:
        logger.error("download_error", error=str(e))
        print(f"\nError: {e}")
        print(f"\nPlease download manually from: https://oracleselixir.com/tools/downloads")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Download Oracle's Elixir LoL data")
    parser.add_argument(
        "--year",
        type=int,
        default=2024,
        help="Year to download (2014-2025)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Output directory",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = download_data(args.year, output_dir)

    print(f"\n{'='*60}")
    print("DOWNLOAD COMPLETE")
    print(f"{'='*60}")
    print(f"File: {output_path}")
    print(f"\nNext step:")
    print(f"  uv run python scripts/load_oracles_elixir.py \\")
    print(f"      --input {output_path} \\")
    print(f"      --output data/lol_training.csv")


if __name__ == "__main__":
    main()
