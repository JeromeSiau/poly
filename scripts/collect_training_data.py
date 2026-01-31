#!/usr/bin/env python3
# scripts/collect_training_data.py
"""Collect historical match data from PandaScore for ML training.

Usage:
    uv run python scripts/collect_training_data.py --game lol --matches 500 --output data/lol_training.csv
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import structlog

from config.settings import settings
from src.ml.data_collector import HistoricalDataCollector
from src.ml.features import FeatureExtractor

logger = structlog.get_logger()


async def collect_data(
    game: str,
    num_matches: int,
    output_path: Path,
) -> None:
    """Collect historical data and save to CSV."""
    collector = HistoricalDataCollector(api_key=settings.PANDASCORE_API_KEY)
    extractor = FeatureExtractor(game=game)

    await collector.connect()

    all_events = []
    matches_processed = 0
    page = 1

    try:
        while matches_processed < num_matches:
            logger.info(
                "fetching_matches",
                page=page,
                processed=matches_processed,
                target=num_matches,
            )

            matches = await collector.fetch_past_matches(
                game=game,
                limit=100,
                page=page,
            )

            if not matches:
                logger.warning("no_more_matches")
                break

            for match in matches:
                if matches_processed >= num_matches:
                    break

                try:
                    events = await collector.fetch_match_events(
                        game=game,
                        game_id=match.match_id,
                    )

                    # Set winner for all events
                    for event in events:
                        event.winner = match.winner

                    all_events.extend(events)
                    matches_processed += 1

                    logger.debug(
                        "match_processed",
                        match_id=match.match_id,
                        events=len(events),
                    )

                except Exception as e:
                    logger.warning(
                        "match_fetch_failed",
                        match_id=match.match_id,
                        error=str(e),
                    )

                # Rate limiting
                await asyncio.sleep(0.5)

            page += 1

    finally:
        await collector.disconnect()

    if not all_events:
        logger.error("no_events_collected")
        return

    logger.info("extracting_features", events=len(all_events))
    df = extractor.extract_batch(all_events)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info(
        "data_saved",
        path=str(output_path),
        rows=len(df),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Collect historical match data for ML training"
    )
    parser.add_argument(
        "--game",
        type=str,
        default="lol",
        choices=["lol", "csgo", "dota2"],
        help="Game to collect data for",
    )
    parser.add_argument(
        "--matches",
        type=int,
        default=100,
        help="Number of matches to collect",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/training_data.csv",
        help="Output CSV path",
    )

    args = parser.parse_args()

    asyncio.run(
        collect_data(
            game=args.game,
            num_matches=args.matches,
            output_path=Path(args.output),
        )
    )


if __name__ == "__main__":
    main()
