"""Data loading and splitting for TD maker ML models."""

from __future__ import annotations

import pandas as pd
import structlog

from src.ml.features import compute_trend_features

logger = structlog.get_logger()


async def load_snapshots(db_url: str, min_minutes: float = 4.0,
                         max_minutes: float = 10.0) -> pd.DataFrame:
    """Load snapshots for entry model training (filtered to minute window).

    Loads ALL snapshots per slot so that trend features can be computed
    from consecutive measurements.  After trend computation the rows
    are filtered to [min_minutes, max_minutes].
    """
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(db_url, echo=False)
    query = text("""
        SELECT
            s.symbol, s.slot_ts, s.minutes_into_slot, s.captured_at,
            s.bid_up, s.ask_up, s.bid_down, s.ask_down,
            s.bid_size_up, s.ask_size_up, s.bid_size_down, s.ask_size_down,
            s.spread_up, s.spread_down,
            s.chainlink_price, s.dir_move_pct, s.abs_move_pct,
            s.market_volume_usd,
            s.hour_utc, s.day_of_week,
            r.resolved_up, r.prev_resolved_up
        FROM slot_snapshots s
        JOIN slot_resolutions r ON s.symbol = r.symbol AND s.slot_ts = r.slot_ts
        WHERE r.resolved_up IS NOT NULL
        ORDER BY s.symbol, s.slot_ts, s.captured_at
    """)

    async with engine.connect() as conn:
        result = await conn.execute(query)
        rows = result.fetchall()
        cols = result.keys()

    await engine.dispose()

    df = pd.DataFrame(rows, columns=list(cols))
    logger.info("loaded_raw_snapshots", rows=len(df),
                symbols=sorted(df["symbol"].unique().tolist()) if len(df) > 0 else [])

    if df.empty:
        return df

    # Compute trend features from consecutive snapshots within each slot.
    df = compute_trend_features(df)

    # Filter to the target minute window.
    before = len(df)
    df = df[(df["minutes_into_slot"] >= min_minutes) &
            (df["minutes_into_slot"] <= max_minutes)]
    logger.info("filtered_to_window", before=before, after=len(df),
                min_minutes=min_minutes, max_minutes=max_minutes)

    return df


async def load_all_snapshots(db_url: str) -> pd.DataFrame:
    """Load ALL snapshots for resolved slots (no minute filtering).

    Used by the exit model which needs all snapshots to simulate
    fill points and compute running position features.
    """
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(db_url, echo=False)
    query = text("""
        SELECT
            s.symbol, s.slot_ts, s.minutes_into_slot, s.captured_at,
            s.bid_up, s.ask_up, s.bid_down, s.ask_down,
            s.bid_size_up, s.ask_size_up, s.bid_size_down, s.ask_size_down,
            s.spread_up, s.spread_down,
            s.chainlink_price, s.dir_move_pct, s.abs_move_pct,
            s.market_volume_usd,
            s.hour_utc, s.day_of_week,
            r.resolved_up, r.prev_resolved_up
        FROM slot_snapshots s
        JOIN slot_resolutions r ON s.symbol = r.symbol AND s.slot_ts = r.slot_ts
        WHERE r.resolved_up IS NOT NULL
        ORDER BY s.symbol, s.slot_ts, s.captured_at
    """)

    async with engine.connect() as conn:
        result = await conn.execute(query)
        rows = result.fetchall()
        cols = result.keys()

    await engine.dispose()

    df = pd.DataFrame(rows, columns=list(cols))
    logger.info("loaded_raw_snapshots", rows=len(df),
                symbols=sorted(df["symbol"].unique().tolist()) if len(df) > 0 else [])
    return df


def deduplicate_per_slot(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one snapshot per (symbol, slot_ts) — the latest within the window."""
    return (
        df.sort_values("minutes_into_slot", ascending=False)
        .groupby(["symbol", "slot_ts"])
        .first()
        .reset_index()
    )


def temporal_split(df: pd.DataFrame, val_days: int = 3
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by time: last val_days of slot_ts data go to validation."""
    max_ts = df["slot_ts"].max()
    cutoff = max_ts - val_days * 86400
    train = df[df["slot_ts"] <= cutoff].copy()
    val = df[df["slot_ts"] > cutoff].copy()
    logger.info("temporal_split",
                train_rows=len(train), val_rows=len(val),
                cutoff_ts=cutoff)
    return train, val
