"""Feature engineering for TD maker ML models."""

from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# Feature columns (order matters — saved models depend on column order)
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    # Book prices
    "bid_up", "ask_up", "bid_down", "ask_down",
    # Spreads
    "spread_up", "spread_down",
    # Book sizes
    "bid_size_up", "ask_size_up", "bid_size_down", "ask_size_down",
    # Chainlink
    "chainlink_price", "dir_move_pct", "abs_move_pct",
    # Market context
    "market_volume_usd",  # 0.0 when unknown (column added 2026-02-19)
    "prev_resolved_up",   # 1.0=prev slot UP, 0.0=prev slot DOWN, 0.5=unknown
    # Timing
    "minutes_into_slot",
    "hour_utc", "day_of_week",
    # Derived
    "spread_ratio",
    "bid_imbalance",
    "fav_bid",
    "move_velocity",
    "book_pressure",
    # Trend (computed from consecutive snapshots within a slot)
    "bid_trend_30s",
    "bid_trend_2m",
    "ask_trend_30s",
    "spread_trend",
]

EXIT_FEATURE_COLS = [
    *FEATURE_COLS,
    # Exit-specific: position context
    "entry_price",
    "bid_max",           # highest bid since fill
    "bid_drop",          # bid_max - current bid
    "bid_drop_pct",      # bid_drop / bid_max
    "minutes_remaining",
    "minutes_held",
    "pnl_unrealized",    # bid_up - entry_price (per share)
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features to raw snapshot data."""
    df = df.copy()

    # spread_ratio: how skewed are spreads between Up and Down
    df["spread_ratio"] = df["spread_up"] / df["spread_down"].clip(lower=0.001)

    # bid_imbalance: which side the market favours
    df["bid_imbalance"] = df["bid_up"] - df["bid_down"]

    # fav_bid: highest bid (the favourite side's price)
    df["fav_bid"] = df[["bid_up", "bid_down"]].max(axis=1)

    # move_velocity: how fast price moved per minute
    df["move_velocity"] = df["dir_move_pct"] / df["minutes_into_slot"].clip(lower=0.5)

    # book_pressure: relative size advantage of Up vs Down bids
    total_bid_sz = (df["bid_size_up"] + df["bid_size_down"]).clip(lower=0.01)
    df["book_pressure"] = (df["bid_size_up"] - df["bid_size_down"]) / total_bid_sz

    # market_volume_usd: fill missing with 0 (column added 2026-02-19, low coverage)
    if "market_volume_usd" in df.columns:
        df["market_volume_usd"] = df["market_volume_usd"].fillna(0.0)

    # prev_resolved_up: cast bool→float, fill unknown with 0.5 (neutral)
    if "prev_resolved_up" in df.columns:
        df["prev_resolved_up"] = df["prev_resolved_up"].astype("float64").fillna(0.5)

    return df


def compute_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute trend features from consecutive snapshots within each slot.

    Snapshots are captured every ~30s.  For each snapshot we look back to
    find the nearest measurement ~30s and ~120s ago (within the same slot)
    and compute deltas for bid_up, ask_up, and spread_up.
    """
    df = df.copy()
    df.sort_values(["symbol", "slot_ts", "captured_at"], inplace=True)

    group = df.groupby(["symbol", "slot_ts"])

    # shift(1) = previous snapshot (~30s ago), shift(4) = ~2min ago.
    df["bid_trend_30s"] = df["bid_up"] - group["bid_up"].shift(1)
    df["bid_trend_2m"] = df["bid_up"] - group["bid_up"].shift(4)
    df["ask_trend_30s"] = df["ask_up"] - group["ask_up"].shift(1)
    df["spread_trend"] = df["spread_up"] - group["spread_up"].shift(2)

    return df


def compute_exit_features(df: pd.DataFrame, entry_price: float = 0.75) -> pd.DataFrame:
    """Compute position-specific features for exit model.

    For each slot, identifies the simulated fill point (first snapshot
    where bid_up >= entry_price) and computes running features for all
    subsequent snapshots.
    """
    df = df.copy()
    df.sort_values(["symbol", "slot_ts", "captured_at"], inplace=True)

    results = []
    for (sym, slot_ts), group in df.groupby(["symbol", "slot_ts"]):
        rows = group.reset_index(drop=True)
        resolved_up = rows["resolved_up"].iloc[0]

        # Find fill point: first snapshot where bid_up >= entry_price.
        fill_idx = None
        for i, row in rows.iterrows():
            if row["bid_up"] >= entry_price:
                fill_idx = i
                break

        if fill_idx is None:
            continue  # no fill in this slot

        fill_price = rows.loc[fill_idx, "bid_up"]
        fill_minutes = rows.loc[fill_idx, "minutes_into_slot"]

        # Take all snapshots after fill (including fill itself for context).
        post_fill = rows.loc[fill_idx:].copy()
        if len(post_fill) < 2:
            continue

        # Running bid_max since fill.
        post_fill["bid_max"] = post_fill["bid_up"].cummax()
        post_fill["entry_price"] = fill_price
        post_fill["bid_drop"] = post_fill["bid_max"] - post_fill["bid_up"]
        post_fill["bid_drop_pct"] = post_fill["bid_drop"] / post_fill["bid_max"].clip(lower=0.01)
        post_fill["minutes_remaining"] = 15.0 - post_fill["minutes_into_slot"]
        post_fill["minutes_held"] = post_fill["minutes_into_slot"] - fill_minutes
        post_fill["pnl_unrealized"] = post_fill["bid_up"] - fill_price

        # Skip the fill snapshot itself (no exit decision at fill time).
        post_fill = post_fill.iloc[1:]
        if len(post_fill) == 0:
            continue

        results.append(post_fill)

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)
