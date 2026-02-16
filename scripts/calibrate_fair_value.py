#!/usr/bin/env python3
"""Calibrate empirical fair value table for 15-min crypto binary markets.

Fetches BTCUSDT + ETHUSDT 1-min klines from Binance REST API for
Oct 2025 - Feb 2026, splits into non-overlapping 15-min windows,
and computes P(win | dir_move_bucket, minutes_remaining).

Output: a Python dict literal suitable for pasting into src/utils/fair_value.py.

Usage:
    ./run scripts/calibrate_fair_value.py
"""

from __future__ import annotations

import bisect
import time
from collections import defaultdict
from datetime import datetime, timezone

import httpx

# ---------- Config ----------

SYMBOLS = ["BTCUSDT", "ETHUSDT"]
INTERVAL = "1m"
KLINE_LIMIT = 1000  # max per Binance request

# Date range: Oct 1 2025 00:00 UTC - Feb 16 2026 00:00 UTC
START_TS_MS = int(datetime(2025, 10, 1, tzinfo=timezone.utc).timestamp() * 1000)
END_TS_MS = int(datetime(2026, 2, 16, tzinfo=timezone.utc).timestamp() * 1000)

# Bucket edges for directional move (%) — must match src/utils/fair_value._MOVE_EDGES
from src.utils.fair_value import _MOVE_EDGES as MOVE_EDGES

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"


# ---------- Fetch ----------


def fetch_klines(
    client: httpx.Client, symbol: str, start_ms: int, end_ms: int
) -> list[list]:
    """Fetch all 1-min klines for symbol in [start_ms, end_ms) via pagination."""
    all_klines: list[list] = []
    cursor = start_ms

    while cursor < end_ms:
        params = {
            "symbol": symbol,
            "interval": INTERVAL,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": KLINE_LIMIT,
        }
        resp = client.get(BINANCE_KLINES_URL, params=params)
        resp.raise_for_status()
        batch = resp.json()

        if not batch:
            break

        all_klines.extend(batch)

        # Next page starts after last kline open_time + 1ms
        last_open = batch[-1][0]
        cursor = last_open + 60_000  # next minute

        if len(batch) < KLINE_LIMIT:
            break

        # Be polite to Binance
        time.sleep(0.15)

    return all_klines


def kline_close(k: list) -> float:
    """Extract close price from a Binance kline array."""
    return float(k[4])


# ---------- Process ----------


def build_observations(
    klines: list[list],
) -> list[tuple[float, int, int]]:
    """Split klines into 15-min windows and extract observations.

    Returns list of (dir_move_pct, minutes_remaining, win).
    """
    observations: list[tuple[float, int, int]] = []

    # Split into non-overlapping 15-candle windows
    n_windows = len(klines) // 15
    for w in range(n_windows):
        window = klines[w * 15 : (w + 1) * 15]
        if len(window) < 15:
            continue

        open_price = kline_close(window[0])  # first candle close = slot open ref
        final_price = kline_close(window[14])  # last candle close = resolution

        if open_price == 0:
            continue

        win = 1 if final_price > open_price else 0

        for t in range(1, 15):  # t=1..14
            current_price = kline_close(window[t])
            dir_move = (current_price - open_price) / open_price * 100
            minutes_remaining = 15 - t
            observations.append((dir_move, minutes_remaining, win))

    return observations


def bucket_index(dir_move: float) -> int:
    """Map dir_move to bucket index using bisect_right on MOVE_EDGES."""
    return bisect.bisect_right(MOVE_EDGES, dir_move)


def compute_table(
    observations: list[tuple[float, int, int]],
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], int]]:
    """Compute P(win) per (bucket_index, minutes_remaining).

    Returns (table, counts) where table maps keys to P(win) and counts
    maps keys to sample size.
    """
    wins: dict[tuple[int, int], int] = defaultdict(int)
    counts: dict[tuple[int, int], int] = defaultdict(int)

    for dir_move, mins_rem, win in observations:
        key = (bucket_index(dir_move), mins_rem)
        wins[key] += win
        counts[key] += 1

    table: dict[tuple[int, int], float] = {}
    for key in sorted(counts.keys()):
        table[key] = round(wins[key] / counts[key], 4)

    return table, counts


# ---------- Main ----------


def main() -> None:
    print("Calibrating fair value table from Binance 1-min klines...")
    print(f"Date range: {datetime.fromtimestamp(START_TS_MS / 1000, tz=timezone.utc)}"
          f" -> {datetime.fromtimestamp(END_TS_MS / 1000, tz=timezone.utc)}")
    print(f"Symbols: {SYMBOLS}")
    print()

    all_observations: list[tuple[float, int, int]] = []

    with httpx.Client(timeout=30.0) as client:
        for symbol in SYMBOLS:
            print(f"Fetching {symbol}...")
            klines = fetch_klines(client, symbol, START_TS_MS, END_TS_MS)
            print(f"  Got {len(klines)} candles")

            obs = build_observations(klines)
            print(f"  Generated {len(obs)} observations from {len(klines) // 15} windows")
            all_observations.extend(obs)

    print(f"\nTotal observations: {len(all_observations)}")

    table, counts = compute_table(all_observations)

    # Print the table as a Python dict literal
    print("\n# ---------- TABLE (paste into src/utils/fair_value.py) ----------\n")
    print("_TABLE: dict[tuple[int, int], float] = {")
    for (b, m), p in sorted(table.items()):
        count = counts[(b, m)]
        print(f"    ({b:2d}, {m:2d}): {p:.4f},  # n={count}")
    print("}")

    # Print stats
    print("\n# ---------- Stats ----------\n")
    print(f"Total 15-min windows: {len(all_observations) // 14}")

    # Per-bucket stats
    bucket_counts: dict[int, int] = defaultdict(int)
    bucket_wins: dict[int, int] = defaultdict(int)
    for dir_move, _, win in all_observations:
        b = bucket_index(dir_move)
        bucket_counts[b] += 1
        bucket_wins[b] += win

    edge_labels = ["< -0.50"] + [
        f"[{MOVE_EDGES[i]:.2f}, {MOVE_EDGES[i+1]:.2f})"
        for i in range(len(MOVE_EDGES) - 1)
    ] + [f">= {MOVE_EDGES[-1]:.2f}"]

    print(f"\n{'Bucket':>5}  {'Range':>20}  {'Count':>8}  {'Win%':>7}")
    print("-" * 50)
    for b in sorted(bucket_counts.keys()):
        label = edge_labels[b] if b < len(edge_labels) else f"bucket {b}"
        n = bucket_counts[b]
        wr = bucket_wins[b] / n * 100 if n > 0 else 0
        print(f"  {b:>3}  {label:>20}  {n:>8}  {wr:>6.1f}%")


if __name__ == "__main__":
    main()
