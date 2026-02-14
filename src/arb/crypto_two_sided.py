"""Crypto Two-Sided Arbitrage Engine.

Buys both Up and Down on crypto 5-min/15-min markets at market open
when ask_up + ask_down < 1.0 - fees, locking in structural arbitrage.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import structlog

logger = structlog.get_logger()

SYMBOL_TO_SLUG: dict[str, str] = {
    "BTCUSDT": "btc",
    "ETHUSDT": "eth",
}


def next_slots(
    now: float,
    symbols: list[str],
    timeframes: list[int],
) -> list[dict[str, object]]:
    """Return upcoming market slots for each symbol x timeframe."""
    slots: list[dict[str, object]] = []
    for symbol in symbols:
        prefix = SYMBOL_TO_SLUG.get(symbol)
        if not prefix:
            continue
        for tf in timeframes:
            current_slot = int(now // tf) * tf
            next_slot = current_slot + tf
            slug = f"{prefix}-updown-{tf // 60}m-{next_slot}"
            slots.append({
                "symbol": symbol,
                "slug": slug,
                "event_start": next_slot,
                "end_time": next_slot + tf,
                "timeframe": tf,
            })
    return slots


def compute_edge(ask_up: float, ask_down: float, fee_rate: float) -> float:
    """Compute structural edge: 1.0 - (ask_up + ask_down) - 2 * fee_rate."""
    return 1.0 - ask_up - ask_down - 2 * fee_rate
