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


def compute_sweep(
    up_asks: list[tuple[float, float]],
    down_asks: list[tuple[float, float]],
    fee_rate: float,
    max_budget: float,
) -> tuple[float, float, float]:
    """Compute how much to spend on each side while maintaining edge.

    Returns (up_budget_usd, down_budget_usd, best_edge).
    """
    if not up_asks or not down_asks:
        return 0.0, 0.0, 0.0

    best_up = up_asks[0][0]
    best_down = down_asks[0][0]
    edge = compute_edge(best_up, best_down, fee_rate)

    if edge <= 0:
        return 0.0, 0.0, edge

    up_depth_usd = 0.0
    for price, shares in up_asks:
        if compute_edge(price, best_down, fee_rate) <= 0:
            break
        up_depth_usd += price * shares

    down_depth_usd = 0.0
    for price, shares in down_asks:
        if compute_edge(best_up, price, fee_rate) <= 0:
            break
        down_depth_usd += price * shares

    total_depth = up_depth_usd + down_depth_usd
    if total_depth <= 0:
        return 0.0, 0.0, 0.0

    budget = min(max_budget, total_depth)
    up_ratio = up_depth_usd / total_depth
    up_budget = budget * up_ratio
    down_budget = budget * (1 - up_ratio)

    return up_budget, down_budget, edge
