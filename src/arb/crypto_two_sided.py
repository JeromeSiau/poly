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


RESOLUTION_DELAY_S = 60


@dataclass
class MarketPosition:
    """Tracks a two-sided position on a single crypto up/down market."""
    condition_id: str
    slug: str
    symbol: str
    timeframe: int
    up_token_id: str
    down_token_id: str
    up_shares: float = 0.0
    down_shares: float = 0.0
    up_cost: float = 0.0
    down_cost: float = 0.0
    entered_at: float = 0.0
    end_time: float = 0.0
    entry_edge: float = 0.0
    resolved: bool = False
    pnl: float | None = None

    @property
    def total_cost(self) -> float:
        return self.up_cost + self.down_cost

    @property
    def orphan(self) -> bool:
        return self.up_shares == 0 or self.down_shares == 0


class CryptoTwoSidedEngine:
    """Core decision engine for crypto two-sided arbitrage."""

    def __init__(
        self,
        min_edge_pct: float = 0.01,
        budget_per_market: float = 200.0,
        max_concurrent: int = 8,
        entry_window_s: int = 30,
        fee_rate: float = 0.01,
    ) -> None:
        self.min_edge_pct = min_edge_pct
        self.budget_per_market = budget_per_market
        self.max_concurrent = max_concurrent
        self.entry_window_s = entry_window_s
        self.fee_rate = fee_rate
        self._positions: dict[str, MarketPosition] = {}
        self._entered_slugs: set[str] = set()
        self.realized_pnl: float = 0.0

    def should_enter(self, ask_up: float, ask_down: float, market_age_s: float) -> bool:
        if market_age_s > self.entry_window_s:
            return False
        open_count = sum(1 for p in self._positions.values() if not p.resolved)
        if open_count >= self.max_concurrent:
            return False
        return compute_edge(ask_up, ask_down, self.fee_rate) >= self.min_edge_pct

    def record_entry(self, position: MarketPosition) -> None:
        self._positions[position.condition_id] = position
        self._entered_slugs.add(position.slug)

    def already_entered(self, slug: str) -> bool:
        return slug in self._entered_slugs

    def resolve(self, condition_id: str, up_final: float, down_final: float) -> float:
        pos = self._positions.get(condition_id)
        if pos is None or pos.resolved:
            return 0.0
        winning_value = (pos.up_shares * up_final) + (pos.down_shares * down_final)
        pnl = winning_value - pos.total_cost
        pos.resolved = True
        pos.pnl = pnl
        self.realized_pnl += pnl
        return pnl

    def get_pending_resolutions(self, now: float) -> list[MarketPosition]:
        return [p for p in self._positions.values()
                if not p.resolved and now > p.end_time + RESOLUTION_DELAY_S]

    def cleanup_resolved(self) -> list[MarketPosition]:
        resolved = [p for p in self._positions.values() if p.resolved]
        for p in resolved:
            del self._positions[p.condition_id]
        return resolved
