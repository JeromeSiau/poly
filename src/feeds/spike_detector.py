"""Detects price spikes on Polymarket CLOB WebSocket feed."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class SpikeSignal:
    condition_id: str
    outcome: str
    price_before: float
    price_now: float
    delta: float
    direction: str  # "up" | "down"
    timestamp: float


class SpikeDetector:
    """Monitors price observations and emits signals on large moves."""

    def __init__(
        self,
        threshold_pct: float = 0.15,
        window_seconds: float = 60.0,
        cooldown_seconds: float = 120.0,
    ) -> None:
        self._threshold = max(0.01, threshold_pct)
        self._window = max(1.0, window_seconds)
        self._cooldown = max(0.0, cooldown_seconds)
        # key -> deque of (timestamp, price)
        self._history: dict[str, deque[tuple[float, float]]] = {}
        # key -> last spike timestamp
        self._last_spike: dict[str, float] = {}

    def _key(self, condition_id: str, outcome: str) -> str:
        return f"{condition_id}:{outcome}"

    def observe(
        self,
        condition_id: str,
        outcome: str,
        price: float,
        timestamp: float,
    ) -> list[SpikeSignal]:
        key = self._key(condition_id, outcome)

        if key not in self._history:
            self._history[key] = deque(maxlen=500)

        buf = self._history[key]
        buf.append((timestamp, price))

        # Prune old entries outside window
        cutoff = timestamp - self._window
        while buf and buf[0][0] < cutoff:
            buf.popleft()

        if len(buf) < 2:
            return []

        # Compare current price to oldest in window
        oldest_price = buf[0][1]
        delta = price - oldest_price

        if abs(delta) < self._threshold:
            return []

        # Cooldown check
        last_spike = self._last_spike.get(key, 0.0)
        if self._cooldown > 0 and (timestamp - last_spike) < self._cooldown:
            return []

        self._last_spike[key] = timestamp

        return [
            SpikeSignal(
                condition_id=condition_id,
                outcome=outcome,
                price_before=oldest_price,
                price_now=price,
                delta=delta,
                direction="up" if delta > 0 else "down",
                timestamp=timestamp,
            )
        ]
