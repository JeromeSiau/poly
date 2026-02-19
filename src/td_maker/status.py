# src/td_maker/status.py
from __future__ import annotations
import time
from datetime import datetime, timezone
from typing import Any
from src.td_maker.state import MarketRegistry


class StatusLine:

    def __init__(self, registry: MarketRegistry, guard: Any,
                 shadow: Any, config: Any) -> None:
        self.registry = registry
        self.guard = guard
        self.shadow = shadow
        self.config = config
        self._last_print: float = 0.0
        self._interval: float = 30.0

    def print_if_due(self) -> None:
        if time.time() - self._last_print < self._interval:
            return
        self._last_print = time.time()
        print(self._format(), flush=True)

    def _format(self) -> str:
        r = self.registry
        now = datetime.now(timezone.utc).strftime("%H:%M:%S")
        stats = self.guard.stats if hasattr(self.guard, "stats") else {}
        wins = stats.get("wins", 0)
        losses = stats.get("losses", 0)
        pnl = stats.get("pnl", 0.0)
        exp = r.total_exposure()
        n_orders = sum(
            len(m.active_orders) for m in r.active_markets())
        n_pos = len(r.markets_with_positions())
        shadow_line = self.shadow.status_line() if hasattr(
            self.shadow, "status_line") else ""
        return (
            f"[{now}] mkts={len(r)} orders={n_orders} pos={n_pos} "
            f"{wins}W-{losses}L pnl=${pnl:+.2f} exp=${exp:.0f}"
            + (f" | {shadow_line}" if shadow_line else "")
        )
