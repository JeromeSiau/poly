"""Shadow taker tracker — logs what taker orders WOULD have done.

Runs alongside the maker strategy without executing any real orders.
Compares taker-at-ask entry vs maker-at-bid entry to measure
adverse selection cost vs taker fee cost.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import structlog

logger = structlog.get_logger()

def crypto_taker_fee(price: float) -> float:
    """Per-share taker fee for crypto 15-min/5-min markets (feeRateBps=1000).

    On-chain formula (CalculatorHelper.sol + docs):
        fee_per_share = price * 0.25 * (price * (1 - price))^2

    Effective rate on capital at various prices:
        p=0.50 → 1.56%,  p=0.75 → 0.88%,  p=0.80 → 0.64%,
        p=0.85 → 0.41%,  p=0.90 → 0.20%
    """
    return price * 0.25 * (price * (1.0 - price)) ** 2


@dataclass(slots=True)
class ShadowEntry:
    """A shadow taker position (never executed, only tracked)."""
    condition_id: str
    outcome: str
    ask_price: float
    fee_per_share: float
    effective_price: float  # ask + fee
    size_usd: float
    shares: float           # size_usd / effective_price
    entered_at: float
    dir_move: Optional[float] = None


class TakerShadow:
    """Tracks shadow taker positions for comparison with real maker fills."""

    def __init__(self) -> None:
        self.positions: dict[str, ShadowEntry] = {}
        self.wins: int = 0
        self.losses: int = 0
        self.pnl: float = 0.0

    def has(self, cid: str) -> bool:
        return cid in self.positions

    def record(
        self,
        cid: str,
        outcome: str,
        ask: float,
        size_usd: float,
        dir_move: Optional[float] = None,
    ) -> None:
        """Record a shadow taker entry at the current ask price."""
        if cid in self.positions:
            return
        fee = crypto_taker_fee(ask)
        effective = ask + fee
        shares = size_usd / effective
        self.positions[cid] = ShadowEntry(
            condition_id=cid,
            outcome=outcome,
            ask_price=ask,
            fee_per_share=fee,
            effective_price=effective,
            size_usd=size_usd,
            shares=shares,
            entered_at=time.time(),
            dir_move=dir_move,
        )

    def settle(self, cid: str, won: bool) -> Optional[dict]:
        """Settle a shadow position. Returns comparison dict or None."""
        shadow = self.positions.pop(cid, None)
        if shadow is None:
            return None

        if won:
            s_pnl = shadow.shares * 1.0 - shadow.size_usd
            self.wins += 1
        else:
            s_pnl = -shadow.size_usd
            self.losses += 1
        self.pnl += s_pnl

        result = {
            "ask_entry": shadow.ask_price,
            "fee": shadow.fee_per_share,
            "shadow_pnl": round(s_pnl, 4),
            "shadow_total_pnl": round(self.pnl, 4),
            "shadow_record": f"{self.wins}W-{self.losses}L",
        }

        logger.info(
            "td_shadow_settled",
            condition_id=cid[:16],
            outcome=shadow.outcome,
            ask_entry=shadow.ask_price,
            won=won,
            shadow_pnl=round(s_pnl, 4),
            shadow_total=round(self.pnl, 4),
            shadow_record=f"{self.wins}W-{self.losses}L",
            dir_move=shadow.dir_move,
        )

        return result

    def remove(self, cid: str) -> None:
        """Remove a shadow position without settling (market cleanup)."""
        self.positions.pop(cid, None)

    def status_line(self) -> str:
        """One-line summary for periodic log."""
        total = self.wins + self.losses
        wr = self.wins / total * 100 if total else 0
        return f"shadow: {self.wins}W-{self.losses}L wr={wr:.0f}% pnl=${self.pnl:+.2f}"
