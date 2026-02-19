# src/td_maker/filters.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Any

import structlog

logger = structlog.get_logger()


@dataclass(slots=True)
class FilterResult:
    action: str   # "skip" | "maker" | "taker"
    reason: str
    price: float
    outcome: str
    p_win: Optional[float] = None  # set by ML model for Kelly sizing

    @property
    def is_skip(self) -> bool:
        return self.action == "skip"

    @classmethod
    def skip(cls, reason: str, outcome: str = "") -> "FilterResult":
        return cls(action="skip", reason=reason, price=0.0, outcome=outcome)

    @classmethod
    def maker(cls, price: float, outcome: str,
              p_win: Optional[float] = None) -> "FilterResult":
        return cls(action="maker", reason="", price=price, outcome=outcome,
                   p_win=p_win)

    @classmethod
    def taker(cls, price: float, outcome: str,
              p_win: Optional[float] = None) -> "FilterResult":
        return cls(action="taker", reason="", price=price, outcome=outcome,
                   p_win=p_win)


class EntryFilters:
    def __init__(self, chainlink_feed: Any, config: Any,
                 model: Optional[Any] = None) -> None:
        self.chainlink = chainlink_feed
        self.config = config
        self.model = model

    def should_bid(
        self,
        market: Any,
        *,
        outcome: str,
        bid: float,
        ask: float,
    ) -> FilterResult:
        """Returns FilterResult(action, reason, price, outcome)."""
        cfg = self.config

        if not (cfg.target_bid <= bid <= cfg.max_bid):
            return FilterResult.skip("bid_out_of_range", outcome)

        if not self._check_entry_time(market):
            return FilterResult.skip("time_gate", outcome)

        if cfg.min_move_pct > 0 and not self._check_min_move(market, outcome):
            return FilterResult.skip("insufficient_move", outcome)

        if cfg.max_move_pct > 0 and not self._check_max_move(market, outcome):
            return FilterResult.skip("excessive_move", outcome)

        if cfg.entry_fair_margin > 0 and not self._check_fair_value(
                market, outcome, bid):
            return FilterResult.skip("above_fair_value", outcome)

        if self.model:
            return self._check_model(market, outcome, bid, ask)

        return FilterResult.maker(bid, outcome)

    def _check_entry_time(self, market: Any) -> bool:
        cfg = self.config
        if cfg.min_entry_minutes <= 0 and cfg.max_entry_minutes <= 0:
            return True
        elapsed = (time.time() - market.slot_ts) / 60.0
        if cfg.min_entry_minutes > 0 and elapsed < cfg.min_entry_minutes:
            return False
        if cfg.max_entry_minutes > 0 and elapsed > cfg.max_entry_minutes:
            return False
        return True

    def _check_min_move(self, market: Any, outcome: str) -> bool:
        current = self.chainlink.get_price(market.chainlink_symbol)
        if current is None or market.ref_price <= 0:
            return True
        move_pct = (current - market.ref_price) / market.ref_price * 100.0
        if outcome == "Down":
            move_pct = -move_pct
        return move_pct >= self.config.min_move_pct

    def _check_max_move(self, market: Any, outcome: str) -> bool:
        current = self.chainlink.get_price(market.chainlink_symbol)
        if current is None or market.ref_price <= 0:
            return True
        move_pct = abs((current - market.ref_price) / market.ref_price * 100.0)
        return move_pct <= self.config.max_move_pct

    def _check_fair_value(self, market: Any, outcome: str, bid: float) -> bool:
        current = self.chainlink.get_price(market.chainlink_symbol)
        if current is None or market.ref_price <= 0:
            return True
        from src.utils.fair_value import estimate_fair_value
        slot_remaining = max(0, market.slot_end_ts() - time.time())
        dir_move_pct = (current - market.ref_price) / market.ref_price * 100.0
        if outcome == "Down":
            dir_move_pct = -dir_move_pct
        fv = estimate_fair_value(dir_move_pct, slot_remaining / 60.0)
        return bid <= fv + self.config.entry_fair_margin

    def _check_model(self, market: Any, outcome: str,
                     bid: float, ask: float) -> FilterResult:
        """ML model entry filter. Returns FilterResult with p_win for Kelly sizing."""
        try:
            p_win = self.model.predict(market, outcome, bid)
            threshold = getattr(self.config, "hybrid_skip_below", 0.55)
            taker_above = getattr(self.config, "hybrid_taker_above", 0.72)
            if p_win < threshold:
                return FilterResult.skip("model_skip", outcome)
            if p_win >= taker_above:
                return FilterResult.taker(ask, outcome, p_win=p_win)
            return FilterResult.maker(bid, outcome, p_win=p_win)
        except Exception as e:
            logger.warning("model_predict_error", error=str(e))
            return FilterResult.maker(bid, outcome)
