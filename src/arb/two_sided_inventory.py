"""Two-sided inventory arbitrage engine.

This engine models the workflow seen in high-frequency Polymarket wallets:
- trade both outcomes on the same condition when micro-inefficiencies appear
- manage directional inventory explicitly (not only raw win-rate)
- exit inventory when markets overprice held outcomes or positions age out

The module is feed-agnostic: callers provide a market snapshot with quotes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import structlog

logger = structlog.get_logger()


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(slots=True)
class OutcomeQuote:
    """Best quote levels for one outcome."""

    outcome: str
    token_id: str
    bid: Optional[float]
    ask: Optional[float]
    bid_size: Optional[float] = None  # shares
    ask_size: Optional[float] = None  # shares

    @property
    def mid(self) -> Optional[float]:
        if self.bid is None and self.ask is None:
            return None
        if self.bid is None:
            return self.ask
        if self.ask is None:
            return self.bid
        return (self.bid + self.ask) / 2.0


@dataclass(slots=True)
class MarketSnapshot:
    """Current market state required for decisioning."""

    condition_id: str
    title: str
    outcome_order: list[str]
    outcomes: dict[str, OutcomeQuote]
    timestamp: float
    liquidity: float = 0.0
    volume_24h: float = 0.0
    slug: str = ""


@dataclass(slots=True)
class InventoryState:
    """Running inventory state for one (condition, outcome)."""

    shares: float = 0.0
    avg_price: float = 0.0
    opened_at: Optional[float] = None
    realized_pnl: float = 0.0
    traded_notional: float = 0.0

    def notional(self, mark_price: float) -> float:
        return self.shares * mark_price

    def is_open(self) -> bool:
        return self.shares > 1e-9


@dataclass(slots=True)
class TradeIntent:
    """Action recommended by the engine."""

    condition_id: str
    title: str
    outcome: str
    token_id: str
    side: str  # BUY or SELL
    price: float
    size_usd: float
    edge_pct: float
    reason: str
    timestamp: float = field(default_factory=lambda: time.time())

    @property
    def shares(self) -> float:
        return self.size_usd / self.price if self.price > 0 else 0.0


@dataclass(slots=True)
class FillResult:
    """Inventory update result after applying a fill."""

    condition_id: str
    outcome: str
    side: str
    shares: float
    fill_price: float
    realized_pnl_delta: float
    remaining_shares: float
    avg_price: float


class TwoSidedInventoryEngine:
    """Generate two-sided trade intents and maintain inventory."""

    def __init__(
        self,
        min_edge_pct: float = 0.015,
        exit_edge_pct: float = 0.003,
        min_order_usd: float = 25.0,
        max_order_usd: float = 400.0,
        max_outcome_inventory_usd: float = 2500.0,
        max_market_net_usd: float = 1200.0,
        inventory_skew_pct: float = 0.02,
        max_hold_seconds: float = 24 * 3600.0,
        fee_bps: int = 0,
        enable_sells: bool = True,
        allow_pair_exit: bool = True,
    ) -> None:
        self.min_edge_pct = min_edge_pct
        self.exit_edge_pct = exit_edge_pct
        self.min_order_usd = min_order_usd
        self.max_order_usd = max_order_usd
        self.max_outcome_inventory_usd = max_outcome_inventory_usd
        self.max_market_net_usd = max_market_net_usd
        self.inventory_skew_pct = inventory_skew_pct
        self.max_hold_seconds = max_hold_seconds
        self.fee_bps = fee_bps
        self.enable_sells = enable_sells
        self.allow_pair_exit = allow_pair_exit

        self._inventory: dict[tuple[str, str], InventoryState] = {}

    @property
    def fee_pct(self) -> float:
        return self.fee_bps / 10000.0

    def get_state(self, condition_id: str, outcome: str) -> InventoryState:
        key = (condition_id, outcome)
        if key not in self._inventory:
            self._inventory[key] = InventoryState()
        return self._inventory[key]

    def compute_fair_prices(self, snapshot: MarketSnapshot) -> dict[str, float]:
        """Compute fair prices from both legs of a condition.

        Binary markets use cross-leg parity:
            fair_yes ~= avg(mid_yes, 1 - mid_no)
        Multi-outcome markets fallback to normalized mids.
        """
        outcomes = snapshot.outcome_order
        if len(outcomes) < 2:
            return {o: 0.5 for o in outcomes}

        if len(outcomes) == 2:
            first = snapshot.outcomes[outcomes[0]]
            second = snapshot.outcomes[outcomes[1]]

            m1 = first.mid
            m2 = second.mid

            candidates: list[float] = []
            if m1 is not None:
                candidates.append(m1)
            if m2 is not None:
                candidates.append(1.0 - m2)

            fair_first = sum(candidates) / len(candidates) if candidates else 0.5
            fair_first = _clamp(fair_first, 0.001, 0.999)
            fair_second = _clamp(1.0 - fair_first, 0.001, 0.999)
            return {
                outcomes[0]: fair_first,
                outcomes[1]: fair_second,
            }

        mids: list[float] = []
        for outcome in outcomes:
            mid = snapshot.outcomes[outcome].mid
            mids.append(mid if mid is not None else 1.0 / len(outcomes))
        total = sum(mids)
        if total <= 0:
            return {o: 1.0 / len(outcomes) for o in outcomes}
        return {o: _clamp(mids[idx] / total, 0.001, 0.999) for idx, o in enumerate(outcomes)}

    def _outcome_sign(self, snapshot: MarketSnapshot, outcome: str) -> float:
        if len(snapshot.outcome_order) != 2:
            return 0.0
        if outcome == snapshot.outcome_order[0]:
            return 1.0
        if outcome == snapshot.outcome_order[1]:
            return -1.0
        return 0.0

    def _market_directional_inventory(
        self,
        snapshot: MarketSnapshot,
        fair_prices: dict[str, float],
    ) -> float:
        if len(snapshot.outcome_order) != 2:
            return 0.0
        total = 0.0
        for outcome in snapshot.outcome_order:
            sign = self._outcome_sign(snapshot, outcome)
            state = self.get_state(snapshot.condition_id, outcome)
            total += sign * state.notional(fair_prices[outcome])
        return total

    def _room_by_market_net(
        self,
        current_net: float,
        side: str,
        sign: float,
    ) -> float:
        if sign == 0:
            return self.max_order_usd

        # BUY adds sign*size to directional net, SELL subtracts sign*size.
        delta_sign = sign if side == "BUY" else -sign
        same_direction = (current_net == 0) or (current_net * delta_sign > 0)
        if same_direction:
            return max(0.0, self.max_market_net_usd - abs(current_net))
        return max(0.0, self.max_market_net_usd + abs(current_net))

    def _binary_combo_intents(
        self,
        snapshot: MarketSnapshot,
        fair: dict[str, float],
        now: float,
        market_net: float,
    ) -> tuple[list[TradeIntent], float, set[str]]:
        """Detect pair-level arb on binary markets.

        Cases:
        - Entry arb: ask_yes + ask_no < 1 - fees
        - Exit arb for held pair: bid_yes + bid_no > 1 + fees
        """
        if len(snapshot.outcome_order) != 2:
            return [], market_net, set()

        out_a, out_b = snapshot.outcome_order
        q_a = snapshot.outcomes[out_a]
        q_b = snapshot.outcomes[out_b]
        s_a = self.get_state(snapshot.condition_id, out_a)
        s_b = self.get_state(snapshot.condition_id, out_b)

        intents: list[TradeIntent] = []
        consumed: set[str] = set()

        # Pair entry arb: buy equal shares on both legs.
        if q_a.ask is not None and q_b.ask is not None:
            pair_edge = 1.0 - (q_a.ask + q_b.ask) - 2 * self.fee_pct
            if pair_edge >= self.min_edge_pct:
                room_a = max(0.0, self.max_outcome_inventory_usd - s_a.notional(fair[out_a]))
                room_b = max(0.0, self.max_outcome_inventory_usd - s_b.notional(fair[out_b]))

                max_shares = min(
                    self.max_order_usd / q_a.ask,
                    self.max_order_usd / q_b.ask,
                    (q_a.ask_size or (self.max_order_usd / q_a.ask)),
                    (q_b.ask_size or (self.max_order_usd / q_b.ask)),
                    (room_a / q_a.ask) if q_a.ask > 0 else 0.0,
                    (room_b / q_b.ask) if q_b.ask > 0 else 0.0,
                )
                min_shares = max(
                    self.min_order_usd / q_a.ask,
                    self.min_order_usd / q_b.ask,
                )
                if max_shares >= min_shares and max_shares > 0:
                    # Respect market directional cap after each leg.
                    size_a = max_shares * q_a.ask
                    size_b = max_shares * q_b.ask
                    sign_a = self._outcome_sign(snapshot, out_a)
                    sign_b = self._outcome_sign(snapshot, out_b)

                    room_a_net = self._room_by_market_net(market_net, side="BUY", sign=sign_a)
                    size_a = min(size_a, room_a_net if sign_a != 0 else size_a)
                    if size_a >= self.min_order_usd:
                        market_net += sign_a * size_a if sign_a != 0 else 0.0

                        room_b_net = self._room_by_market_net(market_net, side="BUY", sign=sign_b)
                        size_b = min(size_b, room_b_net if sign_b != 0 else size_b)
                        if size_b >= self.min_order_usd:
                            intents.extend(
                                [
                                    TradeIntent(
                                        condition_id=snapshot.condition_id,
                                        title=snapshot.title,
                                        outcome=out_a,
                                        token_id=q_a.token_id,
                                        side="BUY",
                                        price=q_a.ask,
                                        size_usd=size_a,
                                        edge_pct=pair_edge,
                                        reason="pair_arb_entry",
                                        timestamp=now,
                                    ),
                                    TradeIntent(
                                        condition_id=snapshot.condition_id,
                                        title=snapshot.title,
                                        outcome=out_b,
                                        token_id=q_b.token_id,
                                        side="BUY",
                                        price=q_b.ask,
                                        size_usd=size_b,
                                        edge_pct=pair_edge,
                                        reason="pair_arb_entry",
                                        timestamp=now,
                                    ),
                                ]
                            )
                            consumed.update({out_a, out_b})
                            market_net += sign_b * size_b if sign_b != 0 else 0.0

        # Pair exit arb: sell matched shares from both legs if rich enough.
        if (
            self.allow_pair_exit
            and q_a.bid is not None
            and q_b.bid is not None
            and s_a.shares > 0
            and s_b.shares > 0
        ):
            pair_exit_edge = (q_a.bid + q_b.bid) - 1.0 - 2 * self.fee_pct
            if pair_exit_edge >= self.exit_edge_pct:
                shares_cap = min(
                    s_a.shares,
                    s_b.shares,
                    q_a.bid_size or s_a.shares,
                    q_b.bid_size or s_b.shares,
                    self.max_order_usd / q_a.bid,
                    self.max_order_usd / q_b.bid,
                )
                if shares_cap > 0:
                    size_a = shares_cap * q_a.bid
                    size_b = shares_cap * q_b.bid
                    if size_a >= self.min_order_usd and size_b >= self.min_order_usd:
                        intents.extend(
                            [
                                TradeIntent(
                                    condition_id=snapshot.condition_id,
                                    title=snapshot.title,
                                    outcome=out_a,
                                    token_id=q_a.token_id,
                                    side="SELL",
                                    price=q_a.bid,
                                    size_usd=size_a,
                                    edge_pct=pair_exit_edge,
                                    reason="pair_arb_exit",
                                    timestamp=now,
                                ),
                                TradeIntent(
                                    condition_id=snapshot.condition_id,
                                    title=snapshot.title,
                                    outcome=out_b,
                                    token_id=q_b.token_id,
                                    side="SELL",
                                    price=q_b.bid,
                                    size_usd=size_b,
                                    edge_pct=pair_exit_edge,
                                    reason="pair_arb_exit",
                                    timestamp=now,
                                ),
                            ]
                        )
                        consumed.update({out_a, out_b})
                        sign_a = self._outcome_sign(snapshot, out_a)
                        sign_b = self._outcome_sign(snapshot, out_b)
                        market_net -= sign_a * size_a if sign_a != 0 else 0.0
                        market_net -= sign_b * size_b if sign_b != 0 else 0.0

        return intents, market_net, consumed

    def evaluate_market(
        self,
        snapshot: MarketSnapshot,
        fair_prices: Optional[dict[str, float]] = None,
        now_ts: Optional[float] = None,
    ) -> list[TradeIntent]:
        """Evaluate one market and return ordered trade intents."""
        now = now_ts or snapshot.timestamp or time.time()
        fair = fair_prices or self.compute_fair_prices(snapshot)
        intents: list[TradeIntent] = []

        market_net = self._market_directional_inventory(snapshot, fair)
        consumed_outcomes: set[str] = set()

        # Pair-level arb first for binary markets.
        combo_intents, market_net, combo_consumed = self._binary_combo_intents(
            snapshot=snapshot,
            fair=fair,
            now=now,
            market_net=market_net,
        )
        if combo_intents:
            intents.extend(combo_intents)
            consumed_outcomes.update(combo_consumed)

        # 1) Unwind / inventory-control sells first.
        if self.enable_sells:
            for outcome in snapshot.outcome_order:
                if outcome in consumed_outcomes:
                    continue
                quote = snapshot.outcomes[outcome]
                state = self.get_state(snapshot.condition_id, outcome)
                if not state.is_open() or quote.bid is None:
                    continue

                fair_price = fair[outcome]
                inv_value = state.notional(fair_price)
                hold_age = (now - state.opened_at) if state.opened_at else 0.0

                edge_sell = quote.bid - fair_price - self.fee_pct
                stale_exit = (
                    hold_age >= self.max_hold_seconds
                    and quote.bid >= state.avg_price + self.fee_pct
                )
                risk_exit = inv_value > self.max_outcome_inventory_usd

                if edge_sell < self.exit_edge_pct and not stale_exit and not risk_exit:
                    continue

                available_notional = state.shares * quote.bid
                if available_notional <= 0:
                    continue

                size_cap = min(
                    self.max_order_usd,
                    available_notional,
                    (quote.bid_size or (available_notional / quote.bid)) * quote.bid,
                )
                if risk_exit:
                    excess = inv_value - self.max_outcome_inventory_usd
                    size_cap = max(size_cap, min(excess, available_notional))

                sign = self._outcome_sign(snapshot, outcome)
                net_room = self._room_by_market_net(market_net, side="SELL", sign=sign)
                size_usd = min(size_cap, net_room if sign != 0 else size_cap)

                if size_usd < self.min_order_usd:
                    continue

                reason_bits: list[str] = []
                if edge_sell >= self.exit_edge_pct:
                    reason_bits.append("over_fair")
                if stale_exit:
                    reason_bits.append("max_hold")
                if risk_exit:
                    reason_bits.append("inv_cap")

                intents.append(
                    TradeIntent(
                        condition_id=snapshot.condition_id,
                        title=snapshot.title,
                        outcome=outcome,
                        token_id=quote.token_id,
                        side="SELL",
                        price=quote.bid,
                        size_usd=size_usd,
                        edge_pct=edge_sell,
                        reason="+".join(reason_bits) if reason_bits else "rebalance",
                        timestamp=now,
                    )
                )
                consumed_outcomes.add(outcome)
                if sign != 0:
                    market_net -= sign * size_usd

        # 2) New entries / adds.
        for outcome in snapshot.outcome_order:
            if outcome in consumed_outcomes:
                continue

            quote = snapshot.outcomes[outcome]
            if quote.ask is None:
                continue

            fair_price = fair[outcome]
            state = self.get_state(snapshot.condition_id, outcome)
            inv_value = state.notional(fair_price)

            raw_edge = fair_price - quote.ask - self.fee_pct
            # Inventory skew penalizes adding where we're already heavy.
            inv_ratio = (
                inv_value / self.max_outcome_inventory_usd
                if self.max_outcome_inventory_usd > 0
                else 0.0
            )
            skew_penalty = max(0.0, inv_ratio) * self.inventory_skew_pct
            edge_buy = raw_edge - skew_penalty
            if edge_buy < self.min_edge_pct:
                continue

            outcome_room = max(0.0, self.max_outcome_inventory_usd - inv_value)
            if outcome_room < self.min_order_usd:
                continue

            size_cap = min(
                self.max_order_usd,
                outcome_room,
                (quote.ask_size or (self.max_order_usd / quote.ask)) * quote.ask,
            )

            sign = self._outcome_sign(snapshot, outcome)
            net_room = self._room_by_market_net(market_net, side="BUY", sign=sign)
            size_usd = min(size_cap, net_room if sign != 0 else size_cap)

            if size_usd < self.min_order_usd:
                continue

            intents.append(
                TradeIntent(
                    condition_id=snapshot.condition_id,
                    title=snapshot.title,
                    outcome=outcome,
                    token_id=quote.token_id,
                    side="BUY",
                    price=quote.ask,
                    size_usd=size_usd,
                    edge_pct=edge_buy,
                    reason="under_fair",
                    timestamp=now,
                )
            )
            if sign != 0:
                market_net += sign * size_usd

        # Prioritize strongest edges first.
        intents.sort(key=lambda i: i.edge_pct, reverse=True)
        return intents

    def apply_fill(self, intent: TradeIntent) -> FillResult:
        """Apply a fill to local inventory and return state delta."""
        state = self.get_state(intent.condition_id, intent.outcome)
        fill_shares = intent.shares
        realized_delta = 0.0

        if intent.side == "BUY":
            prev_shares = state.shares
            total_shares = prev_shares + fill_shares
            if total_shares <= 0:
                total_shares = 0.0
            if total_shares > 0:
                state.avg_price = (
                    (state.avg_price * prev_shares + intent.price * fill_shares) / total_shares
                    if prev_shares > 0
                    else intent.price
                )
                state.shares = total_shares
            state.opened_at = state.opened_at or intent.timestamp
            state.traded_notional += intent.size_usd

        elif intent.side == "SELL":
            if state.shares <= 0:
                fill_shares = 0.0
            else:
                fill_shares = min(fill_shares, state.shares)
                realized_delta = fill_shares * (intent.price - state.avg_price)
                state.realized_pnl += realized_delta
                state.shares -= fill_shares
                state.traded_notional += fill_shares * intent.price
                if state.shares <= 1e-9:
                    state.shares = 0.0
                    state.avg_price = 0.0
                    state.opened_at = None
        else:
            raise ValueError(f"Unsupported side: {intent.side}")

        return FillResult(
            condition_id=intent.condition_id,
            outcome=intent.outcome,
            side=intent.side,
            shares=fill_shares,
            fill_price=intent.price,
            realized_pnl_delta=realized_delta,
            remaining_shares=state.shares,
            avg_price=state.avg_price,
        )

    def settle_position(
        self,
        condition_id: str,
        outcome: str,
        settlement_price: float,
        *,
        timestamp: Optional[float] = None,
    ) -> FillResult:
        """Force-close an open outcome at final settlement price.

        This is used for resolved markets where final price can be exactly 0.0/1.0.
        """
        del timestamp  # reserved for parity with apply_fill-style callers

        state = self.get_state(condition_id, outcome)
        if state.shares <= 0:
            return FillResult(
                condition_id=condition_id,
                outcome=outcome,
                side="SELL",
                shares=0.0,
                fill_price=settlement_price,
                realized_pnl_delta=0.0,
                remaining_shares=0.0,
                avg_price=state.avg_price,
            )

        price = _clamp(settlement_price, 0.0, 1.0)
        closed_shares = state.shares
        realized_delta = closed_shares * (price - state.avg_price)
        state.realized_pnl += realized_delta
        state.traded_notional += closed_shares * price
        state.shares = 0.0
        state.avg_price = 0.0
        state.opened_at = None

        return FillResult(
            condition_id=condition_id,
            outcome=outcome,
            side="SELL",
            shares=closed_shares,
            fill_price=price,
            realized_pnl_delta=realized_delta,
            remaining_shares=0.0,
            avg_price=0.0,
        )

    def get_open_inventory(self) -> dict[str, dict[str, InventoryState]]:
        """Return nested dict by condition and outcome for open lots only."""
        nested: dict[str, dict[str, InventoryState]] = {}
        for (condition_id, outcome), state in self._inventory.items():
            if not state.is_open():
                continue
            nested.setdefault(condition_id, {})[outcome] = state
        return nested

    def get_realized_pnl(self) -> float:
        return sum(state.realized_pnl for state in self._inventory.values())
