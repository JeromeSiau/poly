"""Tests for the two-sided inventory arbitrage engine."""

import time

import pytest

from src.arb.two_sided_inventory import (
    MarketSnapshot,
    OutcomeQuote,
    TradeIntent,
    TwoSidedInventoryEngine,
)


def make_snapshot(
    yes_bid: float,
    yes_ask: float,
    no_bid: float,
    no_ask: float,
    ts: float | None = None,
) -> MarketSnapshot:
    ts = ts or time.time()
    return MarketSnapshot(
        condition_id="cond-1",
        title="Will Team A win?",
        outcome_order=["Yes", "No"],
        timestamp=ts,
        outcomes={
            "Yes": OutcomeQuote(
                outcome="Yes",
                token_id="tok-yes",
                bid=yes_bid,
                ask=yes_ask,
                bid_size=1000.0,
                ask_size=1000.0,
            ),
            "No": OutcomeQuote(
                outcome="No",
                token_id="tok-no",
                bid=no_bid,
                ask=no_ask,
                bid_size=1000.0,
                ask_size=1000.0,
            ),
        },
    )


class TestFairPrice:
    def test_binary_cross_leg_parity(self):
        engine = TwoSidedInventoryEngine()
        snapshot = make_snapshot(yes_bid=0.57, yes_ask=0.59, no_bid=0.39, no_ask=0.41)
        fair = engine.compute_fair_prices(snapshot)

        # mid_yes=0.58 and 1-mid_no=0.60 => fair_yes=0.59
        assert fair["Yes"] == pytest.approx(0.59, abs=1e-6)
        assert fair["No"] == pytest.approx(0.41, abs=1e-6)


class TestSignals:
    def test_pair_arb_entry_generates_both_legs(self):
        engine = TwoSidedInventoryEngine(
            min_edge_pct=0.01,
            max_order_usd=250.0,
            min_order_usd=25.0,
        )
        snapshot = make_snapshot(yes_bid=0.47, yes_ask=0.48, no_bid=0.48, no_ask=0.49)
        intents = engine.evaluate_market(snapshot)

        pair_buys = [i for i in intents if i.side == "BUY" and i.reason == "pair_arb_entry"]
        assert len(pair_buys) == 2
        assert {i.outcome for i in pair_buys} == {"Yes", "No"}

    def test_buy_signal_when_under_fair(self):
        engine = TwoSidedInventoryEngine(
            min_edge_pct=0.01,
            max_order_usd=200.0,
            min_order_usd=25.0,
            max_outcome_inventory_usd=2000.0,
            max_market_net_usd=1000.0,
        )
        snapshot = make_snapshot(yes_bid=0.50, yes_ask=0.51, no_bid=0.43, no_ask=0.44)
        intents = engine.evaluate_market(snapshot)

        buy_yes = [i for i in intents if i.side == "BUY" and i.outcome == "Yes"]
        assert buy_yes, "Expected BUY signal on underpriced Yes leg"
        assert buy_yes[0].size_usd >= 25.0

    def test_inventory_cap_blocks_new_buy(self):
        engine = TwoSidedInventoryEngine(
            min_edge_pct=0.01,
            max_order_usd=400.0,
            min_order_usd=25.0,
            max_outcome_inventory_usd=100.0,
            max_market_net_usd=1000.0,
        )
        snapshot = make_snapshot(yes_bid=0.50, yes_ask=0.51, no_bid=0.43, no_ask=0.44)

        # Fill once to near inventory cap.
        first = engine.evaluate_market(snapshot)
        assert first
        fill = next(i for i in first if i.side == "BUY" and i.outcome == "Yes")
        seeded = TradeIntent(
            condition_id=fill.condition_id,
            title=fill.title,
            outcome=fill.outcome,
            token_id=fill.token_id,
            side=fill.side,
            price=fill.price,
            size_usd=100.0,
            edge_pct=fill.edge_pct,
            reason=fill.reason,
            timestamp=fill.timestamp,
        )
        engine.apply_fill(seeded)

        second = engine.evaluate_market(snapshot)
        assert not [i for i in second if i.side == "BUY" and i.outcome == "Yes"]

    def test_sell_signal_when_over_fair(self):
        engine = TwoSidedInventoryEngine(
            min_edge_pct=0.01,
            exit_edge_pct=0.005,
            max_order_usd=300.0,
            min_order_usd=10.0,
        )
        # Open a Yes position at 0.45 first.
        open_fill = TradeIntent(
            condition_id="cond-1",
            title="Will Team A win?",
            outcome="Yes",
            token_id="tok-yes",
            side="BUY",
            price=0.45,
            size_usd=200.0,
            edge_pct=0.02,
            reason="seed",
            timestamp=time.time() - 1800,
        )
        engine.apply_fill(open_fill)

        snapshot = make_snapshot(yes_bid=0.60, yes_ask=0.61, no_bid=0.79, no_ask=0.80)
        intents = engine.evaluate_market(snapshot)

        sells = [i for i in intents if i.side == "SELL" and i.outcome == "Yes"]
        assert sells, "Expected SELL signal when bid exceeds fair by exit edge"

    def test_pair_exit_generates_two_sells_when_bids_rich(self):
        engine = TwoSidedInventoryEngine(
            min_edge_pct=0.01,
            exit_edge_pct=0.005,
            max_order_usd=500.0,
            min_order_usd=10.0,
        )
        now = time.time()
        engine.apply_fill(
            TradeIntent(
                condition_id="cond-1",
                title="Will Team A win?",
                outcome="Yes",
                token_id="tok-yes",
                side="BUY",
                price=0.45,
                size_usd=180.0,
                edge_pct=0.02,
                reason="seed",
                timestamp=now - 600,
            )
        )
        engine.apply_fill(
            TradeIntent(
                condition_id="cond-1",
                title="Will Team A win?",
                outcome="No",
                token_id="tok-no",
                side="BUY",
                price=0.44,
                size_usd=176.0,
                edge_pct=0.02,
                reason="seed",
                timestamp=now - 600,
            )
        )

        snapshot = make_snapshot(yes_bid=0.53, yes_ask=0.54, no_bid=0.49, no_ask=0.50)
        intents = engine.evaluate_market(snapshot)
        pair_sells = [i for i in intents if i.side == "SELL" and i.reason == "pair_arb_exit"]
        assert len(pair_sells) == 2
        assert {i.outcome for i in pair_sells} == {"Yes", "No"}


class TestFillAccounting:
    def test_fill_updates_avg_price_and_realized_pnl(self):
        engine = TwoSidedInventoryEngine()

        b1 = TradeIntent(
            condition_id="cond-1",
            title="Will Team A win?",
            outcome="Yes",
            token_id="tok-yes",
            side="BUY",
            price=0.40,
            size_usd=80.0,  # 200 shares
            edge_pct=0.02,
            reason="b1",
            timestamp=time.time(),
        )
        b2 = TradeIntent(
            condition_id="cond-1",
            title="Will Team A win?",
            outcome="Yes",
            token_id="tok-yes",
            side="BUY",
            price=0.50,
            size_usd=50.0,  # 100 shares
            edge_pct=0.02,
            reason="b2",
            timestamp=time.time(),
        )
        s1 = TradeIntent(
            condition_id="cond-1",
            title="Will Team A win?",
            outcome="Yes",
            token_id="tok-yes",
            side="SELL",
            price=0.55,
            size_usd=55.0,  # 100 shares sold
            edge_pct=0.01,
            reason="s1",
            timestamp=time.time(),
        )

        engine.apply_fill(b1)
        engine.apply_fill(b2)
        result = engine.apply_fill(s1)

        state = engine.get_state("cond-1", "Yes")
        # Average before sell = (200*0.4 + 100*0.5)/300 = 0.43333
        assert state.avg_price == pytest.approx(0.4333333333, rel=1e-6)
        # Sold 100 shares at 0.55 => pnl ~ +11.6667
        assert result.realized_pnl_delta == pytest.approx(11.6666666667, rel=1e-6)
        assert state.shares == pytest.approx(200.0, rel=1e-6)

    def test_settle_position_closes_at_zero_or_one(self):
        engine = TwoSidedInventoryEngine()
        engine.apply_fill(
            TradeIntent(
                condition_id="cond-1",
                title="Will Team A win?",
                outcome="Yes",
                token_id="tok-yes",
                side="BUY",
                price=0.60,
                size_usd=120.0,  # 200 shares
                edge_pct=0.02,
                reason="seed",
                timestamp=time.time(),
            )
        )

        lose_fill = engine.settle_position("cond-1", "Yes", 0.0)
        assert lose_fill.shares == pytest.approx(200.0, rel=1e-9)
        assert lose_fill.realized_pnl_delta == pytest.approx(-120.0, rel=1e-9)
        assert engine.get_state("cond-1", "Yes").shares == pytest.approx(0.0, abs=1e-9)

        # Re-open and settle as winner.
        engine.apply_fill(
            TradeIntent(
                condition_id="cond-1",
                title="Will Team A win?",
                outcome="Yes",
                token_id="tok-yes",
                side="BUY",
                price=0.40,
                size_usd=80.0,  # 200 shares
                edge_pct=0.02,
                reason="seed",
                timestamp=time.time(),
            )
        )
        win_fill = engine.settle_position("cond-1", "Yes", 1.0)
        assert win_fill.shares == pytest.approx(200.0, rel=1e-9)
        assert win_fill.realized_pnl_delta == pytest.approx(120.0, rel=1e-9)
        assert engine.get_state("cond-1", "Yes").shares == pytest.approx(0.0, abs=1e-9)
