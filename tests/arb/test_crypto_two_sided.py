import pytest
from src.arb.crypto_two_sided import next_slots, compute_edge, compute_sweep


class TestNextSlots:
    def test_5min_slot(self):
        now = 1739502450.0
        slots = next_slots(now, ["BTCUSDT"], [300])
        assert len(slots) == 1
        s = slots[0]
        assert s["timeframe"] == 300
        assert s["event_start"] % 300 == 0
        assert s["end_time"] == s["event_start"] + 300
        assert "btc-updown-5m-" in s["slug"]

    def test_15min_slot(self):
        now = 1739502450.0
        slots = next_slots(now, ["BTCUSDT"], [900])
        assert len(slots) == 1
        s = slots[0]
        assert s["timeframe"] == 900
        assert s["event_start"] % 900 == 0
        assert "btc-updown-15m-" in s["slug"]

    def test_multi_symbol_timeframe(self):
        now = 1739502450.0
        slots = next_slots(now, ["BTCUSDT", "ETHUSDT"], [300, 900])
        assert len(slots) == 4
        slugs = [s["slug"] for s in slots]
        assert any("btc-updown-5m-" in s for s in slugs)
        assert any("eth-updown-15m-" in s for s in slugs)


class TestComputeEdge:
    def test_positive_edge(self):
        assert compute_edge(0.40, 0.40, 0.01) == pytest.approx(0.18)

    def test_zero_edge(self):
        assert compute_edge(0.49, 0.49, 0.01) == pytest.approx(0.0)

    def test_negative_edge(self):
        assert compute_edge(0.55, 0.50, 0.01) == pytest.approx(-0.07)

    def test_no_fees(self):
        assert compute_edge(0.40, 0.40, 0.0) == pytest.approx(0.20)


class TestComputeSweep:
    def test_basic_sweep(self):
        up_asks = [(0.40, 100.0), (0.45, 200.0)]
        down_asks = [(0.35, 150.0), (0.42, 100.0)]
        up_budget, down_budget, edge = compute_sweep(up_asks, down_asks, 0.01, 200.0)
        assert up_budget > 0
        assert down_budget > 0
        assert up_budget + down_budget <= 200.0
        assert edge > 0

    def test_no_edge(self):
        up_asks = [(0.55, 100.0)]
        down_asks = [(0.50, 100.0)]
        up_budget, down_budget, edge = compute_sweep(up_asks, down_asks, 0.01, 200.0)
        assert up_budget == 0
        assert down_budget == 0

    def test_budget_cap(self):
        up_asks = [(0.30, 10000.0)]
        down_asks = [(0.30, 10000.0)]
        up_budget, down_budget, edge = compute_sweep(up_asks, down_asks, 0.01, 100.0)
        assert up_budget + down_budget <= 100.0
        assert edge > 0

    def test_empty_book(self):
        up_budget, down_budget, edge = compute_sweep([], [(0.40, 100.0)], 0.01, 200.0)
        assert up_budget == 0
        assert down_budget == 0

    def test_proportional_split(self):
        up_asks = [(0.40, 500.0)]
        down_asks = [(0.40, 500.0)]
        up_budget, down_budget, edge = compute_sweep(up_asks, down_asks, 0.01, 200.0)
        assert abs(up_budget - down_budget) < 1.0
