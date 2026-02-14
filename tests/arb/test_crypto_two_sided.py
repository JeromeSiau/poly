import pytest
from src.arb.crypto_two_sided import (
    next_slots, compute_edge, compute_sweep,
    MarketPosition, CryptoTwoSidedEngine,
    SlotScanner, SlotMarket,
)


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


def _make_position(**overrides):
    defaults = dict(
        condition_id="0xabc", slug="btc-updown-5m-123",
        symbol="BTCUSDT", timeframe=300,
        up_token_id="tok_up", down_token_id="tok_down",
        up_shares=100.0, down_shares=80.0,
        up_cost=40.0, down_cost=32.0,
        entered_at=1000.0, end_time=1300.0, entry_edge=0.15,
    )
    defaults.update(overrides)
    return MarketPosition(**defaults)


class TestMarketPosition:
    def test_total_cost(self):
        pos = _make_position()
        assert pos.total_cost == 72.0

    def test_not_orphan_both_sides(self):
        pos = _make_position()
        assert not pos.orphan

    def test_orphan_when_one_side_zero(self):
        pos = _make_position(down_shares=0.0, down_cost=0.0)
        assert pos.orphan


class TestCryptoTwoSidedEngine:
    def make_engine(self, **kw):
        defaults = dict(min_edge_pct=0.01, budget_per_market=200.0,
                        max_concurrent=8, entry_window_s=30, fee_rate=0.01)
        defaults.update(kw)
        return CryptoTwoSidedEngine(**defaults)

    def test_should_enter_positive_edge(self):
        e = self.make_engine()
        assert e.should_enter(ask_up=0.40, ask_down=0.40, market_age_s=5)

    def test_no_enter_negative_edge(self):
        e = self.make_engine()
        assert not e.should_enter(ask_up=0.55, ask_down=0.50, market_age_s=5)

    def test_no_enter_after_window(self):
        e = self.make_engine(entry_window_s=30)
        assert not e.should_enter(ask_up=0.40, ask_down=0.40, market_age_s=35)

    def test_no_enter_max_concurrent(self):
        e = self.make_engine(max_concurrent=1)
        e._positions["x"] = _make_position(condition_id="x")
        assert not e.should_enter(ask_up=0.40, ask_down=0.40, market_age_s=5)

    def test_record_entry(self):
        e = self.make_engine()
        pos = _make_position()
        e.record_entry(pos)
        assert "0xabc" in e._positions
        assert e.already_entered("btc-updown-5m-123")

    def test_resolve_up_wins(self):
        e = self.make_engine()
        e.record_entry(_make_position())
        pnl = e.resolve("0xabc", up_final=1.0, down_final=0.0)
        assert pnl == pytest.approx(28.0)  # 100*1 - 72

    def test_resolve_down_wins(self):
        e = self.make_engine()
        e.record_entry(_make_position())
        pnl = e.resolve("0xabc", up_final=0.0, down_final=1.0)
        assert pnl == pytest.approx(8.0)  # 80*1 - 72

    def test_resolve_unknown(self):
        e = self.make_engine()
        assert e.resolve("unknown", 1.0, 0.0) == 0.0

    def test_pending_resolutions(self):
        e = self.make_engine()
        e.record_entry(_make_position())
        assert e.get_pending_resolutions(now=1200) == []
        pending = e.get_pending_resolutions(now=1361)
        assert len(pending) == 1

    def test_cleanup_resolved(self):
        e = self.make_engine()
        e.record_entry(_make_position())
        e.resolve("0xabc", 1.0, 0.0)
        resolved = e.cleanup_resolved()
        assert len(resolved) == 1
        assert "0xabc" not in e._positions


class TestSlotScanner:
    def test_parse_market(self):
        scanner = SlotScanner(["BTCUSDT"], [300])
        raw = {
            "conditionId": "0xabc123",
            "outcomes": '["Up", "Down"]',
            "outcomePrices": '["0.45", "0.55"]',
            "clobTokenIds": '["token_up", "token_down"]',
            "eventStartTime": "2026-02-14T03:10:00Z",
            "endDate": "2026-02-14T03:15:00Z",
        }
        result = scanner._parse_market(raw, "BTCUSDT", "btc-updown-5m-123")
        assert result is not None
        assert result.condition_id == "0xabc123"
        assert result.token_ids["Up"] == "token_up"
        assert result.token_ids["Down"] == "token_down"
        assert result.outcome_prices["Up"] == pytest.approx(0.45)

    def test_parse_missing_fields(self):
        scanner = SlotScanner(["BTCUSDT"], [300])
        assert scanner._parse_market({}, "BTCUSDT", "slug") is None

    def test_cleanup_expired(self):
        scanner = SlotScanner(["BTCUSDT"], [300])
        scanner._markets["old"] = SlotMarket(
            condition_id="x", slug="old", symbol="BTCUSDT",
            event_start=1000, end_time=1100, timeframe=300,
            token_ids={}, outcome_prices={},
        )
        scanner._markets["fresh"] = SlotMarket(
            condition_id="y", slug="fresh", symbol="BTCUSDT",
            event_start=9000, end_time=9200, timeframe=300,
            token_ids={}, outcome_prices={},
        )
        scanner._cleanup_expired(9300)
        assert "old" not in scanner._markets
        assert "fresh" in scanner._markets
