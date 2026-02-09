"""Tests for CryptoMinuteEngine — 15-minute binary market strategies."""

import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.arb.crypto_minute import (
    BinanceSpotPoller,
    CryptoMinuteEngine,
    MarketScanner,
    MinuteMarket,
    MinuteOpportunity,
    PaperTrade,
)
from src.db.database import init_db, reset_engines


# === PaperTrade ===


class TestPaperTrade:
    def test_gap_bucket_small(self):
        assert PaperTrade.gap_to_bucket(0.1) == "small"

    def test_gap_bucket_medium(self):
        assert PaperTrade.gap_to_bucket(0.3) == "medium"

    def test_gap_bucket_large(self):
        assert PaperTrade.gap_to_bucket(0.6) == "large"

    def test_time_bucket_2_3min(self):
        assert PaperTrade.time_to_bucket(150) == "2-3min"

    def test_time_bucket_3_4min(self):
        assert PaperTrade.time_to_bucket(200) == "3-4min"

    def test_time_bucket_4_5min(self):
        assert PaperTrade.time_to_bucket(280) == "4-5min"


# === MarketScanner ===


class TestMarketScanner:
    def test_parse_iso(self):
        ts = MarketScanner._parse_iso("2026-02-09T15:45:00Z")
        assert ts is not None
        assert ts > 0

    def test_parse_iso_empty(self):
        assert MarketScanner._parse_iso("") is None

    def test_parse_market(self):
        scanner = MarketScanner(["BTCUSDT"])
        now = time.time()
        mkt_data = {
            "conditionId": "0xabc123",
            "outcomes": '["Up", "Down"]',
            "outcomePrices": '["0.92", "0.08"]',
            "clobTokenIds": '["tok1", "tok2"]',
            "eventStartTime": "2026-02-09T15:45:00Z",
            "endDate": "2026-02-09T16:00:00Z",
        }
        result = scanner._parse_market(mkt_data, "BTCUSDT", "btc-updown-15m-123")
        assert result is not None
        assert result.condition_id == "0xabc123"
        assert result.symbol == "BTCUSDT"
        assert result.outcome_prices["Up"] == 0.92
        assert result.outcome_prices["Down"] == 0.08
        assert result.token_ids["Up"] == "tok1"

    def test_parse_market_missing_outcomes(self):
        scanner = MarketScanner(["BTCUSDT"])
        mkt_data = {
            "conditionId": "0xabc",
            "outcomes": "[]",
            "outcomePrices": "[]",
            "clobTokenIds": "[]",
            "eventStartTime": "2026-02-09T15:45:00Z",
            "endDate": "2026-02-09T16:00:00Z",
        }
        result = scanner._parse_market(mkt_data, "BTCUSDT", "btc-updown-15m-123")
        assert result is None

    def test_cleanup_expired(self):
        scanner = MarketScanner(["BTCUSDT"])
        now = time.time()
        scanner._markets["old"] = MinuteMarket(
            condition_id="old",
            slug="old",
            symbol="BTCUSDT",
            event_start=now - 1200,
            end_time=now - 120,
            token_ids={},
            outcome_prices={},
        )
        scanner._markets["active"] = MinuteMarket(
            condition_id="active",
            slug="active",
            symbol="BTCUSDT",
            event_start=now - 300,
            end_time=now + 600,
            token_ids={},
            outcome_prices={},
        )
        scanner._cleanup_expired(now)
        assert "old" not in scanner._markets
        assert "active" in scanner._markets


# === CryptoMinuteEngine ===


def _make_market(
    slug: str = "btc-updown-15m-123",
    symbol: str = "BTCUSDT",
    up_price: float = 0.92,
    down_price: float = 0.08,
    time_offset: int = 180,
) -> MinuteMarket:
    """Create a test market expiring in `time_offset` seconds."""
    now = time.time()
    return MinuteMarket(
        condition_id="0xtest",
        slug=slug,
        symbol=symbol,
        event_start=now - 720,
        end_time=now + time_offset,
        token_ids={"Up": "tok-up", "Down": "tok-down"},
        outcome_prices={"Up": up_price, "Down": down_price},
    )


class TestCryptoMinuteEngine:
    def _make_engine(self, tmp_path: Path) -> CryptoMinuteEngine:
        db_path = tmp_path / "test_arb.db"
        db_url = f"sqlite:///{db_path}"
        reset_engines()
        init_db(db_url)
        poller = BinanceSpotPoller(["BTCUSDT", "ETHUSDT"])
        # Stub poll() so it doesn't hit real Binance and override test prices
        poller.poll = AsyncMock(side_effect=lambda: dict(poller._prices))
        scanner = MarketScanner(["BTCUSDT", "ETHUSDT"])
        engine = CryptoMinuteEngine(poller=poller, scanner=scanner, database_url=db_url)
        engine.paper_file = tmp_path / "test_trades.jsonl"
        return engine

    @pytest.mark.asyncio
    async def test_scan_finds_time_decay(self, tmp_path):
        """Time decay: expensive side > threshold, gap large enough."""
        engine = self._make_engine(tmp_path)

        # Setup market with Up at 92c
        market = _make_market(up_price=0.92, down_price=0.08, time_offset=180)
        engine.scanner._markets["btc-updown-15m-123"] = market

        # Spot price moved away from start (large gap)
        engine.poller._prices = {"BTCUSDT": 70000.0, "ETHUSDT": 3500.0}
        engine._start_prices["btc-updown-15m-123"] = 69700.0  # 0.43% gap

        opps = await engine.scan_once()
        td_opps = [o for o in opps if o.strategy == "time_decay"]
        assert len(td_opps) == 1
        assert td_opps[0].side == "Up"
        assert td_opps[0].entry_price == 0.92

    @pytest.mark.asyncio
    async def test_scan_finds_long_vol(self, tmp_path):
        """Long vol: cheap side < threshold, gap small enough."""
        engine = self._make_engine(tmp_path)

        # Setup market with Down at 8c (cheap)
        market = _make_market(up_price=0.92, down_price=0.08, time_offset=180)
        engine.scanner._markets["btc-updown-15m-123"] = market

        # Spot barely moved (small gap)
        engine.poller._prices = {"BTCUSDT": 70050.0, "ETHUSDT": 3500.0}
        engine._start_prices["btc-updown-15m-123"] = 69900.0  # 0.21% gap

        opps = await engine.scan_once()
        lv_opps = [o for o in opps if o.strategy == "long_vol"]
        assert len(lv_opps) == 1
        assert lv_opps[0].side == "Down"
        assert lv_opps[0].entry_price == 0.08

    @pytest.mark.asyncio
    async def test_scan_symmetric_yes_cheap(self, tmp_path):
        """Long vol works when Up is the cheap side too."""
        engine = self._make_engine(tmp_path)

        # Up is cheap (8c), Down is expensive (92c)
        market = _make_market(up_price=0.08, down_price=0.92, time_offset=180)
        engine.scanner._markets["btc-updown-15m-123"] = market

        engine.poller._prices = {"BTCUSDT": 69950.0, "ETHUSDT": 3500.0}
        engine._start_prices["btc-updown-15m-123"] = 70000.0  # 0.07% gap

        opps = await engine.scan_once()
        lv_opps = [o for o in opps if o.strategy == "long_vol"]
        assert len(lv_opps) == 1
        assert lv_opps[0].side == "Up"  # Up is cheap this time

    @pytest.mark.asyncio
    async def test_scan_outside_time_window(self, tmp_path):
        """No opps if time remaining is outside entry window."""
        engine = self._make_engine(tmp_path)

        # 10 minutes remaining - outside the 2-5 min window
        market = _make_market(up_price=0.92, down_price=0.08, time_offset=600)
        engine.scanner._markets["btc-updown-15m-123"] = market

        engine.poller._prices = {"BTCUSDT": 70000.0}
        engine._start_prices["btc-updown-15m-123"] = 69700.0

        opps = await engine.scan_once()
        assert len(opps) == 0

    @pytest.mark.asyncio
    async def test_scan_gap_too_small_for_time_decay(self, tmp_path):
        """Time decay rejected if gap is too small."""
        engine = self._make_engine(tmp_path)

        market = _make_market(up_price=0.92, down_price=0.08, time_offset=180)
        engine.scanner._markets["btc-updown-15m-123"] = market

        # Tiny gap: 0.01%
        engine.poller._prices = {"BTCUSDT": 70007.0}
        engine._start_prices["btc-updown-15m-123"] = 70000.0

        opps = await engine.scan_once()
        td_opps = [o for o in opps if o.strategy == "time_decay"]
        assert len(td_opps) == 0

    @pytest.mark.asyncio
    async def test_scan_gap_too_large_for_long_vol(self, tmp_path):
        """Long vol rejected if gap is too large (reversal unlikely)."""
        engine = self._make_engine(tmp_path)

        market = _make_market(up_price=0.92, down_price=0.08, time_offset=180)
        engine.scanner._markets["btc-updown-15m-123"] = market

        # Large gap: 1%
        engine.poller._prices = {"BTCUSDT": 70700.0}
        engine._start_prices["btc-updown-15m-123"] = 70000.0

        opps = await engine.scan_once()
        lv_opps = [o for o in opps if o.strategy == "long_vol"]
        assert len(lv_opps) == 0

    @pytest.mark.asyncio
    async def test_no_duplicate_entries(self, tmp_path):
        """Same market+strategy only entered once."""
        engine = self._make_engine(tmp_path)

        market = _make_market(up_price=0.92, down_price=0.08, time_offset=180)
        engine.scanner._markets["btc-updown-15m-123"] = market
        engine.poller._prices = {"BTCUSDT": 70000.0}
        engine._start_prices["btc-updown-15m-123"] = 69700.0

        # First scan
        opps1 = await engine.scan_once()
        for opp in opps1:
            engine.enter_paper_trade(opp)

        # Second scan - same market
        opps2 = await engine.scan_once()
        # Should not find any more opps for already entered strategies
        new_strategies = {o.strategy for o in opps2}
        entered_strategies = engine._entered_markets.get("btc-updown-15m-123", set())
        assert new_strategies.isdisjoint(entered_strategies)

    def test_enter_paper_trade(self, tmp_path):
        engine = self._make_engine(tmp_path)
        market = _make_market()

        opp = MinuteOpportunity(
            market=market,
            strategy="long_vol",
            side="Down",
            entry_price=0.08,
            spot_price=70050.0,
            gap_pct=0.21,
            time_remaining_s=180,
            potential_profit=0.92,
        )

        trade = engine.enter_paper_trade(opp)
        assert trade.strategy == "long_vol"
        assert trade.side == "Down"
        assert trade.entry_price == 0.08
        assert trade.gap_bucket == "medium"
        assert trade.time_bucket == "3-4min"  # 180s = 3.0min → "3-4min" bucket
        assert len(engine._open_trades) == 1

    @pytest.mark.asyncio
    async def test_resolve_winning_trade(self, tmp_path):
        engine = self._make_engine(tmp_path)

        # Create an expired market where Down won
        now = time.time()
        market = MinuteMarket(
            condition_id="0xtest",
            slug="btc-updown-15m-done",
            symbol="BTCUSDT",
            event_start=now - 1200,
            end_time=now - 10,  # already expired
            token_ids={"Up": "tok-up", "Down": "tok-down"},
            outcome_prices={"Up": 0.0, "Down": 1.0},  # Down won
        )
        engine.scanner._markets["btc-updown-15m-done"] = market
        engine.poller._prices = {"BTCUSDT": 69500.0}

        # Add an open long_vol trade that bought Down at 8c
        trade = PaperTrade(
            id="test-001",
            strategy="long_vol",
            symbol="BTCUSDT",
            market_slug="btc-updown-15m-done",
            side="Down",
            entry_price=0.08,
            size_usd=10.0,
        )
        engine._open_trades.append(trade)

        resolved = await engine.resolve_expired_trades()
        assert len(resolved) == 1
        assert resolved[0].won is True
        assert resolved[0].pnl_usd > 0
        assert len(engine._open_trades) == 0

        # Check JSONL backup was written
        assert engine.paper_file.exists()
        lines = engine.paper_file.read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["strategy"] == "long_vol"
        assert data["won"] is True

        # Check DB was written with correct strategy_tag
        from sqlalchemy import select
        from src.db.database import get_sync_session
        from src.db.models import LiveObservation as LO, PaperTrade as PT

        session = get_sync_session(engine._database_url)
        obs = session.execute(select(LO)).scalars().all()
        assert len(obs) == 1
        assert obs[0].game_state["strategy_tag"] == "crypto_minute_long_vol"
        trades_db = session.execute(select(PT)).scalars().all()
        assert len(trades_db) == 1
        assert trades_db[0].pnl > 0
        session.close()

    @pytest.mark.asyncio
    async def test_resolve_losing_trade(self, tmp_path):
        engine = self._make_engine(tmp_path)

        now = time.time()
        market = MinuteMarket(
            condition_id="0xtest",
            slug="btc-updown-15m-lost",
            symbol="BTCUSDT",
            event_start=now - 1200,
            end_time=now - 10,
            token_ids={"Up": "tok-up", "Down": "tok-down"},
            outcome_prices={"Up": 1.0, "Down": 0.0},  # Up won
        )
        engine.scanner._markets["btc-updown-15m-lost"] = market
        engine.poller._prices = {"BTCUSDT": 71000.0}

        # Long vol bought Down at 8c, but Up won
        trade = PaperTrade(
            id="test-002",
            strategy="long_vol",
            symbol="BTCUSDT",
            market_slug="btc-updown-15m-lost",
            side="Down",
            entry_price=0.08,
            size_usd=10.0,
        )
        engine._open_trades.append(trade)

        resolved = await engine.resolve_expired_trades()
        assert len(resolved) == 1
        assert resolved[0].won is False
        assert resolved[0].pnl_usd < 0

    def test_stats(self, tmp_path):
        engine = self._make_engine(tmp_path)
        engine._stats["time_decay"] = {"trades": 10, "wins": 8, "pnl": 15.0}
        engine._stats["long_vol"] = {"trades": 10, "wins": 2, "pnl": 25.0}

        stats = engine.get_stats()
        assert stats["time_decay"]["winrate"] == 80.0
        assert stats["long_vol"]["winrate"] == 20.0
        assert stats["long_vol"]["pnl_usd"] == 25.0

    def test_stats_empty(self, tmp_path):
        engine = self._make_engine(tmp_path)
        stats = engine.get_stats()
        assert stats["time_decay"]["winrate"] == 0.0
        assert stats["long_vol"]["trades"] == 0
