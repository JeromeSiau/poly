"""TDD tests for market_volume_usd tracking in SlotCollector."""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scripts.run_slot_collector import ActiveSlot, SlotCollector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_collector() -> SlotCollector:
    polymarket = MagicMock()
    polymarket.subscribe_market = AsyncMock()
    polymarket.flush_subscriptions = AsyncMock()
    polymarket.get_best_levels = MagicMock(return_value=(None, None, None, None))
    chainlink = MagicMock()
    chainlink.get_price = MagicMock(return_value=None)
    # Patch DB engine so SQLite pool-size incompatibility doesn't surface in unit tests
    with (
        patch("scripts.run_slot_collector.create_async_engine"),
        patch("scripts.run_slot_collector.async_sessionmaker"),
    ):
        collector = SlotCollector(
            polymarket=polymarket,
            chainlink=chainlink,
            symbols=["BTC"],
            db_url="sqlite+aiosqlite:///:memory:",
            slot_duration=900,
        )
    return collector


def _current_slot_ts() -> int:
    """Return the start timestamp of the current 15-min slot."""
    return int(time.time() // 900) * 900


def _make_raw_market(
    cid: str = "cid1",
    slug: str | None = None,
    volume: float = 50000.0,
) -> dict[str, Any]:
    if slug is None:
        slug = f"btc-updown-15m-{_current_slot_ts()}"
    return {
        "conditionId": cid,
        "outcomes": '["Up","Down"]',
        "clobTokenIds": '["tok1","tok2"]',
        "events": [{"slug": slug}],
        "volume": str(volume),  # Gamma returns volume as numeric string
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMarketVolumeUsd:
    def test_active_slot_accepts_market_volume_usd(self):
        """ActiveSlot stores market_volume_usd when provided."""
        slot = ActiveSlot(
            symbol="BTC",
            slot_ts=1771079400,
            slot_duration=900,
            condition_id="cid1",
            chainlink_sym="btc/usd",
            ref_price=95000.0,
            outcomes=["Up", "Down"],
            token_ids=["tok1", "tok2"],
            market_volume_usd=50000.0,
        )
        assert slot.market_volume_usd == 50000.0

    def test_active_slot_volume_defaults_to_none(self):
        """ActiveSlot.market_volume_usd defaults to None when omitted."""
        slot = ActiveSlot(
            symbol="BTC",
            slot_ts=1771079400,
            slot_duration=900,
            condition_id="cid1",
            chainlink_sym="btc/usd",
            ref_price=95000.0,
            outcomes=["Up", "Down"],
            token_ids=["tok1", "tok2"],
        )
        assert slot.market_volume_usd is None

    @pytest.mark.asyncio
    async def test_discover_stores_volume_on_new_slot(self):
        """_discover() extracts volume from Gamma response for new slots."""
        collector = _make_collector()
        collector._http_client = MagicMock()

        raw_market = _make_raw_market(cid="cid1", volume=75000.0)

        with patch(
            "scripts.run_slot_collector.fetch_crypto_markets",
            new=AsyncMock(return_value=[raw_market]),
        ):
            await collector._discover()

        assert "cid1" in collector._active_slots
        assert collector._active_slots["cid1"].market_volume_usd == 75000.0

    @pytest.mark.asyncio
    async def test_discover_refreshes_volume_on_existing_slot(self):
        """_discover() updates market_volume_usd on already-known slots."""
        collector = _make_collector()
        collector._http_client = MagicMock()

        slot_ts = _current_slot_ts()
        slug = f"btc-updown-15m-{slot_ts}"

        # Pre-populate with stale volume
        collector._known_cids.add("cid1")
        slot = ActiveSlot(
            symbol="BTC",
            slot_ts=slot_ts,
            slot_duration=900,
            condition_id="cid1",
            chainlink_sym="btc/usd",
            ref_price=95000.0,
            outcomes=["Up", "Down"],
            token_ids=["tok1", "tok2"],
            market_volume_usd=30000.0,
        )
        collector._active_slots["cid1"] = slot

        raw_market = _make_raw_market(cid="cid1", slug=slug, volume=80000.0)

        with patch(
            "scripts.run_slot_collector.fetch_crypto_markets",
            new=AsyncMock(return_value=[raw_market]),
        ):
            await collector._discover()

        assert collector._active_slots["cid1"].market_volume_usd == 80000.0

    def test_snapshot_includes_market_volume_usd(self):
        """_take_snapshots() passes market_volume_usd to the enqueued SlotSnapshot."""
        collector = _make_collector()

        slot = ActiveSlot(
            symbol="BTC",
            slot_ts=_current_slot_ts(),
            slot_duration=900,
            condition_id="cid1",
            chainlink_sym="btc/usd",
            ref_price=95000.0,
            outcomes=["Up", "Down"],
            token_ids=["tok1", "tok2"],
            market_volume_usd=55000.0,
        )
        collector._active_slots["cid1"] = slot

        collector._take_snapshots()

        assert not collector._write_queue.empty()
        snapshot = collector._write_queue.get_nowait()
        assert snapshot.market_volume_usd == 55000.0
