"""Tests for CryptoTDMaker paper fill detection and sequential ladder."""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from src.feeds.polymarket import PolymarketFeed, UserTradeEvent
from src.execution.trade_manager import TradeManager
from scripts.run_crypto_td_maker import CryptoTDMaker, PassiveOrder, OpenPosition


def _make_feed_with_book(
    cid: str, outcome: str, token_id: str, bid: float, ask: float,
    bid_sz: float = 100.0, ask_sz: float = 100.0,
) -> PolymarketFeed:
    """Create a PolymarketFeed with a fake orderbook injected."""
    feed = PolymarketFeed.__new__(PolymarketFeed)
    feed._token_map = {(cid, outcome): token_id}
    feed._best_cache = {token_id: (bid, bid_sz, ask, ask_sz)}
    feed.book_updated = asyncio.Event()
    feed._connected = False
    feed._local_orderbook = {}
    feed._subscribed_tokens = set()
    feed._connection_task = None
    feed._shutdown = False
    feed._ws = None
    return feed


def _make_maker(feed: PolymarketFeed, **kwargs) -> CryptoTDMaker:
    """Create a minimal CryptoTDMaker for testing."""
    defaults = dict(
        executor=None,
        polymarket=feed,
        user_feed=None,
        manager=None,
        symbols=["BTCUSDT"],
        target_bid=0.75,
        max_bid=0.85,
        order_size_usd=10.0,
        max_total_exposure_usd=200.0,
        paper_mode=True,
    )
    defaults.update(kwargs)
    return CryptoTDMaker(**defaults)


def _inject_order(maker: CryptoTDMaker, cid: str, outcome: str,
                   token_id: str, price: float, placed_at: float) -> str:
    """Inject a fake pending passive order into the maker."""
    oid = f"paper_{id(maker)}"
    order = PassiveOrder(
        order_id=oid, condition_id=cid, outcome=outcome,
        token_id=token_id, price=price, size_usd=10.0,
        placed_at=placed_at,
    )
    maker.active_orders[oid] = order
    maker._orders_by_cid_outcome[(cid, outcome)] = oid
    maker.market_outcomes[cid] = ["Up", "Down"]
    return oid


# ---------------------------------------------------------------------------
# BUG: current paper fill never triggers with normal spread
# ---------------------------------------------------------------------------


class TestPaperFillBug:
    """Demonstrates that paper fills never trigger when ask > bid (normal spread)."""

    def test_no_fill_with_normal_spread(self):
        """With bid=0.80 ask=0.82, order at 0.80 should NOT fill under old logic
        (ask 0.82 > order price 0.80), confirming the bug."""
        cid, outcome, token_id = "0xcond1", "Up", "tok_up"
        feed = _make_feed_with_book(cid, outcome, token_id, bid=0.80, ask=0.82)
        maker = _make_maker(feed)
        oid = _inject_order(maker, cid, outcome, token_id, price=0.80, placed_at=time.time())

        maker._check_fills_paper(time.time())

        # Bug: order is still pending, never fills
        assert oid in maker.active_orders
        assert maker.total_fills == 0


# ---------------------------------------------------------------------------
# Fix 1: bid-through fill (bid drops below our order price)
# ---------------------------------------------------------------------------


class TestBidThroughFill:
    """When the bid drops below our order price, someone sold through our level."""

    def test_fill_when_bid_drops_below_order(self):
        """Bid was 0.80 when we placed, now dropped to 0.78 → our level got consumed → fill."""
        cid, outcome, token_id = "0xcond1", "Up", "tok_up"
        # Bid dropped from 0.80 to 0.78, ask at 0.81
        feed = _make_feed_with_book(cid, outcome, token_id, bid=0.78, ask=0.81)
        maker = _make_maker(feed)
        oid = _inject_order(maker, cid, outcome, token_id, price=0.80, placed_at=time.time())

        maker._check_fills_paper(time.time())

        assert oid not in maker.active_orders
        assert maker.total_fills == 1
        assert cid in maker.positions
        assert maker.positions[cid].entry_price == 0.80

    def test_no_fill_when_bid_equals_order(self):
        """Bid still at our price → order is in the queue but not yet filled."""
        cid, outcome, token_id = "0xcond1", "Up", "tok_up"
        feed = _make_feed_with_book(cid, outcome, token_id, bid=0.80, ask=0.82)
        maker = _make_maker(feed)
        _inject_order(maker, cid, outcome, token_id, price=0.80, placed_at=time.time())

        maker._check_fills_paper(time.time())

        assert maker.total_fills == 0

    def test_no_fill_when_bid_above_order(self):
        """Bid moved up to 0.81 (above our 0.80 order) → no fill, book improved."""
        cid, outcome, token_id = "0xcond1", "Up", "tok_up"
        feed = _make_feed_with_book(cid, outcome, token_id, bid=0.81, ask=0.83)
        maker = _make_maker(feed)
        _inject_order(maker, cid, outcome, token_id, price=0.80, placed_at=time.time())

        maker._check_fills_paper(time.time())

        assert maker.total_fills == 0


# ---------------------------------------------------------------------------
# Fix 2: time-at-bid fill (order at best bid long enough with tight spread)
# ---------------------------------------------------------------------------


class TestTimeAtBidFill:
    """When we sit at the best bid with a tight spread for long enough, simulate fill."""

    def test_fill_after_timeout_at_best_bid_tight_spread(self):
        """Order at best bid (0.80) for 30+s, spread 0.02 → fill."""
        cid, outcome, token_id = "0xcond1", "Up", "tok_up"
        feed = _make_feed_with_book(cid, outcome, token_id, bid=0.80, ask=0.82)
        maker = _make_maker(feed)
        placed_at = time.time() - 35  # placed 35s ago
        oid = _inject_order(maker, cid, outcome, token_id, price=0.80, placed_at=placed_at)

        maker._check_fills_paper(time.time())

        assert oid not in maker.active_orders
        assert maker.total_fills == 1
        assert maker.positions[cid].entry_price == 0.80

    def test_no_fill_before_timeout(self):
        """Order at best bid for only 10s → not enough time, no fill."""
        cid, outcome, token_id = "0xcond1", "Up", "tok_up"
        feed = _make_feed_with_book(cid, outcome, token_id, bid=0.80, ask=0.82)
        maker = _make_maker(feed)
        placed_at = time.time() - 10  # only 10s ago
        _inject_order(maker, cid, outcome, token_id, price=0.80, placed_at=placed_at)

        maker._check_fills_paper(time.time())

        assert maker.total_fills == 0

    def test_no_fill_wide_spread(self):
        """Order at best bid for 60s but spread is 0.05 → too wide, no fill."""
        cid, outcome, token_id = "0xcond1", "Up", "tok_up"
        feed = _make_feed_with_book(cid, outcome, token_id, bid=0.80, ask=0.85)
        maker = _make_maker(feed)
        placed_at = time.time() - 60
        _inject_order(maker, cid, outcome, token_id, price=0.80, placed_at=placed_at)

        maker._check_fills_paper(time.time())

        assert maker.total_fills == 0

    def test_no_fill_not_at_best_bid(self):
        """Order at 0.80 but best bid is now 0.82 → we're behind in queue, no time fill."""
        cid, outcome, token_id = "0xcond1", "Up", "tok_up"
        feed = _make_feed_with_book(cid, outcome, token_id, bid=0.82, ask=0.84)
        maker = _make_maker(feed)
        placed_at = time.time() - 60
        _inject_order(maker, cid, outcome, token_id, price=0.80, placed_at=placed_at)

        maker._check_fills_paper(time.time())

        assert maker.total_fills == 0


# ---------------------------------------------------------------------------
# Original ask-cross fill still works
# ---------------------------------------------------------------------------


class TestAskCrossFill:
    """The original condition (ask <= order price) should still trigger fills."""

    def test_fill_when_ask_drops_to_order_price(self):
        """Ask collapsed to our bid level → immediate fill."""
        cid, outcome, token_id = "0xcond1", "Up", "tok_up"
        feed = _make_feed_with_book(cid, outcome, token_id, bid=0.79, ask=0.80)
        maker = _make_maker(feed)
        oid = _inject_order(maker, cid, outcome, token_id, price=0.80, placed_at=time.time())

        maker._check_fills_paper(time.time())

        assert oid not in maker.active_orders
        assert maker.total_fills == 1


# ---------------------------------------------------------------------------
# Sequential ladder: rungs placed one at a time, lowest first
# ---------------------------------------------------------------------------

CID = "0xcond_ladder"
TOK_UP = "tok_up_l"
TOK_DOWN = "tok_down_l"


def _make_feed_both_sides(
    cid: str, bid_up: float, ask_up: float,
    bid_down: float | None = None, ask_down: float | None = None,
) -> PolymarketFeed:
    """Create a PolymarketFeed with order books for both Up and Down outcomes."""
    if bid_down is None:
        bid_down = round(1.0 - ask_up, 2)
    if ask_down is None:
        ask_down = round(1.0 - bid_up, 2)
    feed = PolymarketFeed.__new__(PolymarketFeed)
    feed._token_map = {(cid, "Up"): TOK_UP, (cid, "Down"): TOK_DOWN}
    feed._best_cache = {
        TOK_UP: (bid_up, 100.0, ask_up, 100.0),
        TOK_DOWN: (bid_down, 100.0, ask_down, 100.0),
    }
    feed.book_updated = asyncio.Event()
    feed._connected = False
    feed._local_orderbook = {}
    feed._subscribed_tokens = set()
    feed._connection_task = None
    feed._shutdown = False
    feed._ws = None
    feed.last_update_ts = time.monotonic()
    return feed


def _make_ladder_maker(
    feed: PolymarketFeed, ladder_rungs: int = 2, **kwargs,
) -> CryptoTDMaker:
    """Create a CryptoTDMaker wired for ladder testing (paper, no guard)."""
    manager = TradeManager(
        strategy="test_ladder", paper=True,
        notify_bids=False, notify_fills=False, notify_closes=False,
    )
    defaults = dict(
        executor=None,
        polymarket=feed,
        user_feed=None,
        manager=manager,
        symbols=["BTCUSDT"],
        target_bid=0.75,
        max_bid=0.85,
        order_size_usd=5.0,
        max_total_exposure_usd=200.0,
        paper_mode=True,
        ladder_rungs=ladder_rungs,
    )
    defaults.update(kwargs)
    maker = CryptoTDMaker(**defaults)
    # Register a market so _maker_tick can see it.
    maker.known_markets[CID] = {"events": [{"slug": "btc-updown-1700000000"}]}
    maker.market_outcomes[CID] = ["Up", "Down"]
    maker.market_tokens[(CID, "Up")] = TOK_UP
    maker.market_tokens[(CID, "Down")] = TOK_DOWN
    return maker


def _orders_for_outcome(maker: CryptoTDMaker, outcome: str) -> list[PassiveOrder]:
    """Return all active orders for a given outcome."""
    return [o for o in maker.active_orders.values() if o.outcome == outcome]


class TestSequentialLadder:
    """Sequential ladder: rung[0] first, rung[1] only after rung[0] fills, etc."""

    @pytest.mark.asyncio
    async def test_only_first_rung_placed_initially(self):
        """With ladder_rungs=2 and no fills, only rung[0] (0.75) should be placed."""
        feed = _make_feed_both_sides(CID, bid_up=0.20, ask_up=0.22,
                                     bid_down=0.80, ask_down=0.82)
        maker = _make_ladder_maker(feed, ladder_rungs=2)

        await maker._maker_tick()

        down_orders = _orders_for_outcome(maker, "Down")
        assert len(down_orders) == 1, f"Expected 1 order, got {len(down_orders)}"
        assert down_orders[0].price == 0.75

    @pytest.mark.asyncio
    async def test_second_rung_after_first_fill(self):
        """After rung[0] fills (fill_count=1), rung[1] (0.85) should be placed."""
        feed = _make_feed_both_sides(CID, bid_up=0.12, ask_up=0.14,
                                     bid_down=0.80, ask_down=0.88)
        maker = _make_ladder_maker(feed, ladder_rungs=2)

        # Simulate first rung already filled.
        maker._cid_fill_count[CID] = 1
        maker.positions[CID] = OpenPosition(
            condition_id=CID, outcome="Down", token_id=TOK_DOWN,
            entry_price=0.75, size_usd=5.0, shares=6.67,
            filled_at=time.time(),
        )

        await maker._maker_tick()

        down_orders = _orders_for_outcome(maker, "Down")
        assert len(down_orders) == 1, f"Expected 1 order, got {len(down_orders)}"
        assert down_orders[0].price == 0.85

    @pytest.mark.asyncio
    async def test_no_more_orders_after_all_rungs_filled(self):
        """After all rungs filled, no new orders should be placed."""
        feed = _make_feed_both_sides(CID, bid_up=0.20, ask_up=0.22,
                                     bid_down=0.80, ask_down=0.82)
        maker = _make_ladder_maker(feed, ladder_rungs=2)

        maker._cid_fill_count[CID] = 2
        maker.positions[CID] = OpenPosition(
            condition_id=CID, outcome="Down", token_id=TOK_DOWN,
            entry_price=0.80, size_usd=10.0, shares=13.0,
            filled_at=time.time(),
        )

        await maker._maker_tick()

        assert len(maker.active_orders) == 0

    @pytest.mark.asyncio
    async def test_five_rungs_sequential_order(self):
        """With 5 rungs, each tick places only the next unfilled rung."""
        feed = _make_feed_both_sides(CID, bid_up=0.12, ask_up=0.14,
                                     bid_down=0.80, ask_down=0.88)
        maker = _make_ladder_maker(feed, ladder_rungs=5)
        expected_prices = maker.rung_prices  # [0.75, 0.78, 0.80, 0.83, 0.85]

        for i, expected_price in enumerate(expected_prices):
            # Clear orders from previous tick to simulate fill.
            if i > 0:
                maker._cid_fill_count[CID] = i
                # Clear active orders (simulating fill + cleanup).
                maker.active_orders.clear()
                maker._orders_by_cid_outcome.clear()

            await maker._maker_tick()

            down_orders = _orders_for_outcome(maker, "Down")
            assert len(down_orders) == 1, (
                f"Rung {i}: expected 1 order, got {len(down_orders)}"
            )
            assert down_orders[0].price == expected_price, (
                f"Rung {i}: expected price {expected_price}, got {down_orders[0].price}"
            )

    @pytest.mark.asyncio
    async def test_cancel_clears_rung_placed(self):
        """When bid drops below target_bid, orders are cancelled and _rung_placed
        is cleaned so the rung can be re-placed when bid returns to range."""
        # Start with bid in range → rung[0] placed.
        feed = _make_feed_both_sides(CID, bid_up=0.20, ask_up=0.22,
                                     bid_down=0.80, ask_down=0.82)
        maker = _make_ladder_maker(feed, ladder_rungs=2)

        await maker._maker_tick()
        assert len(_orders_for_outcome(maker, "Down")) == 1
        assert len(maker._rung_placed) == 1

        # Book disappears (bid=None) → out of range → cancel.
        # (Can't use bid < target_bid because that triggers paper bid-through fill.)
        del feed._best_cache[TOK_DOWN]
        del feed._best_cache[TOK_UP]
        feed.last_update_ts = time.monotonic()

        await maker._maker_tick()
        assert len(maker.active_orders) == 0
        assert len(maker._rung_placed) == 0, (
            "_rung_placed should be empty after cancel"
        )

        # Bid returns to range → rung[0] should be re-placed.
        feed._best_cache[TOK_DOWN] = (0.80, 100.0, 0.82, 100.0)
        feed._best_cache[TOK_UP] = (0.20, 100.0, 0.22, 100.0)
        feed.last_update_ts = time.monotonic()

        await maker._maker_tick()
        down_orders = _orders_for_outcome(maker, "Down")
        assert len(down_orders) == 1
        assert down_orders[0].price == 0.75

    @pytest.mark.asyncio
    async def test_ladder_not_cancelled_when_bid_above_max(self):
        """When bid rises above max_bid, ladder orders stay — the upper rungs
        may now be placeable as maker (bid > max_bid means ask > max_bid)."""
        feed = _make_feed_both_sides(CID, bid_up=0.20, ask_up=0.22,
                                     bid_down=0.80, ask_down=0.82)
        maker = _make_ladder_maker(feed, ladder_rungs=2)

        await maker._maker_tick()
        assert len(_orders_for_outcome(maker, "Down")) == 1
        assert _orders_for_outcome(maker, "Down")[0].price == 0.75

        # Bid rises above max_bid — order should NOT be cancelled.
        feed._best_cache[TOK_DOWN] = (0.90, 100.0, 0.92, 100.0)
        feed._best_cache[TOK_UP] = (0.10, 100.0, 0.12, 100.0)
        feed.last_update_ts = time.monotonic()

        await maker._maker_tick()
        # Order at 0.75 still on book.
        assert len(_orders_for_outcome(maker, "Down")) == 1
        assert _orders_for_outcome(maker, "Down")[0].price == 0.75


# ---------------------------------------------------------------------------
# Phantom fill prevention: partial fills must not spill to next rung
# ---------------------------------------------------------------------------


def _make_fake_user_feed() -> SimpleNamespace:
    """Create a minimal user_feed with an asyncio.Queue for fills."""
    return SimpleNamespace(
        fills=asyncio.Queue(),
        fill_received=asyncio.Event(),
    )


class TestPartialFillPhantom:
    """Partial fills on a CLOB order must NOT be attributed to a different rung.

    Scenario: order A (0.75) fills in two partials. After the first partial
    consumes order A from active_orders, the second partial must NOT match
    order B (0.85) just because it shares the same condition_id.
    """

    @pytest.mark.asyncio
    async def test_second_partial_does_not_match_next_rung(self):
        """Two partials with same maker_order_id: only first matches; second is unmatched."""
        feed = _make_feed_both_sides(CID, bid_up=0.20, ask_up=0.22,
                                     bid_down=0.80, ask_down=0.82)
        user_feed = _make_fake_user_feed()
        maker = _make_ladder_maker(feed, ladder_rungs=2, user_feed=user_feed)
        maker.paper_mode = False  # live mode to use _fill_listener

        # Inject two rung orders for the same market.
        order_a_id = "real_order_75"
        order_b_id = "real_order_85"
        now = time.time()
        order_a = PassiveOrder(
            order_id=order_a_id, condition_id=CID, outcome="Down",
            token_id=TOK_DOWN, price=0.75, size_usd=5.0, placed_at=now,
        )
        order_b = PassiveOrder(
            order_id=order_b_id, condition_id=CID, outcome="Down",
            token_id=TOK_DOWN, price=0.85, size_usd=5.0, placed_at=now,
        )
        maker.active_orders[order_a_id] = order_a
        maker.active_orders[order_b_id] = order_b
        maker._orders_by_cid_outcome[(CID, "Down")] = order_a_id

        # First partial fill — should match order_a.
        evt1 = UserTradeEvent(
            order_id="taker_xyz", market=CID, asset_id=TOK_DOWN,
            side="SELL", price=0.75, size=3.0, status="MATCHED",
            timestamp=now, maker_order_id=order_a_id,
        )
        # Second partial fill — same maker_order_id, arrives after order_a removed.
        evt2 = UserTradeEvent(
            order_id="taker_xyz", market=CID, asset_id=TOK_DOWN,
            side="SELL", price=0.75, size=3.5, status="MATCHED",
            timestamp=now + 0.1, maker_order_id=order_a_id,
        )

        # Enqueue both fills and a sentinel to stop the loop.
        await user_feed.fills.put(evt1)
        await user_feed.fills.put(evt2)

        # Run _fill_listener with a timeout — it loops forever, so we cancel it.
        task = asyncio.create_task(maker._fill_listener())
        # Let events be processed.
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # order_a should be consumed (removed from active_orders).
        assert order_a_id not in maker.active_orders
        # order_b must STILL be in active_orders — not phantom-filled.
        assert order_b_id in maker.active_orders
        assert maker.active_orders[order_b_id].price == 0.85
        # fill_count should be 2 (both partials counted).
        assert maker._cid_fill_count.get(CID, 0) == 2

    @pytest.mark.asyncio
    async def test_placeholder_matched_by_condition_id(self):
        """A fill during placement (placeholder order) should still match by condition_id."""
        feed = _make_feed_both_sides(CID, bid_up=0.20, ask_up=0.22,
                                     bid_down=0.80, ask_down=0.82)
        user_feed = _make_fake_user_feed()
        maker = _make_ladder_maker(feed, ladder_rungs=2, user_feed=user_feed)
        maker.paper_mode = False

        # Inject a placeholder (API call in-flight, real order_id unknown).
        placeholder_id = f"_placing_{CID}_Down_75"
        now = time.time()
        placeholder = PassiveOrder(
            order_id=placeholder_id, condition_id=CID, outcome="Down",
            token_id=TOK_DOWN, price=0.75, size_usd=5.0, placed_at=now,
        )
        maker.active_orders[placeholder_id] = placeholder

        # Fill arrives with a maker_order_id we don't know yet.
        evt = UserTradeEvent(
            order_id="taker_abc", market=CID, asset_id=TOK_DOWN,
            side="SELL", price=0.75, size=6.7, status="MATCHED",
            timestamp=now, maker_order_id="unknown_real_order_id",
        )
        await user_feed.fills.put(evt)

        task = asyncio.create_task(maker._fill_listener())
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Placeholder should be consumed.
        assert placeholder_id not in maker.active_orders
        assert CID in maker.positions
        assert maker._cid_fill_count.get(CID, 0) == 1

    @pytest.mark.asyncio
    async def test_no_maker_order_id_falls_back_to_condition_id(self):
        """When maker_order_id is empty, fall back to broad condition_id match."""
        feed = _make_feed_both_sides(CID, bid_up=0.20, ask_up=0.22,
                                     bid_down=0.80, ask_down=0.82)
        user_feed = _make_fake_user_feed()
        maker = _make_ladder_maker(feed, ladder_rungs=1, user_feed=user_feed)
        maker.paper_mode = False

        order_id = "real_order_75"
        now = time.time()
        order = PassiveOrder(
            order_id=order_id, condition_id=CID, outcome="Down",
            token_id=TOK_DOWN, price=0.75, size_usd=5.0, placed_at=now,
        )
        maker.active_orders[order_id] = order

        # Fill with no maker_order_id (edge case).
        evt = UserTradeEvent(
            order_id="taker_xyz", market=CID, asset_id=TOK_DOWN,
            side="SELL", price=0.75, size=6.7, status="MATCHED",
            timestamp=now, maker_order_id="",
        )
        await user_feed.fills.put(evt)

        task = asyncio.create_task(maker._fill_listener())
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Should match via condition_id fallback.
        assert order_id not in maker.active_orders
        assert CID in maker.positions


# ---------------------------------------------------------------------------
# Stop loss: Chainlink fair value override
# ---------------------------------------------------------------------------


class TestStopLossFairValue:
    """Stop loss should be skipped when Chainlink fair value contradicts low bid."""

    @pytest.mark.asyncio
    async def test_stoploss_skipped_when_fair_value_high(self):
        """Bid at 0.34 but underlying still in our favor -> don't exit."""
        feed = _make_feed_with_book(CID, "Up", TOK_UP, bid=0.34, ask=0.36)
        mock_chainlink = Mock()
        mock_chainlink.get_price.return_value = 70000.0  # BTC up from ref
        maker = _make_maker(feed, stoploss_peak=0.75, stoploss_exit=0.35,
                            chainlink_feed=mock_chainlink)

        now = time.time()
        maker.positions[CID] = OpenPosition(
            condition_id=CID, outcome="Up", token_id=TOK_UP,
            entry_price=0.75, size_usd=10.0, shares=13.33, filled_at=now,
        )
        maker._position_bid_max[CID] = 0.80
        maker._ref_prices[CID] = 69500.0       # ref price at slot start
        maker._cid_chainlink_symbol[CID] = "btc/usd"
        maker._cid_slot_ts[CID] = int(now - 600)  # 10 min into slot -> 5 min remaining

        await maker._check_stop_losses(now)

        # Position should still exist -- stop loss was overridden
        assert CID in maker.positions

    @pytest.mark.asyncio
    async def test_stoploss_triggers_when_fair_value_confirms(self):
        """Bid at 0.34 and underlying moved against us -> exit."""
        feed = _make_feed_with_book(CID, "Up", TOK_UP, bid=0.34, ask=0.36)
        mock_chainlink = Mock()
        mock_chainlink.get_price.return_value = 69000.0  # BTC down from ref
        maker = _make_maker(feed, stoploss_peak=0.75, stoploss_exit=0.35,
                            chainlink_feed=mock_chainlink)

        now = time.time()
        maker.positions[CID] = OpenPosition(
            condition_id=CID, outcome="Up", token_id=TOK_UP,
            entry_price=0.75, size_usd=10.0, shares=13.33, filled_at=now,
        )
        maker._position_bid_max[CID] = 0.80
        maker._ref_prices[CID] = 69500.0
        maker._cid_chainlink_symbol[CID] = "btc/usd"
        maker._cid_slot_ts[CID] = int(now - 600)

        await maker._check_stop_losses(now)

        # Position should be gone -- stop loss confirmed by fair value
        assert CID not in maker.positions

    @pytest.mark.asyncio
    async def test_stoploss_triggers_when_no_chainlink_data(self):
        """No Chainlink data -> fall back to normal stop loss behavior."""
        feed = _make_feed_with_book(CID, "Up", TOK_UP, bid=0.34, ask=0.36)
        maker = _make_maker(feed, stoploss_peak=0.75, stoploss_exit=0.35)
        # No chainlink_feed set -> _estimate_fair_value returns None

        now = time.time()
        maker.positions[CID] = OpenPosition(
            condition_id=CID, outcome="Up", token_id=TOK_UP,
            entry_price=0.75, size_usd=10.0, shares=13.33, filled_at=now,
        )
        maker._position_bid_max[CID] = 0.80

        await maker._check_stop_losses(now)

        # Should trigger normally without fair value override
        assert CID not in maker.positions

    @pytest.mark.asyncio
    async def test_stoploss_empty_book_triggers_despite_high_fair_value(self):
        """Empty book with last_bid below exit triggers stoploss even if
        Chainlink says hold — empty book means market makers pulled out,
        not a flash crash."""
        # bid=None simulates an empty orderbook
        feed = _make_feed_with_book(CID, "Up", TOK_UP, bid=0.34, ask=0.36)
        feed._best_cache[TOK_UP] = (None, 0, None, 0)  # empty book
        mock_chainlink = Mock()
        mock_chainlink.get_price.return_value = 70000.0  # BTC up from ref
        maker = _make_maker(feed, stoploss_peak=0.75, stoploss_exit=0.35,
                            chainlink_feed=mock_chainlink)

        now = time.time()
        maker.positions[CID] = OpenPosition(
            condition_id=CID, outcome="Up", token_id=TOK_UP,
            entry_price=0.75, size_usd=10.0, shares=13.33, filled_at=now,
        )
        maker._position_bid_max[CID] = 0.80
        maker._position_last_bid[CID] = 0.34  # below stoploss_exit
        maker._ref_prices[CID] = 69500.0
        maker._cid_chainlink_symbol[CID] = "btc/usd"
        maker._cid_slot_ts[CID] = int(now - 600)

        await maker._check_stop_losses(now)

        # Empty book = stoploss triggers regardless of fair value
        assert CID not in maker.positions


class TestFairValueEntry:
    """Fair value entry guard blocks overpaying on flash pumps."""

    def test_entry_blocked_when_overpaying(self):
        """BTC moved against us (-0.3%), fair value ~0.04, entry at 0.78 → blocked."""
        feed = _make_feed_with_book(CID, "Up", TOK_UP, bid=0.78, ask=0.80)
        mock_chainlink = Mock()
        mock_chainlink.get_price.return_value = 69290.0  # BTC down ~0.3% from ref
        maker = _make_maker(feed, entry_fair_margin=0.25,
                            chainlink_feed=mock_chainlink)
        now = time.time()
        maker._ref_prices[CID] = 69500.0
        maker._cid_chainlink_symbol[CID] = "btc/usd"
        maker._cid_slot_ts[CID] = int(now - 600)  # 10 min into slot

        result = maker._check_fair_value_entry(CID, "Up", 0.78, now)
        assert result is False

    def test_entry_allowed_when_price_consistent(self):
        """Price 0.78, fair value 0.65, margin 0.25 → overpay 0.13 < 0.25 → allowed."""
        feed = _make_feed_with_book(CID, "Up", TOK_UP, bid=0.78, ask=0.80)
        mock_chainlink = Mock()
        mock_chainlink.get_price.return_value = 70000.0  # BTC up from ref
        maker = _make_maker(feed, entry_fair_margin=0.25,
                            chainlink_feed=mock_chainlink)
        now = time.time()
        maker._ref_prices[CID] = 69500.0
        maker._cid_chainlink_symbol[CID] = "btc/usd"
        maker._cid_slot_ts[CID] = int(now - 300)

        result = maker._check_fair_value_entry(CID, "Up", 0.78, now)
        assert result is True

    def test_entry_allowed_when_disabled(self):
        """entry_fair_margin=0 → always allowed."""
        feed = _make_feed_with_book(CID, "Up", TOK_UP, bid=0.82, ask=0.84)
        maker = _make_maker(feed, entry_fair_margin=0.0)

        result = maker._check_fair_value_entry(CID, "Up", 0.82, time.time())
        assert result is True

    def test_entry_allowed_when_no_chainlink(self):
        """No Chainlink data → allow (graceful degradation)."""
        feed = _make_feed_with_book(CID, "Up", TOK_UP, bid=0.82, ask=0.84)
        maker = _make_maker(feed, entry_fair_margin=0.25)
        # No chainlink_feed

        result = maker._check_fair_value_entry(CID, "Up", 0.82, time.time())
        assert result is True


# ---------------------------------------------------------------------------
# Regression tests for refactor-safe bug fixes
# ---------------------------------------------------------------------------


class TestModelSizingConsistency:
    """Model-scaled sizes must stay consistent across order lifecycle."""

    @pytest.mark.asyncio
    async def test_place_order_keeps_scaled_size(self):
        """Scaled size should propagate to active order (not reset to base size)."""
        feed = _make_feed_both_sides(CID, bid_up=0.20, ask_up=0.22,
                                     bid_down=0.80, ask_down=0.82)
        manager = TradeManager(
            strategy="test_model_sizing",
            paper=True,
            notify_bids=False,
            notify_fills=False,
            notify_closes=False,
        )
        maker = _make_maker(feed, manager=manager, order_size_usd=10.0)
        maker.known_markets[CID] = {"events": [{"slug": "btc-updown-1700000000"}]}
        maker.market_outcomes[CID] = ["Up", "Down"]
        maker.market_tokens[(CID, "Down")] = TOK_DOWN

        maker._model = object()
        maker._last_p_win[(CID, "Down")] = 0.90  # 1.6x -> 16 USD

        oid = await maker._place_order(
            CID, "Down", TOK_DOWN, price=0.80, now=time.time(), order_type="maker"
        )

        assert oid is not None
        assert oid in maker.active_orders
        assert maker.active_orders[oid].size_usd == pytest.approx(16.0, abs=1e-6)

    @pytest.mark.asyncio
    async def test_budget_checks_scaled_size_not_base(self):
        """If scaled size exceeds budget, order must not be placed."""
        feed = _make_feed_both_sides(CID, bid_up=0.20, ask_up=0.22,
                                     bid_down=0.80, ask_down=0.82)
        maker = _make_ladder_maker(feed, ladder_rungs=1, order_size_usd=10.0,
                                   max_total_exposure_usd=12.0)
        maker._model = object()

        def _fake_check_model(cid: str, outcome: str, bid: float) -> float:
            maker._last_order_type[(cid, outcome)] = "maker"
            return 0.90  # 1.6x -> 16 USD

        maker._check_model = _fake_check_model  # type: ignore[method-assign]

        await maker._maker_tick()

        assert len(maker.active_orders) == 0


class TestDbRestoreLadder:
    """DB restore should rebuild ladder fills without losing legs."""

    @pytest.mark.asyncio
    async def test_load_db_state_aggregates_multi_leg_position(self, monkeypatch):
        now = time.time()
        rows = [
            SimpleNamespace(
                order_id="oid_1",
                condition_id=CID,
                outcome="Down",
                token_id=TOK_DOWN,
                price=0.75,
                size_usd=5.0,
                shares=6.6666667,
                status="filled",
                placed_at=now - 60,
                filled_at=now - 55,
            ),
            SimpleNamespace(
                order_id="oid_2",
                condition_id=CID,
                outcome="Down",
                token_id=TOK_DOWN,
                price=0.85,
                size_usd=8.0,
                shares=9.4117647,
                status="filled",
                placed_at=now - 50,
                filled_at=now - 45,
            ),
        ]

        async def _fake_load_orders(**kwargs):
            return rows

        async def _fake_delete_order(**kwargs):
            return True

        monkeypatch.setattr("src.db.td_orders.load_orders", _fake_load_orders)
        monkeypatch.setattr("src.db.td_orders.delete_order", _fake_delete_order)

        feed = _make_feed_both_sides(CID, bid_up=0.20, ask_up=0.22,
                                     bid_down=0.80, ask_down=0.82)
        maker = _make_maker(feed, db_url="sqlite+aiosqlite:///tmp/test_td.sqlite")

        await maker._load_db_state()

        assert CID in maker.positions
        pos = maker.positions[CID]
        assert pos.size_usd == pytest.approx(13.0, abs=1e-6)
        assert pos.shares == pytest.approx(16.0784314, abs=1e-5)
        expected_entry = (
            (6.6666667 * 0.75 + 9.4117647 * 0.85) / (6.6666667 + 9.4117647)
        )
        assert pos.entry_price == pytest.approx(expected_entry, abs=1e-6)
        assert maker._cid_fill_count[CID] == 2
        assert set(maker._position_order_legs[CID]) == {"oid_1", "oid_2"}

    def test_settlement_splits_pnl_across_legs(self):
        feed = _make_feed_both_sides(CID, bid_up=0.20, ask_up=0.22,
                                     bid_down=0.80, ask_down=0.82)
        maker = _make_maker(feed, db_url="sqlite+aiosqlite:///tmp/test_td.sqlite")
        maker._position_order_legs[CID] = {"oid_small": 5.0, "oid_big": 15.0}

        calls = []

        def _capture_db_fire(job, *args, **kwargs):
            calls.append((job, args, kwargs))

        maker._db_fire = _capture_db_fire  # type: ignore[method-assign]
        maker._mark_position_settled_in_db(CID, pnl=20.0, now=1234567890.0)

        assert len(calls) == 2
        # 5/20 and 15/20 split
        leg_pnls = {args[0]: args[1] for _, args, _ in calls}
        assert leg_pnls["oid_small"] == pytest.approx(5.0, abs=1e-6)
        assert leg_pnls["oid_big"] == pytest.approx(15.0, abs=1e-6)


class TestModelFilterInLadder:
    """With model enabled, ladder should honor ML skip decisions."""

    @pytest.mark.asyncio
    async def test_ladder_respects_model_skip(self):
        feed = _make_feed_both_sides(CID, bid_up=0.20, ask_up=0.22,
                                     bid_down=0.80, ask_down=0.82)
        maker = _make_ladder_maker(feed, ladder_rungs=2)
        maker._model = object()

        def _fake_check_model(cid: str, outcome: str, bid: float) -> float:
            maker._last_order_type[(cid, outcome)] = "skip"
            return 0.40

        maker._check_model = _fake_check_model  # type: ignore[method-assign]

        await maker._maker_tick()

        assert len(maker.active_orders) == 0


# ---------------------------------------------------------------------------
# Fill reconciliation: recover fills missed during WS downtime
# ---------------------------------------------------------------------------


class _FakeExecutor:
    """Minimal executor stub returning canned get_order responses."""

    def __init__(self, order_responses: dict[str, dict] | None = None):
        self._order_responses = order_responses or {}
        self.cancelled: list[str] = []

    async def get_order(self, order_id: str) -> dict | None:
        return self._order_responses.get(order_id)

    async def cancel_order(self, order_id: str) -> dict:
        self.cancelled.append(order_id)
        return {"status": "OK"}

    async def get_open_orders(self, market: str = "") -> list[dict]:
        return []


@pytest.mark.asyncio
async def test_reconcile_detects_filled_order():
    """Reconciliation should detect an order filled while WS was down."""
    cid = "0xabc123"
    token_id = "tok_btc_up"
    feed = _make_feed_with_book(cid, "Up", token_id, bid=0.90, ask=0.91)
    executor = _FakeExecutor(order_responses={
        "order_123": {
            "id": "order_123",
            "status": "MATCHED",
            "side": "BUY",
            "price": "0.78",
            "original_size": "12.82",
            "size_matched": "12.82",
            "asset_id": token_id,
            "market": cid,
            "outcome": "Up",
        },
    })
    maker = _make_maker(feed, executor=executor, paper_mode=False)
    maker.manager = Mock(spec=TradeManager)
    maker.manager._pending = {}
    maker.manager.record_fill_direct = AsyncMock()

    # Inject a pending order the bot thinks is still active
    oid = "order_123"
    order = PassiveOrder(
        order_id=oid, condition_id=cid, outcome="Up",
        token_id=token_id, price=0.78, size_usd=10.0,
        placed_at=time.time() - 60,
    )
    maker.active_orders[oid] = order
    maker._orders_by_cid_outcome[(cid, "Up")] = oid
    maker.market_outcomes[cid] = ["Up", "Down"]

    reconciled = await maker._reconcile_fills()

    assert reconciled == 1
    assert oid not in maker.active_orders
    assert cid in maker.positions
    pos = maker.positions[cid]
    assert pos.outcome == "Up"
    assert pos.entry_price == 0.78


@pytest.mark.asyncio
async def test_reconcile_skips_live_orders():
    """Orders still LIVE on the book should not be touched."""
    cid = "0xdef456"
    token_id = "tok_sol_up"
    feed = _make_feed_with_book(cid, "Up", token_id, bid=0.80, ask=0.81)
    executor = _FakeExecutor(order_responses={
        "order_live": {
            "id": "order_live",
            "status": "LIVE",
            "price": "0.80",
            "original_size": "12.5",
            "size_matched": "0",
            "asset_id": token_id,
            "market": cid,
        },
    })
    maker = _make_maker(feed, executor=executor, paper_mode=False)

    order = PassiveOrder(
        order_id="order_live", condition_id=cid, outcome="Up",
        token_id=token_id, price=0.80, size_usd=10.0,
        placed_at=time.time() - 30,
    )
    maker.active_orders["order_live"] = order
    maker._orders_by_cid_outcome[(cid, "Up")] = "order_live"
    maker.market_outcomes[cid] = ["Up", "Down"]

    reconciled = await maker._reconcile_fills()

    assert reconciled == 0
    assert "order_live" in maker.active_orders
    assert cid not in maker.positions


@pytest.mark.asyncio
async def test_reconcile_handles_cancelled_order():
    """CANCELLED orders should be cleaned up from active_orders."""
    cid = "0xghi789"
    token_id = "tok_xrp_up"
    feed = _make_feed_with_book(cid, "Up", token_id, bid=0.70, ask=0.71)
    executor = _FakeExecutor(order_responses={
        "order_cancel": {
            "id": "order_cancel",
            "status": "CANCELLED",
            "price": "0.76",
            "original_size": "13.16",
            "size_matched": "0",
            "asset_id": token_id,
            "market": cid,
        },
    })
    maker = _make_maker(feed, executor=executor, paper_mode=False)

    order = PassiveOrder(
        order_id="order_cancel", condition_id=cid, outcome="Up",
        token_id=token_id, price=0.76, size_usd=10.0,
        placed_at=time.time() - 30,
    )
    maker.active_orders["order_cancel"] = order
    maker._orders_by_cid_outcome[(cid, "Up")] = "order_cancel"
    maker.market_outcomes[cid] = ["Up", "Down"]

    reconciled = await maker._reconcile_fills()

    assert reconciled == 0
    assert "order_cancel" not in maker.active_orders
    assert cid not in maker.positions


@pytest.mark.asyncio
async def test_reconcile_handles_api_failure():
    """If the API returns None (error), the order stays as-is."""
    cid = "0xjkl012"
    token_id = "tok_eth_up"
    feed = _make_feed_with_book(cid, "Up", token_id, bid=0.80, ask=0.81)
    executor = _FakeExecutor(order_responses={})  # returns None for unknown orders
    maker = _make_maker(feed, executor=executor, paper_mode=False)

    order = PassiveOrder(
        order_id="order_unknown", condition_id=cid, outcome="Up",
        token_id=token_id, price=0.80, size_usd=10.0,
        placed_at=time.time() - 30,
    )
    maker.active_orders["order_unknown"] = order
    maker._orders_by_cid_outcome[(cid, "Up")] = "order_unknown"

    reconciled = await maker._reconcile_fills()

    assert reconciled == 0
    assert "order_unknown" in maker.active_orders


@pytest.mark.asyncio
async def test_reconcile_on_startup_recovers_filled_orders():
    """Simulate startup: DB has pending orders that were actually filled on-chain."""
    cid = "0xstartup"
    token_id = "tok_startup"
    feed = _make_feed_with_book(cid, "Up", token_id, bid=0.90, ask=0.91)
    executor = _FakeExecutor(order_responses={
        "startup_order": {
            "id": "startup_order",
            "status": "MATCHED",
            "price": "0.80",
            "original_size": "12.5",
            "size_matched": "12.5",
            "asset_id": token_id,
            "market": cid,
        },
    })
    maker = _make_maker(feed, executor=executor, paper_mode=False)
    maker.manager = Mock(spec=TradeManager)
    maker.manager.record_fill_direct = AsyncMock()
    maker.manager._pending = {}

    # Simulate what _load_db_state would produce: a pending order in active_orders
    order = PassiveOrder(
        order_id="startup_order", condition_id=cid, outcome="Up",
        token_id=token_id, price=0.80, size_usd=10.0,
        placed_at=time.time() - 120,
    )
    maker.active_orders["startup_order"] = order
    maker._orders_by_cid_outcome[(cid, "Up")] = "startup_order"
    maker.market_outcomes[cid] = ["Up", "Down"]

    reconciled = await maker._reconcile_fills()

    assert reconciled == 1
    assert cid in maker.positions
    assert maker.positions[cid].entry_price == 0.80
