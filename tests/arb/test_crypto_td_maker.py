"""Tests for CryptoTDMaker paper fill detection and sequential ladder."""

import asyncio
import time
from types import SimpleNamespace

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
        feed = _make_feed_both_sides(CID, bid_up=0.20, ask_up=0.22,
                                     bid_down=0.80, ask_down=0.82)
        maker = _make_ladder_maker(feed, ladder_rungs=2)

        # Simulate first rung already filled.
        maker._cid_fill_count[CID] = 1
        maker.positions[CID] = OpenPosition(
            condition_id=CID, outcome="Down", token_id=TOK_DOWN,
            entry_price=0.75, size_usd=5.0, shares=6.67,
            filled_at=time.time(), side="BUY",
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
            filled_at=time.time(), side="BUY",
        )

        await maker._maker_tick()

        assert len(maker.active_orders) == 0

    @pytest.mark.asyncio
    async def test_five_rungs_sequential_order(self):
        """With 5 rungs, each tick places only the next unfilled rung."""
        feed = _make_feed_both_sides(CID, bid_up=0.20, ask_up=0.22,
                                     bid_down=0.80, ask_down=0.82)
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
        """When bid goes out of range and order is cancelled, _rung_placed is cleaned
        so the rung can be re-placed when bid returns to range."""
        # Start with bid in range → rung[0] placed.
        feed = _make_feed_both_sides(CID, bid_up=0.20, ask_up=0.22,
                                     bid_down=0.80, ask_down=0.82)
        maker = _make_ladder_maker(feed, ladder_rungs=2)

        await maker._maker_tick()
        assert len(_orders_for_outcome(maker, "Down")) == 1
        assert len(maker._rung_placed) == 1

        # Bid goes ABOVE max_bid (0.85) → out of range → cancel.
        # (Using bid > max_bid avoids triggering bid-through paper fill.)
        feed._best_cache[TOK_DOWN] = (0.90, 100.0, 0.92, 100.0)
        feed._best_cache[TOK_UP] = (0.10, 100.0, 0.12, 100.0)
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
