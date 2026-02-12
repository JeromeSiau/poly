"""Tests for CryptoTDMaker paper fill detection."""

import asyncio
import time

import pytest

from src.feeds.polymarket import PolymarketFeed
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
