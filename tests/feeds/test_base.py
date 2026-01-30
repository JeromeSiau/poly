# tests/feeds/test_base.py
import pytest
from abc import ABC
from src.feeds.base import BaseFeed, FeedEvent


def test_feed_event_creation():
    event = FeedEvent(
        source="test",
        event_type="kill",
        game="lol",
        data={"player": "Faker", "kills": 5},
        timestamp=1234567890.0
    )
    assert event.source == "test"
    assert event.data["player"] == "Faker"


def test_base_feed_is_abstract():
    with pytest.raises(TypeError):
        BaseFeed()  # Cannot instantiate abstract class


class MockFeed(BaseFeed):
    async def connect(self):
        self._connected = True

    async def disconnect(self):
        self._connected = False

    async def subscribe(self, game: str, match_id: str):
        pass


@pytest.mark.asyncio
async def test_mock_feed_connects():
    feed = MockFeed()
    await feed.connect()
    assert feed._connected is True
