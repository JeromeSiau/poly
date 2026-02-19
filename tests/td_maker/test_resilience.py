from __future__ import annotations
import asyncio
import time
import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock
from src.td_maker.resilience import clob_retry, FeedMonitor


@pytest.mark.asyncio
async def test_clob_retry_success_first_attempt():
    calls = 0
    async def op():
        nonlocal calls
        calls += 1
        return "ok"
    result = await clob_retry(op, operation="test")
    assert result == "ok"
    assert calls == 1


@pytest.mark.asyncio
async def test_clob_retry_retries_on_timeout():
    calls = 0
    async def op():
        nonlocal calls
        calls += 1
        if calls < 3:
            raise httpx.TimeoutException("timeout")
        return "ok"
    result = await clob_retry(op, base_delay=0.001, operation="test")
    assert result == "ok"
    assert calls == 3


@pytest.mark.asyncio
async def test_clob_retry_raises_after_max_attempts():
    async def op():
        raise httpx.TimeoutException("timeout")
    with pytest.raises(httpx.TimeoutException):
        await clob_retry(op, max_attempts=2, base_delay=0.001, operation="test")


def test_feed_monitor_not_stale_initially():
    feed = MagicMock()
    feed.last_message_at = time.time()
    monitor = FeedMonitor(feed, stale_threshold=30, name="test")
    assert monitor.is_stale() is False


def test_feed_monitor_stale_after_threshold():
    feed = MagicMock()
    feed.last_message_at = time.time() - 60
    monitor = FeedMonitor(feed, stale_threshold=30, name="test")
    assert monitor.is_stale() is True
