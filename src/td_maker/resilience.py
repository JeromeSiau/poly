# src/td_maker/resilience.py
from __future__ import annotations

import asyncio
import time
from typing import Callable, Awaitable, TypeVar

import httpx
import structlog

logger = structlog.get_logger()
T = TypeVar("T")

RETRYABLE = (httpx.TimeoutException, httpx.HTTPStatusError, httpx.ConnectError)


async def clob_retry(
    coro_factory: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    operation: str = "",
) -> T:
    """Retry an async CLOB/REST call with exponential backoff."""
    last_exc: Exception = RuntimeError("no attempts")
    for attempt in range(max_attempts):
        try:
            return await coro_factory()
        except RETRYABLE as e:
            last_exc = e
            if attempt == max_attempts - 1:
                logger.error("clob_failed", op=operation, error=str(e),
                             attempts=attempt + 1)
                raise
            delay = base_delay * (2 ** attempt)
            logger.warning("clob_retry", op=operation, attempt=attempt + 1,
                           delay=delay, error=str(e))
            await asyncio.sleep(delay)
    raise last_exc


class FeedMonitor:
    """Detects silent WSS disconnections via last_message_at timestamp."""

    def __init__(self, feed, *, stale_threshold: float = 30.0, name: str = ""):
        self.feed = feed
        self.stale_threshold = stale_threshold
        self.name = name

    def is_stale(self) -> bool:
        return (time.time() - self.feed.last_message_at) > self.stale_threshold

    async def ensure_connected(self) -> bool:
        """Reconnect if stale. Returns True if reconnect was triggered."""
        if self.is_stale():
            logger.warning("feed_stale_reconnecting", feed=self.name)
            await self.feed.reconnect()
            return True
        return False
