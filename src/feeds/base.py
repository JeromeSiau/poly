from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from datetime import datetime
import asyncio


@dataclass
class FeedEvent:
    """Standardized event from any data feed."""
    source: str  # pandascore, sportsdataio, polymarket
    event_type: str  # kill, tower, goal, price_update, etc.
    game: str  # lol, dota2, csgo, nba, nfl
    data: dict[str, Any]
    timestamp: float  # Unix timestamp
    match_id: Optional[str] = None

    @property
    def age_seconds(self) -> float:
        """How old is this event?"""
        return datetime.utcnow().timestamp() - self.timestamp


class BaseFeed(ABC):
    """Abstract base class for all data feeds."""

    def __init__(self):
        self._connected: bool = False
        self._callbacks: list[Callable[[FeedEvent], None]] = []
        self._subscriptions: set[tuple[str, str]] = set()  # (game, match_id)

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the data source."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the data source."""
        pass

    @abstractmethod
    async def subscribe(self, game: str, match_id: str) -> None:
        """Subscribe to events for a specific match."""
        pass

    def on_event(self, callback: Callable[[FeedEvent], None]) -> None:
        """Register a callback for incoming events."""
        self._callbacks.append(callback)

    async def _emit(self, event: FeedEvent) -> None:
        """Emit an event to all registered callbacks."""
        for callback in self._callbacks:
            if asyncio.iscoroutinefunction(callback):
                await callback(event)
            else:
                callback(event)

    @property
    def is_connected(self) -> bool:
        return self._connected
