# src/feeds/pandascore.py
"""PandaScore esports data feed for real-time game events.

PandaScore provides real-time esports data with ~300ms latency from actual
game events, giving 30-40 seconds advantage over Twitch/YouTube viewers.
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional

import httpx

from .base import BaseFeed, FeedEvent


@dataclass
class PandaScoreEvent(FeedEvent):
    """Event from PandaScore esports feed."""

    @classmethod
    def from_raw(cls, game: str, match_id: str, raw_data: dict[str, Any]) -> "PandaScoreEvent":
        """Parse raw PandaScore event data into a standardized event.

        Args:
            game: Game type (lol, dota2, csgo, valorant)
            match_id: Unique match identifier
            raw_data: Raw event data from PandaScore API

        Returns:
            PandaScoreEvent with normalized data
        """
        event_type = raw_data.get("type", "unknown")
        timestamp = raw_data.get("timestamp", datetime.now().timestamp())
        payload = raw_data.get("payload", {})

        # Normalize the data based on event type
        data = cls._normalize_payload(event_type, payload, game)

        return cls(
            source="pandascore",
            event_type=event_type,
            game=game,
            data=data,
            timestamp=float(timestamp),
            match_id=match_id,
        )

    @staticmethod
    def _normalize_payload(event_type: str, payload: dict[str, Any], game: str) -> dict[str, Any]:
        """Normalize payload data based on event type and game.

        Args:
            event_type: Type of event (kill, tower_destroyed, round_end, etc.)
            payload: Raw payload data
            game: Game type

        Returns:
            Normalized data dictionary
        """
        data: dict[str, Any] = {}

        if event_type == "kill":
            # LoL/Dota2 kill events
            killer = payload.get("killer", {})
            victim = payload.get("victim", {})
            assists = payload.get("assists", [])

            data["killer"] = killer.get("name", "Unknown")
            data["killer_team"] = killer.get("team", "Unknown")
            data["victim"] = victim.get("name", "Unknown")
            data["victim_team"] = victim.get("team", "Unknown")
            data["assists"] = [a.get("name", "Unknown") for a in assists]

        elif event_type == "tower_destroyed":
            # LoL tower events
            data["team"] = payload.get("team", "Unknown")
            data["tower"] = payload.get("tower", "Unknown")

        elif event_type == "round_end":
            # CS:GO/Valorant round events
            data["winner"] = payload.get("winner", "Unknown")
            data["score"] = payload.get("score", {})

        elif event_type == "dragon_kill":
            # LoL dragon events
            data["team"] = payload.get("team", "Unknown")
            data["dragon_type"] = payload.get("dragon_type", "Unknown")

        elif event_type == "baron_kill":
            # LoL baron events
            data["team"] = payload.get("team", "Unknown")

        elif event_type == "roshan_kill":
            # Dota2 Roshan events
            data["team"] = payload.get("team", "Unknown")

        elif event_type == "bomb_planted":
            # CS:GO/Valorant bomb events
            data["player"] = payload.get("player", {}).get("name", "Unknown")
            data["site"] = payload.get("site", "Unknown")

        else:
            # Generic fallback - include all payload data
            data = dict(payload)

        return data


class PandaScoreFeed(BaseFeed):
    """Real-time esports data feed from PandaScore API.

    Supports multiple esports titles with polling-based event updates.
    The API provides ~300ms latency from actual game events.
    """

    SUPPORTED_GAMES = ["lol", "dota2", "csgo", "valorant"]
    BASE_URL = "https://api.pandascore.co"
    DEFAULT_POLL_INTERVAL = 1.0  # seconds

    def __init__(
        self,
        api_key: str,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
    ):
        """Initialize PandaScore feed.

        Args:
            api_key: PandaScore API key
            poll_interval: Interval between polls in seconds (default: 1.0)
        """
        super().__init__()
        self._api_key = api_key
        self._poll_interval = poll_interval
        self._client: Optional[httpx.AsyncClient] = None
        self._poll_task: Optional[asyncio.Task] = None
        self._last_event_ids: dict[str, set[str]] = {}  # match_id -> seen event IDs

    async def connect(self) -> None:
        """Establish connection to PandaScore API."""
        if self._connected:
            return

        self._client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Accept": "application/json",
            },
            timeout=10.0,
        )
        self._connected = True

    async def disconnect(self) -> None:
        """Close connection to PandaScore API."""
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None

        if self._client:
            await self._client.aclose()
            self._client = None

        self._connected = False
        self._subscriptions.clear()
        self._last_event_ids.clear()

    async def subscribe(self, game: str, match_id: str) -> None:
        """Subscribe to events for a specific match.

        Args:
            game: Game type (must be in SUPPORTED_GAMES)
            match_id: Unique match identifier

        Raises:
            ValueError: If game is not supported
        """
        if game not in self.SUPPORTED_GAMES:
            raise ValueError(f"Unsupported game: {game}. Supported: {self.SUPPORTED_GAMES}")

        self._subscriptions.add((game, match_id))
        self._last_event_ids.setdefault(match_id, set())

        # Start polling if not already running
        if self._poll_task is None or self._poll_task.done():
            self._poll_task = asyncio.create_task(self._poll_loop())

    async def get_live_matches(self, game: str) -> list[dict[str, Any]]:
        """Get currently running matches for a game.

        Args:
            game: Game type (lol, dota2, csgo, valorant)

        Returns:
            List of match data dictionaries

        Raises:
            ValueError: If game is not supported
            RuntimeError: If not connected
        """
        if game not in self.SUPPORTED_GAMES:
            raise ValueError(f"Unsupported game: {game}. Supported: {self.SUPPORTED_GAMES}")

        if not self._client:
            # Create a temporary client for one-off requests
            async with httpx.AsyncClient(
                base_url=self.BASE_URL,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Accept": "application/json",
                },
                timeout=10.0,
            ) as client:
                response = await client.get(f"/{game}/matches/running")
                response.raise_for_status()
                return response.json()

        response = await self._client.get(f"/{game}/matches/running")
        response.raise_for_status()
        return response.json()

    async def get_match_events(
        self,
        game: str,
        match_id: str,
        since: Optional[float] = None,
    ) -> list[PandaScoreEvent]:
        """Get events for a specific match.

        Args:
            game: Game type
            match_id: Match identifier
            since: Only return events after this timestamp (optional)

        Returns:
            List of PandaScoreEvent objects
        """
        if not self._client:
            raise RuntimeError("Not connected. Call connect() first.")

        response = await self._client.get(
            f"/{game}/matches/{match_id}/events",
            params={"since": since} if since else {},
        )
        response.raise_for_status()

        events = []
        for raw_event in response.json():
            event = PandaScoreEvent.from_raw(game, match_id, raw_event)
            if since is None or event.timestamp > since:
                events.append(event)

        return events

    async def measure_latency(self) -> float:
        """Measure API response latency.

        Returns:
            Latency in milliseconds
        """
        start_time = time.perf_counter()

        if not self._client:
            async with httpx.AsyncClient(
                base_url=self.BASE_URL,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Accept": "application/json",
                },
                timeout=10.0,
            ) as client:
                await client.get("/lol/matches/running")
        else:
            await self._client.get("/lol/matches/running")

        end_time = time.perf_counter()
        return (end_time - start_time) * 1000  # Convert to milliseconds

    async def _poll_loop(self) -> None:
        """Continuously poll for new events on subscribed matches."""
        while self._subscriptions and self._connected:
            for game, match_id in list(self._subscriptions):
                try:
                    events = await self._fetch_new_events(game, match_id)
                    for event in events:
                        await self._emit(event)
                except Exception as e:
                    # Log error but continue polling other matches
                    # In production, use proper logging
                    pass

            await asyncio.sleep(self._poll_interval)

    async def _fetch_new_events(
        self,
        game: str,
        match_id: str,
    ) -> list[PandaScoreEvent]:
        """Fetch only new events for a match (not seen before).

        Args:
            game: Game type
            match_id: Match identifier

        Returns:
            List of new PandaScoreEvent objects
        """
        if not self._client:
            return []

        try:
            response = await self._client.get(f"/{game}/matches/{match_id}/events")
            response.raise_for_status()
        except httpx.HTTPError:
            return []

        new_events = []
        seen_ids = self._last_event_ids.get(match_id, set())

        for raw_event in response.json():
            event_id = raw_event.get("id") or f"{raw_event.get('type')}_{raw_event.get('timestamp')}"
            if event_id not in seen_ids:
                seen_ids.add(event_id)
                event = PandaScoreEvent.from_raw(game, match_id, raw_event)
                new_events.append(event)

        self._last_event_ids[match_id] = seen_ids
        return new_events
