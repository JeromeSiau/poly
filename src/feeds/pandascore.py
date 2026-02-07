# src/feeds/pandascore.py
"""PandaScore esports data feed for real-time game events.

PandaScore provides real-time esports data with ~300ms latency from actual
game events, giving 30-40 seconds advantage over Twitch/YouTube viewers.

Uses WebSockets for lowest-latency event delivery:
- Events endpoint: wss://live.pandascore.co/matches/{match_id}/events
- Frames endpoint: wss://live.pandascore.co/matches/{match_id} (every 2s)
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional

import httpx
import structlog
import websockets
from websockets.exceptions import ConnectionClosed

logger = structlog.get_logger()

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

        # Best-effort normalization of game time
        if "game_time_minutes" not in data:
            raw_time = (
                payload.get("game_time_minutes")
                or payload.get("game_time_seconds")
                or payload.get("game_time")
            )
            if isinstance(raw_time, (int, float)):
                # Heuristic: if value looks like seconds, convert to minutes
                minutes = raw_time / 60 if raw_time > 100 else raw_time
                data["game_time_minutes"] = float(minutes)

        return data


class PandaScoreFeed(BaseFeed):
    """Real-time esports data feed from PandaScore API.

    Supports multiple esports titles with WebSocket-based event updates.
    The API provides ~300ms latency from actual game events.

    WebSocket endpoints:
    - Events: wss://live.pandascore.co/matches/{match_id}/events (real-time)
    - Frames: wss://live.pandascore.co/matches/{match_id} (every 2s snapshot)
    """

    SUPPORTED_GAMES = ["lol", "dota2", "csgo", "valorant"]
    BASE_URL = "https://api.pandascore.co"
    WS_BASE_URL = "wss://live.pandascore.co"

    def __init__(self, api_key: str):
        """Initialize PandaScore feed.

        Args:
            api_key: PandaScore API key
        """
        super().__init__()
        self._api_key = api_key
        self._client: Optional[httpx.AsyncClient] = None
        self._ws_tasks: dict[str, asyncio.Task] = {}  # match_id -> WebSocket task
        self._ws_connections: dict[str, websockets.WebSocketClientProtocol] = {}

    async def connect(self) -> None:
        """Establish HTTP client for REST API calls (discovery, etc.)."""
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
        logger.info("pandascore_connected")

    async def disconnect(self) -> None:
        """Close all connections (HTTP client + WebSockets)."""
        # Cancel all WebSocket tasks
        for match_id, task in list(self._ws_tasks.items()):
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            logger.debug("ws_task_cancelled", match_id=match_id)

        self._ws_tasks.clear()

        # Close all WebSocket connections
        for match_id, ws in list(self._ws_connections.items()):
            try:
                await ws.close()
            except Exception:
                pass
        self._ws_connections.clear()

        # Close HTTP client
        if self._client:
            await self._client.aclose()
            self._client = None

        self._connected = False
        self._subscriptions.clear()
        logger.info("pandascore_disconnected")

    async def subscribe(self, game: str, match_id: str) -> None:
        """Subscribe to real-time events for a specific match via WebSocket.

        Opens a WebSocket connection to receive events as they happen.

        Args:
            game: Game type (must be in SUPPORTED_GAMES)
            match_id: Unique match identifier

        Raises:
            ValueError: If game is not supported
        """
        if game not in self.SUPPORTED_GAMES:
            raise ValueError(f"Unsupported game: {game}. Supported: {self.SUPPORTED_GAMES}")

        self._subscriptions.add((game, match_id))

        # Start WebSocket listener if not already running for this match
        if match_id not in self._ws_tasks or self._ws_tasks[match_id].done():
            self._ws_tasks[match_id] = asyncio.create_task(
                self._ws_listen(game, match_id)
            )
            logger.info("ws_subscription_started", game=game, match_id=match_id)

    async def unsubscribe(self, match_id: str) -> None:
        """Unsubscribe from a match and close its WebSocket.

        Args:
            match_id: Match identifier to unsubscribe from
        """
        # Remove from subscriptions
        self._subscriptions = {
            (g, m) for g, m in self._subscriptions if m != match_id
        }

        # Cancel WebSocket task
        if match_id in self._ws_tasks:
            self._ws_tasks[match_id].cancel()
            try:
                await self._ws_tasks[match_id]
            except asyncio.CancelledError:
                pass
            del self._ws_tasks[match_id]

        # Close WebSocket connection
        if match_id in self._ws_connections:
            try:
                await self._ws_connections[match_id].close()
            except Exception:
                pass
            del self._ws_connections[match_id]

        logger.info("ws_unsubscribed", match_id=match_id)

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

    async def _ws_listen(self, game: str, match_id: str) -> None:
        """Listen to WebSocket events for a match.

        Connects to the events endpoint for real-time event delivery.
        Automatically reconnects on connection loss.

        Args:
            game: Game type
            match_id: Match identifier
        """
        url = f"{self.WS_BASE_URL}/matches/{match_id}/events?token={self._api_key}"
        reconnect_delay = 1.0
        max_reconnect_delay = 30.0

        while (game, match_id) in self._subscriptions:
            try:
                logger.debug("ws_connecting", match_id=match_id)

                async with websockets.connect(
                    url,
                    ping_interval=20,
                    ping_timeout=10,
                ) as ws:
                    self._ws_connections[match_id] = ws
                    reconnect_delay = 1.0  # Reset on successful connection

                    # First message should be "hello"
                    hello = await ws.recv()
                    hello_data = json.loads(hello)
                    if hello_data.get("type") == "hello":
                        logger.info(
                            "ws_connected",
                            match_id=match_id,
                            game=game,
                        )

                    # Listen for events
                    async for message in ws:
                        try:
                            raw_event = json.loads(message)
                            event = PandaScoreEvent.from_raw(game, match_id, raw_event)
                            await self._emit(event)

                            logger.debug(
                                "ws_event_received",
                                match_id=match_id,
                                event_type=event.event_type,
                            )
                        except json.JSONDecodeError as e:
                            logger.warning(
                                "ws_invalid_json",
                                match_id=match_id,
                                error=str(e),
                            )

            except ConnectionClosed as e:
                logger.warning(
                    "ws_connection_closed",
                    match_id=match_id,
                    code=e.code,
                    reason=e.reason,
                )
            except asyncio.CancelledError:
                logger.debug("ws_cancelled", match_id=match_id)
                raise
            except Exception as e:
                logger.error(
                    "ws_error",
                    match_id=match_id,
                    error=str(e),
                )

            # Clean up connection reference
            self._ws_connections.pop(match_id, None)

            # Reconnect with exponential backoff
            if (game, match_id) in self._subscriptions:
                logger.info(
                    "ws_reconnecting",
                    match_id=match_id,
                    delay=reconnect_delay,
                )
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)

    async def subscribe_frames(self, game: str, match_id: str) -> None:
        """Subscribe to frame snapshots (every 2 seconds) via WebSocket.

        Frames contain full game state snapshots, useful for tracking
        scores, gold, objectives, etc.

        Args:
            game: Game type
            match_id: Match identifier
        """
        if game not in self.SUPPORTED_GAMES:
            raise ValueError(f"Unsupported game: {game}")

        # Ensure subscription tracking so frame loop stays active
        self._subscriptions.add((game, match_id))

        task_key = f"{match_id}_frames"
        if task_key not in self._ws_tasks or self._ws_tasks[task_key].done():
            self._ws_tasks[task_key] = asyncio.create_task(
                self._ws_listen_frames(game, match_id)
            )
            logger.info("ws_frames_subscription_started", game=game, match_id=match_id)

    async def _ws_listen_frames(self, game: str, match_id: str) -> None:
        """Listen to frame snapshots via WebSocket.

        Args:
            game: Game type
            match_id: Match identifier
        """
        url = f"{self.WS_BASE_URL}/matches/{match_id}?token={self._api_key}"
        reconnect_delay = 1.0

        while (game, match_id) in self._subscriptions:
            try:
                async with websockets.connect(url, ping_interval=20) as ws:
                    hello = await ws.recv()
                    logger.info("ws_frames_connected", match_id=match_id)

                    async for message in ws:
                        try:
                            frame = json.loads(message)
                            # Emit frame as a special event type
                            event = PandaScoreEvent(
                                source="pandascore",
                                event_type="frame",
                                game=game,
                                data=frame.get("payload", frame),
                                timestamp=time.time(),
                                match_id=match_id,
                            )
                            await self._emit(event)
                        except json.JSONDecodeError:
                            pass

            except ConnectionClosed:
                pass
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error("ws_frames_error", error=str(e))

            if (game, match_id) in self._subscriptions:
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 30.0)
