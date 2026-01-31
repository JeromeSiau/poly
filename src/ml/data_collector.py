"""Historical match data collection from PandaScore API.

Collects past matches with all events and game states for ML training.
"""

from dataclasses import dataclass
from typing import Optional

import httpx
import structlog

logger = structlog.get_logger()


@dataclass
class EventData:
    """Single event with game state snapshot for ML training.

    Note on field limitations:
        - gold_diff: Always 0 because PandaScore events API does not include
          gold data per event. Gold tracking would require the frames endpoint
          which provides periodic snapshots. This field is reserved for future
          implementation.
        - winner: Not populated by fetch_match_events(). The winner is only
          known from MatchData, so callers must set event.winner externally
          after fetching (e.g., event.winner = match.winner).
    """

    event_type: str
    timestamp: float
    team: str
    game_time_minutes: float

    # Game state at event time
    # Note: gold_diff is always 0 - see class docstring for explanation
    gold_diff: int
    kill_diff: int
    tower_diff: int
    dragon_diff: int
    baron_diff: int

    # Ground truth label (must be set externally - see class docstring)
    winner: str


@dataclass
class MatchData:
    """Match metadata."""

    match_id: int
    game: str
    team_a: str
    team_b: str
    winner: str
    game_length_minutes: float


class HistoricalDataCollector:
    """Collects historical match data from PandaScore for ML training."""

    BASE_URL = "https://api.pandascore.co"

    def __init__(self, api_key: str):
        """Initialize collector with API key."""
        self._api_key = api_key
        self._client: Optional[httpx.AsyncClient] = None

    async def connect(self) -> None:
        """Create HTTP client."""
        logger.info("connecting_to_pandascore", base_url=self.BASE_URL)
        self._client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Accept": "application/json",
            },
            timeout=30.0,
        )
        logger.info("connected_to_pandascore")

    async def disconnect(self) -> None:
        """Close HTTP client."""
        logger.info("disconnecting_from_pandascore")
        if self._client:
            await self._client.aclose()
            self._client = None
        logger.info("disconnected_from_pandascore")

    async def fetch_past_matches(
        self,
        game: str,
        limit: int = 100,
        page: int = 1,
    ) -> list[MatchData]:
        """Fetch past matches for a game."""
        if not self._client:
            raise RuntimeError("Not connected. Call connect() first.")

        logger.info(
            "fetching_past_matches", game=game, limit=limit, page=page
        )
        response = await self._client.get(
            f"/{game}/matches/past",
            params={
                "per_page": limit,
                "page": page,
                "sort": "-begin_at",
            },
        )
        response.raise_for_status()

        matches = []
        for raw in response.json():
            if not raw.get("winner"):
                continue

            opponents = raw.get("opponents", [])
            if len(opponents) < 2:
                continue

            team_a = opponents[0].get("opponent", {}).get("name", "Unknown")
            team_b = opponents[1].get("opponent", {}).get("name", "Unknown")
            winner = raw["winner"].get("name", "Unknown")

            games = raw.get("games", [])
            length_minutes = 0.0
            if games and games[0].get("length"):
                length_minutes = games[0]["length"] / 60.0

            matches.append(
                MatchData(
                    match_id=raw["id"],
                    game=game,
                    team_a=team_a,
                    team_b=team_b,
                    winner=winner,
                    game_length_minutes=length_minutes,
                )
            )

        logger.info("fetched_past_matches", game=game, match_count=len(matches))
        return matches

    async def fetch_match_events(
        self,
        game: str,
        game_id: int,
    ) -> list[EventData]:
        """Fetch events for a specific game within a match.

        Note: The returned EventData objects will have:
            - gold_diff: Always 0 (API limitation - see EventData docstring)
            - winner: Empty string (must be set externally from MatchData)

        Args:
            game: Game slug (e.g., "lol" for League of Legends)
            game_id: PandaScore game ID (from MatchData.games)

        Returns:
            List of EventData with game state snapshots at each event.
        """
        if not self._client:
            raise RuntimeError("Not connected. Call connect() first.")

        logger.info("fetching_match_events", game=game, game_id=game_id)
        response = await self._client.get(
            f"/{game}/games/{game_id}/events",
        )
        response.raise_for_status()

        raw_events = response.json()
        logger.debug("received_raw_events", game_id=game_id, count=len(raw_events))

        events = []
        state = {
            "team_a_gold": 0,
            "team_b_gold": 0,
            "team_a_kills": 0,
            "team_b_kills": 0,
            "team_a_towers": 0,
            "team_b_towers": 0,
            "team_a_dragons": 0,
            "team_b_dragons": 0,
            "team_a_barons": 0,
            "team_b_barons": 0,
            "team_a": None,
            "team_b": None,
            "winner": None,
        }

        for raw in raw_events:
            event_type = raw.get("type", "unknown")
            timestamp = raw.get("timestamp", 0)
            payload = raw.get("payload", {})

            team = payload.get("team") or payload.get("killer", {}).get("team")
            if not team:
                continue

            self._update_state(state, event_type, team)

            events.append(
                EventData(
                    event_type=event_type,
                    timestamp=timestamp,
                    team=team,
                    game_time_minutes=timestamp / 60.0,
                    gold_diff=state["team_a_gold"] - state["team_b_gold"],
                    kill_diff=state["team_a_kills"] - state["team_b_kills"],
                    tower_diff=state["team_a_towers"] - state["team_b_towers"],
                    dragon_diff=state["team_a_dragons"] - state["team_b_dragons"],
                    baron_diff=state["team_a_barons"] - state["team_b_barons"],
                    winner=state["winner"] or "",
                )
            )

        logger.info("fetched_match_events", game_id=game_id, event_count=len(events))
        return events

    def _update_state(
        self,
        state: dict,
        event_type: str,
        team: str,
    ) -> None:
        """Update tracked state based on event."""
        if state["team_a"] is None:
            state["team_a"] = team
        elif state["team_b"] is None and team != state["team_a"]:
            state["team_b"] = team

        is_team_a = team == state["team_a"]

        if event_type == "kill":
            if is_team_a:
                state["team_a_kills"] += 1
            else:
                state["team_b_kills"] += 1

        elif event_type in ("tower_destroyed", "tower"):
            if is_team_a:
                state["team_a_towers"] += 1
            else:
                state["team_b_towers"] += 1

        elif event_type in ("dragon_kill", "dragon"):
            if is_team_a:
                state["team_a_dragons"] += 1
            else:
                state["team_b_dragons"] += 1

        elif event_type in ("baron_kill", "baron"):
            if is_team_a:
                state["team_a_barons"] += 1
            else:
                state["team_b_barons"] += 1
