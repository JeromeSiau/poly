# src/feeds/azuro.py
"""Azuro Protocol GraphQL feed for cross-market arbitrage.

Azuro is a decentralized betting protocol on Polygon and Gnosis chains.
This feed provides access to betting markets via their subgraph API.
"""

import asyncio
from dataclasses import dataclass
from typing import Any, Optional

import structlog
from gql import Client, gql
from gql.transport.aiohttp import AIOHTTPTransport

from config.settings import settings
from .base import BaseFeed, FeedEvent

logger = structlog.get_logger()


@dataclass
class AzuroEvent:
    """Represents an Azuro betting condition/market.

    Attributes:
        condition_id: Unique identifier for the betting condition
        game_id: Unique identifier for the game/match
        sport: Sport type (football, basketball, etc.)
        league: League name (NFL, NBA, etc.)
        home_team: Home team name
        away_team: Away team name
        starts_at: Unix timestamp when the event starts
        outcomes: Dict mapping outcome_id to implied probability
    """
    condition_id: str
    game_id: str
    sport: str
    league: str
    home_team: str
    away_team: str
    starts_at: float
    outcomes: dict[str, float]


# GraphQL queries for Azuro subgraph
ACTIVE_CONDITIONS_QUERY = gql("""
    query GetActiveConditions($first: Int!, $skip: Int!) {
        conditions(
            first: $first
            skip: $skip
            where: {
                status: Created
                game_: { startsAt_gt: 0 }
            }
            orderBy: game__startsAt
            orderDirection: asc
        ) {
            conditionId
            gameId
            outcomes {
                outcomeId
                odds
            }
            game {
                sport {
                    name
                }
                league {
                    name
                }
                participants {
                    name
                }
                startsAt
            }
        }
    }
""")

CONDITION_ODDS_QUERY = gql("""
    query GetConditionOdds($conditionId: String!) {
        condition(id: $conditionId) {
            outcomes {
                outcomeId
                odds
            }
        }
    }
""")

CONDITIONS_BY_SPORT_QUERY = gql("""
    query GetConditionsBySport($sport: String!, $first: Int!) {
        conditions(
            first: $first
            where: {
                status: Created
                game_: {
                    sport_: { name: $sport }
                    startsAt_gt: 0
                }
            }
            orderBy: game__startsAt
            orderDirection: asc
        ) {
            conditionId
            gameId
            outcomes {
                outcomeId
                odds
            }
            game {
                sport {
                    name
                }
                league {
                    name
                }
                participants {
                    name
                }
                startsAt
            }
        }
    }
""")


class AzuroFeed(BaseFeed):
    """GraphQL feed for Azuro Protocol betting markets.

    Fetches betting conditions and odds from the Azuro subgraph on The Graph.
    Supports cross-market arbitrage detection with Polymarket.
    """

    DEFAULT_PAGE_SIZE = 100

    def __init__(self, subgraph_url: Optional[str] = None):
        """Initialize AzuroFeed.

        Args:
            subgraph_url: Optional custom subgraph URL. Defaults to settings.AZURO_SUBGRAPH_URL.
        """
        super().__init__()
        self._subgraph_url = subgraph_url or settings.AZURO_SUBGRAPH_URL
        self._client: Optional[Client] = None
        self._transport: Optional[AIOHTTPTransport] = None
        self._poll_task: Optional[asyncio.Task] = None
        self._poll_interval: float = 5.0  # seconds

    def _init_client(self) -> None:
        """Initialize the GraphQL client."""
        self._transport = AIOHTTPTransport(url=self._subgraph_url)
        self._client = Client(
            transport=self._transport,
            fetch_schema_from_transport=False,
        )

    async def connect(self) -> None:
        """Establish connection to Azuro subgraph."""
        if self._connected:
            return

        self._init_client()
        self._connected = True
        logger.info("azuro_feed_connected", url=self._subgraph_url)

    async def disconnect(self) -> None:
        """Close connection to Azuro subgraph."""
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None

        if self._transport:
            await self._transport.close()
            self._transport = None

        self._client = None
        self._connected = False
        self._subscriptions.clear()
        logger.info("azuro_feed_disconnected")

    async def subscribe(self, game: str, match_id: str) -> None:
        """Subscribe to updates for a specific condition.

        Args:
            game: Sport type (e.g., "football", "basketball")
            match_id: Condition ID to monitor
        """
        self._subscriptions.add((game, match_id))

        # Start polling if not already running
        if self._poll_task is None or self._poll_task.done():
            self._poll_task = asyncio.create_task(self._poll_loop())

    async def _poll_loop(self) -> None:
        """Continuously poll for odds updates on subscribed conditions."""
        while self._subscriptions and self._connected:
            for game, condition_id in list(self._subscriptions):
                try:
                    odds = await self.get_odds(condition_id)
                    event = FeedEvent(
                        source="azuro",
                        event_type="odds_update",
                        game=game,
                        data={"condition_id": condition_id, "odds": odds},
                        timestamp=asyncio.get_event_loop().time(),
                        match_id=condition_id,
                    )
                    await self._emit(event)
                except Exception as e:
                    logger.error("azuro_poll_error", condition_id=condition_id, error=str(e))

            await asyncio.sleep(self._poll_interval)

    async def _execute_query(self, query, variables: Optional[dict] = None) -> dict[str, Any]:
        """Execute a GraphQL query against the Azuro subgraph.

        Args:
            query: The gql query object
            variables: Optional query variables

        Returns:
            Query result as a dictionary

        Raises:
            RuntimeError: If not connected
        """
        if not self._connected or not self._client:
            raise RuntimeError("Not connected. Call connect() first.")

        async with self._client as session:
            result = await session.execute(query, variable_values=variables)
            return result

    async def get_active_events(
        self,
        sport: Optional[str] = None,
        page_size: int = DEFAULT_PAGE_SIZE,
        skip: int = 0,
    ) -> list[AzuroEvent]:
        """Fetch active betting conditions from Azuro.

        Args:
            sport: Optional filter by sport name
            page_size: Number of conditions to fetch
            skip: Number of conditions to skip (pagination)

        Returns:
            List of AzuroEvent objects
        """
        if sport:
            result = await self._execute_query(
                CONDITIONS_BY_SPORT_QUERY,
                {"sport": sport, "first": page_size},
            )
        else:
            result = await self._execute_query(
                ACTIVE_CONDITIONS_QUERY,
                {"first": page_size, "skip": skip},
            )

        events = []
        for condition in result.get("conditions", []):
            event = self._parse_condition(condition)
            if event:
                events.append(event)

        return events

    async def get_odds(self, condition_id: str) -> dict[str, float]:
        """Get current odds for a specific condition.

        Args:
            condition_id: The condition ID to fetch odds for

        Returns:
            Dict mapping outcome_id to implied probability (1/decimal_odds)
        """
        result = await self._execute_query(
            CONDITION_ODDS_QUERY,
            {"conditionId": condition_id},
        )

        condition = result.get("condition", {})
        outcomes = condition.get("outcomes", [])

        odds_map = {}
        for outcome in outcomes:
            outcome_id = outcome.get("outcomeId")
            decimal_odds = float(outcome.get("odds", "1.0"))
            # Convert decimal odds to implied probability
            implied_prob = 1.0 / decimal_odds if decimal_odds > 0 else 0.0
            odds_map[outcome_id] = implied_prob

        return odds_map

    async def get_events_by_sport(self, sport: str, limit: int = 100) -> list[AzuroEvent]:
        """Get active events filtered by sport.

        Args:
            sport: Sport name (e.g., "Football", "Basketball")
            limit: Maximum number of events to return

        Returns:
            List of AzuroEvent objects for the specified sport
        """
        return await self.get_active_events(sport=sport, page_size=limit)

    def _parse_condition(self, condition: dict[str, Any]) -> Optional[AzuroEvent]:
        """Parse a raw condition from the subgraph into an AzuroEvent.

        Args:
            condition: Raw condition data from GraphQL response

        Returns:
            AzuroEvent object or None if parsing fails
        """
        try:
            game = condition.get("game", {})
            sport = game.get("sport", {}).get("name", "Unknown")
            league = game.get("league", {}).get("name", "Unknown")
            participants = game.get("participants", [])

            # Extract home and away teams
            home_team = participants[0].get("name", "Unknown") if len(participants) > 0 else "Unknown"
            away_team = participants[1].get("name", "Unknown") if len(participants) > 1 else "Unknown"

            # Parse outcomes into probability map
            outcomes_map = {}
            for outcome in condition.get("outcomes", []):
                outcome_id = outcome.get("outcomeId")
                decimal_odds = float(outcome.get("odds", "1.0"))
                # Convert decimal odds to implied probability
                implied_prob = 1.0 / decimal_odds if decimal_odds > 0 else 0.0
                outcomes_map[outcome_id] = implied_prob

            return AzuroEvent(
                condition_id=condition.get("conditionId", ""),
                game_id=condition.get("gameId", ""),
                sport=sport,
                league=league,
                home_team=home_team,
                away_team=away_team,
                starts_at=float(game.get("startsAt", 0)),
                outcomes=outcomes_map,
            )
        except Exception as e:
            logger.error("azuro_parse_error", error=str(e), condition=condition)
            return None
