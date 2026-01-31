# src/feeds/overtime.py
"""Overtime Markets (Thales) GraphQL feed for cross-market arbitrage.

Overtime is a decentralized sports prediction market built on Thales Protocol,
operating on Optimism. This feed provides access to sports betting markets
via their subgraph API.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional

import structlog
from gql import Client, gql
from gql.transport.aiohttp import AIOHTTPTransport

from config.settings import settings
from .base import BaseFeed, FeedEvent

logger = structlog.get_logger()


# Sport tag mappings for Overtime Markets
SPORT_TAGS = {
    "9001": "MLB",
    "9002": "NBA",
    "9003": "NHL",
    "9004": "NFL",
    "9005": "MLS",
    "9006": "EPL",
    "9007": "La Liga",
    "9008": "Serie A",
    "9009": "Bundesliga",
    "9010": "Ligue 1",
    "9011": "Champions League",
    "9012": "Europa League",
    "9013": "NCAA Football",
    "9014": "NCAA Basketball",
    "9015": "Tennis",
    "9016": "UFC",
    "9017": "Boxing",
    "9018": "Formula 1",
    "9019": "NASCAR",
    "9020": "Golf",
}


@dataclass
class OvertimeGame:
    """Represents an Overtime sports market.

    Attributes:
        game_id: Unique identifier for the game
        market_address: Contract address for the market (optional)
        sport: Sport type (NFL, NBA, etc.)
        home_team: Home team name
        away_team: Away team name
        starts_at: Unix timestamp when the game starts
        home_odds: Implied probability for home team win (0.0-1.0)
        away_odds: Implied probability for away team win (0.0-1.0)
        draw_odds: Implied probability for draw (0.0-1.0), optional
        is_resolved: Whether the market has been resolved
    """
    game_id: str
    sport: str
    home_team: str
    away_team: str
    starts_at: float
    home_odds: float
    away_odds: float
    is_resolved: bool
    market_address: Optional[str] = None
    draw_odds: Optional[float] = None


# GraphQL queries for Overtime subgraph
ACTIVE_MARKETS_QUERY = gql("""
    query GetActiveMarkets($first: Int!, $skip: Int!) {
        sportMarkets(
            first: $first
            skip: $skip
            where: {
                isResolved: false
                isCanceled: false
            }
            orderBy: maturityDate
            orderDirection: asc
        ) {
            id
            gameId
            tags
            homeTeam
            awayTeam
            maturityDate
            homeOdds
            awayOdds
            drawOdds
            isResolved
        }
    }
""")

MARKET_ODDS_QUERY = gql("""
    query GetMarketOdds($marketId: String!) {
        sportMarket(id: $marketId) {
            homeOdds
            awayOdds
            drawOdds
        }
    }
""")

MARKETS_BY_SPORT_QUERY = gql("""
    query GetMarketsBySport($tag: String!, $first: Int!) {
        sportMarkets(
            first: $first
            where: {
                isResolved: false
                isCanceled: false
                tags_contains: [$tag]
            }
            orderBy: maturityDate
            orderDirection: asc
        ) {
            id
            gameId
            tags
            homeTeam
            awayTeam
            maturityDate
            homeOdds
            awayOdds
            drawOdds
            isResolved
        }
    }
""")


class OvertimeFeed(BaseFeed):
    """GraphQL feed for Overtime Markets (Thales) sports prediction markets.

    Fetches sports betting markets and odds from the Overtime subgraph on The Graph.
    Supports cross-market arbitrage detection with Polymarket and other platforms.
    """

    DEFAULT_PAGE_SIZE = 100
    WEI_DIVISOR = 1e18  # Odds are stored in wei-like format

    def __init__(self, subgraph_url: Optional[str] = None):
        """Initialize OvertimeFeed.

        Args:
            subgraph_url: Optional custom subgraph URL. Defaults to settings.OVERTIME_SUBGRAPH_URL.
        """
        super().__init__()
        self._subgraph_url = subgraph_url or settings.OVERTIME_SUBGRAPH_URL
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
        """Establish connection to Overtime subgraph."""
        if self._connected:
            return

        self._init_client()
        self._connected = True
        logger.info("overtime_feed_connected", url=self._subgraph_url)

    async def disconnect(self) -> None:
        """Close connection to Overtime subgraph."""
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
        logger.info("overtime_feed_disconnected")

    async def subscribe(self, game: str, match_id: str) -> None:
        """Subscribe to updates for a specific market.

        Args:
            game: Sport type (e.g., "NFL", "NBA")
            match_id: Market address to monitor
        """
        self._subscriptions.add((game, match_id))

        # Start polling if not already running
        if self._poll_task is None or self._poll_task.done():
            self._poll_task = asyncio.create_task(self._poll_loop())

    async def _poll_loop(self) -> None:
        """Continuously poll for odds updates on subscribed markets."""
        while self._subscriptions and self._connected:
            for game, market_id in list(self._subscriptions):
                try:
                    odds = await self.get_odds(market_id)
                    event = FeedEvent(
                        source="overtime",
                        event_type="odds_update",
                        game=game,
                        data={"market_id": market_id, "odds": odds},
                        timestamp=asyncio.get_event_loop().time(),
                        match_id=market_id,
                    )
                    await self._emit(event)
                except Exception as e:
                    logger.error("overtime_poll_error", market_id=market_id, error=str(e))

            await asyncio.sleep(self._poll_interval)

    async def _execute_query(self, query, variables: Optional[dict] = None) -> dict[str, Any]:
        """Execute a GraphQL query against the Overtime subgraph.

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

    def _parse_odds_from_wei(self, odds_str: str) -> float:
        """Parse odds from wei-like format to probability (0.0-1.0).

        Args:
            odds_str: Odds in wei format (e.g., "550000000000000000" = 0.55)

        Returns:
            Probability as float between 0.0 and 1.0
        """
        try:
            odds_wei = int(odds_str)
            return odds_wei / self.WEI_DIVISOR
        except (ValueError, TypeError):
            return 0.0

    def _get_sport_from_tags(self, tags: list[str]) -> str:
        """Get sport name from market tags.

        Args:
            tags: List of sport tags from the market

        Returns:
            Sport name or "Unknown" if not found
        """
        for tag in tags:
            if tag in SPORT_TAGS:
                return SPORT_TAGS[tag]
        return "Unknown"

    async def get_active_games(
        self,
        sport: Optional[str] = None,
        page_size: int = DEFAULT_PAGE_SIZE,
        skip: int = 0,
    ) -> list[OvertimeGame]:
        """Fetch active sports markets from Overtime.

        Args:
            sport: Optional filter by sport name (uses sport tag lookup)
            page_size: Number of markets to fetch
            skip: Number of markets to skip (pagination)

        Returns:
            List of OvertimeGame objects
        """
        if sport:
            # Find the tag for the given sport
            sport_tag = None
            for tag, name in SPORT_TAGS.items():
                if name.upper() == sport.upper():
                    sport_tag = tag
                    break

            if sport_tag:
                result = await self._execute_query(
                    MARKETS_BY_SPORT_QUERY,
                    {"tag": sport_tag, "first": page_size},
                )
            else:
                result = await self._execute_query(
                    ACTIVE_MARKETS_QUERY,
                    {"first": page_size, "skip": skip},
                )
        else:
            result = await self._execute_query(
                ACTIVE_MARKETS_QUERY,
                {"first": page_size, "skip": skip},
            )

        games = []
        for market in result.get("sportMarkets", []):
            game = self._parse_market(market)
            if game:
                games.append(game)

        return games

    async def get_odds(self, market_id: str) -> dict[str, float]:
        """Get current odds for a specific market.

        Args:
            market_id: The market address to fetch odds for

        Returns:
            Dict with 'home', 'away', and optionally 'draw' probabilities
        """
        result = await self._execute_query(
            MARKET_ODDS_QUERY,
            {"marketId": market_id},
        )

        market = result.get("sportMarket", {})

        odds = {
            "home": self._parse_odds_from_wei(market.get("homeOdds", "0")),
            "away": self._parse_odds_from_wei(market.get("awayOdds", "0")),
        }

        draw_odds = market.get("drawOdds", "0")
        if draw_odds and draw_odds != "0":
            odds["draw"] = self._parse_odds_from_wei(draw_odds)

        return odds

    async def get_games_by_sport(self, sport: str, limit: int = 100) -> list[OvertimeGame]:
        """Get active games filtered by sport.

        Args:
            sport: Sport name (e.g., "NFL", "NBA")
            limit: Maximum number of games to return

        Returns:
            List of OvertimeGame objects for the specified sport
        """
        return await self.get_active_games(sport=sport, page_size=limit)

    def _parse_market(self, market: dict[str, Any]) -> Optional[OvertimeGame]:
        """Parse a raw market from the subgraph into an OvertimeGame.

        Args:
            market: Raw market data from GraphQL response

        Returns:
            OvertimeGame object or None if parsing fails
        """
        try:
            tags = market.get("tags", [])
            sport = self._get_sport_from_tags(tags)

            home_odds = self._parse_odds_from_wei(market.get("homeOdds", "0"))
            away_odds = self._parse_odds_from_wei(market.get("awayOdds", "0"))

            draw_odds_str = market.get("drawOdds", "0")
            draw_odds = None
            if draw_odds_str and draw_odds_str != "0":
                draw_odds = self._parse_odds_from_wei(draw_odds_str)

            return OvertimeGame(
                game_id=market.get("gameId", ""),
                market_address=market.get("id", ""),
                sport=sport,
                home_team=market.get("homeTeam", "Unknown"),
                away_team=market.get("awayTeam", "Unknown"),
                starts_at=float(market.get("maturityDate", 0)),
                home_odds=home_odds,
                away_odds=away_odds,
                draw_odds=draw_odds,
                is_resolved=market.get("isResolved", False),
            )
        except Exception as e:
            logger.error("overtime_parse_error", error=str(e), market=market)
            return None
