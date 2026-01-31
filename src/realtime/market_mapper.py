"""Market Mapper - Links game events to Polymarket markets.

Handles:
1. Manual mapping registration
2. Automatic discovery from Polymarket API
3. Team name normalization with aliases
4. Fuzzy matching between PandaScore and Polymarket team names
"""

import json
import re
from dataclasses import dataclass, field
from typing import Optional

import httpx
import structlog

from src.utils.team_matching import normalize_team_name, match_team_name, TEAM_ALIASES

logger = structlog.get_logger()


@dataclass
class MarketMapping:
    """Represents a mapping between a game event and a Polymarket market."""

    polymarket_id: str
    game: str
    event_identifier: str
    team_to_outcome: dict[str, str] = field(default_factory=dict)

    def get_outcome_for_team(self, team: str) -> Optional[str]:
        """Get the Polymarket outcome token for a given team.

        Args:
            team: Team name to look up (will be normalized).

        Returns:
            The outcome string (e.g., "YES", "No") or None if not found.
        """
        normalized = self._normalize_team_name(team)
        # Try exact match first
        if normalized in self.team_to_outcome:
            return self.team_to_outcome[normalized]
        # Try case-insensitive match
        for stored_team, outcome in self.team_to_outcome.items():
            if self._normalize_team_name(stored_team) == normalized:
                return outcome
        return None

    @staticmethod
    def _normalize_team_name(name: str) -> str:
        """Normalize team name for comparison using shared utility.

        Args:
            name: Raw team name.

        Returns:
            Normalized team name (lowercase, with alias resolution).
        """
        return normalize_team_name(name)

    @staticmethod
    def _detect_game(market_data: dict) -> str:
        """Detect the game type from market data.

        Args:
            market_data: Polymarket market data dictionary.

        Returns:
            Detected game identifier (e.g., "lol", "csgo", "dota2").
        """
        tags = market_data.get("tags", [])
        question = market_data.get("question", "").lower()

        # Check tags first
        game_keywords = {
            "lol": ["lol", "league of legends", "league"],
            "csgo": ["csgo", "cs2", "counter-strike", "counterstrike"],
            "dota2": ["dota", "dota2", "dota 2"],
            "valorant": ["valorant", "val"],
        }

        for tag in tags:
            tag_lower = tag.lower()
            for game, keywords in game_keywords.items():
                if tag_lower in keywords:
                    return game

        # Check question text
        for game, keywords in game_keywords.items():
            for keyword in keywords:
                if keyword in question:
                    return game

        return "unknown"

    @classmethod
    def from_polymarket(
        cls,
        market_data: dict,
        team_a: str,
        team_b: str
    ) -> "MarketMapping":
        """Create a MarketMapping from Polymarket market data.

        Args:
            market_data: Dictionary containing Polymarket market information.
            team_a: The first team (typically the favorite/subject of "Will X beat Y").
            team_b: The second team.

        Returns:
            A new MarketMapping instance.
        """
        polymarket_id = market_data.get("id", "")
        outcomes = market_data.get("outcomes", ["Yes", "No"])

        # team_a is typically YES (the subject of "Will X beat Y?")
        team_to_outcome = {
            team_a: outcomes[0] if outcomes else "Yes",
            team_b: outcomes[1] if len(outcomes) > 1 else "No",
        }

        game = cls._detect_game(market_data)
        question = market_data.get("question", "")
        event_identifier = f"{game}_{team_a}_vs_{team_b}"

        return cls(
            polymarket_id=polymarket_id,
            game=game,
            event_identifier=event_identifier,
            team_to_outcome=team_to_outcome,
        )


class MarketMapper:
    """Registry for mapping game events to Polymarket markets."""

    GAMMA_API = "https://gamma-api.polymarket.com"
    LOL_SERIES_ID = "10311"

    def __init__(self):
        """Initialize an empty MarketMapper."""
        self._mappings: list[MarketMapping] = []
        self._by_polymarket_id: dict[str, MarketMapping] = {}

    def add_mapping(
        self,
        game: str,
        event_identifier: str,
        polymarket_id: str,
        outcomes: dict[str, str]
    ) -> MarketMapping:
        """Add a new mapping to the registry.

        Args:
            game: Game identifier (e.g., "lol", "csgo").
            event_identifier: Unique identifier for the event.
            polymarket_id: The Polymarket market ID.
            outcomes: Dictionary mapping team names to outcome tokens.

        Returns:
            The created MarketMapping.
        """
        mapping = MarketMapping(
            polymarket_id=polymarket_id,
            game=game,
            event_identifier=event_identifier,
            team_to_outcome=outcomes,
        )
        self._mappings.append(mapping)
        self._by_polymarket_id[polymarket_id] = mapping
        return mapping

    def find_market(
        self,
        game: str,
        teams: list[str],
        league: Optional[str] = None,
        match_id: Optional[str] = None
    ) -> Optional[MarketMapping]:
        """Find a market mapping for a given event.

        Args:
            game: Game identifier (e.g., "lol", "csgo").
            teams: List of team names involved in the match.
            league: Optional league identifier for additional filtering.
            match_id: Optional specific match ID.

        Returns:
            The matching MarketMapping or None if not found.
        """
        normalized_teams = {self._normalize_team(t) for t in teams}

        for mapping in self._mappings:
            # Check game matches
            if mapping.game.lower() != game.lower():
                continue

            # Check if teams match
            mapping_teams = {
                self._normalize_team(t) for t in mapping.team_to_outcome.keys()
            }

            if normalized_teams == mapping_teams:
                # If league is specified, check it's in the event identifier
                if league and league.lower() not in mapping.event_identifier.lower():
                    continue
                # If match_id is specified, check it matches
                if match_id and match_id not in mapping.event_identifier:
                    continue
                return mapping

        return None

    def find_by_polymarket_id(self, polymarket_id: str) -> Optional[MarketMapping]:
        """Find a mapping by its Polymarket market ID.

        Args:
            polymarket_id: The Polymarket market ID to look up.

        Returns:
            The matching MarketMapping or None if not found.
        """
        return self._by_polymarket_id.get(polymarket_id)

    def _normalize_team(self, team: str) -> str:
        """Normalize a team name using shared team_matching utility.

        Args:
            team: Raw team name.

        Returns:
            Normalized/canonical team name (lowercase).
        """
        return normalize_team_name(team)

    def get_all_mappings(self) -> list[MarketMapping]:
        """Get all registered mappings.

        Returns:
            List of all MarketMapping instances.
        """
        return list(self._mappings)

    @staticmethod
    def parse_teams_from_title(title: str) -> tuple[str, str] | None:
        """Extract team names from a Polymarket market title.

        Handles formats like:
        - "LoL: T1 vs G2 Esports"
        - "LoL: Fnatic vs SK Gaming (BO3)"
        - "LoL: Team A vs Team B (BO3) - LEC Regular Season"

        Args:
            title: Market title string.

        Returns:
            Tuple of (team_a, team_b) or None if parsing fails.
        """
        # Remove game prefix
        clean = re.sub(r"^(LoL|Dota2?|CS:?GO|CS2|Valorant):\s*", "", title, flags=re.I)

        # Split by " vs " first
        if " vs " not in clean.lower():
            return None

        parts = re.split(r"\s+vs\s+", clean, flags=re.I)
        if len(parts) != 2:
            return None

        team_a = parts[0].strip()
        team_b = parts[1].strip()

        # Remove (BOx) from team_b (it's usually attached to team_b)
        team_b = re.sub(r"\s*\(BO\d+\)\s*", "", team_b)

        # Remove tournament/league info after " - "
        team_b = re.sub(r"\s*-\s+.*$", "", team_b)

        if not team_a or not team_b:
            return None

        return (team_a, team_b)

    def sync_from_polymarket(
        self,
        game: str = "lol",
        limit: int = 50,
        only_active: bool = True,
    ) -> int:
        """Fetch markets from Polymarket API and create mappings.

        Args:
            game: Game type to fetch ("lol", "csgo", "dota2").
            limit: Maximum number of markets to fetch.
            only_active: If True, only fetch markets accepting orders.

        Returns:
            Number of new mappings created.
        """
        series_id = self.LOL_SERIES_ID if game == "lol" else None

        params: dict = {
            "closed": "false",
            "limit": limit,
        }
        if series_id:
            params["series_id"] = series_id

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.get(f"{self.GAMMA_API}/events", params=params)
                response.raise_for_status()
                events = response.json()
        except Exception as e:
            logger.error("polymarket_fetch_failed", error=str(e))
            return 0

        new_count = 0

        for event in events:
            title = event.get("title", "")
            markets = event.get("markets", [])

            if not markets:
                continue

            market = markets[0]

            # Skip if not accepting orders
            if only_active and not market.get("acceptingOrders", False):
                continue

            # Skip if already resolved (price is 0 or 1)
            prices_str = market.get("outcomePrices", "[]")
            try:
                prices = json.loads(prices_str)
                if prices and prices[0] in ["0", "1", 0, 1]:
                    continue
            except (json.JSONDecodeError, IndexError):
                pass

            # Parse teams from title
            teams = self.parse_teams_from_title(title)
            if not teams:
                logger.debug("cannot_parse_teams", title=title)
                continue

            team_a, team_b = teams

            # Get CLOB token IDs and outcomes
            try:
                clob_ids = json.loads(market.get("clobTokenIds", "[]"))
                outcomes = json.loads(market.get("outcomes", "[]"))
            except json.JSONDecodeError:
                continue

            if not clob_ids or len(outcomes) < 2:
                continue

            # Market ID is the first CLOB token ID
            market_id = clob_ids[0]

            # Skip if already registered
            if self.find_by_polymarket_id(market_id):
                continue

            # Skip Over/Under markets (not direct team vs team)
            if outcomes[0].lower() in ("over", "under"):
                logger.debug("skipping_over_under_market", title=title)
                continue

            # Create mapping: Map team names to their Polymarket outcome tokens
            # Two formats exist:
            # 1. Outcomes are team names: ["T1", "Gen.G"] -> map directly
            # 2. Outcomes are Yes/No: ["Yes", "No"] -> team_a=Yes, team_b=No
            if outcomes[0].lower() in ("yes", "no"):
                # Yes/No format: "Will team_a beat team_b?"
                # team_a (subject of question) maps to "Yes"
                outcome_mapping = {
                    team_a: outcomes[0],  # Yes
                    team_b: outcomes[1],  # No
                }
            else:
                # Team name format: outcomes ARE the team names
                outcome_mapping = {
                    outcomes[0]: outcomes[0],
                    outcomes[1]: outcomes[1],
                }

            mapping = self.add_mapping(
                game=game,
                event_identifier=f"{game}_{team_a}_vs_{team_b}",
                polymarket_id=market_id,
                outcomes=outcome_mapping,
            )

            # Also store the alternate CLOB ID for the second outcome
            if len(clob_ids) > 1:
                self._by_polymarket_id[clob_ids[1]] = mapping

            logger.info(
                "mapping_created",
                title=title[:50],
                team_a=team_a,
                team_b=team_b,
                market_id=market_id[:20] + "...",
            )
            new_count += 1

        logger.info(
            "sync_complete",
            game=game,
            total_events=len(events),
            new_mappings=new_count,
        )

        return new_count

    def find_market_fuzzy(
        self,
        game: str,
        team_a: str,
        team_b: str,
    ) -> Optional[MarketMapping]:
        """Find a market using fuzzy team name matching.

        Tries multiple strategies:
        1. Exact match after normalization
        2. Alias resolution
        3. Partial string matching

        Args:
            game: Game identifier.
            team_a: First team name (from PandaScore).
            team_b: Second team name (from PandaScore).

        Returns:
            Matching MarketMapping or None.
        """
        # Try exact match first
        result = self.find_market(game, [team_a, team_b])
        if result:
            return result

        # Normalize using aliases
        norm_a = self._normalize_team(team_a)
        norm_b = self._normalize_team(team_b)

        # Try with normalized names
        for mapping in self._mappings:
            if mapping.game.lower() != game.lower():
                continue

            mapping_teams = set(mapping.team_to_outcome.keys())
            mapping_teams_normalized = {
                self._normalize_team(t) for t in mapping_teams
            }

            if {norm_a, norm_b} == mapping_teams_normalized:
                return mapping

            # Try partial matching (team name contains or is contained)
            for mt in mapping_teams:
                mt_lower = mt.lower()
                if (
                    (norm_a in mt_lower or mt_lower in norm_a)
                    and (norm_b in mt_lower or mt_lower in norm_b)
                ):
                    continue  # Same team matched both, skip

                matched_a = norm_a in mt_lower or mt_lower in norm_a
                matched_b = norm_b in mt_lower or mt_lower in norm_b

                if matched_a or matched_b:
                    # Check the other team
                    other_teams = mapping_teams - {mt}
                    for ot in other_teams:
                        ot_lower = ot.lower()
                        if matched_a and (norm_b in ot_lower or ot_lower in norm_b):
                            return mapping
                        if matched_b and (norm_a in ot_lower or ot_lower in norm_a):
                            return mapping

        return None
