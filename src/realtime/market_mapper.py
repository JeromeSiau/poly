"""Market Mapper - Links game events to Polymarket markets."""

from dataclasses import dataclass, field
from typing import Optional


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
        """Normalize team name for comparison.

        Args:
            name: Raw team name.

        Returns:
            Normalized team name (lowercase, stripped).
        """
        return name.lower().strip()

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

    # Common team name aliases
    TEAM_ALIASES: dict[str, str] = {
        # League of Legends
        "skt": "T1",
        "sk telecom": "T1",
        "sk telecom t1": "T1",
        "skt t1": "T1",
        "damwon": "DK",
        "damwon gaming": "DK",
        "dwg": "DK",
        "dwg kia": "DK",
        "gen.g": "Gen.G",
        "geng": "Gen.G",
        "samsung": "Gen.G",
        "samsung galaxy": "Gen.G",
        # CS:GO / CS2
        "navi": "Navi",
        "natus vincere": "Navi",
        "faze": "FaZe",
        "faze clan": "FaZe",
        "g2": "G2",
        "g2 esports": "G2",
        "vitality": "Vitality",
        "team vitality": "Vitality",
        # Dota 2
        "og": "OG",
        "team spirit": "Spirit",
        "spirit": "Spirit",
        "liquid": "Liquid",
        "team liquid": "Liquid",
        # General
        "c9": "Cloud9",
        "cloud 9": "Cloud9",
    }

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
        """Normalize a team name using aliases.

        Args:
            team: Raw team name.

        Returns:
            Normalized/canonical team name.
        """
        team_lower = team.lower().strip()

        # Check if it's an alias
        if team_lower in self.TEAM_ALIASES:
            return self.TEAM_ALIASES[team_lower].lower()

        # Check if it matches any alias value
        for alias, canonical in self.TEAM_ALIASES.items():
            if canonical.lower() == team_lower:
                return canonical.lower()

        return team_lower

    def get_all_mappings(self) -> list[MarketMapping]:
        """Get all registered mappings.

        Returns:
            List of all MarketMapping instances.
        """
        return list(self._mappings)
