"""Event normalizer for standardizing team names and event descriptions."""

import re
from datetime import datetime
from difflib import SequenceMatcher
from typing import Optional

# Team aliases mapping various names to canonical forms
TEAM_ALIASES: dict[str, str] = {
    # NFL Teams
    "chiefs": "kansas city chiefs",
    "kc chiefs": "kansas city chiefs",
    "kansas city": "kansas city chiefs",
    "eagles": "philadelphia eagles",
    "philly eagles": "philadelphia eagles",
    "philadelphia": "philadelphia eagles",
    "49ers": "san francisco 49ers",
    "niners": "san francisco 49ers",
    "san francisco": "san francisco 49ers",
    "sf 49ers": "san francisco 49ers",
    "ravens": "baltimore ravens",
    "baltimore": "baltimore ravens",
    "bills": "buffalo bills",
    "buffalo": "buffalo bills",
    "cowboys": "dallas cowboys",
    "dallas": "dallas cowboys",
    "packers": "green bay packers",
    "green bay": "green bay packers",
    "lions": "detroit lions",
    "detroit": "detroit lions",
    "dolphins": "miami dolphins",
    "miami": "miami dolphins",
    "jets": "new york jets",
    "ny jets": "new york jets",
    "giants": "new york giants",
    "ny giants": "new york giants",
    "patriots": "new england patriots",
    "pats": "new england patriots",
    "new england": "new england patriots",
    "raiders": "las vegas raiders",
    "lv raiders": "las vegas raiders",
    "steelers": "pittsburgh steelers",
    "pittsburgh": "pittsburgh steelers",
    "broncos": "denver broncos",
    "denver": "denver broncos",
    "chargers": "los angeles chargers",
    "la chargers": "los angeles chargers",
    "rams": "los angeles rams",
    "la rams": "los angeles rams",
    "seahawks": "seattle seahawks",
    "seattle": "seattle seahawks",
    "cardinals": "arizona cardinals",
    "arizona": "arizona cardinals",
    "falcons": "atlanta falcons",
    "atlanta": "atlanta falcons",
    "panthers": "carolina panthers",
    "carolina": "carolina panthers",
    "bears": "chicago bears",
    "chicago": "chicago bears",
    "bengals": "cincinnati bengals",
    "cincinnati": "cincinnati bengals",
    "browns": "cleveland browns",
    "cleveland": "cleveland browns",
    "texans": "houston texans",
    "houston": "houston texans",
    "colts": "indianapolis colts",
    "indianapolis": "indianapolis colts",
    "jaguars": "jacksonville jaguars",
    "jags": "jacksonville jaguars",
    "jacksonville": "jacksonville jaguars",
    "vikings": "minnesota vikings",
    "minnesota": "minnesota vikings",
    "saints": "new orleans saints",
    "new orleans": "new orleans saints",
    "titans": "tennessee titans",
    "tennessee": "tennessee titans",
    "commanders": "washington commanders",
    "washington": "washington commanders",
    "buccaneers": "tampa bay buccaneers",
    "bucs": "tampa bay buccaneers",
    "tampa bay": "tampa bay buccaneers",
    # NBA Teams
    "lakers": "los angeles lakers",
    "la lakers": "los angeles lakers",
    "l.a. lakers": "los angeles lakers",
    "clippers": "los angeles clippers",
    "la clippers": "los angeles clippers",
    "l.a. clippers": "los angeles clippers",
    "celtics": "boston celtics",
    "boston": "boston celtics",
    "warriors": "golden state warriors",
    "golden state": "golden state warriors",
    "gsw": "golden state warriors",
    "heat": "miami heat",
    "knicks": "new york knicks",
    "ny knicks": "new york knicks",
    "nets": "brooklyn nets",
    "brooklyn": "brooklyn nets",
    "bulls": "chicago bulls",
    "spurs": "san antonio spurs",
    "san antonio": "san antonio spurs",
    "mavericks": "dallas mavericks",
    "mavs": "dallas mavericks",
    "rockets": "houston rockets",
    "nuggets": "denver nuggets",
    "suns": "phoenix suns",
    "phoenix": "phoenix suns",
    "bucks": "milwaukee bucks",
    "milwaukee": "milwaukee bucks",
    "76ers": "philadelphia 76ers",
    "sixers": "philadelphia 76ers",
    "philly 76ers": "philadelphia 76ers",
    "raptors": "toronto raptors",
    "toronto": "toronto raptors",
    "thunder": "oklahoma city thunder",
    "okc": "oklahoma city thunder",
    "oklahoma city": "oklahoma city thunder",
    "jazz": "utah jazz",
    "utah": "utah jazz",
    "timberwolves": "minnesota timberwolves",
    "wolves": "minnesota timberwolves",
    "pelicans": "new orleans pelicans",
    "grizzlies": "memphis grizzlies",
    "memphis": "memphis grizzlies",
    "pistons": "detroit pistons",
    "cavaliers": "cleveland cavaliers",
    "cavs": "cleveland cavaliers",
    "pacers": "indiana pacers",
    "indiana": "indiana pacers",
    "hornets": "charlotte hornets",
    "charlotte": "charlotte hornets",
    "hawks": "atlanta hawks",
    "magic": "orlando magic",
    "orlando": "orlando magic",
    "wizards": "washington wizards",
    "kings": "sacramento kings",
    "sacramento": "sacramento kings",
    "blazers": "portland trail blazers",
    "trail blazers": "portland trail blazers",
    "portland": "portland trail blazers",
    # MLB Teams
    "yankees": "new york yankees",
    "ny yankees": "new york yankees",
    "mets": "new york mets",
    "ny mets": "new york mets",
    "red sox": "boston red sox",
    "dodgers": "los angeles dodgers",
    "la dodgers": "los angeles dodgers",
    "cubs": "chicago cubs",
    "white sox": "chicago white sox",
    "astros": "houston astros",
    "braves": "atlanta braves",
    "phillies": "philadelphia phillies",
    "padres": "san diego padres",
    "san diego": "san diego padres",
    "mariners": "seattle mariners",
    "angels": "los angeles angels",
    "la angels": "los angeles angels",
    "athletics": "oakland athletics",
    "as": "oakland athletics",
    "oakland": "oakland athletics",
    "twins": "minnesota twins",
    "royals": "kansas city royals",
    "kc royals": "kansas city royals",
    "orioles": "baltimore orioles",
    "blue jays": "toronto blue jays",
    "jays": "toronto blue jays",
    "rays": "tampa bay rays",
    "guardians": "cleveland guardians",
    "rangers": "texas rangers",
    "texas": "texas rangers",
    "diamondbacks": "arizona diamondbacks",
    "dbacks": "arizona diamondbacks",
    "rockies": "colorado rockies",
    "colorado": "colorado rockies",
    "reds": "cincinnati reds",
    "brewers": "milwaukee brewers",
    "pirates": "pittsburgh pirates",
    "nationals": "washington nationals",
    "marlins": "miami marlins",
    "giants": "san francisco giants",
    "sf giants": "san francisco giants",
}

# Stop words to filter out when normalizing
STOP_WORDS: set[str] = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
    "be", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "shall", "can", "need",
    "vs", "versus", "v", "against", "over", "win", "wins", "winning",
    "beat", "beats", "defeat", "defeats", "lose", "loses", "losing",
}


class EventNormalizer:
    """Normalizes event names and team names for matching across platforms."""

    def __init__(self) -> None:
        """Initialize the normalizer with team aliases."""
        self.team_aliases = TEAM_ALIASES
        self.stop_words = STOP_WORDS
        # Build reverse lookup for full team names
        self._full_team_names: set[str] = set(self.team_aliases.values())

    def normalize_team(self, team: str) -> str:
        """
        Normalize a team name to its canonical form.

        Args:
            team: The team name to normalize.

        Returns:
            The canonical lowercase team name.
        """
        # Convert to lowercase and strip whitespace
        normalized = team.lower().strip()

        # Remove special characters but keep spaces
        normalized = re.sub(r"[.]", "", normalized)  # Remove periods (L.A. -> LA)
        normalized = re.sub(r"\s+", " ", normalized)  # Normalize whitespace

        # Check if it's already a full team name
        if normalized in self._full_team_names:
            return normalized

        # Try to find in aliases
        if normalized in self.team_aliases:
            return self.team_aliases[normalized]

        # Try partial matching for common patterns
        # Handle "LA" -> "Los Angeles" conversion
        if normalized.startswith("la "):
            la_version = "los angeles " + normalized[3:]
            if la_version in self._full_team_names:
                return la_version

        return normalized

    def normalize_event(self, event_name: str) -> str:
        """
        Normalize an event name, including team names within it.

        Args:
            event_name: The event name to normalize.

        Returns:
            The normalized event name in lowercase.
        """
        # Convert to lowercase
        normalized = event_name.lower()

        # Remove special characters except spaces and hyphens
        normalized = re.sub(r"[?!.,]", "", normalized)
        normalized = re.sub(r"\s+", " ", normalized)

        # Try to replace team names with canonical forms
        for alias, canonical in sorted(
            self.team_aliases.items(), key=lambda x: len(x[0]), reverse=True
        ):
            # Use word boundaries to avoid partial matches
            pattern = r"\b" + re.escape(alias) + r"\b"
            normalized = re.sub(pattern, canonical, normalized)

        return normalized.strip()

    def extract_teams(self, event_name: str) -> list[str]:
        """
        Extract team names from an event name.

        Args:
            event_name: The event name to search.

        Returns:
            List of normalized team names found.
        """
        normalized = event_name.lower()
        found_teams: list[str] = []

        # First check for full team names (longer matches first)
        for full_name in sorted(self._full_team_names, key=len, reverse=True):
            if full_name in normalized:
                found_teams.append(full_name)
                # Remove to avoid duplicate detection via aliases
                normalized = normalized.replace(full_name, " ")

        # Then check for aliases
        for alias, canonical in sorted(
            self.team_aliases.items(), key=lambda x: len(x[0]), reverse=True
        ):
            pattern = r"\b" + re.escape(alias) + r"\b"
            if re.search(pattern, normalized) and canonical not in found_teams:
                found_teams.append(canonical)
                normalized = re.sub(pattern, " ", normalized)

        return found_teams

    def extract_date(self, text: str) -> Optional[datetime]:
        """
        Extract a date from text.

        Args:
            text: The text to search for dates.

        Returns:
            A datetime object if a date was found, None otherwise.
        """
        # Common date patterns
        patterns = [
            # February 9, 2025
            (r"(\w+)\s+(\d{1,2}),?\s+(\d{4})", self._parse_month_day_year),
            # 2025-02-09
            (r"(\d{4})-(\d{2})-(\d{2})", self._parse_iso_date),
            # 02/09/2025
            (r"(\d{1,2})/(\d{1,2})/(\d{4})", self._parse_mdy_date),
            # 09-02-2025 (European style)
            (r"(\d{1,2})-(\d{1,2})-(\d{4})", self._parse_dmy_date),
        ]

        for pattern, parser in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    return parser(match)
                except (ValueError, KeyError):
                    continue

        return None

    def _parse_month_day_year(self, match: re.Match) -> datetime:
        """Parse 'Month Day, Year' format."""
        months = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12,
            "jan": 1, "feb": 2, "mar": 3, "apr": 4,
            "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9,
            "oct": 10, "nov": 11, "dec": 12,
        }
        month_str = match.group(1).lower()
        day = int(match.group(2))
        year = int(match.group(3))
        month = months[month_str]
        return datetime(year, month, day)

    def _parse_iso_date(self, match: re.Match) -> datetime:
        """Parse ISO format 'YYYY-MM-DD'."""
        year = int(match.group(1))
        month = int(match.group(2))
        day = int(match.group(3))
        return datetime(year, month, day)

    def _parse_mdy_date(self, match: re.Match) -> datetime:
        """Parse US format 'MM/DD/YYYY'."""
        month = int(match.group(1))
        day = int(match.group(2))
        year = int(match.group(3))
        return datetime(year, month, day)

    def _parse_dmy_date(self, match: re.Match) -> datetime:
        """Parse European format 'DD-MM-YYYY'."""
        day = int(match.group(1))
        month = int(match.group(2))
        year = int(match.group(3))
        return datetime(year, month, day)

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two event descriptions.

        Uses a combination of sequence matching and Jaccard similarity
        for better semantic comparison.

        Args:
            text1: First event description.
            text2: Second event description.

        Returns:
            Similarity score between 0 and 1.
        """
        # Normalize both texts
        norm1 = self.normalize_event(text1)
        norm2 = self.normalize_event(text2)

        # Remove stop words for comparison
        words1 = [w for w in norm1.split() if w not in self.stop_words]
        words2 = [w for w in norm2.split() if w not in self.stop_words]

        filtered1 = " ".join(words1)
        filtered2 = " ".join(words2)

        # Sequence similarity
        seq_similarity = SequenceMatcher(None, filtered1, filtered2).ratio()

        # Jaccard similarity (set-based word overlap)
        set1 = set(words1)
        set2 = set(words2)
        if set1 or set2:
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            jaccard_similarity = intersection / union if union > 0 else 0.0
        else:
            jaccard_similarity = 0.0

        # Combine both metrics (weighted average favoring Jaccard for semantic matching)
        return 0.4 * seq_similarity + 0.6 * jaccard_similarity
