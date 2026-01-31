"""Cross-market event matcher for identifying identical events across platforms.

This module orchestrates matching identical events across Polymarket, Azuro,
and Overtime using text similarity for candidate generation and LLM for verification.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import structlog

from src.feeds.azuro import AzuroEvent
from src.feeds.overtime import OvertimeGame
from src.matching.llm_verifier import LLMVerifier, MatchResult
from src.matching.normalizer import EventNormalizer

logger = structlog.get_logger()


@dataclass
class MatchedEvent:
    """Represents a matched event across multiple prediction market platforms.

    Attributes:
        name: Normalized name of the matched event
        category: Event category (e.g., 'sports', 'politics', 'crypto')
        polymarket_id: Polymarket market ID if matched
        azuro_condition_id: Azuro condition ID if matched
        overtime_game_id: Overtime game ID if matched
        confidence: LLM verification confidence score (0.0-1.0)
        polymarket_event: Full Polymarket event data
        azuro_event: Full Azuro event data
        overtime_game: Full Overtime game data
    """

    name: str
    category: str
    polymarket_id: Optional[str] = None
    azuro_condition_id: Optional[str] = None
    overtime_game_id: Optional[str] = None
    confidence: float = 0.0
    polymarket_event: Optional[dict[str, Any]] = None
    azuro_event: Optional[AzuroEvent] = None
    overtime_game: Optional[OvertimeGame] = None

    @property
    def has_polymarket(self) -> bool:
        """Check if this event has a Polymarket match."""
        return self.polymarket_id is not None

    @property
    def has_azuro(self) -> bool:
        """Check if this event has an Azuro match."""
        return self.azuro_condition_id is not None

    @property
    def has_overtime(self) -> bool:
        """Check if this event has an Overtime match."""
        return self.overtime_game_id is not None

    @property
    def platforms_count(self) -> int:
        """Count the number of platforms this event is matched on."""
        count = 0
        if self.has_polymarket:
            count += 1
        if self.has_azuro:
            count += 1
        if self.has_overtime:
            count += 1
        return count

    @property
    def is_arbitrageable(self) -> bool:
        """Check if this event can be used for arbitrage (requires 2+ platforms)."""
        return self.platforms_count >= 2


class CrossMarketMatcher:
    """Orchestrates matching events across Polymarket, Azuro, and Overtime.

    Uses text similarity for candidate generation and LLM verification
    to ensure accurate matching of identical events across platforms.
    """

    MIN_SIMILARITY_THRESHOLD = 0.6
    MIN_CONFIDENCE_THRESHOLD = 0.95

    def __init__(
        self,
        llm_verifier: Optional[LLMVerifier] = None,
        normalizer: Optional[EventNormalizer] = None,
        similarity_threshold: float = MIN_SIMILARITY_THRESHOLD,
        confidence_threshold: float = MIN_CONFIDENCE_THRESHOLD,
    ):
        """Initialize the cross-market matcher.

        Args:
            llm_verifier: LLM verifier for match verification. Creates one if not provided.
            normalizer: Event normalizer for text normalization. Creates one if not provided.
            similarity_threshold: Minimum similarity score for candidate generation.
            confidence_threshold: Minimum confidence for accepting LLM verification.
        """
        self._verifier = llm_verifier or LLMVerifier()
        self._normalizer = normalizer or EventNormalizer()
        self._similarity_threshold = similarity_threshold
        self._confidence_threshold = confidence_threshold
        self._cache: dict[str, MatchedEvent] = {}

    def _get_polymarket_title(self, pm_event: dict[str, Any]) -> str:
        """Extract title from Polymarket event.

        Args:
            pm_event: Polymarket event dictionary.

        Returns:
            Event title string.
        """
        return pm_event.get("title", pm_event.get("question", ""))

    def _get_azuro_description(self, azuro_event: AzuroEvent) -> str:
        """Generate description from Azuro event.

        Args:
            azuro_event: Azuro event object.

        Returns:
            Generated description string.
        """
        return f"{azuro_event.home_team} vs {azuro_event.away_team} - {azuro_event.league}"

    def _get_overtime_description(self, game: OvertimeGame) -> str:
        """Generate description from Overtime game.

        Args:
            game: Overtime game object.

        Returns:
            Generated description string.
        """
        return f"{game.home_team} vs {game.away_team} - {game.sport}"

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two event descriptions.

        Args:
            text1: First event description.
            text2: Second event description.

        Returns:
            Similarity score between 0.0 and 1.0.
        """
        return self._normalizer.calculate_similarity(text1, text2)

    async def find_potential_matches(
        self,
        polymarket_events: list[dict[str, Any]],
        azuro_events: list[AzuroEvent],
        overtime_games: list[OvertimeGame],
    ) -> list[dict[str, Any]]:
        """Find potential event matches using text similarity.

        This generates candidate matches based on text similarity scores.
        Candidates should be verified using verify_match() or match_all().

        Args:
            polymarket_events: List of Polymarket event dictionaries.
            azuro_events: List of Azuro events.
            overtime_games: List of Overtime games.

        Returns:
            List of candidate match dictionaries with similarity scores.
        """
        candidates = []

        # Compare Polymarket events with Azuro events
        for pm_event in polymarket_events:
            pm_title = self._get_polymarket_title(pm_event)
            if not pm_title:
                continue

            for azuro_event in azuro_events:
                azuro_desc = self._get_azuro_description(azuro_event)
                similarity = self._calculate_similarity(pm_title, azuro_desc)

                if similarity >= self._similarity_threshold:
                    candidates.append({
                        "pm_event": pm_event,
                        "azuro_event": azuro_event,
                        "similarity": similarity,
                        "type": "pm_azuro",
                    })
                    logger.debug(
                        "candidate_found",
                        type="pm_azuro",
                        pm_title=pm_title,
                        azuro_desc=azuro_desc,
                        similarity=similarity,
                    )

        # Compare Polymarket events with Overtime games
        for pm_event in polymarket_events:
            pm_title = self._get_polymarket_title(pm_event)
            if not pm_title:
                continue

            for ot_game in overtime_games:
                ot_desc = self._get_overtime_description(ot_game)
                similarity = self._calculate_similarity(pm_title, ot_desc)

                if similarity >= self._similarity_threshold:
                    candidates.append({
                        "pm_event": pm_event,
                        "overtime_game": ot_game,
                        "similarity": similarity,
                        "type": "pm_overtime",
                    })
                    logger.debug(
                        "candidate_found",
                        type="pm_overtime",
                        pm_title=pm_title,
                        ot_desc=ot_desc,
                        similarity=similarity,
                    )

        # Compare Azuro events with Overtime games
        for azuro_event in azuro_events:
            azuro_desc = self._get_azuro_description(azuro_event)

            for ot_game in overtime_games:
                ot_desc = self._get_overtime_description(ot_game)
                similarity = self._calculate_similarity(azuro_desc, ot_desc)

                if similarity >= self._similarity_threshold:
                    candidates.append({
                        "azuro_event": azuro_event,
                        "overtime_game": ot_game,
                        "similarity": similarity,
                        "type": "azuro_overtime",
                    })
                    logger.debug(
                        "candidate_found",
                        type="azuro_overtime",
                        azuro_desc=azuro_desc,
                        ot_desc=ot_desc,
                        similarity=similarity,
                    )

        # Sort by similarity score (highest first)
        candidates.sort(key=lambda x: x["similarity"], reverse=True)

        logger.info(
            "potential_matches_found",
            total_candidates=len(candidates),
            pm_azuro=len([c for c in candidates if c["type"] == "pm_azuro"]),
            pm_overtime=len([c for c in candidates if c["type"] == "pm_overtime"]),
            azuro_overtime=len([c for c in candidates if c["type"] == "azuro_overtime"]),
        )

        return candidates

    async def verify_match(self, candidate: dict[str, Any]) -> Optional[MatchedEvent]:
        """Verify a candidate match using LLM verification.

        Args:
            candidate: Candidate match dictionary from find_potential_matches().

        Returns:
            MatchedEvent if verification succeeds, None otherwise.
        """
        match_type = candidate.get("type")

        if match_type == "pm_azuro":
            return await self._verify_pm_azuro(candidate)
        elif match_type == "pm_overtime":
            return await self._verify_pm_overtime(candidate)
        elif match_type == "azuro_overtime":
            return await self._verify_azuro_overtime(candidate)
        else:
            logger.warning("unknown_match_type", match_type=match_type)
            return None

    async def _verify_pm_azuro(self, candidate: dict[str, Any]) -> Optional[MatchedEvent]:
        """Verify a Polymarket-Azuro match candidate.

        Args:
            candidate: Candidate with pm_event and azuro_event.

        Returns:
            MatchedEvent if verification succeeds, None otherwise.
        """
        pm_event = candidate["pm_event"]
        azuro_event = candidate["azuro_event"]

        pm_title = self._get_polymarket_title(pm_event)
        azuro_desc = self._get_azuro_description(azuro_event)

        result = await self._verifier.verify_match(
            event1_name=pm_title,
            event1_platform="polymarket",
            event2_name=azuro_desc,
            event2_platform="azuro",
            additional_context=f"Azuro sport: {azuro_event.sport}, league: {azuro_event.league}",
        )

        if result.is_match and result.confidence >= self._confidence_threshold:
            matched = MatchedEvent(
                name=pm_title,
                category="sports",
                polymarket_id=pm_event.get("id"),
                azuro_condition_id=azuro_event.condition_id,
                confidence=result.confidence,
                polymarket_event=pm_event,
                azuro_event=azuro_event,
            )
            self._cache_match(matched)
            logger.info(
                "match_verified",
                type="pm_azuro",
                name=matched.name,
                confidence=result.confidence,
            )
            return matched

        logger.debug(
            "match_rejected",
            type="pm_azuro",
            is_match=result.is_match,
            confidence=result.confidence,
            reasoning=result.reasoning,
        )
        return None

    async def _verify_pm_overtime(self, candidate: dict[str, Any]) -> Optional[MatchedEvent]:
        """Verify a Polymarket-Overtime match candidate.

        Args:
            candidate: Candidate with pm_event and overtime_game.

        Returns:
            MatchedEvent if verification succeeds, None otherwise.
        """
        pm_event = candidate["pm_event"]
        ot_game = candidate["overtime_game"]

        pm_title = self._get_polymarket_title(pm_event)
        ot_desc = self._get_overtime_description(ot_game)

        result = await self._verifier.verify_match(
            event1_name=pm_title,
            event1_platform="polymarket",
            event2_name=ot_desc,
            event2_platform="overtime",
            additional_context=f"Overtime sport: {ot_game.sport}",
        )

        if result.is_match and result.confidence >= self._confidence_threshold:
            matched = MatchedEvent(
                name=pm_title,
                category="sports",
                polymarket_id=pm_event.get("id"),
                overtime_game_id=ot_game.game_id,
                confidence=result.confidence,
                polymarket_event=pm_event,
                overtime_game=ot_game,
            )
            self._cache_match(matched)
            logger.info(
                "match_verified",
                type="pm_overtime",
                name=matched.name,
                confidence=result.confidence,
            )
            return matched

        logger.debug(
            "match_rejected",
            type="pm_overtime",
            is_match=result.is_match,
            confidence=result.confidence,
            reasoning=result.reasoning,
        )
        return None

    async def _verify_azuro_overtime(self, candidate: dict[str, Any]) -> Optional[MatchedEvent]:
        """Verify an Azuro-Overtime match candidate.

        Args:
            candidate: Candidate with azuro_event and overtime_game.

        Returns:
            MatchedEvent if verification succeeds, None otherwise.
        """
        azuro_event = candidate["azuro_event"]
        ot_game = candidate["overtime_game"]

        azuro_desc = self._get_azuro_description(azuro_event)
        ot_desc = self._get_overtime_description(ot_game)

        result = await self._verifier.verify_match(
            event1_name=azuro_desc,
            event1_platform="azuro",
            event2_name=ot_desc,
            event2_platform="overtime",
            additional_context=f"Azuro: {azuro_event.sport}/{azuro_event.league}, Overtime: {ot_game.sport}",
        )

        if result.is_match and result.confidence >= self._confidence_threshold:
            matched = MatchedEvent(
                name=azuro_desc,
                category="sports",
                azuro_condition_id=azuro_event.condition_id,
                overtime_game_id=ot_game.game_id,
                confidence=result.confidence,
                azuro_event=azuro_event,
                overtime_game=ot_game,
            )
            self._cache_match(matched)
            logger.info(
                "match_verified",
                type="azuro_overtime",
                name=matched.name,
                confidence=result.confidence,
            )
            return matched

        logger.debug(
            "match_rejected",
            type="azuro_overtime",
            is_match=result.is_match,
            confidence=result.confidence,
            reasoning=result.reasoning,
        )
        return None

    async def match_all(
        self,
        polymarket_events: list[dict[str, Any]],
        azuro_events: list[AzuroEvent],
        overtime_games: list[OvertimeGame],
    ) -> list[MatchedEvent]:
        """Find and verify all matches across platforms.

        This is the main entry point for matching. It:
        1. Finds potential matches using text similarity
        2. Verifies each candidate using LLM
        3. Returns verified matches

        Args:
            polymarket_events: List of Polymarket event dictionaries.
            azuro_events: List of Azuro events.
            overtime_games: List of Overtime games.

        Returns:
            List of verified MatchedEvent objects.
        """
        # Find candidates
        candidates = await self.find_potential_matches(
            polymarket_events=polymarket_events,
            azuro_events=azuro_events,
            overtime_games=overtime_games,
        )

        # Verify each candidate
        verified_matches = []
        for candidate in candidates:
            result = await self.verify_match(candidate)
            if result:
                verified_matches.append(result)

        # Merge matches that share platform IDs (create multi-platform matches)
        merged_matches = self._merge_matches(verified_matches)

        logger.info(
            "matching_complete",
            candidates_found=len(candidates),
            verified_matches=len(verified_matches),
            merged_matches=len(merged_matches),
        )

        return merged_matches

    def _merge_matches(self, matches: list[MatchedEvent]) -> list[MatchedEvent]:
        """Merge matches that share platform IDs into multi-platform matches.

        Args:
            matches: List of MatchedEvent objects (may be pairwise).

        Returns:
            List of merged MatchedEvent objects.
        """
        if not matches:
            return []

        # Group matches by platform IDs
        pm_index: dict[str, list[MatchedEvent]] = {}
        azuro_index: dict[str, list[MatchedEvent]] = {}
        overtime_index: dict[str, list[MatchedEvent]] = {}

        for match in matches:
            if match.polymarket_id:
                pm_index.setdefault(match.polymarket_id, []).append(match)
            if match.azuro_condition_id:
                azuro_index.setdefault(match.azuro_condition_id, []).append(match)
            if match.overtime_game_id:
                overtime_index.setdefault(match.overtime_game_id, []).append(match)

        # Merge matches that share IDs
        merged: list[MatchedEvent] = []
        processed: set[int] = set()

        for match in matches:
            if id(match) in processed:
                continue

            # Find all related matches
            related = {match}
            if match.polymarket_id:
                related.update(pm_index.get(match.polymarket_id, []))
            if match.azuro_condition_id:
                related.update(azuro_index.get(match.azuro_condition_id, []))
            if match.overtime_game_id:
                related.update(overtime_index.get(match.overtime_game_id, []))

            # Merge related matches
            merged_match = self._merge_related(list(related))
            merged.append(merged_match)

            # Mark all as processed
            for m in related:
                processed.add(id(m))

        return merged

    def _merge_related(self, matches: list[MatchedEvent]) -> MatchedEvent:
        """Merge a list of related matches into a single MatchedEvent.

        Args:
            matches: List of related MatchedEvent objects.

        Returns:
            Merged MatchedEvent with all platform IDs.
        """
        if len(matches) == 1:
            return matches[0]

        # Collect all IDs and events
        polymarket_id = None
        azuro_condition_id = None
        overtime_game_id = None
        polymarket_event = None
        azuro_event = None
        overtime_game = None
        name = matches[0].name
        category = matches[0].category
        max_confidence = 0.0

        for match in matches:
            if match.polymarket_id:
                polymarket_id = match.polymarket_id
                polymarket_event = match.polymarket_event
            if match.azuro_condition_id:
                azuro_condition_id = match.azuro_condition_id
                azuro_event = match.azuro_event
            if match.overtime_game_id:
                overtime_game_id = match.overtime_game_id
                overtime_game = match.overtime_game
            max_confidence = max(max_confidence, match.confidence)

        return MatchedEvent(
            name=name,
            category=category,
            polymarket_id=polymarket_id,
            azuro_condition_id=azuro_condition_id,
            overtime_game_id=overtime_game_id,
            confidence=max_confidence,
            polymarket_event=polymarket_event,
            azuro_event=azuro_event,
            overtime_game=overtime_game,
        )

    def _cache_match(self, match: MatchedEvent) -> None:
        """Add a match to the internal cache.

        Args:
            match: MatchedEvent to cache.
        """
        cache_key = self._make_cache_key(match)
        self._cache[cache_key] = match

    def _make_cache_key(self, match: MatchedEvent) -> str:
        """Create a cache key for a matched event.

        Args:
            match: MatchedEvent to create key for.

        Returns:
            Cache key string.
        """
        parts = []
        if match.polymarket_id:
            parts.append(f"pm:{match.polymarket_id}")
        if match.azuro_condition_id:
            parts.append(f"az:{match.azuro_condition_id}")
        if match.overtime_game_id:
            parts.append(f"ot:{match.overtime_game_id}")
        return "|".join(sorted(parts))

    def get_cached_matches(self) -> list[MatchedEvent]:
        """Get all cached matches.

        Returns:
            List of cached MatchedEvent objects.
        """
        return list(self._cache.values())

    def clear_cache(self) -> None:
        """Clear the match cache."""
        self._cache.clear()
        logger.debug("match_cache_cleared")
