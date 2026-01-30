"""
Event Detector for Reality Arbitrage Core.

Identifies significant game events that will move the market before
the market fully reacts. This is the core of the Reality Arbitrage strategy.

Event impact scoring:
- 0.0-0.3: Minor event (single kill early game)
- 0.3-0.6: Moderate event (objective, multi-kill)
- 0.6-0.8: Major event (baron, ace, significant score change)
- 0.8-1.0: Game-changing event (base race, match point)
"""

from dataclasses import dataclass
from typing import Optional, Any

from src.feeds.base import FeedEvent


@dataclass
class SignificantEvent:
    """Represents a classified significant game event."""

    original_event: Optional[FeedEvent]
    is_significant: bool
    impact_score: float  # 0-1 scale
    favored_team: Optional[str]
    event_description: str

    @property
    def should_trade(self) -> bool:
        """Determine if this event warrants placing a trade."""
        return self.is_significant and self.impact_score >= 0.4


class EventDetector:
    """
    Classifies game events and estimates their market impact.

    Routes events to game-specific classifiers that understand
    the nuances of each esport's game flow.
    """

    # League of Legends event weights
    LOL_WEIGHTS: dict[str, float] = {
        "kill": 0.15,
        "tower": 0.25,
        "dragon": 0.30,
        "rift_herald": 0.25,
        "baron": 0.85,
        "elder": 0.90,
        "inhibitor": 0.40,
        "ace": 0.60,
        "first_blood": 0.20,
        "nexus_turret": 0.70,
    }

    # CS:GO event weights
    CSGO_WEIGHTS: dict[str, float] = {
        "kill": 0.10,
        "round_end": 0.35,
        "ace": 0.50,
        "clutch": 0.45,
        "bomb_planted": 0.15,
        "bomb_defused": 0.35,
        "overtime_start": 0.60,
        "match_point": 0.80,
    }

    # Dota 2 event weights
    DOTA2_WEIGHTS: dict[str, float] = {
        "kill": 0.12,
        "tower": 0.20,
        "roshan": 0.75,
        "aegis_pickup": 0.70,
        "barracks": 0.50,
        "ancient_damage": 0.80,
        "team_wipe": 0.65,
        "buyback": 0.25,
        "mega_creeps": 0.85,
    }

    def __init__(self):
        """Initialize the event detector."""
        pass

    def classify(self, event: FeedEvent) -> SignificantEvent:
        """
        Classify an incoming feed event and determine its significance.

        Args:
            event: The raw feed event to classify

        Returns:
            SignificantEvent with classification details
        """
        game = event.game.lower()

        if game == "lol":
            return self._classify_lol(event)
        elif game == "csgo":
            return self._classify_csgo(event)
        elif game == "dota2":
            return self._classify_dota2(event)
        else:
            # Unknown game, return minimal significance
            return SignificantEvent(
                original_event=event,
                is_significant=False,
                impact_score=0.0,
                favored_team=None,
                event_description=f"Unknown game event: {event.event_type}"
            )

    def _classify_lol(self, event: FeedEvent) -> SignificantEvent:
        """
        Classify a League of Legends event.

        Late game events are more significant as they have higher
        comeback potential and direct impact on game outcome.
        """
        event_type = event.event_type.lower()
        data = event.data

        # Get base weight for event type
        base_weight = self.LOL_WEIGHTS.get(event_type, 0.10)

        # Get game time if available (in minutes)
        game_time = data.get("game_time_minutes", 15)  # Default to mid-game

        # Calculate time multiplier (late game events are more impactful)
        # Early game (<10min): 0.5x, Mid game (10-25min): 1.0x, Late game (>25min): 2.5-5.0x
        # In late game LoL, every kill matters significantly more as death timers
        # are longer and objectives can be taken during respawn time.
        if game_time < 10:
            time_multiplier = 0.5
        elif game_time < 25:
            time_multiplier = 1.0
        else:
            # Scale from 2.5x at 25min to 5.0x at 50min+
            # At 35min: 2.5 + (35-25) * 0.10 = 3.5x, giving kill 0.15 * 3.5 = 0.525
            time_multiplier = min(2.5 + (game_time - 25) * 0.10, 5.0)

        # Calculate final impact score
        impact_score = min(base_weight * time_multiplier, 1.0)

        # Baron and Elder are always high impact regardless of time
        if event_type in ("baron", "elder"):
            impact_score = max(impact_score, self.LOL_WEIGHTS[event_type])

        # Determine favored team
        favored_team = data.get("killer_team") or data.get("team")

        # Build description
        killer = data.get("killer", "Unknown")
        if event_type == "kill":
            description = f"Kill by {killer} ({favored_team}) at {game_time}min"
        elif event_type == "baron":
            description = f"Baron kill by {favored_team}"
        elif event_type == "elder":
            description = f"Elder Dragon taken by {favored_team}"
        else:
            description = f"{event_type} by {favored_team}"

        is_significant = impact_score >= 0.3

        return SignificantEvent(
            original_event=event,
            is_significant=is_significant,
            impact_score=impact_score,
            favored_team=favored_team,
            event_description=description
        )

    def _classify_csgo(self, event: FeedEvent) -> SignificantEvent:
        """
        Classify a CS:GO event.

        Round-based scoring means momentum and score differential
        are critical factors.
        """
        event_type = event.event_type.lower()
        data = event.data

        # Get base weight for event type
        base_weight = self.CSGO_WEIGHTS.get(event_type, 0.10)

        impact_score = base_weight
        favored_team = data.get("winner") or data.get("team")

        # For round_end events, calculate momentum based on score
        if event_type == "round_end" and "score" in data:
            score = data["score"]
            if isinstance(score, dict) and len(score) == 2:
                teams = list(score.keys())
                scores = list(score.values())

                # Calculate score differential
                diff = abs(scores[0] - scores[1])
                total_rounds = sum(scores)

                # Momentum factor: larger lead = more momentum
                momentum_factor = min(1.0 + diff * 0.1, 1.5)

                # Late game factor: rounds closer to 16 matter more
                max_score = max(scores)
                if max_score >= 15:  # Match point
                    late_game_factor = 2.0
                elif max_score >= 12:  # Close to winning
                    late_game_factor = 1.5
                else:
                    late_game_factor = 1.0

                impact_score = min(base_weight * momentum_factor * late_game_factor, 1.0)

                # Determine favored team (the one with higher score)
                if scores[0] > scores[1]:
                    favored_team = teams[0]
                else:
                    favored_team = teams[1]

        # Build description
        if event_type == "round_end":
            score = data.get("score", {})
            description = f"Round won by {favored_team} (Score: {score})"
        else:
            description = f"{event_type} by {favored_team}"

        is_significant = impact_score >= 0.3

        return SignificantEvent(
            original_event=event,
            is_significant=is_significant,
            impact_score=impact_score,
            favored_team=favored_team,
            event_description=description
        )

    def _classify_dota2(self, event: FeedEvent) -> SignificantEvent:
        """
        Classify a Dota 2 event.

        Similar to LoL but with Roshan as the major objective
        and different game timing considerations.
        """
        event_type = event.event_type.lower()
        data = event.data

        # Get base weight for event type
        base_weight = self.DOTA2_WEIGHTS.get(event_type, 0.10)

        # Get game time if available (in minutes)
        game_time = data.get("game_time_minutes", 20)  # Default to mid-game

        # Calculate time multiplier (Dota games tend to be longer)
        if game_time < 15:
            time_multiplier = 0.5
        elif game_time < 35:
            time_multiplier = 1.0
        else:
            time_multiplier = min(1.5 + (game_time - 35) * 0.03, 2.0)

        # Calculate final impact score
        impact_score = min(base_weight * time_multiplier, 1.0)

        # Roshan and ancient events are always high impact
        if event_type in ("roshan", "ancient_damage", "aegis_pickup"):
            impact_score = max(impact_score, self.DOTA2_WEIGHTS.get(event_type, 0.7))

        # Determine favored team
        favored_team = data.get("killer_team") or data.get("team")

        # Build description
        if event_type == "roshan":
            description = f"Roshan killed by {favored_team}"
        elif event_type == "ancient_damage":
            description = f"Ancient under attack by {favored_team}"
        else:
            description = f"{event_type} by {favored_team}"

        is_significant = impact_score >= 0.3

        return SignificantEvent(
            original_event=event,
            is_significant=is_significant,
            impact_score=impact_score,
            favored_team=favored_team,
            event_description=description
        )

    def estimate_price_impact(
        self,
        event: SignificantEvent,
        current_price: float
    ) -> float:
        """
        Estimate how the market price should move based on the event.

        The price movement is proportional to the impact score, with
        diminishing returns as price approaches extremes (0 or 1).

        Args:
            event: The classified significant event
            current_price: Current market price (0-1)

        Returns:
            Estimated new fair price (0-1)
        """
        if not event.is_significant:
            return current_price

        # Base price movement based on impact score
        # Impact score of 0.85 (baron) should move price by ~15%
        # Impact score of 0.55 (late kill) should move price by ~5-8%
        base_movement = event.impact_score * 0.18

        # Apply diminishing returns near price extremes
        # This models the fact that a 0.90 favorite can't move to 1.10
        if current_price > 0.5:
            # Room to move up is limited
            room_to_move = 1.0 - current_price
            effective_movement = base_movement * (room_to_move / 0.5)
        else:
            # Full movement available (price moving towards 0.5 and beyond)
            effective_movement = base_movement

        # Calculate new price
        new_price = current_price + effective_movement

        # Clamp to valid range
        return max(0.01, min(0.99, new_price))
