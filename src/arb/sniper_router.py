"""Routes spike signals and score changes to trade actions across sibling conditions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from src.analysis.event_condition_mapper import EventConditionMapper
from src.feeds.spike_detector import SpikeSignal
from src.feeds.odds_api import ScoreChange


@dataclass(slots=True)
class SniperAction:
    condition_id: str
    outcome: str
    side: str  # "BUY"
    reason: str  # "spike_primary" | "spike_sibling" | "score_change"
    urgency: float  # higher = more urgent (used for ordering)
    source_event_slug: str


class SniperRouter:
    """Maps signals to concrete buy actions across conditions."""

    def __init__(self, mapper: EventConditionMapper) -> None:
        self._mapper = mapper

    def route_spike(self, spike: SpikeSignal) -> list[SniperAction]:
        actions: list[SniperAction] = []
        slug = self._mapper.event_slug_for(spike.condition_id)
        if not slug:
            return actions

        # Primary: buy the spiking side
        if spike.direction == "up":
            actions.append(SniperAction(
                condition_id=spike.condition_id,
                outcome=spike.outcome,
                side="BUY",
                reason="spike_primary",
                urgency=abs(spike.delta),
                source_event_slug=slug,
            ))

        # Siblings: infer direction for related conditions
        for sibling in self._mapper.siblings_of(spike.condition_id):
            sibling_cid = sibling["conditionId"]
            sibling_outcomes = sibling.get("outcomes", [])
            question = str(sibling.get("question", "")).lower()

            outcome_to_buy = self._infer_sibling_outcome(
                spike=spike,
                sibling_question=question,
                sibling_outcomes=sibling_outcomes,
            )
            if outcome_to_buy:
                actions.append(SniperAction(
                    condition_id=sibling_cid,
                    outcome=outcome_to_buy,
                    side="BUY",
                    reason="spike_sibling",
                    urgency=abs(spike.delta) * 0.7,
                    source_event_slug=slug,
                ))

        return actions

    def route_score_change(self, change: ScoreChange) -> list[SniperAction]:
        actions: list[SniperAction] = []

        # Find matching event slugs
        slugs = self._mapper.find_events_by_teams(change.home_team, change.away_team)
        if not slugs:
            slugs = self._mapper.find_events_by_teams(change.home_team)
        if not slugs:
            return actions

        home_scoring = change.home_score > change.prev_home_score
        away_scoring = change.away_score > change.prev_away_score

        for slug in slugs:
            for cond in self._mapper.conditions_for_event(slug):
                cid = cond["conditionId"]
                question = str(cond.get("question", "")).lower()
                outcomes = cond.get("outcomes", [])

                outcome = self._infer_outcome_from_score(
                    question=question,
                    outcomes=outcomes,
                    home_scoring=home_scoring,
                    away_scoring=away_scoring,
                    completed=change.completed,
                    home_score=change.home_score,
                    away_score=change.away_score,
                )
                if outcome:
                    urgency = 1.0 if change.completed else 0.6
                    actions.append(SniperAction(
                        condition_id=cid,
                        outcome=outcome,
                        side="BUY",
                        reason="score_change",
                        urgency=urgency,
                        source_event_slug=slug,
                    ))

        return actions

    @staticmethod
    def _infer_sibling_outcome(
        spike: SpikeSignal,
        sibling_question: str,
        sibling_outcomes: list[str],
    ) -> Optional[str]:
        if not sibling_outcomes:
            return None

        is_draw = "draw" in sibling_question
        is_ou = "o/u" in sibling_question or "over" in sibling_question

        # If the main market spiked up (team winning), draw becomes less likely
        if is_draw and spike.direction == "up":
            return "No" if "No" in sibling_outcomes else None

        # For O/U and other markets, we can't infer direction from a single spike
        # Just flag the condition but let the engine decide based on orderbook
        if is_ou:
            return None

        # Generic: if spike.direction == "up" on a win market, buy Yes on sibling win markets
        if spike.direction == "up" and "Yes" in sibling_outcomes:
            return "Yes"

        return None

    @staticmethod
    def _infer_outcome_from_score(
        question: str,
        outcomes: list[str],
        home_scoring: bool,
        away_scoring: bool,
        completed: bool,
        home_score: int,
        away_score: int,
    ) -> Optional[str]:
        if not outcomes:
            return None

        is_draw = "draw" in question
        is_win = "win" in question

        if is_draw:
            if home_score == away_score:
                return "Yes" if "Yes" in outcomes else None
            else:
                return "No" if "No" in outcomes else None

        if is_win and (home_scoring or away_scoring):
            # Check which team the question refers to
            if home_scoring and not away_scoring:
                return "Yes" if "Yes" in outcomes else None
            elif away_scoring and not home_scoring:
                return "No" if "No" in outcomes else None

        if completed:
            if home_score > away_score:
                return "Yes" if "Yes" in outcomes else None
            elif away_score > home_score:
                return "No" if "No" in outcomes else None

        return None
