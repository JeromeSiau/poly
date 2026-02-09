"""Maps Polymarket conditions to sporting events for cross-condition trading."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any, Optional


def _parse_json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except (json.JSONDecodeError, TypeError):
            return []
    return []


def _extract_event_slug(raw: dict[str, Any]) -> str:
    events = raw.get("events", [])
    if isinstance(events, str):
        events = _parse_json_list(events)
    if isinstance(events, list) and events:
        slug = events[0].get("slug") if isinstance(events[0], dict) else None
        if isinstance(slug, str):
            return slug
    return str(raw.get("slug", ""))


def _normalize_for_search(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower().strip())


class EventConditionMapper:
    """Groups Polymarket conditions by event_slug for cross-condition trading."""

    def __init__(self) -> None:
        self._cid_to_slug: dict[str, str] = {}
        self._slug_to_cids: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._cid_to_entry: dict[str, dict[str, Any]] = {}
        self._slug_to_teams: dict[str, str] = {}  # slug -> normalized question text

    def build(self, raw_markets: list[dict[str, Any]]) -> None:
        self._cid_to_slug.clear()
        self._slug_to_cids.clear()
        self._cid_to_entry.clear()
        self._slug_to_teams.clear()

        for raw in raw_markets:
            cid = str(raw.get("conditionId") or "")
            if not cid:
                continue
            slug = _extract_event_slug(raw)
            if not slug:
                continue

            outcomes = _parse_json_list(raw.get("outcomes"))
            token_ids = _parse_json_list(raw.get("clobTokenIds"))
            question = str(raw.get("question") or "")

            entry = {
                "conditionId": cid,
                "question": question,
                "outcomes": outcomes,
                "clobTokenIds": token_ids,
                "event_slug": slug,
            }

            self._cid_to_slug[cid] = slug
            self._slug_to_cids[slug].append(entry)
            self._cid_to_entry[cid] = entry
            if slug not in self._slug_to_teams:
                self._slug_to_teams[slug] = _normalize_for_search(question)

    def siblings_of(self, condition_id: str) -> list[dict[str, Any]]:
        slug = self._cid_to_slug.get(condition_id, "")
        if not slug:
            return []
        return [e for e in self._slug_to_cids[slug] if e["conditionId"] != condition_id]

    def event_slug_for(self, condition_id: str) -> str:
        return self._cid_to_slug.get(condition_id, "")

    def all_conditions_for_event(self, event_slug: str) -> int:
        return len(self._slug_to_cids.get(event_slug, []))

    def conditions_for_event(self, event_slug: str) -> list[dict[str, Any]]:
        return list(self._slug_to_cids.get(event_slug, []))

    def token_ids_for(self, condition_id: str) -> list[str]:
        entry = self._cid_to_entry.get(condition_id)
        if entry is None:
            return []
        return list(entry.get("clobTokenIds", []))

    def all_token_ids(self) -> list[str]:
        out: list[str] = []
        for entry in self._cid_to_entry.values():
            out.extend(entry.get("clobTokenIds", []))
        return out

    def find_events_by_teams(self, *team_fragments: str) -> list[str]:
        normalized = [_normalize_for_search(f) for f in team_fragments if f]
        if not normalized:
            return []
        matches: list[str] = []
        for slug, text in self._slug_to_teams.items():
            if all(frag in text for frag in normalized):
                matches.append(slug)
        if not matches:
            for slug, text in self._slug_to_teams.items():
                if any(frag in text for frag in normalized):
                    matches.append(slug)
        return matches
