"""Pure parsing and conversion utilities.

Extracted from ``scripts/run_two_sided_inventory.py`` so that multiple
scripts can import them without pulling in the entire two-sided runner.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def parse_json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
            return decoded if isinstance(decoded, list) else []
        except json.JSONDecodeError:
            return []
    return []


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def _ensure_sync_db_url(database_url: str) -> str:
    if not database_url:
        from config.settings import settings
        return settings.DATABASE_URL
    if "://" not in database_url:
        return database_url
    return database_url.replace("+aiosqlite", "").replace("+aiomysql", "+pymysql")


def _parse_datetime(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str) and value:
        raw = value.strip()
        if raw.endswith("Z"):
            raw = raw.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(raw)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def _first_event_slug(raw: dict[str, Any]) -> str:
    events = raw.get("events", [])
    if isinstance(events, str):
        events = parse_json_list(events)
    if isinstance(events, list) and events:
        slug = events[0].get("slug")
        if isinstance(slug, str):
            return slug
    return str(raw.get("slug", ""))


def _parse_csv_values(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _extract_outcome_price_map(raw: dict[str, Any]) -> dict[str, float]:
    outcomes = [str(o) for o in parse_json_list(raw.get("outcomes", []))]
    prices_raw = parse_json_list(raw.get("outcomePrices", []))
    if len(outcomes) < 2 or len(prices_raw) < 2:
        return {}

    out: dict[str, float] = {}
    for idx, outcome in enumerate(outcomes):
        if idx >= len(prices_raw):
            break
        value = _to_float(prices_raw[idx], default=-1.0)
        if 0.0 <= value <= 1.0:
            out[outcome] = value
    return out


def _normalize_outcome_label(value: str) -> str:
    return " ".join(value.strip().lower().split())
