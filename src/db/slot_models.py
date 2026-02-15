"""Slot models — re-exported from src.db.models for backwards compatibility."""

from src.db.models import Base as SlotBase, SlotSnapshot, SlotResolution  # noqa: F401

__all__ = ["SlotBase", "SlotSnapshot", "SlotResolution"]
