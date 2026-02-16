# src/bot/__init__.py
"""Telegram bot handlers for trading alerts."""

from .crossmarket_handlers import CrossMarketArbHandler

__all__ = ["CrossMarketArbHandler"]
