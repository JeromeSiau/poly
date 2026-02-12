"""Credential and configuration validators."""

from src.exceptions import ConfigError


def validate_polymarket_credentials() -> None:
    """Raise ConfigError if Polymarket execution credentials are missing."""
    from config.settings import settings
    if not settings.POLYMARKET_PRIVATE_KEY:
        raise ConfigError("POLYMARKET_PRIVATE_KEY is required")
    if not settings.POLYMARKET_WALLET_ADDRESS:
        raise ConfigError("POLYMARKET_WALLET_ADDRESS is required")


def validate_odds_api() -> None:
    """Raise ConfigError if Odds API key is missing."""
    from config.settings import settings
    if not settings.ODDS_API_KEY:
        raise ConfigError("ODDS_API_KEY is required for external fair pricing")


def validate_telegram() -> None:
    """Raise ConfigError if Telegram credentials are missing."""
    from config.settings import settings
    if not settings.TELEGRAM_BOT_TOKEN:
        raise ConfigError("TELEGRAM_BOT_TOKEN is required")
    if not settings.TELEGRAM_CHAT_ID:
        raise ConfigError("TELEGRAM_CHAT_ID is required")
