"""Configuration template - copy to settings.py and fill in values."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # === Polymarket ===
    POLYMARKET_API_KEY: str = ""
    POLYMARKET_API_SECRET: str = ""
    POLYMARKET_WALLET_ADDRESS: str = ""
    POLYMARKET_PRIVATE_KEY: str = ""

    # WebSocket endpoints
    POLYMARKET_CLOB_HTTP: str = "https://clob.polymarket.com"
    POLYMARKET_CLOB_WS: str = "wss://ws-subscriptions-clob.polymarket.com/ws/"
    POLYMARKET_RTDS_WS: str = "wss://ws-live-data.polymarket.com"

    # === PandaScore (Esports) ===
    PANDASCORE_API_KEY: str = ""
    PANDASCORE_BASE_URL: str = "https://api.pandascore.co"

    # === SportsDataIO ===
    SPORTSDATAIO_API_KEY: str = ""
    SPORTSDATAIO_BASE_URL: str = "https://api.sportsdata.io/v3"

    # === Telegram ===
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""

    # === Risk Parameters ===
    MAX_POSITION_PCT: float = 0.10  # 10% of capital
    DAILY_LOSS_LIMIT_PCT: float = 0.05  # 5% daily loss cap
    MIN_EDGE_PCT: float = 0.02  # 2% minimum edge
    ANOMALY_THRESHOLD_PCT: float = 0.15  # 15% edge = suspicious
    MIN_BROADCAST_LAG_SECONDS: float = 5.0  # Min lag to trade
    MAX_BROADCAST_LAG_SECONDS: float = 60.0  # Max lag (stale data?)

    # === Execution ===
    ORDER_TIMEOUT_SECONDS: float = 5.0
    MAX_TRADES_PER_HOUR: int = 20
    AUTOPILOT_MODE: bool = False

    # === Database ===
    DATABASE_URL: str = "sqlite+aiosqlite:///data/arb.db"

    model_config = {"env_file": ".env"}


settings = Settings()
