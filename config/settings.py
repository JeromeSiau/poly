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

    # === ML Model ===
    ML_MODEL_PATH: str = "models/impact_model.pkl"
    ML_USE_MODEL: bool = True  # Set to False to use static weights

    # === Database ===
    DATABASE_URL: str = "sqlite+aiosqlite:///data/arb.db"

    # === Azuro (Cross-Market) ===
    AZURO_SUBGRAPH_URL: str = "https://thegraph.azuro.org/subgraphs/name/azuro-protocol/azuro-api-polygon-v3"
    AZURO_POLYGON_RPC: str = ""
    AZURO_GNOSIS_RPC: str = ""

    # === Overtime (Cross-Market) ===
    OVERTIME_SUBGRAPH_URL: str = "https://api.thegraph.com/subgraphs/name/thales-markets/overtime-optimism"
    OVERTIME_OPTIMISM_RPC: str = ""

    # === Anthropic (LLM for event matching) ===
    ANTHROPIC_API_KEY: str = ""
    LLM_MATCH_CONFIDENCE_THRESHOLD: float = 0.95
    LLM_MODEL: str = "claude-3-haiku-20240307"

    # === Cross-Market Arb Settings ===
    CROSSMARKET_SCAN_INTERVAL_SECONDS: float = 5.0
    CROSSMARKET_MIN_EDGE_PCT: float = 0.02
    CROSSMARKET_ALERT_EXPIRY_SECONDS: int = 60

    # === Multi-Chain Wallet ===
    WALLET_PRIVATE_KEY: str = ""
    POLYGON_RPC_URL: str = ""
    OPTIMISM_RPC_URL: str = ""

    # === Capital Allocation ===
    GLOBAL_CAPITAL: float = 10000.0
    CAPITAL_ALLOCATION_REALITY_PCT: float = 50.0
    CAPITAL_ALLOCATION_CROSSMARKET_PCT: float = 50.0

    model_config = {"env_file": ".env"}


settings = Settings()
