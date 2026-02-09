"""Configuration template - copy to settings.py and fill in values."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # === Polymarket ===
    POLYMARKET_API_KEY: str = ""
    POLYMARKET_API_SECRET: str = ""
    POLYMARKET_API_PASSPHRASE: str = ""
    POLYMARKET_WALLET_ADDRESS: str = ""
    POLYMARKET_PRIVATE_KEY: str = ""
    POLYMARKET_CHAIN_ID: int = 137
    POLYMARKET_FEE_BPS: int = 0
    POLYMARKET_ORDER_TYPE: str = "FOK"
    POLYMARKET_POST_ONLY: bool = False

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

    # === The Odds API (External Fair) ===
    ODDS_API_KEY: str = ""
    ODDS_API_BASE_URL: str = "https://api.the-odds-api.com/v4"
    ODDS_API_SPORTS: str = "upcoming"
    ODDS_API_REGIONS: str = "eu"
    ODDS_API_MARKETS: str = "h2h"
    ODDS_API_MIN_REFRESH_SECONDS: float = 14400.0
    ODDS_MATCH_MIN_CONFIDENCE: float = 0.68
    ODDS_SHARED_CACHE_ENABLED: bool = True
    ODDS_SHARED_CACHE_TTL_SECONDS: float = 0.0

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
    REALITY_SYNC_MARKET_LIMIT: int = 200
    REALITY_SYNC_ONLY_ACTIVE: bool = True
    REALITY_USE_FRAMES: bool = True

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

    # === Crypto Reality Arb ===
    CRYPTO_ARB_SYMBOLS: str = "BTCUSDT,ETHUSDT,SOLUSDT"
    CRYPTO_ARB_SCAN_INTERVAL: float = 1.0  # scan every 1 second
    CRYPTO_ARB_FAIR_VALUE_WINDOW: int = 10  # last 10 trades for VWAP
    CRYPTO_ARB_CEX_SENSITIVITY: float = 30.0  # prob shift multiplier
    CRYPTO_ARB_STALE_SECONDS: float = 45.0
    CAPITAL_ALLOCATION_CRYPTO_PCT: float = 0.0  # disabled by default

    # === NO Bet Strategy (NeverYES contrarian approach) ===
    NO_BET_MIN_YES_PRICE: float = 0.35  # lower bound of sweet spot
    NO_BET_MAX_YES_PRICE: float = 0.65  # upper bound of sweet spot
    NO_BET_MIN_LIQUIDITY: float = 1000.0
    NO_BET_MIN_VOLUME_24H: float = 5000.0
    NO_BET_SCAN_INTERVAL: float = 300.0
    NO_BET_MAX_PER_MARKET_PCT: float = 0.02
    CAPITAL_ALLOCATION_NOBET_PCT: float = 0.0

    # === Two-Sided Inventory Arb ===
    TWO_SIDED_SCAN_INTERVAL: float = 20.0
    TWO_SIDED_MIN_EDGE_PCT: float = 0.015
    TWO_SIDED_EXIT_EDGE_PCT: float = 0.003
    TWO_SIDED_MIN_ORDER_USD: float = 25.0
    TWO_SIDED_MAX_ORDER_USD: float = 400.0
    TWO_SIDED_MAX_OUTCOME_INVENTORY_USD: float = 2500.0
    TWO_SIDED_MAX_MARKET_NET_USD: float = 1200.0
    TWO_SIDED_INVENTORY_SKEW_PCT: float = 0.02
    TWO_SIDED_MAX_HOLD_SECONDS: float = 86400.0
    TWO_SIDED_MIN_LIQUIDITY: float = 10000.0
    TWO_SIDED_MIN_VOLUME_24H: float = 5000.0
    TWO_SIDED_MAX_DAYS_TO_END: float = 3.0
    TWO_SIDED_MAX_ORDERS_PER_CYCLE: int = 15
    TWO_SIDED_SIGNAL_COOLDOWN_SECONDS: float = 45.0
    TWO_SIDED_MAX_BOOK_CONCURRENCY: int = 30
    TWO_SIDED_EXTERNAL_FAIR_BLEND: float = 0.85

    # === Market Screener ===
    SCREENER_MIN_ALPHA_SCORE: float = 0.6
    SCREENER_TOP_N: int = 10
    SCREENER_SCAN_INTERVAL: float = 3600.0
    SCREENER_LLM_PROVIDER: str = "claude"
    PERPLEXITY_API_KEY: str = ""

    # === Crypto 15-Minute Strategies ===
    CRYPTO_MINUTE_ENABLED: bool = True
    CRYPTO_MINUTE_SCAN_INTERVAL: float = 3.0
    CRYPTO_MINUTE_SYMBOLS: str = "BTCUSDT,ETHUSDT"
    CRYPTO_MINUTE_BINANCE_URL: str = "https://api.binance.com/api/v3/ticker/price"
    CRYPTO_MINUTE_GAMMA_URL: str = "https://gamma-api.polymarket.com"

    # Entry window (seconds remaining)
    CRYPTO_MINUTE_MIN_ENTRY_TIME: int = 120
    CRYPTO_MINUTE_MAX_ENTRY_TIME: int = 300

    # Time Decay strategy
    CRYPTO_MINUTE_TD_THRESHOLD: float = 0.88
    CRYPTO_MINUTE_TD_MIN_GAP_PCT: float = 0.3

    # Long Vol strategy
    CRYPTO_MINUTE_LV_THRESHOLD: float = 0.15
    CRYPTO_MINUTE_LV_MAX_GAP_PCT: float = 0.5

    # Paper trading
    CRYPTO_MINUTE_PAPER_SIZE_USD: float = 10.0
    CRYPTO_MINUTE_PAPER_CAPITAL: float = 1000.0
    CRYPTO_MINUTE_PAPER_FILE: str = "data/crypto_minute_paper.jsonl"

    # === Combinatorial Arbitrage ===
    COMBO_ARB_SCAN_INTERVAL: float = 60.0
    COMBO_ARB_MIN_PROFIT: float = 0.05
    COMBO_ARB_MAX_PAIRS_PER_SCAN: int = 50
    COMBO_ARB_DEPENDENCY_CONFIDENCE: float = 0.90

    model_config = {"env_file": ".env"}


settings = Settings()
