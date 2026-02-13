"""SQLAlchemy ORM models for game events, markets, trades, and positions."""

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    JSON,
    Text,
    Boolean,
    func,
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class GameEvent(Base):
    """Model for storing game events from data sources like PandaScore."""

    __tablename__ = "game_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    external_id = Column(String(255), unique=True, nullable=False, index=True)
    source = Column(String(100), nullable=False)  # e.g., "pandascore"
    game = Column(String(50), nullable=False)  # e.g., "lol", "csgo", "dota2"
    event_type = Column(String(100), nullable=False)  # e.g., "kill", "objective", "match_end"
    team = Column(String(255), nullable=True)
    player = Column(String(255), nullable=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    raw_data = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<GameEvent(id={self.id}, game={self.game}, event_type={self.event_type})>"


class Market(Base):
    """Model for storing Polymarket market information."""

    __tablename__ = "markets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    polymarket_id = Column(String(255), unique=True, nullable=False, index=True)
    title = Column(Text, nullable=False)
    game = Column(String(50), nullable=True)  # e.g., "lol", "csgo"
    event_name = Column(String(255), nullable=True)  # e.g., "LCK Spring 2026"
    outcomes = Column(JSON, nullable=False)  # List of possible outcomes
    current_prices = Column(JSON, nullable=True)  # Dict mapping outcome to price
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<Market(id={self.id}, title={self.title[:50]}...)>"


class Trade(Base):
    """Model for storing executed trades."""

    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    market_id = Column(String(255), nullable=False, index=True)
    side = Column(String(10), nullable=False)  # "BUY" or "SELL"
    outcome = Column(String(255), nullable=False)
    price = Column(Float, nullable=False)
    size = Column(Float, nullable=False)
    edge_pct = Column(Float, nullable=True)  # Expected edge percentage
    trigger_event = Column(String(100), nullable=True)  # Event that triggered the trade
    status = Column(String(50), nullable=False)  # "PENDING", "FILLED", "CANCELLED", "FAILED"
    execution_time_ms = Column(Integer, nullable=True)  # Execution time in milliseconds
    polymarket_order_id = Column(String(255), nullable=True)  # External order ID from Polymarket
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    filled_at = Column(DateTime, nullable=True)

    def __repr__(self) -> str:
        return f"<Trade(id={self.id}, side={self.side}, outcome={self.outcome}, status={self.status})>"


class Position(Base):
    """Open and closed positions."""

    __tablename__ = "positions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    market_id = Column(String(100), nullable=False, index=True)
    outcome = Column(String(100), nullable=False)
    entry_price = Column(Float, nullable=False)
    current_price = Column(Float, nullable=False)
    size = Column(Float, nullable=False)
    realized_pnl = Column(Float, default=0.0)
    unrealized_pnl = Column(Float, default=0.0)

    is_open = Column(Boolean, default=True)
    opened_at = Column(DateTime, default=datetime.utcnow)
    closed_at = Column(DateTime, nullable=True)

    def __repr__(self) -> str:
        return f"<Position(id={self.id}, market_id={self.market_id}, outcome={self.outcome}, size={self.size})>"


class CrossMarketEvent(Base):
    """Cross-market event pairs for arbitrage."""

    __tablename__ = "cross_market_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(Text, nullable=False)
    category = Column(String(50), nullable=True)  # sports, politics, crypto
    resolution_date = Column(DateTime, nullable=True)

    # Platform-specific IDs (nullable - not all platforms have all events)
    polymarket_id = Column(String(255), nullable=True, index=True)
    azuro_condition_id = Column(String(255), nullable=True, index=True)
    overtime_game_id = Column(String(255), nullable=True, index=True)

    # Matching metadata
    match_confidence = Column(Float, nullable=True)
    match_method = Column(String(50), nullable=True)  # llm, exact, fuzzy
    verified_at = Column(DateTime, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<CrossMarketEvent(id={self.id}, name={self.name[:30]}...)>"


class PriceSnapshot(Base):
    """Price snapshots for arb detection."""

    __tablename__ = "price_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(Integer, nullable=False, index=True)
    platform = Column(String(50), nullable=False)  # polymarket, azuro, overtime
    outcome = Column(String(100), nullable=False)  # YES, NO, team name
    price = Column(Float, nullable=False)
    liquidity = Column(Float, nullable=True)
    captured_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<PriceSnapshot(platform={self.platform}, price={self.price})>"


class CrossMarketOpportunity(Base):
    """Detected cross-market arbitrage opportunities."""

    __tablename__ = "cross_market_opportunities"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(Integer, nullable=False, index=True)

    # Source (buy side)
    source_platform = Column(String(50), nullable=False)
    source_price = Column(Float, nullable=False)
    source_liquidity = Column(Float, nullable=True)

    # Target (sell side)
    target_platform = Column(String(50), nullable=False)
    target_price = Column(Float, nullable=False)
    target_liquidity = Column(Float, nullable=True)

    # Calculations
    gross_edge_pct = Column(Float, nullable=False)
    fees_pct = Column(Float, nullable=True)
    gas_estimate = Column(Float, nullable=True)
    net_edge_pct = Column(Float, nullable=False)

    # Status
    status = Column(String(50), default="detected")  # detected, alerted, approved, executed, expired, skipped
    detected_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)

    def __repr__(self) -> str:
        return f"<CrossMarketOpportunity(id={self.id}, edge={self.net_edge_pct:.2%})>"


class CrossMarketTrade(Base):
    """Executed cross-market trades."""

    __tablename__ = "cross_market_trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    opportunity_id = Column(Integer, nullable=False, index=True)

    # Source leg execution
    source_tx_hash = Column(String(255), nullable=True)
    source_chain = Column(String(50), nullable=True)
    source_amount = Column(Float, nullable=True)
    source_price_filled = Column(Float, nullable=True)
    source_gas_paid = Column(Float, nullable=True)
    source_status = Column(String(50), nullable=True)  # pending, confirmed, failed

    # Target leg execution
    target_tx_hash = Column(String(255), nullable=True)
    target_chain = Column(String(50), nullable=True)
    target_amount = Column(Float, nullable=True)
    target_price_filled = Column(Float, nullable=True)
    target_gas_paid = Column(Float, nullable=True)
    target_status = Column(String(50), nullable=True)

    # Aggregate
    execution_time_ms = Column(Integer, nullable=True)
    realized_edge_pct = Column(Float, nullable=True)
    realized_pnl = Column(Float, nullable=True)

    executed_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<CrossMarketTrade(id={self.id}, pnl={self.realized_pnl})>"


class LiveObservation(Base):
    """Live observations for paper trading."""

    __tablename__ = "live_observations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    match_id = Column(String(255), nullable=False, index=True)
    event_type = Column(String(100), nullable=False)
    game_state = Column(JSON, nullable=False)

    # Model prediction
    model_prediction = Column(Float, nullable=False)

    # Market prices at different times
    polymarket_price = Column(Float, nullable=True)
    polymarket_price_30s = Column(Float, nullable=True)
    polymarket_price_60s = Column(Float, nullable=True)
    polymarket_price_120s = Column(Float, nullable=True)

    # Result
    actual_winner = Column(String(255), nullable=True)
    latency_ms = Column(Integer, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    @property
    def edge_theoretical(self) -> float:
        """Calculate theoretical edge."""
        if self.polymarket_price is None:
            return 0.0
        return self.model_prediction - self.polymarket_price

    def __repr__(self) -> str:
        return f"<LiveObservation(id={self.id}, match={self.match_id}, pred={self.model_prediction:.2f})>"


class PaperTrade(Base):
    """Simulated trades for paper trading."""

    __tablename__ = "paper_trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    observation_id = Column(Integer, nullable=False, index=True)

    side = Column(String(10), nullable=False)  # BUY or SELL
    entry_price = Column(Float, nullable=False)
    simulated_fill_price = Column(Float, nullable=False)
    size = Column(Float, nullable=False)

    edge_theoretical = Column(Float, nullable=False)
    edge_realized = Column(Float, nullable=True)
    exit_price = Column(Float, nullable=True)
    pnl = Column(Float, nullable=True)

    is_open = Column(Boolean, default=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    closed_at = Column(DateTime, nullable=True)

    def __repr__(self) -> str:
        return f"<PaperTrade(id={self.id}, side={self.side}, size={self.size}, pnl={self.pnl})>"


class OddsApiCache(Base):
    """Shared cache for external odds snapshots across daemon processes."""

    __tablename__ = "odds_api_cache"

    id = Column(Integer, primary_key=True, autoincrement=True)
    cache_key = Column(String(255), nullable=False, unique=True, index=True)
    payload = Column(JSON, nullable=False, default=list)
    credits_remaining = Column(Integer, nullable=True)
    credits_used = Column(Integer, nullable=True)
    credits_last_call = Column(Integer, nullable=True)
    fetched_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<OddsApiCache(id={self.id}, key={self.cache_key[:24]}...)>"


class FearPosition(Base):
    """Tracks positions in fear-selling strategy."""

    __tablename__ = "fear_positions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    condition_id = Column(String, nullable=False, index=True)
    token_id = Column(String, nullable=False)
    title = Column(String, nullable=False)
    cluster = Column(String, nullable=False, index=True)
    side = Column(String, nullable=False, default="NO")
    entry_price = Column(Float, nullable=False)
    size_usd = Column(Float, nullable=False)
    shares = Column(Float, nullable=False)
    fear_score = Column(Float, nullable=False)
    yes_price_at_entry = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    realized_pnl = Column(Float, nullable=True)
    unrealized_pnl = Column(Float, default=0.0)
    is_open = Column(Boolean, default=True, index=True)
    entry_trigger = Column(String, nullable=True)  # "scan" | "spike" | "manual"
    opened_at = Column(DateTime, server_default=func.now())
    closed_at = Column(DateTime, nullable=True)


class TDMakerOrder(Base):
    """Persisted TD maker orders — pending bids and open positions.

    Used by CryptoTDMaker and KalshiTDMaker to survive daemon restarts.
    """

    __tablename__ = "td_maker_orders"

    id = Column(Integer, primary_key=True, autoincrement=True)
    platform = Column(String(20), nullable=False, index=True)  # "polymarket" | "kalshi"
    strategy_tag = Column(String(100), nullable=False, index=True)
    order_id = Column(String(255), nullable=False, index=True)
    condition_id = Column(String(255), nullable=False)
    token_id = Column(String(255), nullable=False)
    outcome = Column(String(50), nullable=False)
    price = Column(Float, nullable=False)
    size_usd = Column(Float, nullable=False)
    shares = Column(Float, nullable=True)
    status = Column(String(20), nullable=False, index=True)  # "pending" | "filled" | "settled" | "cancelled"
    placed_at = Column(Float, nullable=True)
    filled_at = Column(Float, nullable=True)
    settled_at = Column(Float, nullable=True)
    pnl = Column(Float, nullable=True)
    extra = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
