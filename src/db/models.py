"""SQLAlchemy ORM models for game events, markets, trades, and positions."""

from datetime import datetime
from typing import Any

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    JSON,
    Text,
    Boolean,
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
    order_id = Column(String(255), nullable=True)  # External order ID from Polymarket
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    filled_at = Column(DateTime, nullable=True)

    def __repr__(self) -> str:
        return f"<Trade(id={self.id}, side={self.side}, outcome={self.outcome}, status={self.status})>"


class Position(Base):
    """Model for tracking current positions in markets."""

    __tablename__ = "positions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    market_id = Column(String(255), nullable=False, index=True)
    outcome = Column(String(255), nullable=False)
    size = Column(Float, nullable=False, default=0.0)
    average_price = Column(Float, nullable=True)
    realized_pnl = Column(Float, default=0.0)
    unrealized_pnl = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<Position(id={self.id}, market_id={self.market_id}, outcome={self.outcome}, size={self.size})>"
