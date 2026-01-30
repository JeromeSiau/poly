"""Database connection and session management with async support."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
    create_async_engine,
    async_sessionmaker,
)
from sqlalchemy.orm import sessionmaker, Session

from .models import Base


# Default database URL (SQLite for development)
DEFAULT_DATABASE_URL = "sqlite:///./data/poly.db"
DEFAULT_ASYNC_DATABASE_URL = "sqlite+aiosqlite:///./data/poly.db"

# Global engine instances
_sync_engine: Optional[create_engine] = None
_async_engine: Optional[AsyncEngine] = None
_sync_session_factory: Optional[sessionmaker] = None
_async_session_factory: Optional[async_sessionmaker] = None


def get_sync_engine(database_url: str = DEFAULT_DATABASE_URL):
    """Get or create synchronous database engine."""
    global _sync_engine
    if _sync_engine is None:
        _sync_engine = create_engine(
            database_url,
            echo=False,
            future=True,
        )
    return _sync_engine


def get_async_engine(database_url: str = DEFAULT_ASYNC_DATABASE_URL) -> AsyncEngine:
    """Get or create asynchronous database engine."""
    global _async_engine
    if _async_engine is None:
        _async_engine = create_async_engine(
            database_url,
            echo=False,
            future=True,
        )
    return _async_engine


def get_sync_session(database_url: str = DEFAULT_DATABASE_URL) -> Session:
    """Get a synchronous database session."""
    global _sync_session_factory
    if _sync_session_factory is None:
        engine = get_sync_engine(database_url)
        _sync_session_factory = sessionmaker(
            bind=engine,
            autocommit=False,
            autoflush=False,
        )
    return _sync_session_factory()


def get_async_session_factory(
    database_url: str = DEFAULT_ASYNC_DATABASE_URL,
) -> async_sessionmaker[AsyncSession]:
    """Get async session factory."""
    global _async_session_factory
    if _async_session_factory is None:
        engine = get_async_engine(database_url)
        _async_session_factory = async_sessionmaker(
            bind=engine,
            class_=AsyncSession,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
        )
    return _async_session_factory


@asynccontextmanager
async def get_session(
    database_url: str = DEFAULT_ASYNC_DATABASE_URL,
) -> AsyncGenerator[AsyncSession, None]:
    """Get an async database session as a context manager."""
    factory = get_async_session_factory(database_url)
    session = factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


# Alias for backwards compatibility
get_async_session = get_session


def init_db(database_url: str = DEFAULT_DATABASE_URL) -> None:
    """Initialize the database by creating all tables."""
    engine = get_sync_engine(database_url)
    Base.metadata.create_all(bind=engine)


async def init_db_async(database_url: str = DEFAULT_ASYNC_DATABASE_URL) -> None:
    """Initialize the database asynchronously by creating all tables."""
    engine = get_async_engine(database_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db_async() -> None:
    """Close async database connections."""
    global _async_engine, _async_session_factory
    if _async_engine is not None:
        await _async_engine.dispose()
        _async_engine = None
        _async_session_factory = None


def close_db() -> None:
    """Close sync database connections."""
    global _sync_engine, _sync_session_factory
    if _sync_engine is not None:
        _sync_engine.dispose()
        _sync_engine = None
        _sync_session_factory = None


def reset_engines() -> None:
    """Reset all engine instances. Useful for testing."""
    global _sync_engine, _async_engine, _sync_session_factory, _async_session_factory
    if _sync_engine is not None:
        _sync_engine.dispose()
    if _async_engine is not None:
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(_async_engine.dispose())
            else:
                loop.run_until_complete(_async_engine.dispose())
        except RuntimeError:
            pass
    _sync_engine = None
    _async_engine = None
    _sync_session_factory = None
    _async_session_factory = None
