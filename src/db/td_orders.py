"""Async CRUD for TDMakerOrder — used by CryptoTDMaker and KalshiTDMaker."""

from __future__ import annotations

from typing import Optional

from sqlalchemy import select, update, delete as sa_delete

from src.db.database import get_session, DEFAULT_ASYNC_DATABASE_URL
from src.db.models import TDMakerOrder


async def save_order(
    *,
    db_url: str = DEFAULT_ASYNC_DATABASE_URL,
    platform: str,
    strategy_tag: str,
    order_id: str,
    condition_id: str,
    token_id: str,
    outcome: str,
    price: float,
    size_usd: float,
    status: str = "pending",
    shares: Optional[float] = None,
    placed_at: Optional[float] = None,
    filled_at: Optional[float] = None,
    extra: Optional[dict] = None,
) -> None:
    """Insert a new order row."""
    async with get_session(db_url) as s:
        row = TDMakerOrder(
            platform=platform,
            strategy_tag=strategy_tag,
            order_id=order_id,
            condition_id=condition_id,
            token_id=token_id,
            outcome=outcome,
            price=price,
            size_usd=size_usd,
            status=status,
            shares=shares,
            placed_at=placed_at,
            filled_at=filled_at,
            extra=extra,
        )
        s.add(row)


async def load_orders(
    *,
    db_url: str = DEFAULT_ASYNC_DATABASE_URL,
    platform: str,
    strategy_tag: str,
    status: Optional[str] = None,
) -> list[TDMakerOrder]:
    """Load orders filtered by platform, tag, and optionally status."""
    async with get_session(db_url) as s:
        q = select(TDMakerOrder).where(
            TDMakerOrder.platform == platform,
            TDMakerOrder.strategy_tag == strategy_tag,
        )
        if status:
            q = q.where(TDMakerOrder.status == status)
        result = await s.execute(q)
        return list(result.scalars().all())


async def delete_order(*, db_url: str = DEFAULT_ASYNC_DATABASE_URL, order_id: str) -> bool:
    """Delete an order by order_id. Returns True if a row was deleted."""
    async with get_session(db_url) as s:
        result = await s.execute(
            sa_delete(TDMakerOrder).where(TDMakerOrder.order_id == order_id)
        )
        return result.rowcount > 0


async def mark_filled(
    *,
    db_url: str = DEFAULT_ASYNC_DATABASE_URL,
    order_id: str,
    shares: float,
    filled_at: float,
) -> None:
    """Transition an order from pending to filled."""
    async with get_session(db_url) as s:
        await s.execute(
            update(TDMakerOrder)
            .where(TDMakerOrder.order_id == order_id)
            .values(status="filled", shares=shares, filled_at=filled_at)
        )


async def mark_settled(
    *,
    db_url: str = DEFAULT_ASYNC_DATABASE_URL,
    order_id: str,
    pnl: float,
    settled_at: float,
) -> None:
    """Transition an order from filled to settled."""
    async with get_session(db_url) as s:
        await s.execute(
            update(TDMakerOrder)
            .where(TDMakerOrder.order_id == order_id)
            .values(status="settled", pnl=pnl, settled_at=settled_at)
        )
