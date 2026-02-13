"""Tests for TDMakerOrder model and CRUD helpers."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.database import init_db_async, reset_engines
from src.db.models import Base, TDMakerOrder
from src.db.td_orders import save_order, load_orders, delete_order, mark_filled, mark_settled

DB_URL = "sqlite+aiosqlite:///:memory:"


# -- Sync model tests --

@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_td_maker_order_table_exists():
    cols = {c.name for c in TDMakerOrder.__table__.columns}
    assert TDMakerOrder.__tablename__ == "td_maker_orders"
    assert "order_id" in cols
    assert "status" in cols
    assert "platform" in cols


def test_td_maker_order_create(db_session):
    row = TDMakerOrder(
        platform="polymarket",
        strategy_tag="test",
        order_id="o1",
        condition_id="c1",
        token_id="t1",
        outcome="Up",
        price=0.80,
        size_usd=5.0,
        status="pending",
    )
    db_session.add(row)
    db_session.commit()
    fetched = db_session.query(TDMakerOrder).first()
    assert fetched.order_id == "o1"
    assert fetched.status == "pending"


# -- Async CRUD tests --

async def _init_db():
    reset_engines()
    await init_db_async(DB_URL)


@pytest.mark.asyncio
async def test_save_and_load():
    await _init_db()
    await save_order(
        db_url=DB_URL,
        platform="polymarket",
        strategy_tag="test_tag",
        order_id="ord_1",
        condition_id="cid_1",
        token_id="tok_1",
        outcome="Up",
        price=0.80,
        size_usd=5.0,
        status="pending",
        placed_at=1000.0,
    )
    rows = await load_orders(
        db_url=DB_URL, platform="polymarket", strategy_tag="test_tag", status="pending",
    )
    assert len(rows) == 1
    assert rows[0].order_id == "ord_1"
    assert rows[0].price == 0.80
    reset_engines()


@pytest.mark.asyncio
async def test_delete():
    await _init_db()
    await save_order(
        db_url=DB_URL,
        platform="polymarket",
        strategy_tag="t",
        order_id="ord_2",
        condition_id="c",
        token_id="t",
        outcome="Up",
        price=0.80,
        size_usd=5.0,
        status="pending",
    )
    deleted = await delete_order(db_url=DB_URL, order_id="ord_2")
    assert deleted
    rows = await load_orders(db_url=DB_URL, platform="polymarket", strategy_tag="t")
    assert len(rows) == 0
    reset_engines()


@pytest.mark.asyncio
async def test_mark_filled():
    await _init_db()
    await save_order(
        db_url=DB_URL,
        platform="polymarket",
        strategy_tag="t",
        order_id="ord_3",
        condition_id="c",
        token_id="t",
        outcome="Up",
        price=0.80,
        size_usd=5.0,
        status="pending",
    )
    await mark_filled(db_url=DB_URL, order_id="ord_3", shares=6.25, filled_at=2000.0)
    rows = await load_orders(db_url=DB_URL, platform="polymarket", strategy_tag="t", status="filled")
    assert len(rows) == 1
    assert rows[0].shares == 6.25
    assert rows[0].filled_at == 2000.0
    reset_engines()


@pytest.mark.asyncio
async def test_mark_settled():
    await _init_db()
    await save_order(
        db_url=DB_URL,
        platform="polymarket",
        strategy_tag="t",
        order_id="ord_4",
        condition_id="c",
        token_id="t",
        outcome="Up",
        price=0.80,
        size_usd=5.0,
        status="filled",
        shares=6.25,
    )
    await mark_settled(db_url=DB_URL, order_id="ord_4", pnl=1.25, settled_at=3000.0)
    rows = await load_orders(db_url=DB_URL, platform="polymarket", strategy_tag="t", status="settled")
    assert len(rows) == 1
    assert rows[0].pnl == 1.25
    assert rows[0].settled_at == 3000.0
    reset_engines()


@pytest.mark.asyncio
async def test_load_filters_by_status():
    await _init_db()
    for i, status in enumerate(["pending", "filled", "settled"]):
        await save_order(
            db_url=DB_URL,
            platform="polymarket",
            strategy_tag="filter_test",
            order_id=f"filt_{i}",
            condition_id="c",
            token_id="t",
            outcome="Up",
            price=0.80,
            size_usd=5.0,
            status=status,
        )
    all_rows = await load_orders(db_url=DB_URL, platform="polymarket", strategy_tag="filter_test")
    assert len(all_rows) == 3
    pending = await load_orders(db_url=DB_URL, platform="polymarket", strategy_tag="filter_test", status="pending")
    assert len(pending) == 1
    reset_engines()
