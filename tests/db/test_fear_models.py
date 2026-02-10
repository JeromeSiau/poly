# tests/db/test_fear_models.py
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.db.models import Base, FearPosition


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


class TestFearPosition:
    def test_create_position(self, db_session):
        pos = FearPosition(
            condition_id="0x123",
            token_id="tok_no",
            title="US strikes Iran by March 31",
            cluster="iran",
            side="NO",
            entry_price=0.65,
            size_usd=10_000.0,
            shares=10_000.0 / 0.65,
            fear_score=0.82,
            yes_price_at_entry=0.35,
        )
        db_session.add(pos)
        db_session.commit()
        fetched = db_session.query(FearPosition).first()
        assert fetched.condition_id == "0x123"
        assert fetched.cluster == "iran"
        assert fetched.is_open is True
        assert fetched.side == "NO"

    def test_close_position(self, db_session):
        pos = FearPosition(
            condition_id="0x456",
            token_id="tok_no",
            title="Khamenei out by Feb 28",
            cluster="iran",
            side="NO",
            entry_price=0.80,
            size_usd=5_000.0,
            shares=5_000.0 / 0.80,
            fear_score=0.75,
            yes_price_at_entry=0.20,
        )
        db_session.add(pos)
        db_session.commit()
        pos.is_open = False
        pos.exit_price = 0.95
        pos.realized_pnl = pos.shares * (0.95 - 0.80)
        db_session.commit()
        fetched = db_session.query(FearPosition).first()
        assert fetched.is_open is False
        assert fetched.realized_pnl > 0
