# tests/db/test_models.py
import pytest
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.models import Base, GameEvent, Market, Trade, Position


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_game_event_creation(db_session):
    event = GameEvent(
        external_id="pandascore_12345",
        source="pandascore",
        game="lol",
        event_type="kill",
        team="T1",
        player="Faker",
        timestamp=datetime.utcnow(),
        raw_data={"kills": 5, "deaths": 1}
    )
    db_session.add(event)
    db_session.commit()

    assert event.id is not None
    assert event.game == "lol"
    assert event.event_type == "kill"


def test_market_creation(db_session):
    market = Market(
        polymarket_id="0x123abc",
        title="T1 vs Gen.G - Winner",
        game="lol",
        event_name="LCK Spring 2026",
        outcomes=["T1", "Gen.G"],
        current_prices={"T1": 0.65, "Gen.G": 0.35}
    )
    db_session.add(market)
    db_session.commit()

    assert market.id is not None
    assert market.outcomes == ["T1", "Gen.G"]


def test_trade_creation(db_session):
    trade = Trade(
        market_id="0x123abc",
        side="BUY",
        outcome="T1",
        price=0.55,
        size=100.0,
        edge_pct=0.08,
        trigger_event="kill",
        status="FILLED",
        execution_time_ms=120
    )
    db_session.add(trade)
    db_session.commit()

    assert trade.id is not None
    assert trade.edge_pct == 0.08
