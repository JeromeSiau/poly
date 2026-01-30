# tests/realtime/test_market_mapper.py
import pytest
from src.realtime.market_mapper import MarketMapper, MarketMapping


class TestMarketMapper:
    @pytest.fixture
    def mapper(self):
        mapper = MarketMapper()
        mapper.add_mapping(
            game="lol",
            event_identifier="LCK_T1_vs_GenG_2026",
            polymarket_id="0x123abc",
            outcomes={"T1": "YES", "Gen.G": "NO"}
        )
        return mapper

    def test_find_market_for_event(self, mapper):
        mapping = mapper.find_market(
            game="lol",
            teams=["T1", "Gen.G"],
            league="LCK"
        )
        assert mapping is not None
        assert mapping.polymarket_id == "0x123abc"
        assert mapping.get_outcome_for_team("T1") == "YES"

    def test_find_market_returns_none_for_unknown(self, mapper):
        mapping = mapper.find_market(
            game="lol",
            teams=["Cloud9", "100T"],
            league="LCS"
        )
        assert mapping is None

    def test_outcome_mapping_correct(self, mapper):
        mapping = mapper.find_market(game="lol", teams=["T1", "Gen.G"], league="LCK")
        assert mapping.get_outcome_for_team("T1") == "YES"
        assert mapping.get_outcome_for_team("Gen.G") == "NO"


class TestMarketMappingCreation:
    def test_create_mapping_from_polymarket_market(self):
        polymarket_data = {
            "id": "0x456def",
            "question": "Will T1 beat Gen.G in LCK Spring Finals?",
            "outcomes": ["Yes", "No"],
            "tags": ["esports", "lol", "lck"]
        }
        mapping = MarketMapping.from_polymarket(polymarket_data, team_a="T1", team_b="Gen.G")
        assert mapping.polymarket_id == "0x456def"
        assert mapping.get_outcome_for_team("T1") == "Yes"
