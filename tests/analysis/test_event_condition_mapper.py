import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.analysis.event_condition_mapper import EventConditionMapper


def test_build_from_gamma_markets():
    """Mapper groups conditions by event_slug."""
    raw_markets = [
        {
            "conditionId": "cid_win",
            "question": "Will Jaguares win?",
            "events": [{"slug": "col1-jag-dep-2026-02-08"}],
            "outcomes": '["Yes","No"]',
            "clobTokenIds": '["tok_win_yes","tok_win_no"]',
        },
        {
            "conditionId": "cid_draw",
            "question": "Will it be a draw?",
            "events": [{"slug": "col1-jag-dep-2026-02-08"}],
            "outcomes": '["Yes","No"]',
            "clobTokenIds": '["tok_draw_yes","tok_draw_no"]',
        },
        {
            "conditionId": "cid_ou15",
            "question": "O/U 1.5 goals",
            "events": [{"slug": "col1-jag-dep-2026-02-08"}],
            "outcomes": '["Over","Under"]',
            "clobTokenIds": '["tok_ou_over","tok_ou_under"]',
        },
        {
            "conditionId": "cid_other",
            "question": "Will Liverpool win?",
            "events": [{"slug": "epl-liv-ars-2026-02-09"}],
            "outcomes": '["Yes","No"]',
            "clobTokenIds": '["tok_other_yes","tok_other_no"]',
        },
    ]

    mapper = EventConditionMapper()
    mapper.build(raw_markets)

    siblings = mapper.siblings_of("cid_win")
    sibling_cids = {c["conditionId"] for c in siblings}
    assert sibling_cids == {"cid_draw", "cid_ou15"}

    assert mapper.event_slug_for("cid_win") == "col1-jag-dep-2026-02-08"
    assert mapper.event_slug_for("cid_other") == "epl-liv-ars-2026-02-09"
    assert mapper.all_conditions_for_event("col1-jag-dep-2026-02-08") == 3

    # token lookup
    assert mapper.token_ids_for("cid_win") == ["tok_win_yes", "tok_win_no"]


def test_match_score_change_to_event():
    """Mapper can find event_slug from team names."""
    raw_markets = [
        {
            "conditionId": "cid_win",
            "question": "Will Jaguares de Córdoba FC win on 2026-02-08?",
            "events": [{"slug": "col1-jag-dep-2026-02-08"}],
            "outcomes": '["Yes","No"]',
            "clobTokenIds": '["tok_a","tok_b"]',
        },
    ]
    mapper = EventConditionMapper()
    mapper.build(raw_markets)

    slugs = mapper.find_events_by_teams("Jaguares", "Deportivo Pereira")
    # "jaguares" is in the normalized question, but "deportivo pereira" is not
    # So the exact-all match won't work, it should fall back to any-match
    assert "col1-jag-dep-2026-02-08" in slugs
