"""Tests for FearMarketScanner — fear/tail-risk market discovery and scoring."""
import pytest

from src.arb.fear_scanner import (
    FEAR_KEYWORDS,
    CLUSTER_PATTERNS,
    FearMarketCandidate,
    FearMarketScanner,
)


class TestFearKeywords:
    def test_high_fear_keywords_present(self):
        """High-tier fear keywords must include core geopolitical/violence terms."""
        high = FEAR_KEYWORDS["high"]
        for kw in [
            "strike", "invade", "nuclear", "war", "attack",
            "bomb", "collapse", "fall", "regime change",
            "die", "killed", "coup", "assassinate",
        ]:
            assert kw in high, f"Missing high-tier keyword: {kw}"

    def test_medium_fear_keywords_present(self):
        """Medium-tier fear keywords must include diplomatic/economic terms."""
        medium = FEAR_KEYWORDS["medium"]
        for kw in [
            "ceasefire", "resign", "impeach", "default", "recession",
            "shutdown", "sanctions", "regime", "annex", "deploy", "mobilize",
        ]:
            assert kw in medium, f"Missing medium-tier keyword: {kw}"


class TestFearMarketScanner:
    def test_score_high_fear_market(self):
        """High-fear geopolitical market should score >= 0.7."""
        scanner = FearMarketScanner()
        score = scanner.score_market(
            title="US strikes Iran by March 31, 2026",
            yes_price=0.45,
            volume_24h=200_000,
            end_date_days=30,
        )
        assert score >= 0.7

    def test_score_low_fear_market(self):
        """Non-fear entertainment market should score < 0.4."""
        scanner = FearMarketScanner()
        score = scanner.score_market(
            title="Will Taylor Swift release album",
            yes_price=0.50,
            volume_24h=10_000,
            end_date_days=60,
        )
        assert score < 0.4

    def test_score_medium_fear_market(self):
        """Medium-fear diplomatic market should score >= 0.5."""
        scanner = FearMarketScanner()
        score = scanner.score_market(
            title="Russia Ukraine ceasefire by June 2026",
            yes_price=0.35,
            volume_24h=150_000,
            end_date_days=120,
        )
        assert score >= 0.5

    def test_detect_cluster_iran(self):
        scanner = FearMarketScanner()
        cluster = scanner.detect_cluster("Will the US strike Iran before July?")
        assert cluster == "iran"

    def test_detect_cluster_russia_ukraine(self):
        scanner = FearMarketScanner()
        cluster = scanner.detect_cluster(
            "Russia Ukraine ceasefire by end of 2026"
        )
        assert cluster == "russia_ukraine"

    def test_detect_cluster_unknown(self):
        scanner = FearMarketScanner()
        cluster = scanner.detect_cluster("Will Taylor Swift release a new album?")
        assert cluster == "other"

    def test_estimate_base_rate_short_deadline(self):
        """Short deadline (14 days) for Iran cluster should yield low probability."""
        scanner = FearMarketScanner()
        rate = scanner.estimate_base_rate(
            end_date_days=14,
            cluster="iran",
        )
        assert rate < 0.15

    def test_estimate_base_rate_long_deadline(self):
        """Long deadline (300 days) for Iran cluster should be moderate."""
        scanner = FearMarketScanner()
        rate = scanner.estimate_base_rate(
            end_date_days=300,
            cluster="iran",
        )
        assert rate > 0.05
        assert rate < 0.40

    def test_filter_candidates_yes_range(self):
        """Only markets with YES price in the configured range should pass."""
        scanner = FearMarketScanner()  # default range: 0.05 - 0.65
        markets = [
            {
                "condition_id": "a",
                "title": "War market",
                "tokens": [
                    {"outcome": "Yes", "price": 0.30},
                    {"outcome": "No", "price": 0.70, "token_id": "tok-a"},
                ],
            },
            {
                "condition_id": "b",
                "title": "Near certain",
                "tokens": [
                    {"outcome": "Yes", "price": 0.90},
                    {"outcome": "No", "price": 0.10, "token_id": "tok-b"},
                ],
            },
            {
                "condition_id": "c",
                "title": "Very unlikely fear",
                "tokens": [
                    {"outcome": "Yes", "price": 0.03},
                    {"outcome": "No", "price": 0.97, "token_id": "tok-c"},
                ],
            },
            {
                "condition_id": "d",
                "title": "Edge of range",
                "tokens": [
                    {"outcome": "Yes", "price": 0.65},
                    {"outcome": "No", "price": 0.35, "token_id": "tok-d"},
                ],
            },
        ]
        candidates = scanner.filter_candidates(markets)
        ids = [c["condition_id"] for c in candidates]
        assert "a" in ids  # 0.30 in [0.05, 0.65]
        assert "d" in ids  # 0.65 in [0.05, 0.65]
        assert "b" not in ids  # 0.90 too high
        assert "c" not in ids  # 0.03 too low


class TestFearMarketCandidate:
    def test_expected_return(self):
        """Expected return calculation for NO bet."""
        candidate = FearMarketCandidate(
            condition_id="cond-001",
            token_id="tok-001",
            title="Will Iran be invaded?",
            yes_price=0.20,
            no_price=0.80,
            estimated_no_probability=0.90,
            edge_pct=0.10,
            volume_24h=50000.0,
            liquidity=10000.0,
            end_date_iso="2026-06-01",
            fear_score=0.75,
            cluster="iran",
        )
        # EV = p_no * win_payout - (1-p_no) * loss
        # win_payout = 1.0 - 0.80 = 0.20
        # loss = 0.80
        # EV = 0.90 * 0.20 - 0.10 * 0.80 = 0.18 - 0.08 = 0.10
        assert candidate.expected_return == pytest.approx(0.10, abs=0.01)

    def test_is_valid(self):
        candidate = FearMarketCandidate(
            condition_id="cond-001",
            token_id="tok-001",
            title="Nuclear war by 2027?",
            yes_price=0.15,
            no_price=0.85,
            estimated_no_probability=0.95,
            edge_pct=0.10,
            volume_24h=50000.0,
            liquidity=10000.0,
            end_date_iso="2027-01-01",
            fear_score=0.85,
            cluster="other",
        )
        assert candidate.is_valid

    def test_is_invalid_low_volume(self):
        candidate = FearMarketCandidate(
            condition_id="cond-001",
            token_id="tok-001",
            title="Test",
            yes_price=0.15,
            no_price=0.85,
            estimated_no_probability=0.95,
            edge_pct=0.10,
            volume_24h=100.0,
            liquidity=50.0,
            end_date_iso="2027-01-01",
            fear_score=0.5,
            cluster="other",
        )
        assert not candidate.is_valid
