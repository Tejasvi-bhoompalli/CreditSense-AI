"""Recommendation rules: determinism, boundaries and escalations."""

from __future__ import annotations

import pytest

from src import config
from src.recommendation import recommend
from src.risk_scoring import score_prediction


def _recommend(probability: float, derived: dict | None = None, inputs: dict | None = None):
    risk = score_prediction(probability)
    return recommend(probability, risk, inputs or {}, derived or {})


@pytest.mark.parametrize(
    "probability,expected_key",
    [
        (0.05, "eligible"),
        (0.30, "eligible"),
        (0.3001, "conditional"),
        (0.50, "conditional"),
        (0.5001, "manual"),
        (0.70, "manual"),
        (0.7001, "elevated"),
        (0.95, "elevated"),
    ],
)
def test_probability_band_rules(probability, expected_key):
    result = _recommend(probability)
    assert result.key == expected_key
    assert result.label == config.RECOMMENDATION_LABELS[expected_key]
    assert result.rules_applied


def test_severe_delay_escalates_to_manual_review():
    result = _recommend(0.05, {"num_severely_delayed_months": 2, "max_repayment_delay": 4})
    assert result.key == "manual"
    assert any(rule.startswith("M1") for rule in result.rules_applied)
    assert result.concerns


def test_repeated_moderate_delay_escalates_to_conditional():
    result = _recommend(0.10, {"num_delayed_months": 3, "num_severely_delayed_months": 0})
    assert result.key == "conditional"
    assert any(rule.startswith("M2") for rule in result.rules_applied)


def test_low_repayment_ratio_escalates_to_conditional():
    result = _recommend(0.10, {"repayment_ratio": 0.001, "num_delayed_months": 0})
    assert result.key == "conditional"


def test_escalation_never_de_escalates():
    """A favourable secondary rule must not lower a high probability band."""
    result = _recommend(
        0.9,
        {
            "num_severely_delayed_months": 0,
            "num_delayed_months": 0,
            "repayment_ratio": 0.9,
            "credit_utilization": 0.05,
            "max_repayment_delay": -1,
        },
    )
    assert result.key == "elevated"


def test_recommendation_is_deterministic():
    first = _recommend(0.42, {"num_delayed_months": 1}).as_dict()
    second = _recommend(0.42, {"num_delayed_months": 1}).as_dict()
    assert first == second


def test_recommendation_content_is_complete():
    result = _recommend(0.55, {"credit_utilization": 0.98, "num_delayed_months": 1})
    assert result.headline
    assert result.reasons
    assert result.supporting_indicators
    assert result.concerns
    assert result.next_steps
    assert "does not approve" in result.disclaimer
    assert "human" in result.disclaimer


def test_recommendation_never_mentions_protected_attributes():
    result = _recommend(0.4, {}, {"sex": 1, "marriage": 2, "credit_limit": 50_000})
    serialised = " ".join(result.as_dict()["reasons"] + result.as_dict()["concerns"]).lower()
    assert "sex" not in serialised
    assert "marriage" not in serialised
    assert "gender" not in serialised


def test_all_recommendation_labels_are_defined():
    for key in ("eligible", "conditional", "manual", "elevated"):
        assert key in config.RECOMMENDATION_LABELS
