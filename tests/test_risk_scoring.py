"""Risk score mapping, band boundaries and category consistency."""

from __future__ import annotations

import pytest

from src import config
from src.risk_scoring import (
    band_bounds,
    category_transition,
    classify_risk_score,
    describe_bands,
    risk_color,
    risk_score_from_probability,
    score_prediction,
)


def test_risk_score_is_a_hundred_times_probability():
    assert risk_score_from_probability(0.0) == 0.0
    assert risk_score_from_probability(0.5) == 50.0
    assert risk_score_from_probability(1.0) == 100.0
    assert risk_score_from_probability(0.1234) == pytest.approx(12.3)


def test_risk_score_clamps_outside_unit_interval():
    assert risk_score_from_probability(-0.4) == 0.0
    assert risk_score_from_probability(1.7) == 100.0


def test_risk_score_rounds_to_one_decimal():
    assert risk_score_from_probability(0.3066) == pytest.approx(30.7, abs=0.05)
    assert risk_score_from_probability(0.00001) == 0.0


@pytest.mark.parametrize(
    "score,expected",
    [
        (0.0, "LOW RISK"),
        (0.1, "LOW RISK"),
        (30.0, "LOW RISK"),
        (30.1, "MEDIUM RISK"),
        (45.0, "MEDIUM RISK"),
        (60.0, "MEDIUM RISK"),
        (60.1, "HIGH RISK"),
        (100.0, "HIGH RISK"),
    ],
)
def test_category_boundaries(score, expected):
    assert classify_risk_score(score) == expected


def test_boundaries_are_stable_across_probabilities():
    # 0.30 and 0.60 sit exactly on a boundary; remember the score is rounded to one
    # decimal before it is classified, so the step over the boundary must survive
    # that rounding.
    assert classify_risk_score(risk_score_from_probability(0.30)) == "LOW RISK"
    assert classify_risk_score(risk_score_from_probability(0.305)) == "MEDIUM RISK"
    assert classify_risk_score(risk_score_from_probability(0.60)) == "MEDIUM RISK"
    assert classify_risk_score(risk_score_from_probability(0.605)) == "HIGH RISK"


def test_score_prediction_is_self_consistent():
    result = score_prediction(0.421)
    assert result.risk_score == pytest.approx(42.1)
    assert result.risk_category == "MEDIUM RISK"
    assert result.band_lower == 30.0
    assert result.band_upper == 60.0
    assert result.as_dict()["probability"] == pytest.approx(0.421)


def test_band_bounds_and_colours_cover_every_category():
    for category in config.RISK_CATEGORIES:
        lower, upper = band_bounds(category)
        assert lower < upper
        assert risk_color(category).startswith("#")


def test_category_transition_wording():
    assert "unchanged" in category_transition("LOW RISK", "LOW RISK")
    assert "improved" in category_transition("HIGH RISK", "LOW RISK")
    assert "deteriorated" in category_transition("LOW RISK", "HIGH RISK")


def test_risk_categories_are_ordered():
    assert config.RISK_ORDER["LOW RISK"] < config.RISK_ORDER["MEDIUM RISK"]
    assert config.RISK_ORDER["MEDIUM RISK"] < config.RISK_ORDER["HIGH RISK"]


def test_describe_bands_lists_all_three():
    description = describe_bands()
    for category in config.RISK_CATEGORIES:
        assert category in description
