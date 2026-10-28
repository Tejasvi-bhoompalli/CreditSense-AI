"""What-if comparison: baseline preservation, recalculation, validation."""

from __future__ import annotations

import pytest

from src import config
from src.data_validation import InputValidationError
from src.what_if import (
    SUPPORTED_ADJUSTMENTS,
    apply_adjustments,
    compare_scenarios,
    illustrative_profile,
    repayment_schedule,
    typical_scenario_from_frame,
)


def test_apply_adjustments_does_not_mutate_baseline(valid_inputs):
    original = dict(valid_inputs)
    apply_adjustments(valid_inputs, {"credit_limit": 200_000.0})
    assert valid_inputs == original


def test_apply_adjustments_overrides_and_validates(valid_inputs):
    adjusted = apply_adjustments(valid_inputs, {"credit_limit": 200_000.0, "age": 44.0})
    assert adjusted["credit_limit"] == 200_000.0
    assert adjusted["age"] == 44.0
    assert adjusted["education"] == valid_inputs["education"]


def test_apply_adjustments_rejects_invalid_value(valid_inputs):
    with pytest.raises(InputValidationError):
        apply_adjustments(valid_inputs, {"age": 3.0}, strict=True)


def test_apply_adjustments_ignores_unknown_fields(valid_inputs):
    adjusted = apply_adjustments(valid_inputs, {"annual_income": 500_000}, strict=True)
    assert "annual_income" not in adjusted


def test_compare_reports_measured_change_and_preserves_baseline(predictor, valid_inputs):
    baseline = dict(valid_inputs)
    comparison = compare_scenarios(
        baseline,
        {"credit_limit": baseline["credit_limit"] * 2, "repayment_ratio_unused": 1},
        predictor,
    )

    assert baseline == valid_inputs  # untouched
    assert comparison.baseline_inputs["credit_limit"] == valid_inputs["credit_limit"]
    assert comparison.alternative_inputs["credit_limit"] == pytest.approx(
        valid_inputs["credit_limit"] * 2
    )
    assert comparison.probability_change_points == pytest.approx(
        (comparison.alternative_probability - comparison.baseline_probability) * 100
    )
    assert comparison.risk_score_change == pytest.approx(
        comparison.alternative_risk_score - comparison.baseline_risk_score
    )
    assert comparison.baseline_category in config.RISK_CATEGORIES
    assert comparison.alternative_category in config.RISK_CATEGORIES
    assert comparison.narrative
    assert comparison.caveats
    assert comparison.direction in {"increased", "decreased", "unchanged"}


def test_compare_lists_changed_features_only(predictor, valid_inputs):
    comparison = compare_scenarios(
        valid_inputs, {"credit_limit": valid_inputs["credit_limit"] + 50_000}, predictor
    )
    changed = {item.feature for item in comparison.changed_features}
    assert changed == {"credit_limit"}
    assert comparison.to_frame().shape[0] == 1


def test_compare_with_no_changes_is_identical(predictor, valid_inputs):
    comparison = compare_scenarios(valid_inputs, {}, predictor)
    assert comparison.probability_change_points == pytest.approx(0.0, abs=1e-9)
    assert comparison.category_transition.startswith("Category unchanged")


def test_every_numeric_input_is_adjustable():
    # Categorical fields are selected rather than slid, so they are excluded here.
    assert set(SUPPORTED_ADJUSTMENTS) == set(config.USER_INPUT_FEATURES) - set(
        config.CATEGORICAL_FEATURES
    )


def test_illustrative_profile_is_complete_and_derived_from_data(sample_frame):
    profile = illustrative_profile(sample_frame)
    assert set(profile) == set(config.USER_INPUT_FEATURES)
    assert profile["education"] in config.EDUCATION_CATEGORIES
    median = sample_frame["credit_limit"].median()
    assert profile["credit_limit"] == pytest.approx(float(median))


def test_typical_scenario_from_frame(sample_frame):
    scenario = typical_scenario_from_frame(sample_frame, index=3)
    assert set(scenario) == set(config.USER_INPUT_FEATURES)
    assert scenario["age"] == sample_frame.iloc[3]["age"]


def test_repayment_schedule_zero_interest():
    schedule = repayment_schedule(120_000, 0.0, 12)
    assert schedule["monthly_payment"] == pytest.approx(10_000.0)
    assert schedule["total_interest"] == pytest.approx(0.0)


def test_repayment_schedule_standard_formula():
    schedule = repayment_schedule(100_000, 12.0, 12)
    monthly_rate = 0.01
    factor = (1 + monthly_rate) ** 12
    expected = 100_000 * monthly_rate * factor / (factor - 1)
    assert schedule["monthly_payment"] == pytest.approx(expected)
    assert schedule["total_repayment"] == pytest.approx(expected * 12)
    assert schedule["total_interest"] == pytest.approx(expected * 12 - 100_000)


@pytest.mark.parametrize(
    "principal,rate,term",
    [(-1, 5, 12), (1000, -1, 12), (1000, 5, 0)],
)
def test_repayment_schedule_rejects_invalid_input(principal, rate, term):
    with pytest.raises(ValueError):
        repayment_schedule(principal, rate, term)
