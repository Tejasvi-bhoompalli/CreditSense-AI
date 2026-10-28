"""Explanation shape, labels, model consistency and graceful fallback."""

from __future__ import annotations

import numpy as np
import pytest

from src import config
from src.data_validation import build_input_frame
from src.explainability import (
    SHAP_CAVEAT,
    CreditExplainer,
    narrate_global,
    narrate_local,
)


@pytest.fixture(scope="module")
def explainer(predictor, sample_frame):
    background = sample_frame[list(config.USER_INPUT_FEATURES)].head(120)
    return CreditExplainer(predictor.bundle["base_pipeline"], background=background)


def test_explainer_uses_the_serving_estimator(explainer, predictor):
    assert explainer.estimator is predictor.bundle["base_pipeline"].named_steps["model"]
    assert explainer.model_name


def test_local_explanation_covers_every_model_feature(explainer, valid_inputs):
    """Every feature the model consumes is explained, including derived ones."""
    explanation = explainer.explain_local(build_input_frame(valid_inputs))
    asserted = {item.feature for item in explanation.contributions}
    assert asserted == set(config.MODEL_FEATURES)
    assert asserted >= set(config.USER_INPUT_FEATURES)
    assert explanation.method in {
        "shap_tree",
        "shap_linear",
        "shap_generic",
        "local_occlusion_fallback",
    }
    assert 0.0 <= explanation.probability <= 1.0


def test_shap_contributions_are_additive(explainer, valid_inputs):
    explanation = explainer.explain_local(build_input_frame(valid_inputs))
    if explanation.method == "local_occlusion_fallback":
        pytest.skip("additivity is not claimed for the fallback method")
    total = sum(item.contribution for item in explanation.contributions)
    assert total + explanation.base_value == pytest.approx(explanation.output_value, abs=1e-6)


def test_contributions_carry_labels_values_and_direction(explainer, valid_inputs):
    explanation = explainer.explain_local(build_input_frame(valid_inputs))
    for item in explanation.contributions:
        assert item.label and item.label != item.feature
        assert item.display_value
        assert item.direction in {"increases_risk", "decreases_risk"}
        assert (item.contribution > 0) == (item.direction == "increases_risk")


def test_ranking_orders_by_magnitude(explainer, valid_inputs):
    explanation = explainer.explain_local(build_input_frame(valid_inputs))
    ranked = explanation.ranked()
    magnitudes = [abs(item.contribution) for item in ranked]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_explanation_carries_the_caveat(explainer, valid_inputs):
    explanation = explainer.explain_local(build_input_frame(valid_inputs))
    assert any(SHAP_CAVEAT in note for note in explanation.notes)


def test_fallback_is_labelled_when_shap_is_unavailable(explainer, valid_inputs):
    """A broken explainer must degrade to a labelled fallback, never crash."""
    original = explainer.explainer
    original_method = explainer.method
    try:
        explainer.explainer = None
        explainer.method = "unavailable"
        explanation = explainer.explain_local(build_input_frame(valid_inputs))
        assert explanation.method == "local_occlusion_fallback"
        assert "not SHAP" in explanation.method_label
        assert len(explanation.contributions) == len(config.USER_INPUT_FEATURES)
        assert all(np.isfinite(item.contribution) for item in explanation.contributions)
        assert explainer.global_importance(build_input_frame(valid_inputs)) is None
    finally:
        explainer.explainer = original
        explainer.method = original_method


def test_global_importance_ranking(explainer, sample_frame):
    sample = sample_frame[list(config.USER_INPUT_FEATURES)].head(200)
    importance = explainer.global_importance(sample, max_rows=200)
    if importance is None:
        pytest.skip("no SHAP explainer available in this environment")
    assert importance.rows_used <= 200
    assert importance.ranking
    shares = [row["share_of_total"] for row in importance.ranking]
    assert shares == sorted(shares, reverse=True)
    assert sum(shares) == pytest.approx(1.0, abs=1e-6)
    assert all(row["feature"] in config.MODEL_FEATURES for row in importance.ranking)


def test_narration_uses_measured_contributions(explainer, valid_inputs):
    explanation = explainer.explain_local(build_input_frame(valid_inputs))
    sentences = narrate_local(explanation)
    assert sentences
    assert any("risk" in sentence.lower() for sentence in sentences)


def test_global_narration(explainer, sample_frame):
    importance = explainer.global_importance(
        sample_frame[list(config.USER_INPUT_FEATURES)].head(150), max_rows=150
    )
    if importance is None:
        pytest.skip("no SHAP explainer available in this environment")
    sentences = narrate_global(importance)
    assert sentences
    assert "contribution" in sentences[0].lower()
