"""Predictor loading, scoring, validation and artifact-mismatch handling."""

from __future__ import annotations

import pandas as pd
import pytest

from src import config
from src.data_validation import InputValidationError
from src.predictor import CreditRiskPredictor, ModelNotReadyError, model_status


def test_predictor_loads_bundle(predictor):
    assert predictor.is_ready
    assert predictor.model_name == "Test Random Forest"
    assert predictor.is_calibrated


def test_predict_one_returns_bounded_probability(predictor, valid_inputs):
    result = predictor.predict_one(valid_inputs, explain=False)
    assert 0.0 <= result.probability <= 1.0
    assert 0.0 <= result.probability_uncalibrated <= 1.0
    assert 0.0 <= result.risk.risk_score <= 100.0
    assert result.risk.risk_category in config.RISK_CATEGORIES
    assert result.model_name == "Test Random Forest"
    assert set(result.derived) == set(config.DERIVED_FEATURES)


def test_risk_score_matches_probability(predictor, valid_inputs):
    result = predictor.predict_one(valid_inputs, explain=False)
    assert result.risk.risk_score == pytest.approx(round(result.probability * 100, 1), abs=1e-9)


def test_predict_frame_returns_one_result_per_row(predictor, sample_frame):
    frame = sample_frame[list(config.USER_INPUT_FEATURES)].head(20)
    results = predictor.predict_frame(frame)
    assert len(results) == 20
    assert all(0.0 <= item.probability <= 1.0 for item in results)


def test_feature_order_does_not_change_prediction(predictor, valid_inputs):
    reordered = pd.DataFrame(
        [valid_inputs], columns=list(reversed(config.USER_INPUT_FEATURES))
    )
    first = predictor.predict_frame(reordered)[0].probability
    second = predictor.predict_one(valid_inputs, explain=False).probability
    assert first == pytest.approx(second)


def test_predict_many_is_deterministic(predictor, valid_inputs):
    first = predictor.predict_one(valid_inputs, explain=False).probability
    second = predictor.predict_one(valid_inputs, explain=False).probability
    assert first == pytest.approx(second)


def test_out_of_range_input_raises(predictor, valid_inputs):
    broken = dict(valid_inputs)
    broken["credit_limit"] = 5.0
    with pytest.raises(InputValidationError):
        predictor.predict_one(broken, explain=True, strict=True)


def test_missing_input_raises(predictor, valid_inputs):
    broken = dict(valid_inputs)
    broken.pop("age")
    with pytest.raises(InputValidationError):
        predictor.predict_one(broken, strict=True)


def test_frame_missing_columns_raises(predictor, sample_frame):
    with pytest.raises(InputValidationError):
        predictor.predict_frame(pd.DataFrame({"credit_limit": [10_000.0]}))


def test_missing_bundle_raises_instructive_error(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODEL_BUNDLE_PATH", tmp_path / "absent.joblib")
    predictor = CreditRiskPredictor()
    assert not predictor.is_ready
    with pytest.raises(ModelNotReadyError) as error:
        predictor.load()
    assert "train_model" in str(error.value)


def test_schema_mismatch_is_detected(tmp_path, monkeypatch, fast_bundle_path):
    import joblib

    bundle = joblib.load(fast_bundle_path)
    bundle["feature_names"] = ["wrong_feature"]
    broken_path = tmp_path / "broken_bundle.joblib"
    joblib.dump(bundle, broken_path)

    predictor = CreditRiskPredictor(bundle_path=broken_path)
    with pytest.raises(ModelNotReadyError) as error:
        predictor.load()
    assert "schema" in str(error.value).lower()


def test_model_status_reports_absence(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODEL_BUNDLE_PATH", tmp_path / "absent.joblib")
    status = model_status()
    assert status["ready"] is False
    assert "train_model" in status["error"]


def test_explainability_attached_on_request(predictor, valid_inputs):
    result = predictor.predict_one(valid_inputs, explain=True)
    assert result.explanation is not None
    explained = {item.feature for item in result.explanation.contributions}
    # The explanation covers everything the model consumed, derived features included.
    assert explained == set(config.MODEL_FEATURES)
