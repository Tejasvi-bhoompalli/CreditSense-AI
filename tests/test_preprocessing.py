"""Derived features and the scikit-learn preprocessing pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src import config
from src.preprocessing import (
    DerivedFeatureAdder,
    PlattCalibrator,
    aggregate_contributions,
    build_design_matrix,
    build_pipeline,
    compute_derived_features,
    source_feature_of,
    transformed_feature_names,
)


def _row(**overrides) -> pd.DataFrame:
    base = {
        "credit_limit": 100_000.0,
        "age": 35.0,
        "education": "university",
        "repay_status_1": 0.0,
        "repay_status_2": 0.0,
        "repay_status_3": 0.0,
        "repay_status_4": 0.0,
        "repay_status_5": 0.0,
        "repay_status_6": 0.0,
        "bill_amount_1": 50_000.0,
        "bill_amount_2": 40_000.0,
        "bill_amount_3": 30_000.0,
        "bill_amount_4": 20_000.0,
        "bill_amount_5": 10_000.0,
        "bill_amount_6": 5_000.0,
        "payment_amount_1": 5_000.0,
        "payment_amount_2": 4_000.0,
        "payment_amount_3": 3_000.0,
        "payment_amount_4": 2_000.0,
        "payment_amount_5": 1_000.0,
        "payment_amount_6": 500.0,
    }
    base.update(overrides)
    return pd.DataFrame([base])


def test_derived_feature_formulas_are_exact():
    frame = _row()
    derived = compute_derived_features(frame).iloc[0]

    assert derived["avg_bill_amount"] == pytest.approx(
        np.mean([50_000, 40_000, 30_000, 20_000, 10_000, 5_000])
    )
    assert derived["avg_payment_amount"] == pytest.approx(
        np.mean([5_000, 4_000, 3_000, 2_000, 1_000, 500])
    )
    assert derived["credit_utilization"] == pytest.approx(50_000 / 100_000)
    total_bill = 50_000 + 40_000 + 30_000 + 20_000 + 10_000 + 5_000
    total_payment = 5_000 + 4_000 + 3_000 + 2_000 + 1_000 + 500
    assert derived["repayment_ratio"] == pytest.approx(total_payment / total_bill)
    assert derived["max_repayment_delay"] == 0
    assert derived["num_delayed_months"] == 0
    assert derived["num_severely_delayed_months"] == 0
    assert derived["bill_trend"] == pytest.approx(50_000 - 5_000)
    assert derived["payment_trend"] == pytest.approx(5_000 - 500)


def test_delay_counts_and_severity():
    frame = _row(
        repay_status_1=2.0,
        repay_status_2=3.0,
        repay_status_3=-1.0,
        repay_status_4=1.0,
        repay_status_5=0.0,
        repay_status_6=-2.0,
    )
    derived = compute_derived_features(frame).iloc[0]
    assert derived["max_repayment_delay"] == 3
    assert derived["num_delayed_months"] == 3
    assert derived["num_severely_delayed_months"] == 1


def test_utilization_and_repayment_ratio_are_clipped():
    frame = _row(bill_amount_1=9_000_000.0, bill_amount_6=1.0)
    derived = compute_derived_features(frame).iloc[0]
    assert derived["credit_utilization"] == pytest.approx(5.0)

    # No balances and no payments: the guarded denominator must yield 0 rather than
    # an error or an unbounded ratio.
    zero_bill = _row(
        **{
            **{f"bill_amount_{i}": 0.0 for i in range(1, 7)},
            **{f"payment_amount_{i}": 0.0 for i in range(1, 7)},
        }
    )
    zero_derived = compute_derived_features(zero_bill).iloc[0]
    assert zero_derived["repayment_ratio"] == 0.0


def test_derived_adder_returns_model_feature_order():
    adder = DerivedFeatureAdder()
    enriched = adder.transform(_row())
    assert list(enriched.columns) == list(config.MODEL_FEATURES)
    assert adder.get_feature_names_out().tolist() == list(config.MODEL_FEATURES)


def test_derived_adder_tolerates_missing_columns():
    frame = pd.DataFrame({"credit_limit": [50_000.0], "age": [40.0]})
    enriched = DerivedFeatureAdder().transform(frame)
    assert list(enriched.columns) == list(config.MODEL_FEATURES)
    assert enriched["repay_status_1"].isna().all()


@pytest.mark.parametrize("scale_numeric", [True, False])
def test_pipeline_produces_probabilities_in_range(sample_frame, scale_numeric):
    pipeline = build_pipeline(
        LogisticRegression(max_iter=500, random_state=0), scale_numeric=scale_numeric
    )
    features = sample_frame[list(config.USER_INPUT_FEATURES)]
    target = sample_frame[config.TARGET_COLUMN]
    pipeline.fit(features, target)
    probabilities = pipeline.predict_proba(features.head(25))[:, 1]
    assert probabilities.shape == (25,)
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_tree_pipeline_accepts_dense_matrix(sample_frame):
    """HistGradientBoosting requires dense input, so encoding must stay dense."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    pipeline = build_pipeline(
        HistGradientBoostingClassifier(max_iter=20, random_state=0), scale_numeric=False
    )
    pipeline.fit(
        sample_frame[list(config.USER_INPUT_FEATURES)],
        sample_frame[config.TARGET_COLUMN],
    )
    assert pipeline.predict_proba(
        sample_frame[list(config.USER_INPUT_FEATURES)].head(5)
    ).shape == (5, 2)


def test_transformed_feature_names_are_prefixed(sample_frame):
    pipeline = build_pipeline(RandomForestClassifier(n_estimators=5, random_state=0))
    pipeline.fit(
        sample_frame[list(config.USER_INPUT_FEATURES)],
        sample_frame[config.TARGET_COLUMN],
    )
    names = transformed_feature_names(pipeline)
    assert any(name.startswith("num__credit_utilization") for name in names)
    assert any(name.startswith("cat__education_") for name in names)
    assert len(names) == len(config.NUMERIC_FEATURES) + len(config.EDUCATION_CATEGORIES)


def test_source_feature_mapping_and_aggregation():
    assert source_feature_of("num__credit_limit") == "credit_limit"
    assert source_feature_of("cat__education_university") == "education"
    assert source_feature_of("avg_payment_amount") == "avg_payment_amount"

    totals = aggregate_contributions(
        ["num__age", "cat__education_university", "cat__education_other"], [0.5, 0.2, 0.1]
    )
    assert totals["age"] == pytest.approx(0.5)
    assert totals["education"] == pytest.approx(0.3)


def test_build_design_matrix_columns(sample_frame):
    design = build_design_matrix(sample_frame)
    assert list(design.columns) == list(config.MODEL_FEATURES)
    assert len(design) == len(sample_frame)


def test_platt_calibrator_is_monotone(sample_frame):
    calibrator = PlattCalibrator()
    probabilities = np.linspace(0.01, 0.99, 50)
    target = (probabilities > 0.5).astype(int)
    calibrator.fit(probabilities, target)
    calibrated = calibrator.calibrate(probabilities)
    # Monotone non-decreasing is the guarantee. With a separable calibration set the
    # fitted slope can saturate the sigmoid, so strict increase is not promised.
    assert np.all(np.diff(calibrated) >= 0)
    assert np.any(np.diff(calibrated) > 0)
    assert np.all((calibrated >= 0) & (calibrated <= 1))
    assert calibrator.as_dict()["method"] == "platt_sigmoid"


def test_platt_calibrator_handles_extreme_probabilities():
    calibrator = PlattCalibrator().fit(np.array([0.0, 1.0]), np.array([0, 1]))
    output = calibrator.calibrate(np.array([0.0, 1.0]))
    assert np.all(np.isfinite(output))
