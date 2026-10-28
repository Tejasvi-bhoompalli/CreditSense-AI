"""Data loading, adaptation, detection, profiling and validation."""

from __future__ import annotations

import pandas as pd
import pytest

from src import config
from src.data_loader import (
    DatasetSchemaError,
    _clean_types,
    canonicalize_frame,
    describe_missing_features,
    detect_target_column,
    generate_synthetic_dataset,
    normalize_column_name,
    profile_frame,
)
from src.data_validation import validate_dataset, validate_user_inputs


def test_normalize_column_name():
    assert normalize_column_name("default.payment.next.month") == "default_payment_next_month"
    assert normalize_column_name("PAY_0") == "pay_0"
    assert normalize_column_name("  LIMIT__BAL ") == "limit_bal"


def test_detect_target_column_on_source_schema():
    frame = pd.DataFrame(
        {
            "ID": [1, 2, 3],
            "LIMIT_BAL": [10_000, 20_000, 30_000],
            "default.payment.next.month": [0, 1, 0],
        }
    )
    assert detect_target_column(frame) == "default.payment.next.month"


def test_detect_target_column_refuses_approval_style_target():
    """A loan-approval column must never be mistaken for a default label."""
    frame = pd.DataFrame({"person_income": [1, 2, 3], "loan_status": [0, 1, 0]})
    with pytest.raises(DatasetSchemaError):
        detect_target_column(frame)


def test_detect_target_column_rejects_non_binary():
    frame = pd.DataFrame({"default": [0, 1, 2]})
    with pytest.raises(DatasetSchemaError):
        detect_target_column(frame)


def test_canonicalize_maps_verified_columns():
    frame = pd.DataFrame(
        {
            "ID": [1],
            "LIMIT_BAL": [50_000],
            "SEX": [1.0],
            "EDUCATION": [2.0],
            "MARRIAGE": [1.0],
            "AGE": [30],
            **{f"PAY_{index}": [0] for index in [0, 2, 3, 4, 5, 6]},
            **{f"BILL_AMT{index}": [1000.0] for index in range(1, 7)},
            **{f"PAY_AMT{index}": [200.0] for index in range(1, 7)},
            "default.payment.next.month": [0],
        }
    )
    frame = frame.rename(columns={c: normalize_column_name(c) for c in frame.columns})
    canonical, mapping, unmapped = canonicalize_frame(frame)
    assert "credit_limit" in canonical.columns
    assert config.TARGET_COLUMN in canonical.columns
    assert "repay_status_1" in canonical.columns
    assert "bill_amount_6" in canonical.columns
    assert "record_id" not in canonical.columns  # identifier dropped
    assert mapping["limit_bal"] == "credit_limit"
    assert unmapped == []


def test_canonicalize_raises_when_columns_missing():
    frame = pd.DataFrame({"default": [0, 1], "age": [30, 40]})
    with pytest.raises(DatasetSchemaError):
        canonicalize_frame(frame)


def test_synthetic_dataset_is_reproducible():
    first = generate_synthetic_dataset(rows=400, seed=99)
    second = generate_synthetic_dataset(rows=400, seed=99)
    different = generate_synthetic_dataset(rows=400, seed=100)
    pd.testing.assert_frame_equal(first, second)
    assert not first[config.TARGET_COLUMN].equals(different[config.TARGET_COLUMN])


def test_synthetic_dataset_has_canonical_schema():
    frame = generate_synthetic_dataset(rows=300, seed=7)
    for column in config.USER_INPUT_FEATURES:
        assert column in frame.columns
    assert config.TARGET_COLUMN in frame.columns
    assert set(frame[config.TARGET_COLUMN].unique()) <= {0, 1}
    assert frame["credit_limit"].min() >= 10_000
    assert frame["age"].min() >= 18


def test_profile_frame_reports_structure(sample_frame):
    profile = profile_frame(sample_frame)
    assert profile["rows"] == len(sample_frame)
    assert profile["missing_total"] == 0
    assert set(profile["target_counts"]).issubset({0, 1})
    assert 0.0 < profile["target_positive_rate"] < 1.0
    assert profile["class_imbalance_ratio"] is not None


def test_validate_dataset_flags_missing_values(sample_frame):
    frame = sample_frame.copy()
    frame.loc[frame.index[:5], "credit_limit"] = None
    report = validate_dataset(frame)
    codes = {issue.code for issue in report.issues}
    assert "missing_values" in codes
    assert report.context["rows"] == len(frame)


def test_validate_dataset_rejects_non_binary_target(sample_frame):
    frame = sample_frame.copy()
    frame[config.TARGET_COLUMN] = 2
    report = validate_dataset(frame)
    assert not report.is_valid
    assert any(issue.code == "target_not_binary" for issue in report.errors)


def test_validate_dataset_rejects_empty():
    report = validate_dataset(pd.DataFrame())
    assert not report.is_valid
    assert any(issue.code == "dataset_empty" for issue in report.errors)


def test_validate_dataset_flags_non_positive_credit_limit(sample_frame):
    frame = sample_frame.copy()
    frame.loc[frame.index[0], "credit_limit"] = 0
    report = validate_dataset(frame)
    assert any(issue.code == "non_positive_credit_limit" for issue in report.errors)


def test_validate_dataset_notes_synthetic_provenance(small_dataset):
    report = validate_dataset(small_dataset.frame, small_dataset.provenance)
    assert any(issue.code == "synthetic_data" for issue in report.issues)


def test_describe_missing_features_lists_absent_canonical_fields(sample_frame):
    missing = describe_missing_features(sample_frame)
    assert "annual_income" in missing
    assert "credit_score" in missing
    assert "loan_purpose" in missing


def test_real_dataset_adapts_and_validates(real_dataset):
    frame = real_dataset.frame
    assert config.TARGET_COLUMN in frame.columns
    for column in config.USER_INPUT_FEATURES:
        assert column in frame.columns
    assert frame[config.TARGET_COLUMN].nunique() == 2
    report = validate_dataset(frame, real_dataset.provenance)
    assert report.is_valid
    assert report.context["rows"] > 1000


def test_user_input_validation_accepts_and_coerces(valid_inputs):
    cleaned, report = validate_user_inputs(valid_inputs, strict=True)
    assert report.is_valid
    assert set(cleaned) == set(config.USER_INPUT_FEATURES)


def test_user_input_validation_rejects_out_of_range(valid_inputs):
    broken = dict(valid_inputs)
    broken["age"] = 250
    from src.data_validation import InputValidationError

    with pytest.raises(InputValidationError):
        validate_user_inputs(broken, strict=True)


def test_user_input_validation_rejects_unknown_category(valid_inputs):
    from src.data_validation import InputValidationError

    broken = dict(valid_inputs)
    broken["education"] = "doctorate"
    with pytest.raises(InputValidationError):
        validate_user_inputs(broken, strict=True)


def test_user_input_validation_requires_every_field(valid_inputs):
    from src.data_validation import InputValidationError

    broken = dict(valid_inputs)
    broken.pop("bill_amount_3")
    with pytest.raises(InputValidationError):
        validate_user_inputs(broken, strict=True)


def test_synthetic_dataset_keeps_every_education_category():
    """Regression: textual education values were once collapsed into one category."""
    frame = generate_synthetic_dataset(rows=800, seed=5)
    categories = set(frame["education"].dropna().unique())
    assert categories <= set(config.EDUCATION_CATEGORIES)
    assert len(categories) > 1
    # Every row carries a usable category (nunique is the distinct count, 4).
    assert frame["education"].notna().all()
    assert set(frame["education"].value_counts().index) == categories


def test_education_cleaning_handles_numeric_codes_and_text_labels():
    numeric = _clean_types(pd.DataFrame({"education": [1.0, 2.0, 3.0, 4.0, 5.0, 0.0]}))
    assert set(numeric["education"]) <= set(config.EDUCATION_CATEGORIES)
    assert "graduate_school" in set(numeric["education"])

    textual = _clean_types(
        pd.DataFrame({"education": ["Graduate School", "High-School", "nonsense", " Other "]})
    )
    assert set(textual["education"]) <= set(config.EDUCATION_CATEGORIES)
    assert "nonsense" not in set(textual["education"])
