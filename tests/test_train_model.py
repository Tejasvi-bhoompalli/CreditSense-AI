"""Training pipeline integrity: split discipline, selection, artifacts."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data_loader import CreditDataset
from src.train_model import (
    TrainingError,
    build_candidates,
    evaluate_probabilities,
    split_dataset,
    train_and_save,
)


def test_split_uses_the_configured_fractions_and_is_stratified(small_dataset):
    train, validation, test = split_dataset(small_dataset)
    total = len(small_dataset.frame)
    assert len(train) + len(validation) + len(test) == total
    assert len(train) / total == pytest.approx(config.SPLIT_FRACTIONS[0], abs=0.02)
    for frame in (train, validation, test):
        rate = frame[config.TARGET_COLUMN].mean()
        assert rate == pytest.approx(
            small_dataset.frame[config.TARGET_COLUMN].mean(), abs=0.05
        )


def test_splits_are_disjoint(sample_frame):
    frame = sample_frame.copy()
    frame["marker"] = np.arange(len(frame))
    dataset = CreditDataset(
        frame=frame,
        provenance=None,  # type: ignore[arg-type]
        mapping={},
        unmapped_columns=[],
        profile={},
    )
    train, validation, test = split_dataset(dataset)
    markers = [
        set(train["marker"]),
        set(validation["marker"]),
        set(test["marker"]),
    ]
    assert markers[0].isdisjoint(markers[1])
    assert markers[1].isdisjoint(markers[2])
    assert markers[0].isdisjoint(markers[2])
    assert len(markers[0] | markers[1] | markers[2]) == len(frame)


def test_split_is_reproducible(small_dataset):
    first = split_dataset(small_dataset)
    second = split_dataset(small_dataset)
    pd.testing.assert_frame_equal(first[0], second[0])
    pd.testing.assert_frame_equal(first[2], second[2])


def test_candidates_cover_the_required_families(seed=config.RANDOM_SEED):
    names = " ".join(candidate.name for candidate in build_candidates(seed=seed))
    assert "Logistic Regression" in names
    assert "Random Forest" in names
    assert "HistGradientBoosting" in names


def test_evaluate_probabilities_bounds_and_counts():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.6, 0.4, 0.9])
    metrics = evaluate_probabilities(y_true, y_prob, threshold=0.5)
    assert metrics["n"] == 4
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["brier_score"] <= 1.0
    assert metrics["false_positives"] + metrics["false_negatives"] == 2


def test_metrics_do_not_shuffle_confusion_counts():
    y_true = np.array([0, 1])
    y_prob = np.array([0.1, 0.9])
    metrics = evaluate_probabilities(y_true, y_prob)
    assert metrics["true_negatives"] == 1
    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 0


def test_training_refuses_invalid_data(small_dataset):
    frame = small_dataset.frame.copy()
    frame[config.TARGET_COLUMN] = 1
    broken = CreditDataset(
        frame=frame,
        provenance=small_dataset.provenance,
        mapping={},
        unmapped_columns=[],
        profile={},
    )
    with pytest.raises(TrainingError):
        train_and_save(dataset=broken, save=False, verbose=False, rows=400)


def test_selection_uses_validation_only(trained_result):
    metrics = trained_result["metrics"]
    assert metrics["test_split_used_for_tuning"] is False
    assert metrics["test_split_metrics_are_measured"] is True
    for result in metrics["candidates"].values():
        assert "validation" in result
        assert "test" not in result  # the test split played no part in selection


def test_selected_model_is_best_on_validation(trained_result):
    candidates = trained_result["metrics"]["candidates"]
    best = max(
        candidates.items(),
        key=lambda item: item[1]["validation"]["roc_auc"],
    )[0]
    assert trained_result["selected_model"] == best


def test_test_metrics_are_finite_and_measured(trained_result):
    test = trained_result["metrics"]["selected"]["test"]
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier_score"):
        assert np.isfinite(test[key]), key
        assert 0.0 <= test[key] <= 1.0
    assert test["n"] > 0
    assert test["true_positives"] + test["false_negatives"] == test["positives"]


def test_calibration_and_overfitting_are_recorded(trained_result):
    metrics = trained_result["metrics"]
    quality = metrics["calibration_quality"]
    assert quality["brier_score_calibrated"] is not None
    assert np.isfinite(quality["expected_calibration_error_calibrated"])
    overfitting = metrics["overfitting"]
    assert set(overfitting) >= {"train_roc_auc", "validation_roc_auc", "test_roc_auc", "warning"}


def test_metadata_records_disclosures_and_provenance(trained_result):
    metadata = trained_result["metadata"]
    assert metadata["selected_model"]
    assert metadata["random_seed"] == config.RANDOM_SEED
    assert "Taiwan" in metadata["geographic_disclosure"]
    assert metadata["dataset"]["is_synthetic"] is True
    assert config.SYNTHETIC_DISCLOSURE in metadata["synthetic_disclosure"]
    assert metadata["split_sizes"]["train"] > metadata["split_sizes"]["test"]


def test_schema_lists_features_and_exclusions(trained_result):
    schema = trained_result["schema"]
    assert schema["model_features"] == list(config.MODEL_FEATURES)
    assert schema["user_input_features"] == list(config.USER_INPUT_FEATURES)
    assert schema["protected_excluded_features"] == list(config.PROTECTED_EXCLUDED_FEATURES)
    assert "sex" not in schema["model_features"]
    assert "marriage" not in schema["model_features"]
    assert len(schema["derived_formulas"]) == len(config.DERIVED_FEATURES)


def test_permutation_importance_is_measured(trained_result):
    rows = trained_result["permutation_importance"]
    assert rows
    assert [row["feature"] for row in rows][:1] in [
        [config.USER_INPUT_FEATURES[index]] for index in range(len(config.USER_INPUT_FEATURES))
    ] or rows[0]["feature"] in config.USER_INPUT_FEATURES
    assert all(np.isfinite(row["mean_importance"]) for row in rows)


def test_artifacts_are_written_and_reloadable(tmp_path, monkeypatch, small_dataset):
    monkeypatch.setattr(config, "MODEL_BUNDLE_PATH", tmp_path / "model_bundle.joblib")
    monkeypatch.setattr(config, "FEATURE_SCHEMA_PATH", tmp_path / "feature_schema.json")
    monkeypatch.setattr(config, "TRAINING_METADATA_PATH", tmp_path / "training_metadata.json")
    monkeypatch.setattr(config, "EVALUATION_METRICS_PATH", tmp_path / "evaluation_metrics.json")
    monkeypatch.setattr(config, "PERMUTATION_IMPORTANCE_PATH", tmp_path / "permutation.json")
    monkeypatch.setattr(config, "GLOBAL_SHAP_PATH", tmp_path / "global_shap.json")
    monkeypatch.setattr(config, "DATASET_PROFILE_PATH", tmp_path / "dataset_profile.json")

    train_and_save(dataset=small_dataset, rows=700, save=True, verbose=False)

    for path in (
        config.MODEL_BUNDLE_PATH,
        config.FEATURE_SCHEMA_PATH,
        config.TRAINING_METADATA_PATH,
        config.EVALUATION_METRICS_PATH,
        config.PERMUTATION_IMPORTANCE_PATH,
        config.GLOBAL_SHAP_PATH,
        config.DATASET_PROFILE_PATH,
    ):
        assert path.exists(), path

    from src.predictor import CreditRiskPredictor

    predictor = CreditRiskPredictor(bundle_path=config.MODEL_BUNDLE_PATH).load()
    sample = small_dataset.frame[list(config.USER_INPUT_FEATURES)].head(5)
    probabilities = [item.probability for item in predictor.predict_frame(sample)]
    assert len(probabilities) == 5
    assert all(0.0 <= value <= 1.0 for value in probabilities)

    metrics = json.loads(config.EVALUATION_METRICS_PATH.read_text())
    assert metrics["selected"]["test"]["n"] > 0
    assert metrics["test_split_used_for_tuning"] is False

    schema = json.loads(config.FEATURE_SCHEMA_PATH.read_text())
    assert schema["model_features"] == list(config.MODEL_FEATURES)
