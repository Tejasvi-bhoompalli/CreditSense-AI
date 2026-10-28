"""Shared pytest fixtures.

The suite runs against a small synthetic dataset so it is fast and fully
deterministic. Where a test genuinely needs the shipped research dataset, it uses
the ``real_dataset`` fixture, which skips cleanly when the file is absent.
"""

from __future__ import annotations

import joblib
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src import config
from src.data_loader import (
    CreditDataset,
    DatasetProvenance,
    generate_synthetic_dataset,
    profile_frame,
)
from src.preprocessing import PlattCalibrator, build_pipeline


@pytest.fixture(scope="session")
def sample_frame():
    """Deterministic synthetic frame with the canonical schema."""
    return generate_synthetic_dataset(rows=1500, seed=1234)


@pytest.fixture(scope="session")
def small_dataset(sample_frame):
    provenance = DatasetProvenance(
        path="<generated>",
        filename="synthetic_test_dataset.csv",
        sha256="synthetic-test",
        rows=int(sample_frame.shape[0]),
        columns=int(sample_frame.shape[1]),
        is_synthetic=True,
        notes=[config.SYNTHETIC_DISCLOSURE],
    )
    return CreditDataset(
        frame=sample_frame,
        provenance=provenance,
        mapping={},
        unmapped_columns=[],
        profile=profile_frame(sample_frame),
    )


@pytest.fixture(scope="session")
def valid_inputs(sample_frame):
    row = sample_frame.iloc[0]
    return {
        feature: (
            str(row[feature])
            if feature in config.CATEGORICAL_FEATURES
            else float(row[feature])
        )
        for feature in config.USER_INPUT_FEATURES
    }


@pytest.fixture(scope="session")
def fast_bundle_path(tmp_path_factory, sample_frame):
    """A small but genuinely fitted bundle, written to a temporary directory."""
    directory = tmp_path_factory.mktemp("creditsense-bundle")
    path = directory / "model_bundle.joblib"

    features = sample_frame[list(config.USER_INPUT_FEATURES)]
    target = sample_frame[config.TARGET_COLUMN]
    x_train, x_val, y_train, y_val = train_test_split(
        features, target, test_size=0.4, random_state=0, stratify=target
    )

    pipeline = build_pipeline(
        RandomForestClassifier(
            n_estimators=40, min_samples_leaf=4, random_state=0, n_jobs=1
        ),
        scale_numeric=False,
    )
    pipeline.fit(x_train, y_train)
    calibrator = PlattCalibrator().fit(pipeline.predict_proba(x_val)[:, 1], y_val)

    bundle = {
        "base_pipeline": pipeline,
        "calibrator": calibrator,
        "feature_names": list(config.USER_INPUT_FEATURES),
        "model_features": list(config.MODEL_FEATURES),
        "model_name": "Test Random Forest",
        "model_version": config.MODEL_VERSION,
        "schema_version": config.SCHEMA_VERSION,
        "trained_at": "2026-01-01T00:00:00+00:00",
        "random_seed": 0,
        "target_column": config.TARGET_COLUMN,
        "dataset_fingerprint": "synthetic-test",
        "dataset_filename": "synthetic_test_dataset.csv",
        "is_synthetic_dataset": True,
        "test_metrics": {"roc_auc": 0.70, "n": int(len(y_val))},
        "calibration_info": {"method": "platt_sigmoid"},
    }
    joblib.dump(bundle, path, compress=("zlib", 3))
    return path


@pytest.fixture(scope="session")
def predictor(fast_bundle_path):
    from src.predictor import CreditRiskPredictor

    return CreditRiskPredictor(bundle_path=fast_bundle_path).load()


@pytest.fixture(scope="session")
def trained_result(small_dataset):
    """Full training pipeline executed once for the session on synthetic data."""
    from src.train_model import train_and_save

    return train_and_save(dataset=small_dataset, save=False, verbose=False)


@pytest.fixture(scope="session")
def real_dataset():
    """The shipped research dataset, skipped if it is not available."""
    from src.data_loader import DatasetError, discover_dataset_files, load_dataset

    if not discover_dataset_files():
        pytest.skip("no dataset file present in data/")
    try:
        return load_dataset()
    except DatasetError as exc:  # pragma: no cover - depends on local files
        pytest.skip(f"dataset could not be loaded: {exc}")
