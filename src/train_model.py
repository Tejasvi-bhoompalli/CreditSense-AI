"""Reproducible training, model comparison, calibration and artifact generation.

Run with::

    python -m src.train_model

Methodological contract enforced by this module:

1. The dataset is split once into train / validation / test (60 / 20 / 20,
   stratified, fixed seed).
2. Candidate models are fitted on the **training split only** and compared on
   the **validation split only**. The test split plays no part in model
   selection, so describing it as untouched is accurate.
3. Probability calibration (Platt/sigmoid) is fitted on the **validation split**
   using the already-selected base model. It is never fitted on the test split.
4. The test split is used once, at the end, to report the metrics that the
   application displays. Nothing is tuned on those numbers.
5. Metric fields are written only after the model actually runs; nothing here
   writes a placeholder or an estimated value.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from src import config
from src.data_loader import CreditDataset, dataset_fingerprint, load_dataset
from src.data_validation import validate_dataset
from src.explainability import CreditExplainer, SHAP_AVAILABLE
from src.preprocessing import PlattCalibrator, build_pipeline
from src.utils import feature_label, get_logger, save_json, utc_now_iso

logger = get_logger(__name__)


class TrainingError(RuntimeError):
    """Raised when training cannot produce trustworthy artifacts."""


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def evaluate_probabilities(
    y_true, y_prob, threshold: float = 0.5
) -> Dict[str, Any]:
    """Measured classification metrics for a probability vector."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics: Dict[str, Any] = {
        "n": int(y_true.size),
        "positives": int(y_true.sum()),
        "negatives": int((y_true == 0).sum()),
        "positive_rate": float(y_true.mean()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, np.clip(y_prob, 1e-9, 1 - 1e-9), labels=[0, 1])),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "mean_predicted_probability": float(y_prob.mean()),
    }
    metrics["precision_recall_gap"] = metrics["precision"] - metrics["recall"]
    return metrics


def _roc_points(y_true, y_prob, max_points: int = 120) -> List[Dict[str, float]]:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return _downsample_curve(
        {"fpr": fpr, "tpr": tpr, "threshold": thresholds}, max_points
    )


def _pr_points(y_true, y_prob, max_points: int = 120) -> List[Dict[str, float]]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    precision, recall = precision[:-1], recall[:-1]
    return _downsample_curve(
        {"recall": recall, "precision": precision, "threshold": thresholds}, max_points
    )


def _downsample_curve(arrays: Dict[str, np.ndarray], max_points: int) -> List[Dict[str, float]]:
    length = len(next(iter(arrays.values())))
    if length > max_points:
        index = np.unique(np.linspace(0, length - 1, max_points).astype(int))
    else:
        index = np.arange(length)
    keys = list(arrays.keys())
    return [
        {key: _finite_float(arrays[key][i]) for key in keys} for i in index
    ]


def _finite_float(value) -> float:
    number = float(value)
    if not np.isfinite(number):
        return 0.0
    return number


def calibration_curve_points(
    y_true, y_prob, n_bins: int = 10
) -> List[Dict[str, float]]:
    """Reliability curve using equal-count (quantile) bins."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    order = np.argsort(y_prob)
    chunks = np.array_split(order, n_bins)
    points: List[Dict[str, float]] = []
    for chunk in chunks:
        if chunk.size == 0:
            continue
        points.append(
            {
                "mean_predicted_probability": float(y_prob[chunk].mean()),
                "observed_default_rate": float(y_true[chunk].mean()),
                "count": int(chunk.size),
            }
        )
    return points


def expected_calibration_error(points: List[Dict[str, float]], n_total: int) -> float:
    if not points or n_total <= 0:
        return float("nan")
    return float(
        sum(
            abs(point["observed_default_rate"] - point["mean_predicted_probability"])
            * (point["count"] / n_total)
            for point in points
        )
    )


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    name: str
    estimator: Any
    scale_numeric: bool


def build_candidates(seed: int = config.RANDOM_SEED) -> List[Candidate]:
    """The three model families required by the brief, plus weighted variants.

    Class weighting is included as a *candidate* rather than applied
    unconditionally: it changes the precision/recall operating point noticeably
    and should be justified by the validation comparison rather than assumed.
    """
    return [
        Candidate(
            "Logistic Regression",
            LogisticRegression(max_iter=3000, random_state=seed, solver="lbfgs"),
            scale_numeric=True,
        ),
        Candidate(
            "Logistic Regression (class-weighted)",
            LogisticRegression(
                max_iter=3000, random_state=seed, solver="lbfgs", class_weight="balanced"
            ),
            scale_numeric=True,
        ),
        Candidate(
            "Random Forest",
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=3,
                max_features="sqrt",
                n_jobs=-1,
                random_state=seed,
            ),
            scale_numeric=False,
        ),
        Candidate(
            "Random Forest (class-weighted)",
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=3,
                max_features="sqrt",
                n_jobs=-1,
                random_state=seed,
                class_weight="balanced_subsample",
            ),
            scale_numeric=False,
        ),
        Candidate(
            "HistGradientBoosting",
            HistGradientBoostingClassifier(
                max_iter=350,
                learning_rate=0.06,
                max_leaf_nodes=31,
                l2_regularization=1.0,
                early_stopping=False,
                random_state=seed,
            ),
            scale_numeric=False,
        ),
    ]


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def split_dataset(
    dataset: CreditDataset, seed: int = config.RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = dataset.frame
    train_fraction, validation_fraction, test_fraction = config.SPLIT_FRACTIONS
    if not np.isclose(train_fraction + validation_fraction + test_fraction, 1.0):
        raise TrainingError("Split fractions must sum to 1.0")

    train_frame, remainder = train_test_split(
        frame,
        test_size=(validation_fraction + test_fraction),
        random_state=seed,
        stratify=frame[config.TARGET_COLUMN],
        shuffle=True,
    )
    relative_test = test_fraction / (validation_fraction + test_fraction)
    validation_frame, test_frame = train_test_split(
        remainder,
        test_size=relative_test,
        random_state=seed,
        stratify=remainder[config.TARGET_COLUMN],
        shuffle=True,
    )
    return (
        train_frame.reset_index(drop=True),
        validation_frame.reset_index(drop=True),
        test_frame.reset_index(drop=True),
    )


def library_versions() -> Dict[str, str]:
    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    for module_name in ("numpy", "pandas", "sklearn", "joblib", "shap", "reportlab"):
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", "unknown")
        except Exception:  # pragma: no cover
            versions[module_name] = "not installed"
    if not SHAP_AVAILABLE:
        versions["shap"] = versions.get("shap", "not installed")
    return versions


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------


def train_and_save(
    dataset: Optional[CreditDataset] = None,
    data_path: Optional[str] = None,
    seed: int = config.RANDOM_SEED,
    rows: Optional[int] = None,
    allow_synthetic_fallback: bool = False,
    save: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    started = time.time()
    if dataset is None:
        dataset = load_dataset(
            path=data_path, allow_synthetic_fallback=allow_synthetic_fallback
        )

    frame = dataset.frame
    if rows is not None and rows < len(frame):
        frame = frame.sample(n=int(rows), random_state=seed).reset_index(drop=True)
        dataset = CreditDataset(
            frame=frame,
            provenance=dataset.provenance,
            mapping=dataset.mapping,
            unmapped_columns=dataset.unmapped_columns,
            profile=dataset.profile,
        )

    validation_report = validate_dataset(frame, dataset.provenance)
    if not validation_report.is_valid:
        raise TrainingError(
            "Dataset validation failed, refusing to train: "
            + "; ".join(issue.message for issue in validation_report.errors)
        )

    train_frame, validation_frame, test_frame = split_dataset(dataset, seed=seed)
    X_train, y_train = train_frame[list(config.USER_INPUT_FEATURES)], train_frame[config.TARGET_COLUMN]
    X_val, y_val = validation_frame[list(config.USER_INPUT_FEATURES)], validation_frame[config.TARGET_COLUMN]
    X_test, y_test = test_frame[list(config.USER_INPUT_FEATURES)], test_frame[config.TARGET_COLUMN]

    if verbose:
        print("=" * 78)
        print(f"CreditSense AI - training {config.MODEL_VERSION}")
        print("=" * 78)
        print(f"dataset      : {dataset.provenance.filename} (synthetic={dataset.is_synthetic})")
        print(f"rows         : {len(frame)}  positive rate: {y_train.mean():.4f}")
        print(f"splits       : train={len(train_frame)} val={len(validation_frame)} test={len(test_frame)}")
        print(f"model inputs : {len(config.USER_INPUT_FEATURES)} raw + "
              f"{len(config.DERIVED_FEATURES)} derived = {len(config.MODEL_FEATURES)}")
        print(f"seed         : {seed}")

    # ---- Candidate comparison on the validation split --------------------
    candidate_results: Dict[str, Dict[str, Any]] = {}
    fitted: Dict[str, Any] = {}
    for candidate in build_candidates(seed=seed):
        if verbose:
            print(f"  fitting {candidate.name} ...", end="", flush=True)
        pipeline = build_pipeline(candidate.estimator, scale_numeric=candidate.scale_numeric)
        pipeline.fit(X_train, y_train)
        train_metrics = evaluate_probabilities(y_train, pipeline.predict_proba(X_train)[:, 1])
        val_metrics = evaluate_probabilities(y_val, pipeline.predict_proba(X_val)[:, 1])
        candidate_results[candidate.name] = {
            "train": train_metrics,
            "validation": val_metrics,
            "scale_numeric": candidate.scale_numeric,
        }
        fitted[candidate.name] = pipeline
        if verbose:
            print(f" val ROC-AUC={val_metrics['roc_auc']:.4f} PR-AUC={val_metrics['pr_auc']:.4f}")

    # Selection criterion: validation ROC-AUC, then validation PR-AUC, then Brier.
    selection_order = sorted(
        candidate_results.items(),
        key=lambda item: (
            -_nan_safe(item[1]["validation"]["roc_auc"]),
            -_nan_safe(item[1]["validation"]["pr_auc"]),
            _nan_safe(item[1]["validation"]["brier_score"]),
        ),
    )
    selected_name = selection_order[0][0]
    base_pipeline = fitted[selected_name]
    if verbose:
        print(f"selected     : {selected_name} (validation ROC-AUC criterion)")

    # ---- Calibration on the validation split ----------------------------
    base_val_probability = base_pipeline.predict_proba(X_val)[:, 1]
    calibrator = PlattCalibrator().fit(base_val_probability, y_val)
    calibrated_val_probability = calibrator.transform(base_val_probability)

    # ---- Final evaluation on the untouched test split -------------------
    base_test_probability = base_pipeline.predict_proba(X_test)[:, 1]
    test_probability = calibrator.transform(base_test_probability)
    base_train_probability = base_pipeline.predict_proba(X_train)[:, 1]
    train_probability = calibrator.transform(base_train_probability)

    selected_metrics = {
        "train": evaluate_probabilities(y_train, train_probability),
        "validation": evaluate_probabilities(y_val, calibrated_val_probability),
        "test": evaluate_probabilities(y_test, test_probability),
        "test_uncalibrated": evaluate_probabilities(y_test, base_test_probability),
        "validation_uncalibrated": evaluate_probabilities(y_val, base_val_probability),
    }

    # Majority-class reference point (no invented numbers).
    baseline_model = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    baseline_metrics = evaluate_probabilities(y_test, baseline_model.predict_proba(X_test)[:, 1])

    curves = {
        "roc_test": _roc_points(y_test, test_probability),
        "pr_test": _pr_points(y_test, test_probability),
        "calibration_test": calibration_curve_points(y_test, test_probability),
        "calibration_test_uncalibrated": calibration_curve_points(y_test, base_test_probability),
    }
    ece_calibrated = expected_calibration_error(curves["calibration_test"], int(y_test.size))
    ece_uncalibrated = expected_calibration_error(
        curves["calibration_test_uncalibrated"], int(y_test.size)
    )

    overfitting = {
        "train_roc_auc": selected_metrics["train"]["roc_auc"],
        "validation_roc_auc": selected_metrics["validation"]["roc_auc"],
        "test_roc_auc": selected_metrics["test"]["roc_auc"],
        "generalisation_gap_train_minus_test": float(
            _nan_safe(selected_metrics["train"]["roc_auc"])
            - _nan_safe(selected_metrics["test"]["roc_auc"])
        ),
        "validation_minus_test": float(
            _nan_safe(selected_metrics["validation"]["roc_auc"])
            - _nan_safe(selected_metrics["test"]["roc_auc"])
        ),
        "warning": (
            "Train ROC-AUC exceeds test ROC-AUC by more than 0.05; the model shows "
            "signs of overfitting."
            if _nan_safe(selected_metrics["train"]["roc_auc"])
            - _nan_safe(selected_metrics["test"]["roc_auc"])
            > 0.05
            else "No material train/test ROC-AUC gap detected (>0.05)."
        ),
    }

    # ---- Global explanations -------------------------------------------
    explainer = CreditExplainer(
        base_pipeline,
        background=X_train.sample(
            n=min(config.SHAP_BACKGROUND_SIZE, len(X_train)), random_state=seed
        ),
        random_seed=seed,
    )
    shap_importance = explainer.global_importance(X_test)
    shap_status = {
        "available": bool(shap_importance is not None),
        "method": shap_importance.method if shap_importance else "unavailable",
        "method_label": explainer.method_label,
        "notes": list(explainer.notes),
    }

    permutation = _permutation_importance(base_pipeline, X_test, y_test, seed=seed)

    duration = time.time() - started

    # ---- Metadata / schema artifacts -----------------------------------
    metadata = {
        "app_name": config.APP_NAME,
        "model_version": config.MODEL_VERSION,
        "schema_version": config.SCHEMA_VERSION,
        "trained_at": utc_now_iso(),
        "training_duration_seconds": round(duration, 3),
        "random_seed": seed,
        "split_fractions": {
            "train": config.SPLIT_FRACTIONS[0],
            "validation": config.SPLIT_FRACTIONS[1],
            "test": config.SPLIT_FRACTIONS[2],
        },
        "split_sizes": {
            "train": int(len(train_frame)),
            "validation": int(len(validation_frame)),
            "test": int(len(test_frame)),
        },
        "class_counts": {
            split: {
                "0": int((values == 0).sum()),
                "1": int((values == 1).sum()),
            }
            for split, values in (
                ("train", y_train),
                ("validation", y_val),
                ("test", y_test),
            )
        },
        "dataset": dataset.provenance.as_dict(),
        "dataset_fingerprint": dataset_fingerprint(dataset),
        "target_column": config.TARGET_COLUMN,
        "target_definition": config.TARGET_DEFINITION,
        "candidate_models": [candidate.name for candidate in build_candidates(seed=seed)],
        "selection_criterion": "Highest validation ROC-AUC (ties: validation PR-AUC, then Brier score)",
        "selected_model": selected_name,
        "calibration": {
            "method": "Platt scaling (sigmoid) fitted on the validation split",
            "fitted_on": "validation split",
            "base_model_trained_on": "training split only",
            **calibrator.as_dict(),
        },
        "explainability": shap_status,
        "library_versions": library_versions(),
        "geographic_disclosure": config.GEOGRAPHIC_DISCLOSURE,
        "synthetic_disclosure": config.SYNTHETIC_DISCLOSURE if dataset.is_synthetic else None,
        "notes": [
            "Preprocessing (imputation, encoding, scaling) is fitted on the training split only.",
            "The test split was used once, after selection and calibration, and never for tuning.",
            "Protected attributes (sex, marital status) are excluded from the model feature set.",
            "Feature order is fixed by config.USER_INPUT_FEATURES and persisted in the bundle.",
        ],
    }

    schema = {
        "schema_version": config.SCHEMA_VERSION,
        "target_column": config.TARGET_COLUMN,
        "target_definition": config.TARGET_DEFINITION,
        "model_features": list(config.MODEL_FEATURES),
        "user_input_features": list(config.USER_INPUT_FEATURES),
        "numeric_features": list(config.NUMERIC_FEATURES),
        "categorical_features": list(config.CATEGORICAL_FEATURES),
        "derived_features": list(config.DERIVED_FEATURES),
        "feature_labels": {feature: feature_label(feature) for feature in config.MODEL_FEATURES},
        "categorical_categories": {
            "education": list(config.EDUCATION_CATEGORIES),
        },
        "input_ranges": {key: list(value) for key, value in config.INPUT_RANGES.items()},
        "protected_excluded_features": list(config.PROTECTED_EXCLUDED_FEATURES),
        "group_analysis_columns": list(config.GROUP_COLUMNS),
        "unavailable_canonical_features": list(config.UNAVAILABLE_FEATURES),
        "derived_formulas": {
            "avg_bill_amount": "mean(bill_amount_1..6)",
            "avg_payment_amount": "mean(payment_amount_1..6)",
            "credit_utilization": "clip(bill_amount_1 / credit_limit, -2, 5)",
            "repayment_ratio": "clip(sum(payment_amount_1..6) / max(sum(max(bill_amount_i,0)), 1), 0, 2)",
            "max_repayment_delay": "max(repay_status_1..6)",
            "num_delayed_months": "count(repay_status_i >= 1)",
            "num_severely_delayed_months": "count(repay_status_i >= 3)",
            "bill_trend": "bill_amount_1 - bill_amount_6",
            "payment_trend": "payment_amount_1 - payment_amount_6",
        },
    }

    metrics_payload = {
        "reported_at": utc_now_iso(),
        "selected_model": selected_name,
        "selection_criterion": metadata["selection_criterion"],
        "candidates": {
            name: {"train": result["train"], "validation": result["validation"]}
            for name, result in candidate_results.items()
        },
        "selected": selected_metrics,
        "baseline_majority_class": baseline_metrics,
        "curves": curves,
        "calibration_quality": {
            "brier_score_calibrated": selected_metrics["test"]["brier_score"],
            "brier_score_uncalibrated": selected_metrics["test_uncalibrated"]["brier_score"],
            "expected_calibration_error_calibrated": ece_calibrated,
            "expected_calibration_error_uncalibrated": ece_uncalibrated,
            "interpretation": (
                "A lower Brier score and a reliability curve close to the diagonal indicate "
                "better-calibrated probabilities. A calibrated probability is still an estimate "
                "for this dataset and period, not a guarantee about an individual applicant."
            ),
        },
        "overfitting": overfitting,
        "explainability": shap_status,
        "permutation_importance_method": (
            "sklearn.inspection.permutation_importance, scoring=roc_auc, "
            f"n_repeats={config.PERMUTATION_IMPORTANCE_REPEATS}, on the test split"
        ),
        "test_split_metrics_are_measured": True,
        "test_split_used_for_tuning": False,
        "disclosures": {
            "geographic": config.GEOGRAPHIC_DISCLOSURE,
            "synthetic": config.SYNTHETIC_DISCLOSURE if dataset.is_synthetic else None,
            "prototype": config.APP_DISCLAIMER,
        },
    }

    if save:
        bundle = {
            "base_pipeline": base_pipeline,
            "calibrator": calibrator,
            "feature_names": list(config.USER_INPUT_FEATURES),
            "model_features": list(config.MODEL_FEATURES),
            "model_name": selected_name,
            "model_version": config.MODEL_VERSION,
            "schema_version": config.SCHEMA_VERSION,
            "trained_at": metadata["trained_at"],
            "random_seed": seed,
            "target_column": config.TARGET_COLUMN,
            "dataset_fingerprint": metadata["dataset_fingerprint"],
            "dataset_filename": dataset.provenance.filename,
            "is_synthetic_dataset": dataset.is_synthetic,
            "test_metrics": selected_metrics["test"],
            "calibration_info": metadata["calibration"],
        }
        config.MODEL_BUNDLE_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Compressed: a 300-tree forest serialises to ~78 MB uncompressed, which is
        # an unreasonable artifact to ship. Loading cost is a fraction of a second.
        joblib.dump(bundle, config.MODEL_BUNDLE_PATH, compress=("zlib", 3))
        save_json(schema, config.FEATURE_SCHEMA_PATH)
        save_json(metadata, config.TRAINING_METADATA_PATH)
        save_json(metrics_payload, config.EVALUATION_METRICS_PATH)
        save_json({"importances": permutation}, config.PERMUTATION_IMPORTANCE_PATH)
        save_json(
            shap_importance.as_dict() if shap_importance else {"ranking": [], "method": "unavailable"},
            config.GLOBAL_SHAP_PATH,
        )
        save_json(dataset.profile, config.DATASET_PROFILE_PATH)
        if verbose:
            print(f"artifacts    : {config.MODEL_BUNDLE_PATH}")
            print(f"               {config.EVALUATION_METRICS_PATH}")

    result = {
        "selected_model": selected_name,
        "metrics": metrics_payload,
        "metadata": metadata,
        "schema": schema,
        "explainer": explainer,
        "bundle": {
            "base_pipeline": base_pipeline,
            "calibrator": calibrator,
            "model_name": selected_name,
        },
        "frames": {
            "train": train_frame,
            "validation": validation_frame,
            "test": test_frame,
        },
        "dataset": dataset,
        "shap_importance": shap_importance,
        "permutation_importance": permutation,
    }

    if verbose:
        test = selected_metrics["test"]
        print("-" * 78)
        print(f"TEST  accuracy={test['accuracy']:.4f} precision={test['precision']:.4f} "
              f"recall={test['recall']:.4f} f1={test['f1']:.4f}")
        print(f"      ROC-AUC={test['roc_auc']:.4f}  PR-AUC={test['pr_auc']:.4f}  "
              f"Brier={test['brier_score']:.4f}")
        print(f"      confusion [[TN={test['true_negatives']} FP={test['false_positives']}] "
              f"[FN={test['false_negatives']} TP={test['true_positives']}]]")
        print(f"baseline majority-class accuracy={baseline_metrics['accuracy']:.4f} "
              f"ROC-AUC={baseline_metrics['roc_auc']:.4f}")
        print(f"calibration: Brier {selected_metrics['test_uncalibrated']['brier_score']:.4f} "
              f"-> {test['brier_score']:.4f}, ECE {ece_uncalibrated:.4f} -> {ece_calibrated:.4f}")
        print(f"shap         : {shap_status['method_label']}")
        print(f"duration     : {duration:.1f}s")
        print("=" * 78)
    return result


def _nan_safe(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("-inf")
    if not np.isfinite(number):
        return float("-inf")
    return number


def _permutation_importance(
    pipeline, X_test: pd.DataFrame, y_test, seed: int, max_rows: int = 2000
) -> List[Dict[str, Any]]:
    """Permutation importance on source features (not on one-hot columns)."""
    if len(X_test) > max_rows:
        sample = X_test.sample(n=max_rows, random_state=seed)
        sample_y = y_test.loc[sample.index]
    else:
        sample, sample_y = X_test, y_test
    try:
        result = permutation_importance(
            pipeline,
            sample,
            sample_y,
            scoring="roc_auc",
            n_repeats=config.PERMUTATION_IMPORTANCE_REPEATS,
            random_state=seed,
            n_jobs=1,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Permutation importance failed: %s", exc)
        return []
    rows = [
        {
            "feature": feature,
            "label": feature_label(feature),
            "mean_importance": float(mean),
            "std_importance": float(std),
        }
        for feature, mean, std in zip(
            config.USER_INPUT_FEATURES, result.importances_mean, result.importances_std
        )
    ]
    rows.sort(key=lambda item: item["mean_importance"], reverse=True)
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train the CreditSense AI credit-default risk model."
    )
    parser.add_argument("--data", default=None, help="Explicit path to a dataset file.")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument(
        "--rows", type=int, default=None, help="Subsample N rows (for a quick run)."
    )
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="Permit the documented synthetic fallback when no dataset is found.",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    try:
        train_and_save(
            data_path=args.data,
            seed=args.seed,
            rows=args.rows,
            allow_synthetic_fallback=args.allow_synthetic,
            verbose=not args.quiet,
        )
    except Exception as exc:
        print(f"TRAINING FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
