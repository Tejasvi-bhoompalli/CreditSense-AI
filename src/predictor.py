"""Inference layer: load the saved bundle and score borrowers.

The loaded bundle contains the fitted preprocessing pipeline, the fitted base
model and the fitted calibration transform, so a single call reproduces exactly
the procedure used during training. The predictor validates its inputs and fails
loudly (with an actionable message) rather than silently substituting values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from src import config
from src.data_validation import (
    InputValidationError,
    build_input_frame,
    validate_user_inputs,
)
from src.preprocessing import compute_derived_features
from src.risk_scoring import RiskScore, score_prediction
from src.utils import load_json, utc_now_iso


class ModelNotReadyError(RuntimeError):
    """Raised when model artifacts are missing or incompatible.

    The message always contains the exact command needed to fix the problem, so
    the UI can display it verbatim.
    """

    RETRAIN_INSTRUCTION = (
        "Train the model first:\n\n"
        "    python -m src.train_model\n\n"
        "Artifacts are written to artifacts/model, artifacts/metrics and artifacts/metadata."
    )


@dataclass
class PredictionResult:
    """A complete, reproducible result for one borrower profile."""

    probability: float
    probability_uncalibrated: float
    risk: RiskScore
    model_name: str
    model_version: str
    score_type: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    derived: Dict[str, float] = field(default_factory=dict)
    explanation: Optional[Any] = None
    validation_report: Optional[Any] = None
    predicted_at: str = field(default_factory=utc_now_iso)
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "probability": self.probability,
            "probability_uncalibrated": self.probability_uncalibrated,
            "risk": self.risk.as_dict(),
            "model_name": self.model_name,
            "model_version": self.model_version,
            "score_type": self.score_type,
            "inputs": dict(self.inputs),
            "derived": dict(self.derived),
            "predicted_at": self.predicted_at,
            "notes": list(self.notes),
        }


class CreditRiskPredictor:
    """Wraps the persisted pipeline + calibrator."""

    def __init__(self, bundle_path: Optional[Path] = None):
        self.bundle_path = Path(bundle_path or config.MODEL_BUNDLE_PATH)
        self._bundle: Optional[Dict[str, Any]] = None

    # -- loading ---------------------------------------------------------

    def load(self, force: bool = False) -> "CreditRiskPredictor":
        if self._bundle is not None and not force:
            return self
        if not self.bundle_path.exists():
            raise ModelNotReadyError(
                f"Model bundle not found at '{self.bundle_path}'.\n\n"
                + ModelNotReadyError.RETRAIN_INSTRUCTION
            )
        try:
            bundle = joblib.load(self.bundle_path)
        except Exception as exc:
            raise ModelNotReadyError(
                f"The saved model bundle could not be loaded ({type(exc).__name__}: {exc}). "
                "It may have been produced by an incompatible library version.\n\n"
                + ModelNotReadyError.RETRAIN_INSTRUCTION
            ) from exc

        required = {"base_pipeline", "calibrator", "feature_names", "model_name"}
        missing = required - set(bundle)
        if missing:
            raise ModelNotReadyError(
                f"The saved model bundle is incomplete (missing {sorted(missing)}).\n\n"
                + ModelNotReadyError.RETRAIN_INSTRUCTION
            )
        if list(bundle["feature_names"]) != list(config.USER_INPUT_FEATURES):
            raise ModelNotReadyError(
                "The saved model expects a different feature schema than this version of "
                "the application provides. This usually means the schema changed after the "
                "model was trained.\n\n" + ModelNotReadyError.RETRAIN_INSTRUCTION
            )
        self._bundle = bundle
        return self

    @property
    def bundle(self) -> Dict[str, Any]:
        if self._bundle is None:
            self.load()
        assert self._bundle is not None
        return self._bundle

    @property
    def is_ready(self) -> bool:
        try:
            self.load()
            return True
        except ModelNotReadyError:
            return False

    @property
    def model_name(self) -> str:
        return str(self.bundle.get("model_name", "unknown"))

    @property
    def model_version(self) -> str:
        return str(self.bundle.get("model_version", config.MODEL_VERSION))

    @property
    def trained_at(self) -> str:
        return str(self.bundle.get("trained_at", ""))

    @property
    def is_calibrated(self) -> bool:
        return self.bundle.get("calibrator") is not None

    @property
    def test_metrics(self) -> Dict[str, Any]:
        return dict(self.bundle.get("test_metrics", {}))

    @property
    def dataset_info(self) -> Dict[str, Any]:
        metadata = load_json(config.TRAINING_METADATA_PATH)
        return dict(metadata.get("dataset", {}))

    # -- scoring ---------------------------------------------------------

    def predict_frame(self, frame: pd.DataFrame, explain: bool = False) -> List[PredictionResult]:
        """Score a frame that already contains the canonical raw input columns."""
        self.load()
        missing = [column for column in config.USER_INPUT_FEATURES if column not in frame.columns]
        if missing:
            raise InputValidationError(
                _missing_columns_report(missing)
            )
        ordered = frame[list(config.USER_INPUT_FEATURES)].copy()
        for column in config.CATEGORICAL_FEATURES:
            ordered[column] = ordered[column].astype("string")

        base_probabilities = self.bundle["base_pipeline"].predict_proba(ordered)[:, 1]
        probabilities = (
            self._calibrate(base_probabilities)
            if self.is_calibrated
            else np.asarray(base_probabilities, dtype=float)
        )
        derived = compute_derived_features(ordered)

        results: List[PredictionResult] = []
        for position in range(len(ordered)):
            probability = float(probabilities[position])
            results.append(
                PredictionResult(
                    probability=probability,
                    probability_uncalibrated=float(base_probabilities[position]),
                    risk=score_prediction(probability),
                    model_name=self.model_name,
                    model_version=self.model_version,
                    score_type=(
                        "Calibrated predicted default probability (Platt scaling fitted on the "
                        "validation split)"
                        if self.is_calibrated
                        else "Model-derived risk index (uncalibrated score, not a validated probability)"
                    ),
                    inputs=ordered.iloc[position].to_dict(),
                    derived={
                        feature: float(derived.iloc[position][feature])
                        for feature in config.DERIVED_FEATURES
                    },
                    notes=list(self._bundle_notes()),
                )
            )
        return results

    def predict_one(
        self, inputs: Dict[str, Any], explain: bool = False, strict: bool = True
    ) -> PredictionResult:
        """Validate, then score a single borrower profile."""
        cleaned, report = validate_user_inputs(inputs, strict=strict)
        if strict and not report.is_valid:
            raise InputValidationError(report)
        frame = build_input_frame(cleaned)
        result = self.predict_frame(frame, explain=explain)[0]
        result.validation_report = report
        if explain:
            result.explanation = self.explain(result.inputs, probability=result.probability)
        return result

    def _calibrate(self, probabilities: np.ndarray) -> np.ndarray:
        calibrator = self.bundle.get("calibrator")
        if calibrator is None:
            return np.asarray(probabilities, dtype=float)
        calibrated = np.asarray(calibrator.transform(np.asarray(probabilities, dtype=float)))
        return np.clip(calibrated, 1e-6, 1 - 1e-6)

    def _bundle_notes(self) -> List[str]:
        notes = [
            "A predicted probability is an estimate, not a certainty. A 30% predicted "
            "probability does not mean 30% of similar applicants will default.",
        ]
        if self.bundle.get("is_synthetic_dataset"):
            notes.append(config.SYNTHETIC_DISCLOSURE)
        else:
            notes.append(config.GEOGRAPHIC_DISCLOSURE)
        return notes

    # -- explainability --------------------------------------------------

    def explainer(self, background: Optional[pd.DataFrame] = None):
        """Build a :class:`CreditExplainer` bound to the fitted base model."""
        from src.explainability import CreditExplainer

        self.load()
        if background is None:
            background = self._default_background()
        return CreditExplainer(self.bundle["base_pipeline"], background=background)

    def _default_background(self) -> pd.DataFrame:
        """Background sample for SHAP, taken from the training data."""
        from src.data_loader import load_dataset

        dataset = load_dataset()
        frame = dataset.frame
        if self.bundle.get("dataset_fingerprint"):
            sample_size = min(config.SHAP_BACKGROUND_SIZE, len(frame))
        else:
            sample_size = config.SHAP_BACKGROUND_SIZE
        return frame.sample(n=sample_size, random_state=config.RANDOM_SEED).reset_index(drop=True)

    def explain(self, inputs: Dict[str, Any], probability: Optional[float] = None):
        self.load()
        frame = build_input_frame(inputs)
        explainer = self.explainer()
        return explainer.explain_local(frame, probability=probability)


def _missing_columns_report(missing: List[str]):
    from src.data_validation import ValidationReport

    report = ValidationReport()
    report.add(
        "missing_input_columns",
        "error",
        f"Input frame is missing required columns: {missing}",
    )
    return report


# ---------------------------------------------------------------------------
# Cached accessors (Streamlit-friendly)
# ---------------------------------------------------------------------------

_PREDICTOR_SINGLETON: Optional[CreditRiskPredictor] = None


def get_predictor(force_reload: bool = False) -> CreditRiskPredictor:
    global _PREDICTOR_SINGLETON
    if _PREDICTOR_SINGLETON is None or force_reload:
        _PREDICTOR_SINGLETON = CreditRiskPredictor().load(force=force_reload)
    return _PREDICTOR_SINGLETON


def model_status() -> Dict[str, Any]:
    """Readiness information for the UI header and the Overview page."""
    status: Dict[str, Any] = {
        "bundle_path": str(config.MODEL_BUNDLE_PATH),
        "bundle_exists": config.MODEL_BUNDLE_PATH.exists(),
        "metrics_exists": config.EVALUATION_METRICS_PATH.exists(),
        "metadata_exists": config.TRAINING_METADATA_PATH.exists(),
        "schema_exists": config.FEATURE_SCHEMA_PATH.exists(),
        "ready": False,
        "error": None,
        "model_name": None,
        "model_version": None,
        "trained_at": None,
    }
    if not status["bundle_exists"]:
        status["error"] = (
            "No trained model found. Run `python -m src.train_model` to create it."
        )
        return status
    try:
        predictor = CreditRiskPredictor().load()
        status.update(
            {
                "ready": True,
                "model_name": predictor.model_name,
                "model_version": predictor.model_version,
                "trained_at": predictor.trained_at,
                "is_calibrated": predictor.is_calibrated,
                "test_roc_auc": predictor.test_metrics.get("roc_auc"),
            }
        )
    except ModelNotReadyError as exc:
        status["error"] = str(exc)
    return status
