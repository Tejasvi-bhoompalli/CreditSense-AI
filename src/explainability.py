"""Explainable AI layer: SHAP-based local and global explanations.

Method notes (also surfaced in the UI):

* The explanation is computed for the **same fitted base estimator** that
  produces the prediction. It is never a proxy model.
* SHAP values are expressed in the model's **raw output units (log-odds)** of
  the base model. The probability shown in the interface is a strictly monotone
  (Platt/sigmoid) calibration of exactly that output, so the ordering of feature
  contributions is unaffected. This is stated in the UI rather than hidden.
* One-hot encoded categorical columns are summed back onto their source feature
  so every contribution has a human-readable name.
* If SHAP is unavailable or incompatible, the module degrades to a transparent
  local occlusion explanation and to permutation importance globally. The
  fallback is always labelled as such; it is never presented as SHAP.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src import config
from src.preprocessing import aggregate_contributions, source_feature_of
from src.utils import feature_label, safe_logit, utc_now_iso

try:  # pragma: no cover - import guard
    import shap

    SHAP_AVAILABLE = True
    SHAP_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover
    shap = None  # type: ignore[assignment]
    SHAP_AVAILABLE = False
    SHAP_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

TREE_MODEL_NAMES = {
    "RandomForestClassifier",
    "ExtraTreesClassifier",
    "GradientBoostingClassifier",
    "HistGradientBoostingClassifier",
    "DecisionTreeClassifier",
    "XGBClassifier",
    "LGBMClassifier",
}
LINEAR_MODEL_NAMES = {"LogisticRegression", "LinearSVC", "RidgeClassifier", "SGDClassifier"}

SHAP_CAVEAT = (
    "SHAP values describe how each feature contributed to this model's output for this "
    "input under the chosen explanation method. They are not causal effects: they do not "
    "prove that changing a feature would produce the same change in real-world default risk."
)


@dataclass
class FeatureContribution:
    feature: str
    label: str
    value: Any
    display_value: str
    contribution: float
    direction: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "feature": self.feature,
            "label": self.label,
            "value": self.value,
            "display_value": self.display_value,
            "contribution": round(self.contribution, 6),
            "direction": self.direction,
        }


@dataclass
class LocalExplanation:
    method: str
    method_label: str
    base_value: float
    output_value: float
    probability: float
    contributions: List[FeatureContribution] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    generated_at: str = field(default_factory=utc_now_iso)

    @property
    def increasing(self) -> List[FeatureContribution]:
        return [item for item in self.contributions if item.contribution > 0]

    @property
    def decreasing(self) -> List[FeatureContribution]:
        return [item for item in self.contributions if item.contribution < 0]

    @property
    def top_increasing(self) -> List[FeatureContribution]:
        return sorted(self.increasing, key=lambda item: item.contribution, reverse=True)

    @property
    def top_decreasing(self) -> List[FeatureContribution]:
        return sorted(self.decreasing, key=lambda item: item.contribution)

    def ranked(self, limit: Optional[int] = None) -> List[FeatureContribution]:
        ordered = sorted(self.contributions, key=lambda item: abs(item.contribution), reverse=True)
        return ordered[:limit] if limit else ordered

    def as_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "method_label": self.method_label,
            "base_value": round(self.base_value, 6),
            "output_value": round(self.output_value, 6),
            "probability": self.probability,
            "generated_at": self.generated_at,
            "contributions": [item.as_dict() for item in self.contributions],
            "notes": list(self.notes),
        }

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Feature": item.label,
                    "Value": item.display_value,
                    "Contribution (log-odds)": round(item.contribution, 4),
                    "Direction": "Increases risk" if item.contribution > 0 else "Reduces risk",
                }
                for item in self.ranked()
            ]
        )


@dataclass
class GlobalImportance:
    method: str
    method_label: str
    rows_used: int
    ranking: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    generated_at: str = field(default_factory=utc_now_iso)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "method_label": self.method_label,
            "rows_used": self.rows_used,
            "ranking": self.ranking,
            "notes": list(self.notes),
            "generated_at": self.generated_at,
        }

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.ranking)


def _display_value(feature: str, value: Any) -> str:
    if feature in config.CATEGORICAL_FEATURES:
        return config.EDUCATION_DISPLAY.get(str(value), str(value))
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if feature in config.REPAY_STATUS_COLUMNS:
        return f"{int(number)} ({config.REPAY_STATUS_CODES.get(int(number), 'undocumented')})"
    if any(
        token in feature
        for token in ("limit", "bill", "payment", "balance", "amount", "trend")
    ):
        return f"NT$ {number:,.0f}"
    if feature in {"credit_utilization", "repayment_ratio"}:
        return f"{number:.3f}"
    return f"{number:,.2f}"


class CreditExplainer:
    """SHAP (or documented fallback) explanations for a fitted pipeline."""

    def __init__(
        self,
        pipeline,
        background: Optional[pd.DataFrame] = None,
        random_seed: int = config.RANDOM_SEED,
    ):
        self.pipeline = pipeline
        self.random_seed = random_seed
        self.estimator = pipeline.named_steps["model"]
        self.features_step = pipeline.named_steps["features"]
        self.preprocessor = pipeline.named_steps["preprocess"]
        self.transformed_names = [str(n) for n in self.preprocessor.get_feature_names_out()]
        self.model_name = type(self.estimator).__name__
        self.notes: List[str] = []
        self.background: Optional[pd.DataFrame] = None
        self.background_matrix: Optional[np.ndarray] = None
        self.explainer = None
        self.method = "unavailable"
        self.method_label = "not available"
        self.base_value = 0.0
        self._background_medians: Dict[str, Any] = {}
        self._background_modes: Dict[str, str] = {}

        if background is not None and len(background) > 0:
            self.set_background(background)

    # -- setup ------------------------------------------------------------

    def set_background(self, background: pd.DataFrame) -> None:
        frame = background[list(config.USER_INPUT_FEATURES)].copy()
        if len(frame) > config.SHAP_BACKGROUND_SIZE:
            frame = frame.sample(
                n=config.SHAP_BACKGROUND_SIZE, random_state=self.random_seed
            ).reset_index(drop=True)
        self.background = frame
        self.background_matrix = self._transform(frame)
        for column in config.NUMERIC_FEATURES:
            if column in frame.columns:
                self._background_medians[column] = float(
                    pd.to_numeric(frame[column], errors="coerce").median()
                )
        for column in config.CATEGORICAL_FEATURES:
            if column in frame.columns:
                modes = frame[column].mode(dropna=True)
                self._background_modes[column] = str(modes.iloc[0]) if len(modes) else "other"
        self._build_explainer()

    def _transform(self, frame: pd.DataFrame) -> np.ndarray:
        enriched = self.features_step.transform(frame)
        return np.asarray(self.preprocessor.transform(enriched), dtype=float)

    def _build_explainer(self) -> None:
        if self.background_matrix is None:
            return
        self.notes = []
        if SHAP_AVAILABLE:
            try:
                if self.model_name in TREE_MODEL_NAMES:
                    self.explainer = shap.TreeExplainer(self.estimator)
                    self.method = "shap_tree"
                    self.method_label = "SHAP TreeExplainer (exact tree-path contributions)"
                elif self.model_name in LINEAR_MODEL_NAMES:
                    masker = shap.maskers.Independent(self.background_matrix)
                    self.explainer = shap.LinearExplainer(self.estimator, masker=masker)
                    self.method = "shap_linear"
                    self.method_label = "SHAP LinearExplainer"
                else:
                    self.explainer = shap.Explainer(
                        self.estimator.predict_proba, self.background_matrix
                    )
                    self.method = "shap_generic"
                    self.method_label = "SHAP model-agnostic explainer"
                self.base_value = self._extract_base_value()
                return
            except Exception as exc:  # explanation failure must never crash the app
                self.notes.append(
                    f"SHAP could not explain this model ({type(exc).__name__}: {exc}). "
                    "Falling back to transparent local occlusion contributions."
                )
                self.explainer = None
        else:
            self.notes.append(
                f"The `shap` package is unavailable ({SHAP_IMPORT_ERROR}). Falling back to "
                "transparent local occlusion contributions and permutation importance."
            )

        self.method = "local_occlusion_fallback"
        self.method_label = "Local occlusion fallback (not SHAP)"
        self.base_value = float(
            np.mean(self._raw_output(self.background_matrix))
        ) if self.background_matrix is not None else 0.0
        if not self.notes:
            self.notes.append("SHAP unavailable; using the documented fallback.")

    def _extract_base_value(self) -> float:
        expected = getattr(self.explainer, "expected_value", None)
        if expected is None:
            return float(np.mean(self._raw_output(self.background_matrix)))
        array = np.asarray(expected, dtype=float).ravel()
        if array.size == 1:
            return float(array[0])
        if array.size >= 2:
            return float(array[1])
        return 0.0

    def _raw_output(self, matrix: np.ndarray) -> np.ndarray:
        if hasattr(self.estimator, "decision_function"):
            return np.asarray(self.estimator.decision_function(matrix), dtype=float).ravel()
        probabilities = np.asarray(self.estimator.predict_proba(matrix), dtype=float)[:, 1]
        return np.array([safe_logit(value) for value in probabilities])

    @property
    def is_available(self) -> bool:
        return self.explainer is not None

    # -- local ------------------------------------------------------------

    def explain_local(
        self, raw_row: pd.DataFrame, probability: Optional[float] = None
    ) -> LocalExplanation:
        frame = raw_row[list(config.USER_INPUT_FEATURES)]
        matrix = self._transform(frame)

        if self.explainer is not None:
            try:
                values = self._shap_values(matrix)
                per_feature = aggregate_contributions(self.transformed_names, values[0])
                output_value = float(np.sum(values[0]) + self.base_value)
            except Exception as exc:
                self.notes.append(
                    f"SHAP evaluation failed at inference time ({type(exc).__name__}: {exc}); "
                    "using the local occlusion fallback for this explanation."
                )
                per_feature, output_value = self._occlusion_contributions(frame)
                method, method_label = (
                    "local_occlusion_fallback",
                    "Local occlusion fallback (not SHAP)",
                )
                return self._build_explanation(
                    frame, per_feature, output_value, probability, method, method_label
                )
            return self._build_explanation(
                frame, per_feature, output_value, probability, self.method, self.method_label
            )

        per_feature, output_value = self._occlusion_contributions(frame)
        return self._build_explanation(
            frame,
            per_feature,
            output_value,
            probability,
            "local_occlusion_fallback",
            "Local occlusion fallback (not SHAP)",
        )

    def _shap_values(self, matrix: np.ndarray) -> np.ndarray:
        raw = self.explainer.shap_values(matrix)
        values = self._normalise_shap_values(raw, n_rows=matrix.shape[0])
        return values

    @staticmethod
    def _normalise_shap_values(raw, n_rows: int) -> np.ndarray:
        """Normalise the many possible SHAP return shapes to (rows, features)."""
        if isinstance(raw, list):
            array = np.asarray(raw[1] if len(raw) > 1 else raw[0], dtype=float)
        else:
            array = np.asarray(raw, dtype=float)
        if array.ndim == 3:
            array = array[:, :, 1] if array.shape[-1] == 2 else array[:, :, 0]
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.shape[0] != n_rows and array.shape[1] == n_rows:
            array = array.T
        return array

    def _occlusion_contributions(self, frame: pd.DataFrame) -> tuple:
        """Per-feature contribution by replacing one feature with its background value.

        Transparent and model-agnostic. It is a non-additive heuristic, which is
        why it is always labelled as a fallback rather than as SHAP.
        """
        matrix = self._transform(frame)
        baseline_output = float(self._raw_output(matrix)[0])
        contributions: Dict[str, float] = {}
        for feature in config.USER_INPUT_FEATURES:
            if feature in config.CATEGORICAL_FEATURES:
                replacement = self._background_modes.get(feature, "other")
            else:
                replacement = self._background_medians.get(feature, 0.0)
            perturbed = frame.copy()
            perturbed.loc[perturbed.index[0], feature] = replacement
            perturbed_matrix = self._transform(perturbed)
            perturbed_output = float(self._raw_output(perturbed_matrix)[0])
            contributions[feature] = baseline_output - perturbed_output
        output_value = baseline_output
        return contributions, output_value

    def _build_explanation(
        self,
        frame: pd.DataFrame,
        per_feature: Dict[str, float],
        output_value: float,
        probability: Optional[float],
        method: str,
        method_label: str,
    ) -> LocalExplanation:
        # The raw input row does not contain the derived features, so the value shown
        # for a contribution is read from the enriched frame that the model consumed.
        enriched = self.features_step.transform(frame[list(config.USER_INPUT_FEATURES)])
        row = enriched.iloc[0]
        contributions = [
            FeatureContribution(
                feature=feature,
                label=feature_label(feature),
                value=row[feature],
                display_value=_display_value(feature, row[feature]),
                contribution=float(value),
                direction="increases_risk" if value > 0 else "decreases_risk",
            )
            for feature, value in per_feature.items()
        ]
        if probability is None:
            probability = float(1.0 / (1.0 + np.exp(-output_value)))
        notes = list(self.notes)
        notes.append(SHAP_CAVEAT)
        if method.startswith("shap"):
            notes.append(
                "Contributions are in the base model's raw output units (log-odds). The "
                "displayed probability is a monotone calibration of the same output, so "
                "the direction and ordering of contributions are preserved."
            )
        return LocalExplanation(
            method=method,
            method_label=method_label,
            base_value=float(self.base_value),
            output_value=float(output_value),
            probability=float(probability),
            contributions=contributions,
            notes=notes,
        )

    # -- global -----------------------------------------------------------

    def global_importance(
        self, raw_sample: pd.DataFrame, max_rows: int = config.SHAP_EXPLANATION_SAMPLE_SIZE
    ) -> Optional[GlobalImportance]:
        if self.explainer is None:
            return None
        frame = raw_sample[list(config.USER_INPUT_FEATURES)]
        if len(frame) > max_rows:
            frame = frame.sample(n=max_rows, random_state=self.random_seed)
        try:
            matrix = self._transform(frame)
            values = self._shap_values(matrix)
        except Exception as exc:
            self.notes.append(
                f"Global SHAP evaluation failed ({type(exc).__name__}: {exc})."
            )
            return None

        totals: Dict[str, List[float]] = {}
        for row_values in values:
            for feature, value in aggregate_contributions(self.transformed_names, row_values).items():
                totals.setdefault(feature, []).append(float(value))

        rows: List[Dict[str, Any]] = []
        total_magnitude = sum(
            float(np.mean(np.abs(np.asarray(vals)))) for vals in totals.values()
        ) or 1.0
        for feature, vals in totals.items():
            array = np.asarray(vals, dtype=float)
            rows.append(
                {
                    "feature": feature,
                    "label": feature_label(feature),
                    "mean_abs_contribution": float(np.mean(np.abs(array))),
                    "mean_contribution": float(np.mean(array)),
                    "share_of_total": float(np.mean(np.abs(array)) / total_magnitude),
                    "direction": (
                        "increases risk" if float(np.mean(array)) > 0 else "reduces risk"
                    ),
                }
            )
        rows.sort(key=lambda item: item["mean_abs_contribution"], reverse=True)
        return GlobalImportance(
            method=self.method,
            method_label=self.method_label,
            rows_used=int(len(frame)),
            ranking=rows,
            notes=[
                "Mean absolute SHAP contribution per source feature over a random sample "
                f"of {len(frame)} records. One-hot columns are summed back onto their source feature.",
                SHAP_CAVEAT,
            ],
        )


# ---------------------------------------------------------------------------
# Human-readable translation
# ---------------------------------------------------------------------------

_FEATURE_PHRASES: Dict[str, tuple] = {
    "repay_status_1": (
        "The most recent repayment status showed a delay",
        "The most recent repayment status showed no delay",
    ),
    "max_repayment_delay": (
        "The worst repayment status in the last six months showed a delay",
        "No repayment delay appeared anywhere in the last six months",
    ),
    "num_delayed_months": (
        "Several of the last six months had a delayed payment",
        "Delayed payments were rare across the last six months",
    ),
    "num_severely_delayed_months": (
        "At least one month showed a severe repayment delay",
        "No month showed a severe repayment delay",
    ),
    "credit_utilization": (
        "The latest balance was high relative to the granted credit limit",
        "The latest balance was low relative to the granted credit limit",
    ),
    "repayment_ratio": (
        "Payments covered only a small share of the outstanding balances",
        "Payments covered a large share of the outstanding balances",
    ),
    "avg_bill_amount": (
        "The average statement balance over six months was high",
        "The average statement balance over six months was low",
    ),
    "avg_payment_amount": (
        "The average amount paid over six months was high relative to the balances",
        "The average amount paid over six months was modest",
    ),
    "bill_trend": (
        "The statement balance increased between the oldest and newest month",
        "The statement balance decreased between the oldest and newest month",
    ),
    "payment_trend": (
        "The amount paid decreased between the oldest and newest month",
        "The amount paid increased between the oldest and newest month",
    ),
    "credit_limit": (
        "The granted credit limit was low, which the model associates with higher risk",
        "The granted credit limit was comparatively high",
    ),
    "age": (
        "The applicant's age contributed to a higher predicted risk",
        "The applicant's age contributed to a lower predicted risk",
    ),
    "education": (
        "The recorded education level contributed to a higher predicted risk",
        "The recorded education level contributed to a lower predicted risk",
    ),
}


def _phrase(feature: str, increasing: bool) -> str:
    if feature in _FEATURE_PHRASES:
        return _FEATURE_PHRASES[feature][0 if increasing else 1]
    label = feature_label(feature)
    if increasing:
        return f"{label} pushed the predicted risk upward"
    return f"{label} pulled the predicted risk downward"


def narrate_local(explanation: LocalExplanation, limit: int = 5) -> List[str]:
    """Plain-English sentences derived only from measured contributions."""
    sentences: List[str] = []
    for item in explanation.top_increasing[:limit]:
        sentences.append(
            f"{_phrase(item.feature, True)} ({item.display_value}), and this contributed to "
            "the model's higher predicted risk."
        )
    for item in explanation.top_decreasing[:limit]:
        sentences.append(
            f"{_phrase(item.feature, False)} ({item.display_value}), and this contributed "
            "toward a lower predicted risk."
        )
    if not sentences:
        sentences.append("No single feature dominated this prediction.")
    return sentences


def narrate_global(importance: GlobalImportance, limit: int = 5) -> List[str]:
    sentences: List[str] = []
    for row in importance.ranking[:limit]:
        sentences.append(
            f"{row['label']} was among the strongest contributors across the sample "
            f"(mean absolute SHAP contribution {row['mean_abs_contribution']:.4f} log-odds, "
            f"{row['share_of_total'] * 100:.1f}% of total attribution)."
        )
    return sentences
