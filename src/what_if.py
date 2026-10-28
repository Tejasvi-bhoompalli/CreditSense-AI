"""What-if scenario simulation.

The simulator never mutates a stored assessment. It copies the baseline inputs,
applies the requested adjustments, re-validates them and re-scores through the
same saved pipeline. Every displayed number is produced by an actual model call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src import config
from src.data_validation import InputValidationError, validate_user_inputs
from src.risk_scoring import category_transition
from src.utils import feature_label

#: Adjustable inputs, with the bounds and step used by the UI sliders.
#: Bounds are the documented dataset-supported ranges (config.INPUT_RANGES) plus
#: a sensible step for exploration.
SUPPORTED_ADJUSTMENTS: Dict[str, Dict[str, Any]] = {
    "credit_limit": {"step": 10_000, "kind": "currency", "group": "Account"},
    "age": {"step": 1, "kind": "years", "group": "Account"},
    "repay_status_1": {"step": 1, "kind": "status", "group": "Repayment history"},
    "repay_status_2": {"step": 1, "kind": "status", "group": "Repayment history"},
    "repay_status_3": {"step": 1, "kind": "status", "group": "Repayment history"},
    "repay_status_4": {"step": 1, "kind": "status", "group": "Repayment history"},
    "repay_status_5": {"step": 1, "kind": "status", "group": "Repayment history"},
    "repay_status_6": {"step": 1, "kind": "status", "group": "Repayment history"},
    "bill_amount_1": {"step": 1_000, "kind": "currency", "group": "Statement balances"},
    "bill_amount_2": {"step": 1_000, "kind": "currency", "group": "Statement balances"},
    "bill_amount_3": {"step": 1_000, "kind": "currency", "group": "Statement balances"},
    "bill_amount_4": {"step": 1_000, "kind": "currency", "group": "Statement balances"},
    "bill_amount_5": {"step": 1_000, "kind": "currency", "group": "Statement balances"},
    "bill_amount_6": {"step": 1_000, "kind": "currency", "group": "Statement balances"},
    "payment_amount_1": {"step": 500, "kind": "currency", "group": "Payments"},
    "payment_amount_2": {"step": 500, "kind": "currency", "group": "Payments"},
    "payment_amount_3": {"step": 500, "kind": "currency", "group": "Payments"},
    "payment_amount_4": {"step": 500, "kind": "currency", "group": "Payments"},
    "payment_amount_5": {"step": 500, "kind": "currency", "group": "Payments"},
    "payment_amount_6": {"step": 500, "kind": "currency", "group": "Payments"},
}

#: The controls shown without expanding an "advanced" section.
PRIMARY_ADJUSTMENTS: List[str] = [
    "credit_limit",
    "repay_status_1",
    "bill_amount_1",
    "payment_amount_1",
]


@dataclass
class ChangedFeature:
    feature: str
    label: str
    before: Any
    after: Any
    display_before: str
    display_after: str
    delta: Optional[float]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "feature": self.feature,
            "label": self.label,
            "before": self.before,
            "after": self.after,
            "display_before": self.display_before,
            "display_after": self.display_after,
            "delta": self.delta,
        }


@dataclass
class ScenarioComparison:
    baseline_inputs: Dict[str, Any]
    alternative_inputs: Dict[str, Any]
    baseline_probability: float
    alternative_probability: float
    baseline_risk_score: float
    alternative_risk_score: float
    baseline_category: str
    alternative_category: str
    probability_change_points: float
    risk_score_change: float
    category_transition: str
    changed_features: List[ChangedFeature] = field(default_factory=list)
    baseline_derived: Dict[str, float] = field(default_factory=dict)
    alternative_derived: Dict[str, float] = field(default_factory=dict)
    narrative: str = ""
    caveats: List[str] = field(default_factory=list)
    baseline_model: str = ""
    generated_at: str = ""

    @property
    def direction(self) -> str:
        if self.probability_change_points > 0.0001:
            return "increased"
        if self.probability_change_points < -0.0001:
            return "decreased"
        return "unchanged"

    def to_frame(self):
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "Input": item.label,
                    "Baseline": item.display_before,
                    "Alternative": item.display_after,
                }
                for item in self.changed_features
            ]
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "baseline_inputs": self.baseline_inputs,
            "alternative_inputs": self.alternative_inputs,
            "baseline_probability": self.baseline_probability,
            "alternative_probability": self.alternative_probability,
            "baseline_risk_score": self.baseline_risk_score,
            "alternative_risk_score": self.alternative_risk_score,
            "baseline_category": self.baseline_category,
            "alternative_category": self.alternative_category,
            "probability_change_points": self.probability_change_points,
            "risk_score_change": self.risk_score_change,
            "category_transition": self.category_transition,
            "changed_features": [item.as_dict() for item in self.changed_features],
            "narrative": self.narrative,
            "caveats": list(self.caveats),
            "baseline_model": self.baseline_model,
            "generated_at": self.generated_at,
        }


def adjustment_bounds(feature: str) -> tuple:
    return config.INPUT_RANGES.get(feature, (0.0, 1.0))


def apply_adjustments(
    baseline_inputs: Dict[str, Any], adjustments: Dict[str, Any], strict: bool = True
) -> Dict[str, Any]:
    """Copy the baseline, override the supplied fields, and re-validate."""
    candidate = dict(baseline_inputs)
    for feature, value in adjustments.items():
        if feature not in config.USER_INPUT_FEATURES:
            continue
        if value is None:
            continue
        candidate[feature] = value
    cleaned, report = validate_user_inputs(candidate, strict=strict)
    if strict and not report.is_valid:
        raise InputValidationError(report)
    return cleaned


def _display(feature: str, value: Any) -> str:
    if feature in config.CATEGORICAL_FEATURES:
        return config.EDUCATION_DISPLAY.get(str(value), str(value))
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if feature in config.REPAY_STATUS_COLUMNS:
        return f"{int(number)} - {config.REPAY_STATUS_CODES.get(int(number), 'undocumented')}"
    return f"NT$ {number:,.0f}"


def compare_scenarios(
    baseline_inputs: Dict[str, Any],
    adjustments: Dict[str, Any],
    predictor,
    explain: bool = False,
) -> ScenarioComparison:
    """Score the baseline and the adjusted profile and describe the difference."""
    from src.risk_scoring import score_prediction

    clean_baseline, _ = validate_user_inputs(baseline_inputs, strict=True)
    alternative = apply_adjustments(clean_baseline, adjustments, strict=True)

    baseline_result = predictor.predict_one(clean_baseline, explain=False, strict=True)
    alternative_result = predictor.predict_one(alternative, explain=False, strict=True)

    changed: List[ChangedFeature] = []
    for feature in config.USER_INPUT_FEATURES:
        before = clean_baseline[feature]
        after = alternative[feature]
        same = (
            str(before) == str(after)
            if feature in config.CATEGORICAL_FEATURES
            else abs(float(before) - float(after)) < 1e-9
        )
        if same:
            continue
        delta: Optional[float]
        try:
            delta = float(after) - float(before)
        except (TypeError, ValueError):
            delta = None
        changed.append(
            ChangedFeature(
                feature=feature,
                label=feature_label(feature),
                before=before,
                after=after,
                display_before=_display(feature, before),
                display_after=_display(feature, after),
                delta=delta,
            )
        )

    probability_change = (alternative_result.probability - baseline_result.probability) * 100.0
    baseline_score = score_prediction(baseline_result.probability)
    alternative_score = score_prediction(alternative_result.probability)

    narrative = (
        f"Under the alternative input scenario, the model predicts a "
        f"{'higher' if probability_change > 0 else 'lower' if probability_change < 0 else 'unchanged'} "
        f"default probability ({baseline_result.probability * 100:.2f}% -> "
        f"{alternative_result.probability * 100:.2f}%, a change of "
        f"{probability_change:+.2f} percentage points)."
        if abs(probability_change) > 0.005
        else "The alternative scenario produced no measurable change in the predicted "
        "default probability."
    )

    return ScenarioComparison(
        baseline_inputs=clean_baseline,
        alternative_inputs=alternative,
        baseline_probability=baseline_result.probability,
        alternative_probability=alternative_result.probability,
        baseline_risk_score=baseline_score.risk_score,
        alternative_risk_score=alternative_score.risk_score,
        baseline_category=baseline_score.risk_category,
        alternative_category=alternative_score.risk_category,
        probability_change_points=probability_change,
        risk_score_change=alternative_score.risk_score - baseline_score.risk_score,
        category_transition=category_transition(
            baseline_score.risk_category, alternative_score.risk_category
        ),
        changed_features=changed,
        baseline_derived=dict(baseline_result.derived),
        alternative_derived=dict(alternative_result.derived),
        narrative=narrative,
        baseline_model=baseline_result.model_name,
        generated_at=alternative_result.predicted_at,
        caveats=[
            "This is a model-based comparison for the same fitted model, not a causal "
            "statement. It does not promise that changing these inputs in the real world "
            "causes the same change in default risk.",
            "The original stored assessment is unchanged. Saving the alternative creates a "
            "separate record only if you explicitly ask for it.",
        ],
    )


def typical_scenario_from_frame(frame, index: int = 0) -> Dict[str, Any]:
    """Build a supported baseline scenario from a dataset row (clearly labelled)."""
    row = frame.iloc[index]
    return {feature: row[feature] for feature in config.USER_INPUT_FEATURES}


def illustrative_profile(frame) -> Dict[str, Any]:
    """A profile built from dataset medians/modes.

    This is a *statistical summary of the dataset*, not a real applicant record and
    not fabricated customer data. Every value is reproducible from the dataset's own
    descriptive statistics, and the UI always labels it as illustrative.
    """
    import pandas as pd

    profile: Dict[str, Any] = {}
    for feature in config.USER_INPUT_FEATURES:
        if feature in config.CATEGORICAL_FEATURES:
            modes = frame[feature].mode(dropna=True)
            profile[feature] = str(modes.iloc[0]) if len(modes) else config.EDUCATION_CATEGORIES[0]
        else:
            series = pd.to_numeric(frame[feature], errors="coerce")
            value = float(series.median())
            if feature in config.REPAY_STATUS_COLUMNS:
                value = float(int(round(value)))
            profile[feature] = value
    return profile


def repayment_schedule(
    principal: float, annual_interest_rate: float, term_months: int
) -> Dict[str, float]:
    """Standard amortising-loan payment.

    This dataset contains no interest rate or loan term, so any figure produced
    here is a hypothetical estimate entered by the user and is labelled as such.
    """
    principal = float(principal)
    term_months = int(term_months)
    annual = float(annual_interest_rate)
    if principal < 0 or term_months <= 0 or annual < 0:
        raise ValueError("Principal must be >= 0, term must be > 0 and rate must be >= 0.")

    monthly_rate = annual / 100.0 / 12.0
    if monthly_rate == 0:
        monthly_payment = principal / term_months
    else:
        factor = (1.0 + monthly_rate) ** term_months
        monthly_payment = principal * monthly_rate * factor / (factor - 1.0)

    total_repayment = monthly_payment * term_months
    return {
        "monthly_payment": monthly_payment,
        "total_repayment": total_repayment,
        "total_interest": total_repayment - principal,
        "principal": principal,
        "term_months": term_months,
        "annual_interest_rate": annual,
    }
