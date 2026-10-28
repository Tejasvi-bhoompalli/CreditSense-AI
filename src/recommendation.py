"""Transparent, rule-based loan recommendation engine.

Every branch is a documented rule operating on measured model output and
available features. Nothing here claims to be bank policy, nothing approves or
rejects a real loan, and no protected attribute is ever used.

Rules (also displayed on the About & Methodology page):

R0  predicted probability > 0.70                      -> ELEVATED RISK
R1  predicted probability in (0.50, 0.70]             -> MANUAL UNDERWRITING REVIEW
R2  predicted probability in (0.30, 0.50]             -> CONDITIONAL REVIEW
R3  predicted probability <= 0.30                     -> LOWER RISK / FURTHER REVIEW

Escalation modifiers applied on top of the probability band:
M1  >= 1 severely delayed month (status >= 3), or >= 4 delayed months -> at least MANUAL
M2  >= 3 delayed months, or repayment ratio < 0.02                   -> at least CONDITIONAL
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src import config
from src.risk_scoring import RiskScore
from src.utils import format_currency_twd, format_probability

ESCALATION_ORDER = ["eligible", "conditional", "manual", "elevated"]


@dataclass
class Recommendation:
    key: str
    label: str
    headline: str
    reasons: List[str] = field(default_factory=list)
    supporting_indicators: List[str] = field(default_factory=list)
    concerns: List[str] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)
    rules_applied: List[str] = field(default_factory=list)
    disclaimer: str = (
        "This is an automated, rule-based prototype recommendation. It does not approve, "
        "decline or price a loan and must be reviewed by a qualified human underwriter "
        "together with verified documents and applicable local regulation."
    )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "headline": self.headline,
            "reasons": list(self.reasons),
            "supporting_indicators": list(self.supporting_indicators),
            "concerns": list(self.concerns),
            "next_steps": list(self.next_steps),
            "rules_applied": list(self.rules_applied),
            "disclaimer": self.disclaimer,
        }


def _escalate(index: int) -> int:
    return min(index, len(ESCALATION_ORDER) - 1)


def _number(inputs: Dict[str, Any], key: str, default: Optional[float] = None) -> Optional[float]:
    value = inputs.get(key, default)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def recommend(
    probability: float,
    risk: RiskScore,
    inputs: Optional[Dict[str, Any]] = None,
    derived: Optional[Dict[str, float]] = None,
) -> Recommendation:
    """Produce the documented recommendation for one scored borrower."""
    inputs = dict(inputs or {})
    derived = dict(derived or {})
    rules = config.RECOMMENDATION_RULES

    probability = float(probability)
    if probability <= rules["lower_risk_max_probability"]:
        key = "eligible"
        rules_applied = ["R3: predicted probability <= 0.30"]
    elif probability <= rules["conditional_max_probability"]:
        key = "conditional"
        rules_applied = ["R2: predicted probability in (0.30, 0.50]"]
    elif probability <= rules["manual_review_max_probability"]:
        key = "manual"
        rules_applied = ["R1: predicted probability in (0.50, 0.70]"]
    else:
        key = "elevated"
        rules_applied = ["R0: predicted probability > 0.70"]

    severity = ESCALATION_ORDER.index(key)
    reasons: List[str] = [
        f"The model's predicted default probability for this profile is "
        f"{format_probability(probability)}, placing it in the {risk.risk_category} band "
        f"(model-derived risk score {risk.risk_score:.1f}/100)."
    ]

    # --- Escalation modifiers ------------------------------------------
    severe_months = derived.get("num_severely_delayed_months")
    delayed_months = derived.get("num_delayed_months")
    repayment_ratio = derived.get("repayment_ratio")
    max_delay = derived.get("max_repayment_delay")
    utilization = derived.get("credit_utilization")

    if severe_months is not None and (
        severe_months >= 1
        or (delayed_months or 0) >= 4
    ):
        if severity < ESCALATION_ORDER.index("manual"):
            key = "manual"
            severity = ESCALATION_ORDER.index("manual")
        rules_applied.append(
            "M1: severe repayment delay observed "
            f"({int(severe_months)} month(s) with status >= 3, worst status {int(max_delay or 0)})"
        )
        reasons.append(
            "Repayment behaviour shows a severe or repeated delay pattern, which is the "
            "dominant risk signal in this dataset."
        )
    elif delayed_months is not None and delayed_months >= 3:
        if severity < ESCALATION_ORDER.index("conditional"):
            key = "conditional"
            severity = ESCALATION_ORDER.index("conditional")
        rules_applied.append(f"M2: {int(delayed_months)} months with a delayed payment")

    if repayment_ratio is not None and repayment_ratio < rules["min_tolerable_repayment_ratio"]:
        if severity < ESCALATION_ORDER.index("conditional"):
            key = "conditional"
            severity = ESCALATION_ORDER.index("conditional")
        rules_applied.append(
            f"M2: repayment ratio {repayment_ratio:.3f} is below the "
            f"{rules['min_tolerable_repayment_ratio']:.2f} reference level"
        )

    # --- Supporting indicators ----------------------------------------
    supporting: List[str] = []
    if max_delay is not None and max_delay <= 0:
        supporting.append(
            "No repayment delay appears in the six observed months "
            f"(worst status {int(max_delay)})."
        )
    if delayed_months is not None:
        supporting.append(f"{int(delayed_months)} of the last 6 months show a delayed payment.")
    if repayment_ratio is not None:
        supporting.append(
            f"Payments covered {repayment_ratio * 100:.1f}% of the outstanding statement "
            "balances over the observed six months."
        )
    if utilization is not None:
        supporting.append(
            f"Latest balance is {utilization * 100:.1f}% of the granted credit limit."
        )
    limit = _number(inputs, "credit_limit")
    if limit is not None:
        supporting.append(f"Granted credit limit is {format_currency_twd(limit)}.")

    # --- Concerns ------------------------------------------------------
    concerns: List[str] = []
    if utilization is not None and utilization > rules["max_tolerable_credit_utilization"]:
        concerns.append(
            f"Latest balance uses {utilization * 100:.1f}% of the credit limit, above the "
            f"{rules['max_tolerable_credit_utilization'] * 100:.0f}% reference level."
        )
    if repayment_ratio is not None and repayment_ratio < rules["min_tolerable_repayment_ratio"]:
        concerns.append(
            "Repayments cover only a small share of the outstanding balances, so the balance "
            "is not reducing over time."
        )
    if severe_months is not None and severe_months >= 1:
        concerns.append(
            f"{int(severe_months)} month(s) show a severe repayment delay (status >= 3)."
        )
    if risk.risk_category == "HIGH RISK":
        concerns.append("The model places this profile in the HIGH RISK band.")
    if not concerns:
        concerns.append("No single rule-based concern was triggered by the available fields.")

    # --- Next steps ----------------------------------------------------
    next_steps: List[str] = []
    if key in {"eligible", "conditional"}:
        next_steps.append("Continue with standard verification of identity and income documents.")
    if key in {"manual", "elevated"}:
        next_steps.append("Route the file to manual underwriting review before any decision.")
    if concerns:
        next_steps.append("Request supporting documentation for the flagged indicators above.")
    if utilization is not None and utilization > rules["max_tolerable_credit_utilization"]:
        next_steps.append(
            "Consider discussing a lower requested exposure or a reduced credit line; this is a "
            "scenario to explore, not a guaranteed method of lowering real default risk."
        )
    next_steps.append(
        "Record the human review outcome. The model has no authority to approve or decline."
    )

    return Recommendation(
        key=key,
        label=config.RECOMMENDATION_LABELS[key],
        headline=_headline(key),
        reasons=reasons,
        supporting_indicators=supporting,
        concerns=concerns,
        next_steps=next_steps,
        rules_applied=rules_applied,
    )


def _headline(key: str) -> str:
    return {
        "eligible": (
            "The measured risk indicators are comparatively favourable. Eligibility for "
            "further review does not mean the loan will be approved."
        ),
        "conditional": (
            "The profile can proceed only after the flagged conditions are resolved and "
            "documented."
        ),
        "manual": (
            "The measured indicators require a human underwriter to review the file in detail."
        ),
        "elevated": (
            "The measured indicators are unfavourable; additional assessment and a "
            "documented human decision are required."
        ),
    }[key]
