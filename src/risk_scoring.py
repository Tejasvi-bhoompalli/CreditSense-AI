"""Risk scoring: probability -> score -> category.

The mapping is defined exactly once, here, and imported by every page so the
same borrower can never receive two different categories in two places.

Important: the bands below are *presentation* thresholds chosen for this
academic prototype. They are not official banking thresholds and must not be
described as such.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from src import config


@dataclass
class RiskScore:
    probability: float
    risk_score: float
    risk_category: str
    band_lower: float
    band_upper: float

    def as_dict(self) -> Dict[str, object]:
        return {
            "probability": self.probability,
            "risk_score": self.risk_score,
            "risk_category": self.risk_category,
            "band_lower": self.band_lower,
            "band_upper": self.band_upper,
        }


def risk_score_from_probability(probability: float, decimals: int = 1) -> float:
    """risk_score = 100 x predicted default probability, rounded half-up (Python's
    default banker's rounding is avoided so that 0.305 -> 30.5 consistently)."""
    from decimal import Decimal, ROUND_HALF_UP

    p = min(max(float(probability), 0.0), 1.0)
    raw = Decimal(str(p * 100.0))
    quantum = Decimal("1").scaleb(-decimals)
    return float(raw.quantize(quantum, rounding=ROUND_HALF_UP))


def classify_risk_score(risk_score: float) -> str:
    """0-30 LOW, >30-60 MEDIUM, >60-100 HIGH (inclusive upper bound)."""
    score = min(max(float(risk_score), 0.0), 100.0)
    for lower, upper, label in config.RISK_BANDS:
        if lower < score <= upper or (score == 0.0 and lower == 0.0):
            return label
    return config.RISK_BANDS[-1][2]


def band_bounds(risk_category: str) -> tuple:
    for lower, upper, label in config.RISK_BANDS:
        if label == risk_category:
            return lower, upper
    return 0.0, 100.0


def risk_color(risk_category: str) -> str:
    return config.RISK_COLORS.get(risk_category, "#64748B")


def risk_severity_index(risk_category: str) -> int:
    return config.RISK_ORDER.get(risk_category, 0)


def score_prediction(probability: float) -> RiskScore:
    """Single entry point used by every page and by the PDF report."""
    score = risk_score_from_probability(probability)
    category = classify_risk_score(score)
    lower, upper = band_bounds(category)
    return RiskScore(
        probability=float(probability),
        risk_score=score,
        risk_category=category,
        band_lower=lower,
        band_upper=upper,
    )


def category_transition(from_category: str, to_category: str) -> str:
    """Human-readable description of a risk-category movement."""
    start = risk_severity_index(from_category)
    end = risk_severity_index(to_category)
    if start == end:
        return f"Category unchanged ({from_category})"
    direction = "improved" if end < start else "deteriorated"
    return f"Category {direction}: {from_category} -> {to_category}"


def describe_bands() -> str:
    return " | ".join(
        f"{label}: {lower:.0f}-{upper:.0f}" for lower, upper, label in config.RISK_BANDS
    )
