"""Narrative generation for an assessment.

Priority order (the application works fully with none of the optional parts):

1. Deterministic, template-based narrative - always available, always safe.
2. Optional local Ollama model, when the operator enables it.
3. Optional OpenAI-compatible endpoint, only when credentials are supplied.

The machine-learning model remains the single source of truth for every
quantity. The language model receives a fixed, verified fact pack and is only
asked to phrase it. Any number that appears in generated text but not in the
fact pack invalidates the response, and the deterministic template is used
instead. Secrets are never logged and never rendered.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src import config
from src.utils import get_logger, utc_now_iso

logger = get_logger(__name__)

NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")

SYSTEM_INSTRUCTION = (
    "You are writing a short credit-risk summary for an academic prototype. "
    "Use ONLY the facts provided in the JSON block. Do not invent, estimate or "
    "restate any number that is not present in the JSON. Do not approve, decline "
    "or price a loan. Do not mention any protected personal characteristic. "
    "Write three short paragraphs of plain professional English: (1) what the model "
    "predicted, (2) which measured factors drove it, (3) what a human reviewer should "
    "do next. Never claim causality and never claim the model is authoritative."
)


@dataclass
class NarrativeResult:
    text: str
    source: str
    model: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    generated_at: str = field(default_factory=utc_now_iso)

    @property
    def source_label(self) -> str:
        return {
            "template": "Deterministic template (no language model used)",
            "template_fallback": "Deterministic template (language model output rejected)",
            "ollama": f"Local Ollama model ({self.model})",
            "llm_api": f"Configured LLM endpoint ({self.model})",
        }.get(self.source, self.source)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "source": self.source,
            "source_label": self.source_label,
            "model": self.model,
            "warnings": list(self.warnings),
            "generated_at": self.generated_at,
        }


def build_facts(
    prediction,
    recommendation=None,
    explanation=None,
    scenario=None,
    metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the verified fact pack. Every value here is measured."""
    facts: Dict[str, Any] = {
        "predicted_default_probability_percent": round(prediction.probability * 100, 2),
        "risk_score_out_of_100": round(prediction.risk.risk_score, 1),
        "risk_category": prediction.risk.risk_category,
        "model_name": prediction.model_name,
        "model_version": prediction.model_version,
        "score_type": prediction.score_type,
        "dataset_name": str(config.DATASET_INFO["name"]),
        "dataset_geography": str(config.DATASET_INFO["geography"]),
        "dataset_period": str(config.DATASET_INFO["observation_period"]),
        "derived_indicators": {
            key: (round(value, 4) if isinstance(value, float) else value)
            for key, value in (prediction.derived or {}).items()
        },
    }
    if recommendation is not None:
        facts["recommendation_label"] = recommendation.label
        facts["recommendation_reasons"] = list(recommendation.reasons)
        facts["recommendation_concerns"] = list(recommendation.concerns)
        facts["recommendation_next_steps"] = list(recommendation.next_steps)
    if explanation is not None:
        facts["explanation_method"] = explanation.method_label
        facts["top_factors_increasing_risk"] = [
            {
                "feature": item.feature,
                "value": item.display_value,
                "contribution_log_odds": round(item.contribution, 4),
            }
            for item in explanation.top_increasing[: config.TOP_FACTORS]
        ]
        facts["top_factors_reducing_risk"] = [
            {
                "feature": item.feature,
                "value": item.display_value,
                "contribution_log_odds": round(item.contribution, 4),
            }
            for item in explanation.top_decreasing[: config.TOP_FACTORS]
        ]
    if scenario is not None:
        facts["what_if"] = {
            "baseline_probability_percent": round(scenario.baseline_probability * 100, 2),
            "alternative_probability_percent": round(scenario.alternative_probability * 100, 2),
            "probability_change_points": round(scenario.probability_change_points, 2),
            "category_transition": scenario.category_transition,
        }
    if metrics:
        facts["measured_test_metrics"] = {
            "roc_auc": round(float(metrics.get("roc_auc", float("nan"))), 4),
            "pr_auc": round(float(metrics.get("pr_auc", float("nan"))), 4),
            "accuracy": round(float(metrics.get("accuracy", float("nan"))), 4),
            "precision": round(float(metrics.get("precision", float("nan"))), 4),
            "recall": round(float(metrics.get("recall", float("nan"))), 4),
            "brier_score": round(float(metrics.get("brier_score", float("nan"))), 4),
            "test_records": int(metrics.get("n", 0)),
        }
    facts["disclosures"] = {
        "geographic": config.GEOGRAPHIC_DISCLOSURE,
        "prototype": config.APP_DISCLAIMER,
        "shap_caveat": (
            "Contributions describe model behaviour for this input and are not causal effects."
        ),
    }
    return facts


def template_narrative(facts: Dict[str, Any]) -> str:
    """Deterministic prose built only from the fact pack."""
    probability = facts["predicted_default_probability_percent"]
    score = facts["risk_score_out_of_100"]
    category = facts["risk_category"]
    recommendation = facts.get("recommendation_label", "n/a")

    first = (
        f"The fitted model ({facts['model_name']} v{facts['model_version']}) estimates a "
        f"default probability of {probability}% for this profile, which corresponds to a "
        f"model-derived risk score of {score} out of 100 and places the profile in the "
        f"{category} band. The rule-based engine returns: {recommendation}. "
        f"These values were computed by the trained pipeline; none of them is illustrative."
    )

    increasing = facts.get("top_factors_increasing_risk", [])
    decreasing = facts.get("top_factors_reducing_risk", [])
    driver_parts: List[str] = []
    if increasing:
        top = increasing[0]
        driver_parts.append(
            f"The strongest measured upward contributor was {top['feature']} "
            f"(value {top['value']}, contribution {top['contribution_log_odds']:+.4f} log-odds)"
        )
    if len(increasing) > 1:
        driver_parts.append(
            "followed by "
            + ", ".join(
                f"{item['feature']} ({item['contribution_log_odds']:+.4f})"
                for item in increasing[1:4]
            )
        )
    if decreasing:
        driver_parts.append(
            "Factors pulling the estimate downward were "
            + ", ".join(
                f"{item['feature']} ({item['contribution_log_odds']:+.4f})"
                for item in decreasing[:3]
            )
        )
    if driver_parts:
        second = (
            f"Explanation method: {facts.get('explanation_method', 'not computed')}. "
            + ". ".join(driver_parts)
            + ". These contributions describe how the model formed this estimate for this "
            "input; they are not causal effects."
        )
    else:
        second = (
            "No feature-contribution explanation was attached to this assessment, so no "
            "driver statement is made."
        )

    third_bits = [
        f"A human underwriter should review this file together with verified documents and "
        f"the applicable local policy. The model has no authority to approve or decline credit.",
    ]
    if facts.get("recommendation_next_steps"):
        third_bits.append(
            "Suggested next steps from the rule engine: "
            + "; ".join(facts["recommendation_next_steps"][:3])
            + "."
        )
    third_bits.append(
        f"Scope note: the training data is the {facts['dataset_name']} dataset collected in "
        f"{facts['dataset_geography']} ({facts['dataset_period']}). It is not an "
        f"India-specific lending dataset, and these predictions are not validated for Indian "
        f"borrowers."
    )
    third = " ".join(third_bits)
    return "\n\n".join([first, second, third])


def _allowed_numbers(facts: Dict[str, Any]) -> set:
    """Every numeric token that legitimately appears inside the fact pack."""
    serialised = json.dumps(facts, default=str)
    allowed = set(NUMBER_PATTERN.findall(serialised))
    # Also allow the rounded forms a writer might reasonably produce.
    for token in list(allowed):
        try:
            value = float(token)
        except ValueError:
            continue
        allowed.add(f"{value:.0f}")
        allowed.add(f"{value:.1f}")
        allowed.add(f"{value:.2f}")
    return allowed


def validate_narrative(text: str, facts: Dict[str, Any]) -> tuple:
    """Reject generated text containing any number absent from the fact pack."""
    allowed = _allowed_numbers(facts)
    offenders = [
        token for token in NUMBER_PATTERN.findall(text) if token not in allowed
    ]
    return (not offenders), sorted(set(offenders))[:10]


def _call_ollama(prompt: str) -> tuple:
    payload = json.dumps(
        {
            "model": config.OLLAMA_MODEL,
            "prompt": f"{SYSTEM_INSTRUCTION}\n\nFACTS (JSON):\n{prompt}",
            "stream": False,
            "options": {"temperature": 0.2},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{config.OLLAMA_HOST.rstrip('/')}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=config.OLLAMA_TIMEOUT_SECONDS) as response:
        body = json.loads(response.read().decode("utf-8"))
    return str(body.get("response", "")).strip(), config.OLLAMA_MODEL


def _call_llm_api(prompt: str) -> tuple:
    payload = json.dumps(
        {
            "model": config.LLM_MODEL,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": f"FACTS (JSON):\n{prompt}"},
            ],
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{config.LLM_API_BASE.rstrip('/')}/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.LLM_API_KEY}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=config.OLLAMA_TIMEOUT_SECONDS) as response:
        body = json.loads(response.read().decode("utf-8"))
    text = body["choices"][0]["message"]["content"]
    return str(text).strip(), config.LLM_MODEL


def generate_narrative(
    prediction,
    recommendation=None,
    explanation=None,
    scenario=None,
    metrics: Optional[Dict[str, Any]] = None,
    prefer_llm: bool = True,
) -> NarrativeResult:
    """Produce the narrative, degrading safely to the template at every step."""
    facts = build_facts(prediction, recommendation, explanation, scenario, metrics)
    warnings: List[str] = []
    fallback_text = template_narrative(facts)

    if not prefer_llm:
        return NarrativeResult(text=fallback_text, source="template")

    prompt = json.dumps(facts, indent=2, default=str)
    providers = []
    if config.OLLAMA_ENABLED:
        providers.append(("ollama", _call_ollama))
    if config.LLM_API_BASE and config.LLM_API_KEY and config.LLM_MODEL:
        providers.append(("llm_api", _call_llm_api))

    if not providers:
        return NarrativeResult(
            text=fallback_text,
            source="template",
            warnings=[
                "No language model is configured. The deterministic template is used; this "
                "is the default and fully supported mode."
            ],
        )

    for source, caller in providers:
        try:
            text, model_name = caller(prompt)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            warnings.append(f"{source} is unavailable ({type(exc).__name__}); using template.")
            continue
        except Exception as exc:  # pragma: no cover - provider specific
            warnings.append(f"{source} failed ({type(exc).__name__}); using template.")
            continue

        if not text:
            warnings.append(f"{source} returned an empty response; using template.")
            continue

        valid, offenders = validate_narrative(text, facts)
        if not valid:
            warnings.append(
                f"{source} produced values that are not present in the verified fact pack "
                f"({offenders}); that response was rejected and the deterministic template "
                "was used instead."
            )
            continue
        return NarrativeResult(text=text, source=source, model=model_name, warnings=warnings)

    return NarrativeResult(text=fallback_text, source="template_fallback", warnings=warnings)


def narrative_provider_status() -> Dict[str, Any]:
    """Describe which narrative providers are configured, without leaking secrets."""
    return {
        "template": {"available": True, "label": "Deterministic template"},
        "ollama": {
            "available": bool(config.OLLAMA_ENABLED),
            "label": f"Ollama at {config.OLLAMA_HOST} (model {config.OLLAMA_MODEL})",
        },
        "llm_api": {
            "available": bool(config.LLM_API_BASE and config.LLM_API_KEY and config.LLM_MODEL),
            "label": "Configured OpenAI-compatible endpoint",
        },
    }
