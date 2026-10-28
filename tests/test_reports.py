"""PDF report generation and narrative safety."""

from __future__ import annotations

import io

import joblib
import pytest

from src import config
from src.ai_report import (
    build_facts,
    generate_narrative,
    narrative_provider_status,
    template_narrative,
    validate_narrative,
)
from src.predictor import CreditRiskPredictor
from src.recommendation import recommend
from src.report_generator import (
    ReportError,
    build_assessment_report,
    metrics_context,
    report_filename,
)
from src.what_if import compare_scenarios


@pytest.fixture(scope="module")
def trained_predictor(trained_result, tmp_path_factory):
    """A predictor built from the session's real training output."""
    bundle = dict(trained_result["bundle"])
    bundle.update(
        {
            "feature_names": list(config.USER_INPUT_FEATURES),
            "model_features": list(config.MODEL_FEATURES),
            "model_version": config.MODEL_VERSION,
            "schema_version": config.SCHEMA_VERSION,
            "trained_at": trained_result["metadata"]["trained_at"],
            "random_seed": config.RANDOM_SEED,
            "target_column": config.TARGET_COLUMN,
            "dataset_fingerprint": trained_result["metadata"]["dataset_fingerprint"],
            "dataset_filename": "synthetic_test_dataset.csv",
            "is_synthetic_dataset": True,
            "test_metrics": trained_result["metrics"]["selected"]["test"],
            "calibration_info": trained_result["metadata"]["calibration"],
        }
    )
    path = tmp_path_factory.mktemp("trained-bundle") / "model_bundle.joblib"
    joblib.dump(bundle, path, compress=("zlib", 3))
    return CreditRiskPredictor(bundle_path=path).load()


@pytest.fixture(scope="module")
def assessment(trained_predictor, valid_inputs):
    result = trained_predictor.predict_one(valid_inputs, explain=True, strict=True)
    recommendation = recommend(result.probability, result.risk, result.inputs, result.derived)
    return result, recommendation


def _context(assessment, trained_result, **overrides) -> dict:
    result, recommendation = assessment
    context = {
        "assessment_id": "CSA-TEST0000001",
        "generated_at": result.predicted_at,
        "inputs": result.inputs,
        "derived": result.derived,
        "probability": result.probability,
        "risk_score": result.risk.risk_score,
        "risk_category": result.risk.risk_category,
        "recommendation": recommendation,
        "recommendation_label": recommendation.label,
        "explanation": result.explanation,
        "model_name": result.model_name,
        "model_version": result.model_version,
        "dataset": trained_result["metadata"]["dataset"],
        "metrics": metrics_context(trained_result["metrics"], trained_result["metadata"]),
    }
    context.update(overrides)
    return context


def _pdf_text(pdf_bytes: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(pdf_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def test_report_is_a_valid_pdf(assessment, trained_result):
    pdf_bytes = build_assessment_report(_context(assessment, trained_result))
    assert pdf_bytes.startswith(b"%PDF")
    assert pdf_bytes.rstrip().endswith(b"%%EOF")
    assert len(pdf_bytes) > 5_000


def test_report_contains_branding_attribution_and_disclaimers(assessment, trained_result):
    context = _context(assessment, trained_result)
    text = _pdf_text(build_assessment_report(context))

    assert "CreditSense AI" in text
    assert context["assessment_id"] in text
    assert "Taiwan" in text
    assert "Geographic scope" in text
    assert "not a bank-issued credit report" in text
    assert "academic research and decision-support prototype" in text
    assert "Risk score" in text
    assert context["recommendation"].label.replace("-", "-") in text.replace("\n", " ")
    assert f"{context['probability'] * 100:.2f}%" in text


def test_report_includes_measured_metrics(assessment, trained_result):
    context = _context(assessment, trained_result)
    text = _pdf_text(build_assessment_report(context))
    assert "Model evaluation context" in text
    assert "ROC-AUC" in text
    assert f"{context['metrics']['test_roc_auc']:.4f}" in text


def test_report_without_explanation_is_still_valid(assessment, trained_result):
    context = _context(assessment, trained_result, explanation=None)
    text = _pdf_text(build_assessment_report(context))
    assert "No explanation was attached" in text


def test_report_with_missing_metrics_is_still_valid(assessment, trained_result):
    context = _context(assessment, trained_result, metrics={})
    text = _pdf_text(build_assessment_report(context))
    assert "metrics were not available" in text


def test_report_includes_what_if_comparison(assessment, trained_result, trained_predictor, valid_inputs):
    scenario = compare_scenarios(
        valid_inputs,
        {"credit_limit": valid_inputs["credit_limit"] * 1.5},
        trained_predictor,
    )
    context = _context(assessment, trained_result, scenario=scenario)
    text = _pdf_text(build_assessment_report(context))
    assert "What-if scenario comparison" in text
    assert "Baseline" in text and "Alternative" in text


def test_report_handles_long_content(assessment, trained_result):
    """A very long application reference must not break the layout."""
    context = _context(
        assessment,
        trained_result,
        assessment_id="CSA-" + "X" * 300,
        explanatory_note="padding " * 400,
    )
    pdf_bytes = build_assessment_report(context)
    assert pdf_bytes.startswith(b"%PDF")
    assert len(_pdf_text(pdf_bytes)) > 500


def test_report_filename_is_sanitised(assessment, trained_result):
    name = report_filename(
        {"assessment_id": "../../etc/passwd;C:\\evil name"}
    )
    assert name.startswith("CreditSenseAI-")
    assert name.endswith(".pdf")
    assert "/" not in name and "\\" not in name and ".." not in name


def test_metrics_context_pulls_measured_values(trained_result):
    context = metrics_context(trained_result["metrics"], trained_result["metadata"])
    test = trained_result["metrics"]["selected"]["test"]
    assert context["test_roc_auc"] == pytest.approx(test["roc_auc"])
    assert context["test_n"] == test["n"]
    assert context["test_false_positives"] == test["false_positives"]
    assert "Taiwan" not in context["interpretation"]  # no invented claims here
    assert "held-out test records" in context["interpretation"]


def test_metrics_context_empty_without_artifacts():
    assert metrics_context({}, {}) == {}


# ---------------------------------------------------------------------------
# Narrative safety
# ---------------------------------------------------------------------------


def test_fact_pack_and_template_use_measured_values(assessment, trained_result):
    result, recommendation = assessment
    facts = build_facts(
        result,
        recommendation=recommendation,
        explanation=result.explanation,
        metrics=trained_result["metrics"]["selected"]["test"],
    )
    assert facts["predicted_default_probability_percent"] == pytest.approx(
        round(result.probability * 100, 2)
    )
    text = template_narrative(facts)
    assert f"{facts['predicted_default_probability_percent']}" in text
    assert facts["risk_category"] in text
    assert "Taiwan" in text
    assert "not an India-specific lending dataset" in text


def test_template_without_explanation_does_not_invent_drivers(assessment):
    result, recommendation = assessment
    facts = build_facts(result, recommendation=recommendation)
    text = template_narrative(facts)
    assert "No feature-contribution explanation was attached" in text


def test_validate_narrative_rejects_invented_numbers(assessment):
    result, recommendation = assessment
    facts = build_facts(result, recommendation=recommendation)
    assert validate_narrative("The predicted probability is consistent with the facts.", facts)[0]
    valid, offenders = validate_narrative(
        "The borrower has a 97.63% chance of default and 512 previous defaults.", facts
    )
    assert valid is False
    assert "97.63" in offenders


def test_generate_narrative_uses_template_by_default(assessment, trained_result):
    result, recommendation = assessment
    narrative = generate_narrative(
        result, recommendation=recommendation, explanation=result.explanation
    )
    assert narrative.source in {"template", "template_fallback"}
    assert narrative.text
    assert "template" in narrative.source_label.lower()
    assert narrative.as_dict()["source"] == narrative.source


def test_generate_narrative_respects_preference(assessment):
    result, recommendation = assessment
    narrative = generate_narrative(
        result, recommendation=recommendation, prefer_llm=False
    )
    assert narrative.source == "template"


def test_provider_status_never_exposes_secrets():
    status = narrative_provider_status()
    assert set(status) == {"template", "ollama", "llm_api"}
    assert status["template"]["available"] is True
    serialised = str(status)
    if config.LLM_API_KEY:
        assert config.LLM_API_KEY not in serialised
    assert "api_key" not in serialised.lower()
    assert "bearer" not in serialised.lower()
