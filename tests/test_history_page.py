"""Regression test for a crash on the Assessment History page.

The page read the raw ``top_risk_factors_json`` / ``protective_factors_json``
columns and iterated them. The database layer keeps those columns as JSON
*strings* (the parsed values live in ``top_risk_factors`` /
``protective_factors``), so iterating the raw column yielded characters and the
page raised ``AttributeError: 'str' object has no attribute 'get'`` for every
record that had stored factors.

Only rendering the real page catches that, so this test drives the page through
Streamlit's own AppTest harness with the model bundle and database redirected to
temporary fixtures.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src import config, database
from src.explainability import FeatureContribution
from src.recommendation import recommend

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _StubExplanation:
    """Minimal explanation-shaped object, so the stored factors are deterministic."""

    method_label = "test stub explanation"

    def __init__(self) -> None:
        self.top_increasing = [
            FeatureContribution(
                feature="repay_status_1",
                label="Repayment status - most recent month (Sept)",
                value=2.0,
                display_value="2 (payment delayed by 2 months)",
                contribution=0.5123,
                direction="increases_risk",
            )
        ]
        self.top_decreasing = [
            FeatureContribution(
                feature="credit_limit",
                label="Credit limit granted (NT$)",
                value=50000.0,
                display_value="NT$ 50,000",
                contribution=-0.2071,
                direction="decreases_risk",
            )
        ]


@pytest.fixture()
def stored_record(predictor, valid_inputs, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "history_page_test.db")
    monkeypatch.setattr(config, "MODEL_BUNDLE_PATH", predictor.bundle_path)
    database.init_db()
    result = predictor.predict_one(valid_inputs, explain=False, strict=True)
    recommendation = recommend(
        result.probability, result.risk, result.inputs, result.derived
    )
    identifier = database.insert_assessment(
        prediction=result,
        recommendation=recommendation,
        application_ref="APP-REGRESSION-1",
        explanation=_StubExplanation(),
    )
    return identifier


def test_stored_factors_are_dicts_not_a_raw_json_string(stored_record):
    """The rows the page reads must be mappings, never a JSON string."""
    record = database.get_assessment(stored_record)
    assert isinstance(record["top_risk_factors_json"], str)  # the trap
    assert isinstance(record["top_risk_factors"], list)
    assert isinstance(record["protective_factors"], list)
    for item in record["top_risk_factors"] + record["protective_factors"]:
        assert isinstance(item, dict)
        assert item.get("label", item.get("feature"))  # the page's access pattern


def test_history_page_renders_a_record_that_has_stored_factors(stored_record):
    """The page previously crashed here; it must render the factor lines."""
    if not (PROJECT_ROOT / "data").exists():
        pytest.skip("no data directory available for the dataset requirement")
    app_test = pytest.importorskip("streamlit.testing.v1")

    record = database.get_assessment(stored_record)
    assert record["top_risk_factors"] and record["protective_factors"]

    app = app_test.AppTest.from_file(
        str(PROJECT_ROOT / "pages" / "assessment_history.py"), default_timeout=240
    ).run()

    exceptions = [item.value for item in app.exception]
    assert not exceptions, exceptions

    rendered = "\n".join(item.value for item in app.markdown)
    assert "Stored factors increasing risk" in rendered
    assert "Stored factors reducing risk" in rendered
    assert "Repayment status - most recent month (Sept)" in rendered
    assert record["assessment_id"] in rendered
    # the raw JSON string must never be what the page iterates
    assert json.dumps(record["top_risk_factors"])[:20] not in rendered
