"""SQLite persistence: schema, parameterised queries, filters, deletion."""

from __future__ import annotations

import pytest

from src import config, database
from src.recommendation import recommend


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    """Point the whole database module at a throwaway file."""
    path = tmp_path / "creditsense_test.db"
    monkeypatch.setattr(config, "DB_PATH", path)
    database.init_db()
    return path


@pytest.fixture()
def stored(predictor, valid_inputs, temp_db):
    result = predictor.predict_one(valid_inputs, explain=False)
    recommendation = recommend(result.probability, result.risk, result.inputs, result.derived)
    identifier = database.insert_assessment(
        prediction=result,
        recommendation=recommendation,
        application_ref="APP-TEST-001",
        dataset_name="synthetic-test",
    )
    return identifier, result


def test_schema_is_created_automatically(temp_db):
    assert temp_db.exists()
    assert database.is_available()


def test_insert_and_read_back(stored):
    identifier, result = stored
    record = database.get_assessment(identifier)
    assert record is not None
    assert record["assessment_id"] == identifier
    assert record["probability"] == pytest.approx(result.probability)
    assert record["risk_score"] == pytest.approx(result.risk.risk_score)
    assert record["risk_category"] == result.risk.risk_category
    assert record["application_ref"] == "APP-TEST-001"
    assert set(record["inputs"]) == set(config.USER_INPUT_FEATURES)


def test_stable_id_and_distinct_inserts(predictor, valid_inputs, temp_db):
    first = predictor.predict_one(valid_inputs, explain=False)
    identifiers = {
        database.insert_assessment(prediction=first),
        database.insert_assessment(prediction=first),
    }
    assert len(identifiers) == 2  # each explicit save is its own record
    assert all(identifier.startswith("CSA-") for identifier in identifiers)


def test_count_and_list(stored):
    assert database.count_assessments() == 1
    frame = database.list_assessments()
    assert len(frame) == 1
    assert "assessment_id" in frame.columns


def test_filters_by_risk_category(stored):
    identifier, result = stored
    matching = database.list_assessments(risk_category=result.risk.risk_category)
    assert len(matching) == 1
    other = [
        category for category in config.RISK_CATEGORIES
        if category != result.risk.risk_category
    ][0]
    assert database.list_assessments(risk_category=other).empty


def test_filters_by_probability_range(stored):
    _, result = stored
    below = database.list_assessments(max_probability=result.probability - 0.01)
    assert below.empty or below["probability"].max() <= result.probability - 0.01
    wide = database.list_assessments(min_probability=0.0, max_probability=1.0)
    assert len(wide) == 1


def test_sorting_and_pagination(predictor, valid_inputs, temp_db):
    result = predictor.predict_one(valid_inputs, explain=False)
    for _ in range(4):
        database.insert_assessment(prediction=result)
    ascending = database.list_assessments(sort_by="risk_score", ascending=True, limit=2)
    descending = database.list_assessments(sort_by="risk_score", ascending=False, limit=2)
    assert len(ascending) == 2
    assert len(descending) == 2
    assert list(ascending["risk_score"]) == sorted(ascending["risk_score"])
    page_two = database.list_assessments(limit=2, offset=2)
    assert len(page_two) == 2
    assert set(page_two["assessment_id"]).isdisjoint(set(ascending["assessment_id"]))


def test_unknown_sort_column_falls_back_safely(predictor, valid_inputs, temp_db):
    result = predictor.predict_one(valid_inputs, explain=False)
    database.insert_assessment(prediction=result)
    # A hostile sort key must not reach SQL; it silently falls back.
    frame = database.list_assessments(sort_by="risk_score; DROP TABLE assessments")
    assert len(frame) == 1
    assert database.count_assessments() == 1


def test_search_is_parameterised_and_injection_safe(predictor, valid_inputs, temp_db):
    result = predictor.predict_one(valid_inputs, explain=False)
    database.insert_assessment(prediction=result, application_ref="REF-42")
    assert len(database.list_assessments(search="REF-42")) == 1
    hostile = database.list_assessments(search="' OR 1=1 --")
    assert hostile.empty
    assert database.count_assessments() == 1


def test_delete_requires_confirmation(stored):
    identifier, _ = stored
    with pytest.raises(database.DatabaseError):
        database.delete_assessment(identifier, confirm=False)
    assert database.count_assessments() == 1
    assert database.delete_assessment(identifier, confirm=True) is True
    assert database.count_assessments() == 0
    assert database.get_assessment(identifier) is None


def test_statistics_are_computed_from_rows(stored):
    _, result = stored
    stats = database.get_statistics()
    assert stats["total"] == 1
    assert stats["average_probability"] == pytest.approx(result.probability)
    assert stats["average_risk_score"] == pytest.approx(result.risk.risk_score)
    assert sum(stats["by_category"].values()) == 1
    assert stats["earliest"] and stats["latest"]


def test_empty_statistics_shape(temp_db):
    stats = database.get_statistics()
    assert stats["total"] == 0
    assert stats["average_probability"] is None
    assert set(stats["by_category"]) == set(config.RISK_CATEGORIES)


def test_export_csv(stored):
    payload = database.export_csv_bytes()
    assert isinstance(payload, bytes)
    text = payload.decode("utf-8")
    assert "assessment_id" in text
    assert "risk_category" in text


def test_export_csv_when_empty(temp_db):
    assert b"no assessments stored" in database.export_csv_bytes()


def test_demonstration_records_are_flagged(predictor, sample_frame, temp_db):
    scenarios = [
        {
            feature: row[feature]
            for feature in config.USER_INPUT_FEATURES
        }
        for _, row in sample_frame.head(3).iterrows()
    ]
    inserted = database.seed_demonstration_records(predictor, scenarios)
    assert inserted == 3
    frame = database.list_assessments()
    assert set(frame["is_demo"]) == {1}
    stats = database.get_statistics()
    assert stats["demo_records"] == 3
    assert stats["user_records"] == 0


def test_narrative_and_report_flags_update(stored):
    identifier, _ = stored
    database.set_narrative_source(identifier, "template")
    database.set_report_available(identifier)
    record = database.get_assessment(identifier)
    assert record["narrative_source"] == "template"
    assert record["report_available"] == 1
