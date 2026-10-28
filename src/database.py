"""SQLite persistence for assessment history.

Security and correctness notes:

* every statement uses bound parameters - no string interpolation of values,
* sorting columns are chosen from a fixed allow-list,
* raw database exceptions are converted into a friendly message so that schema
  details are never shown to a user,
* no API key, token or password is ever written to the database,
* no unnecessary personal identifier is stored (no name, address, e-mail or
  national ID field exists in the schema).
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from src import config
from src.utils import get_logger, utc_now_iso

logger = get_logger(__name__)

SCHEMA_VERSION = 1
SORTABLE_COLUMNS = {
    "created_at": "created_at",
    "risk_score": "risk_score",
    "probability": "probability",
    "risk_category": "risk_category",
    "assessment_id": "assessment_id",
}

_SCHEMA_STATEMENTS: List[str] = [
    """
    CREATE TABLE IF NOT EXISTS schema_meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS assessments (
        assessment_id           TEXT PRIMARY KEY,
        created_at              TEXT NOT NULL,
        application_ref         TEXT,
        model_name              TEXT NOT NULL,
        model_version           TEXT NOT NULL,
        dataset_name            TEXT,
        dataset_version         TEXT,
        dataset_is_synthetic    INTEGER NOT NULL DEFAULT 0,
        inputs_json             TEXT NOT NULL,
        derived_json            TEXT,
        probability             REAL NOT NULL,
        probability_uncalibrated REAL,
        risk_score              REAL NOT NULL,
        risk_category           TEXT NOT NULL,
        recommendation_key      TEXT,
        recommendation_label    TEXT,
        top_risk_factors_json   TEXT,
        protective_factors_json TEXT,
        explanation_method      TEXT,
        narrative_source        TEXT,
        report_available        INTEGER NOT NULL DEFAULT 0,
        is_demo                 INTEGER NOT NULL DEFAULT 0
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_assessments_created ON assessments(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_assessments_category ON assessments(risk_category)",
]

_STATS_COLUMNS = [
    "assessment_id",
    "created_at",
    "application_ref",
    "model_name",
    "model_version",
    "risk_score",
    "probability",
    "risk_category",
    "recommendation_label",
    "is_demo",
]


class DatabaseError(RuntimeError):
    """Friendly wrapper so no raw SQL detail reaches the interface."""


@contextmanager
def connection(db_path: Optional[Path] = None):
    path = Path(db_path or config.DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=20.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        yield conn
        conn.commit()
    except sqlite3.Error as exc:
        conn.rollback()
        logger.error("Database operation failed: %s", exc)
        raise DatabaseError(
            "The local assessment database could not be updated. The file may be locked by "
            "another process or the disk may be unavailable. No data was changed."
        ) from exc
    finally:
        conn.close()


def init_db(db_path: Optional[Path] = None) -> None:
    with connection(db_path) as conn:
        for statement in _SCHEMA_STATEMENTS:
            conn.execute(statement)
        current = conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'schema_version'"
        ).fetchone()
        if current is None:
            conn.execute(
                "INSERT INTO schema_meta (key, value) VALUES (?, ?)",
                ("schema_version", str(SCHEMA_VERSION)),
            )
        elif int(current["value"]) != SCHEMA_VERSION:
            logger.info(
                "Existing database schema version %s (application expects %s); additive "
                "statements keep the schema compatible.",
                current["value"],
                SCHEMA_VERSION,
            )


def is_available() -> bool:
    try:
        init_db()
        return True
    except DatabaseError:
        return False


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------


def insert_assessment(
    prediction,
    recommendation=None,
    application_ref: Optional[str] = None,
    explanation: Optional[Any] = None,
    narrative_source: Optional[str] = None,
    report_available: bool = False,
    is_demo: bool = False,
    dataset_name: Optional[str] = None,
    dataset_version: Optional[str] = None,
    dataset_is_synthetic: bool = False,
    assessment_id: Optional[str] = None,
) -> str:
    """Persist one assessment. Returns the stable assessment id.

    Callers must only invoke this on an explicit user action; re-running a page
    therefore cannot duplicate a record.
    """
    init_db()
    identifier = assessment_id or f"CSA-{uuid.uuid4().hex[:12].upper()}"

    top_factors: List[Dict[str, Any]] = []
    protective: List[Dict[str, Any]] = []
    explanation_method = None
    if explanation is not None:
        explanation_method = getattr(explanation, "method_label", None) or getattr(
            explanation, "method", None
        )
        top_factors = [
            {"feature": item.feature, "label": item.label, "contribution": round(item.contribution, 6)}
            for item in getattr(explanation, "top_increasing", [])[:5]
        ]
        protective = [
            {"feature": item.feature, "label": item.label, "contribution": round(item.contribution, 6)}
            for item in getattr(explanation, "top_decreasing", [])[:5]
        ]

    record = (
        identifier,
        utc_now_iso(),
        (application_ref or "").strip()[:120] or None,
        prediction.model_name,
        prediction.model_version,
        dataset_name,
        dataset_version,
        1 if dataset_is_synthetic else 0,
        json.dumps(prediction.inputs, default=str),
        json.dumps(prediction.derived, default=str),
        float(prediction.probability),
        float(prediction.probability_uncalibrated),
        float(prediction.risk.risk_score),
        prediction.risk.risk_category,
        getattr(recommendation, "key", None),
        getattr(recommendation, "label", None),
        json.dumps(top_factors, default=str),
        json.dumps(protective, default=str),
        explanation_method,
        narrative_source,
        1 if report_available else 0,
        1 if is_demo else 0,
    )

    with connection() as conn:
        conn.execute(
            """
            INSERT INTO assessments (
                assessment_id, created_at, application_ref, model_name, model_version,
                dataset_name, dataset_version, dataset_is_synthetic, inputs_json, derived_json,
                probability, probability_uncalibrated, risk_score, risk_category,
                recommendation_key, recommendation_label, top_risk_factors_json,
                protective_factors_json, explanation_method, narrative_source,
                report_available, is_demo
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            record,
        )
    return identifier


def delete_assessment(assessment_id: str, confirm: bool = False) -> bool:
    """Delete one record. Requires an explicit confirmation flag."""
    if not confirm:
        raise DatabaseError("Deletion requires explicit confirmation.")
    init_db()
    with connection() as conn:
        cursor = conn.execute(
            "DELETE FROM assessments WHERE assessment_id = ?", (assessment_id,)
        )
        return cursor.rowcount > 0


def set_report_available(assessment_id: str) -> None:
    init_db()
    with connection() as conn:
        conn.execute(
            "UPDATE assessments SET report_available = 1 WHERE assessment_id = ?",
            (assessment_id,),
        )


def set_narrative_source(assessment_id: str, source: str) -> None:
    init_db()
    with connection() as conn:
        conn.execute(
            "UPDATE assessments SET narrative_source = ? WHERE assessment_id = ?",
            (source, assessment_id),
        )


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


def _build_filters(
    search: Optional[str] = None,
    risk_category: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    min_probability: Optional[float] = None,
    max_probability: Optional[float] = None,
) -> tuple:
    clauses: List[str] = []
    params: List[Any] = []
    if search:
        clauses.append("(assessment_id LIKE ? OR IFNULL(application_ref, '') LIKE ?)")
        pattern = f"%{search.strip()}%"
        params.extend([pattern, pattern])
    if risk_category and risk_category != "All":
        clauses.append("risk_category = ?")
        params.append(risk_category)
    if date_from:
        clauses.append("created_at >= ?")
        params.append(f"{date_from}T00:00:00+00:00")
    if date_to:
        clauses.append("created_at <= ?")
        params.append(f"{date_to}T23:59:59+00:00")
    if min_probability is not None:
        clauses.append("probability >= ?")
        params.append(float(min_probability))
    if max_probability is not None:
        clauses.append("probability <= ?")
        params.append(float(max_probability))
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    return where, params


def list_assessments(
    search: Optional[str] = None,
    risk_category: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    min_probability: Optional[float] = None,
    max_probability: Optional[float] = None,
    sort_by: str = "created_at",
    ascending: bool = False,
    limit: int = 200,
    offset: int = 0,
) -> pd.DataFrame:
    init_db()
    where, params = _build_filters(
        search, risk_category, date_from, date_to, min_probability, max_probability
    )
    column = SORTABLE_COLUMNS.get(sort_by, "created_at")
    direction = "ASC" if ascending else "DESC"
    # Deterministic tie-breakers: without them, rows sharing a sort key can move
    # between pages on consecutive queries.
    query = (
        f"SELECT {', '.join(_STATS_COLUMNS)} FROM assessments {where} "
        f"ORDER BY {column} {direction}, created_at DESC, assessment_id ASC "
        f"LIMIT ? OFFSET ?"
    )
    bound = list(params) + [int(limit), int(offset)]
    with connection() as conn:
        frame = pd.read_sql_query(query, conn, params=bound)
    return frame


def count_assessments(
    search: Optional[str] = None,
    risk_category: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    min_probability: Optional[float] = None,
    max_probability: Optional[float] = None,
) -> int:
    init_db()
    where, params = _build_filters(
        search, risk_category, date_from, date_to, min_probability, max_probability
    )
    with connection() as conn:
        row = conn.execute(
            f"SELECT COUNT(*) AS total FROM assessments {where}", params
        ).fetchone()
    return int(row["total"]) if row else 0


def get_assessment(assessment_id: str) -> Optional[Dict[str, Any]]:
    init_db()
    with connection() as conn:
        row = conn.execute(
            "SELECT * FROM assessments WHERE assessment_id = ?", (assessment_id,)
        ).fetchone()
    if row is None:
        return None
    record = dict(row)
    for key in ("inputs_json", "derived_json", "top_risk_factors_json", "protective_factors_json"):
        if record.get(key):
            try:
                record[key.replace("_json", "")] = json.loads(record[key])
            except (json.JSONDecodeError, TypeError):
                record[key.replace("_json", "")] = {}
    return record


def assessment_ids(limit: int = 500) -> List[str]:
    init_db()
    with connection() as conn:
        rows = conn.execute(
            "SELECT assessment_id FROM assessments ORDER BY created_at DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
    return [row["assessment_id"] for row in rows]


def get_statistics() -> Dict[str, Any]:
    """Aggregates for the Overview KPIs - always computed from stored rows."""
    init_db()
    with connection() as conn:
        total_row = conn.execute("SELECT COUNT(*) AS total FROM assessments").fetchone()
        total = int(total_row["total"]) if total_row else 0
        stats: Dict[str, Any] = {
            "total": total,
            "by_category": {category: 0 for category in config.RISK_CATEGORIES},
            "average_probability": None,
            "average_risk_score": None,
            "median_risk_score": None,
            "demo_records": 0,
            "user_records": 0,
            "earliest": None,
            "latest": None,
            "high_risk_share": None,
        }
        if total == 0:
            return stats

        rows = conn.execute(
            "SELECT risk_category, COUNT(*) AS n FROM assessments GROUP BY risk_category"
        ).fetchall()
        for row in rows:
            stats["by_category"][row["risk_category"]] = int(row["n"])

        summary = conn.execute(
            """
            SELECT AVG(probability) AS avg_probability,
                   AVG(risk_score)   AS avg_risk_score,
                   MIN(created_at)   AS earliest,
                   MAX(created_at)   AS latest,
                   SUM(is_demo)      AS demo_records
            FROM assessments
            """
        ).fetchone()
        stats["average_probability"] = (
            float(summary["avg_probability"]) if summary["avg_probability"] is not None else None
        )
        stats["average_risk_score"] = (
            float(summary["avg_risk_score"]) if summary["avg_risk_score"] is not None else None
        )
        stats["earliest"] = summary["earliest"]
        stats["latest"] = summary["latest"]
        stats["demo_records"] = int(summary["demo_records"] or 0)
        stats["user_records"] = total - stats["demo_records"]
        stats["high_risk_share"] = stats["by_category"].get("HIGH RISK", 0) / total
        return stats


def get_assessment_frame(limit: int = 2000) -> pd.DataFrame:
    """Full stored records (for dashboard distributions), ordered newest first."""
    init_db()
    with connection() as conn:
        frame = pd.read_sql_query(
            "SELECT * FROM assessments ORDER BY created_at DESC LIMIT ?", conn, params=[int(limit)]
        )
    if frame.empty:
        return frame
    for column in ("inputs_json", "derived_json"):
        if column in frame.columns:
            parsed = frame[column].map(_safe_json)
            frame[column.replace("_json", "")] = parsed
    return frame


def _safe_json(value: Any) -> Dict[str, Any]:
    if not value:
        return {}
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return {}


def export_csv_bytes(frame: Optional[pd.DataFrame] = None) -> bytes:
    """CSV export of assessment metadata (no feature-level personal data)."""
    if frame is None:
        frame = list_assessments(limit=100000)
    if frame.empty:
        return "no assessments stored\n".encode("utf-8")
    return frame.to_csv(index=False).encode("utf-8")


def seed_demonstration_records(predictor, rows: Iterable[Dict[str, Any]], application_prefix: str = "DEMO") -> int:
    """Insert clearly-labelled demonstration records on an explicit user action.

    Every record is written with ``is_demo = 1`` and an application reference
    beginning with "DEMO" so that it can never be mistaken for a real
    assessment. Nothing is seeded automatically.
    """
    from src.recommendation import recommend

    inserted = 0
    for index, scenario in enumerate(rows):
        result = predictor.predict_one(scenario, explain=False, strict=True)
        recommendation = recommend(
            result.probability, result.risk, result.inputs, result.derived
        )
        insert_assessment(
            prediction=result,
            recommendation=recommendation,
            application_ref=f"{application_prefix}-{index + 1:03d}",
            is_demo=True,
            dataset_name=predictor.dataset_info.get("name"),
            dataset_version=predictor.model_version,
            dataset_is_synthetic=bool(
                predictor.dataset_info.get("is_synthetic", False)
            ),
        )
        inserted += 1
    return inserted
