"""Assessment history: search, inspect, export, and delete stored assessments."""

from __future__ import annotations

import json
from datetime import date

import pandas as pd
import streamlit as st

from src import config, database
from src.recommendation import recommend
from src.report_generator import (
    ReportError,
    build_assessment_report,
    metrics_context,
    report_filename,
)
from src.ui import (
    get_cached_metadata,
    get_cached_metrics,
    model_status,
    notice,
    page_header,
    require_dataset,
    require_model,
    section,
)
from src.utils import feature_label, format_probability
from src.what_if import illustrative_profile

PAGE_SIZE_OPTIONS = [10, 25, 50, 100]


def _stored_factor_list(value) -> list:
    """Return stored factor rows as a list of dictionaries.

    ``database.get_assessment`` parses the ``*_json`` columns into
    ``top_risk_factors`` / ``protective_factors``. Reading the raw JSON column
    instead would iterate the string character by character, so a JSON string is
    parsed here too and any element that is not a mapping is dropped.
    """
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError, ValueError):
            return []
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _stored_report_bytes(record: dict, predictor, recompute_explanation: bool) -> bytes:
    inputs = record.get("inputs") or {}
    explanation = None
    if recompute_explanation and predictor is not None:
        try:
            explanation = predictor.explain(inputs)
        except Exception:  # pragma: no cover - explanation is optional in the PDF
            explanation = None

    risk_score = float(record["risk_score"])
    risk_category = record["risk_category"]
    probability = float(record["probability"])
    derived = record.get("derived") or {}

    stored_factors = record.get("top_risk_factors_json") or []
    context = {
        "assessment_id": record["assessment_id"],
        "generated_at": record["created_at"],
        "inputs": inputs,
        "derived": derived,
        "probability": probability,
        "risk_score": risk_score,
        "risk_category": risk_category,
        "recommendation": None,
        "recommendation_label": record.get("recommendation_label"),
        "explanation": explanation,
        "model_name": record.get("model_name"),
        "model_version": record.get("model_version"),
        "dataset": get_cached_metadata().get("dataset", {}),
        "metrics": metrics_context(get_cached_metrics(), get_cached_metadata()),
        "is_demo": bool(record.get("is_demo")),
    }
    if not explanation and stored_factors:
        context["stored_risk_factors"] = stored_factors
    return build_assessment_report(context)


def render() -> None:
    status = model_status()
    page_header(
        "Assessment History",
        "Every record here was created by an explicit save action. The database stores no "
        "names, addresses, e-mail addresses or national identifiers - only the assessment "
        "fields and the model's output.",
        eyebrow="Local database",
        model_status=status,
    )

    if not database.is_available():
        notice("The local SQLite database is unavailable, so no history can be shown.",
               kind="danger")
        st.stop()

    total = database.count_assessments()
    if total == 0:
        st.markdown(
            "<div class='csa-empty'><div class='csa-empty-title'>No assessments stored yet"
            "</div><div class='csa-empty-text'>Save an assessment from the Risk Assessment "
            "page to build a history. Demonstration records are never inserted automatically; "
            "you can add clearly-labelled ones below if you want to see the pages populated."
            "</div></div>",
            unsafe_allow_html=True,
        )
        st.page_link("pages/risk_assessment.py", label="Start New Assessment",
                     icon=":material/assignment:")

        section("Demonstration records (optional)")
        st.caption(
            "These records are built from dataset rows and marked with an application "
            "reference beginning with DEMO. They are labelled 'demonstration' everywhere they "
            "appear and should be deleted before any real use."
        )
        if st.button("Insert 5 labelled demonstration records"):
            try:
                predictor = require_model()
                dataset = require_dataset()
                sample = dataset.frame.sample(n=5, random_state=config.RANDOM_SEED)
                scenarios = [
                    {feature: row[feature] for feature in config.USER_INPUT_FEATURES}
                    for _, row in sample.iterrows()
                ]
                inserted = database.seed_demonstration_records(predictor, scenarios)
                st.success(f"Inserted {inserted} labelled demonstration records.")
                st.rerun()
            except Exception as exc:  # pragma: no cover - defensive
                st.error(f"Could not insert demonstration records ({type(exc).__name__}).")
        return

    # ---- Filters --------------------------------------------------------
    section("Filters")
    columns = st.columns([1.3, 1, 1, 1])
    with columns[0]:
        search = st.text_input("Search ID or application reference", key="ah_search")
    with columns[1]:
        categories = st.multiselect(
            "Risk category", options=config.RISK_CATEGORIES,
            default=list(config.RISK_CATEGORIES), key="ah_categories",
        )
    with columns[2]:
        date_from = st.date_input("From", value=None, key="ah_from")
    with columns[3]:
        date_to = st.date_input("To", value=None, key="ah_to")

    columns = st.columns([1.4, 1, 1, 1])
    with columns[0]:
        probability_range = st.slider(
            "Predicted probability range (%)", min_value=0.0, max_value=100.0,
            value=(0.0, 100.0), step=1.0, key="ah_probability",
        )
    with columns[1]:
        sort_label = st.selectbox(
            "Sort by", options=["Newest first", "Oldest first", "Highest risk score",
                                "Lowest risk score", "Risk category"],
            key="ah_sort",
        )
    with columns[2]:
        page_size = st.selectbox("Rows per page", options=PAGE_SIZE_OPTIONS, index=1,
                                 key="ah_page_size")
    with columns[3]:
        include_demo = st.checkbox("Include demonstration records", value=True,
                                   key="ah_include_demo")

    sort_map = {
        "Newest first": ("created_at", False),
        "Oldest first": ("created_at", True),
        "Highest risk score": ("risk_score", False),
        "Lowest risk score": ("risk_score", True),
        "Risk category": ("risk_category", False),
    }
    sort_by, ascending = sort_map[sort_label]

    filters = dict(
        search=search or None,
        risk_category=None,
        date_from=date_from.isoformat() if isinstance(date_from, date) else None,
        date_to=date_to.isoformat() if isinstance(date_to, date) else None,
        min_probability=probability_range[0] / 100.0,
        max_probability=probability_range[1] / 100.0,
    )

    filtered_count = database.count_assessments(**filters)
    pages = max(1, (filtered_count + page_size - 1) // page_size)
    page_number = st.number_input(
        "Page", min_value=1, max_value=pages, value=1, step=1, key="ah_page"
    )
    offset = (int(page_number) - 1) * page_size

    frame = database.list_assessments(
        **filters, sort_by=sort_by, ascending=ascending, limit=page_size, offset=offset
    )
    if categories and len(categories) < len(config.RISK_CATEGORIES):
        frame = frame[frame["risk_category"].isin(categories)]
    if not include_demo:
        frame = frame[frame["is_demo"] == 0]

    st.caption(f"{filtered_count:,} record(s) match the filters. Showing page "
               f"{int(page_number)} of {pages}.")

    if frame.empty:
        notice("No stored assessment matches the current filters.", kind="warning")
    else:
        display = frame.copy()
        display["predicted probability"] = display["probability"].map(
            lambda value: f"{value * 100:.2f}%"
        )
        display["created"] = pd.to_datetime(
            display["created_at"], errors="coerce", utc=True
        ).dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(
            display[
                ["assessment_id", "created", "application_ref", "model_name",
                 "predicted probability", "risk_score", "risk_category",
                 "recommendation_label", "is_demo"]
            ].rename(
                columns={
                    "assessment_id": "Assessment ID",
                    "application_ref": "Application reference",
                    "model_name": "Model",
                    "risk_score": "Risk score",
                    "risk_category": "Risk category",
                    "recommendation_label": "Recommendation",
                    "is_demo": "Demonstration",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    # ---- Export ---------------------------------------------------------
    section("Export metadata")
    export_frame = database.list_assessments(
        **filters, sort_by=sort_by, ascending=ascending, limit=100000, offset=0
    )
    if not include_demo and not export_frame.empty:
        export_frame = export_frame[export_frame["is_demo"] == 0]
    st.download_button(
        "Download assessment metadata (CSV)",
        data=database.export_csv_bytes(export_frame),
        file_name="creditsense-assessments.csv",
        mime="text/csv",
    )
    st.caption(
        "The export contains assessment metadata only (identifier, timestamp, model, "
        "probability, score, category, recommendation). Feature-level inputs are not "
        "included, to limit unnecessary data replication."
    )

    # ---- Detail view ----------------------------------------------------
    section("Inspect a record")
    identifiers = list(frame["assessment_id"]) if not frame.empty else database.assessment_ids(200)
    if not identifiers:
        return
    selected = st.selectbox("Assessment ID", options=identifiers, key="ah_selected")
    record = database.get_assessment(selected)
    if record is None:
        notice("That record could not be loaded.", kind="warning")
        return

    if record.get("is_demo"):
        notice(
            "This is a labelled <strong>demonstration</strong> record built from dataset rows. "
            "It does not describe a real applicant.",
            kind="warning",
        )

    left, right = st.columns([1, 1])
    with left:
        st.markdown(
            f"""
            <div class="csa-card">
              <div class="csa-card-title">{record['assessment_id']}</div>
              <div class="csa-card-body">
                <b>Created (UTC)</b> {record['created_at']}<br/>
                <b>Application reference</b> {record.get('application_ref') or 'not set'}<br/>
                <b>Model</b> {record['model_name']} v{record['model_version']}<br/>
                <b>Predicted probability</b> {record['probability'] * 100:.2f}%
                (uncalibrated {float(record.get('probability_uncalibrated') or 0) * 100:.2f}%)<br/>
                <b>Risk score</b> {record['risk_score']:.1f} / 100<br/>
                <b>Risk category</b> {record['risk_category']}<br/>
                <b>Recommendation</b> {record.get('recommendation_label') or 'n/a'}<br/>
                <b>Explanation method</b> {record.get('explanation_method') or 'not stored'}<br/>
                <b>Narrative source</b> {record.get('narrative_source') or 'not generated'}<br/>
                <b>Dataset</b> {record.get('dataset_name') or 'n/a'}
                {'(synthetic)' if record.get('dataset_is_synthetic') else ''}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        inputs = record.get("inputs") or {}
        if inputs:
            rows = []
            for feature in config.USER_INPUT_FEATURES:
                if feature not in inputs:
                    continue
                value = inputs[feature]
                if feature in config.CATEGORICAL_FEATURES:
                    display_value = config.EDUCATION_DISPLAY.get(str(value), str(value))
                elif feature in config.REPAY_STATUS_COLUMNS:
                    display_value = (
                        f"{int(float(value))} - "
                        f"{config.REPAY_STATUS_CODES.get(int(float(value)), '')}"
                    )
                else:
                    display_value = f"NT$ {float(value):,.0f}"
                rows.append({"Input": feature_label(feature), "Value": display_value})
            st.dataframe(rows, use_container_width=True, hide_index=True)

    factors = _stored_factor_list(
        record.get("top_risk_factors", record.get("top_risk_factors_json"))
    )
    protective = _stored_factor_list(
        record.get("protective_factors", record.get("protective_factors_json"))
    )
    if factors or protective:
        left, right = st.columns(2)
        with left:
            st.markdown("**Stored factors increasing risk**")
            for item in factors:
                st.markdown(f"- {item.get('label', item.get('feature'))}: "
                            f"{item.get('contribution', 0):+.5f}")
        with right:
            st.markdown("**Stored factors reducing risk**")
            for item in protective:
                st.markdown(f"- {item.get('label', item.get('feature'))}: "
                            f"{item.get('contribution', 0):+.5f}")

    # ---- PDF ------------------------------------------------------------
    section("Report for this record")
    recompute = st.checkbox(
        "Recompute the feature explanation for the report (uses the active model)",
        value=status.get("ready", False),
        key="ah_recompute",
        help="Contributions are not stored per record; recomputing uses the same model that "
             "produced the original prediction.",
    )
    predictor = None
    if recompute and status.get("ready"):
        try:
            predictor = require_model()
        except Exception:  # pragma: no cover - defensive
            predictor = None
    try:
        pdf_bytes = _stored_report_bytes(record, predictor, recompute)
        st.download_button(
            "Download PDF report",
            data=pdf_bytes,
            file_name=report_filename({"assessment_id": record["assessment_id"]}),
            mime="application/pdf",
        )
        st.caption(
            "Generating the report does not change the stored assessment. The PDF is an "
            "academic artefact and is explicitly not a bank-issued credit report."
        )
    except ReportError as exc:
        notice(f"The PDF could not be generated: {exc}", kind="danger")

    # ---- Delete ---------------------------------------------------------
    section("Delete a record")
    st.caption(
        "Deletion is permanent and cannot be undone. It requires an explicit confirmation."
    )
    confirm = st.checkbox(
        f"I understand that deleting {selected} permanently removes it",
        key="ah_confirm_delete",
    )
    if st.button("Delete assessment", disabled=not confirm, type="primary"):
        try:
            deleted = database.delete_assessment(selected, confirm=confirm)
            if deleted:
                st.success(f"Deleted {selected}.")
                st.rerun()
            else:
                st.warning("That record was already removed.")
        except database.DatabaseError as exc:
            st.error(str(exc))


render()
