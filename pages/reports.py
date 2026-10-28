"""Reports: narrative generation and the downloadable PDF assessment report."""

from __future__ import annotations

import streamlit as st

from src import config, database
from src.ai_report import generate_narrative, narrative_provider_status
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
    require_model,
    section,
)
from src.utils import format_probability


def _record_context(record: dict, predictor, explanation, narrative_source: str) -> dict:
    inputs = record.get("inputs") or {}
    return {
        "assessment_id": record["assessment_id"],
        "generated_at": record["created_at"],
        "inputs": inputs,
        "derived": record.get("derived") or {},
        "probability": float(record["probability"]),
        "risk_score": float(record["risk_score"]),
        "risk_category": record["risk_category"],
        "recommendation": None,
        "recommendation_label": record.get("recommendation_label"),
        "explanation": explanation,
        "model_name": record.get("model_name"),
        "model_version": record.get("model_version"),
        "dataset": get_cached_metadata().get("dataset", {}),
        "metrics": metrics_context(get_cached_metrics(), get_cached_metadata()),
        "narrative": st.session_state.get("report_narrative_text"),
        "narrative_source": narrative_source,
        "scenario": st.session_state.get("report_include_scenario")
        and st.session_state.get("scenario_comparison") or None,
        "is_demo": bool(record.get("is_demo")),
    }


def render() -> None:
    status = model_status()
    page_header(
        "Reports",
        "Turn a stored assessment into a narrative summary and a professionally typeset PDF. "
        "The language model (when configured) only phrases measured values; it never invents "
        "them, and the deterministic template is always available.",
        eyebrow="Reporting",
        model_status=status,
    )

    identifiers = database.assessment_ids(limit=300)
    last_result = st.session_state.get("assessment_result")

    if not identifiers and not last_result:
        notice(
            "There is no stored assessment to report on yet. Save one from the Risk Assessment "
            "page first; this page will not invent a report for a record that does not exist.",
            kind="warning",
            strong="Nothing to report.",
        )
        st.page_link("pages/risk_assessment.py", label="Start New Assessment",
                     icon=":material/assignment:")
        return

    options = ["Latest assessment computed in this session"] if last_result else []
    options += [f"Stored assessment {identifier}" for identifier in identifiers]
    choice = st.selectbox("Assessment to report on", options=options, key="rp_choice")

    if choice.startswith("Stored"):
        identifier = choice.replace("Stored assessment ", "").strip()
        record = database.get_assessment(identifier)
        if record is None:
            notice("That record is no longer in the database.", kind="danger")
            st.stop()
        inputs = record.get("inputs") or {}
        result = None
        probability = float(record["probability"])
        risk_score = float(record["risk_score"])
        risk_category = record["risk_category"]
        model_name = record.get("model_name")
        model_version = record.get("model_version")
        is_demo = bool(record.get("is_demo"))
        stored_created = record["created_at"]
        derived = record.get("derived") or {}
    else:
        payload = last_result
        result = payload["result"]
        record = None
        inputs = result.inputs
        derived = result.derived
        probability = result.probability
        risk_score = result.risk.risk_score
        risk_category = result.risk.risk_category
        model_name = result.model_name
        model_version = result.model_version
        is_demo = False
        stored_created = result.predicted_at

    st.markdown(
        f"""
        <div class="csa-card">
          <div class="csa-card-title">
            {'Stored assessment ' + record['assessment_id'] if record else 'Session assessment'}
          </div>
          <div class="csa-card-body">
            <b>Created (UTC)</b> {stored_created}<br/>
            <b>Model</b> {model_name} v{model_version}<br/>
            <b>Predicted default probability</b> {format_probability(probability)}<br/>
            <b>Risk score</b> {risk_score:.1f} / 100 &middot; <b>Category</b> {risk_category}<br/>
            {'<b>Demonstration record</b> yes' if is_demo else ''}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if is_demo:
        notice(
            "This is a labelled demonstration record built from dataset rows, not a real "
            "applicant.",
            kind="warning",
        )

    predictor = None
    if status.get("ready"):
        try:
            predictor = require_model()
        except Exception:  # pragma: no cover - defensive
            predictor = None

    # ---- Explanation (needed for the report and the narrative) ----------
    section("Feature explanation used in the report")
    st.caption(
        "Contributions are recomputed with the active model, because a stored record keeps "
        "only a short summary of its factors."
    )
    cache_key = f"report_explanation::{choice}"
    if predictor is not None:
        if st.button("Compute feature explanation", key="rp_explain"):
            with st.spinner("Computing feature attributions..."):
                try:
                    st.session_state[cache_key] = predictor.explain(
                        inputs, probability=probability
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    st.session_state[cache_key] = None
                    st.error(f"Explanation unavailable ({type(exc).__name__}).")
        explanation = st.session_state.get(cache_key)
        if explanation is not None:
            st.caption(f"Method: {explanation.method_label}")
            ranked = explanation.ranked(8)
            st.dataframe(explanation.to_frame().head(8), use_container_width=True,
                         hide_index=True)
            left, right = st.columns(2)
            with left:
                st.markdown("**Increases risk**")
                for item in explanation.top_increasing[:5]:
                    st.markdown(f"- {item.label} ({item.display_value}): {item.contribution:+.5f}")
            with right:
                st.markdown("**Reduces risk**")
                for item in explanation.top_decreasing[:5]:
                    st.markdown(f"- {item.label} ({item.display_value}): {item.contribution:+.5f}")
        else:
            notice(
                "No explanation computed yet. The PDF will still be generated and will state "
                "that no explanation was attached.",
                kind="info",
            )
    else:
        explanation = None
        notice("The model is not available, so no explanation can be computed.", kind="warning")

    # ---- Narrative ------------------------------------------------------
    section("Narrative summary")
    providers = narrative_provider_status()
    st.caption(
        " · ".join(
            f"{name}: {'configured' if details['available'] else 'not configured'}"
            for name, details in providers.items()
        )
    )
    st.caption(
        "Providers are optional. The deterministic template is used by default and requires no "
        "external service, no API key and no network access."
    )

    if predictor is None:
        notice("The model is required to build an accurate narrative.", kind="warning")
    else:
        if st.button("Generate narrative", type="primary"):
            base_prediction = predictor.predict_one(inputs, explain=False, strict=True)
            recommendation = recommend(
                base_prediction.probability,
                base_prediction.risk,
                base_prediction.inputs,
                base_prediction.derived,
            )
            with st.spinner("Writing the narrative..."):
                narrative = generate_narrative(
                    base_prediction,
                    recommendation=recommendation,
                    explanation=st.session_state.get(cache_key),
                    scenario=st.session_state.get("scenario_comparison"),
                    metrics=get_cached_metrics().get("selected", {}).get("test"),
                )
            st.session_state["report_narrative"] = narrative
            st.session_state["report_narrative_text"] = narrative.text
            st.session_state["report_narrative_source"] = narrative.source_label
            if record is not None:
                try:
                    database.set_narrative_source(record["assessment_id"], narrative.source)
                except database.DatabaseError:
                    st.caption("The narrative source could not be written back to the record.")

        narrative = st.session_state.get("report_narrative")
        if narrative:
            st.markdown(
                f"<div class='csa-card'><div class='csa-kpi-label'>Source</div>"
                f"<div class='csa-card-body'>{narrative.source_label}</div></div>",
                unsafe_allow_html=True,
            )
            for warning in narrative.warnings:
                st.caption(f"- {warning}")
            st.markdown(
                "<div class='csa-quote'>"
                + narrative.text.replace("\n\n", "</div><div class='csa-quote'>")
                + "</div>",
                unsafe_allow_html=True,
            )
            st.download_button(
                "Download narrative (Markdown)",
                data=narrative.text.encode("utf-8"),
                file_name=f"creditsense-narrative-{stored_created[:10]}.md",
                mime="text/markdown",
            )
        else:
            notice(
                "Press <strong>Generate narrative</strong>. Nothing is displayed beforehand, "
                "because a written summary of numbers that were never computed would be "
                "misleading.",
                kind="info",
            )

    # ---- PDF ------------------------------------------------------------
    section("PDF assessment report")
    include_scenario = st.checkbox(
        "Include the What-if comparison currently held in this session",
        value=bool(st.session_state.get("scenario_comparison")),
        key="rp_include_scenario",
        disabled=st.session_state.get("scenario_comparison") is None,
    )
    st.session_state["report_include_scenario"] = include_scenario
    st.caption(
        "The report contains the dataset attribution, the measured evaluation context, the "
        "recommendation, the explanation, the data limitations and the academic disclaimer."
    )

    context = {
        "assessment_id": record["assessment_id"] if record else "session-assessment",
        "generated_at": stored_created,
        "inputs": inputs,
        "derived": derived,
        "probability": probability,
        "risk_score": risk_score,
        "risk_category": risk_category,
        "recommendation": None,
        "recommendation_label": record.get("recommendation_label") if record else None,
        "explanation": st.session_state.get(cache_key),
        "model_name": model_name,
        "model_version": model_version,
        "dataset": get_cached_metadata().get("dataset", {}),
        "metrics": metrics_context(get_cached_metrics(), get_cached_metadata()),
        "narrative": st.session_state.get("report_narrative_text"),
        "narrative_source": st.session_state.get("report_narrative_source", "template"),
        "scenario": st.session_state.get("scenario_comparison") if include_scenario else None,
        "is_demo": is_demo,
    }
    if record is not None and not context["recommendation_label"]:
        context["recommendation_label"] = "not stored"

    if st.button("Generate PDF report", type="primary"):
        try:
            with st.spinner("Typesetting the report..."):
                pdf_bytes = build_assessment_report(context)
            st.session_state["report_pdf"] = pdf_bytes
            if record is not None:
                try:
                    database.set_report_available(record["assessment_id"])
                except database.DatabaseError:
                    pass
            st.success("Report generated. Use the download button below.")
        except ReportError as exc:
            st.error(str(exc))

    pdf_bytes = st.session_state.get("report_pdf")
    if pdf_bytes:
        st.download_button(
            "Download PDF report",
            data=pdf_bytes,
            file_name=report_filename(context),
            mime="application/pdf",
        )
        st.caption(f"{len(pdf_bytes) / 1024:.0f} KB. This document is an academic artefact; it "
                   "is not a bank-issued credit report and creates no credit decision.")
    else:
        st.caption("No PDF has been generated in this session yet.")


render()
