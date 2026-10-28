"""Risk assessment workflow.

Only fields that genuinely exist in the training dataset are requested. Derived
ratios are computed by the saved pipeline, never typed in. The assessment is
written to SQLite only when the user clicks Save.
"""

from __future__ import annotations

import json

import streamlit as st

from src import config, database
from src.data_validation import InputValidationError, issues_to_frame
from src.predictor import ModelNotReadyError
from src.recommendation import recommend
from src.ui import (
    notice,
    page_header,
    require_dataset,
    require_model,
    risk_badge,
    risk_meter,
    model_status,
    section,
)
from src.utils import feature_label, format_probability
from src.what_if import illustrative_profile

REPAY_OPTIONS = sorted(config.REPAY_STATUS_CODES.keys())


def repay_label(value: int) -> str:
    return f"{value} - {config.REPAY_STATUS_CODES[value]}"


def _defaults(profile: dict) -> None:
    for feature in config.USER_INPUT_FEATURES:
        st.session_state[f"input_{feature}"] = profile[feature]


def _render_form(profile: dict) -> dict:
    submitted_values: dict = {}
    with st.form("risk_assessment_form", clear_on_submit=False):
        st.markdown(
            "<div class='csa-muted'>Fields marked with a currency unit are in New Taiwan "
            "Dollars (NT$), the currency of the source dataset. Every field is required and "
            "is validated before any prediction is made.</div>",
            unsafe_allow_html=True,
        )

        st.markdown("**A. Account and applicant profile**")
        cols = st.columns(3)
        with cols[0]:
            submitted_values["credit_limit"] = st.number_input(
                feature_label("credit_limit"),
                min_value=float(config.INPUT_RANGES["credit_limit"][0]),
                max_value=float(config.INPUT_RANGES["credit_limit"][1]),
                step=10_000.0,
                key="input_credit_limit",
                help="Amount of credit granted, including supplementary family credit.",
            )
        with cols[1]:
            submitted_values["age"] = st.number_input(
                feature_label("age"),
                min_value=18.0,
                max_value=100.0,
                step=1.0,
                key="input_age",
            )
        with cols[2]:
            submitted_values["education"] = st.selectbox(
                feature_label("education"),
                options=config.EDUCATION_CATEGORIES,
                format_func=lambda value: config.EDUCATION_DISPLAY.get(value, value),
                key="input_education",
                help="Potential proxy variable; included but disclosed as such.",
            )

        st.markdown("**B. Repayment history (most recent six months)**")
        st.markdown(
            "<div class='csa-muted'>Scale: -1 paid in full, 0 revolving credit with the "
            "minimum paid, 1-8 payment delayed by that many months. Codes -2 was not "
            "documented in the source paper but appears in the distributed extract; it is "
            "treated as 'no consumption / not applicable'.</div>",
            unsafe_allow_html=True,
        )
        cols = st.columns(6)
        for index, feature in enumerate(config.REPAY_STATUS_COLUMNS):
            with cols[index]:
                submitted_values[feature] = st.selectbox(
                    feature_label(feature),
                    options=REPAY_OPTIONS,
                    format_func=repay_label,
                    key=f"input_{feature}",
                )

        st.markdown("**C. Statement balances**")
        st.markdown(
            "<div class='csa-muted'>A negative balance means credit in the client's favour "
            "(overpayment or refund). The dataset contains such client-months.</div>",
            unsafe_allow_html=True,
        )
        cols = st.columns(6)
        for index, feature in enumerate(config.BILL_AMOUNT_COLUMNS):
            with cols[index]:
                submitted_values[feature] = st.number_input(
                    feature_label(feature),
                    min_value=float(config.INPUT_RANGES[feature][0]),
                    max_value=float(config.INPUT_RANGES[feature][1]),
                    step=1_000.0,
                    key=f"input_{feature}",
                )

        st.markdown("**D. Amounts paid**")
        cols = st.columns(6)
        for index, feature in enumerate(config.PAYMENT_AMOUNT_COLUMNS):
            with cols[index]:
                submitted_values[feature] = st.number_input(
                    feature_label(feature),
                    min_value=0.0,
                    max_value=float(config.INPUT_RANGES[feature][1]),
                    step=500.0,
                    key=f"input_{feature}",
                )

        buttons = st.columns([1.3, 1, 3.4])
        with buttons[0]:
            submitted = st.form_submit_button(
                "Assess Credit Risk", type="primary", use_container_width=True
            )
        with buttons[1]:
            reset = st.form_submit_button("Reset form", use_container_width=True)

    if reset:
        _defaults(profile)
        st.session_state.assessment_result = None
        st.rerun()

    return submitted_values if submitted else {}


def _result_panel(payload: dict, predictor) -> None:
    result = payload["result"]
    recommendation = payload["recommendation"]
    explanation = payload.get("explanation")

    risk = result.risk
    st.markdown(
        f"<div class='csa-card'><div style='display:flex;justify-content:space-between;"
        f"align-items:center;flex-wrap:wrap;gap:0.6rem'>"
        f"<div><div class='csa-kpi-label'>Predicted default probability</div>"
        f"<div class='csa-kpi-value' style='font-size:2rem'>"
        f"{format_probability(result.probability)}</div>"
        f"<div class='csa-muted'>Risk score {risk.risk_score:.1f} / 100 &middot; "
        f"Model: {result.model_name} v{result.model_version}</div></div>"
        f"<div>{risk_badge(risk.risk_category)}</div></div>",
        unsafe_allow_html=True,
    )
    risk_meter(result.probability, risk.risk_category)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        f"<div class='csa-muted'>Interpretation of the score: "
        f"{result.score_type}. The uncalibrated model output for this profile was "
        f"{format_probability(result.probability_uncalibrated)}; probability calibration is "
        f"fitted on the validation split and is described on the Model Intelligence page. A "
        f"predicted probability is an estimate, not a guarantee, and not a statement about "
        f"any individual's future behaviour.</div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 1])
    with left:
        section("Recommendation")
        st.markdown(f"**{recommendation.label}**")
        st.markdown(recommendation.headline)
        for reason in recommendation.reasons:
            st.markdown(f"- {reason}")
        st.markdown(recommendation.disclaimer)
        with st.expander("Rules applied"):
            for rule in recommendation.rules_applied:
                st.markdown(f"- {rule}")
            st.caption(
                "Thresholds are prototype presentation rules documented on the "
                "About & Methodology page, not bank policy."
            )
    with right:
        section("Next steps and concerns")
        st.markdown("**Key concerns**")
        for item in recommendation.concerns:
            st.markdown(f"- {item}")
        st.markdown("**Suggested next steps**")
        for item in recommendation.next_steps:
            st.markdown(f"- {item}")
        st.markdown("**Supporting indicators**")
        for item in recommendation.supporting_indicators:
            st.markdown(f"- {item}")

    # ---- Explanation ----------------------------------------------------
    section(
        "Why the model produced this estimate",
        "Computed on the same fitted model that produced the prediction above.",
    )
    if explanation is None:
        notice(
            "No explanation could be produced for this prediction. The prediction itself "
            "remains valid; only the explanation is unavailable.",
            kind="warning",
        )
    else:
        from src.explainability import narrate_local
        import plotly.graph_objects as go

        from src.ui import PLOT_COLORS, apply_chart_theme, show_chart

        st.caption(f"Explanation method: {explanation.method_label}")
        ranked = explanation.ranked(10)
        figure = go.Figure(
            go.Bar(
                x=[item.contribution for item in ranked][::-1],
                y=[item.label for item in ranked][::-1],
                orientation="h",
                marker_color=[
                    PLOT_COLORS["high"] if item.contribution > 0 else PLOT_COLORS["low"]
                    for item in ranked
                ][::-1],
                hovertemplate="%{y}<br>contribution %{x:+.4f} log-odds<extra></extra>",
            )
        )
        figure.update_xaxes(title_text="Contribution to the model's raw output (log-odds)")
        show_chart(
            apply_chart_theme(
                figure, height=380,
                title="Top feature contributions for this profile",
                show_legend=False,
            ),
            key="ra_contrib",
        )

        left, right = st.columns(2)
        with left:
            st.markdown("**Factors increasing the predicted risk**")
            if explanation.top_increasing:
                for item in explanation.top_increasing[:5]:
                    st.markdown(
                        f"- **{item.label}** &middot; value {item.display_value} "
                        f"&middot; {item.contribution:+.4f}"
                    )
            else:
                st.markdown("- None of the measured features pushed the estimate upward.")
        with right:
            st.markdown("**Factors reducing the predicted risk**")
            if explanation.top_decreasing:
                for item in explanation.top_decreasing[:5]:
                    st.markdown(
                        f"- **{item.label}** &middot; value {item.display_value} "
                        f"&middot; {item.contribution:+.4f}"
                    )
            else:
                st.markdown("- None of the measured features pulled the estimate downward.")

        st.markdown("**Plain-English summary**")
        for sentence in narrate_local(explanation):
            st.markdown(f"- {sentence}")
        for note in explanation.notes:
            st.caption(note)

    # ---- Derived values and save ---------------------------------------
    with st.expander("Derived indicators computed by the pipeline"):
        derived_rows = []
        for feature in config.DERIVED_FEATURES:
            value = result.derived.get(feature)
            if value is None:
                continue
            if feature in {"credit_utilization", "repayment_ratio"}:
                display = f"{value:.3f}"
            elif feature.startswith("num_") or feature == "max_repayment_delay":
                display = f"{value:.0f}"
            else:
                display = f"NT$ {value:,.0f}"
            derived_rows.append({"Indicator": feature_label(feature), "Value": display})
        st.dataframe(derived_rows, use_container_width=True, hide_index=True)
        st.caption(
            "Formulas are documented in README.md and in artifacts/metadata/feature_schema.json."
        )

    # ---- Save -----------------------------------------------------------
    section("Save this assessment")
    saved_id = st.session_state.get("saved_assessment_id")
    ref = st.text_input(
        "Optional application reference",
        key="save_application_ref",
        max_chars=120,
        placeholder="e.g. APP-2026-0173 (do not enter personal identifiers)",
        help="Stored so you can find the record later. Do not enter names, e-mail addresses "
             "or national identifiers - the database is not designed to hold them.",
    )
    if saved_id:
        notice(
            f"Saved as <span class='csa-mono'>{saved_id}</span>. Find it under Assessment "
            "History and export it from Reports.",
            kind="success",
        )
    else:
        if st.button("Save assessment to history", type="primary"):
            try:
                identifier = database.insert_assessment(
                    prediction=result,
                    recommendation=recommendation,
                    application_ref=ref or None,
                    explanation=explanation,
                    narrative_source=st.session_state.get("narrative_source"),
                    dataset_name=str(config.DATASET_INFO["name"]),
                    dataset_version=result.model_version,
                    dataset_is_synthetic=bool(
                        predictor.bundle.get("is_synthetic_dataset", False)
                    ),
                )
                st.session_state.saved_assessment_id = identifier
                st.success(f"Assessment saved as {identifier}.")
                st.rerun()
            except database.DatabaseError as exc:
                st.error(str(exc))

    download_payload = {
        "assessment": result.as_dict(),
        "recommendation": recommendation.as_dict(),
        "explanation": explanation.as_dict() if explanation else None,
    }
    st.download_button(
        "Download assessment as JSON",
        data=json.dumps(download_payload, indent=2, default=str),
        file_name=f"creditsense-{result.predicted_at[:10]}.json",
        mime="application/json",
    )


def render() -> None:
    status = model_status()
    page_header(
        "Risk Assessment",
        "Enter the borrower and account information that the trained model actually uses. "
        "The form contains no field the dataset does not provide, and no value is ever "
        "silently invented to complete a prediction.",
        eyebrow="Assessment",
        model_status=status,
    )

    predictor = require_model()
    dataset = require_dataset()
    profile = illustrative_profile(dataset.frame)

    if "assessment_result" not in st.session_state:
        st.session_state.assessment_result = None
    if "saved_assessment_id" not in st.session_state:
        st.session_state.saved_assessment_id = None
    if not all(f"input_{feature}" in st.session_state for feature in config.USER_INPUT_FEATURES):
        _defaults(profile)

    controls = st.columns([1, 1, 2.2])
    with controls[0]:
        if st.button("Load illustrative profile", help="Median/modal values from the dataset."):
            _defaults(profile)
            st.session_state.assessment_result = None
            st.session_state.saved_assessment_id = None
            st.rerun()
    with controls[1]:
        if st.button("Clear result"):
            st.session_state.assessment_result = None
            st.session_state.saved_assessment_id = None
            st.rerun()
    with controls[2]:
        notice(
            "The illustrative profile is a statistical summary of the dataset "
            "(median/modal values), not a real applicant record.",
            kind="info",
        )

    submitted = _render_form(profile)

    if submitted:
        try:
            result = predictor.predict_one(submitted, explain=True, strict=True)
            explanation = result.explanation
            recommendation = recommend(
                result.probability, result.risk, result.inputs, result.derived
            )
            st.session_state.assessment_result = {
                "result": result,
                "recommendation": recommendation,
                "explanation": explanation,
            }
            st.session_state.saved_assessment_id = None
        except InputValidationError as exc:
            st.session_state.assessment_result = None
            st.error("The assessment was not run because some inputs are invalid.")
            st.dataframe(issues_to_frame(exc.report), use_container_width=True, hide_index=True)
        except ModelNotReadyError as exc:
            st.error(str(exc))

    if st.session_state.assessment_result:
        payload = st.session_state.assessment_result
        st.divider()
        _result_panel(payload, predictor)
        st.page_link("pages/what_if_analysis.py",
                     label="Explore this profile in the What-if Simulator",
                     icon=":material/tune:")
    else:
        st.markdown(
            "<div class='csa-muted'>Submit the form to produce a prediction, an explanation "
            "and a recommendation. No example prediction is shown here, because a fabricated "
            "result would be misleading.</div>",
            unsafe_allow_html=True,
        )


render()
