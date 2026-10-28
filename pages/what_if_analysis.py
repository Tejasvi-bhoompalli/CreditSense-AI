"""What-if simulator.

The stored baseline is never modified. Predictions are re-computed by the same
saved pipeline for every comparison, and the expensive explanation step runs only
when the user explicitly asks for it.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config, database
from src.data_validation import InputValidationError, issues_to_frame
from src.recommendation import recommend
from src.ui import (
    PLOT_COLORS,
    apply_chart_theme,
    get_cached_dataset,
    model_status,
    notice,
    page_header,
    require_dataset,
    require_model,
    section,
    show_chart,
)
from src.utils import feature_label, format_probability
from src.what_if import (
    PRIMARY_ADJUSTMENTS,
    SUPPORTED_ADJUSTMENTS,
    apply_adjustments,
    compare_scenarios,
    illustrative_profile,
)


def _control(feature: str, current: float, key_prefix: str) -> float:
    """Slider control whose bounds come from config.INPUT_RANGES."""
    low, high = config.INPUT_RANGES[feature]
    step = SUPPORTED_ADJUSTMENTS[feature]["step"]
    is_status = feature in config.REPAY_STATUS_COLUMNS
    value = float(current)
    if is_status:
        options = list(range(int(low), int(high) + 1))
        nearest = min(options, key=lambda option: abs(option - value))
        index = options.index(nearest)
        return float(
            st.slider(
                feature_label(feature),
                min_value=options[0],
                max_value=options[-1],
                value=options[index],
                step=1,
                key=f"{key_prefix}_{feature}",
                format="%d",
            )
        )
    return float(
        st.slider(
            feature_label(feature),
            min_value=float(low),
            max_value=float(high),
            value=float(min(max(value, low), high)),
            step=float(step),
            key=f"{key_prefix}_{feature}",
        )
    )


def _comparison_cards(comparison) -> None:
    columns = st.columns(3)
    with columns[0]:
        st.markdown(
            f"<div class='csa-card'><div class='csa-kpi-label'>Baseline scenario</div>"
            f"<div class='csa-kpi-value'>{format_probability(comparison.baseline_probability)}</div>"
            f"<div class='csa-muted'>Score {comparison.baseline_risk_score:.1f} &middot; "
            f"{comparison.baseline_category}</div></div>",
            unsafe_allow_html=True,
        )
    with columns[1]:
        st.markdown(
            f"<div class='csa-card'><div class='csa-kpi-label'>Alternative scenario</div>"
            f"<div class='csa-kpi-value'>{format_probability(comparison.alternative_probability)}</div>"
            f"<div class='csa-muted'>Score {comparison.alternative_risk_score:.1f} &middot; "
            f"{comparison.alternative_category}</div></div>",
            unsafe_allow_html=True,
        )
    with columns[2]:
        arrow = "&#8593;" if comparison.probability_change_points > 0 else (
            "&#8595;" if comparison.probability_change_points < 0 else "&#8594;"
        )
        st.markdown(
            f"<div class='csa-card'><div class='csa-kpi-label'>Difference</div>"
            f"<div class='csa-kpi-value'>{comparison.probability_change_points:+.2f} pp {arrow}</div>"
            f"<div class='csa-muted'>Score change {comparison.risk_score_change:+.1f} &middot; "
            f"{comparison.category_transition}</div></div>",
            unsafe_allow_html=True,
        )


def render() -> None:
    status = model_status()
    page_header(
        "What-if Simulator",
        "Adjust the supported inputs of a borrower profile and see how the fitted model's "
        "estimate moves. The baseline record is never modified, and every figure is produced "
        "by an actual model call.",
        eyebrow="Scenario analysis",
        model_status=status,
    )

    predictor = require_model()
    dataset = require_dataset()

    if "scenario_comparison" not in st.session_state:
        st.session_state.scenario_comparison = None
    if "scenario_adjustments" not in st.session_state:
        st.session_state.scenario_adjustments = {}

    # ---- Baseline selection --------------------------------------------
    section("Baseline profile")
    saved_ids = database.assessment_ids(limit=200)
    options = ["Illustrative profile (dataset medians)"] + [
        f"Stored assessment {identifier}" for identifier in saved_ids
    ]
    choice = st.selectbox("Baseline source", options=options, key="whatif_baseline_choice")

    baseline: dict
    baseline_label: str
    if choice.startswith("Stored"):
        identifier = choice.replace("Stored assessment ", "").strip()
        record = database.get_assessment(identifier)
        if record is None:
            notice("That assessment no longer exists in the database.", kind="danger")
            st.stop()
        baseline = {
            feature: record["inputs"][feature]
            for feature in config.USER_INPUT_FEATURES
            if feature in record["inputs"]
        }
        baseline_label = f"Stored assessment {identifier} (created {record['created_at'][:19]}Z)"
    else:
        baseline = illustrative_profile(dataset.frame)
        baseline_label = (
            "Illustrative profile built from dataset medians and modes - a statistical "
            "summary, not a real applicant record."
        )

    if len(baseline) != len(config.USER_INPUT_FEATURES):
        notice(
            "The stored baseline is missing required fields and cannot be used for a scenario "
            "comparison.",
            kind="danger",
        )
        st.stop()

    st.caption(baseline_label)

    with st.expander("Baseline values (read-only)", expanded=False):
        rows = []
        for feature in config.USER_INPUT_FEATURES:
            value = baseline[feature]
            if feature in config.CATEGORICAL_FEATURES:
                display = config.EDUCATION_DISPLAY.get(str(value), str(value))
            elif feature in config.REPAY_STATUS_COLUMNS:
                display = f"{int(float(value))} - {config.REPAY_STATUS_CODES.get(int(float(value)), '')}"
            else:
                display = f"NT$ {float(value):,.0f}"
            rows.append({"Input": feature_label(feature), "Value": display})
        st.dataframe(rows, use_container_width=True, hide_index=True)

    # ---- Adjustments ----------------------------------------------------
    section(
        "Adjust supported inputs",
        "Only fields present in the model's feature schema are adjustable. Derived ratios "
        "(credit utilisation and repayment ratio) are recomputed automatically and are not "
        "editable, because they are not independent inputs.",
    )
    adjustments: dict = {}
    key_prefix = f"wf_{abs(hash(choice)) % 10**8}"
    primary_columns = st.columns(2)
    for index, feature in enumerate(PRIMARY_ADJUSTMENTS):
        with primary_columns[index % 2]:
            adjustments[feature] = _control(feature, baseline[feature], key_prefix)

    remaining = [
        feature
        for feature in SUPPORTED_ADJUSTMENTS
        if feature not in PRIMARY_ADJUSTMENTS
    ]
    with st.expander(f"Advanced: adjust the other {len(remaining)} supported fields"):
        columns = st.columns(3)
        for index, feature in enumerate(remaining):
            with columns[index % 3]:
                adjustments[feature] = _control(feature, baseline[feature], key_prefix)

    actions = st.columns([1.2, 1.2, 3])
    with actions[0]:
        compare_clicked = st.button("Compare scenarios", type="primary", use_container_width=True)
    with actions[1]:
        reset_clicked = st.button("Reset to baseline", use_container_width=True)

    if reset_clicked:
        for feature in SUPPORTED_ADJUSTMENTS:
            st.session_state.pop(f"{key_prefix}_{feature}", None)
        st.session_state.scenario_comparison = None
        st.rerun()

    if compare_clicked:
        try:
            comparison = compare_scenarios(baseline, adjustments, predictor)
            st.session_state.scenario_comparison = comparison
            st.session_state.scenario_adjustments = dict(adjustments)
        except InputValidationError as exc:
            st.session_state.scenario_comparison = None
            st.error("The comparison was not run because an adjusted value is invalid.")
            st.dataframe(issues_to_frame(exc.report), use_container_width=True, hide_index=True)
        except Exception as exc:  # pragma: no cover - defensive
            st.session_state.scenario_comparison = None
            st.error(f"The comparison could not be completed ({type(exc).__name__}).")

    comparison = st.session_state.scenario_comparison
    if comparison is None:
        notice(
            "Adjust the sliders and press <strong>Compare scenarios</strong>. No example "
            "comparison is displayed, because a fabricated one would misrepresent the model.",
            kind="info",
        )
        return

    stale = any(
        abs(float(adjustments.get(feature, baseline[feature])) - float(value)) > 1e-9
        for feature, value in st.session_state.scenario_adjustments.items()
    )
    if stale:
        notice(
            "Some sliders have changed since this comparison was computed. Press "
            "<strong>Compare scenarios</strong> again to refresh it.",
            kind="warning",
        )

    st.divider()
    section("Scenario comparison")
    _comparison_cards(comparison)
    st.markdown(comparison.narrative)

    figure = go.Figure()
    figure.add_bar(
        name="Baseline",
        x=["Default probability (%)", "Risk score"],
        y=[comparison.baseline_probability * 100, comparison.baseline_risk_score],
        marker_color=PLOT_COLORS["muted"],
        hovertemplate="%{x}: %{y:.2f}<extra>Baseline</extra>",
    )
    figure.add_bar(
        name="Alternative",
        x=["Default probability (%)", "Risk score"],
        y=[comparison.alternative_probability * 100, comparison.alternative_risk_score],
        marker_color=PLOT_COLORS["indigo"],
        hovertemplate="%{x}: %{y:.2f}<extra>Alternative</extra>",
    )
    figure.update_layout(barmode="group")
    show_chart(
        apply_chart_theme(figure, height=320, title="Before and after"),
        key="wi_before_after",
    )

    left, right = st.columns([1.4, 1])
    with left:
        st.markdown("**Adjusted inputs**")
        if comparison.changed_features:
            st.dataframe(comparison.to_frame(), use_container_width=True, hide_index=True)
        else:
            st.markdown("No input was changed, so the two scenarios are identical.")
    with right:
        st.markdown("**Derived indicators**")
        derived_rows = []
        for feature in config.DERIVED_FEATURES:
            before = comparison.baseline_derived.get(feature)
            after = comparison.alternative_derived.get(feature)
            if before is None or after is None:
                continue
            derived_rows.append(
                {
                    "Indicator": feature_label(feature),
                    "Baseline": round(before, 4),
                    "Alternative": round(after, 4),
                }
            )
        st.dataframe(derived_rows, use_container_width=True, hide_index=True)

    for caveat in comparison.caveats:
        st.caption(caveat)

    # ---- Optional explanation of the alternative -----------------------
    with st.expander("Explain the alternative scenario (optional, slower)"):
        st.caption(
            "Feature attributions are expensive to compute, so they are generated only when "
            "you ask for them, never on every slider movement."
        )
        if st.button("Generate explanation for the alternative scenario"):
            with st.spinner("Computing feature attributions..."):
                try:
                    explanation = predictor.explain(
                        comparison.alternative_inputs,
                        probability=comparison.alternative_probability,
                    )
                    st.session_state["scenario_explanation"] = explanation
                except Exception as exc:  # pragma: no cover - defensive
                    st.session_state["scenario_explanation"] = None
                    st.error(f"Explanation unavailable ({type(exc).__name__}).")
        explanation = st.session_state.get("scenario_explanation")
        if explanation is not None:
            st.caption(f"Method: {explanation.method_label}")
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
                    hovertemplate="%{y}<br>%{x:+.4f} log-odds<extra></extra>",
                )
            )
            show_chart(
                apply_chart_theme(figure, height=360, show_legend=False,
                                  title="Contributions for the alternative scenario"),
                key="wi_explanation",
            )
            for note in explanation.notes:
                st.caption(note)

    # ---- Export / persist ----------------------------------------------
    section("Export or persist the alternative scenario")
    st.download_button(
        "Download comparison (CSV)",
        data=pd.DataFrame(
            [
                {"Scenario": "Baseline", "Probability": comparison.baseline_probability,
                 "Risk score": comparison.baseline_risk_score,
                 "Risk category": comparison.baseline_category},
                {"Scenario": "Alternative", "Probability": comparison.alternative_probability,
                 "Risk score": comparison.alternative_risk_score,
                 "Risk category": comparison.alternative_category},
                {"Scenario": "Difference (pp)", "Probability": comparison.probability_change_points,
                 "Risk score": comparison.risk_score_change,
                 "Risk category": comparison.category_transition},
            ]
        ).to_csv(index=False).encode("utf-8"),
        file_name="creditsense-what-if-comparison.csv",
        mime="text/csv",
    )

    st.markdown("**Save the alternative as a new assessment**")
    st.caption(
        "The original baseline record is left untouched. Saving creates a separate record."
    )
    save_columns = st.columns([2, 1, 3])
    with save_columns[0]:
        reference = st.text_input(
            "Application reference for the saved alternative",
            key="whatif_save_ref",
            placeholder="e.g. APP-2026-0184-whatif",
            help="Do not enter personal identifiers.",
        )
    with save_columns[1]:
        if st.button("Save alternative"):
            result = predictor.predict_one(
                comparison.alternative_inputs, explain=False, strict=True
            )
            recommendation = recommend(
                result.probability, result.risk, result.inputs, result.derived
            )
            identifier = database.insert_assessment(
                prediction=result,
                recommendation=recommendation,
                application_ref=reference or None,
                dataset_name=str(config.DATASET_INFO["name"]),
                dataset_version=result.model_version,
            )
            st.success(f"Saved as {identifier}.")


render()
