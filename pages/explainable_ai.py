"""Explainable AI page: global and local SHAP explanations.

Every explanation is produced for the same fitted estimator that serves
predictions. When SHAP cannot run, the page says so explicitly and shows the
documented fallback instead of silently substituting a different method.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from src import config, database
from src.explainability import SHAP_AVAILABLE, SHAP_IMPORT_ERROR, narrate_global, narrate_local
from src.ui import (
    PLOT_COLORS,
    apply_chart_theme,
    get_cached_dataset,
    get_cached_global_shap,
    get_cached_metrics,
    get_cached_permutation_importance,
    model_status,
    notice,
    page_header,
    require_dataset,
    require_model,
    section,
    show_chart,
)
from src.utils import feature_label
from src.what_if import illustrative_profile


def _global_tab(explainer, dataset) -> None:
    section(
        "Global explanation",
        "How much each feature moved the model's output across a sample of records, "
        "averaged and reported as mean absolute contribution.",
    )

    importance = get_cached_global_shap()
    ranking = importance.get("ranking") or []
    using_fallback = not ranking

    if using_fallback:
        notice(
            "No SHAP global ranking is stored for the active model. Falling back to "
            "permutation importance, which is a different method and is labelled as such.",
            kind="warning",
        )
        permutation = get_cached_permutation_importance().get("importances") or []
        ranking = [
            {
                "feature": row["feature"],
                "label": row["label"],
                "mean_abs_contribution": row["mean_importance"],
                "share_of_total": 0.0,
                "direction": "mixed",
            }
            for row in permutation
        ]
        method_label = "Permutation importance (ROC-AUC drop) - not SHAP"
    else:
        method_label = importance.get("method_label", "SHAP")
        notice(
            f"Method: <strong>{method_label}</strong>. Computed on "
            f"{importance.get('rows_used', 0)} records drawn from the held-out evaluation split.",
            kind="info",
        )

    if not ranking:
        notice(
            "No global importance artifact is available. Run `python -m src.train_model` to "
            "produce it.",
            kind="danger",
        )
        return

    top = ranking[:15]
    figure = go.Figure(
        go.Bar(
            x=[row["mean_abs_contribution"] for row in top][::-1],
            y=[row["label"] for row in top][::-1],
            orientation="h",
            marker_color=PLOT_COLORS["indigo"],
            customdata=[
                [row.get("share_of_total", 0) * 100, row.get("direction", "")]
                for row in top
            ][::-1],
            hovertemplate="%{y}<br>mean |contribution| %{x:.4f}<br>"
                          "share %{customdata[0]:.1f}%<br>%{customdata[1]}<extra></extra>",
        )
    )
    figure.update_xaxes(title_text="Mean absolute contribution (log-odds)")
    show_chart(
        apply_chart_theme(figure, height=460, title="Global feature importance",
                          show_legend=False),
        key="xai_global",
    )

    st.dataframe(
        [
            {
                "Feature": row["label"],
                "Feature key": row["feature"],
                "Mean |contribution|": round(row["mean_abs_contribution"], 5),
                "Share of total": f"{row.get('share_of_total', 0) * 100:.1f}%",
                "Typical direction": row.get("direction", ""),
            }
            for row in ranking[:20]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("**Plain-English reading of the ranking**")
    if not using_fallback and importance.get("ranking"):
        from src.explainability import GlobalImportance

        for sentence in narrate_global(
            GlobalImportance(
                method=importance.get("method", "shap"),
                method_label=method_label,
                rows_used=importance.get("rows_used", 0),
                ranking=importance.get("ranking", []),
            )
        ):
            st.markdown(f"- {sentence}")
    else:
        for row in ranking[:5]:
            st.markdown(
                f"- Shuffling **{row['label']}** reduces the measured ROC-AUC by "
                f"{row['mean_abs_contribution']:.4f} on average."
            )

    for note in importance.get("notes", []):
        st.caption(note)


def _local_tab(predictor, dataset) -> None:
    section(
        "Local explanation",
        "Feature contributions for one specific profile, computed on demand.",
    )

    source_options = ["Illustrative profile (dataset medians)"]
    last_result = st.session_state.get("assessment_result")
    if last_result:
        source_options.append("Profile from the last Risk Assessment")
    saved_ids = database.assessment_ids(limit=200)
    source_options += [f"Stored assessment {identifier}" for identifier in saved_ids]

    choice = st.selectbox("Profile to explain", options=source_options, key="xai_profile")
    profile: dict | None = None
    label = ""
    if choice.startswith("Stored"):
        record = database.get_assessment(choice.replace("Stored assessment ", "").strip())
        if record:
            profile = {
                feature: record["inputs"][feature]
                for feature in config.USER_INPUT_FEATURES
                if feature in record["inputs"]
            }
            label = f"stored assessment {record['assessment_id']}"
    elif choice.startswith("Profile from"):
        profile = last_result["result"].inputs if last_result else None
        label = "the profile scored on the Risk Assessment page"
    else:
        profile = illustrative_profile(dataset.frame)
        label = "an illustrative profile built from dataset medians"

    if profile is None or len(profile) != len(config.USER_INPUT_FEATURES):
        notice("That profile is not available; choose another source.", kind="warning")
        return

    st.caption(f"Explaining {label}. This is a dataset-derived or user-entered profile, "
               "not a real customer record.")

    with st.expander("Profile values"):
        st.dataframe(
            [
                {"Input": feature_label(feature), "Value": profile[feature]}
                for feature in config.USER_INPUT_FEATURES
            ],
            use_container_width=True,
            hide_index=True,
        )

    if st.button("Generate explanation", type="primary"):
        with st.spinner("Computing feature attributions..."):
            try:
                result = predictor.predict_one(profile, explain=True, strict=True)
                st.session_state["xai_local"] = {
                    "result": result,
                    "explanation": result.explanation,
                    "profile": dict(profile),
                }
            except Exception as exc:  # pragma: no cover - defensive
                st.session_state["xai_local"] = None
                st.error(f"The explanation could not be generated ({type(exc).__name__}).")

    payload = st.session_state.get("xai_local")
    if not payload:
        notice(
            "Press <strong>Generate explanation</strong>. Nothing is displayed beforehand, "
            "because an invented contribution chart would be misleading.",
            kind="info",
        )
        return
    if payload["profile"] != profile:
        notice(
            "The profile changed since the last explanation was generated. Press the button "
            "again to refresh.",
            kind="warning",
        )

    result = payload["result"]
    explanation = payload["explanation"]
    if explanation is None:
        notice("No explanation is available for this profile.", kind="warning")
        return

    st.markdown(
        f"Predicted default probability **{result.probability * 100:.2f}%** "
        f"({result.risk.risk_category}, risk score {result.risk.risk_score:.1f}/100). "
        f"Base model output {explanation.output_value:+.4f} log-odds from a background "
        f"expectation of {explanation.base_value:+.4f}."
    )
    st.caption(f"Explanation method: {explanation.method_label}")

    ranked = explanation.ranked(12)
    figure = go.Figure(
        go.Bar(
            x=[item.contribution for item in ranked][::-1],
            y=[f"{item.label} = {item.display_value}" for item in ranked][::-1],
            orientation="h",
            marker_color=[
                PLOT_COLORS["high"] if item.contribution > 0 else PLOT_COLORS["low"]
                for item in ranked
            ][::-1],
            hovertemplate="%{y}<br>contribution %{x:+.5f} log-odds<extra></extra>",
        )
    )
    figure.add_vline(x=0, line=dict(color=PLOT_COLORS["border"], width=1.4))
    figure.update_xaxes(title_text="Contribution to the model's raw output (log-odds)")
    show_chart(
        apply_chart_theme(
            figure, height=440,
            title="Local contributions - red pushes risk up, green pulls it down",
            show_legend=False,
        ),
        key="xai_local_chart",
    )

    left, right = st.columns(2)
    with left:
        st.markdown("**Increases predicted risk**")
        for item in explanation.top_increasing[:6]:
            st.markdown(f"- {item.label} ({item.display_value}): {item.contribution:+.5f}")
        if not explanation.top_increasing:
            st.markdown("- none")
    with right:
        st.markdown("**Reduces predicted risk**")
        for item in explanation.top_decreasing[:6]:
            st.markdown(f"- {item.label} ({item.display_value}): {item.contribution:+.5f}")
        if not explanation.top_decreasing:
            st.markdown("- none")

    st.markdown("**Plain-English translation**")
    for sentence in narrate_local(explanation):
        st.markdown(f"- {sentence}")

    st.dataframe(
        explanation.to_frame(), use_container_width=True, hide_index=True
    )
    for note in explanation.notes:
        st.caption(note)


def _method_tab() -> None:
    section("How explanations are produced")
    st.markdown(
        f"""
        **Implementation.** Explanations are computed with the `shap` package on the exact
        fitted estimator that serves predictions ({'SHAP is importable' if SHAP_AVAILABLE else 'SHAP is not importable in this environment'}).
        Tree-based models use `shap.TreeExplainer` (exact tree-path contributions), logistic
        regression uses `shap.LinearExplainer`. Contributions are expressed in the base
        model's raw output units (log-odds).

        **Categorical handling.** `education` is one-hot encoded inside the pipeline, so its
        indicator columns are summed back onto the single source feature before display. This
        means every contribution in the interface corresponds to a field the user actually
        supplied.

        **Calibration and explanation.** The displayed probability is a monotone (Platt/sigmoid)
        calibration of the same model output that SHAP explains. Monotone transforms preserve
        the ordering of contributions, which is why the explanation is valid for the displayed
        prediction. This is stated rather than assumed.

        **Fallback.** If SHAP is unavailable or incompatible, the application does not stop and
        does not pretend: it falls back to (a) a local occlusion heuristic - each feature is
        replaced by its background value and the change in the model output is reported - and
        (b) globally, permutation importance measured as the drop in ROC-AUC. Both fallbacks
        are labelled as fallbacks on every screen that shows them, and the occlusion fallback
        is explicitly non-additive.
        """
    )
    if not SHAP_AVAILABLE:
        notice(f"SHAP import error recorded: {SHAP_IMPORT_ERROR}", kind="warning")
    notice(
        "A SHAP value describes how a feature contributed to <em>this model's output for this "
        "input</em> under the chosen method. It is not a causal effect, and it does not imply "
        "that changing the feature would change the real-world outcome.",
        kind="warning",
        strong="What a contribution is not.",
    )


def render() -> None:
    status = model_status()
    page_header(
        "Explainable AI",
        "Global importance for the model as a whole, and local feature attributions for an "
        "individual profile. Explanations are always computed on the model that produced the "
        "prediction.",
        eyebrow="Transparency",
        model_status=status,
    )

    predictor = require_model()
    dataset = require_dataset()
    metrics = get_cached_metrics()

    if metrics.get("explainability"):
        info = metrics["explainability"]
        notice(
            f"Recorded during training: <strong>{info.get('method_label', 'unknown')}</strong>."
            + (" " + " ".join(info.get("notes", [])[:1]) if info.get("notes") else ""),
            kind="info",
        )

    explainer = None
    try:
        explainer = predictor.explainer()
    except Exception as exc:  # pragma: no cover - defensive
        notice(f"The explainer could not be initialised ({type(exc).__name__}).", kind="warning")

    tabs = st.tabs(["Global explanation", "Local explanation", "Method and caveats"])
    with tabs[0]:
        _global_tab(explainer, dataset)
    with tabs[1]:
        _local_tab(predictor, dataset)
    with tabs[2]:
        _method_tab()


render()
