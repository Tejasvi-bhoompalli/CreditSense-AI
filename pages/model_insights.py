"""Model Intelligence page.

Everything here is read from the artifacts produced by the training run. No
metric is typed in by hand and no value is estimated for display.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.ui import (
    PLOT_COLORS,
    apply_chart_theme,
    get_cached_global_shap,
    get_cached_metadata,
    get_cached_metrics,
    get_cached_permutation_importance,
    get_cached_profile,
    kpi_row,
    model_status,
    notice,
    page_header,
    require_model,
    section,
    show_chart,
)
from src.utils import format_percent


def _metric_glossary() -> None:
    with st.expander("What do these metrics mean?"):
        st.markdown(
            """
| Metric | Plain-language meaning | How to read it |
| --- | --- | --- |
| **Accuracy** | Share of all records classified correctly. | Misleading on its own here: a model that always predicts "no default" already reaches ~77.9%. |
| **Balanced accuracy** | Average of the recall on each class, so both classes count equally. | More honest than accuracy with a 22% positive rate. |
| **Precision** | Of the cases the model flags as high risk, how many really defaulted. | High precision means fewer false alarms. |
| **Recall (sensitivity)** | Of the clients who really defaulted, how many the model flagged. | The most business-relevant number for credit risk; missing a default is usually the expensive error. |
| **F1** | Harmonic mean of precision and recall. | A single balance figure when both matter. |
| **ROC-AUC** | Probability that a random defaulter is ranked riskier than a random non-defaulter. | 0.5 = random ordering, 1.0 = perfect. Threshold-independent. |
| **PR-AUC** | Area under the precision-recall curve. | More informative than ROC-AUC when the positive class is the minority. |
| **Brier score** | Mean squared error of the predicted probabilities. | Lower is better. Measures calibration, not just ranking. |
| **ECE** | Expected calibration error across probability bins. | How far the average predicted probability sits from the observed default rate in that bin. |
| **Confusion matrix** | Counts of true/false positives and negatives at the 0.5 threshold. | Shows the actual error trade-off rather than a single score. |
            """
        )
        st.caption(
            "No model is declared universally best. The selection criterion used here is the "
            "validation ROC-AUC, with validation PR-AUC and Brier score as tie-breakers."
        )


def render() -> None:
    status = model_status()
    page_header(
        "Model Intelligence",
        "The model in use, how it was selected, the measured test performance, and the "
        "explanation of its behaviour. All figures come from the saved training artifacts.",
        eyebrow="Model & evaluation",
        model_status=status,
    )

    metrics = get_cached_metrics()
    metadata = get_cached_metadata()

    if not metrics or not metadata:
        notice(
            "No evaluation artifacts were found. Run the training command to generate them; "
            "this page will not display placeholder metrics.",
            kind="danger",
            strong="Artifacts missing.",
        )
        st.code("python -m src.train_model", language="bash")
        st.stop()

    require_model()
    dataset_info = metadata.get("dataset", {})
    profile = get_cached_profile()
    selected = metrics.get("selected", {})
    test = selected.get("test", {})
    candidates = metrics.get("candidates", {})

    # ---- Model and dataset cards ---------------------------------------
    section("Model in use")
    columns = st.columns(2)
    with columns[0]:
        st.markdown(
            f"""
            <div class="csa-card">
              <div class="csa-card-title">{metadata.get('selected_model', 'unknown')}</div>
              <div class="csa-card-body">
                <b>Version</b> {metadata.get('model_version')}<br/>
                <b>Trained (UTC)</b> {metadata.get('trained_at')}<br/>
                <b>Selected by</b> {metadata.get('selection_criterion')}<br/>
                <b>Candidate models compared</b> {', '.join(metadata.get('candidate_models', []))}<br/>
                <b>Calibration</b> {metadata.get('calibration', {}).get('method', 'none')}<br/>
                <b>Feature inputs</b> {len(metadata.get('schema_version', '')) and ''}
                {len(config.MODEL_FEATURES)} model features from {len(config.USER_INPUT_FEATURES)}
                user inputs and {len(config.DERIVED_FEATURES)} derived features<br/>
                <b>Random seed</b> {metadata.get('random_seed')}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with columns[1]:
        st.markdown(
            f"""
            <div class="csa-card">
              <div class="csa-card-title">Dataset</div>
              <div class="csa-card-body">
                <b>{dataset_info.get('name')}</b><br/>
                {dataset_info.get('publisher')} &middot; {dataset_info.get('authors')} (
                {config.DATASET_INFO['publication_year']})<br/>
                <b>DOI</b> {dataset_info.get('doi')}<br/>
                <b>Licence</b> {dataset_info.get('license')}<br/>
                <b>Geography</b> {dataset_info.get('geography')} &middot;
                {dataset_info.get('observation_period')}<br/>
                <b>Records</b> {dataset_info.get('rows')} rows &times; {dataset_info.get('columns')} columns<br/>
                <b>File fingerprint</b> <span class="csa-mono">
                {str(dataset_info.get('sha256', ''))[:24]}...</span><br/>
                {'<b>Data type</b> SYNTHETIC demonstration data' if dataset_info.get('is_synthetic') else ''}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    notice(config.GEOGRAPHIC_DISCLOSURE, kind="warning", strong="Scope of these results.")
    if dataset_info.get("is_synthetic"):
        notice(config.SYNTHETIC_DISCLOSURE, kind="warning", strong="Synthetic data.")

    # ---- Headline metrics ----------------------------------------------
    if test:
        section(
            "Measured performance on the untouched test split",
            f"{metadata.get('split_sizes', {}).get('test', 0)} records held out from "
            "selection and calibration, evaluated exactly once.",
        )
        kpi_row(
            [
                {"label": "ROC-AUC", "value": f"{test['roc_auc']:.4f}",
                 "help": "Ranking ability (0.5 = random)"},
                {"label": "PR-AUC", "value": f"{test['pr_auc']:.4f}",
                 "help": f"Baseline if random = {test['positive_rate']:.4f}"},
                {"label": "Accuracy", "value": f"{test['accuracy']:.4f}",
                 "help": f"Majority-class baseline = "
                         f"{metrics.get('baseline_majority_class', {}).get('accuracy', float('nan')):.4f}"},
                {"label": "Precision", "value": f"{test['precision']:.4f}",
                 "help": f"{test['false_positives']} false positives at threshold 0.5"},
                {"label": "Recall", "value": f"{test['recall']:.4f}",
                 "help": f"{test['true_positives']} of {test['positives']} defaults detected",
                 "tone": "medium"},
                {"label": "F1", "value": f"{test['f1']:.4f}"},
                {"label": "Brier score", "value": f"{test['brier_score']:.4f}",
                 "help": "Lower is better; probability quality"},
            ]
        )

        overfitting = metrics.get("overfitting", {})
        if overfitting:
            line = (
                f"Train ROC-AUC {overfitting.get('train_roc_auc', float('nan')):.4f} &middot; "
                f"validation {overfitting.get('validation_roc_auc', float('nan')):.4f} &middot; "
                f"test {overfitting.get('test_roc_auc', float('nan')):.4f} &middot; "
                f"train-minus-test gap "
                f"{overfitting.get('generalisation_gap_train_minus_test', float('nan')):+.4f}"
            )
            gap = overfitting.get("generalisation_gap_train_minus_test", 0.0)
            notice(
                line + ". " + str(overfitting.get("warning", "")),
                kind="warning" if gap > 0.05 else "success",
            )

        tradeoff = metrics.get("curves", {})
        roc_points = tradeoff.get("roc_test", [])
        pr_points = tradeoff.get("pr_test", [])
        calibration_points = tradeoff.get("calibration_test", [])
        calibration_uncalibrated = tradeoff.get("calibration_test_uncalibrated", [])

        left, right = st.columns(2)
        with left:
            figure = go.Figure()
            if roc_points:
                figure.add_scatter(
                    x=[point["fpr"] for point in roc_points],
                    y=[point["tpr"] for point in roc_points],
                    mode="lines",
                    name=f"Model (AUC {test['roc_auc']:.4f})",
                    line=dict(color=PLOT_COLORS["indigo"], width=2.4),
                    hovertemplate="FPR %{x:.3f}<br>TPR %{y:.3f}<extra></extra>",
                )
            figure.add_scatter(
                x=[0, 1], y=[0, 1], mode="lines", name="Random (AUC 0.5)",
                line=dict(color=PLOT_COLORS["muted"], width=1.4, dash="dash"),
            )
            figure.update_xaxes(title_text="False positive rate")
            figure.update_yaxes(title_text="True positive rate")
            show_chart(apply_chart_theme(figure, height=340, title="ROC curve - test split"),
                       key="mi_roc")

        with right:
            figure = go.Figure()
            if pr_points:
                figure.add_scatter(
                    x=[point["recall"] for point in pr_points],
                    y=[point["precision"] for point in pr_points],
                    mode="lines",
                    name=f"Model (PR-AUC {test['pr_auc']:.4f})",
                    line=dict(color=PLOT_COLORS["teal"], width=2.4),
                    hovertemplate="Recall %{x:.3f}<br>Precision %{y:.3f}<extra></extra>",
                )
                figure.add_scatter(
                    x=[0, 1], y=[test["positive_rate"], test["positive_rate"]],
                    mode="lines", name=f"Random baseline ({test['positive_rate']:.3f})",
                    line=dict(color=PLOT_COLORS["muted"], width=1.4, dash="dash"),
                )
            figure.update_xaxes(title_text="Recall")
            figure.update_yaxes(title_text="Precision")
            show_chart(apply_chart_theme(figure, height=340,
                                         title="Precision-recall curve - test split"),
                       key="mi_pr")

        left, right = st.columns(2)
        with left:
            figure = go.Figure()
            if calibration_uncalibrated:
                figure.add_scatter(
                    x=[point["mean_predicted_probability"] for point in calibration_uncalibrated],
                    y=[point["observed_default_rate"] for point in calibration_uncalibrated],
                    mode="lines+markers",
                    name="Before calibration",
                    line=dict(color=PLOT_COLORS["muted"], width=1.8, dash="dot"),
                    hovertemplate="predicted %{x:.3f}<br>observed %{y:.3f}<extra></extra>",
                )
            if calibration_points:
                figure.add_scatter(
                    x=[point["mean_predicted_probability"] for point in calibration_points],
                    y=[point["observed_default_rate"] for point in calibration_points],
                    mode="lines+markers",
                    name="After Platt calibration",
                    line=dict(color=PLOT_COLORS["indigo"], width=2.4),
                    hovertemplate="predicted %{x:.3f}<br>observed %{y:.3f}<extra></extra>",
                )
            figure.add_scatter(
                x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                line=dict(color=PLOT_COLORS["border"], width=1.6, dash="dash"),
            )
            figure.update_xaxes(title_text="Mean predicted probability")
            figure.update_yaxes(title_text="Observed default rate")
            show_chart(apply_chart_theme(figure, height=340,
                                         title="Reliability curve - test split (10 quantile bins)"),
                       key="mi_cal")

        with right:
            matrix = test.get("confusion_matrix", [[0, 0], [0, 0]])
            figure = go.Figure(
                go.Heatmap(
                    z=matrix,
                    x=["Predicted no default", "Predicted default"],
                    y=["Actual no default", "Actual default"],
                    colorscale=[[0, "#F5F7FB"], [1, PLOT_COLORS["indigo"]]],
                    text=matrix,
                    texttemplate="%{text}",
                    textfont=dict(size=15, color="#172033"),
                    hovertemplate="%{y}<br>%{x}<br>%{z} records<extra></extra>",
                    showscale=False,
                )
            )
            figure.update_yaxes(autorange="reversed")
            show_chart(apply_chart_theme(figure, height=340,
                                         title="Confusion matrix at threshold 0.5",
                                         show_legend=False),
                       key="mi_cm")

        calibration_quality = metrics.get("calibration_quality", {})
        if calibration_quality:
            st.markdown(
                f"**Calibration.** Brier score "
                f"{calibration_quality.get('brier_score_uncalibrated', float('nan')):.4f} before "
                f"calibration and {calibration_quality.get('brier_score_calibrated', float('nan')):.4f} "
                f"after; expected calibration error "
                f"{calibration_quality.get('expected_calibration_error_uncalibrated', float('nan')):.4f} "
                f"before and {calibration_quality.get('expected_calibration_error_calibrated', float('nan')):.4f} "
                f"after. {calibration_quality.get('interpretation', '')}"
            )

    # ---- Candidate comparison ------------------------------------------
    if candidates:
        section(
            "Model comparison",
            "Fitted on the training split and compared on the validation split. The test split "
            "played no part in this selection.",
        )
        rows = []
        for name, result in candidates.items():
            train = result.get("train", {})
            validation = result.get("validation", {})
            rows.append(
                {
                    "Model": name,
                    "Val ROC-AUC": round(validation.get("roc_auc", 0.0), 4),
                    "Val PR-AUC": round(validation.get("pr_auc", 0.0), 4),
                    "Val Brier": round(validation.get("brier_score", 0.0), 4),
                    "Val precision": round(validation.get("precision", 0.0), 4),
                    "Val recall": round(validation.get("recall", 0.0), 4),
                    "Train ROC-AUC": round(train.get("roc_auc", 0.0), 4),
                    "Selected": "yes" if name == metrics.get("selected_model") else "",
                }
            )
        table = pd.DataFrame(rows).sort_values("Val ROC-AUC", ascending=False)
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.caption(
            "Class-weighted variants are included as candidates rather than applied by "
            "default, because weighting shifts the precision/recall operating point and should "
            "be justified by validation evidence."
        )
        baseline = metrics.get("baseline_majority_class", {})
        if baseline:
            st.caption(
                f"Reference point: predicting the majority class for every record scores "
                f"{baseline.get('accuracy', 0.0):.4f} accuracy with ROC-AUC "
                f"{baseline.get('roc_auc', 0.0):.4f} and recall "
                f"{baseline.get('recall', 0.0):.4f}. A model must beat that to be worth using."
            )

    # ---- Feature importance --------------------------------------------
    section("Feature importance")
    global_shap = get_cached_global_shap()
    ranking = global_shap.get("ranking") or []
    if ranking:
        st.caption(f"Method: {global_shap.get('method_label', 'SHAP')} "
                   f"({global_shap.get('rows_used', 0)} records)")
        top = ranking[:15]
        figure = go.Figure(
            go.Bar(
                x=[row["mean_abs_contribution"] for row in top][::-1],
                y=[row["label"] for row in top][::-1],
                orientation="h",
                marker_color=PLOT_COLORS["indigo"],
                hovertemplate="%{y}<br>mean |contribution| %{x:.4f}<extra></extra>",
            )
        )
        figure.update_xaxes(title_text="Mean absolute SHAP contribution (log-odds)")
        show_chart(apply_chart_theme(figure, height=440, show_legend=False,
                                     title="Global SHAP importance"),
                   key="mi_shap")
    else:
        notice(
            "No SHAP ranking is stored for the active model; see the permutation importance "
            "below, which uses a different method.",
            kind="warning",
        )

    permutation = get_cached_permutation_importance().get("importances") or []
    if permutation:
        figure = go.Figure(
            go.Bar(
                x=[row["mean_importance"] for row in permutation[:15]][::-1],
                y=[row["label"] for row in permutation[:15]][::-1],
                orientation="h",
                marker_color=PLOT_COLORS["blue"],
                error_x=dict(
                    type="data",
                    array=[row["std_importance"] for row in permutation[:15]][::-1],
                    color=PLOT_COLORS["muted"],
                ),
                hovertemplate="%{y}<br>ROC-AUC drop %{x:.5f}<extra></extra>",
            )
        )
        figure.update_xaxes(title_text="Drop in ROC-AUC when the feature is shuffled")
        show_chart(apply_chart_theme(figure, height=420, show_legend=False,
                                     title="Permutation importance on the test split"),
                   key="mi_perm")

    # ---- Class distribution --------------------------------------------
    left, right = st.columns(2)
    with left:
        counts = metadata.get("class_counts", {})
        for split_name, values in counts.items():
            pass
        figure = go.Figure()
        for split_name in ("train", "validation", "test"):
            values = counts.get(split_name, {})
            figure.add_bar(
                name=split_name,
                x=["No default (0)", "Default (1)"],
                y=[values.get("0", 0), values.get("1", 0)],
                hovertemplate="%{x}<br>%{y} records<extra>" + split_name + "</extra>",
            )
        figure.update_layout(barmode="group")
        figure.update_yaxes(title_text="Records")
        show_chart(apply_chart_theme(figure, height=320, title="Class distribution by split"),
                   key="mi_class")
    with right:
        if profile:
            st.markdown("**Dataset profile recorded at training time**")
            rows = [
                {"Property": "Rows", "Value": f"{profile.get('rows', 0):,}"},
                {"Property": "Columns", "Value": profile.get("columns")},
                {"Property": "Missing values", "Value": profile.get("missing_total", 0)},
                {"Property": "Duplicate feature rows",
                 "Value": profile.get("duplicate_rows_excluding_target", 0)},
                {"Property": "Observed default rate",
                 "Value": f"{profile.get('target_positive_rate', 0) * 100:.2f}%"},
                {"Property": "Class imbalance ratio",
                 "Value": f"{profile.get('class_imbalance_ratio', 0):.2f} : 1"},
            ]
            st.dataframe(rows, use_container_width=True, hide_index=True)

    _metric_glossary()

    st.caption(
        "Prototype disclaimer: " + config.APP_DISCLAIMER
    )


render()
