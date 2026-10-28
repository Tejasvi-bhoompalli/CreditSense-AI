"""Overview dashboard.

Every figure on this page is computed from stored assessments or from the loaded
dataset. The two sources are always labelled separately, because they answer
different questions:

* "Your assessment activity" - aggregated from the SQLite assessment history.
* "Reference distributions"  - the historical training dataset itself.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config, database
from src.ui import (
    PLOT_COLORS,
    apply_chart_theme,
    dataset_status,
    empty_state,
    feature_availability_html,
    get_cached_dataset,
    get_cached_metrics,
    kpi_row,
    model_status,
    notice,
    page_header,
    section,
    show_chart,
)
from src.utils import format_percent, format_probability


def _risk_donut(counts: dict) -> go.Figure:
    labels = [category for category in config.RISK_CATEGORIES if counts.get(category, 0) > 0]
    values = [counts.get(category, 0) for category in labels]
    if not labels:
        labels, values = ["No assessments"], [1]
    colors = [config.RISK_COLORS.get(label, PLOT_COLORS["muted"]) for label in labels]
    figure = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.62,
            marker=dict(colors=colors, line=dict(color="#FFFFFF", width=2)),
            textinfo="label+percent",
            textposition="outside",
            hovertemplate="%{label}<br>%{value} assessments (%{percent})<extra></extra>",
        )
    )
    figure.update_layout(showlegend=False)
    return apply_chart_theme(figure, height=330, show_legend=False)


def _probability_histogram(frame: pd.DataFrame) -> go.Figure:
    figure = go.Figure(
        go.Histogram(
            x=frame["probability"] * 100,
            nbinsx=20,
            marker=dict(color=PLOT_COLORS["indigo"], line=dict(color="#FFFFFF", width=0.8)),
            hovertemplate="Probability %{x:.1f}%<br>%{y} assessments<extra></extra>",
        )
    )
    for boundary, color in ((30, PLOT_COLORS["medium"]), (60, PLOT_COLORS["high"])):
        figure.add_vline(x=boundary, line=dict(color=color, width=1.4, dash="dot"))
    figure.update_xaxes(title_text="Predicted default probability (%)")
    figure.update_yaxes(title_text="Assessments")
    return apply_chart_theme(figure, height=330, show_legend=False)


def _category_by_month(frame: pd.DataFrame) -> go.Figure:
    frame = frame.copy()
    frame["month"] = pd.to_datetime(frame["created_at"], errors="coerce", utc=True).dt.strftime("%Y-%m")
    grouped = (
        frame.dropna(subset=["month"])
        .groupby(["month", "risk_category"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    figure = go.Figure()
    for category in config.RISK_CATEGORIES:
        if category not in grouped.columns:
            continue
        figure.add_bar(
            x=grouped.index,
            y=grouped[category],
            name=category,
            marker_color=config.RISK_COLORS[category],
            hovertemplate="%{x}<br>%{y} " + category + "<extra></extra>",
        )
    figure.update_layout(barmode="stack")
    figure.update_xaxes(title_text="Month")
    figure.update_yaxes(title_text="Assessments")
    return apply_chart_theme(figure, height=320)


def _dataset_reference_charts(dataset) -> None:
    frame = dataset.frame
    left, right = st.columns(2)

    with left:
        figure = go.Figure(
            go.Histogram(
                x=frame["credit_limit"],
                nbinsx=40,
                marker=dict(color=PLOT_COLORS["blue"], line=dict(color="#FFFFFF", width=0.6)),
                hovertemplate="Credit limit NT$ %{x:,.0f}<br>%{y} clients<extra></extra>",
            )
        )
        figure.update_xaxes(title_text="Granted credit limit (NT$)")
        figure.update_yaxes(title_text="Clients")
        show_chart(
            apply_chart_theme(figure, height=300, title="Credit limit - historical dataset", show_legend=False),
            key="dash_limit_hist",
        )

    with right:
        delay_counts = (
            frame["repay_status_1"].value_counts().sort_index()
        )
        figure = go.Figure(
            go.Bar(
                x=[str(int(index)) for index in delay_counts.index],
                y=delay_counts.values,
                marker_color=[
                    PLOT_COLORS["high"] if index >= 1 else PLOT_COLORS["low"]
                    for index in delay_counts.index
                ],
                hovertemplate="Status %{x}<br>%{y} clients<extra></extra>",
            )
        )
        figure.update_xaxes(title_text="Most recent repayment status level")
        figure.update_yaxes(title_text="Clients")
        show_chart(
            apply_chart_theme(
                figure,
                height=300,
                title="Repayment status - historical dataset (level >= 1 means a delay)",
                show_legend=False,
            ),
            key="dash_status_bar",
        )

    left, right = st.columns(2)
    with left:
        figure = go.Figure(
            go.Histogram(
                x=frame["age"],
                nbinsx=45,
                marker=dict(color=PLOT_COLORS["teal"], line=dict(color="#FFFFFF", width=0.6)),
                hovertemplate="Age %{x}<br>%{y} clients<extra></extra>",
            )
        )
        figure.update_xaxes(title_text="Age (years)")
        figure.update_yaxes(title_text="Clients")
        show_chart(
            apply_chart_theme(figure, height=280, title="Age - historical dataset", show_legend=False),
            key="dash_age_hist",
        )
    with right:
        target_counts = frame[config.TARGET_COLUMN].value_counts().sort_index()
        figure = go.Figure(
            go.Bar(
                x=["No recorded default (0)", "Recorded default (1)"],
                y=[int(target_counts.get(0, 0)), int(target_counts.get(1, 0))],
                marker_color=[PLOT_COLORS["low"], PLOT_COLORS["high"]],
                hovertemplate="%{x}<br>%{y} clients<extra></extra>",
            )
        )
        figure.update_xaxes(title_text="Observed outcome")
        figure.update_yaxes(title_text="Clients")
        show_chart(
            apply_chart_theme(
                figure, height=280, title="Observed target distribution - historical dataset",
                show_legend=False,
            ),
            key="dash_target_bar",
        )


def render() -> None:
    status = model_status()
    page_header(
        "Credit Risk Intelligence",
        "Monitor assessments, understand risk patterns, and explore model-driven "
        "lending insights. Every figure below is calculated from stored assessments or "
        "from the loaded dataset - nothing on this page is illustrative.",
        eyebrow="Overview",
        model_status=status,
    )

    # ---- Readiness ------------------------------------------------------
    data_status = dataset_status()
    dataset_ready = bool(data_status.get("ready"))
    dataset = get_cached_dataset() if dataset_ready else None
    dataset_summary = (
        f"{data_status['filename']} &middot; {int(data_status['rows']):,} rows"
        if dataset_ready
        else (data_status.get("error") or "not found")
    )

    readiness = [
        ("Dataset", dataset_ready, dataset_summary),
        ("Model artifacts", bool(status.get("ready")), status.get("model_name") or "not trained"),
        ("Database", database.is_available(), str(config.DB_PATH.name)),
    ]
    chips = "".join(
        f'<div style="margin-bottom:0.3rem">'
        f'<span style="font-size:0.78rem;color:#64748B;display:inline-block;min-width:120px">'
        f"{label}</span>"
        f'<span class="csa-chip {"ok" if ok else "missing"}">'
        f'{"ready" if ok else "unavailable"}</span>'
        f'<span class="csa-muted">{detail}</span></div>'
        for label, ok, detail in readiness
    )
    st.markdown(f'<div class="csa-card">{chips}</div>', unsafe_allow_html=True)

    if not status.get("ready"):
        notice(
            "Train the model to enable risk assessment, explainability and reporting:",
            kind="warning",
            strong="Model not trained.",
        )
        st.code("python -m src.train_model", language="bash")

    # ---- Assessment activity -------------------------------------------
    section(
        "Assessment activity",
        "Aggregated only from assessments you have explicitly saved to the local database.",
    )

    stats = database.get_statistics()
    filter_columns = st.columns([1.1, 1.1, 1.4])
    with filter_columns[0]:
        category_filter = st.multiselect(
            "Risk category",
            options=config.RISK_CATEGORIES,
            default=list(config.RISK_CATEGORIES),
            key="dash_categories",
        )
    with filter_columns[1]:
        date_filter = st.date_input(
            "Created on or after",
            value=None,
            key="dash_date_from",
            help="Leave empty to include every stored assessment.",
        )
    with filter_columns[2]:
        search = st.text_input(
            "Search assessment ID or application reference",
            key="dash_search",
            placeholder="e.g. CSA-1A2B3C4D5E6F",
        )

    if stats["total"] == 0:
        empty_state(
            "No assessments stored yet",
            "The dashboard shows live aggregates, so it stays empty until you save a real "
            "assessment. Nothing has been pre-filled with invented records.",
        )
        st.page_link("pages/risk_assessment.py", label="Start New Assessment",
                     icon=":material/assignment:")
        st.page_link("pages/what_if_analysis.py", label="Open the What-if Simulator",
                     icon=":material/tune:")
        st.page_link("pages/model_insights.py", label="Review Model Intelligence",
                     icon=":material/monitoring:")
    else:
        frame = database.get_assessment_frame(limit=5000)
        filtered = frame.copy()
        if category_filter:
            filtered = filtered[filtered["risk_category"].isin(category_filter)]
        else:
            filtered = filtered.iloc[0:0]
        if date_filter is not None:
            filtered = filtered[
                pd.to_datetime(filtered["created_at"], errors="coerce", utc=True)
                >= pd.Timestamp(date_filter, tz="UTC")
            ]
        if search:
            pattern = search.strip().lower()
            matches = filtered["assessment_id"].str.lower().str.contains(pattern, na=False)
            if "application_ref" in filtered.columns:
                matches |= filtered["application_ref"].fillna("").str.lower().str.contains(
                    pattern, na=False
                )
            filtered = filtered[matches]

        counts = {
            category: int((filtered["risk_category"] == category).sum())
            for category in config.RISK_CATEGORIES
        }
        average_probability = (
            float(filtered["probability"].mean()) if len(filtered) else None
        )
        average_score = float(filtered["risk_score"].mean()) if len(filtered) else None

        kpi_row(
            [
                {"label": "Assessments shown", "value": f"{len(filtered):,}",
                 "help": f"{stats['total']:,} stored in total", "tone": "neutral"},
                {"label": "Low risk", "value": f"{counts['LOW RISK']:,}",
                 "help": "Risk score 0-30", "tone": "low"},
                {"label": "Medium risk", "value": f"{counts['MEDIUM RISK']:,}",
                 "help": "Risk score above 30 up to 60", "tone": "medium"},
                {"label": "High risk", "value": f"{counts['HIGH RISK']:,}",
                 "help": "Risk score above 60 up to 100", "tone": "high"},
                {
                    "label": "Mean predicted probability",
                    "value": format_probability(average_probability) if average_probability is not None else "n/a",
                    "help": "Model prediction, not a confirmed default",
                },
                {
                    "label": "Mean risk score",
                    "value": f"{average_score:.1f}" if average_score is not None else "n/a",
                    "help": f"Over {len(filtered):,} shown assessments",
                },
            ]
        )

        if len(filtered) == 0:
            notice(
                "No stored assessment matches the current filters. Adjust or clear them.",
                kind="warning",
            )
        else:
            left, right = st.columns([1, 1])
            with left:
                show_chart(_risk_donut(counts), key="dash_donut")
            with right:
                show_chart(_probability_histogram(filtered), key="dash_prob_hist")
            if filtered["created_at"].nunique() > 1:
                show_chart(_category_by_month(filtered), key="dash_month")
            if stats["demo_records"]:
                notice(
                    f"{stats['demo_records']} stored record(s) are labelled demonstration data "
                    "(application reference beginning with DEMO). They are included above and "
                    "are marked in Assessment History.",
                    kind="info",
                )

        notice(
            "These probabilities are model <em>predictions</em> for profiles you scored. They "
            "are not confirmed real-world defaults, and no ground-truth outcome is known for "
            "an assessment. The dataset's own observed default rate is shown separately below.",
            kind="info",
        )

    # ---- Model performance summary --------------------------------------
    metrics = get_cached_metrics()
    if metrics:
        selected = metrics.get("selected", {}).get("test", {})
        if selected:
            section(
                "Measured model performance",
                "Read directly from artifacts/metrics/evaluation_metrics.json, produced by the "
                "training run on a held-out test split that was never used for tuning.",
            )
            kpi_row(
                [
                    {"label": "Test ROC-AUC", "value": f"{selected['roc_auc']:.4f}",
                     "help": "Ranking ability; 0.5 is random", "tone": "neutral"},
                    {"label": "Test PR-AUC", "value": f"{selected['pr_auc']:.4f}",
                     "help": "More informative than ROC-AUC when defaults are the minority"},
                    {"label": "Recall (defaults caught)", "value": format_percent(selected['recall'] * 100),
                     "help": f"{selected['true_positives']} of {selected['positives']} defaults flagged at the 0.5 threshold",
                     "tone": "medium"},
                    {"label": "Precision", "value": format_percent(selected['precision'] * 100),
                     "help": f"{selected['false_positives']} false positives at the 0.5 threshold"},
                    {"label": "Brier score", "value": f"{selected['brier_score']:.4f}",
                     "help": "Lower is better; measures probability calibration"},
                ]
            )
        st.page_link("pages/model_insights.py", label="Full model intelligence and evaluation",
                     icon=":material/monitoring:")

    # ---- Dataset reference distributions --------------------------------
    if dataset_ready and dataset is not None:
        section(
            "Reference distributions from the training dataset",
            f"{dataset.provenance.name} - {dataset.provenance.rows:,} historical credit-card "
            f"clients from {dataset.provenance.geography} ({dataset.provenance.observation_period}). "
            "This is the model's training data, not your assessments.",
        )
        notice(
            config.GEOGRAPHIC_DISCLOSURE,
            kind="warning",
            strong="Geographic scope.",
        )
        _dataset_reference_charts(dataset)

        with st.expander("Which requested risk factors does this dataset actually contain?"):
            st.markdown(feature_availability_html(), unsafe_allow_html=True)
            st.markdown(
                "<div class='csa-muted'>Requested fields absent from this dataset are never "
                "estimated or defaulted; the assessment form simply does not ask for them. "
                "See About &amp; Methodology for the full list.</div>",
                unsafe_allow_html=True,
            )

    st.page_link("pages/risk_assessment.py", label="Start New Assessment",
                 icon=":material/assignment:")


render()
