"""Dataset Explorer.

This page documents the data the model was trained on: what it contains, what it
does not contain, and how it behaves. Protected attributes appear only as
group-level aggregates on the Fairness tab and are never presented as model inputs.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config, fairness
from src.data_loader import describe_missing_features, dataset_fingerprint
from src.data_validation import issues_to_frame, validate_dataset
from src.preprocessing import build_design_matrix
from src.ui import (
    PLOT_COLORS,
    apply_chart_theme,
    get_cached_profile,
    kpi_row,
    model_status,
    notice,
    page_header,
    require_dataset,
    require_model,
    section,
    show_chart,
)
from src.utils import feature_label, format_percent


def _filters(frame: pd.DataFrame) -> pd.DataFrame:
    filtered = frame.copy()
    columns = st.columns([1.2, 1, 1, 1])
    with columns[0]:
        age_range = st.slider(
            "Age range",
            min_value=int(frame["age"].min()),
            max_value=int(frame["age"].max()),
            value=(int(frame["age"].min()), int(frame["age"].max())),
            key="de_age",
        )
    with columns[1]:
        education = st.multiselect(
            "Education",
            options=config.EDUCATION_CATEGORIES,
            default=list(config.EDUCATION_CATEGORIES),
            format_func=lambda value: config.EDUCATION_DISPLAY.get(value, value),
            key="de_edu",
        )
    with columns[2]:
        target = st.multiselect(
            "Observed outcome",
            options=[0, 1],
            default=[0, 1],
            format_func=lambda value: "Default (1)" if value == 1 else "No default (0)",
            key="de_target",
        )
    with columns[3]:
        limit_min, limit_max = float(frame["credit_limit"].min()), float(frame["credit_limit"].max())
        limit_range = st.slider(
            "Credit limit range (NT$)",
            min_value=limit_min,
            max_value=limit_max,
            value=(limit_min, limit_max),
            step=10_000.0,
            key="de_limit",
        )
    filtered = filtered[filtered["age"].between(*age_range)]
    filtered = filtered[filtered["education"].isin(education or config.EDUCATION_CATEGORIES)]
    filtered = filtered[filtered[config.TARGET_COLUMN].isin(target or [0, 1])]
    filtered = filtered[filtered["credit_limit"].between(*limit_range)]
    return filtered


def render() -> None:
    status = model_status()
    page_header(
        "Dataset Explorer",
        "What the model was trained on, in detail: schema, quality, distributions and the "
        "group-level behaviour of the fitted model. Nothing here is synthetic unless the "
        "banner says so.",
        eyebrow="Data provenance and quality",
        model_status=status,
    )

    dataset = require_dataset()
    frame = dataset.frame
    provenance = dataset.provenance

    # ---- Attribution ----------------------------------------------------
    section("Source and attribution")
    st.markdown(
        f"""
        <div class="csa-card">
          <div class="csa-card-title">{provenance.name} - {provenance.publisher}</div>
          <div class="csa-card-body">
            <b>Authors</b> {provenance.authors} ({config.DATASET_INFO['publication_year']})<br/>
            <b>DOI</b> {provenance.doi}<br/>
            <b>Licence</b> {provenance.license}<br/>
            <b>Source</b> <span class="csa-mono">{provenance.url}</span><br/>
            <b>Geography</b> {provenance.geography}<br/>
            <b>Observation period</b> {provenance.observation_period}<br/>
            <b>Units</b> {provenance.currency}<br/>
            <b>Local file</b> <span class="csa-mono">{provenance.filename}</span><br/>
            <b>SHA-256</b> <span class="csa-mono">{provenance.sha256[:32]}...</span><br/>
            <b>Dataset fingerprint (used to tag models)</b>
            <span class="csa-mono">{dataset_fingerprint(dataset)}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"Required attribution: {provenance.citation}")
    notice(config.GEOGRAPHIC_DISCLOSURE, kind="warning", strong="Geographic scope.")
    if dataset.is_synthetic:
        notice(config.SYNTHETIC_DISCLOSURE, kind="danger", strong="Synthetic data in use.")

    notice(
        "A licence that permits sharing does not make the data about a real population you "
        "intend to serve. This sample describes Taiwanese credit-card clients in 2005 and is "
        "not representative of Indian borrowers or of any current portfolio.",
        kind="info",
    )

    # ---- Headline numbers ----------------------------------------------
    profile = dataset.profile
    kpi_row(
        [
            {"label": "Rows", "value": f"{profile['rows']:,}"},
            {"label": "Columns", "value": profile["columns"]},
            {"label": "Missing values", "value": f"{profile['missing_total']:,}"},
            {"label": "Duplicate feature rows",
             "value": f"{profile['duplicate_rows_excluding_target']:,}",
             "help": "Identical feature vectors; retained rather than silently dropped"},
            {"label": "Observed default rate",
             "value": format_percent(profile["target_positive_rate"] * 100),
             "tone": "medium"},
            {"label": "Class imbalance",
             "value": f"{profile['class_imbalance_ratio']:.2f} : 1"},
        ]
    )

    tabs = st.tabs(
        ["Overview", "Distributions", "Relationships", "Data quality", "Fairness", "Export"]
    )

    # ---- Overview -------------------------------------------------------
    with tabs[0]:
        section("Schema")
        dtypes = pd.DataFrame(
            {
                "Column": frame.columns,
                "Type": [str(dtype) for dtype in frame.dtypes],
                "Non-null": [int(frame[column].notna().sum()) for column in frame.columns],
                "Unique": [int(frame[column].nunique(dropna=True)) for column in frame.columns],
                "Example": [frame[column].dropna().iloc[0] for column in frame.columns],
            }
        )
        st.dataframe(dtypes, use_container_width=True, hide_index=True)

        section("Target definition")
        st.markdown(f"**Canonical target:** `{config.TARGET_COLUMN}`")
        st.markdown(config.TARGET_DEFINITION)
        st.markdown(
            "The source column is `default.payment.next.month`. It records an observed "
            "default, not a loan-approval decision, so the model is a default model and not "
            "an approval model."
        )

        section("Feature classes")
        observed = frame[list(config.RAW_NUMERIC_FEATURES + config.CATEGORICAL_FEATURES)]
        st.markdown(
            f"- **Source-observed features used by the model:** "
            f"{len(config.RAW_NUMERIC_FEATURES) + len(config.CATEGORICAL_FEATURES)}"
        )
        st.markdown(f"- **Derived features used by the model:** {len(config.DERIVED_FEATURES)}")
        st.markdown(
            f"- **Present in the dataset but excluded as protected:** "
            f"{', '.join(config.PROTECTED_EXCLUDED_FEATURES)}"
        )
        missing_features = describe_missing_features(frame)
        st.markdown(
            f"- **Requested canonical features that this dataset does not contain:** "
            f"{len(missing_features)}"
        )
        with st.expander("Which requested risk factors are unavailable?"):
            st.markdown(
                "These fields do not exist in the dataset. They are shown as unavailable in "
                "the application rather than being estimated or defaulted:"
            )
            st.markdown(", ".join(f"`{name}`" for name in missing_features))
            st.caption(
                "Because of this, the assessment form asks only for the fields the model "
                "actually uses. A richer product would need a dataset that contains those "
                "factors, and would need to be re-validated."
            )

        section("Descriptive statistics")
        # Computed from the design matrix, which carries the derived features that the
        # raw source frame does not contain.
        design = build_design_matrix(frame)
        stats_frame = (
            design[list(config.NUMERIC_FEATURES)].describe().T.reset_index()
            if not design.empty
            else pd.DataFrame()
        )
        if not stats_frame.empty:
            stats_frame = stats_frame.rename(columns={"index": "feature"})
            st.dataframe(stats_frame, use_container_width=True, hide_index=True)
            st.caption(
                f"{len(config.RAW_NUMERIC_FEATURES)} source-observed numeric features plus "
                f"{len(config.DERIVED_FEATURES)} derived features, computed with the same "
                "formulas the model uses."
            )

    # ---- Distributions --------------------------------------------------
    with tabs[1]:
        section("Filtered distribution explorer")
        filtered = _filters(frame)
        st.caption(f"{len(filtered):,} of {len(frame):,} rows match the current filters.")

        if filtered.empty:
            notice("No rows match the current filters.", kind="warning")
        else:
            # Plot from the design matrix so the derived features are selectable too;
            # they are not present in the raw source frame.
            filtered_design = build_design_matrix(filtered)
            filtered_design[config.TARGET_COLUMN] = filtered[config.TARGET_COLUMN].values
            feature = st.selectbox(
                "Feature to plot",
                options=list(config.NUMERIC_FEATURES),
                format_func=feature_label,
                key="de_feature",
            )
            left, right = st.columns(2)
            with left:
                figure = go.Figure(
                    go.Histogram(
                        x=filtered_design[feature],
                        nbinsx=45,
                        marker=dict(color=PLOT_COLORS["blue"],
                                    line=dict(color="#FFFFFF", width=0.5)),
                        hovertemplate="%{x}<br>%{y} rows<extra></extra>",
                    )
                )
                figure.update_xaxes(title_text=feature_label(feature))
                figure.update_yaxes(title_text="Rows")
                show_chart(apply_chart_theme(figure, height=330, show_legend=False,
                                             title="Histogram"), key="de_hist")
            with right:
                figure = go.Figure()
                for outcome, color, label in ((0, PLOT_COLORS["low"], "No default"),
                                              (1, PLOT_COLORS["high"], "Default")):
                    subset = filtered_design[
                        filtered_design[config.TARGET_COLUMN] == outcome
                    ]
                    figure.add_box(
                        y=subset[feature],
                        name=label,
                        marker_color=color,
                        boxpoints="outliers",
                        hovertemplate=label + "<br>%{y}<extra></extra>",
                    )
                figure.update_yaxes(title_text=feature_label(feature))
                show_chart(apply_chart_theme(figure, height=330, title="By observed outcome"),
                           key="de_box")

            figure = go.Figure()
            for outcome, color, label in ((0, PLOT_COLORS["low"], "No default"),
                                          (1, PLOT_COLORS["high"], "Default")):
                subset = filtered_design[
                    filtered_design[config.TARGET_COLUMN] == outcome
                ]
                figure.add_histogram(
                    x=subset[feature], name=label, opacity=0.62, nbinsx=45,
                    marker_color=color,
                    hovertemplate=label + "<br>%{x}<br>%{y} rows<extra></extra>",
                )
            figure.update_layout(barmode="overlay")
            figure.update_xaxes(title_text=feature_label(feature))
            figure.update_yaxes(title_text="Rows")
            show_chart(apply_chart_theme(figure, height=340,
                                         title="Distribution by observed outcome"),
                       key="de_overlay")

            section("Sample records")
            sample_size = st.slider(
                "Number of rows to display", min_value=5, max_value=50, value=10, step=5,
                key="de_sample",
            )
            display_columns = [
                config.TARGET_COLUMN,
                "credit_limit",
                "age",
                "education",
                "repay_status_1",
                "bill_amount_1",
                "payment_amount_1",
            ]
            if feature not in display_columns:
                display_columns.append(feature)
            st.dataframe(
                filtered_design[display_columns].head(sample_size),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "These are records from the public research dataset above, not customer "
                "records collected by this application. The identifier column was dropped "
                "during loading."
            )

    # ---- Relationships --------------------------------------------------
    with tabs[2]:
        section("Correlation structure")
        design = build_design_matrix(frame)
        design[config.TARGET_COLUMN] = frame[config.TARGET_COLUMN].values
        numeric = design.select_dtypes(include="number")
        correlation = numeric.corr(numeric_only=True)
        figure = go.Figure(
            go.Heatmap(
                z=correlation.values,
                x=correlation.columns,
                y=correlation.columns,
                colorscale="RdBu",
                zmid=0,
                zmin=-1,
                zmax=1,
                hovertemplate="%{y} vs %{x}<br>correlation %{z:.3f}<extra></extra>",
                colorbar=dict(title="r", thickness=12),
            )
        )
        figure.update_xaxes(tickangle=-45, tickfont=dict(size=9))
        figure.update_yaxes(tickfont=dict(size=9))
        show_chart(apply_chart_theme(figure, height=760, show_legend=False,
                                     title="Correlation matrix including derived features"),
                   key="de_corr")

        target_correlation = (
            correlation[config.TARGET_COLUMN]
            .drop(config.TARGET_COLUMN)
            .sort_values(key=lambda series: series.abs(), ascending=False)
        )
        top_correlations = pd.DataFrame(
            {
                "Feature": [feature_label(name) for name in target_correlation.index[:12]],
                "Correlation with observed default": [
                    round(float(value), 4) for value in target_correlation.values[:12]
                ],
            }
        )
        st.markdown("**Strongest linear relationships with the observed outcome**")
        st.dataframe(top_correlations, use_container_width=True, hide_index=True)
        st.caption(
            "Correlation describes linear association in this sample only. It is not a causal "
            "relationship, and it does not by itself justify including a variable in a credit "
            "model."
        )

        section("Repayment delay and outcome")
        delay_frame = (
            frame.assign(months_delayed=(frame[config.REPAY_STATUS_COLUMNS] >= 1).sum(axis=1))
            .groupby("months_delayed", observed=True)[config.TARGET_COLUMN]
            .agg(["count", "mean"])
            .reset_index()
        )
        figure = go.Figure(
            go.Bar(
                x=delay_frame["months_delayed"],
                y=delay_frame["mean"],
                marker_color=PLOT_COLORS["high"],
                customdata=delay_frame["count"],
                hovertemplate="Months delayed %{x}<br>observed default rate %{y:.3f}"
                              "<br>%{customdata} clients<extra></extra>",
            )
        )
        figure.update_xaxes(title_text="Number of the last 6 months with a delayed payment")
        figure.update_yaxes(title_text="Observed default rate", tickformat=".0%")
        show_chart(apply_chart_theme(figure, height=330, show_legend=False,
                                     title="Observed default rate by delay count"),
                   key="de_delay")

    # ---- Data quality ---------------------------------------------------
    with tabs[3]:
        section("Automated quality report")
        report = validate_dataset(frame, provenance)
        notice(report.summary_line(),
               kind="success" if report.is_valid else "danger")
        st.dataframe(issues_to_frame(report), use_container_width=True, hide_index=True)
        st.caption(
            "Validation never repairs data silently. Records with negative statement balances "
            "are preserved because the source dataset documents them as credit in the client's "
            "favour, and duplicate feature vectors are retained because they may be genuinely "
            "distinct clients with identical profiles."
        )
        if report.context:
            st.markdown("**Measured context**")
            st.dataframe(
                [{"Property": key, "Value": value} for key, value in report.context.items()],
                use_container_width=True,
                hide_index=True,
            )

    # ---- Fairness -------------------------------------------------------
    with tabs[4]:
        section(
            "Group-level behaviour of the fitted model",
            "Descriptive only. Group labels come from the source dataset and are never model "
            "inputs.",
        )
        for caveat in fairness.FAIRNESS_CAVEATS:
            st.caption(f"- {caveat}")

        if not status.get("ready"):
            notice("The model is not available, so group-level predictions cannot be computed.",
                   kind="warning")
        else:
            predictor = require_model()
            sample_size = st.slider(
                "Sample of dataset rows to score",
                min_value=500, max_value=min(5000, len(frame)),
                value=min(3000, len(frame)), step=500, key="de_fair_n",
            )
            group_column = st.selectbox(
                "Grouping field",
                options=[column for column in ["sex", "education", "marriage", "age"]
                         if column in frame.columns],
                format_func=lambda value: (
                    f"{value} (protected attribute - excluded from the model)"
                    if value in config.PROTECTED_EXCLUDED_FEATURES
                    else f"{value} (age bands)"
                    if value == "age"
                    else value
                ),
                key="de_fair_group",
            )
            sample = frame.sample(n=sample_size, random_state=config.RANDOM_SEED)
            predictions = predictor.predict_frame(sample)
            probabilities = [item.probability for item in predictions]

            metric_table = fairness.group_metrics(
                sample,
                sample[config.TARGET_COLUMN],
                probabilities,
                group_column=group_column,
            )
            if metric_table is None:
                notice("No group metrics could be computed for that field.", kind="warning")
            else:
                st.dataframe(metric_table.table, use_container_width=True, hide_index=True)
                disagreement = fairness.disparity_table(metric_table)
                st.markdown("**Relative to the largest group in the sample**")
                st.dataframe(disagreement, use_container_width=True, hide_index=True)
                st.caption(
                    "Rates are computed at the 0.5 decision threshold on a random sample of "
                    "the dataset, using the model's calibrated probabilities. Small groups "
                    "produce unstable rates."
                )

            st.markdown("**Threshold trade-off on the same sample**")
            st.dataframe(
                fairness.threshold_tradeoff(sample[config.TARGET_COLUMN], probabilities),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "The default operating threshold is 0.5. Lowering it catches more defaults "
                "but flags more applicants who would not have defaulted."
            )

    # ---- Export ---------------------------------------------------------
    with tabs[5]:
        section("Export processed data")
        design = build_design_matrix(frame)
        design.insert(0, config.TARGET_COLUMN, frame[config.TARGET_COLUMN].values)
        st.markdown(
            f"The processed matrix contains {design.shape[0]:,} rows and "
            f"{design.shape[1]} columns: the canonical inputs the model consumes plus the "
            "documented derived features and the observed target."
        )
        st.dataframe(design.head(20), use_container_width=True, hide_index=True)
        st.download_button(
            "Download processed dataset (CSV)",
            data=design.to_csv(index=False).encode("utf-8"),
            file_name=f"creditsense-processed-{dataset_fingerprint(dataset).split(':')[-1][:12]}.csv",
            mime="text/csv",
        )
        notice(
            f"The source is licensed {provenance.license}, which permits sharing and adaptation "
            "with attribution. Any redistribution must keep this attribution: "
            f"{provenance.citation}",
            kind="info",
            strong="Attribution is required.",
        )
        st.caption(
            "No confidential data is bundled by this project: the dataset is a public research "
            "file, and the application stores only the assessment fields you enter. Protected "
            "attributes are not included in the exported model matrix because the model does "
            "not use them."
        )


render()
