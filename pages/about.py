"""About & Methodology.

The methodological contract of the project, written so that a reviewer can check
the implementation against the claims.
"""

from __future__ import annotations

import streamlit as st

from src import config
from src.ai_report import narrative_provider_status
from src.data_loader import describe_missing_features
from src.explainability import SHAP_AVAILABLE
from src.risk_scoring import describe_bands
from src.ui import (
    get_cached_metadata,
    get_cached_metrics,
    notice,
    page_header,
    require_dataset,
    section,
)


def render() -> None:
    page_header(
        "About & Methodology",
        "CreditSense AI: An Explainable AI-Driven Credit Risk Assessment Framework for "
        "Intelligent Loan Decision Support. What was built, how it decides, and precisely "
        "what it cannot tell you.",
        eyebrow="Documentation",
        model_status=get_cached_metadata() and {"ready": True} or {"ready": False},
    )

    notice(
        f"<strong>{config.APP_NAME}</strong> - {config.APP_TAGLINE}. "
        f"{config.APP_SUPPORTING_TEXT}",
        kind="info",
    )
    notice(config.APP_DISCLAIMER, kind="danger", strong="Academic prototype.")

    tabs = st.tabs(
        [
            "Overview",
            "Data & attribution",
            "Modelling",
            "Explainability",
            "Risk & recommendations",
            "History & reports",
            "Limitations",
            "Running it",
        ]
    )

    # ---- Overview -------------------------------------------------------
    with tabs[0]:
        section("What this application is")
        st.markdown(
            """
            CreditSense AI is an academic decision-support prototype for credit-default risk. It
            trains a machine-learning model on a real public credit-risk dataset, explains every
            prediction, simulates alternative borrower scenarios, applies a transparent
            rule-based review policy, stores assessments locally, and produces a professional
            assessment report.

            The application is built so that a reviewer can verify each claim:

            * every metric on the Model Intelligence page is read from the artifact written by
              the training run - none is typed in;
            * every explanation is computed on the same fitted estimator that produced the
              prediction;
            * every dashboard figure is computed from stored assessments or the loaded dataset;
            * fields the dataset does not contain are shown as unavailable, never estimated;
            * the geographic scope of the data is disclosed on every page that displays a
              prediction.
            """
        )
        section("Feature classes")
        dataset = require_dataset()
        left, right = st.columns(2)
        with left:
            st.markdown(
                f"""
                **Used by the model**
                - Source-observed: {len(config.RAW_NUMERIC_FEATURES) + len(config.CATEGORICAL_FEATURES)}
                  fields actually present in the dataset
                - Derived: {len(config.DERIVED_FEATURES)} fields computed by documented formulas
                - Total model features: {len(config.MODEL_FEATURES)}
                """
            )
        with right:
            st.markdown(
                f"""
                **Deliberately not used**
                - Protected attributes excluded: {', '.join(config.PROTECTED_EXCLUDED_FEATURES)}
                - Identifier column dropped at load time: `record_id`
                - Requested canonical fields absent from this dataset:
                  {len(describe_missing_features(dataset.frame))}
                """
            )
        with st.expander("Requested fields that this dataset does not contain"):
            st.markdown(
                ", ".join(f"`{name}`" for name in config.UNAVAILABLE_FEATURES)
            )
            st.caption(
                "These are not estimated, imputed from elsewhere, or faked in the interface. "
                "The assessment form asks only for fields the model can actually use."
            )

        section("Derived feature formulas")
        st.markdown(
            """
| Derived feature | Formula |
| --- | --- |
| `avg_bill_amount` | mean of the six monthly statement balances |
| `avg_payment_amount` | mean of the six monthly payment amounts |
| `credit_utilization` | latest statement balance / granted credit limit, clipped to [-2, 5] |
| `repayment_ratio` | total payments / total positive balances over six months, clipped to [0, 2] |
| `max_repayment_delay` | maximum repayment status level across the six months |
| `num_delayed_months` | count of months with repayment status >= 1 |
| `num_severely_delayed_months` | count of months with repayment status >= 3 |
| `bill_trend` | latest balance minus the balance six months earlier |
| `payment_trend` | latest payment minus the payment six months earlier |
            """
        )
        st.caption(
            "Derived features are computed inside the saved pipeline, so training and inference "
            "apply exactly the same transformation. Clipping bounds are fixed, documented, and "
            "chosen to limit outlier influence without discarding records."
        )

    # ---- Data -----------------------------------------------------------
    with tabs[1]:
        dataset = require_dataset()
        provenance = dataset.provenance
        section("Dataset")
        st.markdown(
            f"""
            - **Name:** {provenance.name}
            - **Publisher:** {provenance.publisher}
            - **Authors:** {provenance.authors} ({config.DATASET_INFO['publication_year']})
            - **DOI:** {provenance.doi}
            - **Licence:** {provenance.license}
            - **Source:** {provenance.url}
            - **Geography:** {provenance.geography}
            - **Observation period:** {provenance.observation_period}
            - **Currency / units:** {provenance.currency}
            - **Local file:** `{provenance.filename}` with {provenance.rows:,} rows and
              {provenance.columns} columns
            - **SHA-256:** `{provenance.sha256}`
            """
        )
        st.markdown(f"**Required attribution:** {provenance.citation}")
        st.markdown(f"**Source paper:** {config.DATASET_INFO['source_paper']}")

        section("Geographic scope - read this before interpreting any prediction")
        notice(config.GEOGRAPHIC_DISCLOSURE, kind="warning")

        section("Target definition")
        st.markdown(f"Canonical target `{config.TARGET_COLUMN}`. {config.TARGET_DEFINITION}")
        st.markdown(
            "The model is therefore a **default** classifier, not an approval classifier. The "
            "source column was mapped only after its documented meaning was verified."
        )

        section("Data quality")
        st.markdown(
            f"""
            - Missing values in the source file: {dataset.profile['missing_total']}
            - Duplicate feature vectors: {dataset.profile['duplicate_rows_excluding_target']}
              (retained; they may be genuinely distinct clients)
            - Observed default rate: {dataset.profile['target_positive_rate'] * 100:.2f}%
            - Class imbalance ratio: {dataset.profile['class_imbalance_ratio']:.2f} : 1
            - Negative statement balances occur and are preserved: the source documents them as
              credit in the client's favour
            - The published repayment-status scale documents -1, 0 and 1..9; the distributed
              extract also contains -2, which is documented here as 'no consumption / not
              applicable' rather than silently recoded
            """
        )

    # ---- Modelling ------------------------------------------------------
    with tabs[2]:
        metadata = get_cached_metadata()
        metrics = get_cached_metrics()
        section("Training procedure")
        st.markdown(
            f"""
            1. The dataset is loaded, adapted to the canonical schema and validated.
            2. A single stratified split is taken: {int(config.SPLIT_FRACTIONS[0] * 100)}% train,
               {int(config.SPLIT_FRACTIONS[1] * 100)}% validation,
               {int(config.SPLIT_FRACTIONS[2] * 100)}% test, with seed {config.RANDOM_SEED}.
            3. Candidate models are fitted on the **training split only**:
               {', '.join(metadata.get('candidate_models', ['Logistic Regression', 'Random Forest', 'HistGradientBoosting']))}.
            4. The winner is chosen by **validation ROC-AUC** ({metadata.get('selection_criterion', '')}).
            5. Probability calibration (Platt/sigmoid) is fitted on the **validation split** only.
            6. The selected, calibrated model is evaluated **once** on the untouched test split.
               Those numbers are what the application displays.
            7. Preprocessing (imputation, one-hot encoding, scaling) is fitted inside the training
               split only, which is why it is a scikit-learn `Pipeline` saved with the model.
            """
        )
        if metrics.get("overfitting"):
            overfitting = metrics["overfitting"]
            st.markdown(
                f"**Overfitting check:** train ROC-AUC "
                f"{overfitting.get('train_roc_auc', float('nan')):.4f}, validation "
                f"{overfitting.get('validation_roc_auc', float('nan')):.4f}, test "
                f"{overfitting.get('test_roc_auc', float('nan')):.4f}. "
                f"{overfitting.get('warning', '')}"
            )
        section("Class imbalance")
        st.markdown(
            "Defaults are the minority class, so accuracy alone is misleading and is never used "
            "alone. Class-weighted variants are included as *candidates* and compared on the "
            "validation split, because weighting shifts the precision/recall operating point "
            "rather than simply improving the model. Recall, precision, PR-AUC and the full "
            "confusion matrix are all reported."
        )
        section("Probability calibration")
        st.markdown(
            "An uncalibrated score is a ranking, not a probability. A sigmoid (Platt) map is "
            "fitted on the validation split so that predicted probabilities are closer to the "
            "observed default rates. Because the map is monotone, ranking metrics such as "
            "ROC-AUC are unaffected. The reliability curve, Brier score and expected calibration "
            "error are shown on the Model Intelligence page before and after calibration."
        )
        st.markdown(
            "Even calibrated, a predicted probability is an estimate for records resembling this "
            "dataset. A 40% predicted probability does not mean that 40% of similar applicants "
            "will default."
        )

    # ---- Explainability -------------------------------------------------
    with tabs[3]:
        section("How explanations are produced")
        st.markdown(
            f"""
            * Explanations are computed on the **same fitted estimator** that serves predictions.
              SHAP is importable in this environment: **{SHAP_AVAILABLE}**.
            * Tree models use exact tree-path SHAP (`shap.TreeExplainer`); logistic regression
              uses `shap.LinearExplainer`.
            * Contributions are reported in the base model's raw output units (log-odds). The
              displayed probability is a monotone calibration of exactly that output, which
              preserves the direction and ordering of contributions.
            * One-hot encoded categories are summed back onto their source feature so every
              contribution names a field the user actually supplied.
            * If SHAP is unavailable or incompatible, the application falls back to a local
              occlusion heuristic and to permutation importance - always labelled as a fallback,
              never presented as SHAP.
            """
        )
        notice(
            "SHAP values describe how a feature contributed to <em>this model's output for this "
            "input</em>. They are not causal effects and they are not a statement about the "
            "real world.",
            kind="warning",
        )

    # ---- Risk & recommendations ----------------------------------------
    with tabs[4]:
        section("Risk score and categories")
        st.markdown(
            f"""
            The risk score is a linear presentation of the calibrated probability:
            `risk_score = 100 x predicted_default_probability`, rounded half-up to one decimal.
            The bands are {describe_bands()}.

            These thresholds are **prototype presentation choices**. They are not official
            banking thresholds, no regulator defines them, and the underlying probability is
            always reported alongside the score. The same function produces the category
            everywhere in the application, so a profile cannot be classified two different ways
            on two different pages.
            """
        )
        rules = config.RECOMMENDATION_RULES
        section("Recommendation rules")
        st.markdown(
            f"""
            | Rule | Condition | Outcome |
            | --- | --- | --- |
            | R0 | predicted probability > {rules['manual_review_max_probability']:.2f} | ELEVATED RISK - ADDITIONAL ASSESSMENT REQUIRED |
            | R1 | {rules['conditional_max_probability']:.2f} < probability <= {rules['manual_review_max_probability']:.2f} | MANUAL UNDERWRITING REVIEW |
            | R2 | {rules['lower_risk_max_probability']:.2f} < probability <= {rules['conditional_max_probability']:.2f} | CONDITIONAL REVIEW |
            | R3 | probability <= {rules['lower_risk_max_probability']:.2f} | LOWER RISK - ELIGIBLE FOR FURTHER REVIEW |
            | M1 | >= 1 month with status >= 3, or >= 4 delayed months | escalate to at least MANUAL |
            | M2 | >= 3 delayed months, or repayment ratio < {rules['min_tolerable_repayment_ratio']} | escalate to at least CONDITIONAL |

            Modifiers escalate but never de-escalate, so a strong probability signal cannot be
            talked down by a favourable secondary rule. No protected attribute participates in
            any rule, and the recommendation never approves, declines or prices a loan.
            """
        )
        section("What-if analysis")
        st.markdown(
            "The simulator copies the baseline, applies the user's adjustments, re-validates the "
            "result and re-scores it through the same saved pipeline. The baseline record is "
            "never mutated, and saving the alternative creates a separate record. Explanations "
            "are only computed when explicitly requested, so dragging a slider costs nothing."
        )

    # ---- History & reports ---------------------------------------------
    with tabs[5]:
        section("Database")
        st.markdown(
            f"""
            SQLite at `{config.DB_PATH.name}` in the `database/` directory. The schema is created
            automatically, every write uses bound parameters, sorting columns come from a fixed
            allow-list, and raw SQL errors are never shown to the user.

            Stored per assessment: identifier, timestamp, optional application reference, model
            name and version, dataset name and version, the input fields, derived values, the
            calibrated and uncalibrated probabilities, the risk score and category, the
            recommendation, short factor summaries, and flags for demonstration records.

            Not stored: any API key, token or password, and any direct personal identifier. The
            application never asks for a name, address, e-mail address or national identifier,
            and the optional application reference is explicitly documented as a place not to
            put them.
            """
        )
        section("Report generation")
        st.markdown(
            "The PDF is produced locally with ReportLab and contains the attribution, the model "
            "and dataset context, the input summary, the measured probability and score, the "
            "explanation, the recommendation, the limitations and the academic disclaimer. It "
            "is not a bank-issued credit report."
        )
        section("Narrative generation")
        providers = narrative_provider_status()
        st.markdown(
            "Priority: deterministic template (default, always available), then an optional "
            "local Ollama model, then an optional configured endpoint. The language model "
            "receives a fixed fact pack and is never asked to predict risk. Any number in its "
            "output that is not present in the fact pack invalidates the response, and the "
            "template is used instead."
        )
        st.markdown(
            " · ".join(
                f"**{name}**: {'configured' if details['available'] else 'not configured'}"
                for name, details in providers.items()
            )
        )

    # ---- Limitations ----------------------------------------------------
    with tabs[6]:
        section("Known limitations")
        for item in [
            config.GEOGRAPHIC_DISCLOSURE,
            "The data is one historical sample of Taiwanese credit-card clients observed "
            "between April and September 2005, with default measured in October 2005. Economic "
            "conditions, credit products, customer behaviour and reporting standards have all "
            "changed since.",
            "No income, employment, loan purpose, interest rate, loan term or external "
            "credit-score field exists in the dataset, so those canonical risk factors could "
            "not be modelled at all and are never requested.",
            "Class imbalance (about 22% positives) means precision and recall trade off sharply; "
            "the confusion matrix is the honest summary, not a single accuracy number.",
            "Probability calibration is fitted on a single validation split of one dataset and "
            "is not guaranteed to transfer to a different population or period.",
            "The model is a static artifact: there is no monitoring, drift detection or "
            "retraining schedule in this prototype.",
            "Excluding protected attributes does not remove bias. Age, education, credit limit "
            "and repayment behaviour can all act as proxies, and the dataset itself carries "
            "historical selection bias.",
            "Explanations are model behaviour, not causality. A contribution does not license a "
            "claim that changing a field would change the real-world outcome.",
            "The recommendation rules are prototype heuristics. They are not a lending policy, "
            "not compliance advice, and not jurisdiction-aware.",
            "No legal, regulatory or fairness compliance is claimed anywhere in this application. "
            "Real use would require legal review and validation on locally representative, "
            "lawfully obtained data.",
            "Model artifacts, the database and generated reports are all local files; this "
            "prototype has no authentication, no multi-user isolation and no audit trail.",
        ]:
            st.markdown(f"- {item}")

        section("Responsible-use statement")
        st.markdown(
            f"{config.APP_DISCLAIMER} No output of this application may be presented as a real "
            "credit decision, and demonstration records are labelled so they cannot be confused "
            "with real assessments."
        )

    # ---- Running --------------------------------------------------------
    with tabs[7]:
        section("Commands")
        st.code(
            "# 1. install dependencies\n"
            "pip install -r requirements.txt\n\n"
            "# 2. train the model and write the evaluation artifacts\n"
            "python -m src.train_model\n\n"
            "# 3. run the application\n"
            "streamlit run app.py",
            language="bash",
        )
        st.caption(
            "The test suite runs with `python -m pytest -q`. The application refuses to invent a "
            "model: if the artifacts are missing it shows the training command instead of "
            "predicting."
        )
        section("Optional configuration")
        st.markdown(
            """
            * `CREDITSENSE_DB_PATH` - move the SQLite file.
            * `OLLAMA_ENABLED=1`, `OLLAMA_HOST`, `OLLAMA_MODEL` - enable local narrative
              generation.
            * `LLM_API_BASE`, `LLM_API_KEY`, `LLM_MODEL` - enable an OpenAI-compatible endpoint.
            * `CREDITSENSE_LOG_LEVEL` - logging verbosity.

            All optional integrations are off by default; none is required for the core
            application, and secrets belong in environment variables or the deployment's secret
            store, never in the repository.
            """
        )


render()
