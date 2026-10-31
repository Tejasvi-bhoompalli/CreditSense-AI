# CreditSense AI

### An Explainable AI-Driven Credit Risk Assessment Framework for Intelligent Loan Decision Support

**Explainable Intelligence for Smarter Credit Decisions**

> **CreditSense AI is an academic research and decision-support prototype.** Its predictions
> are estimates produced from the available dataset and may be inaccurate or biased. It is
> not a credit bureau, lender, or substitute for a regulated underwriting process. It must
> not be used as the sole basis for an actual lending decision.

---

## 1. Abstract

CreditSense AI is an end-to-end, explainable credit-default risk application. It trains and
evaluates a machine-learning pipeline on a real public credit-risk dataset, explains every
prediction with SHAP, simulates alternative borrower scenarios, applies a transparent
rule-based review policy, stores assessments in a local SQLite database, and produces a
professionally typeset PDF report.

The project is deliberately conservative about what it claims. Every metric in the interface
is read from the artifact written by the training run on a held-out test split. Every
explanation is computed on the same fitted estimator that produced the prediction. Fields
that the dataset does not contain are shown as **unavailable** rather than estimated. The
geographic scope of the data is disclosed everywhere a prediction appears.

**Dataset:** Default of Credit Card Clients (UCI Machine Learning Repository), collected in
**Taiwan** in 2005. This is **not an India-specific lending dataset** and the model is **not
validated for Indian borrowers**.

---

## 2. Problem statement

Consumer credit underwriting must decide, before a facility is granted, whether a borrower is
likely to default. Two problems make this hard in practice:

1. **Discrimination is easy to claim and hard to demonstrate.** A ranking model can be accurate
   and still be useless operationally, because the decision depends on a threshold, a cost
   asymmetry, and whether the predicted probability is calibrated rather than merely ordered.
2. **Opacity blocks adoption and review.** A score without an explanation cannot be
   challenged, audited, or argued with by the person it affects, and offers no diagnostic
   signal when it is wrong.

CreditSense AI addresses both: a measured, calibrated model with an explicit evaluation
contract, and a first-class explainability layer that is never a proxy for the serving model.

---

## 3. Motivation

Public research datasets make it possible to build and demonstrate a full credit-risk pipeline
without touching confidential customer data. Doing it well means resisting three temptations:
reporting a flattering metric, presenting a placeholder as a measurement, and failing to say
which population the numbers describe. This project is organised around refusing all three.

---

## 4. Objectives

1. Adapt a real credit-risk dataset to a canonical schema, with verified provenance.
2. Validate data quality and document every anomaly instead of silently repairing it.
3. Build a reproducible, leakage-free training pipeline with an untouched test split.
4. Estimate default probability with measured calibration quality.
5. Explain predictions with SHAP on the serving model, with labelled fallbacks.
6. Provide scenario ("what-if") analysis that re-scores rather than guesses.
7. Implement a transparent, testable review-recommendation engine.
8. Persist assessments locally with parameterised SQL and no unnecessary personal data.
9. Generate a professional PDF assessment report.
10. Disclose limitations, geographic scope and responsible-use constraints prominently.

---

## 5. Key features

| Area | What it does |
| --- | --- |
| **Risk assessment** | Validated multi-section form; only fields the model actually uses; probability, risk score, category, recommendation and explanation in one panel |
| **Explainable AI** | Global SHAP importance for the model, local feature attributions per profile, plain-English translation, explicit method caveats |
| **What-if simulator** | Adjust any supported input, re-score through the same saved pipeline, compare baseline vs alternative, export the comparison, optionally save the alternative as a new record |
| **Model intelligence** | Candidate comparison, ROC / PR / reliability curves, confusion matrix, overfitting check, permutation importance, class balance, metric glossary |
| **Dataset explorer** | Provenance and licence, schema, quality report, filters, distributions, correlation matrix, group-level behaviour, processed-data export |
| **Assessment history** | Search, date and probability filters, sorting, pagination, detail view, CSV export, PDF export, confirmation-gated deletion |
| **Reports** | Deterministic (or optionally LLM-assisted) narrative plus a ReportLab PDF with attribution, measured metrics, limitations and disclaimer |
| **Guardrails** | No fabricated metrics, no fake applicants, labelled demonstration records, labelled fallbacks, retrain instructions instead of silent failure |

---

## 6. System architecture

```
                       data/UCI_Credit_Card.csv
                                 |
                    src/data_loader.py  (discover, detect target,
                                 |       canonicalise, profile)
                                 v
                  src/data_validation.py  (errors / warnings / notes)
                                 |
      +--------------------------+---------------------------+
      |                                                      |
      v                                                      v
 src/train_model.py  ---------------->  artifacts/  -->  src/predictor.py
 (split, compare, select,               model_bundle.joblib     |
  calibrate, evaluate, persist)         metrics/*.json          |
                                        metadata/*.json         |
      |                                                         |
      v                                                         v
 artifacts/metrics/evaluation_metrics.json            PredictionResult
      |                                                         |
      |                            +----------------------------+-------------------+
      |                            |               |                |              |
      v                            v               v                v              v
 src/explainability.py      src/risk_scoring.py  src/recommendation.py  src/what_if.py  src/fairness.py
 (SHAP + fallbacks)         (single mapping)     (documented rules)     (scenarios)     (group metrics)
      |                            |               |                |              |
      +----------------------------+---------------+---------------+--------------+
                                                 |
                        +------------------------+------------------------+
                        v                                                 v
              src/database.py (SQLite)                       src/report_generator.py (ReportLab PDF)
                        |                                                 ^
                        v                                                 |
                  pages/*.py  <-- Streamlit UI (app.py, st.navigation) ---+
                        |
                        v
              src/ai_report.py (deterministic narrative; optional local LLM)
```

---

## 7. Technology stack

| Layer | Choice |
| --- | --- |
| UI | Streamlit 1.66 with a custom design system (`assets/styles.css`), Plotly 7 charts |
| Data | pandas 3.1, NumPy 2.5 |
| ML | scikit-learn 1.9 (`Pipeline`, `ColumnTransformer`, logistic regression, random forest, histogram gradient boosting), joblib |
| Explainability | SHAP 0.54 (`TreeExplainer`, `LinearExplainer`) with a documented occlusion fallback and permutation importance |
| Database | SQLite (standard library, WAL mode, parameterised queries) |
| Reporting | ReportLab 5.1 (PDF; charts drawn natively, no image dependencies) |
| Validation | explicit validation functions with severities (Pydantic available but not required at runtime) |
| Testing | pytest 9 (plus pypdf for PDF text verification) |
| Optional AI | Ollama locally, or any OpenAI-compatible endpoint, only when explicitly configured |

---

## 8. Dataset source and licence (verified)

| Property | Value |
| --- | --- |
| Name | Default of Credit Card Clients |
| Publisher | UCI Machine Learning Repository |
| Authors | I-Cheng Yeh and Che-hui Lien |
| Year | 2009 (donated 2016) |
| DOI | [10.24432/C55S3H](https://doi.org/10.24432/C55S3H) |
| URL | <https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients> |
| **Licence** | **CC BY 4.0** (Creative Commons Attribution 4.0 International) |
| Geography | **Taiwan** |
| Period | April - September 2005 observed; target is default in October 2005 |
| Records | 30,000 rows; 23 feature columns plus the target |
| Currency | New Taiwan Dollar (NT$) |
| Local file | `data/UCI_Credit_Card.csv`, SHA-256 `a0f0ab49d6326671...`, 2,862,995 bytes |
| Local fingerprint | `UCI_Credit_Card.csv:a0f0ab49d6326671` |

**Required attribution** (also printed in every generated PDF):

> Yeh, I. (2009). *Default of Credit Card Clients* [Dataset]. UCI Machine Learning Repository.
> <https://doi.org/10.24432/C55S3H>

**Source paper:**

> Yeh, I-C., & Lien, C-h. (2009). The comparisons of data mining techniques for the predictive
> accuracy of probability of default of credit card clients. *Expert Systems with Applications*,
> 36(2), 2473-2480.

### 8.1 Geographic scope (read before interpreting any prediction)

> **This dataset was collected in Taiwan in 2005. It is NOT an India-specific lending dataset
> and contains no Indian borrower records. Model outputs are therefore not validated for
> Indian borrowers, Indian credit products, Indian income levels, or Indian regulatory
> definitions of default. Any application of this prototype to an Indian lending portfolio
> would require re-training and validation on locally representative, lawfully obtained data.**

Additional context: a licence that permits sharing does not make the data representative of the
population you intend to serve. Historical selection bias is baked into who was granted credit
in 2005 in one country.

---

## 9. Data dictionary

### 9.1 Source-observed features used by the model

| Canonical name | Source | Meaning | Units |
| --- | --- | --- | --- |
| `credit_limit` | `LIMIT_BAL` | Amount of credit granted (includes family/supplementary credit) | NT$ |
| `age` | `AGE` | Age of the client | years |
| `education` | `EDUCATION` | Education level (`graduate_school`, `university`, `high_school`, `other`) | category |
| `repay_status_1` .. `repay_status_6` | `PAY_0`, `PAY_2` .. `PAY_6` | Repayment status, most recent month back to six months prior | level (-2..8) |
| `bill_amount_1` .. `bill_amount_6` | `BILL_AMT1` .. `BILL_AMT6` | Statement balance, most recent month back | NT$ |
| `payment_amount_1` .. `payment_amount_6` | `PAY_AMT1` .. `PAY_AMT6` | Amount paid, most recent month back | NT$ |

Repayment-status scale: `-1` paid duly, `0` revolving credit (minimum paid, no delay), `1..8`
payment delayed by that many months. The distributed extract also contains `-2`, documented
here as "no consumption / not applicable".

### 9.2 Derived features (computed inside the saved pipeline)

| Feature | Formula | Clipping |
| --- | --- | --- |
| `avg_bill_amount` | mean of `bill_amount_1..6` | none |
| `avg_payment_amount` | mean of `payment_amount_1..6` | none |
| `credit_utilization` | `bill_amount_1 / credit_limit` | `[-2, 5]` |
| `repayment_ratio` | `sum(payment_amount) / max(sum(max(bill_amount, 0)), 1)` | `[0, 2]` |
| `max_repayment_delay` | `max(repay_status_1..6)` | none |
| `num_delayed_months` | `count(repay_status_i >= 1)` | none |
| `num_severely_delayed_months` | `count(repay_status_i >= 3)` | none |
| `bill_trend` | `bill_amount_1 - bill_amount_6` | none |
| `payment_trend` | `payment_amount_1 - payment_amount_6` | none |

**Total model features: 30** (21 raw inputs + 9 derived). Clipping bounds are fixed and
documented; they limit outlier influence without discarding records.

### 9.3 Present in the dataset but excluded from the model

`sex` and `marriage` are protected attributes. They are excluded from the feature set and
appear only as group-level descriptors on the Dataset Explorer's fairness tab. Excluding an
attribute does **not** remove bias: `age`, `education`, `credit_limit` and repayment behaviour
can all act as proxies, and this is disclosed wherever group results are shown.

### 9.4 Requested canonical features that this dataset does not contain

`annual_income`, `employment_status`, `employment_years`, `credit_score`, `credit_history_length`,
`loan_amount`, `loan_term`, `interest_rate`, `debt_to_income_ratio`, `existing_loans`,
`existing_monthly_obligations`, `savings_balance`, `previous_late_payments`, `previous_defaults`,
`loan_purpose`, `home_ownership`, `loan_to_income_ratio`.

These are **not** estimated, imputed from elsewhere, or faked in the interface. The assessment
form asks only for the fields the model can actually use, and this list is shown in the
application (Overview and About pages) and in the README so the gap between the brief's
canonical schema and the data actually available is explicit.

---

## 10. Target definition

Canonical target: **`default`**, mapped from `default.payment.next.month`.

* `1` = the client defaulted on the credit-card payment in the following month (October 2005)
* `0` = no recorded default

This is a genuine **observed default** outcome, not a loan-approval decision. The target
column is detected by name and verified to be binary before use; a loan-approval column
(`loan_status`-style) is deliberately **not** accepted as a default label, and detection fails
loudly rather than guessing.

Observed distribution over the whole file: **23,364 non-defaults (77.88%)** and **6,636
defaults (22.12%)**; class imbalance ratio **3.52 : 1**.

---

## 11. Data preprocessing

1. **Discovery.** Supported files under `data/` are found automatically; known filenames are
   preferred, then the largest file.
2. **Target detection and canonicalisation.** Source columns are normalised to snake_case,
   the target is detected and verified binary, and verified source columns are mapped onto the
   canonical schema. The `ID` column is dropped.
3. **Domain cleaning.** `EDUCATION` codes `0, 4, 5, 6` are grouped into `other` (documented
   codes are `1` graduate school, `2` university, `3` high school). Nothing else is recoded.
4. **Validation.** 13 structural and domain checks produce errors, warnings and notes
   (see §11.2).
5. **Derived features** are added by `DerivedFeatureAdder` **inside the pipeline**, so training
   and inference cannot diverge.
6. **Imputation and encoding** happen inside a `ColumnTransformer`: median imputation for
   numeric features, most-frequent imputation plus one-hot encoding for `education`.
   `OneHotEncoder` is configured with `handle_unknown="ignore"` and `sparse_output=False`
   (histogram gradient boosting requires dense input). Scaling is applied only for the linear
   model candidate.
7. **Fitting discipline.** The preprocessor is fitted on the training split only. Validation and
   test data never contribute an imputation statistic or an encoding category.

### 11.1 Documented quirks (preserved, not repaired)

* Negative `bill_amount_*` values (1,930 client-months) are credit in the client's favour and
  are kept as recorded.
* The published repayment-status scale omits `-2`, which occurs in the distributed extract; it
  is documented rather than recoded.
* 56 rows share identical feature values excluding the target. They are retained: two different
  clients can legitimately have identical profiles.
* The shipped file has **no missing values**, so no imputation was needed for it; the imputers
  exist so the pipeline behaves correctly on data that does have gaps.

### 11.2 Validation checks performed

Required columns present · target present · target numeric · target binary · missing values per
feature · duplicate feature rows · positive credit limit · age within 18-100 · repayment status
within documented scale · no negative payments · negative statement balances (informational) ·
constant columns · provenance disclosure (geographic scope or synthetic-data notice).

---

## 12. Model training methodology

Command: `python -m src.train_model`

1. Load, adapt and validate the dataset (training refuses to start if validation reports an
   error).
2. **Single stratified split, 60 / 20 / 20, seed 42** → 18,000 train / 6,000 validation /
   6,000 test rows. Class counts: train 14,018 / 3,982; validation 4,673 / 1,327; test
   4,673 / 1,327.
3. Fit five candidate pipelines on the **training split only**:
   logistic regression, class-weighted logistic regression, random forest, class-weighted random
   forest, histogram gradient boosting.
4. Select by **validation ROC-AUC** (ties broken by validation PR-AUC, then Brier score).
5. Fit **Platt (sigmoid) calibration on the validation split only**, using the already-selected
   model.
6. Evaluate the selected calibrated model **once** on the untouched test split. Those numbers are
   what the application displays.
7. Persist the pipeline, calibrator, feature schema, metadata, metrics, permutation importance
   and global SHAP ranking.

`artifacts/metrics/evaluation_metrics.json` records `test_split_used_for_tuning: false` and
`test_split_metrics_are_measured: true`. No hyper-parameter or threshold decision is made on
test data.

**Leakage controls.** No identifier is used. No post-outcome field exists in the dataset to
leak (all features are prior behaviour). Preprocessing statistics are fitted on the training
split only. The calibrator sees only validation data.

---

## 13. Model comparison (measured)

Validation-split results, 6,000 rows, calibrated probabilities at threshold 0.5:

| Model | Val ROC-AUC | Val PR-AUC | Val Brier | Val precision | Val recall | Selected |
| --- | --- | --- | --- | --- | --- | --- |
| **Random Forest (class-weighted)** | **0.7674** | **0.5254** | 0.1389 | 0.6562 | 0.3165 | **yes** |
| HistGradientBoosting | 0.7668 | 0.5290 | — | — | — | no |
| Random Forest | 0.7648 | 0.5212 | — | — | — | no |
| Logistic Regression (class-weighted) | 0.7428 | 0.4904 | — | — | — | no |
| Logistic Regression | 0.7374 | 0.4935 | — | — | — | no |

(All five candidates' train and validation metrics are stored in
`artifacts/metrics/evaluation_metrics.json` and rendered on the Model Intelligence page. The
`—` cells are shown on that page; they are omitted here for brevity.)

Selection criterion: highest validation ROC-AUC. Class-weighted variants were included as
**candidates** rather than applied by default, because class weighting moves the
precision/recall operating point and should be justified by evidence rather than assumed.

Reference point: predicting the majority class for every record scores **0.7788 accuracy with
ROC-AUC 0.5000 and recall 0.0000**. Accuracy is therefore never reported alone.

---

## 14. Evaluation metrics (measured, untouched test split)

Selected model: **Random Forest (class-weighted)**, calibrated, threshold 0.5, 6,000 test rows
(1,327 defaults, 22.12%).

| Metric | Test (calibrated) | Test (uncalibrated) | Train |
| --- | --- | --- | --- |
| Accuracy | 0.8185 | 0.8100 | 0.9223 |
| Balanced accuracy | 0.6453 | 0.6867 | 0.8329 |
| Precision | 0.6831 | 0.5891 | 0.9657 |
| Recall | 0.3346 | 0.4657 | 0.6725 |
| F1 | 0.4492 | 0.5202 | 0.7929 |
| ROC-AUC | 0.7812 | 0.7812 | 0.9925 |
| PR-AUC | 0.5633 | 0.5633 | 0.9698 |
| Brier score | 0.1346 | 0.1400 | 0.0586 |
| Specificity | 0.9559 | 0.9078 | 0.9932 |
| TN / FP / FN / TP | 4467 / 206 / 883 / 444 | 4242 / 431 / 709 / 618 | 13923 / 95 / 1304 / 2678 |

**How to read this honestly.**

* ROC-AUC is identical before and after calibration (0.7812) because Platt scaling is a monotone
  transform — it changes the probability values, not the ranking. That is a correctness check,
  not a coincidence.
* Calibration **does** change which records cross the 0.5 line, because it rescales the
  probabilities. Uncalibrated, the model flags far more applicants at 0.5 (recall 0.4657 but
  431 false positives); calibrated, it becomes more conservative (recall 0.3346, 206 false
  positives). **The threshold, not the metric, is the operational decision** — see §16.
* Accuracy (0.8185) exceeds the majority-class baseline (0.7788) but is the least informative
  number here: the positive class is 22% of the data. Balanced accuracy (0.6453) and the confusion
  matrix are the honest summary.
* **Overfitting is present and reported, not hidden.** Train ROC-AUC 0.9925 versus test 0.7812
  is a 0.2113 gap, above the 0.05 threshold the application warns at. The random forest memorises
  the training split (min_samples_leaf = 3, unbounded depth). Validation (0.7674) and test
  (0.7812) agree closely, which indicates the *selection* did not overfit — the gap is
  memorisation, not leakage. A production system would regularise harder, which is listed under
  future enhancements.

---

## 15. Probability calibration methodology

* **Method:** Platt scaling (sigmoid), `p_calibrated = sigmoid(a * logit(p_base) + b)`.
* **Fitted on:** the validation split (6,000 rows). The base model never trains on it.
* **Fitted parameters:** slope `a = 0.8780`, intercept `b = -0.5049`.
* **Why this shape:** the validation split is held out from the base model, so calibration is
  fitted on data the model did not memorise. Fitting on training scores would produce optimistic,
  over-confident calibration.
* **Measured effect on the test split:**

  | Quantity | Uncalibrated | Calibrated |
  | --- | --- | --- |
  | Brier score | 0.1400 | **0.1346** |
  | Expected calibration error (10 quantile bins) | 0.0661 | **0.0171** |

* **Interpretation limits.** A calibrated probability is still an estimate for records resembling
  this dataset and period. A 40% predicted probability does **not** mean that 40% of comparable
  applicants will default. Calibration is validated on one split of one historical sample and is
  not guaranteed to transfer to another population, product or year.

A reliability curve (observed default rate versus mean predicted probability, before and after
calibration, against the diagonal) is rendered on the Model Intelligence page, built from the
stored curve points.

---

## 16. Risk-scoring methodology

Defined exactly once, in `src/risk_scoring.py`, and imported by every page, the report and the
tests, so a profile can never be classified two different ways:

```
risk_score = 100 x predicted_default_probability      (rounded half-up to 1 decimal)
0    - 30    -> LOW RISK
> 30 - 60    -> MEDIUM RISK
> 60 - 100    -> HIGH RISK
```

* The underlying probability is always displayed alongside the score; the score is a linear
  presentation of it, not a separate model.
* Because the displayed probability **is** calibrated (§15), it is presented as a predicted
  default probability. Had calibration been unavailable, the same value would have to be labelled
  a model-derived risk index instead — the `score_type` field carries that distinction.
* The bands are **prototype presentation thresholds** chosen for this project. They are not
  official banking thresholds, no regulator defines them, and the application says so on the
  About page and in every PDF.
* Boundary values (0, 30, 30.1, 60, 60.1, 100) and the half-up rounding convention are covered by
  tests in `tests/test_risk_scoring.py`.

---

## 17. SHAP explanation methodology

**Active method (recorded in the artifacts): `shap_tree` — SHAP `TreeExplainer`, exact tree-path
contributions.**

* Explanations are computed on the **same fitted estimator** that serves predictions. A proxy
  model is never explained and presented as an explanation of the serving model.
* Tree ensembles use `shap.TreeExplainer`; logistic regression would use `shap.LinearExplainer`
  with an independent masker over a background sample of 200 training rows.
* Contributions are expressed in the base model's **raw output units (log-odds)**.
* The displayed probability is a strictly monotone (Platt) calibration of exactly that output,
  so the direction and ordering of contributions are preserved. The application states this
  relationship rather than assuming it.
* One-hot `education` columns are **summed back onto the single source feature**, so every
  contribution corresponds to a field the user actually supplied.
* Local explanations state the base value (background expectation) and the model output for the
  row; the contributions plus the base value reproduce the output (asserted in the test suite).
* **Fallback, always labelled:** if SHAP is unavailable or incompatible, the application degrades
  to (a) a local occlusion heuristic — each feature is replaced by its background value and the
  change in model output is reported — and (b) globally, permutation importance measured as the
  ROC-AUC drop. Neither is ever presented as SHAP, and the occlusion fallback is explicitly
  non-additive.
* **Caveat shown on every explanation:** SHAP values describe how a feature contributed to this
  model's output for this input under the chosen method. They are not causal effects and do not
  prove that changing a feature would change the real-world outcome.

**Global ranking (mean absolute contribution, 600 records from the test split):**

| Feature | Mean \|SHAP\| | Share |
| --- | --- | --- |
| Repayment status, most recent month (Sept) | 0.0481 | 11.7% |
| Worst repayment status in 6 months | 0.0466 | 11.4% |
| Number of months with a delayed payment | 0.0411 | 10.0% |
| Credit utilisation (latest balance / limit) | 0.0189 | 4.6% |
| Repayment status, 2 months prior (Aug) | 0.0177 | 4.3% |
| Credit limit granted | 0.0172 | 4.2% |
| Average payment over 6 months | 0.0154 | 3.8% |
| Repayment ratio | 0.0149 | 3.6% |

Repayment punctuality dominates, which is consistent with the published literature on this
dataset. Permutation importance (ROC-AUC drop on the test split) agrees on the top feature
(recent repayment status, 0.0829) and then ranks `credit_limit` (0.0243) higher than SHAP does —
the two methods answer subtly different questions, which is exactly why both are shown.

---

## 18. What-if analysis

* Baseline profiles come from an explicitly labelled *illustrative* profile (dataset medians and
  modes — a statistical summary, **not** a real applicant record) or from a stored assessment.
* All 21 raw inputs are adjustable, with the four highest-signal fields surfaced first and the
  rest behind an "advanced" expander. Ranges come from the documented supported bounds.
* Derived ratios (`credit_utilization`, `repayment_ratio`) are **not** editable: they are
  computed from the raw inputs, so allowing them to be set independently would create
  internally inconsistent profiles.
* Pressing *Compare scenarios* re-validates the inputs and re-scores both profiles through the
  same saved pipeline. A stale-comparison warning appears if a slider moves afterwards.
* The original record is never mutated. Saving the alternative creates a **separate** record.
* Feature attributions for the alternative are computed **only on explicit request**, so dragging
  a slider does not trigger an expensive SHAP run.
* Comparisons are downloadable as CSV (and can be embedded in the PDF).
* The UI wording and the code both state that the comparison is a model statement, not a causal
  claim.

An optional amortising-loan repayment calculator (standard annuity formula with a correct
zero-interest special case) is provided for scenario exploration. **The dataset contains no
interest rate or loan term**, so any figure from that calculator is a user-supplied hypothetical
and is labelled as such.

---

## 19. Recommendation rules

| Rule | Condition | Outcome |
| --- | --- | --- |
| R0 | probability > 0.70 | ELEVATED RISK — ADDITIONAL ASSESSMENT REQUIRED |
| R1 | 0.50 < probability <= 0.70 | MANUAL UNDERWRITING REVIEW |
| R2 | 0.30 < probability <= 0.50 | CONDITIONAL REVIEW |
| R3 | probability <= 0.30 | LOWER RISK — ELIGIBLE FOR FURTHER REVIEW |
| M1 | >= 1 month with repayment status >= 3, or >= 4 delayed months | escalate to at least MANUAL |
| M2 | >= 3 delayed months, or repayment ratio < 0.05 | escalate to at least CONDITIONAL |
| — | credit utilisation > 0.90 | recorded as a concern and a suggested review point |

Modifiers escalate but never de-escalate: a strong probability signal cannot be argued down by a
favourable secondary rule. Every output contains the label, main reasons, supporting indicators,
key concerns, suggested next steps, the rules applied, and a human-review disclaimer.

**These are prototype heuristics, not bank policy.** They do not approve, decline or price a
loan, no protected attribute participates in any rule, and the logic is independently testable
(`tests/test_recommendation.py` covers every boundary and escalation path).

---

## 20. Database design

SQLite, created automatically at `database/creditsense.db` (configurable via
`CREDITSENSE_DB_PATH`), WAL journal mode.

```sql
CREATE TABLE assessments (
    assessment_id            TEXT PRIMARY KEY,   -- CSA-<12 hex>, stable
    created_at               TEXT NOT NULL,      -- ISO-8601 UTC
    application_ref          TEXT,               -- optional, user-supplied
    model_name               TEXT NOT NULL,
    model_version            TEXT NOT NULL,
    dataset_name             TEXT,
    dataset_version          TEXT,
    dataset_is_synthetic     INTEGER NOT NULL DEFAULT 0,
    inputs_json              TEXT NOT NULL,      -- canonical raw inputs
    derived_json             TEXT,               -- derived indicators
    probability              REAL NOT NULL,      -- calibrated
    probability_uncalibrated REAL,
    risk_score               REAL NOT NULL,
    risk_category            TEXT NOT NULL,
    recommendation_key       TEXT,
    recommendation_label     TEXT,
    top_risk_factors_json    TEXT,
    protective_factors_json  TEXT,
    explanation_method       TEXT,
    narrative_source         TEXT,
    report_available         INTEGER NOT NULL DEFAULT 0,
    is_demo                  INTEGER NOT NULL DEFAULT 0
);
```

Security and privacy decisions:

* Every statement uses **bound parameters**; sorting columns come from a fixed allow-list, so a
  hostile sort key or search string cannot reach SQL (tested).
* Raw SQL errors are converted to a friendly message; schema detail is never shown to a user.
* **Not stored:** API keys, tokens, passwords, and any direct personal identifier. The schema has
  no name, address, e-mail or national-identifier field. The optional application reference is
  documented in the UI as a place **not** to put personal identifiers.
* Records are inserted **only** on an explicit user action, so page reruns cannot duplicate them.
  Each save is a distinct record with its own identifier.
* Demonstration records are never seeded automatically; they require an explicit button and are
  written with `is_demo = 1` and an application reference beginning `DEMO`, then flagged
  everywhere they appear.
* Deletion is permanent and requires an explicit confirmation checkbox.

---

## 21. Report generation

ReportLab builds the PDF locally — no network, no external service, no image files (charts are
drawn as vector shapes and a native horizontal bar chart).

Sections: branding header and footer with page numbers → title block with the assessment ID,
generation timestamp, model, probability, score, category and recommendation → a risk badge with
a band meter → dataset attribution (name, publisher, authors, DOI, licence, geography, period,
currency, file fingerprint, row count, required citation) → geographic disclosure → applicant
input summary and derived indicators → recommendation with reasons, indicators, concerns, next
steps and applied rules → explanation with a contribution chart, ranked table and plain-English
summary → optional narrative (labelled with its source) → optional what-if comparison → measured
model evaluation context → data limitations → fairness considerations → risk-band explanation →
academic disclaimer.

`tests/test_reports.py` verifies the output is a valid PDF, extracts its text with pypdf, and
asserts that the branding, assessment ID, Taiwan attribution, measured metrics, disclaimer and
"not a bank-issued credit report" statement are all present, including with a missing explanation
or missing metrics.

---

## 22. AI-generated narrative

Priority order — the core application works with none of the optional parts:

1. **Deterministic template** (default, always available, no network).
2. **Local Ollama** when `OLLAMA_ENABLED=1`.
3. **OpenAI-compatible endpoint** when `LLM_API_BASE`, `LLM_API_KEY` and `LLM_MODEL` are set.

The machine-learning model remains the sole source of every quantity. The language model receives
a fixed, structured fact pack and is asked only to phrase it. **Any number in its output that is
not present in the fact pack invalidates the response**, and the deterministic template is used
instead — this is implemented (`validate_narrative`) and tested. The interface always shows
whether the text came from the template or a language model. Secrets are never logged, echoed or
rendered.

---

## 23. Installation

```bash
git clone <this repository>
cd creditsense-ai

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Verified with Python 3.12.3. Requirements are pinned to the exact versions the project was
trained, evaluated and tested against (NumPy 2.5.4, pandas 3.1.0, scikit-learn 1.9.1,
SHAP 0.54.1, Streamlit 1.66.1, Plotly 7.1.0, ReportLab 5.1.0, pytest 9.1.1). If a wheel is
unavailable for your interpreter version, relax that single pin — only documented, stable APIs
are used.

Python 3.11 or newer is required (pandas 3 and scikit-learn 1.9 constraints).

---

## 24. Environment setup

No environment variables are required. All optional settings are documented in `.env.example`:

| Variable | Purpose |
| --- | --- |
| `CREDITSENSE_DB_PATH` | Move the SQLite file |
| `CREDITSENSE_LOG_LEVEL` | Logging verbosity |
| `OLLAMA_ENABLED`, `OLLAMA_HOST`, `OLLAMA_MODEL`, `OLLAMA_TIMEOUT_SECONDS` | Local Ollama narrative generation |
| `LLM_API_BASE`, `LLM_API_KEY`, `LLM_MODEL` | OpenAI-compatible narrative endpoint |

The `.env` file is git-ignored. Never commit a key, and never paste one into an issue or a
screenshot.

---

## 25. Training instructions

```bash
python -m src.train_model                 # full run (~3 minutes on 30,000 rows)
python -m src.train_model --rows 5000     # fast iteration
python -m src.train_model --seed 7        # different split/seeds
python -m src.train_model --data path/to/other.csv
python -m src.train_model --allow-synthetic   # documented fallback if data/ is empty
```

Artifacts written:

```
artifacts/model/model_bundle.joblib        fitted pipeline + calibrator + schema fingerprint (compressed)
artifacts/metrics/evaluation_metrics.json   all candidate and selected metrics, curves, calibration quality
artifacts/metrics/permutation_importance.json
artifacts/metadata/training_metadata.json   splits, seed, versions, provenance, disclosures
artifacts/metadata/feature_schema.json      feature lists, labels, ranges, derived formulas
artifacts/metadata/global_shap_importance.json
artifacts/metadata/dataset_profile.json
```

The application detects missing or incompatible artifacts and displays the exact retrain command
instead of guessing. Changing the feature schema invalidates older bundles deliberately: the
predictor refuses to load a bundle whose `feature_names` differ from
`config.USER_INPUT_FEATURES`.

---

## 26. Running instructions

```bash
streamlit run app.py
```

Default address `http://localhost:8501` (this workspace's dev server is configured for port
3000 via `.streamlit/config.toml`). Nine pages: Overview, Risk Assessment, What-if Simulator,
Explainable AI, Model Intelligence, Dataset Explorer, Assessment History, Reports, About &
Methodology. Navigation is defined with `st.navigation` in `app.py`, so order and labels are
explicit rather than derived from the filesystem.

---

## 27. Testing instructions

```bash
python -m pytest -q
python -m pytest -q tests/test_risk_scoring.py -k boundary
python -m pytest -q -m "not slow"
```

The suite uses a small synthetic fixture dataset so it runs quickly and deterministically; tests
that need the real file use the `real_dataset` fixture, which **skips** cleanly when `data/` is
empty. Coverage:

| File | Focus |
| --- | --- |
| `test_data.py` | Loading, target detection (including refusing an approval-style target), column mapping, missing-value handling, schema validation, synthetic reproducibility, real-file adaptation |
| `test_preprocessing.py` | Exact derived-feature formulas, clipping, feature order, dense encoding for histogram boosting, Platt monotonicity |
| `test_prediction.py` | Bundle loading, probability bounds, score/probability consistency, feature-order invariance, determinism, invalid input rejection, missing/mismatched artifact handling |
| `test_risk_scoring.py` | Score = 100 x probability, rounding, every band boundary, category transitions |
| `test_explainability.py` | Explanation covers every input, additivity, ranking order, caveat present, labelled fallback when SHAP is removed |
| `test_what_if.py` | Baseline preservation, recalculation, delta arithmetic, changed-feature detection, repayment schedule (including zero interest) |
| `test_recommendation.py` | Every probability boundary, both escalation rules, no de-escalation, determinism, no protected attributes |
| `test_database.py` | Schema creation, insert/read, filters, sorting allow-list, SQL-injection-safe search, confirmation-gated deletion, statistics, demonstration flags |
| `test_train_model.py` | Split sizes, stratification, disjointness, reproducibility, validation-only selection, candidate families, artifact writing and reload |
| `test_reports.py` | Valid PDF, extracted text contents, missing explanation/metrics tolerance, what-if inclusion, filename sanitisation, narrative number validation |
| `test_history_page.py` | Page-level regression: the Assessment History page renders a record that has stored risk factors (it previously iterated a raw JSON string and raised `AttributeError`) |

Measured result on the delivery environment (Python 3.12.3, 30,000-row dataset present): **155 tests collected and passing, 0 failures** (pytest exit status 0). The suite runs in about twenty seconds; re-run `python -m pytest -q` to reproduce it.

---

## 28. Project structure

```
creditsense-ai/
├── app.py                     entry point; st.navigation defines the nine pages
├── requirements.txt           pinned, verified dependency set
├── README.md                  this document
├── AGENTS.md                  working notes and invariants for automated contributors
├── pytest.ini                 test configuration (pythonpath, markers)
├── .env.example               optional configuration template
├── .gitignore                 excludes the database, model artifacts, secrets and caches
├── .streamlit/config.toml      server binding, light theme
├── data/
│   ├── UCI_Credit_Card.csv     the research dataset (CC BY 4.0)
│   ├── README.md               provenance, licence, schema, quirks, geographic scope
│   ├── raw/                    staging for additional source files
│   └── processed/              staging for derived exports
├── artifacts/
│   ├── model/                  model_bundle.joblib (generated)
│   ├── metrics/                evaluation_metrics.json, permutation_importance.json (generated)
│   └── metadata/               training_metadata.json, feature_schema.json, global_shap_importance.json, dataset_profile.json
├── database/                   creditsense.db (created on first use)
├── src/
│   ├── config.py               paths, palette, features, thresholds, disclosures
│   ├── data_loader.py          discovery, target detection, canonical mapping, synthetic fallback
│   ├── data_validation.py      dataset and input validation with severities
│   ├── preprocessing.py        DerivedFeatureAdder, ColumnTransformer, PlattCalibrator
│   ├── train_model.py          split, compare, select, calibrate, evaluate, persist
│   ├── predictor.py            bundle loading, scoring, readiness
│   ├── risk_scoring.py         the single probability -> score -> category mapping
│   ├── explainability.py       SHAP explainers, fallbacks, narration
│   ├── recommendation.py       documented rule engine
│   ├── what_if.py              scenario comparison, illustrative profile, repayment schedule
│   ├── fairness.py             group-level descriptive metrics
│   ├── database.py             SQLite persistence
│   ├── report_generator.py     ReportLab PDF
│   ├── ai_report.py            deterministic narrative + validated optional LLM
│   ├── ui.py                   shared Streamlit components and chart theme
│   └── utils.py                formatting, hashing, JSON, numeric helpers
├── pages/
│   ├── dashboard.py            Overview
│   ├── risk_assessment.py      Risk Assessment
│   ├── what_if_analysis.py     What-if Simulator
│   ├── explainable_ai.py       Explainable AI
│   ├── model_insights.py       Model Intelligence
│   ├── dataset_explorer.py     Dataset Explorer
│   ├── assessment_history.py   Assessment History
│   ├── reports.py              Reports
│   └── about.py                About & Methodology
├── assets/
│   ├── styles.css              design system
│   └── logo.svg                brand mark
└── tests/                      10 test modules + conftest fixtures
```

---

## 29. Screenshots section

The application is a running Streamlit app; the screenshots below are the ones worth capturing
for a demonstration or a submission. Each is a real screen produced by real computations — none
is a mock-up.

| # | Where | What to capture | Why it matters |
| --- | --- | --- | --- |
| 1 | Overview | Header, readiness chips, KPI row, risk-category donut | Shows the product identity and that empty/populated states are honest |
| 2 | Risk Assessment | The five-section form with field help text | Shows only dataset-supported fields are requested |
| 3 | Risk Assessment | Result panel: probability, score, category badge, band meter, recommendation | The core workflow output |
| 4 | Risk Assessment | Feature-contribution bar chart with the plain-English summary | Explainability in the workflow |
| 5 | What-if Simulator | Baseline vs alternative vs difference cards plus the before/after chart | Shows real recalculation, not a story |
| 6 | Explainable AI | Global SHAP ranking and local contribution chart side by side | The transparency story |
| 7 | Model Intelligence | KPI row, ROC, PR and reliability curves, confusion matrix, overfitting warning | The evaluation contract, including the overfitting warning |
| 8 | Dataset Explorer | Attribution card with licence and DOI, plus the geographic-scope notice | Provenance and disclosure |
| 9 | Dataset Explorer → Fairness | Group metrics table with sample sizes and the caveats above it | Responsible presentation of group behaviour |
| 10 | Assessment History | Filtered table, detail view, delete confirmation | Persistence with a confirmation gate |
| 11 | Reports | Generated narrative with its source label and the PDF download button | Narrative provenance |
| 12 | The generated PDF | First page and the limitations/disclaimer page | The professional deliverable |

Suggested captions should state the dataset (30,000 Taiwan credit-card clients, 2005) and, where
a prediction appears, the model version from the artifacts.

---

## 30. Known limitations

1. **Geographic and temporal scope.** Taiwan, 2005. Not validated for India or anywhere else, and
   not validated for any current portfolio.
2. **Missing factors.** No income, employment, loan purpose, interest rate, loan term or external
   credit score exists in the dataset, so those standard credit-risk factors are absent from the
   model entirely. This is the single largest limitation.
3. **Class imbalance.** 22.12% positive rate, 3.52:1. Precision and recall trade off sharply.
4. **Overfitting.** Train ROC-AUC 0.9925 versus test 0.7812 (gap 0.2113). Validation and test
   agree, so selection is sound, but the model memorises its training split.
5. **Low recall at the default threshold.** 0.3346 of defaults are caught at 0.5. The threshold is
   a business decision that this prototype does not make for you; the trade-off table on the
   Dataset Explorer page shows the alternatives.
6. **Calibration transfer.** Platt scaling is fitted on one validation split of one historical
   sample; it is not guaranteed to transfer to another population, product or year.
7. **No monitoring.** The model is a static artifact: no drift detection, no retraining schedule,
   no performance monitoring.
8. **Residual bias.** Excluding `sex` and `marriage` does not remove bias. Age, education, credit
   limit and repayment behaviour can act as proxies, and the dataset carries historical selection
   bias about who was granted credit.
9. **Explanation limits.** SHAP explains model behaviour, not causality. Contributions are in
   log-odds and are not additive with the calibrated probability.
10. **Prototype heuristics.** Risk bands and recommendation rules are presentation choices, not
    policy, and are not jurisdiction-aware.
11. **No compliance claim.** No regulatory, banking or fairness-law compliance is claimed or
    implied. No fairness audit has been performed.
12. **Single-user, local, unauthenticated.** The application has no authentication, no multi-user
    isolation and no audit trail. Assessment records live in a local SQLite file.
13. **Synthetic fallback.** If no dataset is present and `--allow-synthetic` is used, performance
    figures describe a generated dataset and demonstrate software correctness only, never
    real-world lending accuracy.
14. **Unverified UI interactions.** Page loading and navigation are verified programmatically;
    exhaustive interactive testing of every widget in a browser was not performed in the build
    environment (see §33).

---

## 31. Responsible-use statement

> **CreditSense AI is an academic research and decision-support prototype. Its predictions are
> estimates produced from the available dataset and may be inaccurate or biased. It is not a
> credit bureau, lender, or substitute for a regulated underwriting process. It must not be used
> as the sole basis for an actual lending decision.**

Additional commitments made by this implementation:

* No protected attribute is used as a model input, and no protected attribute participates in any
  recommendation rule.
* No sensitive demographic information is collected from users. Group labels exist only because
  the public research dataset contains them.
* Fields the dataset lacks are never invented to make the interface look richer.
* Every metric displayed is read from a measured artifact; demonstration records are labelled;
  fallback methods are labelled.
* The geographic scope and dataset limitations are disclosed wherever a prediction appears,
  including inside generated PDFs.

Before any real-world use: obtain legal and regulatory review for the jurisdiction, validate on
locally representative and lawfully obtained data, audit for disparate impact, add monitoring and
human oversight, and provide a route for affected individuals to challenge an automated estimate.

---

## 32. Future enhancements

1. **Reduce overfitting** with deeper regularisation (`min_samples_leaf`, `max_depth`,
   `max_features`), or use boosting with early stopping on the validation split.
2. **Threshold optimisation** against an explicit cost matrix (cost of a false negative versus a
   false positive) instead of a fixed 0.5, and expose the chosen operating point in the UI.
3. **Model monitoring**: drift detection for input distributions, calibration decay tracking, and
   scheduled retraining with a champion/challenger comparison.
4. **Richer data**: a dataset containing income, employment, loan purpose and external credit
   history, which would allow a genuinely richer canonical schema.
5. **Fairness hardening**: proper group-fairness metrics with confidence intervals, plus an
   analysis of proxy leakage from included features.
6. **Counterfactual explanations** ("the smallest change that would move this profile to the
   lower band"), clearly separated from the correlational SHAP values.
7. **Authentication and multi-tenancy** with per-analyst data isolation and an audit trail.
8. **API layer** (FastAPI) so the scoring pipeline can be consumed by another system, with the
   same validation and explanation semantics.
9. **i18n and currency-agnostic formatting** so the interface is not tied to NT$ presentation.
10. **Interactive test automation** (Playwright) covering navigation, form validation and download
    flows end to end.

---

## 33. Deployment notes

Local execution is the supported, verified path. For a hosted deployment, note that:

* The shipped `data/UCI_Credit_Card.csv` is 2.8 MB and is redistributed under CC BY 4.0 **with
  attribution**, which must be preserved.
* `database/` and `artifacts/` are git-ignored. A hosted deployment needs a persistent volume for
  the database, or a replacement persistence layer, and artifacts must be produced during build
  (`python -m src.train_model`) because a 30,000-row training run takes about three minutes and
  produces a large model file.
* The application has **no authentication**. Do not expose it publicly with a writable database
  without adding auth and per-user isolation first.
* No deployment was performed as part of this build, and none is claimed.

---

## 34. Measured versus illustrative

| Category | Status |
| --- | --- |
| Model metrics (§13, §14) | **Measured** on held-out data; stored in `artifacts/metrics/evaluation_metrics.json` |
| Calibration figures (§15) | **Measured** on the test split |
| SHAP importance (§17) | **Measured** over 600 test-split records |
| Dataset statistics (§10, §11) | **Measured** from the shipped file |
| Risk bands and recommendation thresholds (§16, §19) | **Chosen design**, explicitly labelled as prototype presentation thresholds |
| Illustrative profile in forms and the simulator | **Dataset medians/modes**, labelled as a statistical summary, not a real applicant |
| Demonstration records | **Generated from dataset rows**, flagged `is_demo` with a `DEMO` reference |
| Synthetic fallback data (only with `--allow-synthetic`) | **Generated**, labelled synthetic everywhere |

Nothing in this project fabricates a metric, a customer record, or a model output.
