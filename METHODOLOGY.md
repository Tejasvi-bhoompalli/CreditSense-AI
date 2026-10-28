# CreditSense AI — Methodology and Technical Report

**Project title:** CreditSense AI: An Explainable AI-Driven Credit Risk Assessment Framework for
Intelligent Loan Decision Support

**Domain:** Artificial Intelligence, Machine Learning, Explainable AI, Financial Technology,
Credit Risk Assessment

**Report status:** every number in this document is read from the training artifacts produced by
`python -m src.train_model` (`artifacts/metrics/evaluation_metrics.json`,
`artifacts/metadata/training_metadata.json`) or from a directly executed command. No value here is
estimated, illustrative or copied from a third party.

---

## 1. Problem statement

Consumer credit default risk is normally assessed by institutions that do not explain their
decisions to applicants or reviewers. A loan officer can see *that* an application was declined,
but not *which measured factors* drove the estimate, nor how sensitive it is to a change in one
input.

The problem addressed here is therefore two-part:

1. **Estimation** — predict the probability that a credit-card client will default on the next
   payment, using only information available at assessment time.
2. **Justification** — produce a per-application, feature-level explanation of that estimate, plus
   a transparent, testable recommendation, so that a human reviewer can interrogate the output
   rather than trust it blindly.

The system is explicitly a **decision-support prototype**. It does not approve, decline or price
credit, and it is not a lender, a credit bureau, or a regulated underwriting process.

## 2. Objectives

1. Automatically inspect a supplied dataset, detect the target variable, map source columns onto a
   canonical feature schema, and validate data quality.
2. Build a reproducible binary-classification pipeline with leakage-safe preprocessing fitted only
   on the training split.
3. Compare several model families on a held-out validation split and select one by a declared
   criterion.
4. Calibrate predicted probabilities using a validation-only fitting procedure.
5. Explain both global and individual predictions with SHAP, computed on the serving estimator.
6. Deliver an interactive application: assessment, what-if simulation, explainability, model
   intelligence, dataset exploration, history, PDF reports.
7. Never fabricate a metric, a record or a prediction; label every fallback and limitation.

## 3. System architecture

The application is a Streamlit multipage application with a modular Python backend. Navigation is
declared explicitly in `app.py` via `st.navigation`, so page order and labels are fixed rather than
derived from the filesystem.

```
app.py                      entry point, navigation, CSS, footer
pages/                      one module per navigation entry (9 pages)
src/config.py               paths, palette, feature lists, thresholds, disclosures
src/data_loader.py          discovery, target detection, canonical mapping, synthetic fallback
src/data_validation.py      dataset and user-input validation with severities
src/preprocessing.py        DerivedFeatureAdder, ColumnTransformer, PlattCalibrator
src/train_model.py          split, compare, select, calibrate, evaluate, persist
src/predictor.py            bundle loading, scoring, readiness reporting
src/explainability.py       SHAP explainers, documented fallbacks, narration
src/recommendation.py       rule-based review recommendation
src/what_if.py              scenario comparison, illustrative profile, repayment schedule
src/fairness.py             group-level descriptive metrics
src/database.py             SQLite persistence, parameterised queries only
src/report_generator.py     ReportLab PDF
src/ai_report.py            deterministic narrative + optional LLM with number validation
src/ui.py                   shared components and chart theme
assets/styles.css           design system
tests/                      pytest suite (synthetic fixtures)
```

**Layering rule.** Every page calls into `src/`; no page re-implements a threshold, a derived
formula or a band mapping. `risk_scoring.py` defines the probability → score → category mapping
exactly once and is imported everywhere.

## 4. Dataset

| Property | Value |
|---|---|
| Name | Default of Credit Card Clients |
| Publisher | UCI Machine Learning Repository |
| Authors | Yeh & Lien |
| Year | 2009 (data collected 2005) |
| DOI | 10.24432/C55S3H |
| Licence | CC BY 4.0 |
| File | `data/UCI_Credit_Card.csv` |
| SHA-256 | `a0f0ab49d6326671d6cd83be5c88dcf18007025fe9a53ecd699119c871176ca1` |
| Dimensions | 30,000 rows × 24 columns |
| Target | `default.payment.next.month` → canonical `default` |
| Class balance | 6,636 defaults (22.12%) vs 23,364 non-defaults; ratio 3.52 : 1 |
| Missing values | 0 |
| Duplicate feature rows | 56 (retained; distinct clients can share a profile) |
| Geography | Taiwan, April–September 2005 (default observed October 2005) |
| Currency | New Taiwan Dollars (NT$) |

**Geographic disclosure (mandatory).** This is *not* an India-specific lending dataset. It contains
no Indian borrower records, and model outputs are not validated for Indian borrowers, Indian
income levels, Indian credit products, or Indian regulatory definitions of default.

**Identifier handling.** The `ID` column is dropped before modelling: it is a row identifier with
no predictive meaning and would leak record identity.

**Unusual values preserved.** 1,930 client-months carry a negative statement balance. In the source
data a negative balance means credit in the client's favour (overpayment or refund), so these
values are kept as-is rather than clipped to zero.

## 5. Preprocessing and feature engineering

### 5.1 Canonical schema and adaptation

The loader inspects the supplied file, detects the target by documented heuristics (and refuses an
approval-style target as a proxy for default), then maps verified source columns onto the
canonical schema. Fields that the source genuinely lacks — annual income, employment status, loan
purpose, interest rate, external credit score — are **not invented**. The assessment form only
presents fields the active model consumes, and the UI marks the remainder as unavailable.

### 5.2 Derived features

Nine features are computed by `DerivedFeatureAdder`, which is a **step inside the saved
scikit-learn pipeline**. Training and inference therefore cannot diverge: a derived value can never
be recomputed by hand on a page.

| Derived feature | Definition |
|---|---|
| `avg_bill_amount` | mean of the six monthly statement balances |
| `avg_payment_amount` | mean of the six monthly payment amounts |
| `credit_utilization` | latest statement balance ÷ credit limit |
| `repayment_ratio` | total paid ÷ total billed over six months |
| `max_repayment_delay` | worst repayment status code over six months |
| `num_delayed_months` | count of months with a delay of ≥ 1 level |
| `num_severely_delayed_months` | count of months with a delay of ≥ 3 levels |
| `bill_trend` | latest balance − balance six months prior |
| `payment_trend` | latest payment − payment six months prior |

**Feature count:** 21 raw user inputs + 9 derived = **30 features** fed to the estimator.

### 5.3 Pipeline

`ColumnTransformer` handles numeric imputation plus scaling (where the estimator benefits from it)
and one-hot encoding of the categorical feature (`education`). The entire transformer graph lives
inside the fitted pipeline, so it is fitted on the training split only.

### 5.4 Protected attributes

`sex` and `marriage` are **excluded from the model** entirely; they are used only as reporting-only
grouping keys in the fairness view. `age` and `education` are retained because they carry signal in
this dataset, and they are disclosed in the UI as **potential proxy variables**.

## 6. Experimental design

### 6.1 Splits

| Split | Rows | Purpose |
|---|---|---|
| Train | 18,000 | fit preprocessing and candidate models |
| Validation | 6,000 | model selection and calibration fitting only |
| Test | 6,000 | evaluated **once**, for reporting |

Splitting is stratified on the target with a fixed random seed. The artifact records
`test_split_used_for_tuning = false` and `test_split_metrics_are_measured = true`; both flags are
asserted by the test suite.

### 6.2 Candidates compared

Five candidates were fitted on train and scored on validation only:

| Candidate | Validation ROC-AUC |
|---|---|
| Logistic Regression | 0.7374 |
| Logistic Regression (class-weighted) | 0.7428 |
| Random Forest | 0.7648 |
| **Random Forest (class-weighted)** | **0.7674** ← selected |
| HistGradientBoosting | 0.7668 |

**Selection criterion:** highest validation ROC-AUC, ties broken by validation PR-AUC then Brier
score. The criterion is declared in the artifact before results are read, so selection is not
post-hoc.

### 6.3 Class imbalance handling

The positive class is 22.12% of rows. Handling was made an explicit candidate rather than applied
silently: both class-weighted and unweighted variants were trained and compared on the validation
split, and the class-weighted Random Forest won on the declared criterion. **No oversampling
(SMOTE or otherwise) was used** — synthetic oversampling before splitting would leak across folds,
and the weighted variant achieved the same goal without it.

## 7. Results (measured)

All values below come from `artifacts/metrics/evaluation_metrics.json` on the untouched test split
(n = 6,000; 1,327 positives; 4,673 negatives).

### 7.1 Selected model — test metrics (calibrated)

| Metric | Value |
|---|---|
| ROC-AUC | **0.7812** |
| PR-AUC | **0.5633** |
| Accuracy | 0.8185 |
| Precision | 0.6831 |
| Recall | 0.3346 |
| F1 | 0.4492 |
| Brier score | 0.1346 |
| Confusion matrix | TN 4467 · FP 206 · FN 883 · TP 444 |

### 7.2 Same model, uncalibrated probabilities (test)

Accuracy 0.8100 · precision 0.5891 · recall 0.4657 · F1 0.5202 · Brier 0.13999 ·
confusion matrix [[4242, 431], [709, 618]] · balanced accuracy 0.6867 · specificity 0.9078.

Reporting both matters: calibration shifts the operating point at the 0.5 threshold. The
calibrated model is more precise but recalls fewer defaults; the uncalibrated model catches more
defaults at the cost of more false alarms. Neither is "better" in isolation — the choice is a
business decision about the relative cost of a missed default versus a wrongly flagged good
account.

### 7.3 Majority-class baseline (test)

Accuracy 0.7788 · balanced accuracy 0.5 · precision 0 · recall 0 · F1 0.

This is why accuracy alone is not used for selection: the trivial always-predict-"no default"
classifier already reaches 0.7788 accuracy while detecting nothing at all.

### 7.4 Train / validation / test comparison and overfitting

| Split | ROC-AUC |
|---|---|
| Train | 0.9925 |
| Validation | 0.7674 |
| Test | 0.7812 |

Generalisation gap (train − test) = **0.2113**, which exceeds the 0.05 threshold at which the
reporting layer raises an overfitting warning. The warning is displayed rather than suppressed:
a Random Forest memorises the training split substantially, and the honest reading is that the
0.7812 figure — not the 0.9925 figure — is the one that generalises. Validation − test = −0.0138,
i.e. the test result is marginally *better* than validation, confirming the test split was not
used to tune anything.

### 7.5 Probability calibration

Platt (sigmoid) scaling is fitted **on the validation split only**, then applied to test
probabilities.

| | Uncalibrated | Calibrated |
|---|---|---|
| Expected Calibration Error | 0.0661 | **0.0171** |
| Brier score | 0.1400 | **0.1346** |

ECE improved by roughly 3.9×. The interface therefore labels the displayed value a *calibrated
predicted default probability*, while still stating that a predicted probability is an estimate,
not a guarantee — a 30% estimate does not mean exactly 30% of comparable borrowers will default.

## 8. Explainability

**Method:** `shap.TreeExplainer` (exact tree-path contributions) on the same fitted estimator that
produces the prediction. Verified during end-to-end testing: method `shap_tree`, additivity
residual 5.55e-17 across the 30 features, no fallback notice attached.

- **Local.** Contributions are aggregated from one-hot columns back onto their source feature, so
  every displayed contribution corresponds to a field the user actually supplied. Values are in
  the model's raw output units (log-odds); because Platt calibration is strictly monotone, the
  ordering of contributions is preserved on the probability scale — this is stated in the UI
  rather than assumed.
- **Global.** Mean absolute SHAP contribution per source feature over a 400-record sample drawn
  from the held-out evaluation split. Stored in `artifacts/metadata/global_shap_importance.json`;
  the live recomputation reproduces the stored top-ranked feature exactly and 4 of the top 5.
- **Fallbacks.** If SHAP is unavailable the application degrades to a labelled local occlusion
  heuristic plus permutation importance globally, each explicitly marked "not SHAP". It never
  silently substitutes a different method and never explains a proxy model.
- **Caveat carried in every view:** SHAP values describe how a feature contributed to this model's
  output for this input; they are not causal effects and do not prove that changing a feature
  changes real-world default risk.

## 9. Application workflow

1. **Overview** — KPIs computed from stored assessments and the loaded dataset; empty state when
   the database is empty (no invented counts).
2. **Risk Assessment** — form restricted to features the active model supports; validated before
   scoring; result panel with probability, score, category, recommendation, explanation chart,
   protective/risk factors, derived indicators; SQLite write only on an explicit Save click.
3. **What-if Simulator** — baseline vs alternative scenario with probability delta, score delta and
   category transition; original assessment is never modified unless the user saves a new one.
4. **Explainable AI** — global ranking, local contributions, method and caveats tabs.
5. **Model Intelligence** — dataset, model identity, split sizes, comparison table, ROC/PR curves,
   confusion matrix, calibration chart, overfitting warning.
6. **Dataset Explorer** — profile, missing values, distributions, correlations, target balance,
   descriptive fairness view.
7. **Assessment History** — search, filter, paginated inspection, PDF regeneration, confirmed
   deletion, labelled demo records.
8. **Reports** — deterministic narrative (optional LLM with number validation) and a ReportLab PDF
   carrying dataset attribution, both disclosures and the academic disclaimer.
9. **About & Methodology** — full method description including every threshold used.

**Risk mapping** (`src/risk_scoring.py`, used by every page): `risk_score = 100 × probability`,
rounded half-up to 1 decimal; bands 0–30 LOW, >30–60 MEDIUM, >60–100 HIGH. These are prototype
presentation thresholds, not banking thresholds.

**Recommendation engine:** rule-based, deterministic, documented on the methodology page, returns
one of four labels with reasons, concerns, next steps and a human-review disclaimer. Protected
attributes are never inputs to any rule.

## 10. Testing

**Command:** `python -m pytest -q`
**Result on the delivery environment (Python 3.12.3): 155 tests collected, 155 passed, 0 failed,
0 errors, exit status 0, ~19 seconds.**

The suite uses small synthetic fixtures so it runs in seconds; tests needing the real file use a
`real_dataset` fixture that skips when `data/` is absent.

| File | Focus |
|---|---|
| `test_data.py` | loading, target detection (refusing an approval-style target), column mapping, missing values, schema validation, real-file adaptation |
| `test_preprocessing.py` | derived-feature formulas, clipping, feature order, Platt monotonicity |
| `test_prediction.py` | bundle loading, probability bounds, score/probability consistency, determinism, invalid-input rejection |
| `test_risk_scoring.py` | score = 100 × probability, rounding, every band boundary, category transitions |
| `test_explainability.py` | explanation covers every model feature, additivity, ranking order, caveat present, labelled fallback |
| `test_what_if.py` | baseline preservation, recalculation, delta arithmetic, repayment schedule incl. zero interest |
| `test_recommendation.py` | every probability boundary, escalation rules, determinism, no protected attributes |
| `test_database.py` | schema, insert/read, filters, sort allow-list, injection-safe search, confirmation-gated deletion, statistics, demo flags |
| `test_train_model.py` | split sizes, stratification, disjointness, reproducibility, validation-only selection, artifact writing/reload |
| `test_reports.py` | valid PDF, extracted text, missing explanation/metrics tolerance, filename sanitisation, narrative number validation |
| `test_history_page.py` | page-level regression: Assessment History renders stored risk factors (it previously iterated a raw JSON string and raised `AttributeError`) |

**End-to-end verification performed through Streamlit's `AppTest` harness (106 checks, 106 passed):**
all nine navigation pages render without exception; Overview KPIs reflect stored rows; form
submission produces a prediction; changing repayment status to the worst code raises the predicted
probability from 6.19% to 50.27%; an explicit save inserts exactly one record and a rerun does not
duplicate it; the Reports page generates a PDF whose extracted text contains the dataset name, DOI,
licence, geographic disclosure and application disclaimer; demonstration records are stored with
`is_demo = 1` and a `DEMO` prefix.

The only warning emitted is a scikit-learn `ConvergenceWarning` for the Logistic Regression
candidate hitting `max_iter = 500`. This is a candidate that was *not* selected; it is reported, not
suppressed.

## 11. Limitations

1. **Geography and time.** Taiwan, 2005. Not validated for any other population, regulation or
   credit product. Not India-specific; not validated for Indian borrowers.
2. **Task definition.** The target is next-month payment default on a credit card, **not** loan
   approval and not loan default. The two must not be equated.
3. **Overfitting.** Train ROC-AUC 0.9925 vs test 0.7812 (gap 0.2113) — the model memorises the
   training split; only the test figure should be quoted.
4. **Recall trade-off.** At the 0.5 threshold the calibrated model detects 444 of 1,327 test
   defaults (recall 0.3346). A production system would choose an operating threshold from the
   cost of misses versus false alarms; this prototype does not.
5. **Single dataset, single point in time.** No out-of-time validation, no stress-testing, no
   economic-cycle coverage.
6. **Probability interpretation.** Calibration was fitted on a 6,000-row validation split of this
   same dataset; the ECE of 0.0171 is an in-dataset result, not a field guarantee.
7. **Explanations are attributions, not causes.** SHAP describes the model, not the world.
8. **Fairness view is descriptive.** Group metrics are computed on dataset-provided labels with
   sample sizes and caveats. It is not a fairness audit and does not establish compliance with any
   regulation.
9. **Proxy variables.** `age` and `education` remain in the model and may act as proxies for
   protected characteristics; this is disclosed in the interface.
10. **Prototype scope.** No authentication, no multi-tenancy, no adversarial hardening, local
    SQLite only. Not production-ready and not to be used for real lending decisions.

## 12. Future enhancements

1. Out-of-time and cross-geography validation on a newer, multi-market dataset before any
   generalisation claim.
2. Threshold selection driven by an explicit cost matrix, with the trade-off surface exposed
   interactively.
3. Post-hoc fairness constraints (e.g. calibration within groups) alongside the existing
   descriptive view, with documented residual-proxy analysis.
4. Monotonic gradient boosting to encode domain-known relationships (higher utilisation ⇒ higher
   risk) and reduce overfitting.
5. Isotonic calibration and cross-validated calibration stability checks.
6. Drift monitoring for input distributions and calibration over time.
7. Authentication, audit logging and role-based access before any deployment beyond a local
   prototype.
8. Optional LLM narrative hardening: schema-constrained generation with stronger numeric
   verification, keeping the deterministic template as the default.

## 13. How to run

```bash
pip install -r requirements.txt   # pinned to verified versions
python -m src.train_model         # optional: trained artifacts are bundled
streamlit run app.py              # http://localhost:8501
python -m pytest -q               # 155 passed
```

Verified working environment: **Python 3.12.3**, with the pinned versions in `requirements.txt`
(scikit-learn 1.9.1, streamlit 1.66.1, shap 0.54.1, pandas 3.1.0, numpy 2.5.4, plotly 7.1.0,
reportlab 5.1.0, pytest 9.1.1, pypdf 6.21.2).

## 14. Attribution

Yeh, I.-C. & Lien, C.-H. (2009). *The predictions of credit card defaulting clients*. DOI
**10.24432/C55S3H**. UCI Machine Learning Repository. Licensed **CC BY 4.0**.

Required attribution is reproduced in `data/README.md`, in `README.md`, in the application UI and
inside every generated PDF report.