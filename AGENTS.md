# AGENTS.md - working notes for automated contributors

CreditSense AI is an academic credit-risk decision-support prototype. If you are
an agent working in this repository, read this file first.

## The three rules that matter most

1. **Never fabricate a number.** No synthetic metrics, no placeholder accuracy, no
   invented applicant records presented as real, no illustrative value rendered as
   if it were measured. If a value cannot be computed, the interface must say so.
2. **Never present the model as more than it is.** It is trained on 2005 Taiwanese
   credit-card data. It is not validated for any other population, it is not a
   lender, and it does not approve or decline credit.
3. **Keep the disclosure text.** `config.GEOGRAPHIC_DISCLOSURE`,
   `config.SYNTHETIC_DISCLOSURE` and `config.APP_DISCLAIMER` must keep appearing
   wherever they currently appear, including in generated PDFs.

## Commands

```bash
# install (a project-local virtualenv is expected)
python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

# train: writes artifacts/model, artifacts/metrics, artifacts/metadata
python -m src.train_model
python -m src.train_model --rows 5000      # fast iteration
python -m src.train_model --allow-synthetic  # documented fallback if data/ is empty

# run
streamlit run app.py

# test
python -m pytest -q
python -m pytest -q tests/test_risk_scoring.py -k boundary
```

## Layout

```
app.py            entry point; st.navigation defines the nine pages in order
pages/            one module per navigation entry (dashboard, risk_assessment, ...)
src/config.py     paths, palette, feature lists, thresholds, disclosures - start here
src/data_loader.py       discovery, target detection, canonical mapping, synthetic fallback
src/data_validation.py   dataset + user-input validation with severities
src/preprocessing.py     derived features (DerivedFeatureAdder), ColumnTransformer, PlattCalibrator
src/train_model.py       split, compare, select, calibrate, evaluate, persist
src/predictor.py         bundle loading, scoring, readiness reporting
src/explainability.py    SHAP explainers + documented fallbacks + narration
src/recommendation.py    rule-based review recommendation
src/what_if.py           scenario comparison, illustrative profile, repayment schedule
src/fairness.py          group-level descriptive metrics
src/database.py          SQLite persistence (parameterised queries only)
src/report_generator.py  ReportLab PDF
src/ai_report.py         deterministic narrative + optional LLM with number validation
src/ui.py                shared Streamlit components and chart theme
assets/styles.css        design system (mostly .csa-* classes we own)
tests/                   pytest suite; synthetic fixtures for speed
```

## Invariants you must not break

* **Split discipline.** Candidates are fitted on train only and compared on
  validation only. Calibration is fitted on validation only. The test split is
  used once, for reporting. `metrics.test_split_used_for_tuning` must stay `false`.
* **Preprocessing lives inside the pipeline.** Derived features are computed by
  `DerivedFeatureAdder`, so training and inference cannot diverge. Never compute a
  derived feature in a page or in the predictor by hand.
* **Feature order** comes from `config.USER_INPUT_FEATURES`. The saved bundle
  records it, and the predictor refuses to load a bundle whose order differs.
* **Protected attributes stay out of the model.** `sex` and `marriage` are
  reporting-only grouping keys. `education` and `age` are included but must remain
  disclosed as potential proxy variables.
* **Risk mapping exists once**, in `src/risk_scoring.py`. Do not re-implement band
  thresholds on a page.
* **Explanations come from the serving estimator.** Never explain a proxy model and
  label it as an explanation of the active prediction.
* **Fallbacks are labelled.** If SHAP fails, the UI must say "fallback (not SHAP)".
* **Database writes are explicit.** An assessment is inserted only on a user
  action. Never seed records automatically; demonstration records must be marked
  `is_demo = 1` with an application reference beginning with `DEMO`.
* **SQL is parameterised.** Sorting columns come from `SORTABLE_COLUMNS`; never
  interpolate a user value into a statement.
* **No secrets in the repository or in logs.** Optional integrations read
  environment variables only.

## Conventions

* Python 3.11+, type hints, `from __future__ import annotations`.
* Keep lines under ~100 characters.
* UI text: concise, professional, no emoji, no hype. Say what was measured and
  what was assumed.
* Every module that shows a number should be able to say where the number came
  from: the training artifact, the database, or the loaded dataset.
* Tests use synthetic fixtures (`tests/conftest.py`) so they run in seconds; tests
  that need the real file use the `real_dataset` fixture, which skips when the data
  is absent.

## Before you finish

1. `python -m pytest -q` passes.
2. If you changed features or preprocessing, re-run training (the bundle carries a
   feature-order check that will otherwise refuse to load).
3. If you changed a number that appears in README.md, update the README.
4. Confirm the disclosures still render on Overview, Model Intelligence and in a
   generated PDF.
