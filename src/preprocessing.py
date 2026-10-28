"""Feature engineering and the scikit-learn preprocessing pipeline.

Design decision: derived features are computed *inside* the saved pipeline via
:class:`DerivedFeatureAdder`. That guarantees training and inference apply
byte-for-byte the same transformation and removes the classic train/serve skew
bug. The user therefore only ever supplies the raw, source-observed fields.

Preprocessing is fitted on the training split only; imputation statistics never
see the validation or test data.
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config

# Documented clipping bounds for derived ratio features. They limit the
# influence of extreme outliers without discarding the record.
UTILIZATION_BOUNDS = (-2.0, 5.0)
REPAYMENT_RATIO_BOUNDS = (0.0, 2.0)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce").astype("float64")


def compute_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the documented derived features to a canonical (raw) frame.

    Formulas (all documented in README.md > Data dictionary):
      avg_bill_amount              = mean(bill_amount_1..6)
      avg_payment_amount           = mean(payment_amount_1..6)
      credit_utilization           = clip(bill_amount_1 / credit_limit, -2, 5)
      repayment_ratio              = clip(sum(payments) / max(sum(bills), 1), 0, 2)
      max_repayment_delay          = max(repay_status_1..6)
      num_delayed_months           = count(repay_status_i >= 1)
      num_severely_delayed_months  = count(repay_status_i >= 3)
      bill_trend                   = bill_amount_1 - bill_amount_6
      payment_trend                = payment_amount_1 - payment_amount_6
    """
    result = frame.copy()

    bills = result[config.BILL_AMOUNT_COLUMNS].apply(pd.to_numeric, errors="coerce")
    payments = result[config.PAYMENT_AMOUNT_COLUMNS].apply(pd.to_numeric, errors="coerce")
    statuses = result[config.REPAY_STATUS_COLUMNS].apply(pd.to_numeric, errors="coerce")
    credit_limit = _numeric(result, "credit_limit")

    result["avg_bill_amount"] = bills.mean(axis=1)
    result["avg_payment_amount"] = payments.mean(axis=1)

    utilization = bills.iloc[:, 0] / credit_limit.replace(0.0, np.nan)
    result["credit_utilization"] = utilization.clip(*UTILIZATION_BOUNDS)

    total_bills = bills.clip(lower=0.0).sum(axis=1)
    total_payments = payments.clip(lower=0.0).sum(axis=1)
    ratio = total_payments / total_bills.where(total_bills > 0, 1.0)
    result["repayment_ratio"] = ratio.clip(*REPAYMENT_RATIO_BOUNDS)

    result["max_repayment_delay"] = statuses.max(axis=1)
    result["num_delayed_months"] = (statuses >= 1).sum(axis=1).astype("float64")
    result["num_severely_delayed_months"] = (statuses >= 3).sum(axis=1).astype("float64")
    result["bill_trend"] = bills.iloc[:, 0] - bills.iloc[:, -1]
    result["payment_trend"] = payments.iloc[:, 0] - payments.iloc[:, -1]

    return result


class DerivedFeatureAdder(BaseEstimator, TransformerMixin):
    """Stateless transformer that appends the derived feature block.

    Implements the scikit-learn transformer contract so it can be serialised
    inside the model bundle and applied identically at inference time.
    """

    def fit(self, X, y=None):  # noqa: N803 - sklearn API uses X
        return self

    def transform(self, X, y=None):  # noqa: N803
        frame = pd.DataFrame(X).copy()
        for column in config.USER_INPUT_FEATURES:
            if column not in frame.columns:
                frame[column] = np.nan
        enriched = compute_derived_features(frame[list(config.USER_INPUT_FEATURES)])
        return enriched[list(config.MODEL_FEATURES)]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(list(config.MODEL_FEATURES), dtype=object)

    def set_output(self, *, transform=None):  # pragma: no cover - sklearn plumbing
        return self


def build_preprocessor(scale_numeric: bool = False) -> ColumnTransformer:
    """Build the ColumnTransformer applied to the derived feature frame."""
    numeric_steps: List[tuple] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # dense output: HistGradientBoosting requires dense input.
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, list(config.NUMERIC_FEATURES)),
            ("cat", categorical_pipeline, list(config.CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_pipeline(estimator, scale_numeric: bool = False) -> Pipeline:
    """Full pipeline: derived features -> preprocessing -> estimator."""
    return Pipeline(
        steps=[
            ("features", DerivedFeatureAdder()),
            ("preprocess", build_preprocessor(scale_numeric=scale_numeric)),
            ("model", estimator),
        ]
    )


def transformed_feature_names(pipeline: Pipeline) -> List[str]:
    """Feature names produced by the preprocessing step of a fitted pipeline."""
    preprocessor = pipeline.named_steps["preprocess"]
    return [str(name) for name in preprocessor.get_feature_names_out()]


def source_feature_of(transformed_name: str) -> str:
    """Map a transformed column name back to its source feature.

    ``num__credit_utilization -> credit_utilization``
    ``cat__education_university -> education``
    """
    name = transformed_name
    if "__" in name:
        name = name.split("__", 1)[1]
    if name in config.MODEL_FEATURES:
        return name
    for feature in config.CATEGORICAL_FEATURES:
        if name.startswith(f"{feature}_"):
            return feature
    return name


def aggregate_contributions(
    transformed_names: Iterable[str], values: Iterable[float]
) -> dict:
    """Sum one-hot contributions back onto their source feature.

    A categorical feature encoded as several one-hot columns therefore receives
    a single, human-readable contribution equal to the sum of its indicator
    contributions for that row.
    """
    totals: dict = {}
    for name, value in zip(transformed_names, values):
        source = source_feature_of(name)
        totals[source] = totals.get(source, 0.0) + float(value)
    return totals


def build_design_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Derived feature frame used for exploratory analysis and CSV export."""
    enriched = compute_derived_features(frame[list(config.USER_INPUT_FEATURES)])
    return enriched[list(config.MODEL_FEATURES)]


class PlattCalibrator(BaseEstimator, TransformerMixin):
    """Sigmoid (Platt) probability calibration fitted on a held-out split.

    Implemented explicitly rather than through ``CalibratedClassifierCV`` so
    that the calibration data is unambiguous (the validation split, never the
    test split) and so the fitted transform is trivially serialisable and
    inspectable. It is a strictly monotone map from the base model's log-odds
    to a probability, which means the ranking of borrowers - and therefore
    ROC-AUC - is unchanged by calibration, while the probability values become
    more trustworthy.
    """

    def __init__(self, c: float = 1e6):
        self.c = c

    def fit(self, X, y):  # noqa: N803 - sklearn API
        from sklearn.linear_model import LogisticRegression

        from src.utils import safe_logit

        probabilities = np.asarray(X, dtype=float).ravel()
        margins = np.array([safe_logit(value) for value in probabilities]).reshape(-1, 1)
        self.model_ = LogisticRegression(C=self.c, solver="lbfgs", max_iter=1000)
        self.model_.fit(margins, np.asarray(y).ravel())
        self.n_features_in_ = 1
        self.slope_ = float(self.model_.coef_.ravel()[0])
        self.intercept_ = float(self.model_.intercept_.ravel()[0])
        return self

    def transform(self, X):  # noqa: N803
        from src.utils import safe_logit

        probabilities = np.asarray(X, dtype=float).ravel()
        margins = np.array([safe_logit(value) for value in probabilities]).reshape(-1, 1)
        return self.model_.predict_proba(margins)[:, 1]

    def calibrate(self, probabilities) -> np.ndarray:
        """Array-friendly alias used by the predictor."""
        return self.transform(probabilities)

    def as_dict(self) -> dict:
        return {
            "method": "platt_sigmoid",
            "slope": getattr(self, "slope_", None),
            "intercept": getattr(self, "intercept_", None),
            "c": self.c,
        }
