"""Group-level dispersion measures on the evaluation split.

This module exists to *describe* how the model behaves for groups that the
source dataset itself provides (sex, education, age band). It is deliberately
conservative:

* the protected attribute is never a model input; it is only a grouping key,
* every table states the sample size, because small groups produce unstable rates,
* nothing here is a legal fairness determination or a compliance claim,
* excluding one attribute does not remove bias, since proxy variables remain.

No sensitive attribute is collected from the user. These group values only exist
because the public research dataset contains them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src import config

SEX_LABELS = {1: "Group code 1 (as coded in source data)", 2: "Group code 2 (as coded in source data)"}

FAIRNESS_CAVEATS = [
    "Group labels come from the source dataset's own coded fields. They are not collected "
    "by this application and are never used as model inputs.",
    "These tables describe model behaviour on one historical Taiwanese sample from 2005. "
    "They are not a fairness audit, not a legal compliance determination, and not evidence "
    "of lawful or unlawful treatment in any jurisdiction.",
    "Removing an attribute from the feature set does not remove proxy information. Age, "
    "credit limit and repayment behaviour correlate with protected characteristics.",
    "Small groups produce unstable rates. Always read the sample size in the same row.",
    "Any real-world use would require jurisdiction-specific legal review and validation on "
    "locally representative data.",
]


def age_band(age: float) -> str:
    """Descriptive age bands used only for group-level reporting."""
    if age is None or (isinstance(age, float) and np.isnan(age)):
        return "unknown"
    value = float(age)
    if value < 25:
        return "18-24"
    if value < 35:
        return "25-34"
    if value < 45:
        return "35-44"
    if value < 55:
        return "45-54"
    return "55+"


@dataclass
class GroupMetricTable:
    group_column: str
    table: pd.DataFrame
    reference_group: Optional[str] = None
    notes: List[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "group_column": self.group_column,
            "reference_group": self.reference_group,
            "rows": self.table.to_dict(orient="records"),
            "notes": list(self.notes or []),
        }


def _group_labels(frame: pd.DataFrame, group_column: str) -> pd.Series:
    if group_column == "sex":
        return frame["sex"].map(
            lambda value: _label_for_code(value, "sex")
        ).astype("string")
    if group_column == "marriage":
        return frame["marriage"].map(
            lambda value: _label_for_code(value, "marriage")
        ).astype("string")
    if group_column == "age":
        return frame["age"].map(age_band).astype("string")
    if group_column == "education":
        return frame["education"].astype("string")
    return frame[group_column].astype("string")


def _label_for_code(value: Any, kind: str) -> str:
    try:
        code = int(float(value))
    except (TypeError, ValueError):
        return "unknown"
    if kind == "sex":
        return SEX_LABELS.get(code, f"code {code}")
    if kind == "marriage":
        mapping = {1: "Married (code 1)", 2: "Single (code 2)", 3: "Other (code 3)"}
        return mapping.get(code, f"code {code}")
    return f"code {code}"


def group_metrics(
    frame: pd.DataFrame,
    y_true,
    y_prob,
    group_column: str,
    threshold: float = 0.5,
    min_group_size: int = 30,
) -> Optional[GroupMetricTable]:
    """Per-group confusion-based metrics and ROC-AUC where computable."""
    if group_column not in frame.columns:
        return None

    labels = _group_labels(frame, group_column)
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    predicted = (y_prob >= threshold).astype(int)

    rows: List[Dict[str, Any]] = []
    for group in sorted(labels.dropna().unique().tolist()):
        mask = (labels == group).to_numpy()
        size = int(mask.sum())
        if size == 0:
            continue
        group_true = y_true[mask]
        group_pred = predicted[mask]
        group_prob = y_prob[mask]

        positives = int(group_true.sum())
        negatives = size - positives
        tp = int(((group_true == 1) & (group_pred == 1)).sum())
        fn = int(((group_true == 1) & (group_pred == 0)).sum())
        fp = int(((group_true == 0) & (group_pred == 1)).sum())
        tn = int(((group_true == 0) & (group_pred == 0)).sum())

        rows.append(
            {
                "Group": group,
                "Sample size": size,
                "Actual default rate": round(positives / size, 4) if size else np.nan,
                "Flagged rate (model)": round(float(group_pred.mean()), 4) if size else np.nan,
                "True positive rate (recall)": round(tp / positives, 4) if positives else np.nan,
                "False positive rate": round(fp / negatives, 4) if negatives else np.nan,
                "True positives": tp,
                "False negatives": fn,
                "False positives": fp,
                "True negatives": tn,
                "ROC-AUC": (
                    round(float(roc_auc_score(group_true, group_prob)), 4)
                    if positives and negatives
                    else np.nan
                ),
                "Reliable sample": "yes" if size >= min_group_size else "no - small group",
            }
        )

    if not rows:
        return None

    table = pd.DataFrame(rows).sort_values("Sample size", ascending=False).reset_index(drop=True)
    notes = list(FAIRNESS_CAVEATS)
    if group_column in config.PROTECTED_EXCLUDED_FEATURES:
        notes.append(
            f"'{group_column}' is EXCLUDED from the model features and appears here only as a "
            "descriptive grouping key."
        )
    return GroupMetricTable(
        group_column=group_column, table=table, reference_group=str(table.iloc[0]["Group"]), notes=notes
    )


def disparity_table(metric_table: GroupMetricTable) -> pd.DataFrame:
    """Ratios and differences relative to the largest group."""
    table = metric_table.table
    if table.empty:
        return table
    reference = table.iloc[0]
    rows = []
    for _, row in table.iterrows():
        flag_ratio = (
            row["Flagged rate (model)"] / reference["Flagged rate (model)"]
            if reference["Flagged rate (model)"]
            else np.nan
        )
        tpr_gap = (
            row["True positive rate (recall)"] - reference["True positive rate (recall)"]
            if pd.notna(row["True positive rate (recall)"])
            and pd.notna(reference["True positive rate (recall)"])
            else np.nan
        )
        rows.append(
            {
                "Group": row["Group"],
                "Flagged-rate ratio vs largest group": (
                    round(float(flag_ratio), 3) if pd.notna(flag_ratio) else np.nan
                ),
                "Recall gap vs largest group": (
                    round(float(tpr_gap), 4) if pd.notna(tpr_gap) else np.nan
                ),
                "Sample size": row["Sample size"],
            }
        )
    return pd.DataFrame(rows)


def threshold_tradeoff(y_true, y_prob) -> pd.DataFrame:
    """Precision / recall / FPR trade-off across candidate decision thresholds."""
    from sklearn.metrics import confusion_matrix

    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    rows = []
    for threshold in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70]:
        predicted = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        rows.append(
            {
                "Threshold": threshold,
                "Flagged share": round(float(predicted.mean()), 4),
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
                "False positive rate": round(fp / (fp + tn), 4) if (fp + tn) else 0.0,
                "False negatives": int(fn),
                "False positives": int(fp),
            }
        )
    return pd.DataFrame(rows)
