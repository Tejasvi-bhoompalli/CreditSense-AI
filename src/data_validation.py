"""Data-quality validation for both the dataset and user-supplied inputs.

Validation never repairs data silently. It reports findings with a severity so
the UI can distinguish blocking problems from informational notes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from src import config

SEVERITY_ORDER = {"error": 0, "warning": 1, "info": 2}


@dataclass
class ValidationIssue:
    code: str
    severity: str
    message: str
    column: Optional[str] = None
    count: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "column": self.column,
            "count": self.count,
        }


@dataclass
class ValidationReport:
    issues: List[ValidationIssue] = field(default_factory=list)
    checks_run: int = 0
    context: Dict[str, Any] = field(default_factory=dict)

    def add(self, code: str, severity: str, message: str, column: Optional[str] = None,
            count: Optional[int] = None) -> None:
        self.issues.append(ValidationIssue(code, severity, message, column, count))

    @property
    def errors(self) -> List[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def infos(self) -> List[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "info"]

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def sorted_issues(self) -> List[ValidationIssue]:
        return sorted(self.issues, key=lambda issue: SEVERITY_ORDER.get(issue.severity, 3))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "checks_run": self.checks_run,
            "is_valid": self.is_valid,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "info_count": len(self.infos),
            "issues": [issue.as_dict() for issue in self.sorted_issues()],
            "context": self.context,
        }

    def summary_line(self) -> str:
        return (
            f"{self.checks_run} checks run - {len(self.errors)} error(s), "
            f"{len(self.warnings)} warning(s), {len(self.infos)} note(s)"
        )


class InputValidationError(ValueError):
    """Raised when user-supplied assessment inputs cannot be used."""

    def __init__(self, report: ValidationReport):
        self.report = report
        super().__init__("; ".join(issue.message for issue in report.errors))


# ---------------------------------------------------------------------------
# Dataset validation
# ---------------------------------------------------------------------------


def validate_dataset(
    frame: pd.DataFrame, provenance: Optional[Any] = None
) -> ValidationReport:
    """Run structural and domain checks over a canonical dataset."""
    report = ValidationReport()
    target = config.TARGET_COLUMN

    report.checks_run += 1
    if frame is None or len(frame) == 0:
        report.add("dataset_empty", "error", "The dataset contains no rows.")
        return report

    report.context["rows"] = int(len(frame))
    report.context["columns"] = int(frame.shape[1])

    # Duplicate labels make every downstream column selection ambiguous, so this
    # is refused outright rather than producing a silently wrong feature matrix.
    report.checks_run += 1
    duplicated_labels = frame.columns[frame.columns.duplicated()].unique().tolist()
    if duplicated_labels:
        report.add(
            "duplicate_column_labels",
            "error",
            f"The frame contains duplicate column labels: {duplicated_labels}. Column "
            "selection would be ambiguous, so this frame is refused.",
            count=len(duplicated_labels),
        )
        return report

    # Required columns -----------------------------------------------------
    report.checks_run += 1
    missing = [column for column in config.USER_INPUT_FEATURES if column not in frame.columns]
    if missing:
        report.add(
            "missing_required_columns",
            "error",
            f"Missing required canonical columns: {missing}",
        )

    report.checks_run += 1
    if target not in frame.columns:
        report.add("missing_target", "error", f"Target column '{target}' is absent.")
        return report

    # Target sanity --------------------------------------------------------
    report.checks_run += 1
    target_values = pd.to_numeric(frame[target], errors="coerce")
    if target_values.isna().any():
        report.add(
            "target_not_numeric",
            "error",
            "Target contains non-numeric values.",
            column=target,
            count=int(target_values.isna().sum()),
        )
    unique_target = sorted(target_values.dropna().unique().tolist())
    if unique_target != [0, 1]:
        report.add(
            "target_not_binary",
            "error",
            f"Target must be binary 0/1 but has values {unique_target}.",
            column=target,
        )
    else:
        positive_rate = float(target_values.mean())
        report.context["target_positive_rate"] = round(positive_rate, 6)
        imbalance = float(max(positive_rate, 1 - positive_rate) / max(min(positive_rate, 1 - positive_rate), 1e-9))
        report.context["class_imbalance_ratio"] = round(imbalance, 4)
        if imbalance > 4.0:
            report.add(
                "class_imbalance",
                "warning",
                "The target is imbalanced "
                f"({imbalance:.1f}:1). Accuracy alone is not a trustworthy metric here; "
                "recall, precision, ROC-AUC and PR-AUC are reported as well.",
                column=target,
            )

    # Missing values -------------------------------------------------------
    report.checks_run += 1
    model_columns = [column for column in config.MODEL_FEATURES if column in frame.columns]
    missing_by_column = {
        column: int(frame[column].isna().sum())
        for column in [*model_columns, target]
        if int(frame[column].isna().sum()) > 0
    }
    if missing_by_column:
        report.add(
            "missing_values",
            "warning",
            "Missing values present in model features; the pipeline imputes them "
            f"inside the training split only: {missing_by_column}",
            count=int(sum(missing_by_column.values())),
        )
    else:
        report.add(
            "no_missing_values",
            "info",
            "No missing values detected in any model feature or the target.",
        )

    # Duplicates -----------------------------------------------------------
    report.checks_run += 1
    duplicate_count = int(frame.drop(columns=[target], errors="ignore").duplicated().sum())
    if duplicate_count:
        report.add(
            "duplicate_rows",
            "info",
            f"{duplicate_count} rows share identical feature values (excluding the target). "
            "These may be genuinely distinct clients with identical profiles, so they are "
            "retained rather than silently dropped.",
            count=duplicate_count,
        )

    # Domain checks --------------------------------------------------------
    report.checks_run += 1
    if "credit_limit" in frame.columns:
        bad = int((pd.to_numeric(frame["credit_limit"], errors="coerce") <= 0).sum())
        if bad:
            report.add(
                "non_positive_credit_limit",
                "error",
                "Credit limit must be strictly positive.",
                column="credit_limit",
                count=bad,
            )

    report.checks_run += 1
    if "age" in frame.columns:
        age = pd.to_numeric(frame["age"], errors="coerce")
        outside = int(((age < 18) | (age > 100)).sum())
        if outside:
            report.add(
                "age_out_of_range",
                "warning",
                "Age outside the supported 18-100 range.",
                column="age",
                count=outside,
            )

    report.checks_run += 1
    status_columns = [column for column in config.REPAY_STATUS_COLUMNS if column in frame.columns]
    if status_columns:
        statuses = frame[status_columns].apply(pd.to_numeric, errors="coerce")
        outside = int(((statuses < -2) | (statuses > 8)).sum().sum())
        if outside:
            report.add(
                "repayment_status_out_of_range",
                "warning",
                "Repayment status values outside the documented -2..8 scale were found.",
                count=outside,
            )
        undocumented = int((statuses == -2).sum().sum())
        report.add(
            "repayment_status_scale",
            "info",
            "The published scale documents -1, 0 and 1..9. This extract also contains -2 "
            f"({undocumented} occurrences), which is treated as 'no consumption / not "
            "applicable'. The meaning is documented rather than silently recoded.",
        )

    report.checks_run += 1
    payment_columns = [column for column in config.PAYMENT_AMOUNT_COLUMNS if column in frame.columns]
    if payment_columns:
        payments = frame[payment_columns].apply(pd.to_numeric, errors="coerce")
        negative = int((payments < 0).sum().sum())
        if negative:
            report.add(
                "negative_payment_amount",
                "error",
                "Payment amounts must not be negative.",
                count=negative,
            )

    report.checks_run += 1
    bill_columns = [column for column in config.BILL_AMOUNT_COLUMNS if column in frame.columns]
    if bill_columns:
        bills = frame[bill_columns].apply(pd.to_numeric, errors="coerce")
        negative_rows = int((bills < 0).any(axis=1).sum())
        if negative_rows:
            report.add(
                "negative_bill_amount",
                "info",
                f"{negative_rows} client-months have a negative statement balance. In this "
                "dataset a negative balance represents credit in the client's favour "
                "(overpayment or refund); the value is preserved, not clipped to zero.",
                count=negative_rows,
            )

    report.checks_run += 1
    constant_columns = [
        column for column in model_columns if frame[column].nunique(dropna=True) <= 1
    ]
    if constant_columns:
        report.add(
            "constant_columns",
            "warning",
            f"Constant columns carry no signal: {constant_columns}",
            count=len(constant_columns),
        )

    report.checks_run += 1
    if provenance is not None and getattr(provenance, "is_synthetic", False):
        report.add(
            "synthetic_data",
            "warning",
            config.SYNTHETIC_DISCLOSURE,
        )
    elif provenance is not None:
        report.add(
            "geographic_scope",
            "warning",
            config.GEOGRAPHIC_DISCLOSURE,
        )

    return report


# ---------------------------------------------------------------------------
# User input validation
# ---------------------------------------------------------------------------


def _coerce_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else None
    text = str(value).strip().replace(",", "")
    if text == "":
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if np.isfinite(number) else None


def validate_user_inputs(
    inputs: Dict[str, Any], strict: bool = True
) -> tuple:
    """Validate and coerce raw form values into the canonical schema.

    Returns ``(cleaned_inputs, report)``. When ``strict`` is True an invalid
    payload raises :class:`InputValidationError` so that no prediction is ever
    produced from silently repaired data.
    """
    report = ValidationReport()
    cleaned: Dict[str, Any] = {}

    report.checks_run += 1
    if not isinstance(inputs, dict) or not inputs:
        report.add("empty_input", "error", "No input values were supplied.")
        if strict:
            raise InputValidationError(report)
        return cleaned, report

    for feature in config.USER_INPUT_FEATURES:
        report.checks_run += 1
        if feature not in inputs:
            report.add(
                "missing_input",
                "error",
                f"Required field '{config.FEATURE_LABELS.get(feature, feature)}' is missing.",
                column=feature,
            )
            continue

        value = inputs[feature]

        if feature in config.CATEGORICAL_FEATURES:
            text = str(value).strip()
            if text not in config.EDUCATION_CATEGORIES:
                report.add(
                    "invalid_category",
                    "error",
                    f"'{text}' is not a supported category for {feature}. "
                    f"Expected one of {config.EDUCATION_CATEGORIES}.",
                    column=feature,
                )
                continue
            cleaned[feature] = text
            continue

        number = _coerce_number(value)
        if number is None:
            report.add(
                "invalid_number",
                "error",
                f"'{config.FEATURE_LABELS.get(feature, feature)}' must be a finite number.",
                column=feature,
            )
            continue

        low, high = config.INPUT_RANGES.get(feature, (float("-inf"), float("inf")))
        if number < low or number > high:
            report.add(
                "out_of_range",
                "error",
                f"{config.FEATURE_LABELS.get(feature, feature)} = {number:,.2f} is outside the "
                f"supported range {low:,.0f} to {high:,.0f}.",
                column=feature,
            )
            continue

        if feature in config.REPAY_STATUS_COLUMNS and number != int(number):
            report.add(
                "non_integer_status",
                "error",
                f"{feature} must be a whole-number repayment status level.",
                column=feature,
            )
            continue

        cleaned[feature] = float(number)

    if cleaned.get("payment_amount_1") is not None and cleaned.get("bill_amount_1") is not None:
        report.checks_run += 1
        if cleaned["payment_amount_1"] > 0 and cleaned["bill_amount_1"] < 0:
            report.add(
                "payment_on_credit_balance",
                "info",
                "A payment was recorded against a negative (credit) balance. This is "
                "accepted: the dataset contains such client-months.",
            )

    if report.errors and strict:
        raise InputValidationError(report)

    return cleaned, report


def build_input_frame(cleaned_inputs: Dict[str, Any]) -> pd.DataFrame:
    """Single-row DataFrame in the exact canonical input order."""
    row = {feature: [cleaned_inputs[feature]] for feature in config.USER_INPUT_FEATURES}
    return pd.DataFrame(row)


def issues_to_frame(report: ValidationReport) -> pd.DataFrame:
    issues: Iterable[ValidationIssue] = report.sorted_issues()
    rows = [
        {
            "Severity": issue.severity.upper(),
            "Check": issue.code,
            "Column": issue.column or "-",
            "Count": issue.count if issue.count is not None else "-",
            "Message": issue.message,
        }
        for issue in issues
    ]
    if not rows:
        return pd.DataFrame(columns=["Severity", "Check", "Column", "Count", "Message"])
    return pd.DataFrame(rows)
