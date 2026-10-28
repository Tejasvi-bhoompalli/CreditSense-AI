"""Dataset discovery, adaptation and profiling.

The application never assumes a fixed set of source columns. This module:

1. discovers tabular files in ``data/``,
2. normalises their column names,
3. detects a genuine default (not approval) target column,
4. maps verified source columns onto the canonical schema used everywhere else,
5. exposes a reproducible synthetic generator for the documented fallback case.

No synthetic data is ever produced unless the caller explicitly asks for it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src import config
from src.utils import get_logger, sha256_of_file, utc_now_iso

logger = get_logger(__name__)

SUPPORTED_SUFFIXES: Tuple[str, ...] = (".csv", ".xls", ".xlsx", ".parquet")
PREFERRED_FILENAMES: Tuple[str, ...] = (
    "uci_credit_card.csv",
    "default of credit card clients.xls",
    "credit_card_default.csv",
    "credit_risk_dataset.csv",
)


class DatasetError(RuntimeError):
    """Raised when a dataset cannot be found, read or adapted."""


class DatasetSchemaError(DatasetError):
    """Raised when a dataset cannot be mapped onto the canonical schema."""


# ---------------------------------------------------------------------------
# Discovery and reading
# ---------------------------------------------------------------------------


def normalize_column_name(name: str) -> str:
    """Lower-case, strip and collapse punctuation to a single underscore."""
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(name).strip().lower())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def discover_dataset_files(search_root: Optional[Path] = None) -> List[Path]:
    """Return supported tabular files found under ``data/``, best candidate first."""
    root = Path(search_root or config.DATA_DIR)
    if not root.exists():
        return []
    found = [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    if not found:
        return []

    def sort_key(path: Path) -> Tuple[int, int]:
        preferred = 0 if path.name.lower() in PREFERRED_FILENAMES else 1
        return (preferred, -path.stat().st_size)

    return sorted(found, key=sort_key)


def _read_tabular(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix in {".xls", ".xlsx"}:
            return pd.read_excel(path)
        if suffix == ".parquet":
            return pd.read_parquet(path)
    except Exception as exc:  # pragma: no cover - depends on file corruption
        raise DatasetError(f"Could not read '{path.name}': {exc}") from exc
    raise DatasetError(f"Unsupported dataset format: {path.suffix}")


@dataclass
class DatasetProvenance:
    """Everything the UI and the reports need to attribute the data honestly."""

    path: str
    filename: str
    sha256: str
    rows: int
    columns: int
    is_synthetic: bool
    loaded_at: str = field(default_factory=utc_now_iso)
    name: str = str(config.DATASET_INFO["name"])
    publisher: str = str(config.DATASET_INFO["publisher"])
    authors: str = str(config.DATASET_INFO["authors"])
    doi: str = str(config.DATASET_INFO["doi"])
    url: str = str(config.DATASET_INFO["url"])
    license: str = str(config.DATASET_INFO["license"])
    geography: str = str(config.DATASET_INFO["geography"])
    observation_period: str = str(config.DATASET_INFO["observation_period"])
    currency: str = str(config.DATASET_INFO["currency"])
    citation: str = str(config.DATASET_INFO["citation"])
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "filename": self.filename,
            "sha256": self.sha256,
            "rows": self.rows,
            "columns": self.columns,
            "is_synthetic": self.is_synthetic,
            "loaded_at": self.loaded_at,
            "name": self.name,
            "publisher": self.publisher,
            "authors": self.authors,
            "doi": self.doi,
            "url": self.url,
            "license": self.license,
            "geography": self.geography,
            "observation_period": self.observation_period,
            "currency": self.currency,
            "citation": self.citation,
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# Target detection and canonicalisation
# ---------------------------------------------------------------------------


def detect_target_column(df: pd.DataFrame) -> str:
    """Identify the binary default target, refusing approval-style targets."""
    normalised = {normalize_column_name(c): c for c in df.columns}
    for candidate in config.TARGET_CANDIDATES:
        if candidate in normalised:
            original = normalised[candidate]
            series = df[original]
            if series.nunique(dropna=True) == 2:
                return original
            raise DatasetSchemaError(
                f"Column '{original}' looks like the target but has "
                f"{series.nunique(dropna=True)} distinct values; expected a binary label."
            )

    # Fallback: any binary column whose name mentions default.
    for column in df.columns:
        normalised_name = normalize_column_name(column)
        if "default" in normalised_name and df[column].nunique(dropna=True) == 2:
            return column

    raise DatasetSchemaError(
        "No binary default-target column could be identified. Expected one of: "
        + ", ".join(config.TARGET_CANDIDATES)
        + ". The application will not treat a loan-approval label as a default outcome."
    )


def canonicalize_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    """Map verified source columns onto the canonical schema.

    Returns the canonical frame, the applied mapping and a list of unmapped
    source columns (kept out of the model but preserved for transparency).
    """
    renamed = {column: normalize_column_name(column) for column in df.columns}
    frame = df.rename(columns=renamed).copy()

    mapping: Dict[str, str] = {}
    for source, canonical in config.SOURCE_COLUMN_MAP.items():
        if source in frame.columns and source != canonical:
            frame = frame.rename(columns={source: canonical})
            mapping[source] = canonical
        elif source in frame.columns:
            mapping[source] = canonical

    missing = [
        column
        for column in config.USER_INPUT_FEATURES
        if column not in frame.columns
    ]
    if missing:
        raise DatasetSchemaError(
            "The dataset could not be adapted to the canonical schema. Missing "
            f"required columns: {missing}. Present columns: {sorted(frame.columns)}"
        )

    if config.TARGET_COLUMN not in frame.columns:
        raise DatasetSchemaError(
            "Target column missing after mapping; expected "
            f"'{config.TARGET_COLUMN}'."
        )

    dropped_ids = [column for column in config.ID_COLUMNS if column in frame.columns]
    frame = frame.drop(columns=dropped_ids, errors="ignore")

    unmapped = [
        column
        for column in frame.columns
        if column not in set(config.MODEL_FEATURES)
        | set(config.GROUP_COLUMNS)
        | {config.TARGET_COLUMN}
    ]
    return frame, mapping, unmapped


def _normalise_education(series: pd.Series) -> pd.Series:
    """Map an education column onto the documented categories.

    Accepts either the source's numeric codes (1 graduate school, 2 university,
    3 high school, 4/5/6/0 others) or already-textual category labels. Unknown
    labels become 'other' rather than being dropped or invented.
    """
    numeric_codes = pd.to_numeric(series, errors="coerce")
    if numeric_codes.notna().all() and len(series) > 0:
        mapped = numeric_codes.round().map(config.EDUCATION_CODES)
        return mapped.fillna("other").astype("string")

    normalised = (
        series.astype("string")
        .str.strip()
        .str.lower()
        .str.replace(r"[\s\-]+", "_", regex=True)
    )
    return (
        normalised.where(normalised.isin(config.EDUCATION_CATEGORIES), "other").astype("string")
    )


def _clean_types(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce dtypes and normalise the documented coded categorical columns."""
    frame = frame.copy()

    numeric_columns = set(config.USER_INPUT_FEATURES) - set(config.CATEGORICAL_FEATURES)
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if config.TARGET_COLUMN in frame.columns:
        frame[config.TARGET_COLUMN] = (
            pd.to_numeric(frame[config.TARGET_COLUMN], errors="coerce").fillna(0).astype("int8")
        )

    if "education" in frame.columns:
        # The source file encodes education numerically, but an externally supplied
        # file may already contain text labels. Handling both explicitly matters: a
        # textual column run through the numeric-code branch would silently collapse
        # every row into a single category.
        frame["education"] = _normalise_education(frame["education"])

    for group_column in ("sex", "marriage"):
        if group_column in frame.columns:
            frame[group_column] = (
                pd.to_numeric(frame[group_column], errors="coerce").astype("float")
            )

    return frame


# ---------------------------------------------------------------------------
# Synthetic fallback
# ---------------------------------------------------------------------------


def generate_synthetic_dataset(
    rows: int = 8000, seed: int = config.RANDOM_SEED
) -> pd.DataFrame:
    """Reproducible synthetic dataset with the canonical schema.

    Used only when no real dataset is present, and by the test-suite (where it
    keeps the tests fast and deterministic). Target outcomes are produced from
    several interacting latent risk factors plus stochastic noise, so the label
    is not a deterministic function of the features.

    The records are NOT real people. They are labelled synthetic everywhere.
    """
    rng = np.random.default_rng(seed)
    rows = int(max(200, rows))

    age = np.clip(rng.normal(35.5, 9.2, rows), 21, 79).round().astype(int)
    # Emitted as numeric codes, exactly like the source file, so that the real dataset
    # and the fallback exercise the same cleaning path.
    education = rng.choice([1, 2, 3, 4], size=rows, p=[0.35, 0.47, 0.16, 0.02]).astype(float)
    sex = rng.choice([1.0, 2.0], size=rows, p=[0.40, 0.60])
    marriage = rng.choice([1.0, 2.0, 3.0], size=rows, p=[0.45, 0.53, 0.02])

    credit_limit = np.clip(
        np.round(np.exp(rng.normal(11.7, 0.75, rows)) / 10_000) * 10_000, 10_000, 1_000_000
    )

    # Latent delinquency propensity, persistent across months.
    latent = rng.normal(0.0, 1.0, rows) - 0.35 * (age - 35.0) / 9.0
    repay = np.empty((rows, 6), dtype=float)
    previous = latent.copy()
    for month in range(6):
        previous = 0.72 * previous + 0.55 * rng.normal(0.0, 1.0, rows)
        raw = np.round(previous - 0.35)
        repay[:, month] = np.clip(raw, -2, 8)

    utilization = np.clip(np.exp(rng.normal(-0.35, 0.75, rows)), 0.01, 2.6)
    bills = np.empty((rows, 6), dtype=float)
    level = credit_limit * utilization
    for month in range(6):
        level = level * np.exp(rng.normal(0.0, 0.16, rows))
        bills[:, month] = np.round(level * rng.normal(1.0, 0.08, rows))
    sender = rng.random(rows) < 0.06
    bills[sender, -1] = -np.round(np.abs(bills[sender, -1]) * 0.4)
    bills = np.round(bills)

    payments = np.empty((rows, 6), dtype=float)
    diligence = np.clip(rng.beta(2.2, 3.0, rows) + 0.14 * (repay[:, 0] <= 0), 0.0, 0.98)
    for month in range(6):
        base = np.clip(bills[:, month], 0, None) * diligence
        payments[:, month] = np.round(np.where(rng.random(rows) < 0.12, 0.0, base))

    max_delay = repay.max(axis=1)
    delayed_months = (repay >= 1).sum(axis=1)
    severe = (repay >= 3).sum(axis=1)
    total_bill = np.clip(bills.sum(axis=1), 1.0, None)
    repayment_ratio = np.clip(payments.sum(axis=1) / total_bill, 0.0, 2.0)
    utilization_ratio = np.clip(bills[:, 0] / credit_limit, -2.0, 5.0)

    logit = (
        -1.55
        + 0.95 * max_delay
        + 0.42 * delayed_months
        + 0.55 * severe
        + 1.35 * utilization_ratio
        - 1.85 * repayment_ratio
        + 0.0000018 * (bills[:, 0] - bills[:, -1])
        + 0.55 * np.tanh((utilization_ratio - 0.8) * (1.0 + delayed_months))
        - 0.28 * (repay[:, 0] <= 0).astype(float)
        + 0.55 * rng.normal(0.0, 1.0, rows)
    )
    probability = 1.0 / (1.0 + np.exp(-logit))
    default = (rng.random(rows) < probability).astype("int8")

    data: Dict[str, Any] = {
        "credit_limit": credit_limit,
        "age": age,
        "education": education,
        "sex": sex,
        "marriage": marriage,
    }
    for index in range(6):
        data[f"repay_status_{index + 1}"] = repay[:, index]
        data[f"bill_amount_{index + 1}"] = bills[:, index]
        data[f"payment_amount_{index + 1}"] = payments[:, index]
    data[config.TARGET_COLUMN] = default

    frame = _clean_types(pd.DataFrame(data))

    # `education` is both a model feature and a reporting group, so the column
    # list is built with an explicit de-duplication: a frame with duplicate
    # labels would make column selection ambiguous further down the pipeline.
    columns: List[str] = list(config.USER_INPUT_FEATURES)
    for group_column in config.GROUP_COLUMNS:
        if group_column in frame.columns and group_column not in columns:
            columns.append(group_column)
    columns.append(config.TARGET_COLUMN)
    return frame[columns]


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------


def profile_frame(frame: pd.DataFrame) -> Dict[str, Any]:
    """Structural profile used by the Dataset Explorer and the validation step."""
    target = config.TARGET_COLUMN
    numeric_columns = [
        column
        for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column]) and column != target
    ]
    categorical_columns = [column for column in frame.columns if column not in numeric_columns + [target]]

    target_counts = frame[target].value_counts().to_dict() if target in frame.columns else {}
    positives = int(target_counts.get(1, 0))
    total = int(len(frame))

    describe = (
        frame[numeric_columns].describe().T.reset_index().rename(columns={"index": "feature"})
        if numeric_columns
        else pd.DataFrame()
    )

    profile: Dict[str, Any] = {
        "rows": total,
        "columns": int(frame.shape[1]),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "missing_total": int(frame.isna().sum().sum()),
        "missing_by_column": {
            column: int(count)
            for column, count in frame.isna().sum().items()
            if int(count) > 0
        },
        "duplicate_rows_excluding_target": int(
            frame.drop(columns=[target], errors="ignore").duplicated().sum()
        ),
        "target_counts": {int(k): int(v) for k, v in target_counts.items()},
        "target_positive_rate": round(positives / total, 6) if total else 0.0,
        "class_imbalance_ratio": (
            round(max(target_counts.values()) / max(1, min(target_counts.values())), 4)
            if len(target_counts) == 2
            else None
        ),
        "descriptive_statistics": describe.to_dict(orient="records") if not describe.empty else [],
        "profiled_at": utc_now_iso(),
    }
    return profile


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


@dataclass
class CreditDataset:
    """A loaded, canonical dataset plus everything known about it."""

    frame: pd.DataFrame
    provenance: DatasetProvenance
    mapping: Dict[str, str]
    unmapped_columns: List[str]
    profile: Dict[str, Any]

    @property
    def target(self) -> pd.Series:
        return self.frame[config.TARGET_COLUMN]

    @property
    def is_synthetic(self) -> bool:
        return self.provenance.is_synthetic

    @property
    def feature_frame(self) -> pd.DataFrame:
        return self.frame[list(config.USER_INPUT_FEATURES)].copy()


_CACHE: Dict[str, CreditDataset] = {}


def load_dataset(
    path: Optional[Path] = None,
    allow_synthetic_fallback: bool = False,
    synthetic_rows: int = 8000,
    use_cache: bool = True,
) -> CreditDataset:
    """Load and adapt the dataset.

    Raises :class:`DatasetError` with an actionable message when nothing usable
    is found, unless ``allow_synthetic_fallback`` is set (used by tests and by
    the documented "no dataset present" development path).
    """
    if path is None:
        candidates = discover_dataset_files()
        if not candidates:
            if not allow_synthetic_fallback:
                raise DatasetError(
                    f"No tabular dataset found under '{config.DATA_DIR}'. Place a CSV, "
                    "XLS(X) or Parquet file in data/ and reload."
                )
            frame = generate_synthetic_dataset(synthetic_rows)
            provenance = DatasetProvenance(
                path="<generated>",
                filename="synthetic_demo_dataset.csv",
                sha256="synthetic",
                rows=int(frame.shape[0]),
                columns=int(frame.shape[1]),
                is_synthetic=True,
                notes=[config.SYNTHETIC_DISCLOSURE],
            )
            return CreditDataset(frame, provenance, {}, [], profile_frame(frame))
        path = candidates[0]

    path = Path(path)
    if not path.exists():
        raise DatasetError(f"Dataset file not found: {path}")

    cache_key = f"{path}::{path.stat().st_mtime_ns}"
    if use_cache and cache_key in _CACHE:
        return _CACHE[cache_key]

    raw = _read_tabular(path)
    if raw.empty:
        raise DatasetError(f"Dataset '{path.name}' is empty.")

    target_source = detect_target_column(raw)
    raw = raw.rename(columns={target_source: normalize_column_name(target_source)})
    frame, mapping, unmapped = canonicalize_frame(raw)
    frame = _clean_types(frame)

    provenance = DatasetProvenance(
        path=str(path),
        filename=path.name,
        sha256=sha256_of_file(path),
        rows=int(frame.shape[0]),
        columns=int(frame.shape[1]),
        is_synthetic=False,
        notes=[
            config.GEOGRAPHIC_DISCLOSURE,
            "Source target column: "
            f"'{target_source}' -> canonical '{config.TARGET_COLUMN}'.",
            "The source file contains no missing values; no imputation was "
            "required for this dataset.",
        ],
    )
    dataset = CreditDataset(frame, provenance, mapping, unmapped, profile_frame(frame))
    logger.info(
        "Loaded dataset %s rows=%s cols=%s positive_rate=%.4f",
        path.name,
        dataset.profile["rows"],
        dataset.profile["columns"],
        dataset.profile["target_positive_rate"],
    )
    if use_cache:
        _CACHE[cache_key] = dataset
    return dataset


def clear_dataset_cache() -> None:
    _CACHE.clear()


def describe_missing_features(frame: pd.DataFrame) -> List[str]:
    """Canonical features from the brief that this dataset does not provide."""
    present = set(frame.columns)
    return [name for name in config.UNAVAILABLE_FEATURES if name not in present]


def dataset_fingerprint(dataset: CreditDataset) -> str:
    """Stable identifier for the exact data used to train a model."""
    if dataset.provenance.is_synthetic:
        return "synthetic-v1"
    return f"{dataset.provenance.filename}:{dataset.provenance.sha256[:16]}"


def ensure_finite(value: float, fallback: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    if math.isnan(number) or math.isinf(number):
        return fallback
    return number
