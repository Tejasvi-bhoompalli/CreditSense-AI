"""Small, dependency-light helpers shared across the application.

Formatting helpers are deliberately centralised so that currency, percentage and
probability rendering is identical on every page.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from src import config

_LOGGER_CONFIGURED = False


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def get_logger(name: str = "creditsense") -> logging.Logger:
    """Return a module logger that writes technical detail to stderr only.

    The logger never receives raw user input or secrets; callers are
    responsible for logging only structural information.
    """
    global _LOGGER_CONFIGURED
    if not _LOGGER_CONFIGURED:
        logging.basicConfig(
            level=os.getenv("CREDITSENSE_LOG_LEVEL", "INFO").upper(),
            format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        )
        _LOGGER_CONFIGURED = True
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Time / hashing
# ---------------------------------------------------------------------------


def utc_now_iso() -> str:
    """Current UTC timestamp in ISO-8601 (seconds precision, timezone-aware)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 fingerprint of a file without loading it fully."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def short_hash(value: str, length: int = 12) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


# ---------------------------------------------------------------------------
# JSON persistence
# ---------------------------------------------------------------------------


def save_json(payload: Dict[str, Any], path: Path) -> Path:
    """Write JSON atomically so a crashed write cannot corrupt an artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=str(path.parent), delete=False, suffix=".tmp"
    )
    try:
        json.dump(payload, handle, indent=2, sort_keys=False, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    finally:
        handle.close()
    os.replace(handle.name, path)
    return path


def load_json(path: Path, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {} if default is None else default
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def feature_label(name: str) -> str:
    """Human-readable label for a canonical feature name."""
    if name in config.FEATURE_LABELS:
        return config.FEATURE_LABELS[name]
    return name.replace("_", " ").strip().capitalize()


def format_currency_twd(value: float, decimals: int = 0) -> str:
    """Format an amount in New Taiwan Dollars (the dataset's currency)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"NT$ {value:,.{decimals}f}"


def format_number(value: float, decimals: int = 2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    if abs(value) >= 1000:
        return f"{value:,.{decimals}f}"
    return f"{value:.{decimals}f}"


def format_probability(value: float, decimals: int = 2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value * 100:.{decimals}f}%"


def format_percent(value: float, decimals: int = 1) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.{decimals}f}%"


def format_ratio_as_percent(value: float, decimals: int = 1) -> str:
    """Format a 0-1 ratio as a percentage."""
    return format_probability(value, decimals)


def format_signed(value: float, decimals: int = 2, suffix: str = "") -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:,.{decimals}f}{suffix}"


def format_signed_percent(points: float, decimals: int = 2) -> str:
    sign = "+" if points > 0 else ""
    return f"{sign}{points:.{decimals}f} pp"


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------


def clamp(value: float, low: float, high: float) -> float:
    if value is None:
        return low
    return max(low, min(high, value))


def safe_logit(probability: float, epsilon: float = 1e-6) -> float:
    """Numerically safe log-odds transform."""
    p = clamp(float(probability), epsilon, 1.0 - epsilon)
    return math.log(p / (1.0 - p))


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    try:
        if denominator is None or float(denominator) == 0.0:
            return default
        return float(numerator) / float(denominator)
    except (TypeError, ValueError, ZeroDivisionError):
        return default


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def slugify(text: str, max_length: int = 60) -> str:
    """Sanitise a string for use inside a generated filename."""
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in str(text))
    safe = "-".join(part for part in safe.split("-") if part)
    return (safe or "report")[:max_length]


def unique_preserving_order(values: Iterable[str]) -> list:
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def safe_str(value: Any) -> str:
    """Convert an arbitrary value to a log-safe string."""
    text = str(value)
    for token in (config.LLM_API_KEY, os.getenv("OPENAI_API_KEY", "")):
        if token and len(token) > 6:
            text = text.replace(token, "***redacted***")
    return text
