"""Shared Streamlit UI components and chart theming.

Everything the pages render visually is defined here so the product looks like
one application rather than nine unrelated screens. Components are emitted as the
application's own HTML classes (``csa-*``) so styling does not depend on
fragile Streamlit internals.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import plotly.graph_objects as go
import streamlit as st

from src import config
from src.risk_scoring import risk_color

PLOT_COLORS = {
    "navy": "#101828",
    "navy2": "#172554",
    "indigo": "#4F46E5",
    "blue": "#3B82F6",
    "teal": "#14B8A6",
    "cyan": "#67E8F9",
    "muted": "#64748B",
    "border": "#E2E8F0",
    "surface": "#FFFFFF",
    "low": "#16A34A",
    "medium": "#D97706",
    "high": "#DC2626",
    "grid": "#EDF1F7",
}

CATEGORY_COLORS = [
    PLOT_COLORS["indigo"],
    PLOT_COLORS["blue"],
    PLOT_COLORS["teal"],
    PLOT_COLORS["navy2"],
    PLOT_COLORS["cyan"],
    PLOT_COLORS["muted"],
]

SEQUENTIAL_BLUES = [
    [0.0, "#EEF2FF"],
    [0.5, "#6366F1"],
    [1.0, "#101828"],
]


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------


def configure_page() -> None:
    st.set_page_config(
        page_title=f"{config.APP_NAME} - {config.APP_TAGLINE}",
        page_icon="assets/logo.svg",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "About": (
                f"{config.APP_NAME} - academic prototype.\n\n{config.APP_DISCLAIMER}"
            )
        },
    )


def load_css() -> None:
    css_path = Path(config.ASSETS_DIR) / "styles.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>",
                    unsafe_allow_html=True)


def brand_bar(model_status: Optional[Dict[str, Any]] = None) -> None:
    logo_path = Path(config.ASSETS_DIR) / "logo.svg"
    logo_svg = logo_path.read_text(encoding="utf-8") if logo_path.exists() else ""
    status = model_status or {}
    if status.get("ready"):
        status_html = (
            f"<strong>Model ready</strong><br/>{status.get('model_name', '')} "
            f"v{status.get('model_version', '')}<br/>trained {str(status.get('trained_at', ''))[:19]}Z"
        )
    else:
        status_html = "<strong>Model not trained</strong><br/>run python -m src.train_model"
    st.markdown(
        f"""
        <div class="csa-brandbar">
          <div class="csa-brandmark">{logo_svg}</div>
          <div class="csa-brandtext">
            <span class="csa-wordmark">{config.APP_NAME}</span>
            <span class="csa-brandtag">{config.APP_TAGLINE}</span>
          </div>
          <div class="csa-brandmeta">{status_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_header(
    title: str,
    subtitle: str,
    eyebrow: str = "Credit risk intelligence",
    model_status: Optional[Dict[str, Any]] = None,
) -> None:
    brand_bar(model_status)
    st.markdown(
        f"""
        <div class="csa-eyebrow">{eyebrow}</div>
        <h1 class="csa-pagetitle">{title}</h1>
        <p class="csa-pagesub">{subtitle}</p>
        """,
        unsafe_allow_html=True,
    )


def section(title: str, subtitle: Optional[str] = None) -> None:
    st.markdown(f'<div class="csa-section">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="csa-section-sub">{subtitle}</div>', unsafe_allow_html=True)


def divider() -> None:
    st.markdown('<hr class="csa-hr"/>', unsafe_allow_html=True)


def card(title: str, body_html: str) -> None:
    st.markdown(
        f'<div class="csa-card"><div class="csa-card-title">{title}</div>'
        f'<div class="csa-card-body">{body_html}</div></div>',
        unsafe_allow_html=True,
    )


def notice(message: str, kind: str = "info", strong: Optional[str] = None) -> None:
    prefix = f"<strong>{strong}</strong> " if strong else ""
    st.markdown(
        f'<div class="csa-notice {kind}">{prefix}{message}</div>', unsafe_allow_html=True
    )


def empty_state(title: str, text: str, icon: str = "&#9679;") -> None:
    st.markdown(
        f'<div class="csa-empty"><div class="csa-empty-icon">{icon}</div>'
        f'<div class="csa-empty-title">{title}</div>'
        f'<div class="csa-empty-text">{text}</div></div>',
        unsafe_allow_html=True,
    )


def kpi_row(items: Sequence[Dict[str, Any]]) -> None:
    """Render a responsive row of KPI cards.

    Each item: {label, value, help (optional), tone (low|medium|high|neutral)}
    """
    cards = []
    for item in items:
        tone = item.get("tone", "neutral")
        help_text = (
            f'<div class="csa-kpi-help">{item["help"]}</div>' if item.get("help") else ""
        )
        cards.append(
            f'<div class="csa-kpi is-{tone}">'
            f'<div class="csa-kpi-label">{item["label"]}</div>'
            f'<div class="csa-kpi-value">{item["value"]}</div>'
            f"{help_text}</div>"
        )
    st.markdown(f'<div class="csa-kpi-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def risk_badge(risk_category: str, score: Optional[float] = None) -> str:
    tone = {
        "LOW RISK": "low",
        "MEDIUM RISK": "medium",
        "HIGH RISK": "high",
    }.get(risk_category, "neutral")
    suffix = f" &middot; score {score:.1f}/100" if score is not None else ""
    return f'<span class="csa-badge {tone}">{risk_category}{suffix}</span>'


def risk_meter(probability: float, risk_category: str) -> None:
    """Band meter plus a numeric read-out (never colour alone)."""
    color = risk_color(risk_category)
    percent = max(0.0, min(1.0, float(probability))) * 100
    st.markdown(
        f"""
        <div class="csa-meter">
          <div class="csa-meter-fill" style="width:{percent:.2f}%;background:{color};"></div>
        </div>
        <div class="csa-meter-labels">
          <span>0% &middot; LOW</span><span>30%</span><span>60%</span><span>HIGH &middot; 100%</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def chip(label: str, kind: str = "ok") -> str:
    return f'<span class="csa-chip {kind}">{label}</span>'


def feature_availability_html() -> str:
    """Honest availability table: source-observed, derived, unavailable."""
    observed = [feature for feature in config.RAW_NUMERIC_FEATURES + config.CATEGORICAL_FEATURES]
    derived = list(config.DERIVED_FEATURES)
    excluded = list(config.PROTECTED_EXCLUDED_FEATURES)
    unavailable = list(config.UNAVAILABLE_FEATURES)
    rows = [
        ("Source-observed (used by the model)", len(observed), "ok"),
        ("Derived by documented formula (used by the model)", len(derived), "ok"),
        ("Present in the dataset but excluded as protected", len(excluded), "missing"),
        ("Requested by the brief but absent from this dataset", len(unavailable), "missing"),
    ]
    html = "".join(
        f'<tr><td>{label}</td><td style="text-align:right;font-weight:600">{count}</td>'
        f'<td style="text-align:right">{chip("included" if kind == "ok" else "not used", kind)}</td></tr>'
        for label, count, kind in rows
    )
    return (
        '<table style="width:100%;border-collapse:collapse;font-size:0.85rem">'
        "<thead><tr><th style='text-align:left'>Feature class</th>"
        "<th style='text-align:right'>Count</th><th style='text-align:right'>Status</th></tr></thead>"
        f"<tbody>{html}</tbody></table>"
    )


def footer() -> None:
    st.markdown(
        f"""
        <div class="csa-footer">
          <strong>{config.APP_NAME}</strong> &mdash; {config.APP_TAGLINE}.<br/>
          {config.APP_DISCLAIMER}<br/>
          Training data: {config.DATASET_INFO['name']} ({config.DATASET_INFO['geography']}, 
          {config.DATASET_INFO['observation_period']}), {config.DATASET_INFO['publisher']}, 
          licence {config.DATASET_INFO['license']}. DOI {config.DATASET_INFO['doi']}.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


def apply_chart_theme(
    figure: go.Figure,
    height: int = 330,
    title: Optional[str] = None,
    show_legend: bool = True,
) -> go.Figure:
    figure.update_layout(
        height=height,
        title=dict(
            text=title or "",
            font=dict(size=14, color=PLOT_COLORS["navy"], family="Inter, Segoe UI, sans-serif"),
            x=0,
            xanchor="left",
        ),
        font=dict(family="Inter, Segoe UI, sans-serif", size=12, color="#172033"),
        paper_bgcolor=PLOT_COLORS["surface"],
        plot_bgcolor=PLOT_COLORS["surface"],
        margin=dict(l=10, r=14, t=48 if title else 18, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11),
        ) if show_legend else dict(),
        showlegend=show_legend,
        hoverlabel=dict(
            bgcolor="#101828",
            font=dict(color="#FFFFFF", size=12, family="Inter, Segoe UI, sans-serif"),
        ),
    )
    figure.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor=PLOT_COLORS["border"],
        tickfont=dict(size=11, color="#475569"),
        title_font=dict(size=12, color="#475569"),
    )
    figure.update_yaxes(
        gridcolor=PLOT_COLORS["grid"],
        zeroline=False,
        linecolor=PLOT_COLORS["border"],
        tickfont=dict(size=11, color="#475569"),
        title_font=dict(size=12, color="#475569"),
    )
    return figure


PLOTLY_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
    "responsive": True,
}


def show_chart(figure: go.Figure, key: Optional[str] = None) -> None:
    st.plotly_chart(figure, use_container_width=True, config=PLOTLY_CONFIG, key=key)


# ---------------------------------------------------------------------------
# Guards and cached resources
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading trained model...")
def get_cached_predictor():
    from src.predictor import CreditRiskPredictor

    return CreditRiskPredictor().load()


@st.cache_data(show_spinner="Loading dataset...", persist=False)
def get_cached_dataset():
    from src.data_loader import load_dataset

    return load_dataset()


@st.cache_data(show_spinner=False, persist=False)
def get_cached_metrics() -> Dict[str, Any]:
    from src.utils import load_json

    return load_json(config.EVALUATION_METRICS_PATH)


@st.cache_data(show_spinner=False, persist=False)
def get_cached_metadata() -> Dict[str, Any]:
    from src.utils import load_json

    return load_json(config.TRAINING_METADATA_PATH)


@st.cache_data(show_spinner=False, persist=False)
def get_cached_schema() -> Dict[str, Any]:
    from src.utils import load_json

    return load_json(config.FEATURE_SCHEMA_PATH)


@st.cache_data(show_spinner=False, persist=False)
def get_cached_global_shap() -> Dict[str, Any]:
    from src.utils import load_json

    return load_json(config.GLOBAL_SHAP_PATH)


@st.cache_data(show_spinner=False, persist=False)
def get_cached_permutation_importance() -> Dict[str, Any]:
    from src.utils import load_json

    return load_json(config.PERMUTATION_IMPORTANCE_PATH)


@st.cache_data(show_spinner=False, persist=False)
def get_cached_profile() -> Dict[str, Any]:
    from src.utils import load_json

    return load_json(config.DATASET_PROFILE_PATH)


def model_status() -> Dict[str, Any]:
    from src.predictor import model_status as _model_status

    return _model_status()


def require_model():
    """Return the predictor, or render actionable retrain instructions."""
    from src.predictor import ModelNotReadyError

    status = model_status()
    if not status.get("ready"):
        notice(
            "The trained model could not be loaded, so no prediction can be made. "
            "Nothing is estimated or filled in on your behalf.",
            kind="danger",
            strong="Model unavailable.",
        )
        st.code(status.get("error") or "python -m src.train_model", language="bash")
        st.stop()
    try:
        return get_cached_predictor()
    except ModelNotReadyError as exc:  # pragma: no cover - defensive
        notice(str(exc), kind="danger", strong="Model unavailable.")
        st.stop()


def dataset_status() -> Dict[str, Any]:
    """Non-stopping dataset readiness probe (safe for dashboards)."""
    from src.data_loader import DatasetError, load_dataset

    status: Dict[str, Any] = {
        "ready": False,
        "filename": None,
        "rows": None,
        "is_synthetic": None,
        "error": None,
    }
    try:
        dataset = load_dataset()
        status.update(
            {
                "ready": True,
                "filename": dataset.provenance.filename,
                "rows": dataset.profile["rows"],
                "is_synthetic": dataset.is_synthetic,
                "positive_rate": dataset.profile["target_positive_rate"],
            }
        )
    except DatasetError as exc:
        status["error"] = str(exc)
    return status


def require_dataset():
    """Return the loaded dataset, or render a clear missing-data message."""
    from src.data_loader import DatasetError

    try:
        return get_cached_dataset()
    except DatasetError as exc:
        notice(str(exc), kind="danger", strong="Dataset unavailable.")
        st.code("Place a CSV/XLS(X)/Parquet file in the data/ directory.", language="text")
        st.stop()
