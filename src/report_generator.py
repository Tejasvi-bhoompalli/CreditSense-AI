"""Professional PDF assessment report built with ReportLab.

The report is an academic artefact. It states its own limitations on the page,
labels the dataset and its geography, and never claims to be a bank-issued
credit report or a regulated decision document.
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from reportlab.graphics.charts.barcharts import HorizontalBarChart
from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src import config
from src.risk_scoring import describe_bands
from src.utils import feature_label, format_currency_twd, slugify

NAVY = colors.HexColor("#101828")
NAVY_2 = colors.HexColor("#172554")
INDIGO = colors.HexColor("#4F46E5")
BLUE = colors.HexColor("#3B82F6")
TEAL = colors.HexColor("#14B8A6")
TEXT = colors.HexColor("#172033")
MUTED = colors.HexColor("#64748B")
BORDER = colors.HexColor("#E2E8F0")
SURFACE = colors.HexColor("#F5F7FB")
LOW = colors.HexColor("#16A34A")
MEDIUM = colors.HexColor("#D97706")
HIGH = colors.HexColor("#DC2626")

RISK_COLOR_MAP = {"LOW RISK": LOW, "MEDIUM RISK": MEDIUM, "HIGH RISK": HIGH}


class ReportError(RuntimeError):
    """Raised when a PDF cannot be produced."""


def _styles() -> Dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "CSATitle", parent=base["Title"], fontName="Helvetica-Bold", fontSize=20,
            leading=24, textColor=NAVY, alignment=TA_LEFT, spaceAfter=2,
        ),
        "subtitle": ParagraphStyle(
            "CSASubtitle", parent=base["Normal"], fontName="Helvetica", fontSize=9.5,
            leading=13, textColor=MUTED, spaceAfter=10,
        ),
        "h1": ParagraphStyle(
            "CSAH1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=13,
            leading=16, textColor=NAVY, spaceBefore=14, spaceAfter=6,
        ),
        "h2": ParagraphStyle(
            "CSAH2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=10.5,
            leading=13, textColor=NAVY_2, spaceBefore=10, spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "CSABody", parent=base["BodyText"], fontName="Helvetica", fontSize=9,
            leading=13, textColor=TEXT, alignment=TA_JUSTIFY, spaceAfter=5,
        ),
        "small": ParagraphStyle(
            "CSASmall", parent=base["BodyText"], fontName="Helvetica", fontSize=7.8,
            leading=10.5, textColor=MUTED, alignment=TA_JUSTIFY, spaceAfter=4,
        ),
        "cell": ParagraphStyle(
            "CSACell", parent=base["BodyText"], fontName="Helvetica", fontSize=8.4,
            leading=11, textColor=TEXT, spaceAfter=0,
        ),
        "cellbold": ParagraphStyle(
            "CSACellBold", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=8.4,
            leading=11, textColor=NAVY, spaceAfter=0,
        ),
        "bullet": ParagraphStyle(
            "CSABullet", parent=base["BodyText"], fontName="Helvetica", fontSize=8.8,
            leading=12, textColor=TEXT, leftIndent=10, bulletIndent=2, spaceAfter=3,
        ),
    }


def _header_footer(canvas, doc) -> None:
    canvas.saveState()
    width, height = A4
    canvas.setFillColor(NAVY)
    canvas.rect(0, height - 18 * mm, width, 18 * mm, stroke=0, fill=1)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 11)
    canvas.drawString(18 * mm, height - 12 * mm, "CreditSense AI")
    canvas.setFont("Helvetica", 7.6)
    canvas.drawRightString(
        width - 18 * mm,
        height - 12 * mm,
        "Explainable Intelligence for Smarter Credit Decisions",
    )
    canvas.setStrokeColor(BORDER)
    canvas.setLineWidth(0.6)
    canvas.line(18 * mm, 16 * mm, width - 18 * mm, 16 * mm)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 7.2)
    canvas.drawString(
        18 * mm, 11.5 * mm, "Academic prototype - not a bank-issued credit report"
    )
    canvas.drawRightString(width - 18 * mm, 11.5 * mm, f"Page {doc.page}")
    canvas.restoreState()


def _risk_badge_drawing(probability: float, category: str, risk_score: float) -> Drawing:
    width, height = 500, 74
    drawing = Drawing(width, height)
    drawing.add(Rect(0, 44, width, 26, fillColor=SURFACE, strokeColor=BORDER, strokeWidth=0.7))

    label = String(10, 52, f"Risk score {risk_score:.1f} / 100", fontName="Helvetica-Bold",
                   fontSize=11, fillColor=NAVY)
    drawing.add(label)
    badge_text = String(
        width - 10, 52, f" {category} ", fontName="Helvetica-Bold", fontSize=11,
        fillColor=colors.white,
    )
    category_color = RISK_COLOR_MAP.get(category, MUTED)
    badge_text.textAnchor = "end"
    drawing.add(badge_text)
    drawing.add(
        Rect(width - 152, 47, 142, 19, fillColor=category_color, strokeColor=None)
    )
    # overlay the text on top of the badge
    overlay = String(
        width - 155, 52.5, category, fontName="Helvetica-Bold", fontSize=10,
        fillColor=colors.white, textAnchor="end",
    )
    drawing.add(overlay)

    bar_y = 20
    drawing.add(Rect(0, bar_y, width, 12, fillColor=colors.HexColor("#EEF2F7"),
                     strokeColor=BORDER, strokeWidth=0.5))
    drawing.add(Rect(0, bar_y, width * 0.30, 12, fillColor=LOW, strokeColor=None))
    drawing.add(Rect(width * 0.30, bar_y, width * 0.30, 12, fillColor=MEDIUM, strokeColor=None))
    drawing.add(Rect(width * 0.60, bar_y, width * 0.40, 12, fillColor=HIGH, strokeColor=None))
    marker_x = min(max(probability, 0.0), 1.0) * width
    drawing.add(Line(marker_x, bar_y - 4, marker_x, bar_y + 16, strokeColor=NAVY, strokeWidth=1.8))
    drawing.add(String(0, 6, "0%", fontName="Helvetica", fontSize=7, fillColor=MUTED))
    drawing.add(String(width / 2 - 8, 6, f"Predicted default probability {probability * 100:.2f}%",
                       fontName="Helvetica", fontSize=7, fillColor=MUTED))
    drawing.add(String(width - 60, 6, "Risk bands: 30 / 60", fontName="Helvetica", fontSize=7,
                       fillColor=MUTED))
    return drawing


def _contribution_chart_drawing(explanation) -> Optional[Drawing]:
    if explanation is None:
        return None
    ranked = explanation.ranked(config.TOP_FACTORS)
    if not ranked:
        return None
    labels = [item.label[:34] for item in ranked][::-1]
    values = [item.contribution for item in ranked][::-1]

    drawing = Drawing(500, 24 + 16 * len(labels))
    chart = HorizontalBarChart()
    chart.x = 172
    chart.y = 24
    chart.height = 16 * len(labels)
    chart.width = 310
    chart.data = [values]
    chart.categoryAxis.categoryNames = labels
    chart.categoryAxis.labels.fontName = "Helvetica"
    chart.categoryAxis.labels.fontSize = 7
    chart.valueAxis.labels.fontName = "Helvetica"
    chart.valueAxis.labels.fontSize = 7
    chart.valueAxis.valueMin = min(values + [0.0])
    chart.valueAxis.valueMax = max(values + [0.0]) or 1.0
    chart.bars[0].strokeColor = None
    chart.barWidth = 8
    chart.bars[0].fillColor = INDIGO
    chart.bars[0].getProperties()
    drawing.add(chart)
    return drawing


def _table(rows: List[List[Any]], widths: List[float], header: bool = True) -> Table:
    table = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    style = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.2),
        ("TEXTCOLOR", (0, 0), (-1, -1), TEXT),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("GRID", (0, 0), (-1, -1), 0.4, BORDER),
    ]
    if header:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]
        for row_index in range(1, len(rows)):
            if row_index % 2 == 0:
                style.append(("BACKGROUND", (0, row_index), (-1, row_index), SURFACE))
    table.setStyle(TableStyle(style))
    return table


def _kv_rows(pairs: List[tuple]) -> List[List[Any]]:
    return [["Field", "Value"]] + [
        [Paragraph(str(key), _styles()["cellbold"]), Paragraph(str(value), _styles()["cell"])]
        for key, value in pairs
    ]


def build_assessment_report(context: Dict[str, Any]) -> bytes:
    """Render the complete assessment report and return the PDF bytes.

    ``context`` keys (all optional except inputs/probability/risk):
      assessment_id, generated_at, inputs, derived, probability, risk_score,
      risk_category, recommendation, explanation, narrative, narrative_source,
      model_name, model_version, dataset, metrics, scenario, is_demo, validation
    """
    styles = _styles()
    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=24 * mm,
        bottomMargin=20 * mm,
        title="CreditSense AI - Credit Risk Assessment Report",
        author="CreditSense AI",
        subject="Explainable credit-risk assessment (academic prototype)",
    )

    story: List[Any] = []
    identifier = context.get("assessment_id", "unassigned")
    generated_at = context.get("generated_at") or datetime.now(timezone.utc).replace(
        microsecond=0
    ).isoformat()
    probability = float(context.get("probability", 0.0))
    risk_score = float(context.get("risk_score", probability * 100))
    risk_category = context.get("risk_category", "UNKNOWN")
    recommendation = context.get("recommendation")
    explanation = context.get("explanation")
    inputs: Dict[str, Any] = dict(context.get("inputs") or {})
    derived: Dict[str, Any] = dict(context.get("derived") or {})
    dataset = dict(context.get("dataset") or {})
    metrics = dict(context.get("metrics") or {})

    # ---- Title block -----------------------------------------------------
    story.append(Paragraph("Credit Risk Assessment Report", styles["title"]))
    story.append(
        Paragraph(
            "CreditSense AI - Explainable AI-Driven Credit Risk Assessment Framework "
            "for Intelligent Loan Decision Support",
            styles["subtitle"],
        )
    )
    story.append(_risk_badge_drawing(probability, risk_category, risk_score))
    story.append(Spacer(1, 6))

    if context.get("is_demo"):
        story.append(
            Paragraph(
                "<b>DEMONSTRATION RECORD.</b> This assessment was saved with an explicitly "
                "labelled demonstration reference. It does not describe a real applicant.",
                styles["small"],
            )
        )

    story.append(
        _table(
            _kv_rows(
                [
                    ("Assessment ID", identifier),
                    ("Generated (UTC)", generated_at),
                    ("Model", f"{context.get('model_name', 'unknown')} "
                              f"v{context.get('model_version', config.MODEL_VERSION)}"),
                    ("Predicted default probability", f"{probability * 100:.2f}%"),
                    ("Risk score", f"{risk_score:.1f} / 100"),
                    ("Risk category", risk_category),
                    (
                        "Recommendation",
                        getattr(recommendation, "label", context.get("recommendation_label", "n/a")),
                    ),
                ]
            ),
            [52 * mm, 118 * mm],
        )
    )

    # ---- Dataset attribution --------------------------------------------
    story.append(Paragraph("Dataset and attribution", styles["h1"]))
    story.append(
        Paragraph(
            "Every number in this report was produced by the model described below, trained "
            "on the dataset named below. No value in this report is illustrative or invented. "
            "Model metrics are measured on a held-out test split that was never used for "
            "model selection or calibration.",
            styles["body"],
        )
    )
    dataset_rows = [
        ("Dataset", dataset.get("name", config.DATASET_INFO["name"])),
        ("Publisher", dataset.get("publisher", config.DATASET_INFO["publisher"])),
        ("Authors", dataset.get("authors", config.DATASET_INFO["authors"])),
        ("Year", str(config.DATASET_INFO["publication_year"])),
        ("DOI", dataset.get("doi", config.DATASET_INFO["doi"])),
        ("Licence", dataset.get("license", config.DATASET_INFO["license"])),
        ("Geography", dataset.get("geography", config.DATASET_INFO["geography"])),
        ("Observation period", dataset.get("observation_period",
                                           config.DATASET_INFO["observation_period"])),
        ("Currency", dataset.get("currency", config.DATASET_INFO["currency"])),
        ("Source file", f"{dataset.get('filename', 'n/a')} "
                        f"(SHA-256 {str(dataset.get('sha256', 'n/a'))[:16]}...)"),
        ("Records", f"{dataset.get('rows', 'n/a')} rows"),
    ]
    if dataset.get("is_synthetic"):
        dataset_rows.append(("Data type", "SYNTHETIC - generated demonstration data"))
    story.append(_table(_kv_rows(dataset_rows), [52 * mm, 118 * mm]))
    citation = dataset.get("citation", config.DATASET_INFO["citation"])
    story.append(Spacer(1, 4))
    story.append(Paragraph(f"Required attribution: {citation}", styles["small"]))

    # ---- Geographic scope disclosure ------------------------------------
    story.append(Paragraph("Geographic scope - important", styles["h1"]))
    story.append(Paragraph(config.GEOGRAPHIC_DISCLOSURE, styles["body"]))

    # ---- Applicant input summary ----------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("Applicant input summary", styles["h1"]))
    story.append(
        Paragraph(
            "Only the fields listed here were used. Fields such as annual income, employment "
            "status, loan purpose or an external credit score are not present in this dataset "
            "and are therefore marked unavailable rather than estimated.",
            styles["small"],
        )
    )
    input_rows = [["Field", "Submitted value"]]
    for feature in config.USER_INPUT_FEATURES:
        if feature not in inputs:
            continue
        value = inputs[feature]
        if feature in config.CATEGORICAL_FEATURES:
            display = config.EDUCATION_DISPLAY.get(str(value), str(value))
        elif feature in config.REPAY_STATUS_COLUMNS:
            display = f"{int(float(value))} - {config.REPAY_STATUS_CODES.get(int(float(value)), '')}"
        else:
            display = f"NT$ {float(value):,.0f}"
        input_rows.append([Paragraph(feature_label(feature), styles["cellbold"]),
                           Paragraph(display, styles["cell"])])
    story.append(_table(input_rows, [95 * mm, 75 * mm]))

    if derived:
        story.append(Paragraph("Derived indicators (formulas documented)", styles["h2"]))
        derived_rows = [["Derived indicator", "Value"]]
        for feature in config.DERIVED_FEATURES:
            if feature not in derived:
                continue
            value = float(derived[feature])
            if feature in {"credit_utilization", "repayment_ratio"}:
                display = f"{value:.3f}"
            elif feature.startswith("num_") or feature == "max_repayment_delay":
                display = f"{value:.0f}"
            else:
                display = f"NT$ {value:,.0f}"
            derived_rows.append([Paragraph(feature_label(feature), styles["cellbold"]),
                                 Paragraph(display, styles["cell"])])
        story.append(_table(derived_rows, [95 * mm, 75 * mm]))

    # ---- Recommendation -------------------------------------------------
    if recommendation is not None or context.get("recommendation_label"):
        story.append(PageBreak())
        story.append(Paragraph("Recommendation", styles["h1"]))
        label = getattr(recommendation, "label", context.get("recommendation_label", ""))
        headline = getattr(recommendation, "headline", "")
        story.append(Paragraph(f"<b>{label}</b>", styles["body"]))
        if headline:
            story.append(Paragraph(headline, styles["body"]))
        sections = [
            ("Main reasons", getattr(recommendation, "reasons", [])),
            ("Supporting indicators", getattr(recommendation, "supporting_indicators", [])),
            ("Key concerns", getattr(recommendation, "concerns", [])),
            ("Suggested next steps", getattr(recommendation, "next_steps", [])),
            ("Rules applied", getattr(recommendation, "rules_applied", [])),
        ]
        for title, items in sections:
            if not items:
                continue
            story.append(Paragraph(title, styles["h2"]))
            for item in items:
                story.append(Paragraph(f"- {item}", styles["bullet"]))
        disclaimer = getattr(recommendation, "disclaimer", "")
        if disclaimer:
            story.append(Paragraph(f"<i>{disclaimer}</i>", styles["small"]))

    # ---- Explainability -------------------------------------------------
    story.append(Paragraph("Explanation of this prediction", styles["h1"]))
    if explanation is not None:
        story.append(
            Paragraph(
                f"Explanation method: <b>{explanation.method_label}</b>. "
                f"The explanation was computed on the same fitted model that produced the "
                f"prediction. Contributions are expressed in the base model's raw output "
                f"units (log-odds); positive values push the predicted risk upward.",
                styles["body"],
            )
        )
        chart = _contribution_chart_drawing(explanation)
        if chart is not None:
            story.append(KeepTogether([chart]))
            story.append(Spacer(1, 4))
        contributing_rows = [["Feature", "Value", "Contribution", "Direction"]]
        for item in explanation.ranked(config.TOP_FACTORS):
            contributing_rows.append(
                [
                    Paragraph(item.label, styles["cell"]),
                    Paragraph(str(item.display_value), styles["cell"]),
                    Paragraph(f"{item.contribution:+.4f}", styles["cell"]),
                    Paragraph("Increases risk" if item.contribution > 0 else "Reduces risk",
                              styles["cell"]),
                ]
            )
        story.append(_table(contributing_rows, [70 * mm, 40 * mm, 28 * mm, 32 * mm]))
        from src.explainability import SHAP_CAVEAT, narrate_local

        story.append(Spacer(1, 4))
        story.append(Paragraph("Plain-English summary", styles["h2"]))
        for sentence in narrate_local(explanation):
            story.append(Paragraph(f"- {sentence}", styles["bullet"]))
        story.append(Paragraph(SHAP_CAVEAT, styles["small"]))
    else:
        story.append(
            Paragraph(
                "No explanation was attached to this assessment record. Open the "
                "Explainable AI page for this profile to generate one.",
                styles["small"],
            )
        )

    # ---- Narrative ------------------------------------------------------
    narrative = context.get("narrative")
    if narrative:
        story.append(Paragraph("Narrative summary", styles["h1"]))
        source = context.get("narrative_source", "template")
        story.append(
            Paragraph(
                f"<i>Generated by: {source}. The narrative renders measured model outputs; it "
                f"does not introduce new numbers or make decisions.</i>",
                styles["small"],
            )
        )
        for paragraph in str(narrative).split("\n\n"):
            story.append(Paragraph(paragraph.replace("\n", "<br/>"), styles["body"]))

    # ---- What-if comparison --------------------------------------------
    scenario = context.get("scenario")
    if scenario is not None:
        story.append(PageBreak())
        story.append(Paragraph("What-if scenario comparison", styles["h1"]))
        story.append(Paragraph(scenario.narrative, styles["body"]))
        scenario_rows = [
            ["Scenario", "Default probability", "Risk score", "Risk category"],
            [
                "Baseline",
                f"{scenario.baseline_probability * 100:.2f}%",
                f"{scenario.baseline_risk_score:.1f}",
                scenario.baseline_category,
            ],
            [
                "Alternative",
                f"{scenario.alternative_probability * 100:.2f}%",
                f"{scenario.alternative_risk_score:.1f}",
                scenario.alternative_category,
            ],
            [
                "Difference",
                f"{scenario.probability_change_points:+.2f} pp",
                f"{scenario.risk_score_change:+.1f}",
                scenario.category_transition,
            ],
        ]
        story.append(_table(scenario_rows, [30 * mm, 42 * mm, 30 * mm, 68 * mm]))
        if scenario.changed_features:
            story.append(Paragraph("Adjusted inputs", styles["h2"]))
            changed_rows = [["Input", "Baseline", "Alternative"]]
            for item in scenario.changed_features:
                changed_rows.append(
                    [
                        Paragraph(item.label, styles["cell"]),
                        Paragraph(item.display_before, styles["cell"]),
                        Paragraph(item.display_after, styles["cell"]),
                    ]
                )
            story.append(_table(changed_rows, [70 * mm, 50 * mm, 50 * mm]))
        for caveat in scenario.caveats:
            story.append(Paragraph(f"- {caveat}", styles["small"]))

    # ---- Model evaluation context ---------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("Model evaluation context", styles["h1"]))
    if metrics:
        story.append(
            _table(
                _kv_rows(
                    [
                        ("Selection criterion", metrics.get("selection_criterion", "n/a")),
                        ("Train ROC-AUC", f"{metrics.get('train_roc_auc', float('nan')):.4f}"),
                        ("Test ROC-AUC", f"{metrics.get('test_roc_auc', float('nan')):.4f}"),
                        ("Test PR-AUC", f"{metrics.get('test_pr_auc', float('nan')):.4f}"),
                        ("Test accuracy", f"{metrics.get('test_accuracy', float('nan')):.4f}"),
                        ("Test precision", f"{metrics.get('test_precision', float('nan')):.4f}"),
                        ("Test recall", f"{metrics.get('test_recall', float('nan')):.4f}"),
                        ("Test F1", f"{metrics.get('test_f1', float('nan')):.4f}"),
                        ("Brier score (test)", f"{metrics.get('brier_score', float('nan')):.4f}"),
                        (
                            "Test split size",
                            f"{metrics.get('test_n', 0)} records "
                            f"({metrics.get('test_positive_rate', float('nan')) * 100:.2f}% default rate)",
                        ),
                    ]
                ),
                [52 * mm, 118 * mm],
            )
        )
        tradeoff_note = metrics.get("interpretation")
        if tradeoff_note:
            story.append(Spacer(1, 4))
            story.append(Paragraph(tradeoff_note, styles["small"]))
    else:
        story.append(
            Paragraph(
                "Model metrics were not available when this report was generated. Run "
                "`python -m src.train_model` to produce them.",
                styles["small"],
            )
        )

    # ---- Limitations, fairness, disclaimer ------------------------------
    story.append(Paragraph("Data limitations", styles["h1"]))
    for limitation in [
        config.GEOGRAPHIC_DISCLOSURE,
        "The source dataset is a single historical sample of credit-card clients of one "
        "Taiwanese bank over April-September 2005, with the default outcome observed for "
        "October 2005. Behaviour, products and economic conditions differ today and elsewhere.",
        "The dataset contains no income, employment, loan purpose, interest rate or external "
        "credit-score field, so those canonical risk factors could not be modelled at all.",
        "The positive class is a minority (22.12% of records). Accuracy alone is therefore "
        "misleading, which is why ROC-AUC, PR-AUC, recall and the full confusion matrix are "
        "reported together.",
        "The source table documents repayment-status codes -1 and 1..9, while the distributed "
        "extract also contains -2 and 0. Those codes are documented here rather than silently "
        "recoded. Statement balances can be negative (credit in the client's favour) and are "
        "preserved rather than clipped to zero.",
        "A predicted probability is an estimate for this dataset; it is not a guarantee and "
        "should not be read as the exact real-world default rate of an individual.",
    ]:
        story.append(Paragraph(f"- {limitation}", styles["bullet"]))

    story.append(Paragraph("Fairness considerations", styles["h1"]))
    for note in [
        "Sex and marital status are excluded from the model's feature set. Age and education "
        "are included but are disclosed as potential proxy variables; excluding one attribute "
        "does not remove bias that other correlated features may carry.",
        "Group-level behaviour is described on the evaluation page of the application with "
        "sample sizes, and is presented as a descriptive diagnostic only.",
        "No fairness audit, regulatory compliance claim or legal conclusion is made here. "
        "Any real deployment would require jurisdiction-specific legal review and validation "
        "on locally representative, lawfully obtained data.",
    ]:
        story.append(Paragraph(f"- {note}", styles["bullet"]))

    story.append(Paragraph("Risk bands used in this report", styles["h1"]))
    story.append(
        Paragraph(
            f"{describe_bands()}. These are prototype presentation thresholds of this academic "
            "project. They are NOT official banking thresholds and no regulatory threshold is "
            "claimed. The underlying value remains a probability; the score is only a linear "
            "presentation of it.",
            styles["small"],
        )
    )

    story.append(Paragraph("Disclaimer", styles["h1"]))
    story.append(
        Paragraph(
            f"<b>{config.APP_DISCLAIMER}</b>",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "This document was generated locally by the open-source CreditSense AI prototype "
            "for academic demonstration and decision-support research. It is not a bank-issued "
            "credit report, not a credit bureau report, and it does not create, approve, "
            "decline or price any credit facility.",
            styles["small"],
        )
    )

    try:
        document.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    except Exception as exc:  # pragma: no cover - reportlab internals
        raise ReportError(f"PDF generation failed: {type(exc).__name__}: {exc}") from exc

    pdf_bytes = buffer.getvalue()
    buffer.close()
    if not pdf_bytes.startswith(b"%PDF"):
        raise ReportError("PDF generation produced an invalid document.")
    return pdf_bytes


def metrics_context(
    metrics_payload: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Flatten the stored evaluation artifacts into the report's metrics block."""
    from src.utils import load_json

    payload = metrics_payload if metrics_payload is not None else load_json(
        config.EVALUATION_METRICS_PATH
    )
    meta = metadata if metadata is not None else load_json(config.TRAINING_METADATA_PATH)
    if not payload:
        return {}

    selected = payload.get("selected", {})
    test = selected.get("test", {})
    train = selected.get("train", {})
    validation = selected.get("validation", {})
    calibration = payload.get("calibration_quality", {})
    overfitting = payload.get("overfitting", {})

    if not test:
        return {}

    return {
        "selection_criterion": payload.get("selection_criterion", ""),
        "model": payload.get("selected_model", ""),
        "train_roc_auc": train.get("roc_auc"),
        "validation_roc_auc": validation.get("roc_auc"),
        "test_roc_auc": test.get("roc_auc"),
        "test_pr_auc": test.get("pr_auc"),
        "test_accuracy": test.get("accuracy"),
        "test_precision": test.get("precision"),
        "test_recall": test.get("recall"),
        "test_f1": test.get("f1"),
        "brier_score": test.get("brier_score"),
        "test_n": test.get("n"),
        "test_positive_rate": test.get("positive_rate"),
        "test_false_positives": test.get("false_positives"),
        "test_false_negatives": test.get("false_negatives"),
        "interpretation": (
            f"Measured on {test.get('n')} held-out test records that were never used for model "
            f"selection or calibration. Calibration reduced the Brier score from "
            f"{calibration.get('brier_score_uncalibrated', float('nan')):.4f} to "
            f"{calibration.get('brier_score_calibrated', float('nan')):.4f} and the expected "
            f"calibration error from "
            f"{calibration.get('expected_calibration_error_uncalibrated', float('nan')):.4f} to "
            f"{calibration.get('expected_calibration_error_calibrated', float('nan')):.4f}. "
            f"{overfitting.get('warning', '')} Those figures describe this dataset only; they "
            f"are not a claim about lending performance anywhere else."
        ),
        "training_split_size": meta.get("split_sizes", {}).get("train"),
        "validation_split_size": meta.get("split_sizes", {}).get("validation"),
    }


def report_filename(context: Dict[str, Any]) -> str:
    identifier = slugify(str(context.get("assessment_id", "assessment")), 40)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"CreditSenseAI-{identifier}-{stamp}.pdf"
