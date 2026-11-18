"""Render a concise, evidence-oriented PDF from an integrated result."""
from __future__ import annotations

from html import escape
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from deep_shield.intelligence.schema import IntelligenceResult


def _text(value: object | None) -> str:
    return escape("" if value is None else str(value)).replace("\n", "<br/>")


def render_analysis_report(result: IntelligenceResult, job_id: str) -> bytes:
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer, pagesize=A4, rightMargin=16 * mm, leftMargin=16 * mm,
        topMargin=17 * mm, bottomMargin=17 * mm,
        title=f"DEEP SHIELD analysis {result.media_id}",
        author="DEEP SHIELD",
    )
    base = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DS_Title", parent=base["Title"], alignment=TA_CENTER,
        textColor=colors.HexColor("#17324d"), spaceAfter=5 * mm,
    )
    small = ParagraphStyle("DS_Small", parent=base["BodyText"], fontSize=8, leading=10)
    body = ParagraphStyle("DS_Body", parent=base["BodyText"], fontSize=9, leading=12, spaceAfter=2 * mm)
    section = ParagraphStyle("DS_Section", parent=base["Heading2"], textColor=colors.HexColor("#176b87"))
    story = [
        Paragraph("DEEP SHIELD - Analysis report", title_style),
        Paragraph(f"Job: {_text(job_id)}", body),
        Paragraph(f"Media ID: {_text(result.media_id)}", body),
        Paragraph(f"SHA-256: {_text(result.media_sha256)}", small),
        Paragraph(f"Source: {_text(result.module_results[0].source_name if result.module_results else '')}", body),
        Spacer(1, 4 * mm),
        Paragraph("Module summary", section),
    ]

    summary_rows = [["Module", "Status", "Findings", "Tasks"]]
    for module in result.module_results:
        summary_rows.append([
            Paragraph(_text(module.module), small),
            _text(module.status),
            str(len(module.findings)),
            Paragraph(_text(", ".join(item.task for item in module.task_summaries)), small),
        ])
    summary = Table(summary_rows, colWidths=[39 * mm, 22 * mm, 19 * mm, 94 * mm], repeatRows=1)
    summary.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17324d")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#c7d2dc")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f6f8")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.extend([summary, Spacer(1, 5 * mm), Paragraph("Linked evidence groups", section)])
    if result.groups:
        group_rows = [["Group", "Modules", "Finding IDs"]]
        for group in result.groups:
            group_rows.append([
                _text(group.group_id),
                Paragraph(_text(", ".join(group.modules)), small),
                Paragraph(_text(", ".join(group.finding_ids)), small),
            ])
        groups = Table(group_rows, colWidths=[31 * mm, 46 * mm, 97 * mm], repeatRows=1)
        groups.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#176b87")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#c7d2dc")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(groups)
    else:
        story.append(Paragraph("No cross-module links were produced.", body))

    findings = [finding for module in result.module_results for finding in module.findings]
    story.extend([Spacer(1, 5 * mm), Paragraph("Findings", section)])
    finding_rows = [["Module / task", "Observation", "Time", "Score type / value", "Evidence"]]
    for finding in findings[:300]:
        time_text = ""
        if finding.time_start_s is not None or finding.timestamp_s is not None:
            start = finding.time_start_s if finding.time_start_s is not None else finding.timestamp_s
            end = finding.time_end_s if finding.time_end_s is not None else finding.timestamp_s
            time_text = f"{start:.2f}-{end:.2f}s" if start is not None and end is not None else ""
        score = "-" if finding.score is None else f"{finding.score:.4f}"
        score_type = getattr(finding.score_type, "value", finding.score_type)
        refs = ", ".join(item.path for item in finding.evidence)
        finding_rows.append([
            Paragraph(f"{_text(finding.module)}<br/>{_text(finding.task)}", small),
            Paragraph(_text(finding.classification), small),
            _text(time_text),
            Paragraph(_text(f"{score_type}: {score}"), small),
            Paragraph(_text(refs), small),
        ])
    finding_table = Table(
        finding_rows, colWidths=[34 * mm, 39 * mm, 22 * mm, 36 * mm, 43 * mm], repeatRows=1
    )
    finding_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17324d")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c7d2dc")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f6f8")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(finding_table)
    if len(findings) > 300:
        story.append(Paragraph(f"Report lists the first 300 of {len(findings)} findings. Retrieve the complete JSON result from the API.", small))
    story.extend([
        Spacer(1, 5 * mm),
        Paragraph("Limitations and review notice", section),
        Paragraph(_text(result.disclaimer), body),
    ])
    for module in result.module_results:
        for limitation in module.findings[0].limitations if module.findings else []:
            story.append(Paragraph(f"- {_text(limitation)}", small))

    def page_number(canvas, _doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#516171"))
        canvas.drawString(16 * mm, 9 * mm, "Advisory output for human review; scores remain engine-specific.")
        canvas.drawRightString(A4[0] - 16 * mm, 9 * mm, f"Page {_doc.page}")
        canvas.restoreState()

    document.build(story, onFirstPage=page_number, onLaterPages=page_number)
    return buffer.getvalue()
