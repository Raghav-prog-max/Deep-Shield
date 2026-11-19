"""Build the shareable, evidence-qualified Deep Shield project status PDF."""
from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "Deep_Shield_Project_Status_2026-11-19.pdf"
NAVY = colors.HexColor("#142d3a")
TEAL = colors.HexColor("#137a70")
INK = colors.HexColor("#23343c")
MUTED = colors.HexColor("#62747c")
PALE = colors.HexColor("#eef5f5")
GRID = colors.HexColor("#d6e0e3")
GREEN = colors.HexColor("#e5f3ec")
AMBER = colors.HexColor("#fff3d9")
RED = colors.HexColor("#fde9e6")


def header_footer(canvas, doc):
    canvas.saveState()
    width, height = A4
    canvas.setFillColor(NAVY)
    canvas.rect(0, height - 12 * mm, width, 12 * mm, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 8)
    canvas.drawString(18 * mm, height - 7.8 * mm, "DEEP SHIELD  /  PROJECT STATUS")
    canvas.drawRightString(width - 18 * mm, height - 7.8 * mm, "19 NOV 2026")
    canvas.setStrokeColor(GRID)
    canvas.line(18 * mm, 14 * mm, width - 18 * mm, 14 * mm)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 7.5)
    canvas.drawString(18 * mm, 9 * mm, "github.com/Raghav-prog-max/Deep-Shield")
    canvas.drawRightString(width - 18 * mm, 9 * mm, f"{doc.page}")
    canvas.restoreState()


def build_pdf(out_path: Path = OUT) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=19 * mm,
        bottomMargin=19 * mm,
        title="Deep Shield - Project Status - 19 Nov 2026",
        author="Deep Shield Team",
    )
    base = getSampleStyleSheet()
    title = ParagraphStyle("TitleDS", parent=base["Title"], fontName="Helvetica-Bold", fontSize=26, leading=30, alignment=TA_LEFT, textColor=NAVY, spaceAfter=2 * mm)
    subtitle = ParagraphStyle("SubDS", parent=base["Normal"], fontName="Helvetica", fontSize=10, leading=14, textColor=MUTED, spaceAfter=4 * mm)
    h2 = ParagraphStyle("H2DS", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=15, leading=18, textColor=NAVY, spaceBefore=3 * mm, spaceAfter=2 * mm)
    h3 = ParagraphStyle("H3DS", parent=base["Heading3"], fontName="Helvetica-Bold", fontSize=10, leading=13, textColor=TEAL, spaceBefore=2 * mm, spaceAfter=1.5 * mm)
    body = ParagraphStyle("BodyDS", parent=base["BodyText"], fontName="Helvetica", fontSize=8.6, leading=12, textColor=INK, spaceAfter=2 * mm)
    small = ParagraphStyle("SmallDS", parent=body, fontSize=7.5, leading=10)
    cell = ParagraphStyle("CellDS", parent=small, spaceAfter=0)
    cell_bold = ParagraphStyle("CellBoldDS", parent=cell, fontName="Helvetica-Bold")
    cell_head = ParagraphStyle("CellHeadDS", parent=cell, fontName="Helvetica-Bold", textColor=colors.white)
    banner = ParagraphStyle("BannerDS", parent=body, alignment=TA_CENTER, fontName="Helvetica-Bold", fontSize=8.2, leading=11, textColor=NAVY, spaceAfter=0)
    flow = ParagraphStyle("FlowDS", parent=body, alignment=TA_CENTER, fontName="Helvetica-Bold", fontSize=8, leading=10, spaceAfter=0)
    story = []

    story.append(Paragraph("Deep Shield", title))
    story.append(Paragraph("Project status, verification, and remaining production work", subtitle))
    story.append(Paragraph("Reviewed 19 Nov 2026 | Repository: Raghav-prog-max/Deep-Shield | Commit: 6206fd990feb0fc35dd6311a278c8acd31bb19a1", small))
    story.append(Spacer(1, 2 * mm))

    summary = Table([[Paragraph("IMPLEMENTED", banner), Paragraph("FRONTEND LIVE", banner), Paragraph("HOSTED BACKEND NOT CONNECTED", banner)]], colWidths=[55 * mm, 55 * mm, 64 * mm])
    summary.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), GREEN), ("BACKGROUND", (1, 0), (1, 0), GREEN), ("BACKGROUND", (2, 0), (2, 0), AMBER),
        ("BOX", (0, 0), (-1, -1), 0.6, GRID), ("INNERGRID", (0, 0), (-1, -1), 0.6, GRID),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(summary)
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph("Flowchart and completion", h2))

    flow_rows = [
        [Paragraph("IMAGE / VIDEO / WEBCAM", flow)],
        [Paragraph("M01 Threat  |  M02 Integrity  |  M03 Events  |  M04 Forensics", flow)],
        [Paragraph("Cross-module evidence linker (scores remain separate)", flow)],
        [Paragraph("FastAPI + durable SQLite worker + PDF reports + retention", flow)],
        [Paragraph("Next.js landing + review workspace", flow)],
        [Paragraph("Vercel pages return 200  |  API health on Vercel returns 404", flow)],
        [Paragraph("REMAINING: deploy backend -> configure Vercel -> verify real media jobs", flow)],
    ]
    flow_table = Table(flow_rows, colWidths=[174 * mm])
    flow_styles = [
        ("BOX", (0, 0), (-1, -1), 0.6, GRID), ("INNERGRID", (0, 0), (-1, -1), 0.6, GRID),
        ("BACKGROUND", (0, 0), (0, 4), PALE), ("BACKGROUND", (0, 5), (0, 5), AMBER), ("BACKGROUND", (0, 6), (0, 6), RED),
        ("TOPPADDING", (0, 0), (-1, -1), 4.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 4.2),
    ]
    flow_styles += [("TEXTCOLOR", (0, row), (0, row), TEAL) for row in range(4)]
    flow_styles.append(("TEXTCOLOR", (0, 6), (0, 6), colors.HexColor("#963d31")))
    flow_table.setStyle(TableStyle(flow_styles))
    story.append(flow_table)
    story.append(Spacer(1, 3 * mm))

    story.append(Paragraph("Status by stage", h2))
    status_rows = [
        [Paragraph("STAGE", cell_head), Paragraph("STATUS", cell_head), Paragraph("WHAT THE EVIDENCE SAYS", cell_head)],
        [Paragraph("M01-M04 engines and assets", cell_bold), Paragraph("Implemented", cell_bold), Paragraph("Code and trained M02/M03/M04 assets are present. M01 YOLOS weights are fetched by setup/CI.", cell)],
        [Paragraph("API, worker, reports", cell_bold), Paragraph("Implemented; local run recorded", cell_bold), Paragraph("Previous local integration recorded lease recovery, 11 evidence files, PDF output, and retention cleanup.", cell)],
        [Paragraph("Frontend", cell_bold), Paragraph("Hosted", cell_bold), Paragraph("On 19 Nov: landing HTTP 200 (95,349 bytes); /review HTTP 200 (10,130 bytes).", cell)],
        [Paragraph("GitHub Actions", cell_bold), Paragraph("Passed", cell_bold), Paragraph("Current commit run 44453823617 passed backend CI and frontend typecheck/build. Backend CI uses M02 stub and excludes deep-model media tests.", cell)],
        [Paragraph("Evaluation", cell_bold), Paragraph("Reports present", cell_bold), Paragraph("Benchmarks are documented in reports but were not independently rerun for this status review.", cell)],
        [Paragraph("Cloud packaging", cell_bold), Paragraph("Ready to deploy", cell_bold), Paragraph("Railway, Render, Hugging Face Spaces, and Docker Compose configuration is present; no public backend is verified.", cell)],
        [Paragraph("Hosted end-to-end", cell_bold), Paragraph("Not complete", cell_bold), Paragraph("Same-origin /api/v1/m01/health returns HTTP 404. Hosted image/video analysis is not verified.", cell)],
    ]
    matrix = Table(status_rows, colWidths=[39 * mm, 32 * mm, 103 * mm], repeatRows=1)
    matrix.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f9fa")]),
        ("GRID", (0, 0), (-1, -1), 0.45, GRID), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(matrix)

    story.append(PageBreak())
    story.append(Paragraph("Verification and next steps", h2))
    story.append(Paragraph("Current public and CI evidence", h3))
    story.append(Paragraph("The Vercel landing and review pages are reachable, but the Vercel domain does not serve the FastAPI health route. The review workspace has a field for a backend URL; no public API URL is recorded in the repository. The successful GitHub Actions run validates the configured CI scope only. It is not evidence of real-model media acceptance or a hosted backend.", body))
    story.append(Paragraph("The earlier deployment report contained a mismatch: its Markdown labelled frontend checks passed while its JSON recorded timeouts. The corrected record distinguishes the earlier local backend integration from the current public frontend check and marks hosted analysis unverified.", body))

    story.append(Paragraph("Remaining work", h3))
    todos = [
        "Deploy the API and durable worker; confirm model loading, health checks, persistent storage, restart, and retention behavior.",
        "Set NEXT_PUBLIC_API_BASE_URL in Vercel to the backend URL and redeploy the frontend.",
        "Submit consented representative images and videos from /review. Check real engine results, evidence links, PDF download, retention, and error paths.",
        "Reproduce real-model acceptance tests; record the exact command, media protocol, model/config revision, counts, skips, and results. Make CI coverage limitations visible.",
        "Review thresholds against the intended cameras and media; keep outputs advisory and require a human reviewer.",
    ]
    for index, item in enumerate(todos, 1):
        story.append(Paragraph(f"<b>{index}.</b> {item}", body))

    story.append(Paragraph("Evaluation figures recorded in repository", h3))
    story.append(Paragraph("The project reports record M02 face ROC-AUC 0.8561, M02 synthetic-image ROC-AUC 0.9409, and M04 22/30 edited-image detections with 0/30 untouched false alarms. These are previously recorded results, not freshly reproduced here. See reports/m02_training/report.md, reports/m02_synth_training/report.md, reports/DEEP_STRESS_TEST_REPORT.md, and reports/VIDEO_BENCHMARK_REPORT.md for scope and caveats.", body))

    notice = Table([[Paragraph("ADVISORY USE ONLY. The system must not identify people or infer intent, criminality, or wrongdoing. Results may fail on new cameras, populations, compression levels, or manipulation methods. Evaluate on consented media representative of the intended use before deployment.", cell_bold)]], colWidths=[174 * mm])
    notice.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), AMBER), ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#d5b96f")),
        ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(Spacer(1, 2 * mm))
    story.append(notice)
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph("More detail: PROJECT_STATUS.md | DEPLOYMENT.md | EVALUATION.md | reports/DEPLOYMENT_VERIFICATION.md", small))

    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    print(f"Generated {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    build_pdf()
