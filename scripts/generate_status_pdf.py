"""Generate the updated 3-page DEEP SHIELD Project Status PDF."""
from __future__ import annotations

from pathlib import Path
from reportlab.lib import colors
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


def _header_footer(canvas, doc):
    canvas.saveState()
    w, h = A4
    # Top dark header bar
    canvas.setFillColor(colors.HexColor("#112936"))
    canvas.rect(0, h - 12 * mm, w, 12 * mm, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 8)
    canvas.drawString(22 * mm, h - 7.8 * mm, "DEEP SHIELD  /  PROJECT STATUS")
    canvas.drawRightString(w - 22 * mm, h - 7.8 * mm, "19 NOV 2026")

    # Bottom footer
    canvas.setStrokeColor(colors.HexColor("#d3dce0"))
    canvas.setLineWidth(0.5)
    canvas.line(22 * mm, 15 * mm, w - 22 * mm, 15 * mm)
    canvas.setFillColor(colors.HexColor("#687780"))
    canvas.setFont("Helvetica", 8)
    canvas.drawString(22 * mm, 10 * mm, "github.com/Raghav-prog-max/Deep-Shield")
    canvas.drawRightString(w - 22 * mm, 10 * mm, str(doc.page))
    canvas.restoreState()


def build_pdf(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=22 * mm,
        rightMargin=22 * mm,
        topMargin=24 * mm,
        bottomMargin=22 * mm,
        title="DEEP SHIELD — Project Status (19 Nov 2026)",
        author="DEEP SHIELD Team",
    )
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("H1", parent=styles["Title"], alignment=0, fontName="Helvetica-Bold", fontSize=24, leading=28, textColor=colors.HexColor("#112936"), spaceAfter=2 * mm)
    sub = ParagraphStyle("Sub", parent=styles["Normal"], fontName="Helvetica", fontSize=11, leading=14, textColor=colors.HexColor("#5c6d75"), spaceAfter=6 * mm)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=16, leading=20, textColor=colors.HexColor("#112936"), spaceBefore=4 * mm, spaceAfter=3 * mm)
    h3 = ParagraphStyle("H3", parent=styles["Heading3"], fontName="Helvetica-Bold", fontSize=11.5, leading=15, textColor=colors.HexColor("#0e7265"), spaceBefore=4 * mm, spaceAfter=2 * mm)
    body = ParagraphStyle("Body", parent=styles["Normal"], fontName="Helvetica", fontSize=9.5, leading=14, textColor=colors.HexColor("#223138"), spaceAfter=2.5 * mm)
    cell = ParagraphStyle("Cell", parent=styles["Normal"], fontName="Helvetica", fontSize=8.5, leading=12, textColor=colors.HexColor("#223138"))
    cell_bold = ParagraphStyle("CellBold", parent=cell, fontName="Helvetica-Bold")
    cell_head = ParagraphStyle("CellHead", parent=cell, fontName="Helvetica-Bold", textColor=colors.white)
    banner_cell = ParagraphStyle("BannerCell", parent=cell, alignment=1, fontName="Helvetica-Bold", textColor=colors.HexColor("#0e7265"), fontSize=8.5, leading=12)
    flow_cell = ParagraphStyle("FlowCell", parent=cell, alignment=1, fontName="Helvetica-Bold", textColor=colors.HexColor("#112936"), fontSize=8.5, leading=11)

    story = []

    # ---------------- PAGE 1 ----------------
    story.append(Paragraph("DEEP SHIELD", h1))
    story.append(Paragraph("Project status and completed verification", sub))

    banner = Table(
        [[
            Paragraph("CORE IMPLEMENTATION<br/>Complete and merged to main", banner_cell),
            Paragraph("MODEL EVALUATION &amp; HEAD<br/>Completed (ROC-AUC 0.9057)", banner_cell),
            Paragraph("FRONTEND &amp; HARDENING<br/>Landing + Workspace + Retention Done", banner_cell),
        ]],
        colWidths=[55 * mm, 55 * mm, 56 * mm],
    )
    banner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f2f7f6")),
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#cddbd9")),
        ("INNERGRID", (0, 0), (-1, -1), 0.6, colors.HexColor("#cddbd9")),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(banner)
    story.append(Spacer(1, 5 * mm))
    story.append(Paragraph("<font color='#0e7265'><b>Project flow</b></font>", h3))

    def _box(text: str, width_mm: float = 120, bg: str = "#e8f3f1", border: str = "#1b6b61"):
        t = Table([[Paragraph(text, flow_cell)]], colWidths=[width_mm * mm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg)),
            ("BOX", (0, 0), (-1, -1), 1.0, colors.HexColor(border)),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ]))
        return t

    story.append(_box("IMAGE OR VIDEO UPLOAD", 75))
    story.append(Spacer(1, 2.5 * mm))

    mods = Table(
        [[
            Paragraph("M01<br/>Threat and anomaly<br/><font color='#0e7265'>DONE</font>", flow_cell),
            Paragraph("M02<br/>Deepfake + Trained Head<br/><font color='#0e7265'>DONE (AUC 0.906)</font>", flow_cell),
            Paragraph("M03<br/>Video events<br/><font color='#0e7265'>DONE</font>", flow_cell),
            Paragraph("M04<br/>Visual forensics<br/><font color='#0e7265'>DONE</font>", flow_cell),
        ]],
        colWidths=[40 * mm, 43 * mm, 40 * mm, 43 * mm],
    )
    mods.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#e8f3f1")),
        ("BOX", (0, 0), (-1, -1), 1.0, colors.HexColor("#1b6b61")),
        ("INNERGRID", (0, 0), (-1, -1), 1.0, colors.HexColor("#1b6b61")),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(mods)
    story.append(Spacer(1, 2.5 * mm))
    story.append(_box("SHARED FINDINGS, EVIDENCE LINKS AND GROUPS — DONE", 115))
    story.append(Spacer(1, 2.5 * mm))
    story.append(_box("INTEGRATED API  |  DURABLE JOB WORKER  |  PDF REPORTS  |  RETENTION PURGE — DONE", 145))
    story.append(Spacer(1, 2.5 * mm))
    story.append(_box("INTEGRATED FRONTEND (BENNETT LANDING PAGE + LIVE REVIEW WORKSPACE) — DONE", 145))
    story.append(Spacer(1, 2.5 * mm))

    bottom = Table(
        [[
            Paragraph("MULTI-GENERATOR EVALUATION &amp;<br/>THRESHOLD CALIBRATION — DONE", flow_cell),
            Paragraph("DOCKER COMPOSE + CADDY HTTPS<br/>STACK READY FOR LINUX HOST — DONE", flow_cell),
        ]],
        colWidths=[82 * mm, 84 * mm],
    )
    bottom.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#e8f3f1")),
        ("BOX", (0, 0), (-1, -1), 1.0, colors.HexColor("#1b6b61")),
        ("INNERGRID", (0, 0), (-1, -1), 1.0, colors.HexColor("#1b6b61")),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(bottom)
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("All modules, frontend routes (/ and /review), M02 multi-generator trained face head, M01–M04 evaluation reports, and retention controls are implemented, verified, and pushed to main.", body))

    # ---------------- PAGE 2 ----------------
    story.append(PageBreak())
    story.append(Paragraph("What is done", h2))
    story.append(Paragraph("The four analysis modules feed a single review path. Engine scores and findings remain separate, and the results are advisory.", body))

    rows = [
        [Paragraph("AREA", cell_head), Paragraph("IMPLEMENTED", cell_head), Paragraph("STATUS", cell_head)],
        [Paragraph("M01 - Threat and anomaly", cell_bold), Paragraph("YOLO26n object detection, ByteTrack, zone entry, loitering (30s) and unattended-object (60s) candidates.", cell), Paragraph("Built &amp; calibrated via evaluate_all.py (36/36 held-out events matched).", cell)],
        [Paragraph("M02 - Deepfake and integrity", cell_bold), Paragraph("Frozen SigLIP2 backbone + trained multi-generator face head (models/m02_face_head.npz), occlusion heatmaps, spectral plots, ELA/noise.", cell), Paragraph("Built &amp; evaluated on 276 held-out faces: ROC-AUC 0.9057 (90.1% decided acc).", cell)],
        [Paragraph("M03 - Video events", cell_bold), Paragraph("Rule-based zone entry/exit, loitering, crowding (4 people / 5s) and rapid-movement (2.5 BL/s) candidates.", cell), Paragraph("Built &amp; calibrated on calibration/held-out sequences (temporal IoU >= 0.5).", cell)],
        [Paragraph("M04 - Visual forensics", cell_bold), Paragraph("Copy-move (SIFT + RANSAC), ELA/noise maps, JPEG quantisation/grid/ghosts and metadata observations.", cell), Paragraph("Built &amp; evaluated: 10/10 recall (IoU 0.7262, Dice 0.8395), 5/5 JPEG quality.", cell)],
        [Paragraph("Integration and reports", cell_bold), Paragraph("Integrated API, linked findings/evidence, persistent job status, downloadable PDF reports, and retention purge endpoints.", cell), Paragraph("Implemented and verified (0.21 ms linking, 28.5 ms PDF generation).", cell)],
        [Paragraph("Durable worker &amp; retention", cell_bold), Paragraph("Separate SQLite-backed queue, worker leases, heartbeats, crash recovery, and automated DEEPSHIELD_RETENTION_HOURS purge.", cell), Paragraph("Verified with automated lease recovery &amp; evidence purge tests.", cell)],
        [Paragraph("Frontend (Landing + Workspace)", cell_bold), Paragraph("Bennett explainable landing page (GSAP vistas, Agent Canvas, karaoke call) + Live Media Review Workspace (/ and /review).", cell), Paragraph("Typecheck (tsc --noEmit) &amp; Next.js production build pass; 0 vulnerabilities.", cell)],
        [Paragraph("Repository", cell_bold), Paragraph("All branches merged and pushed to Raghav-prog-max/Deep-Shield (main &amp; udbhav-branch) and PROJECT_SHOWCASE.", cell), Paragraph("28 passed, 1 skipped in pytest suite.", cell)],
    ]
    tbl = Table(rows, colWidths=[38 * mm, 76 * mm, 52 * mm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#112936")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f9fa")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d3dce0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("Verification snapshot", h3))
    story.append(Paragraph("Frontend typecheck (tsc --noEmit) and Next.js optimized production build pass with zero vulnerabilities for both / (Landing Page + embedded workspace) and /review. Full backend pytest suite passes (28 passed, 1 skipped). Module 02 trained head lifts held-out pooled test ROC-AUC from 0.6509 to 0.9057 across StyleGAN, StyleGAN2/TPDNE (0.9867 AUC), and DeepFakeFace swaps/diffusion (0.7255 AUC).", body))

    # ---------------- PAGE 3 ----------------
    story.append(PageBreak())
    story.append(Paragraph("Evaluation &amp; Deployment Summary", h2))
    story.append(Paragraph("1. Completed Model Evaluation &amp; Threshold Calibration", h3))
    story.append(Paragraph("All evaluation reports and calibrated weights are committed in the repository under reports/ and models/:", body))
    story.append(Paragraph("- <b>M02 (reports/m02_training/report.md):</b> Frozen SigLIP2 backbone + trained MLP head on multi-depth features across 962 faces from three generator families. Validation ROC-AUC 0.8616 (vs 0.5384 original); held-out test ROC-AUC <b>0.9057</b> (vs 0.6509 original), accuracy @ 0.5 = <b>83.7%</b>, decided accuracy = <b>90.1%</b>.", body))
    story.append(Paragraph("- <b>M01 &amp; M03 (reports/EVALUATION_REPORT.md):</b> Thresholds calibrated on 6 calibration video sequences (DEEPSHIELD_M01_DWELL_SECONDS=30.0s, DEEPSHIELD_M01_UNATTENDED_SECONDS=60.0s, DEEPSHIELD_M03_CROWD_MIN_PEOPLE=4, DEEPSHIELD_M03_CROWD_SECONDS=5.0s, DEEPSHIELD_M03_RUNNING_BODY_LENGTHS_PER_SECOND=2.5) and verified on 6 held-out test sequences.", body))
    story.append(Paragraph("- <b>M04 (reports/EVALUATION_REPORT.md):</b> 10/10 copy-move detection recall on 120–175 px clones (Mean BBox IoU 0.7262, Dice 0.8395), 0/8 false alarms on untouched crops, and 5/5 exact JPEG quality table estimation.", body))

    story.append(Paragraph("2. Production Hardening &amp; Evidence Retention (Implemented)", h3))
    story.append(Paragraph("Uploaded source files are deleted immediately after worker processing. Derived evidence (evidence_store/m01..m04) and completed SQLite job records are automatically purged by the worker based on DEEPSHIELD_RETENTION_HOURS (default 168h / 7 days), and can also be deleted on demand via DELETE /api/v1/intelligence/jobs/{job_id} or POST /api/v1/intelligence/retention/purge.", body))

    story.append(Paragraph("3. Hosting Deployment Steps (DEPLOYMENT.md)", h3))
    story.append(Paragraph("When provisioning a public Linux Docker host with a DNS domain and ports 80/443 open: copy .env.production.example to .env.production, set SITE_ADDRESS, SITE_USER, and SITE_PASSWORD_HASH, and run <b>docker compose --env-file .env.production up -d --build</b>.", body))

    story.append(Spacer(1, 4 * mm))
    interp = Table(
        [[Paragraph("<b>Interpretation</b><br/>No module recognizes identity or infers intent or wrongdoing. Model scores and rule detections remain separate and advisory for human review.", cell)]],
        colWidths=[166 * mm],
    )
    interp.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eef7f5")),
        ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#1b6b61")),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(interp)

    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)


if __name__ == "__main__":
    target = ROOT / "reports" / "Deep_Shield_Project_Status_2026-11-19.pdf"
    build_pdf(target)
    downloads = Path(r"C:\Users\Dell\Downloads\Deep_Shield_Project_Status_2026-11-19_Updated.pdf")
    try:
        downloads.write_bytes(target.read_bytes())
    except OSError:
        pass
    print(f"Generated {target} ({target.stat().st_size} bytes)")
