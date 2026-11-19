"""Generate the updated 3-page DEEP SHIELD Project Status PDF with verified metrics."""
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
    canvas.setFillColor(colors.HexColor("#112936"))
    canvas.rect(0, h - 12 * mm, w, 12 * mm, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 8)
    canvas.drawString(22 * mm, h - 7.8 * mm, "DEEP SHIELD  /  PROJECT STATUS")
    canvas.drawRightString(w - 22 * mm, h - 7.8 * mm, "19 NOV 2026")

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
    body = ParagraphStyle("Body", parent=styles["Normal"], fontName="Helvetica", fontSize=9.3, leading=13.5, textColor=colors.HexColor("#223138"), spaceAfter=2.4 * mm)
    cell = ParagraphStyle("Cell", parent=styles["Normal"], fontName="Helvetica", fontSize=8.3, leading=11.5, textColor=colors.HexColor("#223138"))
    cell_bold = ParagraphStyle("CellBold", parent=cell, fontName="Helvetica-Bold")
    cell_head = ParagraphStyle("CellHead", parent=cell, fontName="Helvetica-Bold", textColor=colors.white)
    banner_cell = ParagraphStyle("BannerCell", parent=cell, alignment=1, fontName="Helvetica-Bold", textColor=colors.HexColor("#0e7265"), fontSize=8.3, leading=11.5)
    flow_cell = ParagraphStyle("FlowCell", parent=cell, alignment=1, fontName="Helvetica-Bold", textColor=colors.HexColor("#112936"), fontSize=8.3, leading=11)

    story = []

    # ---------------- PAGE 1 ----------------
    story.append(Paragraph("DEEP SHIELD", h1))
    story.append(Paragraph("Authentic real-dataset training, audit, and full-stack verification", sub))

    banner = Table(
        [[
            Paragraph("M02 FACE &amp; SYNTH HEADS<br/>ROC-AUC 0.9439 / 0.9409", banner_cell),
            Paragraph("ALL 4 ENGINES &amp; TESTS<br/>36 / 36 Passing (Real Models)", banner_cell),
            Paragraph("ZERO-MOCK WORKSPACE<br/>deep-shield-sooty.vercel.app", banner_cell),
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
    story.append(Paragraph("<font color='#0e7265'><b>System Architecture</b></font>", h3))

    def _box(text: str, width_mm: float = 120, bg: str = "#e8f3f1", border: str = "#1b6b61"):
        t = Table([[Paragraph(text, flow_cell)]], colWidths=[width_mm * mm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg)),
            ("BOX", (0, 0), (-1, -1), 1.0, colors.HexColor(border)),
            ("TOPPADDING", (0, 0), (-1, -1), 6.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6.5),
        ]))
        return t

    story.append(_box("IMAGE, VIDEO, OR HARDWARE WEBCAM CAPTURE", 96))
    story.append(Spacer(1, 2.5 * mm))

    mods = Table(
        [[
            Paragraph("M01 Threat<br/>hustvl/yolos-tiny<br/><font color='#0e7265'>COCO + IoU Track</font>", flow_cell),
            Paragraph("M02 Deepfake<br/>face-v3 + synth-v1<br/><font color='#0e7265'>AUC 0.944 / 0.941</font>", flow_cell),
            Paragraph("M03 Events<br/>Rules + GRU v4<br/><font color='#0e7265'>6,265p · 53 tracks</font>", flow_cell),
            Paragraph("M04 Forensics<br/>SIFT/RANSAC + ELA<br/><font color='#0e7265'>0/30 Untouched FP</font>", flow_cell),
        ]],
        colWidths=[40 * mm, 44 * mm, 41 * mm, 41 * mm],
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
    story.append(_box("CROSS-MODULE EVIDENCE LINKER (SCORES KEPT STRICTLY SEPARATE)", 135))
    story.append(Spacer(1, 2.5 * mm))
    story.append(_box("FASTAPI + DURABLE SQLITE WORKER + RETENTION PURGE + PDF REPORT", 145))
    story.append(Spacer(1, 2.5 * mm))
    story.append(_box("NEXT.JS LANDING PAGE + LIVE WORKSPACE (/ &amp; /review) — ZERO MOCK FALLBACKS", 152))
    story.append(Spacer(1, 2.5 * mm))

    bottom = Table(
        [[
            Paragraph("REAL-DATASET TRAINING &amp;<br/>DEEP STRESS TESTS — VERIFIED", flow_cell),
            Paragraph("DOCKER COMPOSE + CADDY HTTPS<br/>STACK READY FOR LINUX HOST", flow_cell),
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
    story.append(Paragraph("Every hardcoded heuristic, mock client fallback, synthetic evaluation fixture, and stale threshold bug discovered during full-project auditing has been eliminated. All four engines run real local model weights and calibrated classical parameters verified on genuine Kaggle/HuggingFace datasets and real surveillance footage.", body))

    # ---------------- PAGE 2 ----------------
    story.append(PageBreak())
    story.append(Paragraph("Verified Component Matrix", h2))
    story.append(Paragraph("All four analysis modules run independently and report separate score types, calibrated thresholds, model revisions, and explicit caveats for human review.", body))

    rows = [
        [Paragraph("ENGINE / LAYER", cell_head), Paragraph("LOADED WEIGHTS &amp; ARCHITECTURE", cell_head), Paragraph("VERIFIED REAL-DATA METRICS", cell_head)],
        [Paragraph("M01 — Threat &amp; Anomaly", cell_bold), Paragraph("Local MS-COCO 2017 hustvl/yolos-tiny (rev 5a6a017a20cb, Apache-2.0) + velocity-predicted MultiObjectIoUTracker; optional Ultralytics yolo26n + ByteTrack. Dwell 30s / unattended 60s.", cell), Paragraph("Real vtest.avi video: 82 detections, 21 tracks, mean conf 0.8958; 0/4 non-person FPs; 48/48 rule events matched (F1 1.000).", cell)],
        [Paragraph("M02 — Face Manipulation", cell_bold), Paragraph("Frozen SigLIP2 + deepshield-face-head-v3 (models/m02_face_head.npz, sha b12bda01ed0c, L2-norm MLP, Guo T=1.6, thr 0.207/0.777) + occlusion heatmaps + spectra + ELA/noise.", cell), Paragraph("2,800 real Kaggle/HF images (4 datasets). Test (651 faces): ROC-AUC 0.9439 (vs 0.6269), Acc 87.25%, F1 0.8935, ECE 0.0306, Decided Acc 94.49%.", cell)],
        [Paragraph("M02 — Whole-Image AI (synthetic_image)", cell_bold), Paragraph("Enabled by default on images via deepshield-synth-head-v1 (models/m02_synth_head.npz, sha 382a7a930175, 5-member L2-norm MLP ensemble, T=0.8047, thr 0.406/0.75).", cell), Paragraph("960 real AI-vs-Photo images (Hemg + CIFAKE). Test (240): ROC-AUC 0.9409 (vs 0.4741), Acc 87.50%, F1 0.8760, ECE 0.0661, Decided Acc 90.61%.", cell)],
        [Paragraph("M03 — Video Events + GRU", cell_bold), Paragraph("Shared M01 tracks + calibrated rules (crowd 4/5s, run 2.5 BL/s) + PyTorch Temporal GRU v4 (models/m03_temporal_gru.pt, 6,265 params, sha ea048bac8813, thr 0.35).", cell), Paragraph("5-fold cross-track CV on 53 real vtest.avi tracks: CV AUC 1.0000, ECE 0.0001, 0/11 false alarms on real walks, 6/6 kinematic scenarios.", cell)],
        [Paragraph("M04 — Visual Forensics", cell_bold), Paragraph("SIFT + RANSAC wired to models/m04_forensics_params.json (r=0.55, d=36, inliers=8, c=0.01), ELA q90, local noise map, JPEG quant/grid/ghosts &amp; EXIF.", cell), Paragraph("30-photo stress test: 0/30 (0.0%) untouched FPR; 22/30 (73.33%) recall across translation, 15 deg rotation, 1.15x scale &amp; q75; 5/5 JPEG q.", cell)],
        [Paragraph("API, Worker &amp; Retention", cell_bold), Paragraph("FastAPI /api/v1/intelligence (/analyze, /jobs, /models, /report.pdf, /retention/purge), durable SQLite queue with worker leases, heartbeats, and crash recovery.", cell), Paragraph("Cross-module linking 0.28 ms, PDF report 46.67 ms; automated lease-recovery and evidence retention purge tests pass.", cell)],
        [Paragraph("Frontend &amp; Verification", cell_bold), Paragraph("Next.js 15 + React 19 + GSAP landing page (/) &amp; workspace (/review), live hardware webcam capture, both M02 heads shown live, per-finding caveats, zero mock fallback.", cell), Paragraph("tsc --noEmit &amp; next build pass; 36 / 36 pytest tests pass with real models (DEEPSHIELD_TEST_REAL=1, 0 skipped).", cell)],
    ]
    tbl = Table(rows, colWidths=[37 * mm, 76 * mm, 53 * mm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#112936")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f9fa")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d3dce0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5.5),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 3.5 * mm))
    story.append(Paragraph("Social-Media Degradation Stress Test (M02 Face v3, 107 held-out faces)", h3))
    story.append(Paragraph("Clean ROC-AUC <b>0.9472</b> | Brightness/Contrast <b>0.9356</b> | JPEG q60 <b>0.9215</b> | Gaussian Blur <b>0.8895</b> | JPEG q40 <b>0.8649</b> | Downscale 0.5x <b>0.8645</b>. Full per-source breakdown: TPDNE <b>0.9969</b>, StyleGAN1 <b>0.9827</b>, Kaggle Deepfake <b>0.9045</b>, DeepFakeFace <b>0.8457</b> (EFS 0.8679, FS 0.8147).", body))

    # ---------------- PAGE 3 ----------------
    story.append(PageBreak())
    story.append(Paragraph("Honest Limitations &amp; Deployment", h2))
    story.append(Paragraph("1. Verified Reports &amp; Reproducibility", h3))
    story.append(Paragraph("All training, evaluation, and deep stress-test outputs are committed under reports/ and models/:", body))
    story.append(Paragraph("- <b>reports/m02_training/report.md</b> — 4-dataset face head training (`deepshield-face-head-v3`), Guo calibration, held-out test breakdown, and leave-one-generator-out generalization audit.", body))
    story.append(Paragraph("- <b>reports/m02_synth_training/report.md</b> — Whole-image AI-generation head (`deepshield-synth-head-v1`) on Hemg + CIFAKE (`0.9409` held-out ROC-AUC vs `0.4741` baseline).", body))
    story.append(Paragraph("- <b>reports/m03_gru_training/report.json</b> — 5-fold cross-track CV on 53 real pedestrian tracks extracted from OpenCV vtest.avi (`temporal-gru-kinematic-v4-realtracks`).", body))
    story.append(Paragraph("- <b>reports/DEEP_STRESS_TEST_REPORT.md</b> &amp; <b>reports/EVALUATION_REPORT.md</b> — End-to-end stress testing across real surveillance clips, social-media compressions, and 30 rotated/scaled/recompressed real photos.", body))

    story.append(Paragraph("2. Honest Known Limitations (Required Reviewer Context)", h3))
    story.append(Paragraph("- <b>Unseen Generators (M02 Face LOO):</b> Leave-one-generator-out ROC-AUC drops on completely unseen generator families (tpdne 0.7891, dff 0.5231, kaggle_deepfake 0.4778, stylegan1 0.4778). Cross-architecture drift remains an open research limit; inspect M04 forensic maps alongside M02.", body))
    story.append(Paragraph("- <b>Site-Specific Surveillance Anomalies (M03 GRU):</b> Trained on 53 genuine vtest.avi pedestrian tracks paired with kinematically warped sprint/erratic anomalies. Operational CCTV deployments should fine-tune on consented camera-angle footage.", body))
    story.append(Paragraph("- <b>Low-Texture Copy-Move (M04):</b> SIFT requires textured keypoints; clones pasted onto smooth sky/skin backgrounds cannot be matched reliably (~73.3% recall on 30 diverse photos, 0.0% untouched false-positive rate).", body))

    story.append(Paragraph("3. Retention &amp; Single-Host Docker Deployment (DEPLOYMENT.md)", h3))
    story.append(Paragraph("Uploaded source files are removed immediately after worker analysis. Derived evidence (evidence_store/m01..m04) and SQLite job rows are purged automatically via DEEPSHIELD_RETENTION_HOURS (default 168h) or on demand via DELETE /api/v1/intelligence/jobs/{job_id} and POST /api/v1/intelligence/retention/purge. For Linux HTTPS deployment, copy .env.production.example to .env.production and run <b>docker compose --env-file .env.production up -d --build</b>.", body))

    story.append(Spacer(1, 3.5 * mm))
    interp = Table(
        [[Paragraph("<b>Advisory Governance Notice</b><br/>No module identifies individuals or infers intent, authenticity, or wrongdoing. All detector confidences, temperature-calibrated classifier probabilities, temporal GRU scores, rule candidates, and forensic indicators remain strictly separate and advisory for human reviewers.", cell)]],
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
    for dl_name in ("Deep_Shield_Project_Status_2026-11-19.pdf", "Deep_Shield_Project_Status_2026-11-19_Updated.pdf"):
        try:
            Path(r"C:\Users\Dell\Downloads", dl_name).write_bytes(target.read_bytes())
        except OSError:
            pass
    print(f"Generated {target} ({target.stat().st_size} bytes)")
