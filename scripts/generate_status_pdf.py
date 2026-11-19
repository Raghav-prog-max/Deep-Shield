"""Generate the updated 3-page DEEP SHIELD Project Status PDF with verified 6-dataset + 52-video metrics."""
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
    body = ParagraphStyle("Body", parent=styles["Normal"], fontName="Helvetica", fontSize=9.2, leading=13.2, textColor=colors.HexColor("#223138"), spaceAfter=2.2 * mm)
    cell = ParagraphStyle("Cell", parent=styles["Normal"], fontName="Helvetica", fontSize=8.2, leading=11.3, textColor=colors.HexColor("#223138"))
    cell_bold = ParagraphStyle("CellBold", parent=cell, fontName="Helvetica-Bold")
    cell_head = ParagraphStyle("CellHead", parent=cell, fontName="Helvetica-Bold", textColor=colors.white)
    banner_cell = ParagraphStyle("BannerCell", parent=cell, alignment=1, fontName="Helvetica-Bold", textColor=colors.HexColor("#0e7265"), fontSize=8.2, leading=11.5)
    flow_cell = ParagraphStyle("FlowCell", parent=cell, alignment=1, fontName="Helvetica-Bold", textColor=colors.HexColor("#112936"), fontSize=8.2, leading=11)

    story = []

    # ---------------- PAGE 1 ----------------
    story.append(Paragraph("DEEP SHIELD", h1))
    story.append(Paragraph("Authentic 6-dataset image + 52-video real benchmark, audit, and cloud packaging", sub))

    banner = Table(
        [[
            Paragraph("6-DATASET FACE v4 + SYNTH v1<br/>Decided Acc 92.86% / 90.61%", banner_cell),
            Paragraph("52 REAL BENCHMARK VIDEOS<br/>Celeb-DF v2 + FF++/DFD + UCF-Crime", banner_cell),
            Paragraph("36 / 36 TESTS &amp; CLOUD READY<br/>Railway / Render / HF Spaces / Vercel", banner_cell),
        ]],
        colWidths=[55 * mm, 56 * mm, 55 * mm],
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
            Paragraph("M02 Deepfake<br/>face-v4 + synth-v1<br/><font color='#0e7265'>6-Src + 40 Videos</font>", flow_cell),
            Paragraph("M03 Events<br/>Rules + GRU v4<br/><font color='#0e7265'>6,265p · Real CCTV</font>", flow_cell),
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
            Paragraph("6-DATASET IMAGE + 52-VIDEO REAL<br/>BENCHMARKS &amp; STRESS TESTS", flow_cell),
            Paragraph("CLOUD BACKEND PACKAGING<br/>Railway · Render · HF Spaces · CI", flow_cell),
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
    story.append(Paragraph("Every hardcoded heuristic, mock client fallback, synthetic evaluation fixture, and stale threshold bug has been eliminated. All four engines run real local weights verified on genuine Kaggle/HuggingFace images plus 52 real benchmark videos (Celeb-DF v2, FaceForensics++/Google DFD, and UCF-Crime CCTV).", body))

    # ---------------- PAGE 2 ----------------
    story.append(PageBreak())
    story.append(Paragraph("Verified Component Matrix", h2))
    story.append(Paragraph("All four analysis modules run independently and report separate score types, calibrated thresholds, model revisions, and explicit caveats for human review.", body))

    rows = [
        [Paragraph("ENGINE / LAYER", cell_head), Paragraph("LOADED WEIGHTS &amp; ARCHITECTURE", cell_head), Paragraph("VERIFIED REAL-DATA METRICS", cell_head)],
        [Paragraph("M01 — Threat &amp; Anomaly", cell_bold), Paragraph("Local MS-COCO 2017 hustvl/yolos-tiny (rev 5a6a017a20cb, Apache-2.0) + velocity-predicted MultiObjectIoUTracker; optional Ultralytics yolo26n. Dwell 30s / unattended 60s.", cell), Paragraph("vtest.avi: 82 detections, 21 tracks, mean conf 0.8958; 0/4 non-person FPs; 48/48 rule events; live-tested on 12 UCF-Crime CCTV clips.", cell)],
        [Paragraph("M02 — Face Manipulation (v4)", cell_bold), Paragraph("Frozen SigLIP2 + 5-member L2-norm MLP ensemble deepshield-face-head-v4 (sha f12a651f67ec, alpha=0.1, Guo T=1.9768, thr 0.156/0.804) + heatmaps + spectra + ELA/noise.", cell), Paragraph("6 real image+video datasets (3,504 items). 966 test faces: AUC 0.8561 (vs 0.5750), ECE 0.0490, Decided Acc 92.86%. 40 MP4s: FF++/DFD 0.7200, Celeb-DF 0.6200, pooled 80.0% Decided Acc.", cell)],
        [Paragraph("M02 — Whole-Image AI (synthetic_image)", cell_bold), Paragraph("Enabled by default on images via deepshield-synth-head-v1 (models/m02_synth_head.npz, sha 382a7a930175, 5-member L2-norm MLP ensemble, T=0.8047, thr 0.406/0.75).", cell), Paragraph("960 real AI-vs-Photo images (Hemg + CIFAKE). Test (240): ROC-AUC 0.9409 (vs 0.4741), Acc 87.50%, F1 0.8760, ECE 0.0661, Decided Acc 90.61%.", cell)],
        [Paragraph("M03 — Video Events + GRU", cell_bold), Paragraph("Shared M01 tracks + calibrated rules (crowd 4/5s, run 2.5 BL/s) + PyTorch Temporal GRU v4 (models/m03_temporal_gru.pt, 6,265 params, sha ea048bac8813, thr 0.35).", cell), Paragraph("5-fold CV on 53 real vtest.avi tracks: CV AUC 1.0000, ECE 0.0001, 0/11 false alarms on held-out walks, 6/6 kinematic scenarios; tested on 12 UCF-Crime clips.", cell)],
        [Paragraph("M04 — Visual Forensics", cell_bold), Paragraph("SIFT + RANSAC wired to models/m04_forensics_params.json (r=0.55, d=36, inliers=8, c=0.01), ELA q90, local noise map, JPEG quant/grid/ghosts &amp; EXIF.", cell), Paragraph("30-photo stress test: 0/30 (0.0%) untouched FPR; 22/30 (73.33%) recall across translation, 15 deg rotation, 1.15x scale &amp; q75; 5/5 JPEG q.", cell)],
        [Paragraph("Cloud &amp; Durable Worker", cell_bold), Paragraph("FastAPI /api/v1/intelligence + SQLite durable queue + scripts/cloud_entrypoint.py supervisor + Dockerfile, railway.json, render.yaml, Dockerfile.hf, GitHub Actions CI.", cell), Paragraph("cloud_entrypoint.py live-verified on custom PORT; linking 0.28 ms, PDF 46.67 ms; lease-recovery &amp; retention purge verified.", cell)],
        [Paragraph("Frontend &amp; Tests", cell_bold), Paragraph("Next.js 15 + React 19 + GSAP landing (/) &amp; workspace (/review), real webcam capture, both M02 heads live, visible caveats, zero mock fallback.", cell), Paragraph("tsc --noEmit &amp; next build pass; live at deep-shield-sooty.vercel.app; 36 / 36 real-model pytest tests pass (0 skipped).", cell)],
    ]
    tbl = Table(rows, colWidths=[37 * mm, 76 * mm, 53 * mm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#112936")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f9fa")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d3dce0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5.0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5.0),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 3.5 * mm))
    story.append(Paragraph("Social-Media Degradation Stress Test (M02 Face v4, 107 held-out faces)", h3))
    story.append(Paragraph("Clean ROC-AUC <b>0.9493</b> (Acc 87.85%, F1 0.8926) | Brightness <b>0.9391</b> | JPEG q60 <b>0.9324</b> | Gaussian Blur <b>0.8944</b> | Downscale 0.5x <b>0.8666</b> | JPEG q40 <b>0.8543</b>. Per-source test AUCs: TPDNE <b>0.9952</b>, StyleGAN1 <b>0.9802</b>, Kaggle-Deepfake <b>0.8835</b>, DFF <b>0.8519</b>, FF++/DFD Video <b>0.7182</b> (LOO unseen <b>0.7410</b>), Celeb-DF v2 <b>0.6238</b>.", body))

    # ---------------- PAGE 3 ----------------
    story.append(PageBreak())
    story.append(Paragraph("Honest Limitations &amp; Cloud Deployment", h2))
    story.append(Paragraph("1. Verified Reports &amp; Reproducibility", h3))
    story.append(Paragraph("All training, real-video benchmark, and stress-test reports are committed under reports/ and models/:", body))
    story.append(Paragraph("- <b>reports/VIDEO_BENCHMARK_REPORT.md</b> — 40 actor-disjoint real deepfake MP4s (Celeb-DF v2 + FF++/Google DFD) &amp; 12 real UCF-Crime CCTV surveillance videos.", body))
    story.append(Paragraph("- <b>reports/m02_training/report.md</b> — 6-dataset real image+video training (deepshield-face-head-v4, 966 held-out test faces, Guo calibration, 6-source LOO audit).", body))
    story.append(Paragraph("- <b>reports/m02_synth_training/report.md</b> — Whole-image AI-generation head (deepshield-synth-head-v1 on Hemg + CIFAKE, 0.9409 ROC-AUC vs 0.4741 baseline).", body))
    story.append(Paragraph("- <b>reports/DEEP_STRESS_TEST_REPORT.md</b> &amp; <b>reports/EVALUATION_REPORT.md</b> — Social-media degradations, 30-photo M04 stress test, M03 5-fold CV &amp; system timings.", body))

    story.append(Paragraph("2. Honest Known Limitations (Required Reviewer Context)", h3))
    story.append(Paragraph("- <b>Compressed Video Deepfakes &amp; Unseen Generators (M02):</b> Video-level AUC reaches 0.7200 on FF++/DFD and 0.6200 on Celeb-DF v2 (80.0% pooled Decided Acc; unseen LOO transfer: tpdne 0.8137, dfd_video 0.7410, dff 0.5855, stylegan1 0.5417, kaggle_deepfake 0.5120, celebdf_video 0.4559). Heavily recompressed video swaps often fall in the inconclusive band by design; pair M02 with M04 visual forensics.", body))
    story.append(Paragraph("- <b>Multi-Camera CCTV Perspective (M03):</b> On uncalibrated UCF-Crime cameras, foreground perspective crossings can elevate GRU kinematic scores on normal scenes, while subtle non-kinematic anomalies (e.g. distant shoplifting) do not trigger running kinetics. Site-specific calibration is required before operational CCTV deployment.", body))
    story.append(Paragraph("- <b>Low-Texture Copy-Move (M04):</b> SIFT requires texture keypoints (~73.3% recall across 30 diverse photos, 0.0% untouched false-alarm rate).", body))

    story.append(Paragraph("3. Cloud &amp; Self-Hosted Deployment (DEPLOYMENT.md)", h3))
    story.append(Paragraph("For multi-device remote access from deep-shield-sooty.vercel.app, deploy the single-container backend (Dockerfile + scripts/cloud_entrypoint.py) to Railway (railway.json), Render (render.yaml), or HuggingFace Spaces (Dockerfile.hf, free 16 GB RAM CPU tier) and set NEXT_PUBLIC_API_BASE_URL in Vercel. For self-hosted Linux HTTPS, run <b>docker compose --env-file .env.production up -d --build</b>.", body))

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
