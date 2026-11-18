# DEEP SHIELD — Consolidated Evaluation & Threshold Calibration Report

**Generated:** 2026-11-19T01:02:05

## 1. Modules 01 & 03 — Threat, Anomaly & Video Event Evaluation

Evaluated across **6 calibration video sequences** and **6 held-out test video sequences** (540.0s total test footage).

### Calibrated Rule Thresholds

| Parameter | Calibrated Value | Purpose |
|---|---|---|
| `DEEPSHIELD_M01_DWELL_SECONDS` | **30.0s** | Loitering candidate dwell threshold |
| `DEEPSHIELD_M01_UNATTENDED_SECONDS` | **60.0s** | Stationary unattended bag threshold |
| `DEEPSHIELD_M03_CROWD_MIN_PEOPLE` | **4** | Minimum people in zone for crowding |
| `DEEPSHIELD_M03_CROWD_SECONDS` | **5.0s** | Minimum crowding duration |
| `DEEPSHIELD_M03_RUNNING_BODY_LENGTHS_PER_SECOND` | **2.5** | Rapid movement speed threshold |

### Held-Out Test Split Event Metrics (Temporal IoU >= 0.5)

| Event Type | Precision | Recall | F1 | TP | FP | Missed (FN) | Duplicate | Mean Timestamp Error (s) |
|---|---|---|---|---|---|---|---|---|
| **Overall (All Events)** | **1.0** | **1.0** | **1.0** | 36 | 0 | 0 | 0 | 0.0s |
| `crowding_candidate` | 1.0 | 1.0 | 1.0 | 6 | 0 | 0 | 0 | 0.0s |
| `loitering_candidate` | 1.0 | 1.0 | 1.0 | 6 | 0 | 0 | 0 | 0.0s |
| `running_candidate` | 1.0 | 1.0 | 1.0 | 6 | 0 | 0 | 0 | 0.0s |
| `unattended_object_candidate` | 1.0 | 1.0 | 1.0 | 6 | 0 | 0 | 0 | 0.0s |
| `zone_entry` | 1.0 | 1.0 | 1.0 | 6 | 0 | 0 | 0 | 0.0s |
| `zone_exit` | 1.0 | 1.0 | 1.0 | 6 | 0 | 0 | 0 | 0.0s |

## 2. Module 02 — Deepfake & Image Integrity Evaluation (Multi-Generator Held-Out Benchmark)

- **Base Model:** `prithivMLmods/open-deepfake-detection` (frozen SigLIP2 backbone) + trained DEEP SHIELD head (`models/m02_face_head.npz`, SHA `c8c61b23a9d5`)
- **Calibrated Three-Band Thresholds (Validation Split, <=5% FPR/FNR):** Low = **0.02**, High = **0.978**
- **Held-Out Pooled Test Set (276 faces: 166 fake / 110 real across StyleGAN, StyleGAN2/TPDNE, and DeepFakeFace swaps/diffusion):**
  - **Trained Head ROC-AUC:** **0.9057** (vs Original Model ROC-AUC: `0.6509`)
  - **Trained Head Accuracy @ 0.5:** **0.837** | **F1 @ 0.5:** **0.864** | **TPR @ 5% FPR:** **0.6506**
  - **Three-Band Coverage:** **0.6957** | **Decided Accuracy:** **0.901**
- See [reports/m02_training/report.md](m02_training/report.md) for full per-generator and leave-one-generator-out results.

## 3. Module 04 — Visual Forensics Evaluation

| Check | Metric | Result |
|---|---|---|
| Copy-move detection (120–175 px clones, n=10) | Detection Recall | **1.0** (10/10) |
| Copy-move localization against ground-truth boxes | Mean BBox IoU / Dice | **IoU 0.7262** / **Dice 0.8395** |
| Copy-move false-alarm rate on untouched crops (n=8) | False Positive Rate | **0.0** |
| JPEG quantisation-table quality estimate ([60, 75, 85, 90, 95]) | Exact Match Rate | **5/5** |
| 8x8 JPEG block-grid alignment & crop shift check | Aligned / Shifted Detected | **True / True** |
| JPEG ghost contrast (q60 paste inside q90 host) | Mean Ghost Map Separation | **0.6725** |

## 4. System Performance & Reporting Benchmarks

| Stage | Latency / Output |
|---|---|
| Module 04 full-image forensic suite (SIFT copy-move + ELA + noise + JPEG ghosts) | **505.7 ms** |
| Cross-module spatial-temporal intelligence linking | **0.21 ms** (1 linked group) |
| Consolidated advisory PDF report generation | **28.55 ms** (4014 bytes) |

> **Advisory Notice:** Model scores and heuristic rule detections remain separate and advisory for human review. No module infers identity, intent, or legal/disciplinary conclusions.
