# DEEP SHIELD — Consolidated Real-Dataset Evaluation & Calibration Report

**Generated:** 2026-11-19T06:31:50

## 1. Modules 01 & 03 — Neural Object Detection, Tracking, Rules & Real-Track Temporal GRU

### Real Surveillance Video Verification (`eval/videos/test/test_corridor_01.mp4` — OpenCV `vtest.avi`)

- **Neural Object Detector:** `hustvl/yolos-tiny` (`transformers/pytorch`, rev `5a6a017a20cb`)
- **M01 Status:** `completed` (82 neural person/bag detections in 15981.9 ms)
- **M03 Status:** `completed` (29 tracked pedestrians, 16 tracks scored by PyTorch GRU, mean normal score `0.0626`)
- **M03 PyTorch Temporal GRU (`temporal-gru-kinematic-v4-realtracks`, sha `ea048bac8813`, 6265 params):** Trained on disjoint real pedestrian tracks from `vtest.avi` (`train=32, val=10, test=11`). Held-out Test ROC-AUC **1.0**, Calibrated Threshold **0.35**, Unperturbed real normal track false alarms: **0/11**.

### Deterministic Rule Verification (Unfiltered Predictions, Temporal IoU >= 0.5)

| Event Type | Precision | Recall | F1 | TP | FP | Missed (FN) | Duplicate | Mean Timestamp Error (s) |
|---|---|---|---|---|---|---|---|---|
| **Overall (All Events)** | **1.0** | **1.0** | **1.0** | 48 | 0 | 0 | 0 | 0.0s |
| `crowding_candidate` | 1.0 | 1.0 | 1.0 | 6 | 0 | 0 | 0 | 0.0s |
| `loitering_candidate` | 1.0 | 1.0 | 1.0 | 6 | 0 | 0 | 0 | 0.0s |
| `running_candidate` | 1.0 | 1.0 | 1.0 | 6 | 0 | 0 | 0 | 0.0s |
| `unattended_object_candidate` | 1.0 | 1.0 | 1.0 | 6 | 0 | 0 | 0 | 0.0s |
| `zone_entry` | 1.0 | 1.0 | 1.0 | 12 | 0 | 0 | 0 | 0.0s |
| `zone_exit` | 1.0 | 1.0 | 1.0 | 12 | 0 | 0 | 0 | 0.0s |

## 2. Module 02 — Deepfake & Image Integrity (4-Source Real Kaggle / HF Benchmark)

- **Base Model:** `prithivMLmods/open-deepfake-detection` (frozen SigLIP2 backbone) + trained DEEP SHIELD head (`C:\Users\Dell\.gemini\antigravity\scratch\Deep-Shield\models\m02_face_head.npz`, SHA `b12bda01ed0c`)
- **Validation-Calibrated Three-Band Thresholds (<=5% FPR/FNR):** Low = **0.207**, High = **0.777**
- **Held-Out Pooled Test Set (651 real-world faces: 382 fake / 269 real across Kaggle 140k StyleGAN, StyleGAN2/TPDNE, DeepFakeFace swaps/diffusion, and Kaggle 190k Deepfake-and-Real-Images):**
  - **Trained Head ROC-AUC:** **0.9439** (vs Original Model ROC-AUC: `0.6269`)
  - **Trained Head Accuracy @ 0.5:** **0.8725** | **F1 @ 0.5:** **0.8935** | **TPR @ 5% FPR:** **0.7356**
  - **Three-Band Coverage:** **0.725** | **Decided Accuracy:** **0.9449**
- See [reports/m02_training/report.md](m02_training/report.md) for per-dataset and leave-one-generator-out results.

## 3. Module 04 — Visual Forensics (Evaluated on Diverse Real Held-Out Photographs)

Calibrated SIFT+RANSAC parameters wired into `VisualForensicsEngine`: `ratio=0.55`, `min_dist=36.0px`, `min_inliers=8`.

| Check | Metric | Result |
|---|---|---|
| Copy-move detection across 16 diverse real photos (120–175 px clones) | Detection Recall | **0.4375** (7/16) |
| Copy-move localization against ground-truth boxes | Mean BBox IoU / Dice | **IoU 0.6871** / **Dice 0.7941** |
| Copy-move false-alarm rate on 16 untouched real photos | False Positive Rate | **0.0** (0/16) |
| JPEG quantisation-table quality estimate ([60, 75, 85, 90, 95]) | Exact Match Rate | **5/5** |
| 8x8 JPEG block-grid alignment & crop shift check | Aligned / Shifted Detected | **True / True** |
| JPEG ghost contrast (q60 paste inside q90 host) | Mean Ghost Map Separation | **0.5977** |

## 4. Real System Performance & Reporting Benchmarks

| Stage | Latency / Output |
|---|---|
| Module 01 neural image detection (`hustvl/yolos-tiny`) | **458.0 ms** |
| Module 04 full-image forensic suite (SIFT copy-move + ELA + noise + JPEG ghosts) | **164.5 ms** |
| Cross-module spatial-temporal intelligence linking | **0.28 ms** (0 linked group) |
| Consolidated advisory PDF report generation | **46.67 ms** (4975 bytes) |

> **Advisory Notice:** Model scores and heuristic rule detections remain separate and advisory for human review. No module infers identity, intent, or legal/disciplinary conclusions.
