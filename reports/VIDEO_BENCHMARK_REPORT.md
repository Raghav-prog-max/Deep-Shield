# DEEP SHIELD — Real Video Benchmark Report (Celeb-DF v2, FaceForensics++/DFD & UCF-Crime)

**Generated:** 2026-11-19T08:43:04
**M02 Trained Video Model:** `prithivMLmods/open-deepfake-detection + deepshield-face-head-v4` (thresholds `{'low': 0.156, 'high': 0.804}`)
**M02 Baseline Model:** `prithivMLmods/open-deepfake-detection`

---

## 1. Module 02 — Real Deepfake Video Benchmark (40 Actor-Disjoint Held-Out `.mp4` Videos)

All videos were evaluated through the full `DeepfakeImageIntegrityEngine.analyze(clip.mp4)` video pipeline (timestamped frame sampling → YuNet face detection → multi-frame face tracking → track temporal mean aggregation). Every test video actor/identity is strictly disjoint from training.

| Video Benchmark Split | Videos Scored (`Fake/Real`) | Baseline SigLIP2 ROC-AUC | Trained Head ROC-AUC | Acc @ 0.5 | F1 @ 0.5 | ECE | Decided Acc | Mean Real Score | Mean Fake Score |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **Pooled (40 Real Videos)** | 20 / 20 | 0.5425 | **0.6125** | 0.5750 | 0.5405 | 0.2288 | 0.8000 | 0.3636 | 0.4668 |
| Celeb-DF v2 (Official Test) | 10 / 10 | 0.6200 | **0.6200** | 0.4500 | 0.1538 | 0.3377 | 0.8333 | 0.2443 | 0.2905 |
| FaceForensics++ / Google DFD | 10 / 10 | 0.5500 | **0.7200** | 0.7000 | 0.7500 | 0.1485 | 0.7500 | 0.4829 | 0.6431 |

---

## 2. Modules 01 & 03 — Real Multi-Camera CCTV Surveillance Benchmark (`UCF-Crime`)

Evaluated `hustvl/yolos-tiny` + `MultiObjectIoUTracker` (M01) and calibrated rules + PyTorch Temporal Kinematic GRU (M03) across **12 real UCF-Crime CCTV videos** (6 `Testing_Normal`, 6 real anomaly categories).

| Clip | Ground Truth | M01 Detections | M03 Tracks | GRU Max Score | GRU Flagged Tracks | M03 Events Triggered |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `Normal_Videos_345_x264.mp4` | `normal` | 57 | 10 | 0.0006 | 0 | zone_exit |
| `Normal_Videos_745_x264.mp4` | `normal` | 186 | 31 | 0.9998 | 5 | crowding_candidate, zone_entry, zone_exit |
| `Normal_Videos_878_x264.mp4` | `normal` | 0 | 1 | - | 0 | none |
| `Normal_Videos_881_x264.mp4` | `normal` | 67 | 18 | 0.0061 | 0 | zone_entry, zone_exit |
| `Normal_Videos_888_x264.mp4` | `normal` | 48 | 7 | 0.9987 | 1 | zone_exit |
| `Normal_Videos_889_x264.mp4` | `normal` | 221 | 43 | 0.9676 | 2 | crowding_candidate, running_candidate, zone_entry, zone_exit |
| `Assault018_x264.mp4` | `anomaly` | 21 | 8 | 0.9998 | 2 | none |
| `Burglary081_x264.mp4` | `anomaly` | 0 | 1 | 0.0002 | 0 | none |
| `Fighting023_x264.mp4` | `anomaly` | 2 | 1 | 0.0150 | 0 | none |
| `Robbery089_x264.mp4` | `anomaly` | 35 | 9 | 0.9549 | 1 | zone_entry, zone_exit |
| `Shoplifting017_x264.mp4` | `anomaly` | 21 | 7 | 0.0006 | 0 | none |
| `Stealing095_x264.mp4` | `anomaly` | 7 | 6 | 0.0002 | 0 | zone_entry |

> **Honest CCTV Note:** M01 + M03 operate on person/bag kinematic tracks only (no violence/weapon/intent classification). Low-resolution CCTV angles, rapid camera cuts, or non-pedestrian anomalies (e.g., distant indoor shoplifting or occlusion) do not always exhibit high-speed running kinematics; every alert remains advisory for human review.
