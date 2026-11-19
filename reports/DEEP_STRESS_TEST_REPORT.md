# DEEP SHIELD — Deep Robustness & Stress-Test Report

**Generated:** 2026-11-19T06:25:33

## 1. M01 Neural Object Detector & Multi-Object Tracker (`hustvl/yolos-tiny`)
- **Real Surveillance Video (`vtest.avi` test clip):** `completed`, **82** detections, **21** persistent tracks (max track length **8** frames), mean confidence **0.8958** (12486.2 ms).
- **Negative Non-Person False-Positive Stress Test:** **0/4** false-positive persons on non-person scenes.

## 2. M02 Face Manipulation Robustness Matrix (`prithivMLmods/open-deepfake-detection + deepshield-face-head-v3` · rev `aaa680d8ff799424ba5d092b6d5e0b999f0e0ab6+head:b12bda01ed0c`)
Tested across **107 balanced real held-out faces** from all 4 Kaggle/HF sources under social-media degradations:

| Degradation Condition | Held-Out ROC-AUC | Accuracy @ 0.5 | F1 @ 0.5 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| `clean` | **0.9472** | 0.8505 | 0.871 | 0.8182 | 0.931 |
| `jpeg_q60` | **0.9215** | 0.8411 | 0.8496 | 0.8727 | 0.8276 |
| `jpeg_q40` | **0.8649** | 0.757 | 0.7547 | 0.8333 | 0.6897 |
| `downscale_0.5x` | **0.8645** | 0.7383 | 0.8 | 0.6829 | 0.9655 |
| `gaussian_blur` | **0.8895** | 0.7103 | 0.7832 | 0.6588 | 0.9655 |
| `brightness_contrast` | **0.9356** | 0.8411 | 0.8595 | 0.8254 | 0.8966 |

- **User Face-Swap Check (`download.webp`):** `face_manipulation_indicated` · score **0.9266** (thresholds `{'low': 0.207, 'high': 0.777}`).

## 3. M02 Whole-Image AI Generation (`prithivMLmods/open-deepfake-detection + deepshield-synth-head-v1` · rev `aaa680d8ff799424ba5d092b6d5e0b999f0e0ab6+head:382a7a930175`)
- **`eval/synthetic/test` (16 real vs AI images):** ROC-AUC **1.0**, Acc@0.5 **1.0**, F1 **1.0**.
- **Full Held-Out `Hemg + CIFAKE` Test Set (240 images):** ROC-AUC **0.9409**, Acc@0.5 **0.875**, ECE **0.0661**, Decided Acc **0.9061**.

## 4. M03 PyTorch Temporal GRU v4 (`temporal-gru-kinematic-v4-realtracks` · `trained:ea048bac8813`, 6265 params)
- **Kinematic Stress Scenarios Passed:** **6/6** (calibrated threshold `0.35`)

| Scenario | GRU Score | Peak Speed (BL/s) | Flagged | Expected | Pass |
|---|---:|---:|---|---|---|
| `normal_slow_walk_0.6bls` | 0.0001 | 0.6 | False | False | PASS |
| `normal_brisk_walk_1.4bls` | 0.0001 | 1.4 | False | False | PASS |
| `normal_pause_and_stop` | 0.0001 | 0.545 | False | False | PASS |
| `anomalous_sustained_sprint_3.4bls` | 0.9999 | 3.4 | True | True | PASS |
| `anomalous_calm_then_burst_sprint` | 0.9999 | 3.545 | True | True | PASS |
| `anomalous_erratic_zigzag_reversals` | 0.9999 | 4.556 | True | True | PASS |

## 5. M04 Visual Forensics Stress Test (30 Diverse Real Photographs)
Parameters: `{'ratio': 0.55, 'min_dist': 36.0, 'min_inliers': 8, 'contrast': 0.01}`

| Attack / Stress Regime | Result | Rate |
|---|---|---:|
| Translation Clone (128px, JPEG q=90) | 22/30 detected | **0.7333** |
| Rotated Clone (15°, 128px) | 22/30 detected | **0.7333** |
| Scaled Clone (1.15×, 128px) | 22/30 detected | **0.7333** |
| Social-Media Recompressed Clone (JPEG q=75) | 22/30 detected | **0.7333** |
| Untouched Diverse Real Photographs (False Alarm Rate) | 0/30 false alarms | **0.0** |
