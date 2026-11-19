# DEEP SHIELD — Deep Robustness & Stress-Test Report

**Generated:** 2026-11-19T08:43:32

## 1. M01 Neural Object Detector & Multi-Object Tracker (`hustvl/yolos-tiny`)
- **Real Surveillance Video (`vtest.avi` test clip):** `completed`, **82** detections, **21** persistent tracks (max track length **8** frames), mean confidence **0.8958** (15077.1 ms).
- **Negative Non-Person False-Positive Stress Test:** **0/4** false-positive persons on non-person scenes.

## 2. M02 Face Manipulation Robustness Matrix (`prithivMLmods/open-deepfake-detection + deepshield-face-head-v4` · rev `aaa680d8ff799424ba5d092b6d5e0b999f0e0ab6+head:f12a651f67ec`)
Tested across **107 balanced real held-out faces** from all 4 Kaggle/HF sources under social-media degradations:

| Degradation Condition | Held-Out ROC-AUC | Accuracy @ 0.5 | F1 @ 0.5 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| `clean` | **0.9493** | 0.8785 | 0.8926 | 0.8571 | 0.931 |
| `jpeg_q60` | **0.9324** | 0.8131 | 0.8214 | 0.8519 | 0.7931 |
| `jpeg_q40` | **0.8543** | 0.7664 | 0.7706 | 0.8235 | 0.7241 |
| `downscale_0.5x` | **0.8666** | 0.785 | 0.8244 | 0.7397 | 0.931 |
| `gaussian_blur` | **0.8944** | 0.7757 | 0.8235 | 0.7179 | 0.9655 |
| `brightness_contrast` | **0.9391** | 0.8411 | 0.8547 | 0.8475 | 0.8621 |

- **User Face-Swap Check (`download.webp`):** `face_manipulation_indicated` · score **0.8614** (thresholds `{'low': 0.156, 'high': 0.804}`).

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
