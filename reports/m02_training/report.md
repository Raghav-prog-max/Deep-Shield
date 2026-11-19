# Module 02 face head - deep training and test report

Created 2026-11-19T06:04:28. Base model `prithivMLmods/open-deepfake-detection` (frozen). Head `C:\Users\Dell\.gemini\antigravity\scratch\Deep-Shield\models\m02_face_head.npz` sha `b12bda01ed0c`.
Selected head: **mlp** on **multi** features (l2_norm=True, reg=0.05, validation AUC=0.9338).
Calibration (validation): base_logit_alpha **0.0**, temperature T **1.6**, val ECE **0.0545**, val Brier **0.0992**.
Three-band thresholds (set on validation): low **0.207**, high **0.777**.
Validation AUC: trained head 0.9338 vs original 0.5708 -> head **ACTIVATED**.

## Held-out test: original model vs trained head (same faces)

| Test set | n (fake/real) | Original AUC | Trained AUC | Acc@0.5 | TPR@5%FPR | ECE | Brier | Coverage | Decided Acc |
|---|---|---|---|---|---|---|---|---|---|
| pooled | 382/269 | 0.6269 | **0.9439** | 0.8725 | 0.7356 | 0.0306 | 0.0929 | 0.725 | 0.9449 |
| dff | 84/72 | 0.5319 | **0.8457** | 0.7692 | 0.369 | 0.0649 | 0.1609 | 0.5897 | 0.8587 |
| kaggle_deepfake | 98/97 | 0.5052 | **0.9045** | 0.8103 | 0.6327 | 0.074 | 0.1296 | 0.6 | 0.9231 |
| stylegan1 | 100/100 | 0.9603 | **0.9827** | 0.95 | 0.97 | 0.0701 | 0.0492 | 0.82 | 0.9756 |
| tpdne | 100/100 | 0.6051 | **0.9969** | 0.965 | 1.0 | 0.067 | 0.0357 | 0.885 | 0.9831 |
| dff:EFS | 49/72 | 0.5031 | **0.8679** | 0.7769 | 0.3878 | 0.0705 | 0.147 | 0.5868 | 0.8873 |
| dff:FS | 35/72 | 0.5722 | **0.8147** | 0.7664 | 0.3429 | 0.1142 | 0.1662 | 0.5981 | 0.8281 |

## Unseen generator (leave-one-generator-out)

Head retrained WITHOUT the named source, then tested on it. This is the honest estimate for a new generator.

| Held-out source | n (fake/real) | Original AUC | Head-without-it AUC | Head-without-it TPR@5%FPR |
|---|---|---|---|---|
| dff | 84/72 | 0.5319 | **0.5231** | 0.0714 |
| kaggle_deepfake | 98/97 | 0.5052 | **0.4778** | 0.051 |
| stylegan1 | 100/100 | 0.9603 | **0.4778** | 0.01 |
| tpdne | 100/100 | 0.6051 | **0.7891** | 0.32 |

## Data

| source/split/label | faces used |
|---|---|
| dff/test/fake | 84 |
| dff/test/real | 72 |
| dff/train/fake | 243 |
| dff/train/real | 208 |
| kaggle_deepfake/test/fake | 98 |
| kaggle_deepfake/test/real | 97 |
| kaggle_deepfake/train/fake | 299 |
| kaggle_deepfake/train/real | 296 |
| stylegan1/test/fake | 100 |
| stylegan1/test/real | 100 |
| stylegan1/train/fake | 300 |
| stylegan1/train/real | 300 |
| tpdne/test/fake | 100 |
| tpdne/train/fake | 299 |

No face detected (skipped, as the engine would): {'tpdne/train': 1, 'dff/train': 149, 'dff/test': 44, 'kaggle_deepfake/train': 5, 'kaggle_deepfake/test': 5}. Test images identical to a train image removed: 0.