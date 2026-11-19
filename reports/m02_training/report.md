# Module 02 face head - deep training and test report

Created 2026-11-19T08:26:59. Base model `prithivMLmods/open-deepfake-detection` (frozen). Head `C:\Users\Dell\.gemini\antigravity\scratch\Deep-Shield\models\m02_face_head.npz` sha `f12a651f67ec`.
Selected head: **mlp_ens** on **multi** features (l2_norm=True, reg=0.1, validation AUC=0.8875).
Calibration (validation): base_logit_alpha **0.1**, temperature T **1.9768**, val ECE **0.057**, val Brier **0.1337**.
Three-band thresholds (set on validation): low **0.156**, high **0.804**.
Validation AUC: trained head 0.8876 vs original 0.5575 -> head **ACTIVATED**.

## Held-out test: original model vs trained head (same faces)

| Test set | n (fake/real) | Original AUC | Trained AUC | Acc@0.5 | TPR@5%FPR | ECE | Brier | Coverage | Decided Acc |
|---|---|---|---|---|---|---|---|---|---|
| pooled | 542/424 | 0.575 | **0.8561** | 0.7629 | 0.5461 | 0.049 | 0.1554 | 0.4638 | 0.9286 |
| celebdf_video | 80/80 | 0.5892 | **0.6238** | 0.4437 | 0.0125 | 0.3101 | 0.3139 | 0.3125 | 0.76 |
| dfd_video | 80/75 | 0.5358 | **0.7182** | 0.6452 | 0.325 | 0.0835 | 0.2157 | 0.2194 | 0.8529 |
| dff | 84/72 | 0.5319 | **0.8519** | 0.7692 | 0.3571 | 0.1074 | 0.1591 | 0.3397 | 0.9057 |
| kaggle_deepfake | 98/97 | 0.5052 | **0.8835** | 0.8154 | 0.551 | 0.084 | 0.143 | 0.4256 | 0.9036 |
| stylegan1 | 100/100 | 0.9603 | **0.9802** | 0.935 | 0.91 | 0.1048 | 0.0636 | 0.66 | 0.9848 |
| tpdne | 100/100 | 0.6051 | **0.9952** | 0.945 | 0.98 | 0.1032 | 0.0474 | 0.765 | 0.9869 |
| dff:EFS | 49/72 | 0.5031 | **0.8625** | 0.8017 | 0.3673 | 0.0955 | 0.1482 | 0.3388 | 0.9024 |
| dff:FS | 35/72 | 0.5722 | **0.8369** | 0.7477 | 0.3429 | 0.1246 | 0.1588 | 0.3738 | 0.9 |

## Unseen generator (leave-one-generator-out)

Head retrained WITHOUT the named source, then tested on it. This is the honest estimate for a new generator.

| Held-out source | n (fake/real) | Original AUC | Head-without-it AUC | Head-without-it TPR@5%FPR |
|---|---|---|---|---|
| celebdf_video | 80/80 | 0.5892 | **0.4559** | 0.0125 |
| dfd_video | 80/75 | 0.5358 | **0.741** | 0.275 |
| dff | 84/72 | 0.5319 | **0.5855** | 0.0595 |
| kaggle_deepfake | 98/97 | 0.5052 | **0.512** | 0.0408 |
| stylegan1 | 100/100 | 0.9603 | **0.5417** | 0.0 |
| tpdne | 100/100 | 0.6051 | **0.8137** | 0.36 |

## Data

| source/split/label | faces used |
|---|---|
| celebdf_video/test/fake | 80 |
| celebdf_video/test/real | 80 |
| celebdf_video/train/fake | 96 |
| celebdf_video/train/real | 96 |
| dfd_video/test/fake | 80 |
| dfd_video/test/real | 75 |
| dfd_video/train/fake | 96 |
| dfd_video/train/real | 84 |
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

No face detected (skipped, as the engine would): {'tpdne/train': 1, 'dff/train': 149, 'dff/test': 44, 'kaggle_deepfake/train': 5, 'kaggle_deepfake/test': 5, 'dfd_video/test': 5, 'dfd_video/train': 12}. Test images identical to a train image removed: 0.