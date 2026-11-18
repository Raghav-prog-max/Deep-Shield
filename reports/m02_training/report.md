# Module 02 face head - training and test report

Created 2026-11-19T00:53:21. Base model `prithivMLmods/open-deepfake-detection` (frozen). Head `models/m02_face_head.npz` sha `c8c61b23a9d5`.
Selected head: **mlp** on **multi** features (reg 0.001, validation AUC 0.8616).
Three-band thresholds (set on validation): low **0.02**, high **0.978**.
Validation AUC: trained head 0.8616 vs original 0.5384 -> head **ACTIVATED**.

## Held-out test: original model vs trained head (same faces)

| Test set | n (fake/real) | Original AUC | Trained AUC | Original acc@0.5 | Trained acc@0.5 | Original TPR@5%FPR | Trained TPR@5%FPR | Trained coverage | Trained decided acc |
|---|---|---|---|---|---|---|---|---|---|
| pooled | 166/110 | 0.6509 | **0.9057** | 0.5761 | 0.837 | 0.2651 | 0.6506 | 0.6957 | 0.901 |
| dff | 47/50 | 0.4932 | **0.7255** | 0.5052 | 0.6701 | 0.1277 | 0.3191 | 0.5361 | 0.75 |
| stylegan1 | 60/60 | 0.9592 | **0.9478** | 0.85 | 0.9 | 0.7833 | 0.65 | 0.7167 | 0.9302 |
| tpdne | 59/60 | 0.5534 | **0.9867** | 0.5462 | 0.9244 | 0.1356 | 0.8983 | 0.8151 | 0.9588 |
| dff:EFS | 25/50 | 0.5336 | **0.8272** | 0.6 | 0.76 | 0.16 | 0.36 | 0.52 | 0.8462 |
| dff:FS | 22/50 | 0.4473 | **0.61** | 0.5556 | 0.6111 | 0.0909 | 0.2727 | 0.5833 | 0.7143 |

## Unseen generator (leave-one-generator-out)

Head retrained WITHOUT the named source, then tested on it. This is the honest estimate for a new generator.

| Held-out source | n (fake/real) | Original AUC | Head-without-it AUC | Head-without-it TPR@5%FPR |
|---|---|---|---|---|
| dff | 47/50 | 0.4932 | 0.5038 | 0.0426 |
| stylegan1 | 60/60 | 0.9592 | 0.275 | 0.0 |
| tpdne | 59/60 | 0.5534 | 0.6737 | 0.1186 |

## Data

| source/split/label | faces used |
|---|---|
| dff/test/fake | 47 |
| dff/test/real | 50 |
| dff/train/fake | 120 |
| dff/train/real | 117 |
| stylegan1/test/fake | 60 |
| stylegan1/test/real | 60 |
| stylegan1/train/fake | 150 |
| stylegan1/train/real | 150 |
| tpdne/test/fake | 59 |
| tpdne/train/fake | 149 |

No face detected (skipped, as the engine would): {'tpdne/train': 1, 'dff/train': 63, 'dff/test': 23}. Test images identical to a train image removed: 0.

Notes: tpdne has fake images only; it is tested against the stylegan1 (FFHQ) real faces. dff category names are as labelled by the dataset uploader. Small test sets: treat differences below ~0.02 AUC as noise. Not a formal benchmark.