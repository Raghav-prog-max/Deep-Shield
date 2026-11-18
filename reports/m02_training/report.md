# Module 02 face head - training and test report

Created 2026-11-19T04:06:47. Base model `prithivMLmods/open-deepfake-detection` (frozen). Head `C:\Users\Dell\.gemini\antigravity\scratch\Deep-Shield\models\m02_face_head.npz` sha `cf00a24531d0`.
Selected head: **mlp** on **multi** features (reg 0.5, validation AUC 0.9302).
Three-band thresholds (set on validation): low **0.173**, high **0.816**.
Validation AUC: trained head 0.9302 vs original 0.5708 -> head **ACTIVATED**.

## Held-out test: original model vs trained head (same faces)

| Test set | n (fake/real) | Original AUC | Trained AUC | Original acc@0.5 | Trained acc@0.5 | Original TPR@5%FPR | Trained TPR@5%FPR | Trained coverage | Trained decided acc |
|---|---|---|---|---|---|---|---|---|---|
| pooled | 382/269 | 0.6269 | **0.937** | 0.5822 | 0.8602 | 0.1361 | 0.7435 | 0.6452 | 0.9571 |
| dff | 84/72 | 0.5319 | **0.8186** | 0.5256 | 0.7308 | 0.1548 | 0.4167 | 0.4487 | 0.9 |
| kaggle_deepfake | 98/97 | 0.5052 | **0.9028** | 0.5179 | 0.8154 | 0.0816 | 0.6429 | 0.5231 | 0.9314 |
| stylegan1 | 100/100 | 0.9603 | **0.9758** | 0.875 | 0.935 | 0.8 | 0.92 | 0.76 | 0.9737 |
| tpdne | 100/100 | 0.6051 | **0.9964** | 0.58 | 0.965 | 0.21 | 0.97 | 0.84 | 0.9821 |
| dff:EFS | 49/72 | 0.5031 | **0.8339** | 0.5785 | 0.7438 | 0.1224 | 0.4286 | 0.4545 | 0.9091 |
| dff:FS | 35/72 | 0.5722 | **0.7972** | 0.6262 | 0.7196 | 0.2 | 0.4 | 0.4486 | 0.8958 |

## Unseen generator (leave-one-generator-out)

Head retrained WITHOUT the named source, then tested on it. This is the honest estimate for a new generator.

| Held-out source | n (fake/real) | Original AUC | Head-without-it AUC | Head-without-it TPR@5%FPR |
|---|---|---|---|---|
| dff | 84/72 | 0.5319 | 0.5351 | 0.0595 |
| kaggle_deepfake | 98/97 | 0.5052 | 0.5317 | 0.0408 |
| stylegan1 | 100/100 | 0.9603 | 0.3737 | 0.01 |
| tpdne | 100/100 | 0.6051 | 0.7883 | 0.22 |

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

Notes: tpdne has fake images only; it is tested against the stylegan1 (FFHQ) real faces. dff category names are as labelled by the dataset uploader. Small test sets: treat differences below ~0.02 AUC as noise. Not a formal benchmark.