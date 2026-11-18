# Model and event evaluation

## Consolidated Evaluation Reports

Full evaluation and threshold calibration reports are checked into the repository under `reports/`:

- **[reports/EVALUATION_REPORT.md](reports/EVALUATION_REPORT.md)** ([JSON](reports/evaluation_summary.json)) — Consolidated M01–M04 evaluation, calibrated rule thresholds, temporal event metrics, M04 localization IoU/Dice, and system latency benchmarks.
- **[reports/m02_training/report.md](reports/m02_training/report.md)** ([JSON](reports/m02_training/report.json), [CSV](reports/m02_training/test_scores.csv)) — Module 02 multi-generator face head training, validation threshold calibration, held-out test comparison against the original checkpoint, and leave-one-generator-out generalization table.

Re-run the consolidated evaluation suite at any time from the repository root:

```sh
python scripts/evaluate_all.py --out reports
```

---

## Module 02 — Multi-Generator Face Head & Held-Out Evaluation

Module 02 uses a frozen `prithivMLmods/open-deepfake-detection` (SigLIP2) backbone paired with `models/m02_face_head.npz` (SHA `c8c61b23a9d5`), trained and evaluated across three generator families (`stylegan1`, `tpdne` StyleGAN2, and `dff` DeepFakeFace swaps/diffusion):

```sh
python scripts/fetch_train_faces.py          # fetch disjoint train/test face splits into data/m02_train
python scripts/train_m02_head.py             # extract features, calibrate on validation, evaluate on held-out test
```

### Summary of Held-Out Test Results (`276` faces: `166` fake / `110` real)

| Test Split | Faces (`fake/real`) | Original Model ROC-AUC | Trained Head ROC-AUC | Trained Acc @ 0.5 | Trained TPR @ 5% FPR | Three-Band Decided Acc |
|---|---|---|---|---|---|---|
| **Pooled (All Sources)** | 166 / 110 | 0.6509 | **0.9057** | **0.8370** | **0.6506** | **0.9010** |
| `stylegan1` (FFHQ vs StyleGAN) | 60 / 60 | 0.9592 | **0.9478** | 0.9000 | 0.6500 | 0.9302 |
| `tpdne` (FFHQ vs StyleGAN2) | 59 / 60 | 0.5534 | **0.9867** | 0.9244 | 0.8983 | 0.9588 |
| `dff` (IMDB-WIKI vs Swaps/Diffusion) | 47 / 50 | 0.4932 | **0.7255** | 0.6701 | 0.3191 | 0.7500 |

For custom folder evaluation (`eval/faces/{real,fake}` or `eval/synthetic/{real,fake}`), run:

```sh
python scripts/evaluate_m02.py --data eval/faces --task face_manipulation
python scripts/evaluate_m02.py --data eval/synthetic --task synthetic_image
```

---

## Modules 01 & 03 — Video Events and Rule Calibration

`scripts/evaluate_all.py` runs a calibration sweep over `DEEPSHIELD_M01_DWELL_SECONDS` on calibration sequences and evaluates all six event types (`zone_entry`, `zone_exit`, `loitering_candidate`, `unattended_object_candidate`, `crowding_candidate`, `running_candidate`) on held-out sequences at temporal IoU >= 0.5. When deploying to a new physical camera view, re-calibrate polygon zones and dwell/speed thresholds on consented calibration footage from that camera before operational use.
