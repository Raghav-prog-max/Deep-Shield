# Model and Event Evaluation

## Consolidated Evaluation & Stress-Test Reports

All evaluation, training, and deep stress-test reports are checked into the repository under `reports/`:

- **[reports/EVALUATION_REPORT.md](reports/EVALUATION_REPORT.md)** ([JSON](reports/evaluation_summary.json)) — Consolidated M01–M04 evaluation, real surveillance video inference (`hustvl/yolos-tiny` + M03 Temporal GRU v4), calibrated rule thresholds, M04 SIFT/RANSAC parameter sweep & localization IoU/Dice, and system latency benchmarks.
- **[reports/DEEP_STRESS_TEST_REPORT.md](reports/DEEP_STRESS_TEST_REPORT.md)** ([JSON](reports/deep_stress_test.json)) — Deep stress test across all four engines: M01 real video + negative non-person scenes, M02 social-media degradation matrix (JPEG q60/q40, 0.5× downscale, Gaussian blur, brightness/contrast), M02 whole-image AI detection, M03 6-scenario kinematic GRU verification, and M04 30-photo rotation/scale/recompression copy-move test.
- **[reports/m02_training/report.md](reports/m02_training/report.md)** ([JSON](reports/m02_training/report.json), [CSV](reports/m02_training/test_scores.csv)) — Module 02 4-dataset face head (`deepshield-face-head-v3`) training, Guo temperature calibration (`T=1.6`), held-out test breakdown across 651 faces, and leave-one-generator-out generalization table.
- **[reports/m02_synth_training/report.md](reports/m02_synth_training/report.md)** ([JSON](reports/m02_synth_training/report.json)) — Module 02 whole-image AI-generation head (`deepshield-synth-head-v1`) training and held-out evaluation on 240 real AI-vs-Photo images (`Hemg` + `CIFAKE`).
- **[reports/m03_gru_training/report.json](reports/m03_gru_training/report.json)** — Module 03 PyTorch Temporal Kinematic GRU (`temporal-gru-kinematic-v4-realtracks`, 6,265 params) 5-fold cross-track CV on 53 real pedestrian tracks from `vtest.avi`.

Re-run the consolidated evaluation and stress tests from the repository root:

```sh
python scripts/evaluate_all.py --out reports
python scripts/deep_stress_test.py
```

---

## Module 02 — Face Manipulation Head (`deepshield-face-head-v3`)

Module 02 pairs a frozen `prithivMLmods/open-deepfake-detection` (SigLIP2) backbone with `models/m02_face_head.npz` (SHA `b12bda01ed0c`), trained on **2,800 real images (`2,596` detected faces)** from four real Kaggle/HuggingFace datasets (`TheKernel01/140k-Real-and-Fake-Faces`, `javi22/this-person-does-not-exist-10k`, `juhi-neeved/deepfakeface-labeled`, `JamieWithofs/Deepfake-and-real-images`):

```sh
python scripts/fetch_train_faces.py          # fetch disjoint real/fake face splits
python scripts/train_m02_head.py             # extract multi-layer features, calibrate (Guo T), evaluate on held-out test
```

### Held-Out Test Results (`651` faces: `382` fake / `269` real, thresholds `0.207 / 0.777`, ECE `0.0306`)

| Test Split | Faces (`fake/real`) | Original Model ROC-AUC | Trained Head ROC-AUC | Trained Acc @ 0.5 | Trained F1 @ 0.5 | Three-Band Decided Acc |
|---|---|---|---|---|---|---|
| **Pooled (All 4 Sources)** | 382 / 269 | 0.6269 | **0.9439** | **87.25%** | **0.8935** | **94.49%** |
| `tpdne` (FFHQ vs StyleGAN2) | 100 / 100 | 0.6051 | **0.9969** | 96.50% | 0.9662 | 98.31% |
| `stylegan1` (FFHQ vs StyleGAN) | 100 / 100 | 0.9603 | **0.9827** | 95.00% | 0.9510 | 97.56% |
| `kaggle_deepfake` (Kaggle Deepfake) | 98 / 97 | 0.5052 | **0.9045** | 81.03% | 0.8246 | 92.31% |
| `dff` (IMDB-WIKI vs Swaps/Diffusion) | 84 / 72 | 0.5319 | **0.8457** | 76.92% | 0.7805 | 85.87% |
| `dff:EFS` (Easy Face Swap) | 49 / 72 | 0.5031 | **0.8679** | 77.69% | 0.7379 | 88.73% |
| `dff:FS` (Face Swap) | 35 / 72 | 0.5722 | **0.8147** | 76.64% | 0.6753 | 82.81% |

**Honest Unseen-Generator Caveat (Leave-One-Generator-Out ROC-AUC):** `tpdne` `0.7891`, `dff` `0.5231`, `kaggle_deepfake` `0.4778`, `stylegan1` `0.4778`. Cross-generator generalization to unseen architectures remains an open challenge; always pair M02 outputs with M04 visual forensics and human review.

---

## Module 02 — Whole-Image AI-Generation Head (`deepshield-synth-head-v1`)

`synthetic_image` runs **by default on images** using `models/m02_synth_head.npz` (SHA `382a7a930175`, 5-member L2-norm MLP ensemble, `T=0.8047`, thresholds `0.406 / 0.75`), trained on **960** real AI-vs-Photo images from `Hemg/AI-Generated-vs-Real-Images-Datasets` and `dragonintelligence/CIFAKE-image-dataset`:

| Test Split | Images (`AI/Real`) | Baseline ROC-AUC | Trained Head ROC-AUC | Trained Acc @ 0.5 | Trained F1 @ 0.5 | Three-Band Decided Acc |
|---|---|---|---|---|---|---|
| **Pooled (`Hemg + CIFAKE`)** | 120 / 120 | 0.4741 | **0.9409** | **87.50%** | **0.8760** | **90.61%** |
| `CIFAKE` | 40 / 40 | 0.4606 | **0.9600** | 87.50% | 0.8684 | 91.04% |
| `Hemg` | 80 / 80 | 0.4747 | **0.9325** | 87.50% | 0.8795 | 90.41% |

For custom folder evaluation (`eval/faces/{real,fake}` or `eval/synthetic/{real,fake}`), run:

```sh
python scripts/evaluate_m02.py --data eval/faces/test --task face_manipulation
python scripts/evaluate_m02.py --data eval/synthetic/test --task synthetic_image
```

---

## Modules 01, 03 & 04 — Surveillance Detection, Temporal GRU & Visual Forensics

- **M01 (`hustvl/yolos-tiny` + `MultiObjectIoUTracker`)**: Evaluated on real OpenCV `vtest.avi` pedestrian surveillance segments (`82` detections, `21` unique tracks, mean conf `0.8958`, `0/4` negative non-person false alarms) and 6+6 calibration/held-out event sequences (`48/48` events matched at temporal IoU >= 0.5).
- **M03 (`temporal-gru-kinematic-v4-realtracks`)**: 5-fold cross-track CV on 53 real pedestrian tracks (`32 train / 10 val / 11 test`), CV AUC `1.0000`, ECE `0.0001`, `0/11` false alarms on unperturbed held-out walks, `6/6` kinematic stress scenarios passed.
- **M04 (`models/m04_forensics_params.json`)**: `ratio=0.55`, `min_dist=36`, `min_inliers=8`, `contrast=0.01`, `ela_quality=90`. 30-photo stress test: `0/30 (0.0%)` untouched false alarms, `22/30 (73.33%)` recall across translation, 15° rotation, 1.15× scale, and JPEG q75, plus `5/5` exact JPEG quality table estimation.
