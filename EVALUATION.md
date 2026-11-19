# Model and Event Evaluation

## Consolidated Evaluation, Real Video Benchmark & Stress-Test Reports

All training, real-video benchmark, and deep stress-test reports are checked into the repository under `reports/`:

- **[reports/VIDEO_BENCHMARK_REPORT.md](reports/VIDEO_BENCHMARK_REPORT.md)** ([JSON](reports/video_benchmark.json), [Video Manifest](reports/real_video_datasets_manifest.json)) — End-to-end real video evaluation across **40 actor-disjoint held-out deepfake `.mp4` videos** (20 `Celeb-DF v2` + 20 `FaceForensics++/Google DFD`) and **12 real `UCF-Crime` CCTV surveillance `.mp4` videos** (6 Normal + 6 Anomaly).
- **[reports/DEEP_STRESS_TEST_REPORT.md](reports/DEEP_STRESS_TEST_REPORT.md)** ([JSON](reports/deep_stress_test.json)) — Social-media degradation matrix (JPEG q60/q40, 0.5× downscale, Gaussian blur, brightness/contrast), negative non-person M01 stress tests, M02 whole-image AI check, M03 6-scenario kinematic GRU tests, and M04 30-photo rotation/scale/recompression copy-move tests.
- **[reports/m02_training/report.md](reports/m02_training/report.md)** ([JSON](reports/m02_training/report.json), [CSV](reports/m02_training/test_scores.csv)) — Module 02 **6-dataset** real image + video face head (`deepshield-face-head-v4`) training, Guo temperature calibration (`T=1.9768`, residual blend `alpha=0.1`), 966-face held-out test breakdown, and 6-source leave-one-generator-out table.
- **[reports/m02_synth_training/report.md](reports/m02_synth_training/report.md)** ([JSON](reports/m02_synth_training/report.json)) — Module 02 whole-image AI-generation head (`deepshield-synth-head-v1`) on `Hemg` + `CIFAKE`.
- **[reports/EVALUATION_REPORT.md](reports/EVALUATION_REPORT.md)** ([JSON](reports/evaluation_summary.json)) & **[reports/m03_gru_training/report.json](reports/m03_gru_training/report.json)** — Consolidated M01–M04 rule calibration, 5-fold cross-track CV on 53 real `vtest.avi` pedestrian tracks, M04 SIFT/RANSAC calibration, and latency benchmarks.

Reproduce from the repository root:

```sh
python scripts/fetch_real_videos.py
python scripts/train_m02_head.py --data <m02_train_dir>
python scripts/benchmark_real_videos.py
python scripts/deep_stress_test.py
python scripts/evaluate_all.py --out reports
```

---

## Module 02 — 6-Dataset Face Manipulation Head (`deepshield-face-head-v4`)

Module 02 pairs frozen `prithivMLmods/open-deepfake-detection` (SigLIP2) with `models/m02_face_head.npz` (SHA `f12a651f67ec`, 5-member per-layer L2-norm MLP ensemble, `alpha=0.1`, Guo `T=1.9768`, thresholds `0.156 / 0.804`), trained with video-group-disjoint validation on **3,504 real items (`3,283` faces)** across **6 real datasets** (4 real image datasets + pair-matched `Celeb-DF v2` videos + actor-disjoint `FaceForensics++/Google DFD` videos).

### Held-Out Frame/Image Test (`966` faces: `542` fake / `424` real, ECE `0.0490`)

| Test Split | Faces (`fake/real`) | Original SigLIP2 AUC | v4 Head AUC | Acc @ 0.5 | Three-Band Decided Acc | LOO Unseen AUC |
|---|---|---|---|---|---|---|
| **Pooled (All 6 Real Sources)** | 542 / 424 | 0.5750 | **0.8561** | **76.29%** | **92.86%** | — |
| `tpdne` (StyleGAN2) | 100 / 100 | 0.6051 | **0.9952** | 94.50% | 98.69% | **0.8137** |
| `stylegan1` (Kaggle 140k) | 100 / 100 | 0.9603 | **0.9802** | 93.50% | 98.48% | 0.5417 |
| `kaggle_deepfake` (Kaggle 190k) | 98 / 97 | 0.5052 | **0.8835** | 81.54% | 90.36% | 0.5120 |
| `dff` (Swaps / Diffusion) | 84 / 72 | 0.5319 | **0.8519** | 76.92% | 90.57% | 0.5855 |
| `dfd_video` (FF++/Google DFD, unseen actors) | 80 / 75 | 0.5358 | **0.7182** | 64.52% | 85.29% | **0.7410** |
| `celebdf_video` (Celeb-DF v2, unseen actors) | 80 / 80 | 0.5892 | **0.6238** | 44.37% | 76.00% | 0.4559 |

### Full M02 Video Pipeline Benchmark (`40` Actor-Disjoint Held-Out `.mp4` Videos)

| Real Video Benchmark | Videos (`Fake/Real`) | Baseline SigLIP2 AUC | v4 Video AUC | Acc @ 0.5 | Recall @ 0.5 | Decided Acc |
|---|---|---|---|---|---|---|
| **Pooled (40 Real Videos)** | 20 / 20 | 0.5425 | **0.6125** | 57.50% | 50.00% | **80.00%** |
| **FaceForensics++ / Google DFD** | 10 / 10 | 0.5500 | **0.7200** | **70.00%** | **90.00%** | **75.00%** |
| **Celeb-DF v2 (Pair-Matched Unseen Actors)** | 10 / 10 | 0.6200 | **0.6200** | 45.00% | 10.00% | **83.33%** |

*Social-Media Degradation Stress Test (`107` held-out faces):* Clean **`0.9493`** · Brightness/Contrast **`0.9391`** · JPEG q60 **`0.9324`** · Gaussian Blur **`0.8944`** · Downscale 0.5× **`0.8666`** · JPEG q40 **`0.8543`**.

---

## Module 02 — Whole-Image AI Generation (`deepshield-synth-head-v1`)

Runs **by default on images** (`models/m02_synth_head.npz`, SHA `382a7a930175`, 5-member L2-norm MLP ensemble, `T=0.8047`, thresholds `0.406 / 0.750`), trained on **960** real AI-vs-Photo images (`Hemg` + `CIFAKE`):

| Test Split | Images (`AI/Real`) | Baseline ROC-AUC | Trained Head ROC-AUC | Acc @ 0.5 | F1 @ 0.5 | Decided Acc |
|---|---|---|---|---|---|---|
| **Pooled (`Hemg + CIFAKE`)** | 120 / 120 | 0.4741 | **0.9409** | **87.50%** | **0.8760** | **90.61%** |
| `CIFAKE` | 40 / 40 | 0.4606 | **0.9600** | 87.50% | 0.8684 | 91.04% |
| `Hemg` | 80 / 80 | 0.4747 | **0.9325** | 87.50% | 0.8795 | 90.41% |

---

## Modules 01, 03 & 04 — Surveillance Detection, Temporal GRU & Visual Forensics

- **M01 (`hustvl/yolos-tiny` + `MultiObjectIoUTracker`)**: Real `vtest.avi` pedestrian clip (`82` detections, `21` tracks, mean conf `0.8958`, `0/4` negative non-person FPs), `48/48` held-out rule events (`F1 1.000`), and live multi-camera execution across **12 real UCF-Crime CCTV clips** ([VIDEO_BENCHMARK_REPORT.md](reports/VIDEO_BENCHMARK_REPORT.md)).
- **M03 (`temporal-gru-kinematic-v4-realtracks`)**: 5-fold cross-track CV on 53 real `vtest.avi` pedestrian tracks (`CV AUC 1.0000`, `ECE 0.0001`, `0/11` false alarms on unperturbed held-out walks, `6/6` kinematic stress scenarios passed). On uncalibrated multi-camera UCF-Crime CCTV clips, M03 flags rapid-motion tracks (`Assault018`, `Robbery089`) while also transparently demonstrating perspective-induced false alerts on busy foreground crossings — confirming why site-specific calibration is mandatory before operational CCTV use.
- **M04 (`models/m04_forensics_params.json`)**: `ratio=0.55`, `min_dist=36`, `min_inliers=8`, `contrast=0.01`, `ela_quality=90`. 30-photo stress test: `0/30 (0.0%)` untouched false alarms, `22/30 (73.33%)` recall across translation, 15° rotation, 1.15× scale, and JPEG q75, plus `5/5` exact JPEG quality recovery.
