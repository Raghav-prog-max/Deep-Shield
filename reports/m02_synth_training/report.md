# Module 02 `synthetic_image` Head — Training & Held-Out Test Report

Created 2026-11-19T06:25:00. Base `prithivMLmods/open-deepfake-detection` (frozen). Head `models/m02_synth_head.npz` sha `382a7a930175`.
Validation AUC: **0.9807** vs baseline `0.4389`. Thresholds: low **0.406**, high **0.75**.

| Test set | n (AI/Real) | Baseline AUC | Trained AUC | Acc@0.5 | TPR@5%FPR | ECE | Decided Acc |
|---|---|---|---|---|---|---|---|
| pooled | 120/120 | 0.4741 | **0.9409** | 0.875 | 0.6833 | 0.0661 | 0.9061 |
| cifake | 40/40 | 0.4606 | **0.96** | 0.875 | 0.775 | 0.0862 | 0.9104 |
| hemg | 80/80 | 0.4747 | **0.9325** | 0.875 | 0.7125 | 0.0646 | 0.9041 |
