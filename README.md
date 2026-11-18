# DEEP SHIELD · Modules 01–04

DEEP SHIELD analyzes images and video for human review. Modules 01–04 return findings in the shared `ModuleResult` schema. Outputs are advisory; detector confidence, classifier scores, forensic indicators and rule events remain separate.

| Module | Capability | Current implementation |
|---|---|---|
| 01 | Threat and anomaly intelligence | YOLO people/bag detection, ByteTrack, zone entry, dwell and unattended-object candidates |
| 02 | Deepfake and image integrity | Face manipulation analysis, frequency plots, ELA/noise maps and metadata |
| 03 | Video event understanding | Reuses M01 tracks for zone entry/exit, loitering, crowding and rapid-movement candidates |
| 04 | Visual forensics | Copy-move, ELA/noise, JPEG observations and metadata |

No module recognizes identity or infers intent or wrongdoing. M01/M03 rule thresholds and M02 classifier thresholds require evaluation on consented, representative footage before operational use.

This project is licensed under AGPL-3.0-only. Ultralytics code and YOLO26n weights use AGPL-3.0 by default; see [LICENSE](LICENSE) and [Ultralytics licensing](https://www.ultralytics.com/license) before private or commercial deployment.

| Acceptance criterion (proposal §9) | How Module 02 meets it |
|---|---|
| Deepfake detection returns a task-specific classification and score | Two separate tasks, each with its own model, labels, thresholds and verdict vocabulary (table below) |
| Image and video uploads work | `POST /api/v1/m02/analyze` accepts jpg/png/webp/bmp/tiff and mp4/avi/mov/mkv/webm |
| Temporal reasoning operates across multiple video frames | Face tracks across sampled frames, plus per-track mean/median/max/std, segments above threshold and a score timeline |
| Suspicious-region map (§4.2 "when supported") | Occlusion-sensitivity heatmap per face (`region_heatmap`), top region mapped to image pixels |
| Spatial/frequency feature analysis (§4.2) | `frequency_analysis`: spectral fingerprint + plot per face crop |
| General image-manipulation analysis, separate task (§4.2) | `image_manipulation`: ELA + noise-level maps + EXIF review, labelled exploratory (§10) |
| Cross-module intelligence can link findings | Every finding carries `media_id`, `frame_index`, `timestamp_s`, `track_id`, `region` and `score_type` |
| One unified report | `ModuleResult.task_summaries` gives the headline per task; `findings` carries the full evidence for the report generator |

---

## 1. What it does

| Task | Input to model | Default model (Hugging Face) | Licence | Verdicts |
|---|---|---|---|---|
| `face_manipulation` | Square face crop (detected face + 25% context) | [`prithivMLmods/open-deepfake-detection`](https://huggingface.co/prithivMLmods/open-deepfake-detection) — SigLIP2, labels `Fake`/`Real`; thresholds 0.35 / 0.65 | Apache-2.0 | `face_manipulation_indicated` · `inconclusive` · `no_face_manipulation_indicated` |
| `synthetic_image` (**experimental**) | Whole image / video keyframe | Same checkpoint; wide inconclusive band 0.25 / 0.85 | Apache-2.0 | `ai_generation_indicated` · `inconclusive` · `no_ai_generation_indicated` |

**Model selection (internal check, 17 Nov 2026, 25 real + 25 fake each — not a formal benchmark):**

| Candidate | Face set (FFHQ vs StyleGAN) ROC-AUC | Web image set (art/photos) ROC-AUC |
|---|---|---|
| `prithivMLmods/open-deepfake-detection` (selected) | **0.96** (20/25 fakes caught, 1/25 false alarm) | 0.48 |
| `dima806/deepfake_vs_real_image_detection` | 0.50 | — |
| `desireemcv/vit-face-real-vs-fake` | 0.52 | — |
| `NYUAD-ComNets/NYUAD_AI-generated_images_detector` | — | 0.58 |
| Face localisation | Frame | OpenCV **YuNet** (OpenCV Zoo); falls back to the OpenCV Haar cascade | Apache-2.0 / BSD | — |

### Task overview (v0.2)

| Task | Default | Method | Output type |
|---|---|---|---|
| `face_manipulation` | images + videos | Face detection → SigLIP2 classifier → **occlusion heatmap** for faces scoring ≥ 0.35 | Verdict (three-band) + score + region map |
| `frequency_analysis` | images | Windowed FFT → azimuthal power spectrum → high-frequency excess vs a 1/f fit (Durall et al., CVPR 2020) | **Spectrum plot only**: the indicator scored ROC-AUC 0.52 on our 50-face check, so no score or verdict is reported |
| `image_manipulation` | images | ELA (JPEG q90 re-compression) + local noise-level map (Immerkær kernel, ratio to image median) + EXIF review | **Visual maps only**, no score, no verdict |
| `synthetic_image` | opt-in | Same classifier on the whole image | Experimental; capped at "inconclusive" on video |

Why `image_manipulation` gives no verdict: in our checks, automatic block-outlier flagging missed real pastes and flagged untouched textured photos. The visual maps do make edits visible to a reviewer: a pasted region with a different noise level shows as a solid red or blue block, and re-compressed pastes stand out in ELA. Following proposal §10, they are therefore labelled exploratory visualizations.

Evidence files per image (under `evidence_store/m02/<media_id>/`): `faces_annotated.jpg`, `face_F1.jpg`, `face_F1_heatmap.jpg`, `spectrum_F1.png`, `integrity_panel.jpg`, `ela_map.jpg`, `noise_map.jpg`. Per video: annotated frames, face crops, `T1_heatmap.jpg`, `face_score_timeline.png`.

**Three-band decision.** score ≥ `high` (0.65) → indicated; score ≤ `low` (0.35) → not indicated; otherwise → **inconclusive**. Thresholds are per task and can be overridden with environment variables. Calibrate them on your held-out set (§6) before the demo.

**Localised tampering is not handled here.** Splicing and copy-move tampering, along with ELA, noise and FFT maps, belong to **Module 04** (Visual Forensics). This keeps deepfake, synthetic-image and tampering outputs separate, as the proposal requires.

### Image pipeline
1. Decode the image and apply EXIF orientation. Compute a SHA-256 hash; `media_id` is its first 16 hex characters.
2. Detect faces (≥ 48 px). Crop each face and classify it, producing one `image`-level finding per face (F1, F2 …).
3. Classify the whole image for AI generation, producing one `image`-level finding.
4. Save evidence: face crops, the annotated image with colour-coded boxes, and the assessed input.

### Video pipeline
1. Sample frames at 2 fps (up to 64 frames or 120 s) and record exact `frame_index` and `timestamp_s`.
2. Detect faces in each sampled frame, then classify every crop in batches.
3. **Temporal reasoning:** link faces across frames by IoU to form tracks (`m02-face-T1` …). Each track gets a `TemporalSummary` with mean, median, max, std, the fraction of frames above `high`, the time **segments above `high`** and the score series.
4. Track verdict = three-band(mean). Tracks with fewer than 3 observations are forced to `inconclusive`.
5. Findings: one `track`-level finding per face track, plus the top-3 `frame`-level findings per track as evidence.
6. Classify 6 evenly spaced keyframes for AI generation. This produces `frame` findings plus one `video`-level aggregate.
7. Save evidence: annotated frames, face crops and `face_score_timeline.png`.

---

## 2. Setup (demo machine or DGX node)

**Windows (PowerShell)** — run inside the extracted `deep_shield_m02` folder:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass        # only if activation is blocked
.\.venv\Scripts\Activate.ps1
pip install torch
pip install -r requirements.txt
python scripts\download_models.py                # M02; ~700 MB, pinned revisions + SHA-256
python scripts\download_m01_model.py             # M01/M03 YOLO26n checkpoint + SHA-256
$env:DEEPSHIELD_TEST_REAL = "1"; pytest -q        # wiring tests + real-model smoke test
python scripts\run_m02.py tests\assets\face_sample.jpg
```

**Linux / DGX (bash)**

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # or /whl/cpu
pip install -r requirements.txt
python scripts/download_models.py
python scripts/download_m01_model.py
DEEPSHIELD_TEST_REAL=1 pytest -q
```

After the download, the engine loads from `./models` and works **offline**. This matters for a live demo on campus Wi-Fi.

## 3. Run

```bash
# CLI
python scripts/run_m02.py samples/face.jpg
python scripts/run_m02.py samples/clip.mp4 --out clip_m02.json

# API (standalone, for module testing)
uvicorn deep_shield.api.main:app --host 0.0.0.0 --port 8002     # Modules 01–04 at /docs
curl -F file=@samples/face.jpg -F tasks=face_manipulation,synthetic_image \
     http://localhost:8002/api/v1/m02/analyze
```

| Endpoint | Purpose |
|---|---|
| `GET  /api/v1/m02/health` | Model ids, revisions, device, thresholds, face-detector backend |
| `POST /api/v1/m02/analyze` | multipart `file`, `tasks` (comma list; empty = defaults), `assume_face_crop` (image only) → `ModuleResult` |
| `GET  /api/v1/m02/evidence/{media_id}/{file}` | Serve evidence images (path-traversal safe) |
| `DELETE /api/v1/m02/evidence/{media_id}` | Retention control: delete all derived evidence for one item |

Uploads are deleted after analysis by default (`DEEPSHIELD_KEEP_UPLOADS=1` keeps them).

## 4. Integration with the DEEP SHIELD backend

```python
# main API app
from deep_shield.api.routes_threat import router as m01_router
from deep_shield.api.routes_deepfake import router as m02_router
from deep_shield.api.routes_video_events import router as m03_router
from deep_shield.api.routes_forensics import router as m04_router
app.include_router(m01_router)
app.include_router(m02_router)
app.include_router(m03_router)
app.include_router(m04_router)

# background worker (RQ / Celery) for long videos — the engine is a pure function of (path, options)
from deep_shield.engines.deepfake import DeepfakeImageIntegrityEngine
ENGINE = DeepfakeImageIntegrityEngine()            # create once per worker process (models stay loaded)
def run_m02_job(path, tasks=None):
    return ENGINE.analyze(path, tasks=tasks).model_dump(mode="json")   # store JSON in PostgreSQL (JSONB)
```

`deep_shield/common/schema.py` is written to be **shared by all four modules**. Modules 01, 03 and 04 should emit the same `Finding` / `ModuleResult` so the intelligence layer can correlate them:

| Link key | Module 02 fills it with | Correlation example |
|---|---|---|
| `media_id` | SHA-256 prefix of the file | Same file analysed by several modules |
| `frame_index`, `timestamp_s` | Exact source frame and time | M03 event at 4.8 s ↔ M02 face score at 4.8 s |
| `time_start_s`, `time_end_s` | Track / segment span | M01 dwell event overlapping a high-score face segment |
| `region` | Face box (x, y, w, h) in original pixels | M01 person box containing the face box (containment / IoU) |
| `track_id` | `m02-face-T{n}` | Map to M01/M03 person tracks by region overlap over time |
| `score_type` | `classifier_probability` or `aggregate_mean_classifier_probability` | Prevents merging with detector / anomaly / rule scores |

## 5. Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `DEEPSHIELD_M02_BACKEND` | `hf` | `hf` = real models; `stub` = placeholder scores for tests/UI wiring (all output is labelled STUB) |
| `DEEPSHIELD_DEVICE` | `auto` | `auto` / `cpu` / `cuda` / `cuda:0` / `mps` |
| `DEEPSHIELD_MODELS_DIR` · `DEEPSHIELD_EVIDENCE_DIR` | `models` · `evidence_store/m02` | Local paths |
| `DEEPSHIELD_M02_FACE_MODEL` · `DEEPSHIELD_M02_SYNTH_MODEL` | see §1 | Swap in another HF image-classification checkpoint |
| `DEEPSHIELD_M02_FACE_POS` · `DEEPSHIELD_M02_SYNTH_POS` | `Fake` · `dalle,sd` | Labels counted as positive; must match the checkpoint's `id2label` (checked at load time) |
| `DEEPSHIELD_M02_FACE_LOW/HIGH` · `DEEPSHIELD_M02_SYNTH_LOW/HIGH` | 0.35 / 0.65 | Three-band thresholds |
| `DEEPSHIELD_M02_SAMPLE_FPS` · `_MAX_FRAMES` · `_MAX_DURATION` | 2 · 64 · 120 | Video sampling limits |
| `DEEPSHIELD_M02_FACE_MIN` · `DEEPSHIELD_M02_FACE_DET_THR` | 48 · 0.75 | Minimum face size (px); YuNet confidence threshold |
| `DEEPSHIELD_M02_BATCH` · `DEEPSHIELD_MAX_UPLOAD_MB` | 16 · 200 | Inference batch size; upload size limit |
| `DEEPSHIELD_M02_DEFAULT_TASKS` | `face_manipulation,image_manipulation,frequency_analysis` | Tasks run when none are requested (videos keep only the face task) |
| `DEEPSHIELD_M02_EXPLAIN` | `1` | `0` disables the occlusion heatmap (saves about 25 extra classifier passes per face) |
| `DEEPSHIELD_M01_MODEL` | `models/yolo26n.pt` | Local Ultralytics checkpoint; download with `scripts/download_m01_model.py` |
| `DEEPSHIELD_M01_ZONES` | empty | Optional JSON array of normalized polygon zones |
| `DEEPSHIELD_M01_SAMPLE_FPS` · `_MAX_DURATION` | 1 · 60 | M01 sample rate and video duration cap |
| `DEEPSHIELD_M01_DWELL_SECONDS` · `_UNATTENDED_SECONDS` | 30 · 60 | M01 loiter and unattended-object candidate thresholds |
| `DEEPSHIELD_M03_SAMPLE_FPS` · `_MAX_DURATION` | 2 · 60 | M03 sample rate (2–5 FPS) and video duration cap |
| `DEEPSHIELD_M03_CROWD_MIN_PEOPLE` · `_CROWD_SECONDS` | 4 · 5 | Minimum people and duration for a crowding candidate |
| `DEEPSHIELD_M03_RUNNING_BODY_LENGTHS_PER_SECOND` | 2.5 | Rapid-movement threshold; requires camera-specific evaluation |
| `DEEPSHIELD_M01_EVIDENCE_DIR` · `DEEPSHIELD_M03_EVIDENCE_DIR` | `evidence_store/m01` · `evidence_store/m03` | Separate derived-evidence paths |

## 6. Evaluation (proposal §10)

```bash
# held-out folders: DATA/real/*  and  DATA/fake/*
python scripts/evaluate_m02.py --data eval/faces     --task face_manipulation --assume-face-crop --limit 500
python scripts/evaluate_m02.py --data eval/synthetic --task synthetic_image  --limit 500
```

The script outputs `metrics.json` with ROC-AUC, accuracy, precision, recall, F1 and the confusion matrix at 0.5. It also reports three-band coverage and how many cases are inconclusive, plus a threshold sweep from 0.05 to 0.95. Per-file scores go to `scores.csv`. Record the dataset source, licence, access date, split and class balance with the numbers. Use FaceForensics++ only under its access agreement and do not redistribute samples. Use consented or synthetic media for the live demo.

## 7. Known limitations — show these in the report

| Area | Limitation |
|---|---|
| Face model | Trained on a c. 2023 real-vs-fake face set. The model author warns of concept drift against newer generators. Expect lower accuracy on recent deepfakes, compressed video and small or profile faces. |
| Synthetic model | Trained on real photos vs DALL·E / Stable Diffusion. Midjourney, newer diffusion/flow models, edited photos and screenshots may be misclassified. Input is downscaled to 224×224. |
| Video | Both classifiers are frame-based. Temporal reasoning aggregates per-frame scores; it does not model temporal artefacts such as blink rate or lip sync. The greedy IoU tracker is suited to short clips with moderate motion. |
| Heatmap | Explains what the *classifier* relied on (occlusion sensitivity). It is not a ground-truth manipulation mask, and its colours are relative (hottest region = 1). The absolute score drop is printed on the image. |
| Frequency analysis | The high-frequency indicator did not separate real from fake faces (ROC-AUC 0.52), so it is shown as a plot for expert reading only. JPEG compression, resizing and sharpening also change the high band. |
| Manipulation maps | Texture, edges and dark or flat areas vary naturally; subtle retouching (blur) shows only as a loss of noise. For human review only. |
| Scores | Uncalibrated softmax outputs. They are not probabilities of manipulation and must not be combined with other modules' scores into one number. |
| Scope | No identity recognition and no inference of intent or sensitive attributes. All output is advisory and needs human review. |

## 8. Attribution

- prithivMLmods, *open-deepfake-detection* (SigLIP2 fine-tune on OpenDeepfake-Preview), Hugging Face, Apache-2.0.
- Candidates evaluated and not selected: dima806 *deepfake_vs_real_image_detection*; desireemcv *vit-face-real-vs-fake*; NYUAD-ComNets *AI-generated_images_detector* (all Apache-2.0).
- OpenCV Zoo **YuNet** face detector, Apache-2.0; OpenCV Haar cascade, BSD-3.
- Rössler et al., *FaceForensics++*, ICCV 2019 — recommended evaluation benchmark (access agreement required).
- Test asset `tests/assets/face_sample.jpg`: NASA portrait of astronaut Eileen Collins (public domain, via scikit-image).

## 9. Layout

```
deep_shield/
  common/schema.py                 shared Finding / ModuleResult schema (all modules)
  engines/deepfake/
    config.py                      model registry, thresholds, env settings
    faces.py                       YuNet / Haar face localisation + crops
    classifiers.py                 HF pipeline backend, stub backend, three-band verdicts
    media.py                       hashing, image decode, timestamped frame sampling
    temporal.py                    IoU + centre-distance face tracker, temporal summary, segments
    explain.py                     occlusion-sensitivity heatmap (suspicious-region map)
    frequency.py                   spectral fingerprint + plot
    manipulation.py                ELA + noise-level maps + EXIF review
    evidence.py                    crops, annotated frames, score timeline
    engine.py                      DeepfakeImageIntegrityEngine.analyze()
    metrics.py                     ROC-AUC, P/R/F1, three-band, threshold sweep
  engines/threat/                  M01 detector, normalized zones and rule findings
  engines/video_events/            M03 temporal event rules over shared M01 tracks
  api/routes_threat.py              M01 router (/api/v1/m01/…)
  api/routes_deepfake.py            M02 router (/api/v1/m02/…)
  api/routes_video_events.py        M03 router (/api/v1/m03/…)
  api/main.py                       shared API app for modules 01–04
scripts/  download_models.py · download_m01_model.py · run_m02.py · run_m04.py
tests/    test_m02.py (+ assets/)
```

---

# DEEP SHIELD · Module 04 — Visual Forensics

Classical image forensics (proposal §4.4), with no model download. It shares Module 02's result schema, evidence store and API app.

| Task | Default | Method | Output |
|---|---|---|---|
| `copy_move` | images + video frames | SIFT keypoints matched within the same image (generalised 2-NN ratio test) → RANSAC similarity clustering (Amerini et al., 2011) | **Candidate cloned region pairs**: source and target boxes, number of matches, shift, scale and rotation, plus a match visualisation |
| `ela_noise` | images + video frames | Error Level Analysis + local noise-level map (shared with Module 02) | Visual maps only, no verdict |
| `jpeg_analysis` | images | Quantisation-table quality estimate; 8×8 block-grid alignment; JPEG ghosts (Farid, 2009) | Observations + ghost maps at qualities *below* the file's own save quality |
| `metadata` | images | EXIF make / model / software / timestamps | Observations |

**Video:** `copy_move` and `ela_noise` run on selected frames, by default 3 evenly spaced keyframes. Pass `--frames` (CLI) or `frames` (API) to analyse a frame flagged by another module; the frame index and timestamp are kept for cross-module linking.

### Internal validation (synthetic, 17 Nov 2026; not a formal benchmark)

| Check | Result |
|---|---|
| Copy-move, cloned regions of 120–200 px | **27 / 28 found** |
| Copy-move, cloned regions of 50–110 px | 16 / 28 found (small or flat regions have too few keypoints) |
| Copy-move false alarms on 14 untouched photos | 2 / 14, both highly repetitive textures (grass, gravel) |
| JPEG quality estimate (q60/75/90/95 OpenCV, q80 Pillow) | Exact in all 5 cases |
| 8×8 grid after a 3×5 px crop | Detected as shifted [5, 3]; untouched JPEG aligned; PNG shows no grid |
| JPEG ghost, region from a q60 save pasted into a q90 image | Pasted region clearly visible in the q60 map |
| **Resampling (resize) detection** (Kirchner p-map) | **Excluded**: after JPEG compression, 0 / 36 resized images separated from originals |

### Run

```powershell
python scripts\make_m04_demo.py                       # demo\m04_original.jpg, m04_clone.jpg, m04_ghost.jpg
python scripts\run_m04.py demo\m04_clone.jpg
python scripts\run_m04.py demo\clip.mp4 --frames 120
uvicorn deep_shield.api.main:app --host 0.0.0.0 --port 8002   # M02 + M04 in one API
```

| Endpoint | Purpose |
|---|---|
| `GET /api/v1/m04/health` | Tasks, keypoint detector (SIFT or ORB) |
| `POST /api/v1/m04/analyze` | multipart `file`, `tasks`, `frames` → `ModuleResult` |
| `GET /api/v1/m04/evidence/{media_id}/{file}` | `copy_move.jpg`, `ela_noise_panel.jpg`, `jpeg_ghosts.jpg` (video: `f000120_…`) |
| `DELETE /api/v1/m04/evidence/{media_id}` | Retention control |

Evidence is stored under `evidence_store/m04/<media_id>/` (override with `DEEPSHIELD_M04_EVIDENCE_DIR`).

### Limitations

| Area | Limitation |
|---|---|
| Copy-move | Candidates for human review. Repeated real structures (tiles, windows, text, foliage) can match. Flat regions (sky, walls) have no keypoints and cannot be checked. |
| JPEG ghosts | Informative only when a region was saved earlier at a *lower* quality than the final save. Flat or saturated areas appear bright in every map; only a region bright at **one** quality is meaningful. |
| Grid alignment | A shifted grid also results from ordinary cropping. |
| ELA / noise / metadata | Visual aids and leads only; metadata is easily stripped or edited. |
| Video | Codec compression limits ELA and noise maps; copy-move works on decoded frames. |

---

# DEEP SHIELD · Module 01 — Threat and Anomaly Intelligence

M01 uses a local YOLO26n COCO checkpoint with ByteTrack to detect people, backpacks, handbags and suitcases. It supports image detection and sampled video tracking. Video defaults to 1 frame/second with a 60-second cap. The checkpoint is not fine-tuned on campus footage.

Download the checkpoint after installing `requirements.txt`:

```powershell
python scripts\download_m01_model.py
```

| Endpoint | Purpose |
|---|---|
| `GET /api/v1/m01/health` | Detector, local checkpoint and rule configuration status |
| `POST /api/v1/m01/analyze` | multipart `file`; optional `zones` JSON field; returns shared `ModuleResult` |
| `GET /api/v1/m01/evidence/{media_id}/{file}` | Annotated image or video evidence |
| `DELETE /api/v1/m01/evidence/{media_id}` | Delete derived M01 evidence |

Zones are normalized polygons, so their coordinates run from 0 to 1 regardless of resolution. Send this JSON value in the multipart `zones` field or configure it as `DEEPSHIELD_M01_ZONES`:

```json
[{"zone_id":"lobby","polygon":[[0.1,0.1],[0.9,0.1],[0.9,0.9],[0.1,0.9]]}]
```

M01 returns detector-confidence findings separately from rule-event findings for zone entry, loitering and stationary unattended-object candidates. Image input supports detection only. These rules do not establish intent or threat; thresholds require local validation.

# DEEP SHIELD · Module 03 — Video Event Understanding

M03 reuses the same M01 detector and ByteTrack tracks. It samples video at 2 fps by default (configurable from 2–5 fps), caps analysis at 60 seconds, and emits timestamped candidates for zone entry/exit, loitering, crowding and rapid movement. The crowd rule uses configured zones, or the full frame if no zones are provided. Short intervals of the same event type are merged.

| Endpoint | Purpose |
|---|---|
| `GET /api/v1/m03/health` | Detector, sampling and event-rule configuration status |
| `POST /api/v1/m03/analyze` | multipart video `file`; optional `zones` JSON field; returns shared `ModuleResult` |
| `GET /api/v1/m03/evidence/{media_id}/{file}` | Annotated frames referenced by events |
| `DELETE /api/v1/m03/evidence/{media_id}` | Delete derived M03 evidence |

This version uses explainable rules; the optional GRU is not included. Crowd and movement thresholds are starting values, not validated behavior. Evaluate event precision/recall and timestamp error on consented, representative video before relying on the output.
