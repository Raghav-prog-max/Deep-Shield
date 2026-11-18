# Model and event evaluation

## Data required

No labeled evaluation media is currently present in this checkout. `tests/assets/face_sample.jpg` is a smoke-test fixture; it is not a consented, representative benchmark and must not be used to claim accuracy or tune thresholds.

For Module 02, provide separate held-out image sets in this layout:

```text
eval/faces/real/       # consented authentic face crops or source images
eval/faces/fake/       # consented manipulated-face images
eval/synthetic/real/   # consented real photographs
eval/synthetic/fake/   # labeled AI-generated images
```

Document the source, consent and usage basis, license, access date, collection period, class counts, and train/calibration/test split. Split by person, source video, or generator where possible to reduce leakage. Never choose thresholds on the held-out test split.

## Existing M02 evaluator

From the repository root, run each task on its own dataset:

```sh
python scripts/evaluate_m02.py --data eval/faces --task face_manipulation
python scripts/evaluate_m02.py --data eval/synthetic --task synthetic_image
```

Use `--assume-face-crop` only when every image is already a face crop. The script writes ROC-AUC, accuracy, precision, recall, F1, three-band coverage, inconclusive count, and a threshold sweep under `eval_out/`. Use calibration data to choose thresholds; run the final selected thresholds once on held-out test data and record both the metrics and model revision.

## Video events and M01/M03

Threshold tuning for zone entry, loitering, unattended-object, crowding, and running events requires consented videos plus human event annotations with clip-relative start/end times. Zone-dependent tasks also require a documented camera view and normalized polygons. Object-detection evaluation needs labeled boxes and object classes. Report event precision/recall/F1 at a predeclared temporal-overlap rule, false alerts per video-hour, missed events, and timestamp error; summarize results by camera and relevant conditions such as lighting and compression.

The repository does not yet contain those labeled videos or a ground-truth event evaluator. Do not tune M01/M03 thresholds from unlabeled demo clips or from one camera and describe them as generally validated.
