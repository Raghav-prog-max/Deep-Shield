# Deep Shield release acceptance

## Local CPU run, 19 November 2026

The test process used the Hugging Face backend, local weights, the trained M02 heads, and the M03 GRU. The complete suite ran without file exclusions. Some unit tests use fixtures or stubs internally.

| Check | Result |
|---|---|
| Full pytest suite with OpenCV 4.12.0 | 36 passed, 0 failed, 0 skipped in 769 seconds |
| Frontend | Typecheck and Next.js production build passed. The local review route returned HTTP 200. |
| Image via the production frontend proxy | M01, M02 and M04 completed. All 12 evidence files downloaded and the PDF contained 6,934 bytes. |
| Video via the production frontend proxy | All four modules completed. All 21 evidence files downloaded and the PDF contained 16,427 bytes. |
| Job cleanup | Both jobs returned 404 after deletion. |

The same two inputs also completed directly against the local API. The frontend proxy run used a separate worker process and the SQLite queue.

## Repository image evaluation

| Task | Scored | ROC-AUC | Result at the deployed thresholds |
|---|---:|---:|---|
| Face manipulation | 15 of 16 | 0.9643 | 8 correct decisions, 7 inconclusive, 1 image without a face finding. Thresholds 0.156 and 0.804. |
| Synthetic image | 16 of 16 | 1.0000 | 16 correct decisions. Thresholds 0.406 and 0.750. |

Each task used eight real and eight fake images from its repository test split. The measured coverage applies to these fixtures. Field calibration requires a separate consented sample and an agreed error tolerance.

## Repeat the checks

Install the CPU dependencies and download the M01 and M02 checkpoints as described in the README. In a shell with the project environment active:

```sh
DEEPSHIELD_TEST_REAL=1 DEEPSHIELD_M02_BACKEND=hf python -m pytest -ra --junitxml=release-tests.xml
python scripts/evaluate_m02.py --data eval/faces/test --task face_manipulation
python scripts/evaluate_m02.py --data eval/synthetic/test --task synthetic_image
```

With the API, worker and frontend running, pass the frontend proxy URL or the backend URL to the live checker. It queues the supplied files, downloads the evidence and PDF, then deletes the completed test jobs.

```sh
python scripts/verify_live_jobs.py --api http://127.0.0.1:3000 --image eval/faces/test/real/test_real_01.jpg --video eval/videos/test/test_corridor_01.mp4
```

## Hosted acceptance

The public site still requires an authenticated backend deployment and a production frontend connection. Verify the production workspace with approved media once both are configured.
