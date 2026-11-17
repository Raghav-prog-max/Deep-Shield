# DEEP SHIELD backend prototype

This is the first local backend milestone: FastAPI application shell, local media upload and validation, SQLite records, a serialized background job queue, job status, and findings retrieval. It does not implement any of the four analysis engines yet. Requested engines without registered implementations are marked unavailable; the API does not invent findings.

## Requirements

- Python 3.10 or newer
- FFmpeg support through the installed OpenCV build for the video formats supported by the host
- CPU-only operation; no GPU dependency

## Run locally

From this directory:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Health and interactive API documentation:

- `GET http://127.0.0.1:8000/api/health`
- `http://127.0.0.1:8000/docs`

Default storage is `backend/data/`. Set `DEEP_SHIELD_DATA_DIR` to change it. The upload defaults are 100 MiB and 60 seconds for video; both can be changed through environment variables. These are prototype defaults, not validated deployment limits or a retention policy.

## Current API

- `POST /api/v1/media` — multipart form field `file`; accepts JPEG, PNG, WebP, MP4, MOV, MKV, or AVI after content validation. Returns the generated media ID and SHA-256. The client MIME header is not trusted.
- `POST /api/v1/jobs` — JSON such as `{"media_id":"...","requested_engines":["engine1"]}`; returns a queued job.
- `GET /api/v1/jobs/{job_id}` — status, progress, and per-engine availability/result state.
- `GET /api/v1/jobs/{job_id}/findings` — findings associated with the job.

The four requested engine names are accepted as job inputs, but none is registered in this starter. Jobs will therefore finish as failed with an explicit `engine_unavailable` result until an implementation is integrated. Engine scores remain separate; there is no combined threat score.

## Important limits

- SQLite and one in-process worker are for a single-machine prototype. Restarting the API does not restore queued jobs.
- Video validation depends on OpenCV codec support and is not a full malware sandbox.
- No report or incident endpoints, authentication, media deletion workflow, face/plate blurring, or model loading is implemented yet.
- Define retention, export handling, access controls, exact limits, and supported media profiles before using personal or operational footage.
- All future findings must remain advisory and include evidence, score type, model version, and limitations. No face recognition, identity lookup, intent/criminality inference, or sensitive-attribute inference.
