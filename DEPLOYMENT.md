# Deployment Guide — Local, Cloud Backend & Single-Host HTTPS

DEEP SHIELD supports two production deployment modes:

1. **Cloud Single-Container Backend (Railway, Render, HuggingFace Spaces) + Vercel Frontend** — fastest way to let any remote browser/device run real M01–M04 PyTorch inference through `https://deep-shield-sooty.vercel.app`.
2. **Single-Host Docker Compose + Caddy HTTPS** — self-hosted Linux server serving both frontend and backend behind Basic Auth + automatic TLS.

---

## Option A: Cloud Backend (Railway / Render / HuggingFace Spaces) + Vercel

Because single-container cloud hosts run one container per service, `Dockerfile` defaults to `python scripts/cloud_entrypoint.py`, which:
- Downloads missing local weights (`hustvl/yolos-tiny`, SigLIP2, YuNet) automatically on first boot if not already baked in.
- Launches and supervises the durable SQLite background worker (`python -m deep_shield.jobs.worker`).
- Serves FastAPI on `0.0.0.0:$PORT` (handles Railway/Render dynamic `PORT` and HuggingFace Spaces port `7860`).

### 1. Railway (`railway.json` included)
1. Create a new Railway project from `Raghav-prog-max/Deep-Shield` — Railway automatically detects `railway.json` + `Dockerfile`.
2. Attach a persistent volume at `/app/data` (recommended `5 GB`) and set environment variables `DEEPSHIELD_DEVICE=cpu`, `DEEPSHIELD_M02_BACKEND=hf`, `DEEPSHIELD_RETENTION_HOURS=168`.
3. Copy the generated `https://<your-app>.up.railway.app` URL and verify `GET /api/v1/intelligence/models`.

### 2. Render (`render.yaml` included)
1. In Render, choose **New → Blueprint** and select `Raghav-prog-max/Deep-Shield` — Render reads `render.yaml` (Docker runtime, `/api/v1/m01/health` check, persistent `/app/data` disk).
2. Verify `GET https://<your-service>.onrender.com/api/v1/intelligence/models`.

### 3. HuggingFace Spaces (Free 16 GB RAM CPU Tier — `Dockerfile.hf` included)
1. Create a new HuggingFace Space with SDK **Docker** (blank template).
2. Push this repo using `Dockerfile.hf` as `Dockerfile` (bakes M01 + M02 weights at build time and listens on port `7860`).
3. Verify `GET https://<your-user>-<space>.hf.space/api/v1/intelligence/models`.

### 4. Connect Vercel Frontend (`https://deep-shield-sooty.vercel.app`)
Set `NEXT_PUBLIC_API_BASE_URL` in Vercel Project Settings → Environment Variables to your cloud backend URL (e.g. `https://<your-app>.up.railway.app`), then redeploy `frontend/` — or paste the URL directly into the workspace's `CONNECT / RETRY` bar at runtime.

---

## Option B: Single-Host Docker Compose + Caddy HTTPS

Runs separate `api`, `worker`, `frontend`, `model-setup`, and `caddy` containers on one Linux host behind HTTPS + Basic Auth.

### Requirements
- A Linux host with Docker Engine and Docker Compose v2.
- A DNS name pointed at the host, with inbound ports `80` and `443` available.
- At least `5–10 GB` free disk space for PyTorch, checkpoints, uploaded media, and evidence.
- Review [LICENSE](LICENSE) (AGPL-3.0-only) before deployment.

### Configure and start

1. Copy `.env.production.example` to `.env.production`.
2. Set `SITE_ADDRESS`, `SITE_USER`, and `SITE_PASSWORD_HASH` (generate with `docker run --rm caddy:2-alpine caddy hash-password --plaintext '<password>'`, doubling each `$` in `.env.production`).
3. Start the services from the repository root:

   ```sh
   docker compose --env-file .env.production up -d --build
   ```

4. Follow first-boot checkpoint downloads with `docker compose logs -f model-setup api worker caddy`.
5. Open `https://<SITE_ADDRESS>`, sign in, and verify `GET https://<SITE_ADDRESS>/api/v1/intelligence/models`.

---

## Operations, Retention & Limits

- **Retention:** Uploaded source files are deleted immediately after worker processing. Derived evidence (`evidence_store/m01..m04`) and completed SQLite job records are automatically purged by the worker after `DEEPSHIELD_RETENTION_HOURS` (default `168` hours / 7 days), and can be deleted immediately via `DELETE /api/v1/intelligence/jobs/{job_id}` or `POST /api/v1/intelligence/retention/purge`.
- **Upload Limit:** Default `200 MB` (`DEEPSHIELD_MAX_UPLOAD_MB`).
- **Scaling:** SQLite is durable for a single host; for multi-replica API scaling, switch job storage to PostgreSQL + Redis/RQ or Celery + shared object storage.
- **Advisory Policy:** Deployment does not make outputs operationally validated; all results remain separate and advisory for human review.

### Stop Docker Compose

```sh
docker compose --env-file .env.production down
```

(`docker compose down -v` permanently removes all stored volumes.)
