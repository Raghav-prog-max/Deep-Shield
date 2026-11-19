# Single-host deployment

This deployment runs the API, a separate SQLite-backed worker, the Next.js review app, model downloads, and Caddy HTTPS on one Docker host. The web app is protected with Caddy Basic Auth; the API and worker are private to the Compose network. The SQLite database, uploaded job files, evidence, downloaded models, Hugging Face cache, and Caddy certificates use persistent Docker volumes.

## Requirements

- A Linux host with Docker Engine and Docker Compose v2.
- A DNS name pointed at the host, with inbound ports 80 and 443 available.
- Enough disk for Python/PyTorch, model downloads, uploaded media and generated evidence. Give the host several GB of free space and monitor it.
- Review the repository's AGPL-3.0-only terms and the Ultralytics licensing terms before deployment.

## Configure and start

1. Copy `.env.production.example` to `.env.production`.
2. Set `SITE_ADDRESS` to the DNS name, choose a unique `SITE_USER`, and generate `SITE_PASSWORD_HASH` with the command in the env file. Double each `$` in the generated hash when putting it in `.env.production` so Compose passes the full hash through. Keep `.env.production` private.
3. Start the services from the repository root:

   ```sh
   docker compose --env-file .env.production up -d --build
   ```

4. The first startup downloads the model checkpoints. Follow it with `docker compose logs -f model-setup api worker caddy`; model setup must complete before the API and worker start.
5. Open `https://<SITE_ADDRESS>`, sign in with the configured Basic Auth credentials, and submit a small consented image before using video.
6. Check `https://<SITE_ADDRESS>/api/v1/m01/health` and `docker compose ps`. A completed analysis job confirms the API, worker, shared volumes, evidence routes and report generation work together.

## Operations and limits

- This queue is durable across API/worker restarts when the Docker volumes remain intact. Run one API replica against the SQLite database. Multiple workers can claim jobs safely, but each loads the inference models into memory; begin with one worker on CPU.
- Back up the `application-data`, `evidence-store`, and `model-store` volumes together. Keep Caddy's data volume for certificate renewal. Test a restore before relying on backups.
- Uploaded source files are deleted immediately after a job finishes. Derived evidence (`evidence_store/m01..m04`) and completed SQLite job records are automatically purged by the worker after `DEEPSHIELD_RETENTION_HOURS` (default `168` hours / 7 days), and can also be removed on demand via `DELETE /api/v1/intelligence/jobs/{job_id}` or `POST /api/v1/intelligence/retention/purge`. Adjust `DEEPSHIELD_RETENTION_HOURS` to match your operational retention policy before storing sensitive media.
- The default upload limit is 200 MB. Tune `DEEPSHIELD_MAX_UPLOAD_MB` and proxy limits together if this changes.
- The single-host SQLite design is not a multi-host/high-availability queue. For multiple API hosts, move job state to PostgreSQL and use a broker such as Redis/RQ or Celery with shared object storage.
- Basic Auth protects the whole site over HTTPS. Add organization identity, access logs and formal retention controls if the deployment will handle sensitive media or multiple reviewers.
- Thresholds remain unvalidated until evaluated against representative, consented, labeled data. Deployment does not make the outputs operationally validated.

## Stop

```sh
docker compose --env-file .env.production down
```

This preserves data volumes. `docker compose down -v` permanently deletes the volumes and all stored job/evidence/model data.
