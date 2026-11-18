# DEEP SHIELD frontend

Next.js client for the local FastAPI integration. It uploads a media file to the queued analysis endpoint, polls job status, then shows module findings, evidence links, cross-module groups, and the generated PDF report.

## Run locally

1. Start the API and its separate durable worker from the repository root (see the root README).
2. From this directory, copy `.env.example` to `.env.local` and set `API_INTERNAL_BASE_URL` if the API uses a different address.
3. Install and start the frontend: `npm install`, then `npm run dev`.
4. Open `http://localhost:3000`. The frontend proxies `/api` requests to the backend, so browser uploads, evidence and reports use one origin.

The backend currently allows CORS from `http://localhost:3000`. For another origin, update the API's explicit CORS allowlist.
