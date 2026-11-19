# Deep Shield - Deployment Verification

**Checked:** 2026-11-19 04:26:22 UTC
**Repository commit:** `6206fd990feb0fc35dd6311a278c8acd31bb19a1`
**Frontend:** [https://deep-shield-sooty.vercel.app](https://deep-shield-sooty.vercel.app)

## Current public checks

| Check | Result | Details |
|---|---|---|
| Vercel landing page `/` | **Reachable** | HTTP 200; response body 95,349 bytes. |
| Vercel review page `/review` | **Reachable** | HTTP 200; response body 10,130 bytes. |
| Same-origin API `/api/v1/m01/health` | **Not available** | HTTP 404. No public backend URL is configured in the checked-in frontend configuration. |
| Hosted image/video analysis workflow | **Not verified** | Requires a deployed backend connected to Vercel. |

The frontend is live, but this is not a complete hosted application until a backend is deployed, connected, and exercised through the review workspace.

## GitHub Actions for the reviewed commit

[Run 44453823617](https://github.com/Raghav-prog-max/Deep-Shield/actions/runs/44453823617), for the commit above, completed successfully:

- `frontend-build`: npm install, typecheck, and production build passed.
- `backend-tests`: configured unit/integration suite passed after M01 weights download.

The backend CI job sets `DEEPSHIELD_M02_BACKEND=stub` and invokes pytest with `--ignore=tests/test_deep_models_media.py`. This run does not substantiate a claim that all real-model tests passed. Any full test count must identify the command, real/stub configuration, skipped or excluded tests, and run result.

## Previously recorded local integration run

The JSON record retains a backend integration run created at `2026-11-19T01:47:21`. It recorded successful engine health, queued job completion, recovery after a worker interruption, 11 evidence responses, a valid PDF response, and retention cleanup. These checks were local/integration checks; the same run timed out when checking the public frontend. The previous Markdown incorrectly labelled those timed-out frontend checks as passed. This report corrects that status and supersedes the old table.

## Remaining production verification

1. Deploy the API and worker on one of the documented hosting targets and record the public URL.
2. Confirm model weights load, health endpoints respond, persistent storage works, and worker restart/recovery behaves as expected.
3. Set `NEXT_PUBLIC_API_BASE_URL` in Vercel to the deployed API URL and redeploy.
4. From `/review`, submit consented representative images and videos; verify real engine results, evidence links, PDF download, retention behavior, and error handling.
5. Save the exact media/test protocol, deployment revision, timestamps, and outcomes here. Keep failed checks marked failed or not run.

See [DEPLOYMENT.md](../DEPLOYMENT.md) for deployment instructions and [PROJECT_STATUS.md](../PROJECT_STATUS.md) for the project flow and completion map.
