"""End-to-end verification of the DEEP SHIELD deployment, API, worker recovery, evidence links, and PDF reports.

Runs local API/worker integration checks and probes the public frontend/API routes.
The TestClient job flow below is local; it does not count as a hosted media test.

Writes reports/DEPLOYMENT_VERIFICATION.md and reports/deployment_verification.json.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("DEEPSHIELD_M02_BACKEND", "stub")

from fastapi.testclient import TestClient  # noqa: E402
from deep_shield.api.main import app  # noqa: E402
from deep_shield.api.routes_intelligence import _analyze_path  # noqa: E402
from deep_shield.engines.threat.config import parse_zones  # noqa: E402
from deep_shield.jobs.store import AnalysisJobStore  # noqa: E402

HOSTED_URL = "https://deep-shield-sooty.vercel.app"


def check_hosted_url(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "deep-shield-verifier/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
            return {
                "url": url,
                "reachable": True,
                "status_code": resp.status,
                "https": url.startswith("https://"),
                "contains_deep_shield": "DEEP SHIELD" in body.upper() or "DEEP-SHIELD" in body.upper(),
                "bytes": len(body),
            }
    except urllib.error.HTTPError as exc:
        return {"url": url, "reachable": False, "status_code": exc.code, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"url": url, "reachable": False, "error": str(exc)}


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="ds_verify_") as tmp:
        tmp_path = Path(tmp)
        os.environ["DEEPSHIELD_DATABASE_PATH"] = str(tmp_path / "verify.sqlite3")
        os.environ["DEEPSHIELD_JOB_UPLOAD_DIR"] = str(tmp_path / "uploads")
        os.environ["DEEPSHIELD_M01_EVIDENCE_DIR"] = str(tmp_path / "ev_m01")
        os.environ["DEEPSHIELD_EVIDENCE_DIR"] = str(tmp_path / "ev_m02")
        os.environ["DEEPSHIELD_M03_EVIDENCE_DIR"] = str(tmp_path / "ev_m03")
        os.environ["DEEPSHIELD_M04_EVIDENCE_DIR"] = str(tmp_path / "ev_m04")

        client = TestClient(app)

        # 1. Hosted HTTPS frontend checks
        hosted_root = check_hosted_url(HOSTED_URL)
        hosted_review = check_hosted_url(f"{HOSTED_URL}/review")
        hosted_api_health = check_hosted_url(f"{HOSTED_URL}/api/v1/m01/health")

        # 2. Engine health endpoints
        h1 = client.get("/api/v1/m01/health").json()
        h2 = client.get("/api/v1/m02/health").json()
        h3 = client.get("/api/v1/m03/health").json()
        h4 = client.get("/api/v1/m04/health").json()

        # 3. Upload media to create queued job
        sample_img = ROOT / "tests" / "assets" / "face_sample.jpg"
        with open(sample_img, "rb") as fh:
            create_resp = client.post(
                "/api/v1/intelligence/jobs",
                files={"file": ("face_sample.jpg", fh, "image/jpeg")},
                data={"zones": '[{"zone_id":"lobby","polygon":[[0.1,0.1],[0.9,0.1],[0.9,0.9],[0.1,0.9]]}]'},
            )
        assert create_resp.status_code == 202, create_resp.text
        job_id = create_resp.json()["job_id"]

        # 4. Simulate worker-1 claiming the job and being interrupted before completion
        store1 = AnalysisJobStore(Path(os.environ["DEEPSHIELD_DATABASE_PATH"]))
        claimed1 = store1.claim_next("worker-1-interrupted", lease_seconds=0)
        assert claimed1 is not None and claimed1.job_id == job_id and claimed1.attempts == 1

        # 5. Simulate service restart + worker-2 recovering the interrupted job
        store2 = AnalysisJobStore(Path(os.environ["DEEPSHIELD_DATABASE_PATH"]))
        claimed2 = store2.claim_next("worker-2-recovery", lease_seconds=-1)
        assert claimed2 is not None and claimed2.job_id == job_id and claimed2.attempts == 2

        result = _analyze_path(claimed2.upload_path, claimed2.source_name, parse_zones(claimed2.zones))
        assert store2.finish(job_id, "worker-2-recovery", status="completed", media_id=result.media_id, result=result)
        claimed2.upload_path.unlink(missing_ok=True)

        # 6. Verify persistence across fresh store reconnect + API status & result retrieval
        status_resp = client.get(f"/api/v1/intelligence/jobs/{job_id}")
        assert status_resp.status_code == 200 and status_resp.json()["status"] == "completed"
        res_resp = client.get(f"/api/v1/intelligence/jobs/{job_id}/result")
        assert res_resp.status_code == 200
        res_body = res_resp.json()

        # 7. Verify evidence link serving
        evidence_checked = 0
        for mod_res in res_body["module_results"]:
            mod_name = mod_res["module"]
            prefix = "m01" if "M01" in mod_name else "m02" if "M02" in mod_name else "m03" if "M03" in mod_name else "m04"
            for f in mod_res["findings"]:
                for ev in f.get("evidence", []):
                    ev_resp = client.get(f"/api/v1/{prefix}/evidence/{ev['path']}")
                    if ev_resp.status_code == 200:
                        evidence_checked += 1

        # 8. Verify PDF report download
        pdf_resp = client.get(f"/api/v1/intelligence/jobs/{job_id}/report.pdf")
        assert pdf_resp.status_code == 200 and pdf_resp.content.startswith(b"%PDF")

        # 9. Verify retention delete endpoint
        del_resp = client.delete(f"/api/v1/intelligence/jobs/{job_id}")
        assert del_resp.status_code == 200 and del_resp.json()["deleted"] == job_id

        configured_backend = os.getenv("NEXT_PUBLIC_API_BASE_URL", "").rstrip("/")
        report = {
            "verified_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "hosted_frontend": {"root": hosted_root, "review_workspace": hosted_review},
            "hosted_backend": {
                "same_origin_m01_health": hosted_api_health,
                "configured_backend_url": configured_backend or None,
                "public_backend_configured": bool(configured_backend),
                "hosted_analysis_verified": False,
            },
            "local_backend_integration": {
                "scope": "Local FastAPI TestClient and temporary SQLite store; no hosted media job was submitted.",
                "engine_health": {
                    "m01": h1.get("module"),
                    "m02": h2.get("module"),
                    "m03": h3.get("module"),
                    "m03_gru_enabled": h3.get("gru_enabled"),
                    "m04": h4.get("module"),
                },
                "job_flow": {
                    "job_id": job_id,
                    "interrupted_worker_attempt": 1,
                    "recovered_worker_attempt": 2,
                    "final_status": "completed",
                    "modules_completed": [m["module"] for m in res_body["module_results"]],
                    "evidence_links_verified": evidence_checked,
                    "pdf_report_bytes": len(pdf_resp.content),
                    "retention_cleanup_verified": True,
                },
            },
        }

        out_dir = ROOT / "reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "deployment_verification.json").write_text(json.dumps(report, indent=2))

        root_status = "PASS" if hosted_root.get("reachable") else "FAIL"
        review_status = "PASS" if hosted_review.get("reachable") else "FAIL"
        api_status = "AVAILABLE" if hosted_api_health.get("reachable") else "NOT AVAILABLE"
        root_http = hosted_root.get("status_code") or "unavailable"
        review_http = hosted_review.get("status_code") or "unavailable"
        api_http = hosted_api_health.get("status_code") or "unavailable"
        md = [
            "# Deep Shield - Deployment Verification",
            "",
            f"**Checked at:** {report['verified_at']}",
            f"**Frontend:** [{HOSTED_URL}]({HOSTED_URL}) ([`/review`]({HOSTED_URL}/review))",
            "",
            "## Public checks",
            "",
            "| Check | Status | Details |",
            "|---|---|---|",
            f"| Landing page (`/`) | **{root_status}** | HTTP {root_http}; {hosted_root.get('bytes', 0)} response bytes |",
            f"| Review workspace (`/review`) | **{review_status}** | HTTP {review_http}; {hosted_review.get('bytes', 0)} response bytes |",
            f"| Same-origin API health (`/api/v1/m01/health`) | **{api_status}** | HTTP {api_http} |",
            "| Hosted image/video analysis | **NOT VERIFIED** | This script does not submit a media job to a hosted backend. |",
            "",
            "## Local backend integration checks",
            "",
            "These checks use FastAPI TestClient and temporary local storage; they do not demonstrate hosted inference.",
            "",
            f"- Engine health routes responded locally; M03 GRU enabled: `{h3.get('gru_enabled')}`.",
            f"- Job `{job_id}` completed after simulated worker lease recovery (attempts 1 -> 2).",
            f"- Verified {evidence_checked} evidence responses and a {len(pdf_resp.content)} byte PDF report.",
            "- Job/evidence deletion endpoint completed successfully.",
            "",
            "## Remaining production checks",
            "",
            "Deploy and configure a backend, connect it to the frontend, then submit consented representative images and videos through `/review`. Record hosted job results, evidence, PDF, and retention outcomes separately.",
        ]
        (out_dir / "DEPLOYMENT_VERIFICATION.md").write_text("\n".join(md) + "\n")
        print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
