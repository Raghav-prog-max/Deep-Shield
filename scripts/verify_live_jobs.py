"""Exercise the running API and worker with an approved image and video."""

import argparse
import json
import time
from pathlib import Path

import httpx


def verify_job(client, path, modules, timeout):
    with path.open("rb") as source:
        created = client.post("jobs", files={"file": (path.name, source)})
    created.raise_for_status()
    if created.status_code != 202:
        raise RuntimeError(f"Expected a queued job, received {created.status_code}")
    route = "jobs/" + created.json()["job_id"]
    deadline = time.monotonic() + timeout
    while True:
        job = client.get(route).raise_for_status().json()
        if job["status"] not in {"queued", "running", "processing"}:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for {route}")
        time.sleep(2)
    if job["status"] != "completed":
        raise RuntimeError(f"{route} finished with status {job['status']}")

    result = client.get(route + "/result").raise_for_status().json()
    outputs = result["module_results"]
    actual = {output["module"].split("_")[0] for output in outputs}
    if actual != modules or any(m["status"] != "completed" for m in outputs):
        raise RuntimeError(f"Incomplete module results for {route}: {actual}")
    evidence = set()
    for output in outputs:
        module = output["module"].split("_")[0].lower()
        for finding in output["findings"]:
            for item in finding["evidence"]:
                evidence.add(f"../{module}/evidence/{item['path']}")
    if not evidence:
        raise RuntimeError(f"No evidence returned for {route}")
    for link in evidence:
        if not client.get(link).raise_for_status().content:
            raise RuntimeError("Empty evidence response")
    pdf = client.get(route + "/report.pdf").raise_for_status().content
    if not pdf.startswith(b"%PDF"):
        raise RuntimeError(f"Invalid PDF for {route}")
    client.delete(route).raise_for_status()
    if client.get(route).status_code != 404:
        raise RuntimeError(f"Deleted job still reachable at {route}")
    return {"input": path.name, "modules": sorted(actual), "evidence_files": len(evidence),
            "pdf_bytes": len(pdf), "job_removed": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", required=True, help="URL of the running backend")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--timeout", type=int, default=600, help="Polling limit per job in seconds")
    args = parser.parse_args()
    with httpx.Client(base_url=args.api.rstrip("/") + "/api/v1/intelligence/", timeout=60) as client:
        for path, expected in [(args.image, {"M01", "M02", "M04"}),
                               (args.video, {"M01", "M02", "M03", "M04"})]:
            print(json.dumps(verify_job(client, path, expected, args.timeout)), flush=True)
