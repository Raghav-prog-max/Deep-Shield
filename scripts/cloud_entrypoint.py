"""Single-container cloud entrypoint for DEEP SHIELD (Railway, Render, Fly.io, HuggingFace Spaces).

In single-container cloud environments (where docker-compose is not used), both the FastAPI HTTP
server and the durable SQLite background worker must run in the same container so queued jobs
(`POST /api/v1/intelligence/jobs`) are processed immediately.

Behavior:
  1. Verifies required local neural checkpoints (`models/hustvl__yolos-tiny` and M02 SigLIP2 + YuNet)
     and runs `scripts/download_m01_model.py` / `scripts/download_models.py` on first boot if missing.
  2. Launches `python -m deep_shield.jobs.worker` in a supervised background subprocess (automatically
     restarts it with backoff if it exits unexpectedly).
  3. Launches `uvicorn deep_shield.api.main:app --host 0.0.0.0 --port $PORT` (defaults to 8002, or
     7860 on HuggingFace Spaces).
  4. Forwards SIGTERM/SIGINT cleanly to both child processes for graceful container shutdown.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _ensure_models() -> None:
    if os.getenv("DEEPSHIELD_SKIP_MODEL_DOWNLOAD", "0") == "1":
        return
    models_dir = Path(os.getenv("DEEPSHIELD_MODELS_DIR", str(ROOT / "models")))
    yolos_cfg = models_dir / "hustvl__yolos-tiny" / "config.json"
    yolo_pt = Path(os.getenv("DEEPSHIELD_M01_MODEL", str(models_dir / "yolo26n.pt")))
    siglip_cfg = models_dir / "prithivMLmods__open-deepfake-detection" / "config.json"

    if not yolos_cfg.is_file() and not yolo_pt.is_file():
        print("[cloud_entrypoint] Downloading M01/M03 hustvl/yolos-tiny weights...", flush=True)
        subprocess.run([sys.executable, str(ROOT / "scripts" / "download_m01_model.py")], check=True, cwd=str(ROOT))

    if os.getenv("DEEPSHIELD_M02_BACKEND", "hf") != "stub" and not siglip_cfg.is_file():
        print("[cloud_entrypoint] Downloading M02 SigLIP2 + YuNet weights...", flush=True)
        subprocess.run([sys.executable, str(ROOT / "scripts" / "download_models.py")], check=True, cwd=str(ROOT))


def main() -> int:
    _ensure_models()

    host = os.getenv("HOST", "0.0.0.0")
    port = os.getenv("PORT", "8002")
    stop_event = threading.Event()
    procs: dict[str, subprocess.Popen | None] = {"worker": None, "api": None}

    def _handle_signal(signum, _frame):
        print(f"[cloud_entrypoint] Received signal {signum}; shutting down API and worker...", flush=True)
        stop_event.set()
        for name in ("api", "worker"):
            p = procs.get(name)
            if p is not None and p.poll() is None:
                p.terminate()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    def _supervise_worker() -> None:
        while not stop_event.is_set():
            print("[cloud_entrypoint] Starting durable SQLite job worker...", flush=True)
            p = subprocess.Popen([sys.executable, "-m", "deep_shield.jobs.worker"], cwd=str(ROOT))
            procs["worker"] = p
            rc = p.wait()
            if stop_event.is_set():
                break
            print(f"[cloud_entrypoint] Worker exited unexpectedly (code {rc}); restarting in 3s...", flush=True)
            stop_event.wait(3.0)

    worker_thread = threading.Thread(target=_supervise_worker, daemon=True)
    worker_thread.start()

    print(f"[cloud_entrypoint] Starting FastAPI server on {host}:{port}...", flush=True)
    api = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "deep_shield.api.main:app",
            "--host",
            host,
            "--port",
            str(port),
            "--workers",
            "1",
        ],
        cwd=str(ROOT),
    )
    procs["api"] = api
    exit_code = api.wait()
    stop_event.set()
    w = procs.get("worker")
    if w is not None and w.poll() is None:
        w.terminate()
        try:
            w.wait(timeout=10)
        except subprocess.TimeoutExpired:
            w.kill()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
