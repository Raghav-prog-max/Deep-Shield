"""Download the official Ultralytics YOLO26n checkpoint into ignored models/."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path


def main() -> None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install requirements.txt before downloading the M01 checkpoint.") from exc

    root = Path(__file__).resolve().parents[1]
    models = root / "models"
    models.mkdir(parents=True, exist_ok=True)
    checkpoint = models / "yolo26n.pt"
    old_cwd = Path.cwd()
    try:
        os.chdir(models)
        YOLO("yolo26n.pt")
    finally:
        os.chdir(old_cwd)
    if not checkpoint.is_file():
        raise SystemExit(f"Ultralytics did not create the expected checkpoint: {checkpoint}")

    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    print(f"Saved {checkpoint}")
    print(f"SHA-256: {digest}")
    print("Checkpoint files are ignored by Git; retain this hash with your deployment records.")


if __name__ == "__main__":
    main()
