"""Download the MS-COCO neural object detector checkpoint into ignored models/.

By default downloads ``hustvl/yolos-tiny`` (PyTorch/HuggingFace MS-COCO 2017 ViT detector, ~26 MB,
Apache-2.0) into ``models/hustvl__yolos-tiny``. Pass ``--yolo`` if ``ultralytics`` is installed and
you prefer ``models/yolo26n.pt``.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path


def download_yolos_tiny(models: Path) -> None:
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    target = models / "hustvl__yolos-tiny"
    target.mkdir(parents=True, exist_ok=True)
    model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-tiny")
    proc = AutoImageProcessor.from_pretrained("hustvl/yolos-tiny")
    model.save_pretrained(target)
    proc.save_pretrained(target)
    weights = target / "model.safetensors"
    sha = hashlib.sha256(weights.read_bytes()).hexdigest() if weights.is_file() else "unknown"
    print(f"Saved hustvl/yolos-tiny -> {target}")
    print(f"SHA-256: {sha}")


def download_yolo(models: Path) -> None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install ultralytics before running with --yolo.") from exc

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo", action="store_true", help="Download Ultralytics yolo26n.pt instead of hustvl/yolos-tiny")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    models = root / "models"
    models.mkdir(parents=True, exist_ok=True)
    if args.yolo:
        download_yolo(models)
    else:
        download_yolos_tiny(models)


if __name__ == "__main__":
    main()
