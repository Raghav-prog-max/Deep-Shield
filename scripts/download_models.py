"""Download and pin Module 02 models into ./models and write models/MANIFEST.json (revision + file hashes).

    python scripts/download_models.py               # both classifiers + YuNet face detector
    python scripts/download_models.py --skip-yunet  # use the Haar cascade bundled with opencv-python 4.x

Run once on the demo machine while online; the engine then loads from ./models offline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deep_shield.engines.deepfake.config import SPECS  # noqa: E402

YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
             "face_detection_yunet_2023mar.onnx")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default=str(ROOT / "models"))
    ap.add_argument("--skip-yunet", action="store_true")
    args = ap.parse_args()
    out = Path(args.models_dir)
    out.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    api = HfApi()
    manifest = {"downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "models": {}}
    for task, spec in SPECS.items():
        info = api.model_info(spec.model_id)
        files = {s.rfilename for s in (info.siblings or [])}
        weight = "model.safetensors" if "model.safetensors" in files else "pytorch_model.bin"
        local = out / spec.model_id.replace("/", "__")
        print(f"[{task}] {spec.model_id} @ {info.sha[:10]}  ({weight})")
        snapshot_download(spec.model_id, revision=info.sha, local_dir=str(local),
                          allow_patterns=["config.json", "preprocessor_config.json", weight])
        pp = local / "preprocessor_config.json"
        if pp.exists():  # transformers 5.x dropped *FeatureExtractor classes; same settings, new name
            txt = pp.read_text()
            if "FeatureExtractor" in txt:
                pp.write_text(txt.replace("FeatureExtractor", "ImageProcessor"))
                print(f"  normalised legacy FeatureExtractor -> ImageProcessor in {pp.name}")
        wpath = local / weight
        manifest["models"][spec.model_id] = {
            "task": task, "revision": info.sha, "license": getattr(info.card_data, "license", None) if info.card_data else None,
            "weights_file": weight, "weights_sha256": sha256(wpath), "local_dir": str(local.relative_to(out.parent)),
        }

    if not args.skip_yunet:
        y = out / "face_detection_yunet_2023mar.onnx"
        if not y.exists():
            print("[face] downloading YuNet (OpenCV Zoo, Apache-2.0)")
            urllib.request.urlretrieve(YUNET_URL, y)
        manifest["face_detector"] = {"model": "YuNet 2023mar", "source": YUNET_URL, "sha256": sha256(y),
                                     "license": "apache-2.0"}

    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {out / 'MANIFEST.json'}")


if __name__ == "__main__":
    main()
