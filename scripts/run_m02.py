"""Command-line runner for Module 02.

    python scripts/run_m02.py samples/face.jpg
    python scripts/run_m02.py samples/clip.mp4 --tasks face_manipulation --out result.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deep_shield.engines.deepfake import ALL_TASKS, DeepfakeImageIntegrityEngine, EngineConfig  # noqa: E402
from deep_shield.engines.deepfake.config import DEFAULT_TASKS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="DEEP SHIELD Module 02 — Deepfake & Image Integrity")
    ap.add_argument("media")
    ap.add_argument("--tasks", default=None,
                    help=f"comma list; default for images: {','.join(DEFAULT_TASKS)} "
                         "(videos: face_manipulation). Add synthetic_image for the experimental AI-image check.")
    ap.add_argument("--assume-face-crop", action="store_true", help="image is already a face crop")
    ap.add_argument("--out", help="write full JSON result here")
    ap.add_argument("--backend", choices=["hf", "stub"], help="override DEEPSHIELD_M02_BACKEND")
    a = ap.parse_args()

    cfg = EngineConfig()
    if a.backend:
        cfg.backend = a.backend
    eng = DeepfakeImageIntegrityEngine(cfg)
    tasks = [t for t in a.tasks.split(",") if t] if a.tasks else None
    r = eng.analyze(a.media, tasks=tasks, assume_face_crop=a.assume_face_crop)

    print(f"\nDEEP SHIELD · {r.module} · {r.engine_version}")
    print(f"media   : {r.source_name} ({r.media_type.value}, id {r.media_id})")
    print(f"status  : {r.status}   time: {r.processing_ms} ms   face detector: {r.media_info.get('face_detector')}")
    if r.error:
        print(f"error   : {r.error}")
    print("-" * 96)
    print(f"{'task':<20}{'classification':<36}{'score':>7}  basis")
    for s in r.task_summaries:
        sc = f"{s.score:.3f}" if s.score is not None else "  -  "
        print(f"{s.task:<20}{s.classification:<36}{sc:>7}  {s.basis}")
    print("-" * 96)
    print(f"{len(r.findings)} findings; evidence under {cfg.evidence_dir / r.media_id}")
    for w in r.warnings:
        print(f"! {w}")
    print(f"\n{r.disclaimer}")
    if a.out:
        Path(a.out).write_text(r.model_dump_json(indent=2))
        print(f"JSON -> {a.out}")
    return 0 if r.status != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
