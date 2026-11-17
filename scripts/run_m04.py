"""Command-line runner for Module 04 — Visual Forensics.

    python scripts/run_m04.py photo.jpg
    python scripts/run_m04.py clip.mp4 --frames 37,120 --out m04.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deep_shield.engines.forensics import ForensicsConfig, VisualForensicsEngine  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="DEEP SHIELD Module 04 — Visual Forensics")
    ap.add_argument("media")
    ap.add_argument("--tasks", default=None, help="copy_move,ela_noise,jpeg_analysis,metadata (default: all for images)")
    ap.add_argument("--frames", default=None, help="video only: comma-separated frame indices")
    ap.add_argument("--out", help="write full JSON result here")
    a = ap.parse_args()

    cfg = ForensicsConfig()
    r = VisualForensicsEngine(cfg).analyze(
        a.media, tasks=[t for t in a.tasks.split(",") if t] if a.tasks else None,
        frames=[int(x) for x in a.frames.split(",") if x] if a.frames else None)

    print(f"\nDEEP SHIELD · {r.module} · {r.engine_version}")
    print(f"media   : {r.source_name} ({r.media_type.value}, id {r.media_id})")
    print(f"status  : {r.status}   time: {r.processing_ms} ms")
    if r.error:
        print(f"error   : {r.error}")
    print("-" * 110)
    print(f"{'task':<15}{'result':<30}basis")
    for s in r.task_summaries:
        print(f"{s.task:<15}{s.classification:<30}{s.basis}")
    print("-" * 110)
    for f in r.findings:
        if f.classification == "copy_move_candidate":
            e = f.explanation
            where = f" frame {f.frame_index}" if f.frame_index is not None else ""
            print(f"copy-move{where}: source {e['source_bbox']} -> target {e['target_bbox']}  "
                  f"({e['inliers']} matches, shift {e['shift_px']} px)")
    print(f"{len(r.findings)} findings; evidence under {cfg.evidence_dir / r.media_id}")
    print("\nAdvisory forensic indicators for human review; not proof of manipulation.")
    if a.out:
        Path(a.out).write_text(r.model_dump_json(indent=2))
        print(f"JSON -> {a.out}")
    return 0 if r.status != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
