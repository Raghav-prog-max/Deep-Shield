"""Record a short webcam clip for the Module 02 video demo (consented footage only).

    python scripts/record_webcam.py                      # 15 s -> demo/clip.mp4
    python scripts/record_webcam.py --seconds 20 --out demo/clip2.mp4 --camera 1
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=15)
    ap.add_argument("--out", default="demo/clip.mp4")
    ap.add_argument("--camera", type=int, default=0)
    a = ap.parse_args()

    backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY  # faster open on Windows
    cap = cv2.VideoCapture(a.camera, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(a.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open the webcam. Close other apps using it (Teams/Zoom/Camera) or try --camera 1.")

    for _ in range(10):              # let auto-exposure settle
        cap.read()
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Webcam opened but returned no frames.")
    h, w = frame.shape[:2]
    fps = 20.0
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    print(f"Recording {a.seconds:.0f}s at {w}x{h} -> {a.out}  (look at the camera, move your head a little)")
    t0, n, next_t = time.time(), 0, time.time()
    while time.time() - t0 < a.seconds:
        ok, frame = cap.read()
        if not ok:
            break
        # write at a steady 20 fps so timestamps in the analysis match real time
        while next_t <= time.time() and time.time() - t0 < a.seconds:
            vw.write(frame)
            n += 1
            next_t += 1.0 / fps
    cap.release()
    vw.release()
    print(f"Saved {n} frames ({n / fps:.1f}s) to {Path(a.out).resolve()}")


if __name__ == "__main__":
    main()
