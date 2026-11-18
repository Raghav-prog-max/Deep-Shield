"""Create Module 04 demo images from the public-domain NASA test portrait (tests/assets/face_sample.jpg).

    python scripts/make_m04_demo.py          # -> demo/m04_clone.jpg, demo/m04_ghost.jpg, demo/m04_original.jpg

m04_clone.jpg : a 160x160 region copied elsewhere in the image (copy-move)
m04_ghost.jpg : a region taken from a quality-60 save pasted into a quality-90 image (JPEG ghost)
These are clearly labelled synthetic manipulations for demonstration only.
"""
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
src = cv2.resize(cv2.imread(str(ROOT / "tests" / "assets" / "face_sample.jpg")), (768, 768))
out = ROOT / "demo"
out.mkdir(exist_ok=True)

cv2.imwrite(str(out / "m04_original.jpg"), src, [cv2.IMWRITE_JPEG_QUALITY, 90])

clone = src.copy()
clone[480:720, 470:710] = src[60:300, 90:330]                     # copy the hair/face-edge region down-right
cv2.imwrite(str(out / "m04_clone.jpg"), clone, [cv2.IMWRITE_JPEG_QUALITY, 90])

low = cv2.imdecode(cv2.imencode(".jpg", src, [cv2.IMWRITE_JPEG_QUALITY, 60])[1], cv2.IMREAD_COLOR)
ghost = src.copy()
ghost[300:540, 300:540] = low[300:540, 300:540]
cv2.imwrite(str(out / "m04_ghost.jpg"), ghost, [cv2.IMWRITE_JPEG_QUALITY, 90])
print("wrote demo/m04_original.jpg, demo/m04_clone.jpg, demo/m04_ghost.jpg")
