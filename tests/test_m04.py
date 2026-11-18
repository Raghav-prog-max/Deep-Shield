"""Module 04 — Visual Forensics tests (no model download needed)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from deep_shield.engines.forensics import ForensicsConfig, VisualForensicsEngine
from deep_shield.engines.forensics import copymove, jpeg

ASSETS = Path(__file__).parent / "assets"


def _img():
    return cv2.resize(cv2.imread(str(ASSETS / "face_sample.jpg")), (512, 512))


@pytest.fixture()
def engine(tmp_path):
    cfg = ForensicsConfig()
    cfg.evidence_dir = tmp_path / "m04"
    return VisualForensicsEngine(cfg)


def test_copy_move_found_and_located(tmp_path):
    b = _img()
    c = b.copy()
    c[300:460, 320:480] = b[40:200, 60:220]           # clone a textured 160x160 region
    p = tmp_path / "clone.jpg"
    cv2.imwrite(str(p), c, [cv2.IMWRITE_JPEG_QUALITY, 90])
    r = copymove.detect(cv2.imread(str(p)))
    assert r["clusters"], "clone not detected"
    boxes = [cl["target_bbox"] for cl in r["clusters"]] + [cl["source_bbox"] for cl in r["clusters"]]
    assert any(abs(x - 320) < 60 and abs(y - 300) < 60 for x, y, _, _ in boxes)


def test_copy_move_no_false_positive_on_untouched():
    assert copymove.detect(_img())["clusters"] == []


def test_jpeg_quality_and_grid(tmp_path):
    b = _img()
    p = tmp_path / "q75.jpg"
    cv2.imwrite(str(p), b, [cv2.IMWRITE_JPEG_QUALITY, 75])
    assert jpeg.quality_from_tables(p)["estimated_quality"] == 75
    j = cv2.imread(str(p))
    assert jpeg.grid_alignment(j)["aligned"]
    assert not jpeg.grid_alignment(j[5:, 3:])["aligned"]


def test_jpeg_ghost_reveals_low_quality_paste():
    b = _img()
    low = cv2.imdecode(cv2.imencode(".jpg", b, [cv2.IMWRITE_JPEG_QUALITY, 60])[1], 1)
    comp = b.copy()
    comp[200:360, 200:360] = low[200:360, 200:360]
    comp = cv2.imdecode(cv2.imencode(".jpg", comp, [cv2.IMWRITE_JPEG_QUALITY, 90])[1], 1)
    g = jpeg.ghosts(comp)
    m = g["_maps"][g["qualities"].index(60)]
    assert m[220:340, 220:340].mean() < m[:150, :150].mean() - 0.2


def test_engine_image_all_tasks(engine, tmp_path):
    b = _img()
    c = b.copy()
    c[300:460, 320:480] = b[40:200, 60:220]
    p = tmp_path / "clone.jpg"
    cv2.imwrite(str(p), c, [cv2.IMWRITE_JPEG_QUALITY, 90])
    r = engine.analyze(p)
    assert r.status == "completed", r.error
    s = {x.task: x for x in r.task_summaries}
    assert set(s) == {"copy_move", "ela_noise", "jpeg_analysis", "metadata"}
    assert s["copy_move"].classification == "copy_move_candidates_found"
    assert "quality ~90" in s["jpeg_analysis"].basis
    cm = [f for f in r.findings if f.classification == "copy_move_candidate"]
    assert cm and cm[0].region is not None and "source_bbox" in cm[0].explanation
    for f in r.findings:
        for e in f.evidence:
            assert (engine.cfg.evidence_dir / e.path).exists()
    assert r.model_dump_json()


def test_engine_video_selected_frames(engine, tmp_path):
    b = _img()
    p = tmp_path / "v.mp4"
    vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 25, (512, 512))
    for i in range(60):
        vw.write(np.roll(b, i, axis=1))
    vw.release()
    r = engine.analyze(p, frames=[10, 40])
    assert r.status == "completed", r.error
    assert r.media_info["frames_analysed"] == [10, 40]
    assert {f.frame_index for f in r.findings} == {10, 40}
    r2 = engine.analyze(p)                                  # default keyframes
    assert len(r2.media_info["frames_analysed"]) == 3


def test_api_m04(engine, tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import deep_shield.api.routes_forensics as routes
    routes.get_engine.cache_clear()
    monkeypatch.setattr(routes, "get_engine", lambda: engine)
    app = FastAPI()
    app.include_router(routes.router)
    c = TestClient(app)
    p = tmp_path / "x.jpg"
    cv2.imwrite(str(p), _img())
    with open(p, "rb") as fh:
        resp = c.post("/api/v1/m04/analyze", files={"file": ("x.jpg", fh, "image/jpeg")})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    ev = [e["path"] for f in body["findings"] for e in f["evidence"]][0]
    assert c.get(f"/api/v1/m04/evidence/{ev}").status_code == 200
    assert c.get("/api/v1/m04/health").json()["module"] == "M04_visual_forensics"
