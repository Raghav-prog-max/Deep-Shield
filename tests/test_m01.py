"""Module 01 — Threat & Anomaly Intelligence engine and API tests."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

import deep_shield.api.routes_threat as routes_threat
from deep_shield.engines.threat import ThreatConfig, ThreatIntelligenceEngine

ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).parent / "assets"
EVAL_VIDEOS = ROOT / "eval" / "videos" / "test"


def test_m01_engine_image_and_video(tmp_path: Path):
    cfg = ThreatConfig(evidence_dir=tmp_path / "m01_ev")
    eng = ThreatIntelligenceEngine(cfg)
    h = eng.health()
    assert h["module"] == "M01_threat_intelligence"
    assert h["detector"]["ready"] is True

    # Image analysis
    img_res = eng.analyze(ASSETS / "face_sample.jpg")
    assert img_res.status == "completed", img_res.error
    assert img_res.findings, "expected at least one person detection on face_sample.jpg"
    for f in img_res.findings:
        for ev in f.evidence:
            assert (cfg.evidence_dir / ev.path).is_file()

    # Video analysis with configured zone
    video_path = EVAL_VIDEOS / "test_corridor_01.mp4"
    zones = [{"zone_id": "restricted_lobby", "polygon": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]}]
    vid_res = eng.analyze(video_path, zones=zones)
    assert vid_res.status == "completed", vid_res.error
    tasks = {s.task for s in vid_res.task_summaries}
    assert tasks == {"object_detection", "zone_entry", "loitering", "unattended_object"}


def test_m01_api_endpoints(tmp_path: Path, monkeypatch):
    cfg = ThreatConfig(evidence_dir=tmp_path / "m01_api_ev")
    eng = ThreatIntelligenceEngine(cfg)
    routes_threat.get_engine.cache_clear()
    monkeypatch.setattr(routes_threat, "get_engine", lambda: eng)

    app = FastAPI()
    app.include_router(routes_threat.router)
    client = TestClient(app)

    assert client.get("/api/v1/m01/health").status_code == 200

    with open(ASSETS / "face_sample.jpg", "rb") as fh:
        resp = client.post("/api/v1/m01/analyze", files={"file": ("face_sample.jpg", fh, "image/jpeg")})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["module"] == "M01_threat_intelligence"
    ev_path = body["findings"][0]["evidence"][0]["path"]
    assert client.get(f"/api/v1/m01/evidence/{ev_path}").status_code == 200
    assert client.delete(f"/api/v1/m01/evidence/{body['media_id']}").status_code == 200
