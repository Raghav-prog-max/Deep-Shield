"""Module 03 — Video Event Understanding & PyTorch Temporal GRU tests."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

import deep_shield.api.routes_video_events as routes_m03
from deep_shield.engines.threat.detector import Detection
from deep_shield.engines.video_events import VideoEventConfig, VideoEventEngine
from deep_shield.engines.video_events.gru import score_tracks_gru

ROOT = Path(__file__).resolve().parents[1]
EVAL_VIDEOS = ROOT / "eval" / "videos" / "test"


def test_m03_temporal_gru_track_scoring():
    dets = [
        Detection("person", 0.92, (50, 80, 40, 100), track_id="T1", frame_index=0, timestamp_s=0.0),
        Detection("person", 0.91, (190, 85, 40, 100), track_id="T1", frame_index=10, timestamp_s=0.5),
        Detection("person", 0.90, (340, 82, 40, 100), track_id="T1", frame_index=20, timestamp_s=1.0),
        Detection("person", 0.93, (490, 88, 40, 100), track_id="T1", frame_index=30, timestamp_s=1.5),
    ]
    obs = score_tracks_gru(dets, width=640, height=360)
    assert len(obs) == 1
    assert obs[0].track_id == "T1"
    assert 0.0 < obs[0].score < 1.0
    assert obs[0].peak_speed_bls > 2.0


def test_m03_engine_and_api(tmp_path: Path, monkeypatch):
    cfg = VideoEventConfig(evidence_dir=tmp_path / "m03_ev")
    eng = VideoEventEngine(cfg)
    h = eng.health()
    assert h["module"] == "M03_video_events"
    assert h["gru_enabled"] is True

    video_path = EVAL_VIDEOS / "test_corridor_01.mp4"
    zones = [{"zone_id": "restricted_lobby", "polygon": [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]}]
    res = eng.analyze(video_path, zones=zones)
    assert res.status == "completed", res.error
    tasks = {s.task for s in res.task_summaries}
    assert tasks == {"entry_exit", "loitering", "crowding", "running", "temporal_gru"}

    routes_m03.get_engine.cache_clear()
    monkeypatch.setattr(routes_m03, "get_engine", lambda: eng)
    app = FastAPI()
    app.include_router(routes_m03.router)
    client = TestClient(app)

    assert client.get("/api/v1/m03/health").json()["gru_enabled"] is True
    with open(video_path, "rb") as fh:
        resp = client.post(
            "/api/v1/m03/analyze",
            files={"file": ("test_corridor_01.mp4", fh, "video/mp4")},
            data={"zones": '[{"zone_id":"restricted_lobby","polygon":[[0.2,0.2],[0.8,0.2],[0.8,0.8],[0.2,0.8]]}]'},
        )
    assert resp.status_code == 200, resp.text
    assert resp.json()["module"] == "M03_video_events"
