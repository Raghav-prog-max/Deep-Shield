"""Deep verification of all loaded neural model weights, calibrated parameters, and real multi-image / multi-video inference."""
from __future__ import annotations

from pathlib import Path

import cv2
from fastapi.testclient import TestClient

from deep_shield.api.main import app
from deep_shield.engines.deepfake import DeepfakeImageIntegrityEngine, EngineConfig
from deep_shield.engines.forensics import ForensicsConfig, VisualForensicsEngine
from deep_shield.engines.threat import ThreatConfig, ThreatIntelligenceEngine
from deep_shield.engines.threat.detector import Detection
from deep_shield.engines.video_events import VideoEventConfig, VideoEventEngine
from deep_shield.engines.video_events.gru import load_trained_gru, score_tracks_gru

ROOT = Path(__file__).resolve().parents[1]


def test_all_weights_and_parameters_loaded(monkeypatch):
    import deep_shield.api.routes_deepfake as routes_deepfake

    # 1. Verify all real model checkpoints & calibrated parameter artifacts exist on disk
    assert (ROOT / "models" / "hustvl__yolos-tiny" / "config.json").is_file()
    assert (ROOT / "models" / "m02_face_head.npz").is_file()
    assert (ROOT / "models" / "m02_synth_head.npz").is_file()
    assert (ROOT / "models" / "m03_temporal_gru.pt").is_file()
    assert (ROOT / "models" / "m04_forensics_params.json").is_file()

    # 2. Verify M03 real-track trained GRU bundle loads trained weights (6,265 params)
    load_trained_gru.cache_clear()
    gru_bundle = load_trained_gru()
    assert gru_bundle["model_id"].startswith("temporal-gru-kinematic-")
    assert gru_bundle["revision"].startswith("trained:")
    assert gru_bundle["n_parameters"] == 6265
    assert gru_bundle["test_auc"] >= 0.90

    # 3. Verify /api/v1/intelligence/models reports real neural backends (force hf regardless of test_m02 import order)
    monkeypatch.setenv("DEEPSHIELD_M02_BACKEND", "hf")
    routes_deepfake.get_engine.cache_clear()
    client = TestClient(app)
    resp = client.get("/api/v1/intelligence/models")
    routes_deepfake.get_engine.cache_clear()
    assert resp.status_code == 200, resp.text
    models_data = resp.json()
    m01_det = models_data["m01_threat_intelligence"]["detector"]
    assert m01_det["ready"] is True
    assert m01_det["backend"] in ("ultralytics/pytorch", "transformers/pytorch")
    assert models_data["m02_deepfake_image_integrity"]["tasks"]["face_manipulation"]["trained_head"] is not None
    assert models_data["m02_deepfake_image_integrity"]["tasks"]["synthetic_image"]["trained_head"] is not None
    assert models_data["m03_video_events"]["gru_enabled"] is True
    assert models_data["m03_video_events"]["gru_model"]["n_parameters"] == 6265
    assert models_data["m04_visual_forensics"]["calibrated_parameters"]["ela_jpeg_quality"] == 90


def test_fine_tuned_gru_separates_normal_from_anomalous_video_tracks():
    # Normal corridor walk (~0.4 body-lengths/s)
    normal_dets = [
        Detection("person", 0.92, (100 + i * 22, 180, 45, 110), track_id="T_WALK", frame_index=i * 12, timestamp_s=i * 0.5)
        for i in range(12)
    ]
    # Rapid sprint (~3.4 body-lengths/s)
    sprint_dets = [
        Detection("person", 0.94, (60 + i * 185, 180, 45, 110), track_id="T_SPRINT", frame_index=i * 12, timestamp_s=i * 0.5)
        for i in range(10)
    ]
    obs = {o.track_id: o for o in score_tracks_gru(normal_dets + sprint_dets, width=1920, height=1080)}
    assert set(obs) == {"T_WALK", "T_SPRINT"}
    assert obs["T_WALK"].score < obs["T_WALK"].threshold
    assert obs["T_SPRINT"].score >= obs["T_SPRINT"].threshold


def test_deep_multi_image_and_video_across_all_four_engines(tmp_path: Path):
    # 1. M02 REAL SigLIP2 + trained 4-source face head across held-out real & fake Kaggle/HF faces
    cfg2 = EngineConfig(backend="hf", evidence_dir=tmp_path / "ev_m02")
    eng2 = DeepfakeImageIntegrityEngine(cfg2)
    for img_path in [
        ROOT / "eval" / "faces" / "test" / "real" / "test_real_01.jpg",
        ROOT / "eval" / "faces" / "test" / "fake" / "test_fake_01.jpg",
        ROOT / "eval" / "synthetic" / "test" / "real" / "test_real_01.jpg",
        ROOT / "eval" / "synthetic" / "test" / "fake" / "test_fake_01.jpg",
    ]:
        res = eng2.analyze(img_path, tasks=["face_manipulation", "synthetic_image", "frequency_analysis", "image_manipulation"])
        assert res.status == "completed", res.error
        assert len(res.task_summaries) == 4
        if "faces" in str(img_path):
            face_f = [f for f in res.findings if f.task == "face_manipulation"]
            assert face_f and face_f[0].model.backend == "transformers+trained_head"
        synth_f = [f for f in res.findings if f.task == "synthetic_image"]
        assert synth_f and synth_f[0].model.backend == "transformers+trained_head"

    # 2. M04 calibrated parameters wired to copymove.detect() on real tampered photo
    eng4 = VisualForensicsEngine(ForensicsConfig(evidence_dir=tmp_path / "ev_m04"))
    base = cv2.resize(cv2.imread(str(ROOT / "eval" / "faces" / "test" / "real" / "test_real_01.jpg")), (512, 512))
    tampered = base.copy()
    tampered[290:450, 310:470] = base[30:190, 50:210]
    t_img = tmp_path / "tampered.jpg"
    cv2.imwrite(str(t_img), tampered, [cv2.IMWRITE_JPEG_QUALITY, 90])
    r4_img = eng4.analyze(t_img)
    assert r4_img.status == "completed"
    assert any(s.classification == "copy_move_candidates_found" for s in r4_img.task_summaries)
    cm_finding = [f for f in r4_img.findings if f.task == "copy_move"][0]
    assert cm_finding.explanation["params"]["ratio"] == eng4.cfg.sift_ratio

    # 3. M01 (real YOLOS-Tiny), M03 (+ real-track GRU), and M04 on real OpenCV vtest.avi surveillance clips
    eng1 = ThreatIntelligenceEngine(ThreatConfig(evidence_dir=tmp_path / "ev_m01"))
    eng3 = VideoEventEngine(VideoEventConfig(evidence_dir=tmp_path / "ev_m03"))
    zones = [{"zone_id": "restricted_lobby", "polygon": [[0.25, 0.20], [0.75, 0.20], [0.75, 0.80], [0.25, 0.80]]}]
    for vid_path in [
        ROOT / "eval" / "videos" / "calibration" / "calibration_corridor_01.mp4",
        ROOT / "eval" / "videos" / "test" / "test_corridor_01.mp4",
    ]:
        r1 = eng1.analyze(vid_path, zones=zones)
        r3 = eng3.analyze(vid_path, zones=zones)
        r4 = eng4.analyze(vid_path)
        assert r1.status == "completed" and len(r1.findings) > 0
        assert r1.findings[0].model.backend in ("ultralytics/pytorch", "transformers/pytorch")
        assert r3.status == "completed" and any(s.task == "temporal_gru" and s.score is not None for s in r3.task_summaries)
        assert r4.status == "completed" and len(r4.media_info["frames_analysed"]) == 3
