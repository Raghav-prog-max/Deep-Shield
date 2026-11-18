"""Module 02 tests. Run with the stub backend (no model download needed):

    DEEPSHIELD_M02_BACKEND=stub pytest -q

Set DEEPSHIELD_TEST_REAL=1 (after scripts/download_models.py) to also smoke-test the real models.
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

os.environ.setdefault("DEEPSHIELD_M02_BACKEND", "stub")

from deep_shield.engines.deepfake import DeepfakeImageIntegrityEngine, EngineConfig  # noqa: E402
from deep_shield.engines.deepfake.metrics import binary_metrics, roc_auc  # noqa: E402
from deep_shield.engines.deepfake.temporal import FaceObs, build_tracks, summarise  # noqa: E402

ASSETS = Path(__file__).parent / "assets"


@pytest.fixture(scope="session")
def media(tmp_path_factory):
    d = tmp_path_factory.mktemp("media")
    face = ASSETS / "face_sample.jpg"
    img = cv2.imread(str(face))
    assert img is not None, "tests/assets/face_sample.jpg missing"
    cv2.imwrite(str(d / "face.jpg"), img)
    rng = np.random.default_rng(0)
    cv2.imwrite(str(d / "noface.png"), (rng.random((240, 320, 3)) * 255).astype(np.uint8))
    vw = cv2.VideoWriter(str(d / "clip.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 25, (img.shape[1], img.shape[0]))
    for i in range(100):
        m = np.float32([[1, 0, int(20 * np.sin(i / 15))], [0, 1, 0]])
        vw.write(cv2.warpAffine(img, m, (img.shape[1], img.shape[0])))
    vw.release()
    (d / "bad.txt").write_text("x")
    return d


@pytest.fixture(scope="session")
def engine(tmp_path_factory):
    cfg = EngineConfig()
    cfg.backend = os.getenv("DEEPSHIELD_M02_BACKEND", "stub")
    cfg.evidence_dir = tmp_path_factory.mktemp("evidence")
    eng = DeepfakeImageIntegrityEngine(cfg)
    if not eng.detector.available:
        pytest.skip("No face detector (install opencv-python 4.x or run scripts/download_models.py)")
    return eng


def test_image_two_tasks(engine, media):
    r = engine.analyze(media / "face.jpg", tasks=["face_manipulation", "synthetic_image"])
    assert r.status == "completed"
    tasks = {s.task: s for s in r.task_summaries}
    assert set(tasks) == {"face_manipulation", "synthetic_image"}
    face = [f for f in r.findings if f.task == "face_manipulation"]
    assert face and face[0].region is not None and 0 <= face[0].score <= 1
    assert face[0].score_type.value == "classifier_probability"
    for f in r.findings:
        for e in f.evidence:
            assert (engine.cfg.evidence_dir / e.path).exists()


def test_no_face(engine, media):
    r = engine.analyze(media / "noface.png", tasks=["face_manipulation", "synthetic_image"])
    s = {x.task: x for x in r.task_summaries}
    assert s["face_manipulation"].classification == "not_applicable_no_face_detected"
    assert s["synthetic_image"].score is not None


def test_video_temporal(engine, media):
    r = engine.analyze(media / "clip.mp4", tasks=["face_manipulation", "synthetic_image"])
    assert r.status == "completed", r.error
    tracks = [f for f in r.findings if f.level == "track"]
    assert tracks, "expected at least one face track"
    t = tracks[0]
    assert t.temporal.n_observations >= 3 and t.time_end_s > t.time_start_s
    assert any(e.kind == "score_timeline" for e in t.evidence)
    frames = [f for f in r.findings if f.level == "frame" and f.task == "face_manipulation"]
    assert all(f.timestamp_s is not None and f.track_id == t.track_id for f in frames if f.track_id == t.track_id)
    vid = [f for f in r.findings if f.level == "video" and f.task == "synthetic_image"]
    assert vid and vid[0].classification != "ai_generation_indicated"   # capped on video


def test_default_image_tasks_full(engine, media):
    r = engine.analyze(media / "face.jpg")
    tasks = [s.task for s in r.task_summaries]
    assert tasks == ["face_manipulation", "frequency_analysis", "image_manipulation"]
    face = [f for f in r.findings if f.task == "face_manipulation"][0]
    if face.score >= 0.35:   # heatmap is produced unless the face is clearly 'real'
        assert any(e.kind == "region_heatmap" for e in face.evidence)
        assert face.explanation["method"] == "occlusion_sensitivity"
        if face.explanation["informative"]:
            assert len(face.explanation["top_region_image"]) == 4
        else:
            assert "top_region_image" not in face.explanation
    fq = [f for f in r.findings if f.task == "frequency_analysis"]
    assert fq and fq[0].track_id == "F1" and "hf_excess" in fq[0].explanation
    mp = [f for f in r.findings if f.task == "image_manipulation"][0]
    assert mp.score is None and mp.classification == "exploratory_visualization"
    assert "metadata" in mp.explanation
    for f in r.findings:
        for e in f.evidence:
            assert (engine.cfg.evidence_dir / e.path).exists(), e.path
    assert r.model_dump_json()  # serialisable


def test_video_default_face_only(engine, media):
    r = engine.analyze(media / "clip.mp4")
    assert [s.task for s in r.task_summaries] == ["face_manipulation"]
    tr = [f for f in r.findings if f.level == "track"][0]
    if tr.score >= 0.35:
        assert any(e.kind == "region_heatmap" for e in tr.evidence)


def test_video_explicit_image_task_not_applicable(engine, media):
    r = engine.analyze(media / "clip.mp4", tasks=["face_manipulation", "image_manipulation"])
    s = {x.task: x.classification for x in r.task_summaries}
    assert s["image_manipulation"] == "not_applicable_video_input" and r.status == "completed"


def test_manipulation_maps_show_paste(tmp_path):
    """The noise-level map must make a noisy pasted region stand out (visual aid, no verdict)."""
    import cv2 as _cv
    from deep_shield.engines.deepfake.manipulation import analyse
    rng = np.random.default_rng(0)
    base = _cv.imread(str(ASSETS / "face_sample.jpg"))
    base = _cv.resize(base, (512, 512))
    a = base.copy()
    patch = a[300:460, 300:460].astype(np.float32) + rng.normal(0, 12, (160, 160, 3))
    a[300:460, 300:460] = np.clip(patch, 0, 255).astype(np.uint8)
    r = analyse(a, block=16)
    logr = r["_logr"]
    inside = logr[300 // 16:460 // 16, 300 // 16:460 // 16].mean()
    assert inside > 1.5            # > ~3x the image's median noise level
    assert analyse(base, block=16)["_logr"][300 // 16:460 // 16, 300 // 16:460 // 16].mean() < inside - 1


def test_spectral_fingerprint_detects_upsampling_artifacts():
    from PIL import Image as _I
    from deep_shield.engines.deepfake.frequency import spectral_fingerprint
    rng = np.random.default_rng(1)
    smooth = _I.fromarray((rng.random((64, 64)) * 255).astype(np.uint8)).resize((256, 256), _I.BICUBIC)
    nn = _I.fromarray((rng.random((64, 64)) * 255).astype(np.uint8)).resize((256, 256), _I.NEAREST)
    # nearest-neighbour up-sampling leaves high-frequency energy (like checkerboard artefacts)
    assert spectral_fingerprint(nn)["hf_excess"] > spectral_fingerprint(smooth)["hf_excess"]


def test_tracker_centre_fallback():
    # face jumps 34 px between sparse samples (IoU 0.28 < 0.3) -> still one track with the fallback
    per = [[FaceObs(i, i * 12, i * 0.5, (100 + 34 * i, 100, 60, 60), 0.9)] for i in range(5)]
    assert len(build_tracks(per, 0.3, 3)) > 1
    assert len(build_tracks(per, 0.3, 3, center_dist=0.6)) == 1


def test_unsupported(engine, media):
    with pytest.raises(ValueError):
        engine.analyze(media / "bad.txt")


def test_tracker_and_segments():
    per = [[FaceObs(i, i * 10, i * 0.4, (100 + i, 100, 80, 80), 0.9, score=s)]
           for i, s in enumerate([0.2, 0.8, 0.9, 0.3, 0.7, 0.75])]
    per.append([FaceObs(6, 60, 2.4, (400, 50, 60, 60), 0.9, score=0.1)])
    tr = build_tracks(per, 0.3, 2)
    assert len(tr) == 2 and len(tr[0].obs) == 6
    ts = summarise([(o.timestamp_s, o.frame_index, o.score) for o in tr[0].obs], 0.65, 0.4)
    assert len(ts.segments_above_high) == 2 and ts.max == 0.9


def test_metrics():
    assert roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)
    m = binary_metrics([0, 1, 1, 0], [0.2, 0.9, 0.4, 0.6])
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)


def test_api(engine, media, monkeypatch):
    from fastapi.testclient import TestClient

    import deep_shield.api.routes_deepfake as routes
    routes.get_engine.cache_clear()
    monkeypatch.setattr(routes, "get_engine", lambda: engine)
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(routes.router)
    c = TestClient(app)
    with open(media / "face.jpg", "rb") as f:
        resp = c.post("/api/v1/m02/analyze", files={"file": ("face.jpg", f, "image/jpeg")},
                      data={"tasks": "face_manipulation"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["task_summaries"][0]["task"] == "face_manipulation"
    ev = body["findings"][0]["evidence"][0]["path"]
    assert c.get(f"/api/v1/m02/evidence/{ev}").status_code == 200
    assert c.get("/api/v1/m02/evidence/..%2F..%2Fetc/passwd").status_code in (400, 404)
    assert c.post("/api/v1/m02/analyze", files={"file": ("x.txt", b"hi", "text/plain")}).status_code == 415
    assert c.delete(f"/api/v1/m02/evidence/{body['media_id']}").status_code == 200


@pytest.mark.skipif(os.getenv("DEEPSHIELD_TEST_REAL") != "1", reason="real models not requested")
def test_real_models_load(tmp_path, media):
    cfg = EngineConfig()
    cfg.backend = "hf"
    cfg.evidence_dir = tmp_path
    r = DeepfakeImageIntegrityEngine(cfg).analyze(media / "face.jpg")
    assert r.status == "completed", r.error
    face = [s for s in r.task_summaries if s.task == "face_manipulation"][0]
    assert face.model and face.model.backend == "transformers"


def test_heatmap_honest_scaling():
    """Tiny score drops must not be drawn as a confident hot spot; a real local cue must be."""
    from PIL import Image as _I
    from deep_shield.engines.deepfake.explain import occlusion_map

    class _P:
        def __init__(self, v):
            self.score = v

    class _Flat:   # score barely reacts to any occlusion (like a fully synthetic face)
        def classify(self, ims):
            return [_P(0.98 - 0.0004 * (np.asarray(i.convert("L")).std() < 60)) for i in ims]

    class _Local:  # score depends on one region (eye band)
        def classify(self, ims):
            out = []
            for i in ims:
                a = np.asarray(i.convert("L").resize((224, 224)), np.float32)
                out.append(_P(float(np.clip(a[60:110, 40:184].std() / 60, 0, 1))))
            return out

    im = _I.open(ASSETS / "face_sample.jpg").crop((150, 40, 330, 220))
    heat_f, info_f = occlusion_map(_Flat(), im)
    heat_l, info_l = occlusion_map(_Local(), im)
    assert not info_f["informative"] and heat_f.max() < 0.2
    assert info_l["informative"] and heat_l.max() > 0.9
