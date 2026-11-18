"""End-to-end integration tests for M01/M03 rules, cross-module intelligence linking,
durable SQLite job queue (with stale worker recovery), PDF reporting, and evaluation runner."""
from __future__ import annotations

from pathlib import Path

from deep_shield.common.schema import BBox, Finding, MediaType, ModelInfo, ModuleResult, ScoreType, TaskSummary
from deep_shield.engines.threat.config import Zone
from deep_shield.engines.threat.detector import Detection
from deep_shield.engines.threat.rules import (
    FrameSnapshot,
    crowding_events,
    loitering_events,
    running_events,
    unattended_object_events,
    zone_transition_events,
)
from deep_shield.intelligence.linker import link_module_results
from deep_shield.jobs.store import AnalysisJobStore
from deep_shield.reporting.pdf import render_analysis_report


def test_m01_m03_rules_all_events():
    zone = Zone(zone_id="lobby", polygon=[(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)])
    w, h = 1000, 1000
    dets = [
        Detection("person", 0.9, (50, 400, 60, 150), track_id="T1", frame_index=0, timestamp_s=0.0),
        Detection("person", 0.9, (400, 400, 60, 150), track_id="T1", frame_index=25, timestamp_s=1.0),
        Detection("person", 0.9, (410, 400, 60, 150), track_id="T1", frame_index=800, timestamp_s=32.0),
        Detection("person", 0.9, (900, 400, 60, 150), track_id="T1", frame_index=825, timestamp_s=33.0),
    ]
    transitions = zone_transition_events(dets, [zone], w, h, include_exits=True, max_gap_s=35.0)
    kinds = [e.event_type for e in transitions]
    assert "zone_entry" in kinds and "zone_exit" in kinds

    loiter = loitering_events(dets, [zone], w, h, dwell_seconds=30.0, sample_interval_s=16.0)
    assert len(loiter) == 1 and loiter[0].event_type == "loitering_candidate"

    # Running track
    run_dets = [
        Detection("person", 0.9, (100, 300, 50, 100), track_id="R1", frame_index=0, timestamp_s=0.0),
        Detection("person", 0.9, (260, 300, 50, 100), track_id="R1", frame_index=12, timestamp_s=0.5),
        Detection("person", 0.9, (430, 300, 50, 100), track_id="R1", frame_index=25, timestamp_s=1.0),
    ]
    runs = running_events(run_dets, threshold_body_lengths_per_second=2.5, sample_interval_s=0.5)
    assert len(runs) == 1 and runs[0].event_type == "running_candidate"

    # Unattended bag + crowding
    snaps = []
    for step, t in enumerate([0.0, 20.0, 40.0, 65.0]):
        fd = [
            Detection("backpack", 0.88, (500, 500, 40, 40), track_id="B1", frame_index=step * 50, timestamp_s=t),
        ]
        for p_idx in range(4):
            fd.append(
                Detection(
                    "person",
                    0.85,
                    (300 + p_idx * 40, 250, 50, 120),
                    track_id=f"P{p_idx}",
                    frame_index=step * 50,
                    timestamp_s=t,
                )
            )
        snaps.append(FrameSnapshot(step * 50, t, w, h, fd))

    unattended = unattended_object_events(snaps, unattended_seconds=60.0, stationary_body_lengths=0.35, sample_interval_s=25.0)
    assert len(unattended) == 1 and unattended[0].event_type == "unattended_object_candidate"

    crowds = crowding_events(snaps, [zone], min_people=4, min_duration_s=5.0, merge_gap_s=3.0)
    assert len(crowds) == 1 and crowds[0].event_type == "crowding_candidate"


def test_intelligence_linking_and_pdf_and_durable_store(tmp_path: Path):
    sha = "a" * 64
    mid = sha[:16]
    r1 = ModuleResult(
        module="M01_threat_intelligence",
        engine_version="m01-0.1.0",
        media_id=mid,
        media_sha256=sha,
        media_type=MediaType.VIDEO,
        source_name="corridor.mp4",
        status="completed",
        tasks_requested=["zone_entry"],
        task_summaries=[TaskSummary(task="zone_entry", classification="zone_entry_candidate", basis="rule trigger")],
        findings=[
            Finding(
                module="M01_threat_intelligence",
                task="zone_entry",
                level="event",
                media_id=mid,
                media_type=MediaType.VIDEO,
                classification="zone_entry_candidate",
                score=1.0,
                score_type=ScoreType.RULE_SCORE,
                timestamp_s=10.0,
                time_start_s=10.0,
                time_end_s=12.0,
                region=BBox(x=100, y=100, w=200, h=300),
                model=ModelInfo(model_id="rule-based-v1", task="zone_entry", backend="python-rules"),
            )
        ],
    )
    r2 = ModuleResult(
        module="M02_deepfake_image_integrity",
        engine_version="m02-0.3.0",
        media_id=mid,
        media_sha256=sha,
        media_type=MediaType.VIDEO,
        source_name="corridor.mp4",
        status="completed",
        tasks_requested=["face_manipulation"],
        task_summaries=[TaskSummary(task="face_manipulation", classification="inconclusive", score=0.48, basis="track mean")],
        findings=[
            Finding(
                module="M02_deepfake_image_integrity",
                task="face_manipulation",
                level="track",
                media_id=mid,
                media_type=MediaType.VIDEO,
                classification="inconclusive",
                score=0.48,
                score_type=ScoreType.AGGREGATE_MEAN_CLASSIFIER,
                timestamp_s=11.0,
                time_start_s=9.5,
                time_end_s=13.0,
                region=BBox(x=130, y=110, w=70, h=70),
                model=ModelInfo(model_id="prithivMLmods/open-deepfake-detection + deepshield-face-head-v1", task="face_manipulation", backend="transformers+trained_head"),
            )
        ],
    )
    linked = link_module_results([r1, r2])
    assert len(linked.links) == 1
    assert linked.links[0].relationship == "spatial_temporal_overlap"
    assert len(linked.groups) == 1

    pdf_bytes = render_analysis_report(linked, job_id="job-123")
    assert pdf_bytes.startswith(b"%PDF") and len(pdf_bytes) > 1000

    # Durable job store + worker lease recovery
    db_path = tmp_path / "jobs.sqlite3"
    store = AnalysisJobStore(db_path)
    upload = tmp_path / "upload.mp4"
    upload.write_bytes(b"fake video bytes")

    created = store.create("job-123", "corridor.mp4", upload, [{"zone_id": "lobby", "polygon": [[0, 0], [1, 0], [1, 1]]}])
    assert created.status == "queued"

    claimed1 = store.claim_next("worker-1", lease_seconds=0)
    assert claimed1 is not None and claimed1.job_id == "job-123" and claimed1.attempts == 1
    assert store.heartbeat("job-123", "worker-1")

    # Simulate stale worker recovery by claiming with lease_seconds=-1
    claimed2 = store.claim_next("worker-2", lease_seconds=-1)
    assert claimed2 is not None and claimed2.job_id == "job-123" and claimed2.attempts == 2

    assert store.finish("job-123", "worker-2", status="completed", media_id=mid, result=linked)
    final_rec = store.get("job-123")
    assert final_rec is not None and final_rec.status == "completed"
    assert final_rec.result is not None and len(final_rec.result.groups) == 1
