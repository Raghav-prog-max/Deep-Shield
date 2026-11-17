"""DEEP SHIELD Module 02 — Deepfake & Image Integrity engine.

Usage
-----
    from deep_shield.engines.deepfake import DeepfakeImageIntegrityEngine
    engine = DeepfakeImageIntegrityEngine()
    result = engine.analyze("clip.mp4")            # -> ModuleResult (pydantic)
    print(result.model_dump_json(indent=2))

Image tasks (default: face_manipulation, image_manipulation, frequency_analysis):
  face_manipulation  - face detection -> per-face deepfake score + suspicious-region heatmap
  frequency_analysis - spectral fingerprint of each face crop (uncalibrated indicator)
  image_manipulation - ELA + local noise-level maps + metadata review (visual, no verdict)
  synthetic_image    - whole-image AI-generation check (experimental, opt-in)
Video (default: face_manipulation): timestamped frame sampling -> face tracks -> per-track
temporal aggregation (mean / median / max / segments above threshold) + heatmap of the top frame.
"""
from __future__ import annotations

import logging
import time
import traceback
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence  # noqa: F401

import cv2
import numpy as np
from PIL import Image

from deep_shield.common.schema import (BBox, Finding, MediaType, ModelInfo, ModuleResult, ScoreType,
                                       TaskSummary)

from .classifiers import get_classifier
from . import frequency as freq
from . import manipulation as manip
from .config import (ALL_TASKS, DEFAULT_TASKS, ENGINE_VERSION, FREQ_LIMITATIONS, IMAGE_ONLY_TASKS,
                     MANIP_LIMITATIONS, MODULE_NAME, SPECS, TASK_FACE, TASK_FREQ, TASK_MANIP, TASK_SYNTH,
                     EngineConfig)
from .evidence import EvidenceStore, band_colour
from .explain import norm_box_to_image, occlusion_map, overlay
from .faces import FaceBox, FaceDetector, crop_box, crop_face
from .media import MediaError, load_image, media_kind, probe_video, read_frames, sample_video, sha256_file
from .temporal import FaceObs, build_tracks, summarise

log = logging.getLogger("deep_shield.m02")


def _bbox(t) -> BBox:
    x, y, w, h = t
    return BBox(x=int(x), y=int(y), w=int(w), h=int(h))


class DeepfakeImageIntegrityEngine:
    def __init__(self, cfg: Optional[EngineConfig] = None):
        self.cfg = cfg or EngineConfig()
        self.detector = FaceDetector(self.cfg)

    # ------------------------------------------------------------------ public
    def warmup(self, tasks: Iterable[str] = (TASK_FACE,)) -> None:
        for t in tasks:
            if t in SPECS:
                get_classifier(t, self.cfg)

    def health(self) -> Dict:
        info = {"module": MODULE_NAME, "engine_version": ENGINE_VERSION,
                "backend": self.cfg.backend, "face_detector": self.detector.backend, "tasks": {}}
        info["tasks"][TASK_FREQ] = {"ready": True, "method": "azimuthal power spectrum (numpy)", "default": TASK_FREQ in DEFAULT_TASKS}
        info["tasks"][TASK_MANIP] = {"ready": True, "method": "ELA + noise-level maps + EXIF (opencv)", "default": TASK_MANIP in DEFAULT_TASKS}
        for t in SPECS:
            try:
                c = get_classifier(t, self.cfg)
                info["tasks"][t] = {"ready": True, "model": c.model_info.model_dump(),
                                    "device": str(c.backend.device), "thresholds": c.thresholds()}
            except Exception as exc:  # noqa: BLE001
                info["tasks"][t] = {"ready": False, "error": f"{type(exc).__name__}: {exc}"}
        return info

    def analyze(self, path, tasks: Optional[Sequence[str]] = None, assume_face_crop: bool = False,
                source_name: Optional[str] = None) -> ModuleResult:
        t0 = time.perf_counter()
        path = Path(path)
        explicit = bool(tasks)
        tasks = list(dict.fromkeys(tasks or DEFAULT_TASKS))
        bad = [t for t in tasks if t not in ALL_TASKS]
        if bad:
            raise ValueError(f"Unknown task(s) {bad}; valid: {list(ALL_TASKS)}")
        if not path.exists():
            raise FileNotFoundError(path)

        kind = media_kind(path)
        if kind == "video" and not explicit:
            tasks = [t for t in tasks if t not in IMAGE_ONLY_TASKS] or [TASK_FACE]
        sha = sha256_file(path)
        res = ModuleResult(module=MODULE_NAME, engine_version=ENGINE_VERSION, media_id=sha[:16],
                           media_sha256=sha, media_type=MediaType(kind),
                           source_name=source_name or path.name, status="completed", tasks_requested=tasks)
        try:
            if kind == "image":
                self._analyze_image(path, tasks, assume_face_crop, res)
            else:
                if assume_face_crop:
                    res.warnings.append("assume_face_crop is ignored for video input.")
                self._analyze_video(path, tasks, res)
            if self.cfg.backend == "stub":
                res.warnings.insert(0, "STUB BACKEND: scores are synthetic placeholders, not model outputs. "
                                       "Do not use for demonstration of detection performance.")
            if any(s.classification.startswith("not_run") for s in res.task_summaries):
                res.status = "partial"
        except MediaError as exc:
            res.status, res.error = "failed", str(exc)
        except Exception as exc:  # noqa: BLE001
            log.error("M02 analysis failed: %s", traceback.format_exc())
            res.status, res.error = "failed", f"{type(exc).__name__}: {exc}"
        res.processing_ms = int((time.perf_counter() - t0) * 1000)
        return res

    # ------------------------------------------------------------------ image
    def _analyze_image(self, path: Path, tasks: List[str], assume_face_crop: bool, res: ModuleResult) -> None:
        bgr, pil = load_image(path)
        h, w = bgr.shape[:2]
        res.media_info = {"width": w, "height": h, "face_detector": self.detector.backend}
        store = EvidenceStore(self.cfg.evidence_dir, res.media_id)

        faces = None
        if TASK_FACE in tasks or TASK_FREQ in tasks:
            faces = self._locate_faces(bgr, pil, assume_face_crop)
        if TASK_FACE in tasks:
            self._image_faces(bgr, faces, assume_face_crop, store, res)
        if TASK_SYNTH in tasks:
            clf = get_classifier(TASK_SYNTH, self.cfg)
            p = clf.classify([pil])[0]
            ref = store.save_annotated(
                bgr, [], "synthetic_image_input.jpg", "annotated_image",
                "Input assessed for whole-image AI generation",
                header=f"synthetic_image score {p.score:.2f} -> {p.classification} (experimental)")
            res.findings.append(Finding(
                module=MODULE_NAME, task=TASK_SYNTH, level="image", media_id=res.media_id,
                media_type=MediaType.IMAGE, classification=p.classification, score=p.score,
                score_type=ScoreType.CLASSIFIER_PROBABILITY, thresholds=clf.thresholds(),
                predicted_label=p.predicted_label, raw_scores=p.raw, model=clf.model_info,
                evidence=[ref], limitations=list(clf.spec.limitations)))
            res.task_summaries.append(TaskSummary(
                task=TASK_SYNTH, classification=p.classification, score=p.score,
                score_type=ScoreType.CLASSIFIER_PROBABILITY, basis="whole image (experimental)",
                model=clf.model_info))
        if TASK_FREQ in tasks:
            self._image_frequency(pil, faces, store, res)
        if TASK_MANIP in tasks:
            self._image_manipulation(path, bgr, store, res)

    def _locate_faces(self, bgr, pil, assume_face_crop):
        """-> (boxes, crops, crop_boxes) or None when no detector is available."""
        h, w = bgr.shape[:2]
        if assume_face_crop:
            return [FaceBox(0, 0, w, h, None)], [pil], [(0, 0, w, h)]
        if not self.detector.available:
            return None
        boxes = self.detector.detect(bgr)
        cbs = [crop_box(bgr.shape, b, self.cfg.face_margin) for b in boxes]
        crops = [crop_face(bgr, b, self.cfg.face_margin) for b in boxes]
        return boxes, crops, cbs

    def _heatmap(self, clf, crop, cbox, label: str, name: str, store, score: float):
        """Occlusion heatmap evidence + explanation dict (only when the score is not clearly 'real')."""
        if not self.cfg.explain_faces or score < clf.spec.threshold_low:
            return None, None
        heat, info = occlusion_map(clf, crop, grid=self.cfg.occlusion_grid)
        if cbox is not None and info["informative"]:
            info["top_region_image"] = list(norm_box_to_image(info["top_region_norm"], cbox))
        if info["informative"]:
            ov = overlay(crop, heat, info["top_region_norm"], f"{label} | max drop {info['max_score_drop']:.2f}")
        else:
            ov = overlay(crop, heat, None, f"{label} | spread across face (max drop {info['max_score_drop']:.2f})")
        ref = store.save_bgr(ov, name, "region_heatmap",
                             f"{label}: occlusion-sensitivity map (red = hiding this area lowers the score most)")
        return ref, info

    def _image_faces(self, bgr, faces, assume_face_crop, store, res) -> None:
        if faces is None:
            res.warnings.append("No face detector available (YuNet model or Haar cascade not found).")
            res.task_summaries.append(TaskSummary(task=TASK_FACE, classification="not_run_no_face_detector",
                                                  basis="face detector unavailable"))
            return
        boxes, crops, cbs = faces
        if not boxes:
            res.warnings.append(f"No face >= {self.cfg.face_min_size}px detected; face-manipulation task not applicable.")
            res.task_summaries.append(TaskSummary(task=TASK_FACE, classification="not_applicable_no_face_detected",
                                                  basis=f"{self.detector.backend} face detector found no face"))
            return

        clf = get_classifier(TASK_FACE, self.cfg)
        preds = clf.classify(crops)
        lo, hi = clf.spec.threshold_low, clf.spec.threshold_high
        ann = store.save_annotated(
            bgr, [(b.as_tuple(), f"F{i + 1} {p.score:.2f}", band_colour(p.score, lo, hi))
                  for i, (b, p) in enumerate(zip(boxes, preds))],
            "faces_annotated.jpg", "annotated_image", "Detected faces with face-manipulation scores")
        extra = (["assume_face_crop: the whole image was treated as one face crop (no localisation)."]
                 if assume_face_crop else [])
        for i, (b, p, c, cb) in enumerate(zip(boxes, preds, crops, cbs)):
            crop_ref = store.save_crop(c, f"face_F{i + 1}.jpg", f"Face crop F{i + 1} given to the classifier")
            evid = [crop_ref, ann]
            href, hinfo = self._heatmap(clf, c, cb, f"F{i + 1} score {p.score:.2f}", f"face_F{i + 1}_heatmap.jpg",
                                        store, p.score)
            if href:
                evid.insert(1, href)
            res.findings.append(Finding(
                module=MODULE_NAME, task=TASK_FACE, level="image", media_id=res.media_id,
                media_type=MediaType.IMAGE, classification=p.classification, score=p.score,
                score_type=ScoreType.CLASSIFIER_PROBABILITY, thresholds=clf.thresholds(),
                predicted_label=p.predicted_label, raw_scores=p.raw, region=_bbox(b.as_tuple()),
                detector_confidence=b.confidence, track_id=f"F{i + 1}", model=clf.model_info,
                explanation=hinfo, evidence=evid, limitations=list(clf.spec.limitations) + extra))
        top = max(preds, key=lambda p: p.score)
        res.task_summaries.append(TaskSummary(
            task=TASK_FACE, classification=top.classification, score=top.score,
            score_type=ScoreType.CLASSIFIER_PROBABILITY,
            basis=f"highest score among {len(preds)} detected face(s)", model=clf.model_info))

    def _image_frequency(self, pil, faces, store, res) -> None:
        targets = []
        if faces and faces[0]:
            for i, (b, c) in enumerate(zip(faces[0], faces[1])):
                targets.append((f"F{i + 1}", c, b))
        else:
            targets.append(("image", pil, None))
        model = ModelInfo(model_id="classical:azimuthal_power_spectrum", task=TASK_FREQ, backend="numpy")
        for label, im, b in targets:
            fp = freq.spectral_fingerprint(im)
            name = f"spectrum_{label}.png"
            evid = []
            if self.cfg.evidence_plots and freq.save_spectrum_plot(fp, store.path_for(name),
                                                                   f"Module 02 spectral fingerprint - {label}"):
                evid.append(store.ref(name, "spectrum_plot", f"{label}: log power spectrum and radial profile"))
            res.findings.append(Finding(
                module=MODULE_NAME, task=TASK_FREQ, level="image", media_id=res.media_id,
                media_type=MediaType.IMAGE, classification="exploratory_visualization",
                score=None, score_type=ScoreType.FORENSIC_INDICATOR,
                region=_bbox(b.as_tuple()) if b is not None else None,
                track_id=label if b is not None else None, model=model,
                explanation=freq.public(fp), evidence=evid, limitations=list(FREQ_LIMITATIONS)))
        basis = (f"spectrum plots for {len(targets)} face crop(s)" if faces and faces[0]
                 else "spectrum plot for the whole image (no face found)")
        res.task_summaries.append(TaskSummary(
            task=TASK_FREQ, classification="exploratory_visualization", score=None,
            score_type=ScoreType.FORENSIC_INDICATOR,
            basis=basis + "; visual aid only (indicator did not separate real/fake faces, ROC-AUC 0.52)",
            model=model))

    def _image_manipulation(self, path, bgr, store, res) -> None:
        r = manip.analyse(bgr, jpeg_quality=self.cfg.ela_quality)
        ela_vis, noise_vis, panel = manip.render(r)
        evid = [store.save_bgr(panel, "integrity_panel.jpg", "integrity_panel",
                               "Input | ELA | local noise-level map, side by side"),
                store.save_bgr(ela_vis, "ela_map.jpg", "ela_map", "Error Level Analysis (JPEG re-compression error)"),
                store.save_bgr(noise_vis, "noise_map.jpg", "noise_map",
                               "Local noise level vs image median (red >= 2x, blue <= 0.5x)")]
        meta = manip.read_metadata(path)
        expl = {**manip.public(r), "metadata": meta}
        model = ModelInfo(model_id="classical:ela+noise+exif", task=TASK_MANIP, backend="opencv")
        res.findings.append(Finding(
            module=MODULE_NAME, task=TASK_MANIP, level="image", media_id=res.media_id,
            media_type=MediaType.IMAGE, classification="exploratory_visualization", score=None,
            score_type=ScoreType.FORENSIC_INDICATOR, model=model, explanation=expl, evidence=evid,
            limitations=list(MANIP_LIMITATIONS)))
        notes = "; ".join(meta.get("notes", [])) or "no metadata notes"
        res.task_summaries.append(TaskSummary(
            task=TASK_MANIP, classification="exploratory_visualization", score=None,
            score_type=ScoreType.FORENSIC_INDICATOR,
            basis=f"ELA + noise-level maps for human review (no automatic verdict); metadata: {notes}",
            model=model))

    # ------------------------------------------------------------------ video
    def _analyze_video(self, path: Path, tasks: List[str], res: ModuleResult) -> None:
        cfg = self.cfg
        meta = probe_video(path, cfg.video_sample_fps)
        period = meta.sample_step / meta.fps
        store = EvidenceStore(cfg.evidence_dir, res.media_id)
        do_face = TASK_FACE in tasks and self.detector.available
        do_synth = TASK_SYNTH in tasks
        for t in tasks:
            if t in IMAGE_ONLY_TASKS:
                res.task_summaries.append(TaskSummary(task=t, classification="not_applicable_video_input",
                                                      basis="image-only analysis; run it on an extracted frame"))
        if TASK_FACE in tasks and not self.detector.available:
            res.warnings.append("No face detector available (YuNet model or Haar cascade not found).")
            res.task_summaries.append(TaskSummary(task=TASK_FACE, classification="not_run_no_face_detector",
                                                  basis="face detector unavailable"))

        per_sample: List[List[FaceObs]] = []
        crops: List[Image.Image] = []
        crop_of: Dict[int, int] = {}          # id(obs) -> crop index
        small_frames = []                      # (frame_index, t, PIL) for synthetic keyframes
        last_t = 0.0
        for si, (fi, t, bgr) in enumerate(sample_video(path, meta, cfg.video_max_frames, cfg.video_max_duration_s)):
            last_t = t
            obs_here: List[FaceObs] = []
            if do_face:
                for b in self.detector.detect(bgr):
                    o = FaceObs(si, fi, t, b.as_tuple(), b.confidence)
                    c = crop_face(bgr, b, cfg.face_margin)
                    c.thumbnail((384, 384))
                    crop_of[id(o)] = len(crops)
                    crops.append(c)
                    obs_here.append(o)
            if do_synth:
                im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                im.thumbnail((640, 640))
                small_frames.append((fi, t, im))
            per_sample.append(obs_here)

        n_sampled = len(per_sample)
        if n_sampled == 0:
            raise MediaError("No frames could be decoded from the video.")
        res.media_info = {**meta.as_dict(), "sampled_frames": n_sampled, "sample_period_s": round(period, 3),
                          "analysed_until_s": last_t, "face_detector": self.detector.backend,
                          "faces_observed": len(crops)}
        if meta.fps_assumed:
            res.warnings.append("Video FPS missing/invalid in container; 25 fps assumed for timestamps.")
        if meta.duration_s and meta.duration_s - last_t > 1.5 * period:
            res.warnings.append(f"Analysis covered 0-{last_t:.1f}s of {meta.duration_s:.1f}s "
                                f"(limits: {cfg.video_max_frames} sampled frames, {cfg.video_max_duration_s:.0f}s).")

        if do_face:
            self._video_faces(path, per_sample, crops, crop_of, period, meta.duration_s or last_t, n_sampled, store, res)
        if do_synth:
            self._video_synthetic(small_frames, period, store, res)

    def _video_faces(self, path, per_sample, crops, crop_of, period, duration, n_sampled, store, res) -> None:
        cfg = self.cfg
        if not crops:
            res.warnings.append(f"No face >= {cfg.face_min_size}px detected in {n_sampled} sampled frames.")
            res.task_summaries.append(TaskSummary(task=TASK_FACE, classification="not_applicable_no_face_detected",
                                                  basis=f"{self.detector.backend} detector, {n_sampled} sampled frames"))
            return
        clf = get_classifier(TASK_FACE, cfg)
        preds = clf.classify(crops)
        all_obs = [o for lst in per_sample for o in lst]
        for o in all_obs:
            p = preds[crop_of[id(o)]]
            o.score, o.raw, o.predicted_label = p.score, p.raw, p.predicted_label

        tracks = build_tracks(per_sample, cfg.track_iou, cfg.track_max_gap, center_dist=cfg.track_center_dist)
        lo, hi = clf.spec.threshold_low, clf.spec.threshold_high

        # choose evidence observations per track
        ev_obs = {}
        for tr in tracks:
            ev_obs[tr.track_id] = sorted(tr.obs, key=lambda o: o.score, reverse=True)[: cfg.evidence_frames_per_track]
        need = {o.frame_index for lst in ev_obs.values() for o in lst}
        frames = read_frames(path, need)

        obs_track = {id(o): tr.track_id for tr in tracks for o in tr.obs}
        frame_ref = {}
        for fi in sorted(need):
            if fi not in frames:
                continue
            here = [o for o in all_obs if o.frame_index == fi]
            t = here[0].timestamp_s
            frame_ref[fi] = store.save_annotated(
                frames[fi],
                [(o.box, f"{obs_track[id(o)].split('-')[-1]} {o.score:.2f}", band_colour(o.score, lo, hi)) for o in here],
                f"frame_{fi:06d}.jpg", "annotated_frame",
                f"Frame {fi} at {t:.2f}s with face-manipulation scores",
                header=f"frame {fi}  t={t:.2f}s")

        timeline_series = []
        track_findings = []
        for tr in tracks:
            pts = [(o.timestamp_s, o.frame_index, o.score) for o in tr.obs]
            ts = summarise(pts, hi, period)
            enough = len(tr.obs) >= cfg.min_track_observations
            verdict = clf.verdict(ts.mean) if enough else clf.spec.verdicts[1]
            peak = max(tr.obs, key=lambda o: o.score)
            lim = list(clf.spec.limitations)
            if not enough:
                lim.insert(0, f"Track has only {len(tr.obs)} observation(s) (< {cfg.min_track_observations}); "
                              f"verdict forced to inconclusive.")
            evid = []
            for o in ev_obs[tr.track_id]:
                c = store.save_crop(crops[crop_of[id(o)]], f"{tr.track_id.split('-')[-1]}_f{o.frame_index:06d}.jpg",
                                    f"{tr.track_id} face crop at {o.timestamp_s:.2f}s (score {o.score:.2f})")
                evid.append(c)
                fr = [frame_ref[o.frame_index]] if o.frame_index in frame_ref else []
                res.findings.append(Finding(
                    module=MODULE_NAME, task=TASK_FACE, level="frame", media_id=res.media_id,
                    media_type=MediaType.VIDEO, classification=clf.verdict(o.score), score=o.score,
                    score_type=ScoreType.CLASSIFIER_PROBABILITY, thresholds=clf.thresholds(),
                    predicted_label=o.predicted_label, raw_scores=o.raw, frame_index=o.frame_index,
                    timestamp_s=o.timestamp_s, track_id=tr.track_id, region=_bbox(o.box),
                    detector_confidence=o.det_conf, model=clf.model_info, evidence=[c] + fr,
                    limitations=["Single-frame score; see the track-level finding for the temporal aggregate."]))
            for o in ev_obs[tr.track_id]:
                if o.frame_index in frame_ref and frame_ref[o.frame_index] not in evid:
                    evid.append(frame_ref[o.frame_index])
            hinfo = None
            if ev_obs[tr.track_id]:
                o = ev_obs[tr.track_id][0]
                fshape = frames[o.frame_index].shape if o.frame_index in frames else None
                cb = crop_box(fshape, FaceBox(*o.box), cfg.face_margin) if fshape is not None else None
                href, hinfo = self._heatmap(clf, crops[crop_of[id(o)]], cb,
                                            f"{tr.track_id} @ {o.timestamp_s:.1f}s score {o.score:.2f}",
                                            f"{tr.track_id.split('-')[-1]}_heatmap.jpg", store, o.score)
                if href:
                    evid.append(href)
                    hinfo["frame_index"], hinfo["timestamp_s"] = o.frame_index, o.timestamp_s
            f = Finding(
                module=MODULE_NAME, task=TASK_FACE, level="track", media_id=res.media_id,
                media_type=MediaType.VIDEO, classification=verdict, score=ts.mean,
                score_type=ScoreType.AGGREGATE_MEAN_CLASSIFIER, thresholds=clf.thresholds(),
                frame_index=peak.frame_index, timestamp_s=peak.timestamp_s,
                time_start_s=tr.obs[0].timestamp_s, time_end_s=tr.obs[-1].timestamp_s,
                track_id=tr.track_id, region=_bbox(peak.box), temporal=ts, model=clf.model_info,
                explanation=hinfo, evidence=evid, limitations=lim)
            track_findings.append(f)
            timeline_series.append((tr.track_id, [o.timestamp_s for o in tr.obs], [o.score for o in tr.obs]))

        tl = store.save_timeline(timeline_series, lo, hi, duration)
        if tl:
            for f in track_findings:
                f.evidence.append(tl)
        res.findings[0:0] = track_findings  # track-level findings first

        eligible = [f for f in track_findings if f.temporal.n_observations >= cfg.min_track_observations]
        if eligible:
            worst = max(eligible, key=lambda f: f.score)
            res.task_summaries.append(TaskSummary(
                task=TASK_FACE, classification=worst.classification, score=worst.score,
                score_type=ScoreType.AGGREGATE_MEAN_CLASSIFIER,
                basis=(f"highest track mean across {len(tracks)} face track(s) "
                       f"({worst.track_id}, {worst.time_start_s:.1f}-{worst.time_end_s:.1f}s) "
                       f"over {n_sampled} sampled frames"), model=clf.model_info))
        else:
            res.task_summaries.append(TaskSummary(
                task=TASK_FACE, classification=clf.spec.verdicts[1], basis=(
                    f"{len(tracks)} face track(s) found but none had >= {cfg.min_track_observations} observations"),
                model=clf.model_info))

    def _video_synthetic(self, small_frames, period, store, res) -> None:
        """Experimental on video: the classifier was trained on still images and reacts to video
        compression, so on video its verdict is capped at 'inconclusive' (scores still reported)."""
        clf = get_classifier(TASK_SYNTH, self.cfg)
        ind, inc, _neg = clf.spec.verdicts
        cap = lambda v: inc if v == ind else v  # noqa: E731
        k = min(self.cfg.synthetic_keyframes, len(small_frames))
        idx = sorted(set(np.linspace(0, len(small_frames) - 1, k).round().astype(int).tolist()))
        keys = [small_frames[i] for i in idx]
        preds = clf.classify([im for _, _, im in keys])
        note = ("EXPERIMENTAL on video: still-image classifier; compression can raise scores. "
                "Verdict capped at 'inconclusive'.")
        for (fi, t, im), p in zip(keys, preds):
            bgr = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
            ref = store.save_annotated(bgr, [], f"synthetic_key_{fi:06d}.jpg", "annotated_frame",
                                       f"Keyframe {fi} at {t:.2f}s assessed for AI generation",
                                       header=f"t={t:.2f}s synthetic_image {p.score:.2f} (experimental)")
            res.findings.append(Finding(
                module=MODULE_NAME, task=TASK_SYNTH, level="frame", media_id=res.media_id,
                media_type=MediaType.VIDEO, classification=cap(p.classification), score=p.score,
                score_type=ScoreType.CLASSIFIER_PROBABILITY, thresholds=clf.thresholds(),
                predicted_label=p.predicted_label, raw_scores=p.raw, frame_index=fi, timestamp_s=t,
                model=clf.model_info, evidence=[ref], limitations=[note]))
        ts = summarise([(t, fi, p.score) for (fi, t, _), p in zip(keys, preds)], clf.spec.threshold_high,
                       period * max(1, len(small_frames) // max(1, k)))
        v = cap(clf.verdict(ts.mean))
        res.findings.append(Finding(
            module=MODULE_NAME, task=TASK_SYNTH, level="video", media_id=res.media_id,
            media_type=MediaType.VIDEO, classification=v, score=ts.mean,
            score_type=ScoreType.AGGREGATE_MEAN_CLASSIFIER, thresholds=clf.thresholds(),
            time_start_s=keys[0][1], time_end_s=keys[-1][1], temporal=ts, model=clf.model_info,
            limitations=[note] + list(clf.spec.limitations)))
        res.task_summaries.append(TaskSummary(
            task=TASK_SYNTH, classification=v, score=ts.mean, score_type=ScoreType.AGGREGATE_MEAN_CLASSIFIER,
            basis=f"mean over {len(keys)} keyframes (experimental on video; verdict capped)", model=clf.model_info))
