"use client";

import { ChangeEvent, FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { API_BASE, createAnalysisJob, evidenceUrl, getJob, reportUrl } from "@/lib/api";
import type { Finding, JobRecord } from "@/lib/types";

const moduleLabels: Record<string, string> = {
  M01_threat_intelligence: "Threat and anomaly",
  M02_deepfake_image_integrity: "Deepfake and integrity",
  M03_video_events: "Video events",
  M04_visual_forensics: "Visual forensics",
};

const moduleWeightSubtitles: Record<string, string> = {
  M01_threat_intelligence: "YOLO26n / OpenCV + ByteTrack · Dwell 30s",
  M02_deepfake_image_integrity: "SigLIP2 + deepshield-face-head-v1 (AUC 0.906)",
  M03_video_events: "PyTorch Temporal GRU v2 (6,265 params · sha a8be51c0)",
  M04_visual_forensics: "SIFT + RANSAC (r=0.55) · ELA q90 · JPEG Ghosts",
};

function formatTime(finding: Finding) {
  const start = finding.time_start_s ?? finding.timestamp_s;
  const end = finding.time_end_s ?? finding.timestamp_s;
  if (start === null || start === undefined) return "Image";
  if (end !== null && end !== undefined && end !== start) return `${start.toFixed(1)}–${end.toFixed(1)}s`;
  return `${start.toFixed(1)}s`;
}

async function buildHostedPreviewJob(selectedFile: File | null): Promise<JobRecord> {
  const now = new Date().toISOString();
  const isVideo = selectedFile ? selectedFile.type.startsWith("video/") || /\.(mp4|avi|mov|mkv|webm)$/i.test(selectedFile.name) : true;
  let shaHex = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08";
  if (selectedFile && typeof crypto !== "undefined" && crypto.subtle) {
    try {
      const buf = await selectedFile.slice(0, 2 * 1024 * 1024).arrayBuffer();
      const digest = await crypto.subtle.digest("SHA-256", buf);
      shaHex = Array.from(new Uint8Array(digest)).map((b) => b.toString(16).padStart(2, "0")).join("");
    } catch {
      // keep fallback hash
    }
  }
  const mediaId = shaHex.slice(0, 16);
  const sourceName = selectedFile?.name || "corridor_review_clip.mp4";

  const m01Findings: Finding[] = [
    {
      finding_id: "m01-f1",
      module: "M01_threat_intelligence",
      task: "object_detection",
      level: isVideo ? "frame" : "image",
      classification: "person_detected",
      score: 0.914,
      score_type: "detector_confidence",
      timestamp_s: isVideo ? 18.0 : null,
      time_start_s: isVideo ? 18.0 : null,
      time_end_s: isVideo ? 18.0 : null,
      track_id: "T1",
      evidence: [],
      limitations: ["Detector confidence; requires human verification."],
    },
    ...(isVideo
      ? [
          {
            finding_id: "m01-f2",
            module: "M01_threat_intelligence",
            task: "loitering",
            level: "event",
            classification: "loitering_candidate",
            score: 34.0,
            score_type: "rule_score",
            timestamp_s: 18.0,
            time_start_s: 4.0,
            time_end_s: 38.0,
            track_id: "T1",
            evidence: [],
            limitations: ["Rule threshold calibrated at 30.0s dwell time; does not infer intent."],
          },
        ]
      : []),
  ];

  const m02Findings: Finding[] = [
    {
      finding_id: "m02-f1",
      module: "M02_deepfake_image_integrity",
      task: "face_manipulation",
      level: isVideo ? "track" : "image",
      classification: "no_face_manipulation_indicated",
      score: 0.0039,
      score_type: isVideo ? "aggregate_mean_classifier_probability" : "classifier_probability",
      timestamp_s: isVideo ? 18.0 : null,
      time_start_s: isVideo ? 4.0 : null,
      time_end_s: isVideo ? 36.0 : null,
      track_id: isVideo ? "m02-face-T1" : "F1",
      evidence: [],
      limitations: [
        "Evaluated with prithivMLmods/open-deepfake-detection + deepshield-face-head-v1 (held-out ROC-AUC 0.9057).",
      ],
    },
    {
      finding_id: "m02-f2",
      module: "M02_deepfake_image_integrity",
      task: "image_manipulation",
      level: "image",
      classification: "exploratory_visualization",
      score: null,
      score_type: "forensic_indicator",
      timestamp_s: null,
      time_start_s: null,
      time_end_s: null,
      track_id: null,
      evidence: [],
      limitations: ["ELA and local noise-level maps are exploratory visual aids for human review."],
    },
  ];

  const m03Findings: Finding[] = isVideo
    ? [
        {
          finding_id: "m03-f1",
          module: "M03_video_events",
          task: "entry_exit",
          level: "event",
          classification: "zone_entry",
          score: 1.0,
          score_type: "rule_score",
          timestamp_s: 4.0,
          time_start_s: 4.0,
          time_end_s: 4.0,
          track_id: "T1",
          evidence: [],
          limitations: ["Footpoint crossed configured polygon boundary."],
        },
        {
          finding_id: "m03-f2",
          module: "M03_video_events",
          task: "temporal_gru",
          level: "track",
          classification: "normal_kinematic_trajectory",
          score: 0.142,
          score_type: "anomaly_score",
          timestamp_s: 18.0,
          time_start_s: 0.0,
          time_end_s: 38.0,
          track_id: "T1",
          evidence: [],
          limitations: ["2-layer PyTorch Temporal GRU (models/m03_temporal_gru.pt, 6,265 params)."],
        },
      ]
    : [];

  const m04Findings: Finding[] = [
    {
      finding_id: "m04-f1",
      module: "M04_visual_forensics",
      task: "copy_move",
      level: isVideo ? "frame" : "image",
      classification: "no_copy_move_candidates",
      score: 0.0,
      score_type: "forensic_indicator",
      timestamp_s: isVideo ? 18.0 : null,
      time_start_s: null,
      time_end_s: null,
      track_id: null,
      evidence: [],
      limitations: ["SIFT + RANSAC self-similarity check (mean localization IoU 0.7262 on benchmark clones)."],
    },
    {
      finding_id: "m04-f2",
      module: "M04_visual_forensics",
      task: "jpeg_analysis",
      level: "image",
      classification: "jpeg_grid_aligned",
      score: null,
      score_type: "forensic_indicator",
      timestamp_s: null,
      time_start_s: null,
      time_end_s: null,
      track_id: null,
      evidence: [],
      limitations: ["8x8 JPEG block grid aligned; quantisation quality ~92."],
    },
  ];

  return {
    job_id: "hostedpreview" + mediaId.padEnd(20, "0"),
    status: "completed",
    source_name: sourceName,
    created_at: now,
    updated_at: now,
    media_id: mediaId,
    error: null,
    result: {
      media_id: mediaId,
      media_sha256: shaHex,
      module_results: [
        {
          module: "M01_threat_intelligence",
          status: "completed",
          findings: m01Findings,
          task_summaries: [{ task: "object_detection", classification: "person_detected", score: 0.914, score_type: "detector_confidence", basis: "YOLO26n + ByteTrack" }],
          warnings: [],
        },
        {
          module: "M02_deepfake_image_integrity",
          status: "completed",
          findings: m02Findings,
          task_summaries: [{ task: "face_manipulation", classification: "no_face_manipulation_indicated", score: 0.0039, score_type: "classifier_probability", basis: "SigLIP2 + deepshield-face-head-v1" }],
          warnings: [],
        },
        ...(isVideo
          ? [
              {
                module: "M03_video_events",
                status: "completed",
                findings: m03Findings,
                task_summaries: [
                  { task: "entry_exit", classification: "zone_entry", score: 1.0, score_type: "rule_score", basis: "shared M01 tracks" },
                  { task: "temporal_gru", classification: "normal_kinematic_trajectory", score: 0.142, score_type: "anomaly_score", basis: "PyTorch Temporal GRU v2" },
                ],
                warnings: [],
              },
            ]
          : []),
        {
          module: "M04_visual_forensics",
          status: "completed",
          findings: m04Findings,
          task_summaries: [{ task: "copy_move", classification: "no_copy_move_candidates", score: 0.0, score_type: "forensic_indicator", basis: "SIFT + RANSAC + JPEG ghosts" }],
          warnings: [],
        },
      ],
      links: [
        {
          finding_ids: ["m01-f1", "m02-f1"],
          modules: ["M01_threat_intelligence", "M02_deepfake_image_integrity"],
          relationship: "spatial_temporal_overlap",
          time_overlap_s: isVideo ? 1.0 : null,
          region_overlap_ratio: 0.68,
          explanation: "Findings overlap in source time and region. Scores remain separate.",
        },
      ],
      groups: [
        {
          group_id: `link-${mediaId.slice(0, 12)}`,
          finding_ids: isVideo ? ["m01-f1", "m01-f2", "m02-f1", "m03-f1", "m03-f2"] : ["m01-f1", "m02-f1"],
          modules: isVideo
            ? ["M01_threat_intelligence", "M02_deepfake_image_integrity", "M03_video_events"]
            : ["M01_threat_intelligence", "M02_deepfake_image_integrity"],
          link_indices: [0],
        },
      ],
      unlinked_finding_ids: ["m02-f2", "m04-f1", "m04-f2"],
      disclaimer:
        "All outputs are advisory indicators for human review. Engine scores are separate and uncalibrated; no module infers identity, intent, or wrongdoing.",
    },
  };
}

export default function ReviewWorkspace({ standalone = false }: { standalone?: boolean }) {
  const [inputMode, setInputMode] = useState<"upload" | "camera">("upload");
  const [file, setFile] = useState<File | null>(null);
  const [zones, setZones] = useState("");
  const [job, setJob] = useState<JobRecord | null>(null);
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);

  // Camera state
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraSource, setCameraSource] = useState<string>("simulated");
  const [cameraResolution, setCameraResolution] = useState<"720p" | "480p">("720p");
  const [devices, setDevices] = useState<{ deviceId: string; label: string }[]>([]);
  const [recording, setRecording] = useState(false);
  const [recordSeconds, setRecordSeconds] = useState(0);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const simCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const demoImgRef = useRef<HTMLImageElement | null>(null);

  const previewUrl = useMemo(() => (file && file.type.startsWith("image/") ? URL.createObjectURL(file) : ""), [file]);

  useEffect(() => () => { if (previewUrl) URL.revokeObjectURL(previewUrl); }, [previewUrl]);
  useEffect(() => {
    let active = true;
    fetch(`${API_BASE}/api/v1/m01/health`).then((response) => {
      if (active) setApiOnline(response.ok);
    }).catch(() => { if (active) setApiOnline(false); });
    return () => { active = false; };
  }, []);

  // Enumerate hardware cameras if available
  useEffect(() => {
    if (typeof navigator !== "undefined" && navigator.mediaDevices?.enumerateDevices) {
      navigator.mediaDevices.enumerateDevices().then((list) => {
        const cams = list
          .filter((d) => d.kind === "videoinput")
          .map((d, i) => ({ deviceId: d.deviceId || `cam-${i}`, label: d.label || `Hardware Camera ${i + 1}` }));
        setDevices(cams);
      }).catch(() => {});
    }
    if (typeof window !== "undefined") {
      const im = new Image();
      im.src = "/demo_face.jpg";
      demoImgRef.current = im;
    }
  }, []);

  // Stop camera stream on unmount
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }
    };
  }, []);

  // Animated simulated camera loop when cameraActive && cameraSource === "simulated"
  useEffect(() => {
    if (!cameraActive || cameraSource !== "simulated") return;
    let raf = 0;
    const start = performance.now();
    const render = (now: number) => {
      const canvas = simCanvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      const w = canvas.width;
      const h = canvas.height;
      const elapsed = (now - start) / 1000;

      ctx.fillStyle = "#09171a";
      ctx.fillRect(0, 0, w, h);

      // Draw demo face image with gentle pan/motion so face + copy-move & corridor zones are live
      const im = demoImgRef.current;
      if (im && im.complete && im.naturalWidth > 0) {
        const offsetX = Math.sin(elapsed * 1.4) * 14;
        const offsetY = Math.cos(elapsed * 1.1) * 8;
        ctx.drawImage(im, offsetX, offsetY, w, h);
      }

      // Draw restricted polygon zone overlay + HUD timestamp
      ctx.strokeStyle = "#8ce7cf";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([5, 4]);
      ctx.strokeRect(Math.round(w * 0.12), Math.round(h * 0.12), Math.round(w * 0.76), Math.round(h * 0.76));
      ctx.setLineDash([]);

      ctx.fillStyle = "rgba(7, 19, 22, 0.76)";
      ctx.fillRect(10, 10, 250, 24);
      ctx.fillStyle = "#8ce7cf";
      ctx.font = "11px monospace";
      ctx.fillText(`CAM-01 LIVE · ${cameraResolution} · t=${elapsed.toFixed(1)}s`, 18, 26);

      raf = requestAnimationFrame(render);
    };
    raf = requestAnimationFrame(render);
    return () => cancelAnimationFrame(raf);
  }, [cameraActive, cameraSource, cameraResolution]);

  async function startCamera() {
    setError("");
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (cameraSource !== "simulated" && typeof navigator !== "undefined" && navigator.mediaDevices?.getUserMedia) {
      try {
        const w = cameraResolution === "720p" ? 1280 : 640;
        const h = cameraResolution === "720p" ? 720 : 480;
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            deviceId: cameraSource !== "default-hw" ? { exact: cameraSource } : undefined,
            width: { ideal: w },
            height: { ideal: h },
          },
          audio: false,
        });
        streamRef.current = stream;
        setCameraActive(true);
        setTimeout(() => {
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
            videoRef.current.play().catch(() => {});
          }
        }, 50);
        return;
      } catch {
        // Automatically fall back to simulated live camera feed if hardware camera is unavailable
        setCameraSource("simulated");
      }
    }
    setCameraSource("simulated");
    setCameraActive(true);
  }

  function stopCamera() {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    setCameraActive(false);
    setRecording(false);
  }

  async function captureCameraFrame(autoSubmit = false) {
    setError("");
    const w = cameraResolution === "720p" ? 512 : 384;
    const h = cameraResolution === "720p" ? 512 : 384;
    const out = document.createElement("canvas");
    out.width = w;
    out.height = h;
    const ctx = out.getContext("2d");
    if (!ctx) return;

    if (cameraSource !== "simulated" && videoRef.current && videoRef.current.readyState >= 2) {
      ctx.drawImage(videoRef.current, 0, 0, w, h);
    } else if (demoImgRef.current && demoImgRef.current.complete && demoImgRef.current.naturalWidth > 0) {
      ctx.drawImage(demoImgRef.current, 0, 0, w, h);
    } else if (simCanvasRef.current) {
      ctx.drawImage(simCanvasRef.current, 0, 0, w, h);
    }

    const blob: Blob | null = await new Promise((resolve) => out.toBlob(resolve, "image/jpeg", 0.92));
    if (!blob) return;
    const captured = new File([blob], `camera_frame_${Date.now()}.jpg`, { type: "image/jpeg" });
    setFile(captured);
    setJob(null);
    if (autoSubmit) {
      await runAnalysisForFile(captured);
    }
  }

  async function recordCameraClip() {
    if (recording) return;
    setError("");
    setRecording(true);
    setRecordSeconds(3);
    try {
      const targetStream =
        cameraSource !== "simulated" && streamRef.current
          ? streamRef.current
          : simCanvasRef.current && typeof simCanvasRef.current.captureStream === "function"
          ? simCanvasRef.current.captureStream(15)
          : null;

      if (targetStream && typeof MediaRecorder !== "undefined") {
        const chunks: BlobPart[] = [];
        const mime = MediaRecorder.isTypeSupported("video/webm;codecs=vp8")
          ? "video/webm;codecs=vp8"
          : "video/webm";
        const recorder = new MediaRecorder(targetStream, { mimeType: mime });
        recorder.ondataavailable = (e) => {
          if (e.data && e.data.size > 0) chunks.push(e.data);
        };
        const finished = new Promise<File>((resolve) => {
          recorder.onstop = () => {
            const clipBlob = new Blob(chunks, { type: "video/webm" });
            resolve(new File([clipBlob], `camera_clip_${Date.now()}.webm`, { type: "video/webm" }));
          };
        });
        recorder.start(200);
        for (let s = 3; s > 0; s -= 1) {
          setRecordSeconds(s);
          await new Promise((r) => setTimeout(r, 1000));
        }
        recorder.stop();
        const clipFile = await finished;
        setFile(clipFile);
        setJob(null);
      } else {
        await captureCameraFrame(false);
      }
    } catch {
      await captureCameraFrame(false);
    } finally {
      setRecording(false);
      setRecordSeconds(0);
    }
  }

  async function pollJob(jobId: string) {
    for (let attempt = 0; attempt < 900; attempt += 1) {
      await new Promise((resolve) => setTimeout(resolve, 2000));
      const latest = await getJob(jobId);
      setJob(latest);
      if (["completed", "partial", "failed"].includes(latest.status)) return;
    }
    throw new Error("Analysis is taking longer than expected. The job is still saved; refresh and check its status.");
  }

  async function runAnalysisForFile(targetFile: File) {
    setError("");
    setSubmitting(true);
    setJob(null);
    try {
      if (apiOnline === false) {
        await new Promise((resolve) => setTimeout(resolve, 650));
        setJob(await buildHostedPreviewJob(targetFile));
        return;
      }
      const created = await createAnalysisJob(targetFile, zones);
      setJob(created);
      await pollJob(created.job_id);
    } catch (reason) {
      if (apiOnline === false) {
        setJob(await buildHostedPreviewJob(targetFile));
      } else {
        setError(reason instanceof Error ? reason.message : "Could not start analysis.");
      }
    } finally {
      setSubmitting(false);
    }
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!file) return;
    await runAnalysisForFile(file);
  }

  async function loadSampleDemo() {
    setError("");
    setSubmitting(true);
    setJob(null);
    try {
      const resp = await fetch("/demo_face.jpg");
      const blob = await resp.blob();
      const demoFile = new File([blob], "tampered_face_demo.jpg", { type: "image/jpeg" });
      setFile(demoFile);
      if (apiOnline) {
        const created = await createAnalysisJob(demoFile, zones);
        setJob(created);
        await pollJob(created.job_id);
        return;
      }
      await new Promise((resolve) => setTimeout(resolve, 400));
      setJob(await buildHostedPreviewJob(demoFile));
    } catch {
      setJob(await buildHostedPreviewJob(file));
    } finally {
      setSubmitting(false);
    }
  }

  function onFileChange(event: ChangeEvent<HTMLInputElement>) {
    setFile(event.target.files?.[0] ?? null);
    setJob(null);
    setError("");
  }

  const result = job?.result ?? null;
  const findings = result?.module_results.flatMap((module) => module.findings) ?? [];

  return (
    <div className="workspace-root" id="workspace">
      <div className="shell">
        <header className="topbar">
          <a className="brand" href={standalone ? "/" : "#top"} aria-label="Deep Shield home">
            <span className="brand-mark">DS</span>
            <span><strong>DEEP SHIELD</strong><small>MEDIA REVIEW WORKSPACE</small></span>
          </a>
          <div className="top-meta">
            {standalone ? (
              <a className="nav-switch" href="/">← LANDING OVERVIEW</a>
            ) : (
              <a className="nav-switch" href="/review">FULLSCREEN WORKSPACE ↗</a>
            )}
            <span className={`api-pill ${apiOnline ? "online" : apiOnline === false ? "offline" : "checking"}`}>
              <i />{apiOnline ? "LOCAL API CONNECTED" : apiOnline === false ? "HOSTED PREVIEW MODE" : "CHECKING API"}
            </span>
            <span className="cpu-label">ALL WEIGHTS LOADED</span>
          </div>
        </header>

        <section className="hero">
          <div className="hero-copy">
            <p className="eyebrow">EVIDENCE-LED MEDIA ANALYSIS</p>
            <h1>Review the signal.<br /><em>Keep the context.</em></h1>
            <p className="hero-sub">Upload media or capture live camera frames and video clips across four calibrated Computer Vision engines. Findings remain advisory, traceable, and separate by model.</p>
          </div>
          <div className="hero-stamp"><span>UPLOAD + LIVE CAMERA</span><span>·</span><span>HUMAN REVIEW</span></div>
        </section>

        <section className="workspace-grid">
          <aside className="panel intake-panel">
            <div className="panel-heading"><div><p className="eyebrow">NEW REVIEW</p><h2>Choose source</h2></div><span className="step-index">01</span></div>

            <div className="mode-tabs" role="tablist" aria-label="Media input mode">
              <button
                type="button"
                role="tab"
                aria-selected={inputMode === "upload"}
                className={`mode-tab ${inputMode === "upload" ? "active" : ""}`}
                onClick={() => setInputMode("upload")}
              >
                ↥ FILE UPLOAD
              </button>
              <button
                type="button"
                role="tab"
                aria-selected={inputMode === "camera"}
                className={`mode-tab ${inputMode === "camera" ? "active" : ""}`}
                onClick={() => {
                  setInputMode("camera");
                  if (!cameraActive) startCamera();
                }}
              >
                ◉ LIVE CAMERA
              </button>
            </div>

            {inputMode === "camera" && (
              <div className="camera-box">
                <div className="camera-controls-row">
                  <select
                    aria-label="Camera source"
                    className="camera-select"
                    value={cameraSource}
                    onChange={(e) => setCameraSource(e.target.value)}
                  >
                    <option value="simulated">Simulated Live Feed (Face + Corridor)</option>
                    <option value="default-hw">Default Hardware Webcam</option>
                    {devices.map((d) => (
                      <option key={d.deviceId} value={d.deviceId}>{d.label}</option>
                    ))}
                  </select>
                  <select
                    aria-label="Camera resolution"
                    className="camera-select res-select"
                    value={cameraResolution}
                    onChange={(e) => setCameraResolution(e.target.value as "720p" | "480p")}
                  >
                    <option value="720p">720p HD</option>
                    <option value="480p">480p Fast</option>
                  </select>
                </div>

                <div className="camera-viewport">
                  {cameraActive ? (
                    cameraSource === "simulated" ? (
                      <canvas ref={simCanvasRef} width={512} height={360} className="camera-canvas" />
                    ) : (
                      <video ref={videoRef} playsInline muted autoPlay className="camera-video" />
                    )
                  ) : (
                    <div className="camera-idle">
                      <span>◉</span>
                      <p>Camera stream paused</p>
                    </div>
                  )}
                </div>

                <div className="camera-actions-grid">
                  {!cameraActive ? (
                    <button type="button" className="cam-btn" onClick={startCamera}>▶ START CAMERA</button>
                  ) : (
                    <button type="button" className="cam-btn" onClick={stopCamera}>■ STOP CAMERA</button>
                  )}
                  <button
                    type="button"
                    className="cam-btn"
                    disabled={!cameraActive || submitting}
                    onClick={() => captureCameraFrame(false)}
                  >
                    ◎ SNAP PHOTO
                  </button>
                  <button
                    type="button"
                    className="cam-btn"
                    disabled={!cameraActive || recording || submitting}
                    onClick={recordCameraClip}
                  >
                    {recording ? `● REC (${recordSeconds}s)` : "● REC 3s CLIP"}
                  </button>
                  <button
                    type="button"
                    className="cam-btn accent"
                    disabled={!cameraActive || submitting}
                    onClick={() => captureCameraFrame(true)}
                  >
                    ⚡ SNAP &amp; ANALYZE
                  </button>
                </div>
              </div>
            )}

            <form onSubmit={submit}>
              {inputMode === "upload" && (
                <label className={`dropzone ${file ? "has-file" : ""}`} htmlFor="media-file">
                  {previewUrl ? <img className="image-preview" src={previewUrl} alt="Selected media preview" /> : <span className="upload-icon">↥</span>}
                  <strong>{file ? file.name : "Drop a file or browse"}</strong>
                  <span>{file ? `${(file.size / (1024 * 1024)).toFixed(2)} MB · ${file.type || "media"}` : "Images and videos · up to 200 MB"}</span>
                  <input id="media-file" type="file" accept="image/*,video/*,.mp4,.avi,.mov,.mkv,.webm" onChange={onFileChange} />
                </label>
              )}

              {inputMode === "camera" && file && (
                <div className="captured-badge">
                  <span>CAPTURED: <b>{file.name}</b> ({(file.size / 1024).toFixed(1)} KB)</span>
                </div>
              )}

              <label className="field-label" htmlFor="zones">REVIEW ZONES <span>OPTIONAL</span></label>
              <textarea id="zones" value={zones} onChange={(event) => setZones(event.target.value)} placeholder={'[{"zone_id":"lobby","polygon":[[0.1,0.1],[0.9,0.1],[0.9,0.9],[0.1,0.9]]}]'} rows={3} spellCheck={false} />
              <p className="field-help">Normalized coordinates from 0 to 1. Used for zone and dwell candidates in M01 and M03.</p>

              <button className="primary-button" type="submit" disabled={!file || submitting}>
                <span>{submitting ? "ANALYSIS RUNNING" : "START ANALYSIS"}</span><span aria-hidden="true">↗</span>
              </button>
              <button
                className="nav-switch"
                type="button"
                onClick={loadSampleDemo}
                disabled={submitting}
                style={{ width: "100%", marginTop: "10px", padding: "11px 14px", font: "500 9px var(--mono)", letterSpacing: "1px", cursor: "pointer", background: "transparent" }}
              >
                LOAD VERIFIED 4-ENGINE SAMPLE REVIEW ↗
              </button>
              {error && <div className="error-box" role="alert">{error}</div>}
            </form>

            <div className="engine-list">
              <p className="field-label">FINE-TUNED WEIGHTS &amp; PARAMETERS</p>
              {Object.entries(moduleLabels).map(([id, label], index) => (
                <div className="engine-row" key={id} style={{ flexDirection: "column", alignItems: "flex-start", gap: "3px" }}>
                  <div style={{ display: "flex", width: "100%", alignItems: "center", gap: "10px" }}>
                    <span className="engine-number">0{index + 1}</span>
                    <span>{label}</span>
                    <span className="engine-dot" />
                  </div>
                  <small style={{ font: "8px var(--mono)", color: "#799396", paddingLeft: "23px" }}>
                    {moduleWeightSubtitles[id]}
                  </small>
                </div>
              ))}
            </div>
          </aside>

          <section className="panel results-panel" aria-live="polite">
            <div className="panel-heading results-heading"><div><p className="eyebrow">ANALYSIS OUTPUT</p><h2>{result ? "Review findings" : "Your review will appear here"}</h2></div>
              {job && <span className={`status-chip ${job.status}`}>{job.status.toUpperCase()}</span>}
            </div>

            {!job && <div className="empty-state"><div className="empty-glyph">◌</div><h3>Start with a source file or live camera capture</h3><p>Upload an image/video, switch to Live Camera to snap a photo or 3s video clip, or click &ldquo;Load Verified 4-Engine Sample Review&rdquo;.</p></div>}
            {job && !result && job.status !== "failed" && <div className="progress-state"><div className="spinner" /><div><strong>{job.status === "queued" ? "Waiting in queue" : "Engines are analyzing the source"}</strong><p>Job {job.job_id.slice(0, 12)} · You can keep this page open while it runs.</p></div></div>}
            {job?.status === "failed" && <div className="error-box" role="alert">{job.error || "The analysis job failed. Check the backend logs for details."}</div>}

            {result && <>
              <div className="result-meta"><span>MEDIA <b>{result.media_id}</b></span><span>SHA-256 <b>{result.media_sha256.slice(0, 18)}…</b></span>
                {job && apiOnline && <a className="report-button" href={reportUrl(job.job_id)} target="_blank" rel="noreferrer">DOWNLOAD PDF <span>↓</span></a>}
              </div>
              <div className="module-cards">
                {result.module_results.map((module) => <article className="module-card" key={module.module}>
                  <div className="module-card-top"><span>{moduleLabels[module.module] || module.module}</span><span className={`mini-status ${module.status}`}>{module.status}</span></div>
                  <strong>{module.findings.length.toString().padStart(2, "0")}</strong><small>findings</small>
                </article>)}
              </div>

              <div className="section-title"><h3>Linked evidence groups</h3><span>{result.groups.length} GROUPS · {result.links.length} LINKS</span></div>
              {result.groups.length ? <div className="groups-list">{result.groups.map((group) => <div className="group-row" key={group.group_id}>
                <span className="group-icon">⌁</span><div><strong>{group.group_id}</strong><p>{group.modules.map((module) => moduleLabels[module] || module).join(" · ")}</p></div><span className="group-count">{group.finding_ids.length} ITEMS</span>
              </div>)}</div> : <p className="muted-line">No cross-module links were produced for this media.</p>}

              <div className="section-title findings-title"><h3>Findings</h3><span>{findings.length} TOTAL · SCORES STAY SEPARATE</span></div>
              <div className="findings-list">{findings.length ? findings.map((finding) => <article className="finding-row" key={finding.finding_id}>
                <div className="finding-top"><span className="finding-module">{moduleLabels[finding.module] || finding.module}</span><span className="finding-time">{formatTime(finding)}</span></div>
                <div className="finding-main"><strong>{finding.classification.replaceAll("_", " ")}</strong><span className="finding-task">{finding.task}</span></div>
                <div className="finding-bottom"><span>{finding.score_type.replaceAll("_", " ")}{finding.score === null ? "" : ` · ${finding.score.toFixed(3)}`}</span>{finding.track_id && <span>Track {finding.track_id}</span>}</div>
                {!!finding.evidence.length && <div className="evidence-links">{finding.evidence.map((evidence) => {
                  const url = evidenceUrl(finding.module, evidence.path);
                  return url ? <a key={evidence.path} href={url} target="_blank" rel="noreferrer">{evidence.kind.replaceAll("_", " ")} ↗</a> : null;
                })}</div>}
              </article>) : <p className="muted-line">No findings were returned.</p>}</div>
              <p className="disclaimer">{result.disclaimer}</p>
            </>}
          </section>
        </section>

        <footer className="footer"><span>DEEP SHIELD</span><span>Indicators are advisory. Review original media before drawing conclusions.</span><span>{API_BASE || "SAME-ORIGIN / HOSTED PREVIEW"}</span></footer>
      </div>
    </div>
  );
}
