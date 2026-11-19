"use client";

import { ChangeEvent, FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  API_BASE,
  createAnalysisJob,
  evidenceUrl,
  fetchLoadedModels,
  getActiveApiBase,
  getJob,
  reportUrl,
  resolveApiBase,
  setCustomApiBase,
} from "@/lib/api";
import type { Finding, JobRecord } from "@/lib/types";

const moduleLabels: Record<string, string> = {
  M01_threat_intelligence: "Threat and anomaly",
  M02_deepfake_image_integrity: "Deepfake and integrity",
  M03_video_events: "Video events",
  M04_visual_forensics: "Visual forensics",
};

interface M02TaskHealth {
  ready?: boolean;
  default?: boolean;
  model?: { model_id?: string; backend?: string; revision?: string };
  thresholds?: { low?: number; high?: number };
  trained_head?: { name?: string; sha?: string; test_auc?: number; test_ece?: number } | null;
}

interface LoadedModelsHealth {
  m01_threat_intelligence?: {
    detector?: { ready?: boolean; model_id?: string; backend?: string; tracker?: string };
    thresholds?: { loitering_seconds?: number };
  };
  m02_deepfake_image_integrity?: {
    face_detector?: string;
    tasks?: {
      face_manipulation?: M02TaskHealth;
      synthetic_image?: M02TaskHealth;
    };
  };
  m03_video_events?: {
    gru_enabled?: boolean;
    gru_model?: { model_id?: string; revision?: string; n_parameters?: number; threshold?: number } | null;
  };
  m04_visual_forensics?: {
    copy_move_detector?: string;
    calibrated_parameters?: { sift_ratio_threshold?: number; ela_jpeg_quality?: number; ransac_min_inliers?: number };
  };
}

function formatTime(finding: Finding) {
  const start = finding.time_start_s ?? finding.timestamp_s;
  const end = finding.time_end_s ?? finding.timestamp_s;
  if (start === null || start === undefined) return "Image";
  if (end !== null && end !== undefined && end !== start) return `${start.toFixed(1)}–${end.toFixed(1)}s`;
  return `${start.toFixed(1)}s`;
}

function formatM02TaskLine(label: string, task?: M02TaskHealth): string {
  if (!task) return `${label}: unavailable`;
  const head = task.trained_head;
  const thr = task.thresholds;
  const baseText = head
    ? `${head.name || "trained-head"} (sha ${head.sha || "?"}, AUC ${head.test_auc ?? "?"})`
    : task.model?.model_id || "Unavailable";
  const thrText = thr?.low !== undefined && thr?.high !== undefined ? ` · thr ${thr.low}/${thr.high}` : "";
  return `${label}: ${baseText}${thrText}`;
}

function describeEngineLive(id: string, health: LoadedModelsHealth | null, online: boolean | null): { lines: string[]; ready: boolean } {
  if (!online || !health) {
    return {
      lines: [online === false ? "Backend offline — start FastAPI to load model weights" : "Querying backend /api/v1/intelligence/models…"],
      ready: false,
    };
  }
  if (id === "M01_threat_intelligence") {
    const det = health.m01_threat_intelligence?.detector;
    const dwell = health.m01_threat_intelligence?.thresholds?.loitering_seconds;
    return {
      lines: [
        det?.ready
          ? `${det.model_id || det.backend} + ${det.tracker || "tracker"}${dwell ? ` · Dwell ${dwell}s` : ""}`
          : "Detector checkpoint not loaded",
      ],
      ready: Boolean(det?.ready),
    };
  }
  if (id === "M02_deepfake_image_integrity") {
    const face = health.m02_deepfake_image_integrity?.tasks?.face_manipulation;
    const synth = health.m02_deepfake_image_integrity?.tasks?.synthetic_image;
    const lines = [formatM02TaskLine("Face", face)];
    if (synth) {
      lines.push(formatM02TaskLine("Synth", synth));
    }
    return { lines, ready: Boolean(face?.ready && (synth ? synth.ready : true)) };
  }
  if (id === "M03_video_events") {
    const gru = health.m03_video_events?.gru_model;
    return {
      lines: [
        gru
          ? `${gru.model_id} (${gru.n_parameters ?? "?"} params · ${gru.revision} · thr ${gru.threshold ?? "?"})`
          : "Rule engine active (Temporal GRU disabled)",
      ],
      ready: Boolean(health.m03_video_events),
    };
  }
  if (id === "M04_visual_forensics") {
    const m4 = health.m04_visual_forensics;
    const p = m4?.calibrated_parameters;
    return {
      lines: [
        m4
          ? `${(m4.copy_move_detector || "sift").toUpperCase()} + RANSAC (r=${p?.sift_ratio_threshold ?? "?"}, min=${p?.ransac_min_inliers ?? "?"}) · ELA q${p?.ela_jpeg_quality ?? 90}`
          : "Unavailable",
      ],
      ready: Boolean(m4),
    };
  }
  return { lines: ["Unknown module"], ready: false };
}

export default function ReviewWorkspace({ standalone = false }: { standalone?: boolean }) {
  const [inputMode, setInputMode] = useState<"upload" | "camera">("upload");
  const [file, setFile] = useState<File | null>(null);
  const [zones, setZones] = useState("");
  const [job, setJob] = useState<JobRecord | null>(null);
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);
  const [resolvedBase, setResolvedBase] = useState<string>(API_BASE);
  const [customUrl, setCustomUrl] = useState<string>("http://127.0.0.1:8002");
  const [modelsHealth, setModelsHealth] = useState<LoadedModelsHealth | null>(null);

  // Real hardware webcam state only (no simulated fake feeds)
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState("");
  const [cameraSource, setCameraSource] = useState<string>("default-hw");
  const [cameraResolution, setCameraResolution] = useState<"720p" | "480p">("720p");
  const [devices, setDevices] = useState<{ deviceId: string; label: string }[]>([]);
  const [recording, setRecording] = useState(false);
  const [recordSeconds, setRecordSeconds] = useState(0);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const previewUrl = useMemo(() => (file && file.type.startsWith("image/") ? URL.createObjectURL(file) : ""), [file]);

  useEffect(() => () => { if (previewUrl) URL.revokeObjectURL(previewUrl); }, [previewUrl]);

  const checkBackend = useCallback(async (preferred?: string) => {
    setApiOnline(null);
    const { online, base } = await resolveApiBase(preferred);
    setApiOnline(online);
    setResolvedBase(base);
    if (online) {
      const m = await fetchLoadedModels();
      setModelsHealth((m as LoadedModelsHealth | null) ?? null);
    } else {
      setModelsHealth(null);
    }
    return online;
  }, []);

  useEffect(() => {
    void checkBackend();
  }, [checkBackend]);

  // Enumerate physical hardware cameras if available
  const refreshCameraDevices = useCallback(async () => {
    if (typeof navigator === "undefined" || !navigator.mediaDevices?.enumerateDevices) return;
    try {
      const list = await navigator.mediaDevices.enumerateDevices();
      const cams = list
        .filter((d) => d.kind === "videoinput")
        .map((d, i) => ({ deviceId: d.deviceId || `cam-${i}`, label: d.label || `Hardware Camera ${i + 1}` }));
      setDevices(cams);
    } catch {
      setDevices([]);
    }
  }, []);

  useEffect(() => {
    void refreshCameraDevices();
  }, [refreshCameraDevices]);

  // Stop camera stream on unmount
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }
    };
  }, []);

  async function startCamera() {
    setCameraError("");
    setError("");
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (typeof navigator === "undefined" || !navigator.mediaDevices?.getUserMedia) {
      setCameraError("Browser mediaDevices.getUserMedia is unavailable in this environment.");
      setCameraActive(false);
      return;
    }
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
      await refreshCameraDevices();
      setTimeout(() => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play().catch(() => {});
        }
      }, 50);
    } catch (err) {
      setCameraActive(false);
      setCameraError(
        err instanceof Error
          ? `Hardware webcam unavailable: ${err.message}`
          : "Could not access hardware webcam. Check camera permissions or switch to File Upload."
      );
    }
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
    setCameraError("");
    if (!videoRef.current || videoRef.current.readyState < 2) {
      setCameraError("Camera video stream is not ready yet.");
      return;
    }
    const w = videoRef.current.videoWidth || (cameraResolution === "720p" ? 1280 : 640);
    const h = videoRef.current.videoHeight || (cameraResolution === "720p" ? 720 : 480);
    const out = document.createElement("canvas");
    out.width = w;
    out.height = h;
    const ctx = out.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(videoRef.current, 0, 0, w, h);

    const blob: Blob | null = await new Promise((resolve) => out.toBlob(resolve, "image/jpeg", 0.92));
    if (!blob) return;
    const captured = new File([blob], `webcam_frame_${Date.now()}.jpg`, { type: "image/jpeg" });
    setFile(captured);
    setJob(null);
    if (autoSubmit) {
      await runAnalysisForFile(captured);
    }
  }

  async function recordCameraClip() {
    if (recording || !streamRef.current) return;
    setError("");
    setCameraError("");
    if (typeof MediaRecorder === "undefined") {
      setCameraError("MediaRecorder is not supported in this browser; use Snap Photo instead.");
      return;
    }
    setRecording(true);
    setRecordSeconds(3);
    try {
      const chunks: BlobPart[] = [];
      const mime = MediaRecorder.isTypeSupported("video/webm;codecs=vp8") ? "video/webm;codecs=vp8" : "video/webm";
      const recorder = new MediaRecorder(streamRef.current, { mimeType: mime });
      recorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) chunks.push(e.data);
      };
      const finished = new Promise<File>((resolve) => {
        recorder.onstop = () => {
          const clipBlob = new Blob(chunks, { type: "video/webm" });
          resolve(new File([clipBlob], `webcam_clip_${Date.now()}.webm`, { type: "video/webm" }));
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
    } catch (err) {
      setCameraError(err instanceof Error ? `Recording failed: ${err.message}` : "Video recording failed.");
    } finally {
      setRecording(false);
      setRecordSeconds(0);
    }
  }

  async function pollJob(jobId: string) {
    for (let attempt = 0; attempt < 900; attempt += 1) {
      await new Promise((resolve) => setTimeout(resolve, 1500));
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
      let isOnline = apiOnline;
      if (!isOnline) {
        isOnline = await checkBackend();
      }
      if (!isOnline) {
        setError(
          "Backend API is offline. Start FastAPI (uvicorn deep_shield.api.main:app --port 8002) and the worker (python -m deep_shield.jobs.worker), or configure a live backend URL above. Deep Shield does not generate synthetic or mock analysis results."
        );
        return;
      }
      const created = await createAnalysisJob(targetFile, zones);
      setJob(created);
      await pollJob(created.job_id);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Analysis request failed.");
    } finally {
      setSubmitting(false);
    }
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!file) return;
    await runAnalysisForFile(file);
  }

  async function handleConnectCustomUrl(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setCustomApiBase(customUrl);
    const ok = await checkBackend(customUrl);
    if (!ok) {
      setError(`Could not reach Deep Shield FastAPI health endpoint at ${customUrl}/api/v1/m01/health`);
    }
  }

  function onFileChange(event: ChangeEvent<HTMLInputElement>) {
    const selected = event.target.files?.[0] ?? null;
    setFile(selected);
    setJob(null);
    setError("");
    if (selected && apiOnline) {
      void runAnalysisForFile(selected);
    }
  }

  const result = job?.result ?? null;
  const rawFindings = result?.module_results.flatMap((module) => module.findings) ?? [];

  const isManipFlagged = (f: Finding) =>
    f.classification === "face_manipulation_indicated" ||
    f.classification === "ai_generation_indicated" ||
    f.classification === "copy_move_candidate";

  const isInconclusive = (f: Finding) => f.classification === "inconclusive";

  const isNoManipIndicated = (f: Finding) =>
    f.classification === "no_face_manipulation_indicated" ||
    f.classification === "no_ai_generation_indicated";

  const findings = [...rawFindings].sort((a, b) => {
    const rank = (f: Finding) =>
      isManipFlagged(f)
        ? 5
        : isInconclusive(f)
          ? 4
          : f.task === "face_manipulation"
            ? 3
            : f.task === "synthetic_image"
              ? 2
              : f.task === "copy_move"
                ? 1
                : 0;
    return rank(b) - rank(a);
  });

  const faceFindings = findings.filter((f) => f.task === "face_manipulation");
  const synthFindings = findings.filter((f) => f.task === "synthetic_image");
  const scoredIntegrityFindings = [...faceFindings, ...synthFindings];
  const copyMoveFlagged = findings.filter((f) => f.classification === "copy_move_candidate");
  const flaggedManipFindings = findings.filter(isManipFlagged);
  const inconclusiveFindings = findings.filter(isInconclusive);
  const anyModuleFailed = (result?.module_results ?? []).some((m) => m.status === "failed");

  const bannerState = useMemo(() => {
    if (!result) return null;
    if (flaggedManipFindings.length > 0) {
      const parts = flaggedManipFindings.map((f) => {
        const sc = f.score !== null ? ` (score ${f.score.toFixed(3)})` : "";
        return `${f.task}: ${f.classification.replaceAll("_", " ")}${sc}`;
      });
      const topScore = flaggedManipFindings.find((f) => f.score !== null)?.score;
      return {
        tone: "fake" as const,
        title: "⚠ MANIPULATION / AI-GENERATION INDICATOR FLAGGED — HUMAN REVIEW REQUIRED",
        sub: `Advisory engine findings (scores kept separate): ${parts.join(" · ")}.`,
        pill: topScore !== undefined && topScore !== null ? `FLAGGED · ${(topScore * 100).toFixed(1)}%` : "INDICATOR FLAGGED",
      };
    }
    if (inconclusiveFindings.length > 0) {
      const parts = inconclusiveFindings.map(
        (f) => `${f.task}${f.score !== null ? ` (${f.score.toFixed(3)})` : ""}`
      );
      const first = inconclusiveFindings[0];
      return {
        tone: "inconclusive" as const,
        title: "◐ INCONCLUSIVE RESULT — MANUAL FORENSIC REVIEW REQUIRED",
        sub: `Engine task(s) inside the calibrated inconclusive band: ${parts.join(" · ")}. Inspect evidence maps before deciding.`,
        pill: first.score !== null ? `INCONCLUSIVE · ${first.score.toFixed(3)}` : "INCONCLUSIVE",
      };
    }
    if (anyModuleFailed) {
      return {
        tone: "inconclusive" as const,
        title: "⚠ PARTIAL ANALYSIS — ONE OR MORE ENGINES FAILED",
        sub: "Check module statuses and warnings below. Missing engines were not substituted.",
        pill: "PARTIAL",
      };
    }
    if (
      scoredIntegrityFindings.length > 0 &&
      scoredIntegrityFindings.every(isNoManipIndicated) &&
      copyMoveFlagged.length === 0
    ) {
      const scoreParts = scoredIntegrityFindings.map(
        (f) => `${f.task}: ${f.score !== null ? f.score.toFixed(3) : "n/a"}`
      );
      const topScore = Math.max(...scoredIntegrityFindings.map((f) => f.score ?? 0));
      return {
        tone: "real" as const,
        title: "✓ NO MANIPULATION OR AI GENERATION INDICATED (ADVISORY)",
        sub: `Separate M02 scores below validated low thresholds (${scoreParts.join(" · ")}) and M04 found no copy-move clone candidates. Advisory only — not proof of authenticity.`,
        pill: `NO INDICATOR · ${topScore.toFixed(3)}`,
      };
    }
    const m02Summary = result.module_results
      .find((m) => m.module === "M02_deepfake_image_integrity")
      ?.task_summaries.find((s) => s.task === "face_manipulation");
    return {
      tone: "inconclusive" as const,
      title: "ANALYSIS COMPLETED — REVIEW PER-ENGINE INDICATORS",
      sub: m02Summary
        ? `M02 face_manipulation status: ${m02Summary.classification.replaceAll("_", " ")} (${m02Summary.basis}). Inspect all separate engine outputs below.`
        : "Inspect each engine's independent observations and evidence artifacts below.",
      pill: "REVIEW FINDINGS",
    };
  }, [result, flaggedManipFindings, inconclusiveFindings, anyModuleFailed, scoredIntegrityFindings, copyMoveFlagged]);

  function formatFindingTitle(finding: Finding): string {
    if (finding.classification === "face_manipulation_indicated") {
      return "⚠ Face Manipulation Indicated (Deepfake / Face-Swap Signal)";
    }
    if (finding.classification === "no_face_manipulation_indicated") {
      return "✓ No Face Manipulation Indicated";
    }
    if (finding.classification === "ai_generation_indicated") {
      return "⚠ AI-Generated / Synthetic Image Indicated";
    }
    if (finding.classification === "no_ai_generation_indicated") {
      return "✓ No AI Generation Indicated";
    }
    if (finding.classification === "copy_move_candidate") {
      return "⚠ Copy-Move Cloned Region Candidate";
    }
    if (finding.classification === "anomalous_motion_candidate") {
      return "⚠ Anomalous Motion Candidate (Temporal GRU)";
    }
    if (finding.classification === "inconclusive") {
      return "◐ Inconclusive (Calibrated Uncertain Band)";
    }
    return finding.classification.replaceAll("_", " ");
  }

  const loadedEngineCount = useMemo(() => {
    if (!apiOnline || !modelsHealth) return 0;
    return Object.keys(moduleLabels).filter((id) => describeEngineLive(id, modelsHealth, apiOnline).ready).length;
  }, [apiOnline, modelsHealth]);

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
              <i />{apiOnline ? "API CONNECTED" : apiOnline === false ? "BACKEND API OFFLINE" : "CHECKING API"}
            </span>
            <span className="cpu-label">
              {apiOnline ? `${loadedEngineCount}/4 ENGINES READY` : "NO MOCK FALLBACK"}
            </span>
          </div>
        </header>

        <section className="hero">
          <div className="hero-copy">
            <p className="eyebrow">EVIDENCE-LED MEDIA ANALYSIS</p>
            <h1>Review the signal.<br /><em>Keep the context.</em></h1>
            <p className="hero-sub">Upload media or capture from a hardware webcam across four independent Computer Vision engines. All outputs come strictly from loaded backend models — never mock or synthetic scores.</p>
          </div>
          <div className="hero-stamp"><span>UPLOAD + WEBCAM</span><span>·</span><span>SEPARATE SCORES</span></div>
        </section>

        {apiOnline === false && (
          <form onSubmit={handleConnectCustomUrl} className="verdict-banner inconclusive" style={{ marginBottom: "18px" }}>
            <div className="verdict-banner-left" style={{ flex: 1 }}>
              <div className="verdict-banner-title">BACKEND API OFFLINE — REAL MODEL INFERENCE REQUIRED</div>
              <div className="verdict-banner-sub">
                Start the local FastAPI backend (<code>uvicorn deep_shield.api.main:app --port 8002</code> +{" "}
                <code>python -m deep_shield.jobs.worker</code>) or connect to a hosted Deep Shield API URL below. No fake or placeholder scores will be generated.
              </div>
            </div>
            <div style={{ display: "flex", gap: "8px", alignItems: "center", flexWrap: "wrap" }}>
              <input
                type="url"
                value={customUrl}
                onChange={(e) => setCustomUrl(e.target.value)}
                aria-label="FastAPI Backend URL"
                placeholder="http://127.0.0.1:8002"
                style={{
                  background: "#0a171a",
                  color: "#e6f5f0",
                  border: "1px solid #486267",
                  borderRadius: "3px",
                  padding: "7px 10px",
                  font: "10px var(--mono)",
                  minWidth: "220px",
                }}
              />
              <button type="submit" className="cam-btn accent" style={{ padding: "8px 12px" }}>
                CONNECT / RETRY
              </button>
            </div>
          </form>
        )}

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
                onClick={() => setInputMode("camera")}
              >
                ◉ LIVE WEBCAM
              </button>
            </div>

            {inputMode === "camera" && (
              <div className="camera-box">
                <div className="camera-controls-row">
                  <select
                    aria-label="Hardware webcam source"
                    className="camera-select"
                    value={cameraSource}
                    onChange={(e) => setCameraSource(e.target.value)}
                  >
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
                    <video ref={videoRef} playsInline muted autoPlay className="camera-video" />
                  ) : (
                    <div className="camera-idle">
                      <span>◉</span>
                      <p>{cameraError || "Click Start Webcam to request hardware camera access"}</p>
                    </div>
                  )}
                </div>

                <div className="camera-actions-grid">
                  {!cameraActive ? (
                    <button type="button" className="cam-btn" onClick={startCamera}>▶ START WEBCAM</button>
                  ) : (
                    <button type="button" className="cam-btn" onClick={stopCamera}>■ STOP WEBCAM</button>
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
                    disabled={!cameraActive || submitting || !apiOnline}
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

              <button className="primary-button" type="submit" disabled={!file || submitting || apiOnline === false}>
                <span>{submitting ? "ANALYSIS RUNNING" : "START ANALYSIS"}</span><span aria-hidden="true">↗</span>
              </button>
              {error && <div className="error-box" role="alert">{error}</div>}
            </form>

            <div className="engine-list">
              <p className="field-label">LIVE BACKEND ENGINES &amp; WEIGHTS</p>
              {Object.entries(moduleLabels).map(([id, label], index) => {
                const live = describeEngineLive(id, modelsHealth, apiOnline);
                return (
                  <div className="engine-row" key={id} style={{ flexDirection: "column", alignItems: "flex-start", gap: "3px" }}>
                    <div style={{ display: "flex", width: "100%", alignItems: "center", gap: "10px" }}>
                      <span className="engine-number">0{index + 1}</span>
                      <span>{label}</span>
                      <span
                        className="engine-dot"
                        style={{
                          background: live.ready ? "var(--mint)" : "var(--red)",
                          boxShadow: live.ready ? "0 0 10px #8ce7cf70" : "0 0 10px #f4827d70",
                        }}
                      />
                    </div>
                    {live.lines.map((line, lineIdx) => (
                      <small key={lineIdx} style={{ font: "8px var(--mono)", color: "#799396", paddingLeft: "23px", lineHeight: 1.35 }}>
                        {line}
                      </small>
                    ))}
                  </div>
                );
              })}
            </div>
          </aside>

          <section className="panel results-panel" aria-live="polite">
            <div className="panel-heading results-heading"><div><p className="eyebrow">ANALYSIS OUTPUT</p><h2>{result ? "Review findings" : "Your review will appear here"}</h2></div>
              {job && <span className={`status-chip ${job.status}`}>{job.status.toUpperCase()}</span>}
            </div>

            {!job && <div className="empty-state"><div className="empty-glyph">◌</div><h3>Start with a source file or live webcam capture</h3><p>Upload an image or video (or switch to Live Webcam) to queue a real multi-engine backend analysis job.</p></div>}
            {job && !result && job.status !== "failed" && <div className="progress-state"><div className="spinner" /><div><strong>{job.status === "queued" ? "Waiting in queue" : "Engines are analyzing the source"}</strong><p>Job {job.job_id.slice(0, 12)} · Real model inference running on backend.</p></div></div>}
            {job?.status === "failed" && <div className="error-box" role="alert">{job.error || "The analysis job failed. Check the backend logs for details."}</div>}

            {result && <>
              <div className="result-meta"><span>MEDIA <b>{result.media_id}</b></span><span>SHA-256 <b>{result.media_sha256.slice(0, 18)}…</b></span>
                {job && apiOnline && <a className="report-button" href={reportUrl(job.job_id)} target="_blank" rel="noreferrer">DOWNLOAD PDF <span>↓</span></a>}
              </div>

              {bannerState && (
                <div className={`verdict-banner ${bannerState.tone}`}>
                  <div className="verdict-banner-left">
                    <div className="verdict-banner-title">{bannerState.title}</div>
                    <div className="verdict-banner-sub">{bannerState.sub}</div>
                  </div>
                  <span className="verdict-pill">{bannerState.pill}</span>
                </div>
              )}

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
              <div className="findings-list">{findings.length ? findings.map((finding) => <article className={`finding-row ${isManipFlagged(finding) ? "flagged-fake" : isNoManipIndicated(finding) ? "flagged-real" : ""}`} key={finding.finding_id}>
                <div className="finding-top"><span className="finding-module">{moduleLabels[finding.module] || finding.module}</span><span className="finding-time">{formatTime(finding)}</span></div>
                <div className="finding-main">
                  <strong>{formatFindingTitle(finding)}</strong>
                  {isManipFlagged(finding) && (
                    <span className="finding-badge fake">
                      FLAGGED · {finding.score !== null ? `${(finding.score * 100).toFixed(1)}%` : "REVIEW"}
                    </span>
                  )}
                  {isNoManipIndicated(finding) && (
                    <span className="finding-badge real">
                      NO INDICATOR · {finding.score !== null ? finding.score.toFixed(3) : "OK"}
                    </span>
                  )}
                  <span className="finding-task">{finding.task}</span>
                </div>
                <div className="finding-bottom">
                  <span>{finding.score_type.replaceAll("_", " ")}{finding.score === null ? "" : ` · ${finding.score.toFixed(3)}`}</span>
                  {finding.thresholds?.low !== undefined && finding.thresholds?.high !== undefined && (
                    <span>thr {finding.thresholds.low}/{finding.thresholds.high}</span>
                  )}
                  {finding.model?.model_id && <span>{finding.model.model_id}</span>}
                  {finding.track_id && <span>Track {finding.track_id}</span>}
                </div>
                {!!finding.limitations?.length && (
                  <p style={{ margin: "6px 0 0", font: "11px/1.45 var(--sans, system-ui)", color: "#8ba6aa" }}>
                    <strong style={{ color: "#a5c3c7", fontWeight: 600 }}>Caveat:</strong> {finding.limitations[0]}
                  </p>
                )}
                {!!finding.evidence.length && <div className="evidence-links">{finding.evidence.map((evidence) => {
                  const url = evidenceUrl(finding.module, evidence.path);
                  return url ? <a key={evidence.path} href={url} target="_blank" rel="noreferrer">{evidence.kind.replaceAll("_", " ")} ↗</a> : null;
                })}</div>}
              </article>) : <p className="muted-line">No findings were returned.</p>}</div>
              <p className="disclaimer">{result.disclaimer}</p>
            </>}
          </section>
        </section>

        <footer className="footer"><span>DEEP SHIELD</span><span>Indicators are advisory. Review original media before drawing conclusions.</span><span>{resolvedBase || getActiveApiBase() || "SAME-ORIGIN API"}</span></footer>
      </div>
    </div>
  );
}
