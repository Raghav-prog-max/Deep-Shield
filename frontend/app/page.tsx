"use client";

import { ChangeEvent, FormEvent, useEffect, useMemo, useState } from "react";
import { API_BASE, createAnalysisJob, evidenceUrl, getJob, reportUrl } from "@/lib/api";
import type { Finding, JobRecord } from "@/lib/types";

const moduleLabels: Record<string, string> = {
  M01_threat_intelligence: "Threat and anomaly",
  M02_deepfake_image_integrity: "Deepfake and integrity",
  M03_video_events: "Video events",
  M04_visual_forensics: "Visual forensics",
};

function formatTime(finding: Finding) {
  const start = finding.time_start_s ?? finding.timestamp_s;
  const end = finding.time_end_s ?? finding.timestamp_s;
  if (start === null || start === undefined) return "Image";
  if (end !== null && end !== undefined && end !== start) return `${start.toFixed(1)}–${end.toFixed(1)}s`;
  return `${start.toFixed(1)}s`;
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [zones, setZones] = useState("");
  const [job, setJob] = useState<JobRecord | null>(null);
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);
  const previewUrl = useMemo(() => file && file.type.startsWith("image/") ? URL.createObjectURL(file) : "", [file]);

  useEffect(() => () => { if (previewUrl) URL.revokeObjectURL(previewUrl); }, [previewUrl]);
  useEffect(() => {
    let active = true;
    fetch(`${API_BASE}/api/v1/m01/health`).then((response) => {
      if (active) setApiOnline(response.ok);
    }).catch(() => { if (active) setApiOnline(false); });
    return () => { active = false; };
  }, []);

  async function pollJob(jobId: string) {
    for (let attempt = 0; attempt < 900; attempt += 1) {
      await new Promise((resolve) => setTimeout(resolve, 2000));
      const latest = await getJob(jobId);
      setJob(latest);
      if (["completed", "partial", "failed"].includes(latest.status)) return;
    }
    throw new Error("Analysis is taking longer than expected. The job is still saved; refresh and check its status.");
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!file) return;
    setError("");
    setSubmitting(true);
    setJob(null);
    try {
      const created = await createAnalysisJob(file, zones);
      setJob(created);
      await pollJob(created.job_id);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not start analysis.");
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
    <main className="shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="Deep Shield home">
          <span className="brand-mark">DS</span>
          <span><strong>DEEP SHIELD</strong><small>MEDIA REVIEW WORKSPACE</small></span>
        </a>
        <div className="top-meta">
          <span className={`api-pill ${apiOnline ? "online" : apiOnline === false ? "offline" : "checking"}`}>
            <i />{apiOnline ? "LOCAL API CONNECTED" : apiOnline === false ? "API OFFLINE" : "CHECKING API"}
          </span>
          <span className="cpu-label">CPU INFERENCE</span>
        </div>
      </header>

      <section className="hero" id="top">
        <div className="hero-copy">
          <p className="eyebrow">EVIDENCE-LED MEDIA ANALYSIS</p>
          <h1>Review the signal.<br /><em>Keep the context.</em></h1>
          <p className="hero-sub">Run four independent analysis engines on one image or video. Findings remain advisory, traceable, and separate by model.</p>
        </div>
        <div className="hero-stamp"><span>LOCAL</span><span>·</span><span>HUMAN REVIEW</span></div>
      </section>

      <section className="workspace-grid">
        <aside className="panel intake-panel">
          <div className="panel-heading"><div><p className="eyebrow">NEW REVIEW</p><h2>Choose media</h2></div><span className="step-index">01</span></div>
          <form onSubmit={submit}>
            <label className={`dropzone ${file ? "has-file" : ""}`} htmlFor="media-file">
              {previewUrl ? <img className="image-preview" src={previewUrl} alt="Selected media preview" /> : <span className="upload-icon">↥</span>}
              <strong>{file ? file.name : "Drop a file or browse"}</strong>
              <span>{file ? `${(file.size / (1024 * 1024)).toFixed(1)} MB · ${file.type || "media"}` : "Images and videos · up to 200 MB"}</span>
              <input id="media-file" type="file" accept="image/*,video/*,.mp4,.avi,.mov,.mkv,.webm" onChange={onFileChange} />
            </label>

            <label className="field-label" htmlFor="zones">REVIEW ZONES <span>OPTIONAL</span></label>
            <textarea id="zones" value={zones} onChange={(event) => setZones(event.target.value)} placeholder={'[{"zone_id":"lobby","polygon":[[0.1,0.1],[0.9,0.1],[0.9,0.9],[0.1,0.9]]}]'} rows={4} spellCheck={false} />
            <p className="field-help">Normalized coordinates from 0 to 1. Used for zone and dwell candidates in M01 and M03.</p>

            <button className="primary-button" type="submit" disabled={!file || submitting}>
              <span>{submitting ? "ANALYSIS RUNNING" : "START ANALYSIS"}</span><span aria-hidden="true">↗</span>
            </button>
            {error && <div className="error-box" role="alert">{error}</div>}
          </form>

          <div className="engine-list">
            <p className="field-label">ENGINES IN THIS REVIEW</p>
            {Object.entries(moduleLabels).map(([id, label], index) => (
              <div className="engine-row" key={id}><span className="engine-number">0{index + 1}</span><span>{label}</span><span className="engine-dot" /></div>
            ))}
          </div>
        </aside>

        <section className="panel results-panel" aria-live="polite">
          <div className="panel-heading results-heading"><div><p className="eyebrow">ANALYSIS OUTPUT</p><h2>{result ? "Review findings" : "Your review will appear here"}</h2></div>
            {job && <span className={`status-chip ${job.status}`}>{job.status.toUpperCase()}</span>}
          </div>

          {!job && <div className="empty-state"><div className="empty-glyph">◌</div><h3>Start with a source file</h3><p>Engine results, time-linked observations, and evidence references will be collected in this panel.</p></div>}
          {job && !result && job.status !== "failed" && <div className="progress-state"><div className="spinner" /><div><strong>{job.status === "queued" ? "Waiting in queue" : "Engines are analyzing the source"}</strong><p>Job {job.job_id.slice(0, 12)} · You can keep this page open while it runs.</p></div></div>}
          {job?.status === "failed" && <div className="error-box" role="alert">{job.error || "The analysis job failed. Check the backend logs for details."}</div>}

          {result && <>
            <div className="result-meta"><span>MEDIA <b>{result.media_id}</b></span><span>SHA-256 <b>{result.media_sha256.slice(0, 18)}…</b></span>
              {job && <a className="report-button" href={reportUrl(job.job_id)} target="_blank" rel="noreferrer">DOWNLOAD PDF <span>↓</span></a>}
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

      <footer className="footer"><span>DEEP SHIELD</span><span>Indicators are advisory. Review original media before drawing conclusions.</span><span>{API_BASE}</span></footer>
    </main>
  );
}
