import type { JobRecord } from "@/lib/types";

// Requests stay on the frontend origin; Next.js proxies /api to the backend.
export const API_BASE = (process.env.NEXT_PUBLIC_API_BASE_URL || "").replace(/\/$/, "");

async function readJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    const detail = body?.detail;
    const message = typeof detail === "string" ? detail : detail?.error || response.statusText;
    throw new Error(message || `Request failed (${response.status})`);
  }
  return response.json() as Promise<T>;
}

export async function createAnalysisJob(file: File, zones: string): Promise<JobRecord> {
  const form = new FormData();
  form.append("file", file);
  if (zones.trim()) form.append("zones", zones.trim());
  const response = await fetch(`${API_BASE}/api/v1/intelligence/jobs`, {
    method: "POST",
    body: form,
  });
  return readJson<JobRecord>(response);
}

export async function getJob(jobId: string): Promise<JobRecord> {
  const response = await fetch(`${API_BASE}/api/v1/intelligence/jobs/${encodeURIComponent(jobId)}`, {
    cache: "no-store",
  });
  return readJson<JobRecord>(response);
}

export function reportUrl(jobId: string): string {
  return `${API_BASE}/api/v1/intelligence/jobs/${encodeURIComponent(jobId)}/report.pdf`;
}

export function evidenceUrl(module: string, path: string): string | null {
  const route = module.startsWith("M01") ? "m01" : module.startsWith("M02") ? "m02" :
    module.startsWith("M03") ? "m03" : module.startsWith("M04") ? "m04" : null;
  const parts = path.split("/").filter(Boolean);
  if (!route || parts.length < 2) return null;
  const mediaId = encodeURIComponent(parts[parts.length - 2] || "");
  const filename = encodeURIComponent(parts[parts.length - 1] || "");
  return `${API_BASE}/api/v1/${route}/evidence/${mediaId}/${filename}`;
}
