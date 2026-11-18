import type { JobRecord } from "@/lib/types";

// Default configured base; automatically checks same-origin and local FastAPI (127.0.0.1:8002).
export const API_BASE = (process.env.NEXT_PUBLIC_API_BASE_URL || "").replace(/\/$/, "");

let activeApiBase: string = API_BASE;

export function getActiveApiBase(): string {
  return activeApiBase;
}

export function setCustomApiBase(url: string): void {
  activeApiBase = url.trim().replace(/\/$/, "");
}

export async function resolveApiBase(preferredBase?: string): Promise<{ online: boolean; base: string }> {
  const candidates = Array.from(
    new Set(
      [
        preferredBase !== undefined ? preferredBase.trim().replace(/\/$/, "") : null,
        API_BASE,
        "",
        "http://127.0.0.1:8002",
        "http://localhost:8002",
      ].filter((v): v is string => v !== null)
    )
  );
  for (const base of candidates) {
    try {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 2500);
      const resp = await fetch(`${base}/api/v1/m01/health`, {
        signal: controller.signal,
        cache: "no-store",
      });
      clearTimeout(timer);
      if (resp.ok) {
        const data = await resp.json().catch(() => null);
        if (data && data.module) {
          activeApiBase = base;
          return { online: true, base };
        }
      }
    } catch {
      // try next candidate
    }
  }
  return { online: false, base: activeApiBase };
}

export async function fetchLoadedModels(): Promise<Record<string, unknown> | null> {
  try {
    const resp = await fetch(`${activeApiBase}/api/v1/intelligence/models`, { cache: "no-store" });
    if (!resp.ok) return null;
    return (await resp.json()) as Record<string, unknown>;
  } catch {
    return null;
  }
}

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
  const response = await fetch(`${activeApiBase}/api/v1/intelligence/jobs`, {
    method: "POST",
    body: form,
  });
  return readJson<JobRecord>(response);
}

export async function getJob(jobId: string): Promise<JobRecord> {
  const response = await fetch(`${activeApiBase}/api/v1/intelligence/jobs/${encodeURIComponent(jobId)}`, {
    cache: "no-store",
  });
  return readJson<JobRecord>(response);
}

export function reportUrl(jobId: string): string {
  return `${activeApiBase}/api/v1/intelligence/jobs/${encodeURIComponent(jobId)}/report.pdf`;
}

export function evidenceUrl(module: string, path: string): string | null {
  if (!path) return null;
  const route = module.startsWith("M01") ? "m01" : module.startsWith("M02") ? "m02" :
    module.startsWith("M03") ? "m03" : module.startsWith("M04") ? "m04" : null;
  const parts = path.split("/").filter(Boolean);
  if (!route || parts.length < 2) return null;
  const mediaId = encodeURIComponent(parts[parts.length - 2] || "");
  const filename = encodeURIComponent(parts[parts.length - 1] || "");
  return `${activeApiBase}/api/v1/${route}/evidence/${mediaId}/${filename}`;
}
