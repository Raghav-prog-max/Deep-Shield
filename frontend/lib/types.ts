export type EvidenceRef = {
  kind: string;
  path: string;
  description?: string | null;
};

export type ModelInfo = {
  model_id: string;
  revision?: string | null;
  task?: string | null;
  backend?: string | null;
};

export type Finding = {
  finding_id: string;
  module: string;
  task: string;
  level: string;
  classification: string;
  score: number | null;
  score_type: string;
  thresholds?: { low?: number; high?: number } | null;
  timestamp_s: number | null;
  time_start_s: number | null;
  time_end_s: number | null;
  track_id: string | null;
  model?: ModelInfo | null;
  evidence: EvidenceRef[];
  limitations: string[];
};

export type TaskSummary = {
  task: string;
  classification: string;
  score: number | null;
  score_type: string | null;
  basis: string;
};

export type ModuleResult = {
  module: string;
  status: string;
  findings: Finding[];
  task_summaries: TaskSummary[];
  warnings: string[];
};

export type FindingLink = {
  finding_ids: string[];
  modules: string[];
  relationship: string;
  time_overlap_s: number | null;
  region_overlap_ratio: number | null;
  explanation: string;
};

export type FindingGroup = {
  group_id: string;
  finding_ids: string[];
  modules: string[];
  link_indices: number[];
};

export type IntelligenceResult = {
  media_id: string;
  media_sha256: string;
  module_results: ModuleResult[];
  links: FindingLink[];
  groups: FindingGroup[];
  unlinked_finding_ids: string[];
  disclaimer: string;
};

export type JobRecord = {
  job_id: string;
  status: "queued" | "running" | "completed" | "partial" | "failed";
  source_name: string;
  created_at: string;
  updated_at: string;
  media_id: string | null;
  result: IntelligenceResult | null;
  error: string | null;
};
