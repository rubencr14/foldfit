const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      headers: { "Content-Type": "application/json", ...options?.headers },
      ...options,
    });
  } catch {
    throw new Error("Cannot connect to backend. Run: conda activate nomosis && make backend");
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail || body.error || res.statusText);
  }
  return res.json();
}

// ── Datasets ─────────────────────────────────────────────────────────────────

export interface Dataset {
  id: string;
  name: string;
  num_structures: number;
  resolution_max: number;
  antibody_type: string;
  organism: string;
  method: string;
  source: string;
  pdb_paths: string[];
  created_at: string;
}

export interface DatasetListResponse {
  datasets: Dataset[];
  total: number;
}

export function listDatasets(): Promise<DatasetListResponse> {
  return request("/v1/datasets");
}

export function createDataset(body: {
  name: string;
  max_structures: number;
  resolution_max: number;
  antibody_type?: string;
  organism?: string;
  method?: string;
}): Promise<Dataset> {
  return request("/v1/datasets", { method: "POST", body: JSON.stringify(body) });
}

export function deleteDataset(id: string): Promise<void> {
  return request(`/v1/datasets/${id}`, { method: "DELETE" });
}

export function getDataset(id: string): Promise<Dataset> {
  return request(`/v1/datasets/${id}`);
}

// ── Fine-tuning ──────────────────────────────────────────────────────────────

export interface FinetuneJob {
  job_id: string;
  name: string;
  dataset_id: string;
  status: string;
  progress: number;
  epoch: number;
  total_epochs: number;
  train_loss: number;
  val_loss: number | null;
  lora_rank: number;
  config: Record<string, unknown>;
  metrics: Record<string, unknown>;
  created_at: string;
}

export interface FinetuneJobListResponse {
  jobs: FinetuneJob[];
  total: number;
}

export function listFinetuneJobs(): Promise<FinetuneJobListResponse> {
  return request("/v1/finetune");
}

export function startFinetune(body: Record<string, unknown>): Promise<FinetuneJob> {
  return request("/v1/finetune", { method: "POST", body: JSON.stringify(body) });
}

export function getFinetuneJob(jobId: string): Promise<FinetuneJob> {
  return request(`/v1/finetune/${jobId}`);
}

export function deleteFinetuneJob(jobId: string): Promise<void> {
  return request(`/v1/finetune/${jobId}`, { method: "DELETE" });
}

// ── Prediction ───────────────────────────────────────────────────────────────

export interface PredictResult {
  sequence_length: number;
  confidence: number[] | null;
  mean_plddt: number | null;
  pdb_string: string | null;
}

export function predictStructure(body: {
  sequence: string;
  adapter_path?: string;
  device?: string;
}): Promise<PredictResult> {
  return request("/v1/predict", { method: "POST", body: JSON.stringify(body) });
}

// ── MSA ──────────────────────────────────────────────────────────────────────

export interface MsaResult {
  num_sequences: number;
  sequence_length: number;
  backend: string;
}

export function computeMsa(body: {
  sequence: string;
  backend?: string;
}): Promise<MsaResult> {
  return request("/v1/msa", { method: "POST", body: JSON.stringify(body) });
}

// ── Health ────────────────────────────────────────────────────────────────────

export function healthCheck(): Promise<{ status: string }> {
  return request("/health");
}
