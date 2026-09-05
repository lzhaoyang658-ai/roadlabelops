import type { AppError, Dashboard, ReleaseReceipt, Session } from "./types";

export const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "/api/v1";
export const CVAT_BASE = process.env.NEXT_PUBLIC_CVAT_BASE_URL ?? "http://localhost:8080";

type ErrorEnvelope = {
  error?: { code?: string; message?: string; retryable?: boolean; request_id?: string };
};

export class ApiError extends Error implements AppError {
  code: string;
  retryable: boolean;
  requestId?: string;

  constructor(message: string, code = "REQUEST_FAILED", retryable = false, requestId?: string) {
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.retryable = retryable;
    this.requestId = requestId;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const isFormData = init?.body instanceof FormData;
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { ...(isFormData ? {} : { "Content-Type": "application/json" }), ...init?.headers },
  });
  if (!response.ok) {
    const body = (await response.json().catch(() => ({}))) as ErrorEnvelope;
    throw new ApiError(
      body.error?.message ?? "RoadLabelOps 服务暂时不可用",
      body.error?.code ?? "REQUEST_FAILED",
      body.error?.retryable ?? response.status >= 500,
      body.error?.request_id,
    );
  }
  return (await response.json()) as T;
}

export function getDashboard(signal?: AbortSignal): Promise<Dashboard> {
  return request<Dashboard>("/dashboard", { signal, cache: "no-store" });
}

export function createDemo(): Promise<{ session: Session }> {
  return request<{ session: Session }>("/demo", { method: "POST", body: "{}" });
}

export function uploadVideo(file: File, sceneSeconds = 15): Promise<{ session: Session; existing: boolean }> {
  const body = new FormData();
  body.append("video", file);
  return request<{ session: Session; existing: boolean }>(
    `/sessions/upload?scene_seconds=${sceneSeconds}`,
    { method: "POST", body },
  );
}

export function thumbnailUrl(sceneId: string): string {
  return `${API_BASE}/scenes/${encodeURIComponent(sceneId)}/thumbnail`;
}

export function cvatJobUrl(taskId: number, jobId?: number): string {
  return jobId
    ? `${CVAT_BASE}/tasks/${taskId}/jobs/${jobId}`
    : `${CVAT_BASE}/tasks/${taskId}`;
}

export function cvatProjectUrl(projectId: number): string {
  return `${CVAT_BASE}/projects/${projectId}`;
}

export function advanceSession(
  sessionId: string,
  action: string,
  version = "1.0.0",
  approved = false,
): Promise<{ session: Session }> {
  return request<{ session: Session }>(`/sessions/${sessionId}/actions/${action}`, {
    method: "POST",
    body: JSON.stringify({ version, approved }),
  });
}

export function verifySessionRelease(sessionId: string): Promise<{ receipt: ReleaseReceipt }> {
  return request<{ receipt: ReleaseReceipt }>(`/sessions/${sessionId}/release/verify`, {
    method: "POST",
    body: "{}",
  });
}
