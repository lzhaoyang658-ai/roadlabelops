export type Stage =
  | "NEW"
  | "PROBED"
  | "SLICED"
  | "TASKS_CREATED"
  | "PREANNOTATED"
  | "WAITING_FOR_HUMAN_REVIEW"
  | "REVIEW_COMPLETED"
  | "QUALITY_CALCULATED"
  | "RELEASED"
  | "FAILED_RETRYABLE"
  | "WAITING_FOR_PERMISSION"
  | "FAILED_FINAL"
  | "CANCELLED";

export type Scene = {
  scene_id: string;
  start_seconds: number;
  end_seconds: number;
  video_path: string;
  thumbnail_path: string | null;
  cvat_project_id: number | null;
  cvat_task_id: number | null;
  cvat_job_ids: number[];
  status: string;
  prediction_count: number | null;
  final_count: number | null;
};

export type Session = {
  session_id: string;
  name: string;
  duration_seconds: number;
  fps: number;
  width: number;
  height: number;
  status: Stage;
  created_at: string;
  updated_at: string;
  scenes: Scene[];
  demo: boolean;
  resume_stage: Stage | null;
  pending_action: string | null;
  last_error: { code: string; message: string } | null;
};

export type ClassQuality = {
  true_positive_count: number;
  false_positive_count: number;
  false_negative_count: number;
  precision: number | null;
  recall: number | null;
  f1_score: number | null;
};

export type Quality = {
  prediction_count: number;
  final_count: number;
  retained_count: number;
  added_count: number;
  removed_count: number;
  retention_rate: number | null;
  human_addition_rate: number | null;
  precision: number | null;
  recall: number | null;
  f1_score: number | null;
  evaluated_frame_count: number;
  clean_frame_count: number;
  clean_frame_rate: number | null;
  first_pass_acceptance_rate: number | null;
  first_pass_acceptance_reason?: string | null;
  class_distribution: Record<string, number>;
  per_class: Record<string, ClassQuality>;
};

export type ReleaseReceipt = {
  release_id: string;
  version?: string;
  path?: string;
  manifest_sha256?: string;
  file_count?: number;
  verified: boolean;
  checked_at?: string;
  errors?: string[];
};

export type Activity = {
  event: string;
  stage: Stage;
  session_id: string;
  timestamp: string;
  tool_name?: string;
};

export type Dashboard = {
  summary: {
    session_count: number;
    scene_count: number;
    task_count: number;
    ready_for_review: number;
  };
  sessions: Session[];
  quality: Quality | null;
  qualities?: Record<string, Quality>;
  releases?: Record<string, ReleaseReceipt>;
  operational_metrics?: {
    video_slice_success_rate: OperationalMetric;
    task_creation_success_rate: OperationalMetric;
    release_verification_success_rate: OperationalMetric;
  };
  activity: Activity[];
};

export type OperationalMetric = {
  value: number | null;
  numerator: number;
  denominator: number;
  reason: string | null;
};

export type AppError = {
  code: string;
  message: string;
  retryable: boolean;
  requestId?: string;
};
