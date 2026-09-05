import type { Session, Stage } from "./types";

export const stageOrder: Stage[] = [
  "SLICED",
  "TASKS_CREATED",
  "PREANNOTATED",
  "WAITING_FOR_HUMAN_REVIEW",
  "REVIEW_COMPLETED",
  "QUALITY_CALCULATED",
  "RELEASED",
];

export const stageCopy: Record<Stage, { label: string; detail: string }> = {
  NEW: { label: "等待导入", detail: "选择一段道路视频开始" },
  PROBED: { label: "视频已探测", detail: "元数据与来源哈希已记录" },
  SLICED: { label: "场景已切片", detail: "可以创建 CVAT 标注任务" },
  TASKS_CREATED: { label: "任务已创建", detail: "下一步生成目标检测预标注" },
  PREANNOTATED: { label: "预标注完成", detail: "请确认后进入人工审核" },
  WAITING_FOR_HUMAN_REVIEW: { label: "等待人工验收", detail: "在 CVAT 中修正标注，再将 Job 设为 acceptance / completed" },
  REVIEW_COMPLETED: { label: "审核已完成", detail: "可以计算真实质量指标" },
  QUALITY_CALCULATED: { label: "质量已计算", detail: "数据已满足 Release 前检查" },
  RELEASED: { label: "数据集已发布", detail: "Manifest 与校验和已冻结" },
  FAILED_RETRYABLE: { label: "可恢复失败", detail: "已有结果已保留，可安全重试" },
  WAITING_FOR_PERMISSION: { label: "等待确认", detail: "操作可能覆盖现有资产" },
  FAILED_FINAL: { label: "流程已停止", detail: "请根据错误信息修复后新建任务" },
  CANCELLED: { label: "已取消", detail: "已有中间结果仍然保留" },
};

export const nextAction: Partial<Record<Stage, { action: string; label: string }>> = {
  SLICED: { action: "create_tasks", label: "创建 CVAT 任务" },
  TASKS_CREATED: { action: "preannotate", label: "生成预标注" },
  PREANNOTATED: { action: "request_review", label: "提交人工审核" },
  WAITING_FOR_HUMAN_REVIEW: { action: "complete_review", label: "同步 CVAT 验收结果" },
  REVIEW_COMPLETED: { action: "calculate_quality", label: "计算质量指标" },
  QUALITY_CALCULATED: { action: "release", label: "发布 COCO / YOLO Release" },
};

export const actionCopy: Record<string, string> = Object.fromEntries(
  Object.values(nextAction).map((item) => [item.action, item.label]),
);

export function nextSessionAction(
  session: Pick<Session, "status" | "pending_action">,
): { action: string; label: string; needsApproval: boolean } | null {
  if (
    (session.status === "FAILED_RETRYABLE" || session.status === "WAITING_FOR_PERMISSION")
    && session.pending_action
  ) {
    return {
      action: session.pending_action,
      label: session.status === "WAITING_FOR_PERMISSION"
        ? `确认并${actionCopy[session.pending_action] ?? "继续"}`
        : `重试：${actionCopy[session.pending_action] ?? session.pending_action}`,
      needsApproval: session.status === "WAITING_FOR_PERMISSION",
    };
  }
  const action = nextAction[session.status];
  return action ? { ...action, needsApproval: false } : null;
}

export function stageProgress(stage: Stage): number {
  const index = stageOrder.indexOf(stage);
  if (index < 0) return 0;
  return Math.round(((index + 1) / stageOrder.length) * 100);
}
