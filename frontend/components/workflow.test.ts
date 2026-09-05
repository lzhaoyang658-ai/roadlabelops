import { describe, expect, it } from "vitest";
import { nextAction, nextSessionAction, stageProgress } from "@/lib/workflow";

describe("workflow mapping", () => {
  it("maps review state to a meaningful action", () => {
    expect(nextAction.WAITING_FOR_HUMAN_REVIEW).toEqual({
      action: "complete_review",
      label: "同步 CVAT 验收结果",
    });
  });

  it("reports a complete release", () => {
    expect(stageProgress("RELEASED")).toBe(100);
  });

  it("restores the exact retryable action", () => {
    expect(nextSessionAction({ status: "FAILED_RETRYABLE", pending_action: "preannotate" })).toEqual({
      action: "preannotate",
      label: "重试：生成预标注",
      needsApproval: false,
    });
  });

  it("requires explicit approval for a paused permission", () => {
    expect(nextSessionAction({ status: "WAITING_FOR_PERMISSION", pending_action: "release" })).toEqual({
      action: "release",
      label: "确认并发布 COCO / YOLO Release",
      needsApproval: true,
    });
  });
});
