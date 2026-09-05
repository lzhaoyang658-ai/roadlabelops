import { expect, test, type Route } from "@playwright/test";

const emptyQuality = {
  prediction_count: 24,
  final_count: 22,
  retained_count: 20,
  added_count: 2,
  removed_count: 4,
  retention_rate: 0.8333,
  human_addition_rate: 0.0909,
  precision: 0.8333,
  recall: 0.9091,
  f1_score: 0.8695,
  evaluated_frame_count: 24,
  clean_frame_count: 19,
  clean_frame_rate: 0.7917,
  first_pass_acceptance_rate: 1,
  class_distribution: { car: 16, pedestrian: 4, traffic_sign: 2 },
  per_class: {},
};

function session(stage = "SLICED", id = "session_demo_test", name = "上海高架夜间道路采样") {
  return {
    session_id: id,
    name,
    source_path: "data/raw/demo.mp4",
    source_sha256: "a".repeat(64),
    duration_seconds: 60,
    fps: 25,
    width: 1920,
    height: 1080,
    scene_seconds: 15,
    frame_step: 5,
    status: stage,
    created_at: "2026-09-04T04:00:00Z",
    updated_at: "2026-09-04T04:00:00Z",
    demo: true,
    resume_stage: null,
    pending_action: null,
    last_error: null,
    scenes: [
      {
        scene_id: `${id}_scene_001`,
        session_id: id,
        start_seconds: 0,
        end_seconds: 15,
        video_path: "data/scenes/demo/scene_001.mp4",
        thumbnail_path: null as string | null,
        cvat_project_id: stage === "SLICED" ? null : 70,
        cvat_task_id: stage === "SLICED" ? null : 701,
        cvat_job_ids: stage === "SLICED" ? [] : [7011],
        status: stage === "RELEASED" ? "completed" : "annotation",
        prediction_count: stage === "SLICED" ? null : 24,
        final_count: stage === "RELEASED" ? 22 : null,
      },
    ],
  };
}

type MockDashboard = {
  summary: {
    session_count: number;
    scene_count: number;
    task_count: number;
    ready_for_review: number;
  };
  sessions: ReturnType<typeof session>[];
  quality: typeof emptyQuality | null;
  qualities: Record<string, typeof emptyQuality>;
  releases: Record<string, Record<string, unknown>>;
  activity: never[];
};

function dashboard(sessions: ReturnType<typeof session>[] = []): MockDashboard {
  return {
    summary: {
      session_count: sessions.length,
      scene_count: sessions.length,
      task_count: sessions.filter((item) => item.status !== "SLICED").length,
      ready_for_review: sessions.filter((item) => item.status === "WAITING_FOR_HUMAN_REVIEW").length,
    },
    sessions,
    quality: null,
    qualities: {},
    releases: {},
    activity: [],
  };
}

test.beforeEach(async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
});

async function fulfillJson(route: Route, value: unknown, status = 200) {
  await route.fulfill({
    status,
    contentType: "application/json",
    body: JSON.stringify(value),
  });
}

test("renders an offline-safe workspace without horizontal overflow", async ({ page }) => {
  await page.route("**/api/v1/dashboard", (route) => fulfillJson(route, dashboard()));

  await page.goto("/");

  await expect(page.getByRole("heading", { name: /让每一帧道路/ })).toBeVisible();
  await expect(page.getByRole("button", { name: "使用演示数据" }).first()).toBeVisible();
  const sourceLink = page.getByRole("link", { name: "查看源码" });
  await expect(sourceLink).toHaveAttribute(
    "href",
    "https://github.com/lzhaoyang658-ai/roadlabelops",
  );
  await expect(sourceLink).toHaveAttribute("target", "_blank");
  await expect(sourceLink).toHaveAttribute("rel", /noreferrer/);
  await expect(page.locator("main")).toHaveCSS("overflow-x", "clip");
  const dimensions = await page.evaluate(() => ({
    viewport: window.innerWidth,
    document: document.documentElement.scrollWidth,
  }));
  expect(dimensions.document).toBeLessThanOrEqual(dimensions.viewport);
});

test("completes the demo workflow and only passes a verified release", async ({ page }) => {
  let current = session();
  let quality: typeof emptyQuality | null = null;
  let receipt: Record<string, unknown> | null = null;
  const stages: Record<string, string> = {
    create_tasks: "TASKS_CREATED",
    preannotate: "PREANNOTATED",
    request_review: "WAITING_FOR_HUMAN_REVIEW",
    complete_review: "REVIEW_COMPLETED",
    calculate_quality: "QUALITY_CALCULATED",
    release: "RELEASED",
  };

  await page.route("**/api/v1/**", async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    if (request.method() === "GET" && path.endsWith("/dashboard")) {
      const value = dashboard([current]);
      value.quality = quality;
      value.qualities = quality ? { [current.session_id]: quality } : {};
      value.releases = receipt ? { [current.session_id]: receipt } : {};
      await fulfillJson(route, value);
      return;
    }
    if (request.method() === "POST" && path.endsWith("/demo")) {
      await fulfillJson(route, { session: current }, 201);
      return;
    }
    if (request.method() === "POST" && path.endsWith("/release/verify")) {
      await fulfillJson(route, { receipt });
      return;
    }
    const action = path.match(/\/actions\/([^/]+)$/)?.[1];
    if (request.method() === "POST" && action && stages[action]) {
      current = session(stages[action]);
      if (action === "calculate_quality") quality = emptyQuality;
      if (action === "release") {
        receipt = {
          release_id: `${current.session_id}-v1.0.0`,
          version: "1.0.0",
          path: `data/releases/${current.session_id}-v1.0.0`,
          manifest_sha256: "b".repeat(64),
          file_count: 6,
          verified: true,
          checked_at: "2026-09-04T04:10:00Z",
          errors: [],
        };
      }
      await fulfillJson(route, { session: current, receipt });
      return;
    }
    await fulfillJson(route, { error: { code: "NOT_FOUND", message: "not mocked" } }, 404);
  });

  await page.goto("/");
  await page.getByRole("button", { name: "使用演示数据" }).first().click();
  await expect(page.getByRole("heading", { name: current.name })).toBeVisible();

  for (const label of [
    "创建 CVAT 任务",
    "生成预标注",
    "提交人工审核",
    "同步 CVAT 验收结果",
    "计算质量指标",
    "发布 COCO / YOLO Release",
  ]) {
    await page.getByRole("button", { name: label }).click();
  }

  await expect(page.getByText("Release 文件与哈希验证通过")).toBeVisible();
  await page.locator("#quality").scrollIntoViewIfNeeded();
  await expect(page.getByText("PASSED")).toBeVisible();
  await expect(page.getByText("87%")).toBeVisible();
});

test("switches sessions and shows quality for the selected session", async ({ page }) => {
  const first = session("QUALITY_CALCULATED", "session_first", "高架主路");
  const second = session("QUALITY_CALCULATED", "session_second", "城市路口");
  const secondQuality = { ...emptyQuality, f1_score: 0.42 };
  const value = dashboard([first, second]);
  value.quality = emptyQuality;
  value.qualities = {
    [first.session_id]: emptyQuality,
    [second.session_id]: secondQuality,
  };
  await page.route("**/api/v1/dashboard", (route) => fulfillJson(route, value));

  await page.goto("/");
  await page.getByLabel("当前 Session").selectOption(second.session_id);

  await expect(page.getByRole("heading", { name: second.name })).toBeVisible();
  await page.locator("#quality").scrollIntoViewIfNeeded();
  await expect(page.getByText("42%")).toBeVisible();
});

test("restores the selected session from the URL and persists later choices", async ({ page }) => {
  const first = session("QUALITY_CALCULATED", "session_first", "高架主路");
  const second = session("QUALITY_CALCULATED", "session_second", "城市路口");
  const value = dashboard([first, second]);
  await page.route("**/api/v1/dashboard", (route) => fulfillJson(route, value));

  await page.goto("/?session=session_second");
  await expect(page.getByLabel("当前 Session")).toHaveValue(second.session_id);
  await expect(page.getByRole("heading", { name: second.name })).toBeVisible();

  await page.getByLabel("当前 Session").selectOption(first.session_id);
  await expect(page).toHaveURL(/\?session=session_first$/);
  await page.reload();

  await expect(page.getByLabel("当前 Session")).toHaveValue(first.session_id);
  await expect(page.getByRole("heading", { name: first.name })).toBeVisible();
});

test("renders every scene with available thumbnails and direct CVAT links", async ({ page }) => {
  const current = session("WAITING_FOR_HUMAN_REVIEW", "session_scenes", "多场景道路样本");
  current.scenes[0].thumbnail_path = "data/scenes/session_scenes/scene_001.jpg";
  current.scenes.push({
    ...current.scenes[0],
    scene_id: "session_scenes_scene_002",
    start_seconds: 15,
    end_seconds: 30,
    thumbnail_path: null,
    cvat_task_id: 702,
    cvat_job_ids: [7021, 7022],
  });
  const value = dashboard([current]);
  value.summary.scene_count = 2;
  value.summary.task_count = 2;
  await page.route("**/api/v1/dashboard", (route) => fulfillJson(route, value));

  await page.goto("/?session=session_scenes");

  await expect(page.locator(".scene-item")).toHaveCount(2);
  await expect(page.getByAltText("session_scenes_scene_001 缩略图")).toHaveAttribute(
    "src",
    "/api/v1/scenes/session_scenes_scene_001/thumbnail",
  );
  await expect(page.getByRole("link", { name: /session_scenes_scene_001 Task 701/ })).toHaveAttribute(
    "href",
    "http://localhost:8080/tasks/701",
  );
  await expect(page.getByRole("link", { name: /session_scenes_scene_002 Job 7022/ })).toHaveAttribute(
    "href",
    "http://localhost:8080/tasks/702/jobs/7022",
  );
});

test("keeps release integrity failures visible after dashboard refresh", async ({ page }) => {
  const released = session("RELEASED", "session_tampered", "待核验 Release");
  const value = dashboard([released]);
  value.releases = {
    [released.session_id]: {
      release_id: `${released.session_id}-v1.0.0`,
      version: "1.0.0",
      manifest_sha256: "c".repeat(64),
      file_count: 6,
      verified: false,
      checked_at: "2026-09-04T04:12:00Z",
      errors: ["annotations.coco.json hash mismatch"],
    },
  };
  await page.route("**/api/v1/dashboard", (route) => fulfillJson(route, value));

  await page.goto("/?session=session_tampered");
  await expect(page.getByText("Release 完整性验证失败")).toBeVisible();
  await expect(page.getByText("annotations.coco.json hash mismatch").first()).toBeVisible();
  await page.locator("#quality").scrollIntoViewIfNeeded();
  await expect(page.getByText("FAILED")).toBeVisible();

  await page.reload();
  await expect(page.getByText("Release 完整性验证失败")).toBeVisible();
  await expect(page.getByText("annotations.coco.json hash mismatch").first()).toBeVisible();
});

test("recovers dashboard loading with the retry control", async ({ page }) => {
  let shouldFail = true;
  await page.route("**/api/v1/dashboard", async (route) => {
    if (shouldFail) {
      await fulfillJson(
        route,
        { error: { code: "OFFLINE", message: "service unavailable", retryable: true } },
        503,
      );
      return;
    }
    await fulfillJson(route, dashboard());
  });

  await page.goto("/");
  await expect(page.locator(".error-banner")).toContainText("service unavailable");
  shouldFail = false;
  await page.getByRole("button", { name: "重试上一步" }).click();

  await expect(page.getByRole("heading", { name: "还没有道路 Session" })).toBeVisible();
  await expect(page.locator(".error-banner")).toHaveCount(0);
});
