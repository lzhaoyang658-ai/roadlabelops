const os = require("node:os");
const path = require("node:path");
const fs = require("node:fs");
const { chromium } = require("playwright");

const targetUrl = process.env.TARGET_URL || "http://127.0.0.1:3100";
const artifactDir = process.env.PW_ARTIFACT_DIR || path.join(os.tmpdir(), "roadlabelops-browser");
fs.mkdirSync(artifactDir, { recursive: true });

(async () => {
  const browser = await chromium.launch({ headless: process.env.PW_HEADLESS === "true" });
  const context = await browser.newContext({ viewport: { width: 1440, height: 960 } });
  const page = await context.newPage();
  const consoleProblems = [];
  const failedRequests = [];
  const badResponses = [];

  page.on("console", (message) => {
    if (["error", "warning"].includes(message.type())) {
      consoleProblems.push(`${message.type()}: ${message.text()}`);
    }
  });
  page.on("pageerror", (error) => consoleProblems.push(`pageerror: ${error.message}`));
  page.on("requestfailed", (request) => {
    if (request.failure()?.errorText !== "net::ERR_ABORTED") {
      failedRequests.push(`${request.method()} ${request.url()} ${request.failure()?.errorText}`);
    }
  });
  page.on("response", (response) => {
    if (response.status() >= 400) badResponses.push(`${response.status()} ${response.url()}`);
  });

  try {
    await page.goto(targetUrl, { waitUntil: "domcontentloaded" });
    await page.getByRole("heading", { name: /让每一帧道路/ }).waitFor();
    await page.getByRole("navigation", { name: "主导航" }).waitFor();
    await page.waitForFunction(() => (
      Array.from(document.querySelectorAll(".nav-shell, .hero-line, .hero-visual, .hero-support, .hero-actions"))
        .flatMap((element) => element.getAnimations())
        .every((animation) => animation.playState === "finished")
    ));
    await page.screenshot({ path: path.join(artifactDir, "desktop-before.png"), fullPage: true });

    const overflow = await page.evaluate(() => ({
      scrollWidth: document.documentElement.scrollWidth,
      clientWidth: document.documentElement.clientWidth,
      mainOverflowX: getComputedStyle(document.querySelector("main")).overflowX,
    }));

    await page.getByRole("button", { name: "使用演示数据" }).first().click();
    await page.getByRole("heading", { name: "上海高架夜间道路采样" }).waitFor();

    const actions = [
      "创建 CVAT 任务",
      "生成预标注",
      "提交人工审核",
      "同步 CVAT 验收结果",
      "计算质量指标",
      "发布 COCO / YOLO Release",
    ];
    for (const name of actions) {
      const button = page.getByRole("button", { name });
      await button.waitFor();
      await button.click();
    }
    await page.getByText("Release 文件与哈希验证通过").waitFor();
    await page.getByText("PASSED").waitFor();
    await page.screenshot({ path: path.join(artifactDir, "desktop-released.png"), fullPage: true });

    await page.setViewportSize({ width: 390, height: 844 });
    await page.reload({ waitUntil: "domcontentloaded" });
    await page.getByRole("heading", { name: /让每一帧道路/ }).waitFor();
    await page.waitForFunction(() => (
      Array.from(document.querySelectorAll(".nav-shell, .hero-line, .hero-visual, .hero-support, .hero-actions"))
        .flatMap((element) => element.getAnimations())
        .every((animation) => animation.playState === "finished")
    ));
    const mobileOverflow = await page.evaluate(() => ({
      scrollWidth: document.documentElement.scrollWidth,
      clientWidth: document.documentElement.clientWidth,
    }));
    await page.screenshot({ path: path.join(artifactDir, "mobile-released.png"), fullPage: true });

    const navigationTiming = await page.evaluate(() => {
      const entry = performance.getEntriesByType("navigation")[0];
      if (!entry) return null;
      return {
        domContentLoaded: Math.round(entry.domContentLoadedEventEnd),
        load: Math.round(entry.loadEventEnd),
      };
    });

    console.log(JSON.stringify({
      title: await page.title(),
      overflow,
      mobileOverflow,
      navigationTiming,
      consoleProblems,
      failedRequests,
      badResponses,
      artifacts: [
        path.join(artifactDir, "desktop-before.png"),
        path.join(artifactDir, "desktop-released.png"),
        path.join(artifactDir, "mobile-released.png"),
      ],
    }, null, 2));

    if (consoleProblems.length || failedRequests.length || badResponses.length) process.exitCode = 2;
    if (overflow.scrollWidth > overflow.clientWidth || mobileOverflow.scrollWidth > mobileOverflow.clientWidth) process.exitCode = 3;
  } finally {
    await browser.close();
  }
})();
