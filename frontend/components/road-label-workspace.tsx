"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Image from "next/image";
import {
  ArrowLeft,
  ArrowRight,
  ArrowSquareOut,
  ArrowUpRight,
  ArrowsClockwise,
  Check,
  Database,
  FileVideo,
  Gauge,
  GitBranch,
  HardDrives,
  Lightning,
  Play,
  ShieldCheck,
  SpinnerGap,
  UploadSimple,
  VideoCamera,
  Warning,
} from "@phosphor-icons/react";
import {
  advanceSession,
  ApiError,
  createDemo,
  cvatJobUrl,
  cvatProjectUrl,
  getDashboard,
  thumbnailUrl,
  uploadVideo,
  verifySessionRelease,
} from "@/lib/api";
import type {
  Activity,
  Dashboard,
  Quality,
  ReleaseReceipt,
  Session,
  Stage,
} from "@/lib/types";
import { nextSessionAction, stageCopy, stageProgress } from "@/lib/workflow";

const emptyDashboard: Dashboard = {
  summary: { session_count: 0, scene_count: 0, task_count: 0, ready_for_review: 0 },
  sessions: [],
  quality: null,
  qualities: {},
  releases: {},
  activity: [],
};

const workflowWords = [
  "VIDEO PROBE",
  "SCENE SPLIT",
  "CVAT TASK",
  "AUTO LABEL",
  "HUMAN REVIEW",
  "QUALITY",
  "COCO RELEASE",
];

const sourceUrl = process.env.NEXT_PUBLIC_SOURCE_URL
  ?? "https://github.com/lzhaoyang658-ai/roadlabelops";

type RetryIntent =
  | { kind: "load" }
  | { kind: "demo" }
  | { kind: "upload" }
  | { kind: "advance"; sessionId: string }
  | { kind: "verify"; sessionId: string };

function formatTime(value: string): string {
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

function percentage(value: number | null | undefined): string {
  return value == null ? "不可计算" : `${Math.round(value * 100)}%`;
}

function stageTone(stage: Stage): string {
  if (stage === "RELEASED") return "status-released";
  if (stage === "WAITING_FOR_HUMAN_REVIEW" || stage === "WAITING_FOR_PERMISSION") {
    return "status-waiting";
  }
  if (stage.startsWith("FAILED")) return "status-failed";
  return "status-running";
}

function persistSessionSelection(sessionId: string | null): void {
  if (typeof window === "undefined") return;
  const url = new URL(window.location.href);
  if (sessionId) {
    url.searchParams.set("session", sessionId);
  } else {
    url.searchParams.delete("session");
  }
  window.history.replaceState(null, "", `${url.pathname}${url.search}${url.hash}`);
}

function releaseErrors(receipt: ReleaseReceipt | null): string[] {
  if (!receipt || receipt.verified) return [];
  return receipt.errors?.length
    ? receipt.errors
    : ["回执存在，但未通过完整性校验。"];
}

function RoadArtwork({ label, variant = "night" }: { label: string; variant?: "night" | "review" }) {
  return (
    <div className={`road-artwork road-artwork-${variant}`} role="img" aria-label={label}>
      <span className="road-sky" />
      <span className="road-glow road-glow-left" />
      <span className="road-glow road-glow-right" />
      <span className="road-city" />
      <span className="road-plane" />
      <span className="road-lane road-lane-left" />
      <span className="road-lane road-lane-right" />
      <span className="road-car road-car-left" />
      <span className="road-car road-car-right" />
    </div>
  );
}

function ActivityCarousel({ activities }: { activities: Activity[] }) {
  const [index, setIndex] = useState(0);
  const visible = activities.slice(0, 5);
  const safeIndex = visible.length ? index % visible.length : 0;
  const current = visible[safeIndex] ?? null;

  if (!current) {
    return <p className="empty-copy">创建首个 Session 后，工作流证据会按时间写入这里。</p>;
  }

  return (
    <div className="evidence-carousel" aria-live="polite">
      <div className="evidence-rail" aria-hidden="true">
        {visible.map((activity, itemIndex) => (
          <span
            key={`${activity.timestamp}-${itemIndex}`}
            className={itemIndex === safeIndex ? "evidence-dot active" : "evidence-dot"}
          />
        ))}
      </div>
      <p className="evidence-event">{current.event.replaceAll(".", " / ")}</p>
      <p className="evidence-detail">
        {stageCopy[current.stage]?.label ?? current.stage}
        <span>{formatTime(current.timestamp)}</span>
      </p>
      <div className="carousel-controls">
        <button
          type="button"
          aria-label="上一条工作流证据"
          onClick={() => setIndex((value) => (value - 1 + visible.length) % visible.length)}
        >
          <ArrowLeft size={18} />
        </button>
        <button
          type="button"
          aria-label="下一条工作流证据"
          onClick={() => setIndex((value) => (value + 1) % visible.length)}
        >
          <ArrowRight size={18} />
        </button>
      </div>
    </div>
  );
}

function SessionPanel({
  session,
  receipt,
  busy,
  releaseVersion,
  onReleaseVersionChange,
  onAdvance,
  onVerify,
}: {
  session: Session;
  receipt: ReleaseReceipt | null;
  busy: boolean;
  releaseVersion: string;
  onReleaseVersionChange: (value: string) => void;
  onAdvance: (session: Session, approved: boolean) => void;
  onVerify: (session: Session) => void;
}) {
  const action = nextSessionAction(session);
  const integrityErrors = releaseErrors(receipt);
  const firstReviewScene = session.scenes.find((scene) => scene.cvat_task_id);
  const reviewUrl = firstReviewScene?.cvat_task_id
    ? cvatJobUrl(firstReviewScene.cvat_task_id, firstReviewScene.cvat_job_ids[0])
    : firstReviewScene?.cvat_project_id
      ? cvatProjectUrl(firstReviewScene.cvat_project_id)
      : null;

  return (
    <article className="session-panel group">
      <div className="session-media">
        <RoadArtwork label={`${session.name} 道路场景预览`} />
        <div className="media-shade" />
        {session.demo ? <span className="demo-marker">演示数据</span> : null}
      </div>
      <div className="session-body">
        <div className="session-heading">
          <div>
            <p className={`status-line ${stageTone(session.status)}`}>
              <span /> {stageCopy[session.status].label}
            </p>
            <h3>{session.name}</h3>
          </div>
          <span className="session-progress">{stageProgress(session.status)}%</span>
        </div>
        <div className="progress-track" aria-label={`工作流完成 ${stageProgress(session.status)}%`}>
          <span style={{ width: `${stageProgress(session.status)}%` }} />
        </div>
        <p className="session-detail">{stageCopy[session.status].detail}</p>
        {session.last_error ? (
          <div className="session-error" role="status">
            <Warning size={18} weight="fill" />
            <span>{session.last_error.message}</span>
          </div>
        ) : null}
        <dl className="session-metrics">
          <div><dt>场景</dt><dd>{session.scenes.length}</dd></div>
          <div><dt>时长</dt><dd>{Math.round(session.duration_seconds)}s</dd></div>
          <div><dt>画面</dt><dd>{session.width}×{session.height}</dd></div>
        </dl>
        {session.scenes.length ? (
          <section className="scene-register" aria-labelledby={`scenes-${session.session_id}`}>
            <div className="scene-register-heading">
              <h4 id={`scenes-${session.session_id}`}>Scene 清单</h4>
              <span>{session.scenes.length} 个切片</span>
            </div>
            <ol className="scene-list">
              {session.scenes.map((scene, index) => (
                <li className="scene-item group" key={scene.scene_id}>
                  <div className="scene-thumbnail">
                    {scene.thumbnail_path ? (
                      <Image
                        unoptimized
                        src={thumbnailUrl(scene.scene_id)}
                        alt={`${scene.scene_id} 缩略图`}
                        width={320}
                        height={180}
                      />
                    ) : (
                      <RoadArtwork label={`${scene.scene_id} 暂无缩略图`} variant="review" />
                    )}
                    <span>{String(index + 1).padStart(2, "0")}</span>
                  </div>
                  <div className="scene-record">
                    <div>
                      <strong>{scene.scene_id}</strong>
                      <span>{scene.start_seconds.toFixed(1)}s — {scene.end_seconds.toFixed(1)}s</span>
                    </div>
                    <div className="scene-links">
                      {scene.cvat_task_id ? (
                        <a
                          href={cvatJobUrl(scene.cvat_task_id)}
                          target="_blank"
                          rel="noreferrer"
                          aria-label={`在 CVAT 打开 ${scene.scene_id} Task ${scene.cvat_task_id}`}
                        >
                          Task #{scene.cvat_task_id} <ArrowSquareOut size={13} />
                        </a>
                      ) : (
                        <span>尚未创建 Task</span>
                      )}
                      {scene.cvat_task_id
                        ? scene.cvat_job_ids.map((jobId) => (
                            <a
                              href={cvatJobUrl(scene.cvat_task_id!, jobId)}
                              target="_blank"
                              rel="noreferrer"
                              key={jobId}
                              aria-label={`在 CVAT 打开 ${scene.scene_id} Job ${jobId}`}
                            >
                              Job #{jobId} <ArrowSquareOut size={13} />
                            </a>
                          ))
                        : null}
                    </div>
                  </div>
                </li>
              ))}
            </ol>
          </section>
        ) : null}
        {action ? (
          <div className="session-actions">
            {!session.demo && session.status === "WAITING_FOR_HUMAN_REVIEW" && reviewUrl ? (
              <a className="secondary-button full" href={reviewUrl} target="_blank" rel="noreferrer">
                打开 CVAT 验收 <ArrowSquareOut size={18} />
              </a>
            ) : null}
            {action.action === "release" ? (
              <label className="release-version">
                <span>Release 版本</span>
                <input
                  value={releaseVersion}
                  onChange={(event) => onReleaseVersionChange(event.target.value)}
                  inputMode="text"
                  pattern="\d+\.\d+\.\d+"
                  aria-label="Release 版本"
                />
              </label>
            ) : null}
            <button
              className="primary-button full"
              type="button"
              onClick={() => onAdvance(session, action.needsApproval)}
              disabled={busy}
              aria-busy={busy}
            >
              {busy ? <SpinnerGap className="spin" size={19} /> : <Lightning weight="fill" size={19} />}
              {busy ? "正在写入工作流" : action.label}
            </button>
          </div>
        ) : (
          <div className="release-verification">
            {receipt?.verified ? (
              <div className="release-complete"><Check weight="bold" /> Release 文件与哈希验证通过</div>
            ) : receipt ? (
              <div className="release-failed" role="alert">
                <span><Warning weight="fill" /> Release 完整性验证失败</span>
                <ul>{integrityErrors.map((message) => <li key={message}>{message}</li>)}</ul>
              </div>
            ) : (
              <div className="release-failed" role="alert">
                <span><Warning weight="fill" /> Release 完整性回执缺失</span>
                <p>已发布的 Session 必须有可验证回执，请重新验证。</p>
              </div>
            )}
            <button
              className="secondary-button full"
              type="button"
              onClick={() => onVerify(session)}
              disabled={busy}
            >
              {busy ? <SpinnerGap className="spin" size={18} /> : <ShieldCheck size={18} />}
              重新验证 Release
            </button>
          </div>
        )}
      </div>
    </article>
  );
}

function QualitySummary({ quality }: { quality: Quality | null }) {
  return (
    <>
      <div className="quality-mini-grid">
        <div><span>Precision</span><strong>{percentage(quality?.precision)}</strong></div>
        <div><span>Recall</span><strong>{percentage(quality?.recall)}</strong></div>
        <div><span>Clean frames</span><strong>{percentage(quality?.clean_frame_rate)}</strong></div>
        <div><span>First pass</span><strong>{percentage(quality?.first_pass_acceptance_rate)}</strong></div>
      </div>
      {quality?.first_pass_acceptance_rate == null && quality?.first_pass_acceptance_reason ? (
        <p className="quality-unavailable-note">一次审核通过率不可计算：{quality.first_pass_acceptance_reason}</p>
      ) : null}
    </>
  );
}

export function RoadLabelWorkspace() {
  const root = useRef<HTMLElement>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const [dashboard, setDashboard] = useState<Dashboard>(emptyDashboard);
  const [loading, setLoading] = useState(true);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [retryIntent, setRetryIntent] = useState<RetryIntent | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(() => {
    if (typeof window === "undefined") return null;
    return new URLSearchParams(window.location.search).get("session");
  });
  const selectedSessionIdRef = useRef(selectedSessionId);
  const [sceneSeconds, setSceneSeconds] = useState(15);
  const [releaseVersions, setReleaseVersions] = useState<Record<string, string>>({});

  const activeSession = dashboard.sessions.find(
    (session) => session.session_id === selectedSessionId,
  ) ?? dashboard.sessions[0] ?? null;
  const activeQuality = activeSession
    ? dashboard.qualities?.[activeSession.session_id]
      ?? (activeSession.session_id === dashboard.sessions[0]?.session_id ? dashboard.quality : null)
    : null;
  const activeReceipt = activeSession
    ? dashboard.releases?.[activeSession.session_id] ?? null
    : null;
  const activeActivities = activeSession
    ? dashboard.activity.filter((activity) => activity.session_id === activeSession.session_id)
    : dashboard.activity;

  const selectSession = useCallback((sessionId: string | null) => {
    selectedSessionIdRef.current = sessionId;
    setSelectedSessionId(sessionId);
    persistSessionSelection(sessionId);
  }, []);

  const load = useCallback(async (signal?: AbortSignal, preserveError = false) => {
    try {
      const data = await getDashboard(signal);
      const requestedSessionId = selectedSessionIdRef.current;
      const resolvedSessionId = data.sessions.some(
        (session) => session.session_id === requestedSessionId,
      )
        ? requestedSessionId
        : data.sessions[0]?.session_id ?? null;
      if (resolvedSessionId !== requestedSessionId) {
        selectedSessionIdRef.current = resolvedSessionId;
        setSelectedSessionId(resolvedSessionId);
      }
      persistSessionSelection(resolvedSessionId);
      setDashboard(data);
      if (!preserveError) {
        setError(null);
        setRetryIntent(null);
      }
    } catch (caught) {
      if (caught instanceof DOMException && caught.name === "AbortError") return;
      setError(
        caught instanceof ApiError
          ? caught
          : new ApiError("无法连接 RoadLabelOps 服务", "OFFLINE", true),
      );
      setRetryIntent({ kind: "load" });
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    const initialLoad = async () => {
      await load(controller.signal);
    };
    void initialLoad();
    return () => controller.abort();
  }, [load]);

  useEffect(() => {
    if (!activeSession || ![
      "WAITING_FOR_HUMAN_REVIEW",
      "WAITING_FOR_PERMISSION",
      "FAILED_RETRYABLE",
    ].includes(activeSession.status)) return;
    const timer = window.setInterval(() => void load(undefined, true), 15_000);
    return () => window.clearInterval(timer);
  }, [activeSession, load]);

  useEffect(() => {
    const scope = root.current;
    if (!scope || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

    const animations: Animation[] = [];
    const play = (
      element: HTMLElement | null,
      keyframes: Keyframe[],
      options: KeyframeAnimationOptions,
    ): Animation | null => {
      if (!element || typeof element.animate !== "function") return null;
      const animation = element.animate(keyframes, { ...options, fill: "backwards" });
      animations.push(animation);
      return animation;
    };

    const nav = scope.querySelector<HTMLElement>(".nav-shell");
    const heroSection = scope.querySelector<HTMLElement>(".hero-section");
    const heroVisual = scope.querySelector<HTMLElement>(".hero-visual");
    const heroLines = Array.from(scope.querySelectorAll<HTMLElement>(".hero-line"));
    const heroSupportingElements = Array.from(
      scope.querySelectorAll<HTMLElement>(".hero-support, .hero-actions"),
    );
    const qualityVisuals = Array.from(
      scope.querySelectorAll<HTMLElement>(".quality-visual"),
    );

    play(
      nav,
      [
        { opacity: 0, transform: "translate(-50%, -24px)" },
        { opacity: 1, transform: "translate(-50%, 0)" },
      ],
      { duration: 800, easing: "cubic-bezier(.22, 1, .36, 1)" },
    );
    heroLines.forEach((element, index) => {
      play(
        element,
        [
          { transform: "translateY(108%) rotate(2deg)" },
          { transform: "translateY(0) rotate(0deg)" },
        ],
        {
          duration: 1050,
          delay: index * 120,
          easing: "cubic-bezier(.16, 1, .3, 1)",
        },
      );
    });
    heroSupportingElements.forEach((element, index) => {
      play(
        element,
        [
          { opacity: 0, transform: "translateY(26px)" },
          { opacity: 1, transform: "translateY(0)" },
        ],
        {
          duration: 800,
          delay: 350 + index * 120,
          easing: "cubic-bezier(.22, 1, .36, 1)",
        },
      );
    });

    let heroEntranceComplete = heroVisual === null;
    const heroEntrance = play(
      heroVisual,
      [
        { opacity: 0.35, transform: "scale(.8) rotate(4deg)" },
        { opacity: 1, transform: "scale(1) rotate(0deg)" },
      ],
      { duration: 1350, easing: "cubic-bezier(.22, 1, .36, 1)" },
    );

    let frameId: number | null = null;
    const clamp = (value: number): number => Math.min(1, Math.max(0, value));
    const mix = (start: number, end: number, progress: number): number => (
      start + (end - start) * progress
    );
    const renderScrollEffects = () => {
      frameId = null;
      const viewportHeight = window.innerHeight || document.documentElement.clientHeight;

      if (heroSection && heroVisual && heroEntranceComplete) {
        const rect = heroSection.getBoundingClientRect();
        const start = viewportHeight * 0.5;
        const marker = rect.top + rect.height * 0.55;
        const end = rect.height * -0.45;
        const progress = clamp((start - marker) / Math.max(1, start - end));
        heroVisual.style.transform = `scale(${mix(1, 0.88, progress)})`;
        heroVisual.style.opacity = String(mix(1, 0.2, progress));
        heroVisual.style.filter = `brightness(${mix(1, 0.45, progress)})`;
      }

      qualityVisuals.forEach((element) => {
        const rect = element.getBoundingClientRect();
        const enterStart = viewportHeight * 0.88;
        const enterEnd = viewportHeight * 0.55 - rect.height * 0.5;
        const enterProgress = clamp(
          (enterStart - rect.top) / Math.max(1, enterStart - enterEnd),
        );
        const exitStart = viewportHeight * 0.28;
        const exitProgress = clamp((exitStart - rect.bottom) / Math.max(1, exitStart));
        const enteredOpacity = mix(0.35, 1, enterProgress);
        element.style.transform = `scale(${mix(0.8, 1, enterProgress)})`;
        element.style.opacity = String(mix(enteredOpacity, 0.2, exitProgress));
        element.style.filter = `brightness(${mix(1, 0.45, exitProgress)})`;
      });
    };
    const scheduleScrollEffects = () => {
      if (frameId === null) frameId = window.requestAnimationFrame(renderScrollEffects);
    };

    const finishHeroEntrance = () => {
      heroEntranceComplete = true;
      scheduleScrollEffects();
    };
    if (heroEntrance) {
      heroEntrance.addEventListener("finish", finishHeroEntrance, { once: true });
    } else {
      finishHeroEntrance();
    }

    const observedElements = [heroVisual, ...qualityVisuals].filter(
      (element): element is HTMLElement => element !== null,
    );
    const visibilityObserver = typeof IntersectionObserver === "undefined"
      ? null
      : new IntersectionObserver(
        (entries) => {
          entries.forEach((entry) => {
            const element = entry.target as HTMLElement;
            element.style.willChange = entry.isIntersecting
              ? "transform, opacity, filter"
              : "auto";
          });
          scheduleScrollEffects();
        },
        { rootMargin: "20% 0px" },
      );
    observedElements.forEach((element) => visibilityObserver?.observe(element));

    window.addEventListener("scroll", scheduleScrollEffects, { passive: true });
    window.addEventListener("resize", scheduleScrollEffects);
    scheduleScrollEffects();

    return () => {
      window.removeEventListener("scroll", scheduleScrollEffects);
      window.removeEventListener("resize", scheduleScrollEffects);
      if (frameId !== null) window.cancelAnimationFrame(frameId);
      visibilityObserver?.disconnect();
      if (heroEntrance) heroEntrance.removeEventListener("finish", finishHeroEntrance);
      animations.forEach((animation) => animation.cancel());
      observedElements.forEach((element) => {
        element.style.removeProperty("transform");
        element.style.removeProperty("opacity");
        element.style.removeProperty("filter");
        element.style.removeProperty("will-change");
      });
    };
  }, []);

  const createFirstDemo = async () => {
    setBusyId("new");
    setError(null);
    try {
      const result = await createDemo();
      selectSession(result.session.session_id);
      await load();
      document.querySelector("#workspace")?.scrollIntoView({ behavior: "smooth" });
    } catch (caught) {
      setError(caught instanceof ApiError ? caught : new ApiError("无法创建演示 Session"));
      setRetryIntent({ kind: "demo" });
    } finally {
      setBusyId(null);
    }
  };

  const importRealVideo = async () => {
    if (!selectedFile) {
      fileInput.current?.click();
      return;
    }
    setBusyId("upload");
    setError(null);
    try {
      const result = await uploadVideo(selectedFile, sceneSeconds);
      selectSession(result.session.session_id);
      setSelectedFile(null);
      if (fileInput.current) fileInput.current.value = "";
      await load();
      document.querySelector("#workspace")?.scrollIntoView({ behavior: "smooth" });
    } catch (caught) {
      setError(caught instanceof ApiError ? caught : new ApiError("真实视频导入失败"));
      setRetryIntent({ kind: "upload" });
    } finally {
      setBusyId(null);
    }
  };

  const advance = async (session: Session, approved = false) => {
    const action = nextSessionAction(session);
    if (!action) return;
    const releaseVersion = releaseVersions[session.session_id] ?? "1.0.0";
    if (action.action === "release" && !/^\d+\.\d+\.\d+$/.test(releaseVersion)) {
      setError(new ApiError("Release 版本必须使用 x.y.z 格式", "INVALID_VERSION"));
      setRetryIntent(null);
      return;
    }
    setBusyId(session.session_id);
    setError(null);
    try {
      const result = await advanceSession(
        session.session_id,
        action.action,
        releaseVersion,
        approved,
      );
      selectSession(result.session.session_id);
      await load();
    } catch (caught) {
      setError(caught instanceof ApiError ? caught : new ApiError("工作流推进失败"));
      setRetryIntent({ kind: "advance", sessionId: session.session_id });
      await load(undefined, true);
    } finally {
      setBusyId(null);
    }
  };

  const verifyRelease = async (session: Session) => {
    setBusyId(session.session_id);
    setError(null);
    try {
      await verifySessionRelease(session.session_id);
      await load();
    } catch (caught) {
      setError(caught instanceof ApiError ? caught : new ApiError("Release 验证失败"));
      setRetryIntent({ kind: "verify", sessionId: session.session_id });
      await load(undefined, true);
    } finally {
      setBusyId(null);
    }
  };

  const retryLastOperation = () => {
    if (!retryIntent || retryIntent.kind === "load") {
      void load();
      return;
    }
    if (retryIntent.kind === "demo") {
      void createFirstDemo();
      return;
    }
    if (retryIntent.kind === "upload") {
      void importRealVideo();
      return;
    }
    const session = dashboard.sessions.find((item) => item.session_id === retryIntent.sessionId);
    if (!session) return;
    if (retryIntent.kind === "verify") {
      void verifyRelease(session);
      return;
    }
    void advance(session, session.status === "WAITING_FOR_PERMISSION");
  };

  const refresh = async () => {
    setBusyId("refresh");
    await load();
    setBusyId(null);
  };

  const counts = useMemo(
    () => [
      { label: "Sessions", value: dashboard.summary.session_count },
      { label: "Scenes", value: dashboard.summary.scene_count },
      { label: "CVAT Tasks", value: dashboard.summary.task_count },
    ],
    [dashboard.summary],
  );
  const classRows = Object.entries(activeQuality?.class_distribution ?? {}).slice(0, 5);
  const activeIntegrityErrors = releaseErrors(activeReceipt);

  return (
    <main ref={root} className="overflow-x-clip w-full max-w-full">
      <header className="nav-shell">
        <a className="brand" href="#top" aria-label="RoadLabelOps 首页">
          <span className="brand-mark"><GitBranch size={19} weight="bold" /></span>
          <span>ROADLABELOPS</span>
        </a>
        <nav aria-label="主导航">
          <a href="#workspace">工作流</a>
          <a href="#quality">质量</a>
          <a href="#evidence">证据</a>
        </nav>
        <a className="nav-action" href="#workspace">进入控制室 <ArrowUpRight size={17} /></a>
      </header>

      <section id="top" className="hero-section">
        <div className="ambient ambient-one" />
        <div className="ambient ambient-two" />
        <div className="hero-copy">
          <h1 className="hero-title">
            <span className="hero-mask"><span className="hero-line">让每一帧道路</span></span>
            <span className="hero-mask">
              <span className="hero-line">都有可信去向</span>
            </span>
          </h1>
          <p className="hero-support">
            从视频切片、CVAT 预标注到人工验收与 COCO / YOLO Release，
            每一步都有状态、证据和可恢复路径。
          </p>
          <div className="hero-actions">
            <button className="primary-button" type="button" onClick={() => fileInput.current?.click()}>
              <UploadSimple weight="bold" size={19} /> 导入真实视频
            </button>
            <button className="secondary-button" type="button" onClick={createFirstDemo} disabled={busyId === "new"}>
              {busyId === "new" ? <SpinnerGap className="spin" size={19} /> : <Play weight="fill" size={18} />}
              {busyId === "new" ? "正在创建" : "使用演示数据"}
            </button>
          </div>
        </div>
        <div className="hero-visual group">
          <RoadArtwork label="带有目标检测框的夜间道路" />
          <div className="hero-visual-wash" />
          <div className="detection-box box-one"><span>car · 0.94</span></div>
          <div className="detection-box box-two"><span>truck · 0.87</span></div>
          <div className="frame-readout"><span>FRAME 00425</span><span>TRACE 8F2A</span></div>
        </div>
      </section>

      <div className="marquee" aria-label="RoadLabelOps 工作流阶段">
        <div className="marquee-track">
          {[...workflowWords, ...workflowWords].map((word, index) => (
            <span key={`${word}-${index}`}><span className="marquee-node" />{word}</span>
          ))}
        </div>
      </div>

      <section id="workspace" className="workspace-section">
        <div className="section-heading">
          <div>
            <p className="eyebrow">当前运行面</p>
            <h2>一眼看清数据走到哪一步</h2>
          </div>
          <p>后端状态是唯一事实来源。刷新页面后，Session、Scene、CVAT 映射与质量结果都会重新恢复。</p>
        </div>

        <div className="ingest-console" aria-label="导入真实道路视频">
          <div className="ingest-copy">
            <span className="ingest-icon"><FileVideo size={25} weight="duotone" /></span>
            <div>
              <strong>真实视频入口</strong>
              <span>MP4 / MOV / M4V · 自动探测、哈希与场景切片</span>
            </div>
          </div>
          <div className="ingest-fields">
            <label className="file-picker">
              <input
                ref={fileInput}
                type="file"
                accept="video/mp4,video/quicktime,video/x-m4v,.mp4,.mov,.m4v"
                onChange={(event) => setSelectedFile(event.target.files?.[0] ?? null)}
              />
              <span>{selectedFile?.name ?? "选择道路视频"}</span>
            </label>
            <label className="scene-picker">
              <span>Scene</span>
              <select value={sceneSeconds} onChange={(event) => setSceneSeconds(Number(event.target.value))}>
                {[10, 15, 20, 30].map((value) => <option key={value} value={value}>{value} 秒</option>)}
              </select>
            </label>
            <button
              className="primary-button"
              type="button"
              onClick={importRealVideo}
              disabled={!selectedFile || busyId === "upload"}
            >
              {busyId === "upload" ? <SpinnerGap className="spin" size={19} /> : <UploadSimple size={19} weight="bold" />}
              {busyId === "upload" ? "正在切片" : "创建 Session"}
            </button>
          </div>
        </div>

        {dashboard.sessions.length ? (
          <div className="session-toolbar">
            <label>
              <span>当前 Session</span>
              <select
                value={activeSession?.session_id ?? ""}
                onChange={(event) => selectSession(event.target.value)}
              >
                {dashboard.sessions.map((session) => (
                  <option key={session.session_id} value={session.session_id}>
                    {session.name} · {stageCopy[session.status].label}
                  </option>
                ))}
              </select>
            </label>
            <button type="button" onClick={refresh} disabled={busyId === "refresh"}>
              <ArrowsClockwise className={busyId === "refresh" ? "spin" : ""} size={18} />
              刷新状态
            </button>
          </div>
        ) : null}

        {error ? (
          <div className="error-banner" role="alert">
            <Warning size={22} weight="fill" />
            <div>
              <strong>{error.retryable ? "操作可安全重试" : "工作流需要处理"}</strong>
              <span>{error.message}{error.requestId ? ` · 请求 ${error.requestId}` : ""}</span>
            </div>
            <button type="button" onClick={retryLastOperation}>
              {retryIntent ? "重试上一步" : "刷新状态"}
            </button>
          </div>
        ) : null}

        <div className="bento-grid">
          <div className="bento-main">
            {loading ? (
              <div className="loading-state"><SpinnerGap className="spin" size={28} /><span>正在读取持久化工作流</span></div>
            ) : activeSession ? (
              <SessionPanel
                session={activeSession}
                receipt={activeReceipt}
                busy={busyId === activeSession.session_id}
                releaseVersion={releaseVersions[activeSession.session_id] ?? "1.0.0"}
                onReleaseVersionChange={(value) => setReleaseVersions((current) => ({
                  ...current,
                  [activeSession.session_id]: value,
                }))}
                onAdvance={advance}
                onVerify={verifyRelease}
              />
            ) : (
              <div className="empty-state">
                <VideoCamera size={42} weight="thin" />
                <h3>还没有道路 Session</h3>
                <p>从上方导入一段 1–3 分钟道路视频，系统会自动切片并保存来源哈希。</p>
                <button className="primary-button" type="button" onClick={() => fileInput.current?.click()}>选择真实视频</button>
              </div>
            )}
          </div>
          <article className="bento-side bento-metrics">
            <div className="card-topline"><Gauge size={24} weight="duotone" /><span>生产面</span></div>
            <div className="count-row">
              {counts.map((item) => (
                <div key={item.label}>
                  <strong>{item.value.toString().padStart(2, "0")}</strong><span>{item.label}</span>
                </div>
              ))}
            </div>
            <div className="operation-rate-list" aria-label="工作流成功率">
              {[
                ["切片", dashboard.operational_metrics?.video_slice_success_rate],
                ["建任务", dashboard.operational_metrics?.task_creation_success_rate],
                ["Release 校验", dashboard.operational_metrics?.release_verification_success_rate],
              ].map(([label, metric]) => {
                const value = typeof metric === "object" && metric ? metric.value : null;
                const reason = typeof metric === "object" && metric ? metric.reason : null;
                return (
                  <div key={String(label)} title={reason ?? undefined}>
                    <span>{String(label)}</span>
                    <strong>{percentage(value)}</strong>
                    {value == null && reason ? <small>{reason}</small> : null}
                  </div>
                );
              })}
            </div>
            <p>{dashboard.summary.ready_for_review ? `${dashboard.summary.ready_for_review} 个 Session 正等待人工审核。` : "当前没有被遗忘的审核任务。"}</p>
          </article>
          <article id="evidence" className="bento-side bento-evidence">
            <div className="card-topline"><HardDrives size={24} weight="duotone" /><span>运行证据</span></div>
            <ActivityCarousel activities={activeActivities} />
          </article>
        </div>
      </section>

      <section id="quality" className="quality-section">
        <div className="quality-chapter">
          <div className="quality-copy">
            <p className="eyebrow light">质量来自审核后的对象</p>
            <h2>每个百分比，都能回到具体标注。</h2>
            <p>
              {activeSession ? `当前查看 ${activeSession.name}。` : "选择一个 Session 查看结果。"}
              空分母显示“不可计算”，Release 只接收完成审核的 Scene。
            </p>
          </div>
          <div className="quality-stack">
            <article className="quality-visual quality-card group">
              <div className="quality-card-head"><ShieldCheck size={26} weight="duotone" /><span>F1 与审核一致性</span></div>
              <strong>{percentage(activeQuality?.f1_score)}</strong>
              <div className="quality-bar"><span style={{ width: `${(activeQuality?.f1_score ?? 0) * 100}%` }} /></div>
              <QualitySummary quality={activeQuality} />
            </article>
            <article className="quality-visual quality-media group">
              <RoadArtwork label="用于质量审核的道路路口" variant="review" />
              <div className="media-shade" />
              <div className="quality-overlay">
                <div><span>预标注保留率</span><strong>{percentage(activeQuality?.retention_rate)}</strong></div>
                <div><span>人工新增率</span><strong>{percentage(activeQuality?.human_addition_rate)}</strong></div>
              </div>
            </article>
            <article className="quality-visual quality-card dark">
              <div className="quality-card-head"><Database size={26} weight="duotone" /><span>Release 完整性</span></div>
              <strong className={activeReceipt?.verified ? "receipt-passed" : activeReceipt ? "receipt-failed-text" : ""}>
                {activeReceipt?.verified ? "PASSED" : activeReceipt ? "FAILED" : "WAITING"}
              </strong>
              <p>
                {activeReceipt?.verified
                  ? `已验证 ${activeReceipt.file_count ?? 0} 个文件，Release ${activeReceipt.release_id} 可追溯且未被篡改。`
                  : activeReceipt
                    ? "验证回执已记录失败；问题修复且重新验证通过前，该 Release 不可作为可信数据集使用。"
                    : "发布后会逐文件重算 SHA-256；只有验证回执通过，这里才显示 PASSED。"}
              </p>
              {activeIntegrityErrors.length ? (
                <ul className="receipt-errors" aria-label="Release 完整性错误">
                  {activeIntegrityErrors.map((message) => <li key={message}>{message}</li>)}
                </ul>
              ) : null}
              {activeReceipt?.manifest_sha256 ? (
                <code className="receipt-hash">SHA-256 {activeReceipt.manifest_sha256}</code>
              ) : null}
              {classRows.length ? (
                <div className="class-distribution" aria-label="最终标注类别分布">
                  {classRows.map(([label, count]) => (
                    <div key={label}><span>{label}</span><strong>{count}</strong></div>
                  ))}
                </div>
              ) : null}
            </article>
          </div>
        </div>
      </section>

      <section className="action-section">
        <div className="action-copy">
          <p>道路数据不会因为一次断线失去来路。</p>
          <h2>把下一个视频，变成一份能解释的数据集。</h2>
        </div>
        <div className="action-controls">
          <button className="light-button" type="button" onClick={() => fileInput.current?.click()}>
            导入真实视频 <ArrowUpRight size={20} />
          </button>
          <button className="secondary-button" type="button" onClick={createFirstDemo} disabled={busyId === "new"}>
            使用演示数据
          </button>
          <p>真实视频可在上方直接导入；CVAT 审核仍在专用标注界面完成。</p>
        </div>
      </section>

      <footer>
        <div className="brand"><span className="brand-mark"><GitBranch size={19} weight="bold" /></span><span>ROADLABELOPS</span></div>
        <p>Deterministic workflow. Human-reviewed truth.</p>
        <div className="footer-links">
          <a href={sourceUrl} target="_blank" rel="noreferrer">查看源码 <ArrowSquareOut size={16} /></a>
          <a href="#top">返回顶部 <ArrowUpRight size={16} /></a>
        </div>
      </footer>
    </main>
  );
}
