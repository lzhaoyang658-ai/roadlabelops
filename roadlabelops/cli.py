from __future__ import annotations

import argparse
import json
import sys

from .ingest import ingest_video
from .models import Stage
from .runtime import WorkflowRuntime
from .settings import build_cvat_adapter, build_detection_runner, load_settings
from .storage import LocalStore
from .tools.environment import check_environment


def _print(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="roadlabelops")
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor")
    doctor.add_argument(
        "--demo-only",
        action="store_true",
        help="accept Demo readiness without CVAT, FFmpeg, or a real detector",
    )
    sub.add_parser("demo")
    sub.add_parser("list")
    ingest = sub.add_parser("ingest")
    ingest.add_argument("video")
    ingest.add_argument("--scene-seconds", type=int, default=15, choices=range(10, 31))
    ingest.add_argument("--allow-long-video", action="store_true")
    run = sub.add_parser("run-to-review")
    run.add_argument("video")
    run.add_argument("--scene-seconds", type=int, default=15, choices=range(10, 31))
    run.add_argument("--allow-long-video", action="store_true")
    advance = sub.add_parser("advance")
    advance.add_argument("session_id")
    advance.add_argument(
        "action",
        choices=[
            "create_tasks",
            "preannotate",
            "request_review",
            "complete_review",
            "calculate_quality",
            "release",
        ],
    )
    advance.add_argument("--version", default="1.0.0")
    advance.add_argument(
        "--approve",
        action="store_true",
        help="approve the pending action when recovery requires explicit permission",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_settings()
    cvat = build_cvat_adapter(settings)
    if args.command == "doctor":
        result = check_environment(
            settings.roadlabelops_data_dir,
            settings.cvat_host,
            settings.cvat_credentials_configured,
            settings.detection_provider,
            settings.detection_model_path,
            runtime_root=settings.roadlabelops_runtime_dir,
            cvat_health_check=cvat.health if cvat else None,
        )
        result.data["requested_readiness"] = "demo" if args.demo_only else "real"
        _print(result.data)
        ready_key = "demo_ready" if args.demo_only else "real_flow_ready"
        return 0 if result.data[ready_key] else 2
    store = LocalStore(
        settings.roadlabelops_data_dir,
        settings.roadlabelops_runtime_dir,
    )
    runtime = WorkflowRuntime(store, cvat=cvat, detector=build_detection_runner(settings))
    if args.command == "demo":
        _print(runtime.create_demo().to_dict())
        return 0
    if args.command == "list":
        _print([session.to_dict() for session in store.list_sessions()])
        return 0
    if args.command == "ingest":
        result = ingest_video(
            store,
            args.video,
            scene_seconds=args.scene_seconds,
            max_duration_seconds=settings.roadlabelops_max_video_duration_seconds,
            allow_long_video=args.allow_long_video,
            absolute_max_duration_seconds=(
                settings.roadlabelops_absolute_max_video_duration_seconds
            ),
            max_scene_count=settings.roadlabelops_max_scene_count,
            max_split_output_bytes=settings.roadlabelops_max_split_output_bytes,
            max_width=settings.roadlabelops_max_video_width,
            max_height=settings.roadlabelops_max_video_height,
            max_fps=settings.roadlabelops_max_video_fps,
        )
        _print(result.data if result.ok else result.error)
        return 0 if result.ok else 2
    if args.command == "run-to-review":
        ingested = ingest_video(
            store,
            args.video,
            scene_seconds=args.scene_seconds,
            max_duration_seconds=settings.roadlabelops_max_video_duration_seconds,
            allow_long_video=args.allow_long_video,
            absolute_max_duration_seconds=(
                settings.roadlabelops_absolute_max_video_duration_seconds
            ),
            max_scene_count=settings.roadlabelops_max_scene_count,
            max_split_output_bytes=settings.roadlabelops_max_split_output_bytes,
            max_width=settings.roadlabelops_max_video_width,
            max_height=settings.roadlabelops_max_video_height,
            max_fps=settings.roadlabelops_max_video_fps,
        )
        if not ingested.ok:
            _print(ingested.error)
            return 2
        session_id = ingested.data["session"]["session_id"]
        session = store.get_session(session_id)
        action_by_stage = {
            Stage.SLICED: "create_tasks",
            Stage.TASKS_CREATED: "preannotate",
            Stage.PREANNOTATED: "request_review",
        }
        while session.status in action_by_stage:
            advanced = runtime.advance(session_id, action_by_stage[session.status])
            if not advanced.ok:
                _print(advanced.error)
                return 2
            session = store.get_session(session_id)
        _print(session.to_dict())
        successful_stages = {
            Stage.WAITING_FOR_HUMAN_REVIEW,
            Stage.REVIEW_COMPLETED,
            Stage.QUALITY_CALCULATED,
            Stage.RELEASED,
        }
        return 0 if session.status in successful_stages else 2
    if args.command == "advance":
        result = runtime.advance(
            args.session_id,
            args.action,
            version=args.version,
            approved=args.approve,
        )
        _print(result.data if result.ok else result.error)
        return 0 if result.ok else 2
    return 1


if __name__ == "__main__":
    sys.exit(main())
