from __future__ import annotations

import math
import secrets
from pathlib import Path
from typing import Any

from .models import Scene, Session, Stage, ToolResult, utc_now
from .storage import LocalStore
from .tools.video import (
    DEFAULT_ABSOLUTE_MAX_VIDEO_DURATION_SECONDS,
    DEFAULT_MAX_SCENE_COUNT,
    DEFAULT_MAX_SPLIT_OUTPUT_BYTES,
    probe_video,
    split_video,
    verify_split_output,
)


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _ingest_video(
    store: LocalStore,
    video_path: Path | str,
    *,
    scene_seconds: int = 15,
    frame_step: int = 5,
    max_duration_seconds: float | None = None,
    allow_long_video: bool = False,
    absolute_max_duration_seconds: float = DEFAULT_ABSOLUTE_MAX_VIDEO_DURATION_SECONDS,
    max_scene_count: int = DEFAULT_MAX_SCENE_COUNT,
    max_split_output_bytes: int = DEFAULT_MAX_SPLIT_OUTPUT_BYTES,
    max_width: int | None = None,
    max_height: int | None = None,
    max_fps: float | None = None,
    session_name: str | None = None,
    _attempt: dict[str, Any],
) -> ToolResult:
    probe = probe_video(video_path)
    if not probe.ok:
        return probe
    _attempt["probe_succeeded"] = True
    identifier = f"session_{probe.data['sha256'][:10]}"
    _attempt["session_id"] = identifier

    duration = float(probe.data["duration_seconds"])
    width = int(probe.data["width"])
    height = int(probe.data["height"])
    fps = float(probe.data["fps"])
    if scene_seconds <= 0 or frame_step <= 0:
        return ToolResult.failure(
            "INVALID_SPLIT_OPTIONS", "scene_seconds and frame_step must both be positive"
        )
    if (
        absolute_max_duration_seconds <= 0
        or max_scene_count <= 0
        or max_split_output_bytes <= 0
    ):
        return ToolResult.failure(
            "INVALID_SPLIT_OPTIONS", "Absolute duration and scene-count limits must be positive"
        )
    scene_count = max(1, math.ceil(duration / scene_seconds))
    _attempt["planned_scene_count"] = scene_count
    if duration > absolute_max_duration_seconds:
        return ToolResult.failure(
            "VIDEO_DURATION_HARD_LIMIT",
            (
                f"Video duration {duration:.1f}s exceeds the absolute "
                f"{absolute_max_duration_seconds:.1f}s safety limit"
            ),
        )
    if scene_count > max_scene_count:
        return ToolResult.failure(
            "VIDEO_SCENE_COUNT_LIMIT",
            f"Video would create {scene_count} scenes, above the {max_scene_count} scene limit",
        )
    if max_duration_seconds is not None and duration > max_duration_seconds and not allow_long_video:
        return ToolResult.failure(
            "VIDEO_TOO_LONG",
            (
                f"Video duration {duration:.1f}s exceeds the default "
                f"{max_duration_seconds:.1f}s limit; explicitly confirm long-video processing"
            ),
        )
    if (max_width is not None and width > max_width) or (
        max_height is not None and height > max_height
    ):
        configured_limit = "x".join(
            str(value) if value is not None else "unbounded"
            for value in (max_width, max_height)
        )
        return ToolResult.failure(
            "VIDEO_RESOLUTION_TOO_HIGH",
            f"Video resolution {width}x{height} exceeds the {configured_limit} limit",
        )
    if max_fps is not None and fps > max_fps:
        return ToolResult.failure(
            "VIDEO_FPS_TOO_HIGH",
            f"Video frame rate {fps:.3f} exceeds the {max_fps:.3f} fps limit",
        )

    display_name = (session_name or Path(video_path).stem).strip()[:120] or identifier
    source_sha256 = str(probe.data["sha256"])
    source_path = str(Path(video_path).resolve())
    destination = store.scenes_dir / identifier
    with store.operation_lock(f"ingest-{identifier}"):
        for existing in store.list_sessions():
            if existing.source_sha256 == source_sha256:
                existing_source_probe = probe_video(existing.source_path)
                source_recovered = (
                    not existing_source_probe.ok
                    or existing_source_probe.data.get("sha256") != source_sha256
                )
                if source_recovered:
                    existing.source_path = source_path
                    existing.updated_at = utc_now()
                    store.save_session(existing)
                return ToolResult.success(
                    {
                        "session": existing.to_dict(),
                        "existing": True,
                        "recovered": False,
                        "source_recovered": source_recovered,
                    }
                )
        try:
            collision = store.get_session(identifier)
        except FileNotFoundError:
            collision = None
        if collision is not None:
            return ToolResult.failure(
                "SESSION_ID_COLLISION",
                "A different source already uses this derived session id",
            )

        recovered = destination.exists() or destination.is_symlink()
        if not recovered:
            split = split_video(
                video_path,
                destination,
                scene_seconds,
                frame_step,
                absolute_max_duration_seconds=absolute_max_duration_seconds,
                max_scene_count=max_scene_count,
                max_output_bytes=max_split_output_bytes,
            )
            if not split.ok:
                return split

        # Even a freshly published split is checked against this ingest call's
        # original probe. This closes a source-change window between probing and
        # FFmpeg publication and gives recovery the exact same trust gate.
        split = verify_split_output(
            destination,
            source_sha256=source_sha256,
            duration_seconds=duration,
            fps=fps,
            width=width,
            height=height,
            scene_seconds=scene_seconds,
            frame_step=frame_step,
            max_scene_count=max_scene_count,
            max_output_bytes=max_split_output_bytes,
        )
        if not split.ok:
            return split

        raw_scenes = split.data.get("scenes")
        if not isinstance(raw_scenes, list) or len(raw_scenes) != scene_count:
            return ToolResult.failure(
                "SPLIT_MANIFEST_INVALID", "Split output did not return the expected scenes"
            )
        scenes: list[Scene] = []
        for expected_index, item in enumerate(raw_scenes, start=1):
            if not isinstance(item, dict) or item.get("index") != expected_index:
                return ToolResult.failure(
                    "SPLIT_MANIFEST_INVALID", "Split output scene order is invalid"
                )
            scene_video_path = Path(str(item.get("video_path", "")))
            thumbnail_path = Path(str(item.get("thumbnail_path", "")))
            recorded_hash = item.get("video_sha256")
            start_seconds = item.get("start_seconds")
            end_seconds = item.get("end_seconds")
            if (
                not _valid_sha256(recorded_hash)
                or scene_video_path.parent.resolve() != destination.resolve()
                or thumbnail_path.parent.resolve() != destination.resolve()
                or isinstance(start_seconds, bool)
                or not isinstance(start_seconds, (int, float))
                or isinstance(end_seconds, bool)
                or not isinstance(end_seconds, (int, float))
            ):
                return ToolResult.failure(
                    "SPLIT_MANIFEST_INVALID",
                    "Split output scene paths or SHA-256 lineage are invalid",
                )
            scenes.append(
                Scene(
                    scene_id=f"{identifier}_scene_{expected_index:03d}",
                    session_id=identifier,
                    start_seconds=float(start_seconds),
                    end_seconds=float(end_seconds),
                    video_path=str(scene_video_path),
                    thumbnail_path=str(thumbnail_path),
                    video_sha256=str(recorded_hash),
                )
            )

        session = Session(
            session_id=identifier,
            name=display_name,
            source_path=source_path,
            source_sha256=source_sha256,
            duration_seconds=duration,
            fps=fps,
            width=width,
            height=height,
            scene_seconds=scene_seconds,
            frame_step=frame_step,
            status=Stage.SLICED,
            scenes=scenes,
        )
        store.save_session(session)
        journal_persisted = True
        try:
            store.append_journal(
                {
                    "run_id": f"run_{session.session_id}",
                    "session_id": session.session_id,
                    "stage": session.status.value,
                    "event": "stage.succeeded",
                    "tool_name": "split_video",
                    "timestamp": utc_now(),
                    "scene_count": len(session.scenes),
                    "recovered": recovered,
                }
            )
        except Exception:
            # The Session and source lineage are already committed. Reporting
            # failure here would make the API delete the only retained source.
            journal_persisted = False
        return ToolResult.success(
            {
                "session": session.to_dict(),
                "existing": False,
                "recovered": recovered,
                "journal_persisted": journal_persisted,
            },
            side_effects=split.side_effects,
            metrics=split.metrics,
        )


def _record_ingest_completion(
    store: LocalStore,
    result: ToolResult | None,
    attempt: dict[str, Any],
    *,
    error_code: str | None = None,
) -> None:
    """Persist one terminal ingest outcome without changing the workflow result."""

    planned_scene_count = attempt.get("planned_scene_count")
    successful_scene_count = 0
    session_id = attempt.get("session_id")
    existing = False
    success = bool(result and result.ok)
    if result is not None:
        session_payload = result.data.get("session")
        if isinstance(session_payload, dict):
            payload_session_id = session_payload.get("session_id")
            if isinstance(payload_session_id, str):
                session_id = payload_session_id
            scenes = session_payload.get("scenes")
            if result.ok and isinstance(scenes, list):
                successful_scene_count = len(scenes)
                payload_duration = session_payload.get("duration_seconds")
                payload_scene_seconds = session_payload.get("scene_seconds")
                if (
                    isinstance(payload_duration, (int, float))
                    and not isinstance(payload_duration, bool)
                    and math.isfinite(float(payload_duration))
                    and payload_duration > 0
                    and isinstance(payload_scene_seconds, int)
                    and not isinstance(payload_scene_seconds, bool)
                    and payload_scene_seconds > 0
                ):
                    planned_scene_count = max(
                        1,
                        math.ceil(float(payload_duration) / payload_scene_seconds),
                    )
        existing = result.data.get("existing") is True
        if result.error and isinstance(result.error.get("code"), str):
            error_code = str(result.error["code"])
        if _valid_scene_count(planned_scene_count, allow_zero=False):
            result.metrics["planned_scene_count"] = planned_scene_count
            result.metrics["successful_scene_count"] = successful_scene_count

    trace_id = secrets.token_hex(8)
    event = {
        "run_id": f"ingest_{trace_id}",
        "session_id": session_id,
        "stage": "SLICED" if success else "INGEST",
        "event": "ingest.completed",
        "tool_name": "ingest_video",
        "timestamp": utc_now(),
        "trace_id": trace_id,
        "success": success,
        "probe_succeeded": attempt.get("probe_succeeded") is True,
        "planned_scene_count": (
            planned_scene_count
            if _valid_scene_count(planned_scene_count, allow_zero=False)
            else None
        ),
        "successful_scene_count": successful_scene_count,
        "existing": existing,
        "demo": False,
        "error_code": error_code,
    }
    try:
        store.append_journal(event)
    except Exception:
        # Ingest correctness and source retention must not depend on telemetry.
        return


def _valid_scene_count(value: object, *, allow_zero: bool) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and (value >= 0 if allow_zero else value > 0)
    )


def ingest_video(
    store: LocalStore,
    video_path: Path | str,
    *,
    scene_seconds: int = 15,
    frame_step: int = 5,
    max_duration_seconds: float | None = None,
    allow_long_video: bool = False,
    absolute_max_duration_seconds: float = DEFAULT_ABSOLUTE_MAX_VIDEO_DURATION_SECONDS,
    max_scene_count: int = DEFAULT_MAX_SCENE_COUNT,
    max_split_output_bytes: int = DEFAULT_MAX_SPLIT_OUTPUT_BYTES,
    max_width: int | None = None,
    max_height: int | None = None,
    max_fps: float | None = None,
    session_name: str | None = None,
) -> ToolResult:
    """Ingest a video and durably record one success or failure outcome."""

    attempt: dict[str, Any] = {
        "probe_succeeded": False,
        "planned_scene_count": None,
        "session_id": None,
    }
    try:
        result = _ingest_video(
            store,
            video_path,
            scene_seconds=scene_seconds,
            frame_step=frame_step,
            max_duration_seconds=max_duration_seconds,
            allow_long_video=allow_long_video,
            absolute_max_duration_seconds=absolute_max_duration_seconds,
            max_scene_count=max_scene_count,
            max_split_output_bytes=max_split_output_bytes,
            max_width=max_width,
            max_height=max_height,
            max_fps=max_fps,
            session_name=session_name,
            _attempt=attempt,
        )
    except Exception:
        _record_ingest_completion(store, None, attempt, error_code="INGEST_EXCEPTION")
        raise
    _record_ingest_completion(store, result, attempt)
    return result
