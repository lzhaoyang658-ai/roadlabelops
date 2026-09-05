from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .models import Session


def _metric(numerator: int, denominator: int, reason: str) -> dict[str, Any]:
    return {
        "value": round(numerator / denominator, 4) if denominator else None,
        "numerator": numerator,
        "denominator": denominator,
        "reason": None if denominator else reason,
    }


def _planned_scene_count(session: Session) -> int:
    if (
        isinstance(session.duration_seconds, bool)
        or not isinstance(session.duration_seconds, (int, float))
        or not math.isfinite(float(session.duration_seconds))
        or session.duration_seconds <= 0
        or isinstance(session.scene_seconds, bool)
        or not isinstance(session.scene_seconds, int)
        or session.scene_seconds <= 0
    ):
        return max(1, len(session.scenes))
    return max(1, math.ceil(session.duration_seconds / session.scene_seconds))


def _valid_scene_count(value: object, *, allow_zero: bool) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and (value >= 0 if allow_zero else value > 0)
    )


def calculate_operational_metrics(
    sessions: Sequence[Session],
    releases: Mapping[str, Mapping[str, Any]],
    journal: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Calculate the three workflow-wide PRD success rates without fake zeroes."""

    real_sessions = [session for session in sessions if not session.demo]
    real_session_ids = {session.session_id for session in real_sessions}
    demo_session_ids = {session.session_id for session in sessions if session.demo}

    planned_scenes = 0
    successful_scenes = 0
    sessions_covered_by_ingest_events: set[str] = set()
    seen_ingest_attempts: set[str] = set()
    for index, event in enumerate(journal):
        if event.get("event") != "ingest.completed" or event.get("demo") is True:
            continue
        success = event.get("success")
        planned = event.get("planned_scene_count")
        successful = event.get("successful_scene_count")
        if (
            not isinstance(success, bool)
            or not _valid_scene_count(planned, allow_zero=False)
            or not _valid_scene_count(successful, allow_zero=True)
            or successful > planned
        ):
            continue

        raw_session_id = event.get("session_id")
        session_id = raw_session_id if isinstance(raw_session_id, str) else None
        if session_id in demo_session_ids:
            continue
        attempt_key = str(
            event.get("trace_id")
            or event.get("run_id")
            or f"legacy-ingest-event-{index}"
        )
        if attempt_key in seen_ingest_attempts:
            continue
        seen_ingest_attempts.add(attempt_key)

        if success:
            # Successful duplicate uploads refer to the same persisted slice.
            # Count that real Session once, then use events only for failed attempts.
            if (
                session_id not in real_session_ids
                or session_id in sessions_covered_by_ingest_events
            ):
                continue
            sessions_covered_by_ingest_events.add(session_id)
        planned_scenes += planned
        successful_scenes += successful

    # Migration fallback: Sessions created before ingest.completed was introduced
    # remain measurable, but a Session represented by a new event is never added twice.
    for session in real_sessions:
        if session.session_id in sessions_covered_by_ingest_events:
            continue
        planned = _planned_scene_count(session)
        planned_scenes += planned
        successful_scenes += min(len(session.scenes), planned)

    total_scenes = 0
    successful_tasks = 0
    for session in real_sessions:
        total_scenes += len(session.scenes)
        successful_tasks += sum(
            isinstance(scene.cvat_task_id, int) and not isinstance(scene.cvat_task_id, bool)
            for scene in session.scenes
        )

    release_attempts: set[str] = set()
    for event in journal:
        event_session_id = event.get("session_id")
        if (
            event.get("event") == "stage.started"
            and (event.get("action") == "release" or event.get("tool_name") == "release")
            and isinstance(event_session_id, str)
            and event_session_id in real_session_ids
            and event.get("trace_id")
        ):
            release_attempts.add(str(event["trace_id"]))
    verified_releases = sum(
        release.get("verified") is True
        for session_id, release in releases.items()
        if session_id in real_session_ids
    )
    release_attempt_count = max(len(release_attempts), verified_releases)

    return {
        "video_slice_success_rate": _metric(
            successful_scenes,
            planned_scenes,
            "尚未规划任何视频 Session。",
        ),
        "task_creation_success_rate": _metric(
            successful_tasks,
            total_scenes,
            "尚未生成任何 Scene。",
        ),
        "release_verification_success_rate": _metric(
            verified_releases,
            release_attempt_count,
            "尚未发起任何 Release。",
        ),
    }
