from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StrEnum(str, Enum):
    """Small Python 3.10-compatible subset of :class:`enum.StrEnum`."""

    def __str__(self) -> str:
        return self.value


class Stage(StrEnum):
    NEW = "NEW"
    PROBED = "PROBED"
    SLICED = "SLICED"
    TASKS_CREATED = "TASKS_CREATED"
    PREANNOTATED = "PREANNOTATED"
    WAITING_FOR_HUMAN_REVIEW = "WAITING_FOR_HUMAN_REVIEW"
    REVIEW_COMPLETED = "REVIEW_COMPLETED"
    QUALITY_CALCULATED = "QUALITY_CALCULATED"
    RELEASED = "RELEASED"
    FAILED_RETRYABLE = "FAILED_RETRYABLE"
    WAITING_FOR_PERMISSION = "WAITING_FOR_PERMISSION"
    FAILED_FINAL = "FAILED_FINAL"
    CANCELLED = "CANCELLED"


@dataclass(slots=True)
class Scene:
    scene_id: str
    session_id: str
    start_seconds: float
    end_seconds: float
    video_path: str
    thumbnail_path: str | None = None
    cvat_project_id: int | None = None
    cvat_task_id: int | None = None
    cvat_job_ids: list[int] = field(default_factory=list)
    status: str = "ready"
    prediction_count: int | None = None
    final_count: int | None = None
    video_sha256: str | None = None
    scene_tags: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class Session:
    session_id: str
    name: str
    source_path: str
    source_sha256: str
    duration_seconds: float
    fps: float
    width: int
    height: int
    scene_seconds: int = 15
    frame_step: int = 5
    status: Stage = Stage.NEW
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    scenes: list[Scene] = field(default_factory=list)
    demo: bool = False
    resume_stage: str | None = None
    pending_action: str | None = None
    last_error: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Session:
        data = dict(payload)
        data["status"] = Stage(data["status"])
        data["scenes"] = [Scene(**scene) for scene in data.get("scenes", [])]
        return cls(**data)


@dataclass(slots=True)
class ToolResult:
    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] | None = None
    retryable: bool = False
    side_effects: list[str] = field(default_factory=list)
    metrics: dict[str, float | int | str | None] = field(default_factory=dict)

    @classmethod
    def success(cls, data: dict[str, Any] | None = None, **kwargs: Any) -> ToolResult:
        return cls(ok=True, data=data or {}, **kwargs)

    @classmethod
    def failure(
        cls, code: str, message: str, *, retryable: bool = False
    ) -> ToolResult:
        return cls(
            ok=False,
            error={"code": code, "message": message},
            retryable=retryable,
        )
