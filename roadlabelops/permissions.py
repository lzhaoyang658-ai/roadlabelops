from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from .models import StrEnum


class Decision(StrEnum):
    ALLOW = "ALLOW"
    ASK = "ASK"
    DENY = "DENY"


@dataclass(frozen=True, slots=True)
class PermissionResult:
    decision: Decision
    reason: str


class PermissionPolicy:
    ALWAYS_ALLOWED: ClassVar[set[str]] = {
        "create_tasks",
        "preannotate",
        "request_review",
        "complete_review",
        "calculate_quality",
        "release",
    }
    NEVER_ALLOWED: ClassVar[set[str]] = {
        "delete_cvat_project",
        "delete_cvat_task",
        "delete_manual_annotations",
    }

    def __init__(self, data_root: Path | str = "data") -> None:
        self.data_root = Path(data_root).resolve()

    def check(
        self, action: str, target: str | Path | None = None, *, exists: bool = False
    ) -> PermissionResult:
        if action in self.NEVER_ALLOWED:
            return PermissionResult(
                Decision.DENY, "V1 never deletes remote or manual annotation data"
            )
        if action == "write_file":
            if target is None:
                return PermissionResult(Decision.DENY, "write target is required")
            resolved = Path(target).resolve()
            if not resolved.is_relative_to(self.data_root):
                return PermissionResult(Decision.DENY, "writes are limited to the data directory")
            if exists:
                return PermissionResult(
                    Decision.ASK, "target exists and overwrite can break traceability"
                )
            return PermissionResult(Decision.ALLOW, "write is inside the managed data directory")
        if action in {"replace_auto_annotations", "overwrite_scene"}:
            return PermissionResult(Decision.ASK, "the action replaces an existing generated asset")
        if action == "overwrite_release":
            return PermissionResult(Decision.DENY, "dataset releases are immutable")
        if action in self.ALWAYS_ALLOWED:
            return PermissionResult(Decision.ALLOW, "allowed by the V1 policy")
        return PermissionResult(
            Decision.DENY,
            "the action is not covered by the V1 permission policy",
        )
