from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .hooks import HookBus
from .metrics import calculate_operational_metrics
from .models import Scene, Session, Stage, ToolResult, utc_now
from .permissions import Decision, PermissionPolicy
from .storage import LocalStore
from .tool_registry import ToolRegistry
from .tools.detection import run_mock_detection
from .tools.quality import calculate_quality
from .tools.release import (
    CATEGORY_IDS,
    QUALITY_REPORT_FIELDS,
    _normalise_annotations,
    _normalise_scene_tags,
    build_coco_release,
    normalise_evaluated_frames,
    normalise_release_predictions,
    verify_coco_release,
)

ALLOWED_ACTIONS: dict[Stage, set[str]] = {
    Stage.SLICED: {"create_tasks"},
    Stage.TASKS_CREATED: {"preannotate"},
    Stage.PREANNOTATED: {"request_review"},
    Stage.WAITING_FOR_HUMAN_REVIEW: {"complete_review"},
    Stage.REVIEW_COMPLETED: {"calculate_quality"},
    Stage.QUALITY_CALCULATED: {"release"},
}

RECOVERY_STAGES = {Stage.FAILED_RETRYABLE, Stage.WAITING_FOR_PERMISSION}
FIRST_PASS_UNAVAILABLE_REASON = (
    "CVAT exposes the current Job state but not enough rejection history "
    "to prove first-pass acceptance."
)


class WorkflowRuntime:
    def __init__(
        self,
        store: LocalStore,
        cvat: Any | None = None,
        detector: Callable[[Path | str, int], ToolResult] | None = None,
        *,
        policy: PermissionPolicy | None = None,
        hooks: HookBus | None = None,
        registry: ToolRegistry | None = None,
    ) -> None:
        self.store = store
        self.cvat = cvat
        self.detector = detector
        if registry is None:
            self.policy = policy or PermissionPolicy(store.root)
            self.hooks = hooks or HookBus(store.runtime_root.parent / "logs" / "roadlabelops.jsonl")
            self.registry = ToolRegistry(self.policy, self.hooks)
        else:
            if policy is not None or hooks is not None:
                raise ValueError("policy and hooks must be configured on an injected registry")
            self.registry = registry
            self.policy = registry.policy
            self.hooks = registry.hooks
        for action in sorted({item for actions in ALLOWED_ACTIONS.values() for item in actions}):
            self.registry.register(f"workflow.{action}", self._execute_action)

    def _event(
        self,
        session: Session,
        event: str,
        tool: str | None = None,
        *,
        trace_id: str | None = None,
        **extra: Any,
    ) -> None:
        self.store.append_journal(
            {
                "run_id": f"run_{session.session_id}",
                "session_id": session.session_id,
                "stage": session.status.value,
                "event": event,
                "tool_name": tool,
                "timestamp": utc_now(),
                "trace_id": trace_id or secrets.token_hex(8),
                **extra,
            }
        )

    def create_demo(self) -> Session:
        identifier = f"session_demo_{secrets.token_hex(3)}"
        session = Session(
            session_id=identifier,
            name="上海高架夜间道路采样",
            source_path="data/raw/demo-road-night.mp4",
            source_sha256=hashlib.sha256(identifier.encode()).hexdigest(),
            duration_seconds=60,
            fps=25,
            width=1920,
            height=1080,
            scene_seconds=15,
            status=Stage.SLICED,
            demo=True,
            scenes=[
                Scene(
                    scene_id=f"{identifier}_scene_{index:03d}",
                    session_id=identifier,
                    start_seconds=(index - 1) * 15,
                    end_seconds=index * 15,
                    video_path=str(self.store.scenes_dir / identifier / f"scene_{index:03d}.mp4"),
                    thumbnail_path=None,
                )
                for index in range(1, 5)
            ],
        )
        self.store.save_session(session)
        self._event(session, "workflow.started")
        self._event(session, "stage.succeeded", "split_video", scene_count=4)
        return session

    def advance(
        self,
        session_id: str,
        action: str,
        *,
        version: str = "1.0.0",
        approved: bool = False,
    ) -> ToolResult:
        with self.store.operation_lock(f"session-{session_id}"):
            return self._advance_locked(
                session_id,
                action,
                version=version,
                approved=approved,
            )

    def _advance_locked(
        self,
        session_id: str,
        action: str,
        *,
        version: str,
        approved: bool,
    ) -> ToolResult:
        session = self.store.get_session(session_id)
        from_stage = self._resolve_action_stage(session, action)
        if from_stage is None:
            return ToolResult.failure(
                "INVALID_TRANSITION",
                f"Action {action} is not allowed while session is {session.status.value}",
            )
        trace_id = secrets.token_hex(16)
        self._event(
            session,
            "stage.started",
            action,
            trace_id=trace_id,
            action=action,
            from_stage=from_stage.value,
            recovery=session.status in RECOVERY_STAGES,
            approved=approved,
        )
        result = self.registry.execute(
            f"workflow.{action}",
            session_id=session.session_id,
            permission_action=action,
            approved=approved,
            event_trace_id=trace_id,
            session=session,
            action=action,
            version=version,
            workflow_approved=approved,
        )
        if not result.ok:
            self._record_failure(session, action, from_stage, result, trace_id)
            return result

        session.resume_stage = None
        session.pending_action = None
        session.last_error = None
        session.updated_at = utc_now()
        self.store.save_session(session)
        self._event(
            session,
            "stage.succeeded",
            action,
            trace_id=trace_id,
            action=action,
            from_stage=from_stage.value,
        )
        payload = dict(result.data)
        payload["session"] = session.to_dict()
        return ToolResult.success(
            payload,
            side_effects=result.side_effects,
            metrics=result.metrics,
        )

    def _execute_action(
        self,
        session: Session,
        action: str,
        version: str,
        workflow_approved: bool = False,
    ) -> ToolResult:
        if action == "create_tasks":
            if session.demo:
                for index, scene in enumerate(session.scenes, start=1):
                    scene.cvat_project_id = 70
                    scene.cvat_task_id = 700 + index
                    scene.cvat_job_ids = [1700 + index]
                    scene.status = "annotation"
            elif self.cvat:
                scene_paths: dict[str, Path] = {}
                for scene in session.scenes:
                    if scene.cvat_task_id:
                        continue
                    try:
                        scene_paths[scene.scene_id] = self._managed_scene_video_path(
                            scene,
                            require_file=True,
                        )
                    except (OSError, RuntimeError, ValueError) as error:
                        return self._scene_path_failure(scene, error)
                project = self.cvat.ensure_project(f"RoadLabelOps · {session.name}", self._labels())
                if not project.ok:
                    return project
                for scene in session.scenes:
                    if scene.cvat_task_id:
                        continue
                    task = self.cvat.create_task(
                        scene.scene_id,
                        project.data["project_id"],
                        scene_paths[scene.scene_id],
                    )
                    if not task.ok:
                        return task
                    scene.cvat_project_id = project.data["project_id"]
                    scene.cvat_task_id = task.data["task_id"]
                    scene.cvat_job_ids = list(task.data.get("job_ids", []))
                    self.store.save_session(session)
            else:
                return ToolResult.failure(
                    "CVAT_NOT_CONFIGURED", "Configure CVAT before creating real tasks"
                )
            session.status = Stage.TASKS_CREATED
        elif action == "preannotate":
            for scene in session.scenes:
                if scene.prediction_count is not None:
                    continue
                try:
                    scene_video_path = self._managed_scene_video_path(
                        scene,
                        require_file=not session.demo,
                    )
                except (OSError, RuntimeError, ValueError) as error:
                    return self._scene_path_failure(scene, error)
                try:
                    prediction_path = self._managed_sidecar_path(scene, "prediction")
                except (OSError, RuntimeError, ValueError) as error:
                    return self._sidecar_path_failure("prediction", scene, error)
                if session.demo:
                    prediction = run_mock_detection(scene_video_path)
                elif self.detector:
                    prediction = self.detector(scene_video_path, session.frame_step)
                else:
                    prediction = ToolResult.failure(
                        "DETECTOR_UNAVAILABLE", "No real detection provider is configured"
                    )
                if not prediction.ok:
                    predictions: list[dict[str, Any]] = []
                    self._event(
                        session,
                        "model.degraded",
                        "run_object_detection",
                        scene_id=scene.scene_id,
                        error=prediction.error,
                    )
                else:
                    predictions = [
                        {**item, "scene_id": scene.scene_id}
                        for item in prediction.data["predictions"]
                    ]
                try:
                    staged_prediction = self._stage_json_sidecar(
                        prediction_path,
                        predictions,
                    )
                except (OSError, TypeError, ValueError) as error:
                    return ToolResult.failure(
                        "PREDICTION_SIDECAR_WRITE_FAILED",
                        f"Could not stage predictions for scene {scene.scene_id}: {error}",
                    )
                try:
                    if not session.demo:
                        if not self.cvat or not scene.cvat_task_id:
                            return ToolResult.failure(
                                "CVAT_NOT_CONFIGURED",
                                "CVAT tasks are required before pre-annotation",
                            )
                        imported = self.cvat.import_predictions(
                            scene.cvat_task_id,
                            predictions,
                        )
                        if (
                            not imported.ok
                            and (imported.error or {}).get("code") == "AUTO_ANNOTATIONS_EXIST"
                        ):
                            permission = self.policy.check(
                                "replace_auto_annotations",
                                f"cvat:task:{scene.cvat_task_id}",
                                exists=True,
                            )
                            if permission.decision is Decision.DENY:
                                return ToolResult.failure(
                                    "PERMISSION_DENIED",
                                    permission.reason,
                                )
                            if permission.decision is Decision.ASK and not workflow_approved:
                                return ToolResult.failure(
                                    "PERMISSION_REQUIRED",
                                    permission.reason,
                                )
                            imported = self.cvat.import_predictions(
                                scene.cvat_task_id,
                                predictions,
                                allow_replace_auto=True,
                            )
                        if not imported.ok:
                            return imported
                    try:
                        publish_path = self._managed_sidecar_path(scene, "prediction")
                        if publish_path != prediction_path:
                            raise ValueError("prediction sidecar path changed while processing")
                        os.replace(staged_prediction, publish_path)
                    except (OSError, RuntimeError, ValueError) as error:
                        return ToolResult.failure(
                            "PREDICTION_SIDECAR_WRITE_FAILED",
                            f"Could not publish predictions for scene {scene.scene_id}: {error}",
                        )
                finally:
                    staged_prediction.unlink(missing_ok=True)
                scene.prediction_count = len(predictions)
                scene.status = "annotation"
                self.store.save_session(session)
            session.status = Stage.PREANNOTATED
        elif action == "request_review":
            session.status = Stage.WAITING_FOR_HUMAN_REVIEW
        elif action == "complete_review":
            if session.demo:
                for scene in session.scenes:
                    scene.final_count = max(1, int((scene.prediction_count or 6) * 0.88) + 1)
                    scene.status = "completed"
            elif self.cvat:
                pending: list[int] = []
                for scene in session.scenes:
                    if scene.status == "completed" and scene.final_count is not None:
                        continue
                    try:
                        final_path = self._managed_sidecar_path(scene, "final")
                    except (OSError, RuntimeError, ValueError) as error:
                        return self._sidecar_path_failure("final", scene, error)
                    if not scene.cvat_task_id:
                        return ToolResult.failure(
                            "CVAT_TASK_MISSING",
                            f"Scene {scene.scene_id} has no CVAT task to review",
                        )
                    review = self.cvat.get_review_result(scene.cvat_task_id)
                    if not review.ok:
                        return review
                    if not review.data["completed"]:
                        pending.append(scene.cvat_task_id)
                        continue
                    final = [
                        {**item, "scene_id": scene.scene_id} for item in review.data["annotations"]
                    ]
                    try:
                        publish_path = self._managed_sidecar_path(scene, "final")
                        if publish_path != final_path:
                            raise ValueError("final sidecar path changed while processing")
                        self.store.write_json_atomic(publish_path, final)
                    except (OSError, RuntimeError, ValueError) as error:
                        return ToolResult.failure(
                            "FINAL_SIDECAR_WRITE_FAILED",
                            f"Could not publish final annotations for scene {scene.scene_id}: {error}",
                        )
                    scene.scene_tags = [
                        {**item, "scene_id": scene.scene_id}
                        for item in review.data.get("scene_tags", [])
                    ]
                    scene.final_count = len(final)
                    scene.status = "completed"
                    scene.cvat_job_ids = [item["job_id"] for item in review.data["jobs"]]
                    self.store.save_session(session)
                if pending:
                    return ToolResult.failure(
                        "CVAT_REVIEW_INCOMPLETE",
                        f"Complete the CVAT jobs for task(s): {', '.join(map(str, pending))}",
                    )
            else:
                return ToolResult.failure("CVAT_NOT_CONFIGURED", "CVAT is required to sync review")
            session.status = Stage.REVIEW_COMPLETED
        elif action == "calculate_quality":
            if session.demo:
                predictions = self._demo_predictions(session)
                final = self._demo_final(session)
                reviewed_jobs = len(session.scenes)
                accepted_jobs = reviewed_jobs
            else:
                loaded_predictions = self._load_counted_sidecars(session, "prediction")
                if not loaded_predictions.ok:
                    return loaded_predictions
                loaded_final = self._load_counted_sidecars(session, "final", completed_only=True)
                if not loaded_final.ok:
                    return loaded_final
                predictions = list(loaded_predictions.data["annotations"])
                final = list(loaded_final.data["annotations"])
                # CVAT exposes the current state, not enough history to prove first-pass acceptance.
                # Keep the metric unknown instead of presenting a fabricated 100%.
                reviewed_jobs = accepted_jobs = 0
            quality = calculate_quality(
                predictions,
                final,
                accepted_jobs,
                reviewed_jobs,
                evaluated_frame_keys=self._evaluated_frame_keys(session),
                first_pass_acceptance_reason=(
                    None if session.demo else FIRST_PASS_UNAVAILABLE_REASON
                ),
            )
            session.status = Stage.QUALITY_CALCULATED
            self.store.write_json_atomic(
                self.store.sessions_dir / f"{session.session_id}.quality.json", quality.data
            )
        elif action == "release":
            if session.demo:
                predictions = self._demo_predictions(session)
                final = self._demo_final(session)
                reviewed_jobs = len(session.scenes)
                accepted_jobs = reviewed_jobs
                first_pass_reason = None
            else:
                loaded = self._load_release_sidecars(session)
                if not loaded.ok:
                    return loaded
                final = list(loaded.data["annotations"])
                predictions = list(loaded.data["predictions"])
                reviewed_jobs = accepted_jobs = 0
                first_pass_reason = FIRST_PASS_UNAVAILABLE_REASON
            quality_path = self.store.sessions_dir / f"{session.session_id}.quality.json"
            quality = self._read_json_object(quality_path)
            if quality is None:
                return ToolResult.failure(
                    "QUALITY_REPORT_MISSING",
                    "Calculate and persist a valid quality report before releasing",
                )
            if quality.get("final_count") != len(final):
                return ToolResult.failure(
                    "QUALITY_REPORT_MISMATCH",
                    "Quality report final count does not match reviewed annotations",
                )

            recovered = self._adopt_published_release(
                session,
                version,
                predictions=predictions,
                final=final,
                quality=quality,
                accepted_jobs=accepted_jobs,
                reviewed_jobs=reviewed_jobs,
                first_pass_acceptance_reason=first_pass_reason,
            )
            if recovered is None:
                result = build_coco_release(
                    self.store,
                    session,
                    version,
                    final,
                    extract_frames=not session.demo,
                    quality_report=quality,
                    predictions=predictions,
                    evaluated_frame_keys=self._evaluated_frame_keys(session),
                    accepted_jobs=accepted_jobs,
                    reviewed_jobs=reviewed_jobs,
                    first_pass_acceptance_reason=first_pass_reason,
                )
                if not result.ok:
                    return result
                operation_result = result
                release_path = Path(str(result.data["path"]))
                release_summary = self._release_summary(result, release_path, session=session)
            else:
                if not recovered.ok:
                    return recovered
                operation_result = recovered
                release_path = Path(str(recovered.data["path"]))
                release_summary = dict(recovered.data["summary"])
            self.store.write_json_atomic(self._release_record_path(session), release_summary)
            session.status = Stage.RELEASED
            return ToolResult.success(
                {"session": session.to_dict(), "receipt": release_summary},
                side_effects=operation_result.side_effects,
                metrics=operation_result.metrics,
            )
        return ToolResult.success({"session": session.to_dict()})

    @staticmethod
    def _resolve_action_stage(session: Session, action: str) -> Stage | None:
        if session.status not in RECOVERY_STAGES:
            return session.status if action in ALLOWED_ACTIONS.get(session.status, set()) else None
        if session.pending_action != action or session.resume_stage is None:
            return None
        try:
            resume_stage = Stage(session.resume_stage)
        except ValueError:
            return None
        return resume_stage if action in ALLOWED_ACTIONS.get(resume_stage, set()) else None

    def _record_failure(
        self,
        session: Session,
        action: str,
        from_stage: Stage,
        result: ToolResult,
        trace_id: str,
    ) -> None:
        error_code = str((result.error or {}).get("code", "UNKNOWN"))
        if error_code == "PERMISSION_REQUIRED":
            session.status = Stage.WAITING_FOR_PERMISSION
            session.resume_stage = from_stage.value
            session.pending_action = action
        elif result.retryable:
            session.status = Stage.FAILED_RETRYABLE
            session.resume_stage = from_stage.value
            session.pending_action = action
        else:
            # A configuration or validation failure must not strand a previously valid workflow.
            # The caller can fix the input/configuration and invoke the same action again.
            session.status = from_stage
            session.resume_stage = None
            session.pending_action = None
        session.last_error = result.error
        session.updated_at = utc_now()
        self.store.save_session(session)
        self._event(
            session,
            "stage.failed",
            action,
            trace_id=trace_id,
            action=action,
            from_stage=from_stage.value,
            error=result.error,
            retryable=result.retryable,
            recovery_status=session.status.value,
        )

    def dashboard(self) -> dict[str, Any]:
        sessions = self.store.list_sessions()
        # Local single-user V1 returns the complete index so a URL-selected
        # historical Session can always be restored after refresh.
        visible_sessions = sessions
        qualities: dict[str, dict[str, Any]] = {}
        releases: dict[str, dict[str, Any]] = {}
        for session in visible_sessions:
            quality_path = self.store.sessions_dir / f"{session.session_id}.quality.json"
            quality = self._read_json_object(quality_path)
            if quality is not None:
                qualities[session.session_id] = quality
            release = self._load_release_summary(session)
            if release is not None:
                releases[session.session_id] = release
        scene_count = sum(len(session.scenes) for session in sessions)
        task_count = sum(
            1 for session in sessions for scene in session.scenes if scene.cvat_task_id
        )
        journal = self.store.read_journal(None)
        return {
            "summary": {
                "session_count": len(sessions),
                "scene_count": scene_count,
                "task_count": task_count,
                "ready_for_review": sum(
                    1 for session in sessions if session.status is Stage.WAITING_FOR_HUMAN_REVIEW
                ),
            },
            "sessions": [session.to_dict() for session in visible_sessions],
            "quality": (
                qualities.get(visible_sessions[0].session_id) if visible_sessions else None
            ),
            "qualities": qualities,
            "releases": releases,
            "operational_metrics": calculate_operational_metrics(
                sessions,
                releases,
                journal,
            ),
            "activity": list(reversed(journal[-16:])),
        }

    def verify_release(self, session_id: str) -> ToolResult:
        session = self.store.get_session(session_id)
        record_path = self._release_record_path(session)
        if record_path.is_symlink():
            return ToolResult.failure(
                "TRUST_ANCHOR_INVALID", "Trusted release records must not be symbolic links"
            )
        record = self._read_json_object(record_path)
        if record is None:
            code = (
                "TRUST_ANCHOR_MISSING" if session.status is Stage.RELEASED else "RELEASE_NOT_FOUND"
            )
            return ToolResult.failure(
                code,
                "The trusted release record is missing or invalid; automatic re-anchoring is denied",
            )
        release_path = self._find_release_path(session, record)
        if release_path is None:
            return ToolResult.failure(
                "TRUST_ANCHOR_INVALID",
                "The trusted release record points outside the release store or is incomplete",
            )
        expected_manifest_sha256 = record.get("manifest_sha256")
        if not self._valid_sha256(expected_manifest_sha256):
            return ToolResult.failure(
                "TRUST_ANCHOR_INVALID",
                "The trusted release record has no valid manifest SHA-256",
            )
        expected_receipt_sha256 = record.get("receipt_sha256")
        if expected_receipt_sha256 is not None and not self._valid_sha256(expected_receipt_sha256):
            return ToolResult.failure(
                "TRUST_ANCHOR_INVALID",
                "The trusted release record has an invalid receipt SHA-256",
            )
        expected_release_id = record.get("release_id")
        if not isinstance(expected_release_id, str) or expected_release_id != release_path.name:
            return ToolResult.failure(
                "TRUST_ANCHOR_INVALID",
                "The trusted release id does not match its immutable directory",
            )
        expected_version = record.get("version")
        result = verify_coco_release(
            release_path,
            expected_release_id=expected_release_id,
            expected_manifest_sha256=str(expected_manifest_sha256),
            expected_receipt_sha256=(
                str(expected_receipt_sha256) if expected_receipt_sha256 is not None else None
            ),
            expected_session_id=session.session_id,
            expected_version=expected_version
            if isinstance(expected_version, str) and expected_version
            else None,
            expected_source_sha256=session.source_sha256,
        )
        summary = self._release_summary(
            result,
            release_path,
            session=session,
            baseline_manifest_sha256=str(expected_manifest_sha256),
            baseline_receipt_sha256=(
                str(expected_receipt_sha256) if expected_receipt_sha256 is not None else None
            ),
        )
        self.store.write_json_atomic(self._release_record_path(session), summary)
        if not result.ok:
            return ToolResult(
                ok=False,
                data={"receipt": summary},
                error=result.error,
                retryable=False,
            )
        return ToolResult.success({"receipt": summary})

    def _release_record_path(self, session: Session) -> Path:
        return self.store.sessions_dir / f"{session.session_id}.release.json"

    @staticmethod
    def _valid_sha256(value: Any) -> bool:
        return (
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value.lower())
        )

    @staticmethod
    def _read_json_object(path: Path) -> dict[str, Any] | None:
        if path.is_symlink() or not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    def _find_release_path(
        self, session: Session, record: dict[str, Any] | None = None
    ) -> Path | None:
        anchor = (
            record
            if record is not None
            else self._read_json_object(self._release_record_path(session))
        )
        if not anchor or not isinstance(anchor.get("path"), str):
            return None
        recorded_path = Path(anchor["path"])
        if (
            recorded_path.is_symlink()
            or recorded_path.parent.resolve() != self.store.releases_dir.resolve()
            or not recorded_path.is_dir()
        ):
            return None
        return recorded_path

    def _load_release_summary(self, session: Session) -> dict[str, Any] | None:
        record_path = self._release_record_path(session)
        if record_path.is_symlink():
            return {
                "release_id": "",
                "version": "",
                "path": "",
                "manifest_sha256": None,
                "file_count": 0,
                "verified": False,
                "checked_at": utc_now(),
                "errors": ["Trusted release record must not be a symbolic link"],
            }
        record = self._read_json_object(record_path)
        if record is not None:
            release_path = self._find_release_path(session, record)
            anchored_hash = record.get("manifest_sha256")
            anchored_receipt_hash = record.get("receipt_sha256")
            record_session_id = record.get("session_id")
            record_source_sha256 = record.get("source_sha256")
            if (
                release_path is None
                or not self._valid_sha256(anchored_hash)
                or not self._valid_sha256(anchored_receipt_hash)
                or record.get("release_id") != release_path.name
                or (record_session_id not in (None, "", session.session_id))
                or (record_source_sha256 not in (None, "", session.source_sha256))
            ):
                return self._invalid_cached_release(record, "Trusted release record is invalid")
            manifest_path = release_path / "manifest.json"
            receipt_path = release_path / "receipt.json"
            if (
                manifest_path.is_symlink()
                or not manifest_path.is_file()
                or self._file_sha256(manifest_path) != anchored_hash
                or receipt_path.is_symlink()
                or not receipt_path.is_file()
                or self._file_sha256(receipt_path) != anchored_receipt_hash
            ):
                return self._invalid_cached_release(
                    record,
                    "Release manifest no longer matches the trusted record; run full verification",
                )
            return record
        if session.status is not Stage.RELEASED and not record_path.exists():
            return None
        return {
            "release_id": "",
            "version": "",
            "path": "",
            "manifest_sha256": None,
            "file_count": 0,
            "verified": False,
            "checked_at": utc_now(),
            "errors": [
                "Trusted release record is missing or invalid; automatic re-anchoring is denied"
            ],
        }

    @staticmethod
    def _file_sha256(path: Path) -> str | None:
        digest = hashlib.sha256()
        try:
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
        except OSError:
            return None
        return digest.hexdigest()

    @staticmethod
    def _invalid_cached_release(record: dict[str, Any], message: str) -> dict[str, Any]:
        summary = dict(record)
        summary["verified"] = False
        summary["errors"] = [message]
        summary["checked_at"] = utc_now()
        return summary

    def _release_summary(
        self,
        result: ToolResult,
        release_path: Path,
        *,
        session: Session | None = None,
        baseline_manifest_sha256: str | None = None,
        baseline_receipt_sha256: str | None = None,
    ) -> dict[str, Any]:
        verification = result.data.get("receipt", {})
        integrity = self._read_json_object(release_path / "receipt.json") or {}
        manifest = self._read_json_object(release_path / "manifest.json") or {}
        issues = verification.get("issues", [])
        return {
            "release_id": str(verification.get("release_id") or release_path.name),
            "version": str(manifest.get("version", "")),
            "session_id": session.session_id
            if session is not None
            else str((manifest.get("source_session") or {}).get("session_id", "")),
            "source_sha256": session.source_sha256
            if session is not None
            else str((manifest.get("source_session") or {}).get("source_sha256", "")),
            "path": str(release_path),
            "manifest_sha256": baseline_manifest_sha256 or integrity.get("manifest_sha256"),
            "receipt_sha256": baseline_receipt_sha256
            or self._file_sha256(release_path / "receipt.json"),
            "file_count": integrity.get(
                "payload_file_count", len(manifest.get("payload_sha256", {}))
            ),
            "verified": bool(result.ok and verification.get("valid")),
            "checked_at": verification.get("verified_at") or utc_now(),
            "errors": [
                str(issue.get("message", issue)) if isinstance(issue, dict) else str(issue)
                for issue in issues
            ],
        }

    def _published_release_matches_session(
        self,
        session: Session,
        release_path: Path,
        *,
        predictions: list[dict[str, Any]],
        final: list[dict[str, Any]],
        quality: dict[str, Any],
        accepted_jobs: int,
        reviewed_jobs: int,
        first_pass_acceptance_reason: str | None,
    ) -> ToolResult:
        """Bind a pre-anchor crash artifact to the current persisted workflow state."""

        manifest = self._read_json_object(release_path / "manifest.json")
        coco = self._read_json_object(release_path / "annotations.coco.json")
        released_quality = self._read_json_object(release_path / "quality.json")
        try:
            released_predictions = json.loads(
                (release_path / "predictions.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            released_predictions = None
        if manifest is None or coco is None or released_quality is None:
            return ToolResult.failure(
                "RECOVERABLE_RELEASE_INCOMPLETE",
                "Published release metadata is incomplete and cannot be adopted",
            )

        expected_source = {
            "session_id": session.session_id,
            "source_path": Path(session.source_path).name,
            "source_path_kind": "basename",
            "source_sha256": session.source_sha256,
            "duration_seconds": session.duration_seconds,
            "fps": session.fps,
            "width": session.width,
            "height": session.height,
        }
        expected_lineage: list[dict[str, Any]] = []
        for scene in sorted(session.scenes, key=lambda item: item.scene_id):
            tags, tag_error = _normalise_scene_tags(scene, fps=session.fps)
            if tag_error is not None:
                return tag_error
            expected_lineage.append(
                {
                    "scene_id": scene.scene_id,
                    "session_id": scene.session_id,
                    "start_seconds": scene.start_seconds,
                    "end_seconds": scene.end_seconds,
                    "cvat_task_id": scene.cvat_task_id,
                    "cvat_job_ids": list(scene.cvat_job_ids),
                    "video_sha256": None
                    if session.demo
                    else str(scene.video_sha256).lower(),
                    "final_count": scene.final_count,
                    "scene_tags": tags,
                }
            )
        expected_evaluated, evaluated_error = normalise_evaluated_frames(
            session, self._evaluated_frame_keys(session)
        )
        if evaluated_error is not None:
            return evaluated_error
        expected_context = {
            "accepted_jobs": accepted_jobs,
            "reviewed_jobs": reviewed_jobs,
            "first_pass_acceptance_reason": first_pass_acceptance_reason,
        }
        expected_task_ids = [
            item["cvat_task_id"]
            for item in expected_lineage
            if item["cvat_task_id"] is not None
        ]
        if (
            manifest.get("source_session") != expected_source
            or manifest.get("session_ids") != [session.session_id]
            or manifest.get("scene_ids") != sorted(scene.scene_id for scene in session.scenes)
            or manifest.get("scene_lineage") != expected_lineage
            or manifest.get("cvat_task_ids") != expected_task_ids
            or manifest.get("evaluated_frames") != expected_evaluated
            or manifest.get("quality_context") != expected_context
        ):
            return ToolResult.failure(
                "RECOVERABLE_RELEASE_LINEAGE_MISMATCH",
                "Published release lineage differs from the current Session",
            )

        canonical_predictions, prediction_error = normalise_release_predictions(
            session, predictions
        )
        if (
            prediction_error is not None
            or released_predictions != canonical_predictions
            or manifest.get("prediction_count") != len(canonical_predictions)
        ):
            return ToolResult.failure(
                "RECOVERABLE_RELEASE_PREDICTIONS_MISMATCH",
                "Published predictions differ from the current Session sidecars",
            )

        normalised_final, final_error = _normalise_annotations(session, final)
        if final_error is not None:
            return final_error
        images = coco.get("images")
        annotations = coco.get("annotations")
        if not isinstance(images, list) or not isinstance(annotations, list):
            return ToolResult.failure(
                "RECOVERABLE_RELEASE_FINAL_MISMATCH",
                "Published COCO labels are invalid",
            )
        image_by_id = {
            item.get("id"): item for item in images if isinstance(item, dict)
        }
        label_by_id = {identifier: label for label, identifier in CATEGORY_IDS.items()}
        released_final: list[dict[str, Any]] = []
        try:
            for annotation in annotations:
                image = image_by_id[annotation["image_id"]]
                released_final.append(
                    {
                        "scene_id": image["scene_id"],
                        "frame": image["frame"],
                        "label": label_by_id[annotation["category_id"]],
                        "bbox": annotation["bbox"],
                        "source": annotation["source"],
                        "attributes": annotation.get("attributes", {}),
                    }
                )
        except (KeyError, TypeError, ValueError):
            return ToolResult.failure(
                "RECOVERABLE_RELEASE_FINAL_MISMATCH",
                "Published COCO labels cannot be compared to current sidecars",
            )
        expected_final = [
            {
                **item,
                "bbox": [
                    round(float(item["bbox"][0]), 2),
                    round(float(item["bbox"][1]), 2),
                    round(float(item["bbox"][2]) - float(item["bbox"][0]), 2),
                    round(float(item["bbox"][3]) - float(item["bbox"][1]), 2),
                ],
            }
            for item in normalised_final
        ]
        if released_final != expected_final:
            return ToolResult.failure(
                "RECOVERABLE_RELEASE_FINAL_MISMATCH",
                "Published COCO labels differ from the current Session sidecars",
            )
        expected_quality = dict(quality)
        if set(expected_quality) == QUALITY_REPORT_FIELDS - {
            "first_pass_acceptance_reason"
        }:
            expected_quality["first_pass_acceptance_reason"] = (
                first_pass_acceptance_reason if reviewed_jobs == 0 else None
            )
        if released_quality != expected_quality:
            return ToolResult.failure(
                "RECOVERABLE_RELEASE_QUALITY_MISMATCH",
                "Published quality differs from the current Session report",
            )
        return ToolResult.success()

    def _adopt_published_release(
        self,
        session: Session,
        version: str,
        *,
        predictions: list[dict[str, Any]],
        final: list[dict[str, Any]],
        quality: dict[str, Any],
        accepted_jobs: int,
        reviewed_jobs: int,
        first_pass_acceptance_reason: str | None,
    ) -> ToolResult | None:
        """Reconcile a complete release left by a crash before state persistence."""

        release_id = f"{session.session_id}-v{version}"
        if not release_id or not all(
            character.isalnum() or character in "._-" for character in release_id
        ):
            return None
        target = self.store.releases_dir / release_id
        if not target.exists() and not target.is_symlink():
            return None

        record_path = self._release_record_path(session)
        if record_path.is_symlink():
            return ToolResult.failure(
                "TRUST_ANCHOR_INVALID", "Release record must not be a symbolic link"
            )
        record = self._read_json_object(record_path)
        if record_path.exists() and record is None:
            return ToolResult.failure(
                "TRUST_ANCHOR_INVALID",
                "A corrupt release record exists; automatic adoption is denied",
            )
        expected_anchor: str | None = None
        expected_receipt_anchor: str | None = None
        if record is not None:
            anchored_path = self._find_release_path(session, record)
            if anchored_path != target or not self._valid_sha256(record.get("manifest_sha256")):
                return ToolResult.failure(
                    "TRUST_ANCHOR_INVALID",
                    "Existing release record does not match the recoverable release",
                )
            expected_anchor = str(record["manifest_sha256"])
            raw_receipt_anchor = record.get("receipt_sha256")
            if raw_receipt_anchor is not None and not self._valid_sha256(raw_receipt_anchor):
                return ToolResult.failure(
                    "TRUST_ANCHOR_INVALID",
                    "Existing release record has an invalid receipt anchor",
                )
            expected_receipt_anchor = (
                str(raw_receipt_anchor) if raw_receipt_anchor is not None else None
            )

        result = verify_coco_release(
            target,
            expected_release_id=release_id,
            expected_manifest_sha256=expected_anchor,
            expected_receipt_sha256=expected_receipt_anchor,
            expected_session_id=session.session_id,
            expected_version=version,
            expected_source_sha256=session.source_sha256,
        )
        if not result.ok:
            return result
        bound = self._published_release_matches_session(
            session,
            target,
            predictions=predictions,
            final=final,
            quality=quality,
            accepted_jobs=accepted_jobs,
            reviewed_jobs=reviewed_jobs,
            first_pass_acceptance_reason=first_pass_acceptance_reason,
        )
        if not bound.ok:
            return bound
        manifest = self._read_json_object(target / "manifest.json") or {}
        if not (target / "quality.json").is_file() or not manifest.get("quality_sha256"):
            return ToolResult.failure(
                "RECOVERABLE_RELEASE_INCOMPLETE",
                "Published release has no frozen quality report and cannot be adopted",
            )
        summary = self._release_summary(
            result,
            target,
            session=session,
            baseline_manifest_sha256=expected_anchor,
            baseline_receipt_sha256=expected_receipt_anchor,
        )
        return ToolResult.success({"path": str(target), "summary": summary, "recovered": True})

    @staticmethod
    def _labels() -> list[str]:
        return [
            "car",
            "bus",
            "truck",
            "motorcycle",
            "bicycle",
            "pedestrian",
            "traffic_light",
            "traffic_sign",
        ]

    @staticmethod
    def _evaluated_frame_keys(session: Session) -> set[tuple[str, int]]:
        stride = max(1, session.frame_step)
        frame_keys: set[tuple[str, int]] = set()
        for scene in session.scenes:
            frame_count = max(
                1,
                math.ceil((scene.end_seconds - scene.start_seconds) * session.fps),
            )
            frame_keys.update(
                (scene.scene_id, frame) for frame in range(stride - 1, frame_count, stride)
            )
        return frame_keys

    @staticmethod
    def _demo_predictions(session: Session) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for scene in session.scenes:
            for index in range(scene.prediction_count or 6):
                items.append(
                    {
                        "prediction_id": f"{scene.scene_id}_{index}",
                        "scene_id": scene.scene_id,
                        "frame": index * 5,
                        "label": "car" if index % 3 else "pedestrian",
                        "confidence": 0.9,
                        "bbox": [20 + index, 30 + index, 120 + index, 110 + index],
                        "source": "auto",
                    }
                )
        return items

    @staticmethod
    def _demo_final(session: Session) -> list[dict[str, Any]]:
        predictions = WorkflowRuntime._demo_predictions(session)
        final: list[dict[str, Any]] = []
        for scene in session.scenes:
            scene_predictions = [
                item for item in predictions if item["scene_id"] == scene.scene_id
            ]
            target_count = scene.final_count
            if target_count is None:
                target_count = max(1, int(len(scene_predictions) * 0.88) + 1)
            retained_count = min(len(scene_predictions), max(0, target_count - 1))
            for item in scene_predictions[:retained_count]:
                final.append({**item, "id": len(final) + 1})
            for offset in range(target_count - retained_count):
                final.append(
                    {
                        "id": len(final) + 1,
                        "scene_id": scene.scene_id,
                        "frame": offset,
                        "label": "traffic_sign",
                        "bbox": [1 + offset, 1 + offset, 3 + offset, 3 + offset],
                        "source": "manual",
                    }
                )
        return final

    def _managed_scene_video_path(
        self,
        scene: Scene,
        *,
        require_file: bool,
    ) -> Path:
        scenes_root = self.store.scenes_dir
        if scenes_root.is_symlink():
            raise ValueError("managed scenes root must not be a symbolic link")
        scenes_root = scenes_root.resolve(strict=True)
        raw_path = Path(scene.video_path).expanduser()
        scene_path = Path(os.path.abspath(os.fspath(raw_path)))
        if scene_path == scenes_root or not scene_path.is_relative_to(scenes_root):
            raise ValueError("scene video escapes the managed scenes directory")
        expected_prefix = f"{scene.session_id}_"
        if (
            not scene.scene_id.startswith(expected_prefix)
            or not scene.scene_id.removeprefix(expected_prefix)
        ):
            raise ValueError("scene id is not bound to its session")
        expected_path = (
            scenes_root
            / scene.session_id
            / f"{scene.scene_id.removeprefix(expected_prefix)}.mp4"
        )
        if scene_path != expected_path:
            raise ValueError("scene video path does not match its session and scene id")
        self._reject_symlink_components(scenes_root, scene_path)
        resolved = scene_path.resolve(strict=False)
        if not resolved.is_relative_to(scenes_root):
            raise ValueError("scene video resolves outside the managed scenes directory")
        if require_file and not scene_path.is_file():
            raise ValueError("scene video is not a regular file")
        return scene_path

    def _managed_sidecar_path(self, scene: Scene, kind: str) -> Path:
        suffixes = {
            "prediction": ".predictions.json",
            "final": ".final.json",
        }
        if kind not in suffixes:
            raise ValueError("unknown annotation sidecar kind")
        scene_path = self._managed_scene_video_path(scene, require_file=False)
        scenes_root = self.store.scenes_dir.resolve(strict=True)
        sidecar_path = scene_path.with_suffix(suffixes[kind])
        if sidecar_path == scenes_root or not sidecar_path.is_relative_to(scenes_root):
            raise ValueError(f"{kind} sidecar escapes the managed scenes directory")
        self._reject_symlink_components(scenes_root, sidecar_path)
        resolved = sidecar_path.resolve(strict=False)
        if not resolved.is_relative_to(scenes_root):
            raise ValueError(f"{kind} sidecar resolves outside the managed scenes directory")
        return sidecar_path

    @staticmethod
    def _reject_symlink_components(root: Path, path: Path) -> None:
        cursor = root
        for part in path.relative_to(root).parts:
            cursor /= part
            if cursor.is_symlink():
                raise ValueError(f"managed path contains a symbolic link: {cursor}")

    @staticmethod
    def _stage_json_sidecar(path: Path, payload: Any) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.",
            dir=path.parent,
        )
        temporary_path = Path(temporary)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
        return temporary_path

    @staticmethod
    def _sidecar_path_failure(kind: str, scene: Scene, error: Exception) -> ToolResult:
        return ToolResult.failure(
            f"{kind.upper()}_SIDECAR_PATH_INVALID",
            f"{kind.title()} sidecar path is invalid for scene {scene.scene_id}: {error}",
        )

    @staticmethod
    def _scene_path_failure(scene: Scene, error: Exception) -> ToolResult:
        return ToolResult.failure(
            "SCENE_VIDEO_PATH_INVALID",
            f"Scene video path is invalid for scene {scene.scene_id}: {error}",
        )

    def _load_release_sidecars(self, session: Session) -> ToolResult:
        predictions = self._load_counted_sidecars(session, "prediction")
        if not predictions.ok:
            return predictions
        final = self._load_counted_sidecars(session, "final", completed_only=True)
        if not final.ok:
            return final
        return ToolResult.success(
            {
                "annotations": list(final.data["annotations"]),
                "predictions": list(predictions.data["annotations"]),
            }
        )

    def _load_counted_sidecars(
        self,
        session: Session,
        kind: str,
        *,
        completed_only: bool = False,
    ) -> ToolResult:
        if kind not in {"prediction", "final"}:
            return ToolResult.failure("SIDECAR_KIND_INVALID", "Unknown annotation sidecar kind")
        annotations: list[dict[str, Any]] = []
        for scene in session.scenes:
            if completed_only and scene.status != "completed":
                continue
            try:
                path = self._managed_sidecar_path(scene, kind)
            except (OSError, RuntimeError, ValueError) as error:
                return self._sidecar_path_failure(kind, scene, error)
            if not path.is_file():
                return ToolResult.failure(
                    f"{kind.upper()}_SIDECAR_MISSING",
                    f"{kind.title()} annotation sidecar is missing for scene {scene.scene_id}",
                )
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return ToolResult.failure(
                    f"{kind.upper()}_SIDECAR_INVALID",
                    f"{kind.title()} annotation sidecar is invalid for scene {scene.scene_id}",
                )
            if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
                return ToolResult.failure(
                    f"{kind.upper()}_SIDECAR_INVALID",
                    f"{kind.title()} annotation sidecar must be a list for scene {scene.scene_id}",
                )
            expected_count = scene.prediction_count if kind == "prediction" else scene.final_count
            if expected_count is None or expected_count != len(payload):
                return ToolResult.failure(
                    f"{kind.upper()}_COUNT_MISMATCH",
                    f"{kind.title()} object count does not match sidecar for scene {scene.scene_id}",
                )
            if any(item.get("scene_id") != scene.scene_id for item in payload):
                return ToolResult.failure(
                    f"{kind.upper()}_SIDECAR_LINEAGE_INVALID",
                    f"{kind.title()} annotations have invalid scene lineage for {scene.scene_id}",
                )
            annotations.extend(payload)
        return ToolResult.success({"annotations": annotations})
