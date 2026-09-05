import json
from pathlib import Path

from roadlabelops.models import Scene, Session, Stage, ToolResult
from roadlabelops.permissions import Decision, PermissionPolicy, PermissionResult
from roadlabelops.runtime import WorkflowRuntime
from roadlabelops.storage import LocalStore


class FailSecondTaskOnce:
    def __init__(self) -> None:
        self.create_calls: list[str] = []
        self.failed = False

    def ensure_project(self, name: str, labels: list[str]) -> ToolResult:
        return ToolResult.success({"project_id": 70, "created": False})

    def create_task(self, name: str, project_id: int, video_path: str) -> ToolResult:
        self.create_calls.append(name)
        if name.endswith("002") and not self.failed:
            self.failed = True
            return ToolResult.failure(
                "CVAT_TASK_FAILED",
                "temporary CVAT failure",
                retryable=True,
            )
        task_id = 701 if name.endswith("001") else 702
        return ToolResult.success({"task_id": task_id, "job_ids": [task_id + 1000]})


class AskForTaskCreation(PermissionPolicy):
    def check(
        self,
        action: str,
        target: str | Path | None = None,
        *,
        exists: bool = False,
    ) -> PermissionResult:
        if action == "create_tasks":
            return PermissionResult(Decision.ASK, "task creation needs approval")
        return super().check(action, target, exists=exists)


class FailSecondImportOnce:
    def __init__(self) -> None:
        self.import_calls: list[int] = []
        self.failed = False

    def import_predictions(
        self,
        task_id: int,
        predictions: list[dict[str, object]],
    ) -> ToolResult:
        self.import_calls.append(task_id)
        if task_id == 702 and not self.failed:
            self.failed = True
            return ToolResult.failure(
                "CVAT_ANNOTATION_IMPORT_FAILED",
                "temporary CVAT failure",
                retryable=True,
            )
        return ToolResult.success({"task_id": task_id, "annotation_count": len(predictions)})


class ExistingAutoAnnotations:
    def __init__(self) -> None:
        self.replacement_count = 0

    def import_predictions(
        self,
        task_id: int,
        predictions: list[dict[str, object]],
        *,
        allow_replace_auto: bool = False,
    ) -> ToolResult:
        if not allow_replace_auto:
            return ToolResult.failure(
                "AUTO_ANNOTATIONS_EXIST",
                "CVAT already contains generated annotations",
            )
        self.replacement_count += 1
        return ToolResult.success({"task_id": task_id, "annotation_count": len(predictions)})


def _make_real_session(store: LocalStore) -> Session:
    session_id = "session_recovery"
    scene_directory = store.scenes_dir / session_id
    scene_directory.mkdir(parents=True, exist_ok=True)
    for index in range(1, 3):
        (scene_directory / f"scene_{index:03d}.mp4").write_bytes(b"scene")
    session = Session(
        session_id=session_id,
        name="Recovery fixture",
        source_path=str(store.raw_dir / "source.mp4"),
        source_sha256="a" * 64,
        duration_seconds=30,
        fps=25,
        width=1280,
        height=720,
        status=Stage.SLICED,
        scenes=[
            Scene(
                scene_id=f"{session_id}_scene_{index:03d}",
                session_id=session_id,
                start_seconds=(index - 1) * 15,
                end_seconds=index * 15,
                video_path=str(store.scenes_dir / session_id / f"scene_{index:03d}.mp4"),
            )
            for index in range(1, 3)
        ],
    )
    store.save_session(session)
    return session


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_retryable_action_recovers_after_restart_without_repeating_side_effects(
    tmp_path: Path,
) -> None:
    store = LocalStore(tmp_path / "data", tmp_path / "runtime")
    session = _make_real_session(store)
    cvat = FailSecondTaskOnce()
    runtime = WorkflowRuntime(store, cvat=cvat)

    failed = runtime.advance(session.session_id, "create_tasks")

    assert not failed.ok
    assert failed.retryable
    persisted = store.get_session(session.session_id)
    assert persisted.status is Stage.FAILED_RETRYABLE
    assert persisted.resume_stage == Stage.SLICED.value
    assert persisted.pending_action == "create_tasks"
    assert persisted.last_error == failed.error
    assert persisted.scenes[0].cvat_task_id == 701
    assert persisted.scenes[1].cvat_task_id is None
    assert cvat.create_calls == [
        "session_recovery_scene_001",
        "session_recovery_scene_002",
    ]

    journal = store.read_journal()
    started = next(item for item in journal if item["event"] == "stage.started")
    stage_failed = next(item for item in journal if item["event"] == "stage.failed")
    assert started["trace_id"] == stage_failed["trace_id"]
    assert stage_failed["retryable"] is True
    assert stage_failed["recovery_status"] == Stage.FAILED_RETRYABLE.value

    tool_events = _read_jsonl(store.runtime_root.parent / "logs" / "roadlabelops.jsonl")
    assert [item["event"] for item in tool_events] == ["before", "failed", "after"]
    assert {item["trace_id"] for item in tool_events} == {started["trace_id"]}

    restarted = WorkflowRuntime(store, cvat=cvat)
    recovered = restarted.advance(session.session_id, "create_tasks")

    assert recovered.ok, recovered.error
    persisted = store.get_session(session.session_id)
    assert persisted.status is Stage.TASKS_CREATED
    assert persisted.resume_stage is None
    assert persisted.pending_action is None
    assert persisted.last_error is None
    assert [scene.cvat_task_id for scene in persisted.scenes] == [701, 702]
    assert cvat.create_calls == [
        "session_recovery_scene_001",
        "session_recovery_scene_002",
        "session_recovery_scene_002",
    ]


def test_permission_waiting_requires_same_action_and_can_resume_when_approved(
    tmp_path: Path,
) -> None:
    store = LocalStore(tmp_path / "data", tmp_path / "runtime")
    policy = AskForTaskCreation(store.root)
    runtime = WorkflowRuntime(store, policy=policy)
    session = runtime.create_demo()

    waiting = runtime.advance(session.session_id, "create_tasks")

    assert not waiting.ok
    assert waiting.error["code"] == "PERMISSION_REQUIRED"
    persisted = store.get_session(session.session_id)
    assert persisted.status is Stage.WAITING_FOR_PERMISSION
    assert persisted.resume_stage == Stage.SLICED.value
    assert persisted.pending_action == "create_tasks"
    assert all(scene.cvat_task_id is None for scene in persisted.scenes)

    wrong_action = runtime.advance(session.session_id, "preannotate", approved=True)
    assert not wrong_action.ok
    assert wrong_action.error["code"] == "INVALID_TRANSITION"
    assert store.get_session(session.session_id).status is Stage.WAITING_FOR_PERMISSION

    resumed = runtime.advance(session.session_id, "create_tasks", approved=True)

    assert resumed.ok, resumed.error
    persisted = store.get_session(session.session_id)
    assert persisted.status is Stage.TASKS_CREATED
    assert persisted.resume_stage is None
    assert persisted.pending_action is None
    assert persisted.last_error is None
    assert all(scene.cvat_task_id is not None for scene in persisted.scenes)


def test_preannotation_recovery_skips_scenes_already_imported_to_cvat(tmp_path: Path) -> None:
    store = LocalStore(tmp_path / "data", tmp_path / "runtime")
    session = _make_real_session(store)
    session.status = Stage.TASKS_CREATED
    for task_id, scene in zip((701, 702), session.scenes, strict=True):
        scene.cvat_task_id = task_id
    store.save_session(session)
    cvat = FailSecondImportOnce()
    detector_calls: list[str] = []

    def detector(video_path: Path | str, frame_step: int) -> ToolResult:
        detector_calls.append(str(video_path))
        return ToolResult.success(
            {"predictions": [{"frame": 0, "label": "car", "bbox": [1, 2, 3, 4], "confidence": 0.9}]}
        )

    runtime = WorkflowRuntime(store, cvat=cvat, detector=detector)
    failed = runtime.advance(session.session_id, "preannotate")

    assert not failed.ok
    persisted = store.get_session(session.session_id)
    assert persisted.status is Stage.FAILED_RETRYABLE
    assert persisted.scenes[0].prediction_count == 1
    assert persisted.scenes[1].prediction_count is None
    assert cvat.import_calls == [701, 702]
    assert len(detector_calls) == 2

    restarted = WorkflowRuntime(store, cvat=cvat, detector=detector)
    recovered = restarted.advance(session.session_id, "preannotate")

    assert recovered.ok, recovered.error
    persisted = store.get_session(session.session_id)
    assert persisted.status is Stage.PREANNOTATED
    assert [scene.prediction_count for scene in persisted.scenes] == [1, 1]
    assert cvat.import_calls == [701, 702, 702]
    assert len(detector_calls) == 3


def test_non_retryable_failure_keeps_original_stage_available_for_retry(tmp_path: Path) -> None:
    store = LocalStore(tmp_path / "data", tmp_path / "runtime")
    session = _make_real_session(store)
    runtime = WorkflowRuntime(store)

    result = runtime.advance(session.session_id, "create_tasks")

    assert not result.ok
    assert result.error["code"] == "CVAT_NOT_CONFIGURED"
    persisted = store.get_session(session.session_id)
    assert persisted.status is Stage.SLICED
    assert persisted.pending_action is None
    assert persisted.resume_stage is None
    assert persisted.last_error == result.error


def test_replacing_existing_auto_annotations_requires_persisted_approval(
    tmp_path: Path,
) -> None:
    store = LocalStore(tmp_path / "data", tmp_path / "runtime")
    session = _make_real_session(store)
    session.status = Stage.TASKS_CREATED
    session.scenes = session.scenes[:1]
    session.scenes[0].cvat_task_id = 701
    store.save_session(session)
    prediction_path = Path(session.scenes[0].video_path).with_suffix(".predictions.json")
    previous_snapshot = [
        {
            "scene_id": session.scenes[0].scene_id,
            "frame": 0,
            "label": "pedestrian",
            "bbox": [9, 8, 7, 6],
        }
    ]
    store.write_json_atomic(prediction_path, previous_snapshot)
    cvat = ExistingAutoAnnotations()

    def detector(_video_path: Path | str, _frame_step: int) -> ToolResult:
        return ToolResult.success(
            {"predictions": [{"frame": 0, "label": "car", "bbox": [1, 2, 3, 4], "confidence": 0.9}]}
        )

    runtime = WorkflowRuntime(store, cvat=cvat, detector=detector)
    waiting = runtime.advance(session.session_id, "preannotate")

    assert not waiting.ok
    assert waiting.error and waiting.error["code"] == "PERMISSION_REQUIRED"
    persisted = store.get_session(session.session_id)
    assert persisted.status is Stage.WAITING_FOR_PERMISSION
    assert persisted.pending_action == "preannotate"
    assert persisted.scenes[0].prediction_count is None
    assert cvat.replacement_count == 0
    assert json.loads(prediction_path.read_text(encoding="utf-8")) == previous_snapshot
    assert list(prediction_path.parent.glob(f".{prediction_path.name}.*")) == []

    restarted = WorkflowRuntime(store, cvat=cvat, detector=detector)
    resumed = restarted.advance(session.session_id, "preannotate", approved=True)

    assert resumed.ok, resumed.error
    assert store.get_session(session.session_id).status is Stage.PREANNOTATED
    assert cvat.replacement_count == 1
    assert json.loads(prediction_path.read_text(encoding="utf-8")) != previous_snapshot


def test_create_task_rejects_scene_path_outside_managed_scenes(tmp_path: Path) -> None:
    store = LocalStore(tmp_path / "data", tmp_path / "runtime")
    session = _make_real_session(store)
    outside_video = tmp_path / "outside.mp4"
    outside_video.write_bytes(b"outside")
    session.scenes[0].video_path = str(outside_video)
    session.scenes = session.scenes[:1]
    store.save_session(session)
    cvat = FailSecondTaskOnce()

    result = WorkflowRuntime(store, cvat=cvat).advance(
        session.session_id,
        "create_tasks",
    )

    assert not result.ok
    assert result.error and result.error["code"] == "SCENE_VIDEO_PATH_INVALID"
    assert cvat.create_calls == []


def test_create_task_rejects_cross_session_scene_path(tmp_path: Path) -> None:
    store = LocalStore(tmp_path / "data", tmp_path / "runtime")
    session = _make_real_session(store)
    other_directory = store.scenes_dir / "session_other"
    other_directory.mkdir()
    other_video = other_directory / "scene_001.mp4"
    other_video.write_bytes(b"other scene")
    session.scenes[0].video_path = str(other_video)
    store.save_session(session)

    result = WorkflowRuntime(store, cvat=FailSecondTaskOnce()).advance(
        session.session_id,
        "create_tasks",
    )

    assert not result.ok
    assert result.error and result.error["code"] == "SCENE_VIDEO_PATH_INVALID"


def test_detector_rejects_symlinked_scene_video_before_local_read(tmp_path: Path) -> None:
    store = LocalStore(tmp_path / "data", tmp_path / "runtime")
    session = _make_real_session(store)
    outside_video = tmp_path / "outside.mp4"
    outside_video.write_bytes(b"outside")
    scene_path = Path(session.scenes[0].video_path)
    scene_path.unlink()
    scene_path.symlink_to(outside_video)
    session.status = Stage.TASKS_CREATED
    session.scenes = session.scenes[:1]
    session.scenes[0].cvat_task_id = 701
    store.save_session(session)
    detector_calls: list[str] = []

    def detector(video_path: Path | str, _frame_step: int) -> ToolResult:
        detector_calls.append(str(video_path))
        return ToolResult.success({"predictions": []})

    result = WorkflowRuntime(
        store,
        cvat=ExistingAutoAnnotations(),
        detector=detector,
    ).advance(session.session_id, "preannotate")

    assert not result.ok
    assert result.error and result.error["code"] == "SCENE_VIDEO_PATH_INVALID"
    assert detector_calls == []


def test_preannotation_rejects_symlinked_prediction_sidecar(tmp_path: Path) -> None:
    store = LocalStore(tmp_path / "data", tmp_path / "runtime")
    session = _make_real_session(store)
    session.status = Stage.TASKS_CREATED
    session.scenes = session.scenes[:1]
    session.scenes[0].cvat_task_id = 701
    store.save_session(session)
    outside_snapshot = tmp_path / "outside-predictions.json"
    outside_snapshot.write_text('[{"sentinel": true}]', encoding="utf-8")
    sidecar = Path(session.scenes[0].video_path).with_suffix(".predictions.json")
    sidecar.symlink_to(outside_snapshot)
    detector_calls: list[str] = []

    def detector(video_path: Path | str, _frame_step: int) -> ToolResult:
        detector_calls.append(str(video_path))
        return ToolResult.success({"predictions": []})

    result = WorkflowRuntime(
        store,
        cvat=ExistingAutoAnnotations(),
        detector=detector,
    ).advance(session.session_id, "preannotate")

    assert not result.ok
    assert result.error and result.error["code"] == "PREDICTION_SIDECAR_PATH_INVALID"
    assert detector_calls == []
    assert json.loads(outside_snapshot.read_text(encoding="utf-8")) == [{"sentinel": True}]


def test_review_rejects_symlinked_final_sidecar_before_cvat_read(tmp_path: Path) -> None:
    class TrackingReview:
        def __init__(self) -> None:
            self.calls = 0

        def get_review_result(self, _task_id: int) -> ToolResult:
            self.calls += 1
            return ToolResult.success({"completed": True, "annotations": [], "jobs": []})

    store = LocalStore(tmp_path / "data", tmp_path / "runtime")
    session = _make_real_session(store)
    session.status = Stage.WAITING_FOR_HUMAN_REVIEW
    session.scenes = session.scenes[:1]
    session.scenes[0].cvat_task_id = 701
    store.save_session(session)
    outside_snapshot = tmp_path / "outside-final.json"
    outside_snapshot.write_text('[{"sentinel": true}]', encoding="utf-8")
    sidecar = Path(session.scenes[0].video_path).with_suffix(".final.json")
    sidecar.symlink_to(outside_snapshot)
    cvat = TrackingReview()

    result = WorkflowRuntime(store, cvat=cvat).advance(
        session.session_id,
        "complete_review",
    )

    assert not result.ok
    assert result.error and result.error["code"] == "FINAL_SIDECAR_PATH_INVALID"
    assert cvat.calls == 0
    assert json.loads(outside_snapshot.read_text(encoding="utf-8")) == [{"sentinel": True}]
