import json
from pathlib import Path

import pytest

import roadlabelops.tools.release as release_module
from roadlabelops.models import Scene, Session, Stage, ToolResult
from roadlabelops.runtime import WorkflowRuntime
from roadlabelops.storage import LocalStore
from roadlabelops.tools.release import build_coco_release


def test_demo_workflow_persists_and_rejects_invalid_transition(tmp_path: Path) -> None:
    store = LocalStore(tmp_path / "data", tmp_path / "runtime")
    runtime = WorkflowRuntime(store)
    session = runtime.create_demo()
    assert store.get_session(session.session_id).status is Stage.SLICED
    invalid = runtime.advance(session.session_id, "release")
    assert not invalid.ok
    assert invalid.error["code"] == "INVALID_TRANSITION"


def test_demo_workflow_reaches_release(tmp_path: Path) -> None:
    store = LocalStore(tmp_path / "data", tmp_path / "runtime")
    runtime = WorkflowRuntime(store)
    session = runtime.create_demo()
    for action in (
        "create_tasks",
        "preannotate",
        "request_review",
        "complete_review",
        "calculate_quality",
        "release",
    ):
        result = runtime.advance(session.session_id, action)
        assert result.ok, result.error
    assert store.get_session(session.session_id).status is Stage.RELEASED
    duplicate = runtime.advance(session.session_id, "release")
    assert not duplicate.ok


def test_dashboard_ignores_quality_sidecar_files(tmp_path: Path) -> None:
    store = LocalStore(tmp_path / "data", tmp_path / "runtime")
    runtime = WorkflowRuntime(store)
    session = runtime.create_demo()
    for action in (
        "create_tasks",
        "preannotate",
        "request_review",
        "complete_review",
        "calculate_quality",
    ):
        assert runtime.advance(session.session_id, action).ok
    dashboard = runtime.dashboard()
    assert dashboard["summary"]["session_count"] == 1
    assert dashboard["quality"]["final_count"] > 0
    assert dashboard["qualities"][session.session_id]["final_count"] > 0


def test_dashboard_keeps_historical_sessions_addressable(tmp_path: Path) -> None:
    store = LocalStore(tmp_path / "data", tmp_path / "runtime")
    runtime = WorkflowRuntime(store)
    session_ids = {runtime.create_demo().session_id for _ in range(10)}

    dashboard = runtime.dashboard()

    assert dashboard["summary"]["session_count"] == 10
    assert {session["session_id"] for session in dashboard["sessions"]} == session_ids


def test_dashboard_exposes_verified_release_receipt_per_session(tmp_path: Path) -> None:
    store = LocalStore(tmp_path / "data", tmp_path / "runtime")
    runtime = WorkflowRuntime(store)
    session = runtime.create_demo()
    for action in (
        "create_tasks",
        "preannotate",
        "request_review",
        "complete_review",
        "calculate_quality",
        "release",
    ):
        assert runtime.advance(session.session_id, action).ok

    receipt = runtime.dashboard()["releases"][session.session_id]

    assert receipt["verified"] is True
    assert receipt["release_id"] == f"{session.session_id}-v1.0.0"
    assert receipt["manifest_sha256"]
    assert receipt["file_count"] > 0


def test_release_reverification_persists_tamper_result(tmp_path: Path) -> None:
    store = LocalStore(tmp_path / "data", tmp_path / "runtime")
    runtime = WorkflowRuntime(store)
    session = runtime.create_demo()
    for action in (
        "create_tasks",
        "preannotate",
        "request_review",
        "complete_review",
        "calculate_quality",
        "release",
    ):
        assert runtime.advance(session.session_id, action).ok
    release_path = store.releases_dir / f"{session.session_id}-v1.0.0"
    (release_path / "annotations.coco.json").write_text("tampered", encoding="utf-8")

    verified = runtime.verify_release(session.session_id)

    assert not verified.ok
    receipt = runtime.dashboard()["releases"][session.session_id]
    assert receipt["verified"] is False
    assert any("hash mismatch" in message for message in receipt["errors"])


def test_release_reverification_uses_trusted_receipt_hash(tmp_path: Path) -> None:
    store = LocalStore(tmp_path / "data", tmp_path / "runtime")
    runtime = WorkflowRuntime(store)
    session = runtime.create_demo()
    for action in (
        "create_tasks",
        "preannotate",
        "request_review",
        "complete_review",
        "calculate_quality",
        "release",
    ):
        assert runtime.advance(session.session_id, action).ok
    release_path = store.releases_dir / f"{session.session_id}-v1.0.0"
    receipt_path = release_path / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["untrusted_note"] = "tampered after publication"
    store.write_json_atomic(receipt_path, receipt)

    verified = runtime.verify_release(session.session_id)

    assert not verified.ok
    assert verified.error and verified.error["code"] == "RECEIPT_SHA256_MISMATCH"
    summary = verified.data["receipt"]
    assert summary["verified"] is False
    assert any("Receipt hash" in message for message in summary["errors"])
    assert any("canonical integrity document" in message for message in summary["errors"])


def _advance_demo_to_quality(runtime: WorkflowRuntime) -> str:
    session = runtime.create_demo()
    for action in (
        "create_tasks",
        "preannotate",
        "request_review",
        "complete_review",
        "calculate_quality",
    ):
        assert runtime.advance(session.session_id, action).ok
    return session.session_id


def test_release_recovers_when_process_stopped_after_atomic_publication(tmp_path: Path) -> None:
    store = LocalStore(tmp_path / "data", tmp_path / "runtime")
    runtime = WorkflowRuntime(store)
    session_id = _advance_demo_to_quality(runtime)
    session = store.get_session(session_id)
    quality = json.loads(
        (store.sessions_dir / f"{session_id}.quality.json").read_text(encoding="utf-8")
    )
    published = build_coco_release(
        store,
        session,
        "1.0.0",
        runtime._demo_final(session),
        quality_report=quality,
        predictions=runtime._demo_predictions(session),
        evaluated_frame_keys=runtime._evaluated_frame_keys(session),
        accepted_jobs=len(session.scenes),
        reviewed_jobs=len(session.scenes),
        first_pass_acceptance_reason=None,
    )
    assert published.ok, published.error
    assert not (store.sessions_dir / f"{session_id}.release.json").exists()

    recovered = runtime.advance(session_id, "release", version="1.0.0")

    assert recovered.ok, recovered.error
    assert store.get_session(session_id).status is Stage.RELEASED
    assert recovered.data["receipt"]["verified"] is True


def test_release_recovery_rejects_pre_anchor_lineage_forgery(tmp_path: Path) -> None:
    store = LocalStore(tmp_path / "data", tmp_path / "runtime")
    runtime = WorkflowRuntime(store)
    session_id = _advance_demo_to_quality(runtime)
    session = store.get_session(session_id)
    predictions = runtime._demo_predictions(session)
    final = runtime._demo_final(session)
    quality = json.loads(
        (store.sessions_dir / f"{session_id}.quality.json").read_text(encoding="utf-8")
    )
    published = build_coco_release(
        store,
        session,
        "1.0.0",
        final,
        quality_report=quality,
        predictions=predictions,
        evaluated_frame_keys=runtime._evaluated_frame_keys(session),
        accepted_jobs=len(session.scenes),
        reviewed_jobs=len(session.scenes),
        first_pass_acceptance_reason=None,
    )
    assert published.ok, published.error
    release = Path(published.data["path"])
    manifest_path = release / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["scene_lineage"][0]["cvat_task_id"] += 10_000
    manifest["cvat_task_ids"] = [
        item["cvat_task_id"]
        for item in manifest["scene_lineage"]
        if item["cvat_task_id"] is not None
    ]
    store.write_json_atomic(manifest_path, manifest)
    store.write_json_atomic(
        release / "receipt.json",
        release_module._receipt(manifest["release_id"], manifest, manifest_path),
    )
    assert release_module.verify_coco_release(release).ok

    recovered = runtime.advance(session_id, "release", version="1.0.0")

    assert not recovered.ok
    assert recovered.error and recovered.error["code"] == (
        "RECOVERABLE_RELEASE_LINEAGE_MISMATCH"
    )
    assert not (store.sessions_dir / f"{session_id}.release.json").exists()
    assert store.get_session(session_id).status is not Stage.RELEASED


def test_release_recovers_after_release_record_but_before_session_save(
    tmp_path: Path, monkeypatch
) -> None:
    store = LocalStore(tmp_path / "data", tmp_path / "runtime")
    runtime = WorkflowRuntime(store)
    session_id = _advance_demo_to_quality(runtime)
    original_save = store.save_session

    def fail_released_save(session: Session):
        if session.status is Stage.RELEASED:
            raise OSError("simulated crash before session commit")
        return original_save(session)

    monkeypatch.setattr(store, "save_session", fail_released_save)
    with pytest.raises(OSError, match="simulated crash"):
        runtime.advance(session_id, "release", version="1.0.0")
    assert store.get_session(session_id).status is Stage.QUALITY_CALCULATED
    assert (store.sessions_dir / f"{session_id}.release.json").is_file()

    monkeypatch.setattr(store, "save_session", original_save)
    recovered = runtime.advance(session_id, "release", version="1.0.0")

    assert recovered.ok, recovered.error
    assert store.get_session(session_id).status is Stage.RELEASED


def test_released_session_missing_trust_anchor_fails_closed(tmp_path: Path) -> None:
    store = LocalStore(tmp_path / "data", tmp_path / "runtime")
    runtime = WorkflowRuntime(store)
    session_id = _advance_demo_to_quality(runtime)
    assert runtime.advance(session_id, "release").ok
    anchor = store.sessions_dir / f"{session_id}.release.json"
    anchor.unlink()

    verified = runtime.verify_release(session_id)
    dashboard_receipt = runtime.dashboard()["releases"][session_id]

    assert not verified.ok
    assert verified.error and verified.error["code"] == "TRUST_ANCHOR_MISSING"
    assert dashboard_receipt["verified"] is False
    assert "automatic re-anchoring is denied" in dashboard_receipt["errors"][0]
    assert not anchor.exists()


def test_real_release_requires_each_completed_final_sidecar(tmp_path: Path) -> None:
    store = LocalStore(tmp_path / "data", tmp_path / "runtime")
    runtime = WorkflowRuntime(store)
    source = store.raw_dir / "source.mp4"
    source.write_bytes(b"source")
    scene_path = store.scenes_dir / "session_real" / "scene_001.mp4"
    scene_path.parent.mkdir(parents=True)
    scene_path.write_bytes(b"scene")
    session = Session(
        session_id="session_real",
        name="real",
        source_path=str(source),
        source_sha256="0" * 64,
        duration_seconds=1,
        fps=1,
        width=32,
        height=32,
        status=Stage.QUALITY_CALCULATED,
        scenes=[
            Scene(
                "session_real_scene_001",
                "session_real",
                0,
                1,
                str(scene_path),
                cvat_task_id=99,
                cvat_job_ids=[199],
                status="completed",
                prediction_count=0,
                final_count=0,
            )
        ],
    )
    store.save_session(session)
    store.write_json_atomic(scene_path.with_suffix(".predictions.json"), [])
    store.write_json_atomic(
        store.sessions_dir / "session_real.quality.json",
        {"prediction_count": 0, "final_count": 0},
    )

    released = runtime.advance(session.session_id, "release")

    assert not released.ok
    assert released.error and released.error["code"] == "FINAL_SIDECAR_MISSING"


def test_review_sync_persists_scene_tags_in_session_lineage(tmp_path: Path) -> None:
    class ReviewedCvat:
        def get_review_result(self, task_id: int):
            return ToolResult.success(
                {
                    "completed": True,
                    "jobs": [{"job_id": task_id + 1000, "status": "completed"}],
                    "annotations": [],
                    "scene_tags": [
                        {
                            "frame": 0,
                            "label": "weather",
                            "source": "manual",
                            "attributes": [{"name": "weather", "value": "rain"}],
                        }
                    ],
                }
            )

    store = LocalStore(tmp_path / "data", tmp_path / "runtime")
    scene_path = store.scenes_dir / "session_tags" / "scene_001.mp4"
    scene_path.parent.mkdir(parents=True)
    session = Session(
        session_id="session_tags",
        name="tags",
        source_path=str(store.raw_dir / "source.mp4"),
        source_sha256="a" * 64,
        duration_seconds=1,
        fps=25,
        width=32,
        height=32,
        status=Stage.WAITING_FOR_HUMAN_REVIEW,
        scenes=[
            Scene(
                "session_tags_scene_001",
                "session_tags",
                0,
                1,
                str(scene_path),
                cvat_task_id=91,
            )
        ],
    )
    store.save_session(session)
    runtime = WorkflowRuntime(store, cvat=ReviewedCvat())

    result = runtime.advance(session.session_id, "complete_review")

    assert result.ok, result.error
    persisted = store.get_session(session.session_id)
    assert persisted.scenes[0].scene_tags[0]["scene_id"] == "session_tags_scene_001"
    assert persisted.scenes[0].scene_tags[0]["label"] == "weather"


def test_quality_fails_closed_when_a_real_prediction_sidecar_is_missing(tmp_path: Path) -> None:
    store = LocalStore(tmp_path / "data", tmp_path / "runtime")
    scene_path = store.scenes_dir / "session_quality" / "scene_001.mp4"
    scene_path.parent.mkdir(parents=True)
    scene = Scene(
        "session_quality_scene_001",
        "session_quality",
        0,
        1,
        str(scene_path),
        status="completed",
        prediction_count=0,
        final_count=0,
    )
    session = Session(
        session_id="session_quality",
        name="quality",
        source_path=str(store.raw_dir / "source.mp4"),
        source_sha256="a" * 64,
        duration_seconds=1,
        fps=25,
        width=32,
        height=32,
        status=Stage.REVIEW_COMPLETED,
        scenes=[scene],
    )
    store.save_session(session)
    store.write_json_atomic(scene_path.with_suffix(".final.json"), [])

    result = WorkflowRuntime(store).advance(session.session_id, "calculate_quality")

    assert not result.ok
    assert result.error and result.error["code"] == "PREDICTION_SIDECAR_MISSING"
    assert store.get_session(session.session_id).status is Stage.REVIEW_COMPLETED
