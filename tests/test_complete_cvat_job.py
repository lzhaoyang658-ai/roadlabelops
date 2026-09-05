from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.apply_final_review import post_apply_canonical_sha256
from scripts.complete_cvat_job import (
    JobCompletionError,
    complete_job,
    validate_completion_evidence,
    write_completion_receipt,
)
from scripts.snapshot_cvat_task import canonical_sha256, canonicalize_annotations

POST_FILE_SHA = "c" * 64
DECISIONS_FILE_SHA = "d" * 64
BACKUP_FILE_SHA = "e" * 64
LOG_FILE_SHA = "f" * 64
SOURCE_SNAPSHOT_SHA = "a" * 64
REVIEW_PACK_SHA = "b" * 64


def rectangle(*, identifier: int | None, frame: int, label_id: int, points: list[float]) -> dict:
    return {
        "type": "rectangle",
        "label_id": label_id,
        "frame": frame,
        "occluded": False,
        "outside": False,
        "z_order": 0,
        "rotation": 0.0,
        "points": points,
        "id": identifier,
        "group": 0,
        "source": "manual",
        "attributes": [],
        "score": 1.0,
        "elements": [],
    }


def build_evidence(*, frame_count: int = 50) -> tuple[dict, dict, dict, dict, dict, dict]:
    original = rectangle(identifier=100, frame=0, label_id=1, points=[1.0, 2.0, 20.0, 22.0])
    addition = rectangle(identifier=None, frame=0, label_id=2, points=[30.0, 3.0, 50.0, 25.0])
    backup_annotations = {
        "version": 4,
        "tags": [],
        "shapes": [original],
        "tracks": [],
        "intervals": [],
    }
    replayed_annotations = copy.deepcopy(backup_annotations)
    replayed_annotations["shapes"].append(addition)
    original_ids = {("int", "100")}
    expected_post_sha = post_apply_canonical_sha256(
        replayed_annotations, original_shape_ids=original_ids
    )
    post_annotations = copy.deepcopy(replayed_annotations)
    post_annotations["version"] = 5
    post_annotations["shapes"][1]["id"] = 200
    post_annotation_sha = canonical_sha256(canonicalize_annotations(post_annotations))

    labels = [{"id": 1, "name": "car"}, {"id": 2, "name": "bus"}]
    images = [{"cvat_frame": frame, "width": 100, "height": 80} for frame in range(frame_count)]
    source_snapshot = {
        "snapshot_schema": {"name": "roadlabelops.cvat-task-snapshot", "version": 1},
        "task": {"id": 42, "size": frame_count},
        "labels": labels,
        "images": images,
        "annotations": backup_annotations,
        "canonical_annotations_sha256": canonical_sha256(
            canonicalize_annotations(backup_annotations)
        ),
    }
    review_pack = {
        "schema_version": "1.0",
        "task_id": 42,
        "read_only": True,
        "source_snapshot_sha256": SOURCE_SNAPSHOT_SHA,
        "annotation_sha256": source_snapshot["canonical_annotations_sha256"],
        "frames": [
            {
                "frame": frame,
                "width": 100,
                "height": 80,
                "shapes": ([{**copy.deepcopy(original), "label": "car"}] if frame == 0 else []),
                "flags": (
                    [
                        {
                            "flag_id": "manual-frame-0",
                            "frame": 0,
                            "type": "manual_class_check",
                            "shape_ids": [],
                        }
                    ]
                    if frame == 0
                    else []
                ),
            }
            for frame in range(frame_count)
        ],
    }
    job = {
        "id": 77,
        "task_id": 42,
        "type": "annotation",
        "stage": "annotation",
        "state": "new",
        "start_frame": 0,
        "stop_frame": frame_count - 1,
        "frame_count": frame_count,
        "issues": {"count": 0},
    }
    post_snapshot = {
        "snapshot_schema": {"name": "roadlabelops.cvat-task-snapshot", "version": 1},
        "task": {"id": 42, "size": frame_count, "status": "annotation"},
        "manifest": {"cvat_task_id": 42, "sample_size": frame_count},
        "labels": labels,
        "jobs": [job],
        "images": images,
        "annotations": post_annotations,
        "canonical_annotations_sha256": post_annotation_sha,
        "counts": {"images": frame_count, "tags": 0, "shapes": 2, "tracks": 0},
        "final_gate": {"passed": True, "blocking_reasons": [], "warnings": []},
    }
    reviews = [
        {
            "frame": frame,
            "reviewed": True,
            "resolved_flag_ids": ["manual-frame-0"] if frame == 0 else [],
            "actions": (
                [{"action": "add", "label": "bus", "points": [30.0, 3.0, 50.0, 25.0]}]
                if frame == 0
                else []
            ),
        }
        for frame in range(frame_count)
    ]
    decisions = {
        "schema_version": "1.3",
        "scope": "full_review",
        "task_id": 42,
        "snapshot_sha256": SOURCE_SNAPSHOT_SHA,
        "review_pack_sha256": REVIEW_PACK_SHA,
        "canonical_annotations_sha256": canonical_sha256(
            canonicalize_annotations(backup_annotations)
        ),
        "mutation_performed": False,
        "frame_reviews": reviews,
    }
    backup = {
        "schema": {"name": "roadlabelops.final-review-backup", "version": 1},
        "task_id": 42,
        "snapshot_file_sha256": SOURCE_SNAPSHOT_SHA,
        "review_pack_file_sha256": REVIEW_PACK_SHA,
        "decisions_file_sha256": DECISIONS_FILE_SHA,
        "canonical_annotations_sha256": decisions["canonical_annotations_sha256"],
        "annotations": backup_annotations,
    }
    initial_normalized_sha = post_apply_canonical_sha256(
        backup_annotations, original_shape_ids=original_ids
    )
    action_log = {
        "schema": {"name": "roadlabelops.final-review-action-log", "version": 1},
        "status": "planned_and_hash_bound_before_write",
        "task_id": 42,
        "snapshot_file_sha256": SOURCE_SNAPSHOT_SHA,
        "review_pack_file_sha256": REVIEW_PACK_SHA,
        "decisions_file_sha256": DECISIONS_FILE_SHA,
        "expected_post_apply_canonical_sha256": expected_post_sha,
        "stage_hashes": {
            "initial": initial_normalized_sha,
            "delete": initial_normalized_sha,
            "update": initial_normalized_sha,
            "add": expected_post_sha,
        },
        "actions": [
            {
                "action_index": 0,
                "action": "add",
                "frame": 0,
                "label": "bus",
                "after_shape": addition,
            }
        ],
    }
    return source_snapshot, review_pack, post_snapshot, decisions, backup, action_log


def build_plan(*, frame_count: int = 50) -> dict:
    return validate_completion_evidence(
        *build_evidence(frame_count=frame_count),
        job_id=77,
        source_snapshot_file_sha256=SOURCE_SNAPSHOT_SHA,
        review_pack_file_sha256=REVIEW_PACK_SHA,
        post_snapshot_file_sha256=POST_FILE_SHA,
        decisions_file_sha256=DECISIONS_FILE_SHA,
        backup_file_sha256=BACKUP_FILE_SHA,
        action_log_file_sha256=LOG_FILE_SHA,
    )


class Record:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def to_dict(self) -> dict:
        return copy.deepcopy(self.payload)


class FakeJob(Record):
    def __init__(self, payload: dict, task_payload: dict) -> None:
        super().__init__(payload)
        self.task_payload = task_payload
        self.update_calls: list[dict] = []
        self.raise_after_update = False

    def update(self, request: object) -> FakeJob:
        values = request.to_dict()
        self.update_calls.append(values)
        assert values == {"state": "completed"}
        self.payload["state"] = "completed"
        self.task_payload["status"] = "completed"
        if self.raise_after_update:
            raise TimeoutError("response lost after commit")
        return self


class FakeTask(Record):
    def __init__(self, snapshot: dict, job: FakeJob) -> None:
        super().__init__(copy.deepcopy(snapshot["task"]))
        self.labels = copy.deepcopy(snapshot["labels"])
        self.annotations = copy.deepcopy(snapshot["annotations"])
        self.job = job
        self.annotation_reads = 0
        self.drift_on_read: int | None = None

    def get_labels(self) -> list[dict]:
        return copy.deepcopy(self.labels)

    def get_annotations(self) -> Record:
        self.annotation_reads += 1
        value = copy.deepcopy(self.annotations)
        if self.drift_on_read == self.annotation_reads:
            value["version"] += 1
        return Record(value)

    def get_jobs(self) -> list[FakeJob]:
        return [self.job]


class FakeClient:
    def __init__(self, snapshot: dict) -> None:
        task_payload = copy.deepcopy(snapshot["task"])
        self.job = FakeJob(copy.deepcopy(snapshot["jobs"][0]), task_payload)
        self.task = FakeTask(snapshot, self.job)
        self.task.payload = task_payload
        self.tasks = SimpleNamespace(retrieve=self._retrieve_task)
        self.jobs = SimpleNamespace(retrieve=self._retrieve_job)
        self.task_reads = 0
        self.job_reads = 0

    def _retrieve_task(self, task_id: int) -> FakeTask:
        assert task_id == 42
        self.task_reads += 1
        return self.task

    def _retrieve_job(self, job_id: int) -> FakeJob:
        assert job_id == 77
        self.job_reads += 1
        return self.job


def test_valid_evidence_binds_full_review_and_replays_post_apply_state() -> None:
    plan = build_plan()

    assert plan["task_id"] == 42
    assert plan["job_id"] == 77
    assert plan["post_annotation_count"] == 2
    assert plan["decision_validation"]["snapshot_frame_count"] == 50
    assert plan["decision_validation"]["reviewed_frame_count"] == 50
    assert plan["decision_validation"]["action_count"] == 1
    assert plan["decision_validation"]["unresolved_manual_flag_count"] == 0


def test_valid_evidence_accepts_synthetic_250_frame_span() -> None:
    plan = build_plan(frame_count=250)

    assert plan["frame_count"] == 250
    assert plan["job_start_frame"] == 0
    assert plan["job_stop_frame"] == 249
    assert plan["decision_validation"]["snapshot_frame_count"] == 250
    assert plan["decision_validation"]["reviewed_frame_count"] == 250


def test_non_contiguous_post_snapshot_frames_are_rejected() -> None:
    source, pack, snapshot, decisions, backup, log = build_evidence(frame_count=250)
    snapshot["images"][-1]["cvat_frame"] = 250

    with pytest.raises(JobCompletionError, match="contiguous CVAT frame span"):
        validate_completion_evidence(
            source,
            pack,
            snapshot,
            decisions,
            backup,
            log,
            job_id=77,
            source_snapshot_file_sha256=SOURCE_SNAPSHOT_SHA,
            review_pack_file_sha256=REVIEW_PACK_SHA,
            post_snapshot_file_sha256=POST_FILE_SHA,
            decisions_file_sha256=DECISIONS_FILE_SHA,
            backup_file_sha256=BACKUP_FILE_SHA,
            action_log_file_sha256=LOG_FILE_SHA,
        )


@pytest.mark.parametrize("field", ["scope", "reviewed", "gate", "status", "binding"])
def test_invalid_evidence_is_rejected(field: str) -> None:
    source, pack, snapshot, decisions, backup, log = build_evidence()
    if field == "scope":
        decisions["scope"] = "automated_risk_cleanup"
    elif field == "reviewed":
        decisions["frame_reviews"][49]["reviewed"] = False
    elif field == "gate":
        snapshot["final_gate"]["passed"] = False
    elif field == "status":
        log["status"] = "unknown"
    elif field == "binding":
        backup["decisions_file_sha256"] = "0" * 64

    with pytest.raises(JobCompletionError):
        validate_completion_evidence(
            source,
            pack,
            snapshot,
            decisions,
            backup,
            log,
            job_id=77,
            source_snapshot_file_sha256=SOURCE_SNAPSHOT_SHA,
            review_pack_file_sha256=REVIEW_PACK_SHA,
            post_snapshot_file_sha256=POST_FILE_SHA,
            decisions_file_sha256=DECISIONS_FILE_SHA,
            backup_file_sha256=BACKUP_FILE_SHA,
            action_log_file_sha256=LOG_FILE_SHA,
        )


def test_tampered_post_annotation_is_rejected_even_if_raw_snapshot_hash_is_rewritten() -> None:
    source, pack, snapshot, decisions, backup, log = build_evidence()
    snapshot["annotations"]["shapes"][1]["points"][0] += 1
    snapshot["canonical_annotations_sha256"] = canonical_sha256(
        canonicalize_annotations(snapshot["annotations"])
    )

    with pytest.raises(JobCompletionError, match="action-log result"):
        validate_completion_evidence(
            source,
            pack,
            snapshot,
            decisions,
            backup,
            log,
            job_id=77,
            source_snapshot_file_sha256=SOURCE_SNAPSHOT_SHA,
            review_pack_file_sha256=REVIEW_PACK_SHA,
            post_snapshot_file_sha256=POST_FILE_SHA,
            decisions_file_sha256=DECISIONS_FILE_SHA,
            backup_file_sha256=BACKUP_FILE_SHA,
            action_log_file_sha256=LOG_FILE_SHA,
        )


def test_dry_run_reads_live_state_but_never_updates_job() -> None:
    _, _, snapshot, *_ = build_evidence()
    client = FakeClient(snapshot)

    result = complete_job(client, build_plan(), apply=False)

    assert result["dry_run"] is True
    assert result["mutation_performed"] is False
    assert result["job_state_before"] == "new"
    assert client.job.update_calls == []
    assert client.task.annotation_reads == 1


def test_synthetic_dry_run_verifies_live_250_frame_span() -> None:
    _, _, snapshot, *_ = build_evidence(frame_count=250)
    client = FakeClient(snapshot)

    result = complete_job(client, build_plan(frame_count=250), apply=False)

    assert result["dry_run"] is True
    assert result["annotation_count"] == 2
    assert result["job_state_before"] == "new"
    assert client.job.update_calls == []


def test_apply_rechecks_then_updates_only_state_and_verifies_readback() -> None:
    _, _, snapshot, *_ = build_evidence()
    client = FakeClient(snapshot)
    revalidations: list[bool] = []

    result = complete_job(
        client,
        build_plan(),
        apply=True,
        revalidate_evidence=lambda: revalidations.append(True),
    )

    assert revalidations == [True]
    assert client.job.update_calls == [{"state": "completed"}]
    assert client.task.annotation_reads == 3
    assert result["job_state_after"] == "completed"
    assert result["job_stage_before"] == result["job_stage_after"] == "annotation"
    assert (
        result["verified_post_completion_canonical_annotations_sha256"]
        == result["verified_live_canonical_annotations_sha256"]
    )


def test_apply_denies_drift_on_the_second_prewrite_read_without_updating() -> None:
    _, _, snapshot, *_ = build_evidence()
    client = FakeClient(snapshot)
    client.task.drift_on_read = 2

    with pytest.raises(JobCompletionError, match="annotations differ"):
        complete_job(client, build_plan(), apply=True)

    assert client.job.update_calls == []


def test_apply_recovers_only_when_timeout_has_a_verified_completed_readback() -> None:
    _, _, snapshot, *_ = build_evidence()
    client = FakeClient(snapshot)
    client.job.raise_after_update = True

    result = complete_job(client, build_plan(), apply=True)

    assert client.job.update_calls == [{"state": "completed"}]
    assert result["job_state_after"] == "completed"


def test_non_working_job_state_is_rejected() -> None:
    _, _, snapshot, *_ = build_evidence()
    client = FakeClient(snapshot)
    client.job.payload["state"] = "rejected"

    with pytest.raises(JobCompletionError, match="not allowed"):
        complete_job(client, build_plan(), apply=False)


def test_completion_receipt_is_never_overwritten(tmp_path: Path) -> None:
    receipt = tmp_path / "completion.json"
    first_sha = write_completion_receipt(receipt, {"schema": "receipt", "value": 1})

    with pytest.raises(FileExistsError):
        write_completion_receipt(receipt, {"schema": "receipt", "value": 2})

    assert canonical_sha256({"schema": "receipt", "value": 1}) != first_sha
    assert '"value": 1' in receipt.read_text(encoding="utf-8")
    assert '"value": 2' not in receipt.read_text(encoding="utf-8")
