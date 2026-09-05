"""Safely mark one fully reviewed CVAT job as completed.

The command is a dry run unless ``--apply`` is supplied.  Completion is a
separate, evidence-bound operation after annotation writeback: it never edits
annotations and it sends only the CVAT job ``state`` field.  A successful
write is followed by task, job, label, and annotation readback and an
immutable completion receipt.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cvat_sdk import models

from roadlabelops.settings import Settings, build_cvat_adapter
from scripts.apply_final_review import (
    FinalReviewApplyError,
    build_apply_plan,
    post_apply_canonical_sha256,
    validate_review_evidence,
    verify_live_labels,
)
from scripts.snapshot_cvat_task import (
    atomic_write_json_new,
    canonical_sha256,
    canonicalize_annotations,
    sdk_to_json,
)
from scripts.validate_full_review_decisions import canonical_shape_sha256

POST_SNAPSHOT_SCHEMA = {"name": "roadlabelops.cvat-task-snapshot", "version": 1}
BACKUP_SCHEMA = {"name": "roadlabelops.final-review-backup", "version": 1}
ACTION_LOG_SCHEMA = {"name": "roadlabelops.final-review-action-log", "version": 1}
RECEIPT_SCHEMA = {"name": "roadlabelops.cvat-job-completion-receipt", "version": 1}
ALLOWED_PRE_COMPLETION_STATES = frozenset({"new", "in_progress"})
ALLOWED_ACTION_LOG_STATUSES = frozenset({"planned_and_hash_bound_before_write"})
ALLOWED_ACTIONS = frozenset(
    {"add", "delete", "keep_distinct", "relabel", "relabel_bbox", "update_bbox"}
)
ALLOWED_DECISION_SCHEMA_VERSIONS = frozenset({"1.0", "1.1", "1.2", "1.3"})


class JobCompletionError(ValueError):
    """Raised before completion when evidence or live CVAT state is unsafe."""


def _object(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise JobCompletionError(f"{location} must be an object")
    return value


def _list(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise JobCompletionError(f"{location} must be a list")
    return value


def _integer(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise JobCompletionError(f"{location} must be an integer")
    return value


def _sha256(value: Any, location: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise JobCompletionError(f"{location} must be a lowercase SHA-256 hex digest")
    return value


def _identity(value: Any, location: str) -> tuple[str, str]:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise JobCompletionError(f"{location} must be an integer or string")
    if isinstance(value, str) and not value:
        raise JobCompletionError(f"{location} must not be empty")
    return type(value).__name__, str(value)


def _enum_text(value: Any, location: str) -> str:
    normalized = getattr(value, "value", value)
    if not isinstance(normalized, str) or not normalized:
        raise JobCompletionError(f"{location} must be a non-empty string")
    return normalized


def _read_hashed_json(path: Path, description: str) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise JobCompletionError(f"{description} does not exist: {path}")
    try:
        encoded = path.read_bytes()
        payload = json.loads(encoded.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise JobCompletionError(f"Could not read {description}: {error}") from error
    return _object(payload, description), hashlib.sha256(encoded).hexdigest()


def _annotation_count(annotations: Mapping[str, Any]) -> int:
    return sum(
        len(_list(annotations.get(key), f"annotations.{key}"))
        for key in ("tags", "shapes", "tracks")
    )


def _snapshot_frames(snapshot: Mapping[str, Any], *, frame_count: int) -> set[int]:
    frames: set[int] = set()
    for index, raw_image in enumerate(_list(snapshot.get("images"), "post snapshot.images")):
        image = _object(raw_image, f"post snapshot.images[{index}]")
        frame = _integer(
            image.get("frame", image.get("cvat_frame")),
            f"post snapshot.images[{index}].frame",
        )
        if frame in frames:
            raise JobCompletionError(f"post snapshot contains duplicate frame {frame}")
        frames.add(frame)
    expected_frames = set(range(frame_count))
    if frames != expected_frames:
        raise JobCompletionError(
            "post snapshot images must contain exactly the contiguous CVAT frame span "
            f"0 through {frame_count - 1}"
        )
    return frames


def _label_ids(snapshot: Mapping[str, Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    seen_ids: set[int] = set()
    for index, raw_label in enumerate(_list(snapshot.get("labels"), "post snapshot.labels")):
        label = _object(raw_label, f"post snapshot.labels[{index}]")
        identifier = _integer(label.get("id"), f"post snapshot.labels[{index}].id")
        name = label.get("name")
        if not isinstance(name, str) or not name:
            raise JobCompletionError(f"post snapshot.labels[{index}].name must be non-empty")
        if name in result or identifier in seen_ids:
            raise JobCompletionError("post snapshot label IDs and names must be unique")
        result[name] = identifier
        seen_ids.add(identifier)
    if not result:
        raise JobCompletionError("post snapshot.labels must not be empty")
    return result


def _require_no_issues(record: Mapping[str, Any], location: str) -> None:
    issues = _object(record.get("issues"), f"{location}.issues")
    count = _integer(issues.get("count"), f"{location}.issues.count")
    if count:
        raise JobCompletionError(f"{location} still has {count} unresolved issue(s)")


def _points(value: Any, location: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 4:
        raise JobCompletionError(f"{location} must contain exactly four coordinates")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise JobCompletionError(f"{location} coordinates must be numbers")
    return [float(item) for item in value]


def _validate_post_snapshot(
    snapshot: Mapping[str, Any], *, task_id: int, job_id: int
) -> dict[str, Any]:
    if snapshot.get("snapshot_schema") != POST_SNAPSHOT_SCHEMA:
        raise JobCompletionError("post snapshot schema is unsupported")
    task = _object(snapshot.get("task"), "post snapshot.task")
    if _integer(task.get("id"), "post snapshot.task.id") != task_id:
        raise JobCompletionError("post snapshot task does not match decisions.task_id")
    frame_count = _integer(task.get("size"), "post snapshot.task.size")
    if frame_count <= 0:
        raise JobCompletionError("post snapshot task size must be positive")
    manifest = _object(snapshot.get("manifest"), "post snapshot.manifest")
    if _integer(manifest.get("cvat_task_id"), "post snapshot.manifest.cvat_task_id") != task_id:
        raise JobCompletionError("post snapshot manifest is bound to a different task")
    if _integer(manifest.get("sample_size"), "post snapshot.manifest.sample_size") != frame_count:
        raise JobCompletionError("post snapshot manifest sample_size differs from task size")

    frames = _snapshot_frames(snapshot, frame_count=frame_count)
    jobs = _list(snapshot.get("jobs"), "post snapshot.jobs")
    if len(jobs) != 1:
        raise JobCompletionError("post snapshot must contain exactly one CVAT job")
    job = _object(jobs[0], "post snapshot.jobs[0]")
    if _integer(job.get("id"), "post snapshot.jobs[0].id") != job_id:
        raise JobCompletionError("requested job does not match the post snapshot job")
    if _integer(job.get("task_id"), "post snapshot.jobs[0].task_id") != task_id:
        raise JobCompletionError("post snapshot job belongs to a different task")
    if _integer(job.get("start_frame"), "post snapshot.jobs[0].start_frame") != 0:
        raise JobCompletionError("post snapshot job must start at frame 0")
    if _integer(job.get("stop_frame"), "post snapshot.jobs[0].stop_frame") != frame_count - 1:
        raise JobCompletionError("post snapshot job must cover the complete task frame span")
    if _integer(job.get("frame_count"), "post snapshot.jobs[0].frame_count") != frame_count:
        raise JobCompletionError("post snapshot job frame_count differs from task size")
    job_type = _enum_text(job.get("type"), "post snapshot.jobs[0].type")
    if job_type != "annotation":
        raise JobCompletionError("post snapshot job must be an annotation job")
    if job.get("parent_job_id") is not None:
        raise JobCompletionError("post snapshot job must not be a derived/consensus job")
    _require_no_issues(job, "post snapshot.jobs[0]")
    stage = _enum_text(job.get("stage"), "post snapshot.jobs[0].stage")
    state = _enum_text(job.get("state"), "post snapshot.jobs[0].state")
    if state not in ALLOWED_PRE_COMPLETION_STATES:
        raise JobCompletionError(
            "post snapshot job state must be 'new' or 'in_progress' before completion"
        )

    gate = _object(snapshot.get("final_gate"), "post snapshot.final_gate")
    if gate.get("passed") is not True:
        raise JobCompletionError("post snapshot final_gate.passed must be true")
    if _list(gate.get("blocking_reasons"), "post snapshot.final_gate.blocking_reasons"):
        raise JobCompletionError("post snapshot final gate still has blocking reasons")

    annotations = canonicalize_annotations(snapshot.get("annotations"))
    if annotations["tags"] or annotations["tracks"]:
        raise JobCompletionError("post snapshot contains tags or tracks")
    annotation_sha = canonical_sha256(annotations)
    if (
        _sha256(
            snapshot.get("canonical_annotations_sha256"),
            "post snapshot.canonical_annotations_sha256",
        )
        != annotation_sha
    ):
        raise JobCompletionError("post snapshot annotation hash does not match its annotations")
    counts = _object(snapshot.get("counts"), "post snapshot.counts")
    if _integer(counts.get("images"), "post snapshot.counts.images") != len(frames):
        raise JobCompletionError("post snapshot image count is inconsistent")
    for key in ("tags", "shapes", "tracks"):
        if _integer(counts.get(key), f"post snapshot.counts.{key}") != len(annotations[key]):
            raise JobCompletionError(f"post snapshot {key} count is inconsistent")

    return {
        "frames": frames,
        "frame_count": frame_count,
        "job": job,
        "job_type": job_type,
        "stage": stage,
        "state": state,
        "annotations": annotations,
        "annotation_sha256": annotation_sha,
        "annotation_count": _annotation_count(annotations),
        "label_ids": _label_ids(snapshot),
    }


def _validate_decisions(
    decisions: Mapping[str, Any], *, task_id: int, frames: set[int]
) -> dict[str, Any]:
    if decisions.get("schema_version") not in ALLOWED_DECISION_SCHEMA_VERSIONS:
        raise JobCompletionError(
            "decisions.schema_version must be a supported explicit-review schema"
        )
    if decisions.get("scope") != "full_review":
        raise JobCompletionError("decisions.scope must be 'full_review'")
    if decisions.get("mutation_performed") is not False:
        raise JobCompletionError("decisions.mutation_performed must be false")
    if _integer(decisions.get("task_id"), "decisions.task_id") != task_id:
        raise JobCompletionError("decisions are bound to a different task")
    _sha256(decisions.get("snapshot_sha256"), "decisions.snapshot_sha256")
    _sha256(decisions.get("review_pack_sha256"), "decisions.review_pack_sha256")
    _sha256(
        decisions.get("canonical_annotations_sha256"),
        "decisions.canonical_annotations_sha256",
    )

    reviews = _list(decisions.get("frame_reviews"), "decisions.frame_reviews")
    if len(reviews) != len(frames):
        raise JobCompletionError(
            "full_review decisions must contain exactly one review for every snapshot frame"
        )
    reviewed_frames: set[int] = set()
    resolved_flags: set[str] = set()
    acted_shapes: set[tuple[str, str]] = set()
    add_identities: set[tuple[int, str, tuple[float, ...]]] = set()
    action_count = 0
    for review_index, raw_review in enumerate(reviews):
        review = _object(raw_review, f"decisions.frame_reviews[{review_index}]")
        frame = _integer(review.get("frame"), f"decisions.frame_reviews[{review_index}].frame")
        if frame not in frames or frame in reviewed_frames:
            raise JobCompletionError("decision frame reviews must cover every frame exactly once")
        if review.get("reviewed") is not True:
            raise JobCompletionError(f"decision frame {frame} is not explicitly reviewed")
        reviewed_frames.add(frame)
        for flag_index, flag_id in enumerate(
            _list(
                review.get("resolved_flag_ids"),
                f"decisions.frame_reviews[{review_index}].resolved_flag_ids",
            )
        ):
            if not isinstance(flag_id, str) or not flag_id:
                raise JobCompletionError(
                    f"decisions.frame_reviews[{review_index}].resolved_flag_ids[{flag_index}] "
                    "must be non-empty"
                )
            if flag_id in resolved_flags:
                raise JobCompletionError(f"decision flag {flag_id!r} is resolved more than once")
            resolved_flags.add(flag_id)
        actions = _list(review.get("actions"), f"decisions.frame_reviews[{review_index}].actions")
        action_count += len(actions)
        for action_index, raw_action in enumerate(actions):
            action = _object(
                raw_action,
                f"decisions.frame_reviews[{review_index}].actions[{action_index}]",
            )
            if ("action" in action) == ("type" in action):
                raise JobCompletionError(
                    "each decision action must contain exactly one of 'action' or 'type'"
                )
            kind = action.get("action", action.get("type"))
            if kind not in ALLOWED_ACTIONS:
                raise JobCompletionError(f"decision action {kind!r} is unsupported")
            if kind == "add":
                label = action.get("label")
                if not isinstance(label, str) or not label:
                    raise JobCompletionError("decision add label must be non-empty")
                identity = (
                    frame,
                    label,
                    tuple(_points(action.get("points"), "decision add.points")),
                )
                if identity in add_identities:
                    raise JobCompletionError("final decisions contain a duplicate add action")
                add_identities.add(identity)
            else:
                shape_id = _identity(action.get("shape_id"), "decision action.shape_id")
                if shape_id in acted_shapes:
                    raise JobCompletionError("final decisions act on one shape more than once")
                acted_shapes.add(shape_id)
    if reviewed_frames != frames:
        raise JobCompletionError("full_review decisions do not cover every snapshot frame")
    return {
        "frame_count": len(reviewed_frames),
        "action_count": action_count,
        "resolved_flag_count": len(resolved_flags),
        # A schema-v1 action log can only be emitted after the canonical full-review
        # validator reports zero unresolved manual flags.  Its exact binding and
        # action replay are checked below instead of trusting a free-form summary.
        "unresolved_manual_flag_count": 0,
    }


def _validate_action_pair(
    decision: Mapping[str, Any],
    logged: Mapping[str, Any],
    *,
    frame: int,
    action_index: int,
    label_ids: Mapping[str, int],
) -> None:
    kind = decision.get("action", decision.get("type"))
    if logged.get("action_index") != action_index:
        raise JobCompletionError("action log indices must be consecutive and ordered")
    if logged.get("action") != kind or logged.get("frame") != frame:
        raise JobCompletionError("action log order/content does not match final decisions")
    after = logged.get("after_shape")
    if kind == "add":
        label = decision.get("label")
        if label not in label_ids or logged.get("label") != label:
            raise JobCompletionError("logged add label does not match final decisions")
        after_shape = _object(after, f"action log.actions[{action_index}].after_shape")
        expected_add = {
            "type": "rectangle",
            "label_id": label_ids[str(label)],
            "frame": frame,
            "occluded": False,
            "outside": False,
            "z_order": 0,
            "rotation": 0.0,
            "points": _points(decision.get("points"), "decision add.points"),
            "id": None,
            "group": 0,
            "source": "manual",
            "attributes": [],
            "score": 1.0,
            "elements": [],
        }
        if canonical_shape_sha256(after_shape) != canonical_shape_sha256(expected_add):
            raise JobCompletionError("logged add shape does not exactly match final decisions")
        return

    if _identity(logged.get("shape_id"), "logged shape_id") != _identity(
        decision.get("shape_id"), "decision shape_id"
    ):
        raise JobCompletionError("logged shape_id does not match final decisions")
    expected_shape_sha = _sha256(
        decision.get("expected_shape_sha256"), "decision.expected_shape_sha256"
    )
    if logged.get("expected_shape_sha256") != expected_shape_sha:
        raise JobCompletionError("logged shape hash does not match final decisions")
    before = _object(logged.get("before_shape"), "logged before_shape")
    if canonical_shape_sha256(before) != expected_shape_sha:
        raise JobCompletionError("logged before_shape does not match its decision hash")
    if kind == "delete":
        if after is not None:
            raise JobCompletionError("logged delete must have after_shape=null")
        return
    after_shape = _object(after, "logged after_shape")
    expected_after = copy.deepcopy(before)
    if kind in {"relabel", "relabel_bbox"}:
        to_label = decision.get("to_label")
        if to_label not in label_ids:
            raise JobCompletionError("logged relabel target does not match final decisions")
        expected_after["label_id"] = label_ids[str(to_label)]
    if kind in {"update_bbox", "relabel_bbox"}:
        expected_after["points"] = _points(decision.get("points"), "decision bbox.points")
    if canonical_shape_sha256(after_shape) != canonical_shape_sha256(expected_after):
        raise JobCompletionError("logged after_shape contains an unauthorized field change")


def _replay_action_log(
    backup_annotations: Mapping[str, Any], actions: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], set[tuple[str, str]]]:
    result = copy.deepcopy(dict(backup_annotations))
    original_ids = {_identity(shape.get("id"), "backup shape.id") for shape in result["shapes"]}
    by_id = {_identity(shape.get("id"), "backup shape.id"): shape for shape in result["shapes"]}
    additions: list[dict[str, Any]] = []
    for index, action in enumerate(actions):
        kind = action["action"]
        if kind == "add":
            additions.append(copy.deepcopy(_object(action.get("after_shape"), f"actions[{index}]")))
            continue
        shape_id = _identity(action.get("shape_id"), f"actions[{index}].shape_id")
        shape = by_id.get(shape_id)
        if shape is None:
            raise JobCompletionError("action log references a shape absent from the backup")
        if kind == "delete":
            result["shapes"].remove(shape)
            del by_id[shape_id]
        elif kind != "keep_distinct":
            after = copy.deepcopy(_object(action.get("after_shape"), f"actions[{index}]"))
            shape.clear()
            shape.update(after)
    result["shapes"].extend(additions)
    return canonicalize_annotations(result), original_ids


def validate_completion_evidence(
    source_snapshot: Mapping[str, Any],
    review_pack: Mapping[str, Any],
    post_snapshot: Mapping[str, Any],
    decisions: Mapping[str, Any],
    backup: Mapping[str, Any],
    action_log: Mapping[str, Any],
    *,
    job_id: int,
    source_snapshot_file_sha256: str,
    review_pack_file_sha256: str,
    post_snapshot_file_sha256: str,
    decisions_file_sha256: str,
    backup_file_sha256: str,
    action_log_file_sha256: str,
) -> dict[str, Any]:
    """Validate and bind all immutable evidence needed for job completion."""

    task_id = _integer(decisions.get("task_id"), "decisions.task_id")
    post = _validate_post_snapshot(post_snapshot, task_id=task_id, job_id=job_id)
    decision_summary = _validate_decisions(decisions, task_id=task_id, frames=post["frames"])
    try:
        apply_plan = build_apply_plan(
            source_snapshot,
            review_pack,
            decisions,
            snapshot_file_sha256=source_snapshot_file_sha256,
            review_pack_file_sha256=review_pack_file_sha256,
        )
    except FinalReviewApplyError as error:
        raise JobCompletionError(f"source full-review validation failed: {error}") from error
    canonical_summary = apply_plan["decision_validation_summary"]
    if canonical_summary.get("scope") != "full_review":
        raise JobCompletionError("canonical decision validation did not confirm full_review")
    if canonical_summary.get("snapshot_frame_count") != post["frame_count"]:
        raise JobCompletionError(
            "canonical decision validation frame count differs from the post snapshot"
        )
    if canonical_summary.get("reviewed_frame_count") != post["frame_count"]:
        raise JobCompletionError(
            "canonical decision validation review count differs from the post snapshot"
        )
    if canonical_summary.get("unresolved_manual_flag_count") != 0:
        raise JobCompletionError("canonical decision validation still has unresolved manual flags")

    if backup.get("schema") != BACKUP_SCHEMA:
        raise JobCompletionError("apply backup schema is unsupported")
    if action_log.get("schema") != ACTION_LOG_SCHEMA:
        raise JobCompletionError("apply action-log schema is unsupported")
    if action_log.get("status") not in ALLOWED_ACTION_LOG_STATUSES:
        raise JobCompletionError("apply action-log status is not a safe completed-write status")
    for name, artifact in (("backup", backup), ("action log", action_log)):
        if _integer(artifact.get("task_id"), f"{name}.task_id") != task_id:
            raise JobCompletionError(f"{name} belongs to a different task")
        if _sha256(
            artifact.get("decisions_file_sha256"), f"{name}.decisions_file_sha256"
        ) != _sha256(decisions_file_sha256, "actual decisions file SHA-256"):
            raise JobCompletionError(f"{name} is not bound to the supplied decisions file")
        if artifact.get("snapshot_file_sha256") != _sha256(
            source_snapshot_file_sha256, "actual source snapshot file SHA-256"
        ):
            raise JobCompletionError(f"{name} source snapshot hash differs from decisions")
        if artifact.get("review_pack_file_sha256") != _sha256(
            review_pack_file_sha256, "actual review-pack file SHA-256"
        ):
            raise JobCompletionError(f"{name} review-pack hash differs from decisions")

    backup_annotations = canonicalize_annotations(backup.get("annotations"))
    if backup_annotations["tags"] or backup_annotations["tracks"]:
        raise JobCompletionError("apply backup contains tags or tracks")
    backup_sha = canonical_sha256(backup_annotations)
    if (
        _sha256(backup.get("canonical_annotations_sha256"), "backup.canonical_annotations_sha256")
        != backup_sha
    ):
        raise JobCompletionError("apply backup annotation hash is invalid")
    if decisions.get("canonical_annotations_sha256") != backup_sha:
        raise JobCompletionError("apply backup is not the pre-apply state bound by decisions")

    logged_actions = [
        _object(action, f"action log.actions[{index}]")
        for index, action in enumerate(_list(action_log.get("actions"), "action log.actions"))
    ]
    if len(logged_actions) != decision_summary["action_count"]:
        raise JobCompletionError("action log count does not match final decisions")
    if canonical_sha256(logged_actions) != canonical_sha256(apply_plan["action_log"]):
        raise JobCompletionError("action log differs from the canonical apply plan")
    flat_decisions: list[tuple[int, dict[str, Any]]] = []
    for raw_review in decisions["frame_reviews"]:
        review = _object(raw_review, "decision frame review")
        frame = _integer(review.get("frame"), "decision frame")
        flat_decisions.extend(
            (frame, _object(action, "decision action")) for action in review["actions"]
        )
    for index, ((frame, decision), logged) in enumerate(zip(flat_decisions, logged_actions)):
        _validate_action_pair(
            decision,
            logged,
            frame=frame,
            action_index=index,
            label_ids=post["label_ids"],
        )

    replayed, original_ids = _replay_action_log(backup_annotations, logged_actions)
    expected_post_sha = post_apply_canonical_sha256(replayed, original_shape_ids=original_ids)
    logged_expected_sha = _sha256(
        action_log.get("expected_post_apply_canonical_sha256"),
        "action log.expected_post_apply_canonical_sha256",
    )
    stage_hashes = _object(action_log.get("stage_hashes"), "action log.stage_hashes")
    if set(stage_hashes) != {"initial", "delete", "update", "add"}:
        raise JobCompletionError("action-log stage hashes are incomplete")
    for stage, digest in stage_hashes.items():
        _sha256(digest, f"action log.stage_hashes.{stage}")
    if stage_hashes.get("add") != logged_expected_sha:
        raise JobCompletionError("action-log final stage hash is inconsistent")
    if stage_hashes != apply_plan["stage_hashes"]:
        raise JobCompletionError("action-log stage hashes differ from the canonical apply plan")
    if logged_expected_sha != apply_plan["expected_post_apply_canonical_sha256"]:
        raise JobCompletionError("action-log final hash differs from the canonical apply plan")
    expected_initial_sha = post_apply_canonical_sha256(
        backup_annotations, original_shape_ids=original_ids
    )
    if stage_hashes.get("initial") != expected_initial_sha:
        raise JobCompletionError("action-log initial stage does not match the apply backup")
    if expected_post_sha != logged_expected_sha:
        raise JobCompletionError("action log does not replay to its expected post-apply hash")
    actual_post_normalized_sha = post_apply_canonical_sha256(
        post["annotations"], original_shape_ids=original_ids
    )
    if actual_post_normalized_sha != logged_expected_sha:
        raise JobCompletionError("post snapshot does not match the applied action-log result")
    if len(replayed["shapes"]) != len(post["annotations"]["shapes"]):
        raise JobCompletionError("post snapshot annotation count differs from the action replay")

    return {
        "task_id": task_id,
        "job_id": job_id,
        "job_stage": post["stage"],
        "job_type": post["job_type"],
        "frame_count": post["frame_count"],
        "job_start_frame": 0,
        "job_stop_frame": post["frame_count"] - 1,
        "post_snapshot": dict(post_snapshot),
        "source_snapshot_file_sha256": _sha256(
            source_snapshot_file_sha256, "actual source snapshot file SHA-256"
        ),
        "review_pack_file_sha256": _sha256(
            review_pack_file_sha256, "actual review-pack file SHA-256"
        ),
        "post_snapshot_file_sha256": _sha256(
            post_snapshot_file_sha256, "actual post snapshot file SHA-256"
        ),
        "decisions_file_sha256": _sha256(decisions_file_sha256, "actual decisions file SHA-256"),
        "backup_file_sha256": _sha256(backup_file_sha256, "actual backup file SHA-256"),
        "action_log_file_sha256": _sha256(action_log_file_sha256, "actual action-log file SHA-256"),
        "post_annotation_sha256": post["annotation_sha256"],
        "post_annotation_count": post["annotation_count"],
        "expected_post_apply_canonical_sha256": logged_expected_sha,
        "decision_validation": canonical_summary,
    }


def _job_record(value: Any, location: str) -> dict[str, Any]:
    record = sdk_to_json(value)
    return _object(record, location)


def verify_live_completion_state(
    client: Any,
    plan: Mapping[str, Any],
    *,
    allowed_states: frozenset[str],
) -> dict[str, Any]:
    """Read and verify the exact live task/job/annotations state."""

    task_id = int(plan["task_id"])
    job_id = int(plan["job_id"])
    task = client.tasks.retrieve(task_id)
    task_record = _job_record(task, "live task")
    if _integer(task_record.get("id"), "live task.id") != task_id:
        raise JobCompletionError("CVAT returned the wrong task")
    if _integer(task_record.get("size"), "live task.size") != plan["frame_count"]:
        raise JobCompletionError("live task size changed after the post-apply snapshot")
    try:
        verify_live_labels(plan["post_snapshot"], task.get_labels())
    except FinalReviewApplyError as error:
        raise JobCompletionError(str(error)) from error
    annotations = canonicalize_annotations(task.get_annotations())
    live_sha = canonical_sha256(annotations)
    if live_sha != plan["post_annotation_sha256"]:
        raise JobCompletionError(
            "live CVAT annotations differ from the post-apply snapshot; completion denied "
            f"(snapshot={plan['post_annotation_sha256']}, live={live_sha})"
        )
    if _annotation_count(annotations) != plan["post_annotation_count"]:
        raise JobCompletionError("live CVAT annotation count differs from the post snapshot")

    task_jobs = [_job_record(job, "live task job") for job in task.get_jobs()]
    if len(task_jobs) != 1 or _integer(task_jobs[0].get("id"), "live task job.id") != job_id:
        raise JobCompletionError("live task no longer has exactly the snapshot-bound job")
    direct_job = client.jobs.retrieve(job_id)
    direct_record = _job_record(direct_job, "live job")
    for record, location in ((task_jobs[0], "live task job"), (direct_record, "live job")):
        if _integer(record.get("id"), f"{location}.id") != job_id:
            raise JobCompletionError(f"{location} ID does not match the requested job")
        if _integer(record.get("task_id"), f"{location}.task_id") != task_id:
            raise JobCompletionError(f"{location} belongs to a different task")
        if _enum_text(record.get("type"), f"{location}.type") != plan["job_type"]:
            raise JobCompletionError(f"{location} type changed; completion denied")
        if record.get("parent_job_id") is not None:
            raise JobCompletionError(f"{location} unexpectedly became a derived job")
        if (
            _integer(record.get("start_frame"), f"{location}.start_frame")
            != plan["job_start_frame"]
        ):
            raise JobCompletionError(f"{location} start_frame changed; completion denied")
        if _integer(record.get("stop_frame"), f"{location}.stop_frame") != plan["job_stop_frame"]:
            raise JobCompletionError(f"{location} stop_frame changed; completion denied")
        if _integer(record.get("frame_count"), f"{location}.frame_count") != plan["frame_count"]:
            raise JobCompletionError(f"{location} frame_count changed; completion denied")
        _require_no_issues(record, location)
        if _enum_text(record.get("stage"), f"{location}.stage") != plan["job_stage"]:
            raise JobCompletionError(f"{location} stage changed; completion denied")
        state = _enum_text(record.get("state"), f"{location}.state")
        if state not in allowed_states:
            raise JobCompletionError(
                f"{location} state {state!r} is not allowed for this completion phase"
            )
    if _enum_text(task_jobs[0].get("state"), "live task job.state") != _enum_text(
        direct_record.get("state"), "live job.state"
    ):
        raise JobCompletionError("task and direct job readbacks disagree on state")
    return {
        "task": task,
        "job": direct_job,
        "task_status": task_record.get("status"),
        "job_state": _enum_text(direct_record.get("state"), "live job.state"),
        "job_stage": _enum_text(direct_record.get("stage"), "live job.stage"),
        "annotation_sha256": live_sha,
        "annotation_count": _annotation_count(annotations),
    }


def complete_job(
    client: Any,
    plan: Mapping[str, Any],
    *,
    apply: bool,
    revalidate_evidence: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Dry-run or execute the state-only job completion transition."""

    initial = verify_live_completion_state(
        client, plan, allowed_states=ALLOWED_PRE_COMPLETION_STATES
    )
    summary = {
        "task_id": plan["task_id"],
        "job_id": plan["job_id"],
        "dry_run": not apply,
        "mutation_performed": False,
        "job_state_before": initial["job_state"],
        "job_stage_before": initial["job_stage"],
        "task_status_before": initial["task_status"],
        "verified_live_canonical_annotations_sha256": initial["annotation_sha256"],
        "annotation_count": initial["annotation_count"],
        "decision_validation": plan["decision_validation"],
    }
    if not apply:
        return summary

    if revalidate_evidence is not None:
        revalidate_evidence()
    # The second complete read is intentionally adjacent to the only mutation.
    before_write = verify_live_completion_state(
        client, plan, allowed_states=ALLOWED_PRE_COMPLETION_STATES
    )
    try:
        before_write["job"].update(models.PatchedJobWriteRequest(state="completed"))
    except Exception as update_error:
        # A transport timeout can occur after CVAT committed the PATCH.  Only a
        # fresh, fully evidence-bound completed readback can recover that case.
        try:
            after = verify_live_completion_state(
                client, plan, allowed_states=frozenset({"completed"})
            )
        except Exception:
            raise RuntimeError(
                "CVAT job completion result is uncertain; no success receipt was written"
            ) from update_error
    else:
        after = verify_live_completion_state(client, plan, allowed_states=frozenset({"completed"}))
    if after["job_stage"] != before_write["job_stage"]:
        raise RuntimeError("CVAT changed the job stage while completing the job")
    if after["annotation_sha256"] != before_write["annotation_sha256"]:
        raise RuntimeError("CVAT annotations changed while completing the job")
    summary.update(
        {
            "dry_run": False,
            "mutation_performed": True,
            "job_state_after": after["job_state"],
            "job_stage_after": after["job_stage"],
            "task_status_after": after["task_status"],
            "verified_post_completion_canonical_annotations_sha256": after["annotation_sha256"],
        }
    )
    return summary


def write_completion_receipt(path: Path, receipt: Mapping[str, Any]) -> str:
    """Publish one immutable receipt and return its exact file digest."""

    atomic_write_json_new(path, receipt)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_snapshot", type=Path)
    parser.add_argument("review_pack", type=Path)
    parser.add_argument("post_snapshot", type=Path)
    parser.add_argument("decisions", type=Path)
    parser.add_argument("backup", type=Path)
    parser.add_argument("action_log", type=Path)
    parser.add_argument("--job-id", required=True, type=int)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    paths = {
        "source_snapshot": args.source_snapshot.resolve(),
        "review_pack": args.review_pack.resolve(),
        "post_snapshot": args.post_snapshot.resolve(),
        "decisions": args.decisions.resolve(),
        "backup": args.backup.resolve(),
        "action_log": args.action_log.resolve(),
    }
    receipt_path = args.receipt.resolve()
    if receipt_path.exists():
        parser.error(f"completion receipt already exists; completion denied: {receipt_path}")
    try:
        loaded = {
            name: _read_hashed_json(path, name.replace("_", " ")) for name, path in paths.items()
        }
        plan = validate_completion_evidence(
            *(
                loaded[name][0]
                for name in (
                    "source_snapshot",
                    "review_pack",
                    "post_snapshot",
                    "decisions",
                    "backup",
                    "action_log",
                )
            ),
            job_id=args.job_id,
            source_snapshot_file_sha256=loaded["source_snapshot"][1],
            review_pack_file_sha256=loaded["review_pack"][1],
            post_snapshot_file_sha256=loaded["post_snapshot"][1],
            decisions_file_sha256=loaded["decisions"][1],
            backup_file_sha256=loaded["backup"][1],
            action_log_file_sha256=loaded["action_log"][1],
        )
        validate_review_evidence(loaded["decisions"][0], decisions_path=paths["decisions"])
    except (FinalReviewApplyError, JobCompletionError) as error:
        parser.error(str(error))

    expected_file_hashes = {name: digest for name, (_, digest) in loaded.items()}

    def revalidate_evidence() -> None:
        for name, path in paths.items():
            _, actual_sha = _read_hashed_json(path, name.replace("_", " "))
            if actual_sha != expected_file_hashes[name]:
                raise JobCompletionError(f"{name.replace('_', ' ')} changed before completion")
        try:
            validate_review_evidence(loaded["decisions"][0], decisions_path=paths["decisions"])
        except FinalReviewApplyError as error:
            raise JobCompletionError(str(error)) from error

    adapter = build_cvat_adapter(Settings())
    if adapter is None:
        parser.error("CVAT is not configured")
    try:
        with adapter._client() as client:
            summary = complete_job(
                client,
                plan,
                apply=args.apply,
                revalidate_evidence=revalidate_evidence,
            )
    except JobCompletionError as error:
        parser.error(str(error))

    summary["planned_completion_receipt"] = str(receipt_path)
    if args.apply:
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            **summary,
            "evidence": {
                name: {"path": str(paths[name]), "sha256": expected_file_hashes[name]}
                for name in paths
            },
        }
        try:
            receipt_sha256 = write_completion_receipt(receipt_path, receipt)
        except FileExistsError as error:
            raise RuntimeError(
                "CVAT job was completed but the immutable receipt path was concurrently claimed"
            ) from error
        summary["completion_receipt"] = str(receipt_path)
        summary["completion_receipt_sha256"] = receipt_sha256
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
