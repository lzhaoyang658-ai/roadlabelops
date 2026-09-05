"""Validate explicit, hash-bound full-review decisions without mutating any input."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

AUTOMATED_FLAG_TYPES = frozenset(
    {
        "same_class_duplicate",
        "cross_class_conflict",
        "rider_pedestrian",
        "degenerate_box",
        "out_of_bounds_box",
    }
)
MANUAL_FLAG_TYPE = "manual_class_check"
ALLOWED_SCOPES = frozenset({"automated_risk_cleanup", "full_review"})
ALLOWED_ACTIONS = frozenset(
    {"add", "delete", "keep_distinct", "relabel", "relabel_bbox", "update_bbox"}
)
ANNOTATION_COLLECTION_KEYS = frozenset(
    {"attributes", "elements", "intervals", "shapes", "tags", "tracks"}
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
RFC3339_TIMESTAMP_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
)
FORBIDDEN_KEY_TOKENS = frozenset({"rank", "range", "rule", "rules"})
MANUAL_DELETE_APPROVAL_TYPE = "manual_shape_delete_dual_review"
MANUAL_DELETE_APPROVAL_KEYS = frozenset(
    {
        "approval_sha256",
        "approval_type",
        "shape_id",
        "frame",
        "canonical_shape_sha256",
        "reason",
        "reviewers",
    }
)
MANUAL_DELETE_REVIEWER_KEYS = frozenset({"reviewer_id", "reviewed_at", "decision"})
MANUAL_DELETE_REVIEW_DECISION = "approve_manual_delete"


class DecisionValidationError(ValueError):
    """Raised when a decision file is ambiguous, stale, or incomplete."""


def canonical_json_bytes(value: Any) -> bytes:
    """Encode a JSON value using the repository's stable canonical representation."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    """Hash a canonical JSON value."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    """Hash a file without changing it."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonicalize_annotation_value(value: Any, parent_key: str | None = None) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize_annotation_value(item, str(key))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, list):
        normalized = [_canonicalize_annotation_value(item) for item in value]
        if parent_key in ANNOTATION_COLLECTION_KEYS:
            normalized.sort(key=canonical_json_bytes)
        return normalized
    return value


def canonical_annotations_sha256(annotations: Mapping[str, Any]) -> str:
    """Reproduce the snapshot's order-independent annotation hash."""

    return canonical_sha256(_canonicalize_annotation_value(annotations))


def canonical_shape_sha256(shape: Mapping[str, Any]) -> str:
    """Hash one exact snapshot shape after canonical mapping/list normalization."""

    return canonical_sha256(_canonicalize_annotation_value(shape))


def manual_delete_approval_sha256(approval: Mapping[str, Any], *, task_id: int) -> str:
    """Content-address one dual-review approval, including its snapshot task.

    The digest deliberately excludes only ``approval_sha256`` itself.  Every
    security-relevant field, including reviewer attestations and the reason, is
    therefore immutable once the approval is referenced by its digest.
    """

    payload = {
        "task_id": task_id,
        **{str(key): value for key, value in approval.items() if key != "approval_sha256"},
    }
    return canonical_sha256(payload)


def _require_rfc3339_timestamp(value: Any, location: str) -> str:
    if not isinstance(value, str) or RFC3339_TIMESTAMP_PATTERN.fullmatch(value) is None:
        raise DecisionValidationError(
            f"{location} must be an RFC 3339 timestamp with an explicit timezone"
        )
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise DecisionValidationError(f"{location} is not a valid timestamp") from error
    if parsed.tzinfo is None:
        raise DecisionValidationError(f"{location} must include a timezone")
    return value


def validate_manual_delete_approvals(
    raw_approvals: Any,
    *,
    task_id: int,
    snapshot_shapes: Mapping[tuple[str, str], Mapping[str, Any]],
    manual_delete_shape_ids: Sequence[tuple[str, str]] | set[tuple[str, str]],
    location: str = "decisions.manual_delete_approvals",
) -> set[tuple[str, str]]:
    """Validate exact, one-use approval coverage for source=manual deletions.

    An approval is not a permissive switch.  It is bound to one task, snapshot
    shape ID, frame, canonical shape digest, reason, and exactly two distinct
    reviewer attestations.  Missing, duplicate, or unused approvals are denied.
    """

    approvals = _require_list(raw_approvals, location)
    manual_targets = list(manual_delete_shape_ids)
    if len(manual_targets) != len(set(manual_targets)):
        raise DecisionValidationError(
            "a source='manual' shape is requested for deletion more than once"
        )
    expected_targets = set(manual_targets)
    approved_targets: set[tuple[str, str]] = set()

    for index, raw_approval in enumerate(approvals):
        approval_location = f"{location}[{index}]"
        approval = _require_object(raw_approval, approval_location)
        if set(approval) != MANUAL_DELETE_APPROVAL_KEYS:
            raise DecisionValidationError(
                f"{approval_location} has unexpected or missing keys; "
                f"missing={sorted(MANUAL_DELETE_APPROVAL_KEYS - set(approval))}, "
                f"extra={sorted(set(approval) - MANUAL_DELETE_APPROVAL_KEYS)}"
            )
        if approval.get("approval_type") != MANUAL_DELETE_APPROVAL_TYPE:
            raise DecisionValidationError(
                f"{approval_location}.approval_type must be {MANUAL_DELETE_APPROVAL_TYPE!r}"
            )

        shape_id = _identity(approval.get("shape_id"), f"{approval_location}.shape_id")
        if shape_id in approved_targets:
            raise DecisionValidationError(
                f"manual-delete approval for shape {approval.get('shape_id')!r} appears twice"
            )
        shape = snapshot_shapes.get(shape_id)
        if shape is None:
            raise DecisionValidationError(
                f"{approval_location} references unknown shape {approval.get('shape_id')!r}"
            )
        if shape.get("source") != "manual":
            raise DecisionValidationError(
                f"{approval_location} may approve only source='manual' shapes; "
                f"shape {approval.get('shape_id')!r} has source={shape.get('source')!r}"
            )
        frame = _strict_int(approval.get("frame"), f"{approval_location}.frame")
        shape_frame = _strict_int(shape.get("frame"), "snapshot shape frame")
        if frame != shape_frame:
            raise DecisionValidationError(
                f"{approval_location}.frame does not match shape {approval.get('shape_id')!r}"
            )
        declared_shape_sha = _require_sha(
            approval.get("canonical_shape_sha256"),
            f"{approval_location}.canonical_shape_sha256",
        )
        if declared_shape_sha != canonical_shape_sha256(shape):
            raise DecisionValidationError(
                f"{approval_location}.canonical_shape_sha256 does not match snapshot shape "
                f"{approval.get('shape_id')!r}"
            )

        reason = approval.get("reason")
        if not isinstance(reason, str) or reason != reason.strip() or len(reason) < 4:
            raise DecisionValidationError(
                f"{approval_location}.reason must be a trimmed, substantive explanation"
            )

        reviewers = _require_list(approval.get("reviewers"), f"{approval_location}.reviewers")
        if len(reviewers) != 2:
            raise DecisionValidationError(
                f"{approval_location}.reviewers must contain exactly two independent reviews"
            )
        reviewer_identities: set[str] = set()
        for reviewer_index, raw_reviewer in enumerate(reviewers):
            reviewer_location = f"{approval_location}.reviewers[{reviewer_index}]"
            reviewer = _require_object(raw_reviewer, reviewer_location)
            if set(reviewer) != MANUAL_DELETE_REVIEWER_KEYS:
                raise DecisionValidationError(
                    f"{reviewer_location} must contain exactly "
                    f"{sorted(MANUAL_DELETE_REVIEWER_KEYS)}"
                )
            reviewer_id = reviewer.get("reviewer_id")
            if (
                not isinstance(reviewer_id, str)
                or reviewer_id != reviewer_id.strip()
                or len(reviewer_id) < 3
            ):
                raise DecisionValidationError(
                    f"{reviewer_location}.reviewer_id must be a trimmed, stable identity"
                )
            normalized_reviewer_id = reviewer_id.casefold()
            if normalized_reviewer_id in reviewer_identities:
                raise DecisionValidationError(
                    f"{approval_location}.reviewers must identify two distinct reviewers"
                )
            reviewer_identities.add(normalized_reviewer_id)
            if reviewer.get("decision") != MANUAL_DELETE_REVIEW_DECISION:
                raise DecisionValidationError(
                    f"{reviewer_location}.decision must be {MANUAL_DELETE_REVIEW_DECISION!r}"
                )
            _require_rfc3339_timestamp(
                reviewer.get("reviewed_at"), f"{reviewer_location}.reviewed_at"
            )

        declared_approval_sha = _require_sha(
            approval.get("approval_sha256"), f"{approval_location}.approval_sha256"
        )
        expected_approval_sha = manual_delete_approval_sha256(approval, task_id=task_id)
        if declared_approval_sha != expected_approval_sha:
            raise DecisionValidationError(
                f"{approval_location}.approval_sha256 does not match its bound approval payload"
            )
        approved_targets.add(shape_id)

    missing = expected_targets - approved_targets
    extra = approved_targets - expected_targets
    if missing or extra:
        missing_ids = [snapshot_shapes[item].get("id") for item in sorted(missing)]
        extra_ids = [snapshot_shapes[item].get("id") for item in sorted(extra)]
        raise DecisionValidationError(
            "manual-delete approvals must correspond one-for-one with exact judgment deletes; "
            f"missing={missing_ids}, extra={extra_ids}"
        )
    return approved_targets


def _require_object(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DecisionValidationError(f"{location} must be an object")
    return value


def _require_list(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise DecisionValidationError(f"{location} must be a list")
    return value


def _strict_int(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DecisionValidationError(f"{location} must be an integer")
    return value


def _strict_bool(value: Any, location: str) -> bool:
    if not isinstance(value, bool):
        raise DecisionValidationError(f"{location} must be a boolean")
    return value


def _require_sha(value: Any, location: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise DecisionValidationError(f"{location} must be a lowercase SHA-256 hex digest")
    return value


def _identity(value: Any, location: str) -> tuple[str, str]:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise DecisionValidationError(f"{location} must be an integer or string")
    if isinstance(value, str) and not value:
        raise DecisionValidationError(f"{location} must not be empty")
    return type(value).__name__, str(value)


def _find_forbidden_key(value: Any, location: str = "decisions") -> tuple[str, str] | None:
    if isinstance(value, dict):
        for raw_key, item in value.items():
            key = str(raw_key)
            tokens = {token for token in re.split(r"[^a-z0-9]+", key.lower()) if token}
            if tokens & FORBIDDEN_KEY_TOKENS:
                return location, key
            found = _find_forbidden_key(item, f"{location}.{key}")
            if found is not None:
                return found
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found = _find_forbidden_key(item, f"{location}[{index}]")
            if found is not None:
                return found
    return None


def _snapshot_task_id(snapshot: Mapping[str, Any]) -> int:
    task = snapshot.get("task")
    nested = task.get("id") if isinstance(task, dict) else None
    raw = snapshot.get("task_id", nested)
    return _strict_int(raw, "snapshot task_id")


def _snapshot_frames(snapshot: Mapping[str, Any]) -> set[int]:
    images = _require_list(snapshot.get("images"), "snapshot.images")
    frames: set[int] = set()
    for index, raw_image in enumerate(images):
        image = _require_object(raw_image, f"snapshot.images[{index}]")
        raw_frame = image.get("frame", image.get("cvat_frame"))
        frame = _strict_int(raw_frame, f"snapshot.images[{index}].frame")
        if frame in frames:
            raise DecisionValidationError(f"snapshot contains duplicate frame {frame}")
        frames.add(frame)
    if not frames:
        raise DecisionValidationError("snapshot.images must not be empty")
    return frames


def _label_maps(snapshot: Mapping[str, Any]) -> tuple[dict[tuple[str, str], str], set[str]]:
    raw_labels = _require_list(snapshot.get("labels"), "snapshot.labels")
    labels_by_id: dict[tuple[str, str], str] = {}
    label_names: set[str] = set()
    for index, raw_label in enumerate(raw_labels):
        label = _require_object(raw_label, f"snapshot.labels[{index}]")
        identifier = _identity(label.get("id"), f"snapshot.labels[{index}].id")
        name = label.get("name")
        if not isinstance(name, str) or not name:
            raise DecisionValidationError(f"snapshot.labels[{index}].name must be non-empty")
        if identifier in labels_by_id or name in label_names:
            raise DecisionValidationError("snapshot label IDs and names must be unique")
        labels_by_id[identifier] = name
        label_names.add(name)
    return labels_by_id, label_names


def _snapshot_shapes(
    snapshot: Mapping[str, Any], snapshot_frames: set[int]
) -> dict[tuple[str, str], dict[str, Any]]:
    annotations = _require_object(snapshot.get("annotations"), "snapshot.annotations")
    raw_shapes = _require_list(annotations.get("shapes"), "snapshot.annotations.shapes")
    shapes: dict[tuple[str, str], dict[str, Any]] = {}
    for index, raw_shape in enumerate(raw_shapes):
        shape = _require_object(raw_shape, f"snapshot.annotations.shapes[{index}]")
        shape_id = _identity(shape.get("id"), f"snapshot.annotations.shapes[{index}].id")
        if shape_id in shapes:
            raise DecisionValidationError(
                f"snapshot contains duplicate shape id {shape.get('id')!r}"
            )
        frame = _strict_int(shape.get("frame"), f"snapshot.annotations.shapes[{index}].frame")
        if frame not in snapshot_frames:
            raise DecisionValidationError(
                f"snapshot shape {shape.get('id')!r} references unknown frame {frame}"
            )
        shapes[shape_id] = shape
    return shapes


def _pack_index(
    pack: Mapping[str, Any],
    *,
    snapshot_frames: set[int],
    snapshot_shapes: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[
    dict[int, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, int],
    set[str],
    set[str],
]:
    raw_frames = _require_list(pack.get("frames"), "review_pack.frames")
    frames: dict[int, dict[str, Any]] = {}
    flags: dict[str, dict[str, Any]] = {}
    flag_frames: dict[str, int] = {}
    automated_flags: set[str] = set()
    manual_flags: set[str] = set()
    pack_shape_ids: set[tuple[str, str]] = set()

    for frame_index, raw_frame in enumerate(raw_frames):
        frame_record = _require_object(raw_frame, f"review_pack.frames[{frame_index}]")
        frame = _strict_int(frame_record.get("frame"), f"review_pack.frames[{frame_index}].frame")
        if frame in frames:
            raise DecisionValidationError(f"review pack contains duplicate frame {frame}")
        if frame not in snapshot_frames:
            raise DecisionValidationError(f"review pack contains unknown frame {frame}")
        frames[frame] = frame_record

        raw_shapes = _require_list(
            frame_record.get("shapes"), f"review_pack.frames[{frame_index}].shapes"
        )
        for shape_index, raw_shape in enumerate(raw_shapes):
            shape = _require_object(
                raw_shape, f"review_pack.frames[{frame_index}].shapes[{shape_index}]"
            )
            shape_id = _identity(
                shape.get("id"),
                f"review_pack.frames[{frame_index}].shapes[{shape_index}].id",
            )
            if shape_id in pack_shape_ids:
                raise DecisionValidationError(
                    f"review pack contains duplicate shape id {shape.get('id')!r}"
                )
            pack_shape_ids.add(shape_id)
            snapshot_shape = snapshot_shapes.get(shape_id)
            if snapshot_shape is None:
                raise DecisionValidationError(
                    f"review pack shape {shape.get('id')!r} is absent from snapshot"
                )
            if _strict_int(snapshot_shape.get("frame"), "snapshot shape frame") != frame:
                raise DecisionValidationError(
                    f"review pack shape {shape.get('id')!r} is assigned to the wrong frame"
                )

        raw_flags = _require_list(
            frame_record.get("flags"), f"review_pack.frames[{frame_index}].flags"
        )
        for flag_index, raw_flag in enumerate(raw_flags):
            flag = _require_object(
                raw_flag, f"review_pack.frames[{frame_index}].flags[{flag_index}]"
            )
            flag_id = flag.get("flag_id", flag.get("id"))
            if not isinstance(flag_id, str) or not flag_id:
                raise DecisionValidationError("review pack flag ID must be a non-empty string")
            if (
                flag.get("id") is not None
                and flag.get("flag_id") is not None
                and flag["id"] != flag["flag_id"]
            ):
                raise DecisionValidationError(f"review pack flag {flag_id!r} has conflicting IDs")
            if flag_id in flags:
                raise DecisionValidationError(f"review pack contains duplicate flag {flag_id!r}")
            flag_frame = _strict_int(flag.get("frame"), f"review pack flag {flag_id}.frame")
            if flag_frame != frame:
                raise DecisionValidationError(
                    f"review pack flag {flag_id!r} is assigned to the wrong frame"
                )
            flag_type = flag.get("type")
            if flag_type in AUTOMATED_FLAG_TYPES:
                automated_flags.add(flag_id)
            elif flag_type == MANUAL_FLAG_TYPE:
                manual_flags.add(flag_id)
            else:
                raise DecisionValidationError(
                    f"review pack flag {flag_id!r} has unknown type {flag_type!r}"
                )
            raw_flag_shape_ids = _require_list(
                flag.get("shape_ids", []), f"review pack flag {flag_id}.shape_ids"
            )
            if flag_type in AUTOMATED_FLAG_TYPES and not raw_flag_shape_ids:
                raise DecisionValidationError(
                    f"review pack automated flag {flag_id!r} must reference at least one shape"
                )
            for shape_position, raw_shape_id in enumerate(raw_flag_shape_ids):
                shape_id = _identity(
                    raw_shape_id, f"review pack flag {flag_id}.shape_ids[{shape_position}]"
                )
                shape = snapshot_shapes.get(shape_id)
                if shape is None:
                    raise DecisionValidationError(
                        f"review pack flag {flag_id!r} references unknown shape {raw_shape_id!r}"
                    )
                if _strict_int(shape.get("frame"), "snapshot shape frame") != frame:
                    raise DecisionValidationError(
                        f"review pack flag {flag_id!r} references a shape on another frame"
                    )
            flags[flag_id] = flag
            flag_frames[flag_id] = frame

    if set(frames) != snapshot_frames:
        missing = sorted(snapshot_frames - set(frames))
        extra = sorted(set(frames) - snapshot_frames)
        raise DecisionValidationError(
            f"review pack frame coverage differs from snapshot; missing={missing}, extra={extra}"
        )
    if pack_shape_ids != set(snapshot_shapes):
        missing_shapes = [
            snapshot_shapes[shape_id].get("id")
            for shape_id in sorted(set(snapshot_shapes) - pack_shape_ids)
        ]
        raise DecisionValidationError(f"review pack omits snapshot shapes: {missing_shapes}")
    return frames, flags, flag_frames, automated_flags, manual_flags


def _action_kind(action: Mapping[str, Any], location: str) -> str:
    # ``action`` is the public spelling. ``type`` is accepted as a narrow alias so
    # hand-authored evidence can mirror the review pack's existing type convention.
    has_action = "action" in action
    has_type = "type" in action
    if has_action == has_type:
        raise DecisionValidationError(f"{location} must contain exactly one of 'action' or 'type'")
    kind = action.get("action", action.get("type"))
    if kind not in ALLOWED_ACTIONS:
        raise DecisionValidationError(
            f"{location} action must be one of {sorted(ALLOWED_ACTIONS)}, got {kind!r}"
        )
    return str(kind)


def _normalize_points(value: Any, location: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 4:
        raise DecisionValidationError(f"{location} must contain exactly four coordinates")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise DecisionValidationError(f"{location} coordinates must be numbers")
    points = [float(item) for item in value]
    if not all(math.isfinite(item) for item in points):
        raise DecisionValidationError(f"{location} coordinates must be finite")
    return points


def _validate_points(value: Any, location: str, frame: Mapping[str, Any]) -> list[float]:
    points = _normalize_points(value, location)
    x1, y1, x2, y2 = points
    if x2 <= x1 or y2 <= y1:
        raise DecisionValidationError(f"{location} must have positive width and height")
    width = _strict_int(frame.get("width"), "review pack frame width")
    height = _strict_int(frame.get("height"), "review pack frame height")
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        raise DecisionValidationError(f"{location} falls outside frame bounds")
    return points


def validate_decisions(
    snapshot: Mapping[str, Any],
    review_pack: Mapping[str, Any],
    decisions: Mapping[str, Any],
    *,
    snapshot_sha256: str,
    review_pack_sha256: str,
) -> dict[str, Any]:
    """Validate parsed evidence and return a compact, JSON-safe summary."""

    snapshot = _require_object(snapshot, "snapshot")
    review_pack = _require_object(review_pack, "review_pack")
    decisions = _require_object(decisions, "decisions")
    forbidden = _find_forbidden_key(decisions)
    if forbidden is not None:
        location, key = forbidden
        raise DecisionValidationError(
            f"{location} uses forbidden rank/range/rules key {key!r}; decisions must be explicit"
        )

    scope = decisions.get("scope")
    if scope not in ALLOWED_SCOPES:
        raise DecisionValidationError(f"decisions.scope must be one of {sorted(ALLOWED_SCOPES)}")
    task_id = _strict_int(decisions.get("task_id"), "decisions.task_id")
    snapshot_task_id = _snapshot_task_id(snapshot)
    pack_task_id = _strict_int(review_pack.get("task_id"), "review_pack.task_id")
    if task_id != snapshot_task_id or task_id != pack_task_id:
        raise DecisionValidationError(
            f"task_id mismatch: decisions={task_id}, snapshot={snapshot_task_id}, "
            f"review_pack={pack_task_id}"
        )

    actual_snapshot_sha = _require_sha(snapshot_sha256, "actual snapshot SHA-256")
    actual_pack_sha = _require_sha(review_pack_sha256, "actual review-pack SHA-256")
    if (
        _require_sha(decisions.get("snapshot_sha256"), "decisions.snapshot_sha256")
        != actual_snapshot_sha
    ):
        raise DecisionValidationError("decisions.snapshot_sha256 does not match the snapshot file")
    if (
        _require_sha(decisions.get("review_pack_sha256"), "decisions.review_pack_sha256")
        != actual_pack_sha
    ):
        raise DecisionValidationError(
            "decisions.review_pack_sha256 does not match the review-pack file"
        )

    annotations = _require_object(snapshot.get("annotations"), "snapshot.annotations")
    computed_annotation_sha = canonical_annotations_sha256(annotations)
    declared_annotation_sha = _require_sha(
        snapshot.get("canonical_annotations_sha256"),
        "snapshot.canonical_annotations_sha256",
    )
    if declared_annotation_sha != computed_annotation_sha:
        raise DecisionValidationError(
            "snapshot.canonical_annotations_sha256 does not match its annotations"
        )
    if (
        _require_sha(
            decisions.get("canonical_annotations_sha256"),
            "decisions.canonical_annotations_sha256",
        )
        != computed_annotation_sha
    ):
        raise DecisionValidationError(
            "decisions.canonical_annotations_sha256 does not match the snapshot annotations"
        )
    if review_pack.get("source_snapshot_sha256") != actual_snapshot_sha:
        raise DecisionValidationError("review pack is not bound to the supplied snapshot")
    if review_pack.get("annotation_sha256") != computed_annotation_sha:
        raise DecisionValidationError("review pack annotation hash does not match the snapshot")

    snapshot_frames = _snapshot_frames(snapshot)
    labels_by_id, label_names = _label_maps(snapshot)
    snapshot_shapes = _snapshot_shapes(snapshot, snapshot_frames)
    frames, flags, flag_frames, automated_flags, manual_flags = _pack_index(
        review_pack,
        snapshot_frames=snapshot_frames,
        snapshot_shapes=snapshot_shapes,
    )

    raw_reviews = _require_list(decisions.get("frame_reviews"), "decisions.frame_reviews")
    reviews_by_frame: dict[int, dict[str, Any]] = {}
    resolved_flags: set[str] = set()
    acted_shapes: set[tuple[str, str]] = set()
    add_identities: set[tuple[int, str, tuple[float, ...]]] = set()
    action_resolved_automated_flags: set[str] = set()
    action_counts: Counter[str] = Counter()
    reviewed_frames: set[int] = set()
    manual_delete_shape_ids: list[tuple[str, str]] = []

    for review_index, raw_review in enumerate(raw_reviews):
        review = _require_object(raw_review, f"decisions.frame_reviews[{review_index}]")
        frame = _strict_int(review.get("frame"), f"frame_reviews[{review_index}].frame")
        if frame not in frames:
            raise DecisionValidationError(f"frame review references unknown frame {frame}")
        if frame in reviews_by_frame:
            raise DecisionValidationError(f"duplicate frame review for frame {frame}")
        reviews_by_frame[frame] = review
        reviewed = _strict_bool(review.get("reviewed"), f"frame_reviews[{review_index}].reviewed")
        if reviewed:
            reviewed_frames.add(frame)

        raw_resolved = _require_list(
            review.get("resolved_flag_ids"),
            f"frame_reviews[{review_index}].resolved_flag_ids",
        )
        for flag_index, flag_id in enumerate(raw_resolved):
            if not isinstance(flag_id, str) or not flag_id:
                raise DecisionValidationError(
                    f"frame_reviews[{review_index}].resolved_flag_ids[{flag_index}] "
                    "must be a non-empty string"
                )
            if flag_id not in flags:
                raise DecisionValidationError(f"unknown resolved flag {flag_id!r}")
            if flag_frames[flag_id] != frame:
                raise DecisionValidationError(
                    f"resolved flag {flag_id!r} belongs to frame {flag_frames[flag_id]}, not {frame}"
                )
            if flag_id in resolved_flags:
                raise DecisionValidationError(f"flag {flag_id!r} is resolved more than once")
            resolved_flags.add(flag_id)

        raw_actions = _require_list(review.get("actions"), f"frame_reviews[{review_index}].actions")
        if not reviewed and (raw_resolved or raw_actions):
            raise DecisionValidationError(
                f"frame {frame} cannot resolve flags or contain actions when reviewed is false"
            )
        for action_index, raw_action in enumerate(raw_actions):
            location = f"frame_reviews[{review_index}].actions[{action_index}]"
            action = _require_object(raw_action, location)
            kind = _action_kind(action, location)
            action_counts[kind] += 1
            action_flag_ids: list[str] = []
            for flag_index, flag_id in enumerate(
                _require_list(
                    action.get("resolves_flag_ids", []),
                    f"{location}.resolves_flag_ids",
                )
            ):
                flag_location = f"{location}.resolves_flag_ids[{flag_index}]"
                if not isinstance(flag_id, str) or not flag_id:
                    raise DecisionValidationError(f"{flag_location} must be a non-empty string")
                if flag_id not in flags:
                    raise DecisionValidationError(
                        f"{flag_location} references unknown flag {flag_id!r}"
                    )
                if flag_frames[flag_id] != frame:
                    raise DecisionValidationError(
                        f"action-resolved flag {flag_id!r} belongs to frame "
                        f"{flag_frames[flag_id]}, not {frame}"
                    )
                if flag_id not in raw_resolved:
                    raise DecisionValidationError(
                        f"action-resolved flag {flag_id!r} is not listed in the frame's "
                        "resolved_flag_ids"
                    )
                if flag_id in action_flag_ids:
                    raise DecisionValidationError(
                        f"{location} resolves flag {flag_id!r} more than once"
                    )
                action_flag_ids.append(flag_id)
                if flag_id in automated_flags:
                    action_resolved_automated_flags.add(flag_id)

            if kind == "add":
                automated_action_flags = automated_flags.intersection(action_flag_ids)
                if automated_action_flags:
                    raise DecisionValidationError(
                        f"{location} add action cannot by itself resolve existing-box risk "
                        f"flags {sorted(automated_action_flags)}"
                    )
                label = action.get("label")
                if label not in label_names:
                    raise DecisionValidationError(f"{location}.label is unknown: {label!r}")
                points = _validate_points(action.get("points"), f"{location}.points", frames[frame])
                identity = (frame, str(label), tuple(points))
                if identity in add_identities:
                    raise DecisionValidationError(f"duplicate add action on frame {frame}")
                add_identities.add(identity)
                continue

            shape_id = _identity(action.get("shape_id"), f"{location}.shape_id")
            shape = snapshot_shapes.get(shape_id)
            if shape is None:
                raise DecisionValidationError(
                    f"{location} references unknown shape {action.get('shape_id')!r}"
                )
            shape_frame = _strict_int(shape.get("frame"), "snapshot shape frame")
            if shape_frame != frame:
                raise DecisionValidationError(
                    f"shape {action.get('shape_id')!r} belongs to frame {shape_frame}, not {frame}"
                )
            for flag_id in automated_flags.intersection(action_flag_ids):
                flag_shape_ids = {
                    _identity(raw_shape_id, f"review pack flag {flag_id}.shape_ids")
                    for raw_shape_id in flags[flag_id].get("shape_ids", [])
                }
                if flag_shape_ids and shape_id not in flag_shape_ids:
                    raise DecisionValidationError(
                        f"{location} shape {action.get('shape_id')!r} is not referenced by "
                        f"automated flag {flag_id!r}"
                    )
            if shape_id in acted_shapes:
                raise DecisionValidationError(
                    f"shape {action.get('shape_id')!r} has more than one action"
                )
            acted_shapes.add(shape_id)
            expected_shape_sha = _require_sha(
                action.get("expected_shape_sha256"), f"{location}.expected_shape_sha256"
            )
            actual_shape_sha = canonical_shape_sha256(shape)
            if expected_shape_sha != actual_shape_sha:
                raise DecisionValidationError(
                    f"{location}.expected_shape_sha256 does not match snapshot shape "
                    f"{action.get('shape_id')!r}"
                )
            if kind == "delete":
                source = shape.get("source")
                if source == "manual":
                    manual_delete_shape_ids.append(shape_id)
                elif source != "auto":
                    raise DecisionValidationError(
                        f"{location} may delete only source='auto' shapes or explicitly "
                        "approved source='manual' shapes; "
                        f"shape {action.get('shape_id')!r} has source={source!r}"
                    )
            if kind in {"relabel", "relabel_bbox"}:
                to_label = action.get("to_label")
                if to_label not in label_names:
                    raise DecisionValidationError(f"{location}.to_label is unknown: {to_label!r}")
                label_id = _identity(shape.get("label_id"), "snapshot shape label_id")
                current_label = labels_by_id.get(label_id)
                if current_label is None:
                    raise DecisionValidationError(
                        f"snapshot shape {action.get('shape_id')!r} has an unknown label_id"
                    )
                if to_label == current_label:
                    raise DecisionValidationError(
                        f"{location} relabel target equals the shape's current label"
                    )
            if kind in {"relabel_bbox", "update_bbox"}:
                points = _validate_points(action.get("points"), f"{location}.points", frames[frame])
                current_points = _normalize_points(
                    shape.get("points"),
                    f"snapshot shape {action.get('shape_id')!r}.points",
                )
                if points == current_points:
                    raise DecisionValidationError(
                        f"{location}.points equals the shape's current bounding box"
                    )

    raw_manual_delete_approvals = decisions.get("manual_delete_approvals", [])
    if scope != "full_review" and (manual_delete_shape_ids or raw_manual_delete_approvals):
        raise DecisionValidationError(
            "manual-delete approvals and source='manual' deletes are allowed only in "
            "full_review decisions compiled from an explicit human judgment"
        )
    validate_manual_delete_approvals(
        raw_manual_delete_approvals,
        task_id=task_id,
        snapshot_shapes=snapshot_shapes,
        manual_delete_shape_ids=manual_delete_shape_ids,
    )

    missing_automated = automated_flags - resolved_flags
    if missing_automated:
        raise DecisionValidationError(
            "automated risk flags are not resolved exactly once: "
            f"missing={sorted(missing_automated)}"
        )
    missing_automated_actions = automated_flags - action_resolved_automated_flags
    if missing_automated_actions:
        raise DecisionValidationError(
            "automated risk flags lack an explicit bound action: "
            f"missing={sorted(missing_automated_actions)}"
        )
    if scope == "full_review":
        missing_frames = snapshot_frames - reviewed_frames
        if missing_frames:
            raise DecisionValidationError(
                f"full_review requires every frame reviewed; missing={sorted(missing_frames)}"
            )
        missing_flags = set(flags) - resolved_flags
        if missing_flags:
            raise DecisionValidationError(
                f"full_review requires every flag resolved exactly once; missing={sorted(missing_flags)}"
            )

    unresolved_manual = manual_flags - resolved_flags
    return {
        "valid": True,
        "mutation_performed": False,
        "scope": scope,
        "task_id": task_id,
        "snapshot_sha256": actual_snapshot_sha,
        "canonical_annotations_sha256": computed_annotation_sha,
        "review_pack_sha256": actual_pack_sha,
        "snapshot_frame_count": len(snapshot_frames),
        "reviewed_frame_count": len(reviewed_frames),
        "action_count": sum(action_counts.values()),
        "action_counts": dict(sorted(action_counts.items())),
        "resolved_flag_count": len(resolved_flags),
        "resolved_automated_flag_count": len(automated_flags),
        "action_bound_automated_flag_count": len(action_resolved_automated_flags),
        "unresolved_manual_flag_count": len(unresolved_manual),
    }


def _read_json_object(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise DecisionValidationError(f"{description} does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DecisionValidationError(f"Could not read {description}: {error}") from error
    return _require_object(payload, description)


def validate_decision_files(
    snapshot_path: Path | str,
    review_pack_path: Path | str,
    decisions_path: Path | str,
) -> dict[str, Any]:
    """Load and validate exactly three immutable JSON inputs."""

    snapshot_path = Path(snapshot_path).resolve()
    review_pack_path = Path(review_pack_path).resolve()
    decisions_path = Path(decisions_path).resolve()
    snapshot = _read_json_object(snapshot_path, "snapshot")
    review_pack = _read_json_object(review_pack_path, "review pack")
    decisions = _read_json_object(decisions_path, "decisions")
    return validate_decisions(
        snapshot,
        review_pack,
        decisions,
        snapshot_sha256=file_sha256(snapshot_path),
        review_pack_sha256=file_sha256(review_pack_path),
    )


# Descriptive public alias matching the script name.
validate_full_review_decisions = validate_decisions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("review_pack", type=Path)
    parser.add_argument("decisions", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = validate_decision_files(args.snapshot, args.review_pack, args.decisions)
    except DecisionValidationError as error:
        parser.error(str(error))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
