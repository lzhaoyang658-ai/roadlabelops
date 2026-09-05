"""Apply explicit, snapshot-bound final-review decisions to one CVAT task safely.

The command is a dry run unless ``--apply`` is supplied.  It never derives
changes from geometry: every deletion, relabel, and addition must be present in
the reviewed decision file.  Existing shapes are bound by ID, frame, and their
canonical snapshot hash before a mutation plan can be built.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cvat_sdk import models
from cvat_sdk.core.proxies.annotations import AnnotationUpdateAction

from roadlabelops.settings import Settings, build_cvat_adapter
from scripts.snapshot_cvat_task import (
    atomic_write_json_new,
    canonical_sha256,
    canonicalize_annotations,
    sdk_to_json,
)
from scripts.validate_full_review_decisions import (
    DecisionValidationError,
    canonical_shape_sha256,
    validate_decisions,
)

ALLOWED_ACTIONS = frozenset(
    {"add", "delete", "keep_distinct", "relabel", "relabel_bbox", "update_bbox"}
)
MUTATING_ACTIONS = frozenset({"add", "delete", "relabel", "relabel_bbox", "update_bbox"})
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_KEY_TOKENS = frozenset({"rank", "range", "rule", "rules"})
SHAPE_REQUEST_FIELDS = frozenset(models.LabeledShapeRequest.attribute_map)


class FinalReviewApplyError(ValueError):
    """Raised before a mutation when evidence or live CVAT state is unsafe."""


def _require_object(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FinalReviewApplyError(f"{location} must be an object")
    return value


def _require_list(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise FinalReviewApplyError(f"{location} must be a list")
    return value


def _strict_int(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FinalReviewApplyError(f"{location} must be an integer")
    return value


def _identity(value: Any, location: str) -> tuple[str, str]:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise FinalReviewApplyError(f"{location} must be an integer or string")
    if isinstance(value, str) and not value:
        raise FinalReviewApplyError(f"{location} must not be empty")
    return type(value).__name__, str(value)


def _require_sha256(value: Any, location: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise FinalReviewApplyError(f"{location} must be a lowercase SHA-256 hex digest")
    return value


def _find_forbidden_key(value: Any, location: str = "decisions") -> tuple[str, str] | None:
    if isinstance(value, Mapping):
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
    return _strict_int(snapshot.get("task_id", nested), "snapshot task_id")


def _snapshot_frames(snapshot: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    frames: dict[int, dict[str, Any]] = {}
    for index, raw_image in enumerate(_require_list(snapshot.get("images"), "snapshot.images")):
        image = _require_object(raw_image, f"snapshot.images[{index}]")
        frame = _strict_int(
            image.get("frame", image.get("cvat_frame")), f"snapshot.images[{index}].frame"
        )
        if frame in frames:
            raise FinalReviewApplyError(f"snapshot contains duplicate frame {frame}")
        width = _strict_int(image.get("width"), f"snapshot.images[{index}].width")
        height = _strict_int(image.get("height"), f"snapshot.images[{index}].height")
        if width <= 0 or height <= 0:
            raise FinalReviewApplyError(f"snapshot frame {frame} has invalid dimensions")
        frames[frame] = image
    if not frames:
        raise FinalReviewApplyError("snapshot.images must not be empty")
    return frames


def _snapshot_label_maps(
    snapshot: Mapping[str, Any],
) -> tuple[dict[tuple[str, str], str], dict[str, int]]:
    labels_by_id: dict[tuple[str, str], str] = {}
    ids_by_name: dict[str, int] = {}
    for index, raw_label in enumerate(_require_list(snapshot.get("labels"), "snapshot.labels")):
        label = _require_object(raw_label, f"snapshot.labels[{index}]")
        identifier = _strict_int(label.get("id"), f"snapshot.labels[{index}].id")
        name = label.get("name")
        if not isinstance(name, str) or not name:
            raise FinalReviewApplyError(f"snapshot.labels[{index}].name must be non-empty")
        identity = _identity(identifier, f"snapshot.labels[{index}].id")
        if identity in labels_by_id or name in ids_by_name:
            raise FinalReviewApplyError("snapshot label IDs and names must be unique")
        labels_by_id[identity] = name
        ids_by_name[name] = identifier
    if not labels_by_id:
        raise FinalReviewApplyError("snapshot.labels must not be empty")
    return labels_by_id, ids_by_name


def _shape_index(annotations: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    shapes: dict[tuple[str, str], dict[str, Any]] = {}
    for index, raw_shape in enumerate(
        _require_list(annotations.get("shapes"), "annotations.shapes")
    ):
        shape = _require_object(raw_shape, f"annotations.shapes[{index}]")
        shape_id = _identity(shape.get("id"), f"annotations.shapes[{index}].id")
        if shape_id in shapes:
            raise FinalReviewApplyError(
                f"annotations contain duplicate shape id {shape.get('id')!r}"
            )
        if shape.get("type") != "rectangle":
            raise FinalReviewApplyError(
                f"shape {shape.get('id')!r} is not a rectangle; final apply denied"
            )
        unsupported = set(shape) - SHAPE_REQUEST_FIELDS
        if unsupported:
            raise FinalReviewApplyError(
                f"shape {shape.get('id')!r} has unsupported fields {sorted(unsupported)}"
            )
        shapes[shape_id] = shape
    return shapes


def _normalize_rectangle_points(value: Any, *, location: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 4:
        raise FinalReviewApplyError(f"{location} must contain exactly four coordinates")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise FinalReviewApplyError(f"{location} coordinates must be numbers")
    points = [float(item) for item in value]
    if not all(math.isfinite(item) for item in points):
        raise FinalReviewApplyError(f"{location} coordinates must be finite")
    return points


def _validate_rectangle_points(
    value: Any, *, frame: int, frame_record: Mapping[str, Any], location: str
) -> list[float]:
    points = _normalize_rectangle_points(value, location=location)
    x1, y1, x2, y2 = points
    width = int(frame_record["width"])
    height = int(frame_record["height"])
    if x2 <= x1 or y2 <= y1:
        raise FinalReviewApplyError(f"{location} must have positive width and height")
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        raise FinalReviewApplyError(f"{location} falls outside frame {frame} bounds")
    return points


def _action_kind(action: Mapping[str, Any], location: str) -> str:
    has_action = "action" in action
    has_type = "type" in action
    if has_action == has_type:
        raise FinalReviewApplyError(f"{location} must contain exactly one of 'action' or 'type'")
    kind = action.get("action", action.get("type"))
    if kind not in ALLOWED_ACTIONS:
        raise FinalReviewApplyError(
            f"{location} action must be one of {sorted(ALLOWED_ACTIONS)}, got {kind!r}"
        )
    return str(kind)


def _new_shape(*, frame: int, label_id: int, points: list[float]) -> dict[str, Any]:
    """Return the complete deterministic rectangle payload used for one explicit add."""

    return {
        "type": "rectangle",
        "label_id": label_id,
        "frame": frame,
        "occluded": False,
        "outside": False,
        "z_order": 0,
        "rotation": 0.0,
        "points": points,
        "id": None,
        "group": 0,
        "source": "manual",
        "attributes": [],
        "score": 1.0,
        "elements": [],
    }


def post_apply_canonical_sha256(
    annotations: Any, *, original_shape_ids: set[tuple[str, str]]
) -> str:
    """Hash post-apply state while normalizing only server-assigned values.

    CVAT increments ``annotations.version`` and chooses IDs for newly created
    shapes.  Both values are unknowable during a dry run.  Existing shape IDs
    remain hash-bound: only IDs absent from the snapshot's original ID set are
    normalized to ``None``.
    """

    canonical = canonicalize_annotations(annotations)
    canonical["version"] = None
    for shape in canonical["shapes"]:
        raw_id = shape.get("id")
        if raw_id is None:
            shape["id"] = None
            continue
        identity = (type(raw_id).__name__, str(raw_id))
        if identity not in original_shape_ids:
            shape["id"] = None
    return canonical_sha256(canonicalize_annotations(canonical))


def _apply_actions_to_annotations(
    annotations: Mapping[str, Any],
    *,
    delete_ids: set[tuple[str, str]],
    relabel_ids: Mapping[tuple[str, str], int],
    bbox_updates: Mapping[tuple[str, str], Sequence[float]],
    additions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result = copy.deepcopy(dict(annotations))
    retained: list[dict[str, Any]] = []
    for raw_shape in result["shapes"]:
        shape = dict(raw_shape)
        shape_id = _identity(shape.get("id"), "planned shape id")
        if shape_id in delete_ids:
            continue
        if shape_id in relabel_ids:
            shape["label_id"] = relabel_ids[shape_id]
        if shape_id in bbox_updates:
            shape["points"] = list(bbox_updates[shape_id])
        retained.append(shape)
    retained.extend(copy.deepcopy(list(additions)))
    result["shapes"] = retained
    return canonicalize_annotations(result)


def build_apply_plan(
    snapshot: Mapping[str, Any],
    review_pack: Mapping[str, Any],
    decisions: Mapping[str, Any],
    *,
    snapshot_file_sha256: str,
    review_pack_file_sha256: str,
) -> dict[str, Any]:
    """Validate three parsed evidence objects and build an explicit mutation plan."""

    snapshot = _require_object(snapshot, "snapshot")
    review_pack = _require_object(review_pack, "review pack")
    decisions = _require_object(decisions, "decisions")
    try:
        validation_summary = validate_decisions(
            snapshot,
            review_pack,
            decisions,
            snapshot_sha256=snapshot_file_sha256,
            review_pack_sha256=review_pack_file_sha256,
        )
    except DecisionValidationError as error:
        raise FinalReviewApplyError(f"final-review decision validation failed: {error}") from error
    if validation_summary.get("scope") != "full_review":
        raise FinalReviewApplyError("validated decisions.scope must be 'full_review' before apply")
    if validation_summary.get("unresolved_manual_flag_count") != 0:
        raise FinalReviewApplyError(
            "validated full review still contains unresolved manual flags; apply denied"
        )

    forbidden = _find_forbidden_key(decisions)
    if forbidden is not None:
        location, key = forbidden
        raise FinalReviewApplyError(
            f"{location} uses forbidden rank/range/rules key {key!r}; apply actions must be explicit"
        )
    if decisions.get("scope") != "full_review":
        raise FinalReviewApplyError("decisions.scope must be 'full_review' before final apply")

    actual_snapshot_file_sha = _require_sha256(snapshot_file_sha256, "actual snapshot file SHA-256")
    actual_review_pack_file_sha = _require_sha256(
        review_pack_file_sha256, "actual review-pack file SHA-256"
    )
    if (
        _require_sha256(decisions.get("snapshot_sha256"), "decisions.snapshot_sha256")
        != actual_snapshot_file_sha
    ):
        raise FinalReviewApplyError("decisions.snapshot_sha256 does not match the snapshot file")

    task_id = _strict_int(decisions.get("task_id"), "decisions.task_id")
    if task_id != _snapshot_task_id(snapshot):
        raise FinalReviewApplyError("decisions.task_id does not match the snapshot task")

    annotations = canonicalize_annotations(snapshot.get("annotations"))
    if annotations["tags"] or annotations["tracks"]:
        raise FinalReviewApplyError("snapshot contains tags or tracks; rectangle-only apply denied")
    snapshot_annotation_sha = canonical_sha256(annotations)
    if (
        _require_sha256(
            snapshot.get("canonical_annotations_sha256"),
            "snapshot.canonical_annotations_sha256",
        )
        != snapshot_annotation_sha
    ):
        raise FinalReviewApplyError(
            "snapshot.canonical_annotations_sha256 does not match its annotations"
        )
    if (
        _require_sha256(
            decisions.get("canonical_annotations_sha256"),
            "decisions.canonical_annotations_sha256",
        )
        != snapshot_annotation_sha
    ):
        raise FinalReviewApplyError(
            "decisions.canonical_annotations_sha256 does not match the snapshot"
        )

    frames = _snapshot_frames(snapshot)
    labels_by_id, label_ids_by_name = _snapshot_label_maps(snapshot)
    shapes = _shape_index(annotations)
    for shape in shapes.values():
        frame = _strict_int(shape.get("frame"), f"shape {shape.get('id')!r}.frame")
        if frame not in frames:
            raise FinalReviewApplyError(
                f"snapshot shape {shape.get('id')!r} references unknown frame {frame}"
            )
        if _identity(shape.get("label_id"), "snapshot shape label_id") not in labels_by_id:
            raise FinalReviewApplyError(
                f"snapshot shape {shape.get('id')!r} references an unknown label"
            )

    # ``validate_decisions`` has already verified every approval's schema,
    # content digest, task/frame/shape binding, dual-review attestations, and
    # one-for-one correspondence with a source=manual delete action.  Keep the
    # exact IDs here as a second, apply-local allowlist instead of weakening the
    # historical source guard globally.
    approved_manual_delete_ids = {
        _identity(
            _require_object(
                raw_approval, f"decisions.manual_delete_approvals[{approval_index}]"
            ).get("shape_id"),
            f"decisions.manual_delete_approvals[{approval_index}].shape_id",
        )
        for approval_index, raw_approval in enumerate(
            _require_list(
                decisions.get("manual_delete_approvals", []),
                "decisions.manual_delete_approvals",
            )
        )
    }

    raw_reviews = _require_list(decisions.get("frame_reviews"), "decisions.frame_reviews")
    acted_shape_ids: set[tuple[str, str]] = set()
    add_identities: set[tuple[int, int, tuple[float, ...]]] = set()
    delete_ids: set[tuple[str, str]] = set()
    relabel_ids: dict[tuple[str, str], int] = {}
    bbox_updates: dict[tuple[str, str], list[float]] = {}
    additions: list[dict[str, Any]] = []
    action_log: list[dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    seen_review_frames: set[int] = set()

    for review_index, raw_review in enumerate(raw_reviews):
        review = _require_object(raw_review, f"decisions.frame_reviews[{review_index}]")
        frame = _strict_int(review.get("frame"), f"frame_reviews[{review_index}].frame")
        if frame not in frames:
            raise FinalReviewApplyError(f"frame review references unknown frame {frame}")
        if frame in seen_review_frames:
            raise FinalReviewApplyError(f"duplicate frame review for frame {frame}")
        seen_review_frames.add(frame)
        if review.get("reviewed") is not True:
            raise FinalReviewApplyError(f"frame {frame} must be explicitly reviewed")
        actions = _require_list(review.get("actions"), f"frame_reviews[{review_index}].actions")
        for action_index, raw_action in enumerate(actions):
            location = f"frame_reviews[{review_index}].actions[{action_index}]"
            action = _require_object(raw_action, location)
            kind = _action_kind(action, location)
            action_counts[kind] += 1

            if kind == "add":
                label = action.get("label")
                if label not in label_ids_by_name:
                    raise FinalReviewApplyError(f"{location}.label is unknown: {label!r}")
                points = _validate_rectangle_points(
                    action.get("points"),
                    frame=frame,
                    frame_record=frames[frame],
                    location=f"{location}.points",
                )
                label_id = label_ids_by_name[str(label)]
                add_identity = (frame, label_id, tuple(points))
                if add_identity in add_identities:
                    raise FinalReviewApplyError(f"duplicate add action on frame {frame}")
                add_identities.add(add_identity)
                after_shape = _new_shape(frame=frame, label_id=label_id, points=points)
                additions.append(after_shape)
                action_log.append(
                    {
                        "action_index": len(action_log),
                        "action": kind,
                        "frame": frame,
                        "label": label,
                        "after_shape": after_shape,
                    }
                )
                continue

            shape_id = _identity(action.get("shape_id"), f"{location}.shape_id")
            shape = shapes.get(shape_id)
            if shape is None:
                raise FinalReviewApplyError(
                    f"{location} references unknown shape {action.get('shape_id')!r}"
                )
            shape_frame = _strict_int(shape.get("frame"), "snapshot shape frame")
            if shape_frame != frame:
                raise FinalReviewApplyError(
                    f"shape {action.get('shape_id')!r} belongs to frame {shape_frame}, not {frame}"
                )
            if shape_id in acted_shape_ids:
                raise FinalReviewApplyError(
                    f"shape {action.get('shape_id')!r} has more than one action"
                )
            acted_shape_ids.add(shape_id)
            expected_shape_sha = _require_sha256(
                action.get("expected_shape_sha256"), f"{location}.expected_shape_sha256"
            )
            if expected_shape_sha != canonical_shape_sha256(shape):
                raise FinalReviewApplyError(
                    f"{location}.expected_shape_sha256 does not match snapshot shape "
                    f"{action.get('shape_id')!r}"
                )

            after_shape: dict[str, Any] | None = copy.deepcopy(shape)
            if kind == "delete":
                source = shape.get("source")
                if source == "manual" and shape_id not in approved_manual_delete_ids:
                    raise FinalReviewApplyError(
                        f"{location} source='manual' shape {action.get('shape_id')!r} "
                        "lacks its exact dual-review approval"
                    )
                if source not in {"auto", "manual"}:
                    raise FinalReviewApplyError(
                        f"{location} may delete only source='auto' shapes or explicitly "
                        "approved source='manual' shapes; "
                        f"shape {action.get('shape_id')!r} has source={source!r}"
                    )
                delete_ids.add(shape_id)
                after_shape = None
            elif kind in {"relabel", "relabel_bbox"}:
                to_label = action.get("to_label")
                if to_label not in label_ids_by_name:
                    raise FinalReviewApplyError(f"{location}.to_label is unknown: {to_label!r}")
                current_label = labels_by_id[_identity(shape.get("label_id"), "shape label_id")]
                if to_label == current_label:
                    raise FinalReviewApplyError(
                        f"{location} relabel target equals the shape's current label"
                    )
                relabel_ids[shape_id] = label_ids_by_name[str(to_label)]
                after_shape["label_id"] = label_ids_by_name[str(to_label)]
            if kind in {"relabel_bbox", "update_bbox"}:
                points = _validate_rectangle_points(
                    action.get("points"),
                    frame=frame,
                    frame_record=frames[frame],
                    location=f"{location}.points",
                )
                current_points = _normalize_rectangle_points(
                    shape.get("points"),
                    location=f"shape {action.get('shape_id')!r}.points",
                )
                if points == current_points:
                    raise FinalReviewApplyError(
                        f"{location}.points equals the shape's current bounding box"
                    )
                bbox_updates[shape_id] = points
                after_shape["points"] = points

            action_log.append(
                {
                    "action_index": len(action_log),
                    "action": kind,
                    "frame": frame,
                    "shape_id": action.get("shape_id"),
                    "expected_shape_sha256": expected_shape_sha,
                    "before_shape": copy.deepcopy(shape),
                    "after_shape": after_shape,
                }
            )

    missing_review_frames = set(frames) - seen_review_frames
    if missing_review_frames:
        raise FinalReviewApplyError(
            "full_review apply requires every snapshot frame reviewed; "
            f"missing={sorted(missing_review_frames)}"
        )

    original_shape_ids = set(shapes)
    after_delete = _apply_actions_to_annotations(
        annotations,
        delete_ids=delete_ids,
        relabel_ids={},
        bbox_updates={},
        additions=[],
    )
    after_update = _apply_actions_to_annotations(
        annotations,
        delete_ids=delete_ids,
        relabel_ids=relabel_ids,
        bbox_updates=bbox_updates,
        additions=[],
    )
    expected_annotations = _apply_actions_to_annotations(
        annotations,
        delete_ids=delete_ids,
        relabel_ids=relabel_ids,
        bbox_updates=bbox_updates,
        additions=additions,
    )
    stage_hashes = {
        "initial": post_apply_canonical_sha256(annotations, original_shape_ids=original_shape_ids),
        "delete": post_apply_canonical_sha256(after_delete, original_shape_ids=original_shape_ids),
        "update": post_apply_canonical_sha256(after_update, original_shape_ids=original_shape_ids),
        "add": post_apply_canonical_sha256(
            expected_annotations, original_shape_ids=original_shape_ids
        ),
    }
    return {
        "task_id": task_id,
        "snapshot_file_sha256": actual_snapshot_file_sha,
        "review_pack_file_sha256": actual_review_pack_file_sha,
        "snapshot_canonical_annotations_sha256": snapshot_annotation_sha,
        "decision_validation_summary": validation_summary,
        "original_shape_ids": original_shape_ids,
        "snapshot_annotations": annotations,
        "expected_annotations": expected_annotations,
        "expected_post_apply_canonical_sha256": stage_hashes["add"],
        "stage_hashes": stage_hashes,
        "delete_shapes": [
            copy.deepcopy(shapes[shape_id])
            for shape_id in sorted(delete_ids, key=lambda item: (item[0], item[1]))
        ],
        "update_shapes": [
            copy.deepcopy(shape)
            for shape in expected_annotations["shapes"]
            if shape.get("id") is not None
            and _identity(shape.get("id"), "planned update shape id")
            in (set(relabel_ids) | set(bbox_updates))
        ],
        "add_shapes": additions,
        "action_log": action_log,
        "action_counts": dict(sorted(action_counts.items())),
        "mutation_action_count": sum(action_counts[kind] for kind in MUTATING_ACTIONS),
        "manual_delete_approval_count": len(approved_manual_delete_ids),
    }


def verify_live_annotations(snapshot: Mapping[str, Any], live_annotations: Any) -> dict[str, Any]:
    """Require the complete live annotation hash to equal the immutable snapshot."""

    expected = canonicalize_annotations(snapshot.get("annotations"))
    live = canonicalize_annotations(live_annotations)
    if live["tags"] or live["tracks"]:
        raise FinalReviewApplyError("live CVAT task contains tags or tracks; apply denied")
    expected_sha = canonical_sha256(expected)
    live_sha = canonical_sha256(live)
    if live_sha != expected_sha:
        raise FinalReviewApplyError(
            "live CVAT annotations changed after the snapshot; apply denied "
            f"(snapshot={expected_sha}, live={live_sha})"
        )
    return live


def verify_live_labels(snapshot: Mapping[str, Any], live_labels: Sequence[Any]) -> None:
    """Reject label-map drift that an annotation-only hash cannot detect."""

    expected = {
        int(label["id"]): str(label["name"])
        for label in _require_list(snapshot.get("labels"), "snapshot.labels")
    }
    actual_records = [sdk_to_json(label) for label in live_labels]
    try:
        actual = {int(label["id"]): str(label["name"]) for label in actual_records}
    except (KeyError, TypeError, ValueError) as error:
        raise FinalReviewApplyError("live CVAT labels are invalid") from error
    if actual != expected:
        raise FinalReviewApplyError("live CVAT labels changed after the snapshot; apply denied")


def shape_request_from_record(
    shape: Mapping[str, Any], *, include_id: bool
) -> models.LabeledShapeRequest:
    """Convert one complete rectangle record without discarding supported fields."""

    unsupported = set(shape) - SHAPE_REQUEST_FIELDS
    if unsupported:
        raise FinalReviewApplyError(f"shape request has unsupported fields {sorted(unsupported)}")
    payload = copy.deepcopy(dict(shape))
    if not include_id or payload.get("id") is None:
        payload.pop("id", None)
    return models.LabeledShapeRequest(**payload)


def _read_hashed_json_object(path: Path, description: str) -> tuple[dict[str, Any], str]:
    """Parse and hash the exact same bytes, avoiding a read/hash race."""

    if not path.is_file():
        raise FinalReviewApplyError(f"{description} does not exist: {path}")
    try:
        encoded = path.read_bytes()
        payload = json.loads(encoded.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FinalReviewApplyError(f"Could not read {description}: {error}") from error
    return _require_object(payload, description), hashlib.sha256(encoded).hexdigest()


def validate_review_evidence(
    decisions: Mapping[str, Any], *, decisions_path: Path | str
) -> list[dict[str, str]]:
    """Verify every relative review-evidence path against the same bytes read.

    Paths are resolved relative to the decision file, including any intentional
    ``..`` components.  De-duplicating the resolved paths prevents aliases such
    as ``candidate.json`` and ``./candidate.json`` from naming one artifact twice.
    """

    records = _require_list(decisions.get("review_evidence"), "decisions.review_evidence")
    if not records:
        raise FinalReviewApplyError("decisions.review_evidence must not be empty")

    decision_directory = Path(decisions_path).resolve().parent
    seen_paths: set[Path] = set()
    validated: list[dict[str, str]] = []
    for index, raw_record in enumerate(records):
        location = f"decisions.review_evidence[{index}]"
        record = _require_object(raw_record, location)
        if set(record) != {"path", "sha256"}:
            raise FinalReviewApplyError(f"{location} must contain exactly 'path' and 'sha256'")
        raw_path = record.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise FinalReviewApplyError(f"{location}.path must be a non-empty string")
        relative_path = Path(raw_path)
        if relative_path.is_absolute():
            raise FinalReviewApplyError(f"{location}.path must be relative to the decision file")
        evidence_path = (decision_directory / relative_path).resolve()
        if evidence_path in seen_paths:
            raise FinalReviewApplyError(
                f"{location}.path resolves to duplicate review evidence {evidence_path}"
            )
        seen_paths.add(evidence_path)
        if not evidence_path.is_file():
            raise FinalReviewApplyError(f"{location}.path does not exist: {evidence_path}")

        expected_sha = _require_sha256(record.get("sha256"), f"{location}.sha256")
        try:
            encoded = evidence_path.read_bytes()
        except OSError as error:
            raise FinalReviewApplyError(
                f"could not read {location}.path {evidence_path}: {error}"
            ) from error
        actual_sha = hashlib.sha256(encoded).hexdigest()
        if actual_sha != expected_sha:
            raise FinalReviewApplyError(f"{location}.sha256 does not match {evidence_path}")
        validated.append({"path": raw_path, "sha256": expected_sha})
    return validated


def _artifact_paths(
    decisions_path: Path, *, task_id: int, artifact_directory: Path | None
) -> tuple[Path, Path]:
    root = (
        artifact_directory.resolve()
        if artifact_directory is not None
        else (decisions_path.parent / "apply-final-review").resolve()
    )
    stem = decisions_path.stem
    return (
        root / f"{stem}.task-{task_id}.pre-apply-backup.json",
        root / f"{stem}.task-{task_id}.action-log.json",
    )


def _summary(plan: Mapping[str, Any], *, dry_run: bool) -> dict[str, Any]:
    return {
        "task_id": plan["task_id"],
        "dry_run": dry_run,
        "mutation_performed": False,
        "snapshot_file_sha256": plan["snapshot_file_sha256"],
        "review_pack_file_sha256": plan["review_pack_file_sha256"],
        "snapshot_canonical_annotations_sha256": plan["snapshot_canonical_annotations_sha256"],
        "expected_post_apply_canonical_sha256": plan["expected_post_apply_canonical_sha256"],
        "annotation_count_before": len(plan["snapshot_annotations"]["shapes"]),
        "annotation_count_after": len(plan["expected_annotations"]["shapes"]),
        "action_counts": plan["action_counts"],
        "mutation_action_count": plan["mutation_action_count"],
        "manual_delete_approval_count": plan["manual_delete_approval_count"],
    }


def _verify_stage(
    annotations: Any,
    *,
    expected_sha256: str,
    original_shape_ids: set[tuple[str, str]],
    stage: str,
) -> dict[str, Any]:
    canonical = canonicalize_annotations(annotations)
    actual_sha = post_apply_canonical_sha256(canonical, original_shape_ids=original_shape_ids)
    if actual_sha != expected_sha256:
        raise RuntimeError(
            f"CVAT {stage} readback hash differs from the dry-run plan: "
            f"expected={expected_sha256}, actual={actual_sha}"
        )
    return canonical


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("review_pack", type=Path)
    parser.add_argument("decisions", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--artifact-directory", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    snapshot_path = args.snapshot.resolve()
    review_pack_path = args.review_pack.resolve()
    decisions_path = args.decisions.resolve()
    try:
        snapshot, snapshot_file_sha = _read_hashed_json_object(snapshot_path, "snapshot")
        review_pack, review_pack_file_sha = _read_hashed_json_object(
            review_pack_path, "review pack"
        )
        decisions, decisions_file_sha = _read_hashed_json_object(decisions_path, "decisions")
        plan = build_apply_plan(
            snapshot,
            review_pack,
            decisions,
            snapshot_file_sha256=snapshot_file_sha,
            review_pack_file_sha256=review_pack_file_sha,
        )
        validate_review_evidence(decisions, decisions_path=decisions_path)
    except FinalReviewApplyError as error:
        parser.error(str(error))

    summary = _summary(plan, dry_run=not args.apply)
    backup_path, action_log_path = _artifact_paths(
        decisions_path,
        task_id=plan["task_id"],
        artifact_directory=args.artifact_directory,
    )
    summary["planned_backup_path"] = str(backup_path)
    summary["planned_action_log_path"] = str(action_log_path)
    summary["decisions_file_sha256"] = decisions_file_sha

    adapter = build_cvat_adapter(Settings())
    if adapter is None:
        parser.error("CVAT is not configured")
    if backup_path.exists() or action_log_path.exists():
        parser.error("backup or action log already exists; apply denied")

    with adapter._client() as client:
        task = client.tasks.retrieve(plan["task_id"])
        try:
            verify_live_labels(snapshot, task.get_labels())
            live = verify_live_annotations(snapshot, task.get_annotations())
        except FinalReviewApplyError as error:
            parser.error(str(error))
        summary["verified_live_canonical_annotations_sha256"] = canonical_sha256(live)
        if not args.apply:
            print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
            return

        created_at = datetime.now(timezone.utc).isoformat()
        backup_payload = {
            "schema": {"name": "roadlabelops.final-review-backup", "version": 1},
            "created_at": created_at,
            "task_id": plan["task_id"],
            "snapshot_file_sha256": plan["snapshot_file_sha256"],
            "review_pack_file_sha256": plan["review_pack_file_sha256"],
            "decisions_file_sha256": decisions_file_sha,
            "canonical_annotations_sha256": canonical_sha256(live),
            "annotations": live,
        }
        log_payload = {
            "schema": {"name": "roadlabelops.final-review-action-log", "version": 1},
            "created_at": created_at,
            "status": "planned_and_hash_bound_before_write",
            "task_id": plan["task_id"],
            "snapshot_file_sha256": plan["snapshot_file_sha256"],
            "review_pack_file_sha256": plan["review_pack_file_sha256"],
            "decisions_file_sha256": decisions_file_sha,
            "expected_post_apply_canonical_sha256": plan["expected_post_apply_canonical_sha256"],
            "stage_hashes": plan["stage_hashes"],
            "actions": plan["action_log"],
        }
        try:
            validate_review_evidence(decisions, decisions_path=decisions_path)
            atomic_write_json_new(backup_path, backup_payload)
            atomic_write_json_new(action_log_path, log_payload)
        except (FileExistsError, FinalReviewApplyError) as error:
            parser.error(str(error))

        stages = (
            (
                "delete",
                AnnotationUpdateAction.DELETE,
                plan["delete_shapes"],
                True,
            ),
            (
                "update",
                AnnotationUpdateAction.UPDATE,
                plan["update_shapes"],
                True,
            ),
            (
                "add",
                AnnotationUpdateAction.CREATE,
                plan["add_shapes"],
                False,
            ),
        )
        last_stage = "initial"
        for stage, update_action, records, include_id in stages:
            if not records:
                continue
            current = task.get_annotations()
            _verify_stage(
                current,
                expected_sha256=plan["stage_hashes"][last_stage],
                original_shape_ids=plan["original_shape_ids"],
                stage=f"pre-{stage}",
            )
            requests = [
                shape_request_from_record(record, include_id=include_id) for record in records
            ]
            request = models.PatchedLabeledDataRequest(
                version=int(current.version),
                shapes=requests,
            )
            task.update_annotations(request, action=update_action)
            _verify_stage(
                task.get_annotations(),
                expected_sha256=plan["stage_hashes"][stage],
                original_shape_ids=plan["original_shape_ids"],
                stage=stage,
            )
            last_stage = stage

        verified = _verify_stage(
            task.get_annotations(),
            expected_sha256=plan["expected_post_apply_canonical_sha256"],
            original_shape_ids=plan["original_shape_ids"],
            stage="final",
        )
        if verified["tags"] or verified["tracks"]:
            raise RuntimeError("CVAT readback unexpectedly contains tags or tracks")

    summary["dry_run"] = False
    summary["mutation_performed"] = bool(plan["mutation_action_count"])
    summary["verified_post_apply_canonical_sha256"] = plan["expected_post_apply_canonical_sha256"]
    summary["backup_path"] = str(backup_path)
    summary["action_log_path"] = str(action_log_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
