"""Compile an explicit human-review draft into hash-bound full-review decisions."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.snapshot_cvat_task import atomic_write_json_new
from scripts.validate_full_review_decisions import (
    AUTOMATED_FLAG_TYPES,
    DecisionValidationError,
    canonical_shape_sha256,
    validate_decisions,
)

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_ACTIONS = frozenset(
    {"add", "delete", "keep_distinct", "relabel", "relabel_bbox", "update_bbox"}
)
REQUIRED_MANUAL_CHECK_LABELS = frozenset({"traffic_light", "traffic_sign"})
CANDIDATE_EVIDENCE_KEYS = frozenset(
    {
        "candidate_count",
        "candidate_counts_by_label",
        "contact_sheets",
        "frame_count",
        "frames",
        "frames_needing_human_review",
        "frames_with_candidates",
        "model",
        "model_label_mapping",
        "model_sha256",
        "mutation_performed",
        "needs_human_review_count",
        "needs_human_review_counts_by_label",
        "needs_human_review_counts_by_reason",
        "pack_type",
        "parameters",
        "read_only",
        "review_labels",
        "reviewed_by_human",
        "schema_version",
        "source_review_pack",
        "source_review_pack_sha256",
        "source_snapshot_sha256",
        "task_id",
    }
)


class FullReviewCompileError(ValueError):
    """Raised when a human draft is incomplete, stale, or ambiguous."""


def _object(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FullReviewCompileError(f"{location} must be an object")
    return value


def _list(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise FullReviewCompileError(f"{location} must be a list")
    return value


def _strict_int(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FullReviewCompileError(f"{location} must be an integer")
    return value


def _strict_number(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FullReviewCompileError(f"{location} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise FullReviewCompileError(f"{location} must be finite")
    return result


def _sha256(value: Any, location: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise FullReviewCompileError(f"{location} must be a lowercase SHA-256 hex digest")
    return value


def _read_json_bytes(path: Path, location: str) -> tuple[dict[str, Any], bytes]:
    if not path.is_file():
        raise FullReviewCompileError(f"{location} does not exist: {path}")
    try:
        encoded = path.read_bytes()
        payload = json.loads(encoded.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FullReviewCompileError(f"Could not read {location}: {error}") from error
    return _object(payload, location), encoded


def _task_id(snapshot: Mapping[str, Any]) -> int:
    task = snapshot.get("task")
    nested = task.get("id") if isinstance(task, Mapping) else None
    return _strict_int(snapshot.get("task_id", nested), "snapshot task_id")


def _frames(snapshot: Mapping[str, Any]) -> set[int]:
    result: set[int] = set()
    for index, raw in enumerate(_list(snapshot.get("images"), "snapshot.images")):
        image = _object(raw, f"snapshot.images[{index}]")
        frame = _strict_int(
            image.get("frame", image.get("cvat_frame")), f"snapshot.images[{index}].frame"
        )
        if frame in result:
            raise FullReviewCompileError(f"snapshot contains duplicate frame {frame}")
        result.add(frame)
    if not result:
        raise FullReviewCompileError("snapshot must contain at least one frame")
    return result


def _snapshot_frame_metadata(
    snapshot: Mapping[str, Any],
) -> dict[int, tuple[int, int, str]]:
    result: dict[int, tuple[int, int, str]] = {}
    for index, raw in enumerate(_list(snapshot.get("images"), "snapshot.images")):
        image = _object(raw, f"snapshot.images[{index}]")
        frame = _strict_int(
            image.get("frame", image.get("cvat_frame")), f"snapshot.images[{index}].frame"
        )
        if frame in result:
            raise FullReviewCompileError(f"snapshot contains duplicate frame {frame}")
        width = _strict_int(image.get("width"), f"snapshot.images[{index}].width")
        height = _strict_int(image.get("height"), f"snapshot.images[{index}].height")
        if width <= 0 or height <= 0:
            raise FullReviewCompileError(f"snapshot frame {frame} has invalid dimensions")
        source_sha = _sha256(image.get("sha256"), f"snapshot.images[{index}].sha256")
        result[frame] = (width, height, source_sha)
    if not result:
        raise FullReviewCompileError("snapshot must contain at least one frame")
    return result


def _label_names(snapshot: Mapping[str, Any]) -> set[str]:
    names: set[str] = set()
    for index, raw in enumerate(_list(snapshot.get("labels"), "snapshot.labels")):
        label = _object(raw, f"snapshot.labels[{index}]")
        name = label.get("name")
        if not isinstance(name, str) or not name:
            raise FullReviewCompileError(f"snapshot.labels[{index}].name must be non-empty")
        if name in names:
            raise FullReviewCompileError(f"snapshot contains duplicate label {name!r}")
        names.add(name)
    if not names:
        raise FullReviewCompileError("snapshot must contain labels")
    return names


def _shape_index(snapshot: Mapping[str, Any]) -> dict[tuple[type, str], dict[str, Any]]:
    annotations = _object(snapshot.get("annotations"), "snapshot.annotations")
    result: dict[tuple[type, str], dict[str, Any]] = {}
    for index, raw in enumerate(_list(annotations.get("shapes"), "snapshot.annotations.shapes")):
        shape = _object(raw, f"snapshot.annotations.shapes[{index}]")
        shape_id = shape.get("id")
        if isinstance(shape_id, bool) or not isinstance(shape_id, (int, str)):
            raise FullReviewCompileError(f"snapshot.annotations.shapes[{index}].id is invalid")
        identity = (type(shape_id), str(shape_id))
        if identity in result:
            raise FullReviewCompileError(f"snapshot contains duplicate shape ID {shape_id!r}")
        result[identity] = shape
    return result


def _manual_flags_by_frame(
    review_pack: Mapping[str, Any], snapshot_frames: set[int], snapshot_labels: set[str]
) -> dict[int, dict[str, str]]:
    expected_labels_raw = _list(
        review_pack.get("manual_check_labels"), "review_pack.manual_check_labels"
    )
    if (
        not expected_labels_raw
        or any(not isinstance(label, str) or not label for label in expected_labels_raw)
        or len(set(expected_labels_raw)) != len(expected_labels_raw)
    ):
        raise FullReviewCompileError("review_pack.manual_check_labels must be unique and non-empty")
    expected_labels = set(expected_labels_raw)
    if expected_labels != REQUIRED_MANUAL_CHECK_LABELS:
        raise FullReviewCompileError(
            "review_pack.manual_check_labels must be exactly "
            f"{sorted(REQUIRED_MANUAL_CHECK_LABELS)}"
        )
    if not expected_labels <= snapshot_labels:
        raise FullReviewCompileError(
            "review_pack.manual_check_labels contains labels absent from the snapshot"
        )
    result: dict[int, dict[str, str]] = {frame: {} for frame in snapshot_frames}
    seen_frames: set[int] = set()
    for index, raw in enumerate(_list(review_pack.get("frames"), "review_pack.frames")):
        frame_record = _object(raw, f"review_pack.frames[{index}]")
        frame = _strict_int(frame_record.get("frame"), f"review_pack.frames[{index}].frame")
        if frame not in snapshot_frames or frame in seen_frames:
            raise FullReviewCompileError(f"review pack has invalid or duplicate frame {frame}")
        seen_frames.add(frame)
        for flag_index, raw_flag in enumerate(
            _list(frame_record.get("flags"), f"review_pack.frames[{index}].flags")
        ):
            flag = _object(raw_flag, f"review_pack.frames[{index}].flags[{flag_index}]")
            if flag.get("type") != "manual_class_check":
                continue
            label = flag.get("label")
            flag_id = flag.get("flag_id", flag.get("id"))
            if (
                flag.get("id") is not None
                and flag.get("flag_id") is not None
                and flag["id"] != flag["flag_id"]
            ):
                raise FullReviewCompileError(
                    f"manual flag on frame {frame} has conflicting id and flag_id"
                )
            flag_frame = _strict_int(flag.get("frame"), f"manual flag {flag_id!r}.frame")
            if flag_frame != frame:
                raise FullReviewCompileError(
                    f"manual flag {flag_id!r} belongs to frame {flag_frame}, not {frame}"
                )
            if not isinstance(label, str) or not label:
                raise FullReviewCompileError(f"manual flag on frame {frame} has no label")
            if label not in expected_labels:
                raise FullReviewCompileError(
                    f"manual flag on frame {frame} has unexpected label {label!r}"
                )
            if not isinstance(flag_id, str) or not flag_id:
                raise FullReviewCompileError(f"manual flag on frame {frame} has no ID")
            if label in result[frame]:
                raise FullReviewCompileError(
                    f"review pack has duplicate manual flag for frame {frame} label {label}"
                )
            result[frame][label] = flag_id
        if set(result[frame]) != expected_labels:
            raise FullReviewCompileError(
                f"review pack frame {frame} manual flag inventory differs from "
                f"manual_check_labels; missing={sorted(expected_labels - set(result[frame]))}, "
                f"extra={sorted(set(result[frame]) - expected_labels)}"
            )
    if seen_frames != snapshot_frames:
        raise FullReviewCompileError("review pack frame coverage differs from snapshot")
    return result


def _automated_flag_ids(review_pack: Mapping[str, Any]) -> set[str]:
    result: set[str] = set()
    for frame_index, raw_frame in enumerate(_list(review_pack.get("frames"), "review_pack.frames")):
        frame = _object(raw_frame, f"review_pack.frames[{frame_index}]")
        for flag_index, raw_flag in enumerate(
            _list(frame.get("flags"), f"review_pack.frames[{frame_index}].flags")
        ):
            flag = _object(raw_flag, f"review_pack.frames[{frame_index}].flags[{flag_index}]")
            if flag.get("type") not in AUTOMATED_FLAG_TYPES:
                continue
            flag_id = flag.get("flag_id", flag.get("id"))
            if not isinstance(flag_id, str) or not flag_id:
                raise FullReviewCompileError("automated review-pack flag has no ID")
            result.add(flag_id)
    return result


def _counter_mapping(value: Any, location: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for key, raw_count in _object(value, location).items():
        if not isinstance(key, str) or not key:
            raise FullReviewCompileError(f"{location} keys must be non-empty strings")
        count = _strict_int(raw_count, f"{location}.{key}")
        if count < 0:
            raise FullReviewCompileError(f"{location}.{key} must not be negative")
        result[key] = count
    return result


def _validate_evidence_artifact(path: Any, evidence_path: Path, location: str) -> None:
    if not isinstance(path, str) or not path:
        raise FullReviewCompileError(f"{location} must be a non-empty relative path")
    artifact = Path(path)
    if artifact.is_absolute():
        raise FullReviewCompileError(f"{location} must be relative to the candidate pack")
    evidence_dir = evidence_path.parent.resolve()
    resolved = (evidence_dir / artifact).resolve()
    if not resolved.is_relative_to(evidence_dir) or not resolved.is_file():
        raise FullReviewCompileError(f"{location} does not resolve to a candidate-pack artifact")


def _validate_candidate_evidence(
    evidence: Mapping[str, Any],
    evidence_path: Path,
    *,
    task_id: int,
    snapshot_sha256: str,
    review_pack_sha256: str,
    snapshot_frames: Mapping[int, tuple[int, int, str]],
    snapshot_labels: set[str],
    location: str,
) -> set[str]:
    if set(evidence) != CANDIDATE_EVIDENCE_KEYS:
        raise FullReviewCompileError(
            f"{location} has unexpected or missing candidate-pack keys; "
            f"missing={sorted(CANDIDATE_EVIDENCE_KEYS - set(evidence))}, "
            f"extra={sorted(set(evidence) - CANDIDATE_EVIDENCE_KEYS)}"
        )
    if evidence.get("schema_version") != "1.0":
        raise FullReviewCompileError(f"{location}.schema_version is unsupported")
    if evidence.get("pack_type") != "manual_class_model_candidates":
        raise FullReviewCompileError(f"{location} must be a manual class candidate pack")
    if (
        evidence.get("read_only") is not True
        or evidence.get("reviewed_by_human") is not False
        or evidence.get("mutation_performed") is not False
    ):
        raise FullReviewCompileError(
            f"{location} must be generated read-only, unreviewed, and non-mutating"
        )
    if _strict_int(evidence.get("task_id"), f"{location}.task_id") != task_id:
        raise FullReviewCompileError(f"{location} task_id does not match the snapshot")
    if (
        _sha256(evidence.get("source_snapshot_sha256"), f"{location}.source_snapshot_sha256")
        != snapshot_sha256
    ):
        raise FullReviewCompileError(f"{location} is not bound to the supplied snapshot")
    if (
        _sha256(
            evidence.get("source_review_pack_sha256"),
            f"{location}.source_review_pack_sha256",
        )
        != review_pack_sha256
    ):
        raise FullReviewCompileError(f"{location} is not bound to the supplied review pack")
    if (
        not isinstance(evidence.get("source_review_pack"), str)
        or not evidence["source_review_pack"]
    ):
        raise FullReviewCompileError(f"{location}.source_review_pack must be non-empty")
    if not isinstance(evidence.get("model"), str) or not evidence["model"]:
        raise FullReviewCompileError(f"{location}.model must be non-empty")
    _sha256(evidence.get("model_sha256"), f"{location}.model_sha256")
    _object(evidence.get("model_label_mapping"), f"{location}.model_label_mapping")
    _object(evidence.get("parameters"), f"{location}.parameters")

    raw_review_labels = _list(evidence.get("review_labels"), f"{location}.review_labels")
    if (
        not raw_review_labels
        or any(not isinstance(label, str) or not label for label in raw_review_labels)
        or len(set(raw_review_labels)) != len(raw_review_labels)
    ):
        raise FullReviewCompileError(f"{location}.review_labels must be unique and non-empty")
    review_labels = set(raw_review_labels)
    if not review_labels <= snapshot_labels:
        raise FullReviewCompileError(f"{location}.review_labels contains unknown labels")

    raw_frames = _list(evidence.get("frames"), f"{location}.frames")
    if _strict_int(evidence.get("frame_count"), f"{location}.frame_count") != len(raw_frames):
        raise FullReviewCompileError(f"{location}.frame_count does not match frames")
    frames_seen: set[int] = set()
    candidate_ids: set[str] = set()
    candidate_counts: Counter[str] = Counter()
    needs_counts: Counter[str] = Counter()
    needs_reasons: Counter[str] = Counter()
    frames_with_candidates = 0
    frames_needing_review = 0
    total_candidates = 0
    total_needs_review = 0
    for frame_index, raw_frame in enumerate(raw_frames):
        frame_location = f"{location}.frames[{frame_index}]"
        frame_record = _object(raw_frame, frame_location)
        frame = _strict_int(frame_record.get("frame"), f"{frame_location}.frame")
        if frame not in snapshot_frames or frame in frames_seen:
            raise FullReviewCompileError(f"{frame_location} has invalid or duplicate frame {frame}")
        frames_seen.add(frame)
        width, height, source_sha = snapshot_frames[frame]
        if (
            _sha256(frame_record.get("source_sha256"), f"{frame_location}.source_sha256")
            != source_sha
        ):
            raise FullReviewCompileError(f"{frame_location} source hash differs from snapshot")
        if not isinstance(frame_record.get("source_path"), str) or not frame_record["source_path"]:
            raise FullReviewCompileError(f"{frame_location}.source_path must be non-empty")
        _strict_int(frame_record.get("sample_index"), f"{frame_location}.sample_index")
        raw_candidates = _list(frame_record.get("candidates"), f"{frame_location}.candidates")
        candidate_count = _strict_int(
            frame_record.get("candidate_count"), f"{frame_location}.candidate_count"
        )
        if candidate_count != len(raw_candidates):
            raise FullReviewCompileError(f"{frame_location}.candidate_count is inconsistent")
        if candidate_count:
            frames_with_candidates += 1
        frame_needs_review = 0
        for candidate_index, raw_candidate in enumerate(raw_candidates):
            candidate_location = f"{frame_location}.candidates[{candidate_index}]"
            candidate = _object(raw_candidate, candidate_location)
            expected_candidate_keys = {
                "bbox",
                "candidate_id",
                "confidence",
                "existing_overlaps",
                "frame",
                "label",
                "model_label",
                "mutation_performed",
                "review_reason",
                "status",
            }
            if set(candidate) != expected_candidate_keys:
                raise FullReviewCompileError(f"{candidate_location} has unexpected or missing keys")
            candidate_id = candidate.get("candidate_id")
            if (
                not isinstance(candidate_id, str)
                or not candidate_id
                or candidate_id in candidate_ids
            ):
                raise FullReviewCompileError(f"{candidate_location}.candidate_id is invalid")
            candidate_ids.add(candidate_id)
            if _strict_int(candidate.get("frame"), f"{candidate_location}.frame") != frame:
                raise FullReviewCompileError(f"{candidate_location} belongs to another frame")
            label = candidate.get("label")
            if label not in review_labels:
                raise FullReviewCompileError(f"{candidate_location}.label was not requested")
            if not isinstance(candidate.get("model_label"), str) or not candidate["model_label"]:
                raise FullReviewCompileError(f"{candidate_location}.model_label is invalid")
            confidence = _strict_number(
                candidate.get("confidence"), f"{candidate_location}.confidence"
            )
            if not 0 <= confidence <= 1:
                raise FullReviewCompileError(f"{candidate_location}.confidence is outside [0, 1]")
            bbox = _list(candidate.get("bbox"), f"{candidate_location}.bbox")
            if len(bbox) != 4:
                raise FullReviewCompileError(f"{candidate_location}.bbox must have four values")
            x1, y1, x2, y2 = (
                _strict_number(value, f"{candidate_location}.bbox[{index}]")
                for index, value in enumerate(bbox)
            )
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise FullReviewCompileError(f"{candidate_location}.bbox is outside the frame")
            if candidate.get("mutation_performed") is not False:
                raise FullReviewCompileError(f"{candidate_location} must be non-mutating")
            status = candidate.get("status")
            reason = candidate.get("review_reason")
            if status not in {"already_annotated", "needs_human_review"}:
                raise FullReviewCompileError(f"{candidate_location}.status is invalid")
            if reason not in {"same_label_match", "cross_label_overlap", "no_same_label_match"}:
                raise FullReviewCompileError(f"{candidate_location}.review_reason is invalid")
            if (status == "already_annotated") != (reason == "same_label_match"):
                raise FullReviewCompileError(f"{candidate_location} status/reason is inconsistent")
            _list(candidate.get("existing_overlaps"), f"{candidate_location}.existing_overlaps")
            candidate_counts[str(label)] += 1
            if status == "needs_human_review":
                frame_needs_review += 1
                needs_counts[str(label)] += 1
                needs_reasons[str(reason)] += 1
        declared_frame_needs = _strict_int(
            frame_record.get("needs_human_review_count"),
            f"{frame_location}.needs_human_review_count",
        )
        if declared_frame_needs != frame_needs_review:
            raise FullReviewCompileError(
                f"{frame_location}.needs_human_review_count is inconsistent"
            )
        frame_keys = {
            "candidate_count",
            "candidates",
            "frame",
            "needs_human_review_count",
            "sample_index",
            "source_path",
            "source_sha256",
        }
        if frame_needs_review:
            frame_keys.add("overlay")
            _validate_evidence_artifact(
                frame_record.get("overlay"), evidence_path, f"{frame_location}.overlay"
            )
            frames_needing_review += 1
        if set(frame_record) != frame_keys:
            raise FullReviewCompileError(f"{frame_location} has unexpected or missing keys")
        total_candidates += candidate_count
        total_needs_review += frame_needs_review
    if frames_seen != set(snapshot_frames):
        raise FullReviewCompileError(f"{location}.frames does not cover the snapshot exactly")

    expected_scalars = {
        "candidate_count": total_candidates,
        "needs_human_review_count": total_needs_review,
        "frames_with_candidates": frames_with_candidates,
        "frames_needing_human_review": frames_needing_review,
    }
    for key, expected in expected_scalars.items():
        if _strict_int(evidence.get(key), f"{location}.{key}") != expected:
            raise FullReviewCompileError(f"{location}.{key} is inconsistent")
    expected_counters = {
        "candidate_counts_by_label": dict(sorted(candidate_counts.items())),
        "needs_human_review_counts_by_label": dict(sorted(needs_counts.items())),
        "needs_human_review_counts_by_reason": dict(sorted(needs_reasons.items())),
    }
    for key, expected in expected_counters.items():
        if _counter_mapping(evidence.get(key), f"{location}.{key}") != expected:
            raise FullReviewCompileError(f"{location}.{key} is inconsistent")
    contact_sheets = _list(evidence.get("contact_sheets"), f"{location}.contact_sheets")
    if len(set(contact_sheets)) != len(contact_sheets):
        raise FullReviewCompileError(f"{location}.contact_sheets contains duplicates")
    for index, artifact in enumerate(contact_sheets):
        _validate_evidence_artifact(artifact, evidence_path, f"{location}.contact_sheets[{index}]")
    if bool(frames_needing_review) != bool(contact_sheets):
        raise FullReviewCompileError(f"{location}.contact_sheets coverage is inconsistent")
    return review_labels


def _validate_review_evidence(
    draft: Mapping[str, Any],
    draft_path: Path,
    output_path: Path,
    *,
    snapshot: Mapping[str, Any],
    task_id: int,
    snapshot_sha256: str,
    review_pack_sha256: str,
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    resolved_paths: set[Path] = set()
    covered_review_labels: set[str] = set()
    snapshot_frames = _snapshot_frame_metadata(snapshot)
    snapshot_labels = _label_names(snapshot)
    for index, raw in enumerate(_list(draft.get("review_evidence"), "draft.review_evidence")):
        record = _object(raw, f"draft.review_evidence[{index}]")
        if set(record) != {"path", "sha256"}:
            raise FullReviewCompileError(
                f"draft.review_evidence[{index}] must contain exactly path and sha256"
            )
        raw_path = record.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise FullReviewCompileError(f"draft.review_evidence[{index}].path is invalid")
        evidence_path = Path(raw_path)
        if not evidence_path.is_absolute():
            evidence_path = (draft_path.parent / evidence_path).resolve()
        else:
            evidence_path = evidence_path.resolve()
        if evidence_path in resolved_paths:
            raise FullReviewCompileError(
                f"draft.review_evidence[{index}] duplicates {evidence_path}"
            )
        resolved_paths.add(evidence_path)
        if not evidence_path.is_file():
            raise FullReviewCompileError(
                f"draft.review_evidence[{index}] does not exist: {evidence_path}"
            )
        expected = _sha256(record.get("sha256"), f"draft.review_evidence[{index}].sha256")
        evidence, evidence_bytes = _read_json_bytes(
            evidence_path, f"draft.review_evidence[{index}] candidate pack"
        )
        if hashlib.sha256(evidence_bytes).hexdigest() != expected:
            raise FullReviewCompileError(
                f"draft.review_evidence[{index}] SHA-256 does not match {evidence_path}"
            )
        covered_review_labels.update(
            _validate_candidate_evidence(
                evidence,
                evidence_path,
                task_id=task_id,
                snapshot_sha256=snapshot_sha256,
                review_pack_sha256=review_pack_sha256,
                snapshot_frames=snapshot_frames,
                snapshot_labels=snapshot_labels,
                location=f"draft.review_evidence[{index}]",
            )
        )
        output_relative_path = os.path.relpath(evidence_path, start=output_path.parent)
        records.append({"path": output_relative_path, "sha256": expected})
    if not records:
        raise FullReviewCompileError("draft.review_evidence must not be empty")
    if not REQUIRED_MANUAL_CHECK_LABELS <= covered_review_labels:
        raise FullReviewCompileError(
            "draft.review_evidence candidate packs do not cover all required manual labels"
        )
    return records


def _compile_action(
    raw: Any,
    *,
    location: str,
    frame: int,
    shapes: Mapping[tuple[type, str], Mapping[str, Any]],
    labels: set[str],
) -> dict[str, Any]:
    action = _object(raw, location)
    kind = action.get("action")
    if kind not in ALLOWED_ACTIONS:
        raise FullReviewCompileError(f"{location}.action must be one of {sorted(ALLOWED_ACTIONS)}")
    allowed_keys = {
        "add": {"action", "label", "points"},
        "delete": {"action", "shape_id"},
        "keep_distinct": {"action", "shape_id"},
        "relabel": {"action", "shape_id", "to_label"},
        "relabel_bbox": {"action", "shape_id", "to_label", "points"},
        "update_bbox": {"action", "shape_id", "points"},
    }[str(kind)]
    if set(action) != allowed_keys:
        raise FullReviewCompileError(
            f"{location} must contain exactly {sorted(allowed_keys)}, got {sorted(action)}"
        )
    if kind == "add":
        label = action.get("label")
        if label not in labels:
            raise FullReviewCompileError(f"{location}.label is unknown: {label!r}")
        points = action.get("points")
        if not isinstance(points, list) or len(points) != 4:
            raise FullReviewCompileError(f"{location}.points must contain four coordinates")
        return {"action": "add", "label": label, "points": points}

    shape_id = action.get("shape_id")
    if isinstance(shape_id, bool) or not isinstance(shape_id, (int, str)):
        raise FullReviewCompileError(f"{location}.shape_id is invalid")
    shape = shapes.get((type(shape_id), str(shape_id)))
    if shape is None:
        raise FullReviewCompileError(f"{location} references unknown shape {shape_id!r}")
    shape_frame = _strict_int(shape.get("frame"), f"shape {shape_id!r}.frame")
    if shape_frame != frame:
        raise FullReviewCompileError(
            f"{location} shape {shape_id!r} belongs to frame {shape_frame}, not {frame}"
        )
    compiled = {
        "action": kind,
        "shape_id": shape_id,
        "expected_shape_sha256": canonical_shape_sha256(shape),
    }
    if kind in {"relabel", "relabel_bbox"}:
        to_label = action.get("to_label")
        if to_label not in labels:
            raise FullReviewCompileError(f"{location}.to_label is unknown: {to_label!r}")
        compiled["to_label"] = to_label
    if kind in {"relabel_bbox", "update_bbox"}:
        points = action.get("points")
        if not isinstance(points, list) or len(points) != 4:
            raise FullReviewCompileError(f"{location}.points must contain four coordinates")
        compiled["points"] = points
    return compiled


def compile_decisions(
    snapshot: Mapping[str, Any],
    review_pack: Mapping[str, Any],
    automated_decisions: Mapping[str, Any],
    draft: Mapping[str, Any],
    *,
    snapshot_sha256: str,
    review_pack_sha256: str,
    automated_decisions_sha256: str,
    draft_sha256: str,
    review_evidence: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    """Compile parsed inputs and validate the exact resulting full-review payload."""

    snapshot = _object(snapshot, "snapshot")
    review_pack = _object(review_pack, "review_pack")
    automated_decisions = _object(automated_decisions, "automated_decisions")
    draft = _object(draft, "draft")
    draft_schema_version = draft.get("schema_version")
    if draft_schema_version not in {"1.1", "1.2"} or draft.get("draft_type") != "full_review_human":
        raise FullReviewCompileError("draft schema_version/draft_type is unsupported")
    legacy_draft_keys = {
        "schema_version",
        "draft_type",
        "task_id",
        "snapshot_sha256",
        "review_pack_sha256",
        "automated_decisions_sha256",
        "reviewer",
        "reviewed_at",
        "mutation_performed",
        "automated_flag_overrides",
        "review_evidence",
        "frame_reviews",
    }
    expected_draft_keys = (
        legacy_draft_keys
        if draft_schema_version == "1.1"
        else legacy_draft_keys | {"manual_delete_approvals"}
    )
    if set(draft) != expected_draft_keys:
        raise FullReviewCompileError(
            "draft has unexpected or missing top-level keys; "
            f"missing={sorted(expected_draft_keys - set(draft))}, "
            f"extra={sorted(set(draft) - expected_draft_keys)}"
        )
    if draft.get("mutation_performed") is not False:
        raise FullReviewCompileError("draft.mutation_performed must be false")
    task_id = _task_id(snapshot)
    if _strict_int(draft.get("task_id"), "draft.task_id") != task_id:
        raise FullReviewCompileError("draft.task_id does not match snapshot")
    if _sha256(draft.get("snapshot_sha256"), "draft.snapshot_sha256") != snapshot_sha256:
        raise FullReviewCompileError("draft.snapshot_sha256 does not match snapshot file")
    if _sha256(draft.get("review_pack_sha256"), "draft.review_pack_sha256") != review_pack_sha256:
        raise FullReviewCompileError("draft.review_pack_sha256 does not match review pack file")
    if (
        _sha256(draft.get("automated_decisions_sha256"), "draft.automated_decisions_sha256")
        != automated_decisions_sha256
    ):
        raise FullReviewCompileError(
            "draft.automated_decisions_sha256 does not match automated decisions file"
        )
    reviewer = draft.get("reviewer")
    reviewed_at = draft.get("reviewed_at")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise FullReviewCompileError("draft.reviewer must be non-empty")
    if not isinstance(reviewed_at, str) or not reviewed_at.strip():
        raise FullReviewCompileError("draft.reviewed_at must be non-empty")
    manual_delete_approvals = copy.deepcopy(
        _list(
            draft.get("manual_delete_approvals", []),
            "draft.manual_delete_approvals",
        )
    )

    try:
        automated_summary = validate_decisions(
            snapshot,
            review_pack,
            automated_decisions,
            snapshot_sha256=snapshot_sha256,
            review_pack_sha256=review_pack_sha256,
        )
    except DecisionValidationError as error:
        raise FullReviewCompileError(f"automated decisions are invalid: {error}") from error
    if automated_summary.get("scope") != "automated_risk_cleanup":
        raise FullReviewCompileError("automated decisions must use automated_risk_cleanup scope")

    snapshot_frames = _frames(snapshot)
    labels = _label_names(snapshot)
    shapes = _shape_index(snapshot)
    manual_flags = _manual_flags_by_frame(review_pack, snapshot_frames, labels)
    automated_flags = _automated_flag_ids(review_pack)
    base_reviews: dict[int, Mapping[str, Any]] = {}
    for index, raw in enumerate(
        _list(automated_decisions.get("frame_reviews"), "automated_decisions.frame_reviews")
    ):
        review = _object(raw, f"automated_decisions.frame_reviews[{index}]")
        frame = _strict_int(review.get("frame"), f"automated frame review {index}")
        resolved = _list(
            review.get("resolved_flag_ids"),
            f"automated_decisions.frame_reviews[{index}].resolved_flag_ids",
        )
        if not set(resolved) <= automated_flags:
            raise FullReviewCompileError(f"automated frame {frame} resolves non-automated flags")
        for action_index, raw_action in enumerate(
            _list(review.get("actions"), f"automated frame {frame} actions")
        ):
            action = _object(raw_action, f"automated frame {frame} action {action_index}")
            action_flags = _list(
                action.get("resolves_flag_ids", []),
                f"automated frame {frame} action {action_index}.resolves_flag_ids",
            )
            if not action_flags or not set(action_flags) <= automated_flags:
                raise FullReviewCompileError(
                    f"automated frame {frame} action {action_index} is not exclusively bound "
                    "to automated flags"
                )
        base_reviews[frame] = review

    action_overrides: dict[tuple[int, int], dict[str, Any]] = {}
    overridden_flags: set[str] = set()
    for index, raw in enumerate(
        _list(draft.get("automated_flag_overrides", []), "draft.automated_flag_overrides")
    ):
        override = _object(raw, f"draft.automated_flag_overrides[{index}]")
        if set(override) != {"flag_id", "frame", "replacement_action"}:
            raise FullReviewCompileError(
                f"draft.automated_flag_overrides[{index}] has unexpected or missing keys"
            )
        flag_id = override.get("flag_id")
        if not isinstance(flag_id, str) or not flag_id:
            raise FullReviewCompileError(
                f"draft.automated_flag_overrides[{index}].flag_id must be non-empty"
            )
        if flag_id in overridden_flags:
            raise FullReviewCompileError(f"automated flag {flag_id!r} is overridden twice")
        frame = _strict_int(override.get("frame"), f"draft.automated_flag_overrides[{index}].frame")
        base = base_reviews.get(frame)
        if base is None or flag_id not in base.get("resolved_flag_ids", []):
            raise FullReviewCompileError(
                f"automated flag override {flag_id!r} is not resolved on base frame {frame}"
            )
        matching_actions: list[tuple[int, Mapping[str, Any]]] = []
        for action_index, raw_action in enumerate(base.get("actions", [])):
            base_action = _object(raw_action, f"automated frame {frame} action {action_index}")
            if flag_id in base_action.get("resolves_flag_ids", []):
                matching_actions.append((action_index, base_action))
        if len(matching_actions) != 1:
            raise FullReviewCompileError(
                f"automated flag {flag_id!r} must bind exactly one replaceable base action"
            )
        action_index, base_action = matching_actions[0]
        if set(base_action.get("resolves_flag_ids", [])) != {flag_id}:
            raise FullReviewCompileError(
                f"base action for {flag_id!r} also resolves another flag and cannot be overridden"
            )
        replacement = _compile_action(
            override.get("replacement_action"),
            location=f"draft.automated_flag_overrides[{index}].replacement_action",
            frame=frame,
            shapes=shapes,
            labels=labels,
        )
        if replacement["action"] == "add":
            raise FullReviewCompileError("an add action cannot replace an existing-box risk action")
        replacement["resolves_flag_ids"] = [flag_id]
        action_overrides[(frame, action_index)] = replacement
        overridden_flags.add(flag_id)

    draft_reviews: dict[int, dict[str, Any]] = {}
    for index, raw in enumerate(_list(draft.get("frame_reviews"), "draft.frame_reviews")):
        review = _object(raw, f"draft.frame_reviews[{index}]")
        if set(review) != {"frame", "reviewed", "labels_reviewed", "actions"}:
            raise FullReviewCompileError(
                f"draft.frame_reviews[{index}] has unexpected or missing keys"
            )
        frame = _strict_int(review.get("frame"), f"draft.frame_reviews[{index}].frame")
        if frame not in snapshot_frames or frame in draft_reviews:
            raise FullReviewCompileError(f"draft has invalid or duplicate frame {frame}")
        if review.get("reviewed") is not True:
            raise FullReviewCompileError(f"draft frame {frame} must be explicitly reviewed")
        raw_labels = _list(
            review.get("labels_reviewed"), f"draft.frame_reviews[{index}].labels_reviewed"
        )
        if any(not isinstance(label, str) for label in raw_labels) or len(set(raw_labels)) != len(
            raw_labels
        ):
            raise FullReviewCompileError(f"draft frame {frame} labels_reviewed is invalid")
        if set(raw_labels) != labels:
            raise FullReviewCompileError(
                f"draft frame {frame} must explicitly review every label; "
                f"missing={sorted(labels - set(raw_labels))}, extra={sorted(set(raw_labels) - labels)}"
            )
        compiled_actions = [
            _compile_action(
                action,
                location=f"draft.frame_reviews[{index}].actions[{action_index}]",
                frame=frame,
                shapes=shapes,
                labels=labels,
            )
            for action_index, action in enumerate(
                _list(review.get("actions"), f"draft.frame_reviews[{index}].actions")
            )
        ]
        draft_reviews[frame] = {
            "frame": frame,
            "reviewed": True,
            "labels_reviewed": sorted(raw_labels),
            "actions": compiled_actions,
        }
    if set(draft_reviews) != snapshot_frames:
        raise FullReviewCompileError(
            "draft must contain every frame exactly once; "
            f"missing={sorted(snapshot_frames - set(draft_reviews))}"
        )

    frame_reviews: list[dict[str, Any]] = []
    for frame in sorted(snapshot_frames):
        base = base_reviews.get(frame, {})
        resolved_flags = list(base.get("resolved_flag_ids", []))
        for label in sorted(manual_flags[frame]):
            flag_id = manual_flags[frame][label]
            if flag_id not in resolved_flags:
                resolved_flags.append(flag_id)
        base_actions = [
            action_overrides.get((frame, index), dict(action))
            for index, action in enumerate(base.get("actions", []))
        ]
        draft_review = draft_reviews[frame]
        frame_reviews.append(
            {
                "frame": frame,
                "reviewed": True,
                "labels_reviewed": draft_review["labels_reviewed"],
                "resolved_flag_ids": resolved_flags,
                "actions": [*base_actions, *draft_review["actions"]],
            }
        )

    annotations = _object(snapshot.get("annotations"), "snapshot.annotations")
    annotation_sha = _sha256(
        snapshot.get("canonical_annotations_sha256"),
        "snapshot.canonical_annotations_sha256",
    )
    result = {
        "schema_version": "1.3",
        "scope": "full_review",
        "task_id": task_id,
        "snapshot_sha256": snapshot_sha256,
        "canonical_annotations_sha256": annotation_sha,
        "review_pack_sha256": review_pack_sha256,
        "automated_risk_decisions_sha256": automated_decisions_sha256,
        "human_review_draft_sha256": draft_sha256,
        "review_evidence": [dict(item) for item in review_evidence],
        "reviewer": reviewer,
        "reviewed_at": reviewed_at,
        "mutation_performed": False,
        "manual_delete_approvals": manual_delete_approvals,
        "frame_reviews": frame_reviews,
    }
    if not annotations:
        raise FullReviewCompileError("snapshot annotations must not be empty")
    try:
        summary = validate_decisions(
            snapshot,
            review_pack,
            result,
            snapshot_sha256=snapshot_sha256,
            review_pack_sha256=review_pack_sha256,
        )
    except DecisionValidationError as error:
        raise FullReviewCompileError(
            f"compiled full-review decisions are invalid: {error}"
        ) from error
    if summary.get("scope") != "full_review" or summary.get("unresolved_manual_flag_count") != 0:
        raise FullReviewCompileError("compiled decisions did not satisfy the full-review gate")
    return result


def compile_decision_files(
    snapshot_path: Path | str,
    review_pack_path: Path | str,
    automated_decisions_path: Path | str,
    draft_path: Path | str,
    output_path: Path | str,
) -> dict[str, Any]:
    """Read, bind, compile, validate, and exclusively publish five file paths."""

    snapshot_path = Path(snapshot_path).resolve()
    review_pack_path = Path(review_pack_path).resolve()
    automated_decisions_path = Path(automated_decisions_path).resolve()
    draft_path = Path(draft_path).resolve()
    output_path = Path(output_path).resolve()
    snapshot, snapshot_bytes = _read_json_bytes(snapshot_path, "snapshot")
    review_pack, review_pack_bytes = _read_json_bytes(review_pack_path, "review pack")
    automated_decisions, automated_bytes = _read_json_bytes(
        automated_decisions_path, "automated decisions"
    )
    draft, draft_bytes = _read_json_bytes(draft_path, "human review draft")
    snapshot_sha256 = hashlib.sha256(snapshot_bytes).hexdigest()
    review_pack_sha256 = hashlib.sha256(review_pack_bytes).hexdigest()
    automated_decisions_sha256 = hashlib.sha256(automated_bytes).hexdigest()
    review_evidence = _validate_review_evidence(
        draft,
        draft_path,
        output_path,
        snapshot=snapshot,
        task_id=_task_id(snapshot),
        snapshot_sha256=snapshot_sha256,
        review_pack_sha256=review_pack_sha256,
    )
    result = compile_decisions(
        snapshot,
        review_pack,
        automated_decisions,
        draft,
        snapshot_sha256=snapshot_sha256,
        review_pack_sha256=review_pack_sha256,
        automated_decisions_sha256=automated_decisions_sha256,
        draft_sha256=hashlib.sha256(draft_bytes).hexdigest(),
        review_evidence=review_evidence,
    )
    if (
        _validate_review_evidence(
            draft,
            draft_path,
            output_path,
            snapshot=snapshot,
            task_id=_task_id(snapshot),
            snapshot_sha256=snapshot_sha256,
            review_pack_sha256=review_pack_sha256,
        )
        != review_evidence
    ):
        raise FullReviewCompileError("review evidence changed while decisions were compiled")
    atomic_write_json_new(output_path, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("review_pack", type=Path)
    parser.add_argument("automated_decisions", type=Path)
    parser.add_argument("draft", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        result = compile_decision_files(
            args.snapshot,
            args.review_pack,
            args.automated_decisions,
            args.draft,
            args.output,
        )
    except (FileExistsError, FullReviewCompileError) as error:
        raise SystemExit(str(error)) from error
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "task_id": result["task_id"],
                "frame_count": len(result["frame_reviews"]),
                "action_count": sum(len(review["actions"]) for review in result["frame_reviews"]),
                "mutation_performed": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
