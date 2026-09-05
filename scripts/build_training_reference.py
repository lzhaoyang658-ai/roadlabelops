"""Build an immutable, evidence-bound COCO training reference offline.

The source-frame map is deliberately independent of scene segmentation.  Its
version-2 schema is::

    {
      "schema": {"name": "roadlabelops.training-source-frame-map", "version": 2},
      "assets": [{"asset_id": "video-a", ...}],
      "frames": [
        {
          "scene_id": "scene-001", "source_frame": 10,
          "asset_id": "video-a",
          "leakage_group_id": "sha256:<source-sha256>",
          "normalized_asset_frame": 110
        }
      ]
    }

Each asset requires ``asset_id``, its unique source ``sha256``, and the matching
``leakage_group_id``; additional audit metadata remains in the source-map
evidence rather than being copied into COCO.  A frame record maps one manifest
``(scene_id, source_frame)`` identity to one normalized
``(asset_id, normalized_asset_frame)`` identity.  It deliberately does not
claim to recover a native frame number after FPS normalization.  This permits
both a scene to span assets and an asset to span scenes without relying on
naming conventions.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import errno
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import yaml
from PIL import Image

from scripts.snapshot_cvat_task import canonical_sha256, canonicalize_annotations

SNAPSHOT_SCHEMA = {"name": "roadlabelops.cvat-task-snapshot", "version": 1}
RECEIPT_SCHEMA = {"name": "roadlabelops.cvat-job-completion-receipt", "version": 1}
SOURCE_MAP_SCHEMA = {"name": "roadlabelops.training-source-frame-map", "version": 2}
OUTPUT_SCHEMA = {"name": "roadlabelops.training-coco-reference", "version": 2}
REQUIRED_LABELS = (
    "car",
    "bus",
    "truck",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_light",
    "traffic_sign",
)
ALLOWED_PRE_COMPLETION_STATES = frozenset({"new", "in_progress"})


class TrainingReferenceError(ValueError):
    """Raised when immutable training-reference publication is unsafe."""


@dataclass(frozen=True)
class InputPaths:
    """Paths binding one completed task snapshot, manifest, and receipt."""

    snapshot: Path
    image_manifest: Path
    completion_receipt: Path


@dataclass(frozen=True)
class ValidatedImage:
    task_id: int
    cvat_frame: int
    sample_index: int
    scene_id: str
    source_frame: int
    file_name: str
    source_path: Path
    sha256: str
    size_bytes: int
    width: int
    height: int
    asset_id: int | str
    leakage_group_id: str
    normalized_asset_frame: int


@dataclass(frozen=True)
class ValidatedTask:
    task_id: int
    job_id: int
    snapshot_path: Path
    snapshot_sha256: str
    manifest_path: Path
    manifest_sha256: str
    receipt_path: Path
    receipt_sha256: str
    annotations_sha256: str
    label_id_to_name: Mapping[int, str]
    images: tuple[ValidatedImage, ...]
    shapes: tuple[dict[str, Any], ...]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TrainingReferenceError(f"{location} must be an object")
    return value


def _list(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise TrainingReferenceError(f"{location} must be a list")
    return value


def _integer(value: Any, location: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TrainingReferenceError(f"{location} must be an integer")
    if minimum is not None and value < minimum:
        raise TrainingReferenceError(f"{location} must be at least {minimum}")
    return value


def _number(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainingReferenceError(f"{location} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise TrainingReferenceError(f"{location} must be finite")
    return result


def _text(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TrainingReferenceError(f"{location} must be a non-empty string")
    return value


def _sha256(value: Any, location: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TrainingReferenceError(f"{location} must be a lowercase SHA-256 digest")
    return value


def _asset_id(value: Any, location: str) -> int | str:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise TrainingReferenceError(f"{location} must be an integer or string")
    if isinstance(value, str) and not value:
        raise TrainingReferenceError(f"{location} must not be empty")
    return value


def _asset_identity(value: int | str) -> tuple[str, str]:
    return type(value).__name__, str(value)


def _leakage_group_id(value: Any, source_sha256: str, location: str) -> str:
    expected = f"sha256:{source_sha256}"
    if _text(value, location) != expected:
        raise TrainingReferenceError(f"{location} must be the source content identity {expected!r}")
    return expected


def _read_json(path: Path, location: str) -> tuple[dict[str, Any], str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise TrainingReferenceError(f"{location} does not exist: {resolved}")
    try:
        encoded = resolved.read_bytes()
        payload = json.loads(encoded.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TrainingReferenceError(f"Could not read {location}: {error}") from error
    return _object(payload, location), hashlib.sha256(encoded).hexdigest()


def _safe_image_path(manifest_path: Path, file_name: str) -> Path:
    posix_path = PurePosixPath(file_name)
    if posix_path.is_absolute() or ".." in posix_path.parts or not posix_path.name:
        raise TrainingReferenceError(f"Manifest image file_name is unsafe: {file_name!r}")
    image_root = (manifest_path.parent / "images").resolve()
    image_path = (image_root / Path(*posix_path.parts)).resolve()
    if not image_path.is_relative_to(image_root):
        raise TrainingReferenceError(f"Manifest image escapes its images directory: {file_name!r}")
    return image_path


def _read_labels(path: Path) -> tuple[tuple[str, ...], str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise TrainingReferenceError(f"Labels config does not exist: {resolved}")
    encoded = resolved.read_bytes()
    try:
        payload = yaml.safe_load(encoded.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise TrainingReferenceError(f"Could not read labels config: {error}") from error
    root = _object(payload, "labels config")
    records = _list(root.get("labels"), "labels config.labels")
    names = tuple(
        _text(_object(record, f"labels config.labels[{index}]").get("name"), "label name")
        for index, record in enumerate(records)
    )
    if names != REQUIRED_LABELS:
        raise TrainingReferenceError(
            "labels config must contain the fixed eight-class taxonomy in canonical order; "
            f"expected {list(REQUIRED_LABELS)}, got {list(names)}"
        )
    return names, hashlib.sha256(encoded).hexdigest()


def _read_source_map(
    path: Path,
) -> tuple[
    dict[tuple[str, int], tuple[int | str, str, int]],
    dict[tuple[str, str], dict[str, Any]],
    str,
]:
    payload, digest = _read_json(path, "source map")
    if payload.get("schema") != SOURCE_MAP_SCHEMA:
        raise TrainingReferenceError("source map schema is unsupported")

    assets: dict[tuple[str, str], dict[str, Any]] = {}
    source_hashes: dict[str, int | str] = {}
    for index, raw_asset in enumerate(_list(payload.get("assets"), "source map.assets")):
        asset = _object(raw_asset, f"source map.assets[{index}]")
        identifier = _asset_id(asset.get("asset_id"), f"source map.assets[{index}].asset_id")
        identity = _asset_identity(identifier)
        if identity in assets:
            raise TrainingReferenceError(f"source map contains duplicate asset_id {identifier!r}")
        source_sha256 = _sha256(asset.get("sha256"), f"source map.assets[{index}].sha256")
        if source_sha256 in source_hashes:
            raise TrainingReferenceError(
                "source map aliases the same source SHA-256 under multiple asset IDs: "
                f"{source_hashes[source_sha256]!r} and {identifier!r}"
            )
        source_hashes[source_sha256] = identifier
        _leakage_group_id(
            asset.get("leakage_group_id"),
            source_sha256,
            f"source map.assets[{index}].leakage_group_id",
        )
        assets[identity] = asset
    if not assets:
        raise TrainingReferenceError("source map.assets must not be empty")

    frames: dict[tuple[str, int], tuple[int | str, str, int]] = {}
    normalized_identities: set[tuple[tuple[str, str], int]] = set()
    for index, raw_frame in enumerate(_list(payload.get("frames"), "source map.frames")):
        frame = _object(raw_frame, f"source map.frames[{index}]")
        scene_id = _text(frame.get("scene_id"), f"source map.frames[{index}].scene_id")
        source_frame = _integer(
            frame.get("source_frame"),
            f"source map.frames[{index}].source_frame",
            minimum=0,
        )
        identifier = _asset_id(frame.get("asset_id"), f"source map.frames[{index}].asset_id")
        asset_identity = _asset_identity(identifier)
        if asset_identity not in assets:
            raise TrainingReferenceError(
                f"source map frame {scene_id!r}/{source_frame} references unknown asset "
                f"{identifier!r}"
            )
        asset = assets[asset_identity]
        leakage_group_id = _leakage_group_id(
            frame.get("leakage_group_id"),
            _sha256(asset.get("sha256"), "source map asset.sha256"),
            f"source map.frames[{index}].leakage_group_id",
        )
        normalized_asset_frame = _integer(
            frame.get("normalized_asset_frame"),
            f"source map.frames[{index}].normalized_asset_frame",
            minimum=0,
        )
        key = (scene_id, source_frame)
        if key in frames:
            raise TrainingReferenceError(f"source map contains duplicate frame mapping {key!r}")
        normalized_identity = (asset_identity, normalized_asset_frame)
        if normalized_identity in normalized_identities:
            raise TrainingReferenceError(
                "source map maps more than one scene frame to normalized asset frame "
                f"{identifier!r}/{normalized_asset_frame}"
            )
        frames[key] = (identifier, leakage_group_id, normalized_asset_frame)
        normalized_identities.add(normalized_identity)
    if not frames:
        raise TrainingReferenceError("source map.frames must not be empty")
    return frames, assets, digest


def _validate_job(
    snapshot: Mapping[str, Any], *, task_id: int, frame_count: int
) -> tuple[int, dict[str, Any]]:
    jobs = _list(snapshot.get("jobs"), "snapshot.jobs")
    if len(jobs) != 1:
        raise TrainingReferenceError("each post-completion snapshot must contain exactly one job")
    job = _object(jobs[0], "snapshot.jobs[0]")
    job_id = _integer(job.get("id"), "snapshot.jobs[0].id")
    if _integer(job.get("task_id"), "snapshot.jobs[0].task_id") != task_id:
        raise TrainingReferenceError("snapshot job belongs to another task")
    if job.get("state") != "completed":
        raise TrainingReferenceError("snapshot job state must be 'completed'")
    if job.get("type", "annotation") != "annotation" or job.get("parent_job_id") is not None:
        raise TrainingReferenceError("snapshot job must be a primary annotation job")
    issues = _object(job.get("issues"), "snapshot.jobs[0].issues")
    if _integer(issues.get("count"), "snapshot.jobs[0].issues.count", minimum=0) != 0:
        raise TrainingReferenceError("snapshot job must have zero unresolved issues")
    expected = (0, frame_count - 1, frame_count)
    actual = (
        _integer(job.get("start_frame"), "snapshot.jobs[0].start_frame"),
        _integer(job.get("stop_frame"), "snapshot.jobs[0].stop_frame"),
        _integer(job.get("frame_count"), "snapshot.jobs[0].frame_count"),
    )
    if actual != expected:
        raise TrainingReferenceError(
            f"snapshot job range/count {actual!r} does not cover task frames {expected!r}"
        )
    return job_id, job


def _aware_datetime(value: Any, location: str) -> datetime:
    text = _text(value, location)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise TrainingReferenceError(f"{location} is not ISO-8601") from error
    if parsed.tzinfo is None:
        raise TrainingReferenceError(f"{location} must include a timezone")
    return parsed


def _resolve_receipt_evidence_path(value: Any, *, receipt_path: Path) -> Path:
    raw_path = Path(_text(value, "completion receipt.evidence.post_snapshot.path"))
    if not raw_path.is_absolute():
        raw_path = receipt_path.resolve().parent / raw_path
    return raw_path.resolve()


def _validate_pre_completion_snapshot(
    snapshot: Mapping[str, Any],
    *,
    task_id: int,
    job_id: int,
    frame_count: int,
    annotation_sha256: str,
    annotation_count: int,
    receipt: Mapping[str, Any],
    completed_at: datetime,
) -> None:
    """Validate the exact post-apply/pre-completion snapshot named by a receipt."""

    location = "completion receipt pre-completion snapshot"
    if snapshot.get("snapshot_schema") != SNAPSHOT_SCHEMA:
        raise TrainingReferenceError(f"{location} schema is unsupported")
    task = _object(snapshot.get("task"), f"{location}.task")
    if _integer(task.get("id"), f"{location}.task.id") != task_id:
        raise TrainingReferenceError(f"{location} belongs to another task")
    if _integer(task.get("size"), f"{location}.task.size", minimum=1) != frame_count:
        raise TrainingReferenceError(f"{location} task size differs from the completed snapshot")

    jobs = _list(snapshot.get("jobs"), f"{location}.jobs")
    if len(jobs) != 1:
        raise TrainingReferenceError(f"{location} must contain exactly one job")
    job = _object(jobs[0], f"{location}.jobs[0]")
    if _integer(job.get("id"), f"{location}.jobs[0].id") != job_id:
        raise TrainingReferenceError(f"{location} is bound to another job")
    if _integer(job.get("task_id"), f"{location}.jobs[0].task_id") != task_id:
        raise TrainingReferenceError(f"{location} job belongs to another task")
    if job.get("type", "annotation") != "annotation" or job.get("parent_job_id") is not None:
        raise TrainingReferenceError(f"{location} job must be a primary annotation job")
    expected_range = (0, frame_count - 1, frame_count)
    actual_range = (
        _integer(job.get("start_frame"), f"{location}.jobs[0].start_frame"),
        _integer(job.get("stop_frame"), f"{location}.jobs[0].stop_frame"),
        _integer(job.get("frame_count"), f"{location}.jobs[0].frame_count"),
    )
    if actual_range != expected_range:
        raise TrainingReferenceError(
            f"{location} job range/count {actual_range!r} differs from {expected_range!r}"
        )
    issues = _object(job.get("issues"), f"{location}.jobs[0].issues")
    if _integer(issues.get("count"), f"{location}.jobs[0].issues.count", minimum=0) != 0:
        raise TrainingReferenceError(f"{location} job has unresolved issues")

    before_state = _text(receipt.get("job_state_before"), "completion receipt.job_state_before")
    if before_state not in ALLOWED_PRE_COMPLETION_STATES:
        raise TrainingReferenceError(
            "completion receipt job_state_before must be 'new' or 'in_progress'"
        )
    if job.get("state") not in ALLOWED_PRE_COMPLETION_STATES:
        raise TrainingReferenceError(f"{location} job state must be 'new' or 'in_progress'")
    if job.get("state") != before_state:
        raise TrainingReferenceError(f"{location} job state differs from the completion receipt")
    before_stage = _text(receipt.get("job_stage_before"), "completion receipt.job_stage_before")
    if job.get("stage") != before_stage:
        raise TrainingReferenceError(f"{location} job stage differs from the completion receipt")

    gate = _object(snapshot.get("final_gate"), f"{location}.final_gate")
    if gate.get("passed") is not True or _list(
        gate.get("blocking_reasons"), f"{location}.final_gate.blocking_reasons"
    ):
        raise TrainingReferenceError(f"{location} final gate has not passed")
    try:
        annotations = canonicalize_annotations(snapshot.get("annotations"))
    except (TypeError, ValueError) as error:
        raise TrainingReferenceError(f"{location} annotations are invalid: {error}") from error
    if annotations["tags"] or annotations["tracks"]:
        raise TrainingReferenceError(f"{location} must contain no tags or tracks")
    computed_sha256 = canonical_sha256(annotations)
    if (
        _sha256(
            snapshot.get("canonical_annotations_sha256"),
            f"{location}.canonical_annotations_sha256",
        )
        != computed_sha256
    ):
        raise TrainingReferenceError(f"{location} canonical annotation hash is inconsistent")
    if computed_sha256 != annotation_sha256:
        raise TrainingReferenceError(
            f"{location} annotations differ from the completed snapshot and receipt"
        )
    if sum(len(annotations[key]) for key in ("tags", "shapes", "tracks")) != annotation_count:
        raise TrainingReferenceError(f"{location} annotation count differs from the receipt")

    snapshot_created_at = _aware_datetime(snapshot.get("created_at"), f"{location}.created_at")
    if snapshot_created_at > completed_at:
        raise TrainingReferenceError(f"{location} was captured after the completion receipt")


def _validate_receipt(
    receipt: Mapping[str, Any],
    *,
    receipt_path: Path,
    post_completion_snapshot: Mapping[str, Any],
    task_id: int,
    job_id: int,
    annotation_sha256: str,
    annotation_count: int,
    frame_count: int,
    job: Mapping[str, Any],
) -> None:
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise TrainingReferenceError("completion receipt schema is unsupported")
    evidence = _object(receipt.get("evidence"), "completion receipt.evidence")
    post_snapshot = _object(
        evidence.get("post_snapshot"), "completion receipt.evidence.post_snapshot"
    )
    if _integer(receipt.get("task_id"), "completion receipt.task_id") != task_id:
        raise TrainingReferenceError("completion receipt is bound to another task")
    if _integer(receipt.get("job_id"), "completion receipt.job_id") != job_id:
        raise TrainingReferenceError("completion receipt is bound to another job")
    if receipt.get("dry_run") is not False or receipt.get("mutation_performed") is not True:
        raise TrainingReferenceError("completion receipt does not record an applied completion")
    if receipt.get("job_state_after") != "completed":
        raise TrainingReferenceError("completion receipt job_state_after must be 'completed'")
    if receipt.get("job_stage_after") != job.get("stage"):
        raise TrainingReferenceError("completion receipt job stage differs from the snapshot")
    if receipt.get("job_stage_before") != receipt.get("job_stage_after"):
        raise TrainingReferenceError("completion receipt job stage changed during completion")
    if _integer(receipt.get("annotation_count"), "completion receipt.annotation_count") != (
        annotation_count
    ):
        raise TrainingReferenceError("completion receipt annotation count differs from snapshot")
    for field in (
        "verified_live_canonical_annotations_sha256",
        "verified_post_completion_canonical_annotations_sha256",
    ):
        if _sha256(receipt.get(field), f"completion receipt.{field}") != annotation_sha256:
            raise TrainingReferenceError(
                f"completion receipt {field} differs from snapshot annotations"
            )
    completed_at = _aware_datetime(receipt.get("completed_at"), "completion receipt.completed_at")
    post_completion_created_at = _aware_datetime(
        post_completion_snapshot.get("created_at"), "post-completion snapshot.created_at"
    )
    if post_completion_created_at < completed_at:
        raise TrainingReferenceError(
            "post-completion snapshot was captured before the completion receipt"
        )

    evidence_snapshot_path = _resolve_receipt_evidence_path(
        post_snapshot.get("path"), receipt_path=receipt_path
    )
    evidence_snapshot, evidence_snapshot_sha256 = _read_json(
        evidence_snapshot_path, "completion receipt pre-completion snapshot"
    )
    if evidence_snapshot_sha256 != _sha256(
        post_snapshot.get("sha256"), "completion receipt.evidence.post_snapshot.sha256"
    ):
        raise TrainingReferenceError(
            "completion receipt pre-completion snapshot SHA-256 differs from the named file"
        )
    _validate_pre_completion_snapshot(
        evidence_snapshot,
        task_id=task_id,
        job_id=job_id,
        frame_count=frame_count,
        annotation_sha256=annotation_sha256,
        annotation_count=annotation_count,
        receipt=receipt,
        completed_at=completed_at,
    )
    validation = _object(
        receipt.get("decision_validation"), "completion receipt.decision_validation"
    )
    if validation.get("valid") is not True:
        raise TrainingReferenceError("completion receipt decision validation must be valid")
    if validation.get("scope") != "full_review":
        raise TrainingReferenceError(
            "completion receipt decision validation scope must be 'full_review'"
        )
    if (
        _integer(validation.get("task_id"), "completion receipt.decision_validation.task_id")
        != task_id
    ):
        raise TrainingReferenceError("completion receipt decision validation is for another task")
    for field in ("snapshot_frame_count", "reviewed_frame_count"):
        if (
            _integer(validation.get(field), f"completion receipt.decision_validation.{field}")
            != frame_count
        ):
            raise TrainingReferenceError(
                f"completion receipt decision validation {field} differs from task frame count"
            )
    if (
        _integer(
            validation.get("unresolved_manual_flag_count"),
            "completion receipt.decision_validation.unresolved_manual_flag_count",
            minimum=0,
        )
        != 0
    ):
        raise TrainingReferenceError(
            "completion receipt decision validation has unresolved manual flags"
        )


def _validate_snapshot_labels(
    snapshot: Mapping[str, Any], expected_names: Sequence[str]
) -> dict[int, str]:
    by_id: dict[int, str] = {}
    names: list[str] = []
    for index, raw_label in enumerate(_list(snapshot.get("labels"), "snapshot.labels")):
        label = _object(raw_label, f"snapshot.labels[{index}]")
        identifier = _integer(label.get("id"), f"snapshot.labels[{index}].id")
        name = _text(label.get("name"), f"snapshot.labels[{index}].name")
        if identifier in by_id or name in names:
            raise TrainingReferenceError("snapshot label IDs and names must be unique")
        by_id[identifier] = name
        names.append(name)
    if set(names) != set(expected_names) or len(names) != len(expected_names):
        raise TrainingReferenceError("snapshot labels do not match the fixed eight-class taxonomy")
    return by_id


def _validate_shapes(
    snapshot: Mapping[str, Any],
    *,
    images_by_frame: Mapping[int, ValidatedImage],
    label_id_to_name: Mapping[int, str],
) -> tuple[tuple[dict[str, Any], ...], str]:
    try:
        annotations = canonicalize_annotations(snapshot.get("annotations"))
    except (TypeError, ValueError) as error:
        raise TrainingReferenceError(f"snapshot annotations are invalid: {error}") from error
    if annotations["tags"] or annotations["tracks"]:
        raise TrainingReferenceError("snapshot annotations must contain no tags or tracks")
    computed_sha = canonical_sha256(annotations)
    if (
        _sha256(
            snapshot.get("canonical_annotations_sha256"),
            "snapshot.canonical_annotations_sha256",
        )
        != computed_sha
    ):
        raise TrainingReferenceError("snapshot canonical annotation hash is inconsistent")

    shape_ids: set[tuple[str, str]] = set()
    shapes: list[dict[str, Any]] = []
    for index, raw_shape in enumerate(annotations["shapes"]):
        shape = _object(raw_shape, f"snapshot.annotations.shapes[{index}]")
        if shape.get("type") != "rectangle":
            raise TrainingReferenceError(f"shape {index} is not a rectangle")
        if shape.get("outside", False) is not False:
            raise TrainingReferenceError(f"shape {index} is marked outside")
        if _number(shape.get("rotation", 0), f"shape {index}.rotation") != 0:
            raise TrainingReferenceError(f"shape {index} is rotated and cannot become a COCO bbox")
        frame = _integer(shape.get("frame"), f"shape {index}.frame", minimum=0)
        if frame not in images_by_frame:
            raise TrainingReferenceError(f"shape {index} references unknown frame {frame}")
        label_id = _integer(shape.get("label_id"), f"shape {index}.label_id")
        if label_id not in label_id_to_name:
            raise TrainingReferenceError(f"shape {index} references unknown label {label_id}")
        points = _list(shape.get("points"), f"shape {index}.points")
        if len(points) != 4:
            raise TrainingReferenceError(f"shape {index}.points must contain four coordinates")
        x1, y1, x2, y2 = (
            _number(value, f"shape {index}.points[{point_index}]")
            for point_index, value in enumerate(points)
        )
        image = images_by_frame[frame]
        if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
            raise TrainingReferenceError(f"shape {index} has a non-positive or negative bbox")
        if x2 > image.width or y2 > image.height:
            raise TrainingReferenceError(f"shape {index} escapes image bounds")
        if "id" in shape and shape["id"] is not None:
            value = shape["id"]
            if isinstance(value, bool) or not isinstance(value, (int, str)):
                raise TrainingReferenceError(f"shape {index}.id must be an integer or string")
            identity = (type(value).__name__, str(value))
            if identity in shape_ids:
                raise TrainingReferenceError(f"snapshot contains duplicate shape id {value!r}")
            shape_ids.add(identity)
        shapes.append(shape)
    return tuple(shapes), computed_sha


def _validate_images(
    snapshot: Mapping[str, Any],
    manifest: Mapping[str, Any],
    manifest_path: Path,
    *,
    task_id: int,
    source_frames: Mapping[tuple[str, int], tuple[int | str, str, int]],
) -> tuple[ValidatedImage, ...]:
    samples = _list(manifest.get("samples"), "image manifest.samples")
    sample_size = _integer(manifest.get("sample_size"), "image manifest.sample_size", minimum=1)
    if sample_size != len(samples):
        raise TrainingReferenceError("image manifest sample_size differs from samples")
    raw_images = _list(snapshot.get("images"), "snapshot.images")
    if len(raw_images) != sample_size:
        raise TrainingReferenceError("snapshot and image manifest image counts differ")
    images_by_frame: dict[int, dict[str, Any]] = {}
    for index, raw_image in enumerate(raw_images):
        image = _object(raw_image, f"snapshot.images[{index}]")
        frame = _integer(
            image.get("cvat_frame", image.get("frame")),
            f"snapshot.images[{index}].cvat_frame",
            minimum=0,
        )
        if frame in images_by_frame:
            raise TrainingReferenceError(f"snapshot contains duplicate CVAT frame {frame}")
        images_by_frame[frame] = image
    if set(images_by_frame) != set(range(sample_size)):
        raise TrainingReferenceError("snapshot CVAT frames must be consecutive from zero")

    result: list[ValidatedImage] = []
    for position, raw_sample in enumerate(samples):
        sample = _object(raw_sample, f"image manifest.samples[{position}]")
        sample_index = _integer(
            sample.get("sample_index"),
            f"image manifest.samples[{position}].sample_index",
            minimum=1,
        )
        if sample_index != position + 1:
            raise TrainingReferenceError("image manifest sample_index must be consecutive in order")
        frame = position
        image = images_by_frame[frame]
        file_name = _text(sample.get("file_name"), f"image manifest.samples[{position}].file_name")
        scene_id = _text(sample.get("scene_id"), f"image manifest.samples[{position}].scene_id")
        source_frame = _integer(
            sample.get("source_frame"),
            f"image manifest.samples[{position}].source_frame",
            minimum=0,
        )
        for field, expected in (
            ("sample_index", sample_index),
            ("file_name", file_name),
            ("scene_id", scene_id),
            ("source_frame", source_frame),
        ):
            if image.get(field) != expected:
                raise TrainingReferenceError(
                    f"snapshot image frame {frame} {field} differs from the image manifest"
                )
        if image.get("relative_path") != f"images/{file_name}":
            raise TrainingReferenceError(
                f"snapshot image frame {frame} relative_path is inconsistent"
            )
        mapping = source_frames.get((scene_id, source_frame))
        if mapping is None:
            raise TrainingReferenceError(
                f"source map has no unique match for {scene_id!r}/{source_frame}"
            )
        asset_identifier, leakage_group_id, normalized_asset_frame = mapping
        image_path = _safe_image_path(manifest_path, file_name)
        if not image_path.is_file():
            raise TrainingReferenceError(f"manifest image is missing: {image_path}")
        before = image_path.stat()
        digest = file_sha256(image_path)
        try:
            with Image.open(image_path) as opened:
                width, height = opened.size
                opened.verify()
        except Exception as error:
            raise TrainingReferenceError(f"manifest image is invalid: {image_path}") from error
        after = image_path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise TrainingReferenceError(f"manifest image changed while validating: {image_path}")
        if width <= 0 or height <= 0:
            raise TrainingReferenceError(f"manifest image has invalid dimensions: {image_path}")
        expected_sha = _sha256(image.get("sha256"), f"snapshot.images[{frame}].sha256")
        if digest != expected_sha:
            raise TrainingReferenceError(f"snapshot image frame {frame} SHA-256 differs from disk")
        if _integer(image.get("size_bytes"), f"snapshot.images[{frame}].size_bytes") != (
            before.st_size
        ):
            raise TrainingReferenceError(
                f"snapshot image frame {frame} byte size differs from disk"
            )
        if (
            _integer(image.get("width"), f"snapshot.images[{frame}].width") != width
            or _integer(image.get("height"), f"snapshot.images[{frame}].height") != height
        ):
            raise TrainingReferenceError(
                f"snapshot image frame {frame} dimensions differ from disk"
            )
        for optional_field, actual in (("sha256", digest), ("width", width), ("height", height)):
            if optional_field in sample and sample[optional_field] != actual:
                raise TrainingReferenceError(
                    f"image manifest sample {sample_index} {optional_field} differs from disk"
                )
        result.append(
            ValidatedImage(
                task_id=task_id,
                cvat_frame=frame,
                sample_index=sample_index,
                scene_id=scene_id,
                source_frame=source_frame,
                file_name=file_name,
                source_path=image_path,
                sha256=digest,
                size_bytes=before.st_size,
                width=width,
                height=height,
                asset_id=asset_identifier,
                leakage_group_id=leakage_group_id,
                normalized_asset_frame=normalized_asset_frame,
            )
        )
    return tuple(result)


def _validate_task(
    paths: InputPaths,
    *,
    expected_names: Sequence[str],
    source_frames: Mapping[tuple[str, int], tuple[int | str, str, int]],
) -> ValidatedTask:
    snapshot_path = paths.snapshot.resolve()
    manifest_path = paths.image_manifest.resolve()
    receipt_path = paths.completion_receipt.resolve()
    snapshot, snapshot_digest = _read_json(snapshot_path, "post-completion snapshot")
    manifest, manifest_digest = _read_json(manifest_path, "image manifest")
    receipt, receipt_digest = _read_json(receipt_path, "completion receipt")
    if snapshot.get("snapshot_schema") != SNAPSHOT_SCHEMA:
        raise TrainingReferenceError("post-completion snapshot schema is unsupported")
    task = _object(snapshot.get("task"), "snapshot.task")
    task_id = _integer(task.get("id"), "snapshot.task.id")
    frame_count = _integer(task.get("size"), "snapshot.task.size", minimum=1)
    snapshot_manifest = _object(snapshot.get("manifest"), "snapshot.manifest")
    if _sha256(snapshot_manifest.get("sha256"), "snapshot.manifest.sha256") != manifest_digest:
        raise TrainingReferenceError("snapshot is bound to a different image manifest hash")
    if _integer(snapshot_manifest.get("cvat_task_id"), "snapshot.manifest.cvat_task_id") != task_id:
        raise TrainingReferenceError("snapshot manifest is bound to another task")
    if (
        _integer(snapshot_manifest.get("sample_size"), "snapshot.manifest.sample_size")
        != frame_count
    ):
        raise TrainingReferenceError("snapshot manifest sample_size differs from task size")
    cvat = _object(manifest.get("cvat"), "image manifest.cvat")
    if _integer(cvat.get("task_id"), "image manifest.cvat.task_id") != task_id:
        raise TrainingReferenceError("image manifest is bound to another task")
    for field in ("session_id", "purpose", "sampling_revision"):
        if snapshot_manifest.get(field) != manifest.get(field):
            raise TrainingReferenceError(f"snapshot manifest metadata {field} is inconsistent")
    gate = _object(snapshot.get("final_gate"), "snapshot.final_gate")
    if gate.get("passed") is not True or _list(
        gate.get("blocking_reasons"), "snapshot.final_gate.blocking_reasons"
    ):
        raise TrainingReferenceError("snapshot final gate has not passed")
    job_id, job = _validate_job(snapshot, task_id=task_id, frame_count=frame_count)
    label_id_to_name = _validate_snapshot_labels(snapshot, expected_names)
    images = _validate_images(
        snapshot,
        manifest,
        manifest_path,
        task_id=task_id,
        source_frames=source_frames,
    )
    shapes, annotation_digest = _validate_shapes(
        snapshot,
        images_by_frame={image.cvat_frame: image for image in images},
        label_id_to_name=label_id_to_name,
    )
    counts = _object(snapshot.get("counts"), "snapshot.counts")
    expected_counts = {
        "images": len(images),
        "tags": 0,
        "shapes": len(shapes),
        "tracks": 0,
    }
    for key, expected in expected_counts.items():
        if _integer(counts.get(key), f"snapshot.counts.{key}", minimum=0) != expected:
            raise TrainingReferenceError(f"snapshot count {key} is inconsistent")
    class_counts = Counter(
        label_id_to_name[_integer(shape["label_id"], "shape.label_id")] for shape in shapes
    )
    reported_by_label = _object(
        counts.get("annotations_by_label"), "snapshot.counts.annotations_by_label"
    )
    if set(reported_by_label) != set(expected_names) or any(
        _integer(reported_by_label[name], f"snapshot.counts.annotations_by_label.{name}")
        != class_counts[name]
        for name in expected_names
    ):
        raise TrainingReferenceError("snapshot annotation counts by label are inconsistent")
    _validate_receipt(
        receipt,
        receipt_path=receipt_path,
        post_completion_snapshot=snapshot,
        task_id=task_id,
        job_id=job_id,
        annotation_sha256=annotation_digest,
        annotation_count=len(shapes),
        frame_count=frame_count,
        job=job,
    )
    return ValidatedTask(
        task_id=task_id,
        job_id=job_id,
        snapshot_path=snapshot_path,
        snapshot_sha256=snapshot_digest,
        manifest_path=manifest_path,
        manifest_sha256=manifest_digest,
        receipt_path=receipt_path,
        receipt_sha256=receipt_digest,
        annotations_sha256=annotation_digest,
        label_id_to_name=label_id_to_name,
        images=images,
        shapes=shapes,
    )


def _validate_global_uniqueness(tasks: Sequence[ValidatedTask]) -> None:
    if len({task.task_id for task in tasks}) != len(tasks):
        raise TrainingReferenceError("input task IDs must be unique")
    file_names: set[str] = set()
    image_hashes: set[str] = set()
    scene_frames: set[tuple[str, int]] = set()
    normalized_asset_frames: set[tuple[str, int]] = set()
    for task in tasks:
        for image in task.images:
            if image.file_name in file_names:
                raise TrainingReferenceError(
                    f"input images duplicate file_name {image.file_name!r}"
                )
            if image.sha256 in image_hashes:
                raise TrainingReferenceError(f"input images duplicate SHA-256 {image.sha256}")
            scene_frame = (image.scene_id, image.source_frame)
            if scene_frame in scene_frames:
                raise TrainingReferenceError(f"input images duplicate source frame {scene_frame!r}")
            normalized_identity = (
                image.leakage_group_id,
                image.normalized_asset_frame,
            )
            if normalized_identity in normalized_asset_frames:
                raise TrainingReferenceError(
                    "input images duplicate a normalized source asset frame: "
                    f"{image.asset_id!r}/{image.normalized_asset_frame}"
                )
            file_names.add(image.file_name)
            image_hashes.add(image.sha256)
            scene_frames.add(scene_frame)
            normalized_asset_frames.add(normalized_identity)


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True).encode(
            "utf-8"
        )
        + b"\n"
    )


def _evidence_locator(path: Path, digest: str, *, workspace_root: Path) -> dict[str, str]:
    """Return a portable evidence locator without embedding a machine-specific root."""

    resolved = path.resolve()
    try:
        relative = resolved.relative_to(workspace_root)
    except ValueError:
        return {
            "path": f"external/{digest[:16]}/{resolved.name}",
            "path_kind": "content_addressed_external",
        }
    return {"path": relative.as_posix(), "path_kind": "workspace_relative"}


def _reject_absolute_paths(value: Any, location: str = "manifest") -> None:
    """Ensure copied audit metadata cannot reintroduce a personal filesystem path."""

    if isinstance(value, dict):
        for key, item in value.items():
            _reject_absolute_paths(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_absolute_paths(item, f"{location}[{index}]")
    elif isinstance(value, str) and (
        Path(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
        or value.lower().startswith("file://")
    ):
        raise TrainingReferenceError(f"{location} contains an absolute local path")


def _write_bytes(path: Path, encoded: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _atomic_publish_directory_no_replace(staging: Path, output: Path) -> None:
    """Atomically rename a directory while refusing even an empty existing target."""

    source_bytes = os.fsencode(staging)
    target_bytes = os.fsencode(output)
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        rename_exclusive = libc.renamex_np
        rename_exclusive.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename_exclusive.restype = ctypes.c_int
        result = rename_exclusive(source_bytes, target_bytes, 0x00000004)  # RENAME_EXCL
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        rename_no_replace = libc.renameat2
        rename_no_replace.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename_no_replace.restype = ctypes.c_int
        result = rename_no_replace(
            -100,
            source_bytes,
            -100,
            target_bytes,
            0x00000001,  # AT_FDCWD, RENAME_NOREPLACE
        )
    else:
        raise TrainingReferenceError(
            "atomic no-replace directory publication is unsupported on this platform"
        )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise TrainingReferenceError(f"output directory already exists: {output}")
    raise OSError(error_number, os.strerror(error_number), str(output))


def _build_payloads(
    tasks: Sequence[ValidatedTask],
    *,
    source_assets: Mapping[tuple[str, str], Mapping[str, Any]],
    labels_path: Path,
    labels_sha256: str,
    source_map_path: Path,
    source_map_sha256: str,
    workspace_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[tuple[ValidatedImage, str]]]:
    ordered_tasks = sorted(tasks, key=lambda task: task.task_id)
    categories = [
        {"id": index, "name": name, "supercategory": "road_object"}
        for index, name in enumerate(REQUIRED_LABELS, start=1)
    ]
    category_id = {name: index for index, name in enumerate(REQUIRED_LABELS, start=1)}
    coco_images: list[dict[str, Any]] = []
    image_id_by_task_frame: dict[tuple[int, int], int] = {}
    copies: list[tuple[ValidatedImage, str]] = []
    image_id = 1
    for task in ordered_tasks:
        for image in sorted(task.images, key=lambda item: item.cvat_frame):
            relative_path = f"images/task-{task.task_id}/{image.file_name}"
            image_id_by_task_frame[(task.task_id, image.cvat_frame)] = image_id
            coco_images.append(
                {
                    "id": image_id,
                    "file_name": relative_path,
                    "width": image.width,
                    "height": image.height,
                    "sha256": image.sha256,
                    "task_id": task.task_id,
                    "cvat_frame": image.cvat_frame,
                    "sample_index": image.sample_index,
                    "scene_id": image.scene_id,
                    "source_frame": image.source_frame,
                    "source_asset_id": image.asset_id,
                    "source_leakage_group_id": image.leakage_group_id,
                    "source_normalized_asset_frame": image.normalized_asset_frame,
                }
            )
            copies.append((image, relative_path))
            image_id += 1

    coco_annotations: list[dict[str, Any]] = []
    annotation_id = 1
    for task in ordered_tasks:
        ordered_shapes = sorted(
            task.shapes,
            key=lambda shape: (
                _integer(shape["frame"], "shape.frame"),
                category_id[task.label_id_to_name[_integer(shape["label_id"], "shape.label_id")]],
                tuple(float(value) for value in shape["points"]),
                json.dumps(shape, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            ),
        )
        for shape in ordered_shapes:
            frame = _integer(shape["frame"], "shape.frame")
            x1, y1, x2, y2 = (float(value) for value in shape["points"])
            width, height = x2 - x1, y2 - y1
            label_name = task.label_id_to_name[_integer(shape["label_id"], "shape.label_id")]
            coco_annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id_by_task_frame[(task.task_id, frame)],
                    "category_id": category_id[label_name],
                    "bbox": [x1, y1, width, height],
                    "area": width * height,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    coco = {
        "info": {
            "description": "RoadLabelOps evidence-bound training reference",
            "schema": OUTPUT_SCHEMA,
            "labels_sha256": labels_sha256,
            "source_map_sha256": source_map_sha256,
        },
        "licenses": [],
        "categories": categories,
        "images": coco_images,
        "annotations": coco_annotations,
    }

    annotations_by_name = Counter(
        REQUIRED_LABELS[annotation["category_id"] - 1] for annotation in coco_annotations
    )
    annotations_by_image = Counter(annotation["image_id"] for annotation in coco_annotations)
    task_stats: list[dict[str, Any]] = []
    scene_images: Counter[str] = Counter()
    scene_annotations: Counter[str] = Counter()
    asset_images: Counter[tuple[str, str]] = Counter()
    asset_annotations: Counter[tuple[str, str]] = Counter()
    asset_values: dict[tuple[str, str], int | str] = {}
    asset_leakage_groups: dict[tuple[str, str], str] = {}
    asset_scenes: dict[tuple[str, str], set[str]] = defaultdict(set)
    for task in ordered_tasks:
        task_image_ids = {
            image_id_by_task_frame[(task.task_id, image.cvat_frame)] for image in task.images
        }
        task_stats.append(
            {
                "task_id": task.task_id,
                "job_id": task.job_id,
                "image_count": len(task.images),
                "zero_annotation_image_count": sum(
                    annotations_by_image[identifier] == 0 for identifier in task_image_ids
                ),
                "annotation_count": sum(
                    annotations_by_image[identifier] for identifier in task_image_ids
                ),
            }
        )
        for image in task.images:
            identifier = image_id_by_task_frame[(task.task_id, image.cvat_frame)]
            annotation_count = annotations_by_image[identifier]
            scene_images[image.scene_id] += 1
            scene_annotations[image.scene_id] += annotation_count
            asset_identity = _asset_identity(image.asset_id)
            asset_values[asset_identity] = image.asset_id
            asset_leakage_groups[asset_identity] = image.leakage_group_id
            asset_images[asset_identity] += 1
            asset_annotations[asset_identity] += annotation_count
            asset_scenes[asset_identity].add(image.scene_id)

    manifest = {
        "schema": OUTPUT_SCHEMA,
        "gate": {
            "passed": True,
            "blocking_reasons": [],
            "checks": {
                "completed_jobs_with_zero_issues": True,
                "snapshot_and_annotation_hashes_verified": True,
                "completion_receipts_exactly_bound": True,
                "manifest_images_verified": True,
                "rectangle_only_annotations": True,
                "fixed_eight_class_taxonomy": True,
                "bounded_finite_positive_bboxes": True,
                "cross_task_image_identities_unique": True,
                "source_asset_hashes_unique": True,
                "stable_leakage_groups_bound": True,
                "source_frame_map_complete_and_unique": True,
                "all_images_including_zero_annotation_images_copied": True,
            },
        },
        "evidence": {
            "labels": {
                **_evidence_locator(labels_path, labels_sha256, workspace_root=workspace_root),
                "sha256": labels_sha256,
            },
            "source_map": {
                **_evidence_locator(
                    source_map_path, source_map_sha256, workspace_root=workspace_root
                ),
                "sha256": source_map_sha256,
                "assets": [
                    copy.deepcopy(source_assets[identity]) for identity in sorted(source_assets)
                ],
            },
            "task_inputs": [
                {
                    "task_id": task.task_id,
                    "job_id": task.job_id,
                    "snapshot": {
                        **_evidence_locator(
                            task.snapshot_path,
                            task.snapshot_sha256,
                            workspace_root=workspace_root,
                        ),
                        "sha256": task.snapshot_sha256,
                        "canonical_annotations_sha256": task.annotations_sha256,
                    },
                    "image_manifest": {
                        **_evidence_locator(
                            task.manifest_path,
                            task.manifest_sha256,
                            workspace_root=workspace_root,
                        ),
                        "sha256": task.manifest_sha256,
                    },
                    "completion_receipt": {
                        **_evidence_locator(
                            task.receipt_path,
                            task.receipt_sha256,
                            workspace_root=workspace_root,
                        ),
                        "sha256": task.receipt_sha256,
                    },
                }
                for task in ordered_tasks
            ],
        },
        "counts": {
            "tasks": len(ordered_tasks),
            "images": len(coco_images),
            "zero_annotation_images": sum(
                annotations_by_image[image["id"]] == 0 for image in coco_images
            ),
            "annotations": len(coco_annotations),
            "categories": len(categories),
            "annotations_by_category": {
                name: annotations_by_name[name] for name in REQUIRED_LABELS
            },
        },
        "source_statistics": {
            "tasks": task_stats,
            "scenes": [
                {
                    "scene_id": scene_id,
                    "image_count": scene_images[scene_id],
                    "annotation_count": scene_annotations[scene_id],
                }
                for scene_id in sorted(scene_images)
            ],
            "assets": [
                {
                    "asset_id": asset_values[identity],
                    "leakage_group_id": asset_leakage_groups[identity],
                    "image_count": asset_images[identity],
                    "annotation_count": asset_annotations[identity],
                    "scene_ids": sorted(asset_scenes[identity]),
                }
                for identity in sorted(asset_images)
            ],
        },
    }
    return coco, manifest, copies


def build_training_reference(
    inputs: Sequence[InputPaths],
    *,
    source_map_path: Path,
    labels_path: Path,
    output: Path,
) -> dict[str, Any]:
    """Validate all evidence and atomically publish a new COCO reference directory."""

    raw_output = Path(output)
    output = raw_output.parent.resolve() / raw_output.name
    if os.path.lexists(output):
        raise TrainingReferenceError(f"output directory already exists: {output}")
    if not inputs:
        raise TrainingReferenceError("at least one --input is required")
    expected_names, labels_digest = _read_labels(labels_path)
    source_frames, source_assets, source_map_digest = _read_source_map(source_map_path)
    tasks = [
        _validate_task(
            paths,
            expected_names=expected_names,
            source_frames=source_frames,
        )
        for paths in inputs
    ]
    _validate_global_uniqueness(tasks)
    workspace_root = Path.cwd().resolve()
    coco, manifest, copies = _build_payloads(
        tasks,
        source_assets=source_assets,
        labels_path=labels_path,
        labels_sha256=labels_digest,
        source_map_path=source_map_path,
        source_map_sha256=source_map_digest,
        workspace_root=workspace_root,
    )
    _reject_absolute_paths(manifest)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=str(output.parent.resolve()))
    )
    published = False
    try:
        coco_path = staging / "annotations.coco.json"
        _write_bytes(coco_path, _json_bytes(coco))
        managed_files = [
            {
                "path": "annotations.coco.json",
                "sha256": file_sha256(coco_path),
                "size_bytes": coco_path.stat().st_size,
            }
        ]
        for image, relative_path in copies:
            target = staging / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            before = image.source_path.stat()
            with image.source_path.open("rb") as source, target.open("xb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
                destination.flush()
                os.fsync(destination.fileno())
            after = image.source_path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise TrainingReferenceError(
                    f"source image changed while copying: {image.source_path}"
                )
            copied_sha = file_sha256(target)
            if copied_sha != image.sha256:
                raise TrainingReferenceError(f"copied image hash mismatch: {image.source_path}")
            managed_files.append(
                {
                    "path": relative_path,
                    "sha256": copied_sha,
                    "size_bytes": target.stat().st_size,
                }
            )
        manifest["files"] = sorted(managed_files, key=lambda item: item["path"])
        manifest_path = staging / "manifest.json"
        _write_bytes(manifest_path, _json_bytes(manifest))
        directory_fd = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        _atomic_publish_directory_no_replace(staging, output)
        published = True
        parent_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)

    return {
        "output": str(output),
        "annotations_coco": str(output / "annotations.coco.json"),
        "manifest": str(output / "manifest.json"),
        "counts": manifest["counts"],
        "gate": manifest["gate"],
    }


def _parse_receipt_bindings(
    values: Sequence[Sequence[str]], parser: argparse.ArgumentParser
) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for raw_task_id, raw_path in values:
        try:
            task_id = int(raw_task_id)
        except ValueError:
            parser.error(f"completion receipt TASK_ID must be an integer: {raw_task_id!r}")
        if task_id in result:
            parser.error(f"completion receipt task {task_id} is bound more than once")
        result[task_id] = Path(raw_path)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Repeat --input for each task and bind receipts explicitly by task ID:\n"
            "  --input SNAPSHOT MANIFEST --completion-receipt TASK_ID RECEIPT\n"
            "The output path must not already exist."
        ),
    )
    parser.add_argument("--input", nargs=2, action="append", metavar=("SNAPSHOT", "MANIFEST"))
    parser.add_argument(
        "--completion-receipt",
        nargs=2,
        action="append",
        default=[],
        metavar=("TASK_ID", "RECEIPT"),
    )
    parser.add_argument("--source-map", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.input:
        parser.error("at least one --input SNAPSHOT MANIFEST pair is required")
    receipt_bindings = _parse_receipt_bindings(args.completion_receipt, parser)
    input_pairs: list[tuple[Path, Path, int]] = []
    seen_task_ids: set[int] = set()
    for raw_snapshot, raw_manifest in args.input:
        snapshot_path = Path(raw_snapshot)
        try:
            snapshot, _digest = _read_json(snapshot_path, "post-completion snapshot")
            task_id = _integer(_object(snapshot.get("task"), "snapshot.task").get("id"), "task.id")
        except TrainingReferenceError as error:
            parser.error(str(error))
        if task_id in seen_task_ids:
            parser.error(f"input task {task_id} is supplied more than once")
        seen_task_ids.add(task_id)
        input_pairs.append((snapshot_path, Path(raw_manifest), task_id))
    if set(receipt_bindings) != seen_task_ids:
        parser.error(
            "completion receipts must bind every and only input task ID; "
            f"inputs={sorted(seen_task_ids)}, receipts={sorted(receipt_bindings)}"
        )
    inputs = [
        InputPaths(snapshot, manifest, receipt_bindings[task_id])
        for snapshot, manifest, task_id in input_pairs
    ]
    try:
        summary = build_training_reference(
            inputs,
            source_map_path=args.source_map,
            labels_path=args.labels,
            output=args.output,
        )
    except TrainingReferenceError as error:
        parser.error(str(error))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
