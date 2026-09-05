"""Freeze one reviewed COCO reference for offline final-holdout evaluation.

The input must be an immutable ``roadlabelops.training-coco-reference`` v2
directory produced by :mod:`scripts.build_training_reference`.  This command
does not contact CVAT.  It copies the COCO annotations, every managed image,
and hash-bound full-review completion evidence into a new directory, then
publishes that directory atomically without replacing an existing path.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import math
import os
import shutil
import stat
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from roadlabelops.holdout_policy import (
    FinalHoldoutConfigError,
    FinalHoldoutIdentity,
    resolve_final_holdout_identity,
)

INPUT_SCHEMA = {"name": "roadlabelops.training-coco-reference", "version": 2}
OUTPUT_SCHEMA = {"name": "roadlabelops.frozen-holdout-reference", "version": 1}
SNAPSHOT_SCHEMA = {"name": "roadlabelops.cvat-task-snapshot", "version": 1}
RECEIPT_SCHEMA = {"name": "roadlabelops.cvat-job-completion-receipt", "version": 1}
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


class FrozenHoldoutReferenceError(ValueError):
    """Raised when a frozen holdout reference cannot be published safely."""


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True).encode(
            "utf-8"
        )
        + b"\n"
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _object(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FrozenHoldoutReferenceError(f"{location} must be an object")
    return value


def _list(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise FrozenHoldoutReferenceError(f"{location} must be a list")
    return value


def _text(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FrozenHoldoutReferenceError(f"{location} must be a non-empty string")
    return value


def _asset_id(value: Any, location: str) -> tuple[str, str]:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise FrozenHoldoutReferenceError(f"{location} must be a string or integer")
    if isinstance(value, str) and not value:
        raise FrozenHoldoutReferenceError(f"{location} must not be empty")
    return type(value).__name__, str(value)


def _integer(value: Any, location: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FrozenHoldoutReferenceError(f"{location} must be an integer")
    if minimum is not None and value < minimum:
        raise FrozenHoldoutReferenceError(f"{location} must be at least {minimum}")
    return value


def _digest(value: Any, location: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise FrozenHoldoutReferenceError(f"{location} must be a lowercase SHA-256")
    return value


def _safe_relative(value: Any, location: str) -> Path:
    if not isinstance(value, str) or not value:
        raise FrozenHoldoutReferenceError(f"{location} must be a non-empty relative path")
    posix = PurePosixPath(value)
    if (
        posix.is_absolute()
        or PureWindowsPath(value).is_absolute()
        or "\\" in value
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        raise FrozenHoldoutReferenceError(f"{location} is unsafe")
    return Path(*posix.parts)


def _leaf_directory(path: Path, location: str) -> Path:
    raw = Path(path)
    resolved = raw.parent.resolve() / raw.name
    try:
        mode = os.lstat(resolved).st_mode
    except OSError as error:
        raise FrozenHoldoutReferenceError(f"{location} is unavailable: {error}") from error
    if not stat.S_ISDIR(mode) or stat.S_ISLNK(mode):
        raise FrozenHoldoutReferenceError(f"{location} must be a non-symlink directory")
    return resolved


def _stable_file_bytes(path: Path, location: str) -> tuple[bytes, str, int]:
    """Read one input exactly once through a stable, non-following file descriptor."""

    if not hasattr(os, "O_NOFOLLOW"):
        raise FrozenHoldoutReferenceError("O_NOFOLLOW is required for secure input reads")
    resolved = Path(path).parent.resolve() / Path(path).name
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(resolved, flags)
    except OSError as error:
        raise FrozenHoldoutReferenceError(f"{location} is unavailable: {error}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise FrozenHoldoutReferenceError(f"{location} must be a non-symlink regular file")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    except OSError as error:
        raise FrozenHoldoutReferenceError(
            f"{location} could not be read safely: {error}"
        ) from error
    finally:
        os.close(descriptor)

    def identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
        return (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mode,
            metadata.st_nlink,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )

    payload = b"".join(chunks)
    if identity(before) != identity(after) or len(payload) != after.st_size:
        raise FrozenHoldoutReferenceError(f"{location} changed while being read")
    try:
        current = os.stat(resolved, follow_symlinks=False)
    except OSError as error:
        raise FrozenHoldoutReferenceError(
            f"{location} changed while being read: {error}"
        ) from error
    if (
        not stat.S_ISREG(current.st_mode)
        or current.st_dev != after.st_dev
        or current.st_ino != after.st_ino
    ):
        raise FrozenHoldoutReferenceError(f"{location} changed while being read")
    return payload, _sha256_bytes(payload), len(payload)


def _json_from_bytes(payload: bytes, location: str) -> dict[str, Any]:
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise FrozenHoldoutReferenceError(f"{location} is not valid UTF-8 JSON: {error}") from error
    return _object(decoded, location)


def _read_json(path: Path, location: str) -> tuple[dict[str, Any], bytes, str, int]:
    payload, digest, size = _stable_file_bytes(Path(path), location)
    return _json_from_bytes(payload, location), payload, digest, size


def _managed_source(root: Path, relative: Path, location: str) -> Path:
    candidate = root / relative
    resolved = candidate.parent.resolve() / candidate.name
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise FrozenHoldoutReferenceError(f"{location} escapes the reference directory") from error
    return resolved


def _bound(path: str, sha256: str, size_bytes: int) -> dict[str, Any]:
    return {"path": path, "sha256": sha256, "size_bytes": size_bytes}


def _validate_source_reference(
    reference_dir: Path,
    identity: FinalHoldoutIdentity,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = _leaf_directory(reference_dir, "reference directory")
    manifest, _manifest_payload, manifest_sha, _manifest_size = _read_json(
        root / "manifest.json", "reference manifest"
    )
    if manifest.get("schema") != INPUT_SCHEMA:
        raise FrozenHoldoutReferenceError("reference manifest is not schema v2 training COCO")
    gate = _object(manifest.get("gate"), "reference manifest.gate")
    gate_checks = _object(gate.get("checks"), "reference manifest.gate.checks")
    if (
        gate.get("passed") is not True
        or _list(gate.get("blocking_reasons"), "reference manifest.gate.blocking_reasons")
        or not gate_checks
        or any(value is not True for value in gate_checks.values())
    ):
        raise FrozenHoldoutReferenceError("reference manifest gate did not pass")
    counts = _object(manifest.get("counts"), "reference manifest.counts")
    if _integer(counts.get("tasks"), "reference manifest.counts.tasks") != 1:
        raise FrozenHoldoutReferenceError("holdout reference must contain exactly one task")
    statistics = _object(manifest.get("source_statistics"), "reference source_statistics")
    task_stats = _list(statistics.get("tasks"), "reference source_statistics.tasks")
    if len(task_stats) != 1:
        raise FrozenHoldoutReferenceError("reference source statistics must contain one task")
    task = _object(task_stats[0], "reference source_statistics.tasks[0]")
    frame_count = _integer(task.get("image_count"), "reference image_count", minimum=1)
    if (
        _integer(task.get("task_id"), "reference task_id") != identity.task_id
        or _integer(task.get("job_id"), "reference job_id") != identity.job_id
    ):
        raise FrozenHoldoutReferenceError(
            "reference does not match the configured final-holdout identity"
        )

    records: list[dict[str, Any]] = []
    managed_by_path: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    for index, raw in enumerate(_list(manifest.get("files"), "reference manifest.files")):
        record = _object(raw, f"reference manifest.files[{index}]")
        if set(record) != {"path", "sha256", "size_bytes"}:
            raise FrozenHoldoutReferenceError("reference manifest contains malformed file records")
        relative = _safe_relative(record.get("path"), f"reference manifest.files[{index}].path")
        key = relative.as_posix()
        if key == "manifest.json" or relative.parts[0] == "evidence":
            raise FrozenHoldoutReferenceError(
                f"reference managed path uses a reserved output namespace: {key}"
            )
        if key in seen:
            raise FrozenHoldoutReferenceError("reference manifest contains duplicate managed paths")
        seen.add(key)
        expected_sha = _digest(record.get("sha256"), f"reference file {key} SHA")
        expected_size = _integer(record.get("size_bytes"), f"reference file {key} size", minimum=1)
        source = _managed_source(root, relative, f"reference file {key}")
        payload, actual_sha, actual_size = _stable_file_bytes(source, f"reference file {key}")
        if actual_size != expected_size or actual_sha != expected_sha:
            raise FrozenHoldoutReferenceError(f"reference managed file changed: {key}")
        validated_record = {
            "path": key,
            "sha256": expected_sha,
            "size_bytes": expected_size,
            "payload": payload,
        }
        records.append(validated_record)
        managed_by_path[key] = validated_record
    if "annotations.coco.json" not in seen:
        raise FrozenHoldoutReferenceError("reference does not manage annotations.coco.json")
    coco = _json_from_bytes(managed_by_path["annotations.coco.json"]["payload"], "COCO annotations")
    categories = _list(coco.get("categories"), "COCO categories")
    category_by_id: dict[int, str] = {}
    for index, raw in enumerate(categories):
        category = _object(raw, f"COCO categories[{index}]")
        identifier = _integer(category.get("id"), f"COCO categories[{index}].id")
        if identifier in category_by_id:
            raise FrozenHoldoutReferenceError("COCO category IDs are not unique")
        category_by_id[identifier] = _text(category.get("name"), f"COCO categories[{index}].name")
    expected_categories = {index: name for index, name in enumerate(REQUIRED_LABELS, start=1)}
    if category_by_id != expected_categories:
        raise FrozenHoldoutReferenceError("COCO does not use the canonical eight-class ID mapping")
    images = _list(coco.get("images"), "COCO images")
    annotations = _list(coco.get("annotations"), "COCO annotations")
    if len(images) != frame_count:
        raise FrozenHoldoutReferenceError(
            "COCO must contain the complete configured final-holdout frame universe"
        )
    images_by_id: dict[int, dict[str, Any]] = {}
    image_files: set[str] = set()
    scene_frame_keys: set[tuple[str, int]] = set()
    source_frame_keys: set[tuple[str, int]] = set()
    for index, raw in enumerate(images):
        image = _object(raw, f"COCO images[{index}]")
        identifier = _integer(image.get("id"), f"COCO images[{index}].id", minimum=1)
        if identifier in images_by_id:
            raise FrozenHoldoutReferenceError("COCO image IDs are not unique")
        width = _integer(image.get("width"), f"COCO images[{index}].width", minimum=1)
        height = _integer(image.get("height"), f"COCO images[{index}].height", minimum=1)
        if _integer(image.get("task_id"), f"COCO images[{index}].task_id") != identity.task_id:
            raise FrozenHoldoutReferenceError("COCO image belongs to another task")
        file_name = _safe_relative(image.get("file_name"), f"COCO images[{index}].file_name")
        key = file_name.as_posix()
        if key in image_files:
            raise FrozenHoldoutReferenceError("COCO image file names are not unique")
        image_files.add(key)
        if key not in seen:
            raise FrozenHoldoutReferenceError(f"COCO image is unmanaged: {key}")
        image_sha = _digest(image.get("sha256"), f"COCO images[{index}].sha256")
        if managed_by_path[key]["sha256"] != image_sha:
            raise FrozenHoldoutReferenceError(f"COCO image hash differs from managed file: {key}")
        scene_id = _text(image.get("scene_id"), f"COCO images[{index}].scene_id")
        source_frame = _integer(
            image.get("source_frame"), f"COCO images[{index}].source_frame", minimum=0
        )
        scene_frame_key = (scene_id, source_frame)
        if scene_frame_key in scene_frame_keys:
            raise FrozenHoldoutReferenceError("COCO scene/source-frame identities are not unique")
        scene_frame_keys.add(scene_frame_key)
        asset_value = image.get("source_asset_id")
        asset_identity = _asset_id(asset_value, f"COCO images[{index}].source_asset_id")
        leakage = _text(
            image.get("source_leakage_group_id"),
            f"COCO images[{index}].source_leakage_group_id",
        )
        if not leakage.startswith("sha256:"):
            raise FrozenHoldoutReferenceError("COCO source leakage identity must use SHA-256")
        source_sha = _digest(leakage.removeprefix("sha256:"), f"COCO images[{index}] source SHA")
        normalized_frame = _integer(
            image.get("source_normalized_asset_frame"),
            f"COCO images[{index}].source_normalized_asset_frame",
            minimum=0,
        )
        source_frame_key = (source_sha, normalized_frame)
        if source_frame_key in source_frame_keys:
            raise FrozenHoldoutReferenceError(
                "COCO aliases one normalized source frame to multiple images"
            )
        source_frame_keys.add(source_frame_key)
        images_by_id[identifier] = {
            "width": width,
            "height": height,
            "scene_id": scene_id,
            "asset_id": asset_value,
            "asset_identity": asset_identity,
            "leakage_group_id": leakage,
        }
    if _integer(counts.get("images"), "reference manifest.counts.images") != len(images):
        raise FrozenHoldoutReferenceError("reference image count is inconsistent")
    if _integer(counts.get("annotations"), "reference manifest.counts.annotations") != len(
        annotations
    ):
        raise FrozenHoldoutReferenceError("reference annotation count is inconsistent")
    annotation_ids: set[int] = set()
    annotations_by_image: Counter[int] = Counter()
    annotations_by_category: Counter[str] = Counter()
    for index, raw in enumerate(annotations):
        annotation = _object(raw, f"COCO annotations[{index}]")
        annotation_id = _integer(annotation.get("id"), f"COCO annotations[{index}].id", minimum=1)
        if annotation_id in annotation_ids:
            raise FrozenHoldoutReferenceError("COCO annotation IDs are not unique")
        annotation_ids.add(annotation_id)
        image_id = _integer(annotation.get("image_id"), "COCO annotation.image_id")
        if image_id not in images_by_id:
            raise FrozenHoldoutReferenceError("COCO annotation references an unknown image")
        category_id = _integer(annotation.get("category_id"), "COCO annotation.category_id")
        if category_id not in category_by_id:
            raise FrozenHoldoutReferenceError("COCO annotation references an unknown category")
        bbox = _list(annotation.get("bbox"), "COCO annotation.bbox")
        if len(bbox) != 4 or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in bbox
        ):
            raise FrozenHoldoutReferenceError("COCO annotation bbox is malformed")
        if bbox[0] < 0 or bbox[1] < 0 or bbox[2] <= 0 or bbox[3] <= 0:
            raise FrozenHoldoutReferenceError("COCO annotation bbox must have positive area")
        image_dimensions = images_by_id[image_id]
        if (
            bbox[0] + bbox[2] > image_dimensions["width"] + 1e-6
            or bbox[1] + bbox[3] > image_dimensions["height"] + 1e-6
        ):
            raise FrozenHoldoutReferenceError("COCO annotation bbox exceeds image bounds")
        if annotation.get("iscrowd", 0) != 0:
            raise FrozenHoldoutReferenceError("COCO crowd annotations are unsupported")
        annotations_by_image[image_id] += 1
        annotations_by_category[category_by_id[category_id]] += 1
    zero_annotation_images = sum(image_id not in annotations_by_image for image_id in images_by_id)
    if _integer(
        counts.get("zero_annotation_images"),
        "reference manifest.counts.zero_annotation_images",
        minimum=0,
    ) != zero_annotation_images or _integer(
        counts.get("categories"), "reference manifest.counts.categories"
    ) != len(REQUIRED_LABELS):
        raise FrozenHoldoutReferenceError("reference zero-image or category count is inconsistent")
    if (
        _integer(task.get("annotation_count"), "reference task annotation_count", minimum=0)
        != len(annotations)
        or _integer(
            task.get("zero_annotation_image_count"),
            "reference task zero_annotation_image_count",
            minimum=0,
        )
        != zero_annotation_images
    ):
        raise FrozenHoldoutReferenceError("reference task statistics are inconsistent")
    reported_by_category = _object(
        counts.get("annotations_by_category"),
        "reference manifest.counts.annotations_by_category",
    )
    expected_by_category = {name: annotations_by_category[name] for name in REQUIRED_LABELS}
    if reported_by_category != expected_by_category:
        raise FrozenHoldoutReferenceError("reference per-category counts are inconsistent")
    expected_task_statistics = {
        "task_id": identity.task_id,
        "job_id": identity.job_id,
        "image_count": len(images),
        "zero_annotation_image_count": zero_annotation_images,
        "annotation_count": len(annotations),
    }
    if task != expected_task_statistics:
        raise FrozenHoldoutReferenceError("reference task statistics are inconsistent")

    scene_images: Counter[str] = Counter()
    scene_annotations: Counter[str] = Counter()
    asset_images: Counter[tuple[str, str]] = Counter()
    asset_annotations: Counter[tuple[str, str]] = Counter()
    asset_values: dict[tuple[str, str], str | int] = {}
    asset_leakage_groups: dict[tuple[str, str], str] = {}
    asset_scenes: dict[tuple[str, str], set[str]] = {}
    for image_id, image in images_by_id.items():
        scene_id = image["scene_id"]
        asset_identity = image["asset_identity"]
        leakage_group_id = image["leakage_group_id"]
        previous_leakage = asset_leakage_groups.get(asset_identity)
        if previous_leakage is not None and previous_leakage != leakage_group_id:
            raise FrozenHoldoutReferenceError("COCO source asset maps to multiple leakage groups")
        scene_images[scene_id] += 1
        scene_annotations[scene_id] += annotations_by_image[image_id]
        asset_values[asset_identity] = image["asset_id"]
        asset_leakage_groups[asset_identity] = leakage_group_id
        asset_images[asset_identity] += 1
        asset_annotations[asset_identity] += annotations_by_image[image_id]
        asset_scenes.setdefault(asset_identity, set()).add(scene_id)

    expected_scene_statistics = [
        {
            "scene_id": scene_id,
            "image_count": scene_images[scene_id],
            "annotation_count": scene_annotations[scene_id],
        }
        for scene_id in sorted(scene_images)
    ]
    if _list(statistics.get("scenes"), "reference source_statistics.scenes") != (
        expected_scene_statistics
    ):
        raise FrozenHoldoutReferenceError("reference scene statistics are inconsistent")
    expected_asset_statistics = [
        {
            "asset_id": asset_values[identity],
            "leakage_group_id": asset_leakage_groups[identity],
            "image_count": asset_images[identity],
            "annotation_count": asset_annotations[identity],
            "scene_ids": sorted(asset_scenes[identity]),
        }
        for identity in sorted(asset_images)
    ]
    if _list(statistics.get("assets"), "reference source_statistics.assets") != (
        expected_asset_statistics
    ):
        raise FrozenHoldoutReferenceError("reference asset statistics are inconsistent")
    evidence = _object(manifest.get("evidence"), "reference manifest.evidence")
    task_inputs = _list(evidence.get("task_inputs"), "reference manifest.evidence.task_inputs")
    if len(task_inputs) != 1:
        raise FrozenHoldoutReferenceError(
            "reference manifest must contain exactly one task input evidence record"
        )
    task_input = _object(task_inputs[0], "reference manifest.evidence.task_inputs[0]")
    if (
        _integer(task_input.get("task_id"), "reference task input.task_id") != identity.task_id
        or _integer(task_input.get("job_id"), "reference task input.job_id") != identity.job_id
    ):
        raise FrozenHoldoutReferenceError("reference task input belongs to another task or job")
    source_snapshot = _object(task_input.get("snapshot"), "reference task input.snapshot")
    source_receipt = _object(
        task_input.get("completion_receipt"), "reference task input.completion_receipt"
    )
    source_binding = {
        "snapshot_sha256": _digest(
            source_snapshot.get("sha256"), "reference task input snapshot SHA"
        ),
        "canonical_annotations_sha256": _digest(
            source_snapshot.get("canonical_annotations_sha256"),
            "reference task input canonical annotations SHA",
        ),
        "completion_receipt_sha256": _digest(
            source_receipt.get("sha256"), "reference task input completion receipt SHA"
        ),
    }
    return {
        "root": root,
        "manifest": manifest,
        "manifest_sha256": manifest_sha,
        "coco": coco,
        "annotation_count": len(annotations),
        "frame_count": frame_count,
        "task_input_binding": source_binding,
    }, records


def _validate_snapshot(
    snapshot: Mapping[str, Any],
    *,
    annotation_count: int,
    frame_count: int,
    identity: FinalHoldoutIdentity,
    require_completed: bool,
) -> str:
    if snapshot.get("snapshot_schema") != SNAPSHOT_SCHEMA:
        raise FrozenHoldoutReferenceError("snapshot schema is unsupported")
    task = _object(snapshot.get("task"), "snapshot.task")
    if (
        _integer(task.get("id"), "snapshot.task.id") != identity.task_id
        or _integer(task.get("size"), "snapshot.task.size") != frame_count
    ):
        raise FrozenHoldoutReferenceError("snapshot is not the complete final-holdout task")
    jobs = _list(snapshot.get("jobs"), "snapshot.jobs")
    if len(jobs) != 1:
        raise FrozenHoldoutReferenceError("snapshot must contain exactly one job")
    job = _object(jobs[0], "snapshot.jobs[0]")
    issues = _object(job.get("issues"), "snapshot.jobs[0].issues")
    if (
        _integer(job.get("id"), "snapshot job.id") != identity.job_id
        or _integer(job.get("task_id"), "snapshot job.task_id") != identity.task_id
        or _integer(job.get("start_frame"), "snapshot job.start_frame") != 0
        or _integer(job.get("stop_frame"), "snapshot job.stop_frame") != frame_count - 1
        or _integer(job.get("frame_count"), "snapshot job.frame_count") != frame_count
        or job.get("type") != "annotation"
        or job.get("parent_job_id") is not None
        or _integer(issues.get("count"), "snapshot job issue count") != 0
    ):
        raise FrozenHoldoutReferenceError("snapshot job identity or frame span is invalid")
    if require_completed and job.get("state") != "completed":
        raise FrozenHoldoutReferenceError("completed snapshot does not show a completed job")
    gate = _object(snapshot.get("final_gate"), "snapshot.final_gate")
    if gate.get("passed") is not True or _list(gate.get("blocking_reasons"), "snapshot blockers"):
        raise FrozenHoldoutReferenceError("snapshot final gate did not pass")
    counts = _object(snapshot.get("counts"), "snapshot.counts")
    if _integer(counts.get("shapes"), "snapshot.counts.shapes") != annotation_count:
        raise FrozenHoldoutReferenceError("snapshot shape count differs from COCO annotations")
    return _digest(snapshot.get("canonical_annotations_sha256"), "snapshot canonical SHA")


def _validate_review_evidence(
    *,
    receipt_path: Path,
    post_snapshot_path: Path,
    decisions_path: Path,
    completed_snapshot_path: Path | None,
    annotation_count: int,
    frame_count: int,
    identity: FinalHoldoutIdentity,
) -> tuple[dict[str, Any], list[tuple[str, bytes, str, int]], str]:
    receipt, receipt_payload, receipt_sha, receipt_size = _read_json(
        receipt_path, "completion receipt"
    )
    post, post_payload, post_sha, post_size = _read_json(
        post_snapshot_path, "receipt-bound post snapshot"
    )
    decisions, decisions_payload, decisions_sha, decisions_size = _read_json(
        decisions_path, "final decisions"
    )
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise FrozenHoldoutReferenceError("completion receipt schema is unsupported")
    if (
        _integer(receipt.get("task_id"), "receipt.task_id") != identity.task_id
        or _integer(receipt.get("job_id"), "receipt.job_id") != identity.job_id
        or receipt.get("dry_run") is not False
        or receipt.get("mutation_performed") is not True
        or receipt.get("job_state_after") != "completed"
        or _integer(receipt.get("annotation_count"), "receipt.annotation_count") != annotation_count
    ):
        raise FrozenHoldoutReferenceError(
            "receipt does not prove applied final-holdout completion"
        )
    validation = _object(receipt.get("decision_validation"), "receipt.decision_validation")
    if (
        validation.get("valid") is not True
        or validation.get("mutation_performed") is not False
        or validation.get("scope") != "full_review"
        or _integer(validation.get("task_id"), "decision validation task") != identity.task_id
        or _integer(validation.get("snapshot_frame_count"), "decision validation frames")
        != frame_count
        or _integer(validation.get("reviewed_frame_count"), "decision reviewed frames")
        != frame_count
        or _integer(validation.get("unresolved_manual_flag_count"), "unresolved flags") != 0
    ):
        raise FrozenHoldoutReferenceError("receipt does not prove exhaustive full review")
    if (
        decisions.get("scope") != "full_review"
        or decisions.get("mutation_performed") is not False
        or _integer(decisions.get("task_id"), "decisions.task_id") != identity.task_id
    ):
        raise FrozenHoldoutReferenceError(
            "final decisions do not match the configured final holdout"
        )
    frames: set[int] = set()
    for index, raw in enumerate(_list(decisions.get("frame_reviews"), "decisions.frame_reviews")):
        frame = _object(raw, f"decisions.frame_reviews[{index}]")
        number = _integer(frame.get("frame"), f"decisions.frame_reviews[{index}].frame")
        if frame.get("reviewed") is not True or number in frames:
            raise FrozenHoldoutReferenceError("final decisions do not review each frame once")
        frames.add(number)
    if frames != set(range(frame_count)):
        raise FrozenHoldoutReferenceError(
            "final decisions do not cover the complete final-holdout frame span"
        )
    canonical_sha = _validate_snapshot(
        post,
        annotation_count=annotation_count,
        frame_count=frame_count,
        identity=identity,
        require_completed=False,
    )
    if (
        _digest(
            receipt.get("verified_post_completion_canonical_annotations_sha256"),
            "receipt verified annotation SHA",
        )
        != canonical_sha
    ):
        raise FrozenHoldoutReferenceError("receipt and post snapshot annotation hashes differ")
    evidence = _object(receipt.get("evidence"), "receipt.evidence")
    for name, actual_sha in (("post_snapshot", post_sha), ("decisions", decisions_sha)):
        record = _object(evidence.get(name), f"receipt.evidence.{name}")
        if _digest(record.get("sha256"), f"receipt.evidence.{name}.sha256") != actual_sha:
            raise FrozenHoldoutReferenceError(f"receipt is not bound to supplied {name}")

    managed = [
        (
            "evidence/completion-receipt.json",
            receipt_payload,
            receipt_sha,
            receipt_size,
        ),
        (
            "evidence/post-snapshot.json",
            post_payload,
            post_sha,
            post_size,
        ),
        (
            "evidence/final-decisions.json",
            decisions_payload,
            decisions_sha,
            decisions_size,
        ),
    ]
    review = {
        "completion_receipt": _bound("evidence/completion-receipt.json", receipt_sha, receipt_size),
        "post_snapshot": _bound("evidence/post-snapshot.json", post_sha, post_size),
        "final_decisions": _bound("evidence/final-decisions.json", decisions_sha, decisions_size),
        "source_canonical_annotations_sha256": canonical_sha,
    }
    if completed_snapshot_path is not None:
        completed, completed_payload, completed_sha, completed_size = _read_json(
            completed_snapshot_path,
            "optional completed snapshot",
        )
        completed_canonical = _validate_snapshot(
            completed,
            annotation_count=annotation_count,
            frame_count=frame_count,
            identity=identity,
            require_completed=True,
        )
        if completed_canonical != canonical_sha:
            raise FrozenHoldoutReferenceError("completed snapshot annotations differ from receipt")
        managed.append(
            (
                "evidence/completed-snapshot.json",
                completed_payload,
                completed_sha,
                completed_size,
            )
        )
        review["completed_snapshot"] = _bound(
            "evidence/completed-snapshot.json", completed_sha, completed_size
        )
    return review, managed, canonical_sha


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _write_verified_payload(
    target: Path,
    payload: bytes,
    expected_sha: str,
    expected_size: int,
    location: str,
) -> None:
    if len(payload) != expected_size or _sha256_bytes(payload) != expected_sha:
        raise FrozenHoldoutReferenceError(f"validated input bytes changed in memory: {location}")
    _write_bytes(target, payload)


def _publish_no_replace(staging: Path, output: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    source = os.fsencode(staging)
    target = os.fsencode(output)
    if sys.platform == "darwin":
        function = libc.renamex_np
        function.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        function.restype = ctypes.c_int
        result = function(source, target, 0x00000004)
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        function = libc.renameat2
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        result = function(-100, source, -100, target, 0x00000001)
    else:
        raise FrozenHoldoutReferenceError("atomic no-replace publication is unsupported")
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FrozenHoldoutReferenceError(f"output directory already exists: {output}")
    raise OSError(error, os.strerror(error), str(output))


def build_frozen_holdout_reference(
    *,
    reference_dir: Path,
    task_id: int,
    job_id: int,
    completion_receipt: Path,
    post_snapshot: Path,
    final_decisions: Path,
    output_dir: Path,
    completed_snapshot: Path | None = None,
) -> dict[str, Any]:
    """Validate evidence and atomically publish a configured holdout reference."""

    try:
        identity = resolve_final_holdout_identity(task_id=task_id, job_id=job_id)
    except FinalHoldoutConfigError as error:
        raise FrozenHoldoutReferenceError(str(error)) from error
    raw_output = Path(output_dir)
    output = raw_output.parent.resolve() / raw_output.name
    if os.path.lexists(output):
        raise FrozenHoldoutReferenceError(f"output directory already exists: {output}")
    source, source_records = _validate_source_reference(Path(reference_dir), identity)
    review, review_sources, canonical_sha = _validate_review_evidence(
        receipt_path=Path(completion_receipt),
        post_snapshot_path=Path(post_snapshot),
        decisions_path=Path(final_decisions),
        completed_snapshot_path=Path(completed_snapshot) if completed_snapshot else None,
        annotation_count=source["annotation_count"],
        frame_count=source["frame_count"],
        identity=identity,
    )
    source_binding = _object(source["task_input_binding"], "source task input binding")
    if source_binding["canonical_annotations_sha256"] != canonical_sha:
        raise FrozenHoldoutReferenceError(
            "source reference canonical annotations differ from review evidence"
        )
    if source_binding["completion_receipt_sha256"] != review["completion_receipt"]["sha256"]:
        raise FrozenHoldoutReferenceError(
            "source reference completion receipt differs from review evidence"
        )
    if (
        "completed_snapshot" in review
        and source_binding["snapshot_sha256"] != review["completed_snapshot"]["sha256"]
    ):
        raise FrozenHoldoutReferenceError(
            "source reference completed snapshot differs from review evidence"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    published = False
    try:
        files: list[dict[str, Any]] = []
        for record in source_records:
            target = staging / record["path"]
            _write_verified_payload(
                target,
                record["payload"],
                record["sha256"],
                record["size_bytes"],
                f"reference file {record['path']}",
            )
            files.append(_bound(record["path"], record["sha256"], record["size_bytes"]))
        for relative, payload, digest, size in review_sources:
            _write_verified_payload(staging / relative, payload, digest, size, relative)
            files.append(_bound(relative, digest, size))
        manifest = {
            "schema": OUTPUT_SCHEMA,
            "task_id": identity.task_id,
            "gate": {
                "passed": True,
                "blocking_reasons": [],
                "checks": {
                    "source_reference_v2_verified": True,
                    "all_managed_files_verified": True,
                    "fixed_eight_class_taxonomy_verified": True,
                    "complete_final_holdout_frame_span_verified": True,
                    "full_review_decisions_verified": True,
                    "completion_receipt_verified": True,
                    "receipt_bound_post_snapshot_verified": True,
                },
            },
            "source_reference": {
                "schema": INPUT_SCHEMA,
                "manifest_sha256": source["manifest_sha256"],
            },
            "counts": source["manifest"]["counts"],
            "source_statistics": source["manifest"]["source_statistics"],
            "review_completion": {
                key: value for key, value in review.items() if key != "completed_snapshot"
            },
            "files": sorted(files, key=lambda item: item["path"]),
        }
        if "completed_snapshot" in review:
            manifest["additional_evidence"] = {"completed_snapshot": review["completed_snapshot"]}
        _write_bytes(staging / "manifest.json", _json_bytes(manifest))
        directory_fd = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        _publish_no_replace(staging, output)
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
        "manifest": str(output / "manifest.json"),
        "annotations_coco": str(output / "annotations.coco.json"),
        "task_id": identity.task_id,
        "job_id": identity.job_id,
        "annotation_count": source["annotation_count"],
        "managed_file_count": len(files),
        "source_canonical_annotations_sha256": canonical_sha,
        "gate": manifest["gate"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--job-id", type=int, required=True)
    parser.add_argument("--completion-receipt", type=Path, required=True)
    parser.add_argument("--post-snapshot", type=Path, required=True)
    parser.add_argument("--final-decisions", type=Path, required=True)
    parser.add_argument("--completed-snapshot", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        summary = build_frozen_holdout_reference(
            reference_dir=args.reference_dir,
            task_id=args.task_id,
            job_id=args.job_id,
            completion_receipt=args.completion_receipt,
            post_snapshot=args.post_snapshot,
            final_decisions=args.final_decisions,
            completed_snapshot=args.completed_snapshot,
            output_dir=args.output_dir,
        )
    except FrozenHoldoutReferenceError as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
