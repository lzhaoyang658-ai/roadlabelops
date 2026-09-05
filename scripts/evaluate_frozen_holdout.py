#!/usr/bin/env python3
"""Evaluate one frozen candidate against one baseline on a configured final holdout.

The command is preflight-only by default.  ``--apply`` is deliberately a
single-use operation: it creates a durable holdout-owned claim before model
loading, then publishes the result with exclusive atomic creation.  A failed
run keeps the claim and therefore cannot silently consume the frozen holdout
again, even if the candidate freeze directory is relocated.

All paths stored in JSON records are resolved relative to the record that owns
them.  The evaluator never talks to CVAT and never changes holdout artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import stat
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from roadlabelops.holdout_policy import (
    NO_FINAL_HOLDOUT_STATEMENT,
    FinalHoldoutConfigError,
    FinalHoldoutIdentity,
    resolve_final_holdout_identity,
)
from roadlabelops.tools.detection import (
    MODEL_MAPPING,
    ROAD_LABELS,
    postprocess_predictions,
    result_frame_index,
)
from roadlabelops.tools.quality import calculate_quality

PROTOCOL_SCHEMA = {"name": "roadlabelops.frozen-holdout-protocol", "version": 1}
FREEZE_SCHEMA = {"name": "roadlabelops.yolo-frozen-candidate", "version": 1}
HOLDOUT_SCHEMA = {"name": "roadlabelops.frozen-holdout-reference", "version": 1}
TRAINING_SCHEMA = {"name": "roadlabelops.training-coco-reference", "version": 2}
YOLO_DATASET_SCHEMA = {"name": "roadlabelops.yolo-dataset", "version": 2}
TRAINING_PROTOCOL_SCHEMA = {
    "name": "roadlabelops.yolo-candidate-training-protocol",
    "version": 1,
}
OVERLAP_SCHEMA = {"name": "roadlabelops.training-holdout-overlap-evidence", "version": 1}
PREFLIGHT_SCHEMA = {"name": "roadlabelops.frozen-holdout-preflight", "version": 1}
RESULT_SCHEMA = {"name": "roadlabelops.frozen-holdout-evaluation", "version": 1}
CLAIM_SCHEMA = {"name": "roadlabelops.frozen-holdout-consumption-claim", "version": 1}
COMPLETION_RECEIPT_SCHEMA = {
    "name": "roadlabelops.cvat-job-completion-receipt",
    "version": 1,
}
CVAT_SNAPSHOT_SCHEMA = {"name": "roadlabelops.cvat-task-snapshot", "version": 1}

EXPECTED_GATES = {
    "precision_min": 0.90,
    "recall_min": 0.85,
    "clean_frame_rate_min": 0.80,
    "match_iou": 0.50,
    "candidate_f1_strictly_greater_than_baseline": True,
}
CANONICAL_LABELS = tuple(ROAD_LABELS)
SELECTION_ORDER = ("mAP50-95", "mAP50", "recall", "precision", "smaller_seed")
SELECTION_METRICS = frozenset({"precision", "recall", "map50", "map50_95", "fitness"})
EXPECTED_TRAINING_SEEDS = frozenset({42, 43, 44})
NO_HOLDOUT_STATEMENT = NO_FINAL_HOLDOUT_STATEMENT


class FrozenHoldoutError(ValueError):
    """Raised when a final holdout evaluation would be unsafe or ambiguous."""


@dataclass(frozen=True)
class BoundFile:
    path: Path
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class SourceVideo:
    scene_id: str
    frame_step: int
    file: BoundFile


@dataclass(frozen=True)
class ReferenceData:
    manifest: BoundFile
    annotations: BoundFile
    payload: Mapping[str, Any]
    ground_truth: tuple[dict[str, Any], ...]
    frame_keys: tuple[tuple[str, int], ...]
    image_hashes: frozenset[str]
    asset_ids: frozenset[tuple[str, str]]
    source_hashes: frozenset[str]
    source_shas_by_scene: Mapping[str, frozenset[str]]
    source_frame_keys: frozenset[tuple[str, int]]
    zero_annotation_frame_count: int


@dataclass(frozen=True)
class CandidateFreeze:
    record: BoundFile
    candidate_id: str
    weight: BoundFile
    selected_seed: int
    dataset_manifest_sha256: str
    dataset_manifest_size_bytes: int
    base_weight_sha256: str
    base_weight_size_bytes: int


@dataclass(frozen=True)
class ValidatedProtocol:
    protocol: BoundFile
    protocol_id: str
    candidate: CandidateFreeze
    training_reference: ReferenceData
    baseline_weight: BoundFile
    holdout: ReferenceData
    overlap_evidence: BoundFile
    warmup_image: BoundFile
    source_videos: tuple[SourceVideo, ...]
    settings: Mapping[str, Any]
    batch_sha256: str
    validation: Mapping[str, Any]


@dataclass(frozen=True)
class ModelRun:
    raw_predictions: tuple[dict[str, Any], ...]
    observed_frame_keys: frozenset[tuple[str, int]]
    processed_video_frame_count: int
    inference_wall_seconds: float
    model_metadata: Mapping[str, Any]


ModelRunner = Callable[
    [Path, Path, Sequence[SourceVideo], Mapping[str, Any], str],
    ModelRun,
]


def _object(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FrozenHoldoutError(f"{location} must be an object")
    return value


def _list(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise FrozenHoldoutError(f"{location} must be a list")
    return value


def _text(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FrozenHoldoutError(f"{location} must be a non-empty string")
    return value


def _integer(value: Any, location: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FrozenHoldoutError(f"{location} must be an integer")
    if minimum is not None and value < minimum:
        raise FrozenHoldoutError(f"{location} must be at least {minimum}")
    return value


def _number(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FrozenHoldoutError(f"{location} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise FrozenHoldoutError(f"{location} must be finite")
    return result


def _sha256(value: Any, location: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise FrozenHoldoutError(f"{location} must be a lowercase SHA-256 digest")
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _leaf_absolute(path: Path) -> Path:
    """Resolve a path's parent while preserving its leaf for symlink checks."""

    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = Path.cwd() / expanded
    return expanded.parent.resolve() / expanded.name


def _owned_path(raw: Any, *, owner: Path, location: str) -> Path:
    text = _text(raw, location)
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = owner.parent / candidate
    return _leaf_absolute(candidate)


def _read_regular_bytes(path: Path, location: str) -> bytes:
    path = _leaf_absolute(path)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise FrozenHoldoutError(f"could not open {location}: {path}: {error}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise FrozenHoldoutError(f"{location} must be a regular file: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        encoded = b"".join(chunks)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if (
            identity_before != identity_after
            or len(encoded) != before.st_size
            or len(encoded) != after.st_size
        ):
            raise FrozenHoldoutError(f"{location} changed while being read: {path}")
        return encoded
    finally:
        os.close(descriptor)


def _read_json(path: Path, location: str) -> tuple[dict[str, Any], BoundFile]:
    resolved = _leaf_absolute(path)
    encoded = _read_regular_bytes(resolved, location)
    try:
        payload = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FrozenHoldoutError(f"could not decode {location}: {error}") from error
    return _object(payload, location), BoundFile(
        resolved, hashlib.sha256(encoded).hexdigest(), len(encoded)
    )


def _validate_bound_file(
    record: Any,
    *,
    owner: Path,
    location: str,
) -> BoundFile:
    value = _object(record, location)
    if set(value) != {"path", "sha256", "size_bytes"}:
        raise FrozenHoldoutError(f"{location} must contain exactly path, sha256, and size_bytes")
    path = _owned_path(value.get("path"), owner=owner, location=f"{location}.path")
    expected_sha256 = _sha256(value.get("sha256"), f"{location}.sha256")
    expected_size = _integer(value.get("size_bytes"), f"{location}.size_bytes", minimum=1)
    encoded = _read_regular_bytes(path, location)
    actual_sha256 = hashlib.sha256(encoded).hexdigest()
    if actual_sha256 != expected_sha256 or len(encoded) != expected_size:
        raise FrozenHoldoutError(f"{location} does not match its managed hash and size")
    return BoundFile(path, expected_sha256, expected_size)


def _validate_explicit_file(
    path: Path,
    *,
    expected_sha256: str,
    expected_size: int,
    location: str,
) -> BoundFile:
    encoded = _read_regular_bytes(path, location)
    if hashlib.sha256(encoded).hexdigest() != expected_sha256 or len(encoded) != expected_size:
        raise FrozenHoldoutError(f"{location} differs from its frozen record")
    return BoundFile(_leaf_absolute(path), expected_sha256, expected_size)


def _safe_relative_path(value: Any, location: str) -> PurePosixPath:
    text = _text(value, location)
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or PureWindowsPath(text).is_absolute()
        or "\\" in text
        or ".." in path.parts
        or not path.name
    ):
        raise FrozenHoldoutError(f"{location} is not a safe relative POSIX path")
    return path


def _managed_path(root: Path, relative: PurePosixPath, location: str) -> Path:
    candidate = root.joinpath(*relative.parts)
    resolved = _leaf_absolute(candidate)
    if not resolved.is_relative_to(root.resolve()):
        raise FrozenHoldoutError(f"{location} escapes its reference directory")
    return resolved


def _typed_asset(value: Any, location: str) -> tuple[str, str]:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise FrozenHoldoutError(f"{location} must be a string or integer")
    if isinstance(value, str) and not value:
        raise FrozenHoldoutError(f"{location} must not be empty")
    return type(value).__name__, str(value)


def _source_identity(
    image: Mapping[str, Any], location: str
) -> tuple[tuple[str, str], str, tuple[str, int]]:
    asset_id = _typed_asset(image.get("source_asset_id"), f"{location}.source_asset_id")
    leakage_group = _text(
        image.get("source_leakage_group_id"), f"{location}.source_leakage_group_id"
    )
    if not leakage_group.startswith("sha256:"):
        raise FrozenHoldoutError(f"{location}.source_leakage_group_id must use sha256 identity")
    source_sha = _sha256(leakage_group.removeprefix("sha256:"), f"{location} source SHA")
    normalized_frame = _integer(
        image.get("source_normalized_asset_frame"),
        f"{location}.source_normalized_asset_frame",
        minimum=0,
    )
    return asset_id, source_sha, (source_sha, normalized_frame)


def _parse_coco(
    payload: Mapping[str, Any], *, location: str
) -> tuple[
    tuple[dict[str, Any], ...],
    tuple[tuple[str, int], ...],
    dict[str, Mapping[str, Any]],
    frozenset[tuple[str, str]],
    frozenset[str],
    dict[str, frozenset[str]],
    frozenset[tuple[str, int]],
    int,
]:
    categories = _list(payload.get("categories"), f"{location}.categories")
    category_by_id: dict[int, str] = {}
    for index, raw_category in enumerate(categories):
        category = _object(raw_category, f"{location}.categories[{index}]")
        identifier = _integer(category.get("id"), f"{location}.categories[{index}].id")
        name = _text(category.get("name"), f"{location}.categories[{index}].name")
        if identifier in category_by_id:
            raise FrozenHoldoutError(f"{location} contains duplicate category IDs")
        category_by_id[identifier] = name
    expected_categories = {index: name for index, name in enumerate(CANONICAL_LABELS, start=1)}
    if category_by_id != expected_categories:
        raise FrozenHoldoutError(
            f"{location} must use the canonical eight-class ID mapping {expected_categories}"
        )

    images_by_id: dict[int, Mapping[str, Any]] = {}
    image_files: dict[str, Mapping[str, Any]] = {}
    frame_keys: list[tuple[str, int]] = []
    asset_ids: set[tuple[str, str]] = set()
    source_hashes: set[str] = set()
    source_shas_by_scene: dict[str, set[str]] = {}
    source_frame_keys: set[tuple[str, int]] = set()
    for index, raw_image in enumerate(_list(payload.get("images"), f"{location}.images")):
        image = _object(raw_image, f"{location}.images[{index}]")
        identifier = _integer(image.get("id"), f"{location}.images[{index}].id", minimum=1)
        if identifier in images_by_id:
            raise FrozenHoldoutError(f"{location} contains duplicate image IDs")
        file_name = str(
            _safe_relative_path(image.get("file_name"), f"{location}.images[{index}].file_name")
        )
        if file_name in image_files:
            raise FrozenHoldoutError(f"{location} contains duplicate image file names")
        _integer(image.get("width"), f"{location}.images[{index}].width", minimum=1)
        _integer(image.get("height"), f"{location}.images[{index}].height", minimum=1)
        scene_id = _text(image.get("scene_id"), f"{location}.images[{index}].scene_id")
        source_frame = _integer(
            image.get("source_frame"), f"{location}.images[{index}].source_frame", minimum=0
        )
        frame_key = (scene_id, source_frame)
        if frame_key in frame_keys:
            raise FrozenHoldoutError(f"{location} contains duplicate scene/frame identities")
        asset_id, source_sha, source_frame_key = _source_identity(
            image, f"{location}.images[{index}]"
        )
        source_shas_by_scene.setdefault(scene_id, set()).add(source_sha)
        if source_frame_key in source_frame_keys:
            raise FrozenHoldoutError(
                f"{location} aliases one normalized source frame to multiple images"
            )
        images_by_id[identifier] = image
        image_files[file_name] = image
        frame_keys.append(frame_key)
        asset_ids.add(asset_id)
        source_hashes.add(source_sha)
        source_frame_keys.add(source_frame_key)
    if not images_by_id:
        raise FrozenHoldoutError(f"{location}.images must not be empty")

    ground_truth: list[dict[str, Any]] = []
    annotation_ids: set[int] = set()
    annotations_by_image: Counter[int] = Counter()
    for index, raw_annotation in enumerate(
        _list(payload.get("annotations"), f"{location}.annotations")
    ):
        annotation = _object(raw_annotation, f"{location}.annotations[{index}]")
        identifier = _integer(
            annotation.get("id"), f"{location}.annotations[{index}].id", minimum=1
        )
        if identifier in annotation_ids:
            raise FrozenHoldoutError(f"{location} contains duplicate annotation IDs")
        annotation_ids.add(identifier)
        image_id = _integer(annotation.get("image_id"), f"{location}.annotations[{index}].image_id")
        if image_id not in images_by_id:
            raise FrozenHoldoutError(f"{location} annotation references an unknown image")
        category_id = _integer(
            annotation.get("category_id"), f"{location}.annotations[{index}].category_id"
        )
        if category_id not in category_by_id:
            raise FrozenHoldoutError(f"{location} annotation references an unknown category")
        bbox = _list(annotation.get("bbox"), f"{location}.annotations[{index}].bbox")
        if len(bbox) != 4:
            raise FrozenHoldoutError(f"{location} annotation bbox must contain four numbers")
        x, y, width, height = (
            _number(value, f"{location}.annotations[{index}].bbox[{coordinate}]")
            for coordinate, value in enumerate(bbox)
        )
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise FrozenHoldoutError(f"{location} annotation bbox must have positive area")
        image = images_by_id[image_id]
        if x + width > int(image["width"]) + 1e-6 or y + height > int(image["height"]) + 1e-6:
            raise FrozenHoldoutError(f"{location} annotation bbox exceeds image bounds")
        if annotation.get("iscrowd", 0) != 0:
            raise FrozenHoldoutError(f"{location} does not support crowd annotations")
        annotations_by_image[image_id] += 1
        ground_truth.append(
            {
                "scene_id": str(image["scene_id"]),
                "frame": int(image["source_frame"]),
                "label": category_by_id[category_id],
                "bbox": [x, y, x + width, y + height],
            }
        )
    zero_count = sum(identifier not in annotations_by_image for identifier in images_by_id)
    return (
        tuple(ground_truth),
        tuple(frame_keys),
        image_files,
        frozenset(asset_ids),
        frozenset(source_hashes),
        {scene: frozenset(hashes) for scene, hashes in source_shas_by_scene.items()},
        frozenset(source_frame_keys),
        zero_count,
    )


def _validate_managed_manifest(
    manifest: BoundFile,
    *,
    expected_schema: Mapping[str, Any],
    annotations_record: BoundFile | None,
    location: str,
    verify_all_files: bool,
) -> tuple[dict[str, Any], BoundFile, Mapping[str, Any]]:
    payload, actual = _read_json(manifest.path, location)
    if actual.sha256 != manifest.sha256 or actual.size_bytes != manifest.size_bytes:
        raise FrozenHoldoutError(f"{location} changed after its bound record was validated")
    if payload.get("schema") != expected_schema:
        raise FrozenHoldoutError(f"{location} schema is unsupported")
    root = manifest.path.parent.resolve()
    managed: dict[str, BoundFile] = {}
    for index, raw_file in enumerate(_list(payload.get("files"), f"{location}.files")):
        record = _object(raw_file, f"{location}.files[{index}]")
        if set(record) != {"path", "sha256", "size_bytes"}:
            raise FrozenHoldoutError(f"{location}.files[{index}] has unexpected or missing keys")
        relative = _safe_relative_path(record.get("path"), f"{location}.files[{index}].path")
        key = str(relative)
        if key in managed:
            raise FrozenHoldoutError(f"{location} contains duplicate managed paths")
        expected_sha = _sha256(record.get("sha256"), f"{location}.files[{index}].sha256")
        expected_size = _integer(
            record.get("size_bytes"), f"{location}.files[{index}].size_bytes", minimum=1
        )
        path = _managed_path(root, relative, f"{location}.files[{index}]")
        if verify_all_files or key == "annotations.coco.json":
            managed[key] = _validate_explicit_file(
                path,
                expected_sha256=expected_sha,
                expected_size=expected_size,
                location=f"{location}.files[{index}]",
            )
        else:
            managed[key] = BoundFile(path, expected_sha, expected_size)
    if "annotations.coco.json" not in managed:
        raise FrozenHoldoutError(f"{location} does not manage annotations.coco.json")
    managed_annotations = managed["annotations.coco.json"]
    if annotations_record is not None and managed_annotations != annotations_record:
        raise FrozenHoldoutError(
            f"{location} annotations entry differs from the protocol-bound annotations"
        )
    _annotations_payload, actual_annotations = _read_json(
        managed_annotations.path, f"{location} annotations"
    )
    if actual_annotations != managed_annotations:
        raise FrozenHoldoutError(f"{location} annotations changed while being validated")
    return payload, managed_annotations, {key: value for key, value in managed.items()}


def _build_reference_data(
    manifest: BoundFile,
    *,
    expected_schema: Mapping[str, Any],
    annotations_record: BoundFile | None,
    location: str,
    verify_all_files: bool,
) -> ReferenceData:
    manifest_payload, annotations, managed = _validate_managed_manifest(
        manifest,
        expected_schema=expected_schema,
        annotations_record=annotations_record,
        location=location,
        verify_all_files=verify_all_files,
    )
    annotations_payload, _ = _read_json(annotations.path, f"{location} annotations")
    (
        ground_truth,
        frame_keys,
        image_files,
        asset_ids,
        source_hashes,
        source_shas_by_scene,
        source_frame_keys,
        zero_count,
    ) = _parse_coco(annotations_payload, location=f"{location} annotations")
    missing = sorted(set(image_files) - set(managed))
    if missing:
        raise FrozenHoldoutError(f"{location} leaves COCO images unmanaged: {missing[:3]}")
    for file_name, image in image_files.items():
        image_sha = _sha256(image.get("sha256"), f"COCO image {file_name}.sha256")
        if managed[file_name].sha256 != image_sha:
            raise FrozenHoldoutError(f"{location} COCO image hash differs from its manifest")
    return ReferenceData(
        manifest=manifest,
        annotations=annotations,
        payload=manifest_payload,
        ground_truth=ground_truth,
        frame_keys=frame_keys,
        image_hashes=frozenset(managed[name].sha256 for name in image_files),
        asset_ids=asset_ids,
        source_hashes=source_hashes,
        source_shas_by_scene=source_shas_by_scene,
        source_frame_keys=source_frame_keys,
        zero_annotation_frame_count=zero_count,
    )


def _validate_managed_review_file(
    manifest: BoundFile,
    manifest_payload: Mapping[str, Any],
    binding: Any,
    *,
    location: str,
) -> BoundFile:
    record = _object(binding, location)
    if set(record) != {"path", "sha256", "size_bytes"}:
        raise FrozenHoldoutError(f"{location} must contain exactly path, sha256, and size_bytes")
    managed_records = [
        _object(raw, f"holdout manifest.files[{index}]")
        for index, raw in enumerate(_list(manifest_payload.get("files"), "holdout manifest.files"))
    ]
    if record not in managed_records:
        raise FrozenHoldoutError(f"{location} is not an exact managed holdout file")
    return _validate_bound_file(record, owner=manifest.path, location=location)


def _validate_full_review_completion(
    holdout: ReferenceData,
    identity: FinalHoldoutIdentity,
) -> dict[str, Any]:
    """Require independently hash-bound proof of exhaustive review and job completion."""

    review = _object(holdout.payload.get("review_completion"), "holdout review_completion")
    if set(review) != {
        "completion_receipt",
        "post_snapshot",
        "final_decisions",
        "source_canonical_annotations_sha256",
    }:
        raise FrozenHoldoutError("holdout review_completion has unexpected or missing evidence")
    receipt_file = _validate_managed_review_file(
        holdout.manifest,
        holdout.payload,
        review.get("completion_receipt"),
        location="holdout review completion receipt",
    )
    snapshot_file = _validate_managed_review_file(
        holdout.manifest,
        holdout.payload,
        review.get("post_snapshot"),
        location="holdout review post snapshot",
    )
    decisions_file = _validate_managed_review_file(
        holdout.manifest,
        holdout.payload,
        review.get("final_decisions"),
        location="holdout review final decisions",
    )
    receipt, actual_receipt = _read_json(receipt_file.path, "holdout completion receipt")
    snapshot, actual_snapshot = _read_json(snapshot_file.path, "holdout post snapshot")
    decisions, actual_decisions = _read_json(decisions_file.path, "holdout final decisions")
    if actual_receipt != receipt_file or actual_snapshot != snapshot_file:
        raise FrozenHoldoutError("holdout completion evidence changed during validation")
    if actual_decisions != decisions_file:
        raise FrozenHoldoutError("holdout final decisions changed during validation")

    frame_count = len(holdout.frame_keys)
    annotation_count = len(holdout.ground_truth)
    if receipt.get("schema") != COMPLETION_RECEIPT_SCHEMA:
        raise FrozenHoldoutError("holdout completion receipt schema is unsupported")
    if _integer(receipt.get("task_id"), "completion receipt.task_id") != identity.task_id:
        raise FrozenHoldoutError("holdout completion receipt belongs to another task")
    if _integer(receipt.get("job_id"), "completion receipt.job_id") != identity.job_id:
        raise FrozenHoldoutError("holdout completion receipt belongs to another job")
    if receipt.get("dry_run") is not False or receipt.get("mutation_performed") is not True:
        raise FrozenHoldoutError("holdout completion receipt is not an applied completion")
    if receipt.get("job_state_after") != "completed":
        raise FrozenHoldoutError("holdout CVAT job is not proven completed")
    if _integer(receipt.get("annotation_count"), "completion receipt.annotation_count") != (
        annotation_count
    ):
        raise FrozenHoldoutError("holdout completion annotation count differs from COCO")

    decision_validation = _object(
        receipt.get("decision_validation"), "completion receipt.decision_validation"
    )
    if (
        decision_validation.get("valid") is not True
        or decision_validation.get("mutation_performed") is not False
        or decision_validation.get("scope") != "full_review"
        or _integer(decision_validation.get("task_id"), "completion decision validation.task_id")
        != identity.task_id
        or _integer(
            decision_validation.get("snapshot_frame_count"),
            "completion decision validation.snapshot_frame_count",
        )
        != frame_count
        or _integer(
            decision_validation.get("reviewed_frame_count"),
            "completion decision validation.reviewed_frame_count",
        )
        != frame_count
        or _integer(
            decision_validation.get("unresolved_manual_flag_count"),
            "completion decision validation.unresolved_manual_flag_count",
        )
        != 0
    ):
        raise FrozenHoldoutError("completion receipt does not prove exhaustive full review")

    if decisions.get("scope") != "full_review" or decisions.get("mutation_performed") is not False:
        raise FrozenHoldoutError("holdout decisions are not immutable full-review decisions")
    if _integer(decisions.get("task_id"), "holdout decisions.task_id") != identity.task_id:
        raise FrozenHoldoutError("holdout decisions belong to another task")
    reviewed_frames: set[int] = set()
    for index, raw_frame in enumerate(
        _list(decisions.get("frame_reviews"), "holdout decisions.frame_reviews")
    ):
        frame = _object(raw_frame, f"holdout decisions.frame_reviews[{index}]")
        frame_number = _integer(
            frame.get("frame"), f"holdout decisions.frame_reviews[{index}].frame", minimum=0
        )
        if frame.get("reviewed") is not True or frame_number in reviewed_frames:
            raise FrozenHoldoutError("holdout decisions do not review each frame exactly once")
        reviewed_frames.add(frame_number)
    if reviewed_frames != set(range(frame_count)):
        raise FrozenHoldoutError("holdout decisions do not cover the complete CVAT frame span")

    if snapshot.get("snapshot_schema") != CVAT_SNAPSHOT_SCHEMA:
        raise FrozenHoldoutError("holdout post snapshot schema is unsupported")
    snapshot_task = _object(snapshot.get("task"), "holdout post snapshot.task")
    if (
        _integer(snapshot_task.get("id"), "holdout post snapshot.task.id") != identity.task_id
        or _integer(snapshot_task.get("size"), "holdout post snapshot.task.size") != frame_count
    ):
        raise FrozenHoldoutError("holdout post snapshot task identity or size is inconsistent")
    jobs = _list(snapshot.get("jobs"), "holdout post snapshot.jobs")
    if len(jobs) != 1:
        raise FrozenHoldoutError("holdout post snapshot must contain exactly one job")
    job = _object(jobs[0], "holdout post snapshot.jobs[0]")
    issues = _object(job.get("issues"), "holdout post snapshot job.issues")
    if (
        _integer(job.get("id"), "holdout post snapshot job.id") != identity.job_id
        or _integer(job.get("task_id"), "holdout post snapshot job.task_id") != identity.task_id
        or _integer(job.get("start_frame"), "holdout post snapshot job.start_frame") != 0
        or _integer(job.get("stop_frame"), "holdout post snapshot job.stop_frame")
        != frame_count - 1
        or _integer(job.get("frame_count"), "holdout post snapshot job.frame_count") != frame_count
        or job.get("type") != "annotation"
        or job.get("parent_job_id") is not None
        or _integer(issues.get("count"), "holdout post snapshot job.issues.count") != 0
    ):
        raise FrozenHoldoutError("holdout post snapshot job is not a complete clean annotation job")
    final_gate = _object(snapshot.get("final_gate"), "holdout post snapshot.final_gate")
    if final_gate.get("passed") is not True or _list(
        final_gate.get("blocking_reasons"), "holdout post snapshot blocking reasons"
    ):
        raise FrozenHoldoutError("holdout post snapshot final gate did not pass")

    source_annotation_sha = _sha256(
        review.get("source_canonical_annotations_sha256"),
        "holdout review source canonical annotations SHA",
    )
    if (
        _sha256(
            snapshot.get("canonical_annotations_sha256"),
            "holdout post snapshot canonical annotations SHA",
        )
        != source_annotation_sha
    ):
        raise FrozenHoldoutError("holdout manifest is not bound to the post snapshot annotations")
    if (
        _sha256(
            receipt.get("verified_post_completion_canonical_annotations_sha256"),
            "completion receipt post-completion annotations SHA",
        )
        != source_annotation_sha
    ):
        raise FrozenHoldoutError(
            "completion receipt is not bound to the holdout annotations source"
        )

    receipt_evidence = _object(receipt.get("evidence"), "completion receipt.evidence")
    for name, evidence_file in (
        ("post_snapshot", snapshot_file),
        ("decisions", decisions_file),
    ):
        evidence = _object(receipt_evidence.get(name), f"completion receipt.evidence.{name}")
        if (
            _sha256(evidence.get("sha256"), f"completion receipt.evidence.{name}.sha256")
            != evidence_file.sha256
        ):
            raise FrozenHoldoutError(f"completion receipt is not bound to {name}")

    return {
        "task_id": identity.task_id,
        "job_id": identity.job_id,
        "job_state": "completed",
        "scope": "full_review",
        "reviewed_frame_count": frame_count,
        "annotation_count": annotation_count,
        "completion_receipt_sha256": receipt_file.sha256,
        "post_snapshot_sha256": snapshot_file.sha256,
        "final_decisions_sha256": decisions_file.sha256,
        "source_canonical_annotations_sha256": source_annotation_sha,
    }


def _metric_record(value: Any, location: str) -> dict[str, float]:
    metrics = _object(value, location)
    if set(metrics) != SELECTION_METRICS:
        raise FrozenHoldoutError(f"{location} has unexpected or missing metrics")
    result = {name: _number(metrics[name], f"{location}.{name}") for name in metrics}
    if any(not 0 <= metric <= 1 for metric in result.values()):
        raise FrozenHoldoutError(f"{location} metrics must be in [0, 1]")
    return result


def _validate_candidate_freeze(path: Path, binding: Any) -> CandidateFreeze:
    freeze_root = _leaf_absolute(path)
    try:
        root_stat = os.lstat(freeze_root)
    except OSError as error:
        raise FrozenHoldoutError(f"candidate freeze directory is unavailable: {error}") from error
    if not stat.S_ISDIR(root_stat.st_mode) or stat.S_ISLNK(root_stat.st_mode):
        raise FrozenHoldoutError("candidate freeze must be a non-symlink directory")
    payload, record = _read_json(freeze_root / "receipt.json", "candidate freeze receipt")
    binding_record = _object(binding, "protocol.candidate_freeze")
    if set(binding_record) != {"sha256", "size_bytes"}:
        raise FrozenHoldoutError(
            "protocol.candidate_freeze must contain exactly sha256 and size_bytes"
        )
    if record.sha256 != _sha256(binding_record.get("sha256"), "candidate freeze SHA-256"):
        raise FrozenHoldoutError("candidate freeze record SHA-256 differs from the protocol")
    if record.size_bytes != _integer(
        binding_record.get("size_bytes"), "candidate freeze size", minimum=1
    ):
        raise FrozenHoldoutError("candidate freeze record size differs from the protocol")
    if payload.get("schema") != FREEZE_SCHEMA:
        raise FrozenHoldoutError("candidate freeze schema is unsupported")
    if set(payload) != {
        "schema",
        "selection_order",
        "selected_seed",
        "selected_source",
        "selected_metrics",
        "selected_weight",
        "candidate_rankings",
        "contracts",
        "holdout_input_read",
        "holdout_statement",
    }:
        raise FrozenHoldoutError("candidate freeze has unexpected or missing keys")
    if payload.get("selection_order") != list(SELECTION_ORDER):
        raise FrozenHoldoutError("candidate freeze selection order is not fixed")
    selected_seed = _integer(payload.get("selected_seed"), "candidate freeze.selected_seed")
    _metric_record(payload.get("selected_metrics"), "candidate freeze.selected_metrics")
    if payload.get("holdout_input_read") is not False:
        raise FrozenHoldoutError("candidate freeze must record that holdout input was not read")
    if payload.get("holdout_statement") != NO_HOLDOUT_STATEMENT:
        raise FrozenHoldoutError("candidate freeze has an invalid no-holdout statement")

    selected_source = _object(payload.get("selected_source"), "candidate freeze.selected_source")
    if set(selected_source) != {"kind", "directory_name", "seed", "receipt"}:
        raise FrozenHoldoutError("candidate freeze.selected_source has unexpected or missing keys")
    if selected_source.get("kind") != "content_bound_candidate_directory":
        raise FrozenHoldoutError("candidate freeze selected source kind is unsupported")
    _text(selected_source.get("directory_name"), "candidate freeze selected directory")
    if _integer(selected_source.get("seed"), "candidate freeze selected source seed") != (
        selected_seed
    ):
        raise FrozenHoldoutError("candidate freeze selected source seed is inconsistent")
    source_receipt = _object(
        selected_source.get("receipt"), "candidate freeze.selected_source.receipt"
    )
    if set(source_receipt) != {"path", "sha256", "size_bytes"}:
        raise FrozenHoldoutError("candidate freeze selected source receipt is malformed")
    if source_receipt.get("path") != "receipt.json":
        raise FrozenHoldoutError("candidate freeze selected source receipt path is not canonical")
    _sha256(source_receipt.get("sha256"), "candidate freeze selected source receipt SHA")
    _integer(
        source_receipt.get("size_bytes"),
        "candidate freeze selected source receipt size",
        minimum=1,
    )

    weight = _validate_bound_file(
        payload.get("selected_weight"),
        owner=record.path,
        location="candidate freeze.selected_weight",
    )
    if _object(payload.get("selected_weight"), "selected weight").get("path") != "best.pt":
        raise FrozenHoldoutError("candidate freeze selected weight path must be best.pt")

    rankings = _list(payload.get("candidate_rankings"), "candidate freeze.candidate_rankings")
    if len(rankings) != 3:
        raise FrozenHoldoutError("candidate freeze must contain exactly three candidate rankings")
    ranking_seeds: set[int] = set()
    validated_rankings: list[tuple[dict[str, Any], dict[str, float]]] = []
    for index, raw_ranking in enumerate(rankings):
        ranking = _object(raw_ranking, f"candidate freeze.candidate_rankings[{index}]")
        if set(ranking) != {"rank", "seed", "metrics", "source", "best_weights", "contract"}:
            raise FrozenHoldoutError("candidate ranking has unexpected or missing keys")
        if _integer(ranking.get("rank"), f"candidate ranking {index}.rank") != index + 1:
            raise FrozenHoldoutError("candidate ranking order is inconsistent")
        seed = _integer(ranking.get("seed"), f"candidate ranking {index}.seed")
        if seed in ranking_seeds:
            raise FrozenHoldoutError("candidate ranking seeds must be distinct")
        ranking_seeds.add(seed)
        ranking_metrics = _metric_record(
            ranking.get("metrics"), f"candidate ranking {index}.metrics"
        )
        ranking_source = _object(ranking.get("source"), f"candidate ranking {index}.source")
        if set(ranking_source) != {"kind", "directory_name", "seed", "receipt"}:
            raise FrozenHoldoutError("candidate ranking source is malformed")
        if (
            ranking_source.get("kind") != "content_bound_candidate_directory"
            or ranking_source.get("seed") != seed
        ):
            raise FrozenHoldoutError("candidate ranking source identity is inconsistent")
        ranking_weight = _object(
            ranking.get("best_weights"), f"candidate ranking {index}.best_weights"
        )
        if set(ranking_weight) != {"path", "sha256", "size_bytes"}:
            raise FrozenHoldoutError("candidate ranking weight binding is malformed")
        _safe_relative_path(ranking_weight.get("path"), f"candidate ranking {index} weight path")
        _sha256(ranking_weight.get("sha256"), f"candidate ranking {index} weight SHA")
        _integer(
            ranking_weight.get("size_bytes"),
            f"candidate ranking {index} weight size",
            minimum=1,
        )
        _object(ranking.get("contract"), f"candidate ranking {index}.contract")
        validated_rankings.append((ranking, ranking_metrics))
    if ranking_seeds != EXPECTED_TRAINING_SEEDS:
        raise FrozenHoldoutError(
            "candidate freeze seeds must be exactly "
            f"{sorted(EXPECTED_TRAINING_SEEDS)}; got {sorted(ranking_seeds)}"
        )
    expected_ranking_order = sorted(
        validated_rankings,
        key=lambda item: (
            -item[1]["map50_95"],
            -item[1]["map50"],
            -item[1]["recall"],
            -item[1]["precision"],
            int(item[0]["seed"]),
        ),
    )
    if [item[0]["seed"] for item in validated_rankings] != [
        item[0]["seed"] for item in expected_ranking_order
    ]:
        raise FrozenHoldoutError(
            "candidate freeze rankings do not follow the fixed selection order"
        )
    first_ranking = _object(rankings[0], "candidate freeze.candidate_rankings[0]")
    if first_ranking.get("seed") != selected_seed:
        raise FrozenHoldoutError("candidate freeze selected seed is not rank one")
    if first_ranking.get("source") != selected_source:
        raise FrozenHoldoutError("candidate freeze selected source is not rank one")
    if first_ranking.get("metrics") != payload.get("selected_metrics"):
        raise FrozenHoldoutError("candidate freeze selected metrics are not rank one")
    first_weight = _object(first_ranking.get("best_weights"), "rank-one best weights")
    if _sha256(first_weight.get("sha256"), "rank-one weight SHA") != weight.sha256:
        raise FrozenHoldoutError("candidate freeze copied weight differs from rank-one weight")
    if _integer(first_weight.get("size_bytes"), "rank-one weight size", minimum=1) != (
        weight.size_bytes
    ):
        raise FrozenHoldoutError("candidate freeze copied weight size differs from rank one")

    contracts = _object(payload.get("contracts"), "candidate freeze.contracts")
    if set(contracts) != {
        "candidate_count",
        "protocol",
        "protocol_sha256",
        "resolved_args_except_seed",
        "dataset",
        "base_weights",
        "taxonomy",
    }:
        raise FrozenHoldoutError("candidate freeze contracts have unexpected or missing keys")
    if _integer(contracts.get("candidate_count"), "candidate freeze candidate_count") != 3:
        raise FrozenHoldoutError("candidate freeze candidate_count must be three")
    if contracts.get("taxonomy") != list(CANONICAL_LABELS):
        raise FrozenHoldoutError("candidate freeze taxonomy is not the canonical eight classes")
    training_protocol = _object(contracts.get("protocol"), "candidate freeze training protocol")
    if set(training_protocol) != {
        "schema",
        "model_family",
        "taxonomy",
        "training",
        "validation_selection",
        "holdout_access",
    }:
        raise FrozenHoldoutError("candidate training protocol is malformed")
    if (
        training_protocol.get("schema") != TRAINING_PROTOCOL_SCHEMA
        or training_protocol.get("model_family") != "YOLO11n"
        or training_protocol.get("taxonomy") != list(CANONICAL_LABELS)
    ):
        raise FrozenHoldoutError("candidate training protocol model or taxonomy is unsupported")
    _object(training_protocol.get("training"), "candidate training settings")
    if training_protocol.get("validation_selection") != {
        "primary": "mAP50-95",
        "tie_breakers": ["mAP50", "recall", "precision", "smaller_seed"],
    }:
        raise FrozenHoldoutError("candidate training validation selection is not fixed")
    if training_protocol.get("holdout_access") != "prohibited":
        raise FrozenHoldoutError("candidate training protocol did not prohibit holdout access")
    if canonical_sha256(training_protocol) != _sha256(
        contracts.get("protocol_sha256"), "candidate freeze training protocol SHA"
    ):
        raise FrozenHoldoutError("candidate freeze training protocol SHA is inconsistent")
    resolved_args = _object(
        contracts.get("resolved_args_except_seed"), "candidate freeze resolved_args_except_seed"
    )
    if "seed" in resolved_args:
        raise FrozenHoldoutError("candidate freeze resolved_args_except_seed still contains seed")
    dataset = _object(contracts.get("dataset"), "candidate freeze contracts.dataset")
    if set(dataset) != {
        "manifest",
        "dataset_yaml",
        "managed_files_sha256",
        "managed_file_count",
        "counts",
    }:
        raise FrozenHoldoutError("candidate freeze dataset contract is malformed")
    dataset_manifest = _object(dataset.get("manifest"), "candidate freeze dataset.manifest")
    if set(dataset_manifest) != {"schema", "sha256", "size_bytes"}:
        raise FrozenHoldoutError("candidate freeze dataset manifest binding is malformed")
    if dataset_manifest.get("schema") != YOLO_DATASET_SCHEMA:
        raise FrozenHoldoutError("candidate freeze dataset schema is unsupported")
    dataset_sha = _sha256(dataset_manifest.get("sha256"), "candidate freeze dataset SHA")
    dataset_size = _integer(
        dataset_manifest.get("size_bytes"), "candidate freeze dataset size", minimum=1
    )
    dataset_yaml = _object(dataset.get("dataset_yaml"), "candidate freeze dataset_yaml")
    if set(dataset_yaml) != {"sha256", "size_bytes"}:
        raise FrozenHoldoutError("candidate freeze dataset_yaml binding is malformed")
    _sha256(dataset_yaml.get("sha256"), "candidate freeze dataset_yaml SHA")
    _integer(dataset_yaml.get("size_bytes"), "candidate freeze dataset_yaml size", minimum=1)
    _sha256(dataset.get("managed_files_sha256"), "candidate freeze managed-files SHA")
    _integer(dataset.get("managed_file_count"), "candidate freeze managed-file count", minimum=1)
    _object(dataset.get("counts"), "candidate freeze dataset counts")
    base_weights = _object(contracts.get("base_weights"), "candidate freeze base_weights")
    if set(base_weights) != {"file_name", "model_family", "sha256", "size_bytes"}:
        raise FrozenHoldoutError("candidate freeze base weight contract is malformed")
    if base_weights.get("file_name") != "yolo11n.pt" or base_weights.get("model_family") != (
        "YOLO11n"
    ):
        raise FrozenHoldoutError("candidate freeze base weight is not YOLO11n")
    base_sha = _sha256(base_weights.get("sha256"), "candidate freeze base weight SHA")
    base_size = _integer(
        base_weights.get("size_bytes"), "candidate freeze base weight size", minimum=1
    )
    for index, (ranking, _metrics) in enumerate(validated_rankings):
        contract = _object(ranking.get("contract"), f"candidate ranking {index}.contract")
        if set(contract) != {
            "protocol_sha256",
            "dataset",
            "base_weights",
            "taxonomy",
        }:
            raise FrozenHoldoutError("candidate ranking contract is malformed")
        if contract != {
            "protocol_sha256": contracts["protocol_sha256"],
            "dataset": dataset,
            "base_weights": base_weights,
            "taxonomy": list(CANONICAL_LABELS),
        }:
            raise FrozenHoldoutError("candidate rankings do not share one frozen contract")
    candidate_id = f"seed-{selected_seed}-{weight.sha256[:12]}"
    return CandidateFreeze(
        record,
        candidate_id,
        weight,
        selected_seed,
        dataset_sha,
        dataset_size,
        base_sha,
        base_size,
    )


def _universe_digest(values: Iterable[Any]) -> str:
    normalized = sorted(values, key=lambda value: _canonical_bytes(value))
    return canonical_sha256(normalized)


def _validate_overlap_evidence(
    evidence: BoundFile,
    *,
    training: ReferenceData,
    holdout: ReferenceData,
) -> dict[str, Any]:
    payload, actual = _read_json(evidence.path, "overlap evidence")
    if actual != evidence:
        raise FrozenHoldoutError("overlap evidence changed while being validated")
    if payload.get("schema") != OVERLAP_SCHEMA:
        raise FrozenHoldoutError("overlap evidence schema is unsupported")
    if set(payload) != {
        "schema",
        "training_reference_manifest_sha256",
        "holdout_manifest_sha256",
        "computed",
        "gate_result",
    }:
        raise FrozenHoldoutError("overlap evidence has unexpected or missing keys")
    if payload.get("gate_result") != "PASS":
        raise FrozenHoldoutError("overlap evidence gate_result must be PASS")
    if (
        _sha256(
            payload.get("training_reference_manifest_sha256"),
            "overlap evidence.training_reference_manifest_sha256",
        )
        != training.manifest.sha256
    ):
        raise FrozenHoldoutError("overlap evidence is bound to another training reference")
    if (
        _sha256(
            payload.get("holdout_manifest_sha256"),
            "overlap evidence.holdout_manifest_sha256",
        )
        != holdout.manifest.sha256
    ):
        raise FrozenHoldoutError("overlap evidence is bound to another holdout")

    asset_overlap = training.asset_ids & holdout.asset_ids
    image_hash_overlap = training.image_hashes & holdout.image_hashes
    source_overlap = training.source_hashes & holdout.source_hashes
    frame_overlap = training.source_frame_keys & holdout.source_frame_keys
    computed = {
        "training_asset_ids_sha256": _universe_digest(training.asset_ids),
        "holdout_asset_ids_sha256": _universe_digest(holdout.asset_ids),
        "training_image_sha256s_sha256": _universe_digest(training.image_hashes),
        "holdout_image_sha256s_sha256": _universe_digest(holdout.image_hashes),
        "training_source_sha256s_sha256": _universe_digest(training.source_hashes),
        "holdout_source_sha256s_sha256": _universe_digest(holdout.source_hashes),
        "training_frame_keys_sha256": _universe_digest(training.source_frame_keys),
        "holdout_frame_keys_sha256": _universe_digest(holdout.source_frame_keys),
        "asset_id_overlap_count": len(asset_overlap),
        "image_sha256_overlap_count": len(image_hash_overlap),
        "source_sha256_overlap_count": len(source_overlap),
        "frame_overlap_count": len(frame_overlap),
    }
    if _object(payload.get("computed"), "overlap evidence.computed") != computed:
        raise FrozenHoldoutError("overlap evidence does not match independently computed universes")
    if asset_overlap or image_hash_overlap or source_overlap or frame_overlap:
        raise FrozenHoldoutError("training and holdout data overlap")
    return computed


def _validate_settings(value: Any) -> dict[str, Any]:
    settings = _object(value, "protocol.settings")
    expected_keys = {
        "confidence",
        "image_size",
        "device",
        "nms_iou",
        "rider_overlap",
        "match_iou",
    }
    if set(settings) != expected_keys:
        raise FrozenHoldoutError("protocol.settings has unexpected or missing keys")
    confidence = _number(settings.get("confidence"), "protocol.settings.confidence")
    nms_iou = _number(settings.get("nms_iou"), "protocol.settings.nms_iou")
    rider_overlap = _number(settings.get("rider_overlap"), "protocol.settings.rider_overlap")
    match_iou = _number(settings.get("match_iou"), "protocol.settings.match_iou")
    if not 0 < confidence <= 1 or not 0 < nms_iou <= 1 or not 0 < rider_overlap <= 1:
        raise FrozenHoldoutError("protocol confidence and overlap thresholds must be in (0, 1]")
    if match_iou != EXPECTED_GATES["match_iou"]:
        raise FrozenHoldoutError("protocol match_iou must be fixed at 0.50")
    return {
        "confidence": confidence,
        "image_size": _integer(
            settings.get("image_size"), "protocol.settings.image_size", minimum=1
        ),
        "device": _text(settings.get("device"), "protocol.settings.device"),
        "nms_iou": nms_iou,
        "rider_overlap": rider_overlap,
        "match_iou": match_iou,
    }


def _validate_protocol_frames(
    raw_frames: Any,
    expected_sha256: Any,
    holdout: ReferenceData,
) -> tuple[tuple[str, int], ...]:
    frames: list[tuple[str, int]] = []
    canonical_records: list[dict[str, Any]] = []
    for index, raw_frame in enumerate(_list(raw_frames, "protocol.holdout.evaluation_frames")):
        frame = _object(raw_frame, f"protocol.holdout.evaluation_frames[{index}]")
        if set(frame) != {"scene_id", "source_frame"}:
            raise FrozenHoldoutError("protocol evaluation frame has unexpected or missing keys")
        scene_id = _text(frame.get("scene_id"), f"evaluation frame {index}.scene_id")
        source_frame = _integer(
            frame.get("source_frame"), f"evaluation frame {index}.source_frame", minimum=0
        )
        frames.append((scene_id, source_frame))
        canonical_records.append({"scene_id": scene_id, "source_frame": source_frame})
    if len(set(frames)) != len(frames):
        raise FrozenHoldoutError("protocol evaluation frame universe contains duplicates")
    if frames != sorted(frames):
        raise FrozenHoldoutError("protocol evaluation frame universe must be sorted")
    if canonical_sha256(canonical_records) != _sha256(
        expected_sha256, "protocol.holdout.evaluation_frames_sha256"
    ):
        raise FrozenHoldoutError("protocol evaluation frame universe hash is inconsistent")
    if set(frames) != set(holdout.frame_keys) or len(frames) != len(holdout.frame_keys):
        raise FrozenHoldoutError("protocol evaluation frame universe differs from holdout COCO")
    return tuple(frames)


def validate_protocol(
    protocol_path: Path,
    candidate_freeze_path: Path,
    *,
    final_holdout_task_id: int | None = None,
    final_holdout_job_id: int | None = None,
) -> ValidatedProtocol:
    """Validate every immutable input without loading a model or touching CVAT."""

    try:
        identity = resolve_final_holdout_identity(
            task_id=final_holdout_task_id,
            job_id=final_holdout_job_id,
        )
    except FinalHoldoutConfigError as error:
        raise FrozenHoldoutError(str(error)) from error
    protocol_payload, protocol_file = _read_json(protocol_path, "frozen holdout protocol")
    if protocol_payload.get("schema") != PROTOCOL_SCHEMA:
        raise FrozenHoldoutError("frozen holdout protocol schema is unsupported")
    if set(protocol_payload) != {
        "schema",
        "protocol_id",
        "mode",
        "candidate_freeze",
        "training_dataset_manifest",
        "training_reference_manifest",
        "baseline",
        "holdout",
        "overlap_evidence",
        "warmup_image",
        "settings",
        "gates",
    }:
        raise FrozenHoldoutError("frozen holdout protocol has unexpected or missing keys")
    protocol_id = _text(protocol_payload.get("protocol_id"), "protocol.protocol_id")
    if protocol_payload.get("mode") != "production_scene_videos":
        raise FrozenHoldoutError("protocol mode must be production_scene_videos")
    if _object(protocol_payload.get("gates"), "protocol.gates") != EXPECTED_GATES:
        raise FrozenHoldoutError("protocol promotion gates differ from the fixed gates")
    settings = _validate_settings(protocol_payload.get("settings"))

    candidate = _validate_candidate_freeze(
        _leaf_absolute(candidate_freeze_path), protocol_payload.get("candidate_freeze")
    )
    dataset_manifest = _validate_bound_file(
        protocol_payload.get("training_dataset_manifest"),
        owner=protocol_file.path,
        location="protocol.training_dataset_manifest",
    )
    if (
        dataset_manifest.sha256 != candidate.dataset_manifest_sha256
        or dataset_manifest.size_bytes != candidate.dataset_manifest_size_bytes
    ):
        raise FrozenHoldoutError(
            "training dataset manifest differs from the candidate freeze contract"
        )
    dataset_payload, dataset_actual = _read_json(dataset_manifest.path, "training dataset manifest")
    if dataset_actual != dataset_manifest:
        raise FrozenHoldoutError("training dataset manifest changed while being validated")
    if dataset_payload.get("schema") != YOLO_DATASET_SCHEMA:
        raise FrozenHoldoutError("training dataset manifest schema is unsupported")
    dataset_gate = _object(dataset_payload.get("gate"), "training dataset manifest.gate")
    if dataset_gate.get("passed") is not True or not all(
        value is True
        for value in _object(dataset_gate.get("checks"), "training dataset gate.checks").values()
    ):
        raise FrozenHoldoutError("training dataset manifest gate has not passed")
    dataset_inputs = _object(dataset_payload.get("inputs"), "training dataset manifest.inputs")
    reference_input = _object(
        dataset_inputs.get("reference_manifest"),
        "training dataset manifest.inputs.reference_manifest",
    )
    if reference_input.get("schema") != TRAINING_SCHEMA:
        raise FrozenHoldoutError("training dataset names an unsupported COCO reference schema")
    reference_manifest = _validate_bound_file(
        protocol_payload.get("training_reference_manifest"),
        owner=protocol_file.path,
        location="protocol.training_reference_manifest",
    )
    if reference_manifest.sha256 != _sha256(
        reference_input.get("sha256"), "training dataset reference manifest SHA"
    ):
        raise FrozenHoldoutError(
            "training reference manifest differs from the dataset input contract"
        )
    training_reference = _build_reference_data(
        reference_manifest,
        expected_schema=TRAINING_SCHEMA,
        annotations_record=None,
        location="training reference manifest",
        verify_all_files=False,
    )
    coco_input = _object(dataset_inputs.get("coco"), "training dataset manifest.inputs.coco")
    if training_reference.annotations.sha256 != _sha256(
        coco_input.get("sha256"), "training dataset input COCO SHA"
    ):
        raise FrozenHoldoutError("training reference COCO differs from the dataset input")
    source_statistics = _object(
        dataset_payload.get("source_statistics"), "training dataset source_statistics"
    )
    dataset_assets: set[tuple[str, str]] = set()
    dataset_sources: set[str] = set()
    for index, raw_asset in enumerate(
        _list(source_statistics.get("assets"), "training dataset source_statistics.assets")
    ):
        asset = _object(raw_asset, f"training dataset source asset {index}")
        dataset_assets.add(_typed_asset(asset.get("asset_id"), f"training source asset {index}"))
        leakage = _text(
            asset.get("leakage_group_id"), f"training source asset {index}.leakage_group_id"
        )
        if not leakage.startswith("sha256:"):
            raise FrozenHoldoutError("training dataset source identity is not content-based")
        dataset_sources.add(
            _sha256(leakage.removeprefix("sha256:"), f"training source asset {index} SHA")
        )
    if dataset_assets != set(training_reference.asset_ids) or dataset_sources != set(
        training_reference.source_hashes
    ):
        raise FrozenHoldoutError(
            "training dataset source identities differ from the COCO training reference"
        )
    baseline = _object(protocol_payload.get("baseline"), "protocol.baseline")
    if set(baseline) != {"model_id", "weight"}:
        raise FrozenHoldoutError("protocol.baseline must contain model_id and weight")
    _text(baseline.get("model_id"), "protocol.baseline.model_id")
    baseline_weight = _validate_bound_file(
        baseline.get("weight"), owner=protocol_file.path, location="protocol.baseline.weight"
    )
    if (
        baseline_weight.sha256 != candidate.base_weight_sha256
        or baseline_weight.size_bytes != candidate.base_weight_size_bytes
    ):
        raise FrozenHoldoutError("baseline weight differs from the candidate training base weight")
    if baseline_weight.sha256 == candidate.weight.sha256:
        raise FrozenHoldoutError("baseline and candidate weights must be different")

    holdout_record = _object(protocol_payload.get("holdout"), "protocol.holdout")
    if set(holdout_record) != {
        "task_id",
        "job_id",
        "manifest",
        "annotations",
        "evaluation_frames",
        "evaluation_frames_sha256",
        "source_videos",
    }:
        raise FrozenHoldoutError("protocol.holdout has unexpected or missing keys")
    if (
        _integer(holdout_record.get("task_id"), "protocol.holdout.task_id")
        != identity.task_id
        or _integer(holdout_record.get("job_id"), "protocol.holdout.job_id")
        != identity.job_id
    ):
        raise FrozenHoldoutError(
            "protocol holdout identity differs from the configured final holdout"
        )
    manifest_file = _validate_bound_file(
        holdout_record.get("manifest"),
        owner=protocol_file.path,
        location="protocol.holdout.manifest",
    )
    annotations_file = _validate_bound_file(
        holdout_record.get("annotations"),
        owner=protocol_file.path,
        location="protocol.holdout.annotations",
    )
    holdout = _build_reference_data(
        manifest_file,
        expected_schema=HOLDOUT_SCHEMA,
        annotations_record=annotations_file,
        location="holdout manifest",
        verify_all_files=True,
    )
    if _integer(holdout.payload.get("task_id"), "holdout manifest.task_id") != identity.task_id:
        raise FrozenHoldoutError("holdout manifest is not bound to the configured final holdout")
    review_completion = _validate_full_review_completion(holdout, identity)
    frames = _validate_protocol_frames(
        holdout_record.get("evaluation_frames"),
        holdout_record.get("evaluation_frames_sha256"),
        holdout,
    )

    raw_videos = _list(holdout_record.get("source_videos"), "protocol.holdout.source_videos")
    videos: list[SourceVideo] = []
    for index, raw_video in enumerate(raw_videos):
        video = _object(raw_video, f"protocol.holdout.source_videos[{index}]")
        if set(video) != {"scene_id", "frame_step", "file"}:
            raise FrozenHoldoutError("protocol source video has unexpected or missing keys")
        scene_id = _text(video.get("scene_id"), f"source video {index}.scene_id")
        frame_step = _integer(
            video.get("frame_step"), f"source video {index}.frame_step", minimum=1
        )
        videos.append(
            SourceVideo(
                scene_id,
                frame_step,
                _validate_bound_file(
                    video.get("file"),
                    owner=protocol_file.path,
                    location=f"protocol.holdout.source_videos[{index}].file",
                ),
            )
        )
    scenes = {scene for scene, _ in frames}
    if len({video.scene_id for video in videos}) != len(videos):
        raise FrozenHoldoutError("protocol source videos contain duplicate scene IDs")
    if {video.scene_id for video in videos} != scenes:
        raise FrozenHoldoutError("protocol source videos do not exactly cover evaluation scenes")
    if {video.file.sha256 for video in videos} != set(holdout.source_hashes):
        raise FrozenHoldoutError(
            "protocol source video hashes do not match the holdout source identities"
        )
    for video in videos:
        scene_sources = holdout.source_shas_by_scene[video.scene_id]
        if len(scene_sources) != 1:
            raise FrozenHoldoutError(
                f"holdout scene {video.scene_id!r} must map to exactly one source identity"
            )
        if video.file.sha256 != next(iter(scene_sources)):
            raise FrozenHoldoutError(
                f"protocol source video for scene {video.scene_id!r} does not match "
                "that scene's holdout source identity"
            )
    step_by_scene = {video.scene_id: video.frame_step for video in videos}
    for scene_id, frame in frames:
        stride = step_by_scene[scene_id]
        if (frame + 1) % stride or result_frame_index((frame + 1) // stride - 1, stride) != frame:
            raise FrozenHoldoutError(
                f"evaluation frame {scene_id}/{frame} is unreachable with vid_stride={stride}"
            )

    overlap_file = _validate_bound_file(
        protocol_payload.get("overlap_evidence"),
        owner=protocol_file.path,
        location="protocol.overlap_evidence",
    )
    overlap_computed = _validate_overlap_evidence(
        overlap_file, training=training_reference, holdout=holdout
    )

    warmup = _validate_bound_file(
        protocol_payload.get("warmup_image"),
        owner=protocol_file.path,
        location="protocol.warmup_image",
    )
    if warmup.path.is_relative_to(holdout.manifest.path.parent.resolve()):
        raise FrozenHoldoutError("warmup image must be outside the holdout reference directory")
    forbidden_hashes = set(holdout.image_hashes) | {video.file.sha256 for video in videos}
    if warmup.sha256 in forbidden_hashes:
        raise FrozenHoldoutError("warmup image content must not come from the holdout")

    batch_record = {
        "mode": "production_scene_videos",
        "evaluation_frames": [
            {"scene_id": scene_id, "source_frame": frame} for scene_id, frame in frames
        ],
        "source_videos": [
            {
                "scene_id": video.scene_id,
                "frame_step": video.frame_step,
                "sha256": video.file.sha256,
                "size_bytes": video.file.size_bytes,
            }
            for video in videos
        ],
        "settings": settings,
        "frame_mapping": "roadlabelops.tools.detection.result_frame_index",
    }
    return ValidatedProtocol(
        protocol=protocol_file,
        protocol_id=protocol_id,
        candidate=candidate,
        training_reference=training_reference,
        baseline_weight=baseline_weight,
        holdout=holdout,
        overlap_evidence=overlap_file,
        warmup_image=warmup,
        source_videos=tuple(videos),
        settings=settings,
        batch_sha256=canonical_sha256(batch_record),
        validation={
            "task_id": identity.task_id,
            "job_id": identity.job_id,
            "evaluation_frame_count": len(frames),
            "zero_annotation_frame_count": holdout.zero_annotation_frame_count,
            "ground_truth_annotation_count": len(holdout.ground_truth),
            "canonical_labels": list(CANONICAL_LABELS),
            "managed_holdout_file_count": len(_list(holdout.payload.get("files"), "files")),
            "review_completion": review_completion,
            "overlap": overlap_computed,
            "frame_mapping": "roadlabelops.tools.detection.result_frame_index",
        },
    )


def build_preflight(validated: ValidatedProtocol, *, output: Path) -> dict[str, Any]:
    output_path = _leaf_absolute(output)
    claim_path = _consumption_claim_path(validated)
    consumed = os.path.lexists(output_path) or os.path.lexists(claim_path)
    return {
        "schema": PREFLIGHT_SCHEMA,
        "dry_run": True,
        "mutation_performed": False,
        "inference_performed": False,
        "protocol_id": validated.protocol_id,
        "protocol_sha256": validated.protocol.sha256,
        "candidate_id": validated.candidate.candidate_id,
        "candidate_freeze_sha256": validated.candidate.record.sha256,
        "training_dataset_manifest_sha256": validated.candidate.dataset_manifest_sha256,
        "training_reference_manifest_sha256": validated.training_reference.manifest.sha256,
        "baseline_weight_sha256": validated.baseline_weight.sha256,
        "candidate_weight_sha256": validated.candidate.weight.sha256,
        "holdout_manifest_sha256": validated.holdout.manifest.sha256,
        "holdout_annotations_sha256": validated.holdout.annotations.sha256,
        "overlap_evidence_sha256": validated.overlap_evidence.sha256,
        "batch_sha256": validated.batch_sha256,
        "validation": dict(validated.validation),
        "single_use": {
            "consumed": consumed,
            "output_exists": os.path.lexists(output_path),
            "claim_exists": os.path.lexists(claim_path),
        },
        "gate_result": "FAIL" if consumed else "PASS",
    }


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True).encode(
            "utf-8"
        )
        + b"\n"
    )


def _atomic_write_json_new(output: Path, payload: Mapping[str, Any]) -> None:
    output = _leaf_absolute(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(output):
        raise FrozenHoldoutError(f"output already exists: {output}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(_json_bytes(payload))
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError as error:
            raise FrozenHoldoutError(f"output already exists: {output}") from error
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _consumption_claim_path(validated: ValidatedProtocol) -> Path:
    """Return a stable claim path that cannot be bypassed with another output name."""

    registry = validated.holdout.manifest.path.parent
    return registry / (
        f".roadlabelops-frozen-holdout-{validated.holdout.manifest.sha256}.consumed.json"
    )


def _claim_once(output: Path, validated: ValidatedProtocol) -> Path:
    output = _leaf_absolute(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    claim = _consumption_claim_path(validated)
    if os.path.lexists(output):
        raise FrozenHoldoutError(f"evaluation output already exists: {output}")
    payload = {
        "schema": CLAIM_SCHEMA,
        "protocol_id": validated.protocol_id,
        "protocol_sha256": validated.protocol.sha256,
        "candidate_freeze_sha256": validated.candidate.record.sha256,
        "candidate_weight_sha256": validated.candidate.weight.sha256,
        "training_dataset_manifest_sha256": validated.candidate.dataset_manifest_sha256,
        "training_reference_manifest_sha256": validated.training_reference.manifest.sha256,
        "holdout_annotations_sha256": validated.holdout.annotations.sha256,
        "batch_sha256": validated.batch_sha256,
        "claimed_at": datetime.now(timezone.utc).isoformat(),
        "warning": "Do not delete: this claim records consumption of the frozen holdout.",
    }
    _atomic_write_json_new(claim, payload)
    if os.path.lexists(output):
        raise FrozenHoldoutError(f"evaluation output appeared concurrently: {output}")
    return claim


def _copy_frozen(source: BoundFile, destination: Path, location: str) -> BoundFile:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(destination):
        raise FrozenHoldoutError(f"frozen input destination already exists: {destination}")
    source_fd = os.open(
        source.path,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
    )
    try:
        source_stat = os.fstat(source_fd)
        if not stat.S_ISREG(source_stat.st_mode):
            raise FrozenHoldoutError(f"{location} must remain a regular file")
        with destination.open("xb") as target:
            digest = hashlib.sha256()
            size = 0
            while True:
                chunk = os.read(source_fd, 1024 * 1024)
                if not chunk:
                    break
                target.write(chunk)
                digest.update(chunk)
                size += len(chunk)
            target.flush()
            os.fsync(target.fileno())
    finally:
        os.close(source_fd)
    if digest.hexdigest() != source.sha256 or size != source.size_bytes:
        raise FrozenHoldoutError(f"{location} changed while frozen inputs were copied")
    destination.chmod(0o400)
    return BoundFile(destination, source.sha256, source.size_bytes)


def _revalidate_frozen_inputs(files: Iterable[BoundFile]) -> None:
    for index, frozen in enumerate(files):
        _validate_explicit_file(
            frozen.path,
            expected_sha256=frozen.sha256,
            expected_size=frozen.size_bytes,
            location=f"private frozen input {index}",
        )


def _validate_model_names(names: Any, role: str) -> None:
    values = names.values() if isinstance(names, Mapping) else names
    if not isinstance(values, Iterable):
        raise FrozenHoldoutError(f"{role} model does not expose a class-name mapping")
    mapped = {MODEL_MAPPING.get(str(value)) for value in values}
    mapped.discard(None)
    if mapped != set(CANONICAL_LABELS):
        raise FrozenHoldoutError(
            f"{role} model classes do not map onto the canonical eight-class taxonomy"
        )


def ultralytics_runner(
    weight: Path,
    warmup: Path,
    videos: Sequence[SourceVideo],
    settings: Mapping[str, Any],
    role: str,
) -> ModelRun:
    """Run one frozen weight on the protocol's production-video batch."""

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise FrozenHoldoutError(
            "Ultralytics is unavailable; install the detection extra before --apply"
        ) from error

    load_started = time.perf_counter()
    model = YOLO(str(weight))
    load_seconds = time.perf_counter() - load_started
    _validate_model_names(model.names, role)
    model.predict(
        source=str(warmup),
        conf=float(settings["confidence"]),
        imgsz=int(settings["image_size"]),
        device=str(settings["device"]),
        verbose=False,
    )

    raw_predictions: list[dict[str, Any]] = []
    observed: set[tuple[str, int]] = set()
    processed = 0
    inference_started = time.perf_counter()
    model_stem = weight.stem
    for video in videos:
        targets = set(getattr(video, "target_frames", ()))
        # Frozen SourceVideo instances receive target_frames in evaluate() via a
        # private runtime subtype-like attribute wrapper; normal dataclasses stay immutable.
        results = model.predict(
            source=str(video.file.path),
            conf=float(settings["confidence"]),
            imgsz=int(settings["image_size"]),
            device=str(settings["device"]),
            stream=True,
            vid_stride=video.frame_step,
            verbose=False,
        )
        for result_index, result in enumerate(results):
            frame = result_frame_index(result_index, video.frame_step)
            processed += 1
            if (
                targets
                and frame > max(targets)
                and targets <= {item[1] for item in observed if item[0] == video.scene_id}
            ):
                break
            if frame not in targets:
                continue
            frame_key = (video.scene_id, frame)
            if frame_key in observed:
                raise FrozenHoldoutError(f"{role} emitted duplicate result for {frame_key}")
            observed.add(frame_key)
            if result.boxes is None:
                continue
            for box_index, box in enumerate(result.boxes):
                model_label = str(result.names[int(box.cls.item())])
                label = MODEL_MAPPING.get(model_label)
                if label is None:
                    continue
                raw_predictions.append(
                    {
                        "prediction_id": f"{role}_{model_stem}_{video.scene_id}_{frame}_{box_index}",
                        "scene_id": video.scene_id,
                        "frame": frame,
                        "label": label,
                        "confidence": round(float(box.conf.item()), 6),
                        "bbox": [round(float(value), 4) for value in box.xyxy[0].tolist()],
                        "source": "auto",
                    }
                )
    inference_seconds = time.perf_counter() - inference_started
    return ModelRun(
        raw_predictions=tuple(raw_predictions),
        observed_frame_keys=frozenset(observed),
        processed_video_frame_count=processed,
        inference_wall_seconds=inference_seconds,
        model_metadata={"load_seconds": round(load_seconds, 3), "warmup_images": 1},
    )


@dataclass(frozen=True)
class RuntimeSourceVideo(SourceVideo):
    target_frames: tuple[int, ...]


def _model_result(
    role: str,
    run: ModelRun,
    *,
    validated: ValidatedProtocol,
) -> dict[str, Any]:
    expected_frames = frozenset(validated.holdout.frame_keys)
    if run.observed_frame_keys != expected_frames:
        missing = sorted(expected_frames - run.observed_frame_keys)
        unexpected = sorted(run.observed_frame_keys - expected_frames)
        raise FrozenHoldoutError(
            f"{role} did not evaluate the fixed frame universe; "
            f"missing={missing[:3]}, unexpected={unexpected[:3]}"
        )
    if isinstance(run.processed_video_frame_count, bool) or run.processed_video_frame_count < len(
        expected_frames
    ):
        raise FrozenHoldoutError(f"{role} processed frame count is inconsistent")
    prediction_ids: set[str] = set()
    for index, prediction in enumerate(run.raw_predictions):
        prediction_id = _text(prediction.get("prediction_id"), f"{role} prediction {index}.id")
        if prediction_id in prediction_ids:
            raise FrozenHoldoutError(f"{role} emitted duplicate prediction IDs")
        prediction_ids.add(prediction_id)
        scene_id = _text(prediction.get("scene_id"), f"{role} prediction {index}.scene_id")
        frame = _integer(prediction.get("frame"), f"{role} prediction {index}.frame", minimum=0)
        if (scene_id, frame) not in expected_frames:
            raise FrozenHoldoutError(f"{role} prediction {index} is outside the frame universe")
        if prediction.get("label") not in CANONICAL_LABELS:
            raise FrozenHoldoutError(f"{role} prediction {index} has a non-canonical label")
        confidence = _number(prediction.get("confidence"), f"{role} prediction {index}.confidence")
        if not 0 <= confidence <= 1:
            raise FrozenHoldoutError(f"{role} prediction {index} confidence is outside [0, 1]")
        bbox = _list(prediction.get("bbox"), f"{role} prediction {index}.bbox")
        if len(bbox) != 4:
            raise FrozenHoldoutError(f"{role} prediction {index} bbox must contain four numbers")
        left, top, right, bottom = (
            _number(value, f"{role} prediction {index}.bbox[{coordinate}]")
            for coordinate, value in enumerate(bbox)
        )
        if left < 0 or top < 0 or right <= left or bottom <= top:
            raise FrozenHoldoutError(f"{role} prediction {index} bbox is invalid")
    predictions, postprocessing = postprocess_predictions(
        list(run.raw_predictions),
        nms_iou_threshold=float(validated.settings["nms_iou"]),
        rider_overlap_threshold=float(validated.settings["rider_overlap"]),
    )
    quality_result = calculate_quality(
        predictions,
        list(validated.holdout.ground_truth),
        evaluated_frame_keys=expected_frames,
    )
    if not quality_result.ok:
        raise FrozenHoldoutError(f"quality calculation failed for {role}")
    quality = quality_result.data
    if quality.get("evaluated_frame_count") != len(expected_frames):
        raise FrozenHoldoutError(f"quality calculation lost empty frames for {role}")
    observed_per_class = _object(quality.get("per_class"), f"{role} quality.per_class")
    quality["per_class"] = {
        label: observed_per_class.get(
            label,
            {
                "true_positive_count": 0,
                "false_positive_count": 0,
                "false_negative_count": 0,
                "precision": None,
                "recall": None,
                "f1_score": None,
            },
        )
        for label in CANONICAL_LABELS
    }
    observed_distribution = _object(
        quality.get("class_distribution"), f"{role} quality.class_distribution"
    )
    quality["class_distribution"] = {
        label: int(observed_distribution.get(label, 0)) for label in CANONICAL_LABELS
    }
    return {
        "role": role,
        "batch_sha256": validated.batch_sha256,
        "weight_sha256": (
            validated.baseline_weight.sha256
            if role == "baseline"
            else validated.candidate.weight.sha256
        ),
        "raw_prediction_count": len(run.raw_predictions),
        "raw_predictions_sha256": canonical_sha256(list(run.raw_predictions)),
        "predictions_sha256": canonical_sha256(predictions),
        "raw_predictions": list(run.raw_predictions),
        "predictions": predictions,
        "postprocessing": postprocessing,
        "quality": quality,
        "processed_video_frame_count": run.processed_video_frame_count,
        "inference_wall_seconds": round(run.inference_wall_seconds, 3),
        "model_metadata": dict(run.model_metadata),
    }


def _promotion_gates(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, bool]:
    baseline_quality = _object(baseline.get("quality"), "baseline quality")
    candidate_quality = _object(candidate.get("quality"), "candidate quality")
    precision = candidate_quality.get("precision")
    recall = candidate_quality.get("recall")
    clean = candidate_quality.get("clean_frame_rate")
    candidate_f1 = candidate_quality.get("f1_score")
    baseline_f1 = baseline_quality.get("f1_score")
    return {
        "precision_at_least_0_90": isinstance(precision, (int, float)) and precision >= 0.90,
        "recall_at_least_0_85": isinstance(recall, (int, float)) and recall >= 0.85,
        "clean_frame_rate_at_least_0_80": isinstance(clean, (int, float)) and clean >= 0.80,
        "match_iou_equals_0_50": True,
        "candidate_f1_strictly_greater_than_baseline": (
            isinstance(candidate_f1, (int, float))
            and isinstance(baseline_f1, (int, float))
            and candidate_f1 > baseline_f1
        ),
    }


def evaluate(
    protocol_path: Path,
    candidate_freeze_path: Path,
    output: Path,
    *,
    apply: bool = False,
    runner: ModelRunner | None = None,
    final_holdout_task_id: int | None = None,
    final_holdout_job_id: int | None = None,
) -> dict[str, Any]:
    """Preflight by default; consume the holdout exactly once when ``apply`` is true."""

    validated = validate_protocol(
        protocol_path,
        candidate_freeze_path,
        final_holdout_task_id=final_holdout_task_id,
        final_holdout_job_id=final_holdout_job_id,
    )
    preflight = build_preflight(validated, output=output)
    if not apply:
        return preflight
    if preflight["gate_result"] != "PASS":
        raise FrozenHoldoutError("holdout protocol was already consumed or output exists")

    output_path = _leaf_absolute(output)
    claim = _claim_once(output_path, validated)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_path.name}.inputs-", dir=output_path.parent))
    try:
        frozen_baseline = _copy_frozen(
            validated.baseline_weight,
            staging / f"baseline{validated.baseline_weight.path.suffix}",
            "baseline weight",
        )
        frozen_candidate = _copy_frozen(
            validated.candidate.weight,
            staging / f"candidate{validated.candidate.weight.path.suffix}",
            "candidate weight",
        )
        frozen_warmup = _copy_frozen(
            validated.warmup_image,
            staging / f"warmup{validated.warmup_image.path.suffix}",
            "warmup image",
        )
        targets_by_scene: dict[str, list[int]] = {}
        for scene_id, frame in validated.holdout.frame_keys:
            targets_by_scene.setdefault(scene_id, []).append(frame)
        frozen_videos: list[RuntimeSourceVideo] = []
        for index, video in enumerate(validated.source_videos):
            copied = _copy_frozen(
                video.file,
                staging / f"scene-{index:03d}{video.file.path.suffix}",
                f"source video {video.scene_id}",
            )
            frozen_videos.append(
                RuntimeSourceVideo(
                    video.scene_id,
                    video.frame_step,
                    copied,
                    tuple(sorted(targets_by_scene[video.scene_id])),
                )
            )
        frozen_inputs = (
            frozen_baseline,
            frozen_candidate,
            frozen_warmup,
            *(video.file for video in frozen_videos),
        )
        _revalidate_frozen_inputs(frozen_inputs)
        selected_runner = runner or ultralytics_runner
        baseline_run = selected_runner(
            frozen_baseline.path,
            frozen_warmup.path,
            tuple(frozen_videos),
            validated.settings,
            "baseline",
        )
        _revalidate_frozen_inputs(frozen_inputs)
        candidate_run = selected_runner(
            frozen_candidate.path,
            frozen_warmup.path,
            tuple(frozen_videos),
            validated.settings,
            "candidate",
        )
        _revalidate_frozen_inputs(frozen_inputs)
        baseline_result = _model_result("baseline", baseline_run, validated=validated)
        candidate_result = _model_result("candidate", candidate_run, validated=validated)
        gates = _promotion_gates(baseline_result, candidate_result)
        payload = {
            "schema": RESULT_SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dry_run": False,
            "inference_performed": True,
            "protocol_id": validated.protocol_id,
            "protocol_sha256": validated.protocol.sha256,
            "candidate_id": validated.candidate.candidate_id,
            "candidate_freeze_sha256": validated.candidate.record.sha256,
            "training_dataset_manifest_sha256": validated.candidate.dataset_manifest_sha256,
            "training_reference_manifest_sha256": validated.training_reference.manifest.sha256,
            "holdout_manifest_sha256": validated.holdout.manifest.sha256,
            "holdout_annotations_sha256": validated.holdout.annotations.sha256,
            "overlap_evidence_sha256": validated.overlap_evidence.sha256,
            "consumption_claim": {
                "path": claim.name,
                "sha256": hashlib.sha256(
                    _read_regular_bytes(claim, "consumption claim")
                ).hexdigest(),
            },
            "batch_sha256": validated.batch_sha256,
            "settings": dict(validated.settings),
            "validation": dict(validated.validation),
            "results": [baseline_result, candidate_result],
            "promotion_gates": gates,
            "gate_result": "PASS" if all(gates.values()) else "FAIL",
        }
        _atomic_write_json_new(output_path, payload)
        return payload
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument(
        "--candidate-freeze",
        required=True,
        type=Path,
        help="Immutable selector output directory containing receipt.json and best.pt",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--final-holdout-task-id",
        type=int,
        help="expected holdout task id; otherwise resolve from ROADLABELOPS_FINAL_HOLDOUT_TASK_IDS",
    )
    parser.add_argument(
        "--final-holdout-job-id",
        type=int,
        help="expected holdout job id; otherwise resolve from ROADLABELOPS_FINAL_HOLDOUT_JOB_IDS",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Consume the frozen holdout once and publish the immutable result",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        payload = evaluate(
            args.protocol,
            args.candidate_freeze,
            args.output,
            apply=args.apply,
            final_holdout_task_id=args.final_holdout_task_id,
            final_holdout_job_id=args.final_holdout_job_id,
        )
    except (FrozenHoldoutError, OSError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    if not args.apply:
        print("Preflight only: no inference was performed. Re-run with --apply once approved.")


if __name__ == "__main__":
    main()
