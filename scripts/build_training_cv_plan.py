"""Build a deterministic training-only LOSO cross-validation plan.

The sole input is an immutable ``roadlabelops.training-coco-reference`` manifest
inside the current workspace's ``data/ground-truth`` tree.  Its managed COCO and
image files are verified with stable, no-follow reads before one standard
``roadlabelops.training-asset-split`` file per source group is atomically
published.  Final holdout paths are deliberately outside this tool's input
contract.
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
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from roadlabelops.holdout_policy import (
    FINAL_HOLDOUT_REJECTED_SCOPES,
    final_holdout_scope_reason,
    is_configured_final_holdout_job,
    is_configured_final_holdout_task,
)

REFERENCE_SCHEMA = {"name": "roadlabelops.training-coco-reference", "version": 2}
SPLIT_PLAN_SCHEMA = {"name": "roadlabelops.training-asset-split", "version": 1}
OUTPUT_SCHEMA = {"name": "roadlabelops.training-loso-cv-plan", "version": 1}
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
REQUIRED_REFERENCE_GATE_CHECKS = frozenset(
    {
        "completed_jobs_with_zero_issues",
        "snapshot_and_annotation_hashes_verified",
        "completion_receipts_exactly_bound",
        "manifest_images_verified",
        "rectangle_only_annotations",
        "fixed_eight_class_taxonomy",
        "bounded_finite_positive_bboxes",
        "cross_task_image_identities_unique",
        "source_asset_hashes_unique",
        "stable_leakage_groups_bound",
        "source_frame_map_complete_and_unique",
        "all_images_including_zero_annotation_images_copied",
    }
)
SUPPORTED_IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})
MINIMUM_SOURCE_GROUP_COUNT = 3
MINIMUM_CLASS_SOURCE_SUPPORT = 2
MAX_MANIFEST_BYTES = 32 * 1024 * 1024
MAX_COCO_BYTES = 128 * 1024 * 1024
MAX_IMAGE_BYTES = 256 * 1024 * 1024


class TrainingCVPlanError(ValueError):
    """Raised when a training-only LOSO plan cannot be published safely."""


@dataclass(frozen=True)
class FileIdentity:
    device: int
    inode: int
    size_bytes: int
    modified_ns: int
    changed_ns: int


@dataclass(frozen=True)
class VerifiedFile:
    path: Path
    location: str
    sha256: str
    size_bytes: int
    identity: FileIdentity
    maximum_bytes: int


@dataclass(frozen=True)
class SourceAsset:
    value: int | str
    identity: tuple[str, str]
    source_sha256: str
    leakage_group_id: str


@dataclass(frozen=True)
class ReferenceImage:
    identifier: int
    file_name: str
    sha256: str
    width: int
    height: int
    task_id: int
    scene_id: str
    asset_value: int | str
    asset_identity: tuple[str, str]
    leakage_group_id: str
    normalized_asset_frame: int


@dataclass(frozen=True)
class ReferenceAnnotation:
    identifier: int
    image_id: int
    category_id: int
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class ValidatedReference:
    workspace_root: Path
    manifest_path: Path
    manifest_sha256: str
    manifest_size_bytes: int
    coco_path: Path
    coco_sha256: str
    coco_size_bytes: int
    managed_files_sha256: str
    assets: tuple[SourceAsset, ...]
    images: tuple[ReferenceImage, ...]
    annotations: tuple[ReferenceAnnotation, ...]
    verified_files: tuple[VerifiedFile, ...]


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True).encode(
            "utf-8"
        )
        + b"\n"
    )


def _semantic_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _object(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TrainingCVPlanError(f"{location} must be an object")
    return value


def _list(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise TrainingCVPlanError(f"{location} must be a list")
    return value


def _text(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TrainingCVPlanError(f"{location} must be a non-empty string")
    return value


def _integer(value: Any, location: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TrainingCVPlanError(f"{location} must be an integer")
    if minimum is not None and value < minimum:
        raise TrainingCVPlanError(f"{location} must be at least {minimum}")
    return value


def _number(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainingCVPlanError(f"{location} must be a number")
    try:
        result = float(value)
    except OverflowError as error:
        raise TrainingCVPlanError(f"{location} must be finite") from error
    if not math.isfinite(result):
        raise TrainingCVPlanError(f"{location} must be finite")
    return 0.0 if result == 0 else result


def _sha256_value(value: Any, location: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TrainingCVPlanError(f"{location} must be a lowercase SHA-256 digest")
    return value


def _asset_id(value: Any, location: str) -> int | str:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise TrainingCVPlanError(f"{location} must be an integer or string")
    if isinstance(value, str) and not value.strip():
        raise TrainingCVPlanError(f"{location} must be a non-empty string")
    return value


def _asset_identity(value: int | str) -> tuple[str, str]:
    return type(value).__name__, str(value)


def _asset_sort_key(value: int | str) -> tuple[int, int | str]:
    return (0, value) if isinstance(value, int) else (1, value)


def _absolute_lexical(path: Path, workspace_root: Path) -> Path:
    candidate = path if path.is_absolute() else workspace_root / path
    return Path(os.path.abspath(os.fspath(candidate)))


def _workspace_relative(path: Path, workspace_root: Path, location: str) -> str:
    try:
        relative = path.relative_to(workspace_root)
    except ValueError as error:
        raise TrainingCVPlanError(f"{location} must remain inside the workspace") from error
    return PurePosixPath(*relative.parts).as_posix()


def _forbidden_scope(parts: Sequence[str]) -> str | None:
    return final_holdout_scope_reason(PurePosixPath(*parts))


def _forbidden_path_scope(path: Path, workspace_root: Path) -> str | None:
    try:
        relative = path.relative_to(workspace_root)
    except ValueError:
        return None
    return _forbidden_scope(relative.parts)


def _reject_forbidden_path(path: Path, workspace_root: Path, location: str) -> None:
    scope = _forbidden_path_scope(path, workspace_root)
    if scope is not None:
        raise TrainingCVPlanError(
            f"{location} references forbidden {scope} scope outside the training-only contract"
        )


def _reject_symlink_chain(path: Path, workspace_root: Path, location: str) -> None:
    try:
        relative = path.relative_to(workspace_root)
    except ValueError as error:
        raise TrainingCVPlanError(f"{location} must remain inside the workspace") from error
    current = workspace_root
    for part in relative.parts:
        current /= part
        try:
            metadata = os.lstat(current)
        except OSError as error:
            raise TrainingCVPlanError(f"Could not inspect {location}: {error}") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise TrainingCVPlanError(f"{location} must not contain symlinks: {current}")


def _ensure_output_parent(path: Path, workspace_root: Path) -> None:
    try:
        relative = path.relative_to(workspace_root)
    except ValueError as error:
        raise TrainingCVPlanError("output parent must remain inside the workspace") from error
    current = workspace_root
    for part in relative.parts:
        current /= part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            try:
                os.mkdir(current)
                metadata = os.lstat(current)
            except OSError as error:
                raise TrainingCVPlanError(
                    f"Could not create output parent directory {current}: {error}"
                ) from error
        except OSError as error:
            raise TrainingCVPlanError(
                f"Could not inspect output parent {current}: {error}"
            ) from error
        if stat.S_ISLNK(metadata.st_mode):
            raise TrainingCVPlanError(f"output parent must not contain symlinks: {current}")
        if not stat.S_ISDIR(metadata.st_mode):
            raise TrainingCVPlanError(f"output parent component is not a directory: {current}")


def _identity(metadata: os.stat_result) -> FileIdentity:
    return FileIdentity(
        device=metadata.st_dev,
        inode=metadata.st_ino,
        size_bytes=metadata.st_size,
        modified_ns=metadata.st_mtime_ns,
        changed_ns=metadata.st_ctime_ns,
    )


def _read_regular_bytes(
    path: Path,
    location: str,
    *,
    workspace_root: Path,
    maximum_bytes: int,
) -> tuple[bytes, VerifiedFile]:
    _reject_forbidden_path(path, workspace_root, location)
    _reject_symlink_chain(path, workspace_root, location)
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise TrainingCVPlanError(f"Could not open {location}: {error}") from error
    try:
        before_metadata = os.fstat(descriptor)
        if not stat.S_ISREG(before_metadata.st_mode):
            raise TrainingCVPlanError(f"{location} must be a regular file")
        if before_metadata.st_size > maximum_bytes:
            raise TrainingCVPlanError(f"{location} exceeds the maximum supported size")
        chunks: list[bytes] = []
        consumed = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - consumed))
            if not chunk:
                break
            consumed += len(chunk)
            if consumed > maximum_bytes:
                raise TrainingCVPlanError(f"{location} exceeds the maximum supported size")
            chunks.append(chunk)
        after_metadata = os.fstat(descriptor)
    except OSError as error:
        raise TrainingCVPlanError(f"Could not read {location}: {error}") from error
    finally:
        os.close(descriptor)
    before = _identity(before_metadata)
    after = _identity(after_metadata)
    encoded = b"".join(chunks)
    if before != after or len(encoded) != before.size_bytes:
        raise TrainingCVPlanError(f"{location} changed while it was being read")
    return encoded, VerifiedFile(
        path=path,
        location=location,
        sha256=hashlib.sha256(encoded).hexdigest(),
        size_bytes=len(encoded),
        identity=before,
        maximum_bytes=maximum_bytes,
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TrainingCVPlanError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise TrainingCVPlanError(f"JSON contains non-finite number {value}")


def _parse_json(encoded: bytes, location: str) -> dict[str, Any]:
    try:
        decoded = encoded.decode("utf-8")
        payload = json.loads(
            decoded,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TrainingCVPlanError(f"Could not parse {location}: {error}") from error
    return _object(payload, location)


def _safe_managed_path(value: Any, location: str) -> str:
    path_text = _text(value, location)
    if "\\" in path_text or any(ord(character) < 32 for character in path_text):
        raise TrainingCVPlanError(f"{location} is unsafe")
    posix_path = PurePosixPath(path_text)
    windows_path = PureWindowsPath(path_text)
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or posix_path.as_posix() != path_text
        or not posix_path.name
        or any(part in {"", ".", ".."} for part in posix_path.parts)
    ):
        raise TrainingCVPlanError(f"{location} is unsafe")
    forbidden = _forbidden_scope(posix_path.parts)
    if forbidden is not None:
        raise TrainingCVPlanError(
            f"{location} references forbidden {forbidden} scope outside the training-only contract"
        )
    return path_text


def _validate_reference_gate(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema") != REFERENCE_SCHEMA:
        raise TrainingCVPlanError("reference manifest is not a supported training reference")
    gate = _object(manifest.get("gate"), "reference manifest.gate")
    blocking_reasons = _list(
        gate.get("blocking_reasons"), "reference manifest.gate.blocking_reasons"
    )
    if gate.get("passed") is not True or blocking_reasons:
        raise TrainingCVPlanError("reference manifest gate has not passed")
    checks = _object(gate.get("checks"), "reference manifest.gate.checks")
    if not REQUIRED_REFERENCE_GATE_CHECKS.issubset(checks) or any(
        value is not True for value in checks.values()
    ):
        raise TrainingCVPlanError("reference manifest gate checks are incomplete or false")


def _validate_evidence_locator(
    value: Any, location: str, *, workspace_root: Path
) -> tuple[str, str] | None:
    locator = _object(value, location)
    if "path" not in locator:
        return None
    path_text = _safe_managed_path(locator.get("path"), f"{location}.path")
    candidate = _absolute_lexical(Path(path_text), workspace_root)
    _workspace_relative(candidate, workspace_root, f"{location}.path")
    _reject_forbidden_path(candidate, workspace_root, f"{location}.path")
    digest = _sha256_value(locator.get("sha256"), f"{location}.sha256")
    return path_text, digest


def _validate_evidence_firewall(
    manifest: Mapping[str, Any], workspace_root: Path
) -> frozenset[int]:
    evidence = _object(manifest.get("evidence"), "reference manifest.evidence")
    _validate_evidence_locator(
        evidence.get("labels"), "reference manifest.evidence.labels", workspace_root=workspace_root
    )
    source_map = _object(evidence.get("source_map"), "reference manifest.evidence.source_map")
    _validate_evidence_locator(
        source_map, "reference manifest.evidence.source_map", workspace_root=workspace_root
    )
    declared_task_ids: set[int] = set()
    for index, raw_task in enumerate(
        _list(evidence.get("task_inputs"), "reference manifest.evidence.task_inputs")
    ):
        location = f"reference manifest.evidence.task_inputs[{index}]"
        task = _object(raw_task, location)
        task_id = _integer(task.get("task_id"), f"{location}.task_id", minimum=1)
        job_id = _integer(task.get("job_id"), f"{location}.job_id", minimum=1)
        if is_configured_final_holdout_task(task_id) or is_configured_final_holdout_job(job_id):
            raise TrainingCVPlanError(
                "configured final-holdout identity is outside the training-only contract"
            )
        declared_task_ids.add(task_id)
        for field in ("snapshot", "image_manifest", "completion_receipt"):
            _validate_evidence_locator(
                task.get(field), f"{location}.{field}", workspace_root=workspace_root
            )
    return frozenset(declared_task_ids)


def _validate_source_assets(manifest: Mapping[str, Any]) -> tuple[SourceAsset, ...]:
    evidence = _object(manifest.get("evidence"), "reference manifest.evidence")
    source_map = _object(evidence.get("source_map"), "reference manifest.evidence.source_map")
    raw_assets = _list(source_map.get("assets"), "reference manifest.evidence.source_map.assets")
    by_identity: dict[tuple[str, str], SourceAsset] = {}
    by_source_sha: dict[str, int | str] = {}
    by_leakage_group: dict[str, int | str] = {}
    for index, raw_asset in enumerate(raw_assets):
        location = f"reference manifest.evidence.source_map.assets[{index}]"
        asset = _object(raw_asset, location)
        value = _asset_id(asset.get("asset_id"), f"{location}.asset_id")
        identity = _asset_identity(value)
        if identity in by_identity:
            raise TrainingCVPlanError(f"duplicate typed source asset identity: {value!r}")
        source_sha = _sha256_value(asset.get("sha256"), f"{location}.sha256")
        leakage_group = _text(asset.get("leakage_group_id"), f"{location}.leakage_group_id")
        if leakage_group != f"sha256:{source_sha}":
            raise TrainingCVPlanError(
                f"{location}.leakage_group_id must equal the source SHA-256 identity"
            )
        if source_sha in by_source_sha:
            raise TrainingCVPlanError(
                "source SHA-256 is aliased by multiple source assets: "
                f"{by_source_sha[source_sha]!r} and {value!r}"
            )
        if leakage_group in by_leakage_group:
            raise TrainingCVPlanError(
                "source leakage group is aliased by multiple source assets: "
                f"{by_leakage_group[leakage_group]!r} and {value!r}"
            )
        validated = SourceAsset(value, identity, source_sha, leakage_group)
        by_identity[identity] = validated
        by_source_sha[source_sha] = value
        by_leakage_group[leakage_group] = value
    if len(by_leakage_group) < MINIMUM_SOURCE_GROUP_COUNT:
        raise TrainingCVPlanError(
            "training reference must contain at least "
            f"{MINIMUM_SOURCE_GROUP_COUNT} unique source groups"
        )
    return tuple(sorted(by_identity.values(), key=lambda item: _asset_sort_key(item.value)))


def _validate_categories(coco: Mapping[str, Any]) -> None:
    categories = _list(coco.get("categories"), "COCO.categories")
    actual: dict[int, str] = {}
    for index, raw_category in enumerate(categories):
        location = f"COCO.categories[{index}]"
        category = _object(raw_category, location)
        identifier = _integer(category.get("id"), f"{location}.id", minimum=1)
        name = _text(category.get("name"), f"{location}.name")
        if identifier in actual:
            raise TrainingCVPlanError(f"COCO contains duplicate category id {identifier}")
        actual[identifier] = name
    expected = {index: label for index, label in enumerate(REQUIRED_LABELS, start=1)}
    if actual != expected:
        raise TrainingCVPlanError(
            "COCO must contain the fixed eight-class taxonomy with canonical IDs"
        )


def _validate_images(
    coco: Mapping[str, Any],
    assets: Sequence[SourceAsset],
    declared_task_ids: frozenset[int],
) -> tuple[ReferenceImage, ...]:
    asset_by_identity = {asset.identity: asset for asset in assets}
    images: list[ReferenceImage] = []
    identifiers: set[int] = set()
    file_names: dict[str, str] = {}
    hashes: dict[str, str] = {}
    normalized_frames: set[tuple[str, int]] = set()
    for index, raw_image in enumerate(_list(coco.get("images"), "COCO.images")):
        location = f"COCO.images[{index}]"
        image = _object(raw_image, location)
        identifier = _integer(image.get("id"), f"{location}.id", minimum=1)
        if identifier in identifiers:
            raise TrainingCVPlanError(f"COCO contains duplicate image id {identifier}")
        identifiers.add(identifier)
        file_name = _safe_managed_path(image.get("file_name"), f"{location}.file_name")
        task_id = _integer(image.get("task_id"), f"{location}.task_id", minimum=1)
        if is_configured_final_holdout_task(task_id):
            raise TrainingCVPlanError(
                "configured final-holdout task is outside the training-only contract"
            )
        if task_id not in declared_task_ids:
            raise TrainingCVPlanError(
                f"{location}.task_id {task_id} is not declared by reference evidence.task_inputs"
            )
        if PurePosixPath(file_name).suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            raise TrainingCVPlanError(f"{location}.file_name has unsupported image suffix")
        casefolded = file_name.casefold()
        if casefolded in file_names:
            raise TrainingCVPlanError(
                f"COCO contains colliding image paths {file_names[casefolded]!r} and {file_name!r}"
            )
        file_names[casefolded] = file_name
        digest = _sha256_value(image.get("sha256"), f"{location}.sha256")
        if digest in hashes:
            raise TrainingCVPlanError(
                f"COCO contains duplicate image content for {hashes[digest]!r} and {file_name!r}"
            )
        hashes[digest] = file_name
        asset_value = _asset_id(image.get("source_asset_id"), f"{location}.source_asset_id")
        asset_identity = _asset_identity(asset_value)
        asset = asset_by_identity.get(asset_identity)
        if asset is None:
            raise TrainingCVPlanError(
                f"COCO source asset is absent from manifest evidence: {asset_value!r}"
            )
        leakage_group = _text(
            image.get("source_leakage_group_id"), f"{location}.source_leakage_group_id"
        )
        if leakage_group != asset.leakage_group_id:
            raise TrainingCVPlanError(
                f"COCO leakage group differs from manifest evidence for asset {asset_value!r}"
            )
        normalized_frame = _integer(
            image.get("source_normalized_asset_frame"),
            f"{location}.source_normalized_asset_frame",
            minimum=0,
        )
        normalized_identity = (leakage_group, normalized_frame)
        if normalized_identity in normalized_frames:
            raise TrainingCVPlanError(
                "COCO duplicates a normalized frame within one source leakage group"
            )
        normalized_frames.add(normalized_identity)
        images.append(
            ReferenceImage(
                identifier=identifier,
                file_name=file_name,
                sha256=digest,
                width=_integer(image.get("width"), f"{location}.width", minimum=1),
                height=_integer(image.get("height"), f"{location}.height", minimum=1),
                task_id=task_id,
                scene_id=_text(image.get("scene_id"), f"{location}.scene_id"),
                asset_value=asset_value,
                asset_identity=asset_identity,
                leakage_group_id=leakage_group,
                normalized_asset_frame=normalized_frame,
            )
        )
    if not images:
        raise TrainingCVPlanError("COCO.images must not be empty")
    observed_assets = {image.asset_identity for image in images}
    expected_assets = set(asset_by_identity)
    if observed_assets != expected_assets:
        missing = sorted(
            (asset_by_identity[identity].value for identity in expected_assets - observed_assets),
            key=_asset_sort_key,
        )
        raise TrainingCVPlanError(f"manifest source assets without COCO images: {missing!r}")
    return tuple(sorted(images, key=lambda item: item.identifier))


def _validate_annotations(
    coco: Mapping[str, Any], images: Sequence[ReferenceImage]
) -> tuple[ReferenceAnnotation, ...]:
    images_by_id = {image.identifier: image for image in images}
    annotations: list[ReferenceAnnotation] = []
    identifiers: set[int] = set()
    for index, raw_annotation in enumerate(_list(coco.get("annotations"), "COCO.annotations")):
        location = f"COCO.annotations[{index}]"
        annotation = _object(raw_annotation, location)
        identifier = _integer(annotation.get("id"), f"{location}.id", minimum=1)
        if identifier in identifiers:
            raise TrainingCVPlanError(f"COCO contains duplicate annotation id {identifier}")
        identifiers.add(identifier)
        image_id = _integer(annotation.get("image_id"), f"{location}.image_id", minimum=1)
        image = images_by_id.get(image_id)
        if image is None:
            raise TrainingCVPlanError(f"{location} references unknown image id {image_id}")
        category_id = _integer(annotation.get("category_id"), f"{location}.category_id", minimum=1)
        if category_id not in range(1, len(REQUIRED_LABELS) + 1):
            raise TrainingCVPlanError(f"{location} references an unknown category")
        raw_bbox = _list(annotation.get("bbox"), f"{location}.bbox")
        if len(raw_bbox) != 4:
            raise TrainingCVPlanError(f"{location}.bbox must contain exactly four numbers")
        x, y, width, height = (
            _number(value, f"{location}.bbox[{position}]")
            for position, value in enumerate(raw_bbox)
        )
        if (
            x < 0
            or y < 0
            or width <= 0
            or height <= 0
            or x + width > image.width
            or y + height > image.height
        ):
            raise TrainingCVPlanError(f"{location}.bbox escapes image bounds")
        area = _number(annotation.get("area"), f"{location}.area")
        if area <= 0 or area != width * height:
            raise TrainingCVPlanError(f"{location}.area must equal bbox width * height")
        if _integer(annotation.get("iscrowd"), f"{location}.iscrowd", minimum=0) != 0:
            raise TrainingCVPlanError(f"{location}.iscrowd must be 0")
        annotations.append(
            ReferenceAnnotation(identifier, image_id, category_id, (x, y, width, height))
        )
    return tuple(
        sorted(
            annotations,
            key=lambda item: (item.image_id, item.category_id, item.bbox, item.identifier),
        )
    )


def _validate_manifest_counts(
    manifest: Mapping[str, Any],
    images: Sequence[ReferenceImage],
    annotations: Sequence[ReferenceAnnotation],
) -> None:
    annotations_by_image = Counter(annotation.image_id for annotation in annotations)
    by_category = Counter(REQUIRED_LABELS[item.category_id - 1] for item in annotations)
    counts = _object(manifest.get("counts"), "reference manifest.counts")
    expected_scalars = {
        "tasks": len({image.task_id for image in images}),
        "images": len(images),
        "zero_annotation_images": sum(
            annotations_by_image[image.identifier] == 0 for image in images
        ),
        "annotations": len(annotations),
        "categories": len(REQUIRED_LABELS),
    }
    for key, expected in expected_scalars.items():
        actual = _integer(counts.get(key), f"reference manifest.counts.{key}", minimum=0)
        if actual != expected:
            raise TrainingCVPlanError(f"reference manifest.counts.{key} differs from managed COCO")
    reported = _object(
        counts.get("annotations_by_category"),
        "reference manifest.counts.annotations_by_category",
    )
    if set(reported) != set(REQUIRED_LABELS):
        raise TrainingCVPlanError(
            "reference manifest category counts do not cover the canonical taxonomy"
        )
    for label in REQUIRED_LABELS:
        actual = _integer(
            reported.get(label),
            f"reference manifest.counts.annotations_by_category.{label}",
            minimum=0,
        )
        if actual != by_category[label]:
            raise TrainingCVPlanError(f"reference manifest category count differs for {label!r}")


def _validate_source_statistics(
    manifest: Mapping[str, Any],
    assets: Sequence[SourceAsset],
    images: Sequence[ReferenceImage],
    annotations: Sequence[ReferenceAnnotation],
) -> None:
    annotations_by_image = Counter(annotation.image_id for annotation in annotations)
    images_by_asset: dict[tuple[str, str], list[ReferenceImage]] = defaultdict(list)
    for image in images:
        images_by_asset[image.asset_identity].append(image)
    asset_by_identity = {asset.identity: asset for asset in assets}
    statistics = _object(manifest.get("source_statistics"), "reference manifest.source_statistics")
    records: dict[tuple[str, str], Mapping[str, Any]] = {}
    for index, raw_record in enumerate(
        _list(statistics.get("assets"), "reference manifest.source_statistics.assets")
    ):
        location = f"reference manifest.source_statistics.assets[{index}]"
        record = _object(raw_record, location)
        value = _asset_id(record.get("asset_id"), f"{location}.asset_id")
        identity = _asset_identity(value)
        if identity in records:
            raise TrainingCVPlanError(
                f"reference manifest source statistics duplicate asset {value!r}"
            )
        records[identity] = record
    if set(records) != set(asset_by_identity):
        raise TrainingCVPlanError(
            "reference manifest source statistics must exactly cover source assets"
        )
    for identity, asset in asset_by_identity.items():
        record = records[identity]
        location = f"reference manifest.source_statistics asset {asset.value!r}"
        source_images = images_by_asset[identity]
        expected_annotations = sum(
            annotations_by_image[image.identifier] for image in source_images
        )
        if _text(record.get("leakage_group_id"), f"{location}.leakage_group_id") != (
            asset.leakage_group_id
        ):
            raise TrainingCVPlanError(f"{location} leakage group differs")
        if _integer(record.get("image_count"), f"{location}.image_count", minimum=0) != len(
            source_images
        ):
            raise TrainingCVPlanError(f"{location} image count differs")
        if (
            _integer(record.get("annotation_count"), f"{location}.annotation_count", minimum=0)
            != expected_annotations
        ):
            raise TrainingCVPlanError(f"{location} annotation count differs")
        scene_ids = [
            _text(item, f"{location}.scene_ids")
            for item in _list(record.get("scene_ids"), f"{location}.scene_ids")
        ]
        if scene_ids != sorted({image.scene_id for image in source_images}):
            raise TrainingCVPlanError(f"{location} scene IDs differ")


def _validate_reference(
    reference_manifest_path: Path, *, workspace_root: Path
) -> ValidatedReference:
    workspace_root = workspace_root.resolve(strict=True)
    manifest_path = _absolute_lexical(reference_manifest_path, workspace_root)
    _workspace_relative(manifest_path, workspace_root, "reference manifest")
    _reject_forbidden_path(manifest_path, workspace_root, "reference manifest")
    ground_truth_root = workspace_root / "data" / "ground-truth"
    try:
        manifest_path.relative_to(ground_truth_root)
    except ValueError as error:
        raise TrainingCVPlanError(
            "reference manifest must be inside workspace data/ground-truth"
        ) from error
    if manifest_path.name != "manifest.json":
        raise TrainingCVPlanError("reference manifest file name must be manifest.json")
    manifest_bytes, manifest_record = _read_regular_bytes(
        manifest_path,
        "reference manifest",
        workspace_root=workspace_root,
        maximum_bytes=MAX_MANIFEST_BYTES,
    )
    manifest = _parse_json(manifest_bytes, "reference manifest")
    _validate_reference_gate(manifest)
    declared_task_ids = _validate_evidence_firewall(manifest, workspace_root)
    assets = _validate_source_assets(manifest)

    file_records: dict[str, tuple[str, int]] = {}
    for index, raw_record in enumerate(_list(manifest.get("files"), "reference manifest.files")):
        location = f"reference manifest.files[{index}]"
        record = _object(raw_record, location)
        relative_path = _safe_managed_path(record.get("path"), f"{location}.path")
        if relative_path in file_records:
            raise TrainingCVPlanError(
                f"reference manifest contains duplicate managed path {relative_path!r}"
            )
        file_records[relative_path] = (
            _sha256_value(record.get("sha256"), f"{location}.sha256"),
            _integer(record.get("size_bytes"), f"{location}.size_bytes", minimum=0),
        )
    coco_relative = "annotations.coco.json"
    if coco_relative not in file_records:
        raise TrainingCVPlanError("reference manifest does not manage annotations.coco.json")
    coco_path = manifest_path.parent / coco_relative
    coco_bytes, coco_record = _read_regular_bytes(
        coco_path,
        "managed COCO",
        workspace_root=workspace_root,
        maximum_bytes=MAX_COCO_BYTES,
    )
    if (coco_record.sha256, coco_record.size_bytes) != file_records[coco_relative]:
        raise TrainingCVPlanError("managed COCO hash or size differs from reference manifest")
    coco = _parse_json(coco_bytes, "managed COCO")
    info = _object(coco.get("info"), "COCO.info")
    if info.get("schema") != REFERENCE_SCHEMA:
        raise TrainingCVPlanError("managed COCO is not a supported training reference")
    evidence = _object(manifest.get("evidence"), "reference manifest.evidence")
    labels = _object(evidence.get("labels"), "reference manifest.evidence.labels")
    source_map = _object(evidence.get("source_map"), "reference manifest.evidence.source_map")
    if _sha256_value(info.get("labels_sha256"), "COCO.info.labels_sha256") != (
        _sha256_value(labels.get("sha256"), "reference manifest.evidence.labels.sha256")
    ):
        raise TrainingCVPlanError("COCO labels hash differs from reference manifest evidence")
    if _sha256_value(info.get("source_map_sha256"), "COCO.info.source_map_sha256") != (
        _sha256_value(source_map.get("sha256"), "reference manifest.evidence.source_map.sha256")
    ):
        raise TrainingCVPlanError("COCO source-map hash differs from reference manifest evidence")
    _validate_categories(coco)
    images = _validate_images(coco, assets, declared_task_ids)
    annotations = _validate_annotations(coco, images)
    expected_paths = {coco_relative, *(image.file_name for image in images)}
    if set(file_records) != expected_paths:
        missing = sorted(expected_paths - set(file_records))
        unexpected = sorted(set(file_records) - expected_paths)
        raise TrainingCVPlanError(
            "reference manifest managed files must exactly cover COCO and images; "
            f"missing={missing!r}, unexpected={unexpected!r}"
        )

    verified_files = [manifest_record, coco_record]
    for image in images:
        image_path = manifest_path.parent / Path(*PurePosixPath(image.file_name).parts)
        _workspace_relative(image_path, workspace_root, f"managed image {image.file_name!r}")
        encoded, record = _read_regular_bytes(
            image_path,
            f"managed image {image.file_name!r}",
            workspace_root=workspace_root,
            maximum_bytes=MAX_IMAGE_BYTES,
        )
        del encoded
        declared = file_records[image.file_name]
        if (record.sha256, record.size_bytes) != declared or record.sha256 != image.sha256:
            raise TrainingCVPlanError(f"managed image hash or size differs for {image.file_name!r}")
        verified_files.append(record)
    _validate_manifest_counts(manifest, images, annotations)
    _validate_source_statistics(manifest, assets, images, annotations)
    managed_files_sha256 = _semantic_sha256(
        [
            {"path": path, "sha256": digest, "size_bytes": size}
            for path, (digest, size) in sorted(file_records.items())
        ]
    )
    return ValidatedReference(
        workspace_root=workspace_root,
        manifest_path=manifest_path,
        manifest_sha256=manifest_record.sha256,
        manifest_size_bytes=manifest_record.size_bytes,
        coco_path=coco_path,
        coco_sha256=coco_record.sha256,
        coco_size_bytes=coco_record.size_bytes,
        managed_files_sha256=managed_files_sha256,
        assets=assets,
        images=images,
        annotations=annotations,
        verified_files=tuple(verified_files),
    )


def _source_statistics(
    reference: ValidatedReference,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    annotations_by_image: dict[int, list[ReferenceAnnotation]] = defaultdict(list)
    for annotation in reference.annotations:
        annotations_by_image[annotation.image_id].append(annotation)
    images_by_asset: dict[tuple[str, str], list[ReferenceImage]] = defaultdict(list)
    for image in reference.images:
        images_by_asset[image.asset_identity].append(image)

    asset_records: list[dict[str, Any]] = []
    class_sources: dict[str, list[int | str]] = {label: [] for label in REQUIRED_LABELS}
    class_boxes: Counter[str] = Counter()
    class_positive_images: dict[str, set[int]] = defaultdict(set)
    for asset in reference.assets:
        source_images = images_by_asset[asset.identity]
        boxes: Counter[str] = Counter()
        positive_images: dict[str, set[int]] = defaultdict(set)
        for image in source_images:
            for annotation in annotations_by_image.get(image.identifier, ()):
                label = REQUIRED_LABELS[annotation.category_id - 1]
                boxes[label] += 1
                positive_images[label].add(image.identifier)
                class_boxes[label] += 1
                class_positive_images[label].add(image.identifier)
        for label in REQUIRED_LABELS:
            if boxes[label] > 0:
                class_sources[label].append(asset.value)
        asset_records.append(
            {
                "asset_id": asset.value,
                "source_sha256": asset.source_sha256,
                "leakage_group_id": asset.leakage_group_id,
                "image_count": len(source_images),
                "zero_annotation_image_count": sum(
                    not annotations_by_image.get(image.identifier) for image in source_images
                ),
                "annotation_count": sum(boxes.values()),
                "classes": {
                    label: {
                        "box_count": boxes[label],
                        "positive_image_count": len(positive_images[label]),
                        "zero_image_count": len(source_images) - len(positive_images[label]),
                    }
                    for label in REQUIRED_LABELS
                },
            }
        )
    support = {
        label: {
            "source_count": len(class_sources[label]),
            "source_asset_ids": sorted(class_sources[label], key=_asset_sort_key),
            "box_count": class_boxes[label],
            "positive_image_count": len(class_positive_images[label]),
        }
        for label in REQUIRED_LABELS
    }
    return asset_records, support


def _split_statistics(
    reference: ValidatedReference,
    selected_assets: set[tuple[str, str]],
) -> dict[str, Any]:
    selected_images = [
        image for image in reference.images if image.asset_identity in selected_assets
    ]
    selected_image_ids = {image.identifier for image in selected_images}
    annotations = [
        annotation
        for annotation in reference.annotations
        if annotation.image_id in selected_image_ids
    ]
    by_image: dict[int, list[ReferenceAnnotation]] = defaultdict(list)
    for annotation in annotations:
        by_image[annotation.image_id].append(annotation)
    box_counts: Counter[str] = Counter()
    positive_images: dict[str, set[int]] = defaultdict(set)
    for annotation in annotations:
        label = REQUIRED_LABELS[annotation.category_id - 1]
        box_counts[label] += 1
        positive_images[label].add(annotation.image_id)
    return {
        "source_count": len(selected_assets),
        "image_count": len(selected_images),
        "zero_annotation_image_count": sum(
            not by_image.get(image.identifier) for image in selected_images
        ),
        "annotation_count": len(annotations),
        "classes": {
            label: {
                "box_count": box_counts[label],
                "positive_image_count": len(positive_images[label]),
                "zero_image_count": len(selected_images) - len(positive_images[label]),
            }
            for label in REQUIRED_LABELS
        },
    }


def _build_payloads(
    reference: ValidatedReference,
) -> tuple[dict[str, Any], list[tuple[str, bytes]]]:
    asset_records, class_support = _source_statistics(reference)
    unsupported = [
        label
        for label in REQUIRED_LABELS
        if class_support[label]["source_count"] < MINIMUM_CLASS_SOURCE_SUPPORT
    ]
    if unsupported:
        raise TrainingCVPlanError(
            "each class must be represented by at least two source assets; "
            f"insufficient={unsupported!r}"
        )
    all_identities = {asset.identity for asset in reference.assets}
    split_files: list[tuple[str, bytes]] = []
    folds: list[dict[str, Any]] = []
    validation_counts: Counter[tuple[str, str]] = Counter()
    for index, validation_asset in enumerate(reference.assets, start=1):
        fold_id = f"fold-{index:02d}"
        validation_identities = {validation_asset.identity}
        training_identities = all_identities - validation_identities
        training_assets = sorted(
            (asset.value for asset in reference.assets if asset.identity in training_identities),
            key=_asset_sort_key,
        )
        split_plan = {
            "schema": SPLIT_PLAN_SCHEMA,
            "train_asset_ids": training_assets,
            "val_asset_ids": [validation_asset.value],
        }
        split_bytes = _json_bytes(split_plan)
        split_path = f"folds/{fold_id}.split.json"
        split_files.append((split_path, split_bytes))
        train_stats = _split_statistics(reference, training_identities)
        val_stats = _split_statistics(reference, validation_identities)
        train_missing = [
            label for label in REQUIRED_LABELS if train_stats["classes"][label]["box_count"] == 0
        ]
        if train_missing:
            raise TrainingCVPlanError(
                f"{fold_id} training partition lacks canonical classes: {train_missing!r}"
            )
        train_groups = {
            asset.leakage_group_id
            for asset in reference.assets
            if asset.identity in training_identities
        }
        val_groups = {validation_asset.leakage_group_id}
        if training_identities & validation_identities:
            raise TrainingCVPlanError(f"{fold_id} source asset crosses train and val")
        if train_groups & val_groups:
            raise TrainingCVPlanError(f"{fold_id} source leakage group crosses train and val")
        evaluability = {
            label: (
                "evaluable" if val_stats["classes"][label]["box_count"] > 0 else "not_evaluable"
            )
            for label in REQUIRED_LABELS
        }
        validation_counts[validation_asset.identity] += 1
        folds.append(
            {
                "fold_id": fold_id,
                "val_asset_id": validation_asset.value,
                "train": {"asset_ids": training_assets, **train_stats},
                "val": {"asset_ids": [validation_asset.value], **val_stats},
                "validation_evaluability": evaluability,
                "split_plan": {
                    "path": split_path,
                    "sha256": hashlib.sha256(split_bytes).hexdigest(),
                    "size_bytes": len(split_bytes),
                    "schema": SPLIT_PLAN_SCHEMA,
                },
                "gate": {
                    "passed": True,
                    "checks": {
                        "training_partition_contains_all_classes": True,
                        "source_asset_overlap_count_zero": True,
                        "source_leakage_group_overlap_count_zero": True,
                        "missing_validation_classes_marked_not_evaluable": all(
                            (
                                status == "not_evaluable"
                                and val_stats["classes"][label]["box_count"] == 0
                            )
                            or (
                                status == "evaluable"
                                and val_stats["classes"][label]["box_count"] > 0
                            )
                            for label, status in evaluability.items()
                        ),
                    },
                },
            }
        )
    if set(validation_counts) != all_identities or any(
        count != 1 for count in validation_counts.values()
    ):
        raise TrainingCVPlanError("each source asset must be validation exactly once")

    file_records = [
        {
            "path": path,
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "size_bytes": len(encoded),
        }
        for path, encoded in split_files
    ]
    plan_semantic_sha256 = _semantic_sha256(
        {
            "schema": OUTPUT_SCHEMA,
            "method": "leave-one-source-asset-out",
            "folds": [
                {
                    "fold_id": fold["fold_id"],
                    "split_plan": _parse_json(encoded, path),
                }
                for fold, (path, encoded) in zip(folds, split_files, strict=True)
            ],
        }
    )
    annotations_by_image = Counter(annotation.image_id for annotation in reference.annotations)
    manifest = {
        "schema": OUTPUT_SCHEMA,
        "input_scope": "training_internal_only",
        "method": "deterministic leave-one-source-asset-out",
        "taxonomy": list(REQUIRED_LABELS),
        "inputs": {
            "reference_manifest": {
                "path": _workspace_relative(
                    reference.manifest_path,
                    reference.workspace_root,
                    "reference manifest",
                ),
                "sha256": reference.manifest_sha256,
                "size_bytes": reference.manifest_size_bytes,
                "schema": REFERENCE_SCHEMA,
            },
            "coco": {
                "path": _workspace_relative(
                    reference.coco_path, reference.workspace_root, "managed COCO"
                ),
                "sha256": reference.coco_sha256,
                "size_bytes": reference.coco_size_bytes,
                "schema": REFERENCE_SCHEMA,
            },
            "managed_files_sha256": reference.managed_files_sha256,
        },
        "plan_semantic_sha256": plan_semantic_sha256,
        "counts": {
            "source_assets": len(reference.assets),
            "source_groups": len({asset.leakage_group_id for asset in reference.assets}),
            "folds": len(folds),
            "images": len(reference.images),
            "zero_annotation_images": sum(
                annotations_by_image[image.identifier] == 0 for image in reference.images
            ),
            "annotations": len(reference.annotations),
            "categories": len(REQUIRED_LABELS),
        },
        "source_statistics": {
            "assets": asset_records,
            "class_source_support": class_support,
        },
        "folds": folds,
        "holdout_firewall": {
            "allowed_reference_scope": "data/ground-truth",
            "rejected_scopes": list(FINAL_HOLDOUT_REJECTED_SCOPES),
            "final_holdout_input_read": False,
        },
        "gate": {
            "passed": True,
            "blocking_reasons": [],
            "checks": {
                "workspace_ground_truth_training_reference_verified": True,
                "reference_schema_and_gate_verified": True,
                "reference_managed_files_verified_with_stable_reads": True,
                "fixed_eight_class_taxonomy": True,
                "minimum_unique_source_group_count_met": True,
                "each_class_supported_by_at_least_two_sources": True,
                "each_source_exactly_once_as_validation": True,
                "every_training_fold_contains_all_classes": True,
                "source_asset_never_crosses_train_and_val": True,
                "source_leakage_group_never_crosses_train_and_val": True,
                "missing_validation_classes_marked_not_evaluable": True,
                "standard_split_plans_are_deterministic": True,
                "holdout_path_outside_input_contract": True,
            },
        },
        "files": file_records,
    }
    return manifest, split_files


def _verify_inputs_unchanged(files: Sequence[VerifiedFile], workspace_root: Path) -> None:
    for expected in files:
        _encoded, actual = _read_regular_bytes(
            expected.path,
            expected.location,
            workspace_root=workspace_root,
            maximum_bytes=expected.maximum_bytes,
        )
        if (
            actual.identity != expected.identity
            or actual.sha256 != expected.sha256
            or actual.size_bytes != expected.size_bytes
        ):
            raise TrainingCVPlanError(f"{expected.location} changed during plan construction")


def _write_new(path: Path, encoded: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directories(root: Path) -> None:
    directories = [path for path in root.rglob("*") if path.is_dir()]
    for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True) + [root]:
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _atomic_publish_directory_no_replace(staging: Path, output: Path) -> None:
    source_bytes = os.fsencode(staging)
    target_bytes = os.fsencode(output)
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        rename_exclusive = libc.renamex_np
        rename_exclusive.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename_exclusive.restype = ctypes.c_int
        result = rename_exclusive(source_bytes, target_bytes, 0x00000004)
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
        result = rename_no_replace(-100, source_bytes, -100, target_bytes, 0x00000001)
    else:
        raise TrainingCVPlanError(
            "atomic no-replace directory publication is unsupported on this platform"
        )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise TrainingCVPlanError(f"output directory already exists: {output}")
    raise OSError(error_number, os.strerror(error_number), str(output))


def build_training_cv_plan(
    reference_manifest_path: Path,
    output: Path,
    *,
    workspace_root: Path | None = None,
) -> dict[str, Any]:
    """Validate one training reference and atomically publish its LOSO plans."""

    root = (workspace_root or Path.cwd()).resolve(strict=True)
    reference = _validate_reference(Path(reference_manifest_path), workspace_root=root)
    output_root = _absolute_lexical(Path(output), root)
    _workspace_relative(output_root, root, "output")
    _reject_forbidden_path(output_root, root, "output")
    if output_root.is_relative_to(reference.manifest_path.parent) or (
        reference.manifest_path.parent.is_relative_to(output_root)
    ):
        raise TrainingCVPlanError("output must not overlap the immutable training reference")
    if os.path.lexists(output_root):
        raise TrainingCVPlanError(f"output directory already exists: {output_root}")
    _ensure_output_parent(output_root.parent, root)
    manifest, split_files = _build_payloads(reference)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.staging-", dir=output_root.parent))
    published = False
    try:
        for relative_path, encoded in split_files:
            _write_new(staging / Path(*PurePosixPath(relative_path).parts), encoded)
        _write_new(staging / "manifest.json", _json_bytes(manifest))
        _verify_inputs_unchanged(reference.verified_files, root)
        _fsync_directories(staging)
        _atomic_publish_directory_no_replace(staging, output_root)
        published = True
        parent_descriptor = os.open(output_root.parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    finally:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)
    return {
        "output": str(output_root),
        "manifest": str(output_root / "manifest.json"),
        "plan_semantic_sha256": manifest["plan_semantic_sha256"],
        "counts": manifest["counts"],
        "gate": manifest["gate"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = build_training_cv_plan(args.reference_manifest, args.output)
    except TrainingCVPlanError as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
