"""Build immutable diagnostics for a failed training run without holdout access.

The command accepts only an immutable training-reference manifest, an immutable
YOLO-dataset manifest, and training-candidate receipts.  Any additional file is
read only when it is explicitly hash-bound by one of those inputs.  Configured
final-holdout paths and identities are rejected before any input is opened.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import secrets
import stat
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, BinaryIO

import yaml

from roadlabelops.holdout_policy import (
    final_holdout_scope_reason,
    is_configured_final_holdout_job,
    is_configured_final_holdout_task,
)
from scripts import freeze_model_candidate as candidate_validator
from scripts import train_yolo_candidate as dataset_validator

ANALYSIS_SCHEMA = {"name": "roadlabelops.training-recovery-analysis", "version": 1}
REFERENCE_SCHEMA = {"name": "roadlabelops.training-coco-reference", "version": 2}
SUPPORTED_DATASET_SCHEMAS = (
    dataset_validator.LEGACY_DATASET_SCHEMA,
    dataset_validator.DATASET_SCHEMA,
)
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
AGGREGATE_METRICS = ("precision", "recall", "map50", "map50_95")
LEARNING_ARGS = (
    "optimizer",
    "lr0",
    "lrf",
    "momentum",
    "weight_decay",
    "warmup_epochs",
    "warmup_momentum",
    "warmup_bias_lr",
    "box",
    "cls",
    "cls_pw",
    "dfl",
    "epochs",
    "patience",
    "batch",
    "imgsz",
    "freeze",
    "close_mosaic",
    "mosaic",
    "amp",
    "deterministic",
)
PRETRAINED_CLASS_NAMES = {
    "car": "car",
    "bus": "bus",
    "truck": "truck",
    "motorcycle": "motorcycle",
    "bicycle": "bicycle",
    "pedestrian": "person",
    "traffic_light": "traffic light",
    "traffic_sign": "stop sign",
}
MAX_JSON_BYTES = 128 * 1024 * 1024
MAX_TEXT_ARTIFACT_BYTES = 32 * 1024 * 1024


class TrainingRecoveryAnalysisError(ValueError):
    """Raised when diagnostic evidence cannot be validated or published safely."""


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as error:
            raise TrainingRecoveryAnalysisError("YAML contains an unhashable key") from error
        if duplicate:
            raise TrainingRecoveryAnalysisError(f"YAML contains duplicate key {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


@dataclass(frozen=True)
class BoundBytes:
    path: Path
    payload: bytes
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class ReferenceData:
    root: Path
    manifest: dict[str, Any]
    manifest_bound: BoundBytes
    coco: dict[str, Any]
    coco_bound: BoundBytes


@dataclass(frozen=True)
class CandidateData:
    seed: int
    receipt: dict[str, Any]
    receipt_bound: BoundBytes
    args: dict[str, Any]
    curve: dict[str, Any]


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _semantic_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _same_contract(first: Any, second: Any) -> bool:
    return _semantic_bytes(first) == _semantic_bytes(second)


def _object(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TrainingRecoveryAnalysisError(f"{location} must be an object")
    return value


def _list(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise TrainingRecoveryAnalysisError(f"{location} must be a list")
    return value


def _integer(value: Any, location: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TrainingRecoveryAnalysisError(f"{location} must be an integer >= {minimum}")
    return value


def _number(value: Any, location: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainingRecoveryAnalysisError(f"{location} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise TrainingRecoveryAnalysisError(f"{location} must be a finite number >= {minimum}")
    return 0.0 if result == 0 else result


def _digest(value: Any, location: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise TrainingRecoveryAnalysisError(f"{location} must be a lowercase SHA-256")
    return value


def _reject_constant(value: str) -> None:
    raise TrainingRecoveryAnalysisError(f"JSON contains non-finite value {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TrainingRecoveryAnalysisError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _parse_json(encoded: bytes, location: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            encoded.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_json_object,
        )
    except TrainingRecoveryAnalysisError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TrainingRecoveryAnalysisError(f"could not parse {location}: {error}") from error
    return _object(payload, location)


def _forbidden_scope(path: PurePosixPath) -> bool:
    return final_holdout_scope_reason(path) is not None


def _reject_forbidden(path: PurePosixPath, location: str) -> None:
    if _forbidden_scope(path):
        raise TrainingRecoveryAnalysisError(
            f"{location} is forbidden: final-holdout paths are never accepted"
        )


def _safe_relative(value: Any, location: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise TrainingRecoveryAnalysisError(f"{location} must be a non-empty relative path")
    path = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        path.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or "\\" in value
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise TrainingRecoveryAnalysisError(f"{location} is unsafe")
    _reject_forbidden(path, location)
    return path


def _lexical_workspace_path(path: Path, location: str) -> tuple[Path, PurePosixPath]:
    workspace = Path.cwd().resolve()
    supplied = Path(path)
    absolute = Path(
        os.path.abspath(os.fspath(supplied if supplied.is_absolute() else workspace / supplied))
    )
    try:
        relative_native = absolute.relative_to(workspace)
    except ValueError as error:
        raise TrainingRecoveryAnalysisError(f"{location} must be inside the workspace") from error
    relative = PurePosixPath(relative_native.as_posix())
    if not relative.parts:
        raise TrainingRecoveryAnalysisError(f"{location} must not be the workspace root")
    _reject_forbidden(relative, location)
    return absolute, relative


def _check_existing_chain(path: Path, location: str, *, regular_file: bool) -> None:
    workspace = Path.cwd().resolve()
    try:
        relative = path.relative_to(workspace)
    except ValueError as error:
        raise TrainingRecoveryAnalysisError(f"{location} must be inside the workspace") from error
    current = workspace
    details: os.stat_result | None = None
    for part in relative.parts:
        current /= part
        try:
            details = os.lstat(current)
        except OSError as error:
            raise TrainingRecoveryAnalysisError(f"{location} is unavailable: {error}") from error
        if stat.S_ISLNK(details.st_mode):
            raise TrainingRecoveryAnalysisError(f"{location} must not contain a symlink")
    if details is None:
        raise TrainingRecoveryAnalysisError(f"{location} is invalid")
    expected = stat.S_ISREG if regular_file else stat.S_ISDIR
    if not expected(details.st_mode):
        kind = "regular file" if regular_file else "directory"
        raise TrainingRecoveryAnalysisError(f"{location} must be a non-symlink {kind}")
    try:
        path.resolve(strict=True).relative_to(workspace)
    except (OSError, RuntimeError, ValueError) as error:
        raise TrainingRecoveryAnalysisError(
            f"{location} must resolve inside the workspace"
        ) from error


def _allowed_input(
    path: Path,
    location: str,
    *,
    prefix: tuple[str, ...],
    file_name: str,
) -> tuple[Path, PurePosixPath]:
    absolute, relative = _lexical_workspace_path(path, location)
    if relative.parts[: len(prefix)] != prefix or len(relative.parts) <= len(prefix):
        raise TrainingRecoveryAnalysisError(
            f"{location} must be under {'/'.join(prefix)} inside the workspace"
        )
    if relative.name != file_name:
        raise TrainingRecoveryAnalysisError(f"{location} must be named {file_name!r}")
    _check_existing_chain(absolute, location, regular_file=True)
    return absolute, relative


def _open_regular(path: Path, location: str) -> BinaryIO:
    _check_existing_chain(path, location, regular_file=True)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise TrainingRecoveryAnalysisError(f"could not open {location}: {error}") from error
    stream = os.fdopen(descriptor, "rb")
    details = os.fstat(stream.fileno())
    if not stat.S_ISREG(details.st_mode):
        stream.close()
        raise TrainingRecoveryAnalysisError(f"{location} must be a regular file")
    return stream


def _stable_bytes(path: Path, location: str, *, maximum: int) -> BoundBytes:
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    with _open_regular(path, location) as stream:
        before = os.fstat(stream.fileno())
        if before.st_size > maximum:
            raise TrainingRecoveryAnalysisError(f"{location} exceeds the safe size limit")
        while chunk := stream.read(1024 * 1024):
            chunks.append(chunk)
            digest.update(chunk)
        after = os.fstat(stream.fileno())
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_nlink,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    payload = b"".join(chunks)
    if identity(before) != identity(after) or len(payload) != after.st_size:
        raise TrainingRecoveryAnalysisError(f"{location} changed while it was read")
    current = os.stat(path, follow_symlinks=False)
    if current.st_dev != after.st_dev or current.st_ino != after.st_ino:
        raise TrainingRecoveryAnalysisError(f"{location} changed while it was read")
    return BoundBytes(path, payload, digest.hexdigest(), len(payload))


def _stable_digest(path: Path, location: str) -> tuple[str, int]:
    digest = hashlib.sha256()
    with _open_regular(path, location) as stream:
        before = os.fstat(stream.fileno())
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
        after = os.fstat(stream.fileno())
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_nlink,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if identity(before) != identity(after):
        raise TrainingRecoveryAnalysisError(f"{location} changed while it was hashed")
    current = os.stat(path, follow_symlinks=False)
    if current.st_dev != after.st_dev or current.st_ino != after.st_ino:
        raise TrainingRecoveryAnalysisError(f"{location} changed while it was hashed")
    return digest.hexdigest(), after.st_size


def _bound_record(bound: BoundBytes, relative: PurePosixPath) -> dict[str, Any]:
    return {
        "path": relative.as_posix(),
        "sha256": bound.sha256,
        "size_bytes": bound.size_bytes,
    }


def _managed_path(root: Path, relative: PurePosixPath, location: str) -> Path:
    target = root.joinpath(*relative.parts)
    try:
        target.relative_to(root)
    except ValueError as error:
        raise TrainingRecoveryAnalysisError(f"{location} escapes its immutable root") from error
    _check_existing_chain(target, location, regular_file=True)
    return target


def _verify_bound_file(
    root: Path,
    record: Mapping[str, Any],
    location: str,
    *,
    read: bool = False,
) -> tuple[PurePosixPath, BoundBytes | None]:
    if set(record) != {"path", "sha256", "size_bytes"}:
        raise TrainingRecoveryAnalysisError(f"{location} has unexpected or missing keys")
    relative = _safe_relative(record.get("path"), f"{location}.path")
    expected_sha = _digest(record.get("sha256"), f"{location}.sha256")
    expected_size = _integer(record.get("size_bytes"), f"{location}.size_bytes")
    target = _managed_path(root, relative, location)
    if read:
        bound = _stable_bytes(target, location, maximum=MAX_JSON_BYTES)
        actual_sha, actual_size = bound.sha256, bound.size_bytes
    else:
        bound = None
        actual_sha, actual_size = _stable_digest(target, location)
    if actual_sha != expected_sha or actual_size != expected_size:
        raise TrainingRecoveryAnalysisError(f"{location} hash or size mismatch")
    return relative, bound


def _scan_declared_paths(value: Any, location: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{location}.{key}"
            if isinstance(item, str) and (key == "path" or key.endswith("_path")):
                _reject_forbidden(PurePosixPath(item.replace("\\", "/")), child)
            else:
                _scan_declared_paths(item, child)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _scan_declared_paths(item, f"{location}[{index}]")


def _inventory_regular_tree(root: Path, location: str) -> set[str]:
    result: set[str] = set()
    for current_text, directories, files in os.walk(root, followlinks=False):
        current = Path(current_text)
        for name in directories:
            target = current / name
            details = target.lstat()
            if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
                raise TrainingRecoveryAnalysisError(
                    f"{location} contains a symlink or special directory"
                )
        for name in files:
            target = current / name
            details = target.lstat()
            if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
                raise TrainingRecoveryAnalysisError(
                    f"{location} contains a symlink or special file"
                )
            result.add(target.relative_to(root).as_posix())
    return result


def _validate_reference(manifest_path: Path) -> ReferenceData:
    bound = _stable_bytes(manifest_path, "training reference manifest", maximum=MAX_JSON_BYTES)
    manifest = _parse_json(bound.payload, "training reference manifest")
    _scan_declared_paths(manifest, "training reference manifest")
    if manifest.get("schema") != REFERENCE_SCHEMA:
        raise TrainingRecoveryAnalysisError("training reference manifest schema is unsupported")
    gate = _object(manifest.get("gate"), "training reference manifest.gate")
    checks = _object(gate.get("checks"), "training reference manifest.gate.checks")
    blockers = _list(
        gate.get("blocking_reasons"), "training reference manifest.gate.blocking_reasons"
    )
    if (
        gate.get("passed") is not True
        or blockers
        or not checks
        or any(value is not True for value in checks.values())
    ):
        raise TrainingRecoveryAnalysisError("training reference manifest gate did not pass")

    statistics_payload = _object(
        manifest.get("source_statistics"), "training reference manifest.source_statistics"
    )
    for index, raw_task in enumerate(
        _list(
            statistics_payload.get("tasks"), "training reference manifest.source_statistics.tasks"
        )
    ):
        task = _object(raw_task, f"training reference task[{index}]")
        if is_configured_final_holdout_task(
            task.get("task_id")
        ) or is_configured_final_holdout_job(task.get("job_id")):
            raise TrainingRecoveryAnalysisError(
                "configured final-holdout identity is forbidden in training diagnostics"
            )

    root = manifest_path.parent
    records: dict[str, dict[str, Any]] = {}
    coco_bound: BoundBytes | None = None
    for index, raw_record in enumerate(
        _list(manifest.get("files"), "training reference manifest.files")
    ):
        record = _object(raw_record, f"training reference manifest.files[{index}]")
        rel = _safe_relative(record.get("path"), f"training reference manifest.files[{index}].path")
        key = rel.as_posix()
        if key == "manifest.json" or key in records:
            raise TrainingRecoveryAnalysisError(f"duplicate or reserved reference path {key!r}")
        _, payload = _verify_bound_file(
            root,
            record,
            f"training reference file {key}",
            read=key == "annotations.coco.json",
        )
        records[key] = record
        if payload is not None:
            coco_bound = payload
    if coco_bound is None:
        raise TrainingRecoveryAnalysisError(
            "training reference does not bind annotations.coco.json"
        )
    actual_tree = _inventory_regular_tree(root, "training reference directory")
    expected_tree = set(records) | {"manifest.json"}
    if actual_tree != expected_tree:
        raise TrainingRecoveryAnalysisError(
            "training reference contains unmanaged or missing files"
        )
    coco = _parse_json(coco_bound.payload, "training reference annotations.coco.json")
    _validate_coco(manifest, coco, records)
    return ReferenceData(root, manifest, bound, coco, coco_bound)


def _validate_coco(
    manifest: Mapping[str, Any],
    coco: Mapping[str, Any],
    records: Mapping[str, Mapping[str, Any]],
) -> None:
    categories = _list(coco.get("categories"), "COCO categories")
    expected_categories = [
        {"id": index, "name": name, "supercategory": "road_object"}
        for index, name in enumerate(REQUIRED_LABELS, start=1)
    ]
    if categories != expected_categories:
        raise TrainingRecoveryAnalysisError("COCO categories differ from the canonical taxonomy")
    info = _object(coco.get("info"), "COCO info")
    if info.get("schema") != REFERENCE_SCHEMA:
        raise TrainingRecoveryAnalysisError("COCO info schema differs from its reference manifest")

    images = _list(coco.get("images"), "COCO images")
    annotations = _list(coco.get("annotations"), "COCO annotations")
    image_ids: set[int] = set()
    image_paths: set[str] = set()
    image_hashes: set[str] = set()
    image_by_id: dict[int, dict[str, Any]] = {}
    for index, raw_image in enumerate(images):
        image = _object(raw_image, f"COCO images[{index}]")
        image_id = _integer(image.get("id"), f"COCO images[{index}].id", minimum=1)
        if image_id in image_ids:
            raise TrainingRecoveryAnalysisError("COCO contains duplicate image IDs")
        image_ids.add(image_id)
        if is_configured_final_holdout_task(image.get("task_id")):
            raise TrainingRecoveryAnalysisError(
                "configured final-holdout task is forbidden in training diagnostics"
            )
        width = _integer(image.get("width"), f"COCO images[{index}].width", minimum=1)
        height = _integer(image.get("height"), f"COCO images[{index}].height", minimum=1)
        del width, height
        path = _safe_relative(image.get("file_name"), f"COCO images[{index}].file_name")
        key = path.as_posix()
        digest = _digest(image.get("sha256"), f"COCO images[{index}].sha256")
        if key in image_paths or digest in image_hashes:
            raise TrainingRecoveryAnalysisError("COCO contains duplicate image paths or hashes")
        image_paths.add(key)
        image_hashes.add(digest)
        if key not in records or records[key].get("sha256") != digest:
            raise TrainingRecoveryAnalysisError(
                f"COCO image {key!r} is not hash-bound by the manifest"
            )
        image_by_id[image_id] = image

    annotation_ids: set[int] = set()
    counts: Counter[str] = Counter()
    for index, raw_annotation in enumerate(annotations):
        annotation = _object(raw_annotation, f"COCO annotations[{index}]")
        annotation_id = _integer(annotation.get("id"), f"COCO annotations[{index}].id", minimum=1)
        if annotation_id in annotation_ids:
            raise TrainingRecoveryAnalysisError("COCO contains duplicate annotation IDs")
        annotation_ids.add(annotation_id)
        image_id = _integer(
            annotation.get("image_id"), f"COCO annotations[{index}].image_id", minimum=1
        )
        if image_id not in image_by_id:
            raise TrainingRecoveryAnalysisError("COCO annotation references an unknown image")
        category_id = _integer(
            annotation.get("category_id"), f"COCO annotations[{index}].category_id", minimum=1
        )
        if not 1 <= category_id <= len(REQUIRED_LABELS):
            raise TrainingRecoveryAnalysisError("COCO annotation has an unknown category")
        bbox = _list(annotation.get("bbox"), f"COCO annotations[{index}].bbox")
        if len(bbox) != 4:
            raise TrainingRecoveryAnalysisError("COCO bbox must have four coordinates")
        x, y, width, height = (
            _number(value, f"COCO annotations[{index}].bbox[{position}]")
            for position, value in enumerate(bbox)
        )
        image = image_by_id[image_id]
        if (
            width <= 0
            or height <= 0
            or x < 0
            or y < 0
            or x + width > image["width"] + 1e-9
            or y + height > image["height"] + 1e-9
        ):
            raise TrainingRecoveryAnalysisError("COCO bbox is empty or outside the image")
        if annotation.get("iscrowd") != 0:
            raise TrainingRecoveryAnalysisError("COCO annotations must have iscrowd=0")
        counts[REQUIRED_LABELS[category_id - 1]] += 1

    declared = _object(manifest.get("counts"), "training reference manifest.counts")
    if _integer(declared.get("images"), "reference counts.images") != len(images):
        raise TrainingRecoveryAnalysisError("reference image count differs from COCO")
    if _integer(declared.get("annotations"), "reference counts.annotations") != len(annotations):
        raise TrainingRecoveryAnalysisError("reference annotation count differs from COCO")
    expected_counts = {name: counts[name] for name in REQUIRED_LABELS}
    if declared.get("annotations_by_category") != expected_counts:
        raise TrainingRecoveryAnalysisError("reference per-class counts differ from COCO")
    if set(records) != image_paths | {"annotations.coco.json"}:
        raise TrainingRecoveryAnalysisError("reference manifest file universe differs from COCO")


def _validate_dataset(
    manifest_path: Path,
    reference: ReferenceData,
) -> tuple[dataset_validator.ValidatedDataset, dict[str, Any], dict[int, str]]:
    manifest_bound = _stable_bytes(manifest_path, "YOLO dataset manifest", maximum=MAX_JSON_BYTES)
    manifest = _parse_json(manifest_bound.payload, "YOLO dataset manifest")
    _scan_declared_paths(manifest, "YOLO dataset manifest")
    if manifest.get("schema") not in SUPPORTED_DATASET_SCHEMAS:
        raise TrainingRecoveryAnalysisError("YOLO dataset manifest schema is unsupported")
    try:
        validated = dataset_validator.validate_yolo_dataset(manifest_path.parent)
    except dataset_validator.CandidateTrainingError as error:
        raise TrainingRecoveryAnalysisError(f"YOLO dataset validation failed: {error}") from error
    if validated.manifest_sha256 != manifest_bound.sha256:
        raise TrainingRecoveryAnalysisError("YOLO dataset manifest changed during validation")

    inputs = _object(manifest.get("inputs"), "YOLO dataset manifest.inputs")
    source_reference = _object(
        inputs.get("reference_manifest"), "dataset inputs.reference_manifest"
    )
    if (
        source_reference.get("schema") != REFERENCE_SCHEMA
        or source_reference.get("sha256") != reference.manifest_bound.sha256
        or source_reference.get("file_name") != "manifest.json"
    ):
        raise TrainingRecoveryAnalysisError("YOLO dataset is not bound to the supplied reference")
    source_coco = _object(inputs.get("coco"), "dataset inputs.coco")
    if (
        source_coco.get("file_name") != "annotations.coco.json"
        or source_coco.get("sha256") != reference.coco_bound.sha256
    ):
        raise TrainingRecoveryAnalysisError("YOLO dataset is not bound to the supplied COCO file")

    yaml_record = next(
        (
            record
            for record in _list(manifest.get("files"), "YOLO dataset manifest.files")
            if isinstance(record, dict) and record.get("path") == "dataset.yaml"
        ),
        None,
    )
    if yaml_record is None:
        raise TrainingRecoveryAnalysisError("YOLO dataset does not bind dataset.yaml")
    _, yaml_bound = _verify_bound_file(
        manifest_path.parent,
        yaml_record,
        "YOLO dataset dataset.yaml",
        read=True,
    )
    assert yaml_bound is not None
    try:
        yaml_payload = yaml.load(yaml_bound.payload.decode("utf-8"), Loader=_UniqueKeyLoader)
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise TrainingRecoveryAnalysisError(f"could not parse dataset.yaml: {error}") from error
    yaml_object = _object(yaml_payload, "dataset.yaml")
    raw_names = yaml_object.get("names")
    if isinstance(raw_names, list):
        names = {index: value for index, value in enumerate(raw_names)}
    else:
        name_object = _object(raw_names, "dataset.yaml.names")
        try:
            names = {int(key): value for key, value in name_object.items()}
        except (TypeError, ValueError) as error:
            raise TrainingRecoveryAnalysisError("dataset.yaml class IDs are invalid") from error
    expected_names = {index: name for index, name in enumerate(validated.model_names)}
    if names != expected_names:
        raise TrainingRecoveryAnalysisError(
            "dataset.yaml class names differ from the validated dataset taxonomy"
        )
    return validated, manifest, dict(sorted(names.items()))


def _candidate_receipt(path: Path) -> tuple[candidate_validator.ValidatedCandidate, CandidateData]:
    root = path.parent
    try:
        validated = candidate_validator._validate_candidate(root)
    except candidate_validator.CandidateFreezeError as error:
        raise TrainingRecoveryAnalysisError(f"candidate validation failed: {error}") from error
    receipt_bound = _stable_bytes(path, f"candidate {root.name} receipt", maximum=MAX_JSON_BYTES)
    if (
        validated.receipt_sha256 != receipt_bound.sha256
        or validated.receipt_size_bytes != receipt_bound.size_bytes
    ):
        raise TrainingRecoveryAnalysisError("candidate receipt changed during validation")
    receipt = _parse_json(receipt_bound.payload, f"candidate {root.name} receipt")
    _scan_declared_paths(receipt, f"candidate {root.name} receipt")

    args_bound: BoundBytes | None = None
    results_bound: BoundBytes | None = None
    artifacts = _object(receipt.get("artifacts"), "candidate receipt.artifacts")
    for name, raw_record in artifacts.items():
        record = _object(raw_record, f"candidate artifact {name}")
        rel = _safe_relative(record.get("path"), f"candidate artifact {name}.path")
        expected_sha = _digest(record.get("sha256"), f"candidate artifact {name}.sha256")
        expected_size = _integer(
            record.get("size_bytes"), f"candidate artifact {name}.size_bytes", minimum=1
        )
        target = _managed_path(root, rel, f"candidate artifact {name}")
        if name in {"args", "results"}:
            bound = _stable_bytes(
                target,
                f"candidate artifact {name}",
                maximum=MAX_TEXT_ARTIFACT_BYTES,
            )
            actual_sha, actual_size = bound.sha256, bound.size_bytes
            if name == "args":
                args_bound = bound
            else:
                results_bound = bound
        else:
            actual_sha, actual_size = _stable_digest(target, f"candidate artifact {name}")
        if actual_sha != expected_sha or actual_size != expected_size:
            raise TrainingRecoveryAnalysisError(f"candidate artifact {name} hash or size mismatch")
    if args_bound is None or results_bound is None:
        raise TrainingRecoveryAnalysisError("candidate does not bind args.yaml and results.csv")

    try:
        args_payload = yaml.load(args_bound.payload.decode("utf-8"), Loader=_UniqueKeyLoader)
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise TrainingRecoveryAnalysisError(
            f"could not parse candidate args.yaml: {error}"
        ) from error
    args = _object(args_payload, "candidate args.yaml")
    _validate_args_against_receipt(args, receipt)
    curve = _parse_curve(results_bound.payload, receipt)
    return validated, CandidateData(validated.seed, receipt, receipt_bound, args, curve)


def _normalize_cache(value: Any) -> Any:
    return "none" if value is False else value


def _validate_args_against_receipt(args: Mapping[str, Any], receipt: Mapping[str, Any]) -> None:
    protocol = _object(receipt.get("protocol"), "candidate receipt.protocol")
    training = _object(protocol.get("training"), "candidate receipt.protocol.training")
    resolved = _object(receipt.get("resolved_args"), "candidate receipt.resolved_args")
    seed = _integer(resolved.get("seed"), "candidate receipt.resolved_args.seed")
    if args.get("seed") != seed:
        raise TrainingRecoveryAnalysisError("args.yaml seed differs from its receipt")
    for name, expected in training.items():
        if name not in args:
            raise TrainingRecoveryAnalysisError(f"args.yaml is missing protocol argument {name!r}")
        actual = _normalize_cache(args[name]) if name == "cache" else args[name]
        if not _same_contract(actual, expected):
            raise TrainingRecoveryAnalysisError(
                f"args.yaml argument {name!r} differs from protocol"
            )
    for name in LEARNING_ARGS:
        if name not in args:
            raise TrainingRecoveryAnalysisError(f"args.yaml is missing learning argument {name!r}")
        value = args[name]
        if value is None or not isinstance(value, (str, bool, int, float)):
            raise TrainingRecoveryAnalysisError(
                f"args.yaml argument {name!r} must be a JSON scalar"
            )
        if isinstance(value, float) and not math.isfinite(value):
            raise TrainingRecoveryAnalysisError(f"args.yaml argument {name!r} is non-finite")


def _parse_curve(encoded: bytes, receipt: Mapping[str, Any]) -> dict[str, Any]:
    try:
        text = encoded.decode("utf-8")
    except UnicodeDecodeError as error:
        raise TrainingRecoveryAnalysisError("results.csv is not UTF-8") from error
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise TrainingRecoveryAnalysisError("results.csv is empty")
    fields = [field.strip() for field in reader.fieldnames]
    if len(fields) != len(set(fields)) or any(not field for field in fields):
        raise TrainingRecoveryAnalysisError("results.csv has duplicate or empty columns")
    required = {
        "epoch",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    }
    if not required.issubset(fields):
        raise TrainingRecoveryAnalysisError("results.csv is missing required metric columns")
    rows: list[dict[str, float]] = []
    for row_index, raw in enumerate(reader, start=1):
        row: dict[str, float] = {}
        normalized = {key.strip(): value for key, value in raw.items() if key is not None}
        for key in fields:
            try:
                value = float(normalized[key])
            except (KeyError, TypeError, ValueError) as error:
                raise TrainingRecoveryAnalysisError(
                    f"results.csv row {row_index} field {key!r} is invalid"
                ) from error
            if not math.isfinite(value):
                raise TrainingRecoveryAnalysisError(
                    f"results.csv row {row_index} field {key!r} is non-finite"
                )
            row[key] = value
        rows.append(row)
    if not rows:
        raise TrainingRecoveryAnalysisError("results.csv has no epochs")
    best = max(rows, key=lambda row: (row["metrics/mAP50-95(B)"], -row["epoch"]))
    best_epoch = _object(receipt.get("best_epoch"), "candidate receipt.best_epoch")
    if _integer(best_epoch.get("epochs_recorded"), "receipt.best_epoch.epochs_recorded") != len(
        rows
    ):
        raise TrainingRecoveryAnalysisError("results.csv row count differs from receipt")
    if _integer(best_epoch.get("number"), "receipt.best_epoch.number", minimum=1) != int(
        best["epoch"]
    ):
        raise TrainingRecoveryAnalysisError("results.csv best mAP50-95 epoch differs from receipt")
    lr_values = [row["lr/pg0"] for row in rows if "lr/pg0" in row]
    result = {
        "epochs_recorded": len(rows),
        "best_epoch_by_map50_95": int(best["epoch"]),
        "best_map50_95": _round(best["metrics/mAP50-95(B)"]),
        "end_map50_95": _round(rows[-1]["metrics/mAP50-95(B)"]),
        "best_to_end_change": _round(rows[-1]["metrics/mAP50-95(B)"] - best["metrics/mAP50-95(B)"]),
    }
    if lr_values:
        result["peak_lr_pg0"] = _round(max(lr_values))
        result["lr_pg0_at_best_epoch"] = _round(best["lr/pg0"])
    return result


def _validate_candidate_contracts(
    validated: Sequence[candidate_validator.ValidatedCandidate],
    candidates: Sequence[CandidateData],
    dataset: dataset_validator.ValidatedDataset,
) -> None:
    try:
        candidate_validator._validate_common_contract(validated)
    except candidate_validator.CandidateFreezeError as error:
        raise TrainingRecoveryAnalysisError(f"candidate contracts differ: {error}") from error
    for candidate in candidates:
        dataset_claim = _object(
            _object(candidate.receipt.get("inputs"), "candidate receipt.inputs").get("dataset"),
            "candidate receipt.inputs.dataset",
        )
        manifest_claim = _object(dataset_claim.get("manifest"), "candidate dataset manifest claim")
        if (
            manifest_claim.get("sha256") != dataset.manifest_sha256
            or manifest_claim.get("size_bytes") != dataset.manifest_size_bytes
            or manifest_claim.get("schema") != dataset.schema
            or dataset_claim.get("managed_files_sha256") != dataset.managed_files_sha256
            or dataset_claim.get("managed_file_count") != len(dataset.files)
            or not _same_contract(dataset_claim.get("counts"), dataset.counts)
        ):
            raise TrainingRecoveryAnalysisError(
                f"candidate seed {candidate.seed} is not bound to the supplied YOLO dataset"
            )
        yaml_claim = _object(dataset_claim.get("dataset_yaml"), "candidate dataset.yaml claim")
        if (
            yaml_claim.get("sha256") != dataset.dataset_yaml.sha256
            or yaml_claim.get("size_bytes") != dataset.dataset_yaml.size_bytes
        ):
            raise TrainingRecoveryAnalysisError(
                f"candidate seed {candidate.seed} dataset.yaml contract differs"
            )


def _image_splits(
    coco: Mapping[str, Any], dataset_manifest: Mapping[str, Any]
) -> tuple[dict[int, str], dict[str, Mapping[str, Any]]]:
    by_basename: dict[str, dict[str, Any]] = {}
    for raw_image in _list(coco.get("images"), "COCO images"):
        image = _object(raw_image, "COCO image")
        basename = PurePosixPath(str(image["file_name"])).name.casefold()
        if basename in by_basename:
            raise TrainingRecoveryAnalysisError("COCO image basenames are not unique")
        by_basename[basename] = image
    split_by_id: dict[int, str] = {}
    dataset_images: dict[str, Mapping[str, Any]] = {}
    for index, raw_record in enumerate(_list(dataset_manifest.get("files"), "dataset files")):
        record = _object(raw_record, f"dataset files[{index}]")
        path = _safe_relative(record.get("path"), f"dataset files[{index}].path")
        if (
            len(path.parts) != 3
            or path.parts[0] != "images"
            or path.parts[1]
            not in {
                "train",
                "val",
            }
        ):
            continue
        basename = path.name.casefold()
        image = by_basename.get(basename)
        if image is None:
            raise TrainingRecoveryAnalysisError("YOLO image is absent from supplied COCO reference")
        image_id = int(image["id"])
        if image_id in split_by_id:
            raise TrainingRecoveryAnalysisError("COCO image occurs in multiple YOLO splits")
        if record.get("sha256") != image.get("sha256"):
            raise TrainingRecoveryAnalysisError("YOLO and COCO image hashes differ")
        split_by_id[image_id] = path.parts[1]
        dataset_images[path.as_posix()] = record
    if set(split_by_id) != {int(image["id"]) for image in by_basename.values()}:
        raise TrainingRecoveryAnalysisError(
            "YOLO split does not exactly cover the COCO image universe"
        )
    return split_by_id, dataset_images


def _typed_asset(value: Any, location: str) -> tuple[str, str]:
    if isinstance(value, bool) or not isinstance(value, (str, int)) or value == "":
        raise TrainingRecoveryAnalysisError(f"{location} must be a string or integer")
    return type(value).__name__, str(value)


def _display_asset(key: tuple[str, str]) -> str | int:
    return int(key[1]) if key[0] == "int" else key[1]


def _quantiles(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)

    def value_at(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)

    return {
        name: _round(value_at(fraction))
        for name, fraction in (("min", 0.0), ("p10", 0.1), ("p50", 0.5), ("p90", 0.9), ("max", 1.0))
    }


def _round(value: float, digits: int = 6) -> float:
    result = round(float(value), digits)
    return 0.0 if result == 0 else result


def _ratio(numerator: float, denominator: float) -> float | None:
    return _round(numerator / denominator) if denominator else None


def _dataset_diagnostics(
    reference: ReferenceData,
    dataset_manifest: Mapping[str, Any],
    *,
    imgsz: int,
) -> dict[str, Any]:
    coco = reference.coco
    images = {int(image["id"]): image for image in _list(coco.get("images"), "COCO images")}
    split_by_id, _ = _image_splits(coco, dataset_manifest)
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for raw_annotation in _list(coco.get("annotations"), "COCO annotations"):
        annotation = _object(raw_annotation, "COCO annotation")
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    split_counts: dict[str, Counter[str]] = {"train": Counter(), "val": Counter()}
    source_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    source_images: Counter[tuple[str, str]] = Counter()
    source_splits: dict[tuple[str, str], str] = {}
    density: dict[str, list[float]] = {"train": [], "val": []}
    zero_images = Counter()
    bbox_values: dict[tuple[str, str], list[tuple[float, float, float]]] = defaultdict(list)
    overall_bbox: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    category_names = {index: name for index, name in enumerate(REQUIRED_LABELS, start=1)}

    for image_id, image in images.items():
        split = split_by_id[image_id]
        asset = _typed_asset(image.get("source_asset_id"), f"COCO image {image_id}.source_asset_id")
        previous_split = source_splits.setdefault(asset, split)
        if previous_split != split:
            raise TrainingRecoveryAnalysisError("one source asset crosses YOLO splits")
        source_images[asset] += 1
        image_annotations = annotations_by_image[image_id]
        density[split].append(float(len(image_annotations)))
        if not image_annotations:
            zero_images[split] += 1
        scale = min(imgsz / int(image["width"]), imgsz / int(image["height"]))
        for annotation in image_annotations:
            name = category_names[int(annotation["category_id"])]
            split_counts[split][name] += 1
            source_counts[asset][name] += 1
            _, _, width, height = (float(value) for value in annotation["bbox"])
            resized_width = width * scale
            resized_height = height * scale
            item = (resized_width, resized_height, math.sqrt(resized_width * resized_height))
            bbox_values[(split, name)].append(item)
            overall_bbox[split].append(item)

    declared = _object(dataset_manifest.get("counts"), "YOLO dataset manifest.counts")
    declared_images = _object(declared.get("images"), "YOLO dataset manifest.counts.images")
    declared_annotations = _object(
        declared.get("annotations"), "YOLO dataset manifest.counts.annotations"
    )
    declared_by_split = _object(
        declared_annotations.get("by_split_and_category"),
        "YOLO dataset manifest.counts.annotations.by_split_and_category",
    )
    for split in ("train", "val"):
        computed = {name: split_counts[split][name] for name in REQUIRED_LABELS}
        if declared_by_split.get(split) != computed:
            raise TrainingRecoveryAnalysisError(f"YOLO dataset {split} counts differ from COCO")
        if declared_images.get(split) != len(density[split]):
            raise TrainingRecoveryAnalysisError(
                f"YOLO dataset {split} image count differs from COCO"
            )

    sources = []
    for asset in sorted(source_images, key=lambda item: (item[0], item[1])):
        counts = {name: source_counts[asset][name] for name in REQUIRED_LABELS}
        sources.append(
            {
                "source_asset_id": _display_asset(asset),
                "split": source_splits[asset],
                "image_count": source_images[asset],
                "annotation_count": sum(counts.values()),
                "annotations_by_class": counts,
            }
        )

    total_train = sum(split_counts["train"].values())
    total_val = sum(split_counts["val"].values())
    distribution_shift = (
        0.5
        * sum(
            abs(split_counts["train"][name] / total_train - split_counts["val"][name] / total_val)
            for name in REQUIRED_LABELS
        )
        if total_train and total_val
        else None
    )
    nonzero_train = [value for value in split_counts["train"].values() if value]
    imbalance_ratio = max(nonzero_train) / min(nonzero_train) if nonzero_train else 0.0

    class_source_coverage = {}
    for name in REQUIRED_LABELS:
        counts = [
            (asset, source_counts[asset][name])
            for asset in source_counts
            if source_counts[asset][name]
        ]
        counts.sort(key=lambda item: (-item[1], item[0]))
        total = sum(value for _, value in counts)
        class_source_coverage[name] = {
            "source_count": len(counts),
            "dominant_source_asset_id": _display_asset(counts[0][0]) if counts else None,
            "dominant_source_fraction": _ratio(counts[0][1], total) if counts else None,
        }

    def bbox_summary(values: Sequence[tuple[float, float, float]]) -> dict[str, Any]:
        small = sum(area_side < 32 for _, _, area_side in values)
        min_dimension = sum(width < 8 or height < 8 for width, height, _ in values)
        return {
            "count": len(values),
            "resized_width_px": _quantiles([item[0] for item in values]),
            "resized_height_px": _quantiles([item[1] for item in values]),
            "resized_sqrt_area_px": _quantiles([item[2] for item in values]),
            "sqrt_area_below_32px_count": small,
            "sqrt_area_below_32px_fraction": _ratio(small, len(values)),
            "one_dimension_below_8px_count": min_dimension,
            "one_dimension_below_8px_fraction": _ratio(min_dimension, len(values)),
        }

    split_payload = {}
    bbox_payload = {}
    for split in ("train", "val"):
        class_counts = {name: split_counts[split][name] for name in REQUIRED_LABELS}
        split_payload[split] = {
            "source_count": len(
                {asset for asset, value in source_splits.items() if value == split}
            ),
            "image_count": len(density[split]),
            "annotation_count": sum(class_counts.values()),
            "zero_annotation_image_count": zero_images[split],
            "annotations_per_image": _quantiles(density[split]),
            "annotations_by_class": class_counts,
        }
        bbox_payload[split] = {
            "overall": bbox_summary(overall_bbox[split]),
            "by_class": {
                name: bbox_summary(bbox_values[(split, name)]) for name in REQUIRED_LABELS
            },
        }

    return {
        "totals": {
            "source_count": len(source_images),
            "image_count": len(images),
            "annotation_count": sum(len(value) for value in annotations_by_image.values()),
            "taxonomy": list(REQUIRED_LABELS),
        },
        "splits": split_payload,
        "imbalance": {
            "train_max_to_min_nonzero_class_ratio": _round(imbalance_ratio),
            "train_validation_class_distribution_total_variation": (
                _round(distribution_shift) if distribution_shift is not None else None
            ),
            "validation_source_fraction": _ratio(
                split_payload["val"]["source_count"], len(source_images)
            ),
            "validation_image_fraction": _ratio(split_payload["val"]["image_count"], len(images)),
        },
        "sources": sources,
        "class_source_coverage": class_source_coverage,
        "bounding_boxes": {
            "training_image_size": imgsz,
            "size_definition": (
                "COCO boxes scaled by min(imgsz/image_width, imgsz/image_height), before padding"
            ),
            "splits": bbox_payload,
        },
    }


def _metric_stats(values: Sequence[float]) -> dict[str, Any]:
    mean = statistics.fmean(values)
    standard_deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "count": len(values),
        "mean": _round(mean),
        "sample_standard_deviation": _round(standard_deviation),
        "minimum": _round(min(values)),
        "maximum": _round(max(values)),
        "range": _round(max(values) - min(values)),
        "coefficient_of_variation": _round(standard_deviation / mean) if mean else None,
    }


def _learning_args(candidates: Sequence[CandidateData]) -> dict[str, Any]:
    per_seed: list[dict[str, Any]] = []
    common: dict[str, Any] = {}
    for name in LEARNING_ARGS:
        values = [candidate.args[name] for candidate in candidates]
        common[name] = (
            values[0] if all(_same_contract(values[0], value) for value in values[1:]) else None
        )
    for candidate in sorted(candidates, key=lambda item: item.seed):
        per_seed.append(
            {
                "seed": candidate.seed,
                "args": {name: candidate.args[name] for name in LEARNING_ARGS},
                "curve": candidate.curve,
            }
        )
    optimizer = common.get("optimizer")
    lr0 = common.get("lr0")
    nc = len(REQUIRED_LABELS)
    auto_reference = round(0.002 * 5 / (4 + nc), 6)
    lr_diagnostic = {
        "optimizer": optimizer,
        "configured_lr0": lr0,
        "adamw_documented_reference_lr0": 0.001 if optimizer == "AdamW" else None,
        "ultralytics_auto_reference_lr0_for_eight_classes": auto_reference,
        "ratio_to_adamw_documented_reference": (
            _round(float(lr0) / 0.001)
            if optimizer == "AdamW" and isinstance(lr0, (int, float))
            else None
        ),
        "ratio_to_ultralytics_auto_reference": (
            _round(float(lr0) / auto_reference)
            if optimizer == "AdamW" and isinstance(lr0, (int, float))
            else None
        ),
    }
    protocol_fields = set(
        _object(
            _object(candidates[0].receipt.get("protocol"), "candidate receipt.protocol").get(
                "training"
            ),
            "candidate receipt.protocol.training",
        )
    )
    return {
        "common_args": common,
        "all_learning_args_equal_except_seed": all(value is not None for value in common.values()),
        "per_seed": per_seed,
        "learning_rate_diagnostic": lr_diagnostic,
        "semantic_protocol_fields": sorted(protocol_fields),
        "observed_args_not_semantically_frozen_in_v1_protocol": sorted(
            set(LEARNING_ARGS) - protocol_fields
        ),
    }


def _mapping_diagnostic(training_names: Mapping[int, str]) -> dict[str, Any]:
    rows = []
    exact_count = 0
    for class_id, canonical_name in enumerate(REQUIRED_LABELS):
        training_name = training_names[class_id]
        pretrained_name = PRETRAINED_CLASS_NAMES[canonical_name]
        exact = training_name.strip().casefold() == pretrained_name.strip().casefold()
        exact_count += int(exact)
        relation = "exact"
        if canonical_name in {"pedestrian", "traffic_light"} and not exact:
            relation = "semantic_alias"
        elif canonical_name == "traffic_sign" and not exact:
            relation = "partial_semantic_alias"
        rows.append(
            {
                "class_id": class_id,
                "canonical_name": canonical_name,
                "training_name": training_name,
                "pretrained_name": pretrained_name,
                "exact_name_match": exact,
                "relation_when_not_exact": None if exact else relation,
            }
        )
    return {
        "matching_rule": "case-insensitive exact name after trimming whitespace",
        "exact_match_count": exact_count,
        "class_count": len(REQUIRED_LABELS),
        "exact_match_fraction": _ratio(exact_count, len(REQUIRED_LABELS)),
        "classes": rows,
    }


def _seed_diagnostics(candidates: Sequence[CandidateData]) -> dict[str, Any]:
    sorted_candidates = sorted(candidates, key=lambda item: item.seed)
    aggregate_by_seed = []
    metric_values: dict[str, list[float]] = defaultdict(list)
    per_class_values: dict[str, dict[str, list[float]]] = {
        name: defaultdict(list) for name in REQUIRED_LABELS
    }
    for candidate in sorted_candidates:
        metrics = _object(candidate.receipt.get("metrics"), "candidate receipt.metrics")
        aggregate = _object(metrics.get("aggregate"), "candidate receipt.metrics.aggregate")
        per_class = _object(metrics.get("per_class"), "candidate receipt.metrics.per_class")
        aggregate_row = {}
        for metric in AGGREGATE_METRICS:
            value = _number(aggregate.get(metric), f"candidate aggregate {metric}", minimum=0)
            if value > 1:
                raise TrainingRecoveryAnalysisError("candidate aggregate metric exceeds one")
            metric_values[metric].append(value)
            aggregate_row[metric] = value
        aggregate_by_seed.append({"seed": candidate.seed, **aggregate_row})
        if set(per_class) != set(REQUIRED_LABELS):
            raise TrainingRecoveryAnalysisError("candidate per-class metrics are incomplete")
        for name in REQUIRED_LABELS:
            row = _object(per_class[name], f"candidate per-class {name}")
            for metric in AGGREGATE_METRICS:
                value = _number(row.get(metric), f"candidate {name}.{metric}", minimum=0)
                if value > 1:
                    raise TrainingRecoveryAnalysisError("candidate per-class metric exceeds one")
                per_class_values[name][metric].append(value)
    return {
        "aggregate_by_seed": aggregate_by_seed,
        "aggregate_statistics": {
            metric: _metric_stats(metric_values[metric]) for metric in AGGREGATE_METRICS
        },
        "per_class_statistics": {
            name: {
                metric: _metric_stats(per_class_values[name][metric])
                for metric in AGGREGATE_METRICS
            }
            for name in REQUIRED_LABELS
        },
    }


def _diagnostic_flags(
    dataset: Mapping[str, Any],
    learning: Mapping[str, Any],
    mapping: Mapping[str, Any],
    seeds: Mapping[str, Any],
) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    splits = _object(dataset.get("splits"), "dataset diagnostics.splits")
    imbalance = _object(dataset.get("imbalance"), "dataset diagnostics.imbalance")
    if splits["val"]["source_count"] == 1:
        flags.append({"code": "SINGLE_SOURCE_VALIDATION", "severity": "high"})
    distribution_shift = imbalance["train_validation_class_distribution_total_variation"]
    if isinstance(distribution_shift, (int, float)) and distribution_shift >= 0.25:
        flags.append({"code": "TRAIN_VALIDATION_CLASS_SHIFT", "severity": "high"})
    if imbalance["train_max_to_min_nonzero_class_ratio"] >= 10:
        flags.append({"code": "TRAIN_CLASS_IMBALANCE", "severity": "high"})
    if splits["val"]["zero_annotation_image_count"] == 0:
        flags.append({"code": "VALIDATION_HAS_NO_NEGATIVE_IMAGES", "severity": "medium"})
    if any(
        row["source_count"] < 2
        for row in _object(dataset.get("class_source_coverage"), "class source coverage").values()
    ):
        flags.append({"code": "CLASS_CONFINED_TO_ONE_SOURCE", "severity": "high"})
    lr = _object(learning.get("learning_rate_diagnostic"), "learning rate diagnostic")
    if (
        lr.get("optimizer") == "AdamW"
        and isinstance(lr.get("ratio_to_adamw_documented_reference"), (int, float))
        and lr["ratio_to_adamw_documented_reference"] > 2
    ):
        flags.append({"code": "ADAMW_LR0_ABOVE_REFERENCE", "severity": "high"})
    if mapping.get("exact_match_count") != mapping.get("class_count"):
        flags.append({"code": "INCOMPLETE_PRETRAINED_CLASS_NAME_MAPPING", "severity": "high"})
    map_cv = seeds["aggregate_statistics"]["map50_95"]["coefficient_of_variation"]
    if isinstance(map_cv, (int, float)) and map_cv >= 0.2:
        flags.append({"code": "HIGH_SEED_VARIANCE", "severity": "high"})
    bbox = dataset["bounding_boxes"]["splits"]["val"]["overall"]
    if (
        isinstance(bbox["sqrt_area_below_32px_fraction"], (int, float))
        and bbox["sqrt_area_below_32px_fraction"] >= 0.5
    ):
        flags.append({"code": "SMALL_OBJECT_DOMINATED_VALIDATION", "severity": "high"})
    return sorted(flags, key=lambda item: (item["severity"], item["code"]))


def _open_output_parent(path: Path, *, create: bool) -> int | None:
    """Open an output parent by directory FDs, optionally creating missing levels."""

    workspace = Path.cwd().resolve()
    relative_parent = path.parent.relative_to(workspace)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(workspace, flags)
    try:
        for part in relative_parent.parts:
            try:
                child = os.open(part, flags, dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    os.close(descriptor)
                    return None
                try:
                    os.mkdir(part, mode=0o755, dir_fd=descriptor)
                except FileExistsError:
                    pass
                child = os.open(part, flags, dir_fd=descriptor)
            except OSError as error:
                raise TrainingRecoveryAnalysisError(
                    f"output parent must contain only non-symlink directories: {error}"
                ) from error
            os.close(descriptor)
            descriptor = child
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _prepare_output(path: Path) -> tuple[Path, PurePosixPath]:
    absolute, relative = _lexical_workspace_path(path, "output")
    if os.path.lexists(absolute):
        raise FileExistsError(f"output already exists: {absolute}")
    parent_fd = _open_output_parent(absolute, create=False)
    if parent_fd is not None:
        os.close(parent_fd)
    return absolute, relative


def _atomic_write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = _json_bytes(payload)
    parent_fd = _open_output_parent(path, create=True)
    assert parent_fd is not None
    temporary = f".{path.name}.{secrets.token_hex(12)}.tmp"
    descriptor: int | None = None
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(temporary, flags, 0o600, dir_fd=parent_fd)
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = None
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(
                temporary,
                path.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError as error:
            raise FileExistsError(f"output already exists: {path}") from error
        os.fsync(parent_fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        os.close(parent_fd)


def analyze_training_recovery(
    *,
    training_reference_manifest: Path,
    dataset_manifest: Path,
    candidate_receipts: Sequence[Path],
    output: Path,
) -> dict[str, Any]:
    """Validate training-only inputs and atomically publish deterministic diagnostics."""

    reference_path, reference_relative = _allowed_input(
        training_reference_manifest,
        "training reference manifest",
        prefix=("data", "ground-truth"),
        file_name="manifest.json",
    )
    dataset_path, dataset_relative = _allowed_input(
        dataset_manifest,
        "YOLO dataset manifest",
        prefix=("data", "training"),
        file_name="manifest.json",
    )
    if len(candidate_receipts) != 3:
        raise TrainingRecoveryAnalysisError(
            "exactly three training candidate receipts are required"
        )
    candidate_paths: list[tuple[Path, PurePosixPath]] = []
    seen_candidates: set[PurePosixPath] = set()
    for index, candidate in enumerate(candidate_receipts):
        path, relative = _allowed_input(
            candidate,
            f"candidate receipt[{index}]",
            prefix=("data", "model-candidates"),
            file_name="receipt.json",
        )
        if relative in seen_candidates:
            raise TrainingRecoveryAnalysisError("candidate receipt paths must be distinct")
        seen_candidates.add(relative)
        candidate_paths.append((path, relative))
    output_path, _output_relative = _prepare_output(output)

    reference = _validate_reference(reference_path)
    validated_dataset, dataset_payload, training_names = _validate_dataset(dataset_path, reference)
    validated_candidates: list[candidate_validator.ValidatedCandidate] = []
    candidates: list[CandidateData] = []
    candidate_relatives: dict[int, PurePosixPath] = {}
    for path, relative in candidate_paths:
        validated, candidate = _candidate_receipt(path)
        validated_candidates.append(validated)
        candidates.append(candidate)
        candidate_relatives[candidate.seed] = relative
    _validate_candidate_contracts(validated_candidates, candidates, validated_dataset)

    imgsz_values = {int(candidate.args["imgsz"]) for candidate in candidates}
    if len(imgsz_values) != 1:
        raise TrainingRecoveryAnalysisError("candidate training image sizes differ")
    dataset_diagnostics = _dataset_diagnostics(
        reference,
        dataset_payload,
        imgsz=next(iter(imgsz_values)),
    )
    learning = _learning_args(candidates)
    mapping = _mapping_diagnostic(training_names)
    seeds = _seed_diagnostics(candidates)
    flags = _diagnostic_flags(dataset_diagnostics, learning, mapping, seeds)

    payload = {
        "schema": ANALYSIS_SCHEMA,
        "scope": {
            "input_scope": "training_only",
            "holdout_access": "prohibited",
            "final_holdout_access": "prohibited",
            "mutation_performed": False,
        },
        "inputs": {
            "training_reference_manifest": _bound_record(
                reference.manifest_bound, reference_relative
            ),
            "training_coco": {
                "path": (reference_relative.parent / "annotations.coco.json").as_posix(),
                "sha256": reference.coco_bound.sha256,
                "size_bytes": reference.coco_bound.size_bytes,
                "read_via": "training_reference_manifest.files",
            },
            "yolo_dataset_manifest": {
                "path": dataset_relative.as_posix(),
                "sha256": validated_dataset.manifest_sha256,
                "size_bytes": validated_dataset.manifest_size_bytes,
            },
            "candidate_receipts": [
                {
                    "seed": candidate.seed,
                    "path": candidate_relatives[candidate.seed].as_posix(),
                    "sha256": candidate.receipt_bound.sha256,
                    "size_bytes": candidate.receipt_bound.size_bytes,
                }
                for candidate in sorted(candidates, key=lambda item: item.seed)
            ],
        },
        "verification": {
            "training_reference_gate_and_all_managed_hashes_verified": True,
            "yolo_dataset_gate_and_all_managed_hashes_verified": True,
            "reference_to_dataset_contract_verified": True,
            "candidate_receipts_and_all_artifact_hashes_verified": True,
            "candidate_dataset_and_protocol_contracts_identical": True,
            "final_holdout_path_read": False,
        },
        "dataset": dataset_diagnostics,
        "training": {
            "learning_arguments": learning,
            "pretrained_class_name_mapping": mapping,
            "seed_metrics": seeds,
        },
        "diagnosis": {
            "status": "RECOVERY_REQUIRED" if flags else "NO_BLOCKING_DIAGNOSTIC_FLAG",
            "flag_count": len(flags),
            "flags": flags,
        },
    }
    _atomic_write_json_new(output_path, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-reference-manifest", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument(
        "--candidate-receipt",
        action="append",
        required=True,
        type=Path,
        dest="candidate_receipts",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = analyze_training_recovery(
        training_reference_manifest=args.training_reference_manifest,
        dataset_manifest=args.dataset_manifest,
        candidate_receipts=args.candidate_receipts,
        output=args.output,
    )
    print(
        json.dumps(
            {
                "status": payload["diagnosis"]["status"],
                "flag_count": payload["diagnosis"]["flag_count"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
