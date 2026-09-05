"""Evaluate one training-only YOLO candidate on its immutable validation split.

This evaluator accepts only inputs below ``data/training`` and
``data/model-candidates``.  Final-holdout paths and configured holdout
identifiers are rejected before any input file is opened.  The complete dataset
inventory is hash-verified, while only validation images and labels are decoded
for inference and scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import numbers
import os
import re
import stat
import tempfile
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Protocol

import yaml
from PIL import Image

from roadlabelops.holdout_policy import (
    FINAL_HOLDOUT_REJECTED_SCOPES,
    NO_FINAL_HOLDOUT_STATEMENT,
    final_holdout_scope_reason,
)
from roadlabelops.tools.detection import MODEL_MAPPING, ROAD_LABELS, postprocess_predictions
from roadlabelops.tools.quality import calculate_quality

LEGACY_DATASET_SCHEMA = {"name": "roadlabelops.yolo-dataset", "version": 2}
DATASET_SCHEMA = {"name": "roadlabelops.yolo-dataset", "version": 3}
LEGACY_CANDIDATE_SCHEMA = {"name": "roadlabelops.yolo-candidate-training", "version": 1}
CANDIDATE_SCHEMA = {"name": "roadlabelops.yolo-candidate-training", "version": 2}
LEGACY_PROTOCOL_SCHEMA = {
    "name": "roadlabelops.yolo-candidate-training-protocol",
    "version": 1,
}
PROTOCOL_SCHEMA = {
    "name": "roadlabelops.yolo-candidate-training-protocol",
    "version": 2,
}
OUTPUT_SCHEMA = {"name": "roadlabelops.training-validation-evaluation", "version": 1}
SPLIT_PLAN_SCHEMA = {"name": "roadlabelops.training-asset-split", "version": 1}

CANONICAL_LABELS = tuple(ROAD_LABELS)
MODEL_LABELS = (
    "car",
    "bus",
    "truck",
    "motorcycle",
    "bicycle",
    "person",
    "traffic light",
    "stop sign",
)
MODEL_TO_CANONICAL = dict(zip(MODEL_LABELS, CANONICAL_LABELS, strict=True))
LEGACY_MODEL_TO_CANONICAL = {name: name for name in CANONICAL_LABELS}
NO_HOLDOUT_STATEMENT = NO_FINAL_HOLDOUT_STATEMENT
SUPPORTED_IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})
REQUIRED_DATASET_GATE_CHECKS = frozenset(
    {
        "annotations_and_bboxes_valid",
        "asset_split_is_nonempty_disjoint_and_complete",
        "fixed_eight_class_taxonomy",
        "output_basenames_unique",
        "reference_manifest_gate_and_files_verified",
        "reference_manifest_source_assets_verified",
        "source_asset_never_crosses_split",
        "source_image_hashes_unique",
        "source_images_hash_and_dimensions_verified",
        "source_leakage_group_never_crosses_split",
        "staged_file_hashes_verified",
    }
)
CURRENT_DATASET_GATE_CHECKS = REQUIRED_DATASET_GATE_CHECKS | {
    "canonical_and_model_taxonomies_bound"
}
LEGACY_CANDIDATE_GATE_CHECKS = frozenset(
    {
        "schema_v2_dataset_manifest_verified",
        "dataset_gate_verified",
        "all_managed_dataset_files_verified",
        "dataset_yaml_hash_and_taxonomy_verified",
        "base_yolo11n_weight_hash_verified",
        "isolated_workspace_training_verified",
        "trainer_args_match_frozen_protocol",
        "finite_complete_eight_class_metrics_verified",
        "required_training_artifacts_verified",
        "source_inputs_unchanged",
        "holdout_input_not_read",
    }
)
PRE_SUPPORT_AWARE_CANDIDATE_GATE_CHECKS = (
    LEGACY_CANDIDATE_GATE_CHECKS - {"schema_v2_dataset_manifest_verified"}
) | {"supported_dataset_manifest_schema_verified"}
CURRENT_CANDIDATE_GATE_CHECKS = (
    PRE_SUPPORT_AWARE_CANDIDATE_GATE_CHECKS - {"finite_complete_eight_class_metrics_verified"}
) | {"support_aware_complete_eight_class_metrics_verified"}
SUPPORT_AWARE_CANDIDATE_GATE_CHECKS = CURRENT_CANDIDATE_GATE_CHECKS
CURRENT_CANDIDATE_GATE_CHECKS = CURRENT_CANDIDATE_GATE_CHECKS | {
    "pretrained_class_head_transfer_verified"
}
RECOVERY_ARM_SIGNATURES = {
    "repaired_control": {"imgsz": 640, "lr0": 0.001, "cls_pw": 0.0},
    "small_target_960": {"imgsz": 960, "lr0": 0.001, "cls_pw": 0.0},
    "class_balance_025": {"imgsz": 640, "lr0": 0.001, "cls_pw": 0.25},
}
TRANSFER_LOG_PATTERN = re.compile(
    r"Remapped 8/8 (?:decoder )?cls head rows from pretrained weights by class name"
)
MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_YAML_BYTES = 4 * 1024 * 1024
MAX_LABEL_BYTES = 16 * 1024 * 1024
MAX_IMAGE_BYTES = 512 * 1024 * 1024
MAX_WEIGHT_BYTES = 4 * 1024 * 1024 * 1024


class TrainingValidationError(ValueError):
    """Raised when a training-only evaluation is unsafe or inconsistent."""


@dataclass(frozen=True)
class FileIdentity:
    device: int
    inode: int
    mode: int
    link_count: int
    size_bytes: int
    modified_ns: int
    changed_ns: int


@dataclass(frozen=True)
class BoundFile:
    path: Path
    sha256: str
    size_bytes: int
    identity: FileIdentity


@dataclass(frozen=True)
class ValidationFrame:
    image: BoundFile
    label: BoundFile
    image_relative: str
    label_relative: str
    scene_id: str
    frame: int
    width: int
    height: int
    ground_truth: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ValidatedDataset:
    root: Path
    manifest: BoundFile
    payload: Mapping[str, Any]
    schema: Mapping[str, Any]
    taxonomy: Mapping[str, Any]
    dataset_yaml: BoundFile
    managed_files: tuple[BoundFile, ...]
    managed_records: tuple[dict[str, Any], ...]
    managed_files_sha256: str
    val_frames: tuple[ValidationFrame, ...]
    val_asset_ids: tuple[int | str, ...]
    split_plan: Mapping[str, Any]
    manifest_fold_id: str | None


@dataclass(frozen=True)
class ValidatedCandidate:
    receipt: BoundFile
    weight: BoundFile
    schema: Mapping[str, Any]
    seed: int
    protocol_sha256: str
    map50: float
    map50_95: float
    training_duration_seconds: float
    image_size: int
    training_device: str
    resolved_args: Mapping[str, Any]


@dataclass(frozen=True)
class EvaluationSettings:
    confidence: float = 0.40
    image_size: int | None = None
    device: str = "mps"
    nms_iou: float = 0.75
    rider_overlap: float = 0.25
    match_iou: float = 0.50


@dataclass(frozen=True)
class ModelRun:
    raw_predictions: tuple[dict[str, Any], ...]
    observed_frame_keys: frozenset[tuple[str, int]]
    predict_call_count: int
    inference_wall_seconds: float
    model_load_seconds: float
    model_metadata: Mapping[str, Any]


class ModelRunner(Protocol):
    def __call__(
        self,
        weight: Path,
        frames: Sequence[ValidationFrame],
        settings: Mapping[str, Any],
    ) -> ModelRun: ...


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as error:
            raise TrainingValidationError("YAML contains an unhashable mapping key") from error
        if duplicate:
            raise TrainingValidationError(f"YAML contains duplicate key {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _object(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TrainingValidationError(f"{location} must be an object")
    return value


def _list(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise TrainingValidationError(f"{location} must be a list")
    return value


def _text(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TrainingValidationError(f"{location} must be a non-empty string")
    return value


def _integer(
    value: Any,
    location: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TrainingValidationError(f"{location} must be an integer")
    if minimum is not None and value < minimum:
        raise TrainingValidationError(f"{location} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise TrainingValidationError(f"{location} must be at most {maximum}")
    return value


def _number(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainingValidationError(f"{location} must be a number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise TrainingValidationError(f"{location} must be finite") from error
    if not math.isfinite(result):
        raise TrainingValidationError(f"{location} must be finite")
    return 0.0 if result == 0.0 else result


def _sha256(value: Any, location: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise TrainingValidationError(f"{location} must be a lowercase SHA-256 digest")
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


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True).encode(
            "utf-8"
        )
        + b"\n"
    )


def _workspace_root(path: Path | None) -> Path:
    raw = Path.cwd() if path is None else Path(path)
    absolute = Path(os.path.abspath(os.fspath(raw.expanduser())))
    try:
        details = absolute.lstat()
    except OSError as error:
        raise TrainingValidationError(f"could not stat workspace root: {error}") from error
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
        raise TrainingValidationError("workspace root must be a real directory, not a symlink")
    return absolute.resolve(strict=True)


def _lexical_absolute(path: Path, workspace: Path) -> Path:
    expanded = Path(path).expanduser()
    candidate = expanded if expanded.is_absolute() else workspace / expanded
    return Path(os.path.abspath(os.fspath(candidate)))


def _contains_forbidden_scope(path: Path) -> bool:
    return final_holdout_scope_reason(path) is not None


def _reject_forbidden_scope(path: Path, location: str) -> None:
    if _contains_forbidden_scope(path):
        raise TrainingValidationError(
            f"{location} is forbidden: final-holdout scope is outside training evaluation"
        )


def _ensure_no_symlink_path(
    workspace: Path,
    path: Path,
    location: str,
    *,
    leaf_kind: str,
) -> None:
    try:
        relative = path.relative_to(workspace)
    except ValueError as error:
        raise TrainingValidationError(f"{location} must remain inside the workspace") from error
    current = workspace
    for index, part in enumerate(relative.parts):
        current = current / part
        is_leaf = index == len(relative.parts) - 1
        try:
            details = current.lstat()
        except FileNotFoundError:
            if is_leaf and leaf_kind == "missing":
                return
            raise TrainingValidationError(f"{location} does not exist: {current}") from None
        except OSError as error:
            raise TrainingValidationError(f"could not stat {location}: {error}") from error
        if stat.S_ISLNK(details.st_mode):
            raise TrainingValidationError(f"{location} must not contain symbolic links")
        if not is_leaf and not stat.S_ISDIR(details.st_mode):
            raise TrainingValidationError(f"{location} has a non-directory parent")
        if is_leaf:
            if leaf_kind == "file" and not stat.S_ISREG(details.st_mode):
                raise TrainingValidationError(f"{location} must be a regular file")
            if leaf_kind == "directory" and not stat.S_ISDIR(details.st_mode):
                raise TrainingValidationError(f"{location} must be a directory")
            if leaf_kind == "missing":
                raise TrainingValidationError(f"{location} already exists: {current}")


def _scoped_path(
    raw_path: Path,
    *,
    workspace: Path,
    scope: tuple[str, str],
    location: str,
    leaf_kind: str,
) -> Path:
    path = _lexical_absolute(raw_path, workspace)
    _reject_forbidden_scope(path, location)
    allowed = workspace.joinpath(*scope)
    if not path.is_relative_to(allowed):
        raise TrainingValidationError(
            f"{location} must be below {PurePosixPath(*scope).as_posix()}"
        )
    _ensure_no_symlink_path(workspace, path, location, leaf_kind=leaf_kind)
    return path


def _identity(details: os.stat_result) -> FileIdentity:
    return FileIdentity(
        device=details.st_dev,
        inode=details.st_ino,
        mode=details.st_mode,
        link_count=details.st_nlink,
        size_bytes=details.st_size,
        modified_ns=details.st_mtime_ns,
        changed_ns=details.st_ctime_ns,
    )


def _read_regular_bytes(
    path: Path, location: str, *, maximum_bytes: int
) -> tuple[bytes, BoundFile]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise TrainingValidationError(f"could not open {location}: {error}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise TrainingValidationError(f"{location} must be a regular file")
        if before.st_size > maximum_bytes:
            raise TrainingValidationError(f"{location} exceeds its maximum allowed size")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        extra = os.read(descriptor, 1)
        after = os.fstat(descriptor)
        encoded = b"".join(chunks)
        if _identity(before) != _identity(after) or len(encoded) != before.st_size or extra:
            raise TrainingValidationError(f"{location} changed while it was read")
        bound = BoundFile(
            path=path,
            sha256=hashlib.sha256(encoded).hexdigest(),
            size_bytes=len(encoded),
            identity=_identity(after),
        )
        return encoded, bound
    finally:
        os.close(descriptor)


def _read_json(path: Path, location: str) -> tuple[dict[str, Any], BoundFile]:
    encoded, bound = _read_regular_bytes(path, location, maximum_bytes=MAX_JSON_BYTES)
    try:
        payload = json.loads(
            encoded.decode("utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise TrainingValidationError(f"could not parse {location}: {error}") from error
    return _object(payload, location), bound


def _safe_relative_path(value: Any, location: str) -> str:
    text = _text(value, location)
    posix = PurePosixPath(text)
    windows = PureWindowsPath(text)
    if (
        "\\" in text
        or any(ord(character) < 32 for character in text)
        or posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or posix.as_posix() != text
        or not posix.name
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        raise TrainingValidationError(f"{location} must be a safe relative POSIX path")
    if _contains_forbidden_scope(Path(*posix.parts)):
        raise TrainingValidationError(f"{location} references a forbidden final-holdout path")
    return text


def _workspace_relative(path: Path, workspace: Path, location: str) -> str:
    try:
        relative = path.relative_to(workspace)
    except ValueError as error:
        raise TrainingValidationError(f"{location} must remain inside the workspace") from error
    return PurePosixPath(*relative.parts).as_posix()


def _revalidate(files: Iterable[BoundFile], location: str) -> None:
    for index, expected in enumerate(files):
        try:
            _encoded, actual = _read_regular_bytes(
                expected.path,
                f"{location}[{index}]",
                maximum_bytes=max(expected.size_bytes, 1),
            )
        except TrainingValidationError as error:
            raise TrainingValidationError(
                f"{location}[{index}] changed during evaluation"
            ) from error
        if actual != expected:
            raise TrainingValidationError(f"{location}[{index}] changed during evaluation")


def _taxonomy_contract(model_names: Sequence[str]) -> dict[str, Any]:
    names = tuple(model_names)
    if names == CANONICAL_LABELS:
        mapping = LEGACY_MODEL_TO_CANONICAL
    elif names == MODEL_LABELS:
        mapping = MODEL_TO_CANONICAL
    else:
        raise TrainingValidationError("dataset uses an unsupported model taxonomy")
    return {
        "canonical_names": list(CANONICAL_LABELS),
        "model_names": list(names),
        "model_to_canonical": dict(mapping),
    }


def _validate_taxonomy(manifest: Mapping[str, Any], schema: Mapping[str, Any]) -> dict[str, Any]:
    if schema == LEGACY_DATASET_SCHEMA:
        if "taxonomy" in manifest:
            raise TrainingValidationError("schema-v2 dataset must not declare a v3 taxonomy")
        return _taxonomy_contract(CANONICAL_LABELS)
    taxonomy = _object(manifest.get("taxonomy"), "dataset manifest.taxonomy")
    expected = _taxonomy_contract(MODEL_LABELS)
    if taxonomy != expected:
        raise TrainingValidationError("dataset manifest taxonomy mapping is unsupported")
    return expected


def _managed_target(root: Path, relative: str, location: str) -> Path:
    target = root.joinpath(*PurePosixPath(relative).parts)
    if not target.is_relative_to(root):
        raise TrainingValidationError(f"{location} escapes the dataset root")
    _ensure_no_symlink_path(root, target, location, leaf_kind="file")
    return target


def _inventory_dataset(root: Path) -> tuple[set[str], set[str]]:
    files: set[str] = set()
    directories: set[str] = set()
    for current_text, directory_names, file_names in os.walk(root, followlinks=False):
        current = Path(current_text)
        for name in directory_names:
            target = current / name
            details = target.lstat()
            if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
                raise TrainingValidationError(f"dataset contains unsafe directory: {target}")
            directories.add(target.relative_to(root).as_posix())
        for name in file_names:
            target = current / name
            details = target.lstat()
            if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
                raise TrainingValidationError(f"dataset contains unsafe file: {target}")
            files.add(target.relative_to(root).as_posix())
    return files, directories


def _load_dataset_yaml(encoded: bytes, expected_model_names: Sequence[str]) -> None:
    try:
        payload = yaml.load(encoded.decode("utf-8"), Loader=_UniqueKeyLoader)
    except (UnicodeDecodeError, yaml.YAMLError, TrainingValidationError) as error:
        raise TrainingValidationError(f"could not parse dataset.yaml: {error}") from error
    value = _object(payload, "dataset.yaml")
    if set(value) != {"train", "val", "names"}:
        raise TrainingValidationError("dataset.yaml must contain only train, val, and names")
    if value["train"] != "images/train" or value["val"] != "images/val":
        raise TrainingValidationError("dataset.yaml train/val paths are not canonical")
    expected_names = {index: name for index, name in enumerate(expected_model_names)}
    if _object(value["names"], "dataset.yaml.names") != expected_names:
        raise TrainingValidationError("dataset.yaml names differ from its taxonomy binding")


def _parse_label_bytes(
    encoded: bytes,
    *,
    location: str,
    scene_id: str,
    frame: int,
    width: int,
    height: int,
) -> tuple[dict[str, Any], ...]:
    try:
        text = encoded.decode("utf-8")
    except UnicodeDecodeError as error:
        raise TrainingValidationError(f"{location} is not UTF-8") from error
    annotations: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        tokens = line.split()
        if len(tokens) != 5 or not re.fullmatch(r"[0-7]", tokens[0]):
            raise TrainingValidationError(f"{location}:{line_number} is not a YOLO label")
        coordinates: list[float] = []
        for index, token in enumerate(tokens[1:], start=1):
            try:
                raw_coordinate = float(token)
            except ValueError as error:
                raise TrainingValidationError(
                    f"{location}:{line_number} coordinate {index} is invalid"
                ) from error
            coordinates.append(
                _number(raw_coordinate, f"{location}:{line_number} coordinate {index}")
            )
        center_x, center_y, box_width, box_height = coordinates
        if (
            not 0.0 <= center_x <= 1.0
            or not 0.0 <= center_y <= 1.0
            or not 0.0 < box_width <= 1.0
            or not 0.0 < box_height <= 1.0
            or center_x - box_width / 2 < -1e-12
            or center_y - box_height / 2 < -1e-12
            or center_x + box_width / 2 > 1.0 + 1e-12
            or center_y + box_height / 2 > 1.0 + 1e-12
        ):
            raise TrainingValidationError(f"{location}:{line_number} bbox escapes image bounds")
        left = (center_x - box_width / 2) * width
        top = (center_y - box_height / 2) * height
        right = (center_x + box_width / 2) * width
        bottom = (center_y + box_height / 2) * height
        annotations.append(
            {
                "scene_id": scene_id,
                "frame": frame,
                "label": CANONICAL_LABELS[int(tokens[0])],
                "bbox": [left, top, right, bottom],
                "source": "ground_truth",
            }
        )
    return tuple(annotations)


def _image_dimensions(encoded: bytes, location: str) -> tuple[int, int]:
    try:
        with Image.open(io.BytesIO(encoded)) as image:
            width, height = image.size
            image.verify()
    except Exception as error:
        raise TrainingValidationError(
            f"could not decode {location}: {type(error).__name__}"
        ) from error
    if width <= 0 or height <= 0:
        raise TrainingValidationError(f"{location} has invalid dimensions")
    return width, height


def _asset_id(value: Any, location: str) -> int | str:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise TrainingValidationError(f"{location} must be an integer or string")
    if isinstance(value, str) and not value.strip():
        raise TrainingValidationError(f"{location} must be non-empty")
    return value


def _fold_identifier(value: Any, location: str) -> str:
    identifier = _text(value, location)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", identifier):
        raise TrainingValidationError(f"{location} contains unsafe characters")
    if final_holdout_scope_reason(identifier) is not None:
        raise TrainingValidationError(f"{location} must not identify the final holdout")
    return identifier


def _validate_fold_binding(
    manifest: Mapping[str, Any], requested_fold_id: str
) -> tuple[tuple[int | str, ...], dict[str, Any], str | None]:
    split = _object(manifest.get("split"), "dataset manifest.split")
    val_assets = tuple(
        _asset_id(value, f"dataset manifest.split.val_asset_ids[{index}]")
        for index, value in enumerate(
            _list(split.get("val_asset_ids"), "dataset manifest.split.val_asset_ids")
        )
    )
    if not val_assets or len({(type(value).__name__, str(value)) for value in val_assets}) != len(
        val_assets
    ):
        raise TrainingValidationError("dataset validation asset IDs must be non-empty and unique")
    inputs = _object(manifest.get("inputs"), "dataset manifest.inputs")
    raw_plan = _object(inputs.get("split_plan"), "dataset manifest.inputs.split_plan")
    file_name = _text(raw_plan.get("file_name"), "dataset manifest.inputs.split_plan.file_name")
    if Path(file_name).name != file_name or _contains_forbidden_scope(Path(file_name)):
        raise TrainingValidationError("dataset split-plan file name is unsafe")
    split_plan = {
        "file_name": file_name,
        "sha256": _sha256(raw_plan.get("sha256"), "dataset split-plan sha256"),
        "semantic_sha256": _sha256(
            raw_plan.get("semantic_sha256"), "dataset split-plan semantic_sha256"
        ),
        "schema": raw_plan.get("schema"),
    }
    if split_plan["schema"] != SPLIT_PLAN_SCHEMA:
        raise TrainingValidationError("dataset split-plan schema is unsupported")

    declarations: list[str] = []
    for raw in (manifest.get("fold_id"), split.get("fold_id")):
        if raw is not None:
            declarations.append(_fold_identifier(raw, "dataset manifest fold_id"))
    matched = re.fullmatch(r"(.+?)\.split\.json", file_name)
    if matched and matched.group(1).startswith("fold-"):
        declarations.append(_fold_identifier(matched.group(1), "dataset split-plan fold_id"))
    if declarations and any(value != requested_fold_id for value in declarations):
        raise TrainingValidationError(
            f"requested fold_id {requested_fold_id!r} differs from dataset fold binding"
        )
    if len(set(declarations)) > 1:
        raise TrainingValidationError("dataset manifest contains conflicting fold identifiers")
    return val_assets, split_plan, declarations[0] if declarations else None


def _managed_file_kind(relative: str) -> tuple[str, str]:
    parts = PurePosixPath(relative).parts
    if relative == "dataset.yaml":
        return "metadata", "dataset"
    if (
        len(parts) != 3
        or parts[0] not in {"images", "labels"}
        or parts[1]
        not in {
            "train",
            "val",
        }
    ):
        raise TrainingValidationError(f"dataset manifest manages unexpected path {relative!r}")
    if parts[0] == "images":
        if PurePosixPath(relative).suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            raise TrainingValidationError(f"unsupported managed image {relative!r}")
    elif PurePosixPath(relative).suffix != ".txt":
        raise TrainingValidationError(f"unsupported managed label {relative!r}")
    return parts[0], parts[1]


def _expected_dataset_directories(records: Sequence[Mapping[str, Any]]) -> set[str]:
    directories: set[str] = set()
    for record in records:
        relative = PurePosixPath(str(record["path"]))
        directories.update(
            parent.as_posix() for parent in relative.parents if parent.as_posix() != "."
        )
    return directories


def _validate_dataset_counts(
    manifest: Mapping[str, Any],
    *,
    val_frame_count: int,
    val_annotations: Sequence[Mapping[str, Any]],
) -> None:
    counts = _object(manifest.get("counts"), "dataset manifest.counts")
    image_counts = _object(counts.get("images"), "dataset manifest.counts.images")
    if _integer(image_counts.get("val"), "dataset manifest.counts.images.val", minimum=0) != (
        val_frame_count
    ):
        raise TrainingValidationError("dataset manifest validation image count is inconsistent")
    annotation_counts = _object(counts.get("annotations"), "dataset manifest.counts.annotations")
    if _integer(
        annotation_counts.get("val"), "dataset manifest.counts.annotations.val", minimum=0
    ) != len(val_annotations):
        raise TrainingValidationError(
            "dataset manifest validation annotation count is inconsistent"
        )
    by_split = _object(
        annotation_counts.get("by_split_and_category"),
        "dataset manifest.counts.annotations.by_split_and_category",
    )
    declared_val = _object(
        by_split.get("val"), "dataset manifest.counts.annotations.by_split_and_category.val"
    )
    if set(declared_val) != set(CANONICAL_LABELS):
        raise TrainingValidationError("dataset manifest validation classes are incomplete")
    actual = Counter(str(annotation["label"]) for annotation in val_annotations)
    normalized = {
        label: _integer(
            declared_val[label],
            f"dataset manifest.counts.annotations.by_split_and_category.val.{label}",
            minimum=0,
        )
        for label in CANONICAL_LABELS
    }
    if normalized != {label: actual[label] for label in CANONICAL_LABELS}:
        raise TrainingValidationError("dataset manifest validation class counts are inconsistent")


def validate_dataset(
    dataset_root: Path,
    dataset_manifest_path: Path,
    *,
    fold_id: str,
    workspace_root: Path,
) -> ValidatedDataset:
    """Validate an immutable training dataset without decoding its training split."""

    root = _scoped_path(
        dataset_root,
        workspace=workspace_root,
        scope=("data", "training"),
        location="dataset root",
        leaf_kind="directory",
    )
    manifest_path = _scoped_path(
        dataset_manifest_path,
        workspace=workspace_root,
        scope=("data", "training"),
        location="dataset manifest",
        leaf_kind="file",
    )
    if manifest_path != root / "manifest.json":
        raise TrainingValidationError("dataset manifest must be dataset_root/manifest.json")
    manifest, manifest_file = _read_json(manifest_path, "dataset manifest")
    _scan_declared_paths(manifest, "dataset manifest")
    schema = manifest.get("schema")
    if schema not in (LEGACY_DATASET_SCHEMA, DATASET_SCHEMA):
        raise TrainingValidationError("dataset manifest schema is unsupported")
    taxonomy = _validate_taxonomy(manifest, schema)
    gate = _object(manifest.get("gate"), "dataset manifest.gate")
    if gate.get("passed") is not True:
        raise TrainingValidationError("dataset manifest gate did not pass")
    if "blocking_reasons" in gate and _list(
        gate.get("blocking_reasons"), "dataset manifest.gate.blocking_reasons"
    ):
        raise TrainingValidationError("dataset manifest has blocking reasons")
    checks = _object(gate.get("checks"), "dataset manifest.gate.checks")
    required_checks = (
        REQUIRED_DATASET_GATE_CHECKS
        if schema == LEGACY_DATASET_SCHEMA
        else CURRENT_DATASET_GATE_CHECKS
    )
    if not required_checks.issubset(checks) or any(value is not True for value in checks.values()):
        raise TrainingValidationError("dataset manifest checks are incomplete or false")

    raw_records = _list(manifest.get("files"), "dataset manifest.files")
    if not raw_records:
        raise TrainingValidationError("dataset manifest.files must not be empty")
    managed_records: list[dict[str, Any]] = []
    managed_files: list[BoundFile] = []
    retained_bytes: dict[str, bytes] = {}
    seen: set[str] = set()
    seen_casefold: set[str] = set()
    for index, raw_record in enumerate(raw_records):
        location = f"dataset manifest.files[{index}]"
        record = _object(raw_record, location)
        if set(record) != {"path", "sha256", "size_bytes"}:
            raise TrainingValidationError(f"{location} has unexpected or missing fields")
        relative = _safe_relative_path(record.get("path"), f"{location}.path")
        if relative == "manifest.json" or relative in seen or relative.casefold() in seen_casefold:
            raise TrainingValidationError(f"dataset manifest has duplicate path {relative!r}")
        seen.add(relative)
        seen_casefold.add(relative.casefold())
        kind, split = _managed_file_kind(relative)
        expected_sha = _sha256(record.get("sha256"), f"{location}.sha256")
        expected_size = _integer(record.get("size_bytes"), f"{location}.size_bytes", minimum=0)
        maximum = (
            MAX_YAML_BYTES
            if kind == "metadata"
            else MAX_IMAGE_BYTES
            if kind == "images"
            else MAX_LABEL_BYTES
        )
        target = _managed_target(root, relative, location)
        encoded, bound = _read_regular_bytes(target, location, maximum_bytes=maximum)
        if bound.sha256 != expected_sha or bound.size_bytes != expected_size:
            raise TrainingValidationError(f"{location} differs from its managed hash or size")
        canonical_record = {
            "path": relative,
            "sha256": expected_sha,
            "size_bytes": expected_size,
        }
        managed_records.append(canonical_record)
        managed_files.append(bound)
        if kind == "metadata" or split == "val":
            retained_bytes[relative] = encoded

    actual_files, actual_directories = _inventory_dataset(root)
    expected_files = seen | {"manifest.json"}
    if actual_files != expected_files:
        raise TrainingValidationError(
            "dataset tree differs from its exact managed inventory; "
            f"missing={sorted(expected_files - actual_files)!r}, "
            f"unmanaged={sorted(actual_files - expected_files)!r}"
        )
    expected_directories = _expected_dataset_directories(managed_records)
    if actual_directories != expected_directories:
        raise TrainingValidationError("dataset tree has unexpected or missing directories")

    records_by_path = {record["path"]: record for record in managed_records}
    files_by_path = {
        record["path"]: bound for record, bound in zip(managed_records, managed_files, strict=True)
    }
    if "dataset.yaml" not in records_by_path:
        raise TrainingValidationError("dataset manifest does not manage dataset.yaml")
    _load_dataset_yaml(retained_bytes["dataset.yaml"], taxonomy["model_names"])

    val_images: dict[str, str] = {}
    val_labels: dict[str, str] = {}
    for relative in seen:
        kind, split = _managed_file_kind(relative)
        if split != "val":
            continue
        stem = PurePosixPath(relative).stem.casefold()
        target = val_images if kind == "images" else val_labels
        if stem in target:
            raise TrainingValidationError(f"validation split has duplicate stem {stem!r}")
        target[stem] = relative
    if not val_images or set(val_images) != set(val_labels):
        raise TrainingValidationError(
            "validation images and labels must be non-empty and one-to-one"
        )

    frames: list[ValidationFrame] = []
    all_ground_truth: list[dict[str, Any]] = []
    for frame_index, stem in enumerate(
        sorted(val_images, key=lambda value: (val_images[value].casefold(), val_images[value]))
    ):
        image_relative = val_images[stem]
        label_relative = val_labels[stem]
        image_file = files_by_path[image_relative]
        label_file = files_by_path[label_relative]
        width, height = _image_dimensions(retained_bytes[image_relative], image_relative)
        scene_id = image_relative
        annotations = _parse_label_bytes(
            retained_bytes[label_relative],
            location=label_relative,
            scene_id=scene_id,
            frame=0,
            width=width,
            height=height,
        )
        all_ground_truth.extend(annotations)
        frames.append(
            ValidationFrame(
                image=image_file,
                label=label_file,
                image_relative=image_relative,
                label_relative=label_relative,
                scene_id=scene_id,
                frame=0,
                width=width,
                height=height,
                ground_truth=annotations,
            )
        )
    _validate_dataset_counts(
        manifest,
        val_frame_count=len(frames),
        val_annotations=all_ground_truth,
    )
    val_asset_ids, split_plan, manifest_fold_id = _validate_fold_binding(manifest, fold_id)
    canonical_records = sorted(managed_records, key=lambda item: item["path"])
    return ValidatedDataset(
        root=root,
        manifest=manifest_file,
        payload=manifest,
        schema=dict(schema),
        taxonomy=taxonomy,
        dataset_yaml=files_by_path["dataset.yaml"],
        managed_files=tuple(managed_files),
        managed_records=tuple(canonical_records),
        managed_files_sha256=canonical_sha256(canonical_records),
        val_frames=tuple(frames),
        val_asset_ids=val_asset_ids,
        split_plan=split_plan,
        manifest_fold_id=manifest_fold_id,
    )


def _scan_declared_paths(value: Any, location: str = "input") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{location}.{key}"
            if key in {"path", "file_name"} and isinstance(item, str):
                normalized = Path(item.replace("\\", "/"))
                if _contains_forbidden_scope(normalized):
                    raise TrainingValidationError(
                        f"{child} references a forbidden final-holdout scope"
                    )
            _scan_declared_paths(item, child)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _scan_declared_paths(item, f"{location}[{index}]")


def _same_json(first: Any, second: Any) -> bool:
    return _canonical_bytes(first) == _canonical_bytes(second)


def _validate_metric(value: Any, location: str) -> float:
    result = _number(value, location)
    if not 0.0 <= result <= 1.0:
        raise TrainingValidationError(f"{location} must be within [0, 1]")
    return result


def _validate_candidate_dataset_binding(
    receipt: Mapping[str, Any],
    dataset: ValidatedDataset,
    candidate_schema: Mapping[str, Any],
) -> None:
    inputs = _object(receipt.get("inputs"), "candidate receipt.inputs")
    receipt_dataset = _object(inputs.get("dataset"), "candidate receipt.inputs.dataset")
    expected_keys = {
        "manifest",
        "dataset_yaml",
        "managed_files_sha256",
        "managed_file_count",
        "counts",
    }
    if candidate_schema == CANDIDATE_SCHEMA:
        expected_keys.add("taxonomy")
    if set(receipt_dataset) != expected_keys:
        raise TrainingValidationError(
            "candidate receipt dataset binding has unexpected or missing fields"
        )
    manifest_record = _object(
        receipt_dataset.get("manifest"), "candidate receipt.inputs.dataset.manifest"
    )
    if set(manifest_record) != {"schema", "sha256", "size_bytes"}:
        raise TrainingValidationError("candidate dataset manifest binding is malformed")
    if (
        not _same_json(manifest_record.get("schema"), dataset.schema)
        or _sha256(manifest_record.get("sha256"), "candidate dataset manifest sha256")
        != dataset.manifest.sha256
        or _integer(
            manifest_record.get("size_bytes"),
            "candidate dataset manifest size_bytes",
            minimum=1,
        )
        != dataset.manifest.size_bytes
    ):
        raise TrainingValidationError("candidate receipt is bound to another dataset manifest")
    yaml_record = _object(
        receipt_dataset.get("dataset_yaml"), "candidate receipt.inputs.dataset.dataset_yaml"
    )
    if set(yaml_record) != {"sha256", "size_bytes"}:
        raise TrainingValidationError("candidate dataset.yaml binding is malformed")
    if (
        _sha256(yaml_record.get("sha256"), "candidate dataset.yaml sha256")
        != dataset.dataset_yaml.sha256
        or _integer(yaml_record.get("size_bytes"), "candidate dataset.yaml size_bytes", minimum=1)
        != dataset.dataset_yaml.size_bytes
    ):
        raise TrainingValidationError("candidate receipt dataset.yaml binding differs")
    if _sha256(
        receipt_dataset.get("managed_files_sha256"),
        "candidate managed_files_sha256",
    ) != dataset.managed_files_sha256 or _integer(
        receipt_dataset.get("managed_file_count"),
        "candidate managed_file_count",
        minimum=1,
    ) != len(dataset.managed_files):
        raise TrainingValidationError("candidate receipt managed-file binding differs")
    if not _same_json(receipt_dataset.get("counts"), dataset.payload.get("counts")):
        raise TrainingValidationError("candidate receipt dataset counts differ")
    if candidate_schema == CANDIDATE_SCHEMA and not _same_json(
        receipt_dataset.get("taxonomy"), dataset.taxonomy
    ):
        raise TrainingValidationError("candidate receipt dataset taxonomy differs")


def _validate_pretrained_transfer(
    receipt: Mapping[str, Any],
    *,
    dataset: ValidatedDataset,
    current_gate: bool,
) -> None:
    evidence_present = "pretrained_transfer" in receipt
    if current_gate != evidence_present:
        raise TrainingValidationError(
            "candidate pretrained-transfer evidence and gate check must appear together"
        )
    if not current_gate:
        return
    inputs = _object(receipt.get("inputs"), "candidate receipt.inputs")
    if set(inputs) != {"dataset", "base_weights"}:
        raise TrainingValidationError("current candidate inputs must bind dataset and base weights")
    base = _object(inputs.get("base_weights"), "candidate receipt.inputs.base_weights")
    if set(base) != {"file_name", "model_family", "sha256", "size_bytes"}:
        raise TrainingValidationError("candidate base-weight binding is malformed")
    base_sha = _sha256(base.get("sha256"), "candidate base-weight sha256")
    if (
        base.get("file_name") != "yolo11n.pt"
        or base.get("model_family") != "YOLO11n"
        or _integer(base.get("size_bytes"), "candidate base-weight size", minimum=1) < 1
    ):
        raise TrainingValidationError("candidate base-weight binding is unsupported")

    evidence = _object(receipt.get("pretrained_transfer"), "candidate pretrained_transfer")
    if set(evidence) != {
        "schema",
        "source_model",
        "target",
        "matched_rows",
        "matched_row_count",
        "target_row_count",
        "runtime_observation",
    } or evidence.get("schema") != {
        "name": "roadlabelops.pretrained-class-head-transfer",
        "version": 1,
    }:
        raise TrainingValidationError("candidate pretrained-transfer contract is malformed")
    source = _object(evidence.get("source_model"), "candidate pretrained_transfer.source_model")
    if set(source) != {"family", "base_weights_sha256", "class_count", "names_sha256"}:
        raise TrainingValidationError("candidate pretrained-transfer source binding is malformed")
    source_count = _integer(
        source.get("class_count"), "candidate pretrained source class_count", minimum=8
    )
    if (
        source.get("family") != "YOLO11n"
        or source.get("base_weights_sha256") != base_sha
        or len(_sha256(source.get("names_sha256"), "candidate pretrained names sha256")) != 64
    ):
        raise TrainingValidationError("candidate pretrained-transfer source differs")
    target = _object(evidence.get("target"), "candidate pretrained_transfer.target")
    if target != {
        "class_count": len(CANONICAL_LABELS),
        "model_names": list(dataset.taxonomy["model_names"]),
        "canonical_names": list(CANONICAL_LABELS),
    }:
        raise TrainingValidationError("candidate pretrained-transfer target differs")
    if evidence.get("matched_row_count") != len(CANONICAL_LABELS) or evidence.get(
        "target_row_count"
    ) != len(CANONICAL_LABELS):
        raise TrainingValidationError("candidate pretrained-transfer is not complete")
    rows = _list(evidence.get("matched_rows"), "candidate pretrained_transfer.matched_rows")
    if len(rows) != len(CANONICAL_LABELS):
        raise TrainingValidationError("candidate pretrained-transfer rows are incomplete")
    source_ids: set[int] = set()
    for target_id, (model_name, canonical_name) in enumerate(
        zip(dataset.taxonomy["model_names"], CANONICAL_LABELS, strict=True)
    ):
        row = _object(rows[target_id], f"candidate pretrained_transfer row {target_id}")
        source_id = _integer(row.get("source_id"), f"candidate pretrained source_id {target_id}")
        if (
            set(row)
            != {
                "target_id",
                "target_model_name",
                "canonical_name",
                "source_id",
                "source_model_name",
            }
            or row.get("target_id") != target_id
            or row.get("target_model_name") != model_name
            or row.get("canonical_name") != canonical_name
            or row.get("source_model_name") != model_name
            or source_id >= source_count
            or source_id in source_ids
        ):
            raise TrainingValidationError("candidate pretrained-transfer row is inconsistent")
        source_ids.add(source_id)
    runtime = _object(
        evidence.get("runtime_observation"), "candidate pretrained_transfer.runtime_observation"
    )
    if set(runtime) != {
        "verification_mode",
        "message",
        "message_sha256",
        "matched_row_count",
        "target_row_count",
    }:
        raise TrainingValidationError("candidate pretrained-transfer runtime binding is malformed")
    message = _text(runtime.get("message"), "candidate pretrained-transfer runtime message")
    if (
        runtime.get("verification_mode") != "ultralytics_logger"
        or TRANSFER_LOG_PATTERN.fullmatch(message) is None
        or hashlib.sha256(message.encode("utf-8")).hexdigest()
        != _sha256(runtime.get("message_sha256"), "candidate pretrained runtime message sha256")
        or runtime.get("matched_row_count") != len(CANONICAL_LABELS)
        or runtime.get("target_row_count") != len(CANONICAL_LABELS)
    ):
        raise TrainingValidationError("candidate pretrained-transfer runtime proof differs")


def validate_candidate(
    candidate_receipt_path: Path,
    candidate_weight_path: Path,
    *,
    dataset: ValidatedDataset,
    workspace_root: Path,
) -> ValidatedCandidate:
    receipt_path = _scoped_path(
        candidate_receipt_path,
        workspace=workspace_root,
        scope=("data", "model-candidates"),
        location="candidate receipt",
        leaf_kind="file",
    )
    if receipt_path.name != "receipt.json":
        raise TrainingValidationError("candidate receipt must be named receipt.json")
    weight_path = _scoped_path(
        candidate_weight_path,
        workspace=workspace_root,
        scope=("data", "model-candidates"),
        location="candidate weight",
        leaf_kind="file",
    )
    receipt, receipt_file = _read_json(receipt_path, "candidate receipt")
    _scan_declared_paths(receipt, "candidate receipt")
    candidate_schema = receipt.get("schema")
    if candidate_schema not in (LEGACY_CANDIDATE_SCHEMA, CANDIDATE_SCHEMA):
        raise TrainingValidationError("candidate receipt schema is unsupported")
    gate = _object(receipt.get("gate"), "candidate receipt.gate")
    if gate.get("passed") is not True:
        raise TrainingValidationError("candidate receipt gate did not pass")
    checks = _object(gate.get("checks"), "candidate receipt.gate.checks")
    supported_gate_contracts = (
        (LEGACY_CANDIDATE_GATE_CHECKS,)
        if candidate_schema == LEGACY_CANDIDATE_SCHEMA
        else (
            PRE_SUPPORT_AWARE_CANDIDATE_GATE_CHECKS,
            SUPPORT_AWARE_CANDIDATE_GATE_CHECKS,
            CURRENT_CANDIDATE_GATE_CHECKS,
        )
    )
    if set(checks) not in tuple(set(contract) for contract in supported_gate_contracts) or any(
        result is not True for result in checks.values()
    ):
        raise TrainingValidationError("candidate receipt gate checks are incomplete or false")
    if receipt.get("mutation_performed") is not True:
        raise TrainingValidationError(
            "candidate receipt does not represent a completed training run"
        )
    holdout = _object(receipt.get("holdout"), "candidate receipt.holdout")
    if holdout.get("input_read") is not False or holdout.get("statement") != NO_HOLDOUT_STATEMENT:
        raise TrainingValidationError("candidate receipt does not preserve the holdout firewall")

    protocol = _object(receipt.get("protocol"), "candidate receipt.protocol")
    expected_protocol_schema = (
        LEGACY_PROTOCOL_SCHEMA if candidate_schema == LEGACY_CANDIDATE_SCHEMA else PROTOCOL_SCHEMA
    )
    if protocol.get("schema") != expected_protocol_schema:
        raise TrainingValidationError("candidate receipt and protocol schema versions differ")
    if protocol.get("model_family") != "YOLO11n" or protocol.get("holdout_access") != "prohibited":
        raise TrainingValidationError("candidate protocol model or holdout contract is unsupported")
    protocol_sha256 = _sha256(receipt.get("protocol_sha256"), "candidate protocol_sha256")
    if canonical_sha256(protocol) != protocol_sha256:
        raise TrainingValidationError("candidate protocol semantic hash differs")
    if candidate_schema == LEGACY_CANDIDATE_SCHEMA:
        if protocol.get("taxonomy") != list(CANONICAL_LABELS):
            raise TrainingValidationError("legacy candidate protocol taxonomy is unsupported")
        if dataset.schema != LEGACY_DATASET_SCHEMA:
            raise TrainingValidationError("legacy candidate cannot bind a schema-v3 dataset")
    elif not _same_json(protocol.get("taxonomy"), dataset.taxonomy):
        raise TrainingValidationError("candidate protocol taxonomy differs from the dataset")

    _validate_candidate_dataset_binding(receipt, dataset, candidate_schema)
    _validate_pretrained_transfer(
        receipt,
        dataset=dataset,
        current_gate=set(checks) == set(CURRENT_CANDIDATE_GATE_CHECKS),
    )
    resolved = _object(receipt.get("resolved_args"), "candidate receipt.resolved_args")
    seed = _integer(
        resolved.get("seed"),
        "candidate receipt.resolved_args.seed",
        minimum=0,
        maximum=2**31 - 1,
    )
    if resolved.get("model_family") != "YOLO11n":
        raise TrainingValidationError("candidate resolved model family is unsupported")
    image_size = _integer(
        resolved.get("imgsz"), "candidate receipt.resolved_args.imgsz", minimum=32
    )
    if image_size % 32:
        raise TrainingValidationError("candidate training image size must be divisible by 32")
    training_device = _text(resolved.get("device"), "candidate receipt.resolved_args.device")
    protocol_training = _object(protocol.get("training"), "candidate receipt.protocol.training")
    expected_protocol_training = {
        key: value for key, value in resolved.items() if key not in {"model_family", "seed"}
    }
    if protocol_training != expected_protocol_training:
        raise TrainingValidationError(
            "candidate resolved arguments differ from its training protocol"
        )

    artifacts = _object(receipt.get("artifacts"), "candidate receipt.artifacts")
    best = _object(artifacts.get("best_weights"), "candidate receipt.artifacts.best_weights")
    if set(best) != {"path", "sha256", "size_bytes"}:
        raise TrainingValidationError("candidate best-weight record is malformed")
    best_relative = _safe_relative_path(best.get("path"), "candidate best-weight path")
    recorded_weight_path = receipt_path.parent.joinpath(*PurePosixPath(best_relative).parts)
    if recorded_weight_path != weight_path:
        raise TrainingValidationError("candidate weight path differs from receipt best_weights")
    weight_bytes, weight_file = _read_regular_bytes(
        weight_path, "candidate best weight", maximum_bytes=MAX_WEIGHT_BYTES
    )
    if not weight_bytes:
        raise TrainingValidationError("candidate best weight must not be empty")
    if weight_file.sha256 != _sha256(
        best.get("sha256"), "candidate best-weight sha256"
    ) or weight_file.size_bytes != _integer(
        best.get("size_bytes"), "candidate best-weight size_bytes", minimum=1
    ):
        raise TrainingValidationError("candidate best weight differs from its receipt hash")

    metrics = _object(receipt.get("metrics"), "candidate receipt.metrics")
    aggregate = _object(metrics.get("aggregate"), "candidate receipt.metrics.aggregate")
    map50 = _validate_metric(aggregate.get("map50"), "candidate receipt metrics.map50")
    map50_95 = _validate_metric(aggregate.get("map50_95"), "candidate receipt metrics.map50_95")
    timestamps = _object(receipt.get("timestamps"), "candidate receipt.timestamps")
    training_duration = _number(
        timestamps.get("duration_seconds"), "candidate receipt duration_seconds"
    )
    if training_duration < 0.0:
        raise TrainingValidationError("candidate training duration must be non-negative")
    return ValidatedCandidate(
        receipt=receipt_file,
        weight=weight_file,
        schema=dict(candidate_schema),
        seed=seed,
        protocol_sha256=protocol_sha256,
        map50=map50,
        map50_95=map50_95,
        training_duration_seconds=training_duration,
        image_size=image_size,
        training_device=training_device,
        resolved_args=dict(resolved),
    )


def _validate_settings(
    settings: EvaluationSettings, candidate: ValidatedCandidate, arm_id: str
) -> dict[str, Any]:
    confidence = _number(settings.confidence, "settings.confidence")
    if confidence != 0.40:
        raise TrainingValidationError(
            "settings.confidence must match the fixed protocol value 0.40"
        )
    image_size = (
        candidate.image_size
        if settings.image_size is None
        else _integer(settings.image_size, "settings.image_size", minimum=32)
    )
    if image_size % 32:
        raise TrainingValidationError("settings.image_size must be divisible by 32")
    if image_size != candidate.image_size:
        raise TrainingValidationError(
            "settings.image_size must match candidate receipt.resolved_args.imgsz"
        )
    expected_arm_signature = RECOVERY_ARM_SIGNATURES.get(arm_id)
    if expected_arm_signature is None:
        raise TrainingValidationError(f"arm_id {arm_id!r} is not in the frozen recovery protocol")
    for key, expected in expected_arm_signature.items():
        actual = candidate.resolved_args.get(key)
        if (
            isinstance(actual, bool)
            or not isinstance(actual, numbers.Real)
            or not math.isfinite(float(actual))
            or float(actual) != float(expected)
        ):
            raise TrainingValidationError(
                f"arm_id training signature differs at {key!r} from the candidate receipt"
            )
    device = _text(settings.device, "settings.device")
    if device != "mps":
        raise TrainingValidationError("settings.device must match the fixed protocol value 'mps'")
    if candidate.training_device != device:
        raise TrainingValidationError(
            "candidate receipt training device differs from the evaluation protocol"
        )
    nms_iou = _number(settings.nms_iou, "settings.nms_iou")
    rider_overlap = _number(settings.rider_overlap, "settings.rider_overlap")
    match_iou = _number(settings.match_iou, "settings.match_iou")
    if nms_iou != 0.75:
        raise TrainingValidationError("settings.nms_iou must match the fixed protocol value 0.75")
    if rider_overlap != 0.25:
        raise TrainingValidationError(
            "settings.rider_overlap must match the fixed protocol value 0.25"
        )
    if match_iou != 0.50:
        raise TrainingValidationError("settings.match_iou must be 0.50 to match calculate_quality")
    return {
        "confidence": confidence,
        "image_size": image_size,
        "device": device,
        "nms_iou": nms_iou,
        "rider_overlap": rider_overlap,
        "match_iou": match_iou,
    }


def _normalized_model_names(value: Any, location: str) -> dict[int, str]:
    if isinstance(value, list):
        names = {index: item for index, item in enumerate(value)}
    elif isinstance(value, Mapping):
        names: dict[int, Any] = {}
        for raw_key, item in value.items():
            if isinstance(raw_key, bool):
                raise TrainingValidationError(f"{location} has an invalid class ID")
            try:
                class_id = int(raw_key)
            except (TypeError, ValueError) as error:
                raise TrainingValidationError(f"{location} has an invalid class ID") from error
            if class_id in names:
                raise TrainingValidationError(f"{location} has duplicate class ID {class_id}")
            names[class_id] = item
    else:
        raise TrainingValidationError(f"{location} must be a list or mapping")
    expected_ids = set(range(len(CANONICAL_LABELS)))
    if set(names) != expected_ids or any(not isinstance(item, str) for item in names.values()):
        raise TrainingValidationError(f"{location} must contain exactly eight named classes")
    mapped = tuple(MODEL_MAPPING.get(names[index]) for index in range(len(CANONICAL_LABELS)))
    if mapped != CANONICAL_LABELS:
        raise TrainingValidationError(
            f"{location} does not map exactly onto the canonical eight-class order"
        )
    return {index: str(names[index]) for index in range(len(CANONICAL_LABELS))}


def ultralytics_runner(
    weight: Path,
    frames: Sequence[ValidationFrame],
    settings: Mapping[str, Any],
) -> ModelRun:
    """Load one candidate and call YOLO predict exactly once for all validation images."""

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise TrainingValidationError(
            "Ultralytics is unavailable; install the detection extra before evaluation"
        ) from error
    load_started = time.perf_counter()
    model = YOLO(str(weight))
    load_seconds = time.perf_counter() - load_started
    _normalized_model_names(model.names, "candidate model.names")
    inference_started = time.perf_counter()
    results = model.predict(
        source=[str(frame.image.path) for frame in frames],
        conf=float(settings["confidence"]),
        imgsz=int(settings["image_size"]),
        device=str(settings["device"]),
        iou=float(settings["nms_iou"]),
        stream=True,
        verbose=False,
        save=False,
    )
    raw_predictions: list[dict[str, Any]] = []
    observed: set[tuple[str, int]] = set()
    result_count = 0
    for result_index, result in enumerate(results):
        if result_index >= len(frames):
            raise TrainingValidationError("YOLO returned more results than validation images")
        result_count += 1
        frame = frames[result_index]
        frame_key = (frame.scene_id, frame.frame)
        observed.add(frame_key)
        names = _normalized_model_names(result.names, f"YOLO result[{result_index}].names")
        if result.boxes is None:
            continue
        for box_index, box in enumerate(result.boxes):
            class_id = int(box.cls.item())
            if class_id not in names:
                raise TrainingValidationError("YOLO emitted an unknown class ID")
            raw_predictions.append(
                {
                    "prediction_id": f"validation_{result_index}_{box_index}",
                    "scene_id": frame.scene_id,
                    "frame": frame.frame,
                    "model_label": names[class_id],
                    "confidence": float(box.conf.item()),
                    "bbox": [float(value) for value in box.xyxy[0].tolist()],
                    "source": "auto",
                }
            )
    inference_seconds = time.perf_counter() - inference_started
    if result_count != len(frames):
        raise TrainingValidationError("YOLO did not return one result for every validation image")
    return ModelRun(
        raw_predictions=tuple(raw_predictions),
        observed_frame_keys=frozenset(observed),
        predict_call_count=1,
        inference_wall_seconds=inference_seconds,
        model_load_seconds=load_seconds,
        model_metadata={"provider": "ultralytics", "model_class_count": len(CANONICAL_LABELS)},
    )


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _f1_from_counts(true_positive: int, false_positive: int, false_negative: int) -> float | None:
    denominator = 2 * true_positive + false_positive + false_negative
    return 2 * true_positive / denominator if denominator else None


def _validated_run(
    run: ModelRun,
    *,
    dataset: ValidatedDataset,
    settings: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_frames = frozenset((frame.scene_id, frame.frame) for frame in dataset.val_frames)
    if run.observed_frame_keys != expected_frames:
        missing = sorted(expected_frames - run.observed_frame_keys)
        unexpected = sorted(run.observed_frame_keys - expected_frames)
        raise TrainingValidationError(
            "model did not cover the complete validation frame universe; "
            f"missing={missing[:3]!r}, unexpected={unexpected[:3]!r}"
        )
    if isinstance(run.predict_call_count, bool) or run.predict_call_count != 1:
        raise TrainingValidationError("model runner must make exactly one predict call")
    inference_seconds = _number(run.inference_wall_seconds, "model inference duration")
    load_seconds = _number(run.model_load_seconds, "model load duration")
    if inference_seconds < 0.0 or load_seconds < 0.0:
        raise TrainingValidationError("model durations must be non-negative")
    frame_by_key = {(frame.scene_id, frame.frame): frame for frame in dataset.val_frames}
    prediction_ids: set[str] = set()
    mapped_predictions: list[dict[str, Any]] = []
    for index, raw in enumerate(run.raw_predictions):
        prediction = _object(raw, f"prediction[{index}]")
        prediction_id = _text(prediction.get("prediction_id"), f"prediction[{index}].id")
        if prediction_id in prediction_ids:
            raise TrainingValidationError("model emitted duplicate prediction IDs")
        prediction_ids.add(prediction_id)
        scene_id = _text(prediction.get("scene_id"), f"prediction[{index}].scene_id")
        frame_number = _integer(prediction.get("frame"), f"prediction[{index}].frame", minimum=0)
        frame_key = (scene_id, frame_number)
        if frame_key not in frame_by_key:
            raise TrainingValidationError("model emitted a prediction outside the validation split")
        model_label = _text(prediction.get("model_label"), f"prediction[{index}].model_label")
        label = MODEL_MAPPING.get(model_label)
        if label not in CANONICAL_LABELS:
            raise TrainingValidationError("model emitted an unmapped class name")
        confidence = _number(prediction.get("confidence"), f"prediction[{index}].confidence")
        if not float(settings["confidence"]) <= confidence <= 1.0:
            raise TrainingValidationError("model emitted confidence outside the fixed threshold")
        bbox = _list(prediction.get("bbox"), f"prediction[{index}].bbox")
        if len(bbox) != 4:
            raise TrainingValidationError("model bbox must contain four coordinates")
        left, top, right, bottom = [
            _number(value, f"prediction[{index}].bbox[{coordinate}]")
            for coordinate, value in enumerate(bbox)
        ]
        frame_input = frame_by_key[frame_key]
        if (
            left < 0.0
            or top < 0.0
            or right <= left
            or bottom <= top
            or right > frame_input.width + 1e-6
            or bottom > frame_input.height + 1e-6
        ):
            raise TrainingValidationError("model bbox is invalid or outside the validation image")
        mapped_predictions.append(
            {
                "prediction_id": prediction_id,
                "scene_id": scene_id,
                "frame": frame_number,
                "label": label,
                "confidence": confidence,
                "bbox": [left, top, right, bottom],
                "source": "auto",
            }
        )

    predictions, postprocessing = postprocess_predictions(
        mapped_predictions,
        nms_iou_threshold=float(settings["nms_iou"]),
        rider_overlap_threshold=float(settings["rider_overlap"]),
    )
    ground_truth = [annotation for frame in dataset.val_frames for annotation in frame.ground_truth]
    quality_result = calculate_quality(
        predictions,
        ground_truth,
        evaluated_frame_keys=expected_frames,
    )
    if not quality_result.ok:
        raise TrainingValidationError("quality calculation failed")
    quality = _object(quality_result.data, "quality result")
    if quality.get("evaluated_frame_count") != len(expected_frames):
        raise TrainingValidationError("quality calculation lost empty validation frames")

    true_positive = _integer(quality.get("retained_count"), "quality retained_count", minimum=0)
    false_positive = _integer(quality.get("removed_count"), "quality removed_count", minimum=0)
    false_negative = _integer(quality.get("added_count"), "quality added_count", minimum=0)
    prediction_count = len(predictions)
    ground_truth_count = len(ground_truth)
    if true_positive + false_positive != prediction_count:
        raise TrainingValidationError("quality prediction counts are inconsistent")
    if true_positive + false_negative != ground_truth_count:
        raise TrainingValidationError("quality ground-truth counts are inconsistent")
    precision = _ratio(true_positive, prediction_count)
    recall = _ratio(true_positive, ground_truth_count)
    clean_frames = _integer(
        quality.get("clean_frame_count"), "quality clean_frame_count", minimum=0
    )
    overall = {
        "true_positive_count": true_positive,
        "false_positive_count": false_positive,
        "false_negative_count": false_negative,
        "prediction_count": prediction_count,
        "ground_truth_count": ground_truth_count,
        "precision": precision,
        "recall": recall,
        "f1_score": _f1_from_counts(true_positive, false_positive, false_negative),
        "evaluated_frame_count": len(expected_frames),
        "clean_frame_count": clean_frames,
        "clean_frame_rate": _ratio(clean_frames, len(expected_frames)),
        "complete_frame_coverage": True,
    }

    observed_per_class = _object(quality.get("per_class"), "quality per_class")
    per_class: dict[str, dict[str, Any]] = {}
    for label in CANONICAL_LABELS:
        raw_metrics = _object(observed_per_class.get(label, {}), f"quality per_class.{label}")
        class_tp = _integer(
            raw_metrics.get("true_positive_count", 0), f"quality {label} true positives", minimum=0
        )
        class_fp = _integer(
            raw_metrics.get("false_positive_count", 0),
            f"quality {label} false positives",
            minimum=0,
        )
        class_fn = _integer(
            raw_metrics.get("false_negative_count", 0),
            f"quality {label} false negatives",
            minimum=0,
        )
        support = class_tp + class_fn
        predictions_for_class = class_tp + class_fp
        if support == 0:
            status = "not_evaluable"
            class_precision = None
            class_recall = None
            class_f1 = None
        else:
            status = "evaluable"
            class_precision = _ratio(class_tp, predictions_for_class)
            class_recall = _ratio(class_tp, support)
            class_f1 = _f1_from_counts(class_tp, class_fp, class_fn)
        per_class[label] = {
            "status": status,
            "support_count": support,
            "true_positive_count": class_tp,
            "false_positive_count": class_fp,
            "false_negative_count": class_fn,
            "prediction_count": predictions_for_class,
            "precision": class_precision,
            "recall": class_recall,
            "f1_score": class_f1,
        }
    if (
        sum(item["true_positive_count"] for item in per_class.values()) != true_positive
        or sum(item["false_positive_count"] for item in per_class.values()) != false_positive
        or sum(item["false_negative_count"] for item in per_class.values()) != false_negative
    ):
        raise TrainingValidationError("per-class quality counts do not sum to overall counts")
    return overall, {
        "per_class": per_class,
        "postprocessing": postprocessing,
        "inference_wall_seconds": inference_seconds,
        "model_load_seconds": load_seconds,
        "model_metadata": dict(run.model_metadata),
    }


def _atomic_write_json_new(output: Path, payload: Mapping[str, Any]) -> None:
    encoded = _json_bytes(payload)
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
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, output, follow_symlinks=False)
        except FileExistsError as error:
            raise TrainingValidationError(f"output already exists: {output}") from error
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _revalidate_dataset(dataset: ValidatedDataset) -> None:
    _revalidate((dataset.manifest, *dataset.managed_files), "dataset input")
    actual_files, actual_directories = _inventory_dataset(dataset.root)
    expected_files = {record["path"] for record in dataset.managed_records} | {"manifest.json"}
    if actual_files != expected_files or actual_directories != _expected_dataset_directories(
        dataset.managed_records
    ):
        raise TrainingValidationError("dataset inventory changed during evaluation")


def evaluate_training_validation(
    dataset_root: Path,
    dataset_manifest_path: Path,
    candidate_receipt_path: Path,
    candidate_weight_path: Path,
    output: Path,
    *,
    arm_id: str,
    fold_id: str,
    settings: EvaluationSettings | None = None,
    workspace_root: Path | None = None,
    runner: ModelRunner | None = None,
) -> dict[str, Any]:
    """Evaluate a receipt-bound candidate on every frame in its training val split."""

    workspace = _workspace_root(workspace_root)
    arm = _fold_identifier(arm_id, "arm_id")
    fold = _fold_identifier(fold_id, "fold_id")
    output_path = _scoped_path(
        output,
        workspace=workspace,
        scope=("data", "model-candidates"),
        location="output",
        leaf_kind="missing",
    )
    if output_path.suffix != ".json":
        raise TrainingValidationError("output must be a new .json file")
    dataset = validate_dataset(
        dataset_root,
        dataset_manifest_path,
        fold_id=fold,
        workspace_root=workspace,
    )
    _scan_declared_paths(dataset.payload, "dataset manifest")
    candidate = validate_candidate(
        candidate_receipt_path,
        candidate_weight_path,
        dataset=dataset,
        workspace_root=workspace,
    )
    validated_settings = _validate_settings(settings or EvaluationSettings(), candidate, arm)
    model_runner = runner or ultralytics_runner
    run = model_runner(candidate.weight.path, dataset.val_frames, validated_settings)
    overall, evaluated = _validated_run(run, dataset=dataset, settings=validated_settings)
    ground_truth_count = sum(len(frame.ground_truth) for frame in dataset.val_frames)
    zero_annotation_frames = sum(not frame.ground_truth for frame in dataset.val_frames)
    frame_records = [
        {
            "scene_id": frame.scene_id,
            "frame": frame.frame,
            "image_path": frame.image_relative,
            "image_sha256": frame.image.sha256,
            "label_path": frame.label_relative,
            "label_sha256": frame.label.sha256,
        }
        for frame in dataset.val_frames
    ]
    inference_seconds = evaluated.pop("inference_wall_seconds")
    load_seconds = evaluated.pop("model_load_seconds")
    evaluated.pop("model_metadata")
    payload = {
        "schema": OUTPUT_SCHEMA,
        "gate": {
            "passed": True,
            "checks": {
                "dataset_scope_training_only": True,
                "candidate_scope_model_candidates_only": True,
                "final_holdout_paths_rejected": True,
                "dataset_manifest_and_all_managed_files_verified": True,
                "candidate_receipt_dataset_binding_verified": True,
                "candidate_receipt_and_weight_hashes_verified": True,
                "fold_split_plan_and_validation_assets_bound": True,
                "complete_validation_frame_coverage": True,
                "single_fixed_scoring_threshold_predict_call": True,
                "recovery_arm_training_signature_verified": True,
                "canonical_mapping_and_production_postprocess_applied": True,
                "quality_match_iou_equals_0_50": True,
                "source_inputs_unchanged_before_publish": True,
                "atomic_no_replace_output": True,
            },
        },
        "experiment": {"arm_id": arm, "seed": candidate.seed, "fold_id": fold},
        "settings": validated_settings,
        "bindings": {
            "dataset": {
                "root": _workspace_relative(dataset.root, workspace, "dataset root"),
                "manifest": {
                    "path": "manifest.json",
                    "sha256": dataset.manifest.sha256,
                    "size_bytes": dataset.manifest.size_bytes,
                    "schema": dict(dataset.schema),
                },
                "dataset_yaml": {
                    "path": "dataset.yaml",
                    "sha256": dataset.dataset_yaml.sha256,
                    "size_bytes": dataset.dataset_yaml.size_bytes,
                },
                "managed_files_sha256": dataset.managed_files_sha256,
                "managed_file_count": len(dataset.managed_files),
                "taxonomy": dict(dataset.taxonomy),
            },
            "candidate": {
                "receipt": {
                    "path": _workspace_relative(
                        candidate.receipt.path, workspace, "candidate receipt"
                    ),
                    "sha256": candidate.receipt.sha256,
                    "size_bytes": candidate.receipt.size_bytes,
                    "schema": dict(candidate.schema),
                },
                "weight": {
                    "path": _workspace_relative(
                        candidate.weight.path, workspace, "candidate weight"
                    ),
                    "sha256": candidate.weight.sha256,
                    "size_bytes": candidate.weight.size_bytes,
                    "artifact": "best_weights",
                },
                "protocol_sha256": candidate.protocol_sha256,
            },
            "fold": {
                "binding_mode": "dataset_manifest_split_plan_and_val_assets",
                "manifest_fold_id": dataset.manifest_fold_id,
                "split_plan": dict(dataset.split_plan),
                "val_asset_ids": list(dataset.val_asset_ids),
            },
        },
        "val_source": {
            "split": "val",
            "asset_ids": list(dataset.val_asset_ids),
            "frame_count": len(dataset.val_frames),
            "zero_annotation_frame_count": zero_annotation_frames,
            "annotation_count": ground_truth_count,
            "frames_sha256": canonical_sha256(frame_records),
        },
        "metrics": {
            "map50": candidate.map50,
            "map50_95": candidate.map50_95,
            "overall": overall,
            "per_class": evaluated["per_class"],
        },
        "compute": {
            "training_duration_seconds": candidate.training_duration_seconds,
            "evaluation_inference_seconds": inference_seconds,
            "model_load_seconds": load_seconds,
            "evaluated_frames_per_second": (
                len(dataset.val_frames) / inference_seconds if inference_seconds else None
            ),
            "predict_call_count": run.predict_call_count,
        },
        "holdout_firewall": {
            "input_read": False,
            "statement": NO_HOLDOUT_STATEMENT,
            "allowed_scopes": ["data/training", "data/model-candidates"],
            "rejected_scopes": list(FINAL_HOLDOUT_REJECTED_SCOPES),
        },
    }
    _revalidate_dataset(dataset)
    _revalidate((candidate.receipt, candidate.weight), "candidate input")
    if os.path.lexists(output_path):
        raise TrainingValidationError(f"output already exists: {output_path}")
    _atomic_write_json_new(output_path, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--candidate-receipt", type=Path, required=True)
    parser.add_argument("--candidate-weight", type=Path, required=True)
    parser.add_argument("--arm-id", required=True)
    parser.add_argument("--fold-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--confidence", type=float, default=EvaluationSettings.confidence)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--device", default=EvaluationSettings.device)
    parser.add_argument("--nms-iou", type=float, default=EvaluationSettings.nms_iou)
    parser.add_argument("--rider-overlap", type=float, default=EvaluationSettings.rider_overlap)
    parser.add_argument("--match-iou", type=float, default=EvaluationSettings.match_iou)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        payload = evaluate_training_validation(
            args.dataset_root,
            args.dataset_manifest,
            args.candidate_receipt,
            args.candidate_weight,
            args.output,
            arm_id=args.arm_id,
            fold_id=args.fold_id,
            settings=EvaluationSettings(
                confidence=args.confidence,
                image_size=args.image_size,
                device=args.device,
                nms_iou=args.nms_iou,
                rider_overlap=args.rider_overlap,
                match_iou=args.match_iou,
            ),
            workspace_root=args.workspace_root,
        )
    except TrainingValidationError as error:
        parser.error(str(error))
    print(json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
