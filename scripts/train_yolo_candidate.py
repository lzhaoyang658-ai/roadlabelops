"""Train and atomically publish one evidence-bound YOLO11n candidate.

The published YOLO dataset is treated as immutable input.  Every manifest-managed
file is verified, copied into a disposable workspace, and verified again before
Ultralytics is imported.  Training and cache writes only receive workspace paths.
The final candidate contains the minimum reproducibility artifacts and is
published with an atomic no-replace rename.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import ctypes
import errno
import hashlib
import importlib.metadata
import json
import logging
import math
import numbers
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Protocol

import yaml

from roadlabelops.holdout_policy import NO_FINAL_HOLDOUT_STATEMENT

LEGACY_DATASET_SCHEMA = {"name": "roadlabelops.yolo-dataset", "version": 2}
DATASET_SCHEMA = {"name": "roadlabelops.yolo-dataset", "version": 3}
RECEIPT_SCHEMA = {"name": "roadlabelops.yolo-candidate-training", "version": 2}
PROTOCOL_SCHEMA = {"name": "roadlabelops.yolo-candidate-training-protocol", "version": 2}
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
MODEL_TO_CANONICAL = dict(zip(MODEL_LABELS, REQUIRED_LABELS, strict=True))
LEGACY_MODEL_TO_CANONICAL = {name: name for name in REQUIRED_LABELS}
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
SUPPORTED_IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})
AGGREGATE_METRIC_KEYS = {
    "metrics/precision(B)": "precision",
    "metrics/recall(B)": "recall",
    "metrics/mAP50(B)": "map50",
    "metrics/mAP50-95(B)": "map50_95",
    "fitness": "fitness",
}
RESULT_METRIC_COLUMNS = (
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
)
PER_CLASS_METRICS = ("precision", "recall", "map50", "map50_95")
OUTPUT_ARTIFACT_PATHS = {
    "args": "artifacts/args.yaml",
    "results": "artifacts/results.csv",
    "best_weights": "weights/best.pt",
    "last_weights": "weights/last.pt",
}


class CandidateTrainingError(ValueError):
    """Raised when candidate training cannot proceed or publish safely."""


class Trainer(Protocol):
    def train(self, **kwargs: Any) -> Any: ...


TrainerFactory = Callable[[str], Trainer]

TRANSFER_LOG_PATTERN = re.compile(
    r"^Remapped (?P<matched>\d+)/(?P<target>\d+) "
    r"(?:decoder )?cls head rows from pretrained weights by class name$"
)


@dataclass(frozen=True)
class ManagedFile:
    path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class FileIdentity:
    device: int
    inode: int
    size_bytes: int
    modified_ns: int
    changed_ns: int


@dataclass(frozen=True)
class ValidatedDataset:
    root: Path
    manifest_path: Path
    manifest_sha256: str
    manifest_size_bytes: int
    manifest_identity: FileIdentity
    dataset_yaml: ManagedFile
    files: tuple[ManagedFile, ...]
    managed_files_sha256: str
    counts: Mapping[str, Any]
    schema: Mapping[str, Any]
    taxonomy: Mapping[str, Any]
    model_names: tuple[str, ...]


@dataclass(frozen=True)
class ValidatedWeights:
    path: Path
    sha256: str
    size_bytes: int
    identity: FileIdentity


@dataclass(frozen=True)
class TrainingConfig:
    seed: int
    epochs: int = 100
    patience: int = 20
    imgsz: int = 640
    batch: int = 8
    device: str = "mps"
    workers: int = 0
    optimizer: str = "AdamW"
    deterministic: bool = True
    amp: bool = False
    cache: str = "disk"
    close_mosaic: int = 10
    freeze: int = 0
    lr0: float = 0.001
    lrf: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0005
    warmup_epochs: float = 3.0
    warmup_bias_lr: float = 0.0
    cls_pw: float = 0.0

    def protocol_training(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("seed")
        return payload


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
            raise CandidateTrainingError("YAML contains an unhashable mapping key") from error
        if duplicate:
            raise CandidateTrainingError(f"YAML contains duplicate key {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _object(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CandidateTrainingError(f"{location} must be an object")
    return value


def _list(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise CandidateTrainingError(f"{location} must be a list")
    return value


def _text(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CandidateTrainingError(f"{location} must be a non-empty string")
    return value


def _integer(value: Any, location: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CandidateTrainingError(f"{location} must be an integer >= {minimum}")
    return value


def _digest(value: Any, location: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise CandidateTrainingError(f"{location} must be a lowercase SHA-256 digest")
    return value


def _finite_number(value: Any, location: str, *, unit_interval: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise CandidateTrainingError(f"{location} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise CandidateTrainingError(f"{location} must be a finite number") from error
    if not math.isfinite(result):
        raise CandidateTrainingError(f"{location} must be a finite number")
    if unit_interval and not 0.0 <= result <= 1.0:
        raise CandidateTrainingError(f"{location} must be within [0, 1]")
    return 0.0 if result == 0 else result


def _identity(path: Path, location: str) -> FileIdentity:
    try:
        details = path.lstat()
    except OSError as error:
        raise CandidateTrainingError(f"Could not stat {location}: {error}") from error
    if stat.S_ISLNK(details.st_mode):
        raise CandidateTrainingError(f"{location} must not be a symbolic link")
    if not stat.S_ISREG(details.st_mode):
        raise CandidateTrainingError(f"{location} must be a regular file")
    return FileIdentity(
        details.st_dev,
        details.st_ino,
        details.st_size,
        details.st_mtime_ns,
        details.st_ctime_ns,
    )


def _read_regular_bytes(path: Path, location: str) -> tuple[bytes, FileIdentity]:
    before = _identity(path, location)
    try:
        encoded = path.read_bytes()
    except OSError as error:
        raise CandidateTrainingError(f"Could not read {location}: {error}") from error
    after = _identity(path, location)
    if before != after or len(encoded) != before.size_bytes:
        raise CandidateTrainingError(f"{location} changed while it was read")
    return encoded, after


def _load_json_bytes(encoded: bytes, location: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            encoded.decode("utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise CandidateTrainingError(f"Could not parse {location}: {error}") from error
    return _object(payload, location)


def _safe_relative_path(value: Any, location: str) -> str:
    path_text = _text(value, location)
    if "\\" in path_text or any(ord(character) < 32 for character in path_text):
        raise CandidateTrainingError(f"{location} is unsafe")
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
        raise CandidateTrainingError(f"{location} is unsafe")
    return path_text


def _assert_regular_tree_path(root: Path, relative_path: str, location: str) -> Path:
    current = root
    for part in PurePosixPath(relative_path).parts[:-1]:
        current = current / part
        try:
            details = current.lstat()
        except OSError as error:
            raise CandidateTrainingError(f"Could not stat {location} parent: {error}") from error
        if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
            raise CandidateTrainingError(f"{location} has a non-directory or symlink parent")
    target = root.joinpath(*PurePosixPath(relative_path).parts)
    _identity(target, location)
    try:
        if not target.resolve(strict=True).is_relative_to(root):
            raise CandidateTrainingError(f"{location} escapes the dataset root")
    except (OSError, RuntimeError, ValueError) as error:
        if isinstance(error, CandidateTrainingError):
            raise
        raise CandidateTrainingError(f"Could not resolve {location}: {error}") from error
    return target


def _inventory_dataset(root: Path) -> tuple[set[str], set[str]]:
    files: set[str] = set()
    directories: set[str] = set()
    for current_text, directory_names, file_names in os.walk(root, followlinks=False):
        current = Path(current_text)
        for name in directory_names:
            target = current / name
            if target.is_symlink():
                raise CandidateTrainingError(f"dataset contains symlink directory: {target}")
            relative = target.relative_to(root).as_posix()
            directories.add(relative)
        for name in file_names:
            target = current / name
            if target.is_symlink():
                raise CandidateTrainingError(f"dataset contains symlink file: {target}")
            if not target.is_file():
                raise CandidateTrainingError(f"dataset contains non-regular file: {target}")
            files.add(target.relative_to(root).as_posix())
    return files, directories


def _taxonomy_contract(model_names: Sequence[str]) -> dict[str, Any]:
    names = tuple(model_names)
    if names == REQUIRED_LABELS:
        mapping = LEGACY_MODEL_TO_CANONICAL
    elif names == MODEL_LABELS:
        mapping = MODEL_TO_CANONICAL
    else:
        raise CandidateTrainingError("dataset uses an unsupported model taxonomy")
    return {
        "canonical_names": list(REQUIRED_LABELS),
        "model_names": list(names),
        "model_to_canonical": dict(mapping),
    }


def _validate_dataset_taxonomy(
    manifest: Mapping[str, Any], schema: Mapping[str, Any]
) -> dict[str, Any]:
    if schema == LEGACY_DATASET_SCHEMA:
        if "taxonomy" in manifest:
            raise CandidateTrainingError("schema-v2 dataset manifest must not declare taxonomy")
        return _taxonomy_contract(REQUIRED_LABELS)
    taxonomy = _object(manifest.get("taxonomy"), "dataset manifest.taxonomy")
    if set(taxonomy) != {"canonical_names", "model_names", "model_to_canonical"}:
        raise CandidateTrainingError("dataset manifest.taxonomy has unexpected or missing fields")
    canonical_names = _list(
        taxonomy.get("canonical_names"), "dataset manifest.taxonomy.canonical_names"
    )
    model_names = _list(taxonomy.get("model_names"), "dataset manifest.taxonomy.model_names")
    mapping = _object(
        taxonomy.get("model_to_canonical"),
        "dataset manifest.taxonomy.model_to_canonical",
    )
    expected = _taxonomy_contract(MODEL_LABELS)
    if canonical_names != expected["canonical_names"]:
        raise CandidateTrainingError("dataset manifest canonical taxonomy is not supported")
    if model_names != expected["model_names"] or mapping != expected["model_to_canonical"]:
        raise CandidateTrainingError("dataset manifest model taxonomy mapping is not supported")
    return expected


def _validate_dataset_yaml(path: Path, expected_model_names: Sequence[str]) -> None:
    encoded, _ = _read_regular_bytes(path, "dataset.yaml")
    try:
        payload = yaml.load(encoded.decode("utf-8"), Loader=_UniqueKeyLoader)
    except (UnicodeDecodeError, yaml.YAMLError, CandidateTrainingError) as error:
        raise CandidateTrainingError(f"Could not parse dataset.yaml: {error}") from error
    data = _object(payload, "dataset.yaml")
    if set(data) != {"train", "val", "names"}:
        raise CandidateTrainingError("dataset.yaml must contain only train, val, and names")
    if data["train"] != "images/train" or data["val"] != "images/val":
        raise CandidateTrainingError("dataset.yaml train/val paths are not canonical")
    names = _object(data["names"], "dataset.yaml.names")
    expected_names = {index: name for index, name in enumerate(expected_model_names)}
    if names != expected_names:
        raise CandidateTrainingError(
            f"dataset.yaml must use the bound eight-class model taxonomy: {expected_names!r}"
        )


def _validate_label_file(path: Path, relative_path: str) -> Counter[str]:
    encoded, _ = _read_regular_bytes(path, relative_path)
    try:
        text = encoded.decode("utf-8")
    except UnicodeDecodeError as error:
        raise CandidateTrainingError(f"{relative_path} is not UTF-8") from error
    counts: Counter[str] = Counter()
    for line_number, line in enumerate(text.splitlines(), start=1):
        tokens = line.split()
        if len(tokens) != 5 or not re.fullmatch(r"[0-7]", tokens[0]):
            raise CandidateTrainingError(
                f"{relative_path}:{line_number} is not a canonical YOLO label"
            )
        coordinates: list[float] = []
        for index, token in enumerate(tokens[1:], start=1):
            try:
                value = float(token)
            except ValueError as error:
                raise CandidateTrainingError(
                    f"{relative_path}:{line_number} coordinate {index} is invalid"
                ) from error
            if not math.isfinite(value):
                raise CandidateTrainingError(
                    f"{relative_path}:{line_number} coordinate {index} is non-finite"
                )
            coordinates.append(value)
        center_x, center_y, width, height = coordinates
        if (
            not 0 <= center_x <= 1
            or not 0 <= center_y <= 1
            or not 0 < width <= 1
            or not 0 < height <= 1
            or center_x - width / 2 < -1e-12
            or center_y - height / 2 < -1e-12
            or center_x + width / 2 > 1 + 1e-12
            or center_y + height / 2 > 1 + 1e-12
        ):
            raise CandidateTrainingError(
                f"{relative_path}:{line_number} bounding box escapes normalized bounds"
            )
        counts[REQUIRED_LABELS[int(tokens[0])]] += 1
    return counts


def _category_counts(value: Any, location: str) -> dict[str, int]:
    payload = _object(value, location)
    if set(payload) != set(REQUIRED_LABELS):
        raise CandidateTrainingError(f"{location} must cover the fixed eight-class taxonomy")
    return {name: _integer(payload[name], f"{location}.{name}") for name in REQUIRED_LABELS}


def _validate_dataset_structure(
    root: Path, manifest: Mapping[str, Any], records: Sequence[ManagedFile]
) -> None:
    by_path = {record.path: record for record in records}
    if "dataset.yaml" not in by_path:
        raise CandidateTrainingError("dataset manifest does not manage dataset.yaml")
    image_paths: dict[str, set[str]] = {"train": set(), "val": set()}
    label_paths: dict[str, set[str]] = {"train": set(), "val": set()}
    for record in records:
        parts = PurePosixPath(record.path).parts
        if record.path == "dataset.yaml":
            continue
        if (
            len(parts) != 3
            or parts[0] not in {"images", "labels"}
            or parts[1]
            not in {
                "train",
                "val",
            }
        ):
            raise CandidateTrainingError(
                f"dataset manifest manages unexpected path {record.path!r}"
            )
        split = parts[1]
        if parts[0] == "images":
            if PurePosixPath(record.path).suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
                raise CandidateTrainingError(f"unsupported managed image {record.path!r}")
            stem = PurePosixPath(record.path).stem.casefold()
            if stem in image_paths[split]:
                raise CandidateTrainingError(
                    f"managed {split} images contain duplicate label stem {stem!r}"
                )
            image_paths[split].add(stem)
        else:
            if PurePosixPath(record.path).suffix != ".txt":
                raise CandidateTrainingError(f"unsupported managed label {record.path!r}")
            stem = PurePosixPath(record.path).stem.casefold()
            if stem in label_paths[split]:
                raise CandidateTrainingError(
                    f"managed {split} labels contain duplicate image stem {stem!r}"
                )
            label_paths[split].add(stem)
    for split in ("train", "val"):
        if image_paths[split] != label_paths[split]:
            raise CandidateTrainingError(f"managed {split} images and labels are not one-to-one")
        if not image_paths[split]:
            raise CandidateTrainingError(f"managed {split} split must not be empty")

    counts = _object(manifest.get("counts"), "dataset manifest.counts")
    image_counts = _object(counts.get("images"), "dataset manifest.counts.images")
    annotation_counts = _object(counts.get("annotations"), "dataset manifest.counts.annotations")
    for split in ("train", "val"):
        declared = _integer(image_counts.get(split), f"dataset manifest.counts.images.{split}")
        if declared != len(image_paths[split]):
            raise CandidateTrainingError(f"dataset manifest {split} image count is inconsistent")
    total_images = _integer(image_counts.get("total"), "dataset manifest.counts.images.total")
    if total_images != sum(len(paths) for paths in image_paths.values()):
        raise CandidateTrainingError("dataset manifest total image count is inconsistent")

    actual_split_categories: dict[str, Counter[str]] = {}
    empty_labels = 0
    for split in ("train", "val"):
        category_counts: Counter[str] = Counter()
        for record in records:
            if record.path.startswith(f"labels/{split}/"):
                per_file = _validate_label_file(root / record.path, record.path)
                category_counts.update(per_file)
                if not per_file:
                    empty_labels += 1
        actual_split_categories[split] = category_counts
        declared_total = _integer(
            annotation_counts.get(split), f"dataset manifest.counts.annotations.{split}"
        )
        if declared_total != sum(category_counts.values()):
            raise CandidateTrainingError(
                f"dataset manifest {split} annotation count is inconsistent"
            )

    declared_by_split = _object(
        annotation_counts.get("by_split_and_category"),
        "dataset manifest.counts.annotations.by_split_and_category",
    )
    actual_total: Counter[str] = Counter()
    for split in ("train", "val"):
        declared_categories = _category_counts(
            declared_by_split.get(split),
            f"dataset manifest.counts.annotations.by_split_and_category.{split}",
        )
        actual_categories = {name: actual_split_categories[split][name] for name in REQUIRED_LABELS}
        if declared_categories != actual_categories:
            raise CandidateTrainingError(
                f"dataset manifest {split} per-class annotation counts are inconsistent"
            )
        actual_total.update(actual_split_categories[split])
    if _category_counts(
        annotation_counts.get("by_category"),
        "dataset manifest.counts.annotations.by_category",
    ) != {name: actual_total[name] for name in REQUIRED_LABELS}:
        raise CandidateTrainingError("dataset manifest total per-class counts are inconsistent")
    if _integer(annotation_counts.get("total"), "dataset manifest.counts.annotations.total") != (
        sum(actual_total.values())
    ):
        raise CandidateTrainingError("dataset manifest total annotation count is inconsistent")
    if (
        _integer(
            image_counts.get("zero_annotations"),
            "dataset manifest.counts.images.zero_annotations",
        )
        != empty_labels
    ):
        raise CandidateTrainingError("dataset manifest zero-annotation image count is inconsistent")


def validate_yolo_dataset(dataset_root: Path) -> ValidatedDataset:
    """Validate a schema-v2 immutable YOLO dataset and all managed bytes."""

    supplied_root = Path(dataset_root).absolute()
    if supplied_root.is_symlink():
        raise CandidateTrainingError("dataset root must not be a symbolic link")
    try:
        root = supplied_root.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise CandidateTrainingError(f"Could not resolve dataset root: {error}") from error
    if not root.is_dir():
        raise CandidateTrainingError(f"dataset root is not a directory: {root}")
    manifest_path = root / "manifest.json"
    encoded, manifest_identity = _read_regular_bytes(manifest_path, "dataset manifest")
    manifest = _load_json_bytes(encoded, "dataset manifest")
    schema = manifest.get("schema")
    if schema not in (LEGACY_DATASET_SCHEMA, DATASET_SCHEMA):
        raise CandidateTrainingError("dataset manifest schema is unsupported")
    taxonomy = _validate_dataset_taxonomy(manifest, schema)
    gate = _object(manifest.get("gate"), "dataset manifest.gate")
    if gate.get("passed") is not True:
        raise CandidateTrainingError("dataset manifest gate has not passed")
    if "blocking_reasons" in gate and _list(
        gate.get("blocking_reasons"), "dataset manifest.gate.blocking_reasons"
    ):
        raise CandidateTrainingError("dataset manifest gate contains blocking reasons")
    checks = _object(gate.get("checks"), "dataset manifest.gate.checks")
    required_checks = (
        REQUIRED_DATASET_GATE_CHECKS
        if schema == LEGACY_DATASET_SCHEMA
        else CURRENT_DATASET_GATE_CHECKS
    )
    if not required_checks.issubset(checks) or any(value is not True for value in checks.values()):
        raise CandidateTrainingError("dataset manifest gate checks are incomplete or false")

    records: list[ManagedFile] = []
    seen_paths: set[str] = set()
    casefold_paths: set[str] = set()
    for index, raw_record in enumerate(_list(manifest.get("files"), "dataset manifest.files")):
        location = f"dataset manifest.files[{index}]"
        record = _object(raw_record, location)
        if set(record) != {"path", "sha256", "size_bytes"}:
            raise CandidateTrainingError(f"{location} has unexpected or missing fields")
        path = _safe_relative_path(record.get("path"), f"{location}.path")
        if path == "manifest.json" or path in seen_paths or path.casefold() in casefold_paths:
            raise CandidateTrainingError(f"dataset manifest contains duplicate path {path!r}")
        seen_paths.add(path)
        casefold_paths.add(path.casefold())
        managed = ManagedFile(
            path=path,
            sha256=_digest(record.get("sha256"), f"{location}.sha256"),
            size_bytes=_integer(record.get("size_bytes"), f"{location}.size_bytes"),
        )
        target = _assert_regular_tree_path(root, path, location)
        target_identity = _identity(target, location)
        if target_identity.size_bytes != managed.size_bytes or sha256(target) != managed.sha256:
            raise CandidateTrainingError(f"{location} hash or size differs from disk")
        records.append(managed)
    if not records:
        raise CandidateTrainingError("dataset manifest.files must not be empty")

    actual_files, actual_directories = _inventory_dataset(root)
    expected_files = {record.path for record in records} | {"manifest.json"}
    if actual_files != expected_files:
        raise CandidateTrainingError(
            "dataset tree must contain exactly manifest.json and manifest-managed files; "
            f"missing={sorted(expected_files - actual_files)!r}, "
            f"unmanaged={sorted(actual_files - expected_files)!r}"
        )
    expected_directories = {
        PurePosixPath(record.path).parent.as_posix()
        for record in records
        if PurePosixPath(record.path).parent.as_posix() != "."
    }
    expected_directories |= {
        parent.as_posix()
        for record in records
        for parent in PurePosixPath(record.path).parents
        if parent.as_posix() != "."
    }
    if actual_directories != expected_directories:
        raise CandidateTrainingError("dataset tree contains unexpected or missing directories")

    dataset_yaml_record = next(
        (record for record in records if record.path == "dataset.yaml"), None
    )
    if dataset_yaml_record is None:
        raise CandidateTrainingError("dataset manifest does not manage dataset.yaml")
    _validate_dataset_yaml(root / "dataset.yaml", taxonomy["model_names"])
    _validate_dataset_structure(root, manifest, records)
    canonical_records = sorted(
        (asdict(record) for record in records), key=lambda item: item["path"]
    )
    return ValidatedDataset(
        root=root,
        manifest_path=manifest_path,
        manifest_sha256=hashlib.sha256(encoded).hexdigest(),
        manifest_size_bytes=len(encoded),
        manifest_identity=manifest_identity,
        dataset_yaml=dataset_yaml_record,
        files=tuple(sorted(records, key=lambda item: item.path)),
        managed_files_sha256=_semantic_sha256(canonical_records),
        counts=_object(manifest.get("counts"), "dataset manifest.counts"),
        schema=dict(schema),
        taxonomy=taxonomy,
        model_names=tuple(taxonomy["model_names"]),
    )


def validate_base_weights(path: Path, expected_sha256: str) -> ValidatedWeights:
    supplied = Path(path).absolute()
    if supplied.name != "yolo11n.pt":
        raise CandidateTrainingError("base weights must be the explicitly named yolo11n.pt")
    expected = _digest(expected_sha256, "expected base-weight SHA-256")
    try:
        resolved = supplied.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise CandidateTrainingError(f"Could not resolve base weights: {error}") from error
    identity = _identity(supplied, "base weights")
    actual = sha256(resolved)
    if actual != expected:
        raise CandidateTrainingError(
            f"base-weight SHA-256 mismatch: expected {expected}, got {actual}"
        )
    if identity.size_bytes <= 0:
        raise CandidateTrainingError("base weights must not be empty")
    return ValidatedWeights(resolved, actual, identity.size_bytes, identity)


def _validate_config(config: TrainingConfig) -> None:
    _integer(config.seed, "seed")
    if config.seed > 2**31 - 1:
        raise CandidateTrainingError("seed must be <= 2147483647")
    for field, minimum in (
        ("epochs", 1),
        ("patience", 0),
        ("imgsz", 32),
        ("batch", 1),
        ("workers", 0),
        ("close_mosaic", 0),
        ("freeze", 0),
    ):
        _integer(getattr(config, field), field, minimum=minimum)
    if config.patience > config.epochs:
        raise CandidateTrainingError("patience must not exceed epochs")
    if config.close_mosaic > config.epochs:
        raise CandidateTrainingError("close_mosaic must not exceed epochs")
    if config.imgsz % 32:
        raise CandidateTrainingError("imgsz must be a multiple of 32")
    if not re.fullmatch(r"(?:cpu|mps|[0-9]+)", config.device):
        raise CandidateTrainingError("device must be cpu, mps, or one numeric accelerator index")
    if config.optimizer not in {"SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp"}:
        raise CandidateTrainingError("optimizer is unsupported or nondeterministic auto-selection")
    if config.deterministic is not True:
        raise CandidateTrainingError("deterministic must remain true")
    if not isinstance(config.amp, bool):
        raise CandidateTrainingError("amp must be boolean")
    if config.cache not in {"disk", "ram", "none"}:
        raise CandidateTrainingError("cache must be disk, ram, or none")
    learning_rate = _finite_number(config.lr0, "lr0")
    if not 0.0 < learning_rate <= 1.0:
        raise CandidateTrainingError("lr0 must be within (0, 1]")
    final_rate_fraction = _finite_number(config.lrf, "lrf")
    if not 0.0 < final_rate_fraction <= 1.0:
        raise CandidateTrainingError("lrf must be within (0, 1]")
    momentum = _finite_number(config.momentum, "momentum")
    if not 0.0 <= momentum < 1.0:
        raise CandidateTrainingError("momentum must be within [0, 1)")
    weight_decay = _finite_number(config.weight_decay, "weight_decay")
    if not 0.0 <= weight_decay <= 1.0:
        raise CandidateTrainingError("weight_decay must be within [0, 1]")
    warmup_epochs = _finite_number(config.warmup_epochs, "warmup_epochs")
    if warmup_epochs < 0.0:
        raise CandidateTrainingError("warmup_epochs must be non-negative")
    warmup_bias_lr = _finite_number(config.warmup_bias_lr, "warmup_bias_lr")
    if not 0.0 <= warmup_bias_lr <= 1.0:
        raise CandidateTrainingError("warmup_bias_lr must be within [0, 1]")
    class_weight_power = _finite_number(config.cls_pw, "cls_pw")
    if not 0.0 <= class_weight_power <= 1.0:
        raise CandidateTrainingError("cls_pw must be within [0, 1]")


def _copy_verified(source: Path, target: Path, expected: ManagedFile, location: str) -> None:
    before = _identity(source, location)
    if before.size_bytes != expected.size_bytes:
        raise CandidateTrainingError(f"{location} size changed before isolation copy")
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        source_descriptor = os.open(source, flags)
        with (
            os.fdopen(source_descriptor, "rb") as source_stream,
            target.open("xb") as target_stream,
        ):
            descriptor_identity = os.fstat(source_stream.fileno())
            if (
                descriptor_identity.st_dev != before.device
                or descriptor_identity.st_ino != before.inode
            ):
                raise CandidateTrainingError(f"{location} changed while opening isolation copy")
            while chunk := source_stream.read(1024 * 1024):
                digest.update(chunk)
                target_stream.write(chunk)
            target_stream.flush()
            os.fsync(target_stream.fileno())
    except OSError as error:
        raise CandidateTrainingError(f"Could not isolate {location}: {error}") from error
    after = _identity(source, location)
    copied_size = target.stat().st_size
    if (
        before != after
        or digest.hexdigest() != expected.sha256
        or copied_size != expected.size_bytes
        or sha256(target) != expected.sha256
    ):
        raise CandidateTrainingError(f"{location} changed or failed verification during copy")


def _copy_dataset_to_workspace(dataset: ValidatedDataset, workspace: Path) -> Path:
    destination = workspace / "dataset"
    destination.mkdir()
    for record in dataset.files:
        source = dataset.root.joinpath(*PurePosixPath(record.path).parts)
        target = destination.joinpath(*PurePosixPath(record.path).parts)
        _copy_verified(source, target, record, f"managed dataset file {record.path}")
    copied_files, _directories = _inventory_dataset(destination)
    if copied_files != {record.path for record in dataset.files}:
        raise CandidateTrainingError("isolated dataset contains files outside the manifest")
    return destination


def _verify_source_unchanged(dataset: ValidatedDataset, weights: ValidatedWeights) -> None:
    manifest_encoded, manifest_identity = _read_regular_bytes(
        dataset.manifest_path, "dataset manifest"
    )
    if (
        manifest_identity != dataset.manifest_identity
        or hashlib.sha256(manifest_encoded).hexdigest() != dataset.manifest_sha256
    ):
        raise CandidateTrainingError("dataset manifest changed during candidate operation")
    actual_files, _directories = _inventory_dataset(dataset.root)
    if actual_files != {record.path for record in dataset.files} | {"manifest.json"}:
        raise CandidateTrainingError("dataset inventory changed during candidate operation")
    for record in dataset.files:
        source = dataset.root.joinpath(*PurePosixPath(record.path).parts)
        if _identity(source, record.path).size_bytes != record.size_bytes:
            raise CandidateTrainingError(f"managed dataset file changed: {record.path}")
        if sha256(source) != record.sha256:
            raise CandidateTrainingError(f"managed dataset file changed: {record.path}")
    if _identity(weights.path, "base weights") != weights.identity:
        raise CandidateTrainingError("base weights changed during candidate operation")
    if sha256(weights.path) != weights.sha256:
        raise CandidateTrainingError("base weights changed during candidate operation")


def _default_trainer_factory(weight_path: str) -> Trainer:
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise CandidateTrainingError(
            "Ultralytics is not installed; install the detection extra before training"
        ) from error
    return YOLO(weight_path)


class _TransferLogHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@contextlib.contextmanager
def _capture_transfer_logs() -> Iterator[list[str]]:
    handler = _TransferLogHandler()
    logger: logging.Logger | None = None
    module = sys.modules.get("ultralytics.utils")
    candidate_logger = getattr(module, "LOGGER", None) if module is not None else None
    if isinstance(candidate_logger, logging.Logger):
        logger = candidate_logger
        logger.addHandler(handler)
    try:
        yield handler.messages
    finally:
        if logger is not None:
            logger.removeHandler(handler)


def _source_model_names(trainer: Trainer) -> dict[int, str]:
    raw = getattr(trainer, "names", None)
    if isinstance(raw, Mapping):
        items = raw.items()
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        items = enumerate(raw)
    else:
        raise CandidateTrainingError("pretrained YOLO model names are unavailable")
    names: dict[int, str] = {}
    for raw_id, raw_name in items:
        if isinstance(raw_id, bool):
            raise CandidateTrainingError("pretrained YOLO model class ID is invalid")
        try:
            class_id = int(raw_id)
        except (TypeError, ValueError, OverflowError) as error:
            raise CandidateTrainingError("pretrained YOLO model class ID is invalid") from error
        if class_id < 0 or class_id in names:
            raise CandidateTrainingError("pretrained YOLO model class IDs are invalid")
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise CandidateTrainingError("pretrained YOLO model class name is invalid")
        names[class_id] = raw_name.strip()
    if not names:
        raise CandidateTrainingError("pretrained YOLO model taxonomy is empty")
    return dict(sorted(names.items()))


def _pretrained_transfer_evidence(
    *,
    trainer: Trainer,
    source_names: Mapping[int, str],
    target_model_names: Sequence[str],
    target_canonical_names: Sequence[str],
    base_weights_sha256: str,
    captured_messages: Sequence[str],
    injected_trainer: bool,
    require_complete: bool,
) -> dict[str, Any]:
    source_lookup: dict[str, tuple[int, str]] = {}
    for source_id, source_name in source_names.items():
        normalized = source_name.strip().casefold()
        if normalized in source_lookup:
            raise CandidateTrainingError("pretrained YOLO model class names are not unique")
        source_lookup[normalized] = (source_id, source_name)
    rows: list[dict[str, Any]] = []
    for target_id, (target_name, canonical_name) in enumerate(
        zip(target_model_names, target_canonical_names, strict=True)
    ):
        source = source_lookup.get(target_name.strip().casefold())
        if source is None:
            continue
        source_id, source_name = source
        rows.append(
            {
                "target_id": target_id,
                "target_model_name": target_name,
                "canonical_name": canonical_name,
                "source_id": source_id,
                "source_model_name": source_name,
            }
        )
    if require_complete and len(rows) != len(target_model_names):
        raise CandidateTrainingError(
            "pretrained class-name mapping does not cover every target classifier row"
        )
    if not rows:
        raise CandidateTrainingError("pretrained class-name mapping matched no target rows")

    runtime_message: str | None = None
    verification_mode = "ultralytics_logger"
    for message in captured_messages:
        match = TRANSFER_LOG_PATTERN.fullmatch(message.strip())
        if match is not None:
            if runtime_message is not None:
                raise CandidateTrainingError(
                    "multiple pretrained class-transfer events were observed"
                )
            if int(match.group("matched")) != len(rows) or int(match.group("target")) != len(
                target_model_names
            ):
                raise CandidateTrainingError("runtime pretrained class-transfer count differs")
            runtime_message = message.strip()
    if runtime_message is None and injected_trainer:
        injected = getattr(trainer, "pretrained_transfer_runtime", None)
        if isinstance(injected, str) and TRANSFER_LOG_PATTERN.fullmatch(injected.strip()):
            runtime_message = injected.strip()
            verification_mode = "injected_test_double"
    if runtime_message is None:
        raise CandidateTrainingError(
            "Ultralytics did not report the required pretrained classifier-row transfer"
        )
    return {
        "schema": {"name": "roadlabelops.pretrained-class-head-transfer", "version": 1},
        "source_model": {
            "family": "YOLO11n",
            "base_weights_sha256": base_weights_sha256,
            "class_count": len(source_names),
            "names_sha256": _semantic_sha256(
                [{"id": class_id, "name": name} for class_id, name in source_names.items()]
            ),
        },
        "target": {
            "class_count": len(target_model_names),
            "model_names": list(target_model_names),
            "canonical_names": list(target_canonical_names),
        },
        "matched_rows": rows,
        "matched_row_count": len(rows),
        "target_row_count": len(target_model_names),
        "runtime_observation": {
            "verification_mode": verification_mode,
            "message": runtime_message,
            "message_sha256": hashlib.sha256(runtime_message.encode("utf-8")).hexdigest(),
            "matched_row_count": len(rows),
            "target_row_count": len(target_model_names),
        },
    }


@contextlib.contextmanager
def _isolated_environment(workspace: Path) -> Iterator[None]:
    directories = {
        "YOLO_CONFIG_DIR": workspace / "config" / "ultralytics",
        "ULTRALYTICS_CONFIG_DIR": workspace / "config" / "ultralytics",
        "XDG_CONFIG_HOME": workspace / "config",
        "XDG_CACHE_HOME": workspace / "cache",
        "TORCH_HOME": workspace / "cache" / "torch",
        "MPLCONFIGDIR": workspace / "cache" / "matplotlib",
        "TMPDIR": workspace / "tmp",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    previous = {name: os.environ.get(name) for name in directories}
    safety_environment = {
        "YOLO_AUTOINSTALL": "false",
        "YOLO_OFFLINE": "true",
    }
    previous_safety = {name: os.environ.get(name) for name in safety_environment}
    try:
        for name, directory in directories.items():
            os.environ[name] = str(directory)
        os.environ.update(safety_environment)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        for name, value in previous_safety.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _training_kwargs(config: TrainingConfig, dataset_yaml: Path, workspace: Path) -> dict[str, Any]:
    return {
        "data": str(dataset_yaml),
        "epochs": config.epochs,
        "patience": config.patience,
        "imgsz": config.imgsz,
        "batch": config.batch,
        "device": config.device,
        "workers": config.workers,
        "optimizer": config.optimizer,
        "seed": config.seed,
        "deterministic": config.deterministic,
        "amp": config.amp,
        "cache": False if config.cache == "none" else config.cache,
        "close_mosaic": config.close_mosaic,
        "freeze": config.freeze,
        "lr0": config.lr0,
        "lrf": config.lrf,
        "momentum": config.momentum,
        "weight_decay": config.weight_decay,
        "warmup_epochs": config.warmup_epochs,
        "warmup_bias_lr": config.warmup_bias_lr,
        "cls_pw": config.cls_pw,
        "project": str(workspace / "trainer-output"),
        "name": "train",
        "exist_ok": False,
        "save": True,
        "val": True,
        "plots": False,
        "resume": False,
    }


def _normalize_names(
    value: Any, location: str, expected_model_names: Sequence[str]
) -> dict[int, str]:
    if isinstance(value, list):
        names = {index: name for index, name in enumerate(value)}
    elif isinstance(value, dict):
        names = {}
        for raw_key, raw_name in value.items():
            if isinstance(raw_key, bool):
                raise CandidateTrainingError(f"{location} has an invalid class id")
            try:
                key = int(raw_key)
            except (TypeError, ValueError) as error:
                raise CandidateTrainingError(f"{location} has an invalid class id") from error
            if key in names or str(key) != str(raw_key):
                raise CandidateTrainingError(f"{location} has an invalid or duplicate class id")
            names[key] = raw_name
    else:
        raise CandidateTrainingError(f"{location} must be a list or object")
    if any(not isinstance(name, str) for name in names.values()):
        raise CandidateTrainingError(f"{location} class names must be strings")
    expected = {index: name for index, name in enumerate(expected_model_names)}
    if names != expected:
        raise CandidateTrainingError(
            "trainer metrics use the wrong eight-class names for the bound model taxonomy; "
            f"expected {expected!r}, got {names!r}"
        )
    contract = _taxonomy_contract(tuple(names[index] for index in range(len(names))))
    mapping = contract["model_to_canonical"]
    return {class_id: mapping[model_name] for class_id, model_name in names.items()}


def _sequence(value: Any, location: str) -> list[Any]:
    if isinstance(value, (str, bytes, bytearray)):
        raise CandidateTrainingError(f"{location} must be a numeric sequence")
    try:
        return list(value)
    except TypeError as error:
        raise CandidateTrainingError(f"{location} must be a numeric sequence") from error


def _validation_class_support(dataset: ValidatedDataset) -> dict[str, int]:
    annotations = _object(dataset.counts.get("annotations"), "dataset counts.annotations")
    by_split = _object(
        annotations.get("by_split_and_category"),
        "dataset counts.annotations.by_split_and_category",
    )
    return _category_counts(
        by_split.get("val"),
        "dataset counts.annotations.by_split_and_category.val",
    )


def _extract_metrics(
    metrics: Any,
    expected_model_names: Sequence[str],
    validation_support: Mapping[str, int],
) -> dict[str, Any]:
    names = _normalize_names(
        getattr(metrics, "names", None),
        "trainer metrics.names",
        expected_model_names,
    )
    results_dict = _object(getattr(metrics, "results_dict", None), "trainer metrics.results_dict")
    aggregate: dict[str, float] = {}
    for source_key, output_key in AGGREGATE_METRIC_KEYS.items():
        aggregate[output_key] = _finite_number(
            results_dict.get(source_key),
            f"trainer metrics.results_dict.{source_key}",
            unit_interval=True,
        )
    box = getattr(metrics, "box", None)
    if box is None:
        raise CandidateTrainingError("trainer metrics.box is missing")
    class_indices = _sequence(
        getattr(box, "ap_class_index", None), "trainer metrics.box.ap_class_index"
    )
    try:
        normalized_indices = [int(value) for value in class_indices]
    except (TypeError, ValueError, OverflowError) as error:
        raise CandidateTrainingError("trainer metrics class indices are invalid") from error
    supported_class_ids = [
        class_id
        for class_id, canonical_name in names.items()
        if validation_support[canonical_name] > 0
    ]
    if normalized_indices != supported_class_ids:
        raise CandidateTrainingError(
            "trainer metrics class IDs must exactly match validation-supported classes "
            "in canonical order"
        )
    arrays = {
        "precision": _sequence(getattr(box, "p", None), "trainer metrics.box.p"),
        "recall": _sequence(getattr(box, "r", None), "trainer metrics.box.r"),
        "map50": _sequence(getattr(box, "ap50", None), "trainer metrics.box.ap50"),
        "map50_95": _sequence(getattr(box, "ap", None), "trainer metrics.box.ap"),
    }
    if any(len(values) != len(supported_class_ids) for values in arrays.values()):
        raise CandidateTrainingError(
            "trainer per-class metric arrays must match validation-supported classes"
        )
    metrics_by_class_id = {
        class_id: {
            metric_name: _finite_number(
                arrays[metric_name][position],
                f"trainer per-class {names[class_id]}.{metric_name}",
                unit_interval=True,
            )
            for metric_name in ("precision", "recall", "map50", "map50_95")
        }
        for position, class_id in enumerate(supported_class_ids)
    }
    per_class: dict[str, dict[str, Any]] = {}
    for class_id, name in names.items():
        support_count = validation_support[name]
        per_class[name] = {
            "class_id": class_id,
            "status": "evaluable" if support_count else "not_evaluable",
            "support_count": support_count,
            **(
                metrics_by_class_id[class_id]
                if support_count
                else {metric_name: None for metric_name in PER_CLASS_METRICS}
            ),
        }
    return {"aggregate": aggregate, "per_class": per_class}


def _parse_results_csv(path: Path) -> dict[str, Any]:
    encoded, _identity_record = _read_regular_bytes(path, "trainer results.csv")
    try:
        lines = encoded.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise CandidateTrainingError("trainer results.csv is not UTF-8") from error
    reader = csv.reader(lines)
    try:
        raw_header = next(reader)
    except StopIteration as error:
        raise CandidateTrainingError("trainer results.csv is empty") from error
    header = [value.strip() for value in raw_header]
    if not header or len(set(header)) != len(header) or "epoch" not in header:
        raise CandidateTrainingError("trainer results.csv has an invalid header")
    if not set(RESULT_METRIC_COLUMNS).issubset(header):
        raise CandidateTrainingError("trainer results.csv is missing required validation metrics")
    rows: list[dict[str, float]] = []
    for line_number, raw_values in enumerate(reader, start=2):
        if len(raw_values) != len(header):
            raise CandidateTrainingError(f"trainer results.csv row {line_number} has wrong width")
        row: dict[str, float] = {}
        for column, raw_value in zip(header, raw_values, strict=True):
            try:
                value = float(raw_value.strip())
            except ValueError as error:
                raise CandidateTrainingError(
                    f"trainer results.csv row {line_number} column {column!r} is not numeric"
                ) from error
            if not math.isfinite(value):
                raise CandidateTrainingError(
                    f"trainer results.csv row {line_number} column {column!r} is non-finite"
                )
            row[column] = value
        if not row["epoch"].is_integer() or row["epoch"] < 0:
            raise CandidateTrainingError(f"trainer results.csv row {line_number} epoch is invalid")
        for column in RESULT_METRIC_COLUMNS:
            if not 0 <= row[column] <= 1:
                raise CandidateTrainingError(
                    f"trainer results.csv row {line_number} column {column!r} is outside [0, 1]"
                )
        rows.append(row)
    if not rows:
        raise CandidateTrainingError("trainer results.csv contains no epochs")
    raw_epochs = [int(row["epoch"]) for row in rows]
    first_epoch = raw_epochs[0]
    if first_epoch not in {0, 1} or raw_epochs != list(range(first_epoch, first_epoch + len(rows))):
        raise CandidateTrainingError(
            "trainer results.csv epochs must be contiguous from zero or one"
        )
    best_row = max(
        rows,
        key=lambda row: (
            row["metrics/mAP50-95(B)"],
            row["epoch"],
        ),
    )
    return {
        "index": int(best_row["epoch"]) - first_epoch,
        "number": int(best_row["epoch"]) - first_epoch + 1,
        "selection_fitness": best_row["metrics/mAP50-95(B)"],
        "epochs_recorded": len(rows),
    }


def _load_args_yaml(path: Path) -> dict[str, Any]:
    encoded, _ = _read_regular_bytes(path, "trainer args.yaml")
    try:
        payload = yaml.load(encoded.decode("utf-8"), Loader=_UniqueKeyLoader)
    except (UnicodeDecodeError, yaml.YAMLError, CandidateTrainingError) as error:
        raise CandidateTrainingError(f"Could not parse trainer args.yaml: {error}") from error
    return _object(payload, "trainer args.yaml")


def _validate_trainer_args(
    args: Mapping[str, Any],
    *,
    config: TrainingConfig,
    dataset_yaml: Path,
    isolated_weights: Path,
    workspace: Path,
) -> None:
    expected = {
        "epochs": config.epochs,
        "patience": config.patience,
        "imgsz": config.imgsz,
        "batch": config.batch,
        "device": config.device,
        "workers": config.workers,
        "optimizer": config.optimizer,
        "seed": config.seed,
        "deterministic": config.deterministic,
        "amp": config.amp,
        "cache": False if config.cache == "none" else config.cache,
        "close_mosaic": config.close_mosaic,
        "freeze": config.freeze,
        "lr0": config.lr0,
        "lrf": config.lrf,
        "momentum": config.momentum,
        "weight_decay": config.weight_decay,
        "warmup_epochs": config.warmup_epochs,
        "warmup_bias_lr": config.warmup_bias_lr,
        "cls_pw": config.cls_pw,
        "project": str(workspace / "trainer-output"),
        "name": "train",
        "exist_ok": False,
        "save": True,
        "val": True,
        "plots": False,
        "resume": False,
    }
    for name, value in expected.items():
        if args.get(name) != value:
            raise CandidateTrainingError(
                f"trainer args.yaml changed immutable argument {name!r}: "
                f"expected {value!r}, got {args.get(name)!r}"
            )
    for name, expected_path in (("data", dataset_yaml), ("model", isolated_weights)):
        value = args.get(name)
        if not isinstance(value, str):
            raise CandidateTrainingError(f"trainer args.yaml {name!r} path is missing")
        try:
            resolved = Path(value).resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise CandidateTrainingError(
                f"trainer args.yaml {name!r} path cannot be resolved"
            ) from error
        if resolved != expected_path.resolve(strict=True) or not resolved.is_relative_to(workspace):
            raise CandidateTrainingError(
                f"trainer args.yaml {name!r} did not use the isolated workspace input"
            )
    for name, value in args.items():
        if isinstance(value, numbers.Real) and not isinstance(value, bool):
            try:
                finite = math.isfinite(float(value))
            except (OverflowError, ValueError):
                finite = False
            if not finite:
                raise CandidateTrainingError(
                    f"trainer args.yaml contains non-finite numeric argument {name!r}"
                )
        if isinstance(value, str) and Path(value).is_absolute():
            try:
                resolved_value = Path(value).resolve(strict=False)
            except (OSError, RuntimeError) as error:
                raise CandidateTrainingError(
                    f"trainer args.yaml absolute path argument {name!r} is invalid"
                ) from error
            if not resolved_value.is_relative_to(workspace):
                raise CandidateTrainingError(
                    f"trainer args.yaml absolute path argument {name!r} escapes workspace"
                )


def _artifact_record(path: Path, relative_path: str) -> dict[str, Any]:
    identity = _identity(path, relative_path)
    if identity.size_bytes <= 0:
        raise CandidateTrainingError(f"training artifact is empty: {relative_path}")
    return {
        "path": relative_path,
        "sha256": sha256(path),
        "size_bytes": identity.size_bytes,
    }


def _write_bytes(path: Path, encoded: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _copy_artifact(source: Path, target: Path, relative_path: str) -> dict[str, Any]:
    source_identity = _identity(source, f"trainer artifact {relative_path}")
    if source_identity.size_bytes <= 0:
        raise CandidateTrainingError(f"trainer artifact is empty: {relative_path}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as source_stream, target.open("xb") as target_stream:
        shutil.copyfileobj(source_stream, target_stream, length=1024 * 1024)
        target_stream.flush()
        os.fsync(target_stream.fileno())
    if _identity(source, f"trainer artifact {relative_path}") != source_identity:
        raise CandidateTrainingError(f"trainer artifact changed while copying: {relative_path}")
    source_sha = sha256(source)
    record = _artifact_record(target, relative_path)
    if record["sha256"] != source_sha or record["size_bytes"] != source_identity.size_bytes:
        raise CandidateTrainingError(f"trainer artifact copy verification failed: {relative_path}")
    return record


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
        raise CandidateTrainingError(
            "atomic no-replace directory publication is unsupported on this platform"
        )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise CandidateTrainingError(f"output directory already exists: {output}")
    raise OSError(error_number, os.strerror(error_number), str(output))


def _fsync_directories(root: Path) -> None:
    directories = [path for path in root.rglob("*") if path.is_dir()]
    for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True) + [root]:
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _environment() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for package in ("ultralytics", "torch", "torchvision", "numpy", "PyYAML"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    git: dict[str, Any] = {"commit": None, "dirty": None}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        status_output = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
        git = {"commit": commit, "dirty": bool(status_output)}
    except (OSError, subprocess.SubprocessError):
        pass
    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": packages,
        "git": git,
    }


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise CandidateTrainingError("clock must return timezone-aware timestamps")
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _protocol(config: TrainingConfig, dataset: ValidatedDataset) -> dict[str, Any]:
    return {
        "schema": PROTOCOL_SCHEMA,
        "model_family": "YOLO11n",
        "taxonomy": dict(dataset.taxonomy),
        "training": config.protocol_training(),
        "validation_selection": {
            "primary": "mAP50-95",
            "tie_breakers": ["mAP50", "recall", "precision", "smaller_seed"],
        },
        "holdout_access": "prohibited",
    }


def _resolved_args(config: TrainingConfig) -> dict[str, Any]:
    return {"model_family": "YOLO11n", **asdict(config)}


def _preflight_summary(
    dataset: ValidatedDataset, weights: ValidatedWeights, config: TrainingConfig, output: Path
) -> dict[str, Any]:
    protocol = _protocol(config, dataset)
    return {
        "mode": "preflight",
        "mutation_performed": False,
        "output": str(output),
        "gate": {
            "passed": True,
            "checks": {
                "supported_dataset_manifest_schema_verified": True,
                "dataset_gate_verified": True,
                "all_managed_dataset_files_verified": True,
                "dataset_yaml_hash_and_taxonomy_verified": True,
                "base_yolo11n_weight_hash_verified": True,
                "isolated_workspace_copy_verified": True,
                "source_inputs_unchanged": True,
                "output_target_absent": True,
                "training_not_started": True,
                "holdout_input_not_read": True,
            },
        },
        "protocol": protocol,
        "protocol_sha256": _semantic_sha256(protocol),
        "resolved_args": _resolved_args(config),
        "inputs": {
            "dataset": {
                "manifest_schema": dict(dataset.schema),
                "manifest_sha256": dataset.manifest_sha256,
                "dataset_yaml_sha256": dataset.dataset_yaml.sha256,
                "managed_files_sha256": dataset.managed_files_sha256,
                "managed_file_count": len(dataset.files),
                "taxonomy": dict(dataset.taxonomy),
            },
            "base_weights": {
                "file_name": weights.path.name,
                "sha256": weights.sha256,
                "size_bytes": weights.size_bytes,
            },
        },
        "holdout": {
            "input_read": False,
            "statement": NO_FINAL_HOLDOUT_STATEMENT,
        },
    }


def train_yolo_candidate(
    dataset_root: Path,
    base_weights_path: Path,
    output: Path,
    *,
    dataset_manifest_sha256: str,
    weights_sha256: str,
    config: TrainingConfig,
    preflight: bool = False,
    trainer_factory: TrainerFactory | None = None,
    clock: Callable[[], datetime] = _utc_now,
) -> dict[str, Any]:
    """Validate, isolate, train, and no-replace publish one candidate."""

    _validate_config(config)
    raw_output = Path(output)
    output_root = raw_output.parent.resolve() / raw_output.name
    if os.path.lexists(output_root):
        raise CandidateTrainingError(f"output directory already exists: {output_root}")
    if not output_root.parent.is_dir():
        raise CandidateTrainingError(
            f"output parent directory does not exist: {output_root.parent}"
        )
    dataset = validate_yolo_dataset(Path(dataset_root))
    expected_manifest_sha256 = _digest(dataset_manifest_sha256, "expected dataset-manifest SHA-256")
    if dataset.manifest_sha256 != expected_manifest_sha256:
        raise CandidateTrainingError(
            "dataset-manifest SHA-256 mismatch: "
            f"expected {expected_manifest_sha256}, got {dataset.manifest_sha256}"
        )
    weights = validate_base_weights(Path(base_weights_path), weights_sha256)
    if output_root.is_relative_to(dataset.root) or dataset.root.is_relative_to(output_root):
        raise CandidateTrainingError("output must be outside the immutable dataset tree")
    if weights.path.is_relative_to(dataset.root):
        raise CandidateTrainingError("base weights must be outside the immutable dataset tree")

    workspace = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.workspace-", dir=output_root.parent)
    )
    staging: Path | None = None
    published = False
    try:
        isolated_dataset = _copy_dataset_to_workspace(dataset, workspace)
        isolated_weights = workspace / "inputs" / "yolo11n.pt"
        _copy_verified(
            weights.path,
            isolated_weights,
            ManagedFile("inputs/yolo11n.pt", weights.sha256, weights.size_bytes),
            "base weights",
        )
        _verify_source_unchanged(dataset, weights)
        if preflight:
            return _preflight_summary(dataset, weights, config, output_root)

        started_at = clock()
        monotonic_start = time.monotonic()
        training_kwargs = _training_kwargs(config, isolated_dataset / "dataset.yaml", workspace)
        factory = trainer_factory or _default_trainer_factory
        with _isolated_environment(workspace):
            trainer = factory(str(isolated_weights))
            source_model_names = _source_model_names(trainer)
            with _capture_transfer_logs() as transfer_messages:
                metrics_object = trainer.train(**training_kwargs)
            pretrained_transfer = _pretrained_transfer_evidence(
                trainer=trainer,
                source_names=source_model_names,
                target_model_names=dataset.model_names,
                target_canonical_names=REQUIRED_LABELS,
                base_weights_sha256=weights.sha256,
                captured_messages=transfer_messages,
                injected_trainer=trainer_factory is not None,
                require_complete=dataset.schema == DATASET_SCHEMA,
            )
        finished_at = clock()
        elapsed_seconds = time.monotonic() - monotonic_start
        if elapsed_seconds < 0 or not math.isfinite(elapsed_seconds):
            raise CandidateTrainingError("training duration is invalid")
        if finished_at.astimezone(timezone.utc) < started_at.astimezone(timezone.utc):
            raise CandidateTrainingError("training completion timestamp precedes its start")

        run_dir = workspace / "trainer-output" / "train"
        required_sources = {
            "args": run_dir / "args.yaml",
            "results": run_dir / "results.csv",
            "best_weights": run_dir / "weights" / "best.pt",
            "last_weights": run_dir / "weights" / "last.pt",
        }
        for name, path in required_sources.items():
            _identity(path, f"trainer {name} artifact")
            if not path.resolve(strict=True).is_relative_to(workspace):
                raise CandidateTrainingError(f"trainer {name} artifact escaped workspace")
        trainer_args = _load_args_yaml(required_sources["args"])
        _validate_trainer_args(
            trainer_args,
            config=config,
            dataset_yaml=isolated_dataset / "dataset.yaml",
            isolated_weights=isolated_weights,
            workspace=workspace,
        )
        best_epoch = _parse_results_csv(required_sources["results"])
        if best_epoch["epochs_recorded"] > config.epochs:
            raise CandidateTrainingError("trainer results.csv exceeds configured epoch count")
        metrics = _extract_metrics(
            metrics_object,
            dataset.model_names,
            _validation_class_support(dataset),
        )
        _verify_source_unchanged(dataset, weights)

        protocol = _protocol(config, dataset)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{output_root.name}.publish-", dir=output_root.parent)
        )
        artifact_records: dict[str, dict[str, Any]] = {}
        for name, source in required_sources.items():
            relative_path = OUTPUT_ARTIFACT_PATHS[name]
            artifact_records[name] = _copy_artifact(source, staging / relative_path, relative_path)
        receipt_gate_checks = {
            "supported_dataset_manifest_schema_verified": True,
            "dataset_gate_verified": True,
            "all_managed_dataset_files_verified": True,
            "dataset_yaml_hash_and_taxonomy_verified": True,
            "base_yolo11n_weight_hash_verified": True,
            "isolated_workspace_training_verified": True,
            "trainer_args_match_frozen_protocol": True,
            "support_aware_complete_eight_class_metrics_verified": True,
            "required_training_artifacts_verified": True,
            "source_inputs_unchanged": True,
            "holdout_input_not_read": True,
        }
        if dataset.schema == DATASET_SCHEMA:
            receipt_gate_checks["pretrained_class_head_transfer_verified"] = True
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "gate": {
                "passed": True,
                "checks": receipt_gate_checks,
            },
            "mutation_performed": True,
            "protocol": protocol,
            "protocol_sha256": _semantic_sha256(protocol),
            "resolved_args": _resolved_args(config),
            "inputs": {
                "dataset": {
                    "manifest": {
                        "schema": dict(dataset.schema),
                        "sha256": dataset.manifest_sha256,
                        "size_bytes": dataset.manifest_size_bytes,
                    },
                    "dataset_yaml": {
                        "sha256": dataset.dataset_yaml.sha256,
                        "size_bytes": dataset.dataset_yaml.size_bytes,
                    },
                    "managed_files_sha256": dataset.managed_files_sha256,
                    "managed_file_count": len(dataset.files),
                    "counts": dataset.counts,
                    "taxonomy": dict(dataset.taxonomy),
                },
                "base_weights": {
                    "file_name": weights.path.name,
                    "model_family": "YOLO11n",
                    "sha256": weights.sha256,
                    "size_bytes": weights.size_bytes,
                },
            },
            "timestamps": {
                "started_at": _timestamp(started_at),
                "finished_at": _timestamp(finished_at),
                "duration_seconds": elapsed_seconds,
            },
            "environment": _environment(),
            "metrics": metrics,
            "best_epoch": best_epoch,
            "artifacts": artifact_records,
            "holdout": {
                "input_read": False,
                "statement": NO_FINAL_HOLDOUT_STATEMENT,
            },
        }
        if dataset.schema == DATASET_SCHEMA:
            receipt["pretrained_transfer"] = pretrained_transfer
        _write_bytes(staging / "receipt.json", _json_bytes(receipt))
        _fsync_directories(staging)
        _atomic_publish_directory_no_replace(staging, output_root)
        published = True
        parent_descriptor = os.open(output_root.parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
        return {
            "mode": "train",
            "mutation_performed": True,
            "output": str(output_root),
            "receipt": str(output_root / "receipt.json"),
            "gate": receipt["gate"],
            "metrics": metrics,
            "best_epoch": best_epoch,
            "best_weights": artifact_records["best_weights"],
        }
    finally:
        if staging is not None and not published:
            shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(workspace, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-manifest-sha256", required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--weights-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--optimizer",
        choices=("SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp"),
        default="AdamW",
    )
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cache", choices=("disk", "ram", "none"), default="disk")
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--freeze", type=int, default=0)
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--warmup-bias-lr", type=float, default=0.0)
    parser.add_argument("--cls-pw", type=float, default=0.0)
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate and exercise isolation-copy checks without training or publishing.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = TrainingConfig(
        seed=args.seed,
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        optimizer=args.optimizer,
        deterministic=True,
        amp=args.amp,
        cache=args.cache,
        close_mosaic=args.close_mosaic,
        freeze=args.freeze,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        warmup_bias_lr=args.warmup_bias_lr,
        cls_pw=args.cls_pw,
    )
    try:
        result = train_yolo_candidate(
            args.dataset,
            args.weights,
            args.output,
            dataset_manifest_sha256=args.dataset_manifest_sha256,
            weights_sha256=args.weights_sha256,
            config=config,
            preflight=args.preflight,
        )
    except CandidateTrainingError as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
