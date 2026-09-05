"""Build an immutable YOLO dataset from an evidence-bound reference and split plan.

The split plan is deliberately independent of scene boundaries. Its version-1
schema is::

    {
      "schema": {"name": "roadlabelops.training-asset-split", "version": 1},
      "train_asset_ids": ["asset-a"],
      "val_asset_ids": ["asset-b"]
    }

Asset identities are typed: JSON integer ``1`` and JSON string ``"1"`` are
different assets. Every source asset in COCO must occur exactly once in the
plan, and all frames from one asset are therefore kept in one split even when
assets and scenes have a many-to-many relationship.  The required reference
manifest binds the exact COCO and image bytes, and SHA-256 leakage identities
prevent source-content aliases from crossing or bypassing the split.  The COCO
input must be that manifest's managed sibling ``annotations.coco.json`` rather
than a renamed or detached copy.
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
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import yaml
from PIL import Image

SPLIT_PLAN_SCHEMA = {"name": "roadlabelops.training-asset-split", "version": 1}
REFERENCE_SCHEMA = {"name": "roadlabelops.training-coco-reference", "version": 2}
OUTPUT_SCHEMA = {"name": "roadlabelops.yolo-dataset", "version": 3}
REFERENCE_REQUIRED_GATE_CHECKS = frozenset(
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
SUPPORTED_IMAGE_SUFFIXES = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


class DatasetPreparationError(ValueError):
    """Raised when a YOLO dataset cannot be published safely."""


@dataclass(frozen=True)
class ValidatedImage:
    identifier: int
    file_name: str
    source_path: Path
    sha256: str
    size_bytes: int
    width: int
    height: int
    scene_id: str
    source_asset_id: int | str
    source_leakage_group_id: str
    source_normalized_asset_frame: int

    @property
    def output_name(self) -> str:
        return PurePosixPath(self.file_name).name

    @property
    def label_name(self) -> str:
        return f"{Path(self.output_name).stem}.txt"


@dataclass(frozen=True)
class ValidatedAnnotation:
    identifier: int
    image_id: int
    category_id: int
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class ValidatedInputs:
    coco_path: Path
    coco_sha256: str
    coco_semantic_sha256: str
    reference_manifest_path: Path
    reference_manifest_sha256: str
    split_plan_path: Path
    split_plan_sha256: str
    split_plan_semantic_sha256: str
    images: tuple[ValidatedImage, ...]
    annotations: tuple[ValidatedAnnotation, ...]
    split_by_asset: Mapping[tuple[str, str], str]
    asset_values: Mapping[tuple[str, str], int | str]


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
        raise DatasetPreparationError(f"{location} must be an object")
    return value


def _list(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise DatasetPreparationError(f"{location} must be a list")
    return value


def _integer(value: Any, location: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DatasetPreparationError(f"{location} must be an integer")
    if minimum is not None and value < minimum:
        raise DatasetPreparationError(f"{location} must be at least {minimum}")
    return value


def _number(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DatasetPreparationError(f"{location} must be a number")
    try:
        result = float(value)
    except OverflowError as error:
        raise DatasetPreparationError(f"{location} must be finite") from error
    if not math.isfinite(result):
        raise DatasetPreparationError(f"{location} must be finite")
    return 0.0 if result == 0 else result


def _text(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetPreparationError(f"{location} must be a non-empty string")
    return value


def _sha256(value: Any, location: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DatasetPreparationError(f"{location} must be a lowercase SHA-256 digest")
    return value


def _asset_id(value: Any, location: str) -> int | str:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise DatasetPreparationError(f"{location} must be an integer or string")
    if isinstance(value, str) and not value.strip():
        raise DatasetPreparationError(f"{location} must be a non-empty string")
    return value


def _asset_identity(value: int | str) -> tuple[str, str]:
    return type(value).__name__, str(value)


def _leakage_group_id(value: Any, location: str) -> str:
    text = _text(value, location)
    prefix = "sha256:"
    if not text.startswith(prefix):
        raise DatasetPreparationError(f"{location} must be a sha256: content identity")
    _sha256(text[len(prefix) :], location)
    return text


def _asset_sort_key(value: int | str) -> tuple[int, int | str]:
    return (0, value) if isinstance(value, int) else (1, value)


def _read_json(path: Path, location: str) -> tuple[dict[str, Any], str, int]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise DatasetPreparationError(f"{location} does not exist: {resolved}")
    try:
        encoded = resolved.read_bytes()
        payload = json.loads(encoded.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DatasetPreparationError(f"Could not read {location}: {error}") from error
    return _object(payload, location), hashlib.sha256(encoded).hexdigest(), len(encoded)


def _safe_source_path(coco_path: Path, file_name: str) -> Path:
    if "\\" in file_name or any(ord(character) < 32 for character in file_name):
        raise DatasetPreparationError(f"COCO image file_name is unsafe: {file_name!r}")
    posix_path = PurePosixPath(file_name)
    windows_path = PureWindowsPath(file_name)
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or posix_path.as_posix() != file_name
        or not posix_path.name
        or any(part in {"", ".", ".."} for part in posix_path.parts)
    ):
        raise DatasetPreparationError(f"COCO image file_name is unsafe: {file_name!r}")
    if posix_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        raise DatasetPreparationError(f"COCO image file_name has unsupported suffix: {file_name!r}")
    source_root = coco_path.parent.resolve()
    try:
        source_path = (source_root / Path(*posix_path.parts)).resolve()
    except (OSError, RuntimeError, ValueError) as error:
        raise DatasetPreparationError(
            f"Could not resolve COCO image file_name {file_name!r}: {error}"
        ) from error
    if not source_path.is_relative_to(source_root):
        raise DatasetPreparationError(f"COCO image escapes its source directory: {file_name!r}")
    if not source_path.is_file():
        raise DatasetPreparationError(f"COCO source image does not exist: {file_name!r}")
    return source_path


def _read_image_dimensions(path: Path, location: str) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            dimensions = image.size
            image.verify()
    except (OSError, SyntaxError, ValueError) as error:
        raise DatasetPreparationError(f"{location} is not a valid image: {error}") from error
    width, height = dimensions
    if width <= 0 or height <= 0:
        raise DatasetPreparationError(f"{location} has invalid dimensions {dimensions!r}")
    return width, height


def _validate_categories(payload: Mapping[str, Any]) -> None:
    categories = _list(payload.get("categories"), "COCO.categories")
    by_id: dict[int, str] = {}
    for index, raw_category in enumerate(categories):
        category = _object(raw_category, f"COCO.categories[{index}]")
        identifier = _integer(category.get("id"), f"COCO.categories[{index}].id", minimum=1)
        name = _text(category.get("name"), f"COCO.categories[{index}].name")
        if identifier in by_id:
            raise DatasetPreparationError(f"COCO contains duplicate category id {identifier}")
        by_id[identifier] = name
    expected = {index: name for index, name in enumerate(REQUIRED_LABELS, start=1)}
    if by_id != expected:
        raise DatasetPreparationError(
            "COCO must contain the fixed eight-class taxonomy with canonical IDs; "
            f"expected {expected!r}, got {by_id!r}"
        )


def _validate_images(
    payload: Mapping[str, Any], coco_path: Path
) -> tuple[tuple[ValidatedImage, ...], dict[int, ValidatedImage]]:
    records = _list(payload.get("images"), "COCO.images")
    if not records:
        raise DatasetPreparationError("COCO.images must not be empty")
    images_by_id: dict[int, ValidatedImage] = {}
    output_basenames: dict[str, str] = {}
    output_label_names: dict[str, str] = {}
    hashes: dict[str, str] = {}
    leakage_by_asset: dict[tuple[str, str], str] = {}
    asset_by_leakage: dict[str, tuple[str, str]] = {}
    normalized_frames: set[tuple[str, int]] = set()
    for index, raw_image in enumerate(records):
        location = f"COCO.images[{index}]"
        image = _object(raw_image, location)
        identifier = _integer(image.get("id"), f"{location}.id", minimum=1)
        if identifier in images_by_id:
            raise DatasetPreparationError(f"COCO contains duplicate image id {identifier}")
        file_name = _text(image.get("file_name"), f"{location}.file_name")
        source_path = _safe_source_path(coco_path, file_name)
        declared_sha = _sha256(image.get("sha256"), f"{location}.sha256")
        actual_sha = sha256(source_path)
        if actual_sha != declared_sha:
            raise DatasetPreparationError(f"{location}.sha256 differs from source image on disk")
        if declared_sha in hashes:
            raise DatasetPreparationError(
                "COCO contains duplicate image SHA-256 for "
                f"{hashes[declared_sha]!r} and {file_name!r}"
            )
        hashes[declared_sha] = file_name
        size_bytes = source_path.stat().st_size
        width = _integer(image.get("width"), f"{location}.width", minimum=1)
        height = _integer(image.get("height"), f"{location}.height", minimum=1)
        actual_dimensions = _read_image_dimensions(source_path, location)
        if actual_dimensions != (width, height):
            raise DatasetPreparationError(
                f"{location} dimensions differ from source image on disk: "
                f"declared {(width, height)!r}, actual {actual_dimensions!r}"
            )
        scene_id = _text(image.get("scene_id"), f"{location}.scene_id")
        asset_id = _asset_id(image.get("source_asset_id"), f"{location}.source_asset_id")
        asset_identity = _asset_identity(asset_id)
        leakage_group_id = _leakage_group_id(
            image.get("source_leakage_group_id"),
            f"{location}.source_leakage_group_id",
        )
        normalized_asset_frame = _integer(
            image.get("source_normalized_asset_frame"),
            f"{location}.source_normalized_asset_frame",
            minimum=0,
        )
        existing_group = leakage_by_asset.get(asset_identity)
        if existing_group is not None and existing_group != leakage_group_id:
            raise DatasetPreparationError(
                f"source asset {asset_id!r} maps to more than one leakage group"
            )
        existing_asset = asset_by_leakage.get(leakage_group_id)
        if existing_asset is not None and existing_asset != asset_identity:
            raise DatasetPreparationError(
                f"multiple source_asset_id values alias leakage group {leakage_group_id!r}"
            )
        normalized_identity = (leakage_group_id, normalized_asset_frame)
        if normalized_identity in normalized_frames:
            raise DatasetPreparationError(
                "COCO duplicates source normalized asset frame "
                f"{leakage_group_id!r}/{normalized_asset_frame}"
            )
        leakage_by_asset[asset_identity] = leakage_group_id
        asset_by_leakage[leakage_group_id] = asset_identity
        normalized_frames.add(normalized_identity)
        validated = ValidatedImage(
            identifier=identifier,
            file_name=file_name,
            source_path=source_path,
            sha256=declared_sha,
            size_bytes=size_bytes,
            width=width,
            height=height,
            scene_id=scene_id,
            source_asset_id=asset_id,
            source_leakage_group_id=leakage_group_id,
            source_normalized_asset_frame=normalized_asset_frame,
        )
        basename_key = validated.output_name.casefold()
        if basename_key in output_basenames:
            raise DatasetPreparationError(
                "COCO image basename collision between "
                f"{output_basenames[basename_key]!r} and {file_name!r}"
            )
        output_basenames[basename_key] = file_name
        label_key = validated.label_name.casefold()
        if label_key in output_label_names:
            raise DatasetPreparationError(
                "COCO label basename collision between "
                f"{output_label_names[label_key]!r} and {file_name!r}"
            )
        output_label_names[label_key] = file_name
        images_by_id[identifier] = validated
    ordered = tuple(
        sorted(
            images_by_id.values(),
            key=lambda item: (item.output_name.casefold(), item.output_name, item.identifier),
        )
    )
    return ordered, images_by_id


def _validate_annotations(
    payload: Mapping[str, Any], images_by_id: Mapping[int, ValidatedImage]
) -> tuple[ValidatedAnnotation, ...]:
    records = _list(payload.get("annotations"), "COCO.annotations")
    identifiers: set[int] = set()
    validated: list[ValidatedAnnotation] = []
    for index, raw_annotation in enumerate(records):
        location = f"COCO.annotations[{index}]"
        annotation = _object(raw_annotation, location)
        identifier = _integer(annotation.get("id"), f"{location}.id", minimum=1)
        if identifier in identifiers:
            raise DatasetPreparationError(f"COCO contains duplicate annotation id {identifier}")
        identifiers.add(identifier)
        image_id = _integer(annotation.get("image_id"), f"{location}.image_id", minimum=1)
        if image_id not in images_by_id:
            raise DatasetPreparationError(f"{location} references unknown image id {image_id}")
        category_id = _integer(annotation.get("category_id"), f"{location}.category_id", minimum=1)
        if category_id not in range(1, len(REQUIRED_LABELS) + 1):
            raise DatasetPreparationError(
                f"{location} references unknown category id {category_id}"
            )
        raw_bbox = _list(annotation.get("bbox"), f"{location}.bbox")
        if len(raw_bbox) != 4:
            raise DatasetPreparationError(f"{location}.bbox must contain exactly four numbers")
        x, y, width, height = (
            _number(value, f"{location}.bbox[{position}]")
            for position, value in enumerate(raw_bbox)
        )
        image = images_by_id[image_id]
        if (
            x < 0
            or y < 0
            or width <= 0
            or height <= 0
            or x + width > image.width
            or y + height > image.height
        ):
            raise DatasetPreparationError(f"{location}.bbox escapes image bounds")
        area = _number(annotation.get("area"), f"{location}.area")
        expected_area = width * height
        if area <= 0 or area != expected_area:
            raise DatasetPreparationError(f"{location}.area must equal bbox width * height")
        iscrowd = _integer(annotation.get("iscrowd"), f"{location}.iscrowd", minimum=0)
        if iscrowd != 0:
            raise DatasetPreparationError(f"{location}.iscrowd must be 0 for YOLO export")
        validated.append(
            ValidatedAnnotation(
                identifier=identifier,
                image_id=image_id,
                category_id=category_id,
                bbox=(x, y, width, height),
            )
        )
    return tuple(
        sorted(
            validated,
            key=lambda item: (item.image_id, item.category_id, item.bbox, item.identifier),
        )
    )


def _validate_split_plan(
    payload: Mapping[str, Any], images: Sequence[ValidatedImage]
) -> tuple[dict[tuple[str, str], str], dict[tuple[str, str], int | str], str]:
    if payload.get("schema") != SPLIT_PLAN_SCHEMA:
        raise DatasetPreparationError("split plan schema is unsupported")
    split_by_asset: dict[tuple[str, str], str] = {}
    asset_values: dict[tuple[str, str], int | str] = {}
    semantic_lists: dict[str, list[int | str]] = {}
    for split_name, field in (("train", "train_asset_ids"), ("val", "val_asset_ids")):
        raw_values = _list(payload.get(field), f"split plan.{field}")
        if not raw_values:
            raise DatasetPreparationError(f"split plan.{field} must not be empty")
        values: list[int | str] = []
        local: set[tuple[str, str]] = set()
        for index, raw_value in enumerate(raw_values):
            value = _asset_id(raw_value, f"split plan.{field}[{index}]")
            identity = _asset_identity(value)
            if identity in local:
                raise DatasetPreparationError(
                    f"split plan.{field} contains duplicate typed asset identity {value!r}"
                )
            if identity in split_by_asset:
                raise DatasetPreparationError(
                    f"asset {value!r} occurs in both train_asset_ids and val_asset_ids"
                )
            local.add(identity)
            split_by_asset[identity] = split_name
            asset_values[identity] = value
            values.append(value)
        semantic_lists[field] = sorted(values, key=_asset_sort_key)

    coco_asset_values: dict[tuple[str, str], int | str] = {}
    for image in images:
        identity = _asset_identity(image.source_asset_id)
        coco_asset_values[identity] = image.source_asset_id
    planned = set(split_by_asset)
    actual = set(coco_asset_values)
    if planned != actual:
        missing = sorted(
            (coco_asset_values[identity] for identity in actual - planned), key=_asset_sort_key
        )
        unknown = sorted(
            (asset_values[identity] for identity in planned - actual), key=_asset_sort_key
        )
        raise DatasetPreparationError(
            "split plan asset IDs must exactly cover COCO source_asset_id values; "
            f"missing={missing!r}, unknown={unknown!r}"
        )
    split_by_leakage_group: dict[str, str] = {}
    for image in images:
        split = split_by_asset[_asset_identity(image.source_asset_id)]
        existing = split_by_leakage_group.get(image.source_leakage_group_id)
        if existing is not None and existing != split:
            raise DatasetPreparationError(
                f"source leakage group crosses train and val: {image.source_leakage_group_id!r}"
            )
        split_by_leakage_group[image.source_leakage_group_id] = split
    return (
        split_by_asset,
        coco_asset_values,
        _semantic_sha256({"schema": SPLIT_PLAN_SCHEMA, **semantic_lists}),
    )


def _manifest_relative_path(value: Any, location: str) -> str:
    path_text = _text(value, location)
    if "\\" in path_text or any(ord(character) < 32 for character in path_text):
        raise DatasetPreparationError(f"{location} is unsafe")
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
        raise DatasetPreparationError(f"{location} is unsafe")
    return path_text


def _validate_reference_manifest(
    manifest: Mapping[str, Any],
    *,
    coco_sha256: str,
    coco_size_bytes: int,
    labels_sha256: str,
    source_map_sha256: str,
    images: Sequence[ValidatedImage],
) -> None:
    if manifest.get("schema") != REFERENCE_SCHEMA:
        raise DatasetPreparationError("reference manifest schema is unsupported")
    gate = _object(manifest.get("gate"), "reference manifest.gate")
    if gate.get("passed") is not True or _list(
        gate.get("blocking_reasons"), "reference manifest.gate.blocking_reasons"
    ):
        raise DatasetPreparationError("reference manifest gate has not passed")
    checks = _object(gate.get("checks"), "reference manifest.gate.checks")
    if not REFERENCE_REQUIRED_GATE_CHECKS.issubset(checks) or any(
        value is not True for value in checks.values()
    ):
        raise DatasetPreparationError("reference manifest gate checks are incomplete or false")
    evidence = _object(manifest.get("evidence"), "reference manifest.evidence")
    labels = _object(evidence.get("labels"), "reference manifest.evidence.labels")
    source_map = _object(evidence.get("source_map"), "reference manifest.evidence.source_map")
    if _sha256(labels.get("sha256"), "reference manifest.evidence.labels.sha256") != (
        labels_sha256
    ):
        raise DatasetPreparationError("reference manifest labels hash differs from COCO info")
    if (
        _sha256(source_map.get("sha256"), "reference manifest.evidence.source_map.sha256")
        != source_map_sha256
    ):
        raise DatasetPreparationError("reference manifest source-map hash differs from COCO info")

    source_assets: dict[tuple[str, str], tuple[int | str, str]] = {}
    source_hashes: dict[str, int | str] = {}
    for index, raw_asset in enumerate(
        _list(source_map.get("assets"), "reference manifest.evidence.source_map.assets")
    ):
        location = f"reference manifest.evidence.source_map.assets[{index}]"
        asset = _object(raw_asset, location)
        asset_id = _asset_id(asset.get("asset_id"), f"{location}.asset_id")
        asset_identity = _asset_identity(asset_id)
        if asset_identity in source_assets:
            raise DatasetPreparationError(
                f"reference manifest source map contains duplicate asset_id {asset_id!r}"
            )
        source_sha256 = _sha256(asset.get("sha256"), f"{location}.sha256")
        if source_sha256 in source_hashes:
            raise DatasetPreparationError(
                "reference manifest source map aliases one source SHA-256 under multiple "
                f"asset IDs: {source_hashes[source_sha256]!r} and {asset_id!r}"
            )
        leakage_group_id = _leakage_group_id(
            asset.get("leakage_group_id"), f"{location}.leakage_group_id"
        )
        if leakage_group_id != f"sha256:{source_sha256}":
            raise DatasetPreparationError(
                f"{location}.leakage_group_id must equal the source SHA-256 identity"
            )
        source_assets[asset_identity] = (asset_id, leakage_group_id)
        source_hashes[source_sha256] = asset_id
    if not source_assets:
        raise DatasetPreparationError(
            "reference manifest.evidence.source_map.assets must not be empty"
        )
    for image in images:
        asset_identity = _asset_identity(image.source_asset_id)
        source_asset = source_assets.get(asset_identity)
        if source_asset is None:
            raise DatasetPreparationError(
                "COCO source asset is absent from the reference manifest source map: "
                f"{image.source_asset_id!r}"
            )
        if source_asset[1] != image.source_leakage_group_id:
            raise DatasetPreparationError(
                "COCO source leakage group differs from the reference manifest source map for "
                f"asset {image.source_asset_id!r}"
            )

    files_by_path: dict[str, tuple[str, int]] = {}
    for index, raw_record in enumerate(_list(manifest.get("files"), "reference manifest.files")):
        location = f"reference manifest.files[{index}]"
        record = _object(raw_record, location)
        path = _manifest_relative_path(record.get("path"), f"{location}.path")
        if path in files_by_path:
            raise DatasetPreparationError(f"reference manifest contains duplicate path {path!r}")
        files_by_path[path] = (
            _sha256(record.get("sha256"), f"{location}.sha256"),
            _integer(record.get("size_bytes"), f"{location}.size_bytes", minimum=0),
        )

    expected_paths = {"annotations.coco.json", *(image.file_name for image in images)}
    if set(files_by_path) != expected_paths:
        missing = sorted(expected_paths - set(files_by_path))
        unknown = sorted(set(files_by_path) - expected_paths)
        raise DatasetPreparationError(
            "reference manifest files must exactly cover COCO and source images; "
            f"missing={missing!r}, unknown={unknown!r}"
        )
    if files_by_path["annotations.coco.json"] != (coco_sha256, coco_size_bytes):
        raise DatasetPreparationError(
            "reference manifest annotations.coco.json hash or size differs from COCO input"
        )
    for image in images:
        if files_by_path[image.file_name] != (image.sha256, image.size_bytes):
            raise DatasetPreparationError(
                f"reference manifest hash or size differs for image {image.file_name!r}"
            )


def _validate_inputs(
    coco_path: Path, reference_manifest_path: Path, split_plan_path: Path
) -> ValidatedInputs:
    resolved_reference_manifest = reference_manifest_path.resolve()
    resolved_coco = coco_path.resolve()
    expected_coco = resolved_reference_manifest.parent / "annotations.coco.json"
    if resolved_coco != expected_coco:
        raise DatasetPreparationError(
            "COCO input must be the reference manifest's managed annotations.coco.json"
        )
    coco, coco_digest, coco_size_bytes = _read_json(coco_path, "COCO input")
    info = _object(coco.get("info"), "COCO.info")
    if info.get("schema") != REFERENCE_SCHEMA:
        raise DatasetPreparationError("COCO input is not a supported training reference")
    labels_sha256 = _sha256(info.get("labels_sha256"), "COCO.info.labels_sha256")
    source_map_sha256 = _sha256(info.get("source_map_sha256"), "COCO.info.source_map_sha256")
    _validate_categories(coco)
    images, images_by_id = _validate_images(coco, resolved_coco)
    annotations = _validate_annotations(coco, images_by_id)
    reference_manifest, reference_manifest_digest, _reference_manifest_size = _read_json(
        resolved_reference_manifest, "reference manifest"
    )
    _validate_reference_manifest(
        reference_manifest,
        coco_sha256=coco_digest,
        coco_size_bytes=coco_size_bytes,
        labels_sha256=labels_sha256,
        source_map_sha256=source_map_sha256,
        images=images,
    )
    split_plan, split_digest, _split_plan_size = _read_json(split_plan_path, "split plan")
    split_by_asset, asset_values, split_semantic_digest = _validate_split_plan(split_plan, images)
    semantic_coco = {
        "categories": [
            {"id": index, "name": name} for index, name in enumerate(REQUIRED_LABELS, start=1)
        ],
        "images": [
            {
                "id": image.identifier,
                "file_name": image.file_name,
                "width": image.width,
                "height": image.height,
                "sha256": image.sha256,
                "scene_id": image.scene_id,
                "source_asset_id": image.source_asset_id,
                "source_leakage_group_id": image.source_leakage_group_id,
                "source_normalized_asset_frame": image.source_normalized_asset_frame,
            }
            for image in sorted(images, key=lambda item: item.identifier)
        ],
        "annotations": [
            {
                "id": annotation.identifier,
                "image_id": annotation.image_id,
                "category_id": annotation.category_id,
                "bbox": list(annotation.bbox),
            }
            for annotation in sorted(annotations, key=lambda item: item.identifier)
        ],
    }
    return ValidatedInputs(
        coco_path=resolved_coco,
        coco_sha256=coco_digest,
        coco_semantic_sha256=_semantic_sha256(semantic_coco),
        reference_manifest_path=resolved_reference_manifest,
        reference_manifest_sha256=reference_manifest_digest,
        split_plan_path=split_plan_path.resolve(),
        split_plan_sha256=split_digest,
        split_plan_semantic_sha256=split_semantic_digest,
        images=images,
        annotations=annotations,
        split_by_asset=split_by_asset,
        asset_values=asset_values,
    )


def yolo_line(
    annotation: ValidatedAnnotation | Mapping[str, Any],
    image: ValidatedImage | Mapping[str, Any],
    class_index: int,
) -> str:
    """Return one canonical YOLO line after a final bounds check."""

    raw_bbox = (
        annotation.bbox if isinstance(annotation, ValidatedAnnotation) else annotation["bbox"]
    )
    x, y, width, height = [float(value) for value in raw_bbox]
    if isinstance(image, ValidatedImage):
        image_width, image_height = float(image.width), float(image.height)
    else:
        image_width, image_height = float(image["width"]), float(image["height"])
    annotation_identifier = (
        annotation.identifier
        if isinstance(annotation, ValidatedAnnotation)
        else annotation.get("id", "<unknown>")
    )
    image_identifier = (
        image.identifier
        if isinstance(image, ValidatedImage)
        else image.get(
            "id",
            annotation.image_id
            if isinstance(annotation, ValidatedAnnotation)
            else annotation.get("image_id", "<unknown>"),
        )
    )
    context = f"Annotation {annotation_identifier} for image {image_identifier}"
    if (
        not all(math.isfinite(value) for value in (x, y, width, height, image_width, image_height))
        or image_width <= 0
        or image_height <= 0
        or x < 0
        or y < 0
        or width <= 0
        or height <= 0
        or x + width > image_width
        or y + height > image_height
    ):
        raise DatasetPreparationError(f"{context} escapes image bounds")

    def serialize_axis(start: float, size: float, extent: float, axis: str) -> tuple[str, str]:
        normalized_size = size / extent
        if not math.isfinite(normalized_size) or normalized_size <= 0:
            raise DatasetPreparationError(
                f"{context} loses positive size during YOLO normalization (axis={axis})"
            )

        # The first expression preserves the historical output.  For a box whose
        # far edge is exactly on the image boundary, binary rounding can make the
        # independently normalized center and size reconstruct to 1 + 1 ULP.  In
        # that case only, retry two algebraically equivalent center calculations.
        # The size is never changed, so this resolves representation error without
        # clipping an actual out-of-bounds box or introducing an epsilon tolerance.
        center_candidates = (
            (start + size / 2) / extent,
            start / extent + normalized_size / 2,
            (start + size) / extent - normalized_size / 2,
        )
        serialized_size = repr(0.0 if normalized_size == 0 else normalized_size)
        parsed_size = float(serialized_size)
        for candidate in center_candidates:
            serialized_center = repr(0.0 if candidate == 0 else candidate)
            parsed_center = float(serialized_center)
            if (
                math.isfinite(parsed_center)
                and 0 < parsed_center < 1
                and parsed_center - parsed_size / 2 >= 0
                and parsed_center + parsed_size / 2 <= 1
            ):
                return serialized_center, serialized_size
        raise DatasetPreparationError(
            f"{context} is invalid after deterministic YOLO serialization (axis={axis})"
        )

    center_x, normalized_width = serialize_axis(x, width, image_width, "x")
    center_y, normalized_height = serialize_axis(y, height, image_height, "y")
    return f"{class_index} {center_x} {center_y} {normalized_width} {normalized_height}"


def _reject_absolute_paths(value: Any, location: str = "manifest") -> None:
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
        raise DatasetPreparationError(f"{location} contains an absolute local path")


def _write_bytes(path: Path, encoded: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _source_stat(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _copy_verified(source: Path, target: Path, expected_sha256: str) -> tuple[str, int]:
    target.parent.mkdir(parents=True, exist_ok=True)
    before = _source_stat(source)
    with source.open("rb") as source_stream, target.open("xb") as target_stream:
        shutil.copyfileobj(source_stream, target_stream, length=1024 * 1024)
        target_stream.flush()
        os.fsync(target_stream.fileno())
    after = _source_stat(source)
    if before != after:
        raise DatasetPreparationError(f"source image changed while copying: {source}")
    copied_sha = sha256(target)
    if copied_sha != expected_sha256:
        raise DatasetPreparationError(f"copied image hash mismatch: {source}")
    return copied_sha, target.stat().st_size


def _atomic_publish_directory_no_replace(staging: Path, output: Path) -> None:
    """Atomically publish a directory and refuse any existing target."""

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
        raise DatasetPreparationError(
            "atomic no-replace directory publication is unsupported on this platform"
        )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise DatasetPreparationError(f"output directory already exists: {output}")
    raise OSError(error_number, os.strerror(error_number), str(output))


def _fsync_directories(root: Path) -> None:
    directories = [path for path in root.rglob("*") if path.is_dir()]
    for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True) + [root]:
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _build_manifest(
    inputs: ValidatedInputs,
    *,
    split_images: Mapping[str, Sequence[ValidatedImage]],
    annotations_by_image: Mapping[int, Sequence[ValidatedAnnotation]],
    files: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    split_category_counts = {
        split: Counter(
            REQUIRED_LABELS[annotation.category_id - 1]
            for image in split_images[split]
            for annotation in annotations_by_image.get(image.identifier, ())
        )
        for split in ("train", "val")
    }
    total_category_counts = split_category_counts["train"] + split_category_counts["val"]
    asset_image_counts: Counter[tuple[str, str]] = Counter()
    asset_annotation_counts: Counter[tuple[str, str]] = Counter()
    asset_scenes: dict[tuple[str, str], set[str]] = defaultdict(set)
    asset_leakage_groups: dict[tuple[str, str], str] = {}
    for image in inputs.images:
        identity = _asset_identity(image.source_asset_id)
        asset_image_counts[identity] += 1
        asset_annotation_counts[identity] += len(annotations_by_image.get(image.identifier, ()))
        asset_scenes[identity].add(image.scene_id)
        asset_leakage_groups[identity] = image.source_leakage_group_id
    zero_annotation_images = sum(
        not annotations_by_image.get(image.identifier) for image in inputs.images
    )
    manifest = {
        "schema": OUTPUT_SCHEMA,
        "taxonomy": {
            "canonical_names": list(REQUIRED_LABELS),
            "model_names": list(MODEL_LABELS),
            "model_to_canonical": MODEL_TO_CANONICAL,
        },
        "gate": {
            "passed": True,
            "checks": {
                "fixed_eight_class_taxonomy": True,
                "canonical_and_model_taxonomies_bound": True,
                "source_images_hash_and_dimensions_verified": True,
                "source_image_hashes_unique": True,
                "output_basenames_unique": True,
                "annotations_and_bboxes_valid": True,
                "reference_manifest_gate_and_files_verified": True,
                "reference_manifest_source_assets_verified": True,
                "asset_split_is_nonempty_disjoint_and_complete": True,
                "source_asset_never_crosses_split": True,
                "source_leakage_group_never_crosses_split": True,
                "staged_file_hashes_verified": True,
            },
        },
        "inputs": {
            "coco": {
                "file_name": inputs.coco_path.name,
                "sha256": inputs.coco_sha256,
                "semantic_sha256": inputs.coco_semantic_sha256,
            },
            "reference_manifest": {
                "file_name": inputs.reference_manifest_path.name,
                "sha256": inputs.reference_manifest_sha256,
                "schema": REFERENCE_SCHEMA,
            },
            "split_plan": {
                "file_name": inputs.split_plan_path.name,
                "sha256": inputs.split_plan_sha256,
                "semantic_sha256": inputs.split_plan_semantic_sha256,
                "schema": SPLIT_PLAN_SCHEMA,
            },
        },
        "split": {
            "method": "explicit typed source_asset_id plan constrained by source leakage group",
            "train_asset_ids": sorted(
                (
                    inputs.asset_values[identity]
                    for identity, split in inputs.split_by_asset.items()
                    if split == "train"
                ),
                key=_asset_sort_key,
            ),
            "val_asset_ids": sorted(
                (
                    inputs.asset_values[identity]
                    for identity, split in inputs.split_by_asset.items()
                    if split == "val"
                ),
                key=_asset_sort_key,
            ),
        },
        "counts": {
            "images": {
                "total": len(inputs.images),
                "train": len(split_images["train"]),
                "val": len(split_images["val"]),
                "zero_annotations": zero_annotation_images,
            },
            "annotations": {
                "total": len(inputs.annotations),
                "train": sum(split_category_counts["train"].values()),
                "val": sum(split_category_counts["val"].values()),
                "by_category": {name: total_category_counts[name] for name in REQUIRED_LABELS},
                "by_split_and_category": {
                    split: {name: split_category_counts[split][name] for name in REQUIRED_LABELS}
                    for split in ("train", "val")
                },
            },
            "assets": {
                "total": len(inputs.asset_values),
                "train": sum(split == "train" for split in inputs.split_by_asset.values()),
                "val": sum(split == "val" for split in inputs.split_by_asset.values()),
            },
        },
        "source_statistics": {
            "assets": [
                {
                    "asset_id": inputs.asset_values[identity],
                    "leakage_group_id": asset_leakage_groups[identity],
                    "split": inputs.split_by_asset[identity],
                    "image_count": asset_image_counts[identity],
                    "annotation_count": asset_annotation_counts[identity],
                    "scene_ids": sorted(asset_scenes[identity]),
                }
                for identity in sorted(
                    inputs.asset_values,
                    key=lambda item: _asset_sort_key(inputs.asset_values[item]),
                )
            ]
        },
        "warnings": [
            f"No annotations for class: {name}"
            for name in REQUIRED_LABELS
            if total_category_counts[name] == 0
        ],
        "files": sorted((dict(record) for record in files), key=lambda item: item["path"]),
    }
    _reject_absolute_paths(manifest)
    return manifest


def prepare_yolo_dataset(
    coco_path: Path,
    split_plan_path: Path,
    output: Path,
    *,
    reference_manifest_path: Path,
) -> dict[str, Any]:
    """Validate inputs and atomically publish a new immutable YOLO dataset."""

    raw_output = Path(output)
    output_root = raw_output.parent.resolve() / raw_output.name
    if os.path.lexists(output_root):
        raise DatasetPreparationError(f"output directory already exists: {output_root}")
    inputs = _validate_inputs(Path(coco_path), Path(reference_manifest_path), Path(split_plan_path))
    split_images: dict[str, list[ValidatedImage]] = {"train": [], "val": []}
    for image in inputs.images:
        split = inputs.split_by_asset[_asset_identity(image.source_asset_id)]
        split_images[split].append(image)
    for images in split_images.values():
        images.sort(
            key=lambda item: (item.output_name.casefold(), item.output_name, item.identifier)
        )
    annotations_by_image: dict[int, list[ValidatedAnnotation]] = defaultdict(list)
    for annotation in inputs.annotations:
        annotations_by_image[annotation.image_id].append(annotation)

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.staging-", dir=str(output_root.parent.resolve())
        )
    )
    published = False
    try:
        files: list[dict[str, Any]] = []
        for split in ("train", "val"):
            (staging / "images" / split).mkdir(parents=True, exist_ok=True)
            (staging / "labels" / split).mkdir(parents=True, exist_ok=True)
            for image in split_images[split]:
                image_relative = PurePosixPath("images", split, image.output_name).as_posix()
                image_target = staging / image_relative
                copied_sha, copied_size = _copy_verified(
                    image.source_path, image_target, image.sha256
                )
                files.append(
                    {"path": image_relative, "sha256": copied_sha, "size_bytes": copied_size}
                )
                lines = [
                    yolo_line(annotation, image, annotation.category_id - 1)
                    for annotation in annotations_by_image.get(image.identifier, ())
                ]
                label_relative = PurePosixPath("labels", split, image.label_name).as_posix()
                label_target = staging / label_relative
                _write_bytes(
                    label_target,
                    ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8"),
                )
                files.append(
                    {
                        "path": label_relative,
                        "sha256": sha256(label_target),
                        "size_bytes": label_target.stat().st_size,
                    }
                )

        dataset_payload = {
            "train": "images/train",
            "val": "images/val",
            "names": {index: name for index, name in enumerate(MODEL_LABELS)},
        }
        dataset_path = staging / "dataset.yaml"
        _write_bytes(
            dataset_path,
            yaml.safe_dump(dataset_payload, sort_keys=False, allow_unicode=True).encode("utf-8"),
        )
        files.append(
            {
                "path": "dataset.yaml",
                "sha256": sha256(dataset_path),
                "size_bytes": dataset_path.stat().st_size,
            }
        )
        manifest = _build_manifest(
            inputs,
            split_images=split_images,
            annotations_by_image=annotations_by_image,
            files=files,
        )
        manifest_path = staging / "manifest.json"
        _write_bytes(manifest_path, _json_bytes(manifest))
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
        "dataset_yaml": str(output_root / "dataset.yaml"),
        "manifest": str(output_root / "manifest.json"),
        "counts": manifest["counts"],
        "gate": manifest["gate"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--split-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = prepare_yolo_dataset(
            args.coco,
            args.split_plan,
            args.output,
            reference_manifest_path=args.reference_manifest,
        )
    except DatasetPreparationError as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
