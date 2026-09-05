"""Materialize a validated Open Images V7 acquisition plan into a local training pack.

The command is a dry run unless ``--apply`` is supplied.  Applied runs derive
every download URL from the official CVDF bucket and a validated
``validation/<ImageID>.jpg`` object key; URLs from the plan are never followed.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import io
import json
import math
import os
import re
import shutil
import stat
import sys
import tempfile
import time
import warnings
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any
from urllib.parse import urlsplit

import httpx
from PIL import Image

from roadlabelops.holdout_policy import (
    FINAL_HOLDOUT_REJECTED_SCOPES,
    final_holdout_scope_reason,
)

PLAN_SCHEMA = {"name": "roadlabelops.open-images-v7-acquisition-plan", "version": 1}
OUTPUT_SCHEMA = {"name": "roadlabelops.open-images-v7-training-pack", "version": 1}
DRAFT_MANIFEST_SCHEMA = {
    "name": "roadlabelops.open-images-v7-draft-sample-manifest",
    "version": 1,
}
CANONICAL_CLASSES = (
    "car",
    "bus",
    "truck",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_light",
    "traffic_sign",
)
REQUIRED_PLAN_GATE_CHECKS = frozenset(
    {
        "all_selected_images_are_official_validation_subset",
        "all_selected_images_have_complete_attribution",
        "all_selected_images_have_explicit_cc_by_2_0",
        "all_target_boxes_are_real_instance_boxes",
        "fixed_eight_class_mapping",
        "forbidden_final_holdout_scopes_not_read",
        "immutable_no_replace_publication",
        "local_plan_only_no_network_or_download",
        "max_images_respected",
        "per_class_box_minimums_met",
        "per_class_image_minimums_met",
        "per_class_source_group_minimums_met",
        "source_group_image_cap_met",
        "special_target_row_excludes_entire_image",
    }
)
OFFICIAL_BUCKET = "open-images-dataset"
OFFICIAL_SUBSET = "validation"
OFFICIAL_HTTPS_ORIGIN = "https://open-images-dataset.s3.amazonaws.com"
CC_BY_2_0_HOSTS = frozenset({"creativecommons.org", "www.creativecommons.org"})
IMAGE_ID_PATTERN = re.compile(r"[0-9a-f]{16}")
MAX_PLAN_BYTES = 64 * 1024 * 1024
MAX_PLAN_IMAGES = 100_000
DEFAULT_MAX_IMAGE_BYTES = 32 * 1024 * 1024
MAX_CONFIGURED_IMAGE_BYTES = 256 * 1024 * 1024
DEFAULT_MAX_TOTAL_BYTES = 4 * 1024 * 1024 * 1024
MAX_CONFIGURED_TOTAL_BYTES = 64 * 1024 * 1024 * 1024
DEFAULT_MAX_IMAGE_PIXELS = 100_000_000
MAX_CONFIGURED_IMAGE_PIXELS = 500_000_000
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_CONNECT_TIMEOUT_SECONDS = 10.0
DEFAULT_RETRIES = 2
DEFAULT_RETRY_BACKOFF_SECONDS = 0.25
RETRYABLE_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})


class OpenImagesMaterializationError(ValueError):
    """Raised when a plan or downloaded image cannot be materialized safely."""


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
class PlannedBox:
    class_name: str
    label_id: str
    source: str
    confidence: float
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    is_occluded: int
    is_truncated: int


@dataclass(frozen=True)
class PlannedImage:
    image_id: str
    selection_rank: int
    file_name: str
    object_key: str
    source_group_id: str
    source_group_basis: str
    author: str
    author_profile_url: str | None
    title: str | None
    landing_url: str
    license_name: str
    license_url: str
    attribution: str
    boxes: tuple[PlannedBox, ...]


@dataclass(frozen=True)
class ValidatedPlan:
    payload: Mapping[str, Any]
    file: VerifiedFile
    semantic_sha256: str
    images: tuple[PlannedImage, ...]


@dataclass(frozen=True)
class DownloadResult:
    encoded: bytes
    attempts: int
    content_type: str | None


class _RetryableDownload(Exception):
    pass


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
        raise OpenImagesMaterializationError(f"{location} must be an object")
    return value


def _list(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise OpenImagesMaterializationError(f"{location} must be a list")
    return value


def _text(value: Any, location: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 32 for character in value)
    ):
        raise OpenImagesMaterializationError(f"{location} must be a clean non-empty string")
    return value


def _optional_text(value: Any, location: str) -> str | None:
    if value is None:
        return None
    return _text(value, location)


def _integer(
    value: Any,
    location: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise OpenImagesMaterializationError(f"{location} must be an integer")
    if minimum is not None and value < minimum:
        raise OpenImagesMaterializationError(f"{location} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise OpenImagesMaterializationError(f"{location} must be at most {maximum}")
    return value


def _number(
    value: Any,
    location: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OpenImagesMaterializationError(f"{location} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise OpenImagesMaterializationError(f"{location} must be finite")
    if minimum is not None and result < minimum:
        raise OpenImagesMaterializationError(f"{location} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise OpenImagesMaterializationError(f"{location} must be at most {maximum}")
    return 0.0 if result == 0 else result


def _sha256_value(value: Any, location: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise OpenImagesMaterializationError(f"{location} must be a lowercase SHA-256 digest")
    return value


def _identity(metadata: os.stat_result) -> FileIdentity:
    return FileIdentity(
        device=metadata.st_dev,
        inode=metadata.st_ino,
        size_bytes=metadata.st_size,
        modified_ns=metadata.st_mtime_ns,
        changed_ns=metadata.st_ctime_ns,
    )


def _absolute_lexical(path: Path, workspace_root: Path) -> Path:
    candidate = path if path.is_absolute() else workspace_root / path
    return Path(os.path.abspath(os.fspath(candidate)))


def _workspace_relative(path: Path, workspace_root: Path, location: str) -> str:
    try:
        relative = path.relative_to(workspace_root)
    except ValueError as error:
        raise OpenImagesMaterializationError(
            f"{location} must remain inside the workspace"
        ) from error
    return PurePosixPath(*relative.parts).as_posix()


def _forbidden_scope(parts: Sequence[str]) -> str | None:
    return final_holdout_scope_reason(PurePosixPath(*parts))


def _reject_forbidden_path(path: Path, workspace_root: Path, location: str) -> None:
    try:
        relative = path.relative_to(workspace_root)
    except ValueError:
        return
    scope = _forbidden_scope(relative.parts)
    if scope is not None:
        raise OpenImagesMaterializationError(
            f"{location} references forbidden {scope} scope outside the training-only contract"
        )


def _reject_symlink_chain(path: Path, workspace_root: Path, location: str) -> None:
    try:
        relative = path.relative_to(workspace_root)
    except ValueError as error:
        raise OpenImagesMaterializationError(
            f"{location} must remain inside the workspace"
        ) from error
    current = workspace_root
    for part in relative.parts:
        current /= part
        try:
            metadata = os.lstat(current)
        except OSError as error:
            raise OpenImagesMaterializationError(
                f"could not inspect {location}: {error}"
            ) from error
        if stat.S_ISLNK(metadata.st_mode):
            raise OpenImagesMaterializationError(f"{location} must not contain symlinks: {current}")


def _safe_relative(value: Any, location: str) -> str:
    text = _text(value, location)
    if "\\" in text:
        raise OpenImagesMaterializationError(f"{location} is unsafe")
    posix = PurePosixPath(text)
    windows = PureWindowsPath(text)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or posix.as_posix() != text
        or not posix.name
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        raise OpenImagesMaterializationError(f"{location} is unsafe")
    forbidden = _forbidden_scope(posix.parts)
    if forbidden is not None:
        raise OpenImagesMaterializationError(f"{location} references forbidden {forbidden} scope")
    return text


def _read_regular_bytes(
    path: Path,
    location: str,
    *,
    workspace_root: Path,
    maximum_bytes: int,
) -> tuple[bytes, VerifiedFile]:
    _reject_forbidden_path(path, workspace_root, location)
    _reject_symlink_chain(path, workspace_root, location)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise OpenImagesMaterializationError(f"could not open {location}: {error}") from error
    try:
        before_metadata = os.fstat(descriptor)
        if not stat.S_ISREG(before_metadata.st_mode):
            raise OpenImagesMaterializationError(f"{location} must be a regular file")
        if before_metadata.st_size > maximum_bytes:
            raise OpenImagesMaterializationError(f"{location} exceeds the maximum supported size")
        chunks: list[bytes] = []
        consumed = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - consumed))
            if not chunk:
                break
            consumed += len(chunk)
            if consumed > maximum_bytes:
                raise OpenImagesMaterializationError(
                    f"{location} exceeds the maximum supported size"
                )
            chunks.append(chunk)
        after_metadata = os.fstat(descriptor)
    except OSError as error:
        raise OpenImagesMaterializationError(f"could not read {location}: {error}") from error
    finally:
        os.close(descriptor)
    before = _identity(before_metadata)
    after = _identity(after_metadata)
    encoded = b"".join(chunks)
    if before != after or len(encoded) != before.size_bytes:
        raise OpenImagesMaterializationError(f"{location} changed while it was being read")
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
            raise OpenImagesMaterializationError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise OpenImagesMaterializationError(f"JSON contains non-finite number {value}")


def _parse_json(encoded: bytes, location: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OpenImagesMaterializationError(f"could not parse {location}: {error}") from error
    return _object(payload, location)


def _http_url(value: Any, location: str) -> str:
    text = _text(value, location)
    try:
        parsed = urlsplit(text)
        _port = parsed.port
    except ValueError as error:
        raise OpenImagesMaterializationError(f"{location} must be a valid HTTP(S) URL") from error
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.netloc:
        raise OpenImagesMaterializationError(f"{location} must be an absolute HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise OpenImagesMaterializationError(f"{location} must not contain credentials")
    return text


def _validate_cc_by_2_0(value: Any, location: str) -> str:
    text = _http_url(value, location)
    parsed = urlsplit(text)
    if (
        (parsed.hostname or "").casefold() not in CC_BY_2_0_HOSTS
        or parsed.path.rstrip("/").casefold() != "/licenses/by/2.0"
        or parsed.query
        or parsed.fragment
    ):
        raise OpenImagesMaterializationError(f"{location} must be an explicit CC BY 2.0 URL")
    return text


def _validate_taxonomy(payload: Mapping[str, Any]) -> dict[str, str]:
    records = _list(payload.get("taxonomy"), "plan.taxonomy")
    if len(records) != len(CANONICAL_CLASSES):
        raise OpenImagesMaterializationError("plan taxonomy must contain exactly eight classes")
    label_ids: dict[str, str] = {}
    display_names: set[str] = set()
    for index, (raw_record, expected_class) in enumerate(
        zip(records, CANONICAL_CLASSES, strict=True)
    ):
        location = f"plan.taxonomy[{index}]"
        record = _object(raw_record, location)
        if set(record) != {"display_name", "label_id", "roadlabelops_class"}:
            raise OpenImagesMaterializationError(f"{location} has unexpected or missing keys")
        class_name = _text(record.get("roadlabelops_class"), f"{location}.roadlabelops_class")
        if class_name != expected_class:
            raise OpenImagesMaterializationError("plan taxonomy order or class mapping differs")
        label_id = _text(record.get("label_id"), f"{location}.label_id")
        display_name = _text(record.get("display_name"), f"{location}.display_name")
        if label_id in label_ids.values() or display_name in display_names:
            raise OpenImagesMaterializationError("plan taxonomy identifiers must be unique")
        label_ids[class_name] = label_id
        display_names.add(display_name)
    return label_ids


def _validate_box(
    value: Any,
    location: str,
    *,
    label_ids: Mapping[str, str],
) -> PlannedBox:
    box = _object(value, location)
    expected_keys = {
        "class_name",
        "confidence",
        "is_depiction",
        "is_group_of",
        "is_inside",
        "is_occluded",
        "is_truncated",
        "label_id",
        "source",
        "xmax",
        "xmin",
        "ymax",
        "ymin",
    }
    if set(box) != expected_keys:
        raise OpenImagesMaterializationError(f"{location} has unexpected or missing keys")
    class_name = _text(box.get("class_name"), f"{location}.class_name")
    if class_name not in label_ids:
        raise OpenImagesMaterializationError(f"{location}.class_name is outside the taxonomy")
    label_id = _text(box.get("label_id"), f"{location}.label_id")
    if label_id != label_ids[class_name]:
        raise OpenImagesMaterializationError(f"{location}.label_id differs from the taxonomy")
    confidence = _number(box.get("confidence"), f"{location}.confidence")
    if confidence != 1.0:
        raise OpenImagesMaterializationError(f"{location}.confidence must equal 1")
    flags = {
        name: _integer(box.get(name), f"{location}.{name}", minimum=0, maximum=1)
        for name in (
            "is_depiction",
            "is_group_of",
            "is_inside",
            "is_occluded",
            "is_truncated",
        )
    }
    if any(flags[name] != 0 for name in ("is_depiction", "is_group_of", "is_inside")):
        raise OpenImagesMaterializationError(f"{location} is not a real instance box")
    xmin = _number(box.get("xmin"), f"{location}.xmin", minimum=0, maximum=1)
    xmax = _number(box.get("xmax"), f"{location}.xmax", minimum=0, maximum=1)
    ymin = _number(box.get("ymin"), f"{location}.ymin", minimum=0, maximum=1)
    ymax = _number(box.get("ymax"), f"{location}.ymax", minimum=0, maximum=1)
    if xmin >= xmax or ymin >= ymax:
        raise OpenImagesMaterializationError(f"{location} has an invalid normalized boundary")
    return PlannedBox(
        class_name=class_name,
        label_id=label_id,
        source=_text(box.get("source"), f"{location}.source"),
        confidence=confidence,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        is_occluded=flags["is_occluded"],
        is_truncated=flags["is_truncated"],
    )


def _validate_image(
    value: Any,
    index: int,
    *,
    label_ids: Mapping[str, str],
) -> PlannedImage:
    location = f"plan.images[{index}]"
    image = _object(value, location)
    image_id = _text(image.get("image_id"), f"{location}.image_id")
    if IMAGE_ID_PATTERN.fullmatch(image_id) is None:
        raise OpenImagesMaterializationError(
            f"{location}.image_id must contain exactly 16 lowercase hex characters"
        )
    rank = _integer(image.get("selection_rank"), f"{location}.selection_rank", minimum=1)
    if rank != index + 1:
        raise OpenImagesMaterializationError("plan image selection ranks must be consecutive")
    if image.get("subset") != OFFICIAL_SUBSET:
        raise OpenImagesMaterializationError(f"{location}.subset must equal validation")
    expected_file_name = f"{image_id}.jpg"
    expected_object_key = f"{OFFICIAL_SUBSET}/{expected_file_name}"
    expected_s3_uri = f"s3://{OFFICIAL_BUCKET}/{expected_object_key}"
    download = _object(image.get("cvdf_download"), f"{location}.cvdf_download")
    if download != {
        "bucket": OFFICIAL_BUCKET,
        "file_name": expected_file_name,
        "object_key": expected_object_key,
        "s3_uri": expected_s3_uri,
    }:
        raise OpenImagesMaterializationError(
            f"{location}.cvdf_download must name only the official validation object"
        )
    license_record = _object(image.get("license"), f"{location}.license")
    if set(license_record) != {"name", "url"} or license_record.get("name") != "CC BY 2.0":
        raise OpenImagesMaterializationError(f"{location}.license must declare CC BY 2.0")
    license_url = _validate_cc_by_2_0(license_record.get("url"), f"{location}.license.url")
    author = _text(image.get("author"), f"{location}.author")
    landing_url = _http_url(image.get("landing_url"), f"{location}.landing_url")
    profile = image.get("author_profile_url")
    author_profile_url = (
        _http_url(profile, f"{location}.author_profile_url") if profile is not None else None
    )
    title = _optional_text(image.get("title"), f"{location}.title")
    attribution = _text(image.get("attribution"), f"{location}.attribution")
    if (
        author not in attribution
        or landing_url not in attribution
        or license_url not in attribution
    ):
        raise OpenImagesMaterializationError(f"{location}.attribution is incomplete")
    group_id = _text(image.get("source_group_id"), f"{location}.source_group_id")
    group_basis = _text(image.get("source_group_basis"), f"{location}.source_group_basis")
    raw_boxes = _list(image.get("boxes"), f"{location}.boxes")
    if not raw_boxes:
        raise OpenImagesMaterializationError(f"{location}.boxes must not be empty")
    boxes = tuple(
        _validate_box(raw_box, f"{location}.boxes[{box_index}]", label_ids=label_ids)
        for box_index, raw_box in enumerate(raw_boxes)
    )
    identities = {(box.class_name, box.xmin, box.ymin, box.xmax, box.ymax) for box in boxes}
    if len(identities) != len(boxes):
        raise OpenImagesMaterializationError(f"{location}.boxes contains duplicates")
    raw_counts = _object(image.get("box_counts"), f"{location}.box_counts")
    if set(raw_counts) != set(CANONICAL_CLASSES):
        raise OpenImagesMaterializationError(f"{location}.box_counts must cover the taxonomy")
    actual_counts = Counter(box.class_name for box in boxes)
    for class_name in CANONICAL_CLASSES:
        count = _integer(
            raw_counts.get(class_name), f"{location}.box_counts.{class_name}", minimum=0
        )
        if count != actual_counts[class_name]:
            raise OpenImagesMaterializationError(
                f"{location}.box_counts.{class_name} differs from boxes"
            )
    return PlannedImage(
        image_id=image_id,
        selection_rank=rank,
        file_name=expected_file_name,
        object_key=expected_object_key,
        source_group_id=group_id,
        source_group_basis=group_basis,
        author=author,
        author_profile_url=author_profile_url,
        title=title,
        landing_url=landing_url,
        license_name="CC BY 2.0",
        license_url=license_url,
        attribution=attribution,
        boxes=boxes,
    )


def _validate_source_groups(payload: Mapping[str, Any], images: Sequence[PlannedImage]) -> None:
    selection = _object(payload.get("selection"), "plan.selection")
    raw_groups = _list(selection.get("source_groups"), "plan.selection.source_groups")
    images_by_group: dict[str, list[PlannedImage]] = {}
    for image in images:
        images_by_group.setdefault(image.source_group_id, []).append(image)
    if len(raw_groups) != len(images_by_group):
        raise OpenImagesMaterializationError("plan source-group count differs from selected images")
    seen: set[str] = set()
    for index, raw_group in enumerate(raw_groups):
        location = f"plan.selection.source_groups[{index}]"
        group = _object(raw_group, location)
        semantic = _sha256_value(
            group.get("manifest_semantic_sha256"), f"{location}.manifest_semantic_sha256"
        )
        unsigned = dict(group)
        unsigned.pop("manifest_semantic_sha256")
        if _semantic_sha256(unsigned) != semantic:
            raise OpenImagesMaterializationError(f"{location} semantic hash differs")
        group_id = _text(group.get("source_group_id"), f"{location}.source_group_id")
        if group_id in seen or group_id not in images_by_group:
            raise OpenImagesMaterializationError(f"{location}.source_group_id is invalid")
        seen.add(group_id)
        group_images = images_by_group[group_id]
        image_ids = _list(group.get("image_ids"), f"{location}.image_ids")
        expected_ids = sorted(image.image_id for image in group_images)
        if image_ids != expected_ids:
            raise OpenImagesMaterializationError(
                f"{location}.image_ids differs from selected images"
            )
        if _integer(group.get("image_count"), f"{location}.image_count", minimum=1) != len(
            group_images
        ):
            raise OpenImagesMaterializationError(f"{location}.image_count differs")
        bases = {image.source_group_basis for image in group_images}
        if len(bases) != 1 or group.get("source_group_basis") not in bases:
            raise OpenImagesMaterializationError(f"{location}.source_group_basis differs")
    if seen != set(images_by_group):
        raise OpenImagesMaterializationError("plan source groups do not cover selected images")


def _validate_plan(plan_path: Path, *, workspace_root: Path) -> ValidatedPlan:
    path = _absolute_lexical(plan_path, workspace_root)
    _workspace_relative(path, workspace_root, "acquisition plan")
    _reject_forbidden_path(path, workspace_root, "acquisition plan")
    if path.suffix.casefold() != ".json":
        raise OpenImagesMaterializationError("acquisition plan must have a .json suffix")
    encoded, verified = _read_regular_bytes(
        path,
        "acquisition plan",
        workspace_root=workspace_root,
        maximum_bytes=MAX_PLAN_BYTES,
    )
    payload = _parse_json(encoded, "acquisition plan")
    if payload.get("schema") != PLAN_SCHEMA:
        raise OpenImagesMaterializationError("acquisition plan schema is unsupported")
    semantic = _sha256_value(payload.get("plan_semantic_sha256"), "plan.plan_semantic_sha256")
    unsigned = dict(payload)
    unsigned.pop("plan_semantic_sha256")
    if _semantic_sha256(unsigned) != semantic:
        raise OpenImagesMaterializationError("acquisition plan semantic hash differs")
    if payload.get("dataset") != {
        "name": "Open Images",
        "project_usage": "training_only",
        "upstream_subset": OFFICIAL_SUBSET,
        "version": "V7",
    }:
        raise OpenImagesMaterializationError("acquisition plan dataset declaration differs")
    gate = _object(payload.get("gate"), "plan.gate")
    checks = _object(gate.get("checks"), "plan.gate.checks")
    blocking = _list(gate.get("blocking_reasons"), "plan.gate.blocking_reasons")
    if (
        gate.get("passed") is not True
        or blocking
        or not REQUIRED_PLAN_GATE_CHECKS.issubset(checks)
        or any(value is not True for value in checks.values())
    ):
        raise OpenImagesMaterializationError("acquisition plan gate did not pass every check")
    firewall = _object(payload.get("holdout_firewall"), "plan.holdout_firewall")
    if (
        firewall.get("downloads_performed") is not False
        or firewall.get("network_accessed") is not False
        or firewall.get("rejected_scopes") != list(FINAL_HOLDOUT_REJECTED_SCOPES)
    ):
        raise OpenImagesMaterializationError("acquisition plan must be local-only")
    inputs = _object(payload.get("inputs"), "plan.inputs")
    if set(inputs) != {"boxable_class_descriptions", "bounding_boxes", "image_metadata"}:
        raise OpenImagesMaterializationError("acquisition plan input bindings differ")
    for name, raw_binding in inputs.items():
        binding = _object(raw_binding, f"plan.inputs.{name}")
        if set(binding) != {"path", "sha256", "size_bytes"}:
            raise OpenImagesMaterializationError(f"plan.inputs.{name} has unexpected keys")
        _safe_relative(binding.get("path"), f"plan.inputs.{name}.path")
        _sha256_value(binding.get("sha256"), f"plan.inputs.{name}.sha256")
        _integer(binding.get("size_bytes"), f"plan.inputs.{name}.size_bytes", minimum=1)
    label_ids = _validate_taxonomy(payload)
    raw_images = _list(payload.get("images"), "plan.images")
    if not raw_images or len(raw_images) > MAX_PLAN_IMAGES:
        raise OpenImagesMaterializationError(
            f"plan.images must contain between 1 and {MAX_PLAN_IMAGES} entries"
        )
    images = tuple(
        _validate_image(raw_image, index, label_ids=label_ids)
        for index, raw_image in enumerate(raw_images)
    )
    if len({image.image_id for image in images}) != len(images):
        raise OpenImagesMaterializationError("plan image IDs must be unique")
    _validate_source_groups(payload, images)
    counts = _object(payload.get("counts"), "plan.counts")
    if _integer(counts.get("selected_images"), "plan.counts.selected_images", minimum=1) != len(
        images
    ):
        raise OpenImagesMaterializationError("plan selected-image count differs")
    if _integer(counts.get("selected_boxes"), "plan.counts.selected_boxes", minimum=1) != sum(
        len(image.boxes) for image in images
    ):
        raise OpenImagesMaterializationError("plan selected-box count differs")
    if _integer(
        counts.get("selected_source_groups"),
        "plan.counts.selected_source_groups",
        minimum=1,
    ) != len({image.source_group_id for image in images}):
        raise OpenImagesMaterializationError("plan selected-source-group count differs")
    return ValidatedPlan(payload=payload, file=verified, semantic_sha256=semantic, images=images)


def _validate_limits(
    *,
    max_image_bytes: int,
    max_total_bytes: int,
    max_image_pixels: int,
    timeout_seconds: float,
    connect_timeout_seconds: float,
    retries: int,
    retry_backoff_seconds: float,
) -> None:
    _integer(max_image_bytes, "max_image_bytes", minimum=1, maximum=MAX_CONFIGURED_IMAGE_BYTES)
    _integer(max_total_bytes, "max_total_bytes", minimum=1, maximum=MAX_CONFIGURED_TOTAL_BYTES)
    _integer(max_image_pixels, "max_image_pixels", minimum=1, maximum=MAX_CONFIGURED_IMAGE_PIXELS)
    _number(timeout_seconds, "timeout_seconds", minimum=0.1, maximum=300)
    _number(connect_timeout_seconds, "connect_timeout_seconds", minimum=0.1, maximum=300)
    _integer(retries, "retries", minimum=0, maximum=10)
    _number(retry_backoff_seconds, "retry_backoff_seconds", minimum=0, maximum=60)


def _ensure_output_parent(path: Path, workspace_root: Path) -> None:
    try:
        relative = path.relative_to(workspace_root)
    except ValueError as error:
        raise OpenImagesMaterializationError(
            "output parent must remain inside the workspace"
        ) from error
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
                raise OpenImagesMaterializationError(
                    f"could not create output parent {current}: {error}"
                ) from error
        except OSError as error:
            raise OpenImagesMaterializationError(
                f"could not inspect output parent {current}: {error}"
            ) from error
        if stat.S_ISLNK(metadata.st_mode):
            raise OpenImagesMaterializationError(
                f"output parent must not contain symlinks: {current}"
            )
        if not stat.S_ISDIR(metadata.st_mode):
            raise OpenImagesMaterializationError(
                f"output parent component is not a directory: {current}"
            )


def _validate_existing_output_ancestors(path: Path, workspace_root: Path) -> None:
    relative = path.relative_to(workspace_root)
    current = workspace_root
    for part in relative.parts:
        current /= part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            return
        except OSError as error:
            raise OpenImagesMaterializationError(
                f"could not inspect output path: {error}"
            ) from error
        if stat.S_ISLNK(metadata.st_mode):
            raise OpenImagesMaterializationError(
                f"output path must not contain symlinks: {current}"
            )
        if current != path and not stat.S_ISDIR(metadata.st_mode):
            raise OpenImagesMaterializationError(
                f"output parent component is not a directory: {current}"
            )


def _official_https_url(image: PlannedImage) -> str:
    return f"{OFFICIAL_HTTPS_ORIGIN}/{image.object_key}"


def _download_cvdf_image(
    client: httpx.Client,
    image: PlannedImage,
    *,
    max_image_bytes: int,
    retries: int,
    retry_backoff_seconds: float,
) -> DownloadResult:
    url = _official_https_url(image)
    last_error: Exception | None = None
    for attempt in range(1, retries + 2):
        try:
            with client.stream("GET", url) as response:
                if response.status_code in RETRYABLE_STATUS_CODES:
                    raise _RetryableDownload(f"HTTP {response.status_code}")
                if response.status_code != 200:
                    raise OpenImagesMaterializationError(
                        f"download {image.image_id} returned HTTP {response.status_code}"
                    )
                raw_length = response.headers.get("content-length")
                if raw_length is not None:
                    try:
                        content_length = int(raw_length)
                    except ValueError as error:
                        raise OpenImagesMaterializationError(
                            f"download {image.image_id} has an invalid Content-Length"
                        ) from error
                    if content_length < 1 or content_length > max_image_bytes:
                        raise OpenImagesMaterializationError(
                            f"download {image.image_id} exceeds max_image_bytes"
                        )
                chunks: list[bytes] = []
                consumed = 0
                for chunk in response.iter_bytes():
                    consumed += len(chunk)
                    if consumed > max_image_bytes:
                        raise OpenImagesMaterializationError(
                            f"download {image.image_id} exceeds max_image_bytes"
                        )
                    chunks.append(chunk)
                if consumed == 0:
                    raise OpenImagesMaterializationError(f"download {image.image_id} is empty")
                content_type = response.headers.get("content-type")
                return DownloadResult(
                    encoded=b"".join(chunks),
                    attempts=attempt,
                    content_type=content_type,
                )
        except _RetryableDownload as error:
            last_error = error
        except httpx.TransportError as error:
            last_error = error
        if attempt <= retries:
            if retry_backoff_seconds:
                time.sleep(retry_backoff_seconds * (2 ** (attempt - 1)))
            continue
        break
    raise OpenImagesMaterializationError(
        f"download {image.image_id} failed after {retries + 1} attempts: {last_error}"
    ) from last_error


def _write_new(path: Path, encoded: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as error:
        raise OpenImagesMaterializationError(f"could not write new file {path}: {error}") from error


def _inspect_jpeg(encoded: bytes, location: str, *, max_image_pixels: int) -> tuple[int, int]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(encoded)) as opened:
                if opened.format != "JPEG":
                    raise OpenImagesMaterializationError(f"{location} is not a JPEG image")
                width, height = opened.size
                if width <= 0 or height <= 0 or width * height > max_image_pixels:
                    raise OpenImagesMaterializationError(f"{location} has unsafe dimensions")
                opened.verify()
            with Image.open(io.BytesIO(encoded)) as decoded:
                decoded.load()
    except OpenImagesMaterializationError:
        raise
    except Exception as error:
        raise OpenImagesMaterializationError(f"{location} is not a valid decodable JPEG") from error
    return width, height


def _pixel_annotation(box: PlannedBox, width: int, height: int, index: int) -> dict[str, Any]:
    x1 = box.xmin * width
    x2 = box.xmax * width
    y1 = box.ymin * height
    y2 = box.ymax * height
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise OpenImagesMaterializationError("normalized bbox falls outside decoded image bounds")
    return {
        "annotation_index": index,
        "attributes": {
            "is_occluded": box.is_occluded,
            "is_truncated": box.is_truncated,
        },
        "bbox": [x1, y1, x2 - x1, y2 - y1],
        "bbox_xyxy": [x1, y1, x2, y2],
        "confidence": box.confidence,
        "label": box.class_name,
        "label_id": box.label_id,
        "normalized_bbox_xyxy": [box.xmin, box.ymin, box.xmax, box.ymax],
        "source": box.source,
    }


def _plan_binding(plan: ValidatedPlan, workspace_root: Path) -> dict[str, Any]:
    return {
        "path": _workspace_relative(plan.file.path, workspace_root, "acquisition plan"),
        "plan_semantic_sha256": plan.semantic_sha256,
        "schema": PLAN_SCHEMA,
        "sha256": plan.file.sha256,
        "size_bytes": plan.file.size_bytes,
    }


def _build_draft_manifest(
    plan: ValidatedPlan,
    materialized: Sequence[dict[str, Any]],
    *,
    workspace_root: Path,
) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    annotation_count = 0
    for image, record in zip(plan.images, materialized, strict=True):
        annotations = [
            _pixel_annotation(box, record["width"], record["height"], index)
            for index, box in enumerate(image.boxes, start=1)
        ]
        annotation_count += len(annotations)
        samples.append(
            {
                "annotations": annotations,
                "file_name": image.file_name,
                "height": record["height"],
                "image_id": image.image_id,
                "sample_index": image.selection_rank,
                "scene_id": f"open-images-validation-{image.image_id}",
                "sha256": record["sha256"],
                "size_bytes": record["size_bytes"],
                "source_frame": 0,
                "source_group_id": image.source_group_id,
                "width": record["width"],
            }
        )
    return {
        "annotation_count": annotation_count,
        "cvat": {"created": False, "job_id": None, "task_id": None},
        "frame_pool": "official Open Images V7 validation images selected by bound plan",
        "input_plan": _plan_binding(plan, workspace_root),
        "method": "materialized immutable Open Images validation selection",
        "purpose": "training",
        "sample_size": len(samples),
        "samples": samples,
        "schema": DRAFT_MANIFEST_SCHEMA,
        "status": "draft_requires_cvat_creation_and_full_human_review",
        "taxonomy": list(CANONICAL_CLASSES),
    }


def _build_pack_manifest(
    plan: ValidatedPlan,
    materialized: Sequence[dict[str, Any]],
    draft_file: VerifiedFile,
    *,
    workspace_root: Path,
    max_image_bytes: int,
    max_total_bytes: int,
    max_image_pixels: int,
    timeout_seconds: float,
    connect_timeout_seconds: float,
    retries: int,
) -> dict[str, Any]:
    files = [
        {
            "path": record["path"],
            "sha256": record["sha256"],
            "size_bytes": record["size_bytes"],
        }
        for record in materialized
    ]
    files.append(
        {
            "path": "draft-sample-manifest.json",
            "sha256": draft_file.sha256,
            "size_bytes": draft_file.size_bytes,
        }
    )
    return {
        "counts": {
            "annotations": sum(record["bbox_count"] for record in materialized),
            "images": len(materialized),
            "source_groups": len({record["source_group_id"] for record in materialized}),
            "total_image_bytes": sum(record["size_bytes"] for record in materialized),
        },
        "dataset": {
            "name": "Open Images",
            "project_usage": "training_only",
            "upstream_subset": OFFICIAL_SUBSET,
            "version": "V7",
        },
        "download_policy": {
            "connect_timeout_seconds": connect_timeout_seconds,
            "follow_redirects": False,
            "max_image_bytes": max_image_bytes,
            "max_image_pixels": max_image_pixels,
            "max_total_bytes": max_total_bytes,
            "official_https_origin": OFFICIAL_HTTPS_ORIGIN,
            "retries": retries,
            "timeout_seconds": timeout_seconds,
        },
        "draft_sample_manifest": {
            "path": "draft-sample-manifest.json",
            "schema": DRAFT_MANIFEST_SCHEMA,
            "sha256": draft_file.sha256,
            "size_bytes": draft_file.size_bytes,
        },
        "files": files,
        "gate": {
            "blocking_reasons": [],
            "checks": {
                "all_downloads_derived_from_official_cvdf_validation_keys": True,
                "all_images_are_decodable_jpeg": True,
                "all_image_hashes_and_dimensions_recorded": True,
                "all_normalized_boxes_within_decoded_image_bounds": True,
                "attribution_and_cc_by_2_0_preserved": True,
                "draft_sample_manifest_hash_bound": True,
                "input_plan_stably_read_and_hash_bound": True,
                "source_group_ids_preserved_verbatim": True,
                "temporary_staging_and_atomic_no_replace_publication": True,
            },
            "passed": True,
        },
        "holdout_firewall": {
            "allowed_network_origin": OFFICIAL_HTTPS_ORIGIN,
            "downloads_performed": True,
            "input_scope": "training_only",
            "rejected_scopes": list(FINAL_HOLDOUT_REJECTED_SCOPES),
        },
        "images": list(materialized),
        "input_plan": _plan_binding(plan, workspace_root),
        "schema": OUTPUT_SCHEMA,
        "taxonomy": list(CANONICAL_CLASSES),
    }


def _verify_input_unchanged(expected: VerifiedFile, workspace_root: Path) -> None:
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
        raise OpenImagesMaterializationError("acquisition plan changed during materialization")


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
        raise OpenImagesMaterializationError(
            "atomic no-replace directory publication is unsupported on this platform"
        )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise OpenImagesMaterializationError(f"output directory already exists: {output}")
    raise OSError(error_number, os.strerror(error_number), str(output))


def materialize_open_images_training_pack(
    acquisition_plan: Path,
    output: Path,
    *,
    workspace_root: Path | None = None,
    apply: bool = False,
    max_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
    max_image_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    connect_timeout_seconds: float = DEFAULT_CONNECT_TIMEOUT_SECONDS,
    retries: int = DEFAULT_RETRIES,
    retry_backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, Any]:
    """Validate a plan and, only with ``apply=True``, download and publish its pack."""

    if not isinstance(apply, bool):
        raise OpenImagesMaterializationError("apply must be a boolean")
    _validate_limits(
        max_image_bytes=max_image_bytes,
        max_total_bytes=max_total_bytes,
        max_image_pixels=max_image_pixels,
        timeout_seconds=timeout_seconds,
        connect_timeout_seconds=connect_timeout_seconds,
        retries=retries,
        retry_backoff_seconds=retry_backoff_seconds,
    )
    root = (workspace_root or Path.cwd()).resolve(strict=True)
    plan = _validate_plan(Path(acquisition_plan), workspace_root=root)
    output_root = _absolute_lexical(Path(output), root)
    _workspace_relative(output_root, root, "output")
    _reject_forbidden_path(output_root, root, "output")
    _validate_existing_output_ancestors(output_root, root)
    if os.path.lexists(output_root):
        raise OpenImagesMaterializationError(f"output directory already exists: {output_root}")
    result = {
        "applied": False,
        "downloads_performed": False,
        "input_plan": _plan_binding(plan, root),
        "mode": "dry-run",
        "output": str(output_root),
        "would_download_images": len(plan.images),
    }
    if not apply:
        return result

    _ensure_output_parent(output_root.parent, root)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.staging-", dir=output_root.parent))
    published = False
    try:
        timeout = httpx.Timeout(timeout_seconds, connect=connect_timeout_seconds)
        materialized: list[dict[str, Any]] = []
        total_bytes = 0
        with httpx.Client(
            timeout=timeout,
            follow_redirects=False,
            transport=transport,
            headers={"Accept": "image/jpeg", "User-Agent": "RoadLabelOps/0.1"},
        ) as client:
            for image in plan.images:
                download = _download_cvdf_image(
                    client,
                    image,
                    max_image_bytes=max_image_bytes,
                    retries=retries,
                    retry_backoff_seconds=retry_backoff_seconds,
                )
                total_bytes += len(download.encoded)
                if total_bytes > max_total_bytes:
                    raise OpenImagesMaterializationError("downloads exceed max_total_bytes")
                image_path = staging / "images" / image.file_name
                _write_new(image_path, download.encoded)
                stable_bytes, verified = _read_regular_bytes(
                    image_path,
                    f"downloaded image {image.image_id}",
                    workspace_root=staging,
                    maximum_bytes=max_image_bytes,
                )
                width, height = _inspect_jpeg(
                    stable_bytes,
                    f"downloaded image {image.image_id}",
                    max_image_pixels=max_image_pixels,
                )
                for box in image.boxes:
                    _pixel_annotation(box, width, height, 1)
                materialized.append(
                    {
                        "attribution": image.attribution,
                        "author": image.author,
                        "author_profile_url": image.author_profile_url,
                        "bbox_count": len(image.boxes),
                        "content_type": download.content_type,
                        "download_attempts": download.attempts,
                        "height": height,
                        "image_id": image.image_id,
                        "landing_url": image.landing_url,
                        "license": {"name": image.license_name, "url": image.license_url},
                        "official_object_key": image.object_key,
                        "path": f"images/{image.file_name}",
                        "selection_rank": image.selection_rank,
                        "sha256": verified.sha256,
                        "size_bytes": verified.size_bytes,
                        "source_group_basis": image.source_group_basis,
                        "source_group_id": image.source_group_id,
                        "title": image.title,
                        "width": width,
                    }
                )
        draft = _build_draft_manifest(plan, materialized, workspace_root=root)
        draft_path = staging / "draft-sample-manifest.json"
        _write_new(draft_path, _json_bytes(draft))
        _draft_bytes, draft_file = _read_regular_bytes(
            draft_path,
            "draft sample manifest",
            workspace_root=staging,
            maximum_bytes=MAX_PLAN_BYTES,
        )
        manifest = _build_pack_manifest(
            plan,
            materialized,
            draft_file,
            workspace_root=root,
            max_image_bytes=max_image_bytes,
            max_total_bytes=max_total_bytes,
            max_image_pixels=max_image_pixels,
            timeout_seconds=timeout_seconds,
            connect_timeout_seconds=connect_timeout_seconds,
            retries=retries,
        )
        _write_new(staging / "manifest.json", _json_bytes(manifest))
        _verify_input_unchanged(plan.file, root)
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
        "applied": True,
        "downloads_performed": True,
        "input_plan": _plan_binding(plan, root),
        "manifest": str(output_root / "manifest.json"),
        "mode": "apply",
        "output": str(output_root),
        "published_images": len(plan.images),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--max-image-bytes", type=int, default=DEFAULT_MAX_IMAGE_BYTES)
    parser.add_argument("--max-total-bytes", type=int, default=DEFAULT_MAX_TOTAL_BYTES)
    parser.add_argument("--max-image-pixels", type=int, default=DEFAULT_MAX_IMAGE_PIXELS)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument(
        "--connect-timeout-seconds", type=float, default=DEFAULT_CONNECT_TIMEOUT_SECONDS
    )
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument(
        "--retry-backoff-seconds", type=float, default=DEFAULT_RETRY_BACKOFF_SECONDS
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = materialize_open_images_training_pack(
            args.acquisition_plan,
            args.output,
            apply=args.apply,
            max_image_bytes=args.max_image_bytes,
            max_total_bytes=args.max_total_bytes,
            max_image_pixels=args.max_image_pixels,
            timeout_seconds=args.timeout_seconds,
            connect_timeout_seconds=args.connect_timeout_seconds,
            retries=args.retries,
            retry_backoff_seconds=args.retry_backoff_seconds,
        )
    except OpenImagesMaterializationError as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
