"""Build a deterministic, local-only Open Images V7 acquisition plan.

The command consumes the official boxable class descriptions, bounding-box
annotations, and image metadata CSV files.  It never downloads image bytes.
Instead, it publishes an immutable JSON plan containing only attribution,
official CVDF object identifiers, and verified target-box metadata for the
fixed RoadLabelOps taxonomy.
"""

from __future__ import annotations

import argparse
import csv
import errno
import hashlib
import json
import math
import os
import re
import stat
import tempfile
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO
from urllib.parse import urlsplit, urlunsplit

from roadlabelops.holdout_policy import (
    FINAL_HOLDOUT_REJECTED_SCOPES,
    final_holdout_scope_reason,
)

OUTPUT_SCHEMA = {"name": "roadlabelops.open-images-v7-acquisition-plan", "version": 1}
ROADLABELOPS_CLASSES = (
    "car",
    "bus",
    "truck",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_light",
    "traffic_sign",
)
DISPLAY_NAME_TO_CLASS = {
    "Person": "pedestrian",
    "Bicycle": "bicycle",
    "Bus": "bus",
    "Car": "car",
    "Motorcycle": "motorcycle",
    "Traffic light": "traffic_light",
    "Traffic sign": "traffic_sign",
    "Truck": "truck",
}
REQUIRED_BBOX_COLUMNS = frozenset(
    {
        "ImageID",
        "Source",
        "LabelName",
        "Confidence",
        "XMin",
        "XMax",
        "YMin",
        "YMax",
        "IsOccluded",
        "IsTruncated",
        "IsGroupOf",
        "IsDepiction",
        "IsInside",
    }
)
REQUIRED_METADATA_COLUMNS = frozenset(
    {
        "ImageID",
        "Subset",
        "OriginalLandingURL",
        "License",
        "AuthorProfileURL",
        "Author",
        "Title",
    }
)
MAX_INPUT_BYTES = 32 * 1024 * 1024 * 1024
MAX_CSV_LINE_BYTES = 16 * 1024 * 1024
CVDF_BUCKET = "open-images-dataset"
OPEN_IMAGES_SOURCE_SUBSET = "validation"


class OpenImagesAcquisitionPlanError(ValueError):
    """Raised when a safe, quota-complete acquisition plan cannot be built."""


@dataclass(frozen=True)
class FileIdentity:
    device: int
    inode: int
    size_bytes: int
    modified_ns: int
    changed_ns: int


@dataclass(frozen=True)
class VerifiedInput:
    path: Path
    location: str
    sha256: str
    size_bytes: int
    identity: FileIdentity


@dataclass(frozen=True)
class TargetBox:
    canonical_class: str
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
class ImageMetadata:
    image_id: str
    subset: str
    author: str
    author_profile_url: str | None
    title: str | None
    landing_url: str
    license_url: str
    source_group_id: str
    source_group_basis: str
    source_group_value: str


@dataclass(frozen=True)
class Candidate:
    metadata: ImageMetadata
    boxes: tuple[TargetBox, ...]
    box_counts: Mapping[str, int]


class _DecodedHashedLines(Iterator[str]):
    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream
        self._digest = hashlib.sha256()
        self._bytes_read = 0
        self._line_number = 0

    @property
    def digest(self) -> str:
        return self._digest.hexdigest()

    @property
    def bytes_read(self) -> int:
        return self._bytes_read

    @property
    def line_number(self) -> int:
        return self._line_number

    def __iter__(self) -> _DecodedHashedLines:
        return self

    def __next__(self) -> str:
        raw_line = self._stream.readline(MAX_CSV_LINE_BYTES + 1)
        if not raw_line:
            raise StopIteration
        self._line_number += 1
        if len(raw_line) > MAX_CSV_LINE_BYTES:
            raise OpenImagesAcquisitionPlanError(
                f"CSV line {self._line_number} exceeds the supported size"
            )
        self._digest.update(raw_line)
        self._bytes_read += len(raw_line)
        try:
            decoded = raw_line.decode("utf-8")
        except UnicodeDecodeError as error:
            raise OpenImagesAcquisitionPlanError(
                f"CSV line {self._line_number} is not valid UTF-8"
            ) from error
        if self._line_number == 1:
            decoded = decoded.removeprefix("\ufeff")
        if "\x00" in decoded:
            raise OpenImagesAcquisitionPlanError(
                f"CSV line {self._line_number} contains a NUL byte"
            )
        return decoded


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


def _identity(metadata: os.stat_result) -> FileIdentity:
    return FileIdentity(
        device=metadata.st_dev,
        inode=metadata.st_ino,
        size_bytes=metadata.st_size,
        modified_ns=metadata.st_mtime_ns,
        changed_ns=metadata.st_ctime_ns,
    )


def _integer(value: Any, location: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise OpenImagesAcquisitionPlanError(f"{location} must be an integer >= {minimum}")
    return value


def _forbidden_scope(parts: Sequence[str]) -> str | None:
    return final_holdout_scope_reason(PurePosixPath(*parts))


def _absolute_lexical(path: Path, workspace_root: Path) -> Path:
    candidate = path if path.is_absolute() else workspace_root / path
    return Path(os.path.abspath(os.fspath(candidate)))


def _workspace_relative(path: Path, workspace_root: Path, location: str) -> str:
    try:
        relative = path.relative_to(workspace_root)
    except ValueError as error:
        raise OpenImagesAcquisitionPlanError(
            f"{location} must remain inside the workspace"
        ) from error
    return PurePosixPath(*relative.parts).as_posix()


def _reject_forbidden_path(path: Path, workspace_root: Path, location: str) -> None:
    relative = _workspace_relative(path, workspace_root, location)
    scope = _forbidden_scope(PurePosixPath(relative).parts)
    if scope is not None:
        raise OpenImagesAcquisitionPlanError(f"{location} references forbidden {scope} scope")


def _reject_symlink_chain(path: Path, workspace_root: Path, location: str) -> None:
    try:
        relative = path.relative_to(workspace_root)
    except ValueError as error:
        raise OpenImagesAcquisitionPlanError(
            f"{location} must remain inside the workspace"
        ) from error
    current = workspace_root
    for part in relative.parts:
        current /= part
        try:
            metadata = os.lstat(current)
        except OSError as error:
            raise OpenImagesAcquisitionPlanError(
                f"could not inspect {location}: {error}"
            ) from error
        if stat.S_ISLNK(metadata.st_mode):
            raise OpenImagesAcquisitionPlanError(f"{location} must not contain symlinks")


def _read_csv(
    path: Path,
    location: str,
    workspace_root: Path,
    parser: Any,
) -> tuple[Any, VerifiedInput]:
    candidate = _absolute_lexical(path, workspace_root)
    _workspace_relative(candidate, workspace_root, location)
    _reject_forbidden_path(candidate, workspace_root, location)
    _reject_symlink_chain(candidate, workspace_root, location)
    if candidate.suffix.casefold() != ".csv":
        raise OpenImagesAcquisitionPlanError(f"{location} must be a CSV file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(candidate, flags)
    except OSError as error:
        raise OpenImagesAcquisitionPlanError(f"could not open {location}: {error}") from error
    try:
        before_metadata = os.fstat(descriptor)
        if not stat.S_ISREG(before_metadata.st_mode):
            raise OpenImagesAcquisitionPlanError(f"{location} must be a regular file")
        if before_metadata.st_size > MAX_INPUT_BYTES:
            raise OpenImagesAcquisitionPlanError(f"{location} exceeds the supported size")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            lines = _DecodedHashedLines(stream)
            try:
                result = parser(lines)
                for _remaining in lines:
                    pass
            except csv.Error as error:
                raise OpenImagesAcquisitionPlanError(
                    f"could not parse {location}: {error}"
                ) from error
            after_metadata = os.fstat(descriptor)
        before = _identity(before_metadata)
        after = _identity(after_metadata)
        if before != after or lines.bytes_read != before.size_bytes:
            raise OpenImagesAcquisitionPlanError(f"{location} changed while it was being read")
    except OSError as error:
        raise OpenImagesAcquisitionPlanError(f"could not read {location}: {error}") from error
    finally:
        os.close(descriptor)
    return result, VerifiedInput(
        path=candidate,
        location=location,
        sha256=lines.digest,
        size_bytes=lines.bytes_read,
        identity=before,
    )


def _clean_cell(value: Any, location: str, *, required: bool = True) -> str:
    if not isinstance(value, str):
        raise OpenImagesAcquisitionPlanError(f"{location} must be text")
    cleaned = value.strip()
    if required and not cleaned:
        raise OpenImagesAcquisitionPlanError(f"{location} must not be empty")
    if any(ord(character) < 32 for character in cleaned):
        raise OpenImagesAcquisitionPlanError(f"{location} contains control characters")
    return cleaned


def _image_id(value: Any, location: str) -> str:
    image_id = _clean_cell(value, location)
    if re.fullmatch(r"[0-9a-f]{16}", image_id) is None:
        raise OpenImagesAcquisitionPlanError(
            f"{location} must be a lowercase 16-character Open Images ID"
        )
    return image_id


def _header(reader: csv.DictReader[str], required: frozenset[str], location: str) -> None:
    fields = reader.fieldnames
    if fields is None:
        raise OpenImagesAcquisitionPlanError(f"{location} is missing a header")
    if any(not isinstance(field, str) or not field for field in fields):
        raise OpenImagesAcquisitionPlanError(f"{location} contains an empty header")
    if len(fields) != len(set(fields)):
        raise OpenImagesAcquisitionPlanError(f"{location} contains duplicate columns")
    missing = sorted(required - set(fields))
    if missing:
        raise OpenImagesAcquisitionPlanError(f"{location} is missing columns: {missing!r}")


def _validate_row(row: Mapping[str | None, Any], location: str) -> None:
    if None in row or any(value is None for value in row.values()):
        raise OpenImagesAcquisitionPlanError(f"{location} has the wrong number of columns")


def _parse_class_descriptions(lines: Iterable[str]) -> dict[str, str]:
    reader = csv.reader(lines)
    target_ids: dict[str, str] = {}
    label_ids: set[str] = set()
    display_names: set[str] = set()
    for row_number, row in enumerate(reader, start=1):
        if not row:
            continue
        if row_number == 1 and row == ["LabelName", "DisplayName"]:
            continue
        if len(row) != 2:
            raise OpenImagesAcquisitionPlanError(
                f"class descriptions row {row_number} must contain two columns"
            )
        label_id = _clean_cell(row[0], f"class descriptions row {row_number}.LabelName")
        display_name = _clean_cell(row[1], f"class descriptions row {row_number}.DisplayName")
        if label_id in label_ids:
            raise OpenImagesAcquisitionPlanError(
                f"class descriptions duplicate LabelName {label_id!r}"
            )
        if display_name in display_names:
            raise OpenImagesAcquisitionPlanError(
                f"class descriptions duplicate DisplayName {display_name!r}"
            )
        label_ids.add(label_id)
        display_names.add(display_name)
        canonical = DISPLAY_NAME_TO_CLASS.get(display_name)
        if canonical is not None:
            target_ids[canonical] = label_id
    missing = sorted(set(ROADLABELOPS_CLASSES) - set(target_ids))
    if missing:
        raise OpenImagesAcquisitionPlanError(
            f"class descriptions do not resolve the fixed taxonomy: {missing!r}"
        )
    if len(set(target_ids.values())) != len(ROADLABELOPS_CLASSES):
        raise OpenImagesAcquisitionPlanError(
            "target classes must resolve to unique LabelName values"
        )
    return target_ids


def _finite_number(value: Any, location: str) -> float:
    text = _clean_cell(value, location)
    try:
        result = float(text)
    except ValueError as error:
        raise OpenImagesAcquisitionPlanError(f"{location} must be numeric") from error
    if not math.isfinite(result):
        raise OpenImagesAcquisitionPlanError(f"{location} must be finite")
    return 0.0 if result == 0 else result


def _binary_flag(value: Any, location: str) -> int:
    number = _finite_number(value, location)
    if number not in {0.0, 1.0}:
        raise OpenImagesAcquisitionPlanError(f"{location} must be 0 or 1")
    return int(number)


def _parse_bboxes(
    lines: Iterable[str], target_ids: Mapping[str, str]
) -> tuple[dict[str, tuple[TargetBox, ...]], frozenset[str], int]:
    reader = csv.DictReader(lines)
    _header(reader, REQUIRED_BBOX_COLUMNS, "bbox CSV")
    canonical_by_label = {label_id: canonical for canonical, label_id in target_ids.items()}
    boxes: dict[str, list[TargetBox]] = defaultdict(list)
    special_images: set[str] = set()
    duplicate_keys: set[tuple[Any, ...]] = set()
    target_row_count = 0
    for row_number, row in enumerate(reader, start=2):
        location = f"bbox CSV row {row_number}"
        _validate_row(row, location)
        label_id = _clean_cell(row["LabelName"], f"{location}.LabelName")
        canonical = canonical_by_label.get(label_id)
        if canonical is None:
            continue
        target_row_count += 1
        image_id = _image_id(row["ImageID"], f"{location}.ImageID")
        source = _clean_cell(row["Source"], f"{location}.Source")
        confidence = _finite_number(row["Confidence"], f"{location}.Confidence")
        if confidence != 1.0:
            raise OpenImagesAcquisitionPlanError(
                f"{location}.Confidence must be 1 for a ground-truth target box"
            )
        xmin = _finite_number(row["XMin"], f"{location}.XMin")
        xmax = _finite_number(row["XMax"], f"{location}.XMax")
        ymin = _finite_number(row["YMin"], f"{location}.YMin")
        ymax = _finite_number(row["YMax"], f"{location}.YMax")
        if not (0 <= xmin < xmax <= 1 and 0 <= ymin < ymax <= 1):
            raise OpenImagesAcquisitionPlanError(f"{location} has an invalid normalized box")
        is_occluded = _binary_flag(row["IsOccluded"], f"{location}.IsOccluded")
        is_truncated = _binary_flag(row["IsTruncated"], f"{location}.IsTruncated")
        special = any(
            _binary_flag(row[name], f"{location}.{name}") != 0
            for name in ("IsGroupOf", "IsDepiction", "IsInside")
        )
        duplicate_key = (image_id, label_id, xmin, xmax, ymin, ymax)
        if duplicate_key in duplicate_keys:
            raise OpenImagesAcquisitionPlanError(f"{location} duplicates a target box")
        duplicate_keys.add(duplicate_key)
        if special:
            special_images.add(image_id)
            continue
        boxes[image_id].append(
            TargetBox(
                canonical_class=canonical,
                label_id=label_id,
                source=source,
                confidence=confidence,
                xmin=xmin,
                xmax=xmax,
                ymin=ymin,
                ymax=ymax,
                is_occluded=is_occluded,
                is_truncated=is_truncated,
            )
        )
    for image_id in special_images:
        boxes.pop(image_id, None)
    class_order = {name: index for index, name in enumerate(ROADLABELOPS_CLASSES)}
    return (
        {
            image_id: tuple(
                sorted(
                    image_boxes,
                    key=lambda box: (
                        class_order[box.canonical_class],
                        box.xmin,
                        box.ymin,
                        box.xmax,
                        box.ymax,
                        box.source,
                    ),
                )
            )
            for image_id, image_boxes in boxes.items()
            if image_boxes
        },
        frozenset(special_images),
        target_row_count,
    )


def _http_url(value: str, location: str) -> str:
    try:
        parsed = urlsplit(value)
        _parsed_port = parsed.port
    except ValueError as error:
        raise OpenImagesAcquisitionPlanError(
            f"{location} must be a valid absolute HTTP(S) URL"
        ) from error
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.netloc:
        raise OpenImagesAcquisitionPlanError(f"{location} must be an absolute HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise OpenImagesAcquisitionPlanError(f"{location} must not contain credentials")
    return value


def _is_cc_by_2_0(value: str) -> bool:
    try:
        parsed = urlsplit(value)
        host = (parsed.hostname or "").casefold()
        _parsed_port = parsed.port
    except ValueError:
        return False
    return (
        parsed.scheme.casefold() in {"http", "https"}
        and host in {"creativecommons.org", "www.creativecommons.org"}
        and parsed.path.rstrip("/").casefold() == "/licenses/by/2.0"
        and not parsed.query
        and not parsed.fragment
    )


def _normalize_url_for_group(value: str) -> str:
    parsed = urlsplit(value)
    netloc = (parsed.hostname or "").casefold()
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    path = parsed.path.rstrip("/") or "/"
    return urlunsplit((parsed.scheme.casefold(), netloc, path, parsed.query, ""))


def _normalize_author(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _source_group(author_profile_url: str | None, author: str) -> tuple[str, str, str]:
    if author_profile_url is not None:
        basis = "author_profile_url"
        value = _normalize_url_for_group(author_profile_url)
    else:
        basis = "author"
        value = _normalize_author(author)
    digest = hashlib.sha256(f"{basis}\0{value}".encode()).hexdigest()
    return f"{basis}:sha256:{digest}", basis, value


def _parse_metadata(
    lines: Iterable[str], target_image_ids: frozenset[str]
) -> tuple[dict[str, ImageMetadata], dict[str, int]]:
    reader = csv.DictReader(lines)
    _header(reader, REQUIRED_METADATA_COLUMNS, "image metadata CSV")
    metadata: dict[str, ImageMetadata] = {}
    seen: set[str] = set()
    exclusions = Counter(
        {
            "non_validation_subset": 0,
            "not_explicit_cc_by_2_0": 0,
            "incomplete_attribution": 0,
        }
    )
    for row_number, row in enumerate(reader, start=2):
        location = f"image metadata CSV row {row_number}"
        _validate_row(row, location)
        raw_image_id = row["ImageID"]
        if not isinstance(raw_image_id, str) or raw_image_id.strip() not in target_image_ids:
            continue
        image_id = _image_id(raw_image_id, f"{location}.ImageID")
        if image_id in seen:
            raise OpenImagesAcquisitionPlanError(
                f"image metadata CSV duplicates target ImageID {image_id!r}"
            )
        seen.add(image_id)
        subset = _clean_cell(row["Subset"], f"{location}.Subset")
        if subset.casefold() != OPEN_IMAGES_SOURCE_SUBSET:
            exclusions["non_validation_subset"] += 1
            continue
        license_url = _clean_cell(row["License"], f"{location}.License")
        if not _is_cc_by_2_0(license_url):
            exclusions["not_explicit_cc_by_2_0"] += 1
            continue
        author = _clean_cell(row["Author"], f"{location}.Author", required=False)
        landing_url = _clean_cell(
            row["OriginalLandingURL"], f"{location}.OriginalLandingURL", required=False
        )
        profile = _clean_cell(
            row["AuthorProfileURL"], f"{location}.AuthorProfileURL", required=False
        )
        title = _clean_cell(row["Title"], f"{location}.Title", required=False)
        if not author or not landing_url:
            exclusions["incomplete_attribution"] += 1
            continue
        try:
            landing_url = _http_url(landing_url, f"{location}.OriginalLandingURL")
            profile_value = _http_url(profile, f"{location}.AuthorProfileURL") if profile else None
        except OpenImagesAcquisitionPlanError:
            exclusions["incomplete_attribution"] += 1
            continue
        group_id, group_basis, group_value = _source_group(profile_value, author)
        metadata[image_id] = ImageMetadata(
            image_id=image_id,
            subset=OPEN_IMAGES_SOURCE_SUBSET,
            author=author,
            author_profile_url=profile_value,
            title=title or None,
            landing_url=landing_url,
            license_url=license_url,
            source_group_id=group_id,
            source_group_basis=group_basis,
            source_group_value=group_value,
        )
    exclusions["missing_metadata"] = len(target_image_ids - seen)
    return metadata, dict(exclusions)


def _candidate_box_counts(boxes: Sequence[TargetBox]) -> dict[str, int]:
    counts = Counter(box.canonical_class for box in boxes)
    return {name: counts[name] for name in ROADLABELOPS_CLASSES}


def _validate_parameters(
    *,
    max_images: int,
    min_images_per_class: int,
    min_boxes_per_class: int,
    max_images_per_source_group: int,
    min_source_groups_per_class: int,
) -> None:
    _integer(max_images, "max_images", minimum=1)
    _integer(min_images_per_class, "min_images_per_class", minimum=1)
    _integer(min_boxes_per_class, "min_boxes_per_class", minimum=1)
    _integer(max_images_per_source_group, "max_images_per_source_group", minimum=1)
    _integer(min_source_groups_per_class, "min_source_groups_per_class", minimum=1)
    if max_images_per_source_group > max_images:
        raise OpenImagesAcquisitionPlanError(
            "max_images_per_source_group must not exceed max_images"
        )
    if min_source_groups_per_class > max_images:
        raise OpenImagesAcquisitionPlanError(
            "min_source_groups_per_class must not exceed max_images"
        )


def _eligible_statistics(candidates: Sequence[Candidate]) -> tuple[Counter[str], Counter[str]]:
    images: Counter[str] = Counter()
    boxes: Counter[str] = Counter()
    for candidate in candidates:
        for name in ROADLABELOPS_CLASSES:
            count = candidate.box_counts[name]
            if count:
                images[name] += 1
                boxes[name] += count
    return images, boxes


def _select_candidates(
    candidates: Sequence[Candidate],
    *,
    max_images: int,
    min_images_per_class: int,
    min_boxes_per_class: int,
    max_images_per_source_group: int,
    min_source_groups_per_class: int,
) -> tuple[Candidate, ...]:
    eligible_images, eligible_boxes = _eligible_statistics(candidates)
    groups_by_class: dict[str, set[str]] = {name: set() for name in ROADLABELOPS_CLASSES}
    for candidate in candidates:
        for name, count in candidate.box_counts.items():
            if count:
                groups_by_class[name].add(candidate.metadata.source_group_id)
    insufficient = {
        name: {
            "eligible_images": eligible_images[name],
            "eligible_boxes": eligible_boxes[name],
            "eligible_source_groups": len(groups_by_class[name]),
        }
        for name in ROADLABELOPS_CLASSES
        if eligible_images[name] < min_images_per_class
        or eligible_boxes[name] < min_boxes_per_class
        or len(groups_by_class[name]) < min_source_groups_per_class
    }
    if insufficient:
        raise OpenImagesAcquisitionPlanError(
            f"eligible Open Images candidates cannot meet per-class quotas: {insufficient!r}"
        )

    selected: list[Candidate] = []
    remaining = {candidate.metadata.image_id: candidate for candidate in candidates}
    selected_images: Counter[str] = Counter()
    selected_boxes: Counter[str] = Counter()
    selected_groups: dict[str, set[str]] = {name: set() for name in ROADLABELOPS_CLASSES}
    group_counts: Counter[str] = Counter()

    while len(selected) < max_images:
        viable = [
            candidate
            for candidate in remaining.values()
            if group_counts[candidate.metadata.source_group_id] < max_images_per_source_group
        ]
        if not viable:
            break

        def ranking(candidate: Candidate) -> tuple[Any, ...]:
            group_id = candidate.metadata.source_group_id
            group_gain = Fraction()
            image_gain = Fraction()
            box_gain = Fraction()
            scarcity_gain = Fraction()
            for name in ROADLABELOPS_CLASSES:
                count = candidate.box_counts[name]
                if not count:
                    continue
                scarcity_gain += Fraction(1, eligible_images[name]) + Fraction(
                    min(count, min_boxes_per_class), eligible_boxes[name]
                )
                if (
                    len(selected_groups[name]) < min_source_groups_per_class
                    and group_id not in selected_groups[name]
                ):
                    group_gain += Fraction(1, len(groups_by_class[name]))
                if selected_images[name] < min_images_per_class:
                    image_gain += Fraction(1, eligible_images[name])
                box_deficit = max(min_boxes_per_class - selected_boxes[name], 0)
                if box_deficit:
                    box_gain += Fraction(min(count, box_deficit), eligible_boxes[name])
            return (
                -group_gain,
                -image_gain,
                -box_gain,
                group_counts[group_id],
                -scarcity_gain,
                candidate.metadata.image_id,
            )

        chosen = min(viable, key=ranking)
        selected.append(chosen)
        remaining.pop(chosen.metadata.image_id)
        group_id = chosen.metadata.source_group_id
        group_counts[group_id] += 1
        for name, count in chosen.box_counts.items():
            if count:
                selected_images[name] += 1
                selected_boxes[name] += count
                selected_groups[name].add(group_id)

    deficits = {
        name: {
            "selected_images": selected_images[name],
            "selected_boxes": selected_boxes[name],
            "selected_source_groups": len(selected_groups[name]),
        }
        for name in ROADLABELOPS_CLASSES
        if selected_images[name] < min_images_per_class
        or selected_boxes[name] < min_boxes_per_class
        or len(selected_groups[name]) < min_source_groups_per_class
    }
    if deficits:
        raise OpenImagesAcquisitionPlanError(
            "deterministic selection cannot meet quotas within max-images/source-group caps: "
            f"{deficits!r}"
        )
    return tuple(selected)


def _attribution(metadata: ImageMetadata) -> str:
    prefix = f'"{metadata.title}" by {metadata.author}' if metadata.title else metadata.author
    return (
        f"{prefix}; source: {metadata.landing_url}; "
        f"licensed under CC BY 2.0 ({metadata.license_url})"
    )


def _box_payload(box: TargetBox) -> dict[str, Any]:
    return {
        "class_name": box.canonical_class,
        "confidence": box.confidence,
        "is_depiction": 0,
        "is_group_of": 0,
        "is_inside": 0,
        "is_occluded": box.is_occluded,
        "is_truncated": box.is_truncated,
        "label_id": box.label_id,
        "source": box.source,
        "xmax": box.xmax,
        "xmin": box.xmin,
        "ymax": box.ymax,
        "ymin": box.ymin,
    }


def _build_plan(
    *,
    class_descriptions: VerifiedInput,
    bbox_annotations: VerifiedInput,
    image_metadata: VerifiedInput,
    target_ids: Mapping[str, str],
    candidates: Sequence[Candidate],
    selected: Sequence[Candidate],
    special_image_count: int,
    target_row_count: int,
    metadata_exclusions: Mapping[str, int],
    workspace_root: Path,
    max_images: int,
    min_images_per_class: int,
    min_boxes_per_class: int,
    max_images_per_source_group: int,
    min_source_groups_per_class: int,
) -> dict[str, Any]:
    eligible_images, eligible_boxes = _eligible_statistics(candidates)
    selected_images, selected_boxes = _eligible_statistics(selected)
    selected_groups_by_class: dict[str, set[str]] = {name: set() for name in ROADLABELOPS_CLASSES}
    eligible_groups_by_class: dict[str, set[str]] = {name: set() for name in ROADLABELOPS_CLASSES}
    for candidate in candidates:
        for name, count in candidate.box_counts.items():
            if count:
                eligible_groups_by_class[name].add(candidate.metadata.source_group_id)
    for candidate in selected:
        for name, count in candidate.box_counts.items():
            if count:
                selected_groups_by_class[name].add(candidate.metadata.source_group_id)

    image_payloads: list[dict[str, Any]] = []
    for rank, candidate in enumerate(selected, start=1):
        metadata = candidate.metadata
        image_payloads.append(
            {
                "attribution": _attribution(metadata),
                "author": metadata.author,
                "author_profile_url": metadata.author_profile_url,
                "box_counts": dict(candidate.box_counts),
                "boxes": [_box_payload(box) for box in candidate.boxes],
                "cvdf_download": {
                    "bucket": CVDF_BUCKET,
                    "file_name": f"{metadata.image_id}.jpg",
                    "object_key": f"{OPEN_IMAGES_SOURCE_SUBSET}/{metadata.image_id}.jpg",
                    "s3_uri": (
                        f"s3://{CVDF_BUCKET}/{OPEN_IMAGES_SOURCE_SUBSET}/{metadata.image_id}.jpg"
                    ),
                },
                "image_id": metadata.image_id,
                "landing_url": metadata.landing_url,
                "license": {"name": "CC BY 2.0", "url": metadata.license_url},
                "selection_rank": rank,
                "source_group_basis": metadata.source_group_basis,
                "source_group_id": metadata.source_group_id,
                "subset": metadata.subset,
                "title": metadata.title,
            }
        )

    selected_by_group: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in selected:
        selected_by_group[candidate.metadata.source_group_id].append(candidate)
    source_groups: list[dict[str, Any]] = []
    for group_id in sorted(selected_by_group):
        group_candidates = sorted(
            selected_by_group[group_id], key=lambda candidate: candidate.metadata.image_id
        )
        per_class = {
            name: {
                "box_count": sum(item.box_counts[name] for item in group_candidates),
                "image_count": sum(item.box_counts[name] > 0 for item in group_candidates),
            }
            for name in ROADLABELOPS_CLASSES
        }
        group_payload: dict[str, Any] = {
            "image_count": len(group_candidates),
            "image_ids": [item.metadata.image_id for item in group_candidates],
            "per_class": per_class,
            "source_group_basis": group_candidates[0].metadata.source_group_basis,
            "source_group_id": group_id,
            "source_group_value": group_candidates[0].metadata.source_group_value,
        }
        group_payload["manifest_semantic_sha256"] = _semantic_sha256(group_payload)
        source_groups.append(group_payload)

    inputs = {}
    for name, item in (
        ("boxable_class_descriptions", class_descriptions),
        ("bounding_boxes", bbox_annotations),
        ("image_metadata", image_metadata),
    ):
        inputs[name] = {
            "path": _workspace_relative(item.path, workspace_root, item.location),
            "sha256": item.sha256,
            "size_bytes": item.size_bytes,
        }
    class_statistics = {
        name: {
            "eligible_box_count": eligible_boxes[name],
            "eligible_image_count": eligible_images[name],
            "eligible_source_group_count": len(eligible_groups_by_class[name]),
            "minimum_boxes_required": min_boxes_per_class,
            "minimum_images_required": min_images_per_class,
            "minimum_source_groups_required": min_source_groups_per_class,
            "quota_met": (
                selected_boxes[name] >= min_boxes_per_class
                and selected_images[name] >= min_images_per_class
                and len(selected_groups_by_class[name]) >= min_source_groups_per_class
            ),
            "selected_box_count": selected_boxes[name],
            "selected_image_count": selected_images[name],
            "selected_source_group_count": len(selected_groups_by_class[name]),
        }
        for name in ROADLABELOPS_CLASSES
    }
    plan: dict[str, Any] = {
        "counts": {
            "eligible_images": len(candidates),
            "selected_boxes": sum(len(candidate.boxes) for candidate in selected),
            "selected_images": len(selected),
            "selected_source_groups": len(source_groups),
            "special_target_flag_images_excluded": special_image_count,
            "target_bbox_rows_read": target_row_count,
        },
        "dataset": {
            "name": "Open Images",
            "project_usage": "training_only",
            "upstream_subset": OPEN_IMAGES_SOURCE_SUBSET,
            "version": "V7",
        },
        "gate": {
            "blocking_reasons": [],
            "checks": {
                "all_selected_images_are_official_validation_subset": True,
                "all_selected_images_have_complete_attribution": True,
                "all_selected_images_have_explicit_cc_by_2_0": True,
                "all_target_boxes_are_real_instance_boxes": True,
                "fixed_eight_class_mapping": True,
                "forbidden_final_holdout_scopes_not_read": True,
                "immutable_no_replace_publication": True,
                "local_plan_only_no_network_or_download": True,
                "max_images_respected": len(selected) <= max_images,
                "per_class_box_minimums_met": all(
                    value["selected_box_count"] >= min_boxes_per_class
                    for value in class_statistics.values()
                ),
                "per_class_image_minimums_met": all(
                    value["selected_image_count"] >= min_images_per_class
                    for value in class_statistics.values()
                ),
                "per_class_source_group_minimums_met": all(
                    value["selected_source_group_count"] >= min_source_groups_per_class
                    for value in class_statistics.values()
                ),
                "source_group_image_cap_met": all(
                    value["image_count"] <= max_images_per_source_group for value in source_groups
                ),
                "special_target_row_excludes_entire_image": True,
            },
            "passed": True,
        },
        "holdout_firewall": {
            "allowed_inputs": ["local_official_open_images_v7_csv"],
            "downloads_performed": False,
            "network_accessed": False,
            "rejected_scopes": list(FINAL_HOLDOUT_REJECTED_SCOPES),
        },
        "images": image_payloads,
        "inputs": inputs,
        "metadata_exclusions": dict(sorted(metadata_exclusions.items())),
        "parameters": {
            "max_images": max_images,
            "max_images_per_source_group": max_images_per_source_group,
            "min_boxes_per_class": min_boxes_per_class,
            "min_images_per_class": min_images_per_class,
            "min_source_groups_per_class": min_source_groups_per_class,
        },
        "schema": OUTPUT_SCHEMA,
        "selection": {
            "class_statistics": class_statistics,
            "policy": (
                "quota-first deterministic scarcity weighting; lower selected source-group "
                "occupancy; lexical ImageID tie-break"
            ),
            "source_groups": source_groups,
        },
        "taxonomy": [
            {
                "display_name": next(
                    display
                    for display, mapped in DISPLAY_NAME_TO_CLASS.items()
                    if mapped == canonical
                ),
                "label_id": target_ids[canonical],
                "roadlabelops_class": canonical,
            }
            for canonical in ROADLABELOPS_CLASSES
        ],
    }
    plan["plan_semantic_sha256"] = _semantic_sha256(plan)
    return plan


def _verify_inputs_unchanged(inputs: Sequence[VerifiedInput], workspace_root: Path) -> None:
    for item in inputs:
        _reject_forbidden_path(item.path, workspace_root, item.location)
        _reject_symlink_chain(item.path, workspace_root, item.location)
        try:
            metadata = os.lstat(item.path)
        except OSError as error:
            raise OpenImagesAcquisitionPlanError(
                f"could not re-inspect {item.location}: {error}"
            ) from error
        if not stat.S_ISREG(metadata.st_mode) or _identity(metadata) != item.identity:
            raise OpenImagesAcquisitionPlanError(
                f"{item.location} changed during plan construction"
            )


def _ensure_output_parent(path: Path, workspace_root: Path) -> None:
    try:
        relative = path.relative_to(workspace_root)
    except ValueError as error:
        raise OpenImagesAcquisitionPlanError(
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
                raise OpenImagesAcquisitionPlanError(
                    f"could not create output parent {current}: {error}"
                ) from error
        except OSError as error:
            raise OpenImagesAcquisitionPlanError(
                f"could not inspect output parent {current}: {error}"
            ) from error
        if stat.S_ISLNK(metadata.st_mode):
            raise OpenImagesAcquisitionPlanError("output parent must not contain symlinks")
        if not stat.S_ISDIR(metadata.st_mode):
            raise OpenImagesAcquisitionPlanError(
                f"output parent component is not a directory: {current}"
            )


def _publish_json_no_replace(path: Path, encoded: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except OSError as error:
            if error.errno == errno.EEXIST:
                raise OpenImagesAcquisitionPlanError(f"output already exists: {path}") from error
            raise
        parent_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def build_open_images_acquisition_plan(
    class_descriptions_csv: Path,
    bbox_csv: Path,
    image_metadata_csv: Path,
    output: Path,
    *,
    max_images: int,
    min_images_per_class: int,
    min_boxes_per_class: int,
    max_images_per_source_group: int,
    min_source_groups_per_class: int = 2,
    workspace_root: Path | None = None,
) -> dict[str, Any]:
    """Validate local Open Images CSVs and atomically publish an acquisition plan."""

    _validate_parameters(
        max_images=max_images,
        min_images_per_class=min_images_per_class,
        min_boxes_per_class=min_boxes_per_class,
        max_images_per_source_group=max_images_per_source_group,
        min_source_groups_per_class=min_source_groups_per_class,
    )
    root = (workspace_root or Path.cwd()).resolve(strict=True)
    target_ids, class_input = _read_csv(
        Path(class_descriptions_csv),
        "class descriptions CSV",
        root,
        _parse_class_descriptions,
    )
    (boxes_by_image, special_images, target_row_count), bbox_input = _read_csv(
        Path(bbox_csv),
        "bbox CSV",
        root,
        lambda lines: _parse_bboxes(lines, target_ids),
    )
    metadata_result, metadata_input = _read_csv(
        Path(image_metadata_csv),
        "image metadata CSV",
        root,
        lambda lines: _parse_metadata(lines, frozenset(boxes_by_image)),
    )
    metadata_by_image, metadata_exclusions = metadata_result
    candidates = tuple(
        Candidate(
            metadata=metadata_by_image[image_id],
            boxes=image_boxes,
            box_counts=_candidate_box_counts(image_boxes),
        )
        for image_id, image_boxes in sorted(boxes_by_image.items())
        if image_id in metadata_by_image
    )
    if not candidates:
        raise OpenImagesAcquisitionPlanError("no eligible Open Images candidates remain")
    selected = _select_candidates(
        candidates,
        max_images=max_images,
        min_images_per_class=min_images_per_class,
        min_boxes_per_class=min_boxes_per_class,
        max_images_per_source_group=max_images_per_source_group,
        min_source_groups_per_class=min_source_groups_per_class,
    )
    output_path = _absolute_lexical(Path(output), root)
    _workspace_relative(output_path, root, "output")
    _reject_forbidden_path(output_path, root, "output")
    if output_path.suffix.casefold() != ".json":
        raise OpenImagesAcquisitionPlanError("output must have a .json suffix")
    if os.path.lexists(output_path):
        raise OpenImagesAcquisitionPlanError(f"output already exists: {output_path}")
    _ensure_output_parent(output_path.parent, root)
    plan = _build_plan(
        class_descriptions=class_input,
        bbox_annotations=bbox_input,
        image_metadata=metadata_input,
        target_ids=target_ids,
        candidates=candidates,
        selected=selected,
        special_image_count=len(special_images),
        target_row_count=target_row_count,
        metadata_exclusions=metadata_exclusions,
        workspace_root=root,
        max_images=max_images,
        min_images_per_class=min_images_per_class,
        min_boxes_per_class=min_boxes_per_class,
        max_images_per_source_group=max_images_per_source_group,
        min_source_groups_per_class=min_source_groups_per_class,
    )
    inputs = (class_input, bbox_input, metadata_input)
    _verify_inputs_unchanged(inputs, root)
    _publish_json_no_replace(output_path, _json_bytes(plan))
    return {
        "counts": plan["counts"],
        "gate": plan["gate"],
        "output": str(output_path),
        "plan_semantic_sha256": plan["plan_semantic_sha256"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--class-descriptions-csv",
        "--class-descriptions",
        dest="class_descriptions",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--bbox-csv",
        "--bbox-annotations",
        dest="bbox_annotations",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--image-metadata-csv",
        "--image-metadata",
        dest="image_metadata",
        type=Path,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-images", type=int, required=True)
    parser.add_argument("--min-images-per-class", type=int, required=True)
    parser.add_argument("--min-boxes-per-class", type=int, required=True)
    parser.add_argument("--max-images-per-source-group", type=int, required=True)
    parser.add_argument("--min-source-groups-per-class", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = build_open_images_acquisition_plan(
            args.class_descriptions,
            args.bbox_annotations,
            args.image_metadata,
            args.output,
            max_images=args.max_images,
            min_images_per_class=args.min_images_per_class,
            min_boxes_per_class=args.min_boxes_per_class,
            max_images_per_source_group=args.max_images_per_source_group,
            min_source_groups_per_class=args.min_source_groups_per_class,
        )
    except OpenImagesAcquisitionPlanError as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
