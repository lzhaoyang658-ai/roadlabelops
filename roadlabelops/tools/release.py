"""Immutable, self-verifying COCO release packages."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import struct
import subprocess
import uuid
from collections import Counter
from collections.abc import Iterable
from contextlib import contextmanager
from functools import lru_cache
from itertools import pairwise
from pathlib import Path
from typing import Any

from ..models import Scene, Session, ToolResult, utc_now
from ..storage import LocalStore
from .quality import calculate_quality

# This ordering is the RoadLabelOps public taxonomy contract. It must not be
# inferred from a particular export, otherwise class ids change between releases.
ROAD_LABEL_TAXONOMY: tuple[str, ...] = (
    "car",
    "bus",
    "truck",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_light",
    "traffic_sign",
)
CATEGORY_IDS = {label: index for index, label in enumerate(ROAD_LABEL_TAXONOMY, start=1)}
ROAD_ATTRIBUTE_VALUES: dict[str, tuple[str, ...]] = {
    "occlusion": ("none", "partial", "heavy"),
    "motion": ("moving", "stopped", "parked", "unknown"),
    "direction": ("same", "opposite", "crossing", "unknown"),
}
ROAD_SCENE_TAG_VALUES: dict[str, tuple[str, ...]] = {
    "lighting": ("day", "night"),
    "weather": ("clear", "rain", "fog"),
    "road_type": ("urban", "highway", "intersection"),
    "traffic_density": ("low", "medium", "high"),
}
MANIFEST_FILENAME = "manifest.json"
RECEIPT_FILENAME = "receipt.json"
ANNOTATIONS_FILENAME = "annotations.coco.json"
QUALITY_FILENAME = "quality.json"
PREDICTIONS_FILENAME = "predictions.json"
YOLO_METADATA_FILENAME = "dataset.yaml"
YOLO_LABELS_DIRECTORY = "labels"
MANIFEST_SCHEMA_VERSION = "2.0.0"
RECEIPT_SCHEMA_VERSION = "2.0.0"
QUALITY_REPORT_FIELDS = frozenset(
    {
        "prediction_count",
        "final_count",
        "retained_count",
        "added_count",
        "removed_count",
        "retention_rate",
        "human_addition_rate",
        "precision",
        "recall",
        "f1_score",
        "evaluated_frame_count",
        "clean_frame_count",
        "clean_frame_rate",
        "first_pass_acceptance_rate",
        "first_pass_acceptance_reason",
        "class_distribution",
        "per_class",
    }
)
QUALITY_CLASS_FIELDS = frozenset(
    {
        "true_positive_count",
        "false_positive_count",
        "false_negative_count",
        "precision",
        "recall",
        "f1_score",
    }
)
_SAFE_COMPONENT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SEMVER_PATTERN = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)

# libjpeg, and therefore OpenCV/Ultralytics, rejects dimensions above 65,500.
_JPEG_MAX_DIMENSION = 65_500


def _jpeg_segment(marker: int, payload: bytes) -> bytes:
    return bytes((0xFF, marker)) + struct.pack(">H", len(payload) + 2) + payload


@lru_cache(maxsize=8)
def _demo_jpeg(width: int, height: int) -> bytes:
    """Create a dependency-free, deterministic, baseline grayscale JPEG.

    Every 8x8 block has a zero DC coefficient and an immediate end-of-block
    marker, producing a uniform mid-gray image. Keeping this encoder here
    avoids making Pillow or FFmpeg a requirement for the self-contained demo,
    while still emitting the exact geometry declared in COCO.
    """

    if not (10 <= width <= _JPEG_MAX_DIMENSION and 10 <= height <= _JPEG_MAX_DIMENSION):
        raise ValueError("demo JPEG dimensions must be between 10 and 65500 pixels")

    app0 = b"JFIF\0" + bytes((1, 1, 0)) + struct.pack(">HH", 1, 1) + bytes((0, 0))
    quantisation_table = bytes((0,)) + bytes((1,)) * 64
    start_of_frame = (
        bytes((8,))
        + struct.pack(">HH", height, width)
        + bytes((1, 1, 0x11, 0))
    )

    # One one-bit code in each table: DC category zero and AC end-of-block.
    code_counts = bytes((1,)) + bytes(15)
    huffman_tables = (
        bytes((0,))
        + code_counts
        + bytes((0,))
        + bytes((0x10,))
        + code_counts
        + bytes((0,))
    )
    start_of_scan = bytes((1, 1, 0, 0, 63, 0))

    block_count = ((width + 7) // 8) * ((height + 7) // 8)
    complete_bytes, remaining_bits = divmod(block_count * 2, 8)
    scan = bytearray(complete_bytes)
    if remaining_bits:
        # JPEG entropy scans pad the final byte with one bits.
        scan.append((1 << (8 - remaining_bits)) - 1)

    return (
        b"\xff\xd8"
        + _jpeg_segment(0xE0, app0)
        + _jpeg_segment(0xDB, quantisation_table)
        + _jpeg_segment(0xC0, start_of_frame)
        + _jpeg_segment(0xC4, huffman_tables)
        + _jpeg_segment(0xDA, start_of_scan)
        + bytes(scan)
        + b"\xff\xd9"
    )


# Retained for source compatibility with tests and integrations that used the
# previous private constant. Unlike its 1x1 predecessor, it matches Demo mode.
_DEMO_JPEG = _demo_jpeg(1920, 1080)


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jpeg_dimensions(path: Path) -> tuple[int, int] | None:
    """Validate a complete JPEG stream and return its declared dimensions."""

    start_of_frame_markers = {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        return None

    position = 2
    pending_marker: int | None = None
    dimensions: tuple[int, int] | None = None
    component_ids: set[int] = set()
    frame_quantisation_tables: set[int] = set()
    quantisation_tables: set[int] = set()
    huffman_tables: set[tuple[int, int]] = set()
    seen_scan = False

    while pending_marker is not None or position < len(data):
        if pending_marker is None:
            if data[position] != 0xFF:
                return None
            while position < len(data) and data[position] == 0xFF:
                position += 1
            if position >= len(data):
                return None
            marker = data[position]
            position += 1
        else:
            marker = pending_marker
            pending_marker = None

        if marker == 0xD9:
            if not seen_scan or dimensions is None or position != len(data):
                return None
            return dimensions
        if marker in {0x00, 0xD8, 0x01, *range(0xD0, 0xD8)}:
            return None
        if marker < 0xC0 or position + 2 > len(data):
            return None

        segment_length = struct.unpack(">H", data[position : position + 2])[0]
        position += 2
        payload_length = segment_length - 2
        if segment_length < 2 or position + payload_length > len(data):
            return None
        payload = data[position : position + payload_length]
        position += payload_length

        if marker == 0xDB:
            cursor = 0
            found_table = False
            while cursor < len(payload):
                specification = payload[cursor]
                cursor += 1
                precision = specification >> 4
                table_id = specification & 0x0F
                if precision not in {0, 1} or table_id > 3:
                    return None
                table_bytes = 64 * (precision + 1)
                table = payload[cursor : cursor + table_bytes]
                if len(table) != table_bytes:
                    return None
                if precision == 0:
                    if 0 in table:
                        return None
                elif any(table[index : index + 2] == b"\0\0" for index in range(0, 128, 2)):
                    return None
                quantisation_tables.add(table_id)
                found_table = True
                cursor += table_bytes
            if not found_table:
                return None
            continue

        if marker == 0xC4:
            cursor = 0
            found_table = False
            while cursor < len(payload):
                if cursor + 17 > len(payload):
                    return None
                specification = payload[cursor]
                code_counts = payload[cursor + 1 : cursor + 17]
                cursor += 17
                table_class = specification >> 4
                table_id = specification & 0x0F
                if table_class not in {0, 1} or table_id > 3:
                    return None
                symbol_count = sum(code_counts)
                if symbol_count == 0 or cursor + symbol_count > len(payload):
                    return None
                available_codes = 1
                for count in code_counts:
                    available_codes = available_codes * 2 - count
                    if available_codes < 0:
                        return None
                cursor += symbol_count
                huffman_tables.add((table_class, table_id))
                found_table = True
            if not found_table:
                return None
            continue

        if marker in start_of_frame_markers:
            if dimensions is not None or len(payload) < 6:
                return None
            precision = payload[0]
            height, width = struct.unpack(">HH", payload[1:5])
            component_count = payload[5]
            if (
                precision != 8
                or not (10 <= width <= _JPEG_MAX_DIMENSION)
                or not (10 <= height <= _JPEG_MAX_DIMENSION)
                or not (1 <= component_count <= 4)
                or len(payload) != 6 + component_count * 3
            ):
                return None
            for offset in range(6, len(payload), 3):
                component_id, sampling, table_id = payload[offset : offset + 3]
                horizontal_sampling = sampling >> 4
                vertical_sampling = sampling & 0x0F
                if (
                    component_id in component_ids
                    or not (1 <= horizontal_sampling <= 4)
                    or not (1 <= vertical_sampling <= 4)
                    or table_id > 3
                ):
                    return None
                component_ids.add(component_id)
                frame_quantisation_tables.add(table_id)
            dimensions = (width, height)
            continue

        if marker != 0xDA:
            continue

        if dimensions is None or not huffman_tables or not frame_quantisation_tables.issubset(
            quantisation_tables
        ):
            return None
        if len(payload) < 4:
            return None
        scan_component_count = payload[0]
        if (
            not (1 <= scan_component_count <= len(component_ids))
            or len(payload) != 4 + scan_component_count * 2
        ):
            return None
        scan_component_ids: set[int] = set()
        scan_table_selectors: list[tuple[int, int]] = []
        for offset in range(1, 1 + scan_component_count * 2, 2):
            component_id = payload[offset]
            selectors = payload[offset + 1]
            if (
                component_id not in component_ids
                or component_id in scan_component_ids
                or selectors >> 4 > 3
                or selectors & 0x0F > 3
            ):
                return None
            scan_component_ids.add(component_id)
            scan_table_selectors.append((selectors >> 4, selectors & 0x0F))
        spectral_start = payload[-3]
        spectral_end = payload[-2]
        approximation = payload[-1]
        if (
            spectral_start > spectral_end
            or spectral_end > 63
            or approximation >> 4 > 13
            or approximation & 0x0F > 13
        ):
            return None
        uses_dc_table = spectral_start == 0
        uses_ac_table = spectral_end > 0
        if any(
            (uses_dc_table and (0, dc_table) not in huffman_tables)
            or (uses_ac_table and (1, ac_table) not in huffman_tables)
            for dc_table, ac_table in scan_table_selectors
        ):
            return None

        seen_scan = True
        scan_has_entropy = False
        while position < len(data):
            if data[position] != 0xFF:
                scan_has_entropy = True
                position += 1
                continue
            position += 1
            while position < len(data) and data[position] == 0xFF:
                position += 1
            if position >= len(data):
                return None
            escaped_or_marker = data[position]
            position += 1
            if escaped_or_marker == 0x00:
                scan_has_entropy = True
                continue
            if 0xD0 <= escaped_or_marker <= 0xD7:
                if not scan_has_entropy:
                    return None
                continue
            if not scan_has_entropy:
                return None
            pending_marker = escaped_or_marker
            break
        if pending_marker is None:
            return None
    return None


def _ffmpeg_decodes_release_images(images_directory: Path, image_count: int) -> bool:
    """Require FFmpeg to fully decode every real source-frame JPEG."""

    timeout_seconds = max(90, min(600, image_count * 2))
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-xerror",
        "-err_detect",
        "explode",
        "-pattern_type",
        "glob",
        "-i",
        str(images_directory / "*.jpg"),
        "-map",
        "0:v:0",
        "-f",
        "null",
        "-",
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def _canonical_digest(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _failure(code: str, message: str, *, retryable: bool = False) -> ToolResult:
    return ToolResult.failure(code, message, retryable=retryable)


def _release_id(session: Session, version: str) -> str:
    return f"{session.session_id}-v{version}"


def _safe_release_name(value: str) -> bool:
    return _safe_path_component(value)


def _valid_version(value: str) -> bool:
    return bool(_SEMVER_PATTERN.fullmatch(value))


def _safe_path_component(value: str) -> bool:
    """Return whether *value* is one ordinary, non-special path component."""

    return (
        bool(_SAFE_COMPONENT_PATTERN.fullmatch(value))
        and value not in {".", ".."}
        and "/" not in value
        and "\\" not in value
    )


def _frame_limit(scene: Scene, fps: float) -> int:
    return max(1, math.ceil((scene.end_seconds - scene.start_seconds) * fps))


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _nonnegative_integer(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _positive_finite_manifest_value(value: Any) -> bool:
    number = _as_number(value)
    return number is not None and number > 0


def _quality_ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _quality_f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    return round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0


def _ratio_matches(value: Any, expected: float | None) -> bool:
    if expected is None:
        return value is None
    numeric = _as_number(value)
    return numeric is not None and 0 <= numeric <= 1 and math.isclose(
        numeric, expected, abs_tol=0.00005
    )


def _quality_validation_issues(
    report: Any,
    *,
    expected_final_count: int,
    expected_class_distribution: dict[str, int],
) -> list[str]:
    """Return semantic quality-report violations for build and verification gates."""

    if not isinstance(report, dict):
        return ["Quality report must be a JSON object"]
    issues: list[str] = []
    if set(report) != QUALITY_REPORT_FIELDS:
        issues.append("Quality report fields do not match the canonical schema")

    count_fields = (
        "prediction_count",
        "final_count",
        "retained_count",
        "added_count",
        "removed_count",
        "evaluated_frame_count",
        "clean_frame_count",
    )
    if any(not _nonnegative_integer(report.get(key)) for key in count_fields):
        issues.append("Quality report counts must be non-negative integers")
        return issues
    prediction_count = int(report["prediction_count"])
    final_count = int(report["final_count"])
    retained_count = int(report["retained_count"])
    added_count = int(report["added_count"])
    removed_count = int(report["removed_count"])
    evaluated_frame_count = int(report["evaluated_frame_count"])
    clean_frame_count = int(report["clean_frame_count"])
    matched_count = final_count - added_count
    if final_count != expected_final_count:
        issues.append("Quality final count does not match the COCO annotation count")
    if not (
        0 <= matched_count <= min(prediction_count, final_count)
        and 0 <= retained_count <= matched_count
        and removed_count == prediction_count - retained_count
        and clean_frame_count <= evaluated_frame_count
    ):
        issues.append("Quality count relationships are inconsistent")

    ratio_expectations = {
        "retention_rate": _quality_ratio(retained_count, prediction_count),
        "human_addition_rate": _quality_ratio(added_count, final_count),
        "precision": _quality_ratio(matched_count, prediction_count),
        "recall": _quality_ratio(matched_count, final_count),
        "clean_frame_rate": _quality_ratio(clean_frame_count, evaluated_frame_count),
    }
    for key, expected in ratio_expectations.items():
        if not _ratio_matches(report.get(key), expected):
            issues.append(f"Quality ratio {key} is inconsistent with its counts")
    precision = ratio_expectations["precision"]
    recall = ratio_expectations["recall"]
    if not _ratio_matches(report.get("f1_score"), _quality_f1(precision, recall)):
        issues.append("Quality F1 score is inconsistent with precision and recall")

    first_pass_rate = report.get("first_pass_acceptance_rate")
    if first_pass_rate is not None:
        first_pass_numeric = _as_number(first_pass_rate)
        if first_pass_numeric is None or not 0 <= first_pass_numeric <= 1:
            issues.append("First-pass acceptance rate must be null or within [0, 1]")
    first_pass_reason = report.get("first_pass_acceptance_reason")
    if first_pass_reason is not None and (
        not isinstance(first_pass_reason, str) or not first_pass_reason.strip()
    ):
        issues.append("First-pass acceptance reason must be null or a non-empty string")
    if first_pass_rate is not None and first_pass_reason is not None:
        issues.append("A measured first-pass acceptance rate cannot carry an unavailable reason")

    distribution = report.get("class_distribution")
    if (
        not isinstance(distribution, dict)
        or any(
            label not in CATEGORY_IDS or not _nonnegative_integer(count) or count == 0
            for label, count in distribution.items()
        )
        or distribution != expected_class_distribution
        or sum(distribution.values()) != final_count
    ):
        issues.append("Quality class distribution does not match the final annotations")

    per_class = report.get("per_class")
    if not isinstance(per_class, dict) or any(label not in CATEGORY_IDS for label in per_class):
        issues.append("Quality per-class labels must use the RoadLabelOps taxonomy")
        return issues
    total_true_positive = 0
    total_false_positive = 0
    total_false_negative = 0
    for label, metrics in per_class.items():
        if not isinstance(metrics, dict) or set(metrics) != QUALITY_CLASS_FIELDS:
            issues.append(f"Quality per-class metrics have an invalid schema: {label}")
            continue
        tp = metrics.get("true_positive_count")
        fp = metrics.get("false_positive_count")
        fn = metrics.get("false_negative_count")
        if not all(_nonnegative_integer(value) for value in (tp, fp, fn)):
            issues.append(f"Quality per-class counts are invalid: {label}")
            continue
        tp = int(tp)
        fp = int(fp)
        fn = int(fn)
        class_precision = _quality_ratio(tp, tp + fp)
        class_recall = _quality_ratio(tp, tp + fn)
        if not _ratio_matches(metrics.get("precision"), class_precision):
            issues.append(f"Quality per-class precision is invalid: {label}")
        if not _ratio_matches(metrics.get("recall"), class_recall):
            issues.append(f"Quality per-class recall is invalid: {label}")
        if not _ratio_matches(
            metrics.get("f1_score"), _quality_f1(class_precision, class_recall)
        ):
            issues.append(f"Quality per-class F1 score is invalid: {label}")
        if tp + fn != expected_class_distribution.get(label, 0):
            issues.append(f"Quality per-class final count is invalid: {label}")
        total_true_positive += tp
        total_false_positive += fp
        total_false_negative += fn
    if (
        total_true_positive != matched_count
        or total_false_positive != prediction_count - matched_count
        or total_false_negative != added_count
    ):
        issues.append("Quality per-class totals do not match the aggregate counts")
    return issues


def _normalise_attributes(
    value: Any, *, annotation_index: int
) -> tuple[dict[str, str], ToolResult | None]:
    if value in (None, []):
        return {}, None
    cvat_value_list = isinstance(value, list)
    if isinstance(value, dict):
        pairs = list(value.items())
    elif isinstance(value, list) and all(isinstance(item, dict) for item in value):
        pairs = [(item.get("name"), item.get("value")) for item in value]
    else:
        return {}, _failure(
            "INVALID_ATTRIBUTES",
            f"Annotation {annotation_index} attributes must be a mapping or CVAT value list",
        )
    attributes: dict[str, str] = {}
    for raw_name, raw_value in pairs:
        name = str(raw_name) if raw_name is not None else ""
        item_value = str(raw_value) if raw_value is not None else ""
        # CVAT serializes an unselected optional select attribute as an empty
        # string. In its value-list representation that means "absent", not an
        # invalid taxonomy value. Explicit mapping inputs remain strict.
        if cvat_value_list and not item_value.strip():
            continue
        if name not in ROAD_ATTRIBUTE_VALUES or item_value not in ROAD_ATTRIBUTE_VALUES[name]:
            return {}, _failure(
                "INVALID_ATTRIBUTES",
                f"Annotation {annotation_index} has an invalid {name or 'unnamed'} attribute",
            )
        if name in attributes:
            return {}, _failure(
                "INVALID_ATTRIBUTES",
                f"Annotation {annotation_index} repeats attribute {name}",
            )
        attributes[name] = item_value
    return attributes, None


def _normalise_annotations(
    session: Session, annotations: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], ToolResult | None]:
    """Resolve legacy defaults, then reject invalid source/image references."""

    completed = [scene for scene in session.scenes if scene.status == "completed"]
    if not completed:
        return [], _failure(
            "NO_COMPLETED_SCENES", "Only completed scenes can enter a final release"
        )
    if len(completed) != len(session.scenes):
        return [], _failure(
            "INCOMPLETE_SCENES",
            "Every scene must complete human review before a final release",
        )
    if not session.session_id or not session.source_path or not session.source_sha256:
        return [], _failure(
            "INVALID_SESSION_LINEAGE",
            "A release requires a session id, source path, and source SHA-256 lineage",
        )
    if len(session.source_sha256) != 64 or any(
        character not in "0123456789abcdefABCDEF" for character in session.source_sha256
    ):
        return [], _failure("INVALID_SESSION_LINEAGE", "Session source SHA-256 lineage is invalid")
    if any(scene.session_id != session.session_id for scene in completed):
        return [], _failure(
            "INVALID_SESSION_LINEAGE",
            "Completed scene lineage must belong to the releasing session",
        )
    if not session.demo and any(
        scene.cvat_task_id is None or not scene.cvat_job_ids for scene in completed
    ):
        return [], _failure(
            "CVAT_LINEAGE_INCOMPLETE",
            "Every completed real scene must retain its CVAT Task and Job lineage",
        )
    if (
        session.width < 10
        or session.height < 10
        or session.width > _JPEG_MAX_DIMENSION
        or session.height > _JPEG_MAX_DIMENSION
        or session.fps <= 0
        or not math.isfinite(session.fps)
        or session.duration_seconds <= 0
        or not math.isfinite(session.duration_seconds)
    ):
        return [], _failure(
            "INVALID_SESSION_GEOMETRY",
            "Session dimensions must be between 10 and 65500 pixels and timing must be positive",
        )

    scene_ids = [scene.scene_id for scene in completed]
    if len(scene_ids) != len(set(scene_ids)):
        return [], _failure("DUPLICATE_SCENE_ID", "Completed scene ids must be unique")
    for scene in completed:
        if not _safe_path_component(scene.scene_id):
            return [], _failure(
                "INVALID_SCENE_ID",
                f"Scene id is not a safe path component: {scene.scene_id!r}",
            )
        start = _as_number(scene.start_seconds)
        end = _as_number(scene.end_seconds)
        if (
            start is None
            or end is None
            or start < 0
            or end <= start
            or end > session.duration_seconds + 1e-6
        ):
            return [], _failure(
                "INVALID_SCENE_INTERVAL",
                f"Scene {scene.scene_id} has an invalid source interval",
            )
    ordered_scenes = sorted(completed, key=lambda item: item.start_seconds)
    if any(
        current.start_seconds < previous.end_seconds - 1e-6
        for previous, current in pairwise(ordered_scenes)
    ):
        return [], _failure(
            "OVERLAPPING_SCENES",
            "Completed scene source intervals must not overlap",
        )

    scenes = {scene.scene_id: scene for scene in completed}
    default_scene = completed[0]
    normalised: list[dict[str, Any]] = []
    for index, item in enumerate(annotations):
        if not isinstance(item, dict):
            return [], _failure("INVALID_ANNOTATION", f"Annotation {index} must be an object")
        label = item.get("label")
        if label not in CATEGORY_IDS:
            return [], _failure(
                "UNKNOWN_CATEGORY",
                f"Annotation {index} uses an unknown RoadLabelOps category: {label!r}",
            )
        if not session.demo:
            missing = [key for key in ("scene_id", "frame", "bbox") if key not in item]
            if missing:
                return [], _failure(
                    "INVALID_ANNOTATION",
                    f"Annotation {index} is missing required field(s): {', '.join(missing)}",
                )
        scene_id = str(item.get("scene_id") or default_scene.scene_id)
        scene = scenes.get(scene_id)
        if scene is None:
            return [], _failure(
                "UNKNOWN_IMAGE_REFERENCE",
                f"Annotation {index} refers to a scene that is not completed: {scene_id}",
            )
        raw_frame = item.get("frame", 0)
        if isinstance(raw_frame, bool) or not isinstance(raw_frame, int):
            return [], _failure(
                "INVALID_FRAME_REFERENCE", f"Annotation {index} frame must be an integer"
            )
        if raw_frame < 0 or raw_frame >= _frame_limit(scene, session.fps):
            return [], _failure(
                "INVALID_FRAME_REFERENCE",
                f"Annotation {index} frame {raw_frame} is outside scene {scene_id}",
            )

        # Legacy defaults are deliberately limited to generated demo data. Real
        # CVAT annotations must carry explicit source-frame geometry.
        raw_bbox = item.get("bbox", [0, 0, 1, 1])
        if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
            return [], _failure("INVALID_BBOX", f"Annotation {index} bbox must be [x1, y1, x2, y2]")
        bbox = [_as_number(value) for value in raw_bbox]
        if any(value is None for value in bbox):
            return [], _failure(
                "INVALID_BBOX", f"Annotation {index} bbox must contain finite numbers"
            )
        x1, y1, x2, y2 = (float(value) for value in bbox)
        if not (0 <= x1 < x2 <= session.width and 0 <= y1 < y2 <= session.height):
            return [], _failure(
                "BBOX_OUT_OF_BOUNDS",
                f"Annotation {index} bbox must be non-empty and within {session.width}x{session.height}",
            )
        attributes, attribute_error = _normalise_attributes(
            item.get("attributes"), annotation_index=index
        )
        if attribute_error is not None:
            return [], attribute_error
        normalised.append(
            {
                "scene_id": scene_id,
                "frame": raw_frame,
                "label": label,
                "bbox": [x1, y1, x2, y2],
                "source": str(item.get("source", "manual")),
                "attributes": attributes,
            }
        )
    if not session.demo:
        final_counts = [scene.final_count for scene in completed]
        if any(count is None or isinstance(count, bool) or count < 0 for count in final_counts):
            return [], _failure(
                "FINAL_COUNT_MISSING",
                "Every completed real scene must have a recorded final annotation count",
            )
        if sum(int(count) for count in final_counts if count is not None) != len(normalised):
            return [], _failure(
                "FINAL_COUNT_MISMATCH",
                "Final annotation sidecar counts do not match the release annotations",
            )
    return normalised, None


def normalise_release_predictions(
    session: Session, predictions: Any
) -> tuple[list[dict[str, Any]], ToolResult | None]:
    """Validate and canonicalize the exact detector snapshot frozen in a V2 release."""

    if not isinstance(predictions, list) or not all(isinstance(item, dict) for item in predictions):
        return [], _failure(
            "PREDICTIONS_INVALID", "Release predictions must be a list of objects"
        )
    scenes = {scene.scene_id: scene for scene in session.scenes}
    canonical: list[dict[str, Any]] = []
    prediction_ids: set[str] = set()
    required_fields = {
        "prediction_id",
        "scene_id",
        "frame",
        "label",
        "confidence",
        "bbox",
        "source",
    }
    for index, item in enumerate(predictions):
        if set(item) != required_fields:
            return [], _failure(
                "PREDICTIONS_INVALID",
                f"Prediction {index} fields do not match the canonical schema",
            )
        prediction_id = item.get("prediction_id")
        scene_id = item.get("scene_id")
        scene = scenes.get(scene_id)
        frame = item.get("frame")
        label = item.get("label")
        confidence = _as_number(item.get("confidence"))
        bbox = item.get("bbox")
        if (
            not isinstance(prediction_id, str)
            or not prediction_id
            or prediction_id in prediction_ids
            or not isinstance(scene_id, str)
            or scene is None
            or isinstance(frame, bool)
            or not isinstance(frame, int)
            or frame < 0
            or frame >= _frame_limit(scene, session.fps)
            or label not in CATEGORY_IDS
            or confidence is None
            or not 0 <= confidence <= 1
            or not isinstance(item.get("source"), str)
            or not str(item["source"]).strip()
            or not isinstance(bbox, (list, tuple))
            or len(bbox) != 4
        ):
            return [], _failure(
                "PREDICTIONS_INVALID", f"Prediction {index} has invalid lineage or fields"
            )
        numeric_bbox = [_as_number(value) for value in bbox]
        if any(value is None for value in numeric_bbox):
            return [], _failure(
                "PREDICTIONS_INVALID", f"Prediction {index} has a non-finite bbox"
            )
        x1, y1, x2, y2 = (float(value) for value in numeric_bbox)
        if not (0 <= x1 < x2 <= session.width and 0 <= y1 < y2 <= session.height):
            return [], _failure(
                "PREDICTIONS_INVALID", f"Prediction {index} bbox is out of bounds"
            )
        prediction_ids.add(prediction_id)
        canonical.append(
            {
                "prediction_id": prediction_id,
                "scene_id": scene_id,
                "frame": frame,
                "label": label,
                "confidence": confidence,
                "bbox": [x1, y1, x2, y2],
                "source": str(item["source"]),
            }
        )
    canonical.sort(
        key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )
    return canonical, None


def normalise_evaluated_frames(
    session: Session, frame_keys: Iterable[tuple[str | None, int]] | None
) -> tuple[list[dict[str, Any]], ToolResult | None]:
    if frame_keys is None:
        return [], _failure(
            "EVALUATED_FRAMES_MISSING",
            "A V2 release requires the exact evaluated-frame universe",
        )
    scenes = {scene.scene_id: scene for scene in session.scenes}
    canonical: set[tuple[str, int]] = set()
    try:
        values = list(frame_keys)
    except TypeError:
        return [], _failure(
            "EVALUATED_FRAMES_INVALID", "Evaluated frames must be an iterable of pairs"
        )
    for item in values:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            return [], _failure(
                "EVALUATED_FRAMES_INVALID", "An evaluated-frame key is invalid"
            )
        scene_id, frame = item
        scene = scenes.get(scene_id)
        if (
            not isinstance(scene_id, str)
            or scene is None
            or isinstance(frame, bool)
            or not isinstance(frame, int)
            or frame < 0
            or frame >= _frame_limit(scene, session.fps)
        ):
            return [], _failure(
                "EVALUATED_FRAMES_INVALID", "An evaluated-frame key has invalid lineage"
            )
        canonical.add((scene_id, frame))
    return [
        {"scene_id": scene_id, "frame": frame}
        for scene_id, frame in sorted(canonical)
    ], None


def _image_filename(scene_id: str, frame: int) -> str:
    if not _safe_path_component(scene_id):
        raise ValueError("scene id is not a safe path component")
    return f"images/{scene_id}_frame_{frame:06d}.jpg"


def _safe_output(root: Path, relative_name: str) -> Path:
    output = root / relative_name
    if output.parent.resolve().is_relative_to(root.resolve()):
        return output
    raise ValueError("release output path escapes its staging directory")


def _materialize_image(
    output: Path,
    scene: Scene,
    frame: int,
    *,
    demo: bool,
    width: int,
    height: int,
    source_path: Path | None = None,
) -> ToolResult | None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if demo:
        output.write_bytes(_demo_jpeg(width, height))
        return None
    source = source_path or Path(scene.video_path)
    if not source.is_file():
        return _failure(
            "RELEASE_SOURCE_FRAME_MISSING",
            f"Cannot materialize {scene.scene_id}:{frame}; scene video is not available",
        )
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-vf",
        f"select=eq(n\\,{frame})",
        "-frames:v",
        "1",
        "-y",
        str(output),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=90, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return _failure("RELEASE_FRAME_EXPORT_FAILED", str(exc), retryable=True)
    if completed.returncode != 0 or not output.is_file() or output.stat().st_size == 0:
        return _failure(
            "RELEASE_FRAME_EXPORT_FAILED",
            f"Could not materialize source frame {scene.scene_id}:{frame}",
            retryable=True,
        )
    return None


def _copy_with_digest(source: Path, destination: Path) -> str:
    digest = hashlib.sha256()
    with source.open("rb") as source_handle, destination.open("xb") as output_handle:
        for block in iter(lambda: source_handle.read(1024 * 1024), b""):
            digest.update(block)
            output_handle.write(block)
        output_handle.flush()
        os.fsync(output_handle.fileno())
    return digest.hexdigest()


def _snapshot_and_validate_sources(
    session: Session,
    completed: dict[str, Scene],
    snapshot_root: Path,
) -> tuple[dict[str, str], dict[str, Path], ToolResult | None]:
    """Bind frame extraction to stable, privately copied scene file descriptors."""

    if session.demo:
        return {}, {}, None
    source = Path(session.source_path)
    if source.is_symlink() or not source.is_file():
        return {}, {}, _failure(
            "RELEASE_SOURCE_MISSING", "The original source video is no longer available"
        )
    if _digest(source) != session.source_sha256.lower():
        return {}, {}, _failure(
            "SOURCE_SHA256_MISMATCH",
            "The original source video no longer matches its recorded SHA-256",
        )

    snapshot_root.mkdir(parents=True, exist_ok=False)
    scene_sha256: dict[str, str] = {}
    scene_sources: dict[str, Path] = {}
    for scene in completed.values():
        recorded_digest = getattr(scene, "video_sha256", None)
        if not isinstance(recorded_digest, str) or not re.fullmatch(
            r"[0-9a-fA-F]{64}", recorded_digest
        ):
            return {}, {}, _failure(
                "SCENE_SHA256_INVALID",
                f"Scene has no valid recorded SHA-256 lineage: {scene.scene_id}",
            )
        scene_path = Path(scene.video_path)
        if scene_path.is_symlink() or not scene_path.is_file():
            return {}, {}, _failure(
                "RELEASE_SOURCE_FRAME_MISSING",
                f"Scene video is no longer available: {scene.scene_id}",
            )
        snapshot = snapshot_root / f"{scene.scene_id}.mp4"
        try:
            digest = _copy_with_digest(scene_path, snapshot)
        except OSError:
            return {}, {}, _failure(
                "RELEASE_SOURCE_SNAPSHOT_FAILED",
                f"Scene video could not be snapshotted: {scene.scene_id}",
                retryable=True,
            )
        if digest != recorded_digest.lower():
            return {}, {}, _failure(
                "SCENE_SHA256_MISMATCH",
                f"Scene video no longer matches its recorded SHA-256: {scene.scene_id}",
            )
        scene_sha256[scene.scene_id] = digest
        scene_sources[scene.scene_id] = snapshot
    return scene_sha256, scene_sources, None


def _normalise_scene_tags(
    scene: Scene,
    *,
    fps: float,
) -> tuple[list[dict[str, Any]], ToolResult | None]:
    tags: list[dict[str, Any]] = []
    for index, item in enumerate(scene.scene_tags):
        if not isinstance(item, dict):
            return [], _failure(
                "INVALID_SCENE_TAG", f"Scene {scene.scene_id} tag {index} must be an object"
            )
        if item.get("scene_id", scene.scene_id) != scene.scene_id:
            return [], _failure(
                "INVALID_SCENE_TAG", f"Scene {scene.scene_id} tag {index} has invalid lineage"
            )
        label = item.get("label")
        if label not in ROAD_SCENE_TAG_VALUES:
            return [], _failure(
                "INVALID_SCENE_TAG", f"Scene {scene.scene_id} has unknown tag {label!r}"
            )
        raw_attributes = item.get("attributes")
        if isinstance(raw_attributes, dict):
            values = raw_attributes
        elif isinstance(raw_attributes, list) and all(
            isinstance(attribute, dict) for attribute in raw_attributes
        ):
            values = {
                str(attribute.get("name")): attribute.get("value")
                for attribute in raw_attributes
            }
        else:
            values = {}
        value = values.get(str(label))
        if value not in ROAD_SCENE_TAG_VALUES[str(label)]:
            return [], _failure(
                "INVALID_SCENE_TAG",
                f"Scene {scene.scene_id} tag {label!r} has an invalid value",
            )
        frame = item.get("frame")
        if (
            isinstance(frame, bool)
            or not isinstance(frame, int)
            or frame < 0
            or frame >= _frame_limit(scene, fps)
        ):
            return [], _failure(
                "INVALID_SCENE_TAG",
                f"Scene {scene.scene_id} tag {label!r} has an invalid frame",
            )
        tags.append(
            {
                "frame": frame,
                "label": str(label),
                "value": str(value),
                "source": str(item.get("source", "manual")),
            }
        )
    return tags, None


def _payload_digests(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _digest(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and not path.is_symlink()
        and path.name not in {MANIFEST_FILENAME, RECEIPT_FILENAME}
    }


def _yolo_metadata() -> str:
    names = "\n".join(
        f"  {index}: {label}" for index, label in enumerate(ROAD_LABEL_TAXONOMY)
    )
    # Paths are relative to this dataset.yaml file. Omitting ``path`` avoids
    # Ultralytics resolving ``.`` against the caller's process working directory.
    return f"train: images\nval: images\nnames:\n{names}\n"


def _yolo_label_text(
    image: dict[str, Any], annotations: list[dict[str, Any]]
) -> str:
    width = float(image["width"])
    height = float(image["height"])
    lines: list[str] = []
    for annotation in sorted(annotations, key=lambda item: int(item["id"])):
        x, y, box_width, box_height = (float(value) for value in annotation["bbox"])
        values = (
            x + box_width / 2,
            y + box_height / 2,
            box_width,
            box_height,
        )
        lines.append(
            " ".join(
                [
                    str(int(annotation["category_id"]) - 1),
                    f"{values[0] / width:.8f}",
                    f"{values[1] / height:.8f}",
                    f"{values[2] / width:.8f}",
                    f"{values[3] / height:.8f}",
                ]
            )
        )
    return "\n".join(lines) + ("\n" if lines else "")


def _write_text_durable(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _receipt(release_id: str, manifest: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    payload_sha256 = dict(manifest["payload_sha256"])
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "receipt_type": "roadlabelops.release.integrity",
        "release_id": release_id,
        "manifest_file": MANIFEST_FILENAME,
        "manifest_sha256": _digest(manifest_path),
        "payload_tree_sha256": _canonical_digest(payload_sha256),
        "payload_file_count": len(payload_sha256),
        "payload_sha256": payload_sha256,
    }


@contextmanager
def _publication_lock(releases_dir: Path):
    """Serialize immutable directory publication across threads and processes."""

    lock_path = releases_dir / ".publish.lock"
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _publish_without_replacing(staging: Path, target: Path) -> None:
    """Atomically publish a fully verified directory without exposing partial files."""

    with _publication_lock(target.parent):
        if target.exists() or target.is_symlink():
            raise FileExistsError(target)
        # staging and target live under the same releases directory, therefore
        # this rename is one atomic metadata operation on supported local filesystems.
        os.rename(staging, target)


def build_coco_release(
    store: LocalStore,
    session: Session,
    version: str,
    annotations: list[dict[str, Any]],
    *,
    extract_frames: bool | None = None,
    quality_report: dict[str, Any] | None = None,
    predictions: list[dict[str, Any]] | None = None,
    evaluated_frame_keys: Iterable[tuple[str | None, int]] | None = None,
    accepted_jobs: int = 0,
    reviewed_jobs: int = 0,
    first_pass_acceptance_reason: str | None = None,
) -> ToolResult:
    """Build an immutable COCO release and return its verified integrity receipt.

    Real sessions always materialize referenced video frames. ``extract_frames``
    remains accepted for source compatibility but cannot disable that requirement.
    """

    release_id = _release_id(session, version)
    if not _valid_version(version) or not _safe_release_name(release_id):
        return _failure(
            "INVALID_RELEASE_VERSION", "Release version must be a safe semantic version"
        )
    target = store.releases_dir / release_id
    if target.exists() or target.is_symlink():
        return _failure("RELEASE_EXISTS", "Dataset releases are immutable; choose a new version")
    if not session.demo and extract_frames is False:
        return _failure(
            "RELEASE_IMAGES_REQUIRED",
            "Real releases must materialize the referenced source-video frames",
        )

    normalised, validation_error = _normalise_annotations(session, annotations)
    if validation_error is not None:
        return validation_error
    if quality_report is None:
        return _failure(
            "QUALITY_REPORT_REQUIRED", "A V2 release requires a frozen quality report"
        )
    canonical_predictions, prediction_error = normalise_release_predictions(
        session, predictions
    )
    if prediction_error is not None:
        return prediction_error
    evaluated_frames, evaluated_error = normalise_evaluated_frames(
        session, evaluated_frame_keys
    )
    if evaluated_error is not None:
        return evaluated_error
    if (
        not _nonnegative_integer(accepted_jobs)
        or not _nonnegative_integer(reviewed_jobs)
        or accepted_jobs > reviewed_jobs
    ):
        return _failure(
            "QUALITY_CONTEXT_INVALID", "Reviewed and accepted Job counts are invalid"
        )
    calculated_quality = calculate_quality(
        canonical_predictions,
        normalised,
        accepted_jobs,
        reviewed_jobs,
        evaluated_frame_keys={
            (str(item["scene_id"]), int(item["frame"])) for item in evaluated_frames
        },
        first_pass_acceptance_reason=first_pass_acceptance_reason,
    ).data
    frozen_quality = dict(quality_report)
    legacy_fields = QUALITY_REPORT_FIELDS - {"first_pass_acceptance_reason"}
    if set(frozen_quality) == legacy_fields:
        frozen_quality["first_pass_acceptance_reason"] = (
            first_pass_acceptance_reason if reviewed_jobs == 0 else None
        )
    quality_issues = _quality_validation_issues(
        frozen_quality,
        expected_final_count=len(normalised),
        expected_class_distribution=dict(
            sorted(Counter(item["label"] for item in normalised).items())
        ),
    )
    if quality_issues:
        return _failure("QUALITY_REPORT_INVALID", quality_issues[0])
    if frozen_quality != calculated_quality:
        return _failure(
            "QUALITY_REPORT_MISMATCH",
            "Frozen quality report does not match recomputation from predictions and final labels",
        )
    completed = {scene.scene_id: scene for scene in session.scenes if scene.status == "completed"}
    scene_tags: dict[str, list[dict[str, Any]]] = {}
    for scene in completed.values():
        normalised_tags, tag_error = _normalise_scene_tags(scene, fps=session.fps)
        if tag_error is not None:
            return tag_error
        scene_tags[scene.scene_id] = normalised_tags
    frame_keys = {(item["scene_id"], item["frame"]) for item in normalised}
    # Retain completed, empty scenes as explicit image lineage records.
    frame_keys.update(
        (scene_id, 0) for scene_id in completed if not any(key[0] == scene_id for key in frame_keys)
    )
    ordered_frames = sorted(frame_keys)
    image_ids = {frame_key: index for index, frame_key in enumerate(ordered_frames, start=1)}
    staging = store.releases_dir / f".{release_id}.{uuid.uuid4().hex}.tmp"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        source_snapshot_root = staging / ".source-snapshots"
        scene_sha256, scene_sources, lineage_error = _snapshot_and_validate_sources(
            session, completed, source_snapshot_root
        )
        if lineage_error is not None:
            shutil.rmtree(staging, ignore_errors=True)
            return lineage_error
        images: list[dict[str, Any]] = []
        for scene_id, frame in ordered_frames:
            filename = _image_filename(scene_id, frame)
            image_error = _materialize_image(
                _safe_output(staging, filename),
                completed[scene_id],
                frame,
                demo=session.demo,
                width=session.width,
                height=session.height,
                source_path=scene_sources.get(scene_id),
            )
            if image_error is not None:
                shutil.rmtree(staging, ignore_errors=True)
                return image_error
            images.append(
                {
                    "id": image_ids[(scene_id, frame)],
                    "file_name": filename,
                    "scene_id": scene_id,
                    "frame": frame,
                    "width": session.width,
                    "height": session.height,
                }
            )
        if source_snapshot_root.exists():
            shutil.rmtree(source_snapshot_root)
        coco_annotations = [
            {
                "id": index,
                "image_id": image_ids[(item["scene_id"], item["frame"])],
                "category_id": CATEGORY_IDS[item["label"]],
                "bbox": [
                    round(item["bbox"][0], 2),
                    round(item["bbox"][1], 2),
                    round(item["bbox"][2] - item["bbox"][0], 2),
                    round(item["bbox"][3] - item["bbox"][1], 2),
                ],
                "area": round(
                    (item["bbox"][2] - item["bbox"][0]) * (item["bbox"][3] - item["bbox"][1]), 2
                ),
                "iscrowd": 0,
                "source": item["source"],
                "attributes": item["attributes"],
            }
            for index, item in enumerate(normalised, start=1)
        ]
        coco = {
            "info": {
                "description": "RoadLabelOps release",
                "version": version,
                "created_at": utc_now(),
            },
            "images": images,
            "categories": [
                {"id": CATEGORY_IDS[label], "name": label} for label in ROAD_LABEL_TAXONOMY
            ],
            "annotations": coco_annotations,
        }
        export_path = staging / ANNOTATIONS_FILENAME
        store.write_json_atomic(export_path, coco)
        annotations_by_image: dict[int, list[dict[str, Any]]] = {
            int(image["id"]): [] for image in images
        }
        for annotation in coco_annotations:
            annotations_by_image[int(annotation["image_id"])].append(annotation)
        for image in images:
            label_name = f"{Path(str(image['file_name'])).stem}.txt"
            label_path = _safe_output(
                staging,
                f"{YOLO_LABELS_DIRECTORY}/{label_name}",
            )
            _write_text_durable(
                label_path,
                _yolo_label_text(image, annotations_by_image[int(image["id"])]),
            )
        _write_text_durable(staging / YOLO_METADATA_FILENAME, _yolo_metadata())
        store.write_json_atomic(staging / PREDICTIONS_FILENAME, canonical_predictions)
        store.write_json_atomic(staging / QUALITY_FILENAME, frozen_quality)
        payload_sha256 = _payload_digests(staging)
        manifest = {
            "release_id": release_id,
            "version": version,
            "status": "final",
            "format": "COCO",
            "formats": ["COCO", "YOLO"],
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "taxonomy": list(ROAD_LABEL_TAXONOMY),
            "source_session": {
                "session_id": session.session_id,
                # Basename preserves human-readable lineage without leaking a
                # workstation's absolute home-directory path into shared data.
                "source_path": Path(session.source_path).name,
                "source_path_kind": "basename",
                "source_sha256": session.source_sha256,
                "duration_seconds": session.duration_seconds,
                "fps": session.fps,
                "width": session.width,
                "height": session.height,
            },
            "session_ids": [session.session_id],
            "scene_ids": sorted(completed),
            "scene_lineage": [
                {
                    "scene_id": scene.scene_id,
                    "session_id": scene.session_id,
                    "start_seconds": scene.start_seconds,
                    "end_seconds": scene.end_seconds,
                    "cvat_task_id": scene.cvat_task_id,
                    "cvat_job_ids": list(scene.cvat_job_ids),
                    "video_sha256": scene_sha256.get(scene.scene_id),
                    "final_count": scene.final_count,
                    "scene_tags": scene_tags[scene.scene_id],
                }
                for scene in sorted(completed.values(), key=lambda item: item.scene_id)
            ],
            "cvat_task_ids": [
                scene.cvat_task_id
                for scene in sorted(completed.values(), key=lambda item: item.scene_id)
                if scene.cvat_task_id is not None
            ],
            "object_count": len(normalised),
            "prediction_count": len(canonical_predictions),
            "image_count": len(images),
            "image_materialization": "deterministic_placeholder"
            if session.demo
            else "source_frame",
            "export_sha256": payload_sha256[ANNOTATIONS_FILENAME],
            "quality_sha256": payload_sha256.get(QUALITY_FILENAME),
            "predictions_sha256": payload_sha256[PREDICTIONS_FILENAME],
            "quality_schema_version": "1.0.0",
            "evaluated_frames": evaluated_frames,
            "quality_context": {
                "accepted_jobs": accepted_jobs,
                "reviewed_jobs": reviewed_jobs,
                "first_pass_acceptance_reason": first_pass_acceptance_reason,
            },
            "yolo_metadata_sha256": payload_sha256[YOLO_METADATA_FILENAME],
            "yolo_label_file_count": len(images),
            # file_sha256 is retained for existing dashboard/export consumers.
            "file_sha256": payload_sha256,
            "payload_sha256": payload_sha256,
            "created_at": utc_now(),
        }
        manifest_path = staging / MANIFEST_FILENAME
        store.write_json_atomic(manifest_path, manifest)
        store.write_json_atomic(
            staging / RECEIPT_FILENAME, _receipt(release_id, manifest, manifest_path)
        )
        staged_verification = verify_coco_release(
            staging,
            expected_release_id=release_id,
            expected_session_id=session.session_id,
            expected_version=version,
        )
        if not staged_verification.ok:
            shutil.rmtree(staging, ignore_errors=True)
            return staged_verification
        try:
            _publish_without_replacing(staging, target)
        except FileExistsError:
            shutil.rmtree(staging, ignore_errors=True)
            return _failure(
                "RELEASE_EXISTS", "Dataset releases are immutable; choose a new version"
            )
    except Exception as exc:
        shutil.rmtree(staging, ignore_errors=True)
        return _failure("RELEASE_BUILD_FAILED", str(exc), retryable=False)
    verified = verify_coco_release(
        target,
        expected_release_id=release_id,
        expected_session_id=session.session_id,
        expected_version=version,
        expected_manifest_sha256=_digest(target / MANIFEST_FILENAME),
        expected_receipt_sha256=_digest(target / RECEIPT_FILENAME),
    )
    verified.data["manifest"] = manifest
    verified.side_effects.append(str(target))
    return verified


def _verification_result(
    ok: bool, release_path: Path, release_id: str | None, issues: list[dict[str, str]]
) -> ToolResult:
    receipt = {
        "receipt_type": "roadlabelops.release.verification",
        "release_id": release_id,
        "release_path": str(release_path),
        "verified_at": utc_now(),
        "valid": ok,
        "issues": issues,
    }
    if ok:
        return ToolResult.success({"receipt": receipt, "path": str(release_path)})
    first = (
        issues[0]
        if issues
        else {"code": "RELEASE_INVALID", "message": "Release verification failed"}
    )
    return ToolResult(
        ok=False, data={"receipt": receipt, "path": str(release_path)}, error=first, retryable=False
    )


def _issue(issues: list[dict[str, str]], code: str, message: str) -> None:
    issues.append({"code": code, "message": message})


def _read_coco_semantics(
    root: Path,
    manifest: dict[str, Any],
    expected_payload: dict[str, str],
    issues: list[dict[str, str]],
) -> dict[str, Any] | None:
    """Validate the payload as COCO rather than treating it as opaque bytes."""

    annotation_path = root / ANNOTATIONS_FILENAME
    if ANNOTATIONS_FILENAME not in expected_payload or not annotation_path.is_file():
        _issue(issues, "COCO_EXPORT_MISSING", "Manifest does not contain the COCO export")
        return None
    try:
        coco = json.loads(annotation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        _issue(issues, "COCO_EXPORT_INVALID", "COCO export is not valid JSON")
        return None
    if not isinstance(coco, dict):
        _issue(issues, "COCO_EXPORT_INVALID", "COCO export must be an object")
        return None

    expected_categories = [
        {"id": CATEGORY_IDS[label], "name": label} for label in ROAD_LABEL_TAXONOMY
    ]
    if coco.get("categories") != expected_categories:
        _issue(issues, "COCO_CATEGORIES_INVALID", "COCO categories do not match the taxonomy")

    images = coco.get("images")
    annotations = coco.get("annotations")
    if not isinstance(images, list) or not all(isinstance(item, dict) for item in images):
        _issue(issues, "COCO_IMAGES_INVALID", "COCO images must be a list of objects")
        images = []
    if not isinstance(annotations, list) or not all(
        isinstance(item, dict) for item in annotations
    ):
        _issue(issues, "COCO_ANNOTATIONS_INVALID", "COCO annotations must be a list of objects")
        annotations = []

    image_by_id: dict[int, dict[str, Any]] = {}
    seen_filenames: set[str] = set()
    readable_image_count = 0
    raw_scene_ids = manifest.get("scene_ids", [])
    scene_ids = {
        item for item in raw_scene_ids if isinstance(item, str)
    } if isinstance(raw_scene_ids, list) else set()
    for index, item in enumerate(images):
        image_id = item.get("id")
        if isinstance(image_id, bool) or not isinstance(image_id, int) or image_id <= 0:
            _issue(issues, "COCO_IMAGE_ID_INVALID", f"COCO image {index} has an invalid id")
            continue
        if image_id in image_by_id:
            _issue(issues, "COCO_IMAGE_ID_DUPLICATE", f"Duplicate COCO image id: {image_id}")
        image_by_id[image_id] = item
        filename = item.get("file_name")
        scene_id = item.get("scene_id")
        frame = item.get("frame")
        if (
            not isinstance(filename, str)
            or not filename.startswith("images/")
            or Path(filename).is_absolute()
            or ".." in Path(filename).parts
            or "\\" in filename
            or not isinstance(scene_id, str)
            or not _safe_path_component(scene_id)
            or isinstance(frame, bool)
            or not isinstance(frame, int)
            or frame < 0
            or filename != _image_filename(scene_id, frame)
        ):
            _issue(
                issues,
                "COCO_IMAGE_PATH_INVALID",
                f"COCO image {image_id} does not use its canonical image path",
            )
        elif filename in seen_filenames:
            _issue(issues, "COCO_IMAGE_PATH_DUPLICATE", f"Duplicate COCO image path: {filename}")
        else:
            seen_filenames.add(filename)
            if filename not in expected_payload:
                _issue(
                    issues,
                    "COCO_IMAGE_UNMANIFESTED",
                    f"COCO image is not covered by the manifest: {filename}",
                )
            actual_dimensions = _jpeg_dimensions(root / filename)
            declared_dimensions = (item.get("width"), item.get("height"))
            if actual_dimensions is None:
                _issue(
                    issues,
                    "COCO_IMAGE_FILE_INVALID",
                    f"COCO image is not a readable JPEG: {filename}",
                )
            else:
                readable_image_count += 1
            if actual_dimensions is not None and actual_dimensions != declared_dimensions:
                _issue(
                    issues,
                    "COCO_IMAGE_DIMENSIONS_MISMATCH",
                    f"JPEG dimensions do not match COCO geometry: {filename}",
                )
            elif actual_dimensions is not None and min(actual_dimensions) < 10:
                _issue(
                    issues,
                    "COCO_IMAGE_GEOMETRY_INVALID",
                    f"COCO image is too small for YOLO ingestion: {filename}",
                )
            if manifest.get("image_materialization") == "deterministic_placeholder":
                try:
                    expected_demo_image = _demo_jpeg(
                        int(item.get("width")), int(item.get("height"))
                    )
                    exact_demo_image = (root / filename).read_bytes() == expected_demo_image
                except (OSError, TypeError, ValueError):
                    exact_demo_image = False
                if not exact_demo_image:
                    _issue(
                        issues,
                        "DEMO_IMAGE_CONTENT_INVALID",
                        f"Demo image is not the canonical placeholder: {filename}",
                    )
        if scene_id not in scene_ids:
            _issue(issues, "COCO_SCENE_INVALID", f"COCO image {image_id} has unknown scene lineage")
        if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
            _issue(issues, "COCO_FRAME_INVALID", f"COCO image {image_id} has an invalid frame")
        if not all(
            isinstance(item.get(key), int) and item[key] > 0 for key in ("width", "height")
        ):
            _issue(issues, "COCO_IMAGE_GEOMETRY_INVALID", f"COCO image {image_id} has invalid geometry")

    manifested_image_files = {
        name for name in expected_payload if name.startswith("images/")
    }
    if manifested_image_files != seen_filenames:
        _issue(
            issues,
            "COCO_IMAGE_SET_INVALID",
            "Manifested image files must exactly match canonical COCO image references",
        )

    if (
        manifest.get("image_materialization") == "source_frame"
        and readable_image_count == len(images)
        and images
        and not _ffmpeg_decodes_release_images(root / "images", len(images))
    ):
        _issue(
            issues,
            "COCO_IMAGE_DECODE_FAILED",
            "FFmpeg could not fully decode every materialized source-frame image",
        )

    seen_annotation_ids: set[int] = set()
    for index, item in enumerate(annotations):
        annotation_id = item.get("id")
        if (
            isinstance(annotation_id, bool)
            or not isinstance(annotation_id, int)
            or annotation_id <= 0
        ):
            _issue(
                issues,
                "COCO_ANNOTATION_ID_INVALID",
                f"COCO annotation {index} has an invalid id",
            )
        elif annotation_id in seen_annotation_ids:
            _issue(
                issues,
                "COCO_ANNOTATION_ID_DUPLICATE",
                f"Duplicate COCO annotation id: {annotation_id}",
            )
        else:
            seen_annotation_ids.add(annotation_id)
        image = image_by_id.get(item.get("image_id"))
        if image is None:
            _issue(
                issues,
                "COCO_IMAGE_REFERENCE_INVALID",
                f"COCO annotation {annotation_id or index} refers to an unknown image",
            )
            continue
        if item.get("category_id") not in CATEGORY_IDS.values():
            _issue(
                issues,
                "COCO_CATEGORY_REFERENCE_INVALID",
                f"COCO annotation {annotation_id or index} refers to an unknown category",
            )
        raw_bbox = item.get("bbox")
        if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
            _issue(issues, "COCO_BBOX_INVALID", f"COCO annotation {annotation_id or index} has an invalid bbox")
            continue
        bbox = [_as_number(value) for value in raw_bbox]
        if any(value is None for value in bbox):
            _issue(issues, "COCO_BBOX_INVALID", f"COCO annotation {annotation_id or index} has a non-finite bbox")
            continue
        x, y, width, height = (float(value) for value in bbox)
        image_width = _as_number(image.get("width"))
        image_height = _as_number(image.get("height"))
        if (
            x < 0
            or y < 0
            or width <= 0
            or height <= 0
            or image_width is None
            or image_height is None
            or x + width > image_width + 1e-6
            or y + height > image_height + 1e-6
        ):
            _issue(issues, "COCO_BBOX_INVALID", f"COCO annotation {annotation_id or index} has an out-of-bounds bbox")
        area = _as_number(item.get("area"))
        if area is None or not math.isclose(area, width * height, abs_tol=0.02):
            _issue(issues, "COCO_AREA_INVALID", f"COCO annotation {annotation_id or index} has an invalid area")
        attributes = item.get("attributes", {})
        if not isinstance(attributes, dict) or any(
            name not in ROAD_ATTRIBUTE_VALUES or value not in ROAD_ATTRIBUTE_VALUES[name]
            for name, value in attributes.items()
        ):
            _issue(
                issues,
                "COCO_ATTRIBUTES_INVALID",
                f"COCO annotation {annotation_id or index} has invalid attributes",
            )

    if manifest.get("image_count") != len(images):
        _issue(issues, "IMAGE_COUNT_MISMATCH", "Manifest image count does not match COCO")
    if manifest.get("object_count") != len(annotations):
        _issue(issues, "OBJECT_COUNT_MISMATCH", "Manifest object count does not match COCO")
    if manifest.get("schema_version") == MANIFEST_SCHEMA_VERSION:
        source = manifest.get("source_session")
        lineage = manifest.get("scene_lineage")
        source_fps = _as_number(source.get("fps")) if isinstance(source, dict) else None
        source_width = source.get("width") if isinstance(source, dict) else None
        source_height = source.get("height") if isinstance(source, dict) else None
        lineage_items = lineage if isinstance(lineage, list) else []
        lineage_by_id = {
            item.get("scene_id"): item
            for item in lineage_items
            if isinstance(item, dict)
        }
        observed_image_scenes = {
            image.get("scene_id") for image in images if isinstance(image, dict)
        }
        if observed_image_scenes != scene_ids:
            _issue(
                issues,
                "COCO_SCENE_COVERAGE_INVALID",
                "COCO images must cover every and only manifested scene",
            )
        for image in images:
            if not isinstance(image, dict):
                continue
            lineage_item = lineage_by_id.get(image.get("scene_id"))
            frame = image.get("frame")
            lineage_start = (
                _as_number(lineage_item.get("start_seconds"))
                if isinstance(lineage_item, dict)
                else None
            )
            lineage_end = (
                _as_number(lineage_item.get("end_seconds"))
                if isinstance(lineage_item, dict)
                else None
            )
            if (
                image.get("width") != source_width
                or image.get("height") != source_height
                or lineage_item is None
                or source_fps is None
                or lineage_start is None
                or lineage_end is None
                or isinstance(frame, bool)
                or not isinstance(frame, int)
                or frame
                >= max(
                    1,
                    math.ceil((lineage_end - lineage_start) * source_fps),
                )
            ):
                _issue(
                    issues,
                    "COCO_IMAGE_LINEAGE_INVALID",
                    "A COCO image does not match source scene geometry",
                )
        annotation_counts: Counter[str] = Counter()
        for annotation in annotations:
            if not isinstance(annotation, dict):
                continue
            image = image_by_id.get(annotation.get("image_id"))
            if image is not None and isinstance(image.get("scene_id"), str):
                annotation_counts[str(image["scene_id"])] += 1
        for scene_id, lineage_item in lineage_by_id.items():
            if lineage_item.get("final_count") != annotation_counts.get(str(scene_id), 0):
                _issue(
                    issues,
                    "SCENE_FINAL_COUNT_MISMATCH",
                    "Scene final count does not match COCO annotations",
                )
    return coco


def _validate_yolo_semantics(
    root: Path,
    manifest: dict[str, Any],
    expected_payload: dict[str, str],
    coco: dict[str, Any] | None,
    issues: list[dict[str, str]],
) -> None:
    if manifest.get("formats") != ["COCO", "YOLO"]:
        _issue(issues, "RELEASE_FORMATS_INVALID", "V2 release must contain COCO and YOLO")
    if expected_payload.get(YOLO_METADATA_FILENAME) != manifest.get("yolo_metadata_sha256"):
        _issue(issues, "YOLO_METADATA_SHA256_MISMATCH", "YOLO metadata hash is inconsistent")
    metadata_path = root / YOLO_METADATA_FILENAME
    try:
        metadata = metadata_path.read_text(encoding="utf-8")
    except OSError:
        _issue(issues, "YOLO_METADATA_MISSING", "YOLO dataset metadata is missing")
        metadata = ""
    if metadata != _yolo_metadata():
        _issue(issues, "YOLO_METADATA_INVALID", "YOLO dataset metadata is invalid")
    if coco is None:
        return

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    if not isinstance(images, list) or not isinstance(annotations, list):
        return
    annotations_by_image: dict[int, list[dict[str, Any]]] = {
        int(image["id"]): []
        for image in images
        if isinstance(image, dict) and isinstance(image.get("id"), int)
    }
    for annotation in annotations:
        if not isinstance(annotation, dict) or annotation.get("image_id") not in annotations_by_image:
            continue
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    expected_label_files: set[str] = set()
    for image in images:
        if not isinstance(image, dict) or not isinstance(image.get("id"), int):
            continue
        label_name = f"{Path(str(image.get('file_name', ''))).stem}.txt"
        relative = f"{YOLO_LABELS_DIRECTORY}/{label_name}"
        expected_label_files.add(relative)
        label_path = root / relative
        try:
            actual = label_path.read_text(encoding="utf-8")
            expected = _yolo_label_text(image, annotations_by_image[int(image["id"])])
        except (OSError, KeyError, TypeError, ValueError):
            _issue(issues, "YOLO_LABEL_INVALID", f"YOLO label is unreadable: {relative}")
            continue
        if actual != expected:
            _issue(issues, "YOLO_LABEL_INVALID", f"YOLO label differs from COCO: {relative}")
    manifested_label_files = {
        name
        for name in expected_payload
        if name.startswith(f"{YOLO_LABELS_DIRECTORY}/") and name.endswith(".txt")
    }
    if manifested_label_files != expected_label_files:
        _issue(issues, "YOLO_LABEL_SET_INVALID", "YOLO label files do not match COCO images")
    if manifest.get("yolo_label_file_count") != len(expected_label_files):
        _issue(issues, "YOLO_LABEL_COUNT_MISMATCH", "YOLO label file count is invalid")


def _validate_quality_semantics(
    root: Path,
    manifest: dict[str, Any],
    expected_payload: dict[str, str],
    coco: dict[str, Any] | None,
    issues: list[dict[str, str]],
) -> None:
    strict_v2 = manifest.get("schema_version") == MANIFEST_SCHEMA_VERSION
    if QUALITY_FILENAME not in expected_payload:
        if strict_v2:
            _issue(
                issues,
                "QUALITY_REPORT_MISSING",
                "V2 release must freeze a quality report",
            )
        return
    quality_path = root / QUALITY_FILENAME
    try:
        report = json.loads(quality_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        _issue(issues, "QUALITY_REPORT_INVALID", "Quality report is not valid JSON")
        return
    if coco is None:
        _issue(
            issues,
            "QUALITY_REPORT_INVALID",
            "Quality report cannot be validated without a valid COCO export",
        )
        return
    annotations = coco.get("annotations")
    images = coco.get("images")
    if not isinstance(annotations, list) or not isinstance(images, list):
        _issue(
            issues,
            "QUALITY_REPORT_INVALID",
            "Quality report cannot be validated without COCO annotations",
        )
        return
    labels_by_category = {identifier: label for label, identifier in CATEGORY_IDS.items()}
    expected_distribution = dict(
        sorted(
            Counter(
                labels_by_category.get(annotation.get("category_id"), "__invalid__")
                for annotation in annotations
                if isinstance(annotation, dict)
            ).items()
        )
    )
    if not strict_v2:
        if not isinstance(report, dict):
            _issue(issues, "QUALITY_REPORT_INVALID", "Legacy quality report must be an object")
            return
        if (
            not _nonnegative_integer(report.get("prediction_count"))
            or not _nonnegative_integer(report.get("final_count"))
            or report.get("final_count") != len(annotations)
        ):
            _issue(
                issues,
                "QUALITY_REPORT_INVALID",
                "Legacy quality counts do not match the COCO annotations",
            )
        distribution = report.get("class_distribution")
        if distribution is not None and distribution != expected_distribution:
            _issue(
                issues,
                "QUALITY_REPORT_INVALID",
                "Legacy quality class distribution does not match COCO",
            )
        return

    quality_issues = _quality_validation_issues(
        report,
        expected_final_count=len(annotations),
        expected_class_distribution=expected_distribution,
    )
    for message in quality_issues:
        _issue(issues, "QUALITY_REPORT_INVALID", message)
    if manifest.get("quality_schema_version") != "1.0.0":
        _issue(
            issues,
            "QUALITY_REPORT_INVALID",
            "V2 quality schema version is missing or unsupported",
        )
    if manifest.get("quality_sha256") != expected_payload.get(QUALITY_FILENAME):
        _issue(issues, "QUALITY_SHA256_MISMATCH", "Quality hash is inconsistent")
    if manifest.get("predictions_sha256") != expected_payload.get(PREDICTIONS_FILENAME):
        _issue(issues, "PREDICTIONS_SHA256_MISMATCH", "Predictions hash is inconsistent")
    predictions_path = root / PREDICTIONS_FILENAME
    if PREDICTIONS_FILENAME not in expected_payload or not predictions_path.is_file():
        _issue(
            issues,
            "PREDICTIONS_MISSING",
            "V2 release must freeze the detector predictions",
        )
        return

    source = manifest.get("source_session")
    lineage = manifest.get("scene_lineage")
    if not isinstance(source, dict) or not isinstance(lineage, list):
        _issue(issues, "PREDICTIONS_INVALID", "Prediction lineage cannot be reconstructed")
        return
    try:
        manifest_session = Session(
            session_id=str(source["session_id"]),
            name="release-verification",
            source_path=str(source["source_path"]),
            source_sha256=str(source["source_sha256"]),
            duration_seconds=float(source["duration_seconds"]),
            fps=float(source["fps"]),
            width=int(source["width"]),
            height=int(source["height"]),
            demo=manifest.get("image_materialization") == "deterministic_placeholder",
            scenes=[
                Scene(
                    scene_id=str(item["scene_id"]),
                    session_id=str(item["session_id"]),
                    start_seconds=float(item["start_seconds"]),
                    end_seconds=float(item["end_seconds"]),
                    video_path="",
                )
                for item in lineage
                if isinstance(item, dict)
            ],
        )
        raw_predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        _issue(issues, "PREDICTIONS_INVALID", "Frozen predictions are unreadable")
        return
    canonical_predictions, prediction_error = normalise_release_predictions(
        manifest_session, raw_predictions
    )
    if prediction_error is not None or canonical_predictions != raw_predictions:
        _issue(
            issues,
            "PREDICTIONS_INVALID",
            "Frozen predictions are invalid or not canonically ordered",
        )
        return
    if manifest.get("prediction_count") != len(canonical_predictions):
        _issue(issues, "PREDICTION_COUNT_MISMATCH", "Prediction count is inconsistent")

    evaluated_frames = manifest.get("evaluated_frames")
    if not isinstance(evaluated_frames, list) or not all(
        isinstance(item, dict) and set(item) == {"scene_id", "frame"}
        for item in evaluated_frames
    ):
        _issue(issues, "EVALUATED_FRAMES_INVALID", "Evaluated-frame lineage is invalid")
        return
    frame_records, frame_error = normalise_evaluated_frames(
        manifest_session,
        [(item.get("scene_id"), item.get("frame")) for item in evaluated_frames],
    )
    if frame_error is not None or frame_records != evaluated_frames:
        _issue(
            issues,
            "EVALUATED_FRAMES_INVALID",
            "Evaluated-frame lineage is invalid or not canonical",
        )
        return
    quality_context = manifest.get("quality_context")
    if not isinstance(quality_context, dict) or set(quality_context) != {
        "accepted_jobs",
        "reviewed_jobs",
        "first_pass_acceptance_reason",
    }:
        _issue(issues, "QUALITY_CONTEXT_INVALID", "Quality context is invalid")
        return
    accepted_jobs = quality_context.get("accepted_jobs")
    reviewed_jobs = quality_context.get("reviewed_jobs")
    reason = quality_context.get("first_pass_acceptance_reason")
    if (
        not _nonnegative_integer(accepted_jobs)
        or not _nonnegative_integer(reviewed_jobs)
        or accepted_jobs > reviewed_jobs
        or (reason is not None and (not isinstance(reason, str) or not reason.strip()))
    ):
        _issue(issues, "QUALITY_CONTEXT_INVALID", "Quality context values are invalid")
        return

    image_by_id = {
        item.get("id"): item for item in images if isinstance(item, dict)
    }
    label_by_id = {identifier: label for label, identifier in CATEGORY_IDS.items()}
    final_annotations: list[dict[str, Any]] = []
    try:
        for annotation in annotations:
            image = image_by_id[annotation["image_id"]]
            x, y, width, height = (float(value) for value in annotation["bbox"])
            final_annotations.append(
                {
                    "scene_id": image["scene_id"],
                    "frame": image["frame"],
                    "label": label_by_id[annotation["category_id"]],
                    "bbox": [x, y, x + width, y + height],
                }
            )
    except (KeyError, TypeError, ValueError):
        _issue(issues, "QUALITY_REPORT_INVALID", "COCO final labels cannot be recomputed")
        return
    recalculated = calculate_quality(
        canonical_predictions,
        final_annotations,
        int(accepted_jobs),
        int(reviewed_jobs),
        evaluated_frame_keys={
            (str(item["scene_id"]), int(item["frame"])) for item in frame_records
        },
        first_pass_acceptance_reason=reason,
    ).data
    if report != recalculated:
        _issue(
            issues,
            "QUALITY_RECOMPUTATION_MISMATCH",
            "Quality report does not match the frozen predictions and final annotations",
        )


def _validate_manifest_semantics(
    root: Path,
    manifest: dict[str, Any],
    receipt: dict[str, Any],
    expected_payload: dict[str, str],
    issues: list[dict[str, str]],
    *,
    expected_session_id: str | None,
    expected_version: str | None,
    expected_source_sha256: str | None,
) -> None:
    schema_version = manifest.get("schema_version")
    if schema_version not in {"1.0.0", MANIFEST_SCHEMA_VERSION}:
        _issue(issues, "MANIFEST_SCHEMA_UNSUPPORTED", "Manifest schema version is unsupported")
    if manifest.get("status") != "final" or manifest.get("format") != "COCO":
        _issue(issues, "MANIFEST_STATE_INVALID", "Manifest must describe a final COCO release")
    if manifest.get("taxonomy") != list(ROAD_LABEL_TAXONOMY):
        _issue(issues, "TAXONOMY_MISMATCH", "Manifest taxonomy is not RoadLabelOps taxonomy")

    source = manifest.get("source_session")
    if not isinstance(source, dict) or not all(
        source.get(key) for key in ("session_id", "source_path", "source_sha256")
    ):
        _issue(issues, "SESSION_LINEAGE_INVALID", "Manifest source session lineage is incomplete")
        source = {}
    source_id = source.get("session_id")
    source_sha256 = source.get("source_sha256")
    if not isinstance(source_sha256, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", source_sha256):
        _issue(issues, "SESSION_LINEAGE_INVALID", "Manifest source SHA-256 is invalid")
    if expected_session_id is not None and source_id != expected_session_id:
        _issue(issues, "SESSION_ID_MISMATCH", "Release belongs to a different session")
    if expected_source_sha256 is not None and source_sha256 != expected_source_sha256:
        _issue(issues, "SOURCE_SHA256_MISMATCH", "Release source hash differs from the session")
    if schema_version == MANIFEST_SCHEMA_VERSION and source.get("source_path_kind") != "basename":
        _issue(issues, "SESSION_LINEAGE_INVALID", "V2 source path must be privacy-safe metadata")
    source_path = source.get("source_path")
    if schema_version == MANIFEST_SCHEMA_VERSION and (
        not isinstance(source_path, str)
        or not source_path
        or Path(source_path).name != source_path
        or "/" in source_path
        or "\\" in source_path
    ):
        _issue(issues, "SESSION_LINEAGE_INVALID", "V2 source path must be one basename")
    if schema_version == MANIFEST_SCHEMA_VERSION and (
        not _positive_finite_manifest_value(source.get("duration_seconds"))
        or not _positive_finite_manifest_value(source.get("fps"))
        or not _nonnegative_integer(source.get("width"))
        or not _nonnegative_integer(source.get("height"))
        or source.get("width") == 0
        or source.get("height") == 0
    ):
        _issue(
            issues,
            "SESSION_LINEAGE_INVALID",
            "V2 source geometry and timing metadata are invalid",
        )
    if schema_version == MANIFEST_SCHEMA_VERSION and manifest.get(
        "image_materialization"
    ) not in {"source_frame", "deterministic_placeholder"}:
        _issue(
            issues,
            "IMAGE_MATERIALIZATION_INVALID",
            "V2 release must declare its image materialization mode",
        )
    if manifest.get("session_ids") != [source_id]:
        _issue(issues, "SESSION_LINEAGE_INVALID", "Manifest session ids do not match source lineage")

    version = manifest.get("version")
    if schema_version == MANIFEST_SCHEMA_VERSION and (
        not isinstance(version, str) or not _valid_version(version)
    ):
        _issue(issues, "RELEASE_VERSION_MISSING", "Manifest release version is invalid")
    if (
        schema_version == MANIFEST_SCHEMA_VERSION
        and isinstance(source_id, str)
        and isinstance(version, str)
        and manifest.get("release_id") != f"{source_id}-v{version}"
    ):
        _issue(issues, "RELEASE_ID_MISMATCH", "Release id is inconsistent with session and version")
    if expected_version is not None and version != expected_version:
        _issue(issues, "RELEASE_VERSION_MISMATCH", "Manifest version does not match the request")

    scene_ids = manifest.get("scene_ids")
    if (
        not isinstance(scene_ids, list)
        or not scene_ids
        or not all(isinstance(item, str) and _safe_path_component(item) for item in scene_ids)
        or len(scene_ids) != len(set(scene_ids))
    ):
        _issue(issues, "SCENE_LINEAGE_INVALID", "Manifest scene ids are invalid or duplicated")
        scene_ids = []
    lineage = manifest.get("scene_lineage")
    if schema_version == MANIFEST_SCHEMA_VERSION:
        if not isinstance(lineage, list) or not all(isinstance(item, dict) for item in lineage):
            _issue(issues, "SCENE_LINEAGE_INVALID", "Manifest scene lineage must be a list")
            lineage = []
        lineage_ids = [item.get("scene_id") for item in lineage]
        if (
            not all(isinstance(item, str) for item in lineage_ids)
            or len(lineage_ids) != len(set(lineage_ids))
            or lineage_ids != sorted(scene_ids)
        ):
            _issue(issues, "SCENE_LINEAGE_INVALID", "Scene lineage does not match scene ids")
        scene_lineage_fields = {
            "scene_id",
            "session_id",
            "start_seconds",
            "end_seconds",
            "cvat_task_id",
            "cvat_job_ids",
            "video_sha256",
            "final_count",
            "scene_tags",
        }
        previous_end: float | None = None
        observed_task_ids: set[int] = set()
        observed_job_ids: set[int] = set()
        for item in lineage:
            start = _as_number(item.get("start_seconds"))
            end = _as_number(item.get("end_seconds"))
            if (
                set(item) != scene_lineage_fields
                or item.get("session_id") != source_id
                or start is None
                or end is None
                or start < 0
                or end <= start
                or (
                    _as_number(source.get("duration_seconds")) is not None
                    and end > float(source["duration_seconds"]) + 1e-6
                )
                or (previous_end is not None and start < previous_end - 1e-6)
            ):
                _issue(issues, "SCENE_LINEAGE_INVALID", "A scene lineage record is invalid")
            if end is not None:
                previous_end = end
            digest = item.get("video_sha256")
            real_materialization = manifest.get("image_materialization") == "source_frame"
            if (real_materialization and not isinstance(digest, str)) or (
                digest is not None
                and (
                    not isinstance(digest, str)
                    or not re.fullmatch(r"[0-9a-f]{64}", digest)
                )
            ):
                _issue(issues, "SCENE_LINEAGE_INVALID", "A scene video SHA-256 is invalid")
            task_id = item.get("cvat_task_id")
            job_ids = item.get("cvat_job_ids")
            task_id_valid = task_id is None or (
                not isinstance(task_id, bool)
                and isinstance(task_id, int)
                and task_id > 0
            )
            job_ids_valid = (
                isinstance(job_ids, list)
                and all(
                    not isinstance(job_id, bool)
                    and isinstance(job_id, int)
                    and job_id > 0
                    for job_id in job_ids
                )
                and len(job_ids) == len(set(job_ids))
            )
            if (
                not task_id_valid
                or not job_ids_valid
                or (task_id is None and bool(job_ids))
                or (real_materialization and (task_id is None or not job_ids))
            ):
                _issue(issues, "CVAT_LINEAGE_INVALID", "A scene has invalid CVAT lineage")
            if isinstance(task_id, int) and not isinstance(task_id, bool) and task_id > 0:
                if task_id in observed_task_ids:
                    _issue(issues, "CVAT_LINEAGE_INVALID", "CVAT Task ids must be unique")
                observed_task_ids.add(task_id)
            if isinstance(job_ids, list):
                for job_id in job_ids:
                    if not isinstance(job_id, int) or isinstance(job_id, bool) or job_id <= 0:
                        continue
                    if job_id in observed_job_ids:
                        _issue(issues, "CVAT_LINEAGE_INVALID", "CVAT Job ids must be unique")
                    observed_job_ids.add(job_id)
            if not _nonnegative_integer(item.get("final_count")):
                _issue(issues, "SCENE_LINEAGE_INVALID", "A scene final count is invalid")
            scene_tags = item.get("scene_tags", [])
            if not isinstance(scene_tags, list):
                _issue(issues, "SCENE_TAGS_INVALID", "Scene tags must be a list")
                continue
            for tag in scene_tags:
                frame_limit = (
                    max(1, math.ceil((float(end) - float(start)) * float(source["fps"])))
                    if start is not None
                    and end is not None
                    and _as_number(source.get("fps")) is not None
                    else None
                )
                if (
                    not isinstance(tag, dict)
                    or set(tag) != {"frame", "label", "value", "source"}
                    or tag.get("label") not in ROAD_SCENE_TAG_VALUES
                    or tag.get("value")
                    not in ROAD_SCENE_TAG_VALUES.get(str(tag.get("label")), ())
                    or isinstance(tag.get("frame"), bool)
                    or not isinstance(tag.get("frame"), int)
                    or tag.get("frame", -1) < 0
                    or (frame_limit is not None and tag.get("frame", 0) >= frame_limit)
                    or not isinstance(tag.get("source"), str)
                    or not str(tag.get("source", "")).strip()
                ):
                    _issue(issues, "SCENE_TAGS_INVALID", "A scene tag is invalid")
        expected_task_ids = [
            item.get("cvat_task_id") for item in lineage if item.get("cvat_task_id") is not None
        ]
        if manifest.get("cvat_task_ids") != expected_task_ids:
            _issue(issues, "CVAT_LINEAGE_INVALID", "Manifest CVAT task ids are inconsistent")

    if manifest.get("export_sha256") != expected_payload.get(ANNOTATIONS_FILENAME):
        _issue(issues, "EXPORT_SHA256_MISMATCH", "Manifest export hash is inconsistent")
    quality_sha256 = manifest.get("quality_sha256")
    if (QUALITY_FILENAME in expected_payload) != (quality_sha256 is not None) or (
        quality_sha256 is not None
        and quality_sha256 != expected_payload.get(QUALITY_FILENAME)
    ):
        _issue(issues, "QUALITY_SHA256_MISMATCH", "Manifest quality hash is inconsistent")
    if receipt:
        if receipt.get("schema_version") not in {"1.0.0", RECEIPT_SCHEMA_VERSION}:
            _issue(issues, "RECEIPT_SCHEMA_UNSUPPORTED", "Receipt schema version is unsupported")
        if receipt.get("receipt_type") != "roadlabelops.release.integrity":
            _issue(issues, "RECEIPT_TYPE_INVALID", "Release receipt type is invalid")
        if receipt.get("payload_file_count") != len(expected_payload):
            _issue(issues, "RECEIPT_FILE_COUNT_MISMATCH", "Receipt payload file count is invalid")

    coco = _read_coco_semantics(root, manifest, expected_payload, issues)
    _validate_quality_semantics(root, manifest, expected_payload, coco, issues)
    if schema_version == MANIFEST_SCHEMA_VERSION:
        _validate_yolo_semantics(root, manifest, expected_payload, coco, issues)


def verify_coco_release(
    release_path: Path | str,
    *,
    expected_release_id: str | None = None,
    expected_manifest_sha256: str | None = None,
    expected_receipt_sha256: str | None = None,
    expected_session_id: str | None = None,
    expected_version: str | None = None,
    expected_source_sha256: str | None = None,
) -> ToolResult:
    """Verify payload hashes, the manifest receipt, and the exact release file set.

    ``data['receipt']`` is returned for both successes and failures so a dashboard
    can render a complete integrity state without parsing exception text.
    """

    root = Path(release_path)
    if root.is_symlink():
        return _verification_result(
            False,
            root,
            expected_release_id,
            [
                {
                    "code": "RELEASE_SYMLINK_FORBIDDEN",
                    "message": "Release root must not be a symbolic link",
                }
            ],
        )
    if not root.is_dir():
        return _verification_result(
            False,
            root,
            expected_release_id,
            [{"code": "RELEASE_NOT_FOUND", "message": "Release directory does not exist"}],
        )
    for path in root.rglob("*"):
        if path.is_symlink():
            return _verification_result(
                False,
                root,
                expected_release_id,
                [
                    {
                        "code": "RELEASE_SYMLINK_FORBIDDEN",
                        "message": (
                            "Release contains a symbolic link: "
                            f"{path.relative_to(root)}"
                        ),
                    }
                ],
            )
    issues: list[dict[str, str]] = []
    manifest_path = root / MANIFEST_FILENAME
    receipt_path = root / RECEIPT_FILENAME
    manifest: dict[str, Any] = {}
    receipt_document: dict[str, Any] = {}
    for path, destination, code in (
        (manifest_path, "manifest", "MANIFEST_MISSING"),
        (receipt_path, "receipt", "RECEIPT_MISSING"),
    ):
        if not path.is_file():
            issues.append({"code": code, "message": f"Required {destination} is missing"})
            continue
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            issues.append(
                {
                    "code": f"{destination.upper()}_INVALID",
                    "message": f"{destination} is not valid JSON",
                }
            )
            continue
        if not isinstance(parsed, dict):
            issues.append(
                {
                    "code": f"{destination.upper()}_INVALID",
                    "message": f"{destination} must be an object",
                }
            )
            continue
        if destination == "manifest":
            manifest = parsed
        else:
            receipt_document = parsed
    if (
        expected_manifest_sha256 is not None
        and manifest_path.is_file()
        and _digest(manifest_path) != expected_manifest_sha256
    ):
        issues.append(
            {
                "code": "MANIFEST_SHA256_MISMATCH",
                "message": "Manifest hash does not match the trusted release record",
            }
        )
    if (
        expected_receipt_sha256 is not None
        and receipt_path.is_file()
        and _digest(receipt_path) != expected_receipt_sha256
    ):
        issues.append(
            {
                "code": "RECEIPT_SHA256_MISMATCH",
                "message": "Receipt hash does not match the trusted release record",
            }
        )
    release_id = (
        manifest.get("release_id")
        if isinstance(manifest.get("release_id"), str)
        else expected_release_id
    )
    if not isinstance(release_id, str) or not _safe_release_name(release_id):
        issues.append(
            {"code": "RELEASE_ID_INVALID", "message": "Manifest release id is invalid"}
        )
    if expected_release_id is not None and release_id != expected_release_id:
        issues.append(
            {"code": "RELEASE_ID_MISMATCH", "message": "Release id does not match the requested id"}
        )
    if expected_release_id is None and release_id != root.name:
        issues.append(
            {
                "code": "RELEASE_ID_MISMATCH",
                "message": "Release id does not match the immutable directory name",
            }
        )
    expected_payload = manifest.get("payload_sha256") if manifest else None
    if not isinstance(expected_payload, dict) or not all(
        isinstance(name, str)
        and bool(name)
        and not Path(name).is_absolute()
        and ".." not in Path(name).parts
        and "\\" not in name
        and name not in {MANIFEST_FILENAME, RECEIPT_FILENAME}
        and isinstance(digest, str)
        and bool(re.fullmatch(r"[0-9a-f]{64}", digest))
        for name, digest in (expected_payload or {}).items()
    ):
        issues.append(
            {
                "code": "PAYLOAD_DIGESTS_INVALID",
                "message": "Manifest payload SHA-256 map is invalid",
            }
        )
        expected_payload = {}
    actual_payload = _payload_digests(root)
    expected_files = set(expected_payload) | {MANIFEST_FILENAME, RECEIPT_FILENAME}
    actual_files = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    for name in sorted(expected_files - actual_files):
        issues.append(
            {"code": "RELEASE_FILE_MISSING", "message": f"Release file is missing: {name}"}
        )
    for name in sorted(actual_files - expected_files):
        issues.append(
            {"code": "RELEASE_FILE_EXTRA", "message": f"Release has an unmanifested file: {name}"}
        )
    for name in sorted(set(expected_payload) & set(actual_payload)):
        if actual_payload[name] != expected_payload[name]:
            issues.append(
                {"code": "PAYLOAD_SHA256_MISMATCH", "message": f"Payload hash mismatch: {name}"}
            )

    if receipt_document:
        if receipt_document.get("release_id") != release_id:
            issues.append(
                {
                    "code": "RECEIPT_RELEASE_ID_MISMATCH",
                    "message": "Receipt release id does not match manifest",
                }
            )
        manifest_digest = _digest(manifest_path) if manifest_path.is_file() else None
        if receipt_document.get("manifest_sha256") != manifest_digest:
            issues.append(
                {"code": "MANIFEST_SHA256_MISMATCH", "message": "Manifest does not match receipt"}
            )
        if receipt_document.get("payload_sha256") != expected_payload:
            issues.append(
                {
                    "code": "RECEIPT_PAYLOAD_MISMATCH",
                    "message": "Receipt payload map does not match manifest",
                }
            )
        if receipt_document.get("payload_tree_sha256") != _canonical_digest(expected_payload):
            issues.append(
                {
                    "code": "RECEIPT_TREE_SHA256_MISMATCH",
                    "message": "Receipt payload tree hash is invalid",
                }
            )
        if receipt_document.get("manifest_file") != MANIFEST_FILENAME:
            issues.append(
                {
                    "code": "RECEIPT_MANIFEST_FILE_INVALID",
                    "message": "Receipt must bind the canonical manifest filename",
                }
            )
        receipt_schema = receipt_document.get("schema_version")
        canonical_receipt = {
            "schema_version": receipt_schema,
            "receipt_type": "roadlabelops.release.integrity",
            "release_id": release_id,
            "manifest_file": MANIFEST_FILENAME,
            "manifest_sha256": _digest(manifest_path) if manifest_path.is_file() else None,
            "payload_tree_sha256": _canonical_digest(expected_payload),
            "payload_file_count": len(expected_payload),
            "payload_sha256": expected_payload,
        }
        if receipt_document != canonical_receipt:
            issues.append(
                {
                    "code": "RECEIPT_DOCUMENT_INVALID",
                    "message": "Receipt differs from the canonical integrity document",
                }
            )
    if manifest:
        _validate_manifest_semantics(
            root,
            manifest,
            receipt_document,
            expected_payload,
            issues,
            expected_session_id=expected_session_id,
            expected_version=expected_version,
            expected_source_sha256=expected_source_sha256,
        )
    return _verification_result(not issues, root, release_id, issues)


# A short generic name is convenient for tool registries and dashboard callers.
verify_release = verify_coco_release
