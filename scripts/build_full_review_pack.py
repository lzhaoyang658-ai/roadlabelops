"""Build a read-only, full-frame annotation review pack from a task snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

COLORS = {
    "car": "#22c55e",
    "bus": "#f59e0b",
    "truck": "#ef4444",
    "motorcycle": "#3b82f6",
    "bicycle": "#8b5cf6",
    "pedestrian": "#ec4899",
    "traffic_light": "#06b6d4",
    "traffic_sign": "#a855f7",
}
MANUAL_CHECK_LABELS = ("traffic_light", "traffic_sign")
AUTOMATED_FLAG_TYPES = {
    "same_class_duplicate",
    "cross_class_conflict",
    "rider_pedestrian",
    "degenerate_box",
    "out_of_bounds_box",
}
RIDER_MIN_HEIGHT_RATIO = 0.50


def sha256(path: Path) -> str:
    """Return the SHA-256 of a file without modifying it."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    """Calculate IoU for two ``[x1, y1, x2, y2]`` rectangles."""

    if len(first) != 4 or len(second) != 4:
        raise ValueError("IoU requires two four-coordinate xyxy boxes")
    ax1, ay1, ax2, ay2 = (float(value) for value in first)
    bx1, by1, bx2, by2 = (float(value) for value in second)
    if not all(math.isfinite(value) for value in (ax1, ay1, ax2, ay2, bx1, by1, bx2, by2)):
        return 0.0
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _box_from_mapping(item: dict[str, Any]) -> list[float]:
    for key in ("bbox_xyxy", "bbox", "points"):
        value = item.get(key)
        if value is not None:
            if not isinstance(value, (list, tuple)) or len(value) != 4:
                raise ValueError(f"{key} must contain exactly four coordinates")
            return [float(coordinate) for coordinate in value]
    raise ValueError("Detection is missing bbox_xyxy, bbox, or points")


def class_aware_nms(
    detections: Sequence[dict[str, Any]],
    iou_threshold: float = 0.50,
) -> list[dict[str, Any]]:
    """Apply NMS independently per frame and class.

    The helper deliberately never suppresses a box because it overlaps a box of a
    different class. It is provided for optional model-candidate processing; the
    snapshot annotations themselves are never passed through NMS.
    """

    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be between 0 and 1")

    groups: dict[tuple[str, int | None, str], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, detection in enumerate(detections):
        label = str(detection.get("label", ""))
        if not label:
            raise ValueError("Every detection must have a label")
        frame_value = detection.get("frame")
        frame = int(frame_value) if frame_value is not None else None
        scene = str(detection.get("scene_id", ""))
        _box_from_mapping(detection)
        groups[(scene, frame, label)].append((index, detection))

    kept_indices: set[int] = set()
    for candidates in groups.values():
        accepted: list[tuple[int, dict[str, Any]]] = []
        ranked = sorted(
            candidates,
            key=lambda item: (
                -float(item[1].get("confidence", item[1].get("score", 0.0)) or 0.0),
                item[0],
            ),
        )
        for candidate in ranked:
            candidate_box = _box_from_mapping(candidate[1])
            if any(
                box_iou(candidate_box, _box_from_mapping(previous[1])) >= iou_threshold
                for previous in accepted
            ):
                continue
            accepted.append(candidate)
            kept_indices.add(candidate[0])

    return [dict(item) for index, item in enumerate(detections) if index in kept_indices]


def pedestrian_is_rider(
    pedestrian: Sequence[float],
    motorcycle: Sequence[float],
    overlap_threshold: float = 0.25,
    min_height_ratio: float = RIDER_MIN_HEIGHT_RATIO,
) -> bool:
    """Return whether a pedestrian box is geometrically attached to a motorcycle.

    A rider and their motorcycle must also have compatible image scales.  The
    default remains deliberately permissive: the motorcycle box may be up to
    twice the pedestrian box's height.
    """

    px1, py1, px2, py2 = (float(value) for value in pedestrian)
    mx1, my1, mx2, my2 = (float(value) for value in motorcycle)
    if not all(math.isfinite(value) for value in (px1, py1, px2, py2, mx1, my1, mx2, my2)):
        return False
    pedestrian_height = max(0.0, py2 - py1)
    motorcycle_height = max(0.0, my2 - my1)
    if motorcycle_height <= 0.0:
        return False
    intersection_width = max(0.0, min(px2, mx2) - max(px1, mx1))
    intersection_height = max(0.0, min(py2, my2) - max(py1, my1))
    pedestrian_area = max(0.0, px2 - px1) * pedestrian_height
    overlap = (
        intersection_width * intersection_height / pedestrian_area if pedestrian_area > 0 else 0.0
    )
    bottom_center_x = (px1 + px2) / 2
    return (
        pedestrian_height / motorcycle_height >= min_height_ratio
        and overlap >= overlap_threshold
        and mx1 <= bottom_center_x <= mx2
        and my1 <= py2 <= my2
    )


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return normalized or "flag"


def stable_flag_id(
    frame: int,
    flag_type: str,
    *,
    shape_refs: Iterable[str] = (),
    label: str | None = None,
) -> str:
    """Create a deterministic ID from semantic flag identity, never list position."""

    identity = {
        "frame": int(frame),
        "type": flag_type,
        "shape_refs": sorted(str(value) for value in shape_refs),
        "label": label,
    }
    canonical = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    return f"frame-{int(frame):06d}-{_slug(flag_type)}-{digest}"


def _label_map(snapshot: dict[str, Any]) -> dict[str, str]:
    labels = snapshot.get("labels", [])
    if isinstance(labels, dict):
        return {str(identifier): str(name) for identifier, name in labels.items()}
    if isinstance(labels, list):
        return {
            str(item["id"]): str(item.get("name", item.get("label")))
            for item in labels
            if isinstance(item, dict)
            and item.get("id") is not None
            and (item.get("name") is not None or item.get("label") is not None)
        }
    return {}


def _shape_signature(shape: dict[str, Any]) -> str:
    payload = {key: value for key, value in shape.items() if not key.startswith("_")}
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalize_shapes(
    raw_shapes: Sequence[dict[str, Any]], label_by_id: dict[str, str] | None = None
) -> list[dict[str, Any]]:
    """Normalize historical snapshot rectangle spellings without mutating input data."""

    label_by_id = label_by_id or {}
    normalized: list[dict[str, Any]] = []
    for raw in raw_shapes:
        if not isinstance(raw, dict):
            raise TypeError("Every snapshot shape must be an object")
        shape = dict(raw)
        shape["frame"] = int(shape.get("frame", 0))
        label = shape.get("label", shape.get("label_name"))
        if label is None and shape.get("label_id") is not None:
            label = label_by_id.get(str(shape["label_id"]))
        if label is None:
            raise ValueError(f"Shape {shape.get('id', '<unknown>')} is missing a label")
        shape["label"] = str(label)
        shape["points"] = _box_from_mapping(shape)
        normalized.append(shape)

    # Anonymous/repeated identifiers get deterministic occurrence suffixes after a
    # canonical sort. Snapshot-provided unique IDs remain unchanged in JSON output.
    sorted_shapes = sorted(
        normalized,
        key=lambda item: (
            str(item.get("id", "")),
            item["label"],
            tuple(item["points"]),
            _shape_signature(item),
        ),
    )
    reference_counts: Counter[str] = Counter()
    for shape in sorted_shapes:
        if shape.get("id") is None:
            signature = _shape_signature(shape)
            base = f"anon-{hashlib.sha256(signature.encode('utf-8')).hexdigest()[:12]}"
        else:
            base = str(shape["id"])
        reference_counts[base] += 1
        occurrence = reference_counts[base]
        shape["_shape_ref"] = base if occurrence == 1 else f"{base}#{occurrence}"
    return sorted(sorted_shapes, key=lambda item: item["_shape_ref"])


def _shape_ids(shapes: Sequence[dict[str, Any]]) -> list[Any]:
    return [shape.get("id", shape["_shape_ref"]) for shape in shapes]


def _flag(
    frame: int,
    flag_type: str,
    message: str,
    *,
    shapes: Sequence[dict[str, Any]] = (),
    label: str | None = None,
    **details: Any,
) -> dict[str, Any]:
    shape_refs = [str(shape["_shape_ref"]) for shape in shapes]
    identifier = stable_flag_id(frame, flag_type, shape_refs=shape_refs, label=label)
    payload: dict[str, Any] = {
        "id": identifier,
        "flag_id": identifier,
        "type": flag_type,
        "frame": int(frame),
        "shape_ids": _shape_ids(shapes),
        "requires_human_review": True,
        "status": "open",
        "message": message,
    }
    if label is not None:
        payload["label"] = label
    payload.update(details)
    return payload


def build_frame_flags(
    frame: int,
    shapes: Sequence[dict[str, Any]],
    width: int,
    height: int,
    *,
    duplicate_iou_threshold: float = 0.50,
    conflict_iou_threshold: float = 0.50,
    rider_overlap_threshold: float = 0.25,
) -> list[dict[str, Any]]:
    """Detect existing-box risks and add mandatory per-frame manual checks."""

    if width <= 0 or height <= 0:
        raise ValueError("Frame width and height must be positive")
    if not 0.0 <= duplicate_iou_threshold <= 1.0:
        raise ValueError("duplicate_iou_threshold must be between 0 and 1")
    if not 0.0 <= conflict_iou_threshold <= 1.0:
        raise ValueError("conflict_iou_threshold must be between 0 and 1")
    if not 0.0 <= rider_overlap_threshold <= 1.0:
        raise ValueError("rider_overlap_threshold must be between 0 and 1")

    ordered = sorted(shapes, key=lambda item: str(item["_shape_ref"]))
    flags: list[dict[str, Any]] = []
    for shape in ordered:
        points = shape["points"]
        finite = all(math.isfinite(float(value)) for value in points)
        x1, y1, x2, y2 = points
        if not finite or x2 <= x1 or y2 <= y1:
            flags.append(
                _flag(
                    frame,
                    "degenerate_box",
                    "Box has non-finite coordinates or non-positive width/height.",
                    shapes=[shape],
                    label=shape["label"],
                    points=points,
                )
            )
        if finite and (x1 < 0 or y1 < 0 or x2 > width or y2 > height):
            flags.append(
                _flag(
                    frame,
                    "out_of_bounds_box",
                    "Box extends outside the declared frame bounds.",
                    shapes=[shape],
                    label=shape["label"],
                    points=points,
                    frame_bounds=[0, 0, width, height],
                )
            )

    for first_index, first in enumerate(ordered):
        for second in ordered[first_index + 1 :]:
            iou = box_iou(first["points"], second["points"])
            if first["label"] == second["label"] and iou >= duplicate_iou_threshold:
                flags.append(
                    _flag(
                        frame,
                        "same_class_duplicate",
                        "Same-class boxes overlap at or above the duplicate threshold.",
                        shapes=[first, second],
                        label=first["label"],
                        iou=round(iou, 6),
                        threshold=duplicate_iou_threshold,
                    )
                )
            elif first["label"] != second["label"] and iou >= conflict_iou_threshold:
                flags.append(
                    _flag(
                        frame,
                        "cross_class_conflict",
                        "Different-class boxes overlap at or above the conflict threshold.",
                        shapes=[first, second],
                        labels=sorted([first["label"], second["label"]]),
                        iou=round(iou, 6),
                        threshold=conflict_iou_threshold,
                    )
                )

    pedestrians = [shape for shape in ordered if shape["label"] == "pedestrian"]
    motorcycles = [shape for shape in ordered if shape["label"] == "motorcycle"]
    for pedestrian in pedestrians:
        for motorcycle in motorcycles:
            if pedestrian_is_rider(
                pedestrian["points"], motorcycle["points"], rider_overlap_threshold
            ):
                flags.append(
                    _flag(
                        frame,
                        "rider_pedestrian",
                        "Pedestrian appears attached to a motorcycle rider and may be duplicated.",
                        shapes=[pedestrian, motorcycle],
                        label="pedestrian",
                        rider_vehicle_label="motorcycle",
                        overlap_threshold=rider_overlap_threshold,
                    )
                )

    for label in MANUAL_CHECK_LABELS:
        flags.append(
            _flag(
                frame,
                "manual_class_check",
                f"Inspect the entire frame for missing or incorrect {label} annotations.",
                label=label,
            )
        )
    return sorted(flags, key=lambda item: item["id"])


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _display_box(
    points: Sequence[float], width: int, height: int
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = (float(value) for value in points)
    values = (x1, y1, x2, y2)
    if not all(math.isfinite(value) for value in values):
        return (0.0, 0.0, 0.0, 0.0)
    left, right = sorted((max(0.0, min(float(width - 1), x1)), max(0.0, min(float(width - 1), x2))))
    top, bottom = sorted(
        (max(0.0, min(float(height - 1), y1)), max(0.0, min(float(height - 1), y2)))
    )
    return left, top, right, bottom


def make_overlay(
    image_path: Path,
    shapes: Sequence[dict[str, Any]],
    flags: Sequence[dict[str, Any]],
    output_path: Path,
) -> Path:
    """Render every existing box and highlight boxes referenced by risk flags."""

    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    label_font = _font(18)
    banner_font = _font(17)
    flagged_ids = {
        str(shape_id)
        for flag in flags
        if flag["type"] in AUTOMATED_FLAG_TYPES
        for shape_id in flag["shape_ids"]
    }
    for shape in shapes:
        shape_id = shape.get("id", shape["_shape_ref"])
        label = shape["label"]
        points = _display_box(shape["points"], image.width, image.height)
        color = "#f43f5e" if str(shape_id) in flagged_ids else COLORS.get(label, "#ffffff")
        line_width = 5 if str(shape_id) in flagged_ids else 3
        draw.rectangle(points, outline=color, width=line_width)
        caption = f"{shape_id} {label}"
        bounds = draw.textbbox((points[0], points[1]), caption, font=label_font, stroke_width=1)
        text_width = max(1, bounds[2] - bounds[0])
        text_height = max(1, bounds[3] - bounds[1]) + 6
        text_x = min(max(0.0, points[0]), max(0.0, image.width - text_width - 8))
        text_y = max(0.0, points[1] - text_height)
        draw.rectangle((text_x, text_y, text_x + text_width + 8, text_y + text_height), fill=color)
        draw.text(
            (text_x + 4, text_y + 2),
            caption,
            fill="white",
            font=label_font,
            stroke_width=1,
        )

    automated_count = sum(flag["type"] in AUTOMATED_FLAG_TYPES for flag in flags)
    banner = (
        f"FULL REVIEW · {len(shapes)} boxes · {automated_count} risk flags · "
        "CHECK traffic_light + traffic_sign"
    )
    banner_height = 30
    draw.rectangle((0, 0, image.width, banner_height), fill="#111827")
    draw.text((8, 5), banner, fill="white", font=banner_font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="JPEG", quality=92)
    return output_path


def make_contact_sheets(
    frames: Sequence[dict[str, Any]], overlay_paths: Sequence[Path], output_dir: Path
) -> list[Path]:
    """Create 2x3 sheets that cover all overlay frames without stretching them."""

    if len(frames) != len(overlay_paths):
        raise ValueError("Every frame must have exactly one overlay before contact-sheet rendering")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    title_font = _font(18)
    tile_width, image_height, header_height = 640, 360, 40
    per_sheet = 6
    for start in range(0, len(frames), per_sheet):
        subset = frames[start : start + per_sheet]
        subset_paths = overlay_paths[start : start + per_sheet]
        sheet = Image.new("RGB", (tile_width * 2, (image_height + header_height) * 3), "#111827")
        draw = ImageDraw.Draw(sheet)
        for local_index, (frame, overlay_path) in enumerate(zip(subset, subset_paths)):
            column, row = local_index % 2, local_index // 2
            x = column * tile_width
            y = row * (image_height + header_height)
            automated = sum(flag["type"] in AUTOMATED_FLAG_TYPES for flag in frame["flags"])
            title = (
                f"FRAME {frame['frame']:06d} · SAMPLE {frame['sample_index']:03d} · "
                f"{frame['annotation_count']} boxes · {automated} risks"
            )
            draw.text((x + 10, y + 8), title, fill="white", font=title_font)
            with Image.open(overlay_path) as source:
                tile = ImageOps.contain(source.convert("RGB"), (tile_width, image_height))
            paste_x = x + (tile_width - tile.width) // 2
            paste_y = y + header_height + (image_height - tile.height) // 2
            sheet.paste(tile, (paste_x, paste_y))
        output = output_dir / f"contact-sheet-{len(paths) + 1:03d}.jpg"
        sheet.save(output, format="JPEG", quality=92)
        paths.append(output)
    return paths


def write_json_atomic(path: Path, payload: Any) -> None:
    """Durably replace a JSON file using a temporary file in the same directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _resolve_image_path(
    image: dict[str, Any], snapshot_dir: Path, manifest_dir: Path | None = None
) -> Path:
    candidates: list[Path] = []
    for path_key in ("path", "relative_path"):
        if image.get(path_key):
            candidate = Path(str(image[path_key]))
            if candidate.is_absolute():
                candidates.append(candidate)
            else:
                candidates.append(snapshot_dir / candidate)
                if manifest_dir is not None:
                    candidates.append(manifest_dir / candidate)
    if image.get("file_name"):
        file_name = Path(str(image["file_name"]))
        candidates.extend((snapshot_dir / file_name, snapshot_dir / "images" / file_name))
        if manifest_dir is not None:
            candidates.extend((manifest_dir / file_name, manifest_dir / "images" / file_name))
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    shown = ", ".join(str(candidate) for candidate in candidates) or "<no image path>"
    raise FileNotFoundError(f"Could not resolve snapshot image: {shown}")


def _public_shape(shape: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in shape.items() if not key.startswith("_")}


def build_review_pack(
    snapshot_path: Path | str,
    output_dir: Path | str,
    *,
    duplicate_iou_threshold: float = 0.50,
    conflict_iou_threshold: float = 0.50,
    rider_overlap_threshold: float = 0.25,
) -> Path:
    """Build the pack and return its JSON path; never mutate the source snapshot/task."""

    snapshot_path = Path(snapshot_path).resolve()
    output_dir = Path(output_dir).resolve()
    if not snapshot_path.is_file():
        raise FileNotFoundError(f"Snapshot does not exist: {snapshot_path}")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output path: {output_dir}")

    snapshot_bytes = snapshot_path.read_bytes()
    try:
        snapshot = json.loads(snapshot_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Snapshot is not valid JSON: {exc}") from exc
    if not isinstance(snapshot, dict):
        raise TypeError("Snapshot root must be a JSON object")
    images = snapshot.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError("Snapshot must contain a non-empty images list")
    if "shapes" in snapshot:
        raw_shapes = snapshot["shapes"]
    elif isinstance(snapshot.get("annotations"), dict):
        raw_shapes = snapshot["annotations"].get("shapes", [])
    else:
        raw_shapes = snapshot.get("annotations", [])
    if not isinstance(raw_shapes, list):
        raise TypeError("Snapshot shapes must be a list")
    shapes = normalize_shapes(raw_shapes, _label_map(snapshot))
    shapes_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for shape in shapes:
        shapes_by_frame[int(shape["frame"])].append(shape)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(exist_ok=False)
    try:
        manifest_dir: Path | None = None
        manifest_record = snapshot.get("manifest")
        if isinstance(manifest_record, dict) and manifest_record.get("path"):
            manifest_dir = Path(str(manifest_record["path"])).resolve().parent
        normalized_images: list[dict[str, Any]] = []
        seen_frames: set[int] = set()
        for position, raw_image in enumerate(images):
            if not isinstance(raw_image, dict):
                raise TypeError("Every snapshot image must be an object")
            image = dict(raw_image)
            frame = int(
                image.get(
                    "frame",
                    image.get("cvat_frame", image.get("source_frame", position)),
                )
            )
            if frame in seen_frames:
                raise ValueError(f"Snapshot contains duplicate image frame {frame}")
            seen_frames.add(frame)
            image_path = _resolve_image_path(image, snapshot_path.parent, manifest_dir)
            with Image.open(image_path) as source:
                actual_width, actual_height = source.size
            declared_width = image.get("width")
            declared_height = image.get("height")
            width = int(declared_width or actual_width)
            height = int(declared_height or actual_height)
            if width <= 0 or height <= 0:
                raise ValueError(f"Image frame {frame} has invalid dimensions")
            if declared_width is not None and width != actual_width:
                raise ValueError(
                    f"Image frame {frame} width differs from snapshot: "
                    f"declared={width}, actual={actual_width}"
                )
            if declared_height is not None and height != actual_height:
                raise ValueError(
                    f"Image frame {frame} height differs from snapshot: "
                    f"declared={height}, actual={actual_height}"
                )
            declared_sha = image.get("sha256")
            actual_sha = sha256(image_path)
            if declared_sha is not None and str(declared_sha) != actual_sha:
                raise ValueError(
                    f"Image frame {frame} SHA-256 differs from snapshot: "
                    f"declared={declared_sha}, actual={actual_sha}"
                )
            normalized_images.append(
                {
                    **image,
                    "frame": frame,
                    "sample_index": int(image.get("sample_index", position + 1)),
                    "file_name": str(image.get("file_name", image_path.name)),
                    "_resolved_path": image_path,
                    "_actual_width": actual_width,
                    "_actual_height": actual_height,
                    "_actual_sha256": actual_sha,
                    "width": width,
                    "height": height,
                }
            )

        unknown_shape_frames = sorted(set(shapes_by_frame) - seen_frames)
        if unknown_shape_frames:
            raise ValueError(f"Shapes reference frames missing from images: {unknown_shape_frames}")

        normalized_images.sort(key=lambda item: (item["frame"], item["sample_index"]))
        frames: list[dict[str, Any]] = []
        overlay_paths: list[Path] = []
        for image in normalized_images:
            frame_number = int(image["frame"])
            frame_shapes = shapes_by_frame.get(frame_number, [])
            flags = build_frame_flags(
                frame_number,
                frame_shapes,
                int(image["width"]),
                int(image["height"]),
                duplicate_iou_threshold=duplicate_iou_threshold,
                conflict_iou_threshold=conflict_iou_threshold,
                rider_overlap_threshold=rider_overlap_threshold,
            )
            overlay_relative = Path("overlays") / f"frame-{frame_number:06d}.jpg"
            overlay_path = make_overlay(
                image["_resolved_path"], frame_shapes, flags, output_dir / overlay_relative
            )
            overlay_paths.append(overlay_path)
            declared_sha = image.get("sha256")
            actual_sha = str(image["_actual_sha256"])
            frame_payload: dict[str, Any] = {
                "frame": frame_number,
                "sample_index": int(image["sample_index"]),
                "file_name": image["file_name"],
                "source_path": str(image["_resolved_path"]),
                "width": int(image["width"]),
                "height": int(image["height"]),
                "actual_width": int(image["_actual_width"]),
                "actual_height": int(image["_actual_height"]),
                "sha256": declared_sha or actual_sha,
                "actual_sha256": actual_sha,
                "sha256_matches_snapshot": True,
                "annotation_count": len(frame_shapes),
                "shapes": [_public_shape(shape) for shape in frame_shapes],
                "flag_count": len(flags),
                "automated_risk_flag_count": sum(
                    flag["type"] in AUTOMATED_FLAG_TYPES for flag in flags
                ),
                "flags": flags,
                "overlay": overlay_relative.as_posix(),
            }
            if image.get("source_frame") is not None:
                frame_payload["source_frame"] = int(image["source_frame"])
            frames.append(frame_payload)

        contact_paths = make_contact_sheets(frames, overlay_paths, output_dir / "contact-sheets")
        flag_counts = Counter(flag["type"] for frame in frames for flag in frame["flags"])
        annotation_sha = snapshot.get(
            "annotation_sha256", snapshot.get("canonical_annotations_sha256")
        )
        task_record = snapshot.get("task")
        nested_task_id = task_record.get("id") if isinstance(task_record, dict) else None
        payload = {
            "schema_version": "1.0",
            "pack_type": "full_annotation_review",
            "read_only": True,
            "mutation_performed": False,
            "reviewed_by_human": False,
            "task_id": snapshot.get("task_id", nested_task_id),
            "source_snapshot": str(snapshot_path),
            "source_snapshot_sha256": hashlib.sha256(snapshot_bytes).hexdigest(),
            "annotation_sha256": annotation_sha,
            "thresholds": {
                "same_class_duplicate_iou": duplicate_iou_threshold,
                "cross_class_conflict_iou": conflict_iou_threshold,
                "rider_pedestrian_overlap": rider_overlap_threshold,
            },
            "manual_check_labels": list(MANUAL_CHECK_LABELS),
            "frame_count": len(frames),
            "annotation_count": sum(frame["annotation_count"] for frame in frames),
            "flag_count": sum(frame["flag_count"] for frame in frames),
            "automated_risk_flag_count": sum(
                frame["automated_risk_flag_count"] for frame in frames
            ),
            "manual_check_flag_count": flag_counts["manual_class_check"],
            "flag_counts_by_type": dict(sorted(flag_counts.items())),
            "all_frames_included": len(frames) == len(images),
            "frames": frames,
            "contact_sheets": [path.relative_to(output_dir).as_posix() for path in contact_paths],
        }
        review_pack_path = output_dir / "review-pack.json"
        write_json_atomic(review_pack_path, payload)
        return review_pack_path
    except Exception:
        # This directory was created by this invocation only; remove an incomplete
        # pack so a later safe retry is not confused with a completed review pack.
        shutil.rmtree(output_dir)
        raise


# A descriptive alias for callers that mirror the script name.
build_full_review_pack = build_review_pack


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--output", "--output-dir", dest="output", type=Path)
    parser.add_argument("--duplicate-iou", type=float, default=0.50)
    parser.add_argument("--conflict-iou", type=float, default=0.50)
    parser.add_argument("--rider-overlap", type=float, default=0.25)
    args = parser.parse_args(argv)
    output = args.output or args.snapshot.with_name(f"{args.snapshot.stem}-full-review-pack")
    try:
        review_pack_path = build_review_pack(
            args.snapshot,
            output,
            duplicate_iou_threshold=args.duplicate_iou,
            conflict_iou_threshold=args.conflict_iou,
            rider_overlap_threshold=args.rider_overlap,
        )
    except (FileExistsError, FileNotFoundError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    payload = json.loads(review_pack_path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "review_pack": str(review_pack_path),
                "frame_count": payload["frame_count"],
                "annotation_count": payload["annotation_count"],
                "automated_risk_flag_count": payload["automated_risk_flag_count"],
                "manual_check_flag_count": payload["manual_check_flag_count"],
                "contact_sheet_count": len(payload["contact_sheets"]),
                "mutation_performed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
