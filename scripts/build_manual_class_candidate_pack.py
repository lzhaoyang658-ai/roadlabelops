"""Build a read-only YOLO candidate pack for manually reviewed road classes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

TARGET_MODEL_LABELS = {
    "person": "pedestrian",
    "pedestrian": "pedestrian",
    "bicycle": "bicycle",
    "car": "car",
    "motorcycle": "motorcycle",
    "bus": "bus",
    "truck": "truck",
    "traffic light": "traffic_light",
    "traffic signal": "traffic_light",
    "stop sign": "traffic_sign",
    "traffic sign": "traffic_sign",
    "road sign": "traffic_sign",
}
TARGET_COLORS = {
    "car": "#22c55e",
    "bus": "#f59e0b",
    "truck": "#ef4444",
    "motorcycle": "#3b82f6",
    "bicycle": "#8b5cf6",
    "pedestrian": "#ec4899",
    "traffic_light": "#06b6d4",
    "traffic_sign": "#a855f7",
}
DEFAULT_REVIEW_LABELS = frozenset({"traffic_light", "traffic_sign"})
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

Predictor = Callable[
    [Sequence[Path], Path, float, float, int, str, int, frozenset[str]],
    list[list[dict[str, Any]]],
]


def _strict_int(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{location} must be an integer")
    return value


def _strict_float(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{location} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{location} must be finite")
    return result


def _require_sha256(value: Any, location: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{location} must be a lowercase SHA-256 hex digest")
    return value


def _model_label_key(value: Any) -> str:
    return " ".join(str(value).strip().lower().replace("_", " ").replace("-", " ").split())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    if len(first) != 4 or len(second) != 4:
        raise ValueError("IoU requires two four-coordinate xyxy boxes")
    ax1, ay1, ax2, ay2 = (float(value) for value in first)
    bx1, by1, bx2, by2 = (float(value) for value in second)
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def box_overlap_metrics(
    candidate: Sequence[float], existing: Sequence[float]
) -> tuple[float, float, float]:
    """Return IoU and the intersection share of each input box."""

    if len(candidate) != 4 or len(existing) != 4:
        raise ValueError("Overlap metrics require two four-coordinate xyxy boxes")
    cx1, cy1, cx2, cy2 = (float(value) for value in candidate)
    ex1, ey1, ex2, ey2 = (float(value) for value in existing)
    intersection_width = max(0.0, min(cx2, ex2) - max(cx1, ex1))
    intersection_height = max(0.0, min(cy2, ey2) - max(cy1, ey1))
    intersection = intersection_width * intersection_height
    candidate_area = max(0.0, cx2 - cx1) * max(0.0, cy2 - cy1)
    existing_area = max(0.0, ex2 - ex1) * max(0.0, ey2 - ey1)
    union = candidate_area + existing_area - intersection
    return (
        intersection / union if union > 0 else 0.0,
        intersection / candidate_area if candidate_area > 0 else 0.0,
        intersection / existing_area if existing_area > 0 else 0.0,
    )


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _validate_bbox(value: Any, *, width: int, height: int, location: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{location} must contain four coordinates")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ValueError(f"{location} coordinates must be numbers")
    points = []
    for index, item in enumerate(value):
        point = round(_strict_float(item, f"{location} coordinate {index}"), 2)
        points.append(0.0 if point == 0.0 else point)
    if not all(math.isfinite(item) for item in points):
        raise ValueError(f"{location} coordinates must be finite")
    x1, y1, x2, y2 = points
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"{location} must have positive width and height")
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        raise ValueError(f"{location} falls outside frame bounds")
    return points


def _class_aware_nms(
    candidates: Sequence[dict[str, Any]], *, iou_threshold: float
) -> list[dict[str, Any]]:
    retained: list[dict[str, Any]] = []
    for label in sorted({str(item["label"]) for item in candidates}):
        accepted: list[dict[str, Any]] = []
        same_class = [item for item in candidates if item["label"] == label]
        for candidate in sorted(
            same_class,
            key=lambda item: (-float(item["confidence"]), tuple(item["bbox"])),
        ):
            if any(
                box_iou(candidate["bbox"], previous["bbox"]) >= iou_threshold
                for previous in accepted
            ):
                continue
            accepted.append(candidate)
        retained.extend(accepted)
    return sorted(retained, key=lambda item: (item["label"], -item["confidence"], item["bbox"]))


def normalize_candidates(
    raw_candidates: Sequence[Mapping[str, Any]],
    *,
    task_id: int,
    frame: int,
    width: int,
    height: int,
    model_sha256: str,
    source_image_sha256: str,
    confidence_threshold: float,
    nms_iou_threshold: float,
    existing_shapes: Sequence[Mapping[str, Any]],
    existing_match_iou: float,
    review_labels: frozenset[str] = DEFAULT_REVIEW_LABELS,
) -> list[dict[str, Any]]:
    """Normalize, deduplicate, bind, and match one frame of model candidates."""

    _require_sha256(model_sha256, "model_sha256")
    _require_sha256(source_image_sha256, "source_image_sha256")
    _strict_int(task_id, "task_id")
    _strict_int(frame, "frame")
    _strict_int(width, "width")
    _strict_int(height, "height")
    if task_id <= 0 or frame < 0 or width <= 0 or height <= 0:
        raise ValueError("task_id and dimensions must be positive; frame must be non-negative")

    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_candidates):
        if not isinstance(raw, Mapping):
            raise TypeError(f"frame {frame} candidate {index} must be an object")
        label = raw.get("label")
        if label not in review_labels:
            continue
        confidence = _strict_float(raw.get("confidence", 0.0), f"frame {frame} candidate {index}")
        if confidence < confidence_threshold:
            continue
        bbox = _validate_bbox(
            raw.get("bbox"),
            width=width,
            height=height,
            location=f"frame {frame} candidate {index} bbox",
        )
        normalized.append(
            {
                "label": str(label),
                "model_label": str(raw.get("model_label", label)),
                "confidence": round(confidence, 4),
                "bbox": bbox,
            }
        )

    normalized_existing: list[dict[str, Any]] = []
    for shape_index, shape in enumerate(existing_shapes):
        if not isinstance(shape, Mapping):
            raise TypeError(f"frame {frame} existing shape {shape_index} must be an object")
        label = str(shape.get("label", ""))
        points = _validate_bbox(
            shape.get("points"),
            width=width,
            height=height,
            location=f"frame {frame} existing shape {shape_index}",
        )
        normalized_existing.append(
            {
                "shape_id": shape.get("id"),
                "label": label,
                "bbox": points,
            }
        )

    results: list[dict[str, Any]] = []
    for candidate in _class_aware_nms(normalized, iou_threshold=nms_iou_threshold):
        overlaps: list[dict[str, Any]] = []
        for existing in normalized_existing:
            iou, candidate_coverage, existing_coverage = box_overlap_metrics(
                candidate["bbox"], existing["bbox"]
            )
            if iou <= 0.0:
                continue
            overlaps.append(
                {
                    "shape_id": existing["shape_id"],
                    "label": existing["label"],
                    "iou": round(iou, 4),
                    "candidate_coverage": round(candidate_coverage, 4),
                    "existing_coverage": round(existing_coverage, 4),
                }
            )
        overlaps.sort(
            key=lambda item: (
                -float(item["candidate_coverage"]),
                -float(item["iou"]),
                str(item["label"]),
                str(item["shape_id"]),
            )
        )
        matched = any(
            overlap["label"] == candidate["label"] and float(overlap["iou"]) >= existing_match_iou
            for overlap in overlaps
        )
        cross_label_overlap = any(
            overlap["label"] != candidate["label"]
            and float(overlap["candidate_coverage"]) >= existing_match_iou
            for overlap in overlaps
        )
        identity = {
            "task_id": task_id,
            "frame": frame,
            "label": candidate["label"],
            "bbox": candidate["bbox"],
            "model_sha256": model_sha256,
            "source_image_sha256": source_image_sha256,
        }
        results.append(
            {
                "candidate_id": (
                    f"task-{task_id}-frame-{frame:06d}-{candidate['label'].replace('_', '-')}-"
                    f"{canonical_sha256(identity)[:20]}"
                ),
                "frame": frame,
                **candidate,
                "status": "already_annotated" if matched else "needs_human_review",
                "review_reason": (
                    "same_label_match"
                    if matched
                    else "cross_label_overlap"
                    if cross_label_overlap
                    else "no_same_label_match"
                ),
                "existing_overlaps": overlaps,
                "mutation_performed": False,
            }
        )
    return results


def _run_yolo(
    image_paths: Sequence[Path],
    model_path: Path,
    confidence: float,
    nms_iou: float,
    image_size: int,
    device: str,
    batch_size: int,
    review_labels: frozenset[str],
) -> list[list[dict[str, Any]]]:
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("Install the detection extra before building candidates") from error

    model = YOLO(str(model_path))
    names = model.names
    name_by_id = dict(names) if isinstance(names, dict) else dict(enumerate(names))
    class_ids = [
        identifier
        for identifier, name in name_by_id.items()
        if TARGET_MODEL_LABELS.get(_model_label_key(name)) in review_labels
    ]
    supported_road_labels = {
        TARGET_MODEL_LABELS[key]
        for name in name_by_id.values()
        if (key := _model_label_key(name)) in TARGET_MODEL_LABELS
    }
    missing = sorted(review_labels - supported_road_labels)
    if missing:
        raise ValueError(f"Model has no usable class for road labels: {missing}")

    predictions: list[list[dict[str, Any]]] = []
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        results = model.predict(
            source=[str(path) for path in batch_paths],
            classes=class_ids,
            conf=confidence,
            iou=nms_iou,
            imgsz=image_size,
            device=device,
            batch=len(batch_paths),
            verbose=False,
        )
        if len(results) != len(batch_paths):
            raise RuntimeError("YOLO returned a different result count than the input batch")
        for batch_index, result in enumerate(results):
            result_path = getattr(result, "path", None)
            if (
                result_path is None
                or Path(str(result_path)).resolve() != batch_paths[batch_index].resolve()
            ):
                raise RuntimeError("YOLO result order/path does not match the input batch")
            frame_candidates: list[dict[str, Any]] = []
            for box in result.boxes or []:
                model_label = str(result.names[int(box.cls.item())])
                label = TARGET_MODEL_LABELS.get(_model_label_key(model_label))
                if label is None:
                    continue
                frame_candidates.append(
                    {
                        "label": label,
                        "model_label": model_label,
                        "confidence": float(box.conf.item()),
                        "bbox": [float(value) for value in box.xyxy[0].tolist()],
                    }
                )
            predictions.append(frame_candidates)
    return predictions


def _draw_candidate_overlay(
    source_path: Path,
    candidates: Sequence[Mapping[str, Any]],
    existing_shapes: Sequence[Mapping[str, Any]],
    review_labels: frozenset[str],
    output_path: Path,
) -> None:
    with Image.open(source_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    label_font = _font(17)
    banner_font = _font(18)
    for shape in existing_shapes:
        label = str(shape.get("label", ""))
        if label not in review_labels:
            continue
        bbox = tuple(float(value) for value in shape["points"])
        color = TARGET_COLORS[label]
        draw.rectangle(bbox, outline=color, width=2)
        draw.text(
            (max(0.0, bbox[0]), max(34.0, bbox[1])),
            f"EXISTING {shape.get('id', '?')} {label}",
            fill=color,
            font=label_font,
            stroke_width=1,
        )
    for candidate in candidates:
        color = "#facc15"
        bbox = tuple(float(value) for value in candidate["bbox"])
        draw.rectangle(bbox, outline=color, width=5)
        caption = (
            f"CHECK {candidate['label']} {candidate['confidence']:.2f} "
            f"{candidate['candidate_id'][-6:]}"
        )
        bounds = draw.textbbox((bbox[0], bbox[1]), caption, font=label_font)
        text_width = bounds[2] - bounds[0] + 8
        text_height = bounds[3] - bounds[1] + 8
        text_x = min(max(0.0, bbox[0]), max(0.0, image.width - text_width))
        text_y = max(0.0, bbox[1] - text_height)
        draw.rectangle(
            (text_x, text_y, text_x + text_width, text_y + text_height),
            fill=color,
        )
        draw.text((text_x + 4, text_y + 3), caption, fill="white", font=label_font)
    banner = f"MODEL CANDIDATES · {len(candidates)} unmatched boxes · REVIEW ONLY"
    draw.rectangle((0, 0, image.width, 32), fill="#111827")
    draw.text((8, 6), banner, fill="white", font=banner_font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="JPEG", quality=94)


def _make_contact_sheets(
    frame_records: Sequence[Mapping[str, Any]], staging_dir: Path
) -> list[str]:
    output_dir = staging_dir / "contact-sheets"
    output_dir.mkdir(parents=True, exist_ok=True)
    tile_width, image_height, header_height = 640, 360, 42
    title_font = _font(19)
    outputs: list[str] = []
    for start in range(0, len(frame_records), 6):
        subset = frame_records[start : start + 6]
        sheet = Image.new("RGB", (tile_width * 2, (image_height + header_height) * 3), "#111827")
        draw = ImageDraw.Draw(sheet)
        for local_index, frame_record in enumerate(subset):
            column, row = local_index % 2, local_index // 2
            x = column * tile_width
            y = row * (image_height + header_height)
            counts = Counter(item["label"] for item in frame_record["candidates"])
            compact_counts = "/".join(
                f"{label.replace('_', '-')[:5]}:{count}" for label, count in sorted(counts.items())
            )
            title = (
                f"FRAME {frame_record['frame']:06d} · "
                f"{frame_record['needs_human_review_count']} checks · {compact_counts}"
            )
            draw.text((x + 8, y + 9), title, fill="white", font=title_font)
            with Image.open(staging_dir / str(frame_record["overlay"])) as source:
                tile = ImageOps.contain(source.convert("RGB"), (tile_width, image_height))
            paste_x = x + (tile_width - tile.width) // 2
            paste_y = y + header_height + (image_height - tile.height) // 2
            sheet.paste(tile, (paste_x, paste_y))
        relative = Path("contact-sheets") / f"candidate-sheet-{len(outputs) + 1:03d}.jpg"
        sheet.save(staging_dir / relative, format="JPEG", quality=92)
        outputs.append(relative.as_posix())
    return outputs


def _publish_staged_directory(staging_dir: Path, output_dir: Path) -> None:
    """Publish without replacing even an empty directory created by another process.

    The destination directory is claimed with an atomic ``mkdir``. Top-level
    artifacts are then moved from the private staging directory, with the JSON
    manifest moved last and an incomplete marker retained until the commit ends.
    A consumer must treat a directory without ``candidate-pack.json`` or with the
    marker present as incomplete.
    """

    output_dir.mkdir(exist_ok=False)
    marker = output_dir / ".incomplete"
    try:
        marker.write_text("candidate pack publication in progress\n", encoding="utf-8")
        manifest = staging_dir / "candidate-pack.json"
        if not manifest.is_file():
            raise RuntimeError("Staging directory has no candidate-pack.json")
        for child in sorted(staging_dir.iterdir(), key=lambda path: path.name):
            if child == manifest:
                continue
            os.replace(child, output_dir / child.name)
        os.replace(manifest, output_dir / manifest.name)
        staging_dir.rmdir()
        marker.unlink()
    except Exception:
        # Only remove a destination still carrying this invocation's incomplete
        # marker. A completed or externally modified directory is left untouched.
        if marker.is_file() or (output_dir.is_dir() and not any(output_dir.iterdir())):
            shutil.rmtree(output_dir, ignore_errors=True)
        raise


def build_candidate_pack(
    review_pack_path: Path | str,
    model_path: Path | str,
    output_dir: Path | str,
    *,
    confidence: float = 0.12,
    nms_iou: float = 0.30,
    existing_match_iou: float = 0.30,
    image_size: int = 960,
    device: str = "mps",
    batch_size: int = 16,
    review_labels: Sequence[str] = tuple(sorted(DEFAULT_REVIEW_LABELS)),
    predictor: Predictor | None = None,
) -> Path:
    """Build an immutable auxiliary pack without mutating CVAT or its review pack."""

    review_pack_path = Path(review_pack_path).resolve()
    model_path = Path(model_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output path: {output_dir}")
    if not review_pack_path.is_file():
        raise FileNotFoundError(f"Review pack does not exist: {review_pack_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model does not exist: {model_path}")
    confidence = _strict_float(confidence, "confidence")
    nms_iou = _strict_float(nms_iou, "nms_iou")
    existing_match_iou = _strict_float(existing_match_iou, "existing_match_iou")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    for value, name in ((nms_iou, "nms_iou"), (existing_match_iou, "existing_match_iou")):
        if not 0.0 < value <= 1.0:
            raise ValueError(f"{name} must be greater than 0 and at most 1")
    _strict_int(image_size, "image_size")
    _strict_int(batch_size, "batch_size")
    if image_size <= 0 or batch_size <= 0:
        raise ValueError("image_size and batch_size must be positive")
    selected_labels = frozenset(review_labels)
    unknown_labels = sorted(selected_labels - set(TARGET_COLORS))
    if not selected_labels or unknown_labels:
        raise ValueError(
            "review_labels must be a non-empty subset of supported labels; "
            f"unknown={unknown_labels}"
        )

    pack_bytes = review_pack_path.read_bytes()
    try:
        review_pack = json.loads(pack_bytes)
    except json.JSONDecodeError as error:
        raise ValueError(f"Review pack is not valid JSON: {error}") from error
    if not isinstance(review_pack, dict):
        raise TypeError("Review pack root must be an object")
    if review_pack.get("schema_version") != "1.0":
        raise ValueError("Review pack schema_version must be '1.0'")
    if review_pack.get("pack_type") != "full_annotation_review":
        raise ValueError("Review pack pack_type must be 'full_annotation_review'")
    if (
        review_pack.get("read_only") is not True
        or review_pack.get("mutation_performed") is not False
    ):
        raise ValueError("Review pack must be read-only and non-mutating")
    if review_pack.get("all_frames_included") is not True:
        raise ValueError("Review pack must explicitly include all frames")
    _require_sha256(review_pack.get("source_snapshot_sha256"), "source_snapshot_sha256")
    _require_sha256(review_pack.get("annotation_sha256"), "annotation_sha256")
    frames = review_pack.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("Review pack must contain a non-empty frames list")
    frame_count = _strict_int(review_pack.get("frame_count"), "Review pack frame_count")
    if frame_count != len(frames):
        raise ValueError("Review pack frame_count does not match its frames list")
    task_id = _strict_int(review_pack.get("task_id"), "Review pack task_id")
    if task_id <= 0:
        raise ValueError("Review pack task_id must be positive")

    image_paths: list[Path] = []
    expected_images: list[tuple[Path, str, int, int]] = []
    seen_frames: set[int] = set()
    for index, raw_frame in enumerate(frames):
        if not isinstance(raw_frame, dict):
            raise TypeError("Every review-pack frame must be an object")
        frame = _strict_int(raw_frame.get("frame"), f"Review pack frame {index} number")
        if frame < 0:
            raise ValueError(f"Review pack frame {index} number must be non-negative")
        if frame in seen_frames:
            raise ValueError(f"Review pack contains duplicate frame {frame}")
        seen_frames.add(frame)
        source_path = Path(str(raw_frame.get("source_path", ""))).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"Frame {frame} source image does not exist: {source_path}")
        declared_sha = _require_sha256(
            raw_frame.get("actual_sha256", raw_frame.get("sha256")),
            f"Frame {frame} source SHA-256",
        )
        actual_sha = file_sha256(source_path)
        if declared_sha != actual_sha:
            raise ValueError(f"Frame {frame} source image SHA-256 differs from the review pack")
        width = _strict_int(raw_frame.get("width"), f"Frame {frame} width")
        height = _strict_int(raw_frame.get("height"), f"Frame {frame} height")
        if width <= 0 or height <= 0:
            raise ValueError(f"Frame {frame} dimensions must be positive")
        shapes = raw_frame.get("shapes", [])
        if not isinstance(shapes, list):
            raise TypeError(f"Frame {frame} shapes must be a list")
        sample_index = _strict_int(raw_frame.get("sample_index"), f"Frame {frame} sample_index")
        if sample_index <= 0:
            raise ValueError(f"Frame {frame} sample_index must be positive")
        with Image.open(source_path) as source:
            if source.size != (width, height):
                raise ValueError(
                    f"Frame {frame} source image dimensions differ from the review pack"
                )
        image_paths.append(source_path)
        expected_images.append((source_path, actual_sha, width, height))

    model_sha = file_sha256(model_path)

    def verify_bound_inputs() -> None:
        if file_sha256(model_path) != model_sha:
            raise ValueError("Model SHA-256 changed while the candidate pack was being built")
        for source_path, expected_sha, width, height in expected_images:
            if file_sha256(source_path) != expected_sha:
                raise ValueError(
                    f"Source image SHA-256 changed while building candidates: {source_path}"
                )
            with Image.open(source_path) as source:
                if source.size != (width, height):
                    raise ValueError(
                        f"Source image dimensions changed while building candidates: {source_path}"
                    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        # Freeze every path-based input in a private directory. Inference and
        # rendering use only these verified copies, closing the replace/restore
        # (ABA) window on the user-owned originals.
        frozen_dir = staging_dir / ".inputs"
        frozen_dir.mkdir()
        frozen_model_path = frozen_dir / f"model{model_path.suffix or '.bin'}"
        shutil.copyfile(model_path, frozen_model_path)
        if file_sha256(frozen_model_path) != model_sha:
            raise ValueError("Frozen model copy does not match the verified model SHA-256")
        frozen_image_paths: list[Path] = []
        for index, (source_path, expected_sha, width, height) in enumerate(expected_images):
            suffix = source_path.suffix or ".img"
            frozen_path = frozen_dir / f"frame-{index:06d}{suffix}"
            shutil.copyfile(source_path, frozen_path)
            if file_sha256(frozen_path) != expected_sha:
                raise ValueError(f"Frozen source image does not match SHA-256: {source_path}")
            with Image.open(frozen_path) as source:
                if source.size != (width, height):
                    raise ValueError(f"Frozen source image dimensions differ: {source_path}")
            frozen_image_paths.append(frozen_path)

        def verify_frozen_inputs() -> None:
            if file_sha256(frozen_model_path) != model_sha:
                raise ValueError("Frozen model SHA-256 changed during candidate generation")
            for frozen_path, (_, expected_sha, width, height) in zip(
                frozen_image_paths, expected_images
            ):
                if file_sha256(frozen_path) != expected_sha:
                    raise ValueError(
                        f"Frozen source image SHA-256 changed during candidate generation: "
                        f"{frozen_path}"
                    )
                with Image.open(frozen_path) as source:
                    if source.size != (width, height):
                        raise ValueError(
                            "Frozen source image dimensions changed during candidate generation: "
                            f"{frozen_path}"
                        )

        predict = predictor or _run_yolo
        raw_by_frame = predict(
            frozen_image_paths,
            frozen_model_path,
            confidence,
            nms_iou,
            image_size,
            device,
            batch_size,
            selected_labels,
        )
        if len(raw_by_frame) != len(frames):
            raise RuntimeError("Predictor returned a different frame count than the review pack")
        verify_frozen_inputs()
        verify_bound_inputs()

        frame_records: list[dict[str, Any]] = []
        all_candidates: list[dict[str, Any]] = []
        candidate_ids: set[str] = set()
        for raw_frame, source_path, frozen_path, raw_candidates in zip(
            frames, image_paths, frozen_image_paths, raw_by_frame
        ):
            frame = _strict_int(raw_frame["frame"], "Review pack frame")
            candidates = normalize_candidates(
                raw_candidates,
                task_id=task_id,
                frame=frame,
                width=_strict_int(raw_frame["width"], f"Frame {frame} width"),
                height=_strict_int(raw_frame["height"], f"Frame {frame} height"),
                model_sha256=model_sha,
                source_image_sha256=str(raw_frame.get("actual_sha256", raw_frame.get("sha256"))),
                confidence_threshold=confidence,
                nms_iou_threshold=nms_iou,
                existing_shapes=raw_frame.get("shapes", []),
                existing_match_iou=existing_match_iou,
                review_labels=selected_labels,
            )
            for candidate in candidates:
                candidate_id = str(candidate["candidate_id"])
                if candidate_id in candidate_ids:
                    raise RuntimeError(f"Candidate ID collision: {candidate_id}")
                candidate_ids.add(candidate_id)
            needs_review = [
                candidate for candidate in candidates if candidate["status"] == "needs_human_review"
            ]
            all_candidates.extend(candidates)
            record: dict[str, Any] = {
                "frame": frame,
                "sample_index": _strict_int(
                    raw_frame["sample_index"], f"Frame {frame} sample_index"
                ),
                "source_path": str(source_path),
                "source_sha256": raw_frame.get("actual_sha256", raw_frame.get("sha256")),
                "candidate_count": len(candidates),
                "needs_human_review_count": len(needs_review),
                "candidates": candidates,
            }
            if needs_review:
                overlay = Path("overlays") / f"frame-{frame:06d}.jpg"
                _draw_candidate_overlay(
                    frozen_path,
                    needs_review,
                    raw_frame.get("shapes", []),
                    selected_labels,
                    staging_dir / overlay,
                )
                record["overlay"] = overlay.as_posix()
            frame_records.append(record)

        candidate_frames = [record for record in frame_records if record.get("overlay")]
        contact_sheets = _make_contact_sheets(candidate_frames, staging_dir)
        needs_review_candidates = [
            candidate for candidate in all_candidates if candidate["status"] == "needs_human_review"
        ]
        verify_frozen_inputs()
        shutil.rmtree(frozen_dir)
        payload = {
            "schema_version": "1.0",
            "pack_type": "manual_class_model_candidates",
            "read_only": True,
            "reviewed_by_human": False,
            "mutation_performed": False,
            "task_id": task_id,
            "source_review_pack": str(review_pack_path),
            "source_review_pack_sha256": hashlib.sha256(pack_bytes).hexdigest(),
            "source_snapshot_sha256": review_pack.get("source_snapshot_sha256"),
            "model": str(model_path),
            "model_sha256": model_sha,
            "model_label_mapping": {
                model_label: road_label
                for model_label, road_label in TARGET_MODEL_LABELS.items()
                if road_label in selected_labels
            },
            "review_labels": sorted(selected_labels),
            "parameters": {
                "confidence": confidence,
                "nms_iou": nms_iou,
                "existing_match_iou": existing_match_iou,
                "image_size": image_size,
                "device": device,
                "batch_size": batch_size,
            },
            "frame_count": len(frame_records),
            "frames_with_candidates": sum(
                record["candidate_count"] > 0 for record in frame_records
            ),
            "frames_needing_human_review": len(candidate_frames),
            "candidate_count": len(all_candidates),
            "needs_human_review_count": len(needs_review_candidates),
            "candidate_counts_by_label": dict(
                sorted(Counter(item["label"] for item in all_candidates).items())
            ),
            "needs_human_review_counts_by_label": dict(
                sorted(Counter(item["label"] for item in needs_review_candidates).items())
            ),
            "needs_human_review_counts_by_reason": dict(
                sorted(Counter(item["review_reason"] for item in needs_review_candidates).items())
            ),
            "frames": frame_records,
            "contact_sheets": contact_sheets,
        }
        output_path = staging_dir / "candidate-pack.json"
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        # Reconfirm that the original provenance paths still match before the
        # frozen-input result is published.
        verify_bound_inputs()
        _publish_staged_directory(staging_dir, output_dir)
        return output_dir / "candidate-pack.json"
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review_pack", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--confidence", type=float, default=0.12)
    parser.add_argument("--nms-iou", type=float, default=0.30)
    parser.add_argument("--existing-match-iou", type=float, default=0.30)
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--review-labels",
        nargs="+",
        choices=sorted(TARGET_COLORS),
        default=sorted(DEFAULT_REVIEW_LABELS),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        output = build_candidate_pack(
            args.review_pack,
            args.model,
            args.output,
            confidence=args.confidence,
            nms_iou=args.nms_iou,
            existing_match_iou=args.existing_match_iou,
            image_size=args.image_size,
            device=args.device,
            batch_size=args.batch_size,
            review_labels=args.review_labels,
        )
    except (FileExistsError, FileNotFoundError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))
    payload = json.loads(output.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "output": str(output),
                "frame_count": payload["frame_count"],
                "frames_needing_human_review": payload["frames_needing_human_review"],
                "needs_human_review_count": payload["needs_human_review_count"],
                "needs_human_review_counts_by_label": payload["needs_human_review_counts_by_label"],
                "contact_sheet_count": len(payload["contact_sheets"]),
                "mutation_performed": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
