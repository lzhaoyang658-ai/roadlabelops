from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..models import ToolResult

ROAD_LABELS = [
    "car",
    "bus",
    "truck",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_light",
    "traffic_sign",
]
MODEL_MAPPING = {
    "person": "pedestrian",
    "pedestrian": "pedestrian",
    "car": "car",
    "bus": "bus",
    "truck": "truck",
    "motorcycle": "motorcycle",
    "bicycle": "bicycle",
    "traffic light": "traffic_light",
    "traffic_light": "traffic_light",
    "stop sign": "traffic_sign",
    "traffic sign": "traffic_sign",
    "traffic_sign": "traffic_sign",
}


def result_frame_index(result_index: int, frame_step: int) -> int:
    """Map an Ultralytics vid_stride result to its actual zero-based video frame."""

    stride = max(1, frame_step)
    return (result_index + 1) * stride - 1


def _iou(first: list[float], second: list[float]) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    if not intersection:
        return 0.0
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _pedestrian_is_rider(
    pedestrian: list[float],
    motorcycle: list[float],
    overlap_threshold: float,
) -> bool:
    px1, py1, px2, py2 = pedestrian
    mx1, my1, mx2, my2 = motorcycle
    intersection_width = max(0.0, min(px2, mx2) - max(px1, mx1))
    intersection_height = max(0.0, min(py2, my2) - max(py1, my1))
    pedestrian_area = max(0.0, px2 - px1) * max(0.0, py2 - py1)
    overlap = intersection_width * intersection_height / pedestrian_area if pedestrian_area else 0.0
    bottom_center_x = (px1 + px2) / 2
    return overlap >= overlap_threshold and mx1 <= bottom_center_x <= mx2 and my1 <= py2 <= my2


def postprocess_predictions(
    predictions: list[dict[str, Any]],
    *,
    nms_iou_threshold: float = 0.75,
    rider_overlap_threshold: float = 0.25,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Remove mapped-class duplicates and pedestrian boxes attached to riders."""

    by_frame: dict[tuple[str, int], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, prediction in enumerate(predictions):
        key = (str(prediction.get("scene_id", "")), int(prediction["frame"]))
        by_frame[key].append((index, prediction))

    nms_kept: list[tuple[int, dict[str, Any]]] = []
    nms_suppressed = 0
    for candidates in by_frame.values():
        accepted: list[tuple[int, dict[str, Any]]] = []
        for candidate in sorted(
            candidates,
            key=lambda item: (-float(item[1].get("confidence", 0.0)), item[0]),
        ):
            if any(
                _iou(candidate[1]["bbox"], kept[1]["bbox"]) >= nms_iou_threshold
                for kept in accepted
            ):
                nms_suppressed += 1
                continue
            accepted.append(candidate)
        nms_kept.extend(accepted)

    kept_by_frame: dict[tuple[str, int], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for item in nms_kept:
        key = (str(item[1].get("scene_id", "")), int(item[1]["frame"]))
        kept_by_frame[key].append(item)

    final: list[tuple[int, dict[str, Any]]] = []
    rider_suppressed = 0
    for candidates in kept_by_frame.values():
        motorcycles = [item[1]["bbox"] for item in candidates if item[1]["label"] == "motorcycle"]
        for item in candidates:
            prediction = item[1]
            if prediction["label"] == "pedestrian" and any(
                _pedestrian_is_rider(prediction["bbox"], motorcycle, rider_overlap_threshold)
                for motorcycle in motorcycles
            ):
                rider_suppressed += 1
                continue
            final.append(item)

    processed = [prediction for _, prediction in sorted(final, key=lambda item: item[0])]
    metrics = {
        "input_prediction_count": len(predictions),
        "nms_suppressed_count": nms_suppressed,
        "rider_suppressed_count": rider_suppressed,
        "prediction_count": len(processed),
    }
    return processed, metrics


def run_mock_detection(scene_path: Path | str, confidence: float = 0.4) -> ToolResult:
    path = Path(scene_path)
    seed = int(hashlib.sha256(str(path).encode()).hexdigest()[:8], 16)
    count = 4 + seed % 8
    predictions = []
    for index in range(count):
        label = ROAD_LABELS[(seed + index) % len(ROAD_LABELS)]
        x = 20 + ((seed // (index + 1)) % 360)
        y = 30 + ((seed // (index + 3)) % 180)
        predictions.append(
            {
                "prediction_id": f"pred_{seed:x}_{index}",
                "frame": index * 5,
                "label": label,
                "confidence": round(min(0.98, confidence + ((seed + index) % 38) / 100), 2),
                "bbox": [x, y, x + 80 + index * 2, y + 70],
                "source": "auto",
            }
        )
    return ToolResult.success(
        {"predictions": predictions, "provider": "mock", "model": "deterministic-fixture"},
        metrics={"prediction_count": len(predictions)},
    )


def run_yolo_detection(
    scene_path: Path | str,
    *,
    model_name: str = "yolo11n.pt",
    confidence: float = 0.4,
    frame_step: int = 5,
    nms_iou_threshold: float = 0.75,
    rider_overlap_threshold: float = 0.25,
) -> ToolResult:
    path = Path(scene_path).resolve()
    if not path.is_file():
        return ToolResult.failure("SCENE_NOT_FOUND", "The scene video does not exist")
    model_path = Path(model_name).expanduser().resolve()
    if not model_path.is_file():
        return ToolResult.failure(
            "DETECTOR_MODEL_MISSING",
            (
                f"The configured YOLO weights do not exist: {model_path}. "
                "Provision the .pt file explicitly; runtime downloads are disabled"
            ),
        )
    try:
        from ultralytics import YOLO
    except ImportError:
        return ToolResult.failure(
            "DETECTOR_UNAVAILABLE",
            "Install the detection extra before running YOLO pre-annotation",
        )

    try:
        model = YOLO(str(model_path))
        stride = max(1, frame_step)
        results = model.predict(
            source=str(path),
            conf=confidence,
            stream=True,
            vid_stride=stride,
            verbose=False,
        )
        raw_predictions: list[dict[str, Any]] = []
        for result_index, result in enumerate(results):
            frame = result_frame_index(result_index, stride)
            if result.boxes is None:
                continue
            names = result.names
            for box_index, box in enumerate(result.boxes):
                model_label = str(names[int(box.cls.item())])
                label = MODEL_MAPPING.get(model_label)
                if not label:
                    continue
                points = [round(float(value), 2) for value in box.xyxy[0].tolist()]
                score = round(float(box.conf.item()), 4)
                raw_predictions.append(
                    {
                        "prediction_id": f"{path.stem}_{frame}_{box_index}",
                        "frame": frame,
                        "label": label,
                        "confidence": score,
                        "bbox": points,
                        "source": "auto",
                    }
                )
        predictions, metrics = postprocess_predictions(
            raw_predictions,
            nms_iou_threshold=nms_iou_threshold,
            rider_overlap_threshold=rider_overlap_threshold,
        )
        return ToolResult.success(
            {
                "predictions": predictions,
                "provider": "ultralytics",
                "model": str(model_path),
            },
            metrics=metrics,
        )
    except Exception as exc:
        return ToolResult.failure(
            "DETECTION_FAILED",
            f"YOLO could not process this scene: {type(exc).__name__}",
            retryable=True,
        )
