from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

from ..models import ToolResult

RETENTION_IOU_THRESHOLD = 0.80
EVALUATION_IOU_THRESHOLD = 0.50


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    return round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0


def _iou(first: list[float], second: list[float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _match_score(
    prediction: dict[str, Any],
    final: dict[str, Any],
    *,
    iou_threshold: float,
) -> float | None:
    required = ("frame", "label", "bbox")
    if not all(key in prediction and key in final for key in required):
        # Retain compatibility with the early, ID-only sidecar format. Whenever
        # geometry is available, it must satisfy the same frame/class/IoU rules
        # and an equal ID is not allowed to bypass them.
        if prediction.get("prediction_id") and prediction.get("prediction_id") == final.get(
            "prediction_id"
        ):
            return 1.0 if prediction.get("label") == final.get("label") else None
        return None
    if (
        prediction["frame"] != final["frame"]
        or prediction["label"] != final["label"]
        or prediction.get("scene_id") != final.get("scene_id")
    ):
        return None
    score = _iou(prediction["bbox"], final["bbox"])
    return score if score >= iou_threshold else None


def _match_annotations(
    predictions: list[dict[str, Any]],
    final_annotations: list[dict[str, Any]],
    *,
    iou_threshold: float,
) -> list[tuple[int, int]]:
    """Find a maximum one-to-one match, preferring confidence and higher IoU."""

    candidates: dict[int, list[tuple[float, int]]] = {}
    for prediction_index, prediction in enumerate(predictions):
        for final_index, final in enumerate(final_annotations):
            score = _match_score(prediction, final, iou_threshold=iou_threshold)
            if score is not None:
                candidates.setdefault(prediction_index, []).append((score, final_index))
    for prediction_candidates in candidates.values():
        prediction_candidates.sort(reverse=True)

    matched_final: dict[int, int] = {}

    def assign(prediction_index: int, visited_final: set[int]) -> bool:
        for _, final_index in candidates.get(prediction_index, []):
            if final_index in visited_final:
                continue
            visited_final.add(final_index)
            previous_prediction = matched_final.get(final_index)
            if previous_prediction is None or assign(previous_prediction, visited_final):
                matched_final[final_index] = prediction_index
                return True
        return False

    prediction_order = sorted(
        candidates,
        key=lambda index: (-float(predictions[index].get("confidence", 0.0)), index),
    )
    for prediction_index in prediction_order:
        assign(prediction_index, set())
    return [
        (prediction_index, final_index) for final_index, prediction_index in matched_final.items()
    ]


def calculate_quality(
    predictions: list[dict[str, Any]],
    final_annotations: list[dict[str, Any]],
    accepted_jobs: int = 0,
    reviewed_jobs: int = 0,
    *,
    evaluated_frame_keys: Iterable[tuple[str | None, int]] | None = None,
    first_pass_acceptance_reason: str | None = None,
) -> ToolResult:
    evaluation_pairs = _match_annotations(
        predictions,
        final_annotations,
        iou_threshold=EVALUATION_IOU_THRESHOLD,
    )
    retention_pairs = _match_annotations(
        predictions,
        final_annotations,
        iou_threshold=RETENTION_IOU_THRESHOLD,
    )
    matched = len(evaluation_pairs)
    retained = len(retention_pairs)
    added = len(final_annotations) - matched
    distribution = Counter(str(item.get("label", "unknown")) for item in final_annotations)
    precision = _ratio(matched, len(predictions))
    recall = _ratio(matched, len(final_annotations))
    retention_rate = _ratio(retained, len(predictions))

    observed_frame_keys = {
        (item.get("scene_id"), item["frame"])
        for item in [*predictions, *final_annotations]
        if "frame" in item
    }
    frame_keys = set(evaluated_frame_keys or ()) | observed_frame_keys
    clean_frames = 0
    for frame_key in frame_keys:
        prediction_indices = {
            index
            for index, item in enumerate(predictions)
            if (item.get("scene_id"), item.get("frame")) == frame_key
        }
        final_indices = {
            index
            for index, item in enumerate(final_annotations)
            if (item.get("scene_id"), item.get("frame")) == frame_key
        }
        matched_in_frame = sum(
            prediction_index in prediction_indices and final_index in final_indices
            for prediction_index, final_index in evaluation_pairs
        )
        if matched_in_frame == len(prediction_indices) == len(final_indices):
            clean_frames += 1

    per_class: dict[str, dict[str, int | float | None]] = {}
    labels = sorted(
        {str(item.get("label", "unknown")) for item in [*predictions, *final_annotations]}
    )
    for label in labels:
        prediction_count = sum(item.get("label") == label for item in predictions)
        final_count = sum(item.get("label") == label for item in final_annotations)
        true_positives = sum(
            predictions[prediction_index].get("label") == label
            for prediction_index, _ in evaluation_pairs
        )
        class_precision = _ratio(true_positives, prediction_count)
        class_recall = _ratio(true_positives, final_count)
        per_class[label] = {
            "true_positive_count": true_positives,
            "false_positive_count": prediction_count - true_positives,
            "false_negative_count": final_count - true_positives,
            "precision": class_precision,
            "recall": class_recall,
            "f1_score": _f1(class_precision, class_recall),
        }

    return ToolResult.success(
        {
            "prediction_count": len(predictions),
            "final_count": len(final_annotations),
            "retained_count": retained,
            "added_count": added,
            "removed_count": len(predictions) - retained,
            "retention_rate": retention_rate,
            "human_addition_rate": _ratio(added, len(final_annotations)),
            "precision": precision,
            "recall": recall,
            "f1_score": _f1(precision, recall),
            "evaluated_frame_count": len(frame_keys),
            "clean_frame_count": clean_frames,
            "clean_frame_rate": _ratio(clean_frames, len(frame_keys)),
            "first_pass_acceptance_rate": _ratio(accepted_jobs, reviewed_jobs),
            "first_pass_acceptance_reason": (
                first_pass_acceptance_reason if reviewed_jobs == 0 else None
            ),
            "class_distribution": dict(sorted(distribution.items())),
            "per_class": per_class,
        }
    )
