"""Safely apply an explicit, visually reviewed candidate batch to a CVAT task."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from cvat_sdk import models

from roadlabelops.settings import Settings, build_cvat_adapter

RIDER_MIN_HEIGHT_RATIO = 0.50


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def box_iou(first: list[float], second: list[float]) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


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


def same_box(first: list[float], second: list[float], tolerance: float = 0.02) -> bool:
    return all(abs(float(a) - float(b)) <= tolerance for a, b in zip(first, second))


def matching_relabel_rule(
    sample_index: int, label: str, rules: list[dict[str, Any]]
) -> tuple[int, str] | None:
    matches = [
        (index, str(rule["to_label"]))
        for index, rule in enumerate(rules)
        if label == rule["from_label"]
        and int(rule.get("sample_index_min", sample_index)) <= sample_index
        and int(rule.get("sample_index_max", sample_index)) >= sample_index
    ]
    if len(matches) > 1:
        raise ValueError(f"Multiple relabel rules match sample {sample_index} label {label}")
    return matches[0] if matches else None


def select_candidates(queue: dict[str, Any], decisions: dict[str, Any]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    ranked = queue["ranked_frames"]
    for rule in decisions.get("accepted_rules", []):
        for rank, frame in enumerate(ranked, start=1):
            if rank < int(rule.get("rank_min", 1)) or rank > int(rule["rank_max"]):
                continue
            for candidate in frame["candidates"]:
                if candidate["label"] != rule["label"]:
                    continue
                if float(candidate["confidence"]) < float(rule["min_confidence"]):
                    continue
                selected.append({**candidate, "sample_index": int(frame["sample_index"])})

    for explicit in decisions.get("accepted_explicit", []):
        frame = next(
            item for item in ranked if int(item["sample_index"]) == int(explicit["sample_index"])
        )
        match = next(
            candidate
            for candidate in frame["candidates"]
            if candidate["label"] == explicit["label"]
            and same_box(candidate["bbox_xyxy"], explicit["bbox_xyxy"])
        )
        selected.append({**match, "sample_index": int(frame["sample_index"])})

    unique: list[dict[str, Any]] = []
    for candidate in selected:
        if any(
            item["sample_index"] == candidate["sample_index"]
            and item["label"] == candidate["label"]
            and box_iou(item["bbox_xyxy"], candidate["bbox_xyxy"]) >= 0.50
            for item in unique
        ):
            continue
        unique.append(candidate)
    return unique


def shape_request(shape: Any, *, label_id: int | None = None) -> models.LabeledShapeRequest:
    kwargs: dict[str, Any] = {
        "type": "rectangle",
        "label_id": int(shape.label_id if label_id is None else label_id),
        "frame": int(shape.frame),
        "points": [float(value) for value in shape.points],
        "source": str(shape.source),
        "occluded": bool(shape.occluded),
        "outside": bool(shape.outside),
        "z_order": int(shape.z_order),
        "rotation": float(shape.rotation),
        "group": int(shape.group or 0),
    }
    if shape.score is not None:
        kwargs["score"] = float(shape.score)
    return models.LabeledShapeRequest(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queue", type=Path)
    parser.add_argument("decisions", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    queue_path = args.queue.resolve()
    decisions_path = args.decisions.resolve()
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
    if sha256(queue_path) != decisions["review_queue_sha256"]:
        raise SystemExit("Review queue hash does not match the reviewed decision file")

    selected = select_candidates(queue, decisions)
    task_id = int(decisions["task_id"])
    adapter = build_cvat_adapter(Settings())
    if adapter is None:
        raise SystemExit("CVAT is not configured")

    with adapter._client() as client:
        task = client.tasks.retrieve(task_id)
        annotations = task.get_annotations()
        if annotations.tags or annotations.tracks:
            raise SystemExit("Task contains tags or tracks; rectangle-only review apply denied")
        if len(annotations.shapes) != int(decisions["expected_annotation_count"]):
            raise SystemExit("Task annotation count changed after review; apply denied")
        actual_sources = Counter(str(shape.source) for shape in annotations.shapes)
        expected_sources = decisions.get("expected_source_counts")
        if expected_sources and dict(actual_sources) != expected_sources:
            raise SystemExit("Task annotation sources changed after review; apply denied")
        if not decisions.get("allow_existing_manual", False) and any(
            str(shape.source) != "auto" for shape in annotations.shapes
        ):
            raise SystemExit("Task already contains non-auto shapes; first-batch apply denied")

        labels = {int(label.id): label.name for label in task.get_labels()}
        label_ids = {name: identifier for identifier, name in labels.items()}
        selected_by_frame: dict[int, list[dict[str, Any]]] = {}
        for candidate in selected:
            frame = int(candidate["sample_index"]) - 1
            selected_by_frame.setdefault(frame, []).append(candidate)

        all_motorcycles_by_frame: dict[int, list[list[float]]] = {}
        if decisions.get("remove_rider_conflicts_against_all_motorcycles", False):
            for shape in annotations.shapes:
                if labels[int(shape.label_id)] == "motorcycle":
                    all_motorcycles_by_frame.setdefault(int(shape.frame), []).append(
                        [float(value) for value in shape.points]
                    )
            for frame, candidates in selected_by_frame.items():
                all_motorcycles_by_frame.setdefault(frame, []).extend(
                    candidate["bbox_xyxy"]
                    for candidate in candidates
                    if candidate["label"] == "motorcycle"
                )

        removed: list[dict[str, Any]] = []
        relabeled: list[dict[str, Any]] = []
        relabel_rules = decisions.get("relabel_rules", [])
        relabel_match_counts: Counter[int] = Counter()
        retained: list[tuple[Any, str]] = []
        for shape in annotations.shapes:
            frame_candidates = selected_by_frame.get(int(shape.frame), [])
            label = labels[int(shape.label_id)]
            points = [float(value) for value in shape.points]
            relabel_match = matching_relabel_rule(int(shape.frame) + 1, label, relabel_rules)
            effective_label = relabel_match[1] if relabel_match else label
            if relabel_match:
                rule_index, to_label = relabel_match
                if to_label not in label_ids:
                    raise SystemExit(f"Unknown relabel target: {to_label}")
                relabel_match_counts[rule_index] += 1
                relabeled.append(
                    {
                        "shape_id": int(shape.id),
                        "frame": int(shape.frame),
                        "sample_index": int(shape.frame) + 1,
                        "from_label": label,
                        "to_label": to_label,
                        "bbox_xyxy": points,
                    }
                )
            remove_reason = None
            rider_motorcycles = (
                all_motorcycles_by_frame.get(int(shape.frame), [])
                if decisions.get("remove_rider_conflicts_against_all_motorcycles", False)
                else [
                    candidate["bbox_xyxy"]
                    for candidate in frame_candidates
                    if candidate["label"] == "motorcycle"
                ]
            )
            if label == "pedestrian" and any(
                pedestrian_is_rider(points, motorcycle) for motorcycle in rider_motorcycles
            ):
                remove_reason = "rider_pedestrian_conflict"
            elif label in {"car", "truck"} and any(
                candidate["label"] == "bus" and box_iou(points, candidate["bbox_xyxy"]) >= 0.50
                for candidate in frame_candidates
            ):
                remove_reason = "bus_class_correction"
            if remove_reason:
                removed.append(
                    {
                        "shape_id": int(shape.id),
                        "frame": int(shape.frame),
                        "label": label,
                        "bbox_xyxy": points,
                        "reason": remove_reason,
                    }
                )
            else:
                retained.append((shape, effective_label))

        for index, rule in enumerate(relabel_rules):
            expected = rule.get("expected_match_count")
            if expected is not None and relabel_match_counts[index] != int(expected):
                raise SystemExit(
                    f"Relabel rule {index} matched {relabel_match_counts[index]} shapes; "
                    f"expected {expected}"
                )

        requests = [shape_request(shape, label_id=label_ids[label]) for shape, label in retained]
        for candidate in selected:
            requests.append(
                models.LabeledShapeRequest(
                    type="rectangle",
                    label_id=label_ids[candidate["label"]],
                    frame=int(candidate["sample_index"]) - 1,
                    points=[float(value) for value in candidate["bbox_xyxy"]],
                    source="manual",
                )
            )

        summary = {
            "task_id": task_id,
            "dry_run": not args.apply,
            "annotation_count_before": len(annotations.shapes),
            "accepted_count": len(selected),
            "accepted_by_label": {
                label: sum(item["label"] == label for item in selected)
                for label in sorted({item["label"] for item in selected})
            },
            "removed_count": len(removed),
            "removed": removed,
            "relabeled_count": len(relabeled),
            "relabeled": relabeled,
            "annotation_count_after": len(requests),
        }
        if not args.apply:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return

        artifact_dir = (Path.cwd() / decisions["artifact_directory"]).resolve()
        artifact_dir.mkdir(parents=True, exist_ok=True)
        backup_path = artifact_dir / decisions["backup_file_name"]
        correction_path = artifact_dir / decisions["correction_file_name"]
        if backup_path.exists() or correction_path.exists():
            raise SystemExit("Backup or correction log already exists; apply denied")
        backup = {
            "task_id": task_id,
            "annotation_count": len(annotations.shapes),
            "shapes": [shape.to_dict() for shape in annotations.shapes],
        }
        backup_path.write_text(json.dumps(backup, ensure_ascii=False, indent=2) + "\n")
        task.set_annotations(models.LabeledDataRequest(shapes=requests))
        verified = task.get_annotations()
        if len(verified.shapes) != len(requests):
            raise RuntimeError("CVAT verification count differs after write")
        summary["verified_annotation_count"] = len(verified.shapes)
        summary["backup_path"] = str(backup_path)
        correction_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
