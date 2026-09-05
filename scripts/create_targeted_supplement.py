"""Create a class-targeted, non-duplicate training supplement in CVAT."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from roadlabelops.settings import Settings
from roadlabelops.storage import LocalStore
from scripts.create_qa_sample import (
    create_cvat_task,
    extract_frames,
    make_contact_sheets,
    make_overlay,
)


def prediction_score(predictions: list[dict[str, Any]], target_label: str) -> float:
    """Prioritize target density first, then detector confidence."""
    target = [item for item in predictions if item["label"] == target_label]
    return len(target) * 100.0 + sum(float(item.get("confidence", 1.0)) for item in target)


def select_target_frames(
    candidates: list[dict[str, Any]],
    *,
    target_label: str,
    count: int,
    min_frame_gap: int,
    max_scene_share: float,
    excluded: set[tuple[str, int]] | None = None,
) -> list[dict[str, Any]]:
    """Greedily select dense frames with temporal spacing and scene diversity."""
    if count <= 0:
        return []
    if not 0 < max_scene_share <= 1:
        raise ValueError("max_scene_share must be in (0, 1]")

    excluded = excluded or set()
    scene_cap = max(1, math.ceil(count * max_scene_share))
    ranked = sorted(
        (
            item
            for item in candidates
            if (item["scene_id"], int(item["source_frame"])) not in excluded
            and any(prediction["label"] == target_label for prediction in item["predictions"])
        ),
        key=lambda item: (
            -prediction_score(item["predictions"], target_label),
            item["scene_id"],
            int(item["source_frame"]),
        ),
    )
    selected: list[dict[str, Any]] = []
    scene_counts: Counter[str] = Counter()
    selected_frames: dict[str, list[int]] = defaultdict(list)
    for item in ranked:
        scene_id = str(item["scene_id"])
        source_frame = int(item["source_frame"])
        if scene_counts[scene_id] >= scene_cap:
            continue
        if any(abs(source_frame - other) < min_frame_gap for other in selected_frames[scene_id]):
            continue
        selected.append(item)
        scene_counts[scene_id] += 1
        selected_frames[scene_id].append(source_frame)
        if len(selected) == count:
            return selected

    raise RuntimeError(
        f"Could only select {len(selected)} of {count} {target_label} frames "
        f"with min_frame_gap={min_frame_gap} and max_scene_share={max_scene_share}"
    )


def normalize_annotations(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    annotations: list[dict[str, Any]] = []
    for prediction in predictions:
        x1, y1, x2, y2 = [round(float(value), 2) for value in prediction["bbox"]]
        annotations.append(
            {
                "prediction_id": prediction.get("prediction_id"),
                "label": str(prediction["label"]),
                "confidence": round(float(prediction.get("confidence", 1.0)), 4),
                "bbox_xyxy": [x1, y1, x2, y2],
                "bbox": [x1, y1, round(x2 - x1, 2), round(y2 - y1, 2)],
            }
        )
    return annotations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_id")
    parser.add_argument("baseline_manifest", type=Path)
    parser.add_argument("--motorcycle-frames", type=int, default=30)
    parser.add_argument("--bus-frames", type=int, default=20)
    parser.add_argument("--min-frame-gap", type=int, default=10)
    parser.add_argument("--max-scene-share", type=float, default=0.50)
    parser.add_argument("--revision", default="targeted-supplement-v1")
    args = parser.parse_args()

    if args.motorcycle_frames < 0 or args.bus_frames < 0:
        raise SystemExit("target frame counts cannot be negative")
    if args.min_frame_gap < 1:
        raise SystemExit("min frame gap must be positive")

    settings = Settings()
    store = LocalStore(settings.roadlabelops_data_dir)
    session = store.get_session(args.session_id)
    baseline_path = args.baseline_manifest.resolve()
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if baseline["session_id"] != session.session_id:
        raise SystemExit("baseline manifest belongs to a different session")

    existing = {(str(item["scene_id"]), int(item["source_frame"])) for item in baseline["samples"]}
    scene_by_id = {scene.scene_id: scene for scene in session.scenes}
    candidates: list[dict[str, Any]] = []
    for scene in session.scenes:
        prediction_path = Path(scene.video_path).with_suffix(".predictions.json")
        predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
        by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for prediction in predictions:
            by_frame[int(prediction["frame"])].append(prediction)
        for source_frame, frame_predictions in by_frame.items():
            if (scene.scene_id, source_frame) in existing:
                continue
            candidates.append(
                {
                    "scene_id": scene.scene_id,
                    "source_frame": source_frame,
                    "predictions": frame_predictions,
                }
            )

    motorcycle = select_target_frames(
        candidates,
        target_label="motorcycle",
        count=args.motorcycle_frames,
        min_frame_gap=args.min_frame_gap,
        max_scene_share=args.max_scene_share,
    )
    selected_keys = {(item["scene_id"], int(item["source_frame"])) for item in motorcycle}
    bus = select_target_frames(
        candidates,
        target_label="bus",
        count=args.bus_frames,
        min_frame_gap=args.min_frame_gap,
        max_scene_share=args.max_scene_share,
        excluded=selected_keys,
    )
    selected = [
        *({**item, "selection_target": "motorcycle"} for item in motorcycle),
        *({**item, "selection_target": "bus"} for item in bus),
    ]
    selected.sort(key=lambda item: (item["scene_id"], int(item["source_frame"])))

    safe_revision = args.revision.replace("/", "-").replace("..", "-")
    output_root = store.sessions_dir / f"{session.session_id}.training-{safe_revision}"
    image_dir = output_root / "images"
    overlay_dir = output_root / "overlays"
    contact_sheet_dir = output_root / "contact-sheets"

    targets_by_scene: dict[str, dict[int, Path]] = defaultdict(dict)
    for item in selected:
        file_name = f"{item['scene_id']}_frame_{int(item['source_frame']):06d}.jpg"
        item["file_name"] = file_name
        targets_by_scene[item["scene_id"]][int(item["source_frame"])] = image_dir / file_name
    for scene_id, targets in targets_by_scene.items():
        extract_frames(Path(scene_by_id[scene_id].video_path), targets)

    samples: list[dict[str, Any]] = []
    cvat_shapes: list[dict[str, Any]] = []
    for item in selected:
        sample_index = len(samples) + 1
        annotations = normalize_annotations(item["predictions"])
        samples.append(
            {
                "sample_index": sample_index,
                "scene_id": item["scene_id"],
                "source_frame": int(item["source_frame"]),
                "file_name": item["file_name"],
                "selection_target": item["selection_target"],
                "selection_score": round(
                    prediction_score(item["predictions"], item["selection_target"]), 4
                ),
                "annotations": annotations,
            }
        )
        cvat_shapes.extend(
            {
                "frame": sample_index - 1,
                "label": annotation["label"],
                "confidence": annotation["confidence"],
                "bbox": annotation["bbox_xyxy"],
            }
            for annotation in annotations
        )

    overlay_paths = [make_overlay(sample, image_dir, overlay_dir, "training") for sample in samples]
    contact_sheets = make_contact_sheets(
        samples, overlay_paths, contact_sheet_dir, "training-supplement"
    )
    cvat = create_cvat_task(
        settings,
        session.name,
        session.session_id,
        safe_revision,
        "training",
        len(samples),
        [image_dir / sample["file_name"] for sample in samples],
        cvat_shapes,
    )
    class_counts = Counter(
        annotation["label"] for sample in samples for annotation in sample["annotations"]
    )
    manifest = {
        "session_id": session.session_id,
        "source_sha256": session.source_sha256,
        "purpose": "training",
        "sampling_revision": safe_revision,
        "method": "Class-targeted density ranking with exact-frame exclusion, temporal spacing, and per-scene caps",
        "baseline_manifest": str(baseline_path),
        "baseline_task_id": int(baseline["cvat"]["task_id"]),
        "selection_constraints": {
            "motorcycle_frames": args.motorcycle_frames,
            "bus_frames": args.bus_frames,
            "min_frame_gap_within_supplement": args.min_frame_gap,
            "max_scene_share_per_target": args.max_scene_share,
            "exact_baseline_frame_overlap": 0,
        },
        "sample_size": len(samples),
        "sample_counts_by_scene": dict(Counter(item["scene_id"] for item in samples)),
        "sample_counts_by_target": dict(Counter(item["selection_target"] for item in samples)),
        "predicted_box_count": len(cvat_shapes),
        "predicted_class_counts": dict(class_counts),
        "samples": samples,
        "cvat": cvat,
        "contact_sheets": [str(path.resolve()) for path in contact_sheets],
    }
    manifest_path = output_root / "sample-manifest.json"
    store.write_json_atomic(manifest_path, manifest)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "sample_size": len(samples),
                "predicted_box_count": len(cvat_shapes),
                "predicted_class_counts": dict(class_counts),
                "cvat": cvat,
                "contact_sheet_count": len(contact_sheets),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
