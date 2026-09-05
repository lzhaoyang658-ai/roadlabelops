from __future__ import annotations

import pytest

from scripts.create_targeted_supplement import (
    normalize_annotations,
    prediction_score,
    select_target_frames,
)


def candidate(scene: str, frame: int, label: str, count: int) -> dict:
    return {
        "scene_id": scene,
        "source_frame": frame,
        "predictions": [
            {
                "prediction_id": f"p-{frame}-{index}",
                "frame": frame,
                "label": label,
                "confidence": 0.5,
                "bbox": [1, 2, 11, 22],
            }
            for index in range(count)
        ],
    }


def test_prediction_score_prioritizes_density() -> None:
    dense = candidate("a", 10, "bus", 2)["predictions"]
    sparse = candidate("a", 20, "bus", 1)["predictions"]
    sparse[0]["confidence"] = 0.99

    assert prediction_score(dense, "bus") > prediction_score(sparse, "bus")


def test_select_target_frames_enforces_spacing_diversity_and_exclusion() -> None:
    candidates = [
        candidate("a", 10, "motorcycle", 5),
        candidate("a", 15, "motorcycle", 4),
        candidate("a", 30, "motorcycle", 3),
        candidate("b", 10, "motorcycle", 2),
        candidate("b", 30, "motorcycle", 1),
        candidate("b", 50, "motorcycle", 1),
    ]

    selected = select_target_frames(
        candidates,
        target_label="motorcycle",
        count=4,
        min_frame_gap=10,
        max_scene_share=0.5,
        excluded={("b", 30)},
    )

    keys = {(item["scene_id"], item["source_frame"]) for item in selected}
    assert keys == {("a", 10), ("a", 30), ("b", 10), ("b", 50)}


def test_select_target_frames_reports_infeasible_constraints() -> None:
    with pytest.raises(RuntimeError, match="Could only select"):
        select_target_frames(
            [candidate("a", 10, "bus", 2), candidate("a", 15, "bus", 1)],
            target_label="bus",
            count=2,
            min_frame_gap=10,
            max_scene_share=1,
        )


def test_normalize_annotations_converts_xyxy_to_width_height() -> None:
    annotation = normalize_annotations(candidate("a", 10, "bus", 1)["predictions"])[0]

    assert annotation["bbox_xyxy"] == [1.0, 2.0, 11.0, 22.0]
    assert annotation["bbox"] == [1.0, 2.0, 10.0, 20.0]
