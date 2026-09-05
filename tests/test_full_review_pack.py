import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.build_full_review_pack import (
    AUTOMATED_FLAG_TYPES,
    build_frame_flags,
    build_review_pack,
    class_aware_nms,
    normalize_shapes,
)


def image_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_class_aware_nms_suppresses_only_same_class_in_same_frame() -> None:
    detections = [
        {
            "prediction_id": "car-high",
            "scene_id": "scene-a",
            "frame": 0,
            "label": "car",
            "confidence": 0.9,
            "bbox": [0, 0, 20, 20],
        },
        {
            "prediction_id": "car-low",
            "scene_id": "scene-a",
            "frame": 0,
            "label": "car",
            "confidence": 0.8,
            "bbox": [0, 0, 20, 20],
        },
        {
            "prediction_id": "truck-overlap",
            "scene_id": "scene-a",
            "frame": 0,
            "label": "truck",
            "confidence": 0.7,
            "bbox": [0, 0, 20, 20],
        },
        {
            "prediction_id": "car-other-frame",
            "scene_id": "scene-a",
            "frame": 1,
            "label": "car",
            "confidence": 0.6,
            "bbox": [0, 0, 20, 20],
        },
        {
            "prediction_id": "car-other-scene",
            "scene_id": "scene-b",
            "frame": 0,
            "label": "car",
            "confidence": 0.5,
            "bbox": [0, 0, 20, 20],
        },
    ]

    kept = class_aware_nms(detections, iou_threshold=0.5)

    assert [item["prediction_id"] for item in kept] == [
        "car-high",
        "truck-overlap",
        "car-other-frame",
        "car-other-scene",
    ]


def test_frame_flags_cover_existing_risks_and_have_stable_ids() -> None:
    raw_shapes = [
        {"id": 20, "frame": 0, "label": "car", "points": [10, 10, 60, 60]},
        {"id": 10, "frame": 0, "label": "car", "points": [10, 10, 60, 60]},
        {"id": 30, "frame": 0, "label": "truck", "points": [10, 10, 60, 60]},
        {"id": 40, "frame": 0, "label": "motorcycle", "points": [70, 30, 130, 100]},
        {"id": 50, "frame": 0, "label": "pedestrian", "points": [85, 40, 110, 95]},
        {"id": 60, "frame": 0, "label": "car", "points": [100, 100, 90, 120]},
        {"id": 70, "frame": 0, "label": "bus", "points": [-2, 0, 20, 20]},
    ]
    forward = normalize_shapes(raw_shapes)
    reverse = normalize_shapes(list(reversed(raw_shapes)))

    forward_flags = build_frame_flags(0, forward, 160, 120)
    reverse_flags = build_frame_flags(0, reverse, 160, 120)

    assert [flag["id"] for flag in forward_flags] == [flag["id"] for flag in reverse_flags]
    assert len({flag["id"] for flag in forward_flags}) == len(forward_flags)
    flag_types = {flag["type"] for flag in forward_flags}
    assert AUTOMATED_FLAG_TYPES <= flag_types
    manual_checks = [flag for flag in forward_flags if flag["type"] == "manual_class_check"]
    assert {flag["label"] for flag in manual_checks} == {"traffic_light", "traffic_sign"}
    assert len(manual_checks) == 2


@pytest.mark.parametrize(
    ("pedestrian_points", "motorcycle_points"),
    [
        (
            [936.67, 337.94, 953.42, 392.45],
            [771.00, 376.82, 966.21, 717.89],
        ),
        (
            [819.08, 343.54, 836.14, 402.31],
            [824.74, 373.35, 1258.18, 716.88],
        ),
    ],
)
def test_frame_flags_ignore_depth_mismatched_pedestrian_motorcycle_pairs(
    pedestrian_points: list[float], motorcycle_points: list[float]
) -> None:
    shapes = normalize_shapes(
        [
            {"id": 1, "frame": 0, "label": "pedestrian", "points": pedestrian_points},
            {"id": 2, "frame": 0, "label": "motorcycle", "points": motorcycle_points},
        ]
    )

    flags = build_frame_flags(0, shapes, 1280, 720)

    assert not any(flag["type"] == "rider_pedestrian" for flag in flags)


def test_build_review_pack_covers_every_frame_and_refuses_overwrite(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    first_image = image_dir / "first.jpg"
    second_image = image_dir / "second.jpg"
    Image.new("RGB", (160, 120), "#334155").save(first_image, quality=95)
    Image.new("RGB", (120, 160), "#475569").save(second_image, quality=95)
    snapshot = {
        "schema_version": "1.0",
        "task_id": 42,
        "annotation_sha256": "annotations-v1",
        # Deliberately unordered and with non-contiguous sample indices.
        "images": [
            {
                "frame": 5,
                "sample_index": 9,
                "file_name": "second.jpg",
                "path": "images/second.jpg",
                "width": 120,
                "height": 160,
                "sha256": image_sha256(second_image),
            },
            {
                "frame": 0,
                "sample_index": 3,
                "file_name": "first.jpg",
                "path": "images/first.jpg",
                "width": 160,
                "height": 120,
                "sha256": image_sha256(first_image),
            },
        ],
        "shapes": [
            {"id": 2, "frame": 0, "label": "car", "points": [10, 10, 70, 70]},
            {"id": 1, "frame": 0, "label": "car", "points": [10, 10, 70, 70]},
        ],
    }
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
    snapshot_before = snapshot_path.read_bytes()
    output_dir = tmp_path / "full-review"

    result = build_review_pack(snapshot_path, output_dir)
    payload = json.loads(result.read_text(encoding="utf-8"))

    assert snapshot_path.read_bytes() == snapshot_before
    assert payload["read_only"] is True
    assert payload["mutation_performed"] is False
    assert payload["all_frames_included"] is True
    assert payload["frame_count"] == 2
    assert [frame["frame"] for frame in payload["frames"]] == [0, 5]
    assert [frame["sample_index"] for frame in payload["frames"]] == [3, 9]
    assert all(
        {flag["label"] for flag in frame["flags"] if flag["type"] == "manual_class_check"}
        == {"traffic_light", "traffic_sign"}
        for frame in payload["frames"]
    )
    assert all((output_dir / frame["overlay"]).is_file() for frame in payload["frames"])
    assert len(payload["contact_sheets"]) == 1
    assert (output_dir / payload["contact_sheets"][0]).is_file()
    assert not list(output_dir.glob(".review-pack.json.*"))

    original_pack = result.read_bytes()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        build_review_pack(snapshot_path, output_dir)
    assert result.read_bytes() == original_pack


def test_build_review_pack_accepts_canonical_nested_snapshot_schema(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "dataset"
    image_dir = manifest_dir / "images"
    image_dir.mkdir(parents=True)
    manifest_path = manifest_dir / "sample-manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    image_path = image_dir / "frame.png"
    Image.new("RGB", (64, 48), "#64748b").save(image_path)
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    snapshot_path = evidence_dir / "snapshot.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "task": {"id": 77},
                "manifest": {"path": str(manifest_path)},
                "labels": [{"id": 9, "name": "bus"}],
                "images": [
                    {
                        "cvat_frame": 0,
                        "sample_index": 1,
                        "file_name": "frame.png",
                        "relative_path": "images/frame.png",
                        "width": 64,
                        "height": 48,
                        "sha256": image_sha256(image_path),
                    }
                ],
                "annotations": {
                    "tags": [],
                    "tracks": [],
                    "shapes": [
                        {
                            "id": 1,
                            "frame": 0,
                            "label_id": 9,
                            "points": [2, 3, 30, 40],
                        }
                    ],
                },
                "canonical_annotations_sha256": "canonical-v1",
            }
        ),
        encoding="utf-8",
    )

    result = build_review_pack(snapshot_path, tmp_path / "nested-review")
    payload = json.loads(result.read_text(encoding="utf-8"))

    assert payload["task_id"] == 77
    assert payload["annotation_sha256"] == "canonical-v1"
    assert payload["frames"][0]["frame"] == 0
    assert payload["frames"][0]["shapes"][0]["label"] == "bus"


@pytest.mark.parametrize(
    ("snapshot_change", "match"),
    [
        ({"sha256": "0" * 64}, "SHA-256 differs from snapshot"),
        ({"width": 63}, "width differs from snapshot"),
        ({"height": 47}, "height differs from snapshot"),
    ],
)
def test_build_review_pack_rejects_image_integrity_drift_without_partial_output(
    tmp_path: Path, snapshot_change: dict, match: str
) -> None:
    image_path = tmp_path / "frame.png"
    Image.new("RGB", (64, 48), "#64748b").save(image_path)
    image_record = {
        "frame": 0,
        "sample_index": 1,
        "path": str(image_path),
        "width": 64,
        "height": 48,
        "sha256": image_sha256(image_path),
    }
    image_record.update(snapshot_change)
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "task_id": 42,
                "images": [image_record],
                "shapes": [],
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "full-review"

    with pytest.raises(ValueError, match=match):
        build_review_pack(snapshot_path, output_dir)

    assert not output_dir.exists()
