from pathlib import Path

from roadlabelops.tools.detection import (
    MODEL_MAPPING,
    ROAD_LABELS,
    postprocess_predictions,
    result_frame_index,
    run_yolo_detection,
)


def prediction(
    identifier: str,
    label: str,
    confidence: float,
    bbox: list[float],
) -> dict:
    return {
        "prediction_id": identifier,
        "frame": 0,
        "label": label,
        "confidence": confidence,
        "bbox": bbox,
        "source": "auto",
    }


def test_postprocess_removes_cross_class_duplicate() -> None:
    predictions = [
        prediction("car", "car", 0.81, [10, 10, 110, 110]),
        prediction("truck", "truck", 0.62, [11, 11, 111, 111]),
        prediction("pedestrian", "pedestrian", 0.75, [130, 20, 160, 100]),
    ]

    processed, metrics = postprocess_predictions(predictions)

    assert [item["prediction_id"] for item in processed] == ["car", "pedestrian"]
    assert metrics == {
        "input_prediction_count": 3,
        "nms_suppressed_count": 1,
        "rider_suppressed_count": 0,
        "prediction_count": 2,
    }


def test_postprocess_suppresses_rider_but_keeps_nearby_pedestrian() -> None:
    predictions = [
        prediction("rider", "pedestrian", 0.72, [40, 10, 80, 100]),
        prediction("walker", "pedestrian", 0.78, [120, 10, 150, 100]),
        prediction("motorcycle", "motorcycle", 0.69, [30, 60, 100, 115]),
    ]

    processed, metrics = postprocess_predictions(predictions)

    assert [item["prediction_id"] for item in processed] == ["walker", "motorcycle"]
    assert metrics["rider_suppressed_count"] == 1


def test_result_frame_index_matches_ultralytics_video_stride() -> None:
    assert [result_frame_index(index, 5) for index in range(3)] == [4, 9, 14]
    assert result_frame_index(0, 1) == 0


def test_model_mapping_accepts_base_aliases_and_custom_eight_class_names() -> None:
    assert ROAD_LABELS == [
        "car",
        "bus",
        "truck",
        "motorcycle",
        "bicycle",
        "pedestrian",
        "traffic_light",
        "traffic_sign",
    ]
    assert MODEL_MAPPING["person"] == "pedestrian"
    assert MODEL_MAPPING["pedestrian"] == "pedestrian"
    assert MODEL_MAPPING["traffic light"] == "traffic_light"
    assert MODEL_MAPPING["traffic_light"] == "traffic_light"
    assert MODEL_MAPPING["stop sign"] == "traffic_sign"
    assert MODEL_MAPPING["traffic_sign"] == "traffic_sign"


def test_yolo_requires_explicit_local_weights_before_loading_runtime(
    tmp_path: Path,
) -> None:
    scene = tmp_path / "scene.mp4"
    scene.write_bytes(b"video-fixture")

    result = run_yolo_detection(scene, model_name=str(tmp_path / "missing.pt"))

    assert not result.ok
    assert result.error["code"] == "DETECTOR_MODEL_MISSING"
    assert "runtime downloads are disabled" in result.error["message"]
