from roadlabelops.tools.quality import calculate_quality


def test_quality_uses_none_for_empty_denominator() -> None:
    result = calculate_quality([], [])
    assert result.ok
    assert result.data["retention_rate"] is None
    assert result.data["human_addition_rate"] is None
    assert result.data["first_pass_acceptance_rate"] is None
    assert result.data["first_pass_acceptance_reason"] is None
    assert result.data["precision"] is None
    assert result.data["recall"] is None
    assert result.data["f1_score"] is None
    assert result.data["clean_frame_rate"] is None


def test_quality_explains_uncomputable_first_pass_acceptance() -> None:
    reason = "CVAT history is unavailable."

    result = calculate_quality([], [], first_pass_acceptance_reason=reason)

    assert result.data["first_pass_acceptance_rate"] is None
    assert result.data["first_pass_acceptance_reason"] == reason


def test_quality_counts_retained_and_added() -> None:
    result = calculate_quality(
        [{"prediction_id": "p1", "label": "car"}, {"prediction_id": "p2", "label": "bus"}],
        [{"prediction_id": "p1", "label": "car"}, {"label": "pedestrian"}],
        accepted_jobs=1,
        reviewed_jobs=2,
    )
    assert result.data["retention_rate"] == 0.5
    assert result.data["human_addition_rate"] == 0.5
    assert result.data["first_pass_acceptance_rate"] == 0.5


def test_quality_does_not_retain_a_prediction_id_after_class_change() -> None:
    result = calculate_quality(
        [{"prediction_id": "p1", "label": "truck"}],
        [{"prediction_id": "p1", "label": "bus"}],
    )

    assert result.data["retained_count"] == 0
    assert result.data["removed_count"] == 1
    assert result.data["added_count"] == 1


def test_quality_matches_cvat_boxes_by_geometry() -> None:
    prediction = {
        "scene_id": "scene_1",
        "frame": 5,
        "label": "car",
        "bbox": [10, 10, 110, 110],
    }
    final = {
        "scene_id": "scene_1",
        "frame": 5,
        "label": "car",
        "bbox": [12, 12, 108, 108],
    }
    result = calculate_quality([prediction], [final])
    assert result.data["retained_count"] == 1
    assert result.data["removed_count"] == 0
    assert result.data["added_count"] == 0
    assert result.data["precision"] == 1.0
    assert result.data["recall"] == 1.0
    assert result.data["f1_score"] == 1.0
    assert result.data["clean_frame_rate"] == 1.0


def test_quality_uses_stricter_iou_for_retention_than_evaluation() -> None:
    prediction = {
        "prediction_id": "p1",
        "scene_id": "scene_1",
        "frame": 5,
        "label": "car",
        "bbox": [0, 0, 10, 10],
    }
    final = {
        "prediction_id": "p1",
        "scene_id": "scene_1",
        "frame": 5,
        "label": "car",
        "bbox": [2, 0, 12, 10],
    }

    result = calculate_quality([prediction], [final])

    # IoU is 2/3: it is a valid evaluation match at 0.50 but the box was not
    # retained unchanged under the PRD's 0.80 retention definition.
    assert result.data["retained_count"] == 0
    assert result.data["retention_rate"] == 0.0
    assert result.data["removed_count"] == 1
    assert result.data["added_count"] == 0
    assert result.data["human_addition_rate"] == 0.0
    assert result.data["precision"] == 1.0
    assert result.data["recall"] == 1.0
    assert result.data["f1_score"] == 1.0
    assert result.data["clean_frame_rate"] == 1.0


def test_prediction_id_does_not_bypass_geometry_thresholds() -> None:
    prediction = {
        "prediction_id": "p1",
        "scene_id": "scene_1",
        "frame": 5,
        "label": "car",
        "bbox": [0, 0, 10, 10],
    }
    final = {
        "prediction_id": "p1",
        "scene_id": "scene_1",
        "frame": 5,
        "label": "car",
        "bbox": [20, 20, 30, 30],
    }

    result = calculate_quality([prediction], [final])

    assert result.data["retained_count"] == 0
    assert result.data["added_count"] == 1
    assert result.data["precision"] == 0.0
    assert result.data["recall"] == 0.0
    assert result.data["clean_frame_rate"] == 0.0


def test_quality_reports_ground_truth_metrics_and_dirty_frames() -> None:
    predictions = [
        {"scene_id": "s1", "frame": 0, "label": "car", "bbox": [0, 0, 100, 100]},
        {"scene_id": "s1", "frame": 1, "label": "truck", "bbox": [0, 0, 100, 100]},
    ]
    ground_truth = [
        {"scene_id": "s1", "frame": 0, "label": "car", "bbox": [0, 0, 100, 100]},
        {"scene_id": "s1", "frame": 1, "label": "bus", "bbox": [0, 0, 100, 100]},
        {"scene_id": "s1", "frame": 1, "label": "pedestrian", "bbox": [110, 0, 140, 90]},
    ]

    result = calculate_quality(predictions, ground_truth)

    assert result.data["precision"] == 0.5
    assert result.data["recall"] == 0.3333
    assert result.data["f1_score"] == 0.4
    assert result.data["clean_frame_count"] == 1
    assert result.data["clean_frame_rate"] == 0.5
    assert result.data["per_class"]["truck"]["false_positive_count"] == 1
    assert result.data["per_class"]["bus"]["false_negative_count"] == 1


def test_quality_maximizes_matches_instead_of_taking_local_best() -> None:
    predictions = [
        {
            "scene_id": "s1",
            "frame": 0,
            "label": "car",
            "confidence": 0.9,
            "bbox": [0, 0, 10, 10],
        },
        {
            "scene_id": "s1",
            "frame": 0,
            "label": "car",
            "confidence": 0.8,
            "bbox": [-2, 0, 8, 10],
        },
    ]
    ground_truth = [
        {"scene_id": "s1", "frame": 0, "label": "car", "bbox": [0, 0, 10, 10]},
        {"scene_id": "s1", "frame": 0, "label": "car", "bbox": [2, 0, 12, 10]},
    ]

    result = calculate_quality(predictions, ground_truth)

    assert result.ok is True
    assert result.data["retained_count"] == 1
    assert result.data["precision"] == 1.0
    assert result.data["recall"] == 1.0
    assert result.data["clean_frame_rate"] == 1.0


def test_quality_counts_explicit_empty_evaluation_frames_as_clean() -> None:
    prediction = {
        "scene_id": "s1",
        "frame": 4,
        "label": "car",
        "bbox": [0, 0, 10, 10],
    }
    ground_truth = dict(prediction)

    result = calculate_quality(
        [prediction],
        [ground_truth],
        evaluated_frame_keys={("s1", 4), ("s1", 9)},
    )

    assert result.data["evaluated_frame_count"] == 2
    assert result.data["clean_frame_count"] == 2
    assert result.data["clean_frame_rate"] == 1.0


def test_quality_does_not_hide_observed_frames_outside_explicit_universe() -> None:
    prediction = {
        "scene_id": "s1",
        "frame": 14,
        "label": "car",
        "bbox": [0, 0, 10, 10],
    }

    result = calculate_quality(
        [prediction],
        [],
        evaluated_frame_keys={("s1", 4), ("s1", 9)},
    )

    assert result.data["evaluated_frame_count"] == 3
    assert result.data["clean_frame_count"] == 2
    assert result.data["clean_frame_rate"] == 0.6667
