from scripts.build_annotation_review_queue import box_iou, find_unmatched_candidates


def test_box_iou_handles_overlap_and_disjoint_boxes() -> None:
    assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert box_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert round(box_iou([0, 0, 10, 10], [5, 5, 15, 15]), 4) == 0.1429


def test_find_unmatched_candidates_requires_same_label_and_iou() -> None:
    baseline = [{"label": "car", "bbox_xyxy": [0, 0, 10, 10]}]
    candidates = [
        {"label": "car", "bbox_xyxy": [0, 0, 10, 10]},
        {"label": "bus", "bbox_xyxy": [0, 0, 10, 10]},
        {"label": "car", "bbox_xyxy": [20, 20, 30, 30]},
    ]

    assert find_unmatched_candidates(baseline, candidates, 0.5) == candidates[1:]
