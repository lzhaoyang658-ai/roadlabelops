import pytest

from scripts.apply_review_decisions import (
    RIDER_MIN_HEIGHT_RATIO,
    box_iou,
    matching_relabel_rule,
    same_box,
)
from scripts.apply_review_decisions import (
    pedestrian_is_rider as apply_pedestrian_is_rider,
)
from scripts.build_full_review_pack import (
    RIDER_MIN_HEIGHT_RATIO as PACK_RIDER_MIN_HEIGHT_RATIO,
)
from scripts.build_full_review_pack import pedestrian_is_rider as pack_pedestrian_is_rider


def test_box_iou_and_same_box() -> None:
    assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert box_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert same_box([1, 2, 3, 4], [1.01, 2.01, 3.01, 4.01])


@pytest.mark.parametrize(
    "pedestrian_is_rider",
    [apply_pedestrian_is_rider, pack_pedestrian_is_rider],
)
def test_pedestrian_rider_geometry(pedestrian_is_rider) -> None:
    assert pedestrian_is_rider([40, 10, 60, 80], [30, 50, 70, 100])
    assert not pedestrian_is_rider([0, 0, 20, 80], [50, 50, 90, 100])


@pytest.mark.parametrize(
    "pedestrian_is_rider",
    [apply_pedestrian_is_rider, pack_pedestrian_is_rider],
)
def test_pedestrian_rider_geometry_requires_compatible_height_scale(
    pedestrian_is_rider,
) -> None:
    motorcycle = [0, 0, 100, 100]

    assert pedestrian_is_rider([25, 50, 75, 100], motorcycle)
    assert not pedestrian_is_rider([25, 50.01, 75, 100], motorcycle)
    assert RIDER_MIN_HEIGHT_RATIO == PACK_RIDER_MIN_HEIGHT_RATIO == 0.50


@pytest.mark.parametrize(
    ("pedestrian", "motorcycle"),
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
def test_synthetic_background_pedestrian_is_not_a_rider(
    pedestrian: list[float], motorcycle: list[float]
) -> None:
    # Both pairs pass the legacy overlap/foot-point checks, but their height
    # ratios are only 0.160 and 0.171 because the objects are at different depths.
    assert not apply_pedestrian_is_rider(pedestrian, motorcycle)
    assert not pack_pedestrian_is_rider(pedestrian, motorcycle)


def test_matching_relabel_rule_uses_one_based_sample_range() -> None:
    rules = [
        {
            "from_label": "truck",
            "to_label": "bus",
            "sample_index_min": 1,
            "sample_index_max": 24,
        }
    ]

    assert matching_relabel_rule(24, "truck", rules) == (0, "bus")
    assert matching_relabel_rule(25, "truck", rules) is None
    assert matching_relabel_rule(10, "car", rules) is None
