from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from scripts.apply_final_review import (
    FinalReviewApplyError,
    build_apply_plan,
    post_apply_canonical_sha256,
    shape_request_from_record,
    validate_review_evidence,
    verify_live_annotations,
    verify_live_labels,
)
from scripts.snapshot_cvat_task import canonical_sha256, canonicalize_annotations
from scripts.validate_full_review_decisions import (
    MANUAL_DELETE_APPROVAL_TYPE,
    MANUAL_DELETE_REVIEW_DECISION,
    canonical_shape_sha256,
    manual_delete_approval_sha256,
)

SNAPSHOT_FILE_SHA = "a" * 64
REVIEW_PACK_FILE_SHA = "b" * 64


class ToDict:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def to_dict(self) -> dict:
        return copy.deepcopy(self.payload)


def build_evidence() -> tuple[dict, dict, dict]:
    shapes = [
        {
            "type": "rectangle",
            "label_id": 10,
            "frame": 0,
            "occluded": True,
            "outside": False,
            "z_order": 3,
            "rotation": 2.5,
            "points": [1.0, 2.0, 20.0, 22.0],
            "id": 101,
            "group": 7,
            "source": "auto",
            "attributes": [{"spec_id": 4, "value": "yes"}],
            "score": 0.73,
            "elements": [],
        },
        {
            "type": "rectangle",
            "label_id": 10,
            "frame": 0,
            "occluded": False,
            "outside": False,
            "z_order": 0,
            "rotation": 0.0,
            "points": [25.0, 2.0, 45.0, 22.0],
            "id": 102,
            "group": 0,
            "source": "manual",
            "attributes": [],
            "score": 1.0,
            "elements": [],
        },
        {
            "type": "rectangle",
            "label_id": 11,
            "frame": 1,
            "occluded": False,
            "outside": False,
            "z_order": 1,
            "rotation": 0.0,
            "points": [30.0, 10.0, 60.0, 40.0],
            "id": 103,
            "group": 0,
            "source": "auto",
            "attributes": [],
            "score": 0.91,
            "elements": [],
        },
    ]
    annotations = {
        "version": 8,
        "tags": [],
        "shapes": shapes,
        "tracks": [],
        "intervals": [],
    }
    annotation_sha = canonical_sha256(canonicalize_annotations(annotations))
    snapshot = {
        "snapshot_schema": {"name": "roadlabelops.cvat-task-snapshot", "version": 1},
        "task": {"id": 42},
        "labels": [{"id": 10, "name": "car"}, {"id": 11, "name": "bus"}],
        "images": [
            {"cvat_frame": 0, "width": 100, "height": 80},
            {"cvat_frame": 1, "width": 100, "height": 80},
        ],
        "annotations": annotations,
        "canonical_annotations_sha256": annotation_sha,
    }
    review_pack = {
        "schema_version": "1.0",
        "task_id": 42,
        "read_only": True,
        "source_snapshot_sha256": SNAPSHOT_FILE_SHA,
        "annotation_sha256": annotation_sha,
        "frames": [
            {
                "frame": 0,
                "width": 100,
                "height": 80,
                "shapes": [
                    {**copy.deepcopy(shapes[0]), "label": "car"},
                    {**copy.deepcopy(shapes[1]), "label": "car"},
                ],
                "flags": [],
            },
            {
                "frame": 1,
                "width": 100,
                "height": 80,
                "shapes": [{**copy.deepcopy(shapes[2]), "label": "bus"}],
                "flags": [],
            },
        ],
    }
    decisions = {
        "schema_version": "1.0",
        "scope": "full_review",
        "task_id": 42,
        "snapshot_sha256": SNAPSHOT_FILE_SHA,
        "canonical_annotations_sha256": annotation_sha,
        "review_pack_sha256": REVIEW_PACK_FILE_SHA,
        "frame_reviews": [
            {
                "frame": 0,
                "reviewed": True,
                "resolved_flag_ids": [],
                "actions": [
                    {
                        "action": "delete",
                        "shape_id": 101,
                        "expected_shape_sha256": canonical_shape_sha256(shapes[0]),
                    },
                    {
                        "action": "relabel",
                        "shape_id": 102,
                        "expected_shape_sha256": canonical_shape_sha256(shapes[1]),
                        "to_label": "bus",
                    },
                ],
            },
            {
                "frame": 1,
                "reviewed": True,
                "resolved_flag_ids": [],
                "actions": [
                    {
                        "action": "keep_distinct",
                        "shape_id": 103,
                        "expected_shape_sha256": canonical_shape_sha256(shapes[2]),
                    },
                    {
                        "action": "add",
                        "label": "car",
                        "points": [65, 20, 90, 70],
                    },
                ],
            },
        ],
    }
    return snapshot, review_pack, decisions


def build_plan(snapshot: dict, review_pack: dict, decisions: dict) -> dict:
    return build_apply_plan(
        snapshot,
        review_pack,
        decisions,
        snapshot_file_sha256=SNAPSHOT_FILE_SHA,
        review_pack_file_sha256=REVIEW_PACK_FILE_SHA,
    )


def manual_delete_approval(shape: dict) -> dict:
    approval = {
        "approval_sha256": "0" * 64,
        "approval_type": MANUAL_DELETE_APPROVAL_TYPE,
        "shape_id": shape["id"],
        "frame": shape["frame"],
        "canonical_shape_sha256": canonical_shape_sha256(shape),
        "reason": "Independent visual review confirmed this manual box is erroneous.",
        "reviewers": [
            {
                "reviewer_id": "reviewer-alpha",
                "reviewed_at": "2026-09-01T12:00:00+08:00",
                "decision": MANUAL_DELETE_REVIEW_DECISION,
            },
            {
                "reviewer_id": "reviewer-beta",
                "reviewed_at": "2026-09-01T12:05:00+08:00",
                "decision": MANUAL_DELETE_REVIEW_DECISION,
            },
        ],
    }
    approval["approval_sha256"] = manual_delete_approval_sha256(approval, task_id=42)
    return approval


def test_plan_applies_only_explicit_actions_and_preserves_complete_shape_fields() -> None:
    snapshot, review_pack, decisions = build_evidence()

    plan = build_plan(snapshot, review_pack, decisions)

    assert plan["action_counts"] == {
        "add": 1,
        "delete": 1,
        "keep_distinct": 1,
        "relabel": 1,
    }
    assert plan["mutation_action_count"] == 3
    after_by_id = {shape["id"]: shape for shape in plan["expected_annotations"]["shapes"]}
    assert 101 not in after_by_id
    assert after_by_id[102]["label_id"] == 11
    expected_relabel = copy.deepcopy(snapshot["annotations"]["shapes"][1])
    expected_relabel["label_id"] = 11
    assert after_by_id[102] == expected_relabel
    assert plan["update_shapes"] == [expected_relabel]
    assert after_by_id[103] == snapshot["annotations"]["shapes"][2]
    assert after_by_id[None] == {
        "type": "rectangle",
        "label_id": 10,
        "frame": 1,
        "occluded": False,
        "outside": False,
        "z_order": 0,
        "rotation": 0.0,
        "points": [65.0, 20.0, 90.0, 70.0],
        "id": None,
        "group": 0,
        "source": "manual",
        "attributes": [],
        "score": 1.0,
        "elements": [],
    }
    after_delete = copy.deepcopy(snapshot["annotations"])
    after_delete["shapes"] = [shape for shape in after_delete["shapes"] if shape["id"] != 101]
    after_update = copy.deepcopy(after_delete)
    next(shape for shape in after_update["shapes"] if shape["id"] == 102)["label_id"] = 11
    assert plan["stage_hashes"]["delete"] == post_apply_canonical_sha256(
        after_delete, original_shape_ids=plan["original_shape_ids"]
    )
    assert plan["stage_hashes"]["update"] == post_apply_canonical_sha256(
        after_update, original_shape_ids=plan["original_shape_ids"]
    )


def test_bbox_actions_preserve_shape_fields_and_build_the_exact_update_stage() -> None:
    snapshot, review_pack, decisions = build_evidence()
    original_by_id = {shape["id"]: shape for shape in snapshot["annotations"]["shapes"]}
    decisions["frame_reviews"][0]["actions"] = [
        {
            "action": "update_bbox",
            "shape_id": 101,
            "expected_shape_sha256": canonical_shape_sha256(original_by_id[101]),
            "points": [0, 1, 22, 24],
        },
        {
            "action": "relabel_bbox",
            "shape_id": 102,
            "expected_shape_sha256": canonical_shape_sha256(original_by_id[102]),
            "to_label": "bus",
            "points": [24, 1, 47, 24],
        },
    ]

    plan = build_plan(snapshot, review_pack, decisions)

    expected_101 = copy.deepcopy(original_by_id[101])
    expected_101["points"] = [0.0, 1.0, 22.0, 24.0]
    expected_102 = copy.deepcopy(original_by_id[102])
    expected_102["label_id"] = 11
    expected_102["points"] = [24.0, 1.0, 47.0, 24.0]
    updates_by_id = {shape["id"]: shape for shape in plan["update_shapes"]}
    after_by_id = {shape["id"]: shape for shape in plan["expected_annotations"]["shapes"]}

    assert plan["action_counts"] == {
        "add": 1,
        "keep_distinct": 1,
        "relabel_bbox": 1,
        "update_bbox": 1,
    }
    assert plan["mutation_action_count"] == 3
    assert updates_by_id == {101: expected_101, 102: expected_102}
    assert after_by_id[101] == expected_101
    assert after_by_id[102] == expected_102
    assert after_by_id[102]["id"] == 102
    assert after_by_id[102]["source"] == "manual"

    after_update = copy.deepcopy(snapshot["annotations"])
    for shape in after_update["shapes"]:
        if shape["id"] == 101:
            shape.update(expected_101)
        elif shape["id"] == 102:
            shape.update(expected_102)
    assert plan["stage_hashes"]["delete"] == plan["stage_hashes"]["initial"]
    assert plan["stage_hashes"]["update"] == post_apply_canonical_sha256(
        after_update, original_shape_ids=plan["original_shape_ids"]
    )
    assert plan["stage_hashes"]["add"] == plan["expected_post_apply_canonical_sha256"]
    logged_by_id = {
        action["shape_id"]: action
        for action in plan["action_log"]
        if action.get("shape_id") in {101, 102}
    }
    assert logged_by_id[101]["after_shape"] == expected_101
    assert logged_by_id[102]["after_shape"] == expected_102


@pytest.mark.parametrize(
    "original_points",
    [
        [-5.0, 2.0, 45.0, 22.0],
        [25.0, 2.0, 25.0, 22.0],
    ],
)
def test_bbox_update_can_repair_an_out_of_bounds_or_degenerate_snapshot_shape(
    original_points: list[float],
) -> None:
    snapshot, review_pack, decisions = build_evidence()
    shape = snapshot["annotations"]["shapes"][1]
    shape["points"] = original_points
    review_pack["frames"][0]["shapes"][1]["points"] = copy.deepcopy(original_points)
    annotation_sha = canonical_sha256(canonicalize_annotations(snapshot["annotations"]))
    snapshot["canonical_annotations_sha256"] = annotation_sha
    review_pack["annotation_sha256"] = annotation_sha
    decisions["canonical_annotations_sha256"] = annotation_sha
    decisions["frame_reviews"][0]["actions"][1] = {
        "action": "update_bbox",
        "shape_id": shape["id"],
        "expected_shape_sha256": canonical_shape_sha256(shape),
        "points": [24, 1, 47, 24],
    }

    plan = build_plan(snapshot, review_pack, decisions)

    updated = next(item for item in plan["update_shapes"] if item["id"] == shape["id"])
    assert updated["points"] == [24.0, 1.0, 47.0, 24.0]
    assert updated["id"] == shape["id"]
    assert updated["source"] == "manual"


def test_post_apply_hash_normalizes_only_version_and_new_server_ids() -> None:
    snapshot, review_pack, decisions = build_evidence()
    plan = build_plan(snapshot, review_pack, decisions)
    readback = copy.deepcopy(plan["expected_annotations"])
    readback["version"] = 12
    new_shape = next(shape for shape in readback["shapes"] if shape["id"] is None)
    new_shape["id"] = 9999

    assert (
        post_apply_canonical_sha256(readback, original_shape_ids=plan["original_shape_ids"])
        == plan["expected_post_apply_canonical_sha256"]
    )

    existing = next(shape for shape in readback["shapes"] if shape["id"] == 102)
    existing["id"] = 8888
    assert (
        post_apply_canonical_sha256(readback, original_shape_ids=plan["original_shape_ids"])
        != plan["expected_post_apply_canonical_sha256"]
    )


def test_wrong_review_pack_hash_is_rejected_before_plan() -> None:
    snapshot, review_pack, decisions = build_evidence()
    decisions["review_pack_sha256"] = "c" * 64

    with pytest.raises(FinalReviewApplyError, match="review_pack_sha256 does not match"):
        build_plan(snapshot, review_pack, decisions)


def test_all_frames_reviewed_with_zero_resolved_flags_is_rejected() -> None:
    snapshot, review_pack, decisions = build_evidence()
    review_pack["frames"][0]["flags"] = [
        {
            "id": "manual-check-0",
            "flag_id": "manual-check-0",
            "type": "manual_class_check",
            "frame": 0,
            "shape_ids": [],
        }
    ]
    assert all(review["reviewed"] for review in decisions["frame_reviews"])
    assert not any(review["resolved_flag_ids"] for review in decisions["frame_reviews"])

    with pytest.raises(FinalReviewApplyError, match="every flag resolved"):
        build_plan(snapshot, review_pack, decisions)


def test_automated_cleanup_scope_is_rejected_before_plan() -> None:
    snapshot, review_pack, decisions = build_evidence()
    decisions["scope"] = "automated_risk_cleanup"

    with pytest.raises(FinalReviewApplyError, match="scope must be 'full_review'"):
        build_plan(snapshot, review_pack, decisions)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda _snapshot, decisions: decisions.__setitem__("snapshot_sha256", "b" * 64),
            "does not match the snapshot file",
        ),
        (
            lambda _snapshot, decisions: decisions.__setitem__(
                "canonical_annotations_sha256", "b" * 64
            ),
            "does not match the snapshot",
        ),
        (
            lambda _snapshot, decisions: decisions.__setitem__("task_id", 99),
            "task_id mismatch",
        ),
        (
            lambda snapshot, _decisions: snapshot["annotations"]["tracks"].append({"id": 1}),
            "canonical_annotations_sha256 does not match",
        ),
        (
            lambda _snapshot, decisions: decisions.update({"accepted_rules": []}),
            "forbidden rank/range/rules",
        ),
    ],
)
def test_plan_rejects_stale_wrong_or_non_explicit_evidence(mutate, match: str) -> None:
    snapshot, review_pack, decisions = build_evidence()
    mutate(snapshot, decisions)

    with pytest.raises(FinalReviewApplyError, match=match):
        build_plan(snapshot, review_pack, decisions)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda action: action.__setitem__("shape_id", 999),
            "unknown shape",
        ),
        (
            lambda action: action.__setitem__("expected_shape_sha256", "b" * 64),
            "does not match snapshot shape",
        ),
        (
            lambda action: action.__setitem__("action", "bulk_delete"),
            "action must be one of",
        ),
    ],
)
def test_existing_shape_action_rejects_unknown_id_hash_drift_and_implicit_action(
    mutate, match: str
) -> None:
    snapshot, review_pack, decisions = build_evidence()
    action = decisions["frame_reviews"][0]["actions"][0]
    mutate(action)

    with pytest.raises(FinalReviewApplyError, match=match):
        build_plan(snapshot, review_pack, decisions)


def test_existing_shape_action_rejects_wrong_frame_and_duplicate_action() -> None:
    snapshot, review_pack, decisions = build_evidence()
    action = decisions["frame_reviews"][0]["actions"][0]
    decisions["frame_reviews"][0]["actions"].append(copy.deepcopy(action))
    with pytest.raises(FinalReviewApplyError, match="more than one action"):
        build_plan(snapshot, review_pack, decisions)

    snapshot, review_pack, decisions = build_evidence()
    decisions["frame_reviews"][0]["actions"][0] = copy.deepcopy(
        decisions["frame_reviews"][1]["actions"][0]
    )
    with pytest.raises(FinalReviewApplyError, match="belongs to frame 1, not 0"):
        build_plan(snapshot, review_pack, decisions)


def test_delete_rejects_manual_shape_while_manual_relabel_remains_allowed() -> None:
    snapshot, review_pack, decisions = build_evidence()
    manual_shape = snapshot["annotations"]["shapes"][1]
    decisions["frame_reviews"][0]["actions"][1] = {
        "action": "delete",
        "shape_id": manual_shape["id"],
        "expected_shape_sha256": canonical_shape_sha256(manual_shape),
    }

    with pytest.raises(
        FinalReviewApplyError,
        match=r"manual-delete approvals must correspond one-for-one.*missing=\[102\]",
    ):
        build_plan(snapshot, review_pack, decisions)


def test_delete_allows_only_exact_hash_bound_dual_review_manual_shape() -> None:
    snapshot, review_pack, decisions = build_evidence()
    manual_shape = snapshot["annotations"]["shapes"][1]
    decisions["frame_reviews"][0]["actions"][1] = {
        "action": "delete",
        "shape_id": manual_shape["id"],
        "expected_shape_sha256": canonical_shape_sha256(manual_shape),
    }
    decisions["manual_delete_approvals"] = [manual_delete_approval(manual_shape)]

    plan = build_plan(snapshot, review_pack, decisions)

    assert plan["manual_delete_approval_count"] == 1
    assert plan["action_counts"] == {
        "add": 1,
        "delete": 2,
        "keep_distinct": 1,
    }
    assert {shape["id"] for shape in plan["delete_shapes"]} == {101, 102}
    assert 102 not in {shape["id"] for shape in plan["expected_annotations"]["shapes"]}


def test_apply_rejects_tampered_manual_delete_approval_before_planning() -> None:
    snapshot, review_pack, decisions = build_evidence()
    manual_shape = snapshot["annotations"]["shapes"][1]
    decisions["frame_reviews"][0]["actions"][1] = {
        "action": "delete",
        "shape_id": manual_shape["id"],
        "expected_shape_sha256": canonical_shape_sha256(manual_shape),
    }
    approval = manual_delete_approval(manual_shape)
    approval["reason"] = "Tampered after approval."
    decisions["manual_delete_approvals"] = [approval]

    with pytest.raises(FinalReviewApplyError, match="approval_sha256 does not match"):
        build_plan(snapshot, review_pack, decisions)


def test_full_review_apply_requires_every_snapshot_frame() -> None:
    snapshot, review_pack, decisions = build_evidence()
    decisions["frame_reviews"].pop()

    with pytest.raises(FinalReviewApplyError, match="every frame reviewed"):
        build_plan(snapshot, review_pack, decisions)


@pytest.mark.parametrize(
    ("points", "match"),
    [
        ([1, 2, 1, 4], "positive width"),
        ([-1, 2, 10, 20], "outside frame"),
        ([1, 2, 3], "four coordinates"),
    ],
)
def test_add_requires_an_explicit_valid_in_bounds_rectangle(points, match: str) -> None:
    snapshot, review_pack, decisions = build_evidence()
    decisions["frame_reviews"][1]["actions"][1]["points"] = points

    with pytest.raises(FinalReviewApplyError, match=match):
        build_plan(snapshot, review_pack, decisions)


@pytest.mark.parametrize(
    ("points", "match"),
    [
        ([25, 2, 45, 22], "equals the shape's current bounding box"),
        ([-1, 2, 45, 22], "falls outside frame bounds"),
    ],
)
def test_bbox_update_rejects_unchanged_or_out_of_bounds_points(points, match: str) -> None:
    snapshot, review_pack, decisions = build_evidence()
    shape = snapshot["annotations"]["shapes"][1]
    decisions["frame_reviews"][0]["actions"][1] = {
        "action": "update_bbox",
        "shape_id": shape["id"],
        "expected_shape_sha256": canonical_shape_sha256(shape),
        "points": points,
    }

    with pytest.raises(FinalReviewApplyError, match=match):
        build_plan(snapshot, review_pack, decisions)


def test_review_evidence_paths_are_relative_hash_bound_and_returned_unchanged(
    tmp_path: Path,
) -> None:
    decisions_path = tmp_path / "decisions.json"
    evidence_path = tmp_path / "evidence" / "candidate-pack.json"
    evidence_path.parent.mkdir()
    encoded = b'{"read_only":true}\n'
    evidence_path.write_bytes(encoded)
    record = {
        "path": "evidence/candidate-pack.json",
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }

    assert validate_review_evidence(
        {"review_evidence": [record]}, decisions_path=decisions_path
    ) == [record]


@pytest.mark.parametrize("case", ["drift", "missing", "duplicate", "absolute"])
def test_review_evidence_rejects_drift_missing_duplicate_and_absolute_paths(
    tmp_path: Path, case: str
) -> None:
    decisions_path = tmp_path / "decisions.json"
    evidence_path = tmp_path / "candidate-pack.json"
    original = b'{"version":1}\n'
    evidence_path.write_bytes(original)
    record = {
        "path": evidence_path.name,
        "sha256": hashlib.sha256(original).hexdigest(),
    }
    records = [record]
    match = ""
    if case == "drift":
        evidence_path.write_bytes(b'{"version":2}\n')
        match = "sha256 does not match"
    elif case == "missing":
        record["path"] = "missing.json"
        match = "does not exist"
    elif case == "duplicate":
        records.append({**record, "path": f"./{evidence_path.name}"})
        match = "duplicate review evidence"
    else:
        record["path"] = str(evidence_path.resolve())
        match = "must be relative"

    with pytest.raises(FinalReviewApplyError, match=match):
        validate_review_evidence({"review_evidence": records}, decisions_path=decisions_path)


def test_live_verification_uses_snapshot_canonicalization_and_rejects_drift() -> None:
    snapshot, _review_pack, _decisions = build_evidence()
    reordered = copy.deepcopy(snapshot["annotations"])
    reordered["shapes"].reverse()
    live = ToDict(reordered)

    assert verify_live_annotations(snapshot, live) == canonicalize_annotations(
        snapshot["annotations"]
    )

    drifted = copy.deepcopy(reordered)
    drifted["shapes"][0]["points"][0] += 1
    with pytest.raises(FinalReviewApplyError, match="changed after the snapshot"):
        verify_live_annotations(snapshot, ToDict(drifted))

    tagged = copy.deepcopy(reordered)
    tagged["tags"] = [{"id": 1, "frame": 0, "label_id": 10}]
    with pytest.raises(FinalReviewApplyError, match="tags or tracks"):
        verify_live_annotations(snapshot, ToDict(tagged))


def test_live_labels_and_shape_request_preserve_exact_supported_fields() -> None:
    snapshot, _review_pack, _decisions = build_evidence()
    verify_live_labels(
        snapshot,
        [ToDict({"id": 10, "name": "car"}), ToDict({"id": 11, "name": "bus"})],
    )
    with pytest.raises(FinalReviewApplyError, match="labels changed"):
        verify_live_labels(
            snapshot,
            [ToDict({"id": 10, "name": "truck"}), ToDict({"id": 11, "name": "bus"})],
        )

    shape = snapshot["annotations"]["shapes"][0]
    assert shape_request_from_record(shape, include_id=True).to_dict() == shape
    without_id = shape_request_from_record(shape, include_id=False).to_dict()
    assert "id" not in without_id
    assert {key: value for key, value in shape.items() if key != "id"} == without_id
