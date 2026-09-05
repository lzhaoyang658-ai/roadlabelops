from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.validate_full_review_decisions import (
    MANUAL_DELETE_APPROVAL_TYPE,
    MANUAL_DELETE_REVIEW_DECISION,
    DecisionValidationError,
    canonical_annotations_sha256,
    canonical_shape_sha256,
    file_sha256,
    main,
    manual_delete_approval_sha256,
    validate_decision_files,
)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def manual_delete_approval(shape: dict) -> dict:
    approval = {
        "approval_sha256": "0" * 64,
        "approval_type": MANUAL_DELETE_APPROVAL_TYPE,
        "shape_id": shape["id"],
        "frame": shape["frame"],
        "canonical_shape_sha256": canonical_shape_sha256(shape),
        "reason": "Two independent reviewers confirmed this annotation is wrong.",
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


def build_evidence(tmp_path: Path, scope: str = "automated_risk_cleanup") -> dict:
    annotations = {
        "version": 0,
        "tags": [],
        "tracks": [],
        "shapes": [
            {
                "id": 1,
                "frame": 0,
                "label_id": 10,
                "type": "rectangle",
                "points": [1.0, 2.0, 20.0, 22.0],
                "source": "auto",
                "attributes": [],
                "elements": [],
            },
            {
                "id": 2,
                "frame": 0,
                "label_id": 10,
                "type": "rectangle",
                "points": [2.0, 3.0, 21.0, 23.0],
                "source": "auto",
                "attributes": [],
                "elements": [],
            },
            {
                "id": 3,
                "frame": 1,
                "label_id": 11,
                "type": "rectangle",
                "points": [30.0, 10.0, 60.0, 40.0],
                "source": "manual",
                "attributes": [],
                "elements": [],
            },
        ],
    }
    annotation_sha = canonical_annotations_sha256(annotations)
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
    snapshot_path = tmp_path / "snapshot.json"
    write_json(snapshot_path, snapshot)
    snapshot_sha = file_sha256(snapshot_path)

    automatic_flag = {
        "id": "flag-auto",
        "flag_id": "flag-auto",
        "type": "same_class_duplicate",
        "frame": 0,
        "shape_ids": [1, 2],
    }
    manual_zero = {
        "id": "flag-manual-0",
        "flag_id": "flag-manual-0",
        "type": "manual_class_check",
        "frame": 0,
        "shape_ids": [],
    }
    manual_one = {
        "id": "flag-manual-1",
        "flag_id": "flag-manual-1",
        "type": "manual_class_check",
        "frame": 1,
        "shape_ids": [],
    }
    pack = {
        "schema_version": "1.0",
        "task_id": 42,
        "read_only": True,
        "source_snapshot_sha256": snapshot_sha,
        "annotation_sha256": annotation_sha,
        "frames": [
            {
                "frame": 0,
                "width": 100,
                "height": 80,
                "shapes": [
                    {**annotations["shapes"][0], "label": "car"},
                    {**annotations["shapes"][1], "label": "car"},
                ],
                "flags": [automatic_flag, manual_zero],
            },
            {
                "frame": 1,
                "width": 100,
                "height": 80,
                "shapes": [{**annotations["shapes"][2], "label": "bus"}],
                "flags": [manual_one],
            },
        ],
    }
    pack_path = tmp_path / "review-pack.json"
    write_json(pack_path, pack)
    pack_sha = file_sha256(pack_path)

    decisions = {
        "schema_version": "1.0",
        "scope": scope,
        "task_id": 42,
        "snapshot_sha256": snapshot_sha,
        "canonical_annotations_sha256": annotation_sha,
        "review_pack_sha256": pack_sha,
        "frame_reviews": [
            {
                "frame": 0,
                "reviewed": True,
                "resolved_flag_ids": ["flag-auto"],
                "actions": [
                    {
                        "action": "keep_distinct",
                        "shape_id": shape["id"],
                        "expected_shape_sha256": canonical_shape_sha256(shape),
                        "resolves_flag_ids": ["flag-auto"],
                    }
                    for shape in annotations["shapes"][:2]
                ],
            }
        ],
    }
    if scope == "full_review":
        decisions["frame_reviews"][0]["resolved_flag_ids"].append("flag-manual-0")
        decisions["frame_reviews"].append(
            {
                "frame": 1,
                "reviewed": True,
                "resolved_flag_ids": ["flag-manual-1"],
                "actions": [],
            }
        )
    decisions_path = tmp_path / "decisions.json"
    write_json(decisions_path, decisions)
    return {
        "snapshot": snapshot,
        "snapshot_path": snapshot_path,
        "pack": pack,
        "pack_path": pack_path,
        "decisions": decisions,
        "decisions_path": decisions_path,
    }


def test_automated_cleanup_resolves_every_risk_but_may_leave_manual_checks(
    tmp_path: Path,
) -> None:
    evidence = build_evidence(tmp_path)

    summary = validate_decision_files(
        evidence["snapshot_path"], evidence["pack_path"], evidence["decisions_path"]
    )

    assert summary["valid"] is True
    assert summary["mutation_performed"] is False
    assert summary["scope"] == "automated_risk_cleanup"
    assert summary["snapshot_frame_count"] == 2
    assert summary["reviewed_frame_count"] == 1
    assert summary["resolved_automated_flag_count"] == 1
    assert summary["action_bound_automated_flag_count"] == 1
    assert summary["unresolved_manual_flag_count"] == 2
    assert summary["action_counts"] == {"keep_distinct": 2}


def test_automated_cleanup_cannot_use_manual_delete_approvals(tmp_path: Path) -> None:
    evidence = build_evidence(tmp_path)
    manual_shape = evidence["snapshot"]["annotations"]["shapes"][2]
    evidence["decisions"]["frame_reviews"].append(
        {
            "frame": 1,
            "reviewed": True,
            "resolved_flag_ids": [],
            "actions": [
                {
                    "action": "delete",
                    "shape_id": manual_shape["id"],
                    "expected_shape_sha256": canonical_shape_sha256(manual_shape),
                }
            ],
        }
    )
    evidence["decisions"]["manual_delete_approvals"] = [manual_delete_approval(manual_shape)]
    write_json(evidence["decisions_path"], evidence["decisions"])

    with pytest.raises(DecisionValidationError, match="allowed only in full_review"):
        validate_decision_files(
            evidence["snapshot_path"], evidence["pack_path"], evidence["decisions_path"]
        )


def test_full_review_requires_and_accepts_every_frame_and_flag(tmp_path: Path) -> None:
    evidence = build_evidence(tmp_path, scope="full_review")

    summary = validate_decision_files(
        evidence["snapshot_path"], evidence["pack_path"], evidence["decisions_path"]
    )

    assert summary["scope"] == "full_review"
    assert summary["reviewed_frame_count"] == 2
    assert summary["resolved_flag_count"] == 3
    assert summary["unresolved_manual_flag_count"] == 0


def test_all_four_action_types_are_explicit_and_supported(tmp_path: Path) -> None:
    evidence = build_evidence(tmp_path, scope="full_review")
    shapes = evidence["snapshot"]["annotations"]["shapes"]
    evidence["decisions"]["frame_reviews"][0]["actions"] = [
        {
            "action": "delete",
            "shape_id": 1,
            "expected_shape_sha256": canonical_shape_sha256(shapes[0]),
            "resolves_flag_ids": ["flag-auto"],
        },
        {
            "action": "relabel",
            "shape_id": 2,
            "expected_shape_sha256": canonical_shape_sha256(shapes[1]),
            "to_label": "bus",
        },
    ]
    evidence["decisions"]["frame_reviews"][1]["actions"] = [
        {
            "type": "keep_distinct",
            "shape_id": 3,
            "expected_shape_sha256": canonical_shape_sha256(shapes[2]),
        },
        {"action": "add", "label": "car", "points": [65, 20, 90, 70]},
    ]
    write_json(evidence["decisions_path"], evidence["decisions"])

    summary = validate_decision_files(
        evidence["snapshot_path"], evidence["pack_path"], evidence["decisions_path"]
    )

    assert summary["action_counts"] == {
        "add": 1,
        "delete": 1,
        "keep_distinct": 1,
        "relabel": 1,
    }


def test_bbox_update_actions_are_explicit_and_supported(tmp_path: Path) -> None:
    evidence = build_evidence(tmp_path, scope="full_review")
    shapes = evidence["snapshot"]["annotations"]["shapes"]
    evidence["decisions"]["frame_reviews"][0]["actions"] = [
        {
            "action": "update_bbox",
            "shape_id": 1,
            "expected_shape_sha256": canonical_shape_sha256(shapes[0]),
            "points": [0, 1, 22, 24],
            "resolves_flag_ids": ["flag-auto"],
        },
        {
            "action": "keep_distinct",
            "shape_id": 2,
            "expected_shape_sha256": canonical_shape_sha256(shapes[1]),
        },
    ]
    evidence["decisions"]["frame_reviews"][1]["actions"] = [
        {
            "action": "relabel_bbox",
            "shape_id": 3,
            "expected_shape_sha256": canonical_shape_sha256(shapes[2]),
            "to_label": "car",
            "points": [31, 11, 62, 42],
        }
    ]
    write_json(evidence["decisions_path"], evidence["decisions"])

    summary = validate_decision_files(
        evidence["snapshot_path"], evidence["pack_path"], evidence["decisions_path"]
    )

    assert summary["action_counts"] == {
        "keep_distinct": 1,
        "relabel_bbox": 1,
        "update_bbox": 1,
    }


@pytest.mark.parametrize(
    ("action", "match"),
    [
        (
            {"action": "update_bbox", "points": [30, 10, 60, 40]},
            "points equals the shape's current bounding box",
        ),
        (
            {"action": "update_bbox", "points": [-1, 10, 60, 40]},
            "falls outside frame bounds",
        ),
        (
            {
                "action": "relabel_bbox",
                "points": [31, 11, 62, 42],
                "to_label": "bus",
            },
            "relabel target equals the shape's current label",
        ),
    ],
)
def test_bbox_update_rejects_unchanged_out_of_bounds_or_same_label(
    tmp_path: Path, action: dict, match: str
) -> None:
    evidence = build_evidence(tmp_path, scope="full_review")
    shape = evidence["snapshot"]["annotations"]["shapes"][2]
    evidence["decisions"]["frame_reviews"][1]["actions"] = [
        {
            **action,
            "shape_id": shape["id"],
            "expected_shape_sha256": canonical_shape_sha256(shape),
        }
    ]
    write_json(evidence["decisions_path"], evidence["decisions"])

    with pytest.raises(DecisionValidationError, match=match):
        validate_decision_files(
            evidence["snapshot_path"], evidence["pack_path"], evidence["decisions_path"]
        )


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda value: value.__setitem__("snapshot_sha256", "0" * 64),
            "snapshot_sha256 does not match",
        ),
        (
            lambda value: value.__setitem__("review_pack_sha256", "0" * 64),
            "review_pack_sha256 does not match",
        ),
        (
            lambda value: value.__setitem__("canonical_annotations_sha256", "0" * 64),
            "canonical_annotations_sha256 does not match",
        ),
        (lambda value: value.__setitem__("task_id", 99), "task_id mismatch"),
    ],
)
def test_top_level_bindings_reject_stale_or_wrong_evidence(
    tmp_path: Path, mutate, match: str
) -> None:
    evidence = build_evidence(tmp_path)
    mutate(evidence["decisions"])
    write_json(evidence["decisions_path"], evidence["decisions"])

    with pytest.raises(DecisionValidationError, match=match):
        validate_decision_files(
            evidence["snapshot_path"], evidence["pack_path"], evidence["decisions_path"]
        )


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda value: value["frame_reviews"][0]["actions"].append(
                copy.deepcopy(value["frame_reviews"][0]["actions"][0])
            ),
            "more than one action",
        ),
        (
            lambda value: value["frame_reviews"][0]["actions"][0].__setitem__("shape_id", 999),
            "unknown shape",
        ),
        (
            lambda value: value["frame_reviews"][0]["actions"][0].__setitem__("shape_id", 3),
            "belongs to frame 1",
        ),
        (
            lambda value: value["frame_reviews"][0]["actions"][0].__setitem__(
                "expected_shape_sha256", "0" * 64
            ),
            "does not match snapshot shape",
        ),
        (
            lambda value: value["frame_reviews"][0]["actions"][0].pop("expected_shape_sha256"),
            "must be a lowercase SHA-256",
        ),
        (
            lambda value: value["frame_reviews"][0]["actions"][0].__setitem__(
                "action", "bulk_delete"
            ),
            "action must be one of",
        ),
    ],
)
def test_existing_shape_actions_reject_duplicates_unknowns_frame_drift_and_hash_drift(
    tmp_path: Path, mutate, match: str
) -> None:
    evidence = build_evidence(tmp_path)
    mutate(evidence["decisions"])
    write_json(evidence["decisions_path"], evidence["decisions"])

    with pytest.raises(DecisionValidationError, match=match):
        validate_decision_files(
            evidence["snapshot_path"], evidence["pack_path"], evidence["decisions_path"]
        )


def test_flag_resolution_rejects_unknown_duplicate_and_wrong_frame(tmp_path: Path) -> None:
    for suffix, resolved, match in (
        ("unknown", ["unknown-flag"], "unknown resolved flag"),
        ("duplicate", ["flag-auto", "flag-auto"], "resolved more than once"),
    ):
        case_dir = tmp_path / suffix
        case_dir.mkdir()
        evidence = build_evidence(case_dir)
        evidence["decisions"]["frame_reviews"][0]["resolved_flag_ids"] = resolved
        write_json(evidence["decisions_path"], evidence["decisions"])
        with pytest.raises(DecisionValidationError, match=match):
            validate_decision_files(
                evidence["snapshot_path"], evidence["pack_path"], evidence["decisions_path"]
            )

    case_dir = tmp_path / "wrong-frame"
    case_dir.mkdir()
    evidence = build_evidence(case_dir)
    evidence["decisions"]["frame_reviews"] = [
        {
            "frame": 1,
            "reviewed": True,
            "resolved_flag_ids": ["flag-auto"],
            "actions": [],
        }
    ]
    write_json(evidence["decisions_path"], evidence["decisions"])
    with pytest.raises(DecisionValidationError, match="belongs to frame 0"):
        validate_decision_files(
            evidence["snapshot_path"], evidence["pack_path"], evidence["decisions_path"]
        )


def test_scopes_enforce_their_distinct_completeness_gates(tmp_path: Path) -> None:
    cleanup_dir = tmp_path / "cleanup"
    cleanup_dir.mkdir()
    cleanup = build_evidence(cleanup_dir)
    cleanup["decisions"]["frame_reviews"][0]["resolved_flag_ids"] = []
    for action in cleanup["decisions"]["frame_reviews"][0]["actions"]:
        action["resolves_flag_ids"] = []
    write_json(cleanup["decisions_path"], cleanup["decisions"])
    with pytest.raises(DecisionValidationError, match="automated risk flags"):
        validate_decision_files(
            cleanup["snapshot_path"], cleanup["pack_path"], cleanup["decisions_path"]
        )

    frames_dir = tmp_path / "missing-frame"
    frames_dir.mkdir()
    missing_frame = build_evidence(frames_dir, scope="full_review")
    missing_frame["decisions"]["frame_reviews"].pop()
    missing_frame["decisions"]["frame_reviews"][0]["resolved_flag_ids"] = [
        "flag-auto",
        "flag-manual-0",
    ]
    write_json(missing_frame["decisions_path"], missing_frame["decisions"])
    with pytest.raises(DecisionValidationError, match="every frame reviewed"):
        validate_decision_files(
            missing_frame["snapshot_path"],
            missing_frame["pack_path"],
            missing_frame["decisions_path"],
        )

    flags_dir = tmp_path / "missing-flag"
    flags_dir.mkdir()
    missing_flag = build_evidence(flags_dir, scope="full_review")
    missing_flag["decisions"]["frame_reviews"][1]["resolved_flag_ids"] = []
    write_json(missing_flag["decisions_path"], missing_flag["decisions"])
    with pytest.raises(DecisionValidationError, match="every flag resolved"):
        validate_decision_files(
            missing_flag["snapshot_path"],
            missing_flag["pack_path"],
            missing_flag["decisions_path"],
        )


def test_automated_flags_require_an_explicit_bound_action(tmp_path: Path) -> None:
    evidence = build_evidence(tmp_path)
    evidence["decisions"]["frame_reviews"][0]["actions"] = []
    write_json(evidence["decisions_path"], evidence["decisions"])

    with pytest.raises(DecisionValidationError, match="lack an explicit bound action"):
        validate_decision_files(
            evidence["snapshot_path"], evidence["pack_path"], evidence["decisions_path"]
        )


def test_action_flag_binding_rejects_unknown_undeclared_and_unrelated_flags(
    tmp_path: Path,
) -> None:
    cases = (
        ("unknown", ["does-not-exist"], "references unknown flag"),
        ("undeclared", ["flag-manual-0"], "not listed in the frame's resolved_flag_ids"),
    )
    for suffix, flag_ids, match in cases:
        case_dir = tmp_path / suffix
        case_dir.mkdir()
        evidence = build_evidence(case_dir)
        evidence["decisions"]["frame_reviews"][0]["actions"][0]["resolves_flag_ids"] = flag_ids
        write_json(evidence["decisions_path"], evidence["decisions"])
        with pytest.raises(DecisionValidationError, match=match):
            validate_decision_files(
                evidence["snapshot_path"], evidence["pack_path"], evidence["decisions_path"]
            )

    unrelated_dir = tmp_path / "unrelated-shape"
    unrelated_dir.mkdir()
    evidence = build_evidence(unrelated_dir, scope="full_review")
    for action in evidence["decisions"]["frame_reviews"][0]["actions"]:
        action.pop("resolves_flag_ids")
    evidence["decisions"]["frame_reviews"][1]["actions"] = [
        {
            "action": "keep_distinct",
            "shape_id": 3,
            "expected_shape_sha256": canonical_shape_sha256(
                evidence["snapshot"]["annotations"]["shapes"][2]
            ),
            "resolves_flag_ids": ["flag-auto"],
        }
    ]
    evidence["decisions"]["frame_reviews"][1]["resolved_flag_ids"].append("flag-auto")
    evidence["decisions"]["frame_reviews"][0]["resolved_flag_ids"].remove("flag-auto")
    write_json(evidence["decisions_path"], evidence["decisions"])
    with pytest.raises(DecisionValidationError, match="belongs to frame 0"):
        validate_decision_files(
            evidence["snapshot_path"], evidence["pack_path"], evidence["decisions_path"]
        )


def test_automated_action_shape_must_be_referenced_by_its_flag(tmp_path: Path) -> None:
    evidence = build_evidence(tmp_path)
    evidence["pack"]["frames"][0]["flags"][0]["shape_ids"] = [2]
    write_json(evidence["pack_path"], evidence["pack"])
    evidence["decisions"]["review_pack_sha256"] = file_sha256(evidence["pack_path"])
    write_json(evidence["decisions_path"], evidence["decisions"])

    with pytest.raises(DecisionValidationError, match="is not referenced by automated flag"):
        validate_decision_files(
            evidence["snapshot_path"], evidence["pack_path"], evidence["decisions_path"]
        )


def test_automated_flag_must_reference_at_least_one_shape(tmp_path: Path) -> None:
    evidence = build_evidence(tmp_path)
    evidence["pack"]["frames"][0]["flags"][0]["shape_ids"] = []
    write_json(evidence["pack_path"], evidence["pack"])
    evidence["decisions"]["review_pack_sha256"] = file_sha256(evidence["pack_path"])
    write_json(evidence["decisions_path"], evidence["decisions"])

    with pytest.raises(
        DecisionValidationError,
        match="automated flag 'flag-auto' must reference at least one shape",
    ):
        validate_decision_files(
            evidence["snapshot_path"], evidence["pack_path"], evidence["decisions_path"]
        )


@pytest.mark.parametrize(
    "forbidden",
    [
        {"accepted_rules": []},
        {"metadata": {"rank_min": 1}},
        {"metadata": {"frame_range": [0, 1]}},
    ],
)
def test_rank_range_and_rule_syntax_is_forbidden(tmp_path: Path, forbidden: dict) -> None:
    evidence = build_evidence(tmp_path)
    evidence["decisions"].update(forbidden)
    write_json(evidence["decisions_path"], evidence["decisions"])

    with pytest.raises(DecisionValidationError, match="forbidden rank/range/rules"):
        validate_decision_files(
            evidence["snapshot_path"], evidence["pack_path"], evidence["decisions_path"]
        )


def test_cli_reads_three_inputs_prints_summary_and_never_mutates_them(
    tmp_path: Path, capsys
) -> None:
    evidence = build_evidence(tmp_path)
    before = {
        key: evidence[key].read_bytes() for key in ("snapshot_path", "pack_path", "decisions_path")
    }

    main(
        [
            str(evidence["snapshot_path"]),
            str(evidence["pack_path"]),
            str(evidence["decisions_path"]),
        ]
    )

    summary = json.loads(capsys.readouterr().out)
    assert summary["valid"] is True
    assert summary["mutation_performed"] is False
    assert all(evidence[key].read_bytes() == value for key, value in before.items())
