from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest

from scripts import materialize_full_review_draft as materializer
from scripts.build_manual_class_candidate_pack import canonical_sha256
from scripts.compile_full_review_decisions import compile_decision_files
from scripts.materialize_full_review_draft import (
    FullReviewDraftMaterializationError,
    main,
    materialize_full_review_draft_files,
)
from scripts.validate_full_review_decisions import (
    MANUAL_DELETE_APPROVAL_TYPE,
    MANUAL_DELETE_REVIEW_DECISION,
    canonical_annotations_sha256,
    canonical_shape_sha256,
    file_sha256,
    manual_delete_approval_sha256,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def candidate_id(
    *,
    task_id: int,
    frame: int,
    label: str,
    bbox: list[float],
    model_sha256: str,
    source_sha256: str,
) -> str:
    identity = {
        "task_id": task_id,
        "frame": frame,
        "label": label,
        "bbox": bbox,
        "model_sha256": model_sha256,
        "source_image_sha256": source_sha256,
    }
    return (
        f"task-{task_id}-frame-{frame:06d}-{label.replace('_', '-')}-"
        f"{canonical_sha256(identity)[:20]}"
    )


def manual_delete_approval(shape: dict, *, task_id: int = 42) -> dict:
    approval = {
        "approval_sha256": "0" * 64,
        "approval_type": MANUAL_DELETE_APPROVAL_TYPE,
        "shape_id": shape["id"],
        "frame": shape["frame"],
        "canonical_shape_sha256": canonical_shape_sha256(shape),
        "reason": "Two reviewers confirmed this manual box is a false annotation.",
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
    approval["approval_sha256"] = manual_delete_approval_sha256(approval, task_id=task_id)
    return approval


def build_fixture(tmp_path: Path) -> dict:
    shapes = [
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
            "label_id": 10,
            "type": "rectangle",
            "points": [30.0, 10.0, 60.0, 40.0],
            "source": "manual",
            "attributes": [],
            "elements": [],
        },
    ]
    annotations = {"version": 0, "tags": [], "tracks": [], "shapes": shapes}
    annotation_sha = canonical_annotations_sha256(annotations)
    snapshot = {
        "task": {"id": 42},
        "labels": [
            {"id": 10, "name": "car"},
            {"id": 11, "name": "traffic_light"},
            {"id": 12, "name": "traffic_sign"},
        ],
        "images": [
            {"cvat_frame": 0, "width": 100, "height": 80, "sha256": "a" * 64},
            {"cvat_frame": 1, "width": 100, "height": 80, "sha256": "b" * 64},
        ],
        "annotations": annotations,
        "canonical_annotations_sha256": annotation_sha,
    }
    snapshot_path = tmp_path / "snapshot.json"
    write_json(snapshot_path, snapshot)
    snapshot_sha = file_sha256(snapshot_path)

    review_pack = {
        "schema_version": "1.0",
        "pack_type": "full_annotation_review",
        "task_id": 42,
        "read_only": True,
        "mutation_performed": False,
        "source_snapshot_sha256": snapshot_sha,
        "annotation_sha256": annotation_sha,
        "manual_check_labels": ["traffic_light", "traffic_sign"],
        "frames": [
            {
                "frame": 0,
                "width": 100,
                "height": 80,
                "shapes": [
                    {**copy.deepcopy(shapes[0]), "label": "car"},
                    {**copy.deepcopy(shapes[1]), "label": "car"},
                ],
                "flags": [
                    {
                        "flag_id": "auto-0",
                        "type": "same_class_duplicate",
                        "frame": 0,
                        "shape_ids": [1, 2],
                    },
                    {
                        "flag_id": "light-0",
                        "type": "manual_class_check",
                        "label": "traffic_light",
                        "frame": 0,
                        "shape_ids": [],
                    },
                    {
                        "flag_id": "sign-0",
                        "type": "manual_class_check",
                        "label": "traffic_sign",
                        "frame": 0,
                        "shape_ids": [],
                    },
                ],
            },
            {
                "frame": 1,
                "width": 100,
                "height": 80,
                "shapes": [{**copy.deepcopy(shapes[2]), "label": "car"}],
                "flags": [
                    {
                        "flag_id": "light-1",
                        "type": "manual_class_check",
                        "label": "traffic_light",
                        "frame": 1,
                        "shape_ids": [],
                    },
                    {
                        "flag_id": "sign-1",
                        "type": "manual_class_check",
                        "label": "traffic_sign",
                        "frame": 1,
                        "shape_ids": [],
                    },
                ],
            },
        ],
    }
    review_pack_path = tmp_path / "review-pack.json"
    write_json(review_pack_path, review_pack)
    review_pack_sha = file_sha256(review_pack_path)

    automated_decisions = {
        "schema_version": "1.1",
        "scope": "automated_risk_cleanup",
        "task_id": 42,
        "snapshot_sha256": snapshot_sha,
        "canonical_annotations_sha256": annotation_sha,
        "review_pack_sha256": review_pack_sha,
        "frame_reviews": [
            {
                "frame": 0,
                "reviewed": True,
                "resolved_flag_ids": ["auto-0"],
                "actions": [
                    {
                        "action": "delete",
                        "shape_id": 2,
                        "expected_shape_sha256": canonical_shape_sha256(shapes[1]),
                        "resolves_flag_ids": ["auto-0"],
                    }
                ],
            }
        ],
    }
    automated_path = tmp_path / "automated.json"
    write_json(automated_path, automated_decisions)

    evidence_dir = tmp_path / "candidate-evidence"
    (evidence_dir / "overlays").mkdir(parents=True)
    for artifact in (
        evidence_dir / "overlays" / "frame-000000.jpg",
        evidence_dir / "overlays" / "frame-000001.jpg",
        evidence_dir / "contact-sheet.jpg",
    ):
        artifact.write_bytes(b"fixture-image")
    model_sha = "c" * 64
    light_bbox = [70.0, 5.0, 80.0, 25.0]
    sign_bbox = [65.0, 20.0, 90.0, 70.0]
    existing_bbox = [1.0, 2.0, 20.0, 22.0]
    light_id = candidate_id(
        task_id=42,
        frame=0,
        label="traffic_light",
        bbox=light_bbox,
        model_sha256=model_sha,
        source_sha256="a" * 64,
    )
    sign_id = candidate_id(
        task_id=42,
        frame=1,
        label="traffic_sign",
        bbox=sign_bbox,
        model_sha256=model_sha,
        source_sha256="b" * 64,
    )
    existing_id = candidate_id(
        task_id=42,
        frame=0,
        label="car",
        bbox=existing_bbox,
        model_sha256=model_sha,
        source_sha256="a" * 64,
    )
    candidate_pack = {
        "schema_version": "1.0",
        "pack_type": "manual_class_model_candidates",
        "task_id": 42,
        "read_only": True,
        "reviewed_by_human": False,
        "mutation_performed": False,
        "source_review_pack": str(review_pack_path),
        "source_snapshot_sha256": snapshot_sha,
        "source_review_pack_sha256": review_pack_sha,
        "model": "fixture-model.pt",
        "model_sha256": model_sha,
        "model_label_mapping": {},
        "review_labels": ["car", "traffic_light", "traffic_sign"],
        "parameters": {},
        "frame_count": 2,
        "frames_with_candidates": 2,
        "frames_needing_human_review": 2,
        "candidate_count": 3,
        "needs_human_review_count": 2,
        "candidate_counts_by_label": {"car": 1, "traffic_light": 1, "traffic_sign": 1},
        "needs_human_review_counts_by_label": {"traffic_light": 1, "traffic_sign": 1},
        "needs_human_review_counts_by_reason": {"no_same_label_match": 2},
        "frames": [
            {
                "frame": 0,
                "sample_index": 1,
                "source_path": "frame-0.jpg",
                "source_sha256": "a" * 64,
                "candidate_count": 2,
                "needs_human_review_count": 1,
                "candidates": [
                    {
                        "candidate_id": light_id,
                        "frame": 0,
                        "label": "traffic_light",
                        "model_label": "traffic light",
                        "confidence": 0.91,
                        "bbox": light_bbox,
                        "status": "needs_human_review",
                        "review_reason": "no_same_label_match",
                        "existing_overlaps": [],
                        "mutation_performed": False,
                    },
                    {
                        "candidate_id": existing_id,
                        "frame": 0,
                        "label": "car",
                        "model_label": "car",
                        "confidence": 0.99,
                        "bbox": existing_bbox,
                        "status": "already_annotated",
                        "review_reason": "same_label_match",
                        "existing_overlaps": [],
                        "mutation_performed": False,
                    },
                ],
                "overlay": "overlays/frame-000000.jpg",
            },
            {
                "frame": 1,
                "sample_index": 2,
                "source_path": "frame-1.jpg",
                "source_sha256": "b" * 64,
                "candidate_count": 1,
                "needs_human_review_count": 1,
                "candidates": [
                    {
                        "candidate_id": sign_id,
                        "frame": 1,
                        "label": "traffic_sign",
                        "model_label": "traffic sign",
                        "confidence": 0.88,
                        "bbox": sign_bbox,
                        "status": "needs_human_review",
                        "review_reason": "no_same_label_match",
                        "existing_overlaps": [],
                        "mutation_performed": False,
                    }
                ],
                "overlay": "overlays/frame-000001.jpg",
            },
        ],
        "contact_sheets": ["contact-sheet.jpg"],
    }
    candidate_pack_path = evidence_dir / "candidate-pack.json"
    write_json(candidate_pack_path, candidate_pack)

    judgment = {
        "schema_version": "1.0",
        "judgment_type": "full_review_explicit",
        "task_id": 42,
        "reviewer": "Human reviewer",
        "reviewed_at": "2026-09-01T12:00:00+08:00",
        "mutation_performed": False,
        "automated_flag_overrides": [],
        "accepted_candidate_ids": [light_id],
        "frame_actions": [
            {
                "frame": 1,
                "actions": [{"action": "relabel", "shape_id": 3, "to_label": "traffic_sign"}],
            }
        ],
    }
    judgment_path = tmp_path / "judgment.json"
    write_json(judgment_path, judgment)
    return {
        "snapshot": snapshot,
        "snapshot_path": snapshot_path,
        "review_pack_path": review_pack_path,
        "automated_path": automated_path,
        "candidate_pack_path": candidate_pack_path,
        "candidate_pack": candidate_pack,
        "judgment_path": judgment_path,
        "judgment": judgment,
        "output_path": tmp_path / "drafts" / "full-review-draft.json",
        "light_id": light_id,
        "sign_id": sign_id,
        "existing_id": existing_id,
        "light_bbox": light_bbox,
    }


def materialize_fixture(fixture: dict, candidate_paths: list[Path] | None = None) -> dict:
    return materialize_full_review_draft_files(
        fixture["snapshot_path"],
        fixture["review_pack_path"],
        fixture["automated_path"],
        fixture["judgment_path"],
        candidate_paths or [fixture["candidate_pack_path"]],
        fixture["output_path"],
    )


def test_materialize_builds_complete_hash_bound_compiler_draft(tmp_path: Path) -> None:
    fixture = build_fixture(tmp_path)
    inputs_before = {
        key: fixture[key].read_bytes()
        for key in (
            "snapshot_path",
            "review_pack_path",
            "automated_path",
            "judgment_path",
            "candidate_pack_path",
        )
    }

    draft = materialize_fixture(fixture)

    assert draft["schema_version"] == "1.2"
    assert draft["draft_type"] == "full_review_human"
    assert draft["task_id"] == 42
    assert draft["snapshot_sha256"] == file_sha256(fixture["snapshot_path"])
    assert draft["review_pack_sha256"] == file_sha256(fixture["review_pack_path"])
    assert draft["automated_decisions_sha256"] == file_sha256(fixture["automated_path"])
    assert draft["mutation_performed"] is False
    assert draft["manual_delete_approvals"] == []
    assert draft["review_evidence"] == [
        {
            "path": os.path.relpath(
                fixture["candidate_pack_path"], start=fixture["output_path"].parent
            ),
            "sha256": file_sha256(fixture["candidate_pack_path"]),
        }
    ]
    assert [review["frame"] for review in draft["frame_reviews"]] == [0, 1]
    assert all(review["reviewed"] is True for review in draft["frame_reviews"])
    assert all(
        review["labels_reviewed"] == ["car", "traffic_light", "traffic_sign"]
        for review in draft["frame_reviews"]
    )
    assert draft["frame_reviews"][0]["actions"] == [
        {"action": "add", "label": "traffic_light", "points": fixture["light_bbox"]}
    ]
    assert draft["frame_reviews"][1]["actions"] == [
        {"action": "relabel", "shape_id": 3, "to_label": "traffic_sign"}
    ]
    assert json.loads(fixture["output_path"].read_text(encoding="utf-8")) == draft
    assert all(fixture[key].read_bytes() == value for key, value in inputs_before.items())

    final_path = tmp_path / "full-review-decisions.json"
    decisions = compile_decision_files(
        fixture["snapshot_path"],
        fixture["review_pack_path"],
        fixture["automated_path"],
        fixture["output_path"],
        final_path,
    )
    assert decisions["scope"] == "full_review"
    assert decisions["mutation_performed"] is False
    assert decisions["review_evidence"][0]["sha256"] == file_sha256(fixture["candidate_pack_path"])


def _configure_approved_manual_delete(fixture: dict) -> dict:
    manual_shape = fixture["snapshot"]["annotations"]["shapes"][2]
    approval = manual_delete_approval(manual_shape)
    fixture["judgment"]["schema_version"] = "1.1"
    fixture["judgment"]["manual_delete_approvals"] = [approval]
    fixture["judgment"]["frame_actions"][0]["actions"] = [
        {"action": "delete", "shape_id": manual_shape["id"]}
    ]
    return approval


def test_materialize_propagates_exact_dual_review_manual_delete_approval(
    tmp_path: Path,
) -> None:
    fixture = build_fixture(tmp_path)
    approval = _configure_approved_manual_delete(fixture)
    write_json(fixture["judgment_path"], fixture["judgment"])

    draft = materialize_fixture(fixture)

    assert draft["schema_version"] == "1.2"
    assert draft["manual_delete_approvals"] == [approval]
    assert draft["frame_reviews"][1]["actions"] == [{"action": "delete", "shape_id": 3}]

    final_path = tmp_path / "approved-full-review-decisions.json"
    decisions = compile_decision_files(
        fixture["snapshot_path"],
        fixture["review_pack_path"],
        fixture["automated_path"],
        fixture["output_path"],
        final_path,
    )
    assert decisions["schema_version"] == "1.3"
    assert decisions["manual_delete_approvals"] == [approval]
    assert (
        decisions["frame_reviews"][1]["actions"][0]["expected_shape_sha256"]
        == (approval["canonical_shape_sha256"])
    )


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("missing", "one-for-one.*missing=\\[3\\]"),
        ("legacy_missing", "one-for-one.*missing=\\[3\\]"),
        ("new_schema_missing_field", "unexpected or missing keys"),
        ("unused", "one-for-one.*extra=\\[3\\]"),
        ("duplicate_approval", "appears twice"),
        ("duplicate_delete", "requested for deletion more than once"),
        ("unknown_shape", "references unknown shape"),
        ("auto_shape", "approve only source='manual'"),
        ("wrong_frame", "frame does not match shape"),
        ("wrong_shape_hash", "canonical_shape_sha256 does not match"),
        ("weak_reason", "substantive explanation"),
        ("one_reviewer", "exactly two independent reviews"),
        ("same_reviewer", "two distinct reviewers"),
        ("bad_decision", "decision must be 'approve_manual_delete'"),
        ("bad_timestamp", "RFC 3339 timestamp"),
        ("tampered_approval_hash", "approval_sha256 does not match"),
        ("extra_approval_key", "unexpected or missing keys"),
    ],
)
def test_materialize_rejects_forged_missing_or_unused_manual_delete_approvals(
    tmp_path: Path, case: str, match: str
) -> None:
    fixture = build_fixture(tmp_path)
    approval = _configure_approved_manual_delete(fixture)
    if case == "missing":
        fixture["judgment"]["manual_delete_approvals"] = []
    elif case == "legacy_missing":
        fixture["judgment"]["schema_version"] = "1.0"
        fixture["judgment"].pop("manual_delete_approvals")
    elif case == "new_schema_missing_field":
        fixture["judgment"].pop("manual_delete_approvals")
    elif case == "unused":
        fixture["judgment"]["frame_actions"][0]["actions"] = [
            {"action": "relabel", "shape_id": 3, "to_label": "traffic_sign"}
        ]
    elif case == "duplicate_approval":
        fixture["judgment"]["manual_delete_approvals"].append(copy.deepcopy(approval))
    elif case == "duplicate_delete":
        fixture["judgment"]["frame_actions"][0]["actions"].append(
            {"action": "delete", "shape_id": 3}
        )
    elif case == "unknown_shape":
        approval["shape_id"] = 999
    elif case == "auto_shape":
        auto_shape = fixture["snapshot"]["annotations"]["shapes"][0]
        fixture["judgment"]["frame_actions"][0] = {
            "frame": 0,
            "actions": [{"action": "delete", "shape_id": auto_shape["id"]}],
        }
        fixture["judgment"]["manual_delete_approvals"] = [manual_delete_approval(auto_shape)]
    elif case == "wrong_frame":
        approval["frame"] = 0
    elif case == "wrong_shape_hash":
        approval["canonical_shape_sha256"] = "f" * 64
    elif case == "weak_reason":
        approval["reason"] = "."
    elif case == "one_reviewer":
        approval["reviewers"].pop()
    elif case == "same_reviewer":
        approval["reviewers"][1]["reviewer_id"] = "REVIEWER-ALPHA"
    elif case == "bad_decision":
        approval["reviewers"][0]["decision"] = "approve"
    elif case == "bad_timestamp":
        approval["reviewers"][0]["reviewed_at"] = "2026-09-01"
    elif case == "tampered_approval_hash":
        approval["approval_sha256"] = "f" * 64
    else:
        approval["unexpected"] = True
    write_json(fixture["judgment_path"], fixture["judgment"])

    with pytest.raises(FullReviewDraftMaterializationError, match=match):
        materialize_fixture(fixture)

    assert not fixture["output_path"].exists()


def test_materialize_refuses_to_overwrite_existing_output(tmp_path: Path) -> None:
    fixture = build_fixture(tmp_path)
    materialize_fixture(fixture)
    before = fixture["output_path"].read_bytes()

    with pytest.raises(FileExistsError):
        materialize_fixture(fixture)

    assert fixture["output_path"].read_bytes() == before


def test_materialize_requires_at_least_one_candidate_pack(tmp_path: Path) -> None:
    fixture = build_fixture(tmp_path)

    with pytest.raises(FullReviewDraftMaterializationError, match="at least one"):
        materialize_full_review_draft_files(
            fixture["snapshot_path"],
            fixture["review_pack_path"],
            fixture["automated_path"],
            fixture["judgment_path"],
            [],
            fixture["output_path"],
        )

    assert not fixture["output_path"].exists()


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("unknown_candidate", "accepted candidate_id is unknown"),
        ("duplicate_candidate", "listed more than once"),
        ("already_annotated", "is not needs_human_review"),
        ("duplicate_frame", "duplicate frame"),
        ("unknown_frame", "unknown frame"),
        ("wrong_task", "does not match the snapshot task"),
        ("extra_key", "unexpected or missing keys"),
        ("duplicate_add", "duplicates add from accepted candidate"),
        ("manual_delete", "manual-delete approvals must correspond one-for-one"),
    ],
)
def test_materialize_rejects_invalid_or_ambiguous_judgments(
    tmp_path: Path, case: str, match: str
) -> None:
    fixture = build_fixture(tmp_path)
    judgment = fixture["judgment"]
    if case == "unknown_candidate":
        judgment["accepted_candidate_ids"] = ["does-not-exist"]
    elif case == "duplicate_candidate":
        judgment["accepted_candidate_ids"] *= 2
    elif case == "already_annotated":
        judgment["accepted_candidate_ids"] = [fixture["existing_id"]]
    elif case == "duplicate_frame":
        judgment["frame_actions"].append(copy.deepcopy(judgment["frame_actions"][0]))
    elif case == "unknown_frame":
        judgment["frame_actions"][0]["frame"] = 99
    elif case == "wrong_task":
        judgment["task_id"] = 99
    elif case == "extra_key":
        judgment["unexpected"] = True
    elif case == "duplicate_add":
        judgment["frame_actions"].append(
            {
                "frame": 0,
                "actions": [
                    {
                        "action": "add",
                        "label": "traffic_light",
                        "points": fixture["light_bbox"],
                    }
                ],
            }
        )
    else:
        judgment["frame_actions"][0]["actions"] = [{"action": "delete", "shape_id": 3}]
    write_json(fixture["judgment_path"], judgment)

    with pytest.raises(FullReviewDraftMaterializationError, match=match):
        materialize_fixture(fixture)

    assert not fixture["output_path"].exists()


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("duplicate_across_packs", "duplicate candidate_id across candidate packs"),
        ("forged_id", "contains forged candidate_id"),
        ("incomplete_pack", "unexpected or missing candidate-pack keys"),
        ("wrong_task", "task_id does not match the snapshot"),
    ],
)
def test_materialize_rejects_duplicate_or_forged_candidate_packs(
    tmp_path: Path, case: str, match: str
) -> None:
    fixture = build_fixture(tmp_path)
    candidate_paths = [fixture["candidate_pack_path"]]
    if case == "duplicate_across_packs":
        second_path = fixture["candidate_pack_path"].with_name("candidate-pack-2.json")
        write_json(second_path, fixture["candidate_pack"])
        candidate_paths.append(second_path)
    elif case == "forged_id":
        fixture["candidate_pack"]["frames"][0]["candidates"][0]["candidate_id"] = "forged"
        fixture["judgment"]["accepted_candidate_ids"] = ["forged"]
        write_json(fixture["candidate_pack_path"], fixture["candidate_pack"])
        write_json(fixture["judgment_path"], fixture["judgment"])
    elif case == "incomplete_pack":
        fixture["candidate_pack"].pop("model")
        write_json(fixture["candidate_pack_path"], fixture["candidate_pack"])
    else:
        fixture["candidate_pack"]["task_id"] = 99
        write_json(fixture["candidate_pack_path"], fixture["candidate_pack"])

    with pytest.raises(FullReviewDraftMaterializationError, match=match):
        materialize_fixture(fixture, candidate_paths)

    assert not fixture["output_path"].exists()


def test_materialize_rechecks_input_bytes_immediately_before_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = build_fixture(tmp_path)
    original_compile = materializer.compiler.compile_decisions

    def compile_then_drift(*args, **kwargs):
        result = original_compile(*args, **kwargs)
        fixture["candidate_pack_path"].write_bytes(b'{"changed":true}\n')
        return result

    monkeypatch.setattr(materializer.compiler, "compile_decisions", compile_then_drift)

    with pytest.raises(FullReviewDraftMaterializationError, match="changed while"):
        materialize_fixture(fixture)

    assert not fixture["output_path"].exists()


def test_materialize_cli_prints_read_only_summary(tmp_path: Path, capsys) -> None:
    fixture = build_fixture(tmp_path)

    main(
        [
            str(fixture["snapshot_path"]),
            str(fixture["review_pack_path"]),
            str(fixture["automated_path"]),
            str(fixture["judgment_path"]),
            "--candidate-pack",
            str(fixture["candidate_pack_path"]),
            "--output",
            str(fixture["output_path"]),
        ]
    )

    summary = json.loads(capsys.readouterr().out)
    assert summary == {
        "action_count": 2,
        "candidate_pack_count": 1,
        "frame_count": 2,
        "mutation_performed": False,
        "output": str(fixture["output_path"].resolve()),
        "task_id": 42,
    }
