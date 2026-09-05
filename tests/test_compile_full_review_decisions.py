from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

import scripts.compile_full_review_decisions as compiler_module
from scripts.compile_full_review_decisions import (
    FullReviewCompileError,
    compile_decision_files,
)
from scripts.validate_full_review_decisions import (
    canonical_annotations_sha256,
    canonical_shape_sha256,
    file_sha256,
    validate_decision_files,
)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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

    pack = {
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
    pack_path = tmp_path / "review-pack.json"
    write_json(pack_path, pack)
    pack_sha = file_sha256(pack_path)

    automated = {
        "schema_version": "1.1",
        "scope": "automated_risk_cleanup",
        "task_id": 42,
        "snapshot_sha256": snapshot_sha,
        "canonical_annotations_sha256": annotation_sha,
        "review_pack_sha256": pack_sha,
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
    write_json(automated_path, automated)
    automated_sha = file_sha256(automated_path)

    evidence_path = tmp_path / "candidate-pack.json"
    evidence = {
        "schema_version": "1.0",
        "pack_type": "manual_class_model_candidates",
        "task_id": 42,
        "read_only": True,
        "reviewed_by_human": False,
        "mutation_performed": False,
        "source_review_pack": str(pack_path),
        "source_snapshot_sha256": snapshot_sha,
        "source_review_pack_sha256": pack_sha,
        "model": "fixture-model.pt",
        "model_sha256": "c" * 64,
        "model_label_mapping": {},
        "review_labels": ["car", "traffic_light", "traffic_sign"],
        "parameters": {},
        "frame_count": 2,
        "frames_with_candidates": 0,
        "frames_needing_human_review": 0,
        "candidate_count": 0,
        "needs_human_review_count": 0,
        "candidate_counts_by_label": {},
        "needs_human_review_counts_by_label": {},
        "needs_human_review_counts_by_reason": {},
        "frames": [
            {
                "frame": 0,
                "sample_index": 1,
                "source_path": "frame-0.jpg",
                "source_sha256": "a" * 64,
                "candidate_count": 0,
                "needs_human_review_count": 0,
                "candidates": [],
            },
            {
                "frame": 1,
                "sample_index": 2,
                "source_path": "frame-1.jpg",
                "source_sha256": "b" * 64,
                "candidate_count": 0,
                "needs_human_review_count": 0,
                "candidates": [],
            },
        ],
        "contact_sheets": [],
    }
    write_json(evidence_path, evidence)
    labels = ["car", "traffic_light", "traffic_sign"]
    draft = {
        "schema_version": "1.1",
        "draft_type": "full_review_human",
        "task_id": 42,
        "snapshot_sha256": snapshot_sha,
        "review_pack_sha256": pack_sha,
        "automated_decisions_sha256": automated_sha,
        "reviewer": "Human reviewer",
        "reviewed_at": "2026-09-01",
        "mutation_performed": False,
        "automated_flag_overrides": [],
        "review_evidence": [{"path": evidence_path.name, "sha256": file_sha256(evidence_path)}],
        "frame_reviews": [
            {
                "frame": 0,
                "reviewed": True,
                "labels_reviewed": labels,
                "actions": [{"action": "add", "label": "traffic_light", "points": [70, 5, 80, 25]}],
            },
            {
                "frame": 1,
                "reviewed": True,
                "labels_reviewed": labels,
                "actions": [
                    {"action": "relabel", "shape_id": 3, "to_label": "traffic_sign"},
                    {"action": "add", "label": "traffic_sign", "points": [65, 20, 90, 70]},
                ],
            },
        ],
    }
    draft_path = tmp_path / "draft.json"
    write_json(draft_path, draft)
    return {
        "snapshot": snapshot,
        "snapshot_path": snapshot_path,
        "pack": pack,
        "pack_path": pack_path,
        "automated": automated,
        "automated_path": automated_path,
        "draft": draft,
        "draft_path": draft_path,
        "evidence": evidence,
        "evidence_path": evidence_path,
        "output_path": tmp_path / "full-decisions.json",
    }


def refresh_pack_bindings(fixture: dict) -> None:
    """Rebind all downstream fixture files after a deliberate review-pack mutation."""

    write_json(fixture["pack_path"], fixture["pack"])
    pack_sha = file_sha256(fixture["pack_path"])
    fixture["automated"]["review_pack_sha256"] = pack_sha
    write_json(fixture["automated_path"], fixture["automated"])
    fixture["evidence"]["source_review_pack_sha256"] = pack_sha
    write_json(fixture["evidence_path"], fixture["evidence"])
    fixture["draft"]["review_pack_sha256"] = pack_sha
    fixture["draft"]["automated_decisions_sha256"] = file_sha256(fixture["automated_path"])
    fixture["draft"]["review_evidence"][0]["sha256"] = file_sha256(fixture["evidence_path"])
    write_json(fixture["draft_path"], fixture["draft"])


def compile_fixture(fixture: dict) -> dict:
    return compile_decision_files(
        fixture["snapshot_path"],
        fixture["pack_path"],
        fixture["automated_path"],
        fixture["draft_path"],
        fixture["output_path"],
    )


def test_compile_merges_automated_actions_resolves_manual_flags_and_hashes_shapes(
    tmp_path: Path,
) -> None:
    fixture = build_fixture(tmp_path)

    result = compile_fixture(fixture)
    summary = validate_decision_files(
        fixture["snapshot_path"], fixture["pack_path"], fixture["output_path"]
    )

    assert summary["scope"] == "full_review"
    assert summary["reviewed_frame_count"] == 2
    assert summary["unresolved_manual_flag_count"] == 0
    assert summary["action_counts"] == {"add": 2, "delete": 1, "relabel": 1}
    assert result["frame_reviews"][0]["resolved_flag_ids"] == [
        "auto-0",
        "light-0",
        "sign-0",
    ]
    relabel = result["frame_reviews"][1]["actions"][0]
    assert relabel["expected_shape_sha256"] == canonical_shape_sha256(
        fixture["snapshot"]["annotations"]["shapes"][2]
    )
    assert (
        result["human_review_draft_sha256"]
        == hashlib.sha256(fixture["draft_path"].read_bytes()).hexdigest()
    )
    assert result["automated_risk_decisions_sha256"] == file_sha256(fixture["automated_path"])
    assert result["review_evidence"] == fixture["draft"]["review_evidence"]
    assert result["mutation_performed"] is False


def test_compile_refuses_to_overwrite_existing_output(tmp_path: Path) -> None:
    fixture = build_fixture(tmp_path)
    compile_fixture(fixture)
    before = fixture["output_path"].read_bytes()

    with pytest.raises(FileExistsError):
        compile_fixture(fixture)

    assert fixture["output_path"].read_bytes() == before


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_frame",
        "missing_label",
        "unknown_shape",
        "reviewed_false",
        "duplicate_frame",
        "duplicate_label",
        "extra_label",
    ],
)
def test_compile_rejects_incomplete_or_unbound_human_decisions(
    tmp_path: Path, mutation: str
) -> None:
    fixture = build_fixture(tmp_path)
    if mutation == "missing_frame":
        fixture["draft"]["frame_reviews"].pop()
    elif mutation == "missing_label":
        fixture["draft"]["frame_reviews"][0]["labels_reviewed"].remove("traffic_sign")
    elif mutation == "unknown_shape":
        fixture["draft"]["frame_reviews"][1]["actions"][0]["shape_id"] = 999
    elif mutation == "reviewed_false":
        fixture["draft"]["frame_reviews"][0]["reviewed"] = False
    elif mutation == "duplicate_frame":
        fixture["draft"]["frame_reviews"][1]["frame"] = 0
    elif mutation == "duplicate_label":
        fixture["draft"]["frame_reviews"][0]["labels_reviewed"].append("car")
    else:
        fixture["draft"]["frame_reviews"][0]["labels_reviewed"].append("alien")
    write_json(fixture["draft_path"], fixture["draft"])

    with pytest.raises(FullReviewCompileError):
        compile_fixture(fixture)

    assert not fixture["output_path"].exists()


def test_compile_rejects_review_evidence_hash_drift(tmp_path: Path) -> None:
    fixture = build_fixture(tmp_path)
    fixture["evidence_path"].write_text('{"changed":true}\n', encoding="utf-8")

    with pytest.raises(FullReviewCompileError, match="SHA-256 does not match"):
        compile_fixture(fixture)

    assert not fixture["output_path"].exists()


def test_compile_rejects_review_evidence_source_binding_mismatch(tmp_path: Path) -> None:
    fixture = build_fixture(tmp_path)
    fixture["evidence"]["source_snapshot_sha256"] = "0" * 64
    write_json(fixture["evidence_path"], fixture["evidence"])
    fixture["draft"]["review_evidence"][0]["sha256"] = file_sha256(fixture["evidence_path"])
    write_json(fixture["draft_path"], fixture["draft"])

    with pytest.raises(FullReviewCompileError, match="not bound to the supplied snapshot"):
        compile_fixture(fixture)

    assert not fixture["output_path"].exists()


def test_compile_rejects_duplicate_review_evidence_path(tmp_path: Path) -> None:
    fixture = build_fixture(tmp_path)
    fixture["draft"]["review_evidence"].append(
        copy.deepcopy(fixture["draft"]["review_evidence"][0])
    )
    write_json(fixture["draft_path"], fixture["draft"])

    with pytest.raises(FullReviewCompileError, match="duplicates"):
        compile_fixture(fixture)

    assert not fixture["output_path"].exists()


def test_compile_rechecks_review_evidence_immediately_before_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = build_fixture(tmp_path)
    original_compile = compiler_module.compile_decisions

    def compile_then_mutate_evidence(*args: object, **kwargs: object) -> dict:
        result = original_compile(*args, **kwargs)
        fixture["evidence_path"].write_text('{"changed":true}\n', encoding="utf-8")
        return result

    monkeypatch.setattr(compiler_module, "compile_decisions", compile_then_mutate_evidence)

    with pytest.raises(FullReviewCompileError, match="SHA-256 does not match"):
        compile_fixture(fixture)

    assert not fixture["output_path"].exists()


@pytest.mark.parametrize(
    "mutation", ["missing", "unknown_label", "conflicting_id", "shrunk_declared"]
)
def test_compile_rejects_invalid_manual_flag_inventory(tmp_path: Path, mutation: str) -> None:
    fixture = build_fixture(tmp_path)
    flags = fixture["pack"]["frames"][0]["flags"]
    if mutation == "missing":
        flags.pop()
    elif mutation == "unknown_label":
        flags[-1]["label"] = "alien"
    elif mutation == "conflicting_id":
        flags[-1]["id"] = "different-id"
    else:
        fixture["pack"]["manual_check_labels"] = ["traffic_light"]
        for frame in fixture["pack"]["frames"]:
            frame["flags"] = [
                flag for flag in frame["flags"] if flag.get("label") != "traffic_sign"
            ]
    refresh_pack_bindings(fixture)

    with pytest.raises(FullReviewCompileError):
        compile_fixture(fixture)

    assert not fixture["output_path"].exists()


def test_compile_rejects_minimal_forged_candidate_evidence(tmp_path: Path) -> None:
    fixture = build_fixture(tmp_path)
    fixture["evidence"] = {
        key: fixture["evidence"][key]
        for key in (
            "pack_type",
            "task_id",
            "read_only",
            "mutation_performed",
            "source_snapshot_sha256",
            "source_review_pack_sha256",
        )
    }
    write_json(fixture["evidence_path"], fixture["evidence"])
    fixture["draft"]["review_evidence"][0]["sha256"] = file_sha256(fixture["evidence_path"])
    write_json(fixture["draft_path"], fixture["draft"])

    with pytest.raises(FullReviewCompileError, match="candidate-pack keys"):
        compile_fixture(fixture)

    assert not fixture["output_path"].exists()


def test_compile_rejects_unexpected_or_forbidden_draft_keys(tmp_path: Path) -> None:
    fixture = build_fixture(tmp_path)
    fixture["draft"]["accepted_rules"] = [{"frame_range": [0, 1]}]
    write_json(fixture["draft_path"], fixture["draft"])

    with pytest.raises(FullReviewCompileError, match="top-level keys"):
        compile_fixture(fixture)

    assert not fixture["output_path"].exists()


def test_compile_rejects_draft_bound_to_another_automated_decision_file(
    tmp_path: Path,
) -> None:
    fixture = build_fixture(tmp_path)
    fixture["draft"]["automated_decisions_sha256"] = "0" * 64
    write_json(fixture["draft_path"], fixture["draft"])

    with pytest.raises(FullReviewCompileError, match="automated_decisions_sha256"):
        compile_fixture(fixture)

    assert not fixture["output_path"].exists()


def test_compile_rejects_unbound_automated_base_action(tmp_path: Path) -> None:
    fixture = build_fixture(tmp_path)
    fixture["automated"]["frame_reviews"][0]["actions"].append(
        {
            "action": "keep_distinct",
            "shape_id": 1,
            "expected_shape_sha256": canonical_shape_sha256(
                fixture["snapshot"]["annotations"]["shapes"][0]
            ),
        }
    )
    write_json(fixture["automated_path"], fixture["automated"])
    fixture["draft"]["automated_decisions_sha256"] = file_sha256(fixture["automated_path"])
    write_json(fixture["draft_path"], fixture["draft"])

    with pytest.raises(FullReviewCompileError, match="not exclusively bound"):
        compile_fixture(fixture)

    assert not fixture["output_path"].exists()


def test_compile_rejects_action_conflicting_with_automated_cleanup(tmp_path: Path) -> None:
    fixture = build_fixture(tmp_path)
    fixture["draft"]["frame_reviews"][0]["actions"].append(
        {"action": "relabel", "shape_id": 2, "to_label": "traffic_light"}
    )
    write_json(fixture["draft_path"], fixture["draft"])

    with pytest.raises(FullReviewCompileError, match="more than one action"):
        compile_fixture(fixture)

    assert not fixture["output_path"].exists()


def test_compile_allows_explicit_hash_bound_automated_flag_action_override(
    tmp_path: Path,
) -> None:
    fixture = build_fixture(tmp_path)
    fixture["draft"]["automated_flag_overrides"] = [
        {
            "flag_id": "auto-0",
            "frame": 0,
            "replacement_action": {"action": "delete", "shape_id": 1},
        }
    ]
    write_json(fixture["draft_path"], fixture["draft"])

    result = compile_fixture(fixture)

    first_action = result["frame_reviews"][0]["actions"][0]
    assert first_action["shape_id"] == 1
    assert first_action["expected_shape_sha256"] == canonical_shape_sha256(
        fixture["snapshot"]["annotations"]["shapes"][0]
    )
    assert first_action["resolves_flag_ids"] == ["auto-0"]


@pytest.mark.parametrize("kind", ["update_bbox", "relabel_bbox"])
def test_compile_hash_binds_bbox_updates_without_replacing_shape_identity(
    tmp_path: Path, kind: str
) -> None:
    fixture = build_fixture(tmp_path)
    action = {
        "action": kind,
        "shape_id": 3,
        "points": [28, 8, 64, 44],
    }
    if kind == "relabel_bbox":
        action["to_label"] = "traffic_sign"
    fixture["draft"]["frame_reviews"][1]["actions"][0] = action
    write_json(fixture["draft_path"], fixture["draft"])

    result = compile_fixture(fixture)

    compiled = result["frame_reviews"][1]["actions"][0]
    assert compiled["action"] == kind
    assert compiled["shape_id"] == 3
    assert compiled["points"] == [28, 8, 64, 44]
    assert compiled["expected_shape_sha256"] == canonical_shape_sha256(
        fixture["snapshot"]["annotations"]["shapes"][2]
    )
    if kind == "relabel_bbox":
        assert compiled["to_label"] == "traffic_sign"
