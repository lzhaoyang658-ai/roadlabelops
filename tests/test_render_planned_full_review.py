from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.render_planned_full_review import render_planned_review
from scripts.snapshot_cvat_task import canonical_sha256


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_render_planned_review_is_local_only_complete_and_non_overwriting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path = tmp_path / "frame.jpg"
    Image.new("RGB", (100, 80), "#334155").save(image_path, quality=95)
    original_shape = {
        "type": "rectangle",
        "label_id": 10,
        "frame": 0,
        "occluded": False,
        "outside": False,
        "z_order": 0,
        "rotation": 0.0,
        "points": [5.0, 5.0, 25.0, 25.0],
        "id": 101,
        "group": 0,
        "source": "auto",
        "attributes": [],
        "score": 0.9,
        "elements": [],
    }
    annotations = {
        "version": 1,
        "tags": [],
        "shapes": [original_shape],
        "tracks": [],
        "intervals": [],
    }
    snapshot = {
        "snapshot_schema": {"name": "roadlabelops.cvat-task-snapshot", "version": 1},
        "task": {"id": 42},
        "labels": [{"id": 10, "name": "car"}, {"id": 11, "name": "bus"}],
        "images": [
            {
                "cvat_frame": 0,
                "sample_index": 1,
                "path": str(image_path),
                "file_name": image_path.name,
                "width": 100,
                "height": 80,
                "sha256": file_sha256(image_path),
            }
        ],
        "annotations": annotations,
        "canonical_annotations_sha256": canonical_sha256(annotations),
    }
    snapshot_path = tmp_path / "snapshot.json"
    write_json(snapshot_path, snapshot)
    review_pack_path = tmp_path / "review-pack.json"
    review_pack = {
        "schema_version": "1.0",
        "task_id": 42,
        "source_snapshot_sha256": file_sha256(snapshot_path),
        "annotation_sha256": snapshot["canonical_annotations_sha256"],
        "frames": [],
    }
    write_json(review_pack_path, review_pack)
    evidence_path = tmp_path / "visual-evidence.json"
    write_json(evidence_path, {"read_only": True})
    decisions_path = tmp_path / "decisions.json"
    decisions = {
        "task_id": 42,
        "review_evidence": [
            {"path": evidence_path.name, "sha256": file_sha256(evidence_path)}
        ],
    }
    write_json(decisions_path, decisions)

    added_shape = {
        "type": "rectangle",
        "label_id": 11,
        "frame": 0,
        "occluded": False,
        "outside": False,
        "z_order": 0,
        "rotation": 0.0,
        "points": [40.0, 10.0, 80.0, 60.0],
        "id": None,
        "group": 0,
        "source": "manual",
        "attributes": [],
        "score": 1.0,
        "elements": [],
    }
    expected_annotations = copy.deepcopy(annotations)
    expected_annotations["shapes"].append(added_shape)

    def fake_build_apply_plan(
        supplied_snapshot,
        supplied_pack,
        supplied_decisions,
        *,
        snapshot_file_sha256,
        review_pack_file_sha256,
    ):
        assert supplied_snapshot == snapshot
        assert supplied_pack == review_pack
        assert supplied_decisions == decisions
        assert snapshot_file_sha256 == file_sha256(snapshot_path)
        assert review_pack_file_sha256 == file_sha256(review_pack_path)
        return {
            "task_id": 42,
            "snapshot_annotations": annotations,
            "expected_annotations": expected_annotations,
            "expected_post_apply_canonical_sha256": "f" * 64,
            "action_counts": {"add": 1},
            "mutation_action_count": 1,
            "manual_delete_approval_count": 0,
        }

    monkeypatch.setattr(
        "scripts.render_planned_full_review.build_apply_plan", fake_build_apply_plan
    )
    inputs_before = {
        path: path.read_bytes() for path in (snapshot_path, review_pack_path, decisions_path)
    }
    output_dir = tmp_path / "planned"

    summary_path = render_planned_review(
        snapshot_path, review_pack_path, decisions_path, output_dir
    )

    assert all(path.read_bytes() == encoded for path, encoded in inputs_before.items())
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    planned_snapshot = json.loads((output_dir / summary["planned_snapshot"]).read_text())
    planned_pack = json.loads((output_dir / summary["planned_review_pack"]).read_text())
    assert summary["mutation_performed"] is False
    assert summary["cvat_connection_performed"] is False
    assert summary["annotation_count_before"] == 1
    assert summary["annotation_count_after"] == 2
    assert summary["action_counts"] == {"add": 1}
    assert summary["manual_delete_approval_count"] == 0
    assert planned_snapshot["planned_preview"]["cvat_connection_performed"] is False
    assert planned_snapshot["final_gate"]["passed"] is False
    assert planned_snapshot["counts"]["annotations_by_label"] == {"bus": 1, "car": 1}
    assert planned_snapshot["counts"]["shapes"] == 2
    assert planned_snapshot["canonical_annotations_sha256"] == canonical_sha256(
        expected_annotations
    )
    assert planned_snapshot["images"][0]["path"] == str(image_path.resolve())
    assert planned_pack["read_only"] is True
    assert planned_pack["mutation_performed"] is False
    assert planned_pack["annotation_count"] == 2
    assert (output_dir / "pack" / planned_pack["frames"][0]["overlay"]).is_file()
    assert all((output_dir / "pack" / path).is_file() for path in summary["contact_sheets"])

    sentinel = (output_dir / "plan-summary.json").read_bytes()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        render_planned_review(snapshot_path, review_pack_path, decisions_path, output_dir)
    assert (output_dir / "plan-summary.json").read_bytes() == sentinel
