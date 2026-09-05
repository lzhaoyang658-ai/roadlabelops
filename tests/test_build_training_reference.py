from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml
from PIL import Image

from scripts import build_training_reference as reference
from scripts.build_training_reference import (
    InputPaths,
    TrainingReferenceError,
    build_training_reference,
    file_sha256,
    main,
)
from scripts.snapshot_cvat_task import canonical_sha256, canonicalize_annotations


@dataclass
class Fixture:
    inputs: list[InputPaths]
    source_map: Path
    labels: Path


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_image(path: Path, color: tuple[int, int, int]) -> tuple[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 12), color).save(path)
    return file_sha256(path), path.stat().st_size


def shape(identifier: int, *, frame: int, label_id: int, points: list[float]) -> dict[str, object]:
    return {
        "id": identifier,
        "frame": frame,
        "label_id": label_id,
        "type": "rectangle",
        "points": points,
        "outside": False,
        "rotation": 0.0,
    }


def build_task(
    root: Path,
    *,
    task_id: int,
    job_id: int,
    samples: list[tuple[str, str, int, tuple[int, int, int]]],
    shapes: list[dict[str, object]],
) -> InputPaths:
    task_root = root / f"task-{task_id}"
    manifest_path = task_root / "sample-manifest.json"
    manifest_samples: list[dict[str, object]] = []
    image_inventory: list[dict[str, object]] = []
    for frame, (file_name, scene_id, source_frame, color) in enumerate(samples):
        image_path = task_root / "images" / file_name
        digest, size_bytes = write_image(image_path, color)
        manifest_samples.append(
            {
                "sample_index": frame + 1,
                "file_name": file_name,
                "scene_id": scene_id,
                "source_frame": source_frame,
                "sha256": digest,
                "width": 20,
                "height": 12,
            }
        )
        image_inventory.append(
            {
                "sample_index": frame + 1,
                "cvat_frame": frame,
                "file_name": file_name,
                "relative_path": f"images/{file_name}",
                "scene_id": scene_id,
                "source_frame": source_frame,
                "sha256": digest,
                "size_bytes": size_bytes,
                "width": 20,
                "height": 12,
                "format": "PNG",
            }
        )
    manifest = {
        "session_id": "session-test",
        "purpose": "training",
        "sampling_revision": f"task-{task_id}-v1",
        "sample_size": len(samples),
        "cvat": {"task_id": task_id, "job_ids": [job_id]},
        "samples": manifest_samples,
    }
    write_json(manifest_path, manifest)

    labels = [
        {"id": task_id * 10 + index, "name": name}
        for index, name in enumerate(reference.REQUIRED_LABELS, start=1)
    ]
    label_id_to_name = {record["id"]: record["name"] for record in labels}
    annotations = {"tags": [], "shapes": shapes, "tracks": []}
    annotation_sha = canonical_sha256(canonicalize_annotations(annotations))
    class_counts = {name: 0 for name in reference.REQUIRED_LABELS}
    for record in shapes:
        class_counts[label_id_to_name[int(record["label_id"])]] += 1
    snapshot = {
        "snapshot_schema": reference.SNAPSHOT_SCHEMA,
        "created_at": "2026-09-01T01:01:00+00:00",
        "task": {"id": task_id, "size": len(samples), "status": "annotation"},
        "manifest": {
            "path": str(manifest_path),
            "sha256": file_sha256(manifest_path),
            "session_id": manifest["session_id"],
            "purpose": manifest["purpose"],
            "sampling_revision": manifest["sampling_revision"],
            "sample_size": len(samples),
            "cvat_task_id": task_id,
        },
        "labels": labels,
        "jobs": [
            {
                "id": job_id,
                "task_id": task_id,
                "type": "annotation",
                "parent_job_id": None,
                "start_frame": 0,
                "stop_frame": len(samples) - 1,
                "frame_count": len(samples),
                "stage": "annotation",
                "state": "completed",
                "issues": {"count": 0},
            }
        ],
        "images": image_inventory,
        "annotations": annotations,
        "canonical_annotations_sha256": annotation_sha,
        "counts": {
            "images": len(samples),
            "tags": 0,
            "shapes": len(shapes),
            "tracks": 0,
            "annotations_by_label": class_counts,
        },
        "final_gate": {"passed": True, "blocking_reasons": [], "warnings": []},
    }
    snapshot_path = task_root / "post-completion.snapshot.json"
    write_json(snapshot_path, snapshot)
    pre_completion_snapshot = copy.deepcopy(snapshot)
    pre_completion_snapshot["created_at"] = "2026-09-01T00:59:00+00:00"
    pre_completion_snapshot["jobs"][0]["state"] = "new"
    pre_completion_snapshot_path = task_root / "post-apply.snapshot.json"
    write_json(pre_completion_snapshot_path, pre_completion_snapshot)
    receipt = {
        "schema": reference.RECEIPT_SCHEMA,
        "task_id": task_id,
        "job_id": job_id,
        "dry_run": False,
        "mutation_performed": True,
        "job_state_before": "new",
        "job_state_after": "completed",
        "job_stage_before": "annotation",
        "job_stage_after": "annotation",
        "annotation_count": len(shapes),
        "verified_live_canonical_annotations_sha256": annotation_sha,
        "verified_post_completion_canonical_annotations_sha256": annotation_sha,
        "completed_at": "2026-09-01T01:00:00+00:00",
        "decision_validation": {
            "valid": True,
            "scope": "full_review",
            "task_id": task_id,
            "snapshot_frame_count": len(samples),
            "reviewed_frame_count": len(samples),
            "unresolved_manual_flag_count": 0,
        },
        "evidence": {
            "post_snapshot": {
                "path": str(pre_completion_snapshot_path),
                "sha256": file_sha256(pre_completion_snapshot_path),
            }
        },
    }
    receipt_path = task_root / "completion-receipt.json"
    write_json(receipt_path, receipt)
    return InputPaths(snapshot_path, manifest_path, receipt_path)


@pytest.fixture
def fixture(tmp_path: Path) -> Fixture:
    labels_path = tmp_path / "road_labels.yaml"
    labels_path.write_text(
        yaml.safe_dump(
            {"labels": [{"name": name} for name in reference.REQUIRED_LABELS]},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    task_11 = build_task(
        tmp_path,
        task_id=11,
        job_id=111,
        samples=[
            ("scene-a-000.png", "scene-a", 0, (255, 0, 0)),
            ("scene-a-001.png", "scene-a", 1, (0, 255, 0)),
        ],
        shapes=[shape(1, frame=0, label_id=111, points=[1.0, 2.0, 10.0, 8.0])],
    )
    task_22 = build_task(
        tmp_path,
        task_id=22,
        job_id=222,
        samples=[("scene-b-002.png", "scene-b", 2, (0, 0, 255))],
        shapes=[shape(2, frame=0, label_id=222, points=[2.0, 1.0, 18.0, 11.0])],
    )
    source_map_path = tmp_path / "source-map.json"
    write_json(
        source_map_path,
        {
            "schema": reference.SOURCE_MAP_SCHEMA,
            "assets": [
                {
                    "asset_id": "asset-a",
                    "page_url": "https://example.test/assets/a",
                    "sha256": "a" * 64,
                    "leakage_group_id": f"sha256:{'a' * 64}",
                    "audit": {"license": "test-license"},
                },
                {
                    "asset_id": "asset-b",
                    "page_url": "https://example.test/assets/b",
                    "sha256": "b" * 64,
                    "leakage_group_id": f"sha256:{'b' * 64}",
                },
            ],
            "frames": [
                {
                    "scene_id": "scene-a",
                    "source_frame": 0,
                    "asset_id": "asset-a",
                    "leakage_group_id": f"sha256:{'a' * 64}",
                    "normalized_asset_frame": 100,
                },
                {
                    "scene_id": "scene-a",
                    "source_frame": 1,
                    "asset_id": "asset-b",
                    "leakage_group_id": f"sha256:{'b' * 64}",
                    "normalized_asset_frame": 10,
                },
                {
                    "scene_id": "scene-b",
                    "source_frame": 2,
                    "asset_id": "asset-a",
                    "leakage_group_id": f"sha256:{'a' * 64}",
                    "normalized_asset_frame": 101,
                },
            ],
        },
    )
    return Fixture([task_11, task_22], source_map_path, labels_path)


def test_builds_deterministic_complete_coco_with_many_to_many_source_map(
    tmp_path: Path, fixture: Fixture
) -> None:
    first = tmp_path / "reference-a"
    second = tmp_path / "reference-b"

    summary = build_training_reference(
        fixture.inputs,
        source_map_path=fixture.source_map,
        labels_path=fixture.labels,
        output=first,
    )
    build_training_reference(
        list(reversed(fixture.inputs)),
        source_map_path=fixture.source_map,
        labels_path=fixture.labels,
        output=second,
    )

    assert (first / "annotations.coco.json").read_bytes() == (
        second / "annotations.coco.json"
    ).read_bytes()
    coco = json.loads((first / "annotations.coco.json").read_text(encoding="utf-8"))
    manifest_text = (first / "manifest.json").read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert str(tmp_path) not in manifest_text
    assert all(
        record["path_kind"] == "content_addressed_external"
        for record in (
            manifest["evidence"]["labels"],
            manifest["evidence"]["source_map"],
            *(
                evidence
                for task_input in manifest["evidence"]["task_inputs"]
                for evidence in (
                    task_input["snapshot"],
                    task_input["image_manifest"],
                    task_input["completion_receipt"],
                )
            ),
        )
    )
    assert summary["gate"]["passed"] is True
    assert coco["info"]["schema"] == reference.OUTPUT_SCHEMA
    assert [category["name"] for category in coco["categories"]] == list(reference.REQUIRED_LABELS)
    assert [category["id"] for category in coco["categories"]] == list(range(1, 9))
    assert len(coco["images"]) == 3
    assert len(coco["annotations"]) == 2
    assert all("source_leakage_group_id" in image for image in coco["images"])
    assert all("source_normalized_asset_frame" in image for image in coco["images"])
    assert all("source_asset_frame" not in image for image in coco["images"])
    assert manifest["counts"]["zero_annotation_images"] == 1
    assert all((first / image["file_name"]).is_file() for image in coco["images"])
    scene_a_assets = {
        image["source_asset_id"] for image in coco["images"] if image["scene_id"] == "scene-a"
    }
    assert scene_a_assets == {"asset-a", "asset-b"}
    asset_a_scenes = next(
        item["scene_ids"]
        for item in manifest["source_statistics"]["assets"]
        if item["asset_id"] == "asset-a"
    )
    assert asset_a_scenes == ["scene-a", "scene-b"]
    assert manifest["evidence"]["source_map"]["assets"] == [
        {
            "asset_id": "asset-a",
            "page_url": "https://example.test/assets/a",
            "sha256": "a" * 64,
            "leakage_group_id": f"sha256:{'a' * 64}",
            "audit": {"license": "test-license"},
        },
        {
            "asset_id": "asset-b",
            "page_url": "https://example.test/assets/b",
            "sha256": "b" * 64,
            "leakage_group_id": f"sha256:{'b' * 64}",
        },
    ]
    assert len(manifest["files"]) == 4
    for record in manifest["files"]:
        assert file_sha256(first / record["path"]) == record["sha256"]


def test_workspace_evidence_locator_is_relative_posix(tmp_path: Path) -> None:
    locator = reference._evidence_locator(
        tmp_path / "nested" / "evidence.json",
        "a" * 64,
        workspace_root=tmp_path.resolve(),
    )

    assert locator == {
        "path": "nested/evidence.json",
        "path_kind": "workspace_relative",
    }


def test_cli_binds_receipts_by_task_id(
    tmp_path: Path, fixture: Fixture, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "cli-reference"
    first, second = fixture.inputs

    main(
        [
            "--input",
            str(second.snapshot),
            str(second.image_manifest),
            "--input",
            str(first.snapshot),
            str(first.image_manifest),
            "--completion-receipt",
            "11",
            str(first.completion_receipt),
            "--completion-receipt",
            "22",
            str(second.completion_receipt),
            "--source-map",
            str(fixture.source_map),
            "--labels",
            str(fixture.labels),
            "--output",
            str(output),
        ]
    )

    assert json.loads(capsys.readouterr().out)["gate"]["passed"] is True
    assert (output / "manifest.json").is_file()


def test_rejects_receipt_annotation_hash_drift(tmp_path: Path, fixture: Fixture) -> None:
    receipt_path = fixture.inputs[0].completion_receipt
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["verified_post_completion_canonical_annotations_sha256"] = "0" * 64
    write_json(receipt_path, receipt)

    with pytest.raises(TrainingReferenceError, match="differs from snapshot annotations"):
        build_training_reference(
            fixture.inputs,
            source_map_path=fixture.source_map,
            labels_path=fixture.labels,
            output=tmp_path / "rejected",
        )


def test_rejects_receipt_when_named_pre_completion_snapshot_drifts(
    tmp_path: Path, fixture: Fixture
) -> None:
    receipt_path = fixture.inputs[0].completion_receipt
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    snapshot_path = Path(receipt["evidence"]["post_snapshot"]["path"])
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot["final_gate"]["warnings"] = ["changed after completion receipt"]
    write_json(snapshot_path, snapshot)

    with pytest.raises(TrainingReferenceError, match="SHA-256 differs from the named file"):
        build_training_reference(
            fixture.inputs,
            source_map_path=fixture.source_map,
            labels_path=fixture.labels,
            output=tmp_path / "rejected",
        )


def test_rejects_completed_snapshot_captured_before_receipt(
    tmp_path: Path, fixture: Fixture
) -> None:
    snapshot_path = fixture.inputs[0].snapshot
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot["created_at"] = "2026-09-01T00:59:59+00:00"
    write_json(snapshot_path, snapshot)

    with pytest.raises(TrainingReferenceError, match="captured before the completion receipt"):
        build_training_reference(
            fixture.inputs,
            source_map_path=fixture.source_map,
            labels_path=fixture.labels,
            output=tmp_path / "rejected",
        )


def test_rejects_receipt_evidence_that_is_already_completed(
    tmp_path: Path, fixture: Fixture
) -> None:
    receipt_path = fixture.inputs[0].completion_receipt
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    snapshot_path = Path(receipt["evidence"]["post_snapshot"]["path"])
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot["jobs"][0]["state"] = "completed"
    write_json(snapshot_path, snapshot)
    receipt["evidence"]["post_snapshot"]["sha256"] = file_sha256(snapshot_path)
    write_json(receipt_path, receipt)

    with pytest.raises(TrainingReferenceError, match="must be 'new' or 'in_progress'"):
        build_training_reference(
            fixture.inputs,
            source_map_path=fixture.source_map,
            labels_path=fixture.labels,
            output=tmp_path / "rejected",
        )


def test_rejects_receipt_without_real_post_snapshot_evidence(
    tmp_path: Path, fixture: Fixture
) -> None:
    receipt_path = fixture.inputs[0].completion_receipt
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt.pop("evidence")
    write_json(receipt_path, receipt)

    with pytest.raises(TrainingReferenceError, match="completion receipt.evidence must be"):
        build_training_reference(
            fixture.inputs,
            source_map_path=fixture.source_map,
            labels_path=fixture.labels,
            output=tmp_path / "rejected",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("valid", False, "decision validation must be valid"),
        ("scope", "sample_review", "scope must be 'full_review'"),
        ("task_id", 999, "for another task"),
        ("snapshot_frame_count", 999, "snapshot_frame_count differs"),
        ("reviewed_frame_count", 999, "reviewed_frame_count differs"),
        ("unresolved_manual_flag_count", 1, "unresolved manual flags"),
    ],
)
def test_rejects_incomplete_receipt_decision_validation(
    tmp_path: Path,
    fixture: Fixture,
    field: str,
    value: object,
    message: str,
) -> None:
    receipt_path = fixture.inputs[0].completion_receipt
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["decision_validation"][field] = value
    write_json(receipt_path, receipt)

    with pytest.raises(TrainingReferenceError, match=message):
        build_training_reference(
            fixture.inputs,
            source_map_path=fixture.source_map,
            labels_path=fixture.labels,
            output=tmp_path / "rejected",
        )


def test_rejects_missing_source_frame_mapping(tmp_path: Path, fixture: Fixture) -> None:
    source_map = json.loads(fixture.source_map.read_text(encoding="utf-8"))
    source_map["frames"].pop()
    write_json(fixture.source_map, source_map)

    with pytest.raises(TrainingReferenceError, match="source map has no unique match"):
        build_training_reference(
            fixture.inputs,
            source_map_path=fixture.source_map,
            labels_path=fixture.labels,
            output=tmp_path / "rejected",
        )


def test_rejects_source_sha_aliases_under_different_asset_ids(
    tmp_path: Path, fixture: Fixture
) -> None:
    source_map = json.loads(fixture.source_map.read_text(encoding="utf-8"))
    source_map["assets"][1]["sha256"] = "a" * 64
    source_map["assets"][1]["leakage_group_id"] = f"sha256:{'a' * 64}"
    write_json(fixture.source_map, source_map)

    with pytest.raises(TrainingReferenceError, match="aliases the same source SHA-256"):
        build_training_reference(
            fixture.inputs,
            source_map_path=fixture.source_map,
            labels_path=fixture.labels,
            output=tmp_path / "rejected",
        )


def test_rejects_duplicate_normalized_asset_frame(tmp_path: Path, fixture: Fixture) -> None:
    source_map = json.loads(fixture.source_map.read_text(encoding="utf-8"))
    source_map["frames"][1]["asset_id"] = "asset-a"
    source_map["frames"][1]["leakage_group_id"] = f"sha256:{'a' * 64}"
    source_map["frames"][1]["normalized_asset_frame"] = 100
    write_json(fixture.source_map, source_map)

    with pytest.raises(TrainingReferenceError, match="more than one scene frame"):
        build_training_reference(
            fixture.inputs,
            source_map_path=fixture.source_map,
            labels_path=fixture.labels,
            output=tmp_path / "rejected",
        )


def test_rejects_image_hash_drift(tmp_path: Path, fixture: Fixture) -> None:
    snapshot_path = fixture.inputs[0].snapshot
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot["images"][0]["sha256"] = "0" * 64
    write_json(snapshot_path, snapshot)

    with pytest.raises(TrainingReferenceError, match="SHA-256 differs from disk"):
        build_training_reference(
            fixture.inputs,
            source_map_path=fixture.source_map,
            labels_path=fixture.labels,
            output=tmp_path / "rejected",
        )


def test_rejects_out_of_bounds_bbox(tmp_path: Path, fixture: Fixture) -> None:
    paths = fixture.inputs[0]
    snapshot = json.loads(paths.snapshot.read_text(encoding="utf-8"))
    snapshot["annotations"]["shapes"][0]["points"] = [1.0, 2.0, 21.0, 8.0]
    annotation_sha = canonical_sha256(canonicalize_annotations(snapshot["annotations"]))
    snapshot["canonical_annotations_sha256"] = annotation_sha
    write_json(paths.snapshot, snapshot)
    receipt = json.loads(paths.completion_receipt.read_text(encoding="utf-8"))
    receipt["verified_live_canonical_annotations_sha256"] = annotation_sha
    receipt["verified_post_completion_canonical_annotations_sha256"] = annotation_sha
    write_json(paths.completion_receipt, receipt)

    with pytest.raises(TrainingReferenceError, match="escapes image bounds"):
        build_training_reference(
            fixture.inputs,
            source_map_path=fixture.source_map,
            labels_path=fixture.labels,
            output=tmp_path / "rejected",
        )


def test_rejects_noncanonical_taxonomy_order(tmp_path: Path, fixture: Fixture) -> None:
    labels = yaml.safe_load(fixture.labels.read_text(encoding="utf-8"))
    labels["labels"][0], labels["labels"][1] = labels["labels"][1], labels["labels"][0]
    fixture.labels.write_text(yaml.safe_dump(labels, sort_keys=False), encoding="utf-8")

    with pytest.raises(TrainingReferenceError, match="canonical order"):
        build_training_reference(
            fixture.inputs,
            source_map_path=fixture.source_map,
            labels_path=fixture.labels,
            output=tmp_path / "rejected",
        )


def test_existing_output_is_never_overwritten(tmp_path: Path, fixture: Fixture) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "owned-by-user.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(TrainingReferenceError, match="already exists"):
        build_training_reference(
            fixture.inputs,
            source_map_path=fixture.source_map,
            labels_path=fixture.labels,
            output=output,
        )
    assert marker.read_text(encoding="utf-8") == "keep"


def test_broken_output_symlink_is_never_followed(tmp_path: Path, fixture: Fixture) -> None:
    redirected = tmp_path / "redirected-reference"
    output = tmp_path / "broken-reference-link"
    output.symlink_to(redirected)

    with pytest.raises(TrainingReferenceError, match="already exists"):
        build_training_reference(
            fixture.inputs,
            source_map_path=fixture.source_map,
            labels_path=fixture.labels,
            output=output,
        )

    assert output.is_symlink()
    assert not redirected.exists()


def test_atomic_publish_refuses_an_empty_racing_target(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    output = tmp_path / "output"
    staging.mkdir()
    (staging / "payload.txt").write_text("new", encoding="utf-8")
    output.mkdir()

    with pytest.raises(TrainingReferenceError, match="already exists"):
        reference._atomic_publish_directory_no_replace(staging, output)

    assert staging.is_dir()
    assert output.is_dir()
    assert not (output / "payload.txt").exists()


def test_failed_staged_copy_is_cleaned(
    tmp_path: Path, fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "copy-failure"

    def fail_copy(*_args: object, **_kwargs: object) -> None:
        raise OSError("injected copy failure")

    monkeypatch.setattr(reference.shutil, "copyfileobj", fail_copy)
    with pytest.raises(OSError, match="injected copy failure"):
        build_training_reference(
            fixture.inputs,
            source_map_path=fixture.source_map,
            labels_path=fixture.labels,
            output=output,
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".copy-failure.staging-*"))
