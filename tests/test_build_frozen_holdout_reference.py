from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import build_frozen_holdout_reference as builder
from scripts import evaluate_frozen_holdout as evaluator

FINAL_HOLDOUT_TASK_ID = 91001
FINAL_HOLDOUT_JOB_ID = 92001


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> dict[str, Path]:
    reference = tmp_path / "reference-v2"
    images: list[dict[str, object]] = []
    annotations: list[dict[str, object]] = []
    files: list[dict[str, object]] = []
    source_sha = "a" * 64
    for frame in range(50):
        relative = f"images/final-holdout-fixture/frame-{frame:03d}.jpg"
        image_path = reference / relative
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(f"image-{frame}".encode())
        files.append(
            {
                "path": relative,
                "sha256": _digest(image_path),
                "size_bytes": image_path.stat().st_size,
            }
        )
        images.append(
            {
                "id": frame + 1,
                "file_name": relative,
                "width": 10,
                "height": 10,
                "sha256": _digest(image_path),
                "task_id": FINAL_HOLDOUT_TASK_ID,
                "cvat_frame": frame,
                "sample_index": frame,
                "scene_id": "holdout-scene",
                "source_frame": frame,
                "source_asset_id": "holdout-asset",
                "source_leakage_group_id": f"sha256:{source_sha}",
                "source_normalized_asset_frame": frame,
            }
        )
        annotations.append(
            {
                "id": frame + 1,
                "image_id": frame + 1,
                "category_id": 1,
                "bbox": [1.0, 1.0, 2.0, 2.0],
                "area": 4.0,
                "iscrowd": 0,
            }
        )
    categories = [
        {"id": index, "name": name, "supercategory": "road_object"}
        for index, name in enumerate(builder.REQUIRED_LABELS, start=1)
    ]
    coco = reference / "annotations.coco.json"
    _write_json(
        coco,
        {
            "info": {"schema": builder.INPUT_SCHEMA},
            "licenses": [],
            "categories": categories,
            "images": images,
            "annotations": annotations,
        },
    )
    files.append(
        {
            "path": "annotations.coco.json",
            "sha256": _digest(coco),
            "size_bytes": coco.stat().st_size,
        }
    )
    _write_json(
        reference / "manifest.json",
        {
            "schema": builder.INPUT_SCHEMA,
            "gate": {
                "passed": True,
                "blocking_reasons": [],
                "checks": {"source_reference_verified": True},
            },
            "counts": {
                "tasks": 1,
                "images": 50,
                "zero_annotation_images": 0,
                "annotations": 50,
                "categories": 8,
                "annotations_by_category": {
                    name: 50 if name == "car" else 0 for name in builder.REQUIRED_LABELS
                },
            },
            "source_statistics": {
                "tasks": [
                    {
                        "task_id": FINAL_HOLDOUT_TASK_ID,
                        "job_id": FINAL_HOLDOUT_JOB_ID,
                        "image_count": 50,
                        "zero_annotation_image_count": 0,
                        "annotation_count": 50,
                    }
                ],
                "scenes": [
                    {
                        "scene_id": "holdout-scene",
                        "image_count": 50,
                        "annotation_count": 50,
                    }
                ],
                "assets": [
                    {
                        "asset_id": "holdout-asset",
                        "leakage_group_id": f"sha256:{source_sha}",
                        "image_count": 50,
                        "annotation_count": 50,
                        "scene_ids": ["holdout-scene"],
                    }
                ],
            },
            "files": sorted(files, key=lambda item: str(item["path"])),
        },
    )

    evidence = tmp_path / "evidence"
    decisions = evidence / "final-decisions.json"
    _write_json(
        decisions,
        {
            "schema_version": "1.3",
            "scope": "full_review",
            "mutation_performed": False,
            "task_id": FINAL_HOLDOUT_TASK_ID,
            "frame_reviews": [
                {"frame": frame, "reviewed": True, "actions": [], "resolved_flag_ids": []}
                for frame in range(50)
            ],
        },
    )
    canonical_sha = "b" * 64
    snapshot = evidence / "post-snapshot.json"
    _write_json(
        snapshot,
        {
            "snapshot_schema": builder.SNAPSHOT_SCHEMA,
            "task": {"id": FINAL_HOLDOUT_TASK_ID, "size": 50},
            "jobs": [
                {
                    "id": FINAL_HOLDOUT_JOB_ID,
                    "task_id": FINAL_HOLDOUT_TASK_ID,
                    "start_frame": 0,
                    "stop_frame": 49,
                    "frame_count": 50,
                    "type": "annotation",
                    "parent_job_id": None,
                    "issues": {"count": 0},
                    "stage": "annotation",
                    "state": "new",
                }
            ],
            "counts": {"shapes": 50},
            "final_gate": {"passed": True, "blocking_reasons": []},
            "canonical_annotations_sha256": canonical_sha,
        },
    )
    receipt = evidence / "completion-receipt.json"
    _write_json(
        receipt,
        {
            "schema": builder.RECEIPT_SCHEMA,
            "task_id": FINAL_HOLDOUT_TASK_ID,
            "job_id": FINAL_HOLDOUT_JOB_ID,
            "dry_run": False,
            "mutation_performed": True,
            "job_state_after": "completed",
            "annotation_count": 50,
            "verified_post_completion_canonical_annotations_sha256": canonical_sha,
            "decision_validation": {
                "valid": True,
                "mutation_performed": False,
                "scope": "full_review",
                "task_id": FINAL_HOLDOUT_TASK_ID,
                "snapshot_frame_count": 50,
                "reviewed_frame_count": 50,
                "unresolved_manual_flag_count": 0,
            },
            "evidence": {
                "post_snapshot": {"path": str(snapshot), "sha256": _digest(snapshot)},
                "decisions": {"path": str(decisions), "sha256": _digest(decisions)},
            },
        },
    )
    completed = evidence / "completed-snapshot.json"
    completed_payload = json.loads(snapshot.read_text(encoding="utf-8"))
    completed_payload["jobs"][0]["state"] = "completed"
    _write_json(completed, completed_payload)
    manifest_path = reference / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["evidence"] = {
        "task_inputs": [
            {
                "task_id": FINAL_HOLDOUT_TASK_ID,
                "job_id": FINAL_HOLDOUT_JOB_ID,
                "snapshot": {
                    "sha256": _digest(completed),
                    "canonical_annotations_sha256": canonical_sha,
                },
                "completion_receipt": {"sha256": _digest(receipt)},
            }
        ]
    }
    _write_json(manifest_path, manifest)
    return {
        "reference": reference,
        "receipt": receipt,
        "snapshot": snapshot,
        "completed": completed,
        "decisions": decisions,
        "output": tmp_path / "frozen-v1",
    }


def _build(paths: dict[str, Path]) -> dict[str, object]:
    return builder.build_frozen_holdout_reference(
        reference_dir=paths["reference"],
        task_id=FINAL_HOLDOUT_TASK_ID,
        job_id=FINAL_HOLDOUT_JOB_ID,
        completion_receipt=paths["receipt"],
        post_snapshot=paths["snapshot"],
        final_decisions=paths["decisions"],
        output_dir=paths["output"],
    )


def test_builds_evaluator_compatible_frozen_reference(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)

    summary = _build(paths)

    manifest_path = paths["output"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert summary["annotation_count"] == 50
    assert manifest["schema"] == builder.OUTPUT_SCHEMA
    assert manifest["task_id"] == FINAL_HOLDOUT_TASK_ID
    assert set(manifest["review_completion"]) == {
        "completion_receipt",
        "post_snapshot",
        "final_decisions",
        "source_canonical_annotations_sha256",
    }
    assert len(manifest["files"]) == 54
    bound_manifest = evaluator.BoundFile(
        manifest_path.resolve(), _digest(manifest_path), manifest_path.stat().st_size
    )
    holdout = evaluator._build_reference_data(
        bound_manifest,
        expected_schema=evaluator.HOLDOUT_SCHEMA,
        annotations_record=None,
        location="test holdout",
        verify_all_files=True,
    )
    completion = evaluator._validate_full_review_completion(
        holdout,
        builder.FinalHoldoutIdentity(FINAL_HOLDOUT_TASK_ID, FINAL_HOLDOUT_JOB_ID),
    )
    assert completion["reviewed_frame_count"] == 50
    assert completion["annotation_count"] == 50


def test_refuses_to_replace_existing_output(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    paths["output"].mkdir()
    sentinel = paths["output"] / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(builder.FrozenHoldoutReferenceError, match="already exists"):
        _build(paths)

    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_binds_optional_completed_snapshot_to_source_reference(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)

    builder.build_frozen_holdout_reference(
        reference_dir=paths["reference"],
        task_id=FINAL_HOLDOUT_TASK_ID,
        job_id=FINAL_HOLDOUT_JOB_ID,
        completion_receipt=paths["receipt"],
        post_snapshot=paths["snapshot"],
        final_decisions=paths["decisions"],
        completed_snapshot=paths["completed"],
        output_dir=paths["output"],
    )

    manifest = json.loads((paths["output"] / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["additional_evidence"]["completed_snapshot"]["sha256"] == _digest(
        paths["completed"]
    )
    assert len(manifest["files"]) == 55


@pytest.mark.parametrize("binding", ["canonical", "receipt"])
def test_rejects_source_reference_review_binding_mismatch(tmp_path: Path, binding: str) -> None:
    paths = _fixture(tmp_path)
    manifest_path = paths["reference"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    task_input = manifest["evidence"]["task_inputs"][0]
    if binding == "canonical":
        task_input["snapshot"]["canonical_annotations_sha256"] = "c" * 64
        expected = "canonical annotations differ"
    else:
        task_input["completion_receipt"]["sha256"] = "c" * 64
        expected = "completion receipt differs"
    _write_json(manifest_path, manifest)

    with pytest.raises(builder.FrozenHoldoutReferenceError, match=expected):
        _build(paths)


def test_rejects_false_source_reference_gate_check(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    manifest_path = paths["reference"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["gate"]["checks"]["source_reference_verified"] = False
    _write_json(manifest_path, manifest)

    with pytest.raises(builder.FrozenHoldoutReferenceError, match="gate did not pass"):
        _build(paths)


@pytest.mark.parametrize("statistic", ["scenes", "assets"])
def test_rejects_source_statistics_that_disagree_with_coco(tmp_path: Path, statistic: str) -> None:
    paths = _fixture(tmp_path)
    manifest_path = paths["reference"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_statistics"][statistic][0]["image_count"] = 49
    _write_json(manifest_path, manifest)

    with pytest.raises(builder.FrozenHoldoutReferenceError, match=f"{statistic[:-1]} statistics"):
        _build(paths)


def test_rejects_coco_image_hash_that_disagrees_with_managed_file(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    coco_path = paths["reference"] / "annotations.coco.json"
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    coco["images"][0]["sha256"] = "c" * 64
    _write_json(coco_path, coco)
    manifest_path = paths["reference"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    annotations_record = next(
        record for record in manifest["files"] if record["path"] == "annotations.coco.json"
    )
    annotations_record["sha256"] = _digest(coco_path)
    annotations_record["size_bytes"] = coco_path.stat().st_size
    _write_json(manifest_path, manifest)

    with pytest.raises(builder.FrozenHoldoutReferenceError, match="COCO image hash differs"):
        _build(paths)


def test_rejects_source_managed_path_in_reserved_evidence_namespace(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    injected = paths["reference"] / "evidence" / "untrusted.json"
    _write_json(injected, {"untrusted": True})
    manifest_path = paths["reference"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"].append(
        {
            "path": "evidence/untrusted.json",
            "sha256": _digest(injected),
            "size_bytes": injected.stat().st_size,
        }
    )
    _write_json(manifest_path, manifest)

    with pytest.raises(builder.FrozenHoldoutReferenceError, match="reserved output namespace"):
        _build(paths)


@pytest.mark.parametrize("tamper", ["category_mapping", "bbox_bounds"])
def test_rejects_coco_that_would_fail_evaluator_contract(tmp_path: Path, tamper: str) -> None:
    paths = _fixture(tmp_path)
    coco_path = paths["reference"] / "annotations.coco.json"
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    if tamper == "category_mapping":
        coco["categories"][0]["id"] = 8
        coco["categories"][7]["id"] = 1
        expected = "canonical eight-class ID mapping"
    else:
        coco["annotations"][0]["bbox"] = [9.0, 9.0, 2.0, 2.0]
        expected = "bbox exceeds image bounds"
    _write_json(coco_path, coco)
    manifest_path = paths["reference"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    annotations_record = next(
        record for record in manifest["files"] if record["path"] == "annotations.coco.json"
    )
    annotations_record["sha256"] = _digest(coco_path)
    annotations_record["size_bytes"] = coco_path.stat().st_size
    _write_json(manifest_path, manifest)

    with pytest.raises(builder.FrozenHoldoutReferenceError, match=expected):
        _build(paths)


@pytest.mark.parametrize("tamper", ["receipt", "snapshot"])
def test_rejects_receipt_snapshot_binding_tamper(tmp_path: Path, tamper: str) -> None:
    paths = _fixture(tmp_path)
    if tamper == "receipt":
        receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
        receipt["evidence"]["post_snapshot"]["sha256"] = "0" * 64
        _write_json(paths["receipt"], receipt)
    else:
        snapshot = json.loads(paths["snapshot"].read_text(encoding="utf-8"))
        snapshot["tampered"] = True
        _write_json(paths["snapshot"], snapshot)

    with pytest.raises(builder.FrozenHoldoutReferenceError, match="not bound"):
        _build(paths)


@pytest.mark.parametrize(
    ("source_key", "output_relative"),
    [
        ("managed_image", "images/final-holdout-fixture/frame-000.jpg"),
        ("receipt", "evidence/completion-receipt.json"),
    ],
)
def test_publishes_from_verified_bytes_after_same_size_path_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_key: str,
    output_relative: str,
) -> None:
    paths = _fixture(tmp_path)
    source = (
        paths["reference"] / "images/final-holdout-fixture/frame-000.jpg"
        if source_key == "managed_image"
        else paths["receipt"]
    )
    verified_payload = source.read_bytes()
    attacker_payload = bytes([verified_payload[0] ^ 1]) + verified_payload[1:]
    original_inode = source.stat().st_ino
    original_mkdtemp = builder.tempfile.mkdtemp
    replaced = False

    def replace_after_validation(*args: object, **kwargs: object) -> str:
        nonlocal replaced
        staging = original_mkdtemp(*args, **kwargs)
        replacement = source.with_name(f"{source.name}.replacement")
        replacement.write_bytes(attacker_payload)
        replacement.replace(source)
        assert source.stat().st_size == len(verified_payload)
        assert source.stat().st_ino != original_inode
        replaced = True
        return staging

    monkeypatch.setattr(builder.tempfile, "mkdtemp", replace_after_validation)

    _build(paths)

    assert replaced is True
    assert source.read_bytes() == attacker_payload
    assert (paths["output"] / output_relative).read_bytes() == verified_payload


def test_rejects_same_size_path_replacement_during_stable_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    source = paths["receipt"]
    original_payload = source.read_bytes()
    replacement = source.with_name(f"{source.name}.replacement")
    replacement.write_bytes(bytes([original_payload[0] ^ 1]) + original_payload[1:])
    source_identity = (source.stat().st_dev, source.stat().st_ino)
    original_read = builder.os.read
    replaced = False

    def replace_during_read(descriptor: int, size: int) -> bytes:
        nonlocal replaced
        payload = original_read(descriptor, size)
        metadata = builder.os.fstat(descriptor)
        if not replaced and (metadata.st_dev, metadata.st_ino) == source_identity:
            source.replace(source.with_name(f"{source.name}.original"))
            replacement.replace(source)
            replaced = True
        return payload

    monkeypatch.setattr(builder.os, "read", replace_during_read)

    with pytest.raises(builder.FrozenHoldoutReferenceError, match="changed while being read"):
        _build(paths)

    assert replaced is True


@pytest.mark.parametrize("source_key", ["manifest", "receipt"])
def test_rejects_symlink_inputs(tmp_path: Path, source_key: str) -> None:
    paths = _fixture(tmp_path)
    source = paths["reference"] / "manifest.json" if source_key == "manifest" else paths["receipt"]
    real_source = source.with_name(f"{source.name}.real")
    source.replace(real_source)
    source.symlink_to(real_source)

    with pytest.raises(builder.FrozenHoldoutReferenceError, match="unavailable"):
        _build(paths)
