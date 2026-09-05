from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

import scripts.build_training_cv_plan as cv_plan
from scripts.build_training_cv_plan import TrainingCVPlanError, build_training_cv_plan
from scripts.prepare_yolo_dataset import prepare_yolo_dataset

SYNTHETIC_TRAINING_TASK_ID = 41001
SYNTHETIC_TRAINING_JOB_ID = 51001
OTHER_SYNTHETIC_TASK_ID = 41002
FINAL_HOLDOUT_TASK_ID = 91001
FINAL_HOLDOUT_JOB_ID = 92001


@pytest.fixture(autouse=True)
def _configure_final_holdout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROADLABELOPS_FINAL_HOLDOUT_TASK_IDS", str(FINAL_HOLDOUT_TASK_ID))
    monkeypatch.setenv("ROADLABELOPS_FINAL_HOLDOUT_JOB_IDS", str(FINAL_HOLDOUT_JOB_ID))


@dataclass
class ReferenceFixture:
    workspace: Path
    root: Path
    manifest: Path
    coco: Path
    coco_payload: dict[str, Any]
    manifest_payload: dict[str, Any]

    def write_manifest(self) -> None:
        self.manifest.write_bytes(_json_bytes(self.manifest_payload))

    def refresh(self) -> None:
        self.coco.write_bytes(_json_bytes(self.coco_payload))
        annotations_by_image: dict[int, list[dict[str, Any]]] = {}
        for annotation in self.coco_payload["annotations"]:
            annotations_by_image.setdefault(annotation["image_id"], []).append(annotation)
        by_category = {
            name: sum(
                annotation["category_id"] == index
                for annotation in self.coco_payload["annotations"]
            )
            for index, name in enumerate(cv_plan.REQUIRED_LABELS, start=1)
        }
        self.manifest_payload["counts"] = {
            "tasks": len({image["task_id"] for image in self.coco_payload["images"]}),
            "images": len(self.coco_payload["images"]),
            "zero_annotation_images": sum(
                not annotations_by_image.get(image["id"]) for image in self.coco_payload["images"]
            ),
            "annotations": len(self.coco_payload["annotations"]),
            "categories": len(cv_plan.REQUIRED_LABELS),
            "annotations_by_category": by_category,
        }
        source_records: list[dict[str, Any]] = []
        for asset in self.manifest_payload["evidence"]["source_map"]["assets"]:
            asset_id = asset["asset_id"]
            images = [
                image
                for image in self.coco_payload["images"]
                if type(image["source_asset_id"]) is type(asset_id)
                and image["source_asset_id"] == asset_id
            ]
            image_ids = {image["id"] for image in images}
            source_records.append(
                {
                    "asset_id": asset_id,
                    "leakage_group_id": asset["leakage_group_id"],
                    "image_count": len(images),
                    "annotation_count": sum(
                        annotation["image_id"] in image_ids
                        for annotation in self.coco_payload["annotations"]
                    ),
                    "scene_ids": sorted({image["scene_id"] for image in images}),
                }
            )
        self.manifest_payload["source_statistics"] = {"assets": source_records}
        files = [
            {
                "path": "annotations.coco.json",
                "sha256": _sha256(self.coco),
                "size_bytes": self.coco.stat().st_size,
            }
        ]
        for image in self.coco_payload["images"]:
            image_path = self.root / image["file_name"]
            files.append(
                {
                    "path": image["file_name"],
                    "sha256": _sha256(image_path),
                    "size_bytes": image_path.stat().st_size,
                }
            )
        self.manifest_payload["files"] = files
        self.write_manifest()


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True).encode(
            "utf-8"
        )
        + b"\n"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_image(path: Path, color: tuple[int, int, int]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color).save(path)
    return _sha256(path)


@pytest.fixture
def reference(tmp_path: Path) -> ReferenceFixture:
    workspace = tmp_path / "workspace"
    root = workspace / "data" / "ground-truth" / "training-reference-v1"
    root.mkdir(parents=True)
    labels_sha = hashlib.sha256(b"labels").hexdigest()
    source_map_sha = hashlib.sha256(b"source-map").hexdigest()
    assets: list[dict[str, Any]] = []
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    annotation_id = 1
    for asset_index in range(5):
        asset_id = f"asset-{asset_index + 1}"
        source_sha = hashlib.sha256(asset_id.encode("utf-8")).hexdigest()
        leakage_group = f"sha256:{source_sha}"
        assets.append(
            {
                "asset_id": asset_id,
                "sha256": source_sha,
                "leakage_group_id": leakage_group,
            }
        )
        annotated_id = asset_index * 2 + 1
        zero_id = annotated_id + 1
        for image_id, suffix in ((annotated_id, "positive"), (zero_id, "zero")):
            file_name = f"images/task-41001/{asset_id}-{suffix}.png"
            color = (20 + asset_index * 30, 20 if suffix == "positive" else 40, image_id)
            images.append(
                {
                    "id": image_id,
                    "file_name": file_name,
                    "width": 32,
                    "height": 32,
                    "sha256": _write_image(root / file_name, color),
                    "task_id": SYNTHETIC_TRAINING_TASK_ID,
                    "cvat_frame": image_id - 1,
                    "sample_index": image_id - 1,
                    "scene_id": f"scene-{asset_index + 1}",
                    "source_frame": image_id * 5,
                    "source_asset_id": asset_id,
                    "source_leakage_group_id": leakage_group,
                    "source_normalized_asset_frame": asset_index * 100 + (image_id % 2),
                }
            )
        for category_id, label in enumerate(cv_plan.REQUIRED_LABELS, start=1):
            if label in {"bicycle", "truck"} and asset_index not in {0, 1}:
                continue
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": annotated_id,
                    "category_id": category_id,
                    "bbox": [1, 1, 8, 8],
                    "area": 64,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
    categories = [
        {"id": index, "name": label, "supercategory": "road_object"}
        for index, label in enumerate(cv_plan.REQUIRED_LABELS, start=1)
    ]
    coco_payload = {
        "info": {
            "description": "synthetic training reference",
            "schema": cv_plan.REFERENCE_SCHEMA,
            "labels_sha256": labels_sha,
            "source_map_sha256": source_map_sha,
        },
        "licenses": [],
        "categories": categories,
        "images": images,
        "annotations": annotations,
    }
    manifest_payload = {
        "schema": cv_plan.REFERENCE_SCHEMA,
        "gate": {
            "passed": True,
            "blocking_reasons": [],
            "checks": {name: True for name in cv_plan.REQUIRED_REFERENCE_GATE_CHECKS},
        },
        "evidence": {
            "labels": {
                "path": "config/road_labels.yaml",
                "path_kind": "workspace_relative",
                "sha256": labels_sha,
            },
            "source_map": {
                "path": "docs/evidence/training-source-map.json",
                "path_kind": "workspace_relative",
                "sha256": source_map_sha,
                "assets": assets,
            },
            "task_inputs": [
                {
                    "task_id": SYNTHETIC_TRAINING_TASK_ID,
                    "job_id": SYNTHETIC_TRAINING_JOB_ID,
                    "snapshot": {
                        "path": "data/sessions/training/task-41001/snapshot.json",
                        "path_kind": "workspace_relative",
                        "sha256": "a" * 64,
                    },
                    "image_manifest": {
                        "path": "data/sessions/training/task-41001/images.json",
                        "path_kind": "workspace_relative",
                        "sha256": "b" * 64,
                    },
                    "completion_receipt": {
                        "path": "data/sessions/training/task-41001/completion.json",
                        "path_kind": "workspace_relative",
                        "sha256": "c" * 64,
                    },
                }
            ],
        },
        "counts": {},
        "source_statistics": {},
        "files": [],
    }
    fixture = ReferenceFixture(
        workspace=workspace,
        root=root,
        manifest=root / "manifest.json",
        coco=root / "annotations.coco.json",
        coco_payload=coco_payload,
        manifest_payload=manifest_payload,
    )
    fixture.refresh()
    return fixture


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_builds_deterministic_loso_plans_consumable_by_yolo_preparer(
    reference: ReferenceFixture,
) -> None:
    first = reference.workspace / "docs" / "evidence" / "cv-plan-a"
    second = reference.workspace / "docs" / "evidence" / "cv-plan-b"

    result = build_training_cv_plan(reference.manifest, first, workspace_root=reference.workspace)
    build_training_cv_plan(reference.manifest, second, workspace_root=reference.workspace)

    assert result["gate"]["passed"] is True
    assert _tree_bytes(first) == _tree_bytes(second)
    manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == cv_plan.OUTPUT_SCHEMA
    assert manifest["counts"] == {
        "annotations": 34,
        "categories": 8,
        "folds": 5,
        "images": 10,
        "source_assets": 5,
        "source_groups": 5,
        "zero_annotation_images": 5,
    }
    assert manifest["source_statistics"]["class_source_support"]["bicycle"] == {
        "box_count": 2,
        "positive_image_count": 2,
        "source_asset_ids": ["asset-1", "asset-2"],
        "source_count": 2,
    }
    assert [fold["val_asset_id"] for fold in manifest["folds"]] == [
        "asset-1",
        "asset-2",
        "asset-3",
        "asset-4",
        "asset-5",
    ]
    assert manifest["folds"][2]["validation_evaluability"]["bicycle"] == "not_evaluable"
    assert manifest["folds"][0]["validation_evaluability"]["bicycle"] == "evaluable"
    for fold in manifest["folds"]:
        split_path = first / fold["split_plan"]["path"]
        split = json.loads(split_path.read_text(encoding="utf-8"))
        assert set(split) == {"schema", "train_asset_ids", "val_asset_ids"}
        assert split["schema"] == cv_plan.SPLIT_PLAN_SCHEMA
        assert len(split["train_asset_ids"]) == 4
        assert len(split["val_asset_ids"]) == 1

    yolo_output = reference.workspace / "data" / "training" / "fold-01-yolo"
    prepared = prepare_yolo_dataset(
        reference.coco,
        first / manifest["folds"][0]["split_plan"]["path"],
        yolo_output,
        reference_manifest_path=reference.manifest,
    )
    assert prepared["gate"]["passed"] is True


def test_builds_six_source_loso_plan_from_reference(
    reference: ReferenceFixture,
) -> None:
    asset_index = 5
    asset_id = "asset-6"
    source_sha = hashlib.sha256(asset_id.encode("utf-8")).hexdigest()
    leakage_group = f"sha256:{source_sha}"
    reference.manifest_payload["evidence"]["source_map"]["assets"].append(
        {
            "asset_id": asset_id,
            "sha256": source_sha,
            "leakage_group_id": leakage_group,
        }
    )
    annotated_id = max(image["id"] for image in reference.coco_payload["images"]) + 1
    zero_id = annotated_id + 1
    for image_id, suffix in ((annotated_id, "positive"), (zero_id, "zero")):
        file_name = f"images/task-41001/{asset_id}-{suffix}.png"
        reference.coco_payload["images"].append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": 32,
                "height": 32,
                "sha256": _write_image(
                    reference.root / file_name,
                    (20 + asset_index * 30, 20 if suffix == "positive" else 40, image_id),
                ),
                "task_id": SYNTHETIC_TRAINING_TASK_ID,
                "cvat_frame": image_id - 1,
                "sample_index": image_id - 1,
                "scene_id": "scene-6",
                "source_frame": image_id * 5,
                "source_asset_id": asset_id,
                "source_leakage_group_id": leakage_group,
                "source_normalized_asset_frame": asset_index * 100 + (image_id % 2),
            }
        )
    annotation_id = max(annotation["id"] for annotation in reference.coco_payload["annotations"])
    for category_id, _label in enumerate(cv_plan.REQUIRED_LABELS, start=1):
        annotation_id += 1
        reference.coco_payload["annotations"].append(
            {
                "id": annotation_id,
                "image_id": annotated_id,
                "category_id": category_id,
                "bbox": [1, 1, 8, 8],
                "area": 64,
                "iscrowd": 0,
            }
        )
    reference.refresh()

    output = reference.workspace / "docs" / "evidence" / "six-source-plan"
    build_training_cv_plan(reference.manifest, output, workspace_root=reference.workspace)

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"]["source_assets"] == 6
    assert manifest["counts"]["source_groups"] == 6
    assert manifest["counts"]["folds"] == 6
    assert manifest["gate"]["checks"]["minimum_unique_source_group_count_met"] is True
    assert [fold["val_asset_id"] for fold in manifest["folds"]] == [
        "asset-1",
        "asset-2",
        "asset-3",
        "asset-4",
        "asset-5",
        "asset-6",
    ]
    for fold in manifest["folds"]:
        assert len(fold["train"]["asset_ids"]) == 5
        assert len(fold["val"]["asset_ids"]) == 1
        assert set(fold["train"]["asset_ids"]).isdisjoint(fold["val"]["asset_ids"])


@pytest.mark.parametrize("failure", ["reference_gate", "class_support"])
def test_rejects_failed_reference_and_hard_readiness_gates(
    reference: ReferenceFixture, failure: str
) -> None:
    if failure == "reference_gate":
        reference.manifest_payload["gate"]["passed"] = False
        reference.write_manifest()
    elif failure == "class_support":
        image_ids = {
            image["id"]
            for image in reference.coco_payload["images"]
            if image["source_asset_id"] == "asset-2"
        }
        reference.coco_payload["annotations"] = [
            annotation
            for annotation in reference.coco_payload["annotations"]
            if not (annotation["image_id"] in image_ids and annotation["category_id"] == 5)
        ]
        reference.refresh()
    with pytest.raises(TrainingCVPlanError):
        build_training_cv_plan(
            reference.manifest,
            reference.workspace / "docs" / "evidence" / f"failed-{failure}",
            workspace_root=reference.workspace,
        )


def test_rejects_fewer_than_three_source_groups(reference: ReferenceFixture) -> None:
    assets = reference.manifest_payload["evidence"]["source_map"]["assets"]
    del assets[2:]
    retained_asset_ids = {asset["asset_id"] for asset in assets}
    reference.coco_payload["images"] = [
        image
        for image in reference.coco_payload["images"]
        if image["source_asset_id"] in retained_asset_ids
    ]
    retained_image_ids = {image["id"] for image in reference.coco_payload["images"]}
    reference.coco_payload["annotations"] = [
        annotation
        for annotation in reference.coco_payload["annotations"]
        if annotation["image_id"] in retained_image_ids
    ]
    reference.refresh()

    with pytest.raises(TrainingCVPlanError, match="at least 3 unique source groups"):
        build_training_cv_plan(
            reference.manifest,
            reference.workspace / "docs" / "evidence" / "too-few-source-groups",
            workspace_root=reference.workspace,
        )


@pytest.mark.parametrize("failure", ["duplicate_asset", "duplicate_leakage_group"])
def test_rejects_duplicate_source_identities(reference: ReferenceFixture, failure: str) -> None:
    assets = reference.manifest_payload["evidence"]["source_map"]["assets"]
    if failure == "duplicate_asset":
        assets[-1]["asset_id"] = assets[0]["asset_id"]
    else:
        assets[-1]["sha256"] = assets[0]["sha256"]
        assets[-1]["leakage_group_id"] = assets[0]["leakage_group_id"]
    reference.write_manifest()

    with pytest.raises(TrainingCVPlanError):
        build_training_cv_plan(
            reference.manifest,
            reference.workspace / "docs" / "evidence" / f"failed-{failure}",
            workspace_root=reference.workspace,
        )


@pytest.mark.parametrize("managed_file", ["coco", "image"])
def test_rejects_managed_file_hash_or_size_drift(
    reference: ReferenceFixture, managed_file: str
) -> None:
    record = (
        reference.manifest_payload["files"][0]
        if managed_file == "coco"
        else reference.manifest_payload["files"][1]
    )
    record["sha256"] = "0" * 64
    reference.write_manifest()

    with pytest.raises(TrainingCVPlanError, match="hash or size"):
        build_training_cv_plan(
            reference.manifest,
            reference.workspace / "docs" / "evidence" / f"drift-{managed_file}",
            workspace_root=reference.workspace,
        )


def test_rejects_data_holdout_and_paths_outside_ground_truth(
    reference: ReferenceFixture,
) -> None:
    holdout_reference = reference.workspace / "data" / "holdout" / "reference"
    shutil.copytree(reference.root, holdout_reference)
    with pytest.raises(TrainingCVPlanError, match="data/holdout"):
        build_training_cv_plan(
            holdout_reference / "manifest.json",
            reference.workspace / "docs" / "evidence" / "holdout-attempt",
            workspace_root=reference.workspace,
        )

    outside = reference.workspace / "docs" / "training-reference" / "manifest.json"
    outside.parent.mkdir(parents=True)
    shutil.copy2(reference.manifest, outside)
    with pytest.raises(TrainingCVPlanError, match="data/ground-truth"):
        build_training_cv_plan(
            outside,
            reference.workspace / "docs" / "evidence" / "outside-attempt",
            workspace_root=reference.workspace,
        )


@pytest.mark.parametrize(
    "relative_path",
    [
        "data/holdout/reference/manifest.json",
        "data/ground-truth/task-91001/manifest.json",
        "data/ground-truth/task/91001/manifest.json",
        "data/ground-truth/job-092001/manifest.json",
    ],
)
def test_rejects_forbidden_reference_path_before_any_open(
    reference: ReferenceFixture,
    monkeypatch: pytest.MonkeyPatch,
    relative_path: str,
) -> None:
    opened: list[object] = []

    def unexpected_open(*args: object, **_kwargs: object) -> int:
        opened.append(args[0] if args else None)
        raise AssertionError("forbidden input reached os.open")

    monkeypatch.setattr(cv_plan.os, "open", unexpected_open)
    with pytest.raises(TrainingCVPlanError, match="outside the training-only contract"):
        build_training_cv_plan(
            reference.workspace / relative_path,
            reference.workspace / "docs" / "evidence" / "forbidden-path-attempt",
            workspace_root=reference.workspace,
        )
    assert opened == []


@pytest.mark.parametrize(
    "declared_path",
    [
        "data/holdout/snapshot.json",
        "data/sessions/training/task-91001/snapshot.json",
        "data/sessions/training/task/91001/snapshot.json",
        "data/sessions/training/job-092001/snapshot.json",
    ],
)
def test_rejects_forbidden_declared_path_before_managed_coco_open(
    reference: ReferenceFixture,
    declared_path: str,
) -> None:
    reference.manifest_payload["evidence"]["task_inputs"][0]["snapshot"]["path"] = declared_path
    reference.write_manifest()
    reference.coco.unlink()

    with pytest.raises(TrainingCVPlanError, match="outside the training-only contract"):
        build_training_cv_plan(
            reference.manifest,
            reference.workspace / "docs" / "evidence" / "forbidden-declaration-attempt",
            workspace_root=reference.workspace,
        )


@pytest.mark.parametrize("target", ["manifest", "managed_image"])
def test_rejects_symlink_inputs(reference: ReferenceFixture, target: str) -> None:
    if target == "manifest":
        alias_root = reference.workspace / "data" / "ground-truth" / "alias"
        alias_root.mkdir()
        alias = alias_root / "manifest.json"
        alias.symlink_to(reference.manifest)
        manifest = alias
    else:
        manifest = reference.manifest
        image = reference.root / reference.coco_payload["images"][0]["file_name"]
        replacement = image.with_suffix(".original")
        image.rename(replacement)
        image.symlink_to(replacement.name)

    with pytest.raises(TrainingCVPlanError, match="symlink"):
        build_training_cv_plan(
            manifest,
            reference.workspace / "docs" / "evidence" / f"symlink-{target}",
            workspace_root=reference.workspace,
        )


@pytest.mark.parametrize("existing_kind", ["directory", "file", "symlink"])
def test_never_overwrites_existing_output(reference: ReferenceFixture, existing_kind: str) -> None:
    output = reference.workspace / "docs" / "evidence" / f"existing-{existing_kind}"
    output.parent.mkdir(parents=True, exist_ok=True)
    if existing_kind == "directory":
        output.mkdir()
        marker = output / "marker.txt"
        marker.write_text("keep", encoding="utf-8")
    elif existing_kind == "file":
        output.write_text("keep", encoding="utf-8")
    else:
        target = output.parent / "symlink-target"
        target.mkdir()
        output.symlink_to(target.name)

    with pytest.raises(TrainingCVPlanError, match="already exists"):
        build_training_cv_plan(reference.manifest, output, workspace_root=reference.workspace)
    if existing_kind == "directory":
        assert marker.read_text(encoding="utf-8") == "keep"
    elif existing_kind == "file":
        assert output.read_text(encoding="utf-8") == "keep"
    else:
        assert output.is_symlink()


def test_rejects_symlink_output_parent_without_writing_through_it(
    reference: ReferenceFixture, tmp_path: Path
) -> None:
    outside = tmp_path / "outside-output"
    outside.mkdir()
    linked_parent = reference.workspace / "linked-output"
    linked_parent.symlink_to(outside, target_is_directory=True)
    output = linked_parent / "nested" / "plan"

    with pytest.raises(TrainingCVPlanError, match="output parent must not contain symlinks"):
        build_training_cv_plan(reference.manifest, output, workspace_root=reference.workspace)

    assert not (outside / "nested").exists()


@pytest.mark.parametrize("identifier_field", ["task_id", "job_id"])
def test_rejects_consumed_task_binding_before_managed_files_are_read(
    reference: ReferenceFixture, identifier_field: str
) -> None:
    payload = copy.deepcopy(reference.manifest_payload)
    payload["evidence"]["task_inputs"][0][identifier_field] = (
        FINAL_HOLDOUT_TASK_ID if identifier_field == "task_id" else FINAL_HOLDOUT_JOB_ID
    )
    reference.manifest_payload = payload
    reference.write_manifest()
    image = reference.root / reference.coco_payload["images"][0]["file_name"]
    image.unlink()

    with pytest.raises(TrainingCVPlanError, match="outside the training-only contract"):
        build_training_cv_plan(
            reference.manifest,
            reference.workspace / "docs" / "evidence" / "consumed-task-attempt",
            workspace_root=reference.workspace,
        )


@pytest.mark.parametrize(
    ("task_id", "message"),
    [
        (FINAL_HOLDOUT_TASK_ID, "configured final-holdout"),
        (OTHER_SYNTHETIC_TASK_ID, "not declared"),
    ],
)
def test_rejects_coco_image_task_outside_declared_training_tasks_before_image_open(
    reference: ReferenceFixture,
    task_id: int,
    message: str,
) -> None:
    reference.coco_payload["images"][0]["task_id"] = task_id
    reference.refresh()
    first_image = reference.root / reference.coco_payload["images"][0]["file_name"]
    first_image.unlink()

    with pytest.raises(TrainingCVPlanError, match=message):
        build_training_cv_plan(
            reference.manifest,
            reference.workspace / "docs" / "evidence" / f"coco-task-{task_id}-attempt",
            workspace_root=reference.workspace,
        )


def test_cli_reports_success(
    reference: ReferenceFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    output = reference.workspace / "docs" / "evidence" / "cli-plan"
    previous = Path.cwd()
    os.chdir(reference.workspace)
    try:
        cv_plan.main(
            [
                "--reference-manifest",
                str(reference.manifest.relative_to(reference.workspace)),
                "--output",
                str(output.relative_to(reference.workspace)),
            ]
        )
    finally:
        os.chdir(previous)
    result = json.loads(capsys.readouterr().out)
    assert result["gate"]["passed"] is True
    assert Path(result["manifest"]) == output / "manifest.json"
