from __future__ import annotations

import copy
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml
from PIL import Image

import scripts.prepare_yolo_dataset as prepare
from scripts.prepare_yolo_dataset import (
    DatasetPreparationError,
    prepare_yolo_dataset,
    sha256,
    yolo_line,
)


@dataclass
class Fixture:
    root: Path
    coco: Path
    reference_manifest: Path
    split_plan: Path
    coco_payload: dict[str, Any]
    split_payload: dict[str, Any]

    def write_coco(
        self,
        path: Path | None = None,
        *,
        reference_manifest_path: Path | None = None,
    ) -> Path:
        destination = path or self.coco
        destination.write_text(
            json.dumps(self.coco_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.write_reference_manifest(
            coco_path=destination,
            path=reference_manifest_path or self.reference_manifest,
        )
        return destination

    def write_reference_manifest(
        self,
        *,
        coco_path: Path | None = None,
        path: Path | None = None,
    ) -> Path:
        source_coco = coco_path or self.coco
        destination = path or self.reference_manifest
        files = [
            {
                "path": "annotations.coco.json",
                "sha256": sha256(source_coco),
                "size_bytes": source_coco.stat().st_size,
            }
        ]
        for image in self.coco_payload["images"]:
            file_name = image.get("file_name")
            candidate = source_coco.parent / str(file_name)
            files.append(
                {
                    "path": file_name,
                    "sha256": image.get("sha256", "0" * 64),
                    "size_bytes": candidate.stat().st_size if candidate.is_file() else 0,
                }
            )
        source_assets: list[dict[str, Any]] = []
        seen_assets: set[tuple[str, str]] = set()
        for image in self.coco_payload["images"]:
            asset_id = image.get("source_asset_id")
            identity = (type(asset_id).__name__, str(asset_id))
            if identity in seen_assets:
                continue
            seen_assets.add(identity)
            leakage_group_id = image.get("source_leakage_group_id")
            source_sha256 = (
                leakage_group_id.removeprefix("sha256:")
                if isinstance(leakage_group_id, str)
                and leakage_group_id.startswith("sha256:")
                and len(leakage_group_id) == 71
                else "0" * 64
            )
            source_assets.append(
                {
                    "asset_id": asset_id,
                    "sha256": source_sha256,
                    "leakage_group_id": leakage_group_id,
                }
            )
        payload = {
            "schema": prepare.REFERENCE_SCHEMA,
            "gate": {
                "passed": True,
                "blocking_reasons": [],
                "checks": {name: True for name in prepare.REFERENCE_REQUIRED_GATE_CHECKS},
            },
            "evidence": {
                "labels": {"sha256": self.coco_payload["info"]["labels_sha256"]},
                "source_map": {
                    "sha256": self.coco_payload["info"]["source_map_sha256"],
                    "assets": source_assets,
                },
            },
            "files": files,
        }
        destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return destination

    def write_split(self, path: Path | None = None) -> Path:
        destination = path or self.split_plan
        destination.write_text(
            json.dumps(self.split_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return destination


def _write_image(path: Path, color: tuple[int, int, int]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 10), color).save(path)
    return sha256(path)


@pytest.fixture
def fixture(tmp_path: Path) -> Fixture:
    source = tmp_path / "reference"
    source.mkdir()
    image_specs = [
        ("images/task-1/alpha.png", (255, 0, 0), "scene-shared", "asset-a", "a", 0),
        ("images/task-1/beta.png", (0, 255, 0), "scene-shared", "asset-b", "b", 0),
        ("images/task-2/gamma.png", (0, 0, 255), "scene-other", "asset-a", "a", 1),
        ("images/task-2/delta.png", (255, 255, 0), "scene-third", "asset-c", "c", 0),
    ]
    images = [
        {
            "id": index,
            "file_name": file_name,
            "width": 20,
            "height": 10,
            "sha256": _write_image(source / file_name, color),
            "scene_id": scene_id,
            "source_asset_id": asset_id,
            "source_leakage_group_id": f"sha256:{group_character * 64}",
            "source_normalized_asset_frame": normalized_asset_frame,
        }
        for index, (
            file_name,
            color,
            scene_id,
            asset_id,
            group_character,
            normalized_asset_frame,
        ) in enumerate(image_specs, start=1)
    ]
    categories = [
        {"id": index, "name": name, "supercategory": "road_object"}
        for index, name in enumerate(prepare.REQUIRED_LABELS, start=1)
    ]
    annotations = [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [1, 1, 8, 4],
            "area": 32,
            "iscrowd": 0,
        },
        {
            "id": 2,
            "image_id": 1,
            "category_id": 5,
            "bbox": [10, 2, 5, 6],
            "area": 30,
            "iscrowd": 0,
        },
        {
            "id": 3,
            "image_id": 2,
            "category_id": 8,
            "bbox": [2, 1, 4, 4],
            "area": 16,
            "iscrowd": 0,
        },
        {
            "id": 4,
            "image_id": 4,
            "category_id": 6,
            "bbox": [0, 0, 3, 5],
            "area": 15,
            "iscrowd": 0,
        },
    ]
    coco_payload = {
        "info": {
            "description": "fixture",
            "schema": prepare.REFERENCE_SCHEMA,
            "labels_sha256": "d" * 64,
            "source_map_sha256": "e" * 64,
        },
        "categories": categories,
        "images": images,
        "annotations": annotations,
    }
    split_payload = {
        "schema": prepare.SPLIT_PLAN_SCHEMA,
        "train_asset_ids": ["asset-c", "asset-a"],
        "val_asset_ids": ["asset-b"],
    }
    result = Fixture(
        root=source,
        coco=source / "annotations.coco.json",
        reference_manifest=source / "manifest.json",
        split_plan=source / "asset-split.json",
        coco_payload=coco_payload,
        split_payload=split_payload,
    )
    result.write_coco()
    result.write_split()
    return result


def _read_manifest(output: Path) -> dict[str, Any]:
    return json.loads((output / "manifest.json").read_text(encoding="utf-8"))


def _prepare(
    fixture: Fixture,
    output: Path,
    *,
    coco: Path | None = None,
    reference_manifest: Path | None = None,
    split_plan: Path | None = None,
) -> dict[str, Any]:
    return prepare_yolo_dataset(
        coco or fixture.coco,
        split_plan or fixture.split_plan,
        output,
        reference_manifest_path=reference_manifest or fixture.reference_manifest,
    )


def test_asset_split_builds_portable_verified_dataset(tmp_path: Path, fixture: Fixture) -> None:
    output = tmp_path / "yolo"

    summary = _prepare(fixture, output)

    manifest_text = (output / "manifest.json").read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    dataset = yaml.safe_load((output / "dataset.yaml").read_text(encoding="utf-8"))
    assert summary["gate"]["passed"] is True
    assert str(tmp_path) not in manifest_text
    assert manifest["schema"] == prepare.OUTPUT_SCHEMA
    assert manifest["taxonomy"] == {
        "canonical_names": list(prepare.REQUIRED_LABELS),
        "model_names": list(prepare.MODEL_LABELS),
        "model_to_canonical": prepare.MODEL_TO_CANONICAL,
    }
    assert manifest["gate"]["checks"]["canonical_and_model_taxonomies_bound"] is True
    assert manifest["inputs"]["coco"]["sha256"] == sha256(fixture.coco)
    assert manifest["inputs"]["reference_manifest"] == {
        "file_name": "manifest.json",
        "sha256": sha256(fixture.reference_manifest),
        "schema": prepare.REFERENCE_SCHEMA,
    }
    assert manifest["inputs"]["split_plan"]["sha256"] == sha256(fixture.split_plan)
    assert manifest["split"] == {
        "method": "explicit typed source_asset_id plan constrained by source leakage group",
        "train_asset_ids": ["asset-a", "asset-c"],
        "val_asset_ids": ["asset-b"],
    }
    assert manifest["counts"]["images"] == {
        "total": 4,
        "train": 3,
        "val": 1,
        "zero_annotations": 1,
    }
    assert manifest["counts"]["annotations"]["by_category"]["bicycle"] == 1
    assert manifest["counts"]["annotations"]["by_category"]["bus"] == 0
    assert manifest["warnings"] == [
        "No annotations for class: bus",
        "No annotations for class: truck",
        "No annotations for class: motorcycle",
        "No annotations for class: traffic_light",
    ]
    assert dataset == {
        "train": "images/train",
        "val": "images/val",
        "names": {index: name for index, name in enumerate(prepare.MODEL_LABELS)},
    }
    assert sorted(path.name for path in (output / "images" / "train").iterdir()) == [
        "alpha.png",
        "delta.png",
        "gamma.png",
    ]
    assert [path.name for path in (output / "images" / "val").iterdir()] == ["beta.png"]
    assert (output / "labels" / "train" / "gamma.txt").read_bytes() == b""
    asset_a = next(
        record
        for record in manifest["source_statistics"]["assets"]
        if record["asset_id"] == "asset-a"
    )
    assert asset_a == {
        "asset_id": "asset-a",
        "leakage_group_id": f"sha256:{'a' * 64}",
        "split": "train",
        "image_count": 2,
        "annotation_count": 2,
        "scene_ids": ["scene-other", "scene-shared"],
    }
    # scene-shared crosses assets/splits, while asset-a crosses scenes but never splits.
    assert (output / "images" / "train" / "alpha.png").is_file()
    assert (output / "images" / "val" / "beta.png").is_file()
    assert len(manifest["files"]) == 9
    for record in manifest["files"]:
        path = output / record["path"]
        assert path.stat().st_size == record["size_bytes"]
        assert sha256(path) == record["sha256"]


def test_coco_and_plan_array_order_do_not_change_output_semantics(
    tmp_path: Path, fixture: Fixture
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _prepare(fixture, first)
    fixture.coco_payload["categories"].reverse()
    fixture.coco_payload["images"].reverse()
    fixture.coco_payload["annotations"].reverse()
    fixture.split_payload["train_asset_ids"].reverse()
    reordered_reference = tmp_path / "reordered-reference"
    shutil.copytree(fixture.root, reordered_reference)
    reordered_manifest = reordered_reference / "manifest.json"
    reordered_coco = fixture.write_coco(
        reordered_reference / "annotations.coco.json",
        reference_manifest_path=reordered_manifest,
    )
    reordered_plan = fixture.write_split(fixture.root / "reordered.split.json")

    _prepare(
        fixture,
        second,
        coco=reordered_coco,
        reference_manifest=reordered_manifest,
        split_plan=reordered_plan,
    )

    first_manifest = _read_manifest(first)
    second_manifest = _read_manifest(second)
    assert first_manifest["inputs"]["coco"]["sha256"] != second_manifest["inputs"]["coco"]["sha256"]
    assert (
        first_manifest["inputs"]["coco"]["semantic_sha256"]
        == second_manifest["inputs"]["coco"]["semantic_sha256"]
    )
    assert (
        first_manifest["inputs"]["split_plan"]["semantic_sha256"]
        == second_manifest["inputs"]["split_plan"]["semantic_sha256"]
    )
    for field in ("split", "counts", "source_statistics", "warnings", "files"):
        assert first_manifest[field] == second_manifest[field]
    for record in first_manifest["files"]:
        assert (first / record["path"]).read_bytes() == (second / record["path"]).read_bytes()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda manifest: manifest.update(schema={"name": "wrong", "version": 2}), "schema"),
        (lambda manifest: manifest["gate"].update(passed=False), "gate has not passed"),
        (
            lambda manifest: manifest["gate"]["blocking_reasons"].append("blocked"),
            "gate has not passed",
        ),
        (
            lambda manifest: manifest["gate"]["checks"].update(
                completion_receipts_exactly_bound=False
            ),
            "checks are incomplete or false",
        ),
        (
            lambda manifest: manifest["evidence"]["labels"].update(sha256="0" * 64),
            "labels hash differs",
        ),
        (
            lambda manifest: manifest["files"][0].update(sha256="0" * 64),
            "differs from COCO input",
        ),
        (
            lambda manifest: manifest["files"].append(copy.deepcopy(manifest["files"][0])),
            "duplicate path",
        ),
        (lambda manifest: manifest["files"].pop(), "must exactly cover"),
        (
            lambda manifest: manifest["evidence"]["source_map"]["assets"].pop(0),
            "absent from the reference manifest source map",
        ),
        (
            lambda manifest: manifest["evidence"]["source_map"]["assets"].append(
                copy.deepcopy(manifest["evidence"]["source_map"]["assets"][0])
            ),
            "duplicate asset_id",
        ),
        (
            lambda manifest: manifest["evidence"]["source_map"]["assets"][0].update(
                leakage_group_id=f"sha256:{'f' * 64}"
            ),
            "must equal the source SHA-256 identity",
        ),
        (
            lambda manifest: manifest["evidence"]["source_map"]["assets"][1].update(
                sha256="a" * 64,
                leakage_group_id=f"sha256:{'a' * 64}",
            ),
            "aliases one source SHA-256",
        ),
    ],
)
def test_rejects_unbound_or_failed_reference_manifest(
    tmp_path: Path, fixture: Fixture, mutation, message: str
) -> None:
    manifest = json.loads(fixture.reference_manifest.read_text(encoding="utf-8"))
    mutation(manifest)
    fixture.reference_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with pytest.raises(DatasetPreparationError, match=message):
        _prepare(fixture, tmp_path / "rejected")


def test_rejects_non_reference_coco_schema(tmp_path: Path, fixture: Fixture) -> None:
    fixture.coco_payload["info"]["schema"] = {"name": "wrong", "version": 2}
    fixture.write_coco()

    with pytest.raises(DatasetPreparationError, match="not a supported training reference"):
        _prepare(fixture, tmp_path / "rejected")


def test_rejects_coco_outside_reference_manifest_managed_path(
    tmp_path: Path, fixture: Fixture
) -> None:
    copied_coco = fixture.root / "copied.coco.json"
    shutil.copyfile(fixture.coco, copied_coco)

    with pytest.raises(DatasetPreparationError, match="managed annotations.coco.json"):
        _prepare(fixture, tmp_path / "rejected", coco=copied_coco)


def test_rejects_forged_coco_asset_absent_from_reference_source_map(
    tmp_path: Path, fixture: Fixture
) -> None:
    image = fixture.coco_payload["images"][0]
    image["source_asset_id"] = "forged-asset-a"
    image["source_leakage_group_id"] = f"sha256:{'f' * 64}"
    fixture.coco.write_text(
        json.dumps(fixture.coco_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = json.loads(fixture.reference_manifest.read_text(encoding="utf-8"))
    coco_record = next(
        record for record in manifest["files"] if record["path"] == "annotations.coco.json"
    )
    coco_record.update(sha256=sha256(fixture.coco), size_bytes=fixture.coco.stat().st_size)
    fixture.reference_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fixture.split_payload["val_asset_ids"].append("forged-asset-a")
    fixture.write_split()

    with pytest.raises(DatasetPreparationError, match="absent from the reference manifest"):
        _prepare(fixture, tmp_path / "rejected")


def test_rejects_source_content_alias_across_asset_ids(tmp_path: Path, fixture: Fixture) -> None:
    fixture.coco_payload["images"][1]["source_leakage_group_id"] = f"sha256:{'a' * 64}"
    fixture.write_coco()

    with pytest.raises(DatasetPreparationError, match="alias leakage group"):
        _prepare(fixture, tmp_path / "rejected")


def test_rejects_one_asset_with_multiple_leakage_groups(tmp_path: Path, fixture: Fixture) -> None:
    fixture.coco_payload["images"][2]["source_leakage_group_id"] = f"sha256:{'f' * 64}"
    fixture.write_coco()

    with pytest.raises(DatasetPreparationError, match="more than one leakage group"):
        _prepare(fixture, tmp_path / "rejected")


def test_rejects_duplicate_normalized_source_frame(tmp_path: Path, fixture: Fixture) -> None:
    fixture.coco_payload["images"][2]["source_normalized_asset_frame"] = 0
    fixture.write_coco()

    with pytest.raises(DatasetPreparationError, match="duplicates source normalized asset frame"):
        _prepare(fixture, tmp_path / "rejected")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda plan: plan.update(schema={"name": "wrong", "version": 1}), "schema"),
        (lambda plan: plan.update(train_asset_ids=[]), "must not be empty"),
        (
            lambda plan: plan.update(train_asset_ids=["asset-a", "asset-a"]),
            "duplicate typed asset identity",
        ),
        (
            lambda plan: plan.update(val_asset_ids=["asset-a", "asset-b"]),
            "both train_asset_ids and val_asset_ids",
        ),
        (lambda plan: plan.update(train_asset_ids=["asset-a"]), "exactly cover"),
        (
            lambda plan: plan.update(train_asset_ids=["asset-a", "asset-c", "unknown"]),
            "exactly cover",
        ),
    ],
)
def test_rejects_invalid_or_inexact_split_plan(
    tmp_path: Path, fixture: Fixture, mutate, message: str
) -> None:
    mutate(fixture.split_payload)
    fixture.write_split()

    with pytest.raises(DatasetPreparationError, match=message):
        _prepare(fixture, tmp_path / "rejected")


def test_typed_asset_identities_keep_integer_and_string_distinct(
    tmp_path: Path, fixture: Fixture
) -> None:
    for image in fixture.coco_payload["images"]:
        if image["source_asset_id"] == "asset-a":
            image["source_asset_id"] = 1
        elif image["source_asset_id"] == "asset-b":
            image["source_asset_id"] = "1"
    fixture.split_payload["train_asset_ids"] = [1, "asset-c"]
    fixture.split_payload["val_asset_ids"] = ["1"]
    fixture.write_coco()
    fixture.write_split()
    output = tmp_path / "typed"

    _prepare(fixture, output)

    manifest = _read_manifest(output)
    assert manifest["split"]["train_asset_ids"] == [1, "asset-c"]
    assert manifest["split"]["val_asset_ids"] == ["1"]
    assert (output / "images" / "train" / "alpha.png").is_file()
    assert (output / "images" / "val" / "beta.png").is_file()


@pytest.mark.parametrize(
    "file_name",
    [
        "/absolute.png",
        "../escape.png",
        "C:\\escape.png",
        "C:escape.png",
        "images//alpha.png",
        "bad.txt",
    ],
)
def test_rejects_unsafe_or_unsupported_image_paths(
    tmp_path: Path, fixture: Fixture, file_name: str
) -> None:
    fixture.coco_payload["images"][0]["file_name"] = file_name
    fixture.write_coco()

    with pytest.raises(DatasetPreparationError, match="unsafe|unsupported suffix"):
        _prepare(fixture, tmp_path / "rejected")


def test_rejects_source_symlink_that_escapes_coco_root(tmp_path: Path, fixture: Fixture) -> None:
    outside = tmp_path / "outside.png"
    digest = _write_image(outside, (2, 3, 4))
    link = fixture.root / "images" / "escape.png"
    link.symlink_to(outside)
    fixture.coco_payload["images"][0].update(file_name="images/escape.png", sha256=digest)
    fixture.write_coco()

    with pytest.raises(DatasetPreparationError, match="escapes its source directory"):
        _prepare(fixture, tmp_path / "rejected")


def test_rejects_output_image_basename_collision(tmp_path: Path, fixture: Fixture) -> None:
    replacement = fixture.root / "images" / "other" / "alpha.png"
    digest = _write_image(replacement, (11, 22, 33))
    fixture.coco_payload["images"][1].update(file_name="images/other/alpha.png", sha256=digest)
    fixture.write_coco()

    with pytest.raises(DatasetPreparationError, match="image basename collision"):
        _prepare(fixture, tmp_path / "rejected")


def test_rejects_output_label_basename_collision(tmp_path: Path, fixture: Fixture) -> None:
    replacement = fixture.root / "images" / "other" / "alpha.jpg"
    digest = _write_image(replacement, (11, 22, 33))
    fixture.coco_payload["images"][1].update(file_name="images/other/alpha.jpg", sha256=digest)
    fixture.write_coco()

    with pytest.raises(DatasetPreparationError, match="label basename collision"):
        _prepare(fixture, tmp_path / "rejected")


def test_rejects_duplicate_image_content_hash(tmp_path: Path, fixture: Fixture) -> None:
    first = fixture.root / fixture.coco_payload["images"][0]["file_name"]
    duplicate = fixture.root / "images" / "duplicate.png"
    shutil.copyfile(first, duplicate)
    fixture.coco_payload["images"][1].update(
        file_name="images/duplicate.png", sha256=sha256(duplicate)
    )
    fixture.write_coco()

    with pytest.raises(DatasetPreparationError, match="duplicate image SHA-256"):
        _prepare(fixture, tmp_path / "rejected")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda image: image.pop("sha256"), "lowercase SHA-256"),
        (lambda image: image.update(sha256="0" * 64), "differs from source image"),
        (lambda image: image.update(width=21), "dimensions differ"),
        (lambda image: image.update(source_asset_id=True), "integer or string"),
        (lambda image: image.update(source_leakage_group_id="not-a-hash"), "sha256:"),
        (lambda image: image.update(source_normalized_asset_frame=-1), "at least 0"),
        (lambda image: image.update(scene_id=""), "non-empty string"),
    ],
)
def test_rejects_invalid_or_drifted_image_metadata(
    tmp_path: Path, fixture: Fixture, mutation, message: str
) -> None:
    mutation(fixture.coco_payload["images"][0])
    fixture.write_coco()

    with pytest.raises(DatasetPreparationError, match=message):
        _prepare(fixture, tmp_path / "rejected")


def test_rejects_noncanonical_taxonomy(tmp_path: Path, fixture: Fixture) -> None:
    fixture.coco_payload["categories"][0]["name"] = "automobile"
    fixture.write_coco()

    with pytest.raises(DatasetPreparationError, match="fixed eight-class taxonomy"):
        _prepare(fixture, tmp_path / "rejected")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda annotation: annotation.update(image_id=999), "unknown image id"),
        (lambda annotation: annotation.update(category_id=999), "unknown category id"),
        (lambda annotation: annotation.update(bbox=[1, 2, 3]), "exactly four"),
        (lambda annotation: annotation.update(bbox=[1, 2, float("nan"), 3]), "finite"),
        (lambda annotation: annotation.update(bbox=[19, 1, 2, 2]), "escapes image bounds"),
        (lambda annotation: annotation.update(id=True), "must be an integer"),
        (lambda annotation: annotation.pop("area"), "area must be a number"),
        (lambda annotation: annotation.update(area=31), "area must equal"),
        (lambda annotation: annotation.update(area=32.0000000001), "area must equal"),
        (lambda annotation: annotation.pop("iscrowd"), "iscrowd must be an integer"),
        (lambda annotation: annotation.update(iscrowd=1), "iscrowd must be 0"),
        (lambda annotation: annotation.update(iscrowd=2), "iscrowd must be 0"),
    ],
)
def test_rejects_invalid_annotations(
    tmp_path: Path, fixture: Fixture, mutation, message: str
) -> None:
    mutation(fixture.coco_payload["annotations"][0])
    fixture.write_coco()

    with pytest.raises(DatasetPreparationError, match=message):
        _prepare(fixture, tmp_path / "rejected")


def test_existing_output_is_never_overwritten(tmp_path: Path, fixture: Fixture) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "owned-by-user.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(DatasetPreparationError, match="already exists"):
        _prepare(fixture, output)

    assert marker.read_text(encoding="utf-8") == "keep"


def test_broken_output_symlink_is_never_followed(tmp_path: Path, fixture: Fixture) -> None:
    redirected = tmp_path / "redirected"
    output = tmp_path / "broken-link"
    output.symlink_to(redirected)

    with pytest.raises(DatasetPreparationError, match="already exists"):
        _prepare(fixture, output)

    assert output.is_symlink()
    assert not redirected.exists()


def test_atomic_publish_refuses_an_empty_racing_target(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    output = tmp_path / "output"
    staging.mkdir()
    (staging / "payload.txt").write_text("new", encoding="utf-8")
    output.mkdir()

    with pytest.raises(DatasetPreparationError, match="already exists"):
        prepare._atomic_publish_directory_no_replace(staging, output)

    assert staging.is_dir()
    assert output.is_dir()
    assert not (output / "payload.txt").exists()


def test_failed_staged_copy_is_cleaned(
    tmp_path: Path, fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "copy-failure"

    def fail_copy(*_args: object, **_kwargs: object) -> None:
        raise OSError("injected copy failure")

    monkeypatch.setattr(prepare.shutil, "copyfileobj", fail_copy)

    with pytest.raises(OSError, match="injected copy failure"):
        _prepare(fixture, output)

    assert not output.exists()
    assert not list(tmp_path.glob(".copy-failure.staging-*"))


def test_hash_mismatch_in_staging_is_rejected_and_cleaned(
    tmp_path: Path, fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "hash-failure"
    real_copy = prepare.shutil.copyfileobj

    def corrupt_copy(source, destination, *, length: int) -> None:
        real_copy(source, destination, length=length)
        destination.write(b"corruption")

    monkeypatch.setattr(prepare.shutil, "copyfileobj", corrupt_copy)

    with pytest.raises(DatasetPreparationError, match="copied image hash mismatch"):
        _prepare(fixture, output)

    assert not output.exists()
    assert not list(tmp_path.glob(".hash-failure.staging-*"))


def test_racing_output_is_preserved_and_staging_is_cleaned(
    tmp_path: Path, fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "raced"
    real_publish = prepare._atomic_publish_directory_no_replace

    def race_then_publish(staging: Path, target: Path) -> None:
        target.mkdir()
        (target / "winner.txt").write_text("keep", encoding="utf-8")
        real_publish(staging, target)

    monkeypatch.setattr(prepare, "_atomic_publish_directory_no_replace", race_then_publish)

    with pytest.raises(DatasetPreparationError, match="already exists"):
        _prepare(fixture, output)

    assert (output / "winner.txt").read_text(encoding="utf-8") == "keep"
    assert not list(tmp_path.glob(".raced.staging-*"))


def test_cli_requires_reference_manifest() -> None:
    with pytest.raises(SystemExit):
        prepare.build_parser().parse_args(
            [
                "--coco",
                "input.json",
                "--split-plan",
                "split.json",
                "--output",
                "out",
            ]
        )


def test_cli_builds_with_reference_manifest(
    tmp_path: Path, fixture: Fixture, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "cli-yolo"

    prepare.main(
        [
            "--coco",
            str(fixture.coco),
            "--reference-manifest",
            str(fixture.reference_manifest),
            "--split-plan",
            str(fixture.split_plan),
            "--output",
            str(output),
        ]
    )

    assert json.loads(capsys.readouterr().out)["gate"]["passed"] is True
    assert (output / "manifest.json").is_file()


def test_yolo_line_normalizes_coco_xywh() -> None:
    annotation = {"id": 1, "bbox": [10, 20, 40, 20]}
    image = {"width": 100, "height": 100}

    tokens = yolo_line(annotation, image, 3).split()
    assert tokens[0] == "3"
    assert [float(value) for value in tokens[1:]] == pytest.approx([0.3, 0.3, 0.4, 0.2])


def test_yolo_line_preserves_tiny_positive_bbox() -> None:
    tokens = yolo_line(
        {"id": 1, "bbox": [0.0, 0.0, 1e-8, 1.0]},
        {"width": 20, "height": 10},
        0,
    ).split()

    assert float(tokens[3]) > 0
    assert float(tokens[4]) > 0


@pytest.mark.parametrize(
    ("annotation_id", "image_id", "bbox"),
    [
        (1481, 119, [672.03, 647.11, 207.12, 72.88999999999999]),
        (1567, 122, [345.67, 631.84, 249.2, 88.15999999999997]),
        (1627, 124, [345.13, 631.84, 250.89, 88.15999999999997]),
    ],
)
def test_yolo_line_stabilizes_boundary_aligned_bbox_without_clipping(
    annotation_id: int, image_id: int, bbox: list[float]
) -> None:
    annotation = {"id": annotation_id, "image_id": image_id, "bbox": bbox}
    image = {"id": image_id, "width": 1280, "height": 720}

    line = yolo_line(annotation, image, 0)
    assert yolo_line(annotation, image, 0) == line
    center_x, center_y, normalized_width, normalized_height = map(float, line.split()[1:])

    assert center_x - normalized_width / 2 >= 0
    assert center_x + normalized_width / 2 <= 1
    assert center_y - normalized_height / 2 >= 0
    assert center_y + normalized_height / 2 <= 1
    assert normalized_width == bbox[2] / image["width"]
    assert normalized_height == bbox[3] / image["height"]
    original_center_y = (bbox[1] + bbox[3] / 2) / image["height"]
    assert abs(center_y - original_center_y) <= math.ulp(original_center_y)


def test_yolo_line_boundary_stabilization_does_not_accept_real_overflow() -> None:
    with pytest.raises(
        DatasetPreparationError,
        match="Annotation 91 for image 17 escapes image bounds",
    ):
        yolo_line(
            {"id": 91, "image_id": 17, "bbox": [0.0, 0.0, 20.00000001, 10.0]},
            {"id": 17, "width": 20, "height": 10},
            0,
        )


def test_yolo_line_rejects_normalization_underflow() -> None:
    with pytest.raises(
        DatasetPreparationError,
        match="Annotation 1 for image 7 loses positive size",
    ):
        yolo_line(
            {"id": 1, "image_id": 7, "bbox": [0.0, 0.0, 5e-324, 1.0]},
            {"id": 7, "width": 20, "height": 10},
            0,
        )


def test_semantic_fixture_copy_is_independent(fixture: Fixture) -> None:
    # Guard against accidental shared-object mutations between test cases.
    cloned = copy.deepcopy(fixture.coco_payload)
    cloned["images"][0]["scene_id"] = "changed"
    assert fixture.coco_payload["images"][0]["scene_id"] == "scene-shared"
