from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import scripts.build_open_images_acquisition_plan as acquisition
from scripts.build_open_images_acquisition_plan import (
    OpenImagesAcquisitionPlanError,
    build_open_images_acquisition_plan,
)

LABELS = {
    "Person": "/m/01g317",
    "Bicycle": "/m/0199g",
    "Bus": "/m/01bjv",
    "Car": "/m/0k4j",
    "Motorcycle": "/m/04_sv",
    "Traffic light": "/m/015qff",
    "Traffic sign": "/m/01mqdt",
    "Truck": "/m/07r04",
}
BBOX_FIELDS = [
    "ImageID",
    "Source",
    "LabelName",
    "Confidence",
    "XMin",
    "XMax",
    "YMin",
    "YMax",
    "IsOccluded",
    "IsTruncated",
    "IsGroupOf",
    "IsDepiction",
    "IsInside",
]
METADATA_FIELDS = [
    "ImageID",
    "Subset",
    "OriginalURL",
    "OriginalLandingURL",
    "License",
    "AuthorProfileURL",
    "Author",
    "Title",
    "OriginalSize",
    "OriginalMD5",
    "Thumbnail300KURL",
    "Rotation",
]
CC_BY_2_0 = "https://creativecommons.org/licenses/by/2.0/"


@dataclass
class OpenImagesFixture:
    workspace: Path
    class_descriptions: Path
    bboxes: Path
    metadata: Path
    bbox_rows: list[dict[str, str]]
    metadata_rows: list[dict[str, str]]

    def write_bboxes(self) -> None:
        _write_dict_csv(self.bboxes, BBOX_FIELDS, self.bbox_rows)

    def write_metadata(self) -> None:
        _write_dict_csv(self.metadata, METADATA_FIELDS, self.metadata_rows)

    def build(self, output_name: str, **overrides: Any) -> dict[str, Any]:
        parameters = {
            "max_images": 3,
            "min_images_per_class": 2,
            "min_boxes_per_class": 2,
            "max_images_per_source_group": 1,
            "min_source_groups_per_class": 2,
        }
        parameters.update(overrides)
        return build_open_images_acquisition_plan(
            self.class_descriptions,
            self.bboxes,
            self.metadata,
            self.workspace / "docs" / "evidence" / output_name,
            workspace_root=self.workspace,
            **parameters,
        )


def _write_dict_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _metadata_row(image_id: str, author_number: int) -> dict[str, str]:
    return {
        "ImageID": image_id,
        "Subset": "validation",
        "OriginalURL": f"https://images.example/{image_id}.jpg",
        "OriginalLandingURL": f"https://photos.example/image/{image_id}",
        "License": CC_BY_2_0,
        "AuthorProfileURL": f"https://photos.example/authors/{author_number}",
        "Author": f"Author {author_number}",
        "Title": f"Road scene {image_id}",
        "OriginalSize": "12345",
        "OriginalMD5": "a" * 32,
        "Thumbnail300KURL": f"https://thumbs.example/{image_id}.jpg",
        "Rotation": "0",
    }


def _bbox_row(image_id: str, label_id: str, index: int) -> dict[str, str]:
    offset = index / 100
    return {
        "ImageID": image_id,
        "Source": "xclick",
        "LabelName": label_id,
        "Confidence": "1",
        "XMin": f"{0.01 + offset:.2f}",
        "XMax": f"{0.31 + offset:.2f}",
        "YMin": "0.10",
        "YMax": "0.60",
        "IsOccluded": "0",
        "IsTruncated": "0",
        "IsGroupOf": "0",
        "IsDepiction": "0",
        "IsInside": "0",
    }


@pytest.fixture
def open_images(tmp_path: Path) -> OpenImagesFixture:
    workspace = tmp_path / "workspace"
    inputs = workspace / "data" / "training" / "open-images-v7-inputs"
    inputs.mkdir(parents=True)
    class_descriptions = inputs / "oidv7-class-descriptions-boxable.csv"
    with class_descriptions.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        for display_name, label_id in LABELS.items():
            writer.writerow([label_id, display_name])
        writer.writerow(["/m/not_target", "Aircraft"])

    image_ids = [f"{number:016x}" for number in range(1, 5)]
    # Images 1 and 2 deliberately share a profile/source group.
    metadata_rows = [
        _metadata_row(image_ids[0], 1),
        _metadata_row(image_ids[1], 1),
        _metadata_row(image_ids[2], 2),
        _metadata_row(image_ids[3], 3),
    ]
    metadata_rows[1]["Author"] = "Alias for Author 1"
    bbox_rows: list[dict[str, str]] = []
    for image_id in image_ids:
        for index, label_id in enumerate(LABELS.values()):
            bbox_rows.append(_bbox_row(image_id, label_id, index))
        bbox_rows.append(_bbox_row(image_id, "/m/not_target", 8))
    bboxes = inputs / "oidv7-validation-annotations-bbox.csv"
    metadata = inputs / "validation-images-with-rotation.csv"
    fixture = OpenImagesFixture(
        workspace=workspace,
        class_descriptions=class_descriptions,
        bboxes=bboxes,
        metadata=metadata,
        bbox_rows=bbox_rows,
        metadata_rows=metadata_rows,
    )
    fixture.write_bboxes()
    fixture.write_metadata()
    return fixture


def test_builds_stable_local_only_plan_with_source_group_manifests(
    open_images: OpenImagesFixture,
) -> None:
    first = open_images.build("plan-a.json")
    second = open_images.build("plan-b.json")
    first_path = Path(first["output"])
    second_path = Path(second["output"])

    assert first_path.read_bytes() == second_path.read_bytes()
    plan = json.loads(first_path.read_text(encoding="utf-8"))
    assert plan["schema"] == acquisition.OUTPUT_SCHEMA
    assert plan["dataset"] == {
        "name": "Open Images",
        "project_usage": "training_only",
        "upstream_subset": "validation",
        "version": "V7",
    }
    assert plan["gate"]["passed"] is True
    assert plan["holdout_firewall"] == {
        "allowed_inputs": ["local_official_open_images_v7_csv"],
        "downloads_performed": False,
        "network_accessed": False,
        "rejected_scopes": ["data/holdout", "configured-final-holdout"],
    }
    assert plan["counts"] == {
        "eligible_images": 4,
        "selected_boxes": 24,
        "selected_images": 3,
        "selected_source_groups": 3,
        "special_target_flag_images_excluded": 0,
        "target_bbox_rows_read": 32,
    }
    assert [image["image_id"] for image in plan["images"]] == [
        "0000000000000001",
        "0000000000000003",
        "0000000000000004",
    ]
    for image in plan["images"]:
        assert set(image["box_counts"]) == set(acquisition.ROADLABELOPS_CLASSES)
        assert {box["class_name"] for box in image["boxes"]} == set(
            acquisition.ROADLABELOPS_CLASSES
        )
        assert image["license"]["name"] == "CC BY 2.0"
        assert image["author"] in image["attribution"]
        assert image["landing_url"] in image["attribution"]
        assert image["subset"] == "validation"
        assert image["cvdf_download"]["object_key"] == (f"validation/{image['image_id']}.jpg")
    for class_name, statistics in plan["selection"]["class_statistics"].items():
        assert class_name in acquisition.ROADLABELOPS_CLASSES
        assert statistics["selected_source_group_count"] == 3
        assert statistics["quota_met"] is True
    for group in plan["selection"]["source_groups"]:
        assert group["image_ids"] == sorted(group["image_ids"])
        unsigned = dict(group)
        expected_digest = unsigned.pop("manifest_semantic_sha256")
        assert acquisition._semantic_sha256(unsigned) == expected_digest
    for binding in plan["inputs"].values():
        path = open_images.workspace / binding["path"]
        assert binding["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert binding["size_bytes"] == path.stat().st_size


def test_rejects_non_cc_by_2_0_metadata(open_images: OpenImagesFixture) -> None:
    rejected_id = open_images.metadata_rows[0]["ImageID"]
    open_images.metadata_rows[0]["License"] = "https://creativecommons.org/licenses/by-sa/2.0/"
    open_images.write_metadata()

    result = open_images.build("license-filter.json")
    plan = json.loads(Path(result["output"]).read_text(encoding="utf-8"))

    assert rejected_id not in {image["image_id"] for image in plan["images"]}
    assert plan["metadata_exclusions"]["not_explicit_cc_by_2_0"] == 1
    assert all(image["license"]["url"] == CC_BY_2_0 for image in plan["images"])


@pytest.mark.parametrize("special_flag", ["IsGroupOf", "IsDepiction", "IsInside"])
def test_any_special_target_box_rejects_the_entire_image(
    open_images: OpenImagesFixture, special_flag: str
) -> None:
    rejected_id = open_images.bbox_rows[0]["ImageID"]
    first_target_row = next(
        row
        for row in open_images.bbox_rows
        if row["ImageID"] == rejected_id and row["LabelName"] in LABELS.values()
    )
    first_target_row[special_flag] = "1"
    open_images.write_bboxes()

    result = open_images.build("special-filter.json")
    plan = json.loads(Path(result["output"]).read_text(encoding="utf-8"))

    assert rejected_id not in {image["image_id"] for image in plan["images"]}
    assert plan["counts"]["special_target_flag_images_excluded"] == 1
    assert plan["counts"]["eligible_images"] == 3


@pytest.mark.parametrize("failure", ["box_quota", "source_group_quota"])
def test_fails_when_hard_quotas_cannot_be_met(open_images: OpenImagesFixture, failure: str) -> None:
    if failure == "box_quota":
        overrides = {"min_boxes_per_class": 5}
    else:
        for row in open_images.metadata_rows:
            row["AuthorProfileURL"] = "https://photos.example/authors/one"
            row["Author"] = "One Author"
        open_images.write_metadata()
        overrides = {}

    with pytest.raises(OpenImagesAcquisitionPlanError, match="cannot meet|cannot meet per-class"):
        open_images.build("quota-failure.json", **overrides)


def test_never_replaces_an_existing_plan(open_images: OpenImagesFixture) -> None:
    result = open_images.build("immutable.json")
    output = Path(result["output"])
    before = output.read_bytes()

    with pytest.raises(OpenImagesAcquisitionPlanError, match="already exists"):
        open_images.build("immutable.json")

    assert output.read_bytes() == before


def test_cli_defaults_to_two_source_groups_per_class() -> None:
    args = acquisition.build_parser().parse_args(
        [
            "--class-descriptions-csv",
            "classes.csv",
            "--bbox-csv",
            "boxes.csv",
            "--image-metadata-csv",
            "metadata.csv",
            "--output",
            "plan.json",
            "--max-images",
            "10",
            "--min-images-per-class",
            "2",
            "--min-boxes-per-class",
            "2",
            "--max-images-per-source-group",
            "2",
        ]
    )

    assert args.min_source_groups_per_class == 2
