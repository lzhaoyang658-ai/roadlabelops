from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest
from PIL import Image

import scripts.materialize_open_images_training_pack as materializer
from scripts.materialize_open_images_training_pack import (
    OpenImagesMaterializationError,
    materialize_open_images_training_pack,
)


def _semantic_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True).encode(
            "utf-8"
        )
        + b"\n"
    )


def _jpeg_bytes(*, width: int = 10, height: int = 8) -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (width, height), (32, 96, 160)).save(stream, format="JPEG")
    return stream.getvalue()


def _box(class_name: str, label_id: str) -> dict[str, Any]:
    return {
        "class_name": class_name,
        "confidence": 1.0,
        "is_depiction": 0,
        "is_group_of": 0,
        "is_inside": 0,
        "is_occluded": 0,
        "is_truncated": 0,
        "label_id": label_id,
        "source": "xclick",
        "xmax": 0.6,
        "xmin": 0.1,
        "ymax": 0.8,
        "ymin": 0.2,
    }


def _build_plan_payload() -> dict[str, Any]:
    image_id = "0000000000000001"
    group_id = "author-profile:https://photos.example/authors/one"
    license_url = "https://creativecommons.org/licenses/by/2.0/"
    landing_url = f"https://photos.example/images/{image_id}"
    taxonomy = [
        {
            "display_name": class_name.replace("_", " ").title(),
            "label_id": f"/m/test-{index}",
            "roadlabelops_class": class_name,
        }
        for index, class_name in enumerate(materializer.CANONICAL_CLASSES)
    ]
    boxes = [_box(record["roadlabelops_class"], record["label_id"]) for record in taxonomy]
    image = {
        "attribution": (
            f'"Road scene" by Example Author; source: {landing_url}; '
            f"licensed under CC BY 2.0 ({license_url})"
        ),
        "author": "Example Author",
        "author_profile_url": "https://photos.example/authors/one",
        "box_counts": {class_name: 1 for class_name in materializer.CANONICAL_CLASSES},
        "boxes": boxes,
        "cvdf_download": {
            "bucket": "open-images-dataset",
            "file_name": f"{image_id}.jpg",
            "object_key": f"validation/{image_id}.jpg",
            "s3_uri": f"s3://open-images-dataset/validation/{image_id}.jpg",
        },
        "image_id": image_id,
        "landing_url": landing_url,
        "license": {"name": "CC BY 2.0", "url": license_url},
        "selection_rank": 1,
        "source_group_basis": "AuthorProfileURL",
        "source_group_id": group_id,
        "subset": "validation",
        "title": "Road scene",
    }
    group = {
        "image_count": 1,
        "image_ids": [image_id],
        "per_class": {
            class_name: {"box_count": 1, "image_count": 1}
            for class_name in materializer.CANONICAL_CLASSES
        },
        "source_group_basis": "AuthorProfileURL",
        "source_group_id": group_id,
        "source_group_value": "https://photos.example/authors/one",
    }
    group["manifest_semantic_sha256"] = _semantic_sha256(group)
    binding = {"path": "data/external/input.csv", "sha256": "a" * 64, "size_bytes": 1}
    payload: dict[str, Any] = {
        "counts": {
            "eligible_images": 1,
            "selected_boxes": len(boxes),
            "selected_images": 1,
            "selected_source_groups": 1,
            "special_target_flag_images_excluded": 0,
            "target_bbox_rows_read": len(boxes),
        },
        "dataset": {
            "name": "Open Images",
            "project_usage": "training_only",
            "upstream_subset": "validation",
            "version": "V7",
        },
        "gate": {
            "blocking_reasons": [],
            "checks": {check: True for check in sorted(materializer.REQUIRED_PLAN_GATE_CHECKS)},
            "passed": True,
        },
        "holdout_firewall": {
            "allowed_inputs": ["local_official_open_images_v7_csv"],
            "downloads_performed": False,
            "network_accessed": False,
            "rejected_scopes": ["data/holdout", "configured-final-holdout"],
        },
        "images": [image],
        "inputs": {
            "boxable_class_descriptions": dict(binding),
            "bounding_boxes": dict(binding),
            "image_metadata": dict(binding),
        },
        "metadata_exclusions": {},
        "parameters": {
            "max_images": 1,
            "max_images_per_source_group": 1,
            "min_boxes_per_class": 1,
            "min_images_per_class": 1,
            "min_source_groups_per_class": 1,
        },
        "schema": materializer.PLAN_SCHEMA,
        "selection": {"class_statistics": {}, "policy": "fixture", "source_groups": [group]},
        "taxonomy": taxonomy,
    }
    payload["plan_semantic_sha256"] = _semantic_sha256(payload)
    return payload


@dataclass
class MaterializationFixture:
    workspace: Path
    plan: Path
    output: Path
    payload: dict[str, Any]

    def write_plan(self) -> None:
        unsigned = dict(self.payload)
        unsigned.pop("plan_semantic_sha256", None)
        self.payload["plan_semantic_sha256"] = _semantic_sha256(unsigned)
        self.plan.parent.mkdir(parents=True, exist_ok=True)
        self.plan.write_bytes(_json_bytes(self.payload))


@pytest.fixture
def materialization(tmp_path: Path) -> MaterializationFixture:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    fixture = MaterializationFixture(
        workspace=workspace,
        plan=workspace / "docs/evidence/open-images-plan.json",
        output=workspace / "data/training/open-images-pack",
        payload=_build_plan_payload(),
    )
    fixture.write_plan()
    return fixture


def test_default_dry_run_performs_zero_network_and_writes_nothing(
    materialization: MaterializationFixture,
) -> None:
    def fail_if_called(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("dry-run attempted network access")

    result = materialize_open_images_training_pack(
        materialization.plan,
        materialization.output,
        workspace_root=materialization.workspace,
        transport=httpx.MockTransport(fail_if_called),
    )

    assert result["mode"] == "dry-run"
    assert result["downloads_performed"] is False
    assert result["would_download_images"] == 1
    assert not materialization.output.exists()


def test_apply_uses_only_official_cvdf_validation_url_and_materializes_bboxes(
    materialization: MaterializationFixture,
) -> None:
    jpeg = _jpeg_bytes()
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        assert request.extensions["timeout"] == {
            "connect": 1.25,
            "read": 2.5,
            "write": 2.5,
            "pool": 2.5,
        }
        return httpx.Response(
            200,
            content=jpeg,
            headers={"Content-Type": "image/jpeg", "Content-Length": str(len(jpeg))},
        )

    result = materialize_open_images_training_pack(
        materialization.plan,
        materialization.output,
        workspace_root=materialization.workspace,
        apply=True,
        timeout_seconds=2.5,
        connect_timeout_seconds=1.25,
        transport=httpx.MockTransport(handler),
    )

    assert requested == [
        "https://open-images-dataset.s3.amazonaws.com/validation/0000000000000001.jpg"
    ]
    assert result["published_images"] == 1
    manifest = json.loads((materialization.output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset"]["upstream_subset"] == "validation"
    assert manifest["dataset"]["project_usage"] == "training_only"
    assert manifest["images"][0]["official_object_key"] == ("validation/0000000000000001.jpg")
    draft = json.loads(
        (materialization.output / "draft-sample-manifest.json").read_text(encoding="utf-8")
    )
    first_box = draft["samples"][0]["annotations"][0]
    assert first_box["bbox"] == pytest.approx([1.0, 1.6, 5.0, 4.8])
    assert first_box["normalized_bbox_xyxy"] == [0.1, 0.2, 0.6, 0.8]


def test_apply_retries_retryable_status_and_transport_timeout(
    materialization: MaterializationFixture,
) -> None:
    jpeg = _jpeg_bytes()
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(503, request=request)
        if attempts == 2:
            raise httpx.ReadTimeout("fixture timeout", request=request)
        return httpx.Response(200, content=jpeg, request=request)

    materialize_open_images_training_pack(
        materialization.plan,
        materialization.output,
        workspace_root=materialization.workspace,
        apply=True,
        retries=2,
        retry_backoff_seconds=0,
        transport=httpx.MockTransport(handler),
    )

    assert attempts == 3
    manifest = json.loads((materialization.output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["images"][0]["download_attempts"] == 3


def test_bad_jpeg_aborts_without_publishing(materialization: MaterializationFixture) -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, content=b"not a jpeg", request=request)
    )

    with pytest.raises(OpenImagesMaterializationError, match="valid decodable JPEG|not a JPEG"):
        materialize_open_images_training_pack(
            materialization.plan,
            materialization.output,
            workspace_root=materialization.workspace,
            apply=True,
            transport=transport,
        )

    assert not materialization.output.exists()
    assert not list(materialization.output.parent.glob(".open-images-pack.staging-*"))


def test_invalid_normalized_bbox_is_rejected_before_network(
    materialization: MaterializationFixture,
) -> None:
    box = materialization.payload["images"][0]["boxes"][0]
    box["xmax"] = box["xmin"]
    materialization.write_plan()

    def fail_if_called(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("invalid plan attempted network access")

    with pytest.raises(OpenImagesMaterializationError, match="invalid normalized boundary"):
        materialize_open_images_training_pack(
            materialization.plan,
            materialization.output,
            workspace_root=materialization.workspace,
            apply=True,
            transport=httpx.MockTransport(fail_if_called),
        )

    assert not materialization.output.exists()


def test_existing_output_is_never_replaced_and_performs_zero_network(
    materialization: MaterializationFixture,
) -> None:
    materialization.output.mkdir(parents=True)
    sentinel = materialization.output / "sentinel.txt"
    sentinel.write_text("keep\n", encoding="utf-8")

    def fail_if_called(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("existing output attempted network access")

    with pytest.raises(OpenImagesMaterializationError, match="already exists"):
        materialize_open_images_training_pack(
            materialization.plan,
            materialization.output,
            workspace_root=materialization.workspace,
            apply=True,
            transport=httpx.MockTransport(fail_if_called),
        )

    assert sentinel.read_text(encoding="utf-8") == "keep\n"


def test_plan_drift_aborts_atomic_publication(materialization: MaterializationFixture) -> None:
    jpeg = _jpeg_bytes()

    def handler(request: httpx.Request) -> httpx.Response:
        materialization.plan.write_bytes(materialization.plan.read_bytes() + b" ")
        return httpx.Response(200, content=jpeg, request=request)

    with pytest.raises(OpenImagesMaterializationError, match="changed during materialization"):
        materialize_open_images_training_pack(
            materialization.plan,
            materialization.output,
            workspace_root=materialization.workspace,
            apply=True,
            transport=httpx.MockTransport(handler),
        )

    assert not materialization.output.exists()
    assert not list(materialization.output.parent.glob(".open-images-pack.staging-*"))
