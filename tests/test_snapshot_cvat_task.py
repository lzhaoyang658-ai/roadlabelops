from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.snapshot_cvat_task import (
    REQUIRED_LABELS,
    SNAPSHOT_SCHEMA,
    SnapshotValidationError,
    atomic_write_json_new,
    build_image_inventory,
    build_parser,
    build_snapshot_payload,
    canonical_sha256,
    canonicalize_annotations,
    validate_frame_mapping,
    validate_jobs,
    validate_labels,
    validate_manifest,
)

SYNTHETIC_TASK_ID = 41001
OTHER_SYNTHETIC_TASK_ID = 41002
SYNTHETIC_JOB_ID = 51001
OTHER_SYNTHETIC_JOB_ID = 51002


def labels() -> list[dict]:
    return [
        {"id": index, "name": name, "color": f"#{index:06x}", "attributes": []}
        for index, name in enumerate(REQUIRED_LABELS, start=10)
    ]


def manifest(sample_count: int = 2, task_id: int = SYNTHETIC_TASK_ID) -> dict:
    return {
        "session_id": "session_test",
        "purpose": "training",
        "sampling_revision": "test-v1",
        "sample_size": sample_count,
        "cvat": {"task_id": task_id, "project_id": 9, "job_ids": [SYNTHETIC_JOB_ID]},
        "samples": [
            {
                "sample_index": index + 1,
                "scene_id": "scene_001",
                "source_frame": index * 5,
                "file_name": f"frame_{index:03d}.png",
                "annotations": [],
            }
            for index in range(sample_count)
        ],
    }


def frame_metadata(sample_count: int = 2) -> list[dict]:
    return [
        {"name": f"frame_{index:03d}.png", "width": 32, "height": 24}
        for index in range(sample_count)
    ]


def annotations(*, tracks: list[dict] | None = None) -> dict:
    return {
        "version": 7,
        "tags": [
            {
                "id": 1,
                "frame": 0,
                "label_id": 10,
                "group": 0,
                "source": "manual",
                "attributes": [],
            }
        ],
        "shapes": [
            {
                "id": 2,
                "frame": 1,
                "label_id": 11,
                "type": "rectangle",
                "points": [1.0, 2.0, 10.0, 20.0],
                "occluded": False,
                "outside": False,
                "z_order": 0,
                "rotation": 0.0,
                "group": 0,
                "source": "manual",
                "attributes": [{"spec_id": 90, "value": "yes"}],
                "score": None,
                "elements": [],
            }
        ],
        "tracks": tracks or [],
        "intervals": [],
    }


def write_fixture_images(manifest_path: Path, count: int = 2) -> None:
    image_dir = manifest_path.parent / "images"
    image_dir.mkdir(parents=True)
    for index in range(count):
        Image.new("RGB", (32, 24), (index * 20, 40, 60)).save(image_dir / f"frame_{index:03d}.png")


def write_manifest(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def test_cli_requires_named_snapshot_arguments() -> None:
    args = build_parser().parse_args(
        [
            "--task-id",
            str(SYNTHETIC_TASK_ID),
            "--image-manifest",
            "manifest.json",
            "--output",
            "snap.json",
        ]
    )

    assert args.task_id == SYNTHETIC_TASK_ID
    assert args.image_manifest == Path("manifest.json")
    assert args.output == Path("snap.json")


def test_canonical_annotations_ignore_mapping_and_collection_order() -> None:
    first = annotations()
    second = {
        "tracks": [],
        "shapes": [dict(reversed(list(first["shapes"][0].items())))],
        "tags": list(reversed(first["tags"])),
        "version": 7,
        "intervals": [],
    }

    assert canonicalize_annotations(first) == canonicalize_annotations(second)
    assert canonical_sha256(canonicalize_annotations(first)) == canonical_sha256(
        canonicalize_annotations(second)
    )

    changed = annotations()
    changed["shapes"][0]["points"] = list(reversed(changed["shapes"][0]["points"]))
    assert canonical_sha256(canonicalize_annotations(first)) != canonical_sha256(
        canonicalize_annotations(changed)
    )


def test_validate_manifest_binds_task_and_consecutive_samples() -> None:
    value = manifest()
    assert [
        sample["sample_index"] for sample in validate_manifest(value, SYNTHETIC_TASK_ID)
    ] == [1, 2]

    with pytest.raises(SnapshotValidationError, match="does not match manifest"):
        validate_manifest(value, OTHER_SYNTHETIC_TASK_ID)

    value["samples"][1]["sample_index"] = 3
    with pytest.raises(SnapshotValidationError, match="consecutive"):
        validate_manifest(value, SYNTHETIC_TASK_ID)


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda value: value.pop(), "exactly match"),
        (lambda value: value.append({"id": 99, "name": "rider"}), "exactly match"),
        (lambda value: value.__setitem__(1, {**value[1], "id": value[0]["id"]}), "unique"),
    ],
)
def test_validate_labels_requires_exact_eight_class_taxonomy(mutate, match: str) -> None:
    value = labels()
    mutate(value)

    with pytest.raises(SnapshotValidationError, match=match):
        validate_labels(value)


def test_validate_jobs_binds_manifest_ids_task_and_full_frame_coverage() -> None:
    jobs = [
        {
            "id": SYNTHETIC_JOB_ID,
            "task_id": SYNTHETIC_TASK_ID,
            "start_frame": 0,
            "stop_frame": 1,
            "frame_count": 2,
        }
    ]
    assert validate_jobs(
        jobs,
        task_id=SYNTHETIC_TASK_ID,
        expected_job_ids=[SYNTHETIC_JOB_ID],
        frame_count=2,
    ) == jobs

    with pytest.raises(SnapshotValidationError, match="manifest jobs"):
        validate_jobs(
            jobs,
            task_id=SYNTHETIC_TASK_ID,
            expected_job_ids=[OTHER_SYNTHETIC_JOB_ID],
            frame_count=2,
        )
    with pytest.raises(SnapshotValidationError, match="different task"):
        validate_jobs(
            [{**jobs[0], "task_id": OTHER_SYNTHETIC_TASK_ID}],
            task_id=SYNTHETIC_TASK_ID,
            expected_job_ids=[SYNTHETIC_JOB_ID],
            frame_count=2,
        )
    with pytest.raises(SnapshotValidationError, match="do not cover"):
        validate_jobs(
            [{**jobs[0], "stop_frame": 0, "frame_count": 1}],
            task_id=SYNTHETIC_TASK_ID,
            expected_job_ids=[SYNTHETIC_JOB_ID],
            frame_count=2,
        )


def test_validate_frame_mapping_checks_names_and_annotation_range() -> None:
    samples = manifest()["samples"]
    frames = frame_metadata()
    assert validate_frame_mapping(samples, frames, annotations(), task_size=2) == frames

    with pytest.raises(SnapshotValidationError, match="expected"):
        validate_frame_mapping(
            samples, [{**frames[0], "name": "wrong.png"}, frames[1]], annotations(), task_size=2
        )
    bad_annotations = annotations()
    bad_annotations["shapes"][0]["frame"] = 2
    with pytest.raises(SnapshotValidationError, match="outside"):
        validate_frame_mapping(samples, frames, bad_annotations, task_size=2)


def test_build_image_inventory_hashes_dimensions_and_rejects_drift(tmp_path: Path) -> None:
    manifest_path = tmp_path / "sample-manifest.json"
    value = manifest()
    write_manifest(manifest_path, value)
    write_fixture_images(manifest_path)

    inventory = build_image_inventory(manifest_path, value["samples"], frame_metadata())

    image_bytes = (tmp_path / "images/frame_000.png").read_bytes()
    assert inventory[0]["sha256"] == hashlib.sha256(image_bytes).hexdigest()
    assert (inventory[0]["width"], inventory[0]["height"]) == (32, 24)
    assert inventory[0]["size_bytes"] == len(image_bytes)
    assert inventory[1]["cvat_frame"] == 1

    with pytest.raises(SnapshotValidationError, match="width"):
        build_image_inventory(
            manifest_path,
            value["samples"],
            [{**frame_metadata()[0], "width": 31}, frame_metadata()[1]],
        )


def test_build_snapshot_preserves_complete_annotations_and_sets_track_gate(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "sample-manifest.json"
    value = manifest()
    write_manifest(manifest_path, value)
    write_fixture_images(manifest_path)
    track = {
        "id": 3,
        "frame": 0,
        "label_id": 12,
        "group": 0,
        "source": "manual",
        "attributes": [],
        "elements": [],
        "shapes": [
            {
                "id": 4,
                "frame": 1,
                "type": "rectangle",
                "points": [2.0, 3.0, 11.0, 21.0],
                "outside": False,
                "occluded": True,
                "z_order": 2,
                "rotation": 0.0,
                "attributes": [],
            }
        ],
    }
    raw_annotations = annotations(tracks=[track])

    snapshot = build_snapshot_payload(
        task_id=SYNTHETIC_TASK_ID,
        manifest=value,
        manifest_path=manifest_path,
        task={
            "id": SYNTHETIC_TASK_ID,
            "project_id": 9,
            "name": "training",
            "size": 2,
            "status": "annotation",
        },
        labels=labels(),
        jobs=[
            {
                "id": SYNTHETIC_JOB_ID,
                "task_id": SYNTHETIC_TASK_ID,
                "start_frame": 0,
                "stop_frame": 1,
                "frame_count": 2,
                "stage": "annotation",
                "state": "new",
            }
        ],
        metadata={"size": 2, "start_frame": 0, "stop_frame": 1, "frames": frame_metadata()},
        annotations=raw_annotations,
        created_at="2026-08-31T00:00:00+00:00",
    )

    assert snapshot["snapshot_schema"] == SNAPSHOT_SCHEMA
    assert snapshot["counts"] == {
        "images": 2,
        "tags": 1,
        "shapes": 1,
        "tracks": 1,
        "annotations_by_label": {
            "car": 1,
            "bus": 1,
            "truck": 1,
            "motorcycle": 0,
            "bicycle": 0,
            "pedestrian": 0,
            "traffic_light": 0,
            "traffic_sign": 0,
        },
    }
    assert snapshot["annotations"]["tracks"][0]["shapes"][0]["occluded"] is True
    assert snapshot["canonical_annotations_sha256"] == canonical_sha256(snapshot["annotations"])
    assert snapshot["final_gate"]["passed"] is False
    assert "TRACKS_PRESENT" in snapshot["final_gate"]["blocking_reasons"][0]


def test_build_snapshot_rejects_unknown_annotation_label(tmp_path: Path) -> None:
    manifest_path = tmp_path / "sample-manifest.json"
    value = manifest()
    write_manifest(manifest_path, value)
    write_fixture_images(manifest_path)
    invalid = annotations()
    invalid["shapes"][0]["label_id"] = 999

    with pytest.raises(SnapshotValidationError, match="unknown label"):
        build_snapshot_payload(
            task_id=SYNTHETIC_TASK_ID,
            manifest=value,
            manifest_path=manifest_path,
            task={"id": SYNTHETIC_TASK_ID, "project_id": 9, "size": 2},
            labels=labels(),
            jobs=[
                {
                    "id": SYNTHETIC_JOB_ID,
                    "task_id": SYNTHETIC_TASK_ID,
                    "start_frame": 0,
                    "stop_frame": 1,
                    "frame_count": 2,
                }
            ],
            metadata={"size": 2, "frames": frame_metadata()},
            annotations=invalid,
            created_at="2026-08-31T00:00:00+00:00",
        )


def test_atomic_write_json_new_is_atomic_and_never_overwrites(tmp_path: Path) -> None:
    output = tmp_path / "new" / "snapshot.json"
    atomic_write_json_new(output, {"b": 2, "a": 1})

    assert json.loads(output.read_text(encoding="utf-8")) == {"a": 1, "b": 2}
    assert not list(output.parent.glob(".*.tmp"))
    with pytest.raises(FileExistsError, match="already exists"):
        atomic_write_json_new(output, {"replacement": True})
    assert json.loads(output.read_text(encoding="utf-8")) == {"a": 1, "b": 2}

    directory_target = tmp_path / "directory-target"
    directory_target.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        atomic_write_json_new(directory_target, {"replacement": True})
