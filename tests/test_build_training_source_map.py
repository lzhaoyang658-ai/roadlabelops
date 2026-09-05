from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.build_training_source_map import (
    SOURCE_MAP_SCHEMA,
    TrainingSourceMapError,
    build_training_source_map,
    main,
)


@pytest.fixture(autouse=True)
def use_fixture_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest(
    *,
    session_id: str,
    source_sha256: str,
    revision: str,
    samples: list[tuple[str, int]],
) -> dict[str, object]:
    counts: dict[str, int] = {}
    records: list[dict[str, object]] = []
    for index, (scene_id, source_frame) in enumerate(samples, start=1):
        counts[scene_id] = counts.get(scene_id, 0) + 1
        records.append(
            {
                "sample_index": index,
                "scene_id": scene_id,
                "source_frame": source_frame,
                "file_name": f"{revision}-{scene_id}-{source_frame:04d}.jpg",
            }
        )
    return {
        "session_id": session_id,
        "source_sha256": source_sha256,
        "sampling_revision": revision,
        "purpose": "training",
        "sample_size": len(records),
        "sample_counts_by_scene": counts,
        "samples": records,
    }


def make_inputs(tmp_path: Path) -> dict[str, object]:
    compiled_path = tmp_path / "data" / "raw" / "compiled.mp4"
    compiled_path.parent.mkdir(parents=True)
    compiled_path.write_bytes(b"immutable compiled video fixture")
    compiled_sha = sha256(compiled_path)
    session_id = "session-test"
    session = {
        "session_id": session_id,
        "source_path": str(compiled_path),
        "source_sha256": compiled_sha,
        "duration_seconds": 1.2,
        "fps": 10.0,
        "width": 1280,
        "height": 720,
        "scenes": [
            {
                "scene_id": "scene-a",
                "session_id": session_id,
                "start_seconds": 0.0,
                "end_seconds": 0.3,
                "cvat_task_id": 11,
            },
            {
                "scene_id": "scene-b",
                "session_id": session_id,
                "start_seconds": 0.3,
                "end_seconds": 0.9,
                "cvat_task_id": 12,
            },
            {
                "scene_id": "scene-c",
                "session_id": session_id,
                "start_seconds": 0.9,
                "end_seconds": 1.2,
                "cvat_task_id": 13,
            },
        ],
    }
    session_path = tmp_path / "data" / "sessions" / "session-test.json"
    write_json(session_path, session)
    first_manifest = manifest(
        session_id=session_id,
        source_sha256=compiled_sha,
        revision="primary-v1",
        samples=[
            ("scene-a", 0),
            ("scene-a", 2),
            ("scene-b", 0),
            ("scene-b", 1),
            ("scene-b", 2),
            ("scene-b", 5),
        ],
    )
    second_manifest = manifest(
        session_id=session_id,
        source_sha256=compiled_sha,
        revision="supplement-v1",
        samples=[("scene-c", 0), ("scene-c", 2)],
    )
    first_manifest_path = tmp_path / "data" / "sessions" / "primary" / "manifest.json"
    second_manifest_path = tmp_path / "data" / "sessions" / "supplement" / "manifest.json"
    write_json(first_manifest_path, first_manifest)
    write_json(second_manifest_path, second_manifest)
    evidence = {
        "source_isolation": {"selected_asset_ids": [101, 202]},
        "sources": [
            {
                "asset_id": 101,
                "title": "first source",
                "page_url": "https://example.test/assets/101",
                "download_url": "https://example.test/assets/101.mp4",
                "duration_seconds": 0.5,
                "fps": "25/1",
                "sha256": "a" * 64,
            },
            {
                "asset_id": 202,
                "title": "second source",
                "page_url": "https://example.test/assets/202",
                "download_url": "https://example.test/assets/202.mp4",
                "duration_seconds": 0.7,
                "fps": "30000/1001",
                "sha256": "b" * 64,
            },
        ],
        "compiled_source": {
            "path": "data/raw/compiled.mp4",
            "transform": "normalize to 10 fps and concatenate in source array order",
            "duration_seconds": 1.2,
            "width": 1280,
            "height": 720,
            "fps": 10,
            "frame_count": 12,
            "bytes": compiled_path.stat().st_size,
            "sha256": compiled_sha,
        },
        "workflow": {
            "session_id": session_id,
            "scene_count": 3,
            "scene_task_ids": [11, 12, 13],
        },
        "sample": {
            "manifest_path": "data/sessions/primary/manifest.json",
            "revision": "primary-v1",
            "sample_size": first_manifest["sample_size"],
            "sample_counts_by_scene": first_manifest["sample_counts_by_scene"],
        },
    }
    evidence_path = tmp_path / "docs" / "evidence" / "training.json"
    write_json(evidence_path, evidence)
    return {
        "session": session,
        "session_path": session_path,
        "evidence": evidence,
        "evidence_path": evidence_path,
        "manifests": [first_manifest_path, second_manifest_path],
        "compiled_path": compiled_path,
    }


def build(fixture: dict[str, object], output: Path) -> dict[str, object]:
    return build_training_source_map(
        session_record_path=fixture["session_path"],  # type: ignore[arg-type]
        training_evidence_path=fixture["evidence_path"],  # type: ignore[arg-type]
        manifest_paths=fixture["manifests"],  # type: ignore[arg-type]
        output=output,
    )


def test_maps_scene_across_assets_and_asset_across_scenes_at_boundaries(tmp_path: Path) -> None:
    fixture = make_inputs(tmp_path)
    output = tmp_path / "source-map.json"

    summary = build(fixture, output)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == SOURCE_MAP_SCHEMA
    assert summary["mapped_frame_count"] == 8
    assert [asset["asset_id"] for asset in payload["assets"]] == [101, 202]
    assert payload["assets"][0]["title"] == "first source"
    assert payload["assets"][0]["normalized"] == {
        "fps": 10.0,
        "start_frame": 0,
        "end_frame_exclusive": 5,
        "frame_count": 5,
    }
    assert payload["assets"][1]["normalized"] == {
        "fps": 10.0,
        "start_frame": 5,
        "end_frame_exclusive": 12,
        "frame_count": 7,
    }
    mapping = {
        (frame["scene_id"], frame["source_frame"]): (
            frame["asset_id"],
            frame["normalized_asset_frame"],
        )
        for frame in payload["frames"]
    }
    assert mapping[("scene-b", 1)] == (101, 4)
    assert mapping[("scene-b", 2)] == (202, 0)
    assert mapping[("scene-c", 0)] == (202, 4)
    assert mapping[("scene-c", 2)] == (202, 6)
    assert all(frame["leakage_group_id"].startswith("sha256:") for frame in payload["frames"])
    assert payload["assets"][0]["leakage_group_id"] == f"sha256:{'a' * 64}"
    assert payload["evidence"]["compiled_source"] == {
        "path": "data/raw/compiled.mp4",
        "sha256": sha256(fixture["compiled_path"]),  # type: ignore[arg-type]
        "fps": 10,
        "frame_count": 12,
        "method": "normalize to 10 fps and concatenate in source array order",
    }
    assert payload["evidence"]["session_record"]["path"] == ("data/sessions/session-test.json")
    assert payload["evidence"]["training_evidence"]["path"] == ("docs/evidence/training.json")
    assert [record["path"] for record in payload["evidence"]["manifests"]] == [
        "data/sessions/primary/manifest.json",
        "data/sessions/supplement/manifest.json",
    ]
    assert str(tmp_path) not in output.read_text(encoding="utf-8")


def test_manifest_input_order_does_not_change_output(tmp_path: Path) -> None:
    fixture = make_inputs(tmp_path)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    build(fixture, first)
    reversed_fixture = dict(fixture)
    reversed_fixture["manifests"] = list(reversed(fixture["manifests"]))  # type: ignore[arg-type]
    build(reversed_fixture, second)

    assert first.read_bytes() == second.read_bytes()
    payload = json.loads(first.read_text(encoding="utf-8"))
    assert payload["frames"] == sorted(
        payload["frames"], key=lambda item: (item["scene_id"], item["source_frame"])
    )
    assert [record["sampling_revision"] for record in payload["evidence"]["manifests"]] == [
        "primary-v1",
        "supplement-v1",
    ]


def test_precision_aware_half_up_recovers_serialized_half_frame(tmp_path: Path) -> None:
    fixture = make_inputs(tmp_path)
    compiled_path: Path = fixture["compiled_path"]  # type: ignore[assignment]
    compiled_sha = sha256(compiled_path)
    session = fixture["session"]
    session["duration_seconds"] = 16.7  # type: ignore[index]
    session["fps"] = 30.0  # type: ignore[index]
    session["scenes"] = [  # type: ignore[index]
        {
            "scene_id": "scene-a",
            "session_id": "session-test",
            "start_seconds": 0.0,
            "end_seconds": 16.7,
            "cvat_task_id": 11,
        }
    ]
    write_json(fixture["session_path"], session)  # type: ignore[arg-type]
    only_manifest_path: Path = fixture["manifests"][0]  # type: ignore[index,assignment]
    only_manifest = manifest(
        session_id="session-test",
        source_sha256=compiled_sha,
        revision="primary-v1",
        samples=[("scene-a", 500)],
    )
    write_json(only_manifest_path, only_manifest)
    fixture["manifests"] = [only_manifest_path]
    evidence = fixture["evidence"]
    evidence["sources"] = [  # type: ignore[index]
        {
            "asset_id": 101,
            "page_url": "https://example.test/assets/101",
            "download_url": "https://example.test/assets/101.mp4",
            "duration_seconds": 16.683333,
            "fps": "24000/1001",
            "sha256": "a" * 64,
        }
    ]
    evidence["source_isolation"] = {"selected_asset_ids": [101]}  # type: ignore[index]
    evidence["compiled_source"].update(  # type: ignore[index,union-attr]
        {"duration_seconds": 16.7, "fps": 30, "frame_count": 501}
    )
    evidence["workflow"] = {  # type: ignore[index]
        "session_id": "session-test",
        "scene_count": 1,
        "scene_task_ids": [11],
    }
    evidence["sample"] = {  # type: ignore[index]
        "manifest_path": "data/sessions/primary/manifest.json",
        "revision": "primary-v1",
        "sample_size": 1,
        "sample_counts_by_scene": {"scene-a": 1},
    }
    write_json(fixture["evidence_path"], evidence)  # type: ignore[arg-type]

    output = tmp_path / "precision-aware.json"
    build(fixture, output)

    asset = json.loads(output.read_text(encoding="utf-8"))["assets"][0]
    assert asset["raw_frame_estimate"] == pytest.approx(500.49999)
    assert asset["duration_precision"] == pytest.approx(0.000001)
    assert asset["rounding_tolerance_frames"] == pytest.approx(0.000015)
    assert asset["normalized_frame_count"] == 501


def test_half_frame_shortfall_beyond_duration_precision_is_not_reconciled(tmp_path: Path) -> None:
    fixture = make_inputs(tmp_path)
    evidence = fixture["evidence"]
    # 4.4998 frames is farther from the half boundary than the propagated
    # 0.00005-frame duration precision, so it must remain four frames.
    evidence["sources"][0]["duration_seconds"] = 0.44998  # type: ignore[index]
    write_json(fixture["evidence_path"], evidence)  # type: ignore[arg-type]

    output = tmp_path / "rejected.json"
    with pytest.raises(TrainingSourceMapError, match="normalized source frame total differs"):
        build(fixture, output)

    assert not output.exists()
    assert not list(tmp_path.glob(".rejected.json.*"))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["session"]["scenes"][1].update({"start_seconds": 0.2}),
            "overlaps",
        ),
        (
            lambda value: value["session"]["scenes"][1].update({"start_seconds": 0.4}),
            "missing coverage",
        ),
        (
            lambda value: value["session"]["scenes"][1].update({"start_seconds": 0.35}),
            "not aligned to an integer compiled frame",
        ),
    ],
)
def test_rejects_overlapping_missing_or_non_frame_aligned_scenes(
    tmp_path: Path, mutation: object, message: str
) -> None:
    fixture = make_inputs(tmp_path)
    mutation(fixture)  # type: ignore[operator]
    write_json(fixture["session_path"], fixture["session"])  # type: ignore[arg-type]

    with pytest.raises(TrainingSourceMapError, match=message):
        build(fixture, tmp_path / "rejected.json")


def test_rejects_normalized_asset_total_mismatch_without_partial_output(tmp_path: Path) -> None:
    fixture = make_inputs(tmp_path)
    evidence = fixture["evidence"]
    evidence["sources"][1]["duration_seconds"] = 0.8  # type: ignore[index]
    write_json(fixture["evidence_path"], evidence)  # type: ignore[arg-type]
    output = tmp_path / "rejected.json"

    with pytest.raises(TrainingSourceMapError, match="normalized source frame total differs"):
        build(fixture, output)

    assert not output.exists()


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("asset", "duplicate asset_id"),
        ("source-sha", "aliases the same source SHA-256"),
        ("scene", "duplicate scene_id"),
        ("sample", "duplicate sample"),
        ("manifest", "supplied more than once"),
    ],
)
def test_rejects_duplicate_inputs(tmp_path: Path, kind: str, message: str) -> None:
    fixture = make_inputs(tmp_path)
    if kind == "asset":
        evidence = fixture["evidence"]
        evidence["sources"][1]["asset_id"] = 101  # type: ignore[index]
        write_json(fixture["evidence_path"], evidence)  # type: ignore[arg-type]
    elif kind == "source-sha":
        evidence = fixture["evidence"]
        evidence["sources"][1]["sha256"] = "a" * 64  # type: ignore[index]
        write_json(fixture["evidence_path"], evidence)  # type: ignore[arg-type]
    elif kind == "scene":
        session = fixture["session"]
        session["scenes"][1]["scene_id"] = "scene-a"  # type: ignore[index]
        write_json(fixture["session_path"], session)  # type: ignore[arg-type]
    elif kind == "sample":
        second_path: Path = fixture["manifests"][1]  # type: ignore[index,assignment]
        second = json.loads(second_path.read_text(encoding="utf-8"))
        second["samples"][0]["scene_id"] = "scene-a"
        second["samples"][0]["source_frame"] = 0
        second["samples"][0]["file_name"] = "still-unique.jpg"
        second["sample_counts_by_scene"] = {"scene-a": 1, "scene-c": 1}
        write_json(second_path, second)
    else:
        fixture["manifests"] = [fixture["manifests"][0], fixture["manifests"][0]]  # type: ignore[index]

    with pytest.raises(TrainingSourceMapError, match=message):
        build(fixture, tmp_path / "rejected.json")


@pytest.mark.parametrize("drift", ["session-sha", "manifest-sha", "width", "bytes"])
def test_rejects_hash_and_compiled_metadata_drift(tmp_path: Path, drift: str) -> None:
    fixture = make_inputs(tmp_path)
    if drift == "session-sha":
        fixture["session"]["source_sha256"] = "0" * 64  # type: ignore[index]
        write_json(fixture["session_path"], fixture["session"])  # type: ignore[arg-type]
    elif drift == "manifest-sha":
        path: Path = fixture["manifests"][0]  # type: ignore[index,assignment]
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["source_sha256"] = "0" * 64
        write_json(path, payload)
    elif drift == "width":
        fixture["session"]["width"] = 1920  # type: ignore[index]
        write_json(fixture["session_path"], fixture["session"])  # type: ignore[arg-type]
    else:
        fixture["evidence"]["compiled_source"]["bytes"] += 1  # type: ignore[index,operator]
        write_json(fixture["evidence_path"], fixture["evidence"])  # type: ignore[arg-type]

    with pytest.raises(TrainingSourceMapError, match="differs"):
        build(fixture, tmp_path / "rejected.json")


def test_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    fixture = make_inputs(tmp_path)
    output = tmp_path / "owned.json"
    output.write_text("owned by user\n", encoding="utf-8")

    with pytest.raises(TrainingSourceMapError, match="already exists"):
        build(fixture, output)

    assert output.read_text(encoding="utf-8") == "owned by user\n"


def test_broken_output_symlink_is_never_followed(tmp_path: Path) -> None:
    fixture = make_inputs(tmp_path)
    redirected = tmp_path / "redirected.json"
    output = tmp_path / "broken-link.json"
    output.symlink_to(redirected)

    with pytest.raises(TrainingSourceMapError, match="already exists"):
        build(fixture, output)

    assert output.is_symlink()
    assert not redirected.exists()


def test_cli_supports_repeated_manifests(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture = make_inputs(tmp_path)
    output = tmp_path / "cli.json"
    manifests: list[Path] = fixture["manifests"]  # type: ignore[assignment]

    main(
        [
            "--session-record",
            str(fixture["session_path"]),
            "--training-evidence",
            str(fixture["evidence_path"]),
            "--manifest",
            str(manifests[1]),
            "--manifest",
            str(manifests[0]),
            "--output",
            str(output),
        ]
    )

    summary = json.loads(capsys.readouterr().out)
    assert summary["mapped_frame_count"] == 8
    assert summary["mutation_performed"] is False
    assert summary["source_map_sha256"] == sha256(output)


def test_training_evidence_primary_manifest_metadata_is_bound(tmp_path: Path) -> None:
    fixture = make_inputs(tmp_path)
    evidence = deepcopy(fixture["evidence"])
    evidence["sample"]["sample_size"] = 999
    write_json(fixture["evidence_path"], evidence)  # type: ignore[arg-type]

    with pytest.raises(TrainingSourceMapError, match="sample.sample_size differs"):
        build(fixture, tmp_path / "rejected.json")
