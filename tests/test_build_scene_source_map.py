from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import scripts.build_scene_source_map as builder
from scripts.build_scene_source_map import (
    SOURCE_MAP_SCHEMA,
    SceneSourceMapError,
    build_scene_source_map,
    main,
)


@pytest.fixture(autouse=True)
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        builder,
        "_probe_video",
        lambda _path, *, target_frames, location: (30, 450),
    )


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fixture_inputs(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    scenes = {f"scene-{index}": tmp_path / "scenes" / f"scene-{index}.mp4" for index in range(1, 5)}
    for index, path in enumerate(scenes.values(), start=1):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"unique scene video {index}".encode())
    samples = []
    counts = {}
    for sample_index, (scene_id, source_frame) in enumerate(
        [("scene-1", 4), ("scene-1", 39), ("scene-2", 4), ("scene-3", 44), ("scene-4", 9)],
        start=1,
    ):
        counts[scene_id] = counts.get(scene_id, 0) + 1
        samples.append(
            {
                "sample_index": sample_index,
                "scene_id": scene_id,
                "source_frame": source_frame,
                "file_name": f"{scene_id}-{source_frame}.jpg",
            }
        )
    manifest = tmp_path / "sample-manifest.json"
    write_json(
        manifest,
        {
            "session_id": "session-test",
            "sampling_revision": "frame-aligned-v1",
            "source_sha256": "a" * 64,
            "sample_size": len(samples),
            "sample_counts_by_scene": counts,
            "samples": samples,
        },
    )
    return manifest, scenes


def test_builds_four_scene_v2_map_with_identity_frames(tmp_path: Path) -> None:
    manifest, scenes = fixture_inputs(tmp_path)
    output = tmp_path / "source-map.json"

    summary = build_scene_source_map(
        manifest_path=manifest,
        scene_videos=scenes,
        output=output,
        frame_step=5,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == SOURCE_MAP_SCHEMA
    assert summary == {"output": str(output), "scene_count": 4, "mapped_frame_count": 5}
    assert [asset["asset_id"] for asset in payload["assets"]] == [
        "scene-1",
        "scene-2",
        "scene-3",
        "scene-4",
    ]
    assert all(asset["fps"] == 30 for asset in payload["assets"])
    assert all(asset["frame_count"] == 450 for asset in payload["assets"])
    assert all("frame_step" not in asset and "scene_id" not in asset for asset in payload["assets"])
    assert all(asset["path"].startswith("scenes/") for asset in payload["assets"])
    assert all(
        frame["normalized_asset_frame"] == frame["source_frame"] for frame in payload["frames"]
    )
    by_scene = {asset["asset_id"]: asset for asset in payload["assets"]}
    for frame in payload["frames"]:
        asset = by_scene[frame["scene_id"]]
        assert frame["asset_id"] == frame["scene_id"]
        assert frame["leakage_group_id"] == asset["leakage_group_id"]
        assert asset["sha256"] == hashlib.sha256(scenes[frame["scene_id"]].read_bytes()).hexdigest()
    assert str(tmp_path) not in output.read_text(encoding="utf-8")
    assert payload["evidence"] == {
        "mapping": (
            "Each holdout scene MP4 is a content-addressed production-video asset; "
            "scene source frames are already normalized 30 fps frames."
        ),
        "sample_manifest": {
            "path": "sample-manifest.json",
            "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        },
    }


def test_rejects_incomplete_scene_video_universe(tmp_path: Path) -> None:
    manifest, scenes = fixture_inputs(tmp_path)
    scenes.pop("scene-4")
    with pytest.raises(SceneSourceMapError, match="exactly cover sampled scenes"):
        build_scene_source_map(
            manifest_path=manifest,
            scene_videos=scenes,
            output=tmp_path / "source-map.json",
        )


def test_rejects_duplicate_frame_and_identical_scene_content(tmp_path: Path) -> None:
    manifest, scenes = fixture_inputs(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["samples"][1]["source_frame"] = payload["samples"][0]["source_frame"]
    write_json(manifest, payload)
    with pytest.raises(SceneSourceMapError, match="duplicate sample/frame identity"):
        build_scene_source_map(
            manifest_path=manifest,
            scene_videos=scenes,
            output=tmp_path / "duplicate-frame.json",
        )

    manifest, scenes = fixture_inputs(tmp_path)
    scenes["scene-4"].write_bytes(scenes["scene-3"].read_bytes())
    with pytest.raises(SceneSourceMapError, match="identical content"):
        build_scene_source_map(
            manifest_path=manifest,
            scene_videos=scenes,
            output=tmp_path / "duplicate-video.json",
        )


def test_rejects_unreachable_frame_for_step(tmp_path: Path) -> None:
    manifest, scenes = fixture_inputs(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["samples"][0]["source_frame"] = 5
    write_json(manifest, payload)
    with pytest.raises(SceneSourceMapError, match="unreachable at frame_step=5"):
        build_scene_source_map(
            manifest_path=manifest,
            scene_videos=scenes,
            output=tmp_path / "unreachable.json",
            frame_step=5,
        )


def test_rejects_scene_video_symlink_before_resolving(tmp_path: Path) -> None:
    manifest, scenes = fixture_inputs(tmp_path)
    real_video = scenes["scene-4"].with_name("real-scene-4.mp4")
    scenes["scene-4"].rename(real_video)
    scenes["scene-4"].symlink_to(real_video)
    with pytest.raises(SceneSourceMapError, match="symlink"):
        build_scene_source_map(
            manifest_path=manifest,
            scene_videos=scenes,
            output=tmp_path / "symlink.json",
        )


def test_rejects_parent_directory_symlink_outside_workspace(tmp_path: Path) -> None:
    manifest, scenes = fixture_inputs(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-outside-scenes"
    outside.mkdir()
    outside_video = outside / "scene-4.mp4"
    scenes["scene-4"].replace(outside_video)
    linked = tmp_path / "linked-scenes"
    linked.symlink_to(outside, target_is_directory=True)
    scenes["scene-4"] = linked / "scene-4.mp4"
    with pytest.raises(SceneSourceMapError, match="symlink"):
        build_scene_source_map(
            manifest_path=manifest,
            scene_videos=scenes,
            output=tmp_path / "parent-symlink.json",
        )


def test_rejects_video_changed_between_probe_and_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, scenes = fixture_inputs(tmp_path)

    def mutate(path: Path, *, target_frames: list[int], location: str) -> tuple[int, int]:
        del target_frames, location
        path.write_bytes(path.read_bytes() + b"changed")
        return 30, 450

    monkeypatch.setattr(builder, "_probe_video", mutate)
    with pytest.raises(SceneSourceMapError, match="changed while video metadata was checked"):
        build_scene_source_map(
            manifest_path=manifest,
            scene_videos=scenes,
            output=tmp_path / "changed.json",
        )


def test_exclusive_output_and_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest, scenes = fixture_inputs(tmp_path)
    output = tmp_path / "source-map.json"
    argv = ["--manifest", str(manifest)]
    for scene_id, path in scenes.items():
        argv += ["--scene-video", f"{scene_id}={path}"]
    argv += ["--frame-step", "5", "--output", str(output)]
    assert main(argv) == 0
    assert json.loads(capsys.readouterr().out)["mapped_frame_count"] == 5

    with pytest.raises(FileExistsError, match="already exists"):
        build_scene_source_map(
            manifest_path=manifest,
            scene_videos=scenes,
            output=output,
        )
