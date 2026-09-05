from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from roadlabelops.models import ToolResult
from roadlabelops.tools import video


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _install_fake_media_tools(
    monkeypatch,
    source: Path,
    destination: Path,
    *,
    fail_cut: int | None = None,
) -> list[Path]:
    observed_outputs: list[Path] = []

    def fake_probe(path_value):
        path = Path(path_value).resolve()
        if path == source.resolve():
            return ToolResult.success({
                "path": str(path),
                "sha256": _hash(path),
                "duration_seconds": 25.0,
                "fps": 25.0,
                "width": 1280,
                "height": 720,
                "codec": "h264",
            })
        duration_by_name = {"scene_001.mp4": 10.0, "scene_002.mp4": 10.0, "scene_003.mp4": 5.0}
        return ToolResult.success({
            "path": str(path),
            "sha256": _hash(path),
            "duration_seconds": duration_by_name[path.name],
            "fps": 25.0,
            "width": 1280,
            "height": 720,
            "codec": "h264",
        })

    cut_number = 0

    def fake_run(command, *, timeout):
        nonlocal cut_number
        assert timeout in {60, 180}
        assert not destination.exists()
        output = Path(command[-1])
        observed_outputs.append(output)
        if output.suffix == ".mp4":
            cut_number += 1
            if cut_number == fail_cut:
                return ToolResult.failure("FFMPEG_FAILED", "synthetic cut failure", retryable=True)
        output.write_bytes(output.name.encode("utf-8"))
        return ToolResult.success()

    monkeypatch.setattr(video, "probe_video", fake_probe)
    monkeypatch.setattr(video, "_run_ffmpeg", fake_run)
    return observed_outputs


def test_split_video_validates_in_staging_then_atomically_publishes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "road.mp4"
    source.write_bytes(b"source")
    destination = tmp_path / "scenes" / "session_test"
    observed = _install_fake_media_tools(monkeypatch, source, destination)

    result = video.split_video(source, destination, scene_seconds=10, frame_step=5)

    assert result.ok
    assert len(result.data["scenes"]) == 3
    assert all(path.parent.name.startswith(".session_test.staging-") for path in observed)
    assert [item["video_sha256"] for item in result.data["scenes"]] == [
        _hash(destination / f"scene_{index:03d}.mp4") for index in range(1, 4)
    ]
    assert all(Path(item["video_path"]).parent == destination for item in result.data["scenes"])
    assert not list(destination.parent.glob(".session_test.staging-*"))


def test_split_failure_removes_staging_and_never_publishes_partial_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "road.mp4"
    source.write_bytes(b"source")
    destination = tmp_path / "scenes" / "session_test"
    _install_fake_media_tools(monkeypatch, source, destination, fail_cut=2)

    result = video.split_video(source, destination, scene_seconds=10)

    assert not result.ok
    assert result.error["code"] == "SPLIT_FAILED"
    assert result.retryable
    assert not destination.exists()
    assert not list(destination.parent.glob(".session_test.staging-*"))


def test_split_never_reuses_preexisting_partial_output(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "road.mp4"
    source.write_bytes(b"source")
    destination = tmp_path / "scenes" / "session_test"
    destination.mkdir(parents=True)
    partial = destination / "scene_001.mp4"
    partial.write_bytes(b"partial")
    monkeypatch.setattr(
        video,
        "probe_video",
        lambda _path: ToolResult.success({
            "duration_seconds": 10.0,
            "fps": 25.0,
        }),
    )

    result = video.split_video(source, destination)

    assert not result.ok
    assert result.error["code"] == "SPLIT_OUTPUT_EXISTS"
    assert partial.read_bytes() == b"partial"


def test_split_rejects_symbolic_link_destination(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "road.mp4"
    source.write_bytes(b"source")
    actual = tmp_path / "actual"
    actual.mkdir()
    destination = tmp_path / "scenes"
    destination.symlink_to(actual, target_is_directory=True)
    monkeypatch.setattr(
        video,
        "probe_video",
        lambda _path: ToolResult.success({
            "duration_seconds": 10.0,
            "fps": 25.0,
        }),
    )

    result = video.split_video(source, destination)

    assert not result.ok
    assert result.error["code"] == "SPLIT_OUTPUT_INVALID"
    assert list(actual.iterdir()) == []


def test_probe_video_normalizes_invalid_ffprobe_payload(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "road.mp4"
    source.write_bytes(b"source")
    monkeypatch.setattr(video.shutil, "which", lambda _name: "/usr/bin/ffprobe")
    monkeypatch.setattr(
        video.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, "{\"streams\": []}", ""),
    )

    result = video.probe_video(source)

    assert not result.ok
    assert result.error["code"] == "VIDEO_INVALID"


def test_probe_video_normalizes_timeout_as_retryable(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "road.mp4"
    source.write_bytes(b"source")
    monkeypatch.setattr(video.shutil, "which", lambda _name: "/usr/bin/ffprobe")

    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("ffprobe", 30)

    monkeypatch.setattr(video.subprocess, "run", timeout)

    result = video.probe_video(source)

    assert not result.ok
    assert result.error["code"] == "VIDEO_PROBE_TIMEOUT"
    assert result.retryable


def test_split_hard_duration_and_scene_limits_fail_before_ffmpeg(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "road.mp4"
    source.write_bytes(b"source")
    probe = ToolResult.success(
        {
            "sha256": _hash(source),
            "duration_seconds": 601.0,
            "fps": 25.0,
            "width": 1280,
            "height": 720,
        }
    )
    monkeypatch.setattr(video, "probe_video", lambda _path: probe)
    monkeypatch.setattr(
        video,
        "_run_ffmpeg",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("FFmpeg must not run past a hard safety limit")
        ),
    )

    duration_limited = video.split_video(
        source,
        tmp_path / "duration-limited",
        absolute_max_duration_seconds=600,
    )
    scene_limited = video.split_video(
        source,
        tmp_path / "scene-limited",
        scene_seconds=10,
        absolute_max_duration_seconds=1000,
        max_scene_count=60,
    )

    assert duration_limited.error and duration_limited.error["code"] == (
        "VIDEO_DURATION_HARD_LIMIT"
    )
    assert scene_limited.error and scene_limited.error["code"] == (
        "VIDEO_SCENE_COUNT_LIMIT"
    )
    assert not list(tmp_path.glob(".*.staging-*"))


def test_split_rejects_insufficient_storage_before_staging(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "road.mp4"
    source.write_bytes(b"source")
    probe = ToolResult.success(
        {
            "sha256": _hash(source),
            "duration_seconds": 10.0,
            "fps": 25.0,
            "width": 3840,
            "height": 2160,
        }
    )
    monkeypatch.setattr(video, "probe_video", lambda _path: probe)
    monkeypatch.setattr(
        video.shutil,
        "disk_usage",
        lambda _path: type("DiskUsage", (), {"free": 0})(),
    )

    result = video.split_video(source, tmp_path / "scenes")

    assert not result.ok
    assert result.error and result.error["code"] == "SPLIT_STORAGE_INSUFFICIENT"
    assert result.retryable
    assert not list(tmp_path.glob(".scenes.staging-*"))


def test_split_output_byte_quota_cleans_staging_and_never_publishes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "road.mp4"
    source.write_bytes(b"source")
    destination = tmp_path / "scenes"
    probe = ToolResult.success(
        {
            "sha256": _hash(source),
            "duration_seconds": 10.0,
            "fps": 25.0,
            "width": 1280,
            "height": 720,
        }
    )
    monkeypatch.setattr(video, "probe_video", lambda _path: probe)

    def fill_budget(command, *, timeout):
        assert timeout == 180
        Path(command[-1]).write_bytes(b"1234")
        return ToolResult.success()

    monkeypatch.setattr(video, "_run_ffmpeg", fill_budget)

    result = video.split_video(source, destination, max_output_bytes=4)

    assert not result.ok
    assert result.error and result.error["code"] == "SPLIT_OUTPUT_QUOTA_EXCEEDED"
    assert not destination.exists()
    assert not list(tmp_path.glob(".scenes.staging-*"))


def test_split_rejects_source_change_while_creating_private_snapshot(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "road.mp4"
    source.write_bytes(b"source-a")
    destination = tmp_path / "scenes"
    initial_hash = _hash(source)
    probe = ToolResult.success(
        {
            "sha256": initial_hash,
            "duration_seconds": 10.0,
            "fps": 25.0,
            "width": 1280,
            "height": 720,
        }
    )
    monkeypatch.setattr(video, "probe_video", lambda _path: probe)

    def changed_copy(_source: Path, snapshot: Path) -> str:
        snapshot.write_bytes(b"source-b")
        return _hash(snapshot)

    monkeypatch.setattr(video, "_copy_with_sha256", changed_copy)
    monkeypatch.setattr(
        video,
        "_run_ffmpeg",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("FFmpeg must use only a source snapshot matching the initial probe")
        ),
    )

    result = video.split_video(source, destination)

    assert not result.ok
    assert result.error and result.error["code"] == "SPLIT_SOURCE_CHANGED"
    assert result.retryable
    assert not destination.exists()
    assert not list(tmp_path.glob(".scenes.staging-*"))
