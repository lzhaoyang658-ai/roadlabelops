import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import roadlabelops.ingest as ingest_module
from roadlabelops.ingest import ingest_video
from roadlabelops.models import ToolResult
from roadlabelops.storage import LocalStore
from roadlabelops.tools import video


def _probe(*, duration: float = 15, fps: float = 25, width: int = 1280, height: int = 720):
    return ToolResult.success({
        "sha256": "a" * 64,
        "duration_seconds": duration,
        "fps": fps,
        "width": width,
        "height": height,
    })


def test_ingest_is_idempotent_by_source_hash(tmp_path: Path, monkeypatch) -> None:
    store = LocalStore(tmp_path / "data", tmp_path / "runtime")
    source = tmp_path / "road.mp4"
    source.write_bytes(b"video")
    probe = _probe()
    destination = store.scenes_dir / "session_aaaaaaaaaa"
    split = ToolResult.success({
        "scenes": [{
            "index": 1,
            "start_seconds": 0,
            "end_seconds": 15,
            "video_path": str(destination / "scene_001.mp4"),
            "thumbnail_path": str(destination / "scene_001.jpg"),
            "video_sha256": "b" * 64,
        }]
    })
    monkeypatch.setattr("roadlabelops.ingest.probe_video", lambda _: probe)
    monkeypatch.setattr("roadlabelops.ingest.split_video", lambda *args, **kwargs: split)
    monkeypatch.setattr("roadlabelops.ingest.verify_split_output", lambda *args, **kwargs: split)

    first = ingest_video(store, source)
    second = ingest_video(store, source)

    assert first.ok and not first.data["existing"]
    assert second.ok and second.data["existing"]
    assert len(store.list_sessions()) == 1
    assert store.list_sessions()[0].scenes[0].video_sha256 == "b" * 64


def test_duplicate_ingest_repairs_a_missing_recorded_source_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = LocalStore(tmp_path / "data", tmp_path / "runtime")
    original = tmp_path / "original.mp4"
    replacement = tmp_path / "replacement.mp4"
    original.write_bytes(b"same-video")
    replacement.write_bytes(b"same-video")
    destination = store.scenes_dir / "session_aaaaaaaaaa"
    split = ToolResult.success(
        {
            "scenes": [
                {
                    "index": 1,
                    "start_seconds": 0,
                    "end_seconds": 15,
                    "video_path": str(destination / "scene_001.mp4"),
                    "thumbnail_path": str(destination / "scene_001.jpg"),
                    "video_sha256": "b" * 64,
                }
            ]
        }
    )

    def fake_probe(path_value):
        if not Path(path_value).is_file():
            return ToolResult.failure("VIDEO_NOT_FOUND", "missing")
        return _probe()

    monkeypatch.setattr(ingest_module, "probe_video", fake_probe)
    monkeypatch.setattr(ingest_module, "split_video", lambda *_a, **_k: split)
    monkeypatch.setattr(ingest_module, "verify_split_output", lambda *_a, **_k: split)

    created = ingest_video(store, original)
    assert created.ok
    original.unlink()
    recovered = ingest_video(store, replacement)

    assert recovered.ok
    assert recovered.data["existing"] is True
    assert recovered.data["source_recovered"] is True
    assert store.get_session("session_aaaaaaaaaa").source_path == str(replacement.resolve())


def test_ingest_requires_explicit_confirmation_above_default_duration(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = LocalStore(tmp_path / "data", tmp_path / "runtime")
    source = tmp_path / "road.mp4"
    source.write_bytes(b"video")
    monkeypatch.setattr("roadlabelops.ingest.probe_video", lambda _: _probe(duration=301))
    split_called = False

    def fake_split(*_args):
        nonlocal split_called
        split_called = True
        return ToolResult.failure("UNEXPECTED", "split should not run")

    monkeypatch.setattr("roadlabelops.ingest.split_video", fake_split)

    result = ingest_video(store, source, max_duration_seconds=300)

    assert not result.ok
    assert result.error["code"] == "VIDEO_TOO_LONG"
    assert not split_called


def test_ingest_accepts_confirmed_long_video_and_persists_scene_hash(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = LocalStore(tmp_path / "data", tmp_path / "runtime")
    source = tmp_path / "road.mp4"
    source.write_bytes(b"video")
    monkeypatch.setattr("roadlabelops.ingest.probe_video", lambda _: _probe(duration=301))
    monkeypatch.setattr(
        "roadlabelops.ingest.split_video",
        lambda *_args, **_kwargs: ToolResult.success({
            "scenes": [{
                "index": 1,
                "start_seconds": 0,
                "end_seconds": 301,
                "video_path": str(
                    store.scenes_dir / "session_aaaaaaaaaa" / "scene_001.mp4"
                ),
                "thumbnail_path": str(
                    store.scenes_dir / "session_aaaaaaaaaa" / "scene_001.jpg"
                ),
                "video_sha256": "b" * 64,
            }]
        }),
    )
    monkeypatch.setattr(
        "roadlabelops.ingest.verify_split_output",
        lambda *_args, **_kwargs: ToolResult.success(
            {
                "scenes": [
                    {
                        "index": 1,
                        "start_seconds": 0,
                        "end_seconds": 301,
                        "video_path": str(
                            store.scenes_dir / "session_aaaaaaaaaa" / "scene_001.mp4"
                        ),
                        "thumbnail_path": str(
                            store.scenes_dir / "session_aaaaaaaaaa" / "scene_001.jpg"
                        ),
                        "video_sha256": "b" * 64,
                    }
                ]
            }
        ),
    )

    result = ingest_video(
        store,
        source,
        max_duration_seconds=300,
        allow_long_video=True,
        scene_seconds=301,
    )

    assert result.ok
    assert result.data["session"]["scenes"][0]["video_sha256"] == "b" * 64
    assert store.list_sessions()[0].scenes[0].video_sha256 == "b" * 64


def test_ingest_rejects_excessive_resolution_before_split(tmp_path: Path, monkeypatch) -> None:
    store = LocalStore(tmp_path / "data", tmp_path / "runtime")
    source = tmp_path / "road.mp4"
    source.write_bytes(b"video")
    monkeypatch.setattr("roadlabelops.ingest.probe_video", lambda _: _probe(width=4096))

    result = ingest_video(store, source, max_width=3840, max_height=2160)

    assert not result.ok
    assert result.error["code"] == "VIDEO_RESOLUTION_TOO_HIGH"


def test_ingest_rejects_excessive_frame_rate_before_split(tmp_path: Path, monkeypatch) -> None:
    store = LocalStore(tmp_path / "data", tmp_path / "runtime")
    source = tmp_path / "road.mp4"
    source.write_bytes(b"video")
    monkeypatch.setattr("roadlabelops.ingest.probe_video", lambda _: _probe(fps=120))

    result = ingest_video(store, source, max_fps=60)

    assert not result.ok
    assert result.error["code"] == "VIDEO_FPS_TOO_HIGH"


def _install_fake_split_pipeline(monkeypatch, source: Path) -> None:
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()

    def fake_probe(path_value):
        path = Path(path_value).resolve()
        if path == source.resolve():
            digest = source_hash
        else:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return ToolResult.success(
            {
                "path": str(path),
                "sha256": digest,
                "duration_seconds": 15.0,
                "fps": 25.0,
                "width": 1280,
                "height": 720,
                "codec": "h264",
            }
        )

    def fake_ffmpeg(command, *, timeout):
        assert timeout in {60, 180}
        Path(command[-1]).write_bytes(Path(command[-1]).name.encode())
        return ToolResult.success()

    monkeypatch.setattr(ingest_module, "probe_video", fake_probe)
    monkeypatch.setattr(video, "probe_video", fake_probe)
    monkeypatch.setattr(video, "_run_ffmpeg", fake_ffmpeg)


def test_ingest_adopts_cryptographically_complete_split_after_commit_crash(
    tmp_path: Path, monkeypatch
) -> None:
    store = LocalStore(tmp_path / "data", tmp_path / "runtime")
    source = tmp_path / "road.mp4"
    source.write_bytes(b"source-video")
    _install_fake_split_pipeline(monkeypatch, source)
    original_save = store.save_session
    crashed = False

    def crash_once(session):
        nonlocal crashed
        if not crashed:
            crashed = True
            raise OSError("simulated process stop before Session commit")
        return original_save(session)

    monkeypatch.setattr(store, "save_session", crash_once)
    with pytest.raises(OSError, match="before Session commit"):
        ingest_video(store, source)
    destination = store.scenes_dir / f"session_{hashlib.sha256(source.read_bytes()).hexdigest()[:10]}"
    assert (destination / video.SPLIT_MANIFEST_FILENAME).is_file()
    assert store.list_sessions() == []

    monkeypatch.setattr(store, "save_session", original_save)
    recovered = ingest_video(store, source)

    assert recovered.ok, recovered.error
    assert recovered.data["recovered"] is True
    assert recovered.data["session"]["scenes"][0]["video_sha256"]
    assert len(store.list_sessions()) == 1


def test_ingest_reverifies_fresh_split_against_the_initial_source_probe(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = LocalStore(tmp_path / "data", tmp_path / "runtime")
    source = tmp_path / "road.mp4"
    source.write_bytes(b"source-a")
    destination = store.scenes_dir / "session_aaaaaaaaaa"
    monkeypatch.setattr(ingest_module, "probe_video", lambda _path: _probe())
    monkeypatch.setattr(
        ingest_module,
        "split_video",
        lambda *_args, **_kwargs: ToolResult.success(
            {"scenes": [{"video_sha256": "b" * 64}]}
        ),
    )
    observed_source_hashes: list[str] = []

    def reject_changed_split(_destination, **kwargs):
        observed_source_hashes.append(kwargs["source_sha256"])
        return ToolResult.failure(
            "SPLIT_LINEAGE_MISMATCH",
            "published split belongs to a changed source",
        )

    monkeypatch.setattr(ingest_module, "verify_split_output", reject_changed_split)

    result = ingest_video(store, source)

    assert not result.ok
    assert result.error and result.error["code"] == "SPLIT_LINEAGE_MISMATCH"
    assert observed_source_hashes == ["a" * 64]
    assert store.list_sessions() == []
    assert destination.parent == store.scenes_dir


def test_ingest_journals_planned_scene_counts_for_split_failures(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = LocalStore(tmp_path / "data", tmp_path / "runtime")
    source = tmp_path / "road.mp4"
    source.write_bytes(b"video")
    monkeypatch.setattr(ingest_module, "probe_video", lambda _path: _probe(duration=31))
    monkeypatch.setattr(
        ingest_module,
        "split_video",
        lambda *_args, **_kwargs: ToolResult.failure(
            "VIDEO_SPLIT_FAILED",
            "ffmpeg failed",
            retryable=True,
        ),
    )

    result = ingest_video(store, source, scene_seconds=15)

    assert not result.ok
    assert result.metrics == {
        "planned_scene_count": 3,
        "successful_scene_count": 0,
    }
    completion = store.read_journal(None)[-1]
    assert completion["event"] == "ingest.completed"
    assert completion["success"] is False
    assert completion["probe_succeeded"] is True
    assert completion["session_id"] == "session_aaaaaaaaaa"
    assert completion["planned_scene_count"] == 3
    assert completion["successful_scene_count"] == 0
    assert completion["error_code"] == "VIDEO_SPLIT_FAILED"


def test_ingest_journals_probe_failures_without_a_scene_denominator(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = LocalStore(tmp_path / "data", tmp_path / "runtime")
    source = tmp_path / "broken.mp4"
    source.write_bytes(b"not-video")
    monkeypatch.setattr(
        ingest_module,
        "probe_video",
        lambda _path: ToolResult.failure("VIDEO_PROBE_FAILED", "ffprobe failed"),
    )

    result = ingest_video(store, source)

    assert not result.ok
    completion = store.read_journal(None)[-1]
    assert completion["event"] == "ingest.completed"
    assert completion["success"] is False
    assert completion["probe_succeeded"] is False
    assert completion["planned_scene_count"] is None
    assert completion["successful_scene_count"] == 0
    assert completion["session_id"] is None


def test_ingest_refuses_tampered_published_split_during_recovery(
    tmp_path: Path, monkeypatch
) -> None:
    store = LocalStore(tmp_path / "data", tmp_path / "runtime")
    source = tmp_path / "road.mp4"
    source.write_bytes(b"source-video")
    _install_fake_split_pipeline(monkeypatch, source)
    monkeypatch.setattr(store, "save_session", lambda _session: (_ for _ in ()).throw(OSError()))
    with pytest.raises(OSError):
        ingest_video(store, source)
    destination = store.scenes_dir / f"session_{hashlib.sha256(source.read_bytes()).hexdigest()[:10]}"
    (destination / "scene_001.mp4").write_bytes(b"tampered")

    result = ingest_video(store, source)

    assert not result.ok
    assert result.error and result.error["code"] == "SPLIT_HASH_MISMATCH"
    assert store.list_sessions() == []


def test_ingest_absolute_limits_cannot_be_bypassed_by_confirmation(
    tmp_path: Path, monkeypatch
) -> None:
    store = LocalStore(tmp_path / "data", tmp_path / "runtime")
    source = tmp_path / "road.mp4"
    source.write_bytes(b"video")
    monkeypatch.setattr(ingest_module, "probe_video", lambda _path: _probe(duration=601))
    split_called = False

    def unexpected_split(*_args, **_kwargs):
        nonlocal split_called
        split_called = True
        return ToolResult.success()

    monkeypatch.setattr(ingest_module, "split_video", unexpected_split)
    result = ingest_video(
        store,
        source,
        allow_long_video=True,
        absolute_max_duration_seconds=600,
    )

    assert not result.ok
    assert result.error and result.error["code"] == "VIDEO_DURATION_HARD_LIMIT"
    assert not split_called


def test_ingest_serializes_concurrent_uploads_for_the_same_source(
    tmp_path: Path, monkeypatch
) -> None:
    store = LocalStore(tmp_path / "data", tmp_path / "runtime")
    source = tmp_path / "road.mp4"
    source.write_bytes(b"video")
    probe = _probe()
    destination = store.scenes_dir / "session_aaaaaaaaaa"
    split = ToolResult.success(
        {
            "scenes": [
                {
                    "index": 1,
                    "start_seconds": 0,
                    "end_seconds": 15,
                    "video_path": str(destination / "scene_001.mp4"),
                    "thumbnail_path": str(destination / "scene_001.jpg"),
                    "video_sha256": "b" * 64,
                }
            ]
        }
    )
    calls = 0

    def fake_split(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return split

    monkeypatch.setattr(ingest_module, "probe_video", lambda _path: probe)
    monkeypatch.setattr(ingest_module, "split_video", fake_split)
    monkeypatch.setattr(ingest_module, "verify_split_output", lambda *_a, **_k: split)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: ingest_video(store, source), range(2)))

    assert all(result.ok for result in results)
    assert calls == 1
    assert sum(bool(result.data["existing"]) for result in results) == 1


def test_committed_ingest_survives_noncritical_journal_failure(
    tmp_path: Path, monkeypatch
) -> None:
    store = LocalStore(tmp_path / "data", tmp_path / "runtime")
    source = tmp_path / "road.mp4"
    source.write_bytes(b"video")
    destination = store.scenes_dir / "session_aaaaaaaaaa"
    split = ToolResult.success(
        {
            "scenes": [
                {
                    "index": 1,
                    "start_seconds": 0,
                    "end_seconds": 15,
                    "video_path": str(destination / "scene_001.mp4"),
                    "thumbnail_path": str(destination / "scene_001.jpg"),
                    "video_sha256": "b" * 64,
                }
            ]
        }
    )
    monkeypatch.setattr(ingest_module, "probe_video", lambda _path: _probe())
    monkeypatch.setattr(ingest_module, "split_video", lambda *_a, **_k: split)
    monkeypatch.setattr(ingest_module, "verify_split_output", lambda *_a, **_k: split)
    monkeypatch.setattr(
        store,
        "append_journal",
        lambda _event: (_ for _ in ()).throw(RuntimeError("journal unavailable")),
    )

    result = ingest_video(store, source)

    assert result.ok
    assert result.data["journal_persisted"] is False
    assert store.get_session("session_aaaaaaaaaa").source_path == str(source.resolve())
