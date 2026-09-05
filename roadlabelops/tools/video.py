from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from ..models import ToolResult

SPLIT_MANIFEST_FILENAME = "split-manifest.json"
SPLIT_MANIFEST_SCHEMA_VERSION = "1.0.0"
DEFAULT_ABSOLUTE_MAX_VIDEO_DURATION_SECONDS = 2 * 60 * 60
DEFAULT_MAX_SCENE_COUNT = 720
DEFAULT_MAX_SPLIT_OUTPUT_BYTES = 8 * 1024 * 1024 * 1024
SPLIT_FREE_SPACE_HEADROOM_BYTES = 256 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_with_sha256(source: Path, destination: Path) -> str:
    digest = hashlib.sha256()
    with source.open("rb") as source_handle, destination.open("xb") as output_handle:
        for chunk in iter(lambda: source_handle.read(1024 * 1024), b""):
            digest.update(chunk)
            output_handle.write(chunk)
        output_handle.flush()
        os.fsync(output_handle.fileno())
    return digest.hexdigest()


def _positive_finite(value: object) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric) and numeric > 0


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def probe_video(video_path: Path | str) -> ToolResult:
    path = Path(video_path).resolve()
    if not path.is_file():
        return ToolResult.failure("VIDEO_NOT_FOUND", "The selected video does not exist")
    if shutil.which("ffprobe") is None:
        return ToolResult.failure("FFPROBE_MISSING", "Install FFmpeg before importing a real video")
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,codec_name:format=duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return ToolResult.failure(
            "VIDEO_PROBE_TIMEOUT",
            "ffprobe timed out while reading this video",
            retryable=True,
        )
    except OSError:
        return ToolResult.failure(
            "VIDEO_PROBE_FAILED",
            "ffprobe could not be started",
            retryable=True,
        )
    if completed.returncode != 0:
        return ToolResult.failure("VIDEO_INVALID", "ffprobe could not read this video")
    try:
        payload = json.loads(completed.stdout)
        streams = payload.get("streams", [])
        if not streams:
            raise ValueError("video stream missing")
        stream = streams[0]
        numerator, denominator = str(stream.get("r_frame_rate", "0/1")).split("/", 1)
        denominator_value = float(denominator)
        if denominator_value == 0:
            raise ValueError("invalid frame-rate denominator")
        fps = float(numerator) / denominator_value
        duration = float(payload["format"]["duration"])
        width = int(stream["width"])
        height = int(stream["height"])
        if not all((_positive_finite(duration), _positive_finite(fps), width > 0, height > 0)):
            raise ValueError("invalid video metadata")
        source_sha256 = _sha256(path)
    except (
        AttributeError,
        IndexError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        OSError,
    ):
        return ToolResult.failure("VIDEO_INVALID", "ffprobe returned invalid video metadata")
    return ToolResult.success(
        {
            "path": str(path),
            "sha256": source_sha256,
            "duration_seconds": duration,
            "fps": round(fps, 3),
            "width": width,
            "height": height,
            "codec": stream.get("codec_name", "unknown"),
        }
    )


def _run_ffmpeg(command: list[str], *, timeout: int) -> ToolResult:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return ToolResult.failure(
            "FFMPEG_TIMEOUT",
            "FFmpeg timed out while creating scene media",
            retryable=True,
        )
    except OSError:
        return ToolResult.failure(
            "FFMPEG_FAILED",
            "FFmpeg could not be started",
            retryable=True,
        )
    if completed.returncode != 0:
        return ToolResult.failure(
            "FFMPEG_FAILED",
            "FFmpeg could not create scene media",
            retryable=True,
        )
    return ToolResult.success()


def _regular_nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and not path.is_symlink() and path.stat().st_size > 0
    except OSError:
        return False


def _existing_output_failure(destination: Path) -> ToolResult:
    return ToolResult.failure(
        "SPLIT_OUTPUT_EXISTS",
        (
            f"Scene output already exists at {destination}; it was not reused because "
            "its completeness cannot be trusted"
        ),
    )


def _split_output_failure(code: str, message: str) -> ToolResult:
    return ToolResult.failure(code, message, retryable=False)


def _write_json_durable(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())


def verify_split_output(
    output_dir: Path | str,
    *,
    source_sha256: str,
    duration_seconds: float,
    fps: float,
    width: int,
    height: int,
    scene_seconds: int,
    frame_step: int,
    max_scene_count: int = DEFAULT_MAX_SCENE_COUNT,
    max_output_bytes: int = DEFAULT_MAX_SPLIT_OUTPUT_BYTES,
) -> ToolResult:
    """Validate a published split before adopting it after an interrupted ingest.

    The manifest binds the current source hash and split parameters to every
    expected scene and thumbnail hash. Symbolic links and unmanifested entries
    fail before any manifest or media file is read.
    """

    root = Path(output_dir)
    if root.is_symlink():
        return _split_output_failure(
            "SPLIT_OUTPUT_INVALID", "Scene output directory must not be a symbolic link"
        )
    if not root.is_dir():
        return _split_output_failure(
            "SPLIT_OUTPUT_MISSING", "Published scene output directory does not exist"
        )
    try:
        descendants = list(root.rglob("*"))
    except OSError:
        return _split_output_failure(
            "SPLIT_OUTPUT_INVALID", "Published scene output cannot be inspected"
        )
    for path in descendants:
        if path.is_symlink():
            return _split_output_failure(
                "SPLIT_OUTPUT_INVALID",
                f"Published scene output contains a symbolic link: {path.relative_to(root)}",
            )

    manifest_path = root / SPLIT_MANIFEST_FILENAME
    if not _regular_nonempty_file(manifest_path):
        return _split_output_failure(
            "SPLIT_MANIFEST_MISSING", "Published scene output has no trusted split manifest"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _split_output_failure(
            "SPLIT_MANIFEST_INVALID", "Published split manifest is not valid JSON"
        )
    expected_manifest_keys = {
        "schema_version",
        "source",
        "scene_seconds",
        "frame_step",
        "scene_count",
        "scenes",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected_manifest_keys:
        return _split_output_failure(
            "SPLIT_MANIFEST_INVALID", "Published split manifest has an invalid schema"
        )
    source = manifest.get("source")
    if not isinstance(source, dict) or set(source) != {
        "sha256",
        "duration_seconds",
        "fps",
        "width",
        "height",
    }:
        return _split_output_failure(
            "SPLIT_MANIFEST_INVALID", "Published split source lineage is invalid"
        )
    numeric_source = {
        "duration_seconds": duration_seconds,
        "fps": fps,
    }
    if (
        source.get("sha256") != source_sha256
        or source.get("width") != width
        or source.get("height") != height
        or any(
            isinstance(source.get(key), bool)
            or not isinstance(source.get(key), (int, float))
            or not math.isclose(float(source[key]), expected, abs_tol=1e-6)
            for key, expected in numeric_source.items()
        )
        or manifest.get("scene_seconds") != scene_seconds
        or manifest.get("frame_step") != frame_step
    ):
        return _split_output_failure(
            "SPLIT_LINEAGE_MISMATCH",
            "Published scene output does not match the current source and split options",
        )

    if max_output_bytes <= 0:
        return _split_output_failure(
            "INVALID_SPLIT_OPTIONS", "Split output byte limit must be positive"
        )
    expected_scene_count = max(1, math.ceil(duration_seconds / scene_seconds))
    raw_scenes = manifest.get("scenes")
    if (
        expected_scene_count > max_scene_count
        or isinstance(manifest.get("scene_count"), bool)
        or manifest.get("scene_count") != expected_scene_count
        or not isinstance(raw_scenes, list)
        or len(raw_scenes) != expected_scene_count
    ):
        return _split_output_failure(
            "SPLIT_MANIFEST_INVALID", "Published split manifest has an invalid scene count"
        )

    expected_entries = {SPLIT_MANIFEST_FILENAME}
    scenes: list[dict[str, Any]] = []
    scene_keys = {
        "index",
        "start_seconds",
        "end_seconds",
        "video_file",
        "video_sha256",
        "thumbnail_file",
        "thumbnail_sha256",
    }
    for expected_index, item in enumerate(raw_scenes, start=1):
        if not isinstance(item, dict) or set(item) != scene_keys:
            return _split_output_failure(
                "SPLIT_MANIFEST_INVALID", "Published split scene record has an invalid schema"
            )
        expected_video = f"scene_{expected_index:03d}.mp4"
        expected_thumbnail = f"scene_{expected_index:03d}.jpg"
        expected_start = (expected_index - 1) * scene_seconds
        expected_end = min(expected_index * scene_seconds, duration_seconds)
        start = item.get("start_seconds")
        end = item.get("end_seconds")
        if (
            item.get("index") != expected_index
            or item.get("video_file") != expected_video
            or item.get("thumbnail_file") != expected_thumbnail
            or isinstance(start, bool)
            or not isinstance(start, (int, float))
            or isinstance(end, bool)
            or not isinstance(end, (int, float))
            or not math.isclose(float(start), expected_start, abs_tol=1e-6)
            or not math.isclose(float(end), expected_end, abs_tol=1e-6)
            or not _valid_sha256(item.get("video_sha256"))
            or not _valid_sha256(item.get("thumbnail_sha256"))
        ):
            return _split_output_failure(
                "SPLIT_MANIFEST_INVALID", "Published split scene lineage is invalid"
            )
        expected_entries.update({expected_video, expected_thumbnail})
        scenes.append(
            {
                "index": expected_index,
                "start_seconds": float(start),
                "end_seconds": float(end),
                "video_path": str(root / expected_video),
                "thumbnail_path": str(root / expected_thumbnail),
                "video_sha256": str(item["video_sha256"]),
                "frame_step": frame_step,
            }
        )

    actual_entries = {path.relative_to(root).as_posix() for path in descendants}
    if actual_entries != expected_entries:
        return _split_output_failure(
            "SPLIT_FILE_SET_INVALID",
            "Published scene output has missing or unmanifested entries",
        )

    try:
        total_output_bytes = sum(
            path.stat().st_size
            for path in descendants
            if path.name != SPLIT_MANIFEST_FILENAME
        )
    except OSError:
        return _split_output_failure(
            "SPLIT_FILE_INVALID", "Published scene output size cannot be inspected"
        )
    if total_output_bytes > max_output_bytes:
        return _split_output_failure(
            "SPLIT_OUTPUT_QUOTA_EXCEEDED",
            f"Published scene output exceeds the {max_output_bytes} byte safety limit",
        )

    for scene, record in zip(scenes, raw_scenes, strict=True):
        video_path = Path(scene["video_path"])
        thumbnail_path = Path(scene["thumbnail_path"])
        if not _regular_nonempty_file(video_path) or not _regular_nonempty_file(thumbnail_path):
            return _split_output_failure(
                "SPLIT_FILE_INVALID", "A published scene or thumbnail is missing or empty"
            )
        scene_probe = probe_video(video_path)
        if not scene_probe.ok:
            return _split_output_failure(
                "SPLIT_FILE_INVALID", "A published scene is not a valid video"
            )
        expected_duration = float(scene["end_seconds"]) - float(scene["start_seconds"])
        tolerance = max(0.5, 2 / fps)
        if (
            scene_probe.data.get("sha256") != record["video_sha256"]
            or scene_probe.data.get("width") != width
            or scene_probe.data.get("height") != height
            or abs(float(scene_probe.data.get("duration_seconds", 0)) - expected_duration)
            > tolerance
            or _sha256(thumbnail_path) != record["thumbnail_sha256"]
        ):
            return _split_output_failure(
                "SPLIT_HASH_MISMATCH",
                "Published scene output does not match its split manifest",
            )

    return ToolResult.success(
        {
            "scenes": scenes,
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "output_bytes": total_output_bytes,
        },
        side_effects=[str(root)],
        metrics={"scene_count": len(scenes)},
    )


def split_video(
    video_path: Path | str,
    output_dir: Path | str,
    scene_seconds: int = 15,
    frame_step: int = 5,
    absolute_max_duration_seconds: float = DEFAULT_ABSOLUTE_MAX_VIDEO_DURATION_SECONDS,
    max_scene_count: int = DEFAULT_MAX_SCENE_COUNT,
    max_output_bytes: int = DEFAULT_MAX_SPLIT_OUTPUT_BYTES,
) -> ToolResult:
    if (
        scene_seconds <= 0
        or frame_step <= 0
        or absolute_max_duration_seconds <= 0
        or max_scene_count <= 0
        or max_output_bytes <= 0
    ):
        return ToolResult.failure(
            "INVALID_SPLIT_OPTIONS",
            "Split durations, frame step, and safety limits must all be positive",
        )
    probe = probe_video(video_path)
    if not probe.ok:
        return probe
    source = Path(video_path).resolve()
    requested_destination = Path(output_dir)
    if requested_destination.is_symlink():
        return ToolResult.failure(
            "SPLIT_OUTPUT_INVALID",
            "Scene output directory must not be a symbolic link",
        )
    destination = requested_destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        return _existing_output_failure(destination)

    duration_seconds = float(probe.data["duration_seconds"])
    source_fps = float(probe.data["fps"])
    if duration_seconds > absolute_max_duration_seconds:
        return ToolResult.failure(
            "VIDEO_DURATION_HARD_LIMIT",
            (
                f"Video duration {duration_seconds:.1f}s exceeds the absolute "
                f"{absolute_max_duration_seconds:.1f}s safety limit"
            ),
        )
    scene_count = max(1, math.ceil(duration_seconds / scene_seconds))
    if scene_count > max_scene_count:
        return ToolResult.failure(
            "VIDEO_SCENE_COUNT_LIMIT",
            f"Video would create {scene_count} scenes, above the {max_scene_count} scene limit",
        )
    try:
        source_size = source.stat().st_size
        estimated_transcode_bytes = max(
            source_size * 4,
            math.ceil(
                float(probe.data["width"])
                * float(probe.data["height"])
                * source_fps
                * duration_seconds
                * 0.2
                / 8
            ),
        )
        required_free_bytes = source_size + min(max_output_bytes, estimated_transcode_bytes)
        available_bytes = shutil.disk_usage(destination.parent).free
    except OSError:
        return ToolResult.failure(
            "SPLIT_STORAGE_CHECK_FAILED",
            "Available storage could not be checked before splitting",
            retryable=True,
        )
    if available_bytes < required_free_bytes + SPLIT_FREE_SPACE_HEADROOM_BYTES:
        return ToolResult.failure(
            "SPLIT_STORAGE_INSUFFICIENT",
            "There is not enough free storage for the bounded scene output",
            retryable=True,
        )
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-",
            dir=destination.parent,
        )
    )
    staged_scenes: list[dict[str, Any]] = []
    staged_output_bytes = 0
    try:
        source_snapshot = staging / ".source-snapshot"
        try:
            snapshot_sha256 = _copy_with_sha256(source, source_snapshot)
        except OSError:
            return ToolResult.failure(
                "SPLIT_SOURCE_SNAPSHOT_FAILED",
                "Source video could not be copied into private split staging",
                retryable=True,
            )
        if snapshot_sha256 != str(probe.data["sha256"]):
            return ToolResult.failure(
                "SPLIT_SOURCE_CHANGED",
                "Source video changed after probing; retry with a stable source file",
                retryable=True,
            )
        for index in range(scene_count):
            start = index * scene_seconds
            duration = min(scene_seconds, duration_seconds - start)
            output = staging / f"scene_{index + 1:03d}.mp4"
            thumb = staging / f"scene_{index + 1:03d}.jpg"
            remaining_bytes = max_output_bytes - staged_output_bytes
            if remaining_bytes <= 0:
                return ToolResult.failure(
                    "SPLIT_OUTPUT_QUOTA_EXCEEDED",
                    f"Scene output exceeds the {max_output_bytes} byte safety limit",
                )
            cut = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                str(start),
                "-i",
                str(source_snapshot),
                "-t",
                str(duration),
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-fs",
                str(remaining_bytes),
                "-y",
                str(output),
            ]
            cut_result = _run_ffmpeg(cut, timeout=180)
            if not cut_result.ok:
                if output.exists() and output.stat().st_size >= remaining_bytes:
                    return ToolResult.failure(
                        "SPLIT_OUTPUT_QUOTA_EXCEEDED",
                        f"Scene output exceeds the {max_output_bytes} byte safety limit",
                    )
                return ToolResult.failure(
                    "SPLIT_FAILED",
                    f"Scene {index + 1} could not be created: {cut_result.error['message']}",
                    retryable=cut_result.retryable,
                )
            if not _regular_nonempty_file(output):
                return ToolResult.failure(
                    "SPLIT_VALIDATION_FAILED",
                    f"Scene {index + 1} output is missing or empty",
                    retryable=True,
                )
            staged_output_bytes += output.stat().st_size
            if staged_output_bytes >= max_output_bytes:
                return ToolResult.failure(
                    "SPLIT_OUTPUT_QUOTA_EXCEEDED",
                    f"Scene output exceeds the {max_output_bytes} byte safety limit",
                )

            scene_probe = probe_video(output)
            if not scene_probe.ok:
                return ToolResult.failure(
                    "SPLIT_VALIDATION_FAILED",
                    f"Scene {index + 1} output could not be validated",
                    retryable=True,
                )
            actual_duration = float(scene_probe.data["duration_seconds"])
            tolerance = max(0.5, 2 / source_fps)
            if abs(actual_duration - duration) > tolerance:
                return ToolResult.failure(
                    "SPLIT_VALIDATION_FAILED",
                    (
                        f"Scene {index + 1} duration {actual_duration:.3f}s does not match "
                        f"the expected {duration:.3f}s"
                    ),
                    retryable=True,
                )

            thumbnail = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                str(start + min(1, duration / 2)),
                "-i",
                str(source_snapshot),
                "-frames:v",
                "1",
                "-vf",
                "scale=960:-2",
                "-fs",
                str(max_output_bytes - staged_output_bytes),
                "-y",
                str(thumb),
            ]
            if max_output_bytes - staged_output_bytes <= 0:
                return ToolResult.failure(
                    "SPLIT_OUTPUT_QUOTA_EXCEEDED",
                    f"Scene output exceeds the {max_output_bytes} byte safety limit",
                )
            thumbnail_result = _run_ffmpeg(thumbnail, timeout=60)
            if not thumbnail_result.ok or not _regular_nonempty_file(thumb):
                if (
                    thumb.exists()
                    and thumb.stat().st_size >= max_output_bytes - staged_output_bytes
                ):
                    return ToolResult.failure(
                        "SPLIT_OUTPUT_QUOTA_EXCEEDED",
                        f"Scene output exceeds the {max_output_bytes} byte safety limit",
                    )
                return ToolResult.failure(
                    "THUMBNAIL_FAILED",
                    f"Scene {index + 1} thumbnail could not be created",
                    retryable=True,
                )
            staged_output_bytes += thumb.stat().st_size
            if staged_output_bytes > max_output_bytes:
                return ToolResult.failure(
                    "SPLIT_OUTPUT_QUOTA_EXCEEDED",
                    f"Scene output exceeds the {max_output_bytes} byte safety limit",
                )
            staged_scenes.append(
                {
                    "index": index + 1,
                    "start_seconds": start,
                    "end_seconds": start + duration,
                    "video_file": output.name,
                    "video_sha256": str(scene_probe.data["sha256"]),
                    "thumbnail_file": thumb.name,
                    "thumbnail_sha256": _sha256(thumb),
                }
            )

        try:
            source_snapshot.unlink()
        except OSError:
            return ToolResult.failure(
                "SPLIT_SOURCE_SNAPSHOT_FAILED",
                "Private source snapshot could not be removed before publication",
                retryable=True,
            )

        split_manifest = {
            "schema_version": SPLIT_MANIFEST_SCHEMA_VERSION,
            "source": {
                "sha256": str(probe.data["sha256"]),
                "duration_seconds": duration_seconds,
                "fps": source_fps,
                "width": int(probe.data["width"]),
                "height": int(probe.data["height"]),
            },
            "scene_seconds": scene_seconds,
            "frame_step": frame_step,
            "scene_count": scene_count,
            "scenes": staged_scenes,
        }
        _write_json_durable(staging / SPLIT_MANIFEST_FILENAME, split_manifest)

        lock_path = destination.parent / f".{destination.name}.split.lock"
        try:
            with lock_path.open("a+b") as lock_handle:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                if destination.exists() or destination.is_symlink():
                    return _existing_output_failure(destination)
                staging.rename(destination)
        except OSError:
            return ToolResult.failure(
                "SPLIT_PUBLISH_FAILED",
                "Validated scene outputs could not be published atomically",
                retryable=True,
            )

        return verify_split_output(
            destination,
            source_sha256=str(probe.data["sha256"]),
            duration_seconds=duration_seconds,
            fps=source_fps,
            width=int(probe.data["width"]),
            height=int(probe.data["height"]),
            scene_seconds=scene_seconds,
            frame_step=frame_step,
            max_scene_count=max_scene_count,
            max_output_bytes=max_output_bytes,
        )
    finally:
        if staging.exists():
            shutil.rmtree(staging)
