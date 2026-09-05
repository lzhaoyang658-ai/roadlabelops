"""Build an immutable v2 source-frame map from sampled scene videos.

Unlike the compiled-video mapper, this builder records one normalized asset
per scene.  A sampled ``source_frame`` is therefore already the normalized
asset frame; no FPS or concatenation arithmetic is inferred.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePath
from typing import Any, BinaryIO

import cv2

SOURCE_MAP_SCHEMA = {"name": "roadlabelops.training-source-frame-map", "version": 2}


class SceneSourceMapError(ValueError):
    """Raised when sampled scene evidence is incomplete or ambiguous."""


def _object(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SceneSourceMapError(f"{location} must be an object")
    return value


def _list(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise SceneSourceMapError(f"{location} must be a list")
    return value


def _text(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SceneSourceMapError(f"{location} must be a non-empty string")
    return value.strip()


def _integer(value: Any, location: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SceneSourceMapError(f"{location} must be an integer at least {minimum}")
    return value


def _path_without_final_symlink(path: Path, location: str) -> Path:
    """Return a workspace-contained path with no symlink in its relative chain."""

    workspace = Path.cwd().resolve()
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        relative = absolute.relative_to(workspace)
    except ValueError as error:
        raise SceneSourceMapError(f"{location} must be inside the workspace") from error
    if not relative.parts:
        raise SceneSourceMapError(f"{location} must be a regular file: {path}")
    current = workspace
    for part in relative.parts:
        current /= part
        try:
            component_stat = os.lstat(current)
        except OSError as error:
            raise SceneSourceMapError(f"{location} is unavailable: {error}") from error
        if stat.S_ISLNK(component_stat.st_mode):
            raise SceneSourceMapError(f"{location} must not contain a symlink: {path}")
    if not stat.S_ISREG(component_stat.st_mode):
        raise SceneSourceMapError(f"{location} must be a regular file: {path}")
    try:
        absolute.resolve(strict=True).relative_to(workspace)
    except (OSError, ValueError) as error:
        raise SceneSourceMapError(f"{location} must resolve inside the workspace") from error
    return absolute


def _open_regular_file(path: Path, location: str) -> BinaryIO:
    """Open a regular non-symlink leaf without a resolve/lstat race."""

    absolute = _path_without_final_symlink(path, location)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(absolute, flags)
    except OSError as error:
        raise SceneSourceMapError(f"could not open {location}: {error}") from error
    stream = os.fdopen(descriptor, "rb")
    opened_stat = os.fstat(stream.fileno())
    if not stat.S_ISREG(opened_stat.st_mode):
        stream.close()
        raise SceneSourceMapError(f"{location} must be a regular file: {path}")
    return stream


def _stable_file_bytes(path: Path, location: str) -> tuple[bytes, os.stat_result]:
    """Read once from an fd and reject mutation while the content is read."""

    with _open_regular_file(path, location) as stream:
        before = os.fstat(stream.fileno())
        encoded = stream.read()
        after = os.fstat(stream.fileno())
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after or len(encoded) != before.st_size:
        raise SceneSourceMapError(f"{location} changed while it was being read")
    return encoded, after


def _read_json(path: Path, location: str) -> tuple[dict[str, Any], str]:
    try:
        encoded, _ = _stable_file_bytes(path, location)
        payload = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SceneSourceMapError(f"could not read {location}: {error}") from error
    return _object(payload, location), hashlib.sha256(encoded).hexdigest()


def _stable_file_sha256(
    path: Path,
    location: str,
    *,
    expected_stat: os.stat_result | None = None,
) -> tuple[str, os.stat_result]:
    digest = hashlib.sha256()
    with _open_regular_file(path, location) as stream:
        before = os.fstat(stream.fileno())
        if expected_stat is not None and (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            expected_stat.st_dev,
            expected_stat.st_ino,
            expected_stat.st_size,
            expected_stat.st_mtime_ns,
            expected_stat.st_ctime_ns,
        ):
            raise SceneSourceMapError(f"{location} changed while video metadata was checked")
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        after = os.fstat(stream.fileno())
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise SceneSourceMapError(f"{location} changed while it was being hashed")
    return digest.hexdigest(), after


def _probe_video(
    path: Path,
    *,
    target_frames: Sequence[int],
    location: str,
) -> tuple[int, int]:
    """Read canonical metadata and prove every sampled frame is decodable."""

    absolute = _path_without_final_symlink(path, location)
    before = os.stat(absolute, follow_symlinks=False)
    capture = cv2.VideoCapture(str(absolute))
    try:
        if not capture.isOpened():
            raise SceneSourceMapError(f"{location} could not be opened as a video")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0 or frame_count <= 0:
            raise SceneSourceMapError(f"{location} has invalid fps/frame_count metadata")
        if not math.isclose(fps, 30.0, rel_tol=0.0, abs_tol=1e-3):
            raise SceneSourceMapError(f"{location} must be normalized to exactly 30 fps")
        if target_frames[-1] >= frame_count:
            raise SceneSourceMapError(
                f"{location} has {frame_count} frames but sample {target_frames[-1]} is required"
            )
        for target_frame in target_frames:
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame):
                raise SceneSourceMapError(
                    f"{location} could not seek to sampled frame {target_frame}"
                )
            decoded, _ = capture.read()
            if not decoded:
                raise SceneSourceMapError(
                    f"{location} could not decode sampled frame {target_frame}"
                )
    finally:
        capture.release()
    after = os.stat(absolute, follow_symlinks=False)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise SceneSourceMapError(f"{location} changed while video metadata was checked")
    return 30, frame_count


def _portable_path(path: Path, location: str) -> str:
    value = _path_without_final_symlink(path, location).relative_to(Path.cwd().resolve()).as_posix()
    candidate = PurePath(value)
    if value in {"", "."} or candidate.is_absolute() or ".." in candidate.parts:
        raise SceneSourceMapError(f"{location} must be a safe workspace-relative path")
    return value


def _atomic_write_json_new(output: Path, payload: Mapping[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(output):
        raise FileExistsError(f"output already exists: {output}")
    encoded = (
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError as error:
            raise FileExistsError(f"output already exists: {output}") from error
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build_scene_source_map(
    *,
    manifest_path: Path,
    scene_videos: Mapping[str, Path],
    output: Path,
    frame_step: int = 1,
) -> dict[str, Any]:
    """Validate scene assets and exclusively publish a v2 frame map."""

    frame_step = _integer(frame_step, "frame_step", minimum=1)
    manifest, manifest_sha256 = _read_json(manifest_path, "sample manifest")
    _text(manifest.get("session_id"), "sample manifest.session_id")
    _text(manifest.get("sampling_revision"), "sample manifest.sampling_revision")
    samples = _list(manifest.get("samples"), "sample manifest.samples")
    sample_size = _integer(manifest.get("sample_size"), "sample manifest.sample_size")
    if sample_size != len(samples) or not samples:
        raise SceneSourceMapError("sample manifest.sample_size is inconsistent or empty")

    frames: list[dict[str, Any]] = []
    sample_indices: set[int] = set()
    frame_keys: set[tuple[str, int]] = set()
    counts: Counter[str] = Counter()
    for index, raw_sample in enumerate(samples):
        sample = _object(raw_sample, f"sample manifest.samples[{index}]")
        sample_index = _integer(
            sample.get("sample_index"), f"sample manifest.samples[{index}].sample_index", minimum=1
        )
        scene_id = _text(sample.get("scene_id"), f"sample manifest.samples[{index}].scene_id")
        source_frame = _integer(
            sample.get("source_frame"), f"sample manifest.samples[{index}].source_frame"
        )
        key = (scene_id, source_frame)
        if sample_index in sample_indices or key in frame_keys:
            raise SceneSourceMapError("sample manifest contains duplicate sample/frame identity")
        sample_indices.add(sample_index)
        frame_keys.add(key)
        counts[scene_id] += 1
        frames.append(
            {
                "scene_id": scene_id,
                "source_frame": source_frame,
                "asset_id": scene_id,
                "normalized_asset_frame": source_frame,
            }
        )
    if sample_indices != set(range(1, sample_size + 1)):
        raise SceneSourceMapError("sample indices must exactly cover 1..sample_size")
    declared_counts = _object(
        manifest.get("sample_counts_by_scene"), "sample manifest.sample_counts_by_scene"
    )
    if declared_counts != dict(sorted(counts.items())):
        raise SceneSourceMapError("sample_counts_by_scene differs from manifest samples")

    expected_scenes = set(counts)
    supplied_scenes = set(scene_videos)
    if supplied_scenes != expected_scenes:
        raise SceneSourceMapError(
            "scene videos must exactly cover sampled scenes; "
            f"missing={sorted(expected_scenes - supplied_scenes)}, "
            f"extra={sorted(supplied_scenes - expected_scenes)}"
        )

    assets: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    leakage_by_scene: dict[str, str] = {}
    for scene_id in sorted(expected_scenes):
        requested_path = Path(scene_videos[scene_id])
        location = f"scene video {scene_id!r}"
        unreachable = sorted(
            frame
            for candidate_scene, frame in frame_keys
            if candidate_scene == scene_id and (frame + 1) % frame_step != 0
        )
        if unreachable:
            raise SceneSourceMapError(
                f"{location} has sampled frames unreachable at frame_step={frame_step}: "
                f"{unreachable}"
            )
        target_frames = sorted(
            frame for candidate_scene, frame in frame_keys if candidate_scene == scene_id
        )
        initial_stat = os.stat(
            _path_without_final_symlink(requested_path, location), follow_symlinks=False
        )
        fps, frame_count = _probe_video(
            requested_path,
            target_frames=target_frames,
            location=location,
        )
        digest, _ = _stable_file_sha256(
            requested_path,
            location,
            expected_stat=initial_stat,
        )
        if digest in hashes:
            raise SceneSourceMapError(
                f"scene videos {hashes[digest]!r} and {scene_id!r} have identical content"
            )
        hashes[digest] = scene_id
        leakage = f"sha256:{digest}"
        leakage_by_scene[scene_id] = leakage
        assets.append(
            {
                "asset_id": scene_id,
                "path": _portable_path(requested_path, f"scene video {scene_id!r}"),
                "sha256": digest,
                "fps": fps,
                "frame_count": frame_count,
                "leakage_group_id": leakage,
            }
        )
    for frame in frames:
        frame["leakage_group_id"] = leakage_by_scene[str(frame["scene_id"])]
    frames.sort(key=lambda item: (str(item["scene_id"]), int(item["source_frame"])))

    payload = {
        "schema": SOURCE_MAP_SCHEMA,
        "assets": assets,
        "frames": frames,
        "evidence": {
            "sample_manifest": {
                "path": _portable_path(manifest_path, "sample manifest"),
                "sha256": manifest_sha256,
            },
            "mapping": (
                "Each holdout scene MP4 is a content-addressed production-video asset; "
                "scene source frames are already normalized 30 fps frames."
            ),
        },
    }
    _atomic_write_json_new(output, payload)
    return {
        "output": str(output),
        "scene_count": len(assets),
        "mapped_frame_count": len(frames),
    }


def _scene_video(value: str) -> tuple[str, Path]:
    scene_id, separator, path = value.partition("=")
    if not separator or not scene_id.strip() or not path.strip():
        raise argparse.ArgumentTypeError("scene video must be SCENE_ID=PATH")
    return scene_id.strip(), Path(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--scene-video",
        required=True,
        action="append",
        type=_scene_video,
        metavar="SCENE_ID=PATH",
    )
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    scene_videos: dict[str, Path] = {}
    for scene_id, path in args.scene_video:
        if scene_id in scene_videos:
            parser.error(f"duplicate --scene-video scene ID: {scene_id}")
        scene_videos[scene_id] = path
    try:
        summary = build_scene_source_map(
            manifest_path=args.manifest,
            scene_videos=scene_videos,
            output=args.output,
            frame_step=args.frame_step,
        )
    except (FileExistsError, SceneSourceMapError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
