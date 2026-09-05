"""Build and preflight an immutable configured final-holdout protocol.

The builder derives the sorted evaluation universe from the holdout COCO,
content-binds every input, fixes the promotion gates, validates the complete
candidate/training/holdout contract with the evaluator, and only then
publishes the exact validated bytes with exclusive creation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from roadlabelops.holdout_policy import (
    FinalHoldoutConfigError,
    FinalHoldoutIdentity,
    resolve_final_holdout_identity,
)
from scripts.evaluate_frozen_holdout import (
    EXPECTED_GATES,
    PROTOCOL_SCHEMA,
    FrozenHoldoutError,
    _validate_candidate_freeze,
    _validate_settings,
    canonical_sha256,
    validate_protocol,
)

SOURCE_MAP_SCHEMA = {"name": "roadlabelops.training-source-frame-map", "version": 2}


class FrozenHoldoutProtocolBuildError(ValueError):
    """Raised when an input cannot form the configured final-holdout contract."""


DEFAULT_SETTINGS: dict[str, Any] = {
    "confidence": 0.4,
    "image_size": 640,
    "device": "mps",
    "nms_iou": 0.75,
    "rider_overlap": 0.25,
    "match_iou": 0.5,
}


def _object(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FrozenHoldoutProtocolBuildError(f"{location} must be an object")
    return value


def _list(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise FrozenHoldoutProtocolBuildError(f"{location} must be a list")
    return value


def _text(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FrozenHoldoutProtocolBuildError(f"{location} must be a non-empty string")
    return value.strip()


def _integer(value: Any, location: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise FrozenHoldoutProtocolBuildError(f"{location} must be an integer at least {minimum}")
    return value


def _leaf_absolute(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = Path.cwd() / expanded
    return expanded.parent.resolve() / expanded.name


def _stable_file_bytes(path: Path, location: str) -> tuple[bytes, os.stat_result]:
    """Read a non-symlink regular file and detect in-place mutation."""

    absolute = _leaf_absolute(path)
    try:
        leaf_stat = os.lstat(absolute)
    except OSError as error:
        raise FrozenHoldoutProtocolBuildError(f"{location} is unavailable: {error}") from error
    if stat.S_ISLNK(leaf_stat.st_mode) or not stat.S_ISREG(leaf_stat.st_mode):
        raise FrozenHoldoutProtocolBuildError(
            f"{location} must be a regular non-symlink file: {absolute}"
        )
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(absolute, flags)
    except OSError as error:
        raise FrozenHoldoutProtocolBuildError(f"could not open {location}: {error}") from error
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    encoded = b"".join(chunks)
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
        raise FrozenHoldoutProtocolBuildError(f"{location} changed while it was read")
    return encoded, after


def _read_json(path: Path, location: str) -> tuple[dict[str, Any], str, int]:
    encoded, details = _stable_file_bytes(path, location)
    try:
        payload = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FrozenHoldoutProtocolBuildError(f"could not decode {location}: {error}") from error
    return _object(payload, location), hashlib.sha256(encoded).hexdigest(), details.st_size


def _relative_path(path: Path, *, owner: Path) -> str:
    return os.path.relpath(_leaf_absolute(path), _leaf_absolute(owner).parent).replace(os.sep, "/")


def _workspace_regular_path(raw: str, location: str) -> Path:
    """Resolve a portable v2 asset path without allowing symlink traversal."""

    portable = PurePosixPath(raw)
    if (
        portable.is_absolute()
        or PureWindowsPath(raw).is_absolute()
        or "\\" in raw
        or ".." in portable.parts
        or not portable.name
    ):
        raise FrozenHoldoutProtocolBuildError(
            f"{location} must be a safe workspace-relative POSIX path"
        )
    workspace = Path.cwd().resolve()
    current = workspace
    for part in portable.parts:
        current /= part
        try:
            details = os.lstat(current)
        except OSError as error:
            raise FrozenHoldoutProtocolBuildError(f"{location} is unavailable: {error}") from error
        if stat.S_ISLNK(details.st_mode):
            raise FrozenHoldoutProtocolBuildError(f"{location} must not traverse a symlink")
    if not stat.S_ISREG(details.st_mode):
        raise FrozenHoldoutProtocolBuildError(f"{location} must name a regular file")
    try:
        current.resolve(strict=True).relative_to(workspace)
    except (OSError, ValueError) as error:
        raise FrozenHoldoutProtocolBuildError(
            f"{location} must resolve inside the workspace"
        ) from error
    return current


def _bound_file(path: Path, *, owner: Path, location: str) -> dict[str, Any]:
    encoded, details = _stable_file_bytes(path, location)
    return {
        "path": _relative_path(path, owner=owner),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "size_bytes": details.st_size,
    }


def _candidate_binding(candidate_freeze: Path) -> dict[str, Any]:
    root = _leaf_absolute(candidate_freeze)
    try:
        details = os.lstat(root)
    except OSError as error:
        raise FrozenHoldoutProtocolBuildError(
            f"candidate freeze directory is unavailable: {error}"
        ) from error
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
        raise FrozenHoldoutProtocolBuildError("candidate freeze must be a non-symlink directory")
    encoded, receipt_stat = _stable_file_bytes(root / "receipt.json", "candidate freeze receipt")
    binding = {
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "size_bytes": receipt_stat.st_size,
    }
    try:
        _validate_candidate_freeze(root, binding)
    except FrozenHoldoutError as error:
        raise FrozenHoldoutProtocolBuildError(str(error)) from error
    return binding


def _evaluation_frames(
    coco_path: Path,
    identity: FinalHoldoutIdentity,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    coco, _, _ = _read_json(coco_path, "holdout COCO")
    images = _list(coco.get("images"), "holdout COCO.images")
    if not images:
        raise FrozenHoldoutProtocolBuildError("holdout COCO.images must not be empty")
    records: list[tuple[str, int]] = []
    source_sha_by_scene: dict[str, str] = {}
    for index, raw_image in enumerate(images):
        image = _object(raw_image, f"holdout COCO.images[{index}]")
        if (
            "task_id" in image
            and _integer(image.get("task_id"), f"holdout COCO.images[{index}].task_id")
            != identity.task_id
        ):
            raise FrozenHoldoutProtocolBuildError(
                f"holdout COCO image {index} is not bound to the configured final holdout"
            )
        scene_id = _text(image.get("scene_id"), f"holdout COCO.images[{index}].scene_id")
        source_frame = _integer(
            image.get("source_frame"),
            f"holdout COCO.images[{index}].source_frame",
        )
        normalized_frame = _integer(
            image.get("source_normalized_asset_frame"),
            f"holdout COCO.images[{index}].source_normalized_asset_frame",
        )
        if normalized_frame != source_frame:
            raise FrozenHoldoutProtocolBuildError(
                f"holdout COCO image {index} is not an identity source-frame mapping"
            )
        leakage = _text(
            image.get("source_leakage_group_id"),
            f"holdout COCO.images[{index}].source_leakage_group_id",
        )
        source_sha = leakage.removeprefix("sha256:")
        if (
            not leakage.startswith("sha256:")
            or len(source_sha) != 64
            or any(character not in "0123456789abcdef" for character in source_sha)
        ):
            raise FrozenHoldoutProtocolBuildError(
                f"holdout COCO image {index} has an invalid source leakage identity"
            )
        previous_sha = source_sha_by_scene.setdefault(scene_id, source_sha)
        if previous_sha != source_sha:
            raise FrozenHoldoutProtocolBuildError(
                f"holdout COCO scene {scene_id} has multiple source identities"
            )
        records.append((scene_id, source_frame))
    if len(set(records)) != len(records):
        raise FrozenHoldoutProtocolBuildError(
            "holdout COCO contains duplicate scene/source-frame identities"
        )
    return [
        {"scene_id": scene_id, "source_frame": source_frame}
        for scene_id, source_frame in sorted(records)
    ], source_sha_by_scene


def _source_videos(
    *,
    scene_map_path: Path,
    evaluation_frames: Sequence[Mapping[str, Any]],
    holdout_source_sha_by_scene: Mapping[str, str],
    owner: Path,
    frame_step: int | None,
) -> list[dict[str, Any]]:
    scene_map, _, _ = _read_json(scene_map_path, "scene map")
    if frame_step is not None:
        frame_step = _integer(frame_step, "frame_step", minimum=1)
    schema_present = "schema" in scene_map
    if schema_present and scene_map["schema"] != SOURCE_MAP_SCHEMA:
        raise FrozenHoldoutProtocolBuildError(
            "unsupported scene map schema; expected "
            f"{SOURCE_MAP_SCHEMA!r}, or omit schema for the legacy format"
        )
    if schema_present:
        if frame_step is None:
            raise FrozenHoldoutProtocolBuildError(
                "frame_step is required for a v2 training source-frame map"
            )
        expected_frame_keys = {
            (str(frame["scene_id"]), int(frame["source_frame"])) for frame in evaluation_frames
        }
        mapped_frame_keys: set[tuple[str, int]] = set()
        mapped_source_shas_by_scene: dict[str, set[str]] = {}
        for index, raw_frame in enumerate(_list(scene_map.get("frames"), "scene map.frames")):
            mapped = _object(raw_frame, f"scene map.frames[{index}]")
            scene_id = _text(mapped.get("scene_id"), f"scene map.frames[{index}].scene_id")
            source_frame = _integer(
                mapped.get("source_frame"), f"scene map.frames[{index}].source_frame"
            )
            if mapped.get("asset_id") != scene_id:
                raise FrozenHoldoutProtocolBuildError(
                    f"scene map frame {index} asset_id must equal scene_id"
                )
            if (
                _integer(
                    mapped.get("normalized_asset_frame"),
                    f"scene map.frames[{index}].normalized_asset_frame",
                )
                != source_frame
            ):
                raise FrozenHoldoutProtocolBuildError(
                    f"scene map frame {index} is not an identity frame mapping"
                )
            leakage = _text(
                mapped.get("leakage_group_id"),
                f"scene map.frames[{index}].leakage_group_id",
            )
            mapped_sha = leakage.removeprefix("sha256:")
            if (
                not leakage.startswith("sha256:")
                or len(mapped_sha) != 64
                or any(character not in "0123456789abcdef" for character in mapped_sha)
            ):
                raise FrozenHoldoutProtocolBuildError(
                    f"scene map.frames[{index}] has an invalid leakage identity"
                )
            mapped_source_shas_by_scene.setdefault(scene_id, set()).add(mapped_sha)
            key = (scene_id, source_frame)
            if key in mapped_frame_keys:
                raise FrozenHoldoutProtocolBuildError("scene map contains duplicate frame keys")
            mapped_frame_keys.add(key)
        if mapped_frame_keys != expected_frame_keys:
            raise FrozenHoldoutProtocolBuildError(
                "v2 scene map frame universe differs from the holdout COCO"
            )

        assets: dict[str, tuple[int, Path, str, int]] = {}
        for index, raw_asset in enumerate(_list(scene_map.get("assets"), "scene map.assets")):
            asset = _object(raw_asset, f"scene map.assets[{index}]")
            scene_id = _text(asset.get("asset_id"), f"scene map.assets[{index}].asset_id")
            if scene_id in assets:
                raise FrozenHoldoutProtocolBuildError(f"duplicate scene map asset_id: {scene_id}")
            video_path = _workspace_regular_path(
                _text(asset.get("path"), f"scene map.assets[{index}].path"),
                f"scene map.assets[{index}].path",
            )
            declared_sha = _text(asset.get("sha256"), f"scene map.assets[{index}].sha256")
            if len(declared_sha) != 64 or any(
                character not in "0123456789abcdef" for character in declared_sha
            ):
                raise FrozenHoldoutProtocolBuildError(
                    f"scene map.assets[{index}].sha256 must be a lowercase SHA-256"
                )
            fps = asset.get("fps")
            if isinstance(fps, bool) or not isinstance(fps, (int, float)) or float(fps) != 30.0:
                raise FrozenHoldoutProtocolBuildError(f"scene map.assets[{index}].fps must be 30")
            frame_count = _integer(
                asset.get("frame_count"),
                f"scene map.assets[{index}].frame_count",
                minimum=1,
            )
            leakage = _text(
                asset.get("leakage_group_id"),
                f"scene map.assets[{index}].leakage_group_id",
            )
            if leakage != f"sha256:{declared_sha}":
                raise FrozenHoldoutProtocolBuildError(
                    f"scene map.assets[{index}] leakage identity differs from its SHA-256"
                )
            assets[scene_id] = (frame_step, video_path, declared_sha, frame_count)
        evaluation_scenes = {scene_id for scene_id, _ in expected_frame_keys}
        if set(assets) != evaluation_scenes:
            raise FrozenHoldoutProtocolBuildError(
                "v2 scene map assets do not exactly cover evaluation scenes"
            )
        records: list[dict[str, Any]] = []
        for scene_id in sorted(assets):
            step, video_path, declared_sha, frame_count = assets[scene_id]
            if declared_sha != holdout_source_sha_by_scene[scene_id]:
                raise FrozenHoldoutProtocolBuildError(
                    f"scene map asset {scene_id} differs from the holdout COCO source identity"
                )
            if mapped_source_shas_by_scene.get(scene_id) != {declared_sha}:
                raise FrozenHoldoutProtocolBuildError(
                    f"scene map frames for {scene_id} differ from the asset source identity"
                )
            scene_frames = sorted(
                source_frame
                for candidate_scene, source_frame in expected_frame_keys
                if candidate_scene == scene_id
            )
            unreachable = [value for value in scene_frames if (value + 1) % step != 0]
            if unreachable:
                raise FrozenHoldoutProtocolBuildError(
                    f"evaluation frames for {scene_id} are unreachable at frame_step={step}: "
                    f"{unreachable}"
                )
            if scene_frames[-1] >= frame_count:
                raise FrozenHoldoutProtocolBuildError(
                    f"scene map asset {scene_id} has only {frame_count} frames"
                )
            binding = _bound_file(
                video_path,
                owner=owner,
                location=f"source video {scene_id}",
            )
            if binding["sha256"] != declared_sha:
                raise FrozenHoldoutProtocolBuildError(
                    f"source video {scene_id} differs from the v2 scene map SHA-256"
                )
            records.append({"scene_id": scene_id, "frame_step": step, "file": binding})
        return records

    global_step_raw = scene_map.get("frame_step")
    global_step = (
        _integer(global_step_raw, "scene map.frame_step", minimum=1)
        if global_step_raw is not None
        else None
    )
    if frame_step is not None:
        if global_step is not None and global_step != frame_step:
            raise FrozenHoldoutProtocolBuildError(
                "explicit frame_step differs from scene map.frame_step"
            )
        global_step = frame_step
    raw_scenes = _list(scene_map.get("scenes"), "scene map.scenes")
    scenes: dict[str, tuple[int, Path]] = {}
    for index, raw_scene in enumerate(raw_scenes):
        scene = _object(raw_scene, f"scene map.scenes[{index}]")
        scene_id = _text(scene.get("scene_id"), f"scene map.scenes[{index}].scene_id")
        if scene_id in scenes:
            raise FrozenHoldoutProtocolBuildError(f"duplicate scene map scene_id: {scene_id}")
        raw_step = scene.get("frame_step", global_step)
        frame_step = _integer(raw_step, f"scene map.scenes[{index}].frame_step", minimum=1)
        raw_video_path = scene.get("video_path", scene.get("path"))
        video_path = Path(
            _text(raw_video_path, f"scene map.scenes[{index}].video_path")
        ).expanduser()
        if not video_path.is_absolute():
            video_path = _leaf_absolute(scene_map_path).parent / video_path
        scenes[scene_id] = (frame_step, video_path)

    evaluation_scenes = {str(frame["scene_id"]) for frame in evaluation_frames}
    if set(scenes) != evaluation_scenes:
        raise FrozenHoldoutProtocolBuildError(
            "scene map must exactly cover evaluation scenes; "
            f"missing={sorted(evaluation_scenes - set(scenes))}, "
            f"extra={sorted(set(scenes) - evaluation_scenes)}"
        )
    records: list[dict[str, Any]] = []
    for scene_id in sorted(scenes):
        frame_step, video_path = scenes[scene_id]
        scene_frames = [
            int(frame["source_frame"])
            for frame in evaluation_frames
            if frame["scene_id"] == scene_id
        ]
        unreachable = [frame for frame in scene_frames if (frame + 1) % frame_step != 0]
        if unreachable:
            raise FrozenHoldoutProtocolBuildError(
                f"evaluation frames for {scene_id} are unreachable at frame_step={frame_step}: "
                f"{unreachable}"
            )
        binding = _bound_file(
            video_path,
            owner=owner,
            location=f"source video {scene_id}",
        )
        if binding["sha256"] != holdout_source_sha_by_scene[scene_id]:
            raise FrozenHoldoutProtocolBuildError(
                f"source video {scene_id} differs from the holdout COCO source identity"
            )
        records.append(
            {
                "scene_id": scene_id,
                "frame_step": frame_step,
                "file": binding,
            }
        )
    return records


def _encoded_json(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _validate_and_publish(
    *,
    output: Path,
    payload: Mapping[str, Any],
    candidate_freeze: Path,
    identity: FinalHoldoutIdentity,
) -> None:
    output = _leaf_absolute(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(output):
        raise FileExistsError(f"output already exists: {output}")
    encoded = _encoded_json(payload)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{output.name}.",
            suffix=".preflight.tmp",
            dir=output.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            validate_protocol(
                temporary,
                candidate_freeze,
                final_holdout_task_id=identity.task_id,
                final_holdout_job_id=identity.job_id,
            )
        except FrozenHoldoutError as error:
            raise FrozenHoldoutProtocolBuildError(
                f"evaluator preflight rejected the generated protocol: {error}"
            ) from error
        verified, _ = _stable_file_bytes(temporary, "evaluator-validated generated protocol")
        if verified != encoded:
            raise FrozenHoldoutProtocolBuildError(
                "generated protocol changed after evaluator preflight"
            )
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


def build_frozen_holdout_protocol(
    *,
    protocol_id: str,
    candidate_freeze: Path,
    training_dataset_manifest: Path,
    training_reference_manifest: Path,
    baseline_weight: Path,
    holdout_manifest: Path,
    holdout_annotations: Path,
    overlap_evidence: Path,
    warmup_image: Path,
    scene_map: Path,
    settings: Mapping[str, Any],
    output: Path,
    baseline_model_id: str = "baseline-yolo11n-pretrained",
    frame_step: int | None = None,
    final_holdout_task_id: int | None = None,
    final_holdout_job_id: int | None = None,
) -> dict[str, Any]:
    """Build, preflight, and exclusively publish a final-holdout protocol."""

    if os.path.lexists(_leaf_absolute(output)):
        raise FileExistsError(f"output already exists: {_leaf_absolute(output)}")
    protocol_id = _text(protocol_id, "protocol_id")
    baseline_model_id = _text(baseline_model_id, "baseline_model_id")
    try:
        identity = resolve_final_holdout_identity(
            task_id=final_holdout_task_id,
            job_id=final_holdout_job_id,
        )
    except FinalHoldoutConfigError as error:
        raise FrozenHoldoutProtocolBuildError(str(error)) from error
    candidate_binding = _candidate_binding(candidate_freeze)
    try:
        canonical_settings = _validate_settings(dict(settings))
    except FrozenHoldoutError as error:
        raise FrozenHoldoutProtocolBuildError(str(error)) from error
    frames, holdout_source_sha_by_scene = _evaluation_frames(holdout_annotations, identity)
    source_videos = _source_videos(
        scene_map_path=scene_map,
        evaluation_frames=frames,
        holdout_source_sha_by_scene=holdout_source_sha_by_scene,
        owner=output,
        frame_step=frame_step,
    )
    payload = {
        "schema": PROTOCOL_SCHEMA,
        "protocol_id": protocol_id,
        "mode": "production_scene_videos",
        "candidate_freeze": candidate_binding,
        "training_dataset_manifest": _bound_file(
            training_dataset_manifest,
            owner=output,
            location="training dataset manifest",
        ),
        "training_reference_manifest": _bound_file(
            training_reference_manifest,
            owner=output,
            location="training reference manifest",
        ),
        "baseline": {
            "model_id": baseline_model_id,
            "weight": _bound_file(
                baseline_weight,
                owner=output,
                location="baseline weight",
            ),
        },
        "holdout": {
            "task_id": identity.task_id,
            "job_id": identity.job_id,
            "manifest": _bound_file(
                holdout_manifest,
                owner=output,
                location="holdout manifest",
            ),
            "annotations": _bound_file(
                holdout_annotations,
                owner=output,
                location="holdout COCO",
            ),
            "evaluation_frames": frames,
            "evaluation_frames_sha256": canonical_sha256(frames),
            "source_videos": source_videos,
        },
        "overlap_evidence": _bound_file(
            overlap_evidence,
            owner=output,
            location="training/holdout overlap evidence",
        ),
        "warmup_image": _bound_file(
            warmup_image,
            owner=output,
            location="warmup image",
        ),
        "settings": canonical_settings,
        "gates": EXPECTED_GATES,
    }
    _validate_and_publish(
        output=output,
        payload=payload,
        candidate_freeze=candidate_freeze,
        identity=identity,
    )
    encoded = _encoded_json(payload)
    return {
        "output": str(output),
        "protocol_sha256": hashlib.sha256(encoded).hexdigest(),
        "final_holdout_task_id": identity.task_id,
        "final_holdout_job_id": identity.job_id,
        "evaluation_frame_count": len(frames),
        "source_video_count": len(source_videos),
        "evaluator_preflight": "PASS",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--candidate-freeze", required=True, type=Path)
    parser.add_argument("--training-dataset-manifest", required=True, type=Path)
    parser.add_argument("--training-reference-manifest", required=True, type=Path)
    parser.add_argument("--baseline-weight", required=True, type=Path)
    parser.add_argument("--baseline-model-id", default="baseline-yolo11n-pretrained")
    parser.add_argument("--holdout-manifest", required=True, type=Path)
    parser.add_argument("--holdout-annotations", required=True, type=Path)
    parser.add_argument(
        "--final-holdout-task-id",
        type=int,
        help="holdout task id; otherwise resolve from ROADLABELOPS_FINAL_HOLDOUT_TASK_IDS",
    )
    parser.add_argument(
        "--final-holdout-job-id",
        type=int,
        help="holdout job id; otherwise resolve from ROADLABELOPS_FINAL_HOLDOUT_JOB_IDS",
    )
    parser.add_argument("--overlap-evidence", required=True, type=Path)
    parser.add_argument("--warmup-image", required=True, type=Path)
    parser.add_argument("--scene-map", required=True, type=Path)
    parser.add_argument(
        "--frame-step",
        type=int,
        help="required with a v2 source-frame map; must reach every evaluation frame",
    )
    parser.add_argument(
        "--settings",
        type=Path,
        help="optional JSON settings object; canonical defaults are used when omitted",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    settings: Mapping[str, Any] = DEFAULT_SETTINGS
    if args.settings is not None:
        try:
            settings, _, _ = _read_json(args.settings, "settings")
        except FrozenHoldoutProtocolBuildError as error:
            parser.error(str(error))
    try:
        summary = build_frozen_holdout_protocol(
            protocol_id=args.protocol_id,
            candidate_freeze=args.candidate_freeze,
            training_dataset_manifest=args.training_dataset_manifest,
            training_reference_manifest=args.training_reference_manifest,
            baseline_weight=args.baseline_weight,
            baseline_model_id=args.baseline_model_id,
            holdout_manifest=args.holdout_manifest,
            holdout_annotations=args.holdout_annotations,
            final_holdout_task_id=args.final_holdout_task_id,
            final_holdout_job_id=args.final_holdout_job_id,
            overlap_evidence=args.overlap_evidence,
            warmup_image=args.warmup_image,
            scene_map=args.scene_map,
            frame_step=args.frame_step,
            settings=settings,
            output=args.output,
        )
    except (FileExistsError, FrozenHoldoutProtocolBuildError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
