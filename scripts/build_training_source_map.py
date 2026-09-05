"""Build an immutable source-frame map from local training evidence.

The builder is deliberately read-only.  It binds every sampled
``(scene_id, source_frame)`` to exactly one original source asset and publishes
the resulting JSON with exclusive, atomic creation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from decimal import ROUND_FLOOR, ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path, PurePath
from typing import Any

SOURCE_MAP_SCHEMA = {"name": "roadlabelops.training-source-frame-map", "version": 2}
SHA256_LENGTH = 64
HALF = Decimal("0.5")
ZERO = Decimal(0)
ONE = Decimal(1)


class TrainingSourceMapError(ValueError):
    """Raised when source evidence cannot produce an unambiguous frame map."""


def _object(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TrainingSourceMapError(f"{location} must be an object")
    return value


def _list(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise TrainingSourceMapError(f"{location} must be a list")
    return value


def _text(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TrainingSourceMapError(f"{location} must be a non-empty string")
    return value


def _integer(value: Any, location: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TrainingSourceMapError(f"{location} must be an integer")
    if minimum is not None and value < minimum:
        raise TrainingSourceMapError(f"{location} must be at least {minimum}")
    return value


def _decimal(value: Any, location: str, *, positive: bool = False) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise TrainingSourceMapError(f"{location} must be a number")
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except InvalidOperation as error:
        raise TrainingSourceMapError(f"{location} must be a finite number") from error
    if not result.is_finite():
        raise TrainingSourceMapError(f"{location} must be a finite number")
    if positive and result <= ZERO:
        raise TrainingSourceMapError(f"{location} must be greater than zero")
    return result


def _fps(value: Any, location: str) -> Decimal:
    if isinstance(value, str):
        parts = value.split("/")
        if len(parts) != 2:
            raise TrainingSourceMapError(
                f"{location} must be a positive number or NUMERATOR/DENOMINATOR"
            )
        try:
            numerator = Decimal(parts[0])
            denominator = Decimal(parts[1])
        except InvalidOperation as error:
            raise TrainingSourceMapError(f"{location} contains an invalid rational FPS") from error
        if not numerator.is_finite() or not denominator.is_finite() or denominator == ZERO:
            raise TrainingSourceMapError(f"{location} contains an invalid rational FPS")
        result = numerator / denominator
    else:
        result = _decimal(value, location)
    if result <= ZERO or not result.is_finite():
        raise TrainingSourceMapError(f"{location} must be greater than zero")
    return result


def _sha256(value: Any, location: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TrainingSourceMapError(f"{location} must be a lowercase SHA-256 digest")
    return value


def _asset_id(value: Any, location: str) -> int | str:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise TrainingSourceMapError(f"{location} must be an integer or string")
    if isinstance(value, str) and not value:
        raise TrainingSourceMapError(f"{location} must not be empty")
    return value


def _asset_identity(value: int | str) -> tuple[str, str]:
    return type(value).__name__, str(value)


def _leakage_group_id(source_sha256: str) -> str:
    return f"sha256:{source_sha256}"


def _read_json(path: Path, location: str) -> tuple[dict[str, Any], str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise TrainingSourceMapError(f"{location} does not exist: {resolved}")
    try:
        encoded = resolved.read_bytes()
        payload = json.loads(encoded.decode("utf-8"), parse_float=Decimal)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TrainingSourceMapError(f"Could not read {location}: {error}") from error
    return _object(payload, location), hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True).encode(
            "utf-8"
        )
        + b"\n"
    )


def _atomic_write_json_new(output: Path, payload: Mapping[str, Any]) -> None:
    """Atomically publish JSON without resolving or replacing the leaf path."""

    output.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(output):
        raise FileExistsError(f"output already exists: {output}")
    encoded = _json_bytes(payload)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary_path, output)
        except FileExistsError as error:
            raise FileExistsError(f"output already exists: {output}") from error
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _duration_precision(duration: Decimal) -> Decimal:
    """Return the seconds represented by the duration's last decimal place."""

    exponent = duration.as_tuple().exponent
    if exponent >= 0:
        return ZERO
    return ONE.scaleb(exponent)


def _precision_aware_half_up(
    duration: Decimal, fps: Decimal
) -> tuple[int, Decimal, Decimal, Decimal]:
    """Round ``duration * fps`` without losing a serialized half-frame.

    JSON evidence commonly stores seconds to six decimal places.  The upper
    error bound of a nearest-decimal representation is half its final-place
    quantum.  If that propagated bound reaches an exact half-frame boundary,
    the value is treated as the boundary and rounded up.  No other residual is
    allocated or reconciled later.
    """

    raw = duration * fps
    precision = _duration_precision(duration)
    tolerance_frames = precision * fps / 2
    floor_value = raw.to_integral_value(rounding=ROUND_FLOOR)
    fraction = raw - floor_value
    # Coarsely serialized durations whose uncertainty spans half a frame do
    # not provide enough evidence to reinterpret an otherwise exact/non-half
    # estimate.  The precision-aware exception is only for a narrow apparent
    # half-frame boundary such as 500.49999 +/- 0.000015.
    if ZERO < tolerance_frames < HALF and abs(fraction - HALF) <= tolerance_frames:
        rounded = floor_value + ONE
    else:
        rounded = raw.to_integral_value(rounding=ROUND_HALF_UP)
    return int(rounded), raw, precision, tolerance_frames


def _exact_frame_from_seconds(value: Decimal, fps: Decimal, location: str) -> int:
    raw = value * fps
    precision = _duration_precision(value)
    tolerance_frames = precision * fps / 2
    nearest = raw.to_integral_value(rounding=ROUND_HALF_UP)
    if abs(raw - nearest) > tolerance_frames:
        raise TrainingSourceMapError(
            f"{location} is not aligned to an integer compiled frame: {raw}"
        )
    return int(nearest)


def _path_suffix_matches(first: PurePath, second: PurePath) -> bool:
    first_parts = first.parts
    second_parts = second.parts
    shorter = min(len(first_parts), len(second_parts))
    return first_parts[-shorter:] == second_parts[-shorter:]


def _portable_evidence_path(path: Path, location: str) -> str:
    """Return a normalized workspace-relative POSIX evidence path.

    Relative CLI paths remain relative.  Absolute function inputs are accepted
    only when they resolve inside the current workspace, preventing user-home
    or temporary-root prefixes from leaking into immutable evidence.
    """

    if path.is_absolute():
        workspace = Path.cwd().resolve()
        try:
            relative = path.resolve().relative_to(workspace)
        except ValueError as error:
            raise TrainingSourceMapError(
                f"{location} is outside the current workspace and cannot be recorded portably"
            ) from error
        normalized = relative.as_posix()
    else:
        normalized = os.path.normpath(path.as_posix()).replace(os.sep, "/")
    candidate = PurePath(normalized)
    if normalized in {"", "."} or candidate.is_absolute() or ".." in candidate.parts:
        raise TrainingSourceMapError(f"{location} must resolve to a safe workspace-relative path")
    return normalized


def _json_value(value: Any) -> Any:
    """Convert parsed Decimal values back to ordinary JSON-compatible values."""

    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    return value


def _validate_compiled_source(
    session: Mapping[str, Any], evidence: Mapping[str, Any]
) -> tuple[dict[str, Any], Decimal, int, str, Path, str]:
    compiled = _object(evidence.get("compiled_source"), "training evidence.compiled_source")
    compiled_sha = _sha256(compiled.get("sha256"), "training evidence.compiled_source.sha256")
    compiled_fps = _fps(compiled.get("fps"), "training evidence.compiled_source.fps")
    frame_count = _integer(
        compiled.get("frame_count"),
        "training evidence.compiled_source.frame_count",
        minimum=1,
    )
    transform = _text(compiled.get("transform"), "training evidence.compiled_source.transform")
    compiled_duration = _decimal(
        compiled.get("duration_seconds"),
        "training evidence.compiled_source.duration_seconds",
        positive=True,
    )
    if (
        _exact_frame_from_seconds(
            compiled_duration, compiled_fps, "training evidence compiled duration"
        )
        != frame_count
    ):
        raise TrainingSourceMapError(
            "training evidence compiled duration/FPS differs from compiled frame_count"
        )

    session_sha = _sha256(session.get("source_sha256"), "session.source_sha256")
    if session_sha != compiled_sha:
        raise TrainingSourceMapError("session source SHA-256 differs from training evidence")
    if _fps(session.get("fps"), "session.fps") != compiled_fps:
        raise TrainingSourceMapError("session FPS differs from training evidence compiled FPS")
    session_duration = _decimal(
        session.get("duration_seconds"), "session.duration_seconds", positive=True
    )
    if session_duration != compiled_duration:
        raise TrainingSourceMapError("session duration differs from training evidence")

    for field in ("width", "height"):
        session_value = _integer(session.get(field), f"session.{field}", minimum=1)
        compiled_value = _integer(
            compiled.get(field), f"training evidence.compiled_source.{field}", minimum=1
        )
        if session_value != compiled_value:
            raise TrainingSourceMapError(
                f"session {field} differs from training evidence compiled metadata"
            )

    session_source = Path(_text(session.get("source_path"), "session.source_path")).resolve()
    evidence_source = PurePath(
        _text(compiled.get("path"), "training evidence.compiled_source.path")
    )
    if not _path_suffix_matches(PurePath(session_source), evidence_source):
        raise TrainingSourceMapError("session source path differs from training evidence")
    if not session_source.is_file():
        raise TrainingSourceMapError(f"compiled source does not exist: {session_source}")
    if _file_sha256(session_source) != compiled_sha:
        raise TrainingSourceMapError("compiled source SHA-256 differs from evidence")
    if "bytes" in compiled and (
        _integer(compiled.get("bytes"), "training evidence.compiled_source.bytes", minimum=0)
        != session_source.stat().st_size
    ):
        raise TrainingSourceMapError("compiled source byte size differs from evidence")
    portable_source = _portable_evidence_path(
        Path(str(compiled.get("path"))), "training evidence.compiled_source.path"
    )
    return compiled, compiled_fps, frame_count, transform, session_source, portable_source


def _build_assets(
    evidence: Mapping[str, Any], compiled_fps: Decimal, frame_count: int
) -> tuple[list[dict[str, Any]], list[tuple[int, int, int | str, str]]]:
    raw_sources = _list(evidence.get("sources"), "training evidence.sources")
    if not raw_sources:
        raise TrainingSourceMapError("training evidence.sources must not be empty")
    assets: list[dict[str, Any]] = []
    intervals: list[tuple[int, int, int | str, str]] = []
    identities: set[tuple[str, str]] = set()
    source_hashes: dict[str, int | str] = {}
    cursor = 0
    source_ids: list[int | str] = []
    for index, raw_source in enumerate(raw_sources):
        location = f"training evidence.sources[{index}]"
        source = _object(raw_source, location)
        identifier = _asset_id(source.get("asset_id"), f"{location}.asset_id")
        identity = _asset_identity(identifier)
        if identity in identities:
            raise TrainingSourceMapError(f"training evidence has duplicate asset_id {identifier!r}")
        identities.add(identity)
        source_ids.append(identifier)
        _text(source.get("page_url"), f"{location}.page_url")
        _text(source.get("download_url"), f"{location}.download_url")
        source_sha256 = _sha256(source.get("sha256"), f"{location}.sha256")
        if source_sha256 in source_hashes:
            raise TrainingSourceMapError(
                "training evidence aliases the same source SHA-256 under multiple asset IDs: "
                f"{source_hashes[source_sha256]!r} and {identifier!r}"
            )
        source_hashes[source_sha256] = identifier
        leakage_group_id = _leakage_group_id(source_sha256)
        duration = _decimal(
            source.get("duration_seconds"), f"{location}.duration_seconds", positive=True
        )
        _fps(source.get("fps"), f"{location}.fps")
        if "normalized" in source:
            raise TrainingSourceMapError(f"{location}.normalized is reserved for generated data")
        normalized_count, raw_estimate, precision, tolerance = _precision_aware_half_up(
            duration, compiled_fps
        )
        if normalized_count <= 0:
            raise TrainingSourceMapError(f"{location} normalizes to no frames")
        end = cursor + normalized_count
        output_source = copy.deepcopy(source)
        output_source.update(
            {
                "raw_frame_estimate": float(raw_estimate),
                "duration_precision": float(precision),
                "rounding_tolerance_frames": float(tolerance),
                "normalized_frame_count": normalized_count,
                "leakage_group_id": leakage_group_id,
                "normalized": {
                    "fps": _json_value(compiled_fps),
                    "start_frame": cursor,
                    "end_frame_exclusive": end,
                    "frame_count": normalized_count,
                },
            }
        )
        assets.append(_json_value(output_source))
        intervals.append((cursor, end, identifier, leakage_group_id))
        cursor = end

    isolation = evidence.get("source_isolation")
    if isinstance(isolation, Mapping) and "selected_asset_ids" in isolation:
        selected = _list(
            isolation.get("selected_asset_ids"),
            "training evidence.source_isolation.selected_asset_ids",
        )
        selected_ids = [
            _asset_id(value, f"selected_asset_ids[{index}]") for index, value in enumerate(selected)
        ]
        if [_asset_identity(value) for value in selected_ids] != [
            _asset_identity(value) for value in source_ids
        ]:
            raise TrainingSourceMapError(
                "source_isolation.selected_asset_ids differs from ordered sources"
            )

    if cursor != frame_count:
        counts = [end - start for start, end, _identifier, _group in intervals]
        raise TrainingSourceMapError(
            "normalized source frame total differs from compiled frame_count: "
            f"assets={counts} total={cursor}, compiled={frame_count}"
        )
    return assets, intervals


def _build_scenes(
    session: Mapping[str, Any],
    evidence: Mapping[str, Any],
    compiled_fps: Decimal,
    frame_count: int,
) -> dict[str, tuple[int, int]]:
    session_id = _text(session.get("session_id"), "session.session_id")
    raw_scenes = _list(session.get("scenes"), "session.scenes")
    if not raw_scenes:
        raise TrainingSourceMapError("session.scenes must not be empty")
    scenes: list[tuple[int, int, str, int | None]] = []
    scene_ids: set[str] = set()
    for index, raw_scene in enumerate(raw_scenes):
        location = f"session.scenes[{index}]"
        scene = _object(raw_scene, location)
        scene_id = _text(scene.get("scene_id"), f"{location}.scene_id")
        if scene_id in scene_ids:
            raise TrainingSourceMapError(f"session contains duplicate scene_id {scene_id!r}")
        scene_ids.add(scene_id)
        if _text(scene.get("session_id"), f"{location}.session_id") != session_id:
            raise TrainingSourceMapError(f"scene {scene_id!r} belongs to another session")
        start_seconds = _decimal(scene.get("start_seconds"), f"{location}.start_seconds")
        end_seconds = _decimal(scene.get("end_seconds"), f"{location}.end_seconds")
        if start_seconds < ZERO or end_seconds <= start_seconds:
            raise TrainingSourceMapError(f"scene {scene_id!r} has an invalid time interval")
        start = _exact_frame_from_seconds(
            start_seconds, compiled_fps, f"scene {scene_id!r} start_seconds"
        )
        end = _exact_frame_from_seconds(
            end_seconds, compiled_fps, f"scene {scene_id!r} end_seconds"
        )
        task_id = scene.get("cvat_task_id")
        if task_id is not None:
            task_id = _integer(task_id, f"{location}.cvat_task_id", minimum=1)
        scenes.append((start, end, scene_id, task_id))

    scenes.sort(key=lambda item: (item[0], item[1], item[2]))
    expected_start = 0
    for start, end, scene_id, _task_id in scenes:
        if start < expected_start:
            raise TrainingSourceMapError(
                f"scene {scene_id!r} overlaps the preceding scene at frame {start}"
            )
        if start > expected_start:
            raise TrainingSourceMapError(
                f"session scenes have missing coverage at frames {expected_start}..{start - 1}"
            )
        expected_start = end
    if expected_start != frame_count:
        raise TrainingSourceMapError(
            "session scenes have missing coverage at the end of the compiled source: "
            f"covered={expected_start}, compiled={frame_count}"
        )

    workflow = _object(evidence.get("workflow"), "training evidence.workflow")
    if _text(workflow.get("session_id"), "training evidence.workflow.session_id") != session_id:
        raise TrainingSourceMapError("training evidence workflow belongs to another session")
    if _integer(
        workflow.get("scene_count"), "training evidence.workflow.scene_count", minimum=1
    ) != len(scenes):
        raise TrainingSourceMapError("training evidence workflow scene_count differs from session")
    if "scene_task_ids" in workflow:
        expected_task_ids = [task_id for *_prefix, task_id in scenes]
        supplied_task_ids = [
            _integer(value, f"workflow.scene_task_ids[{index}]", minimum=1)
            for index, value in enumerate(
                _list(workflow.get("scene_task_ids"), "workflow.scene_task_ids")
            )
        ]
        if None in expected_task_ids or supplied_task_ids != expected_task_ids:
            raise TrainingSourceMapError(
                "training evidence workflow scene_task_ids differs from session"
            )
    return {scene_id: (start, end) for start, end, scene_id, _task_id in scenes}


def _manifest_samples(
    manifest_paths: Sequence[Path],
    *,
    session_id: str,
    compiled_sha: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[Path, dict[str, Any]]]:
    if not manifest_paths:
        raise TrainingSourceMapError("at least one manifest is required")
    resolved_paths = [path.resolve() for path in manifest_paths]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise TrainingSourceMapError("the same manifest input was supplied more than once")

    records: list[dict[str, Any]] = []
    evidence_records: list[dict[str, Any]] = []
    payloads: dict[Path, dict[str, Any]] = {}
    sample_identities: set[tuple[str, int]] = set()
    file_names: set[str] = set()
    revisions: set[str] = set()
    for manifest_path in sorted(resolved_paths, key=str):
        manifest, digest = _read_json(manifest_path, f"manifest {manifest_path}")
        payloads[manifest_path] = manifest
        if _text(manifest.get("session_id"), "manifest.session_id") != session_id:
            raise TrainingSourceMapError(f"manifest belongs to another session: {manifest_path}")
        if _sha256(manifest.get("source_sha256"), "manifest.source_sha256") != compiled_sha:
            raise TrainingSourceMapError(
                f"manifest source SHA-256 differs from compiled source: {manifest_path}"
            )
        if manifest.get("purpose") != "training":
            raise TrainingSourceMapError(f"manifest purpose must be 'training': {manifest_path}")
        revision = _text(manifest.get("sampling_revision"), "manifest.sampling_revision")
        if revision in revisions:
            raise TrainingSourceMapError(f"duplicate manifest sampling_revision {revision!r}")
        revisions.add(revision)
        samples = _list(manifest.get("samples"), "manifest.samples")
        if not samples:
            raise TrainingSourceMapError(f"manifest samples must not be empty: {manifest_path}")
        if _integer(manifest.get("sample_size"), "manifest.sample_size", minimum=1) != len(samples):
            raise TrainingSourceMapError(
                f"manifest sample_size differs from samples: {manifest_path}"
            )
        seen_indices: set[int] = set()
        counts: Counter[str] = Counter()
        for index, raw_sample in enumerate(samples):
            location = f"manifest {manifest_path}.samples[{index}]"
            sample = _object(raw_sample, location)
            sample_index = _integer(
                sample.get("sample_index"), f"{location}.sample_index", minimum=1
            )
            if sample_index in seen_indices:
                raise TrainingSourceMapError(
                    f"manifest contains duplicate sample_index {sample_index}: {manifest_path}"
                )
            seen_indices.add(sample_index)
            scene_id = _text(sample.get("scene_id"), f"{location}.scene_id")
            source_frame = _integer(
                sample.get("source_frame"), f"{location}.source_frame", minimum=0
            )
            identity = (scene_id, source_frame)
            if identity in sample_identities:
                raise TrainingSourceMapError(
                    f"manifest inputs contain duplicate sample {scene_id!r}/{source_frame}"
                )
            sample_identities.add(identity)
            file_name = _text(sample.get("file_name"), f"{location}.file_name")
            if file_name in file_names:
                raise TrainingSourceMapError(
                    f"manifest inputs contain duplicate file_name {file_name!r}"
                )
            file_names.add(file_name)
            counts[scene_id] += 1
            records.append(
                {
                    "scene_id": scene_id,
                    "source_frame": source_frame,
                    "manifest_path": manifest_path,
                    "sample_index": sample_index,
                }
            )
        if seen_indices != set(range(1, len(samples) + 1)):
            raise TrainingSourceMapError(
                f"manifest sample_index values must be contiguous and one-based: {manifest_path}"
            )
        declared_counts = manifest.get("sample_counts_by_scene")
        if declared_counts is not None:
            supplied_counts = _object(declared_counts, "manifest.sample_counts_by_scene")
            if supplied_counts != dict(sorted(counts.items())):
                raise TrainingSourceMapError(
                    f"manifest sample_counts_by_scene differs from samples: {manifest_path}"
                )
        evidence_records.append(
            {
                "path": _portable_evidence_path(manifest_path, "manifest path"),
                "sha256": digest,
                "sampling_revision": revision,
                "sample_count": len(samples),
            }
        )
    return records, evidence_records, payloads


def _validate_primary_manifest_metadata(
    evidence: Mapping[str, Any], payloads: Mapping[Path, Mapping[str, Any]]
) -> None:
    raw_sample = evidence.get("sample")
    if raw_sample is None:
        return
    sample = _object(raw_sample, "training evidence.sample")
    declared_path = PurePath(
        _text(sample.get("manifest_path"), "training evidence.sample.manifest_path")
    )
    matching = [
        (path, manifest)
        for path, manifest in payloads.items()
        if _path_suffix_matches(PurePath(path), declared_path)
    ]
    if len(matching) != 1:
        raise TrainingSourceMapError(
            "training evidence sample.manifest_path does not match exactly one input manifest"
        )
    _path, manifest = matching[0]
    checks = (
        ("sample_size", "sample_size"),
        ("revision", "sampling_revision"),
        ("sample_counts_by_scene", "sample_counts_by_scene"),
    )
    for evidence_field, manifest_field in checks:
        if evidence_field in sample and sample[evidence_field] != manifest.get(manifest_field):
            raise TrainingSourceMapError(
                f"training evidence sample.{evidence_field} differs from its manifest"
            )


def build_training_source_map(
    *,
    session_record_path: Path,
    training_evidence_path: Path,
    manifest_paths: Sequence[Path],
    output: Path,
) -> dict[str, Any]:
    """Validate evidence, map all samples, and atomically publish a new JSON file."""

    raw_output = Path(output)
    output = raw_output.parent.resolve() / raw_output.name
    if os.path.lexists(output):
        raise TrainingSourceMapError(f"output already exists: {output}")
    session_path = session_record_path.resolve()
    evidence_path = training_evidence_path.resolve()
    session, session_digest = _read_json(session_path, "session record")
    evidence, evidence_digest = _read_json(evidence_path, "training evidence")
    compiled, compiled_fps, frame_count, transform, _compiled_source_path, portable_source = (
        _validate_compiled_source(session, evidence)
    )
    compiled_sha = _sha256(compiled.get("sha256"), "compiled_source.sha256")
    assets, intervals = _build_assets(evidence, compiled_fps, frame_count)
    scenes = _build_scenes(session, evidence, compiled_fps, frame_count)
    samples, manifest_evidence, manifest_payloads = _manifest_samples(
        manifest_paths,
        session_id=_text(session.get("session_id"), "session.session_id"),
        compiled_sha=compiled_sha,
    )
    _validate_primary_manifest_metadata(evidence, manifest_payloads)

    frames: list[dict[str, Any]] = []
    normalized_asset_frames: set[tuple[tuple[str, str], int]] = set()
    for sample in samples:
        scene_id = sample["scene_id"]
        source_frame = sample["source_frame"]
        if scene_id not in scenes:
            raise TrainingSourceMapError(f"manifest sample references unknown scene {scene_id!r}")
        scene_start, scene_end = scenes[scene_id]
        global_frame = scene_start + source_frame
        if global_frame < scene_start or global_frame >= scene_end:
            raise TrainingSourceMapError(
                f"manifest sample {scene_id!r}/{source_frame} is outside its scene"
            )
        if global_frame < 0 or global_frame >= frame_count:
            raise TrainingSourceMapError(
                f"manifest sample {scene_id!r}/{source_frame} is outside compiled frames"
            )
        matches = [
            (start, end, identifier, leakage_group_id)
            for start, end, identifier, leakage_group_id in intervals
            if start <= global_frame < end
        ]
        if len(matches) != 1:
            raise TrainingSourceMapError(
                f"manifest sample {scene_id!r}/{source_frame} maps to {len(matches)} assets"
            )
        asset_start, _asset_end, identifier, leakage_group_id = matches[0]
        normalized_asset_frame = global_frame - asset_start
        normalized_identity = (_asset_identity(identifier), normalized_asset_frame)
        if normalized_identity in normalized_asset_frames:
            raise TrainingSourceMapError(
                "more than one manifest sample maps to normalized asset frame "
                f"{identifier!r}/{normalized_asset_frame}"
            )
        normalized_asset_frames.add(normalized_identity)
        frames.append(
            {
                "scene_id": scene_id,
                "source_frame": source_frame,
                "asset_id": identifier,
                "leakage_group_id": leakage_group_id,
                "normalized_asset_frame": normalized_asset_frame,
            }
        )
    frames.sort(key=lambda item: (item["scene_id"], item["source_frame"]))

    source_map = {
        "schema": SOURCE_MAP_SCHEMA,
        "method": (
            "Sources use evidence array order and contiguous half-open frame intervals. "
            "normalized_frame_count uses precision-aware Decimal ROUND_HALF_UP of "
            "duration_seconds * compiled FPS; only the duration's final decimal-place "
            "uncertainty may resolve an apparent half-frame. global_frame equals "
            "round(scene.start_seconds * compiled FPS) + source_frame. "
            "normalized_asset_frame is the compiled-FPS offset within that asset interval; "
            "it is not a native source frame number."
        ),
        "evidence": {
            "session_record": {
                "path": _portable_evidence_path(session_record_path, "session record path"),
                "sha256": session_digest,
            },
            "training_evidence": {
                "path": _portable_evidence_path(training_evidence_path, "training evidence path"),
                "sha256": evidence_digest,
            },
            "manifests": manifest_evidence,
            "compiled_source": {
                "path": portable_source,
                "sha256": compiled_sha,
                "fps": _json_value(compiled.get("fps")),
                "frame_count": frame_count,
                "method": transform,
            },
        },
        "assets": assets,
        "frames": frames,
    }
    try:
        _atomic_write_json_new(output, source_map)
    except FileExistsError as error:
        raise TrainingSourceMapError(f"output already exists: {output}") from error
    return {
        "output": str(output),
        "source_map_sha256": _file_sha256(output),
        "asset_count": len(assets),
        "mapped_frame_count": len(frames),
        "compiled_frame_count": frame_count,
        "mutation_performed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-record", required=True, type=Path)
    parser.add_argument("--training-evidence", required=True, type=Path)
    parser.add_argument("--manifest", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        summary = build_training_source_map(
            session_record_path=args.session_record,
            training_evidence_path=args.training_evidence,
            manifest_paths=args.manifest,
            output=args.output,
        )
    except TrainingSourceMapError as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
