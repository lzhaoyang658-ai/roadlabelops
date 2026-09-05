"""Create a read-only, immutable JSON snapshot of a CVAT image task."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from PIL import Image

from roadlabelops.settings import Settings, build_cvat_adapter

SNAPSHOT_SCHEMA = {"name": "roadlabelops.cvat-task-snapshot", "version": 1}
REQUIRED_LABELS = (
    "car",
    "bus",
    "truck",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_light",
    "traffic_sign",
)
ANNOTATION_COLLECTION_KEYS = frozenset(
    {"attributes", "elements", "intervals", "shapes", "tags", "tracks"}
)


class SnapshotValidationError(ValueError):
    """Raised when CVAT, the image manifest, and local media do not agree."""


def sdk_to_json(value: Any) -> Any:
    """Convert CVAT SDK models (or plain test doubles) into JSON values."""
    model = getattr(value, "_model", None)
    if model is not None:
        value = model
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
    if isinstance(value, Mapping):
        return {str(key): sdk_to_json(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [sdk_to_json(item) for item in value]
    if isinstance(value, Enum):
        return sdk_to_json(value.value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"Cannot serialize {type(value).__name__} as snapshot JSON")


def canonical_json_bytes(value: Any) -> bytes:
    """Encode a JSON value with a stable, whitespace-free representation."""
    return json.dumps(
        sdk_to_json(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    """Return the SHA256 of the canonical JSON representation."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonicalize_annotation_value(value: Any, parent_key: str | None = None) -> Any:
    value = sdk_to_json(value)
    if isinstance(value, dict):
        return {
            key: _canonicalize_annotation_value(item, key) for key, item in sorted(value.items())
        }
    if isinstance(value, list):
        normalized = [_canonicalize_annotation_value(item) for item in value]
        if parent_key in ANNOTATION_COLLECTION_KEYS:
            normalized.sort(key=canonical_json_bytes)
        return normalized
    return value


def canonicalize_annotations(annotations: Any) -> dict[str, Any]:
    """Normalize complete CVAT annotations while preserving coordinate order."""
    payload = sdk_to_json(annotations)
    if not isinstance(payload, dict):
        raise SnapshotValidationError("CVAT annotations must be an object")
    for key in ("tags", "shapes", "tracks"):
        if key not in payload or not isinstance(payload[key], list):
            raise SnapshotValidationError(f"CVAT annotations.{key} must be a list")
    return _canonicalize_annotation_value(payload)


def validate_manifest(manifest: Mapping[str, Any], task_id: int) -> list[dict[str, Any]]:
    """Validate task identity and deterministic one-based sample ordering."""
    try:
        manifest_task_id = int(manifest["cvat"]["task_id"])
    except (KeyError, TypeError, ValueError) as error:
        raise SnapshotValidationError("Manifest cvat.task_id is missing or invalid") from error
    if manifest_task_id != task_id:
        raise SnapshotValidationError(
            f"Requested task {task_id} does not match manifest task {manifest_task_id}"
        )

    raw_samples = manifest.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        raise SnapshotValidationError("Manifest samples must be a non-empty list")
    samples = [sdk_to_json(sample) for sample in raw_samples]
    expected_indices = list(range(1, len(samples) + 1))
    try:
        actual_indices = [int(sample["sample_index"]) for sample in samples]
    except (KeyError, TypeError, ValueError) as error:
        raise SnapshotValidationError("Every sample must have an integer sample_index") from error
    if actual_indices != expected_indices:
        raise SnapshotValidationError(
            f"Manifest sample_index must be consecutive in list order: expected {expected_indices}"
        )
    try:
        sample_size = int(manifest["sample_size"])
    except (KeyError, TypeError, ValueError) as error:
        raise SnapshotValidationError("Manifest sample_size is missing or invalid") from error
    if sample_size != len(samples):
        raise SnapshotValidationError(
            f"Manifest sample_size {sample_size} does not match {len(samples)} samples"
        )

    file_names: list[str] = []
    for sample in samples:
        file_name = sample.get("file_name")
        if not isinstance(file_name, str) or not file_name.strip():
            raise SnapshotValidationError("Every sample must have a non-empty file_name")
        file_names.append(file_name)
        expected_frame = int(sample["sample_index"]) - 1
        for frame_key in ("cvat_frame", "frame"):
            if frame_key in sample:
                try:
                    actual_frame = int(sample[frame_key])
                except (TypeError, ValueError) as error:
                    raise SnapshotValidationError(
                        f"Sample {sample['sample_index']} {frame_key} must be an integer"
                    ) from error
                if actual_frame != expected_frame:
                    raise SnapshotValidationError(
                        f"Sample {sample['sample_index']} {frame_key} must be {expected_frame}"
                    )
    if len(set(file_names)) != len(file_names):
        raise SnapshotValidationError("Manifest sample file_name values must be unique")
    return samples


def validate_labels(labels: Sequence[Any]) -> list[dict[str, Any]]:
    """Require the exact eight-class RoadLabelOps taxonomy and unique label IDs."""
    records = [sdk_to_json(label) for label in labels]
    if any(not isinstance(record, dict) for record in records):
        raise SnapshotValidationError("Every CVAT label must be an object")
    try:
        names = [str(record["name"]) for record in records]
        identifiers = [int(record["id"]) for record in records]
    except (KeyError, TypeError, ValueError) as error:
        raise SnapshotValidationError(
            "Every CVAT label must have an integer id and name"
        ) from error
    if len(set(names)) != len(names) or len(set(identifiers)) != len(identifiers):
        raise SnapshotValidationError("CVAT label names and IDs must be unique")
    if len(records) != len(REQUIRED_LABELS) or set(names) != set(REQUIRED_LABELS):
        raise SnapshotValidationError(
            "CVAT labels must exactly match the eight RoadLabelOps classes; "
            f"expected {sorted(REQUIRED_LABELS)}, got {sorted(names)}"
        )
    return sorted(records, key=lambda record: REQUIRED_LABELS.index(str(record["name"])))


def validate_jobs(
    jobs: Sequence[Any],
    *,
    task_id: int,
    expected_job_ids: Sequence[int],
    frame_count: int,
) -> list[dict[str, Any]]:
    """Validate job identity and require its frame ranges to cover the task."""
    records = [sdk_to_json(job) for job in jobs]
    if not records:
        raise SnapshotValidationError("CVAT task has no jobs")
    try:
        actual_ids = [int(record["id"]) for record in records]
        job_task_ids = [int(record["task_id"]) for record in records]
    except (KeyError, TypeError, ValueError) as error:
        raise SnapshotValidationError("Every CVAT job must have integer id and task_id") from error
    try:
        expected_ids = [int(identifier) for identifier in expected_job_ids]
    except (TypeError, ValueError) as error:
        raise SnapshotValidationError("Manifest cvat.job_ids must contain integers") from error
    if len(set(actual_ids)) != len(actual_ids):
        raise SnapshotValidationError("CVAT job IDs must be unique")
    if sorted(actual_ids) != sorted(expected_ids):
        raise SnapshotValidationError(
            f"CVAT jobs {sorted(actual_ids)} do not match manifest jobs {sorted(expected_ids)}"
        )
    if any(identifier != task_id for identifier in job_task_ids):
        raise SnapshotValidationError("A CVAT job belongs to a different task")

    covered: set[int] = set()
    for record in records:
        try:
            start = int(record["start_frame"])
            stop = int(record["stop_frame"])
        except (KeyError, TypeError, ValueError) as error:
            raise SnapshotValidationError("Every CVAT job must have a valid frame range") from error
        if start < 0 or stop < start or stop >= frame_count:
            raise SnapshotValidationError(
                f"CVAT job {record['id']} has invalid frame range {start}..{stop}"
            )
        if "frame_count" in record:
            try:
                reported_frame_count = int(record["frame_count"])
            except (TypeError, ValueError) as error:
                raise SnapshotValidationError(
                    f"CVAT job {record['id']} frame_count must be an integer"
                ) from error
            if reported_frame_count != stop - start + 1:
                raise SnapshotValidationError(
                    f"CVAT job {record['id']} frame_count does not match its frame range"
                )
        covered.update(range(start, stop + 1))
    expected_frames = set(range(frame_count))
    if covered != expected_frames:
        missing = sorted(expected_frames - covered)
        raise SnapshotValidationError(f"CVAT jobs do not cover task frames: missing {missing}")
    return sorted(records, key=lambda record: int(record["id"]))


def _annotation_frames(value: Any, *, parent_key: str | None = None) -> list[int]:
    frames: list[int] = []
    if isinstance(value, dict):
        if parent_key not in {None, "annotations"} and "frame" in value:
            try:
                frames.append(int(value["frame"]))
            except (TypeError, ValueError) as error:
                raise SnapshotValidationError("Annotation frame must be an integer") from error
        for key, item in value.items():
            frames.extend(_annotation_frames(item, parent_key=key))
    elif isinstance(value, list):
        for item in value:
            frames.extend(_annotation_frames(item, parent_key=parent_key))
    return frames


def _annotation_label_ids(value: Any, *, parent_key: str | None = None) -> list[int]:
    identifiers: list[int] = []
    if isinstance(value, dict):
        if parent_key not in {None, "annotations"} and "label_id" in value:
            try:
                identifiers.append(int(value["label_id"]))
            except (TypeError, ValueError) as error:
                raise SnapshotValidationError("Annotation label_id must be an integer") from error
        for key, item in value.items():
            identifiers.extend(_annotation_label_ids(item, parent_key=key))
    elif isinstance(value, list):
        for item in value:
            identifiers.extend(_annotation_label_ids(item, parent_key=parent_key))
    return identifiers


def validate_frame_mapping(
    samples: Sequence[Mapping[str, Any]],
    frame_metadata: Sequence[Any],
    annotations: Mapping[str, Any],
    *,
    task_size: int,
) -> list[dict[str, Any]]:
    """Bind manifest samples, CVAT frame names, and every annotation frame."""
    if task_size != len(samples):
        raise SnapshotValidationError(
            f"CVAT task size {task_size} does not match {len(samples)} manifest samples"
        )
    frames = [sdk_to_json(frame) for frame in frame_metadata]
    if len(frames) != task_size:
        raise SnapshotValidationError(
            f"CVAT returned {len(frames)} frame metadata records for task size {task_size}"
        )
    for index, (sample, frame) in enumerate(zip(samples, frames)):
        if not isinstance(frame, dict) or not frame.get("name"):
            raise SnapshotValidationError(f"CVAT frame {index} has no name")
        expected_name = Path(str(sample["file_name"])).name
        actual_name = Path(str(frame["name"])).name
        if actual_name != expected_name:
            raise SnapshotValidationError(
                f"CVAT frame {index} is {actual_name!r}, expected {expected_name!r}"
            )
    invalid_frames = sorted(
        {frame for frame in _annotation_frames(dict(annotations)) if frame not in range(task_size)}
    )
    if invalid_frames:
        raise SnapshotValidationError(
            f"Annotations reference frames outside 0..{task_size - 1}: {invalid_frames}"
        )
    return frames


def validate_annotation_labels(
    annotations: Mapping[str, Any], labels: Sequence[Mapping[str, Any]]
) -> None:
    valid = {int(label["id"]) for label in labels}
    for collection in ("tags", "shapes", "tracks"):
        if any("label_id" not in item for item in annotations[collection]):
            raise SnapshotValidationError(
                f"Every CVAT annotations.{collection} item must have a label_id"
            )
    unknown = sorted(set(_annotation_label_ids(dict(annotations))) - valid)
    if unknown:
        raise SnapshotValidationError(f"Annotations reference unknown label IDs: {unknown}")


def build_image_inventory(
    manifest_path: Path,
    samples: Sequence[Mapping[str, Any]],
    frame_metadata: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Hash and inspect every manifest image, rejecting traversal and media drift."""
    image_root = (manifest_path.resolve().parent / "images").resolve()
    inventory: list[dict[str, Any]] = []
    for sample, frame in zip(samples, frame_metadata):
        image_path = (image_root / str(sample["file_name"])).resolve()
        if not image_path.is_relative_to(image_root):
            raise SnapshotValidationError(
                f"Sample {sample['sample_index']} file_name escapes the images directory"
            )
        if not image_path.is_file():
            raise SnapshotValidationError(f"Manifest image is missing: {image_path}")
        before = image_path.stat()
        digest = file_sha256(image_path)
        try:
            with Image.open(image_path) as image:
                width, height = image.size
                image_format = image.format
                image.verify()
        except Exception as error:
            raise SnapshotValidationError(f"Manifest image is invalid: {image_path}") from error
        after = image_path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise SnapshotValidationError(f"Manifest image changed while hashing: {image_path}")
        if width <= 0 or height <= 0:
            raise SnapshotValidationError(f"Manifest image has invalid dimensions: {image_path}")
        cvat_width = frame.get("width")
        cvat_height = frame.get("height")
        if cvat_width is not None and int(cvat_width) != width:
            raise SnapshotValidationError(
                f"Image width for frame {int(sample['sample_index']) - 1} differs from CVAT"
            )
        if cvat_height is not None and int(cvat_height) != height:
            raise SnapshotValidationError(
                f"Image height for frame {int(sample['sample_index']) - 1} differs from CVAT"
            )
        inventory.append(
            {
                "sample_index": int(sample["sample_index"]),
                "cvat_frame": int(sample["sample_index"]) - 1,
                "scene_id": sample.get("scene_id"),
                "source_frame": sample.get("source_frame"),
                "file_name": str(sample["file_name"]),
                "relative_path": f"images/{sample['file_name']}",
                "sha256": digest,
                "size_bytes": before.st_size,
                "width": width,
                "height": height,
                "format": image_format,
            }
        )
    return inventory


def build_snapshot_payload(
    *,
    task_id: int,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    task: Any,
    labels: Sequence[Any],
    jobs: Sequence[Any],
    metadata: Any,
    annotations: Any,
    created_at: str,
) -> dict[str, Any]:
    """Build and validate a snapshot from already-fetched CVAT values."""
    samples = validate_manifest(manifest, task_id)
    task_record = sdk_to_json(task)
    if not isinstance(task_record, dict):
        raise SnapshotValidationError("CVAT task must be an object")
    try:
        actual_task_id = int(task_record["id"])
        task_size = int(task_record["size"])
    except (KeyError, TypeError, ValueError) as error:
        raise SnapshotValidationError("CVAT task id or size is missing") from error
    if actual_task_id != task_id:
        raise SnapshotValidationError(
            f"CVAT returned task {actual_task_id} while task {task_id} was requested"
        )
    manifest_project = manifest.get("cvat", {}).get("project_id")
    if manifest_project is not None:
        try:
            project_matches = int(task_record.get("project_id")) == int(manifest_project)
        except (TypeError, ValueError) as error:
            raise SnapshotValidationError("CVAT task project ID is missing or invalid") from error
        if not project_matches:
            raise SnapshotValidationError(
                "CVAT task project does not match manifest cvat.project_id"
            )

    label_records = validate_labels(labels)
    annotation_record = canonicalize_annotations(annotations)
    validate_annotation_labels(annotation_record, label_records)
    metadata_record = sdk_to_json(metadata)
    if not isinstance(metadata_record, dict) or not isinstance(metadata_record.get("frames"), list):
        raise SnapshotValidationError("CVAT frame metadata is missing")
    if "size" in metadata_record:
        try:
            metadata_size = int(metadata_record["size"])
        except (TypeError, ValueError) as error:
            raise SnapshotValidationError("CVAT frame metadata size is invalid") from error
        if metadata_size != task_size:
            raise SnapshotValidationError("CVAT task size differs from frame metadata size")
    frame_records = validate_frame_mapping(
        samples,
        metadata_record["frames"],
        annotation_record,
        task_size=task_size,
    )
    try:
        expected_job_ids = manifest["cvat"]["job_ids"]
    except (KeyError, TypeError) as error:
        raise SnapshotValidationError("Manifest cvat.job_ids is missing") from error
    if not isinstance(expected_job_ids, list) or not expected_job_ids:
        raise SnapshotValidationError("Manifest cvat.job_ids must be a non-empty list")
    job_records = validate_jobs(
        jobs,
        task_id=task_id,
        expected_job_ids=expected_job_ids,
        frame_count=task_size,
    )
    images = build_image_inventory(manifest_path, samples, frame_records)

    labels_by_id = {int(label["id"]): str(label["name"]) for label in label_records}
    class_counts: Counter[str] = Counter()
    for collection in ("tags", "shapes", "tracks"):
        for item in annotation_record[collection]:
            class_counts[labels_by_id[int(item["label_id"])]] += 1
    track_count = len(annotation_record["tracks"])
    blocking_reasons = []
    warnings = []
    if track_count:
        message = (
            f"TRACKS_PRESENT: snapshot preserves {track_count} track(s), but static-image "
            "dataset export requires an explicit track-flattening review"
        )
        blocking_reasons.append(message)
        warnings.append(message)

    annotation_hash = canonical_sha256(annotation_record)
    return {
        "snapshot_schema": SNAPSHOT_SCHEMA,
        "created_at": created_at,
        "task": task_record,
        "manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": file_sha256(manifest_path.resolve()),
            "session_id": manifest.get("session_id"),
            "purpose": manifest.get("purpose"),
            "sampling_revision": manifest.get("sampling_revision"),
            "sample_size": len(samples),
            "cvat_task_id": task_id,
        },
        "labels": label_records,
        "jobs": job_records,
        "frame_metadata": {key: value for key, value in metadata_record.items() if key != "frames"},
        "images": images,
        "annotations": annotation_record,
        "canonical_annotations_sha256": annotation_hash,
        "counts": {
            "images": len(images),
            "tags": len(annotation_record["tags"]),
            "shapes": len(annotation_record["shapes"]),
            "tracks": track_count,
            "annotations_by_label": {
                label: class_counts.get(label, 0) for label in REQUIRED_LABELS
            },
        },
        "final_gate": {
            "passed": not blocking_reasons,
            "blocking_reasons": blocking_reasons,
            "warnings": warnings,
        },
    }


def atomic_write_json_new(output: Path, payload: Mapping[str, Any]) -> None:
    """Atomically publish JSON without ever replacing an existing path."""
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Snapshot output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            sdk_to_json(payload), ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True
        ).encode("utf-8")
        + b"\n"
    )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, delete=False
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary_path, output)
        except FileExistsError as error:
            raise FileExistsError(f"Snapshot output already exists: {output}") from error
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True, type=int)
    parser.add_argument("--image-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest_path = args.image_manifest.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise SystemExit(f"Snapshot output already exists: {output_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"Could not read image manifest: {error}") from error
    try:
        validate_manifest(manifest, args.task_id)
    except SnapshotValidationError as error:
        raise SystemExit(str(error)) from error

    adapter = build_cvat_adapter(Settings())
    if adapter is None:
        raise SystemExit("CVAT is not configured")
    try:
        with adapter._client() as client:
            task = client.tasks.retrieve(args.task_id)
            payload = build_snapshot_payload(
                task_id=args.task_id,
                manifest=manifest,
                manifest_path=manifest_path,
                task=task,
                labels=task.get_labels(),
                jobs=task.get_jobs(),
                metadata=task.get_meta(),
                annotations=task.get_annotations(),
                created_at=datetime.now(timezone.utc).isoformat(),
            )
        atomic_write_json_new(output_path, payload)
    except (FileExistsError, SnapshotValidationError) as error:
        raise SystemExit(str(error)) from error
    print(
        json.dumps(
            {
                "output": str(output_path),
                "task_id": args.task_id,
                "image_count": payload["counts"]["images"],
                "annotation_counts": {
                    key: payload["counts"][key] for key in ("tags", "shapes", "tracks")
                },
                "canonical_annotations_sha256": payload["canonical_annotations_sha256"],
                "final_gate": payload["final_gate"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
