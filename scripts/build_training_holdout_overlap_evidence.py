"""Publish immutable training/holdout leakage evidence.

The builder deliberately reuses the frozen-holdout evaluator's reference
parser and universe digest. This keeps the four leakage identities exactly
aligned with the final gate: typed asset IDs, managed image SHA-256 values,
source-content SHA-256 values, and
``(source_sha256, source_normalized_asset_frame)`` keys.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts import evaluate_frozen_holdout as evaluator

OUTPUT_SCHEMA = evaluator.OVERLAP_SCHEMA


class OverlapEvidenceError(ValueError):
    """Raised when exact overlap evidence cannot be published safely."""


def _leaf_absolute(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = Path.cwd() / expanded
    return expanded.parent.resolve() / expanded.name


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _publish_json_new(output: Path, payload: Mapping[str, Any]) -> None:
    """Atomically create one JSON file without following/replacing its leaf."""

    destination = _leaf_absolute(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(destination):
        raise FileExistsError(f"output already exists: {destination}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(_json_bytes(payload))
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            raise FileExistsError(f"output already exists: {destination}") from error
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _reference(
    manifest_path: Path,
    *,
    schema: Mapping[str, Any],
    location: str,
    verify_all_files: bool,
) -> evaluator.ReferenceData:
    try:
        _payload, manifest = evaluator._read_json(manifest_path, location)
        return evaluator._build_reference_data(
            manifest,
            expected_schema=schema,
            annotations_record=None,
            location=location,
            verify_all_files=verify_all_files,
        )
    except evaluator.FrozenHoldoutError as error:
        raise OverlapEvidenceError(str(error)) from error


def _computed(
    training: evaluator.ReferenceData, holdout: evaluator.ReferenceData
) -> dict[str, Any]:
    return {
        "training_asset_ids_sha256": evaluator._universe_digest(training.asset_ids),
        "holdout_asset_ids_sha256": evaluator._universe_digest(holdout.asset_ids),
        "training_image_sha256s_sha256": evaluator._universe_digest(training.image_hashes),
        "holdout_image_sha256s_sha256": evaluator._universe_digest(holdout.image_hashes),
        "training_source_sha256s_sha256": evaluator._universe_digest(training.source_hashes),
        "holdout_source_sha256s_sha256": evaluator._universe_digest(holdout.source_hashes),
        "training_frame_keys_sha256": evaluator._universe_digest(training.source_frame_keys),
        "holdout_frame_keys_sha256": evaluator._universe_digest(holdout.source_frame_keys),
        "asset_id_overlap_count": len(training.asset_ids & holdout.asset_ids),
        "image_sha256_overlap_count": len(training.image_hashes & holdout.image_hashes),
        "source_sha256_overlap_count": len(training.source_hashes & holdout.source_hashes),
        "frame_overlap_count": len(training.source_frame_keys & holdout.source_frame_keys),
    }


def build_overlap_evidence(
    training_manifest: Path | str,
    holdout_manifest: Path | str,
    output: Path | str,
) -> dict[str, Any]:
    """Validate both references and publish their exact four-universe comparison."""

    training = _reference(
        Path(training_manifest),
        schema=evaluator.TRAINING_SCHEMA,
        location="training reference manifest",
        verify_all_files=False,
    )
    holdout = _reference(
        Path(holdout_manifest),
        schema=evaluator.HOLDOUT_SCHEMA,
        location="holdout manifest",
        verify_all_files=True,
    )
    computed = _computed(training, holdout)
    payload = {
        "schema": OUTPUT_SCHEMA,
        "training_reference_manifest_sha256": training.manifest.sha256,
        "holdout_manifest_sha256": holdout.manifest.sha256,
        "computed": computed,
        "gate_result": (
            "PASS"
            if all(
                computed[name] == 0
                for name in (
                    "asset_id_overlap_count",
                    "image_sha256_overlap_count",
                    "source_sha256_overlap_count",
                    "frame_overlap_count",
                )
            )
            else "FAIL"
        ),
    }
    _publish_json_new(Path(output), payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-manifest", required=True, type=Path)
    parser.add_argument("--holdout-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        payload = build_overlap_evidence(args.training_manifest, args.holdout_manifest, args.output)
    except (FileExistsError, OverlapEvidenceError, OSError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
