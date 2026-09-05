"""Materialize an explicit judgment manifest into a compiler-ready review draft.

This command is read-only with respect to CVAT and every input artifact.  It
accepts candidate IDs only from strongly validated, hash-bound candidate packs,
turns accepted candidates into exact ``add`` actions, fills every snapshot frame
and label, then publishes a schema-1.2 draft with exclusive atomic creation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts import compile_full_review_decisions as compiler
from scripts.build_manual_class_candidate_pack import canonical_sha256
from scripts.snapshot_cvat_task import atomic_write_json_new
from scripts.validate_full_review_decisions import (
    DecisionValidationError,
    validate_manual_delete_approvals,
)

LEGACY_JUDGMENT_KEYS = frozenset(
    {
        "schema_version",
        "judgment_type",
        "task_id",
        "reviewer",
        "reviewed_at",
        "mutation_performed",
        "automated_flag_overrides",
        "accepted_candidate_ids",
        "frame_actions",
    }
)
JUDGMENT_KEYS = LEGACY_JUDGMENT_KEYS | {"manual_delete_approvals"}


class FullReviewDraftMaterializationError(ValueError):
    """Raised when inputs cannot produce one unambiguous compiler draft."""


def _object(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FullReviewDraftMaterializationError(f"{location} must be an object")
    return value


def _list(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise FullReviewDraftMaterializationError(f"{location} must be a list")
    return value


def _strict_int(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FullReviewDraftMaterializationError(f"{location} must be an integer")
    return value


def _read_json_bytes(path: Path, location: str) -> tuple[dict[str, Any], bytes]:
    if not path.is_file():
        raise FullReviewDraftMaterializationError(f"{location} does not exist: {path}")
    try:
        encoded = path.read_bytes()
        payload = json.loads(encoded.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FullReviewDraftMaterializationError(f"could not read {location}: {error}") from error
    return _object(payload, location), encoded


def _compiler_value(description: str, callback):
    try:
        return callback()
    except compiler.FullReviewCompileError as error:
        raise FullReviewDraftMaterializationError(f"{description}: {error}") from error


def _candidate_id_variants(
    candidate: Mapping[str, Any],
    *,
    task_id: int,
    frame: int,
    model_sha256: str,
    source_sha256: str,
) -> set[str]:
    bbox = [float(value) for value in candidate["bbox"]]
    label = str(candidate["label"])
    identity = {
        "task_id": task_id,
        "frame": frame,
        "label": label,
        "bbox": bbox,
        "model_sha256": model_sha256,
        "source_image_sha256": source_sha256,
    }
    prefix = f"task-{task_id}-frame-{frame:06d}-{label.replace('_', '-')}-"
    digest = canonical_sha256(identity)
    # Hardened packs exist with both the original 12-hex suffix and the newer
    # collision-resistant 20-hex suffix.  Both bind the same identity payload.
    return {f"{prefix}{digest[:12]}", f"{prefix}{digest[:20]}"}


def _add_identity(
    action: Mapping[str, Any],
    *,
    frame: int,
    width: int,
    height: int,
    location: str,
) -> tuple[int, str, tuple[float, float, float, float]]:
    points = action.get("points")
    if not isinstance(points, list) or len(points) != 4:
        raise FullReviewDraftMaterializationError(
            f"{location}.points must contain exactly four coordinates"
        )
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in points):
        raise FullReviewDraftMaterializationError(f"{location}.points must contain numbers")
    normalized = tuple(float(value) for value in points)
    if not all(math.isfinite(value) for value in normalized):
        raise FullReviewDraftMaterializationError(f"{location}.points must be finite")
    x1, y1, x2, y2 = normalized
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise FullReviewDraftMaterializationError(f"{location}.points is outside frame {frame}")
    label = action.get("label")
    if not isinstance(label, str) or not label:
        raise FullReviewDraftMaterializationError(f"{location}.label must be non-empty")
    return frame, label, normalized


def _manual_delete_identity(
    raw_action: Mapping[str, Any],
    *,
    shapes: Mapping[tuple[type, str], Mapping[str, Any]],
    location: str,
) -> tuple[str, str] | None:
    if raw_action.get("action") != "delete":
        return None
    shape_id = raw_action.get("shape_id")
    shape = shapes.get((type(shape_id), str(shape_id)))
    if shape is None or shape.get("source") == "auto":
        return None
    if shape.get("source") != "manual":
        raise FullReviewDraftMaterializationError(
            f"{location} may delete only source='auto' shapes or explicitly approved "
            "source='manual' shapes; "
            f"shape {shape_id!r} has source={shape.get('source')!r}"
        )
    return type(shape_id).__name__, str(shape_id)


def _ensure_inputs_unchanged(inputs: Sequence[tuple[str, Path, bytes]]) -> None:
    for location, path, expected in inputs:
        try:
            current = path.read_bytes()
        except OSError as error:
            raise FullReviewDraftMaterializationError(
                f"{location} could not be re-read before publish: {error}"
            ) from error
        if current != expected:
            raise FullReviewDraftMaterializationError(
                f"{location} changed while the draft was materialized: {path}"
            )


def materialize_full_review_draft_files(
    snapshot_path: Path | str,
    review_pack_path: Path | str,
    automated_decisions_path: Path | str,
    judgment_path: Path | str,
    candidate_pack_paths: Sequence[Path | str],
    output_path: Path | str,
) -> dict[str, Any]:
    """Validate, materialize, preflight, and exclusively publish one draft."""

    snapshot_path = Path(snapshot_path).resolve()
    review_pack_path = Path(review_pack_path).resolve()
    automated_decisions_path = Path(automated_decisions_path).resolve()
    judgment_path = Path(judgment_path).resolve()
    output_path = Path(output_path).resolve()
    resolved_candidate_paths = [Path(path).resolve() for path in candidate_pack_paths]
    if not resolved_candidate_paths:
        raise FullReviewDraftMaterializationError("at least one candidate pack is required")
    if len(set(resolved_candidate_paths)) != len(resolved_candidate_paths):
        raise FullReviewDraftMaterializationError("candidate-pack paths must be unique")

    snapshot, snapshot_bytes = _read_json_bytes(snapshot_path, "snapshot")
    review_pack, review_pack_bytes = _read_json_bytes(review_pack_path, "review pack")
    automated_decisions, automated_bytes = _read_json_bytes(
        automated_decisions_path, "automated decisions"
    )
    judgment, judgment_bytes = _read_json_bytes(judgment_path, "judgment manifest")
    input_bytes: list[tuple[str, Path, bytes]] = [
        ("snapshot", snapshot_path, snapshot_bytes),
        ("review pack", review_pack_path, review_pack_bytes),
        ("automated decisions", automated_decisions_path, automated_bytes),
        ("judgment manifest", judgment_path, judgment_bytes),
    ]

    judgment_version = judgment.get("schema_version")
    expected_judgment_keys = LEGACY_JUDGMENT_KEYS if judgment_version == "1.0" else JUDGMENT_KEYS
    if set(judgment) != expected_judgment_keys:
        raise FullReviewDraftMaterializationError(
            "judgment manifest has unexpected or missing keys; "
            f"missing={sorted(expected_judgment_keys - set(judgment))}, "
            f"extra={sorted(set(judgment) - expected_judgment_keys)}"
        )
    if (
        judgment_version not in {"1.0", "1.1"}
        or judgment.get("judgment_type") != "full_review_explicit"
    ):
        raise FullReviewDraftMaterializationError(
            "judgment schema_version/judgment_type is unsupported"
        )
    manual_delete_approvals = copy.deepcopy(
        _list(
            judgment.get("manual_delete_approvals", []),
            "judgment.manual_delete_approvals",
        )
    )
    if judgment.get("mutation_performed") is not False:
        raise FullReviewDraftMaterializationError("judgment.mutation_performed must be false")

    snapshot_sha256 = hashlib.sha256(snapshot_bytes).hexdigest()
    review_pack_sha256 = hashlib.sha256(review_pack_bytes).hexdigest()
    automated_decisions_sha256 = hashlib.sha256(automated_bytes).hexdigest()
    task_id = _compiler_value("snapshot is invalid", lambda: compiler._task_id(snapshot))
    if _strict_int(judgment.get("task_id"), "judgment.task_id") != task_id:
        raise FullReviewDraftMaterializationError(
            "judgment.task_id does not match the snapshot task"
        )
    reviewer = judgment.get("reviewer")
    reviewed_at = judgment.get("reviewed_at")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise FullReviewDraftMaterializationError("judgment.reviewer must be non-empty")
    if not isinstance(reviewed_at, str) or not reviewed_at.strip():
        raise FullReviewDraftMaterializationError("judgment.reviewed_at must be non-empty")

    frame_metadata = _compiler_value(
        "snapshot frame metadata is invalid", lambda: compiler._snapshot_frame_metadata(snapshot)
    )
    labels = _compiler_value("snapshot labels are invalid", lambda: compiler._label_names(snapshot))
    shapes = _compiler_value("snapshot shapes are invalid", lambda: compiler._shape_index(snapshot))

    candidate_index: dict[str, dict[str, Any]] = {}
    covered_labels: set[str] = set()
    review_evidence: list[dict[str, str]] = []
    for pack_index, candidate_path in enumerate(resolved_candidate_paths):
        location = f"candidate pack {pack_index}"
        candidate_pack, candidate_bytes = _read_json_bytes(candidate_path, location)
        input_bytes.append((location, candidate_path, candidate_bytes))
        candidate_sha256 = hashlib.sha256(candidate_bytes).hexdigest()
        pack_labels = _compiler_value(
            f"{location} is invalid",
            lambda candidate_pack=candidate_pack, candidate_path=candidate_path, location=location: (
                compiler._validate_candidate_evidence(
                    candidate_pack,
                    candidate_path,
                    task_id=task_id,
                    snapshot_sha256=snapshot_sha256,
                    review_pack_sha256=review_pack_sha256,
                    snapshot_frames=frame_metadata,
                    snapshot_labels=labels,
                    location=location,
                )
            ),
        )
        covered_labels.update(pack_labels)
        model_sha256 = str(candidate_pack["model_sha256"])
        for raw_frame in candidate_pack["frames"]:
            frame_record = _object(raw_frame, f"{location}.frame")
            frame = int(frame_record["frame"])
            source_sha256 = str(frame_record["source_sha256"])
            for raw_candidate in frame_record["candidates"]:
                candidate = _object(raw_candidate, f"{location}.candidate")
                candidate_id = str(candidate["candidate_id"])
                if candidate_id in candidate_index:
                    raise FullReviewDraftMaterializationError(
                        f"duplicate candidate_id across candidate packs: {candidate_id!r}"
                    )
                expected_candidate_ids = _candidate_id_variants(
                    candidate,
                    task_id=task_id,
                    frame=frame,
                    model_sha256=model_sha256,
                    source_sha256=source_sha256,
                )
                if candidate_id not in expected_candidate_ids:
                    raise FullReviewDraftMaterializationError(
                        f"{location} contains forged candidate_id {candidate_id!r}; "
                        f"expected one of {sorted(expected_candidate_ids)!r}"
                    )
                candidate_index[candidate_id] = copy.deepcopy(candidate)
        review_evidence.append(
            {
                "path": os.path.relpath(candidate_path, start=output_path.parent),
                "sha256": candidate_sha256,
            }
        )
    if not compiler.REQUIRED_MANUAL_CHECK_LABELS <= covered_labels:
        raise FullReviewDraftMaterializationError(
            "candidate packs do not cover all required manual labels; "
            f"missing={sorted(compiler.REQUIRED_MANUAL_CHECK_LABELS - covered_labels)}"
        )

    accepted_ids = _list(judgment.get("accepted_candidate_ids"), "judgment.accepted_candidate_ids")
    seen_accepted: set[str] = set()
    candidate_actions: dict[int, list[dict[str, Any]]] = {frame: [] for frame in frame_metadata}
    add_origins: dict[tuple[int, str, tuple[float, float, float, float]], str] = {}
    for index, candidate_id in enumerate(accepted_ids):
        if not isinstance(candidate_id, str) or not candidate_id:
            raise FullReviewDraftMaterializationError(
                f"judgment.accepted_candidate_ids[{index}] must be a non-empty string"
            )
        if candidate_id in seen_accepted:
            raise FullReviewDraftMaterializationError(
                f"accepted candidate_id is listed more than once: {candidate_id!r}"
            )
        seen_accepted.add(candidate_id)
        candidate = candidate_index.get(candidate_id)
        if candidate is None:
            raise FullReviewDraftMaterializationError(
                f"accepted candidate_id is unknown: {candidate_id!r}"
            )
        if candidate.get("status") != "needs_human_review":
            raise FullReviewDraftMaterializationError(
                f"accepted candidate_id {candidate_id!r} is not needs_human_review"
            )
        frame = int(candidate["frame"])
        action = {
            "action": "add",
            "label": candidate["label"],
            "points": copy.deepcopy(candidate["bbox"]),
        }
        width, height, _source_sha = frame_metadata[frame]
        identity = _add_identity(
            action,
            frame=frame,
            width=width,
            height=height,
            location=f"accepted candidate {candidate_id!r}",
        )
        if identity in add_origins:
            raise FullReviewDraftMaterializationError(
                f"accepted candidate {candidate_id!r} duplicates add from {add_origins[identity]}"
            )
        add_origins[identity] = f"accepted candidate {candidate_id!r}"
        candidate_actions[frame].append(action)

    manual_actions: dict[int, list[dict[str, Any]]] = {frame: [] for frame in frame_metadata}
    manual_delete_shape_ids: list[tuple[str, str]] = []
    seen_action_frames: set[int] = set()
    for frame_index, raw_frame_actions in enumerate(
        _list(judgment.get("frame_actions"), "judgment.frame_actions")
    ):
        location = f"judgment.frame_actions[{frame_index}]"
        frame_record = _object(raw_frame_actions, location)
        if set(frame_record) != {"frame", "actions"}:
            raise FullReviewDraftMaterializationError(
                f"{location} must contain exactly 'frame' and 'actions'"
            )
        frame = _strict_int(frame_record.get("frame"), f"{location}.frame")
        if frame not in frame_metadata:
            raise FullReviewDraftMaterializationError(
                f"{location} references unknown frame {frame}"
            )
        if frame in seen_action_frames:
            raise FullReviewDraftMaterializationError(
                f"judgment.frame_actions contains duplicate frame {frame}"
            )
        seen_action_frames.add(frame)
        for action_index, raw_action in enumerate(
            _list(frame_record.get("actions"), f"{location}.actions")
        ):
            action_location = f"{location}.actions[{action_index}]"
            action = _object(raw_action, action_location)
            _compiler_value(
                f"{action_location} is invalid",
                lambda action=action, action_location=action_location, frame=frame: (
                    compiler._compile_action(
                        action,
                        location=action_location,
                        frame=frame,
                        shapes=shapes,
                        labels=labels,
                    )
                ),
            )
            manual_delete_identity = _manual_delete_identity(
                action, shapes=shapes, location=action_location
            )
            if manual_delete_identity is not None:
                manual_delete_shape_ids.append(manual_delete_identity)
            if action.get("action") == "add":
                width, height, _source_sha = frame_metadata[frame]
                identity = _add_identity(
                    action,
                    frame=frame,
                    width=width,
                    height=height,
                    location=action_location,
                )
                if identity in add_origins:
                    raise FullReviewDraftMaterializationError(
                        f"{action_location} duplicates add from {add_origins[identity]}"
                    )
                add_origins[identity] = action_location
            manual_actions[frame].append(copy.deepcopy(action))

    automated_flag_overrides = copy.deepcopy(
        _list(
            judgment.get("automated_flag_overrides"),
            "judgment.automated_flag_overrides",
        )
    )
    for index, raw_override in enumerate(automated_flag_overrides):
        override_location = f"judgment.automated_flag_overrides[{index}]"
        override = _object(raw_override, override_location)
        replacement = override.get("replacement_action")
        if isinstance(replacement, Mapping):
            replacement_location = f"{override_location}.replacement_action"
            override_frame = _strict_int(override.get("frame"), f"{override_location}.frame")
            _compiler_value(
                f"{replacement_location} is invalid",
                lambda replacement=replacement, replacement_location=replacement_location, override_frame=override_frame: (
                    compiler._compile_action(
                        replacement,
                        location=replacement_location,
                        frame=override_frame,
                        shapes=shapes,
                        labels=labels,
                    )
                ),
            )
            manual_delete_identity = _manual_delete_identity(
                replacement, shapes=shapes, location=replacement_location
            )
            if manual_delete_identity is not None:
                manual_delete_shape_ids.append(manual_delete_identity)

    validator_shapes = {
        (identity_type.__name__, identity_value): shape
        for (identity_type, identity_value), shape in shapes.items()
    }
    try:
        validate_manual_delete_approvals(
            manual_delete_approvals,
            task_id=task_id,
            snapshot_shapes=validator_shapes,
            manual_delete_shape_ids=manual_delete_shape_ids,
            location="judgment.manual_delete_approvals",
        )
    except DecisionValidationError as error:
        raise FullReviewDraftMaterializationError(str(error)) from error

    sorted_labels = sorted(labels)
    draft = {
        "schema_version": "1.2",
        "draft_type": "full_review_human",
        "task_id": task_id,
        "snapshot_sha256": snapshot_sha256,
        "review_pack_sha256": review_pack_sha256,
        "automated_decisions_sha256": automated_decisions_sha256,
        "reviewer": reviewer,
        "reviewed_at": reviewed_at,
        "mutation_performed": False,
        "automated_flag_overrides": automated_flag_overrides,
        "manual_delete_approvals": manual_delete_approvals,
        "review_evidence": review_evidence,
        "frame_reviews": [
            {
                "frame": frame,
                "reviewed": True,
                "labels_reviewed": sorted_labels,
                "actions": [*candidate_actions[frame], *manual_actions[frame]],
            }
            for frame in sorted(frame_metadata)
        ],
    }

    _compiler_value(
        "materialized draft failed compiler preflight",
        lambda: compiler.compile_decisions(
            snapshot,
            review_pack,
            automated_decisions,
            draft,
            snapshot_sha256=snapshot_sha256,
            review_pack_sha256=review_pack_sha256,
            automated_decisions_sha256=automated_decisions_sha256,
            draft_sha256="0" * 64,
            review_evidence=review_evidence,
        ),
    )
    _ensure_inputs_unchanged(input_bytes)
    atomic_write_json_new(output_path, draft)
    return draft


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("review_pack", type=Path)
    parser.add_argument("automated_decisions", type=Path)
    parser.add_argument("judgment", type=Path)
    parser.add_argument("--candidate-pack", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        draft = materialize_full_review_draft_files(
            args.snapshot,
            args.review_pack,
            args.automated_decisions,
            args.judgment,
            args.candidate_pack,
            args.output,
        )
    except (FileExistsError, FullReviewDraftMaterializationError) as error:
        raise SystemExit(str(error)) from error
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "task_id": draft["task_id"],
                "frame_count": len(draft["frame_reviews"]),
                "action_count": sum(
                    len(frame_review["actions"]) for frame_review in draft["frame_reviews"]
                ),
                "candidate_pack_count": len(draft["review_evidence"]),
                "mutation_performed": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
