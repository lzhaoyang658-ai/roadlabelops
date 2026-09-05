"""Merge disjoint visual-review segments into one explicit judgment.

The command is read-only with respect to CVAT and all input evidence. It
requires exact frame coverage, validates every accepted candidate against the
supplied immutable packs, refuses unapproved manual-delete requests, and
publishes the judgment with exclusive atomic creation.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.snapshot_cvat_task import atomic_write_json_new
from scripts.validate_full_review_decisions import (
    DecisionValidationError,
    canonical_shape_sha256,
    validate_manual_delete_approvals,
)

SEGMENT_KEYS = frozenset(
    {
        "schema_version",
        "task_id",
        "frame_start",
        "frame_end",
        "reviewer",
        "reviewed_at",
        "reviewed_frames",
        "accepted_candidate_ids",
        "frame_actions",
        "automated_flag_overrides",
        "manual_delete_requests",
        "candidate_review_summary",
        "qa_notes",
    }
)
MANUAL_DELETE_REQUEST_KEYS = frozenset(
    {"shape_id", "frame", "reason", "canonical_shape_sha256"}
)


class SegmentMergeError(ValueError):
    """Raised when segmented review evidence cannot be safely merged."""


def _object(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SegmentMergeError(f"{location} must be an object")
    return value


def _list(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise SegmentMergeError(f"{location} must be a list")
    return value


def _integer(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SegmentMergeError(f"{location} must be an integer")
    return value


def _text(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SegmentMergeError(f"{location} must be a non-empty string")
    return value.strip()


def _read_json(path: Path, location: str) -> dict[str, Any]:
    if not path.is_file():
        raise SegmentMergeError(f"{location} does not exist: {path}")
    try:
        return _object(json.loads(path.read_text(encoding="utf-8")), location)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SegmentMergeError(f"could not read {location}: {error}") from error


def _snapshot_identity(
    snapshot: Mapping[str, Any],
) -> tuple[int, set[int], dict[tuple[str, str], dict[str, Any]]]:
    task = _object(snapshot.get("task"), "snapshot.task")
    task_id = _integer(task.get("id"), "snapshot.task.id")
    frames: set[int] = set()
    for index, raw_image in enumerate(_list(snapshot.get("images"), "snapshot.images")):
        image = _object(raw_image, f"snapshot.images[{index}]")
        frame = _integer(
            image.get("frame", image.get("cvat_frame")),
            f"snapshot.images[{index}].frame",
        )
        if frame in frames:
            raise SegmentMergeError(f"snapshot contains duplicate frame {frame}")
        frames.add(frame)
    if not frames:
        raise SegmentMergeError("snapshot contains no frames")
    annotations = _object(snapshot.get("annotations", {}), "snapshot.annotations")
    shapes: dict[tuple[str, str], dict[str, Any]] = {}
    for index, raw_shape in enumerate(_list(annotations.get("shapes", []), "snapshot.shapes")):
        shape = _object(raw_shape, f"snapshot.shapes[{index}]")
        shape_id = shape.get("id")
        if isinstance(shape_id, bool) or not isinstance(shape_id, (int, str)):
            raise SegmentMergeError(f"snapshot.shapes[{index}].id must be an integer or string")
        identity = (type(shape_id).__name__, str(shape_id))
        if identity in shapes:
            raise SegmentMergeError(f"snapshot contains duplicate shape ID {shape_id!r}")
        shapes[identity] = shape
    return task_id, frames, shapes


def _candidate_index(
    packs: Sequence[Mapping[str, Any]], *, task_id: int, snapshot_frames: set[int]
) -> dict[str, tuple[int, str]]:
    candidates: dict[str, tuple[int, str]] = {}
    for pack_index, pack in enumerate(packs):
        location = f"candidate pack {pack_index}"
        if _integer(pack.get("task_id"), f"{location}.task_id") != task_id:
            raise SegmentMergeError(f"{location} belongs to a different task")
        for frame_index, raw_frame in enumerate(_list(pack.get("frames"), f"{location}.frames")):
            frame_record = _object(raw_frame, f"{location}.frames[{frame_index}]")
            frame = _integer(
                frame_record.get("frame"), f"{location}.frames[{frame_index}].frame"
            )
            if frame not in snapshot_frames:
                raise SegmentMergeError(f"{location} references unknown frame {frame}")
            for candidate_index, raw_candidate in enumerate(
                _list(
                    frame_record.get("candidates"),
                    f"{location}.frames[{frame_index}].candidates",
                )
            ):
                candidate = _object(
                    raw_candidate,
                    f"{location}.frames[{frame_index}].candidates[{candidate_index}]",
                )
                candidate_id = _text(candidate.get("candidate_id"), "candidate_id")
                candidate_frame = _integer(candidate.get("frame"), "candidate.frame")
                if candidate_frame != frame:
                    raise SegmentMergeError(
                        f"candidate {candidate_id!r} frame differs from its frame record"
                    )
                if candidate_id in candidates:
                    raise SegmentMergeError(
                        f"duplicate candidate_id across candidate packs: {candidate_id!r}"
                    )
                status = _text(candidate.get("status"), "candidate.status")
                candidates[candidate_id] = (frame, status)
    return candidates


def merge_segments(
    snapshot: Mapping[str, Any],
    candidate_packs: Sequence[Mapping[str, Any]],
    segments: Sequence[Mapping[str, Any]],
    manual_delete_approvals: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate and merge already-read review artifacts."""

    if not candidate_packs:
        raise SegmentMergeError("at least one candidate pack is required")
    if not segments:
        raise SegmentMergeError("at least one review segment is required")
    task_id, snapshot_frames, snapshot_shapes = _snapshot_identity(snapshot)
    candidates = _candidate_index(
        candidate_packs, task_id=task_id, snapshot_frames=snapshot_frames
    )

    covered_frames: set[int] = set()
    accepted_ids: list[str] = []
    accepted_seen: set[str] = set()
    frame_actions: list[dict[str, Any]] = []
    action_frames: set[int] = set()
    overrides: list[dict[str, Any]] = []
    override_ids: set[str] = set()
    reviewers: list[str] = []
    reviewed_dates: list[str] = []
    pending_manual_deletes: list[dict[str, Any]] = []
    manual_delete_actions: list[tuple[str, str]] = []

    ordered_segments = sorted(
        segments,
        key=lambda item: _integer(
            _object(item, "segment").get("frame_start"), "segment.frame_start"
        ),
    )
    for segment_index, raw_segment in enumerate(ordered_segments):
        location = f"segment {segment_index}"
        segment = _object(raw_segment, location)
        if set(segment) != SEGMENT_KEYS:
            raise SegmentMergeError(
                f"{location} has unexpected or missing keys; "
                f"missing={sorted(SEGMENT_KEYS - set(segment))}, "
                f"extra={sorted(set(segment) - SEGMENT_KEYS)}"
            )
        if segment.get("schema_version") != "1.0":
            raise SegmentMergeError(f"{location}.schema_version must be '1.0'")
        if _integer(segment.get("task_id"), f"{location}.task_id") != task_id:
            raise SegmentMergeError(f"{location} belongs to a different task")
        frame_start = _integer(segment.get("frame_start"), f"{location}.frame_start")
        frame_end = _integer(segment.get("frame_end"), f"{location}.frame_end")
        if frame_start > frame_end:
            raise SegmentMergeError(f"{location} frame range is reversed")
        expected_frames = set(range(frame_start, frame_end + 1))
        reviewed_list = _list(segment.get("reviewed_frames"), f"{location}.reviewed_frames")
        if any(isinstance(frame, bool) or not isinstance(frame, int) for frame in reviewed_list):
            raise SegmentMergeError(f"{location}.reviewed_frames must contain integers")
        if len(reviewed_list) != len(set(reviewed_list)):
            raise SegmentMergeError(f"{location}.reviewed_frames contains duplicates")
        reviewed_frames = set(reviewed_list)
        if reviewed_frames != expected_frames:
            raise SegmentMergeError(
                f"{location}.reviewed_frames must exactly cover {frame_start}..{frame_end}"
            )
        if not reviewed_frames <= snapshot_frames:
            raise SegmentMergeError(f"{location} covers frames absent from the snapshot")
        overlap = covered_frames & reviewed_frames
        if overlap:
            raise SegmentMergeError(f"review segments overlap on frames {sorted(overlap)}")
        covered_frames.update(reviewed_frames)
        reviewers.append(_text(segment.get("reviewer"), f"{location}.reviewer"))
        reviewed_dates.append(_text(segment.get("reviewed_at"), f"{location}.reviewed_at"))

        for accepted_index, candidate_id_value in enumerate(
            _list(
                segment.get("accepted_candidate_ids"),
                f"{location}.accepted_candidate_ids",
            )
        ):
            candidate_id = _text(
                candidate_id_value,
                f"{location}.accepted_candidate_ids[{accepted_index}]",
            )
            candidate = candidates.get(candidate_id)
            if candidate is None:
                raise SegmentMergeError(f"{location} accepts unknown candidate {candidate_id!r}")
            candidate_frame, status = candidate
            if candidate_frame not in reviewed_frames:
                raise SegmentMergeError(
                    f"{location} accepts candidate {candidate_id!r} from frame {candidate_frame}"
                )
            if status != "needs_human_review":
                raise SegmentMergeError(
                    f"{location} accepts candidate {candidate_id!r} with status {status!r}"
                )
            if candidate_id in accepted_seen:
                raise SegmentMergeError(f"candidate accepted more than once: {candidate_id!r}")
            accepted_seen.add(candidate_id)
            accepted_ids.append(candidate_id)

        for action_index, raw_frame_actions in enumerate(
            _list(segment.get("frame_actions"), f"{location}.frame_actions")
        ):
            record = _object(raw_frame_actions, f"{location}.frame_actions[{action_index}]")
            if set(record) != {"frame", "actions"}:
                raise SegmentMergeError(
                    f"{location}.frame_actions[{action_index}] must contain frame/actions"
                )
            frame = _integer(record.get("frame"), "frame_actions.frame")
            if frame not in reviewed_frames:
                raise SegmentMergeError(f"{location} contains actions for frame {frame}")
            if frame in action_frames:
                raise SegmentMergeError(f"frame {frame} has more than one action record")
            actions = _list(record.get("actions"), "frame_actions.actions")
            for raw_action in actions:
                action = _object(raw_action, "frame_actions.action")
                if action.get("action") != "delete":
                    continue
                shape_id = action.get("shape_id")
                if isinstance(shape_id, bool) or not isinstance(shape_id, (int, str)):
                    raise SegmentMergeError("delete action.shape_id must be an integer or string")
                identity = (type(shape_id).__name__, str(shape_id))
                shape = snapshot_shapes.get(identity)
                if shape is None:
                    raise SegmentMergeError(f"delete action references unknown shape {shape_id!r}")
                if shape.get("source") == "manual":
                    manual_delete_actions.append(identity)
            action_frames.add(frame)
            frame_actions.append(copy.deepcopy(record))

        for override_index, raw_override in enumerate(
            _list(
                segment.get("automated_flag_overrides"),
                f"{location}.automated_flag_overrides",
            )
        ):
            override = _object(
                raw_override, f"{location}.automated_flag_overrides[{override_index}]"
            )
            frame = _integer(override.get("frame"), "automated_flag_override.frame")
            if frame not in reviewed_frames:
                raise SegmentMergeError(f"{location} overrides a flag outside its frame range")
            flag_id = _text(override.get("flag_id"), "automated_flag_override.flag_id")
            if flag_id in override_ids:
                raise SegmentMergeError(f"automated flag overridden more than once: {flag_id!r}")
            override_ids.add(flag_id)
            overrides.append(copy.deepcopy(override))

        requests = _list(
            segment.get("manual_delete_requests"),
            f"{location}.manual_delete_requests",
        )
        pending_manual_deletes.extend(copy.deepcopy(requests))

    if covered_frames != snapshot_frames:
        missing = sorted(snapshot_frames - covered_frames)
        extra = sorted(covered_frames - snapshot_frames)
        raise SegmentMergeError(
            f"segments do not exactly cover the snapshot; missing={missing}, extra={extra}"
        )
    request_by_identity: dict[tuple[str, str], dict[str, Any]] = {}
    for index, raw_request in enumerate(pending_manual_deletes):
        request = _object(raw_request, f"manual_delete_requests[{index}]")
        if set(request) != MANUAL_DELETE_REQUEST_KEYS:
            raise SegmentMergeError(
                f"manual_delete_requests[{index}] has unexpected or missing keys"
            )
        shape_id = request.get("shape_id")
        if isinstance(shape_id, bool) or not isinstance(shape_id, (int, str)):
            raise SegmentMergeError("manual delete request shape_id must be an integer or string")
        identity = (type(shape_id).__name__, str(shape_id))
        if identity in request_by_identity:
            raise SegmentMergeError(f"manual delete requested more than once for shape {shape_id!r}")
        shape = snapshot_shapes.get(identity)
        if shape is None or shape.get("source") != "manual":
            raise SegmentMergeError(
                f"manual delete request must reference a source='manual' shape: {shape_id!r}"
            )
        frame = _integer(request.get("frame"), "manual delete request.frame")
        if frame != shape.get("frame"):
            raise SegmentMergeError(f"manual delete request frame differs for shape {shape_id!r}")
        shape_sha = _text(
            request.get("canonical_shape_sha256"),
            "manual delete request.canonical_shape_sha256",
        )
        if shape_sha != canonical_shape_sha256(shape):
            raise SegmentMergeError(f"manual delete request shape hash differs for {shape_id!r}")
        _text(request.get("reason"), "manual delete request.reason")
        request_by_identity[identity] = request

    action_targets = set(manual_delete_actions)
    request_targets = set(request_by_identity)
    if action_targets != request_targets:
        raise SegmentMergeError(
            "source='manual' delete actions and manual-delete requests differ; "
            f"missing_requests={sorted(action_targets - request_targets)}, "
            f"unused_requests={sorted(request_targets - action_targets)}"
        )
    supplied_approvals = [copy.deepcopy(dict(item)) for item in (manual_delete_approvals or [])]
    if request_targets and not supplied_approvals:
        identities = [
            {"frame": request["frame"], "shape_id": request["shape_id"]}
            for request in request_by_identity.values()
        ]
        raise SegmentMergeError(
            "manual-delete requests require exact dual-review approvals before judgment merge: "
            f"{identities}"
        )
    try:
        validate_manual_delete_approvals(
            supplied_approvals,
            task_id=task_id,
            snapshot_shapes=snapshot_shapes,
            manual_delete_shape_ids=manual_delete_actions,
            location="manual_delete_approvals",
        )
    except DecisionValidationError as error:
        raise SegmentMergeError(str(error)) from error
    for approval in supplied_approvals:
        shape_id = approval.get("shape_id")
        identity = (type(shape_id).__name__, str(shape_id))
        request = request_by_identity.get(identity)
        if request is None:
            continue
        for key in ("frame", "canonical_shape_sha256", "reason"):
            if approval.get(key) != request.get(key):
                raise SegmentMergeError(
                    f"manual delete approval {key} differs from its review request for "
                    f"shape {shape_id!r}"
                )

    candidate_frames = {candidate_id: candidates[candidate_id][0] for candidate_id in accepted_ids}
    return {
        "schema_version": "1.1",
        "judgment_type": "full_review_explicit",
        "task_id": task_id,
        "reviewer": "Structured exhaustive visual review ("
        + "; ".join(reviewers)
        + ")",
        "reviewed_at": max(reviewed_dates),
        "mutation_performed": False,
        "automated_flag_overrides": sorted(
            overrides, key=lambda item: (int(item["frame"]), str(item["flag_id"]))
        ),
        "manual_delete_approvals": supplied_approvals,
        "accepted_candidate_ids": sorted(
            accepted_ids, key=lambda candidate_id: (candidate_frames[candidate_id], candidate_id)
        ),
        "frame_actions": sorted(frame_actions, key=lambda item: int(item["frame"])),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--candidate-pack", action="append", required=True, type=Path)
    parser.add_argument("--segment", action="append", required=True, type=Path)
    parser.add_argument("--manual-delete-approvals", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        snapshot = _read_json(args.snapshot.resolve(), "snapshot")
        candidate_packs = [
            _read_json(path.resolve(), f"candidate pack {index}")
            for index, path in enumerate(args.candidate_pack)
        ]
        segments = [
            _read_json(path.resolve(), f"segment {index}")
            for index, path in enumerate(args.segment)
        ]
        manual_delete_approvals: list[dict[str, Any]] = []
        if args.manual_delete_approvals is not None:
            approval_payload = _read_json(
                args.manual_delete_approvals.resolve(), "manual delete approval file"
            )
            if set(approval_payload) != {"manual_delete_approvals"}:
                raise SegmentMergeError(
                    "manual delete approval file must contain only manual_delete_approvals"
                )
            manual_delete_approvals = [
                _object(item, f"manual_delete_approvals[{index}]")
                for index, item in enumerate(
                    _list(
                        approval_payload.get("manual_delete_approvals"),
                        "manual_delete_approvals",
                    )
                )
            ]
        judgment = merge_segments(
            snapshot,
            candidate_packs,
            segments,
            manual_delete_approvals=manual_delete_approvals,
        )
        atomic_write_json_new(args.output.resolve(), judgment)
    except (FileExistsError, SegmentMergeError) as error:
        raise SystemExit(str(error)) from error
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "task_id": judgment["task_id"],
                "segment_count": len(segments),
                "accepted_candidate_count": len(judgment["accepted_candidate_ids"]),
                "manual_frame_action_count": sum(
                    len(record["actions"]) for record in judgment["frame_actions"]
                ),
                "mutation_performed": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
