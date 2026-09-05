from __future__ import annotations

import copy
import re

import pytest

from scripts.merge_full_review_segments import SegmentMergeError, merge_segments
from scripts.validate_full_review_decisions import (
    canonical_shape_sha256,
    manual_delete_approval_sha256,
)


def manual_shape() -> dict:
    return {
        "id": 99,
        "frame": 0,
        "label_id": 4,
        "type": "rectangle",
        "points": [1.0, 2.0, 10.0, 12.0],
        "source": "manual",
        "attributes": [],
        "elements": [],
    }


def snapshot() -> dict:
    return {
        "task": {"id": 24},
        "images": [
            {"frame": 0},
            {"frame": 1},
            {"frame": 2},
            {"frame": 3},
        ],
        "annotations": {"shapes": [manual_shape()]},
    }


def candidate_pack() -> dict:
    return {
        "task_id": 24,
        "frames": [
            {
                "frame": 0,
                "candidates": [
                    {
                        "candidate_id": "candidate-light-0",
                        "frame": 0,
                        "status": "needs_human_review",
                    }
                ],
            },
            {
                "frame": 2,
                "candidates": [
                    {
                        "candidate_id": "candidate-car-2",
                        "frame": 2,
                        "status": "existing_match",
                    },
                    {
                        "candidate_id": "candidate-sign-2",
                        "frame": 2,
                        "status": "needs_human_review",
                    },
                ],
            },
        ],
    }


def segment(start: int, end: int, *, reviewer: str) -> dict:
    return {
        "schema_version": "1.0",
        "task_id": 24,
        "frame_start": start,
        "frame_end": end,
        "reviewer": reviewer,
        "reviewed_at": "2026-09-01",
        "reviewed_frames": list(range(start, end + 1)),
        "accepted_candidate_ids": [],
        "frame_actions": [],
        "automated_flag_overrides": [],
        "manual_delete_requests": [],
        "candidate_review_summary": {},
        "qa_notes": [],
    }


def request_manual_delete(segments: list[dict]) -> None:
    shape = manual_shape()
    reason = "The fragment duplicates the visible object and is not an independent instance."
    segments[0]["frame_actions"] = [
        {"frame": 0, "actions": [{"action": "delete", "shape_id": shape["id"]}]}
    ]
    segments[0]["manual_delete_requests"] = [
        {
            "shape_id": shape["id"],
            "frame": shape["frame"],
            "reason": reason,
            "canonical_shape_sha256": canonical_shape_sha256(shape),
        }
    ]


def approved_manual_delete() -> dict:
    shape = manual_shape()
    approval = {
        "approval_sha256": "0" * 64,
        "approval_type": "manual_shape_delete_dual_review",
        "shape_id": shape["id"],
        "frame": shape["frame"],
        "canonical_shape_sha256": canonical_shape_sha256(shape),
        "reason": "The fragment duplicates the visible object and is not an independent instance.",
        "reviewers": [
            {
                "reviewer_id": "reviewer-alpha",
                "reviewed_at": "2026-09-01T12:00:00+08:00",
                "decision": "approve_manual_delete",
            },
            {
                "reviewer_id": "reviewer-beta",
                "reviewed_at": "2026-09-01T12:05:00+08:00",
                "decision": "approve_manual_delete",
            },
        ],
    }
    approval["approval_sha256"] = manual_delete_approval_sha256(approval, task_id=24)
    return approval


def test_merge_builds_deterministic_complete_judgment() -> None:
    first = segment(0, 1, reviewer="reviewer-a")
    first["accepted_candidate_ids"] = ["candidate-light-0"]
    first["frame_actions"] = [
        {
            "frame": 1,
            "actions": [
                {"action": "add", "label": "traffic_sign", "points": [1, 2, 3, 4]}
            ],
        }
    ]
    second = segment(2, 3, reviewer="reviewer-b")
    second["accepted_candidate_ids"] = ["candidate-sign-2"]
    second["automated_flag_overrides"] = [
        {
            "flag_id": "flag-2",
            "frame": 2,
            "replacement_action": {"action": "keep_distinct", "shape_id": 9},
        }
    ]

    result = merge_segments(snapshot(), [candidate_pack()], [second, first])

    assert result == {
        "schema_version": "1.1",
        "judgment_type": "full_review_explicit",
        "task_id": 24,
        "reviewer": "Structured exhaustive visual review (reviewer-a; reviewer-b)",
        "reviewed_at": "2026-09-01",
        "mutation_performed": False,
        "automated_flag_overrides": second["automated_flag_overrides"],
        "manual_delete_approvals": [],
        "accepted_candidate_ids": ["candidate-light-0", "candidate-sign-2"],
        "frame_actions": first["frame_actions"],
    }


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda segments: segments[1].update({"reviewed_frames": [2]}),
            "must exactly cover 2..3",
        ),
        (
            lambda segments: segments[1].update(
                {"frame_start": 1, "reviewed_frames": [1, 2, 3]}
            ),
            "overlap on frames [1]",
        ),
        (
            lambda segments: segments[0].update(
                {"accepted_candidate_ids": ["unknown-candidate"]}
            ),
            "accepts unknown candidate",
        ),
        (
            lambda segments: segments[0].update(
                {"accepted_candidate_ids": ["candidate-sign-2"]}
            ),
            "from frame 2",
        ),
        (
            lambda segments: segments[1].update(
                {"accepted_candidate_ids": ["candidate-car-2"]}
            ),
            "with status 'existing_match'",
        ),
        (
            request_manual_delete,
            "require exact dual-review approvals",
        ),
        (
            lambda segments: segments[0].update({"unexpected": True}),
            "unexpected or missing keys",
        ),
    ],
)
def test_merge_rejects_incomplete_or_unbound_segments(mutate, message: str) -> None:
    segments = [
        segment(0, 1, reviewer="reviewer-a"),
        segment(2, 3, reviewer="reviewer-b"),
    ]
    mutate(segments)

    with pytest.raises(SegmentMergeError, match=re.escape(message)):
        merge_segments(snapshot(), [candidate_pack()], segments)


def test_merge_rejects_coverage_gap() -> None:
    with pytest.raises(SegmentMergeError, match=r"missing=\[2\]"):
        merge_segments(
            snapshot(),
            [candidate_pack()],
            [
                segment(0, 1, reviewer="reviewer-a"),
                segment(3, 3, reviewer="reviewer-b"),
            ],
        )


def test_merge_rejects_duplicate_candidate_ids_across_packs() -> None:
    duplicate = copy.deepcopy(candidate_pack())
    with pytest.raises(SegmentMergeError, match="duplicate candidate_id"):
        merge_segments(
            snapshot(),
            [candidate_pack(), duplicate],
            [segment(0, 3, reviewer="reviewer-a")],
        )


def test_merge_rejects_duplicate_frame_action_records() -> None:
    first = segment(0, 1, reviewer="reviewer-a")
    first["frame_actions"] = [
        {"frame": 0, "actions": []},
        {"frame": 0, "actions": []},
    ]
    with pytest.raises(SegmentMergeError, match="more than one action record"):
        merge_segments(
            snapshot(),
            [candidate_pack()],
            [first, segment(2, 3, reviewer="reviewer-b")],
        )


def test_merge_accepts_exact_dual_review_manual_delete_approval() -> None:
    segments = [
        segment(0, 1, reviewer="reviewer-a"),
        segment(2, 3, reviewer="reviewer-b"),
    ]
    request_manual_delete(segments)
    approval = approved_manual_delete()

    result = merge_segments(
        snapshot(),
        [candidate_pack()],
        list(reversed(segments)),
        manual_delete_approvals=[approval],
    )

    assert result["manual_delete_approvals"] == [approval]
    assert result["frame_actions"] == segments[0]["frame_actions"]


def test_merge_rejects_approval_that_differs_from_review_request() -> None:
    segments = [
        segment(0, 1, reviewer="reviewer-a"),
        segment(2, 3, reviewer="reviewer-b"),
    ]
    request_manual_delete(segments)
    approval = approved_manual_delete()
    approval["reason"] = "A different but still substantive reason."
    approval["approval_sha256"] = manual_delete_approval_sha256(approval, task_id=24)

    with pytest.raises(SegmentMergeError, match="reason differs from its review request"):
        merge_segments(
            snapshot(),
            [candidate_pack()],
            segments,
            manual_delete_approvals=[approval],
        )
