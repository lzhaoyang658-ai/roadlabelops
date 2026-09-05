"""Render a hash-bound full-review apply plan without connecting to CVAT.

The command validates the immutable snapshot, review pack, decisions, and all
review-evidence files.  It then applies the explicit decisions to an in-memory
copy of the snapshot annotations and renders a fresh, read-only review pack.
No CVAT adapter or client is created anywhere in this module.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.apply_final_review import (
    FinalReviewApplyError,
    build_apply_plan,
    validate_review_evidence,
)
from scripts.build_full_review_pack import (
    _resolve_image_path,
    build_review_pack,
    write_json_atomic,
)
from scripts.snapshot_cvat_task import canonical_sha256


class PlannedReviewRenderError(ValueError):
    """Raised before or while creating a local-only planned review pack."""


def _read_hashed_object(path: Path, description: str) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise PlannedReviewRenderError(f"{description} does not exist: {path}")
    try:
        encoded = path.read_bytes()
        payload = json.loads(encoded.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PlannedReviewRenderError(f"Could not read {description}: {error}") from error
    if not isinstance(payload, dict):
        raise PlannedReviewRenderError(f"{description} must be a JSON object")
    return payload, hashlib.sha256(encoded).hexdigest()


def _frame_number(image: Mapping[str, Any], position: int) -> int:
    raw_frame = image.get("frame", image.get("cvat_frame", image.get("source_frame", position)))
    if isinstance(raw_frame, bool) or not isinstance(raw_frame, int):
        raise PlannedReviewRenderError(f"snapshot.images[{position}].frame must be an integer")
    return raw_frame


def _planned_snapshot(
    snapshot: Mapping[str, Any],
    expected_annotations: Mapping[str, Any],
    *,
    snapshot_path: Path,
    snapshot_sha256: str,
    review_pack_path: Path,
    review_pack_sha256: str,
    decisions_path: Path,
    decisions_sha256: str,
    expected_post_apply_sha256: str,
) -> dict[str, Any]:
    planned = copy.deepcopy(dict(snapshot))
    planned["annotations"] = copy.deepcopy(dict(expected_annotations))
    planned_annotation_sha = canonical_sha256(planned["annotations"])
    planned["canonical_annotations_sha256"] = planned_annotation_sha

    labels = planned.get("labels")
    if not isinstance(labels, list):
        raise PlannedReviewRenderError("snapshot.labels must be a list")
    label_names_by_id: dict[int, str] = {}
    for index, label in enumerate(labels):
        if not isinstance(label, dict):
            raise PlannedReviewRenderError(f"snapshot.labels[{index}] must be an object")
        raw_id = label.get("id")
        name = label.get("name")
        if isinstance(raw_id, bool) or not isinstance(raw_id, int):
            raise PlannedReviewRenderError(f"snapshot.labels[{index}].id must be an integer")
        if not isinstance(name, str) or not name:
            raise PlannedReviewRenderError(f"snapshot.labels[{index}].name must be non-empty")
        label_names_by_id[raw_id] = name

    shapes = planned["annotations"].get("shapes")
    tags = planned["annotations"].get("tags")
    tracks = planned["annotations"].get("tracks")
    if not isinstance(shapes, list) or not isinstance(tags, list) or not isinstance(tracks, list):
        raise PlannedReviewRenderError("planned annotations shapes, tags, and tracks must be lists")
    class_counts: Counter[str] = Counter()
    for index, shape in enumerate(shapes):
        if not isinstance(shape, dict):
            raise PlannedReviewRenderError(f"planned shape {index} must be an object")
        label_id = shape.get("label_id")
        if isinstance(label_id, bool) or not isinstance(label_id, int):
            raise PlannedReviewRenderError(f"planned shape {index}.label_id must be an integer")
        try:
            class_counts[label_names_by_id[label_id]] += 1
        except KeyError as error:
            raise PlannedReviewRenderError(
                f"planned shape {index} references unknown label_id {label_id}"
            ) from error

    images = planned.get("images")
    if not isinstance(images, list) or not images:
        raise PlannedReviewRenderError("snapshot.images must be a non-empty list")
    manifest_dir: Path | None = None
    manifest = snapshot.get("manifest")
    if isinstance(manifest, dict) and manifest.get("path"):
        manifest_dir = Path(str(manifest["path"])).resolve().parent
    for position, raw_image in enumerate(images):
        if not isinstance(raw_image, dict):
            raise PlannedReviewRenderError(f"snapshot.images[{position}] must be an object")
        image_path = _resolve_image_path(raw_image, snapshot_path.parent, manifest_dir)
        raw_image["path"] = str(image_path)
        _frame_number(raw_image, position)

    planned["counts"] = {
        "annotations_by_label": {
            name: class_counts.get(name, 0) for name in sorted(label_names_by_id.values())
        },
        "images": len(images),
        "shapes": len(shapes),
        "tags": len(tags),
        "tracks": len(tracks),
    }
    planned["final_gate"] = {
        "passed": False,
        "blocking_reasons": ["preview_only_not_live_cvat_state"],
        "warnings": [],
    }
    planned["planned_preview"] = {
        "preview_type": "full_review_apply_plan",
        "mutation_performed": False,
        "cvat_connection_performed": False,
        "source_snapshot": str(snapshot_path),
        "source_snapshot_sha256": snapshot_sha256,
        "source_review_pack": str(review_pack_path),
        "source_review_pack_sha256": review_pack_sha256,
        "source_decisions": str(decisions_path),
        "source_decisions_sha256": decisions_sha256,
        "planned_canonical_annotations_sha256": planned_annotation_sha,
        "expected_post_apply_canonical_sha256": expected_post_apply_sha256,
    }
    return planned


def render_planned_review(
    snapshot_path: Path | str,
    review_pack_path: Path | str,
    decisions_path: Path | str,
    output_dir: Path | str,
) -> Path:
    """Create a local-only planned snapshot and review pack in a new directory."""

    snapshot_path = Path(snapshot_path).resolve()
    review_pack_path = Path(review_pack_path).resolve()
    decisions_path = Path(decisions_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output path: {output_dir}")

    snapshot, snapshot_sha = _read_hashed_object(snapshot_path, "snapshot")
    review_pack, review_pack_sha = _read_hashed_object(review_pack_path, "review pack")
    decisions, decisions_sha = _read_hashed_object(decisions_path, "decisions")
    try:
        plan = build_apply_plan(
            snapshot,
            review_pack,
            decisions,
            snapshot_file_sha256=snapshot_sha,
            review_pack_file_sha256=review_pack_sha,
        )
        evidence = validate_review_evidence(decisions, decisions_path=decisions_path)
    except FinalReviewApplyError as error:
        raise PlannedReviewRenderError(str(error)) from error

    planned = _planned_snapshot(
        snapshot,
        plan["expected_annotations"],
        snapshot_path=snapshot_path,
        snapshot_sha256=snapshot_sha,
        review_pack_path=review_pack_path,
        review_pack_sha256=review_pack_sha,
        decisions_path=decisions_path,
        decisions_sha256=decisions_sha,
        expected_post_apply_sha256=plan["expected_post_apply_canonical_sha256"],
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(exist_ok=False)
    try:
        planned_snapshot_path = output_dir / "planned.snapshot.json"
        write_json_atomic(planned_snapshot_path, planned)
        planned_pack_path = build_review_pack(planned_snapshot_path, output_dir / "pack")
        planned_pack = json.loads(planned_pack_path.read_text(encoding="utf-8"))
        if planned_pack.get("annotation_sha256") != planned["canonical_annotations_sha256"]:
            raise PlannedReviewRenderError("planned review pack annotation hash is inconsistent")

        summary = {
            "schema": {"name": "roadlabelops.planned-full-review", "version": 1},
            "task_id": plan["task_id"],
            "mutation_performed": False,
            "cvat_connection_performed": False,
            "source_snapshot_sha256": snapshot_sha,
            "source_review_pack_sha256": review_pack_sha,
            "source_decisions_sha256": decisions_sha,
            "planned_snapshot": planned_snapshot_path.name,
            "planned_review_pack": str(planned_pack_path.relative_to(output_dir)),
            "planned_canonical_annotations_sha256": planned["canonical_annotations_sha256"],
            "expected_post_apply_canonical_sha256": plan[
                "expected_post_apply_canonical_sha256"
            ],
            "annotation_count_before": len(plan["snapshot_annotations"]["shapes"]),
            "annotation_count_after": len(plan["expected_annotations"]["shapes"]),
            "action_counts": plan["action_counts"],
            "mutation_action_count": plan["mutation_action_count"],
            "manual_delete_approval_count": plan["manual_delete_approval_count"],
            "review_evidence": evidence,
            "frame_count": planned_pack["frame_count"],
            "contact_sheets": planned_pack["contact_sheets"],
        }
        summary_path = output_dir / "plan-summary.json"
        write_json_atomic(summary_path, summary)
        return summary_path
    except Exception:
        shutil.rmtree(output_dir)
        raise


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("review_pack", type=Path)
    parser.add_argument("decisions", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        summary_path = render_planned_review(
            args.snapshot,
            args.review_pack,
            args.decisions,
            args.output,
        )
    except (FileExistsError, PlannedReviewRenderError) as error:
        parser.error(str(error))
    print(summary_path.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
