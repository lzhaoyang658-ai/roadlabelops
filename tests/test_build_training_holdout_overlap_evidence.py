from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_evaluate_frozen_holdout import Fixture, digest, write_json
from test_evaluate_frozen_holdout import fixture as original_evaluator_fixture

from scripts import evaluate_frozen_holdout as evaluator
from scripts.build_training_holdout_overlap_evidence import (
    OverlapEvidenceError,
    _reference,
    build_overlap_evidence,
)


@pytest.fixture
def evaluator_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Fixture:
    return original_evaluator_fixture.__wrapped__(tmp_path, monkeypatch)


def test_builds_exact_evaluator_overlap_payload(evaluator_fixture: Fixture) -> None:
    output = evaluator_fixture.root / "built-overlap.json"

    payload = build_overlap_evidence(
        evaluator_fixture.training_manifest,
        evaluator_fixture.holdout_manifest,
        output,
    )

    expected = json.loads(evaluator_fixture.overlap.read_text(encoding="utf-8"))
    assert payload == expected
    assert json.loads(output.read_text(encoding="utf-8")) == expected
    assert payload["gate_result"] == "PASS"
    training_image = json.loads(evaluator_fixture.training_coco.read_text(encoding="utf-8"))[
        "images"
    ][0]
    expected_frame_digest = evaluator._universe_digest(
        {
            (
                training_image["source_leakage_group_id"].removeprefix("sha256:"),
                training_image["source_normalized_asset_frame"],
            )
        }
    )
    assert payload["computed"]["training_frame_keys_sha256"] == expected_frame_digest

    training = _reference(
        evaluator_fixture.training_manifest,
        schema=evaluator.TRAINING_SCHEMA,
        location="training reference manifest",
        verify_all_files=False,
    )
    holdout = _reference(
        evaluator_fixture.holdout_manifest,
        schema=evaluator.HOLDOUT_SCHEMA,
        location="holdout manifest",
        verify_all_files=True,
    )
    validated = evaluator._validate_overlap_evidence(
        evaluator.BoundFile(output, digest(output), output.stat().st_size),
        training=training,
        holdout=holdout,
    )
    assert validated == payload["computed"]


def test_publishes_fail_when_any_exact_universe_overlaps(
    evaluator_fixture: Fixture,
) -> None:
    training = json.loads(evaluator_fixture.training_coco.read_text(encoding="utf-8"))
    holdout = json.loads(evaluator_fixture.holdout_coco.read_text(encoding="utf-8"))
    training["images"][0]["source_asset_id"] = holdout["images"][0]["source_asset_id"]
    write_json(evaluator_fixture.training_coco, training)
    evaluator_fixture.rebuild_training_manifest()
    output = evaluator_fixture.root / "overlap-fail.json"

    payload = build_overlap_evidence(
        evaluator_fixture.training_manifest,
        evaluator_fixture.holdout_manifest,
        output,
    )

    assert payload["gate_result"] == "FAIL"
    assert payload["computed"]["asset_id_overlap_count"] == 1
    assert payload["computed"]["image_sha256_overlap_count"] == 0


def test_publishes_fail_for_exported_image_hash_overlap(
    evaluator_fixture: Fixture,
) -> None:
    training_image = evaluator_fixture.training_manifest.parent / "images" / "train.jpg"
    training_image.write_bytes(evaluator_fixture.holdout_images[0].read_bytes())
    training = json.loads(evaluator_fixture.training_coco.read_text(encoding="utf-8"))
    training["images"][0]["sha256"] = digest(training_image)
    write_json(evaluator_fixture.training_coco, training)
    evaluator_fixture.rebuild_training_manifest()
    manifest = json.loads(evaluator_fixture.training_manifest.read_text(encoding="utf-8"))
    manifest["files"][1] = {
        "path": "images/train.jpg",
        "sha256": digest(training_image),
        "size_bytes": training_image.stat().st_size,
    }
    write_json(evaluator_fixture.training_manifest, manifest)

    payload = build_overlap_evidence(
        evaluator_fixture.training_manifest,
        evaluator_fixture.holdout_manifest,
        evaluator_fixture.root / "image-overlap.json",
    )

    assert payload["gate_result"] == "FAIL"
    assert payload["computed"]["image_sha256_overlap_count"] == 1
    assert payload["computed"]["source_sha256_overlap_count"] == 0


def test_publishes_fail_for_source_hash_without_frame_overlap(
    evaluator_fixture: Fixture,
) -> None:
    training = json.loads(evaluator_fixture.training_coco.read_text(encoding="utf-8"))
    holdout = json.loads(evaluator_fixture.holdout_coco.read_text(encoding="utf-8"))
    training["images"][0]["source_leakage_group_id"] = holdout["images"][0][
        "source_leakage_group_id"
    ]
    training["images"][0]["source_normalized_asset_frame"] = 999
    write_json(evaluator_fixture.training_coco, training)
    evaluator_fixture.rebuild_training_manifest()

    payload = build_overlap_evidence(
        evaluator_fixture.training_manifest,
        evaluator_fixture.holdout_manifest,
        evaluator_fixture.root / "source-overlap.json",
    )

    assert payload["gate_result"] == "FAIL"
    assert payload["computed"]["source_sha256_overlap_count"] == 1
    assert payload["computed"]["frame_overlap_count"] == 0


def test_publishes_fail_for_normalized_source_frame_overlap(
    evaluator_fixture: Fixture,
) -> None:
    training = json.loads(evaluator_fixture.training_coco.read_text(encoding="utf-8"))
    holdout = json.loads(evaluator_fixture.holdout_coco.read_text(encoding="utf-8"))
    training["images"][0]["source_leakage_group_id"] = holdout["images"][0][
        "source_leakage_group_id"
    ]
    training["images"][0]["source_normalized_asset_frame"] = holdout["images"][0][
        "source_normalized_asset_frame"
    ]
    write_json(evaluator_fixture.training_coco, training)
    evaluator_fixture.rebuild_training_manifest()

    payload = build_overlap_evidence(
        evaluator_fixture.training_manifest,
        evaluator_fixture.holdout_manifest,
        evaluator_fixture.root / "frame-overlap.json",
    )

    assert payload["gate_result"] == "FAIL"
    assert payload["computed"]["source_sha256_overlap_count"] == 1
    assert payload["computed"]["frame_overlap_count"] == 1


def test_refuses_existing_or_broken_symlink_output(evaluator_fixture: Fixture) -> None:
    existing = evaluator_fixture.root / "existing.json"
    existing.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        build_overlap_evidence(
            evaluator_fixture.training_manifest,
            evaluator_fixture.holdout_manifest,
            existing,
        )
    assert existing.read_text(encoding="utf-8") == "keep"

    broken = evaluator_fixture.root / "broken-output.json"
    broken.symlink_to(evaluator_fixture.root / "missing-target.json")
    with pytest.raises(FileExistsError, match="already exists"):
        build_overlap_evidence(
            evaluator_fixture.training_manifest,
            evaluator_fixture.holdout_manifest,
            broken,
        )
    assert broken.is_symlink()


def test_rejects_manifest_leaf_symlink(evaluator_fixture: Fixture) -> None:
    linked = evaluator_fixture.root / "linked-training-manifest.json"
    linked.symlink_to(evaluator_fixture.training_manifest)

    with pytest.raises(OverlapEvidenceError, match="could not open training reference"):
        build_overlap_evidence(
            linked,
            evaluator_fixture.holdout_manifest,
            evaluator_fixture.root / "not-published.json",
        )
