from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from test_evaluate_frozen_holdout import (
    FINAL_HOLDOUT_JOB_ID,
    FINAL_HOLDOUT_TASK_ID,
    Fixture,
    write_json,
)
from test_evaluate_frozen_holdout import fixture as original_evaluator_fixture

from scripts.build_frozen_holdout_protocol import (
    FrozenHoldoutProtocolBuildError,
    build_frozen_holdout_protocol,
)
from scripts.evaluate_frozen_holdout import EXPECTED_GATES, validate_protocol


@pytest.fixture
def evaluator_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Fixture:
    return original_evaluator_fixture.__wrapped__(tmp_path, monkeypatch)


def scene_map(fixture: Fixture, *, frame_step: int = 1) -> Path:
    path = fixture.root / f"scene-map-step-{frame_step}.json"
    write_json(
        path,
        {
            "frame_step": frame_step,
            "scenes": [
                {
                    "scene_id": "holdout-scene",
                    "video_path": str(fixture.videos[0]),
                }
            ],
        },
    )
    return path


def v2_scene_map(
    fixture: Fixture,
    *,
    path: Path,
    asset_sha: str | None = None,
    frame_sha: str | None = None,
) -> Path:
    video = fixture.videos[0]
    actual_sha = hashlib.sha256(video.read_bytes()).hexdigest()
    asset_sha = asset_sha or actual_sha
    frame_sha = frame_sha or actual_sha
    write_json(
        path,
        {
            "schema": {
                "name": "roadlabelops.training-source-frame-map",
                "version": 2,
            },
            "assets": [
                {
                    "asset_id": "holdout-scene",
                    "path": video.name,
                    "sha256": asset_sha,
                    "fps": 30,
                    "frame_count": 2,
                    "leakage_group_id": f"sha256:{asset_sha}",
                }
            ],
            "frames": [
                {
                    "asset_id": "holdout-scene",
                    "scene_id": "holdout-scene",
                    "source_frame": frame,
                    "normalized_asset_frame": frame,
                    "leakage_group_id": f"sha256:{frame_sha}",
                }
                for frame in (0, 1)
            ],
            "evidence": {},
        },
    )
    return path


def build_from_fixture(
    fixture: Fixture,
    *,
    output: Path,
    scene_map_path: Path,
    frame_step: int | None = None,
) -> dict[str, object]:
    expected = json.loads(fixture.protocol.read_text(encoding="utf-8"))
    return build_frozen_holdout_protocol(
        protocol_id=expected["protocol_id"],
        candidate_freeze=fixture.freeze,
        training_dataset_manifest=fixture.dataset_manifest,
        training_reference_manifest=fixture.training_manifest,
        baseline_weight=fixture.baseline_weight,
        baseline_model_id=expected["baseline"]["model_id"],
        holdout_manifest=fixture.holdout_manifest,
        holdout_annotations=fixture.holdout_coco,
        overlap_evidence=fixture.overlap,
        warmup_image=fixture.warmup,
        scene_map=scene_map_path,
        settings=expected["settings"],
        output=output,
        frame_step=frame_step,
    )


def test_builds_canonical_protocol_and_passes_evaluator_preflight(
    evaluator_fixture: Fixture,
) -> None:
    output = evaluator_fixture.root / "built-protocol.json"
    expected = json.loads(evaluator_fixture.protocol.read_text(encoding="utf-8"))

    summary = build_from_fixture(
        evaluator_fixture,
        output=output,
        scene_map_path=scene_map(evaluator_fixture),
    )

    actual = json.loads(output.read_text(encoding="utf-8"))
    assert actual == expected
    assert actual["gates"] == EXPECTED_GATES
    assert actual["holdout"]["evaluation_frames"] == sorted(
        actual["holdout"]["evaluation_frames"],
        key=lambda frame: (frame["scene_id"], frame["source_frame"]),
    )
    assert summary == {
        "output": str(output),
        "protocol_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "final_holdout_task_id": FINAL_HOLDOUT_TASK_ID,
        "final_holdout_job_id": FINAL_HOLDOUT_JOB_ID,
        "evaluation_frame_count": 2,
        "source_video_count": 1,
        "evaluator_preflight": "PASS",
    }
    validated = validate_protocol(output, evaluator_fixture.freeze)
    assert validated.validation["task_id"] == FINAL_HOLDOUT_TASK_ID
    assert validated.validation["job_id"] == FINAL_HOLDOUT_JOB_ID

    with pytest.raises(FileExistsError, match="already exists"):
        build_from_fixture(
            evaluator_fixture,
            output=output,
            scene_map_path=scene_map(evaluator_fixture),
        )


def test_accepts_v2_source_map_and_cross_checks_its_asset_binding(
    evaluator_fixture: Fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(evaluator_fixture.root)
    source_map = v2_scene_map(
        evaluator_fixture,
        path=evaluator_fixture.root / "source-map-v2.json",
    )
    output = evaluator_fixture.root / "built-from-v2-map.json"

    summary = build_from_fixture(
        evaluator_fixture,
        output=output,
        scene_map_path=source_map,
        frame_step=1,
    )

    assert summary["evaluator_preflight"] == "PASS"
    validate_protocol(output, evaluator_fixture.freeze)


@pytest.mark.parametrize(
    "schema",
    [
        {
            "name": "roadlabelops.training-source-frame-map",
            "version": 1,
        },
        {
            "name": "roadlabelops.tampered-source-frame-map",
            "version": 2,
        },
    ],
)
def test_rejects_declared_unsupported_source_map_schema_without_legacy_fallback(
    evaluator_fixture: Fixture,
    monkeypatch: pytest.MonkeyPatch,
    schema: dict[str, object],
) -> None:
    monkeypatch.chdir(evaluator_fixture.root)
    source_map = v2_scene_map(
        evaluator_fixture,
        path=evaluator_fixture.root / f"unsupported-schema-{schema['version']}.json",
    )
    payload = json.loads(source_map.read_text(encoding="utf-8"))
    payload["schema"] = schema
    payload["scenes"] = [
        {
            "scene_id": "holdout-scene",
            "video_path": str(evaluator_fixture.videos[0]),
        }
    ]
    write_json(source_map, payload)
    output = evaluator_fixture.root / f"unsupported-schema-{schema['version']}-protocol.json"

    with pytest.raises(FrozenHoldoutProtocolBuildError, match="unsupported scene map schema"):
        build_from_fixture(
            evaluator_fixture,
            output=output,
            scene_map_path=source_map,
            frame_step=1,
        )
    assert not output.exists()


@pytest.mark.parametrize("tamper", ["asset_sha", "frame_leakage"])
def test_rejects_tampered_v2_source_map_identity(
    evaluator_fixture: Fixture,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    monkeypatch.chdir(evaluator_fixture.root)
    wrong_sha = "f" * 64
    source_map = v2_scene_map(
        evaluator_fixture,
        path=evaluator_fixture.root / f"tampered-{tamper}.json",
        asset_sha=wrong_sha if tamper == "asset_sha" else None,
        frame_sha=wrong_sha if tamper == "frame_leakage" else None,
    )
    output = evaluator_fixture.root / f"tampered-{tamper}-protocol.json"

    with pytest.raises(FrozenHoldoutProtocolBuildError, match="source identity"):
        build_from_fixture(
            evaluator_fixture,
            output=output,
            scene_map_path=source_map,
            frame_step=1,
        )
    assert not output.exists()


def test_rejects_unreachable_stride_without_publishing(
    evaluator_fixture: Fixture,
) -> None:
    output = evaluator_fixture.root / "unreachable-protocol.json"
    with pytest.raises(FrozenHoldoutProtocolBuildError, match="unreachable"):
        build_from_fixture(
            evaluator_fixture,
            output=output,
            scene_map_path=scene_map(evaluator_fixture, frame_step=2),
        )
    assert not output.exists()


def test_generated_protocol_is_rejected_before_publish_if_holdout_identity_differs(
    evaluator_fixture: Fixture,
) -> None:
    manifest = json.loads(evaluator_fixture.holdout_manifest.read_text(encoding="utf-8"))
    manifest["task_id"] = 91002
    write_json(evaluator_fixture.holdout_manifest, manifest)
    evaluator_fixture.rebuild_overlap()
    output = evaluator_fixture.root / "wrong-task-protocol.json"

    with pytest.raises(
        FrozenHoldoutProtocolBuildError,
        match="configured final holdout|holdout identity",
    ):
        build_from_fixture(
            evaluator_fixture,
            output=output,
            scene_map_path=scene_map(evaluator_fixture),
        )
    assert not output.exists()


def test_rejects_scene_video_paths_swapped_between_scene_ids(
    evaluator_fixture: Fixture,
) -> None:
    second_video = evaluator_fixture.root / "holdout-scene-2.mp4"
    second_video.write_bytes(b"synthetic-video-for-second-scene")
    coco = json.loads(evaluator_fixture.holdout_coco.read_text(encoding="utf-8"))
    coco["images"][1].update(
        {
            "scene_id": "holdout-scene-2",
            "source_frame": 0,
            "source_asset_id": "holdout-asset-2",
            "source_leakage_group_id": (
                f"sha256:{hashlib.sha256(second_video.read_bytes()).hexdigest()}"
            ),
            "source_normalized_asset_frame": 0,
        }
    )
    write_json(evaluator_fixture.holdout_coco, coco)
    evaluator_fixture.rebuild_holdout_manifest()
    evaluator_fixture.rebuild_overlap()
    evaluator_fixture.rebuild_protocol()
    swapped_map = evaluator_fixture.root / "swapped-scene-map.json"
    write_json(
        swapped_map,
        {
            "frame_step": 1,
            "scenes": [
                {
                    "scene_id": "holdout-scene",
                    "video_path": str(second_video),
                },
                {
                    "scene_id": "holdout-scene-2",
                    "video_path": str(evaluator_fixture.videos[0]),
                },
            ],
        },
    )
    output = evaluator_fixture.root / "swapped-protocol.json"

    with pytest.raises(
        FrozenHoldoutProtocolBuildError,
        match="differs from the holdout COCO source identity",
    ):
        build_from_fixture(
            evaluator_fixture,
            output=output,
            scene_map_path=swapped_map,
        )
    assert not output.exists()
