from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import evaluate_frozen_holdout as evaluator
from scripts.evaluate_frozen_holdout import (
    FrozenHoldoutError,
    ModelRun,
    RuntimeSourceVideo,
    SourceVideo,
    canonical_sha256,
    evaluate,
    validate_protocol,
)

FINAL_HOLDOUT_TASK_ID = 91001
FINAL_HOLDOUT_JOB_ID = 92001


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def digest(path: Path) -> str:
    return evaluator.hashlib.sha256(path.read_bytes()).hexdigest()


def bound(path: Path, owner: Path) -> dict[str, object]:
    return {
        "path": os.path.relpath(path, owner.parent),
        "sha256": digest(path),
        "size_bytes": path.stat().st_size,
    }


def test_read_regular_bytes_accepts_stable_regular_file(tmp_path: Path) -> None:
    source = tmp_path / "stable.bin"
    expected = b"stable contents"
    source.write_bytes(expected)

    assert evaluator._read_regular_bytes(source, "stable input") == expected


def test_read_regular_bytes_rejects_equal_size_in_place_rewrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "mutable.bin"
    original = b"original contents"
    replacement = b"rewritten content"
    assert len(original) == len(replacement)
    source.write_bytes(original)
    initial = source.stat()
    real_read = os.read
    rewritten = False

    def rewrite_after_first_read(descriptor: int, size: int) -> bytes:
        nonlocal rewritten
        chunk = real_read(descriptor, size)
        if chunk and not rewritten:
            with source.open("r+b") as stream:
                assert os.fstat(stream.fileno()).st_ino == initial.st_ino
                stream.write(replacement)
                stream.flush()
                os.fsync(stream.fileno())
            rewritten = True
        return chunk

    monkeypatch.setattr(evaluator.os, "read", rewrite_after_first_read)

    with pytest.raises(FrozenHoldoutError, match="changed while being read"):
        evaluator._read_regular_bytes(source, "mutable input")

    assert rewritten is True
    final = source.stat()
    assert final.st_ino == initial.st_ino
    assert final.st_size == initial.st_size


def categories() -> list[dict[str, object]]:
    return [
        {"id": index, "name": name}
        for index, name in enumerate(evaluator.CANONICAL_LABELS, start=1)
    ]


def image_record(
    identifier: int,
    file_name: str,
    *,
    scene_id: str,
    source_frame: int,
    image_sha256: str,
    asset_id: str,
    source_sha256: str,
    normalized_frame: int,
) -> dict[str, object]:
    return {
        "id": identifier,
        "file_name": file_name,
        "width": 20,
        "height": 12,
        "scene_id": scene_id,
        "source_frame": source_frame,
        "sha256": image_sha256,
        "source_asset_id": asset_id,
        "source_leakage_group_id": f"sha256:{source_sha256}",
        "source_normalized_asset_frame": normalized_frame,
    }


@dataclass
class Fixture:
    root: Path
    protocol: Path
    freeze: Path
    output: Path
    training_coco: Path
    training_manifest: Path
    dataset_manifest: Path
    holdout_coco: Path
    holdout_manifest: Path
    completion_receipt: Path
    post_snapshot: Path
    final_decisions: Path
    overlap: Path
    candidate_weight: Path
    baseline_weight: Path
    warmup: Path
    holdout_images: tuple[Path, ...]
    videos: tuple[Path, ...]

    @property
    def claim_path(self) -> Path:
        return self.holdout_manifest.parent / (
            f".roadlabelops-frozen-holdout-{digest(self.holdout_manifest)}.consumed.json"
        )

    def rebuild_training_manifest(self) -> None:
        payload = json.loads(self.training_manifest.read_text(encoding="utf-8"))
        payload["files"][0] = {
            "path": "annotations.coco.json",
            "sha256": digest(self.training_coco),
            "size_bytes": self.training_coco.stat().st_size,
        }
        write_json(self.training_manifest, payload)

    def rebuild_holdout_manifest(self) -> None:
        payload = json.loads(self.holdout_manifest.read_text(encoding="utf-8"))
        payload["files"][0] = {
            "path": "annotations.coco.json",
            "sha256": digest(self.holdout_coco),
            "size_bytes": self.holdout_coco.stat().st_size,
        }
        write_json(self.holdout_manifest, payload)

    def rebuild_review_evidence_bindings(self) -> None:
        payload = json.loads(self.holdout_manifest.read_text(encoding="utf-8"))
        review = payload["review_completion"]
        paths = {
            "completion_receipt": self.completion_receipt,
            "post_snapshot": self.post_snapshot,
            "final_decisions": self.final_decisions,
        }
        for key, path in paths.items():
            previous_path = review[key]["path"]
            record = bound(path, self.holdout_manifest)
            review[key] = record
            matches = [
                index
                for index, managed in enumerate(payload["files"])
                if managed["path"] == previous_path
            ]
            assert len(matches) == 1
            payload["files"][matches[0]] = record
        write_json(self.holdout_manifest, payload)

    def rebuild_dataset_manifest(self) -> None:
        payload = json.loads(self.dataset_manifest.read_text(encoding="utf-8"))
        source = json.loads(self.training_coco.read_text(encoding="utf-8"))["images"][0]
        payload["inputs"]["reference_manifest"] = {
            "file_name": "manifest.json",
            "schema": evaluator.TRAINING_SCHEMA,
            "sha256": digest(self.training_manifest),
        }
        payload["inputs"]["coco"] = {
            "file_name": "annotations.coco.json",
            "semantic_sha256": "3" * 64,
            "sha256": digest(self.training_coco),
        }
        payload["source_statistics"]["assets"] = [
            {
                "asset_id": source["source_asset_id"],
                "leakage_group_id": source["source_leakage_group_id"],
            }
        ]
        write_json(self.dataset_manifest, payload)

    def rebuild_freeze(self) -> None:
        receipt = self.freeze / "receipt.json"
        protocol = {
            "schema": {
                "name": "roadlabelops.yolo-candidate-training-protocol",
                "version": 1,
            },
            "model_family": "YOLO11n",
            "taxonomy": list(evaluator.CANONICAL_LABELS),
            "training": {"epochs": 10},
            "validation_selection": {
                "primary": "mAP50-95",
                "tie_breakers": ["mAP50", "recall", "precision", "smaller_seed"],
            },
            "holdout_access": "prohibited",
        }
        metrics = {
            "precision": 0.8,
            "recall": 0.8,
            "map50": 0.8,
            "map50_95": 0.7,
            "fitness": 0.7,
        }
        dataset = {
            "manifest": {
                "schema": evaluator.YOLO_DATASET_SCHEMA,
                "sha256": digest(self.dataset_manifest),
                "size_bytes": self.dataset_manifest.stat().st_size,
            },
            "dataset_yaml": {"sha256": "4" * 64, "size_bytes": 10},
            "managed_files_sha256": "5" * 64,
            "managed_file_count": 3,
            "counts": {"images": {"total": 1}},
        }
        base_weights = {
            "file_name": "yolo11n.pt",
            "model_family": "YOLO11n",
            "sha256": digest(self.baseline_weight),
            "size_bytes": self.baseline_weight.stat().st_size,
        }

        def source(seed: int) -> dict[str, object]:
            return {
                "kind": "content_bound_candidate_directory",
                "directory_name": f"candidate-{seed}",
                "seed": seed,
                "receipt": {
                    "path": "receipt.json",
                    "sha256": evaluator.hashlib.sha256(str(seed).encode()).hexdigest(),
                    "size_bytes": 100,
                },
            }

        rankings = []
        for rank, seed in enumerate((42, 43, 44), start=1):
            rankings.append(
                {
                    "rank": rank,
                    "seed": seed,
                    "metrics": metrics,
                    "source": source(seed),
                    "best_weights": {
                        "path": "weights/best.pt",
                        "sha256": digest(self.candidate_weight) if rank == 1 else f"{rank}" * 64,
                        "size_bytes": self.candidate_weight.stat().st_size,
                    },
                    "contract": {
                        "protocol_sha256": canonical_sha256(protocol),
                        "dataset": dataset,
                        "base_weights": base_weights,
                        "taxonomy": list(evaluator.CANONICAL_LABELS),
                    },
                }
            )
        payload = {
            "schema": evaluator.FREEZE_SCHEMA,
            "selection_order": list(evaluator.SELECTION_ORDER),
            "selected_seed": 42,
            "selected_source": source(42),
            "selected_metrics": metrics,
            "selected_weight": bound(self.candidate_weight, receipt),
            "candidate_rankings": rankings,
            "contracts": {
                "candidate_count": 3,
                "protocol": protocol,
                "protocol_sha256": canonical_sha256(protocol),
                "resolved_args_except_seed": {"epochs": 10},
                "dataset": dataset,
                "base_weights": base_weights,
                "taxonomy": list(evaluator.CANONICAL_LABELS),
            },
            "holdout_input_read": False,
            "holdout_statement": evaluator.NO_HOLDOUT_STATEMENT,
        }
        write_json(receipt, payload)

    def rebuild_overlap(self) -> None:
        training = json.loads(self.training_coco.read_text(encoding="utf-8"))["images"]
        holdout = json.loads(self.holdout_coco.read_text(encoding="utf-8"))["images"]

        def universes(records: list[dict[str, object]]) -> tuple[set, set, set, set]:
            assets = {
                (type(record["source_asset_id"]).__name__, str(record["source_asset_id"]))
                for record in records
            }
            image_hashes = {str(record["sha256"]) for record in records}
            sources = {
                str(record["source_leakage_group_id"]).removeprefix("sha256:") for record in records
            }
            frames = {
                (
                    str(record["source_leakage_group_id"]).removeprefix("sha256:"),
                    int(record["source_normalized_asset_frame"]),
                )
                for record in records
            }
            return assets, image_hashes, sources, frames

        training_assets, training_images, training_sources, training_frames = universes(training)
        holdout_assets, holdout_images, holdout_sources, holdout_frames = universes(holdout)
        computed = {
            "training_asset_ids_sha256": evaluator._universe_digest(training_assets),
            "holdout_asset_ids_sha256": evaluator._universe_digest(holdout_assets),
            "training_image_sha256s_sha256": evaluator._universe_digest(training_images),
            "holdout_image_sha256s_sha256": evaluator._universe_digest(holdout_images),
            "training_source_sha256s_sha256": evaluator._universe_digest(training_sources),
            "holdout_source_sha256s_sha256": evaluator._universe_digest(holdout_sources),
            "training_frame_keys_sha256": evaluator._universe_digest(training_frames),
            "holdout_frame_keys_sha256": evaluator._universe_digest(holdout_frames),
            "asset_id_overlap_count": len(training_assets & holdout_assets),
            "image_sha256_overlap_count": len(training_images & holdout_images),
            "source_sha256_overlap_count": len(training_sources & holdout_sources),
            "frame_overlap_count": len(training_frames & holdout_frames),
        }
        write_json(
            self.overlap,
            {
                "schema": evaluator.OVERLAP_SCHEMA,
                "training_reference_manifest_sha256": digest(self.training_manifest),
                "holdout_manifest_sha256": digest(self.holdout_manifest),
                "computed": computed,
                "gate_result": "PASS",
            },
        )

    def rebuild_protocol(self) -> None:
        payload = json.loads(self.protocol.read_text(encoding="utf-8"))
        receipt = self.freeze / "receipt.json"
        payload["candidate_freeze"] = {
            "sha256": digest(receipt),
            "size_bytes": receipt.stat().st_size,
        }
        payload["training_dataset_manifest"] = bound(self.dataset_manifest, self.protocol)
        payload["training_reference_manifest"] = bound(self.training_manifest, self.protocol)
        payload["baseline"]["weight"] = bound(self.baseline_weight, self.protocol)
        payload["holdout"]["manifest"] = bound(self.holdout_manifest, self.protocol)
        payload["holdout"]["annotations"] = bound(self.holdout_coco, self.protocol)
        payload["overlap_evidence"] = bound(self.overlap, self.protocol)
        payload["warmup_image"] = bound(self.warmup, self.protocol)
        for record, video in zip(payload["holdout"]["source_videos"], self.videos, strict=True):
            record["file"] = bound(video, self.protocol)
        write_json(self.protocol, payload)

    def rebuild_all_bindings(self) -> None:
        self.rebuild_training_manifest()
        self.rebuild_dataset_manifest()
        self.rebuild_holdout_manifest()
        self.rebuild_freeze()
        self.rebuild_overlap()
        self.rebuild_protocol()


@pytest.fixture
def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Fixture:
    monkeypatch.setenv(
        "ROADLABELOPS_FINAL_HOLDOUT_TASK_IDS",
        str(FINAL_HOLDOUT_TASK_ID),
    )
    monkeypatch.setenv(
        "ROADLABELOPS_FINAL_HOLDOUT_JOB_IDS",
        str(FINAL_HOLDOUT_JOB_ID),
    )
    training_root = tmp_path / "training"
    training_image = training_root / "images" / "train.jpg"
    training_image.parent.mkdir(parents=True)
    training_image.write_bytes(b"training-image")
    training_coco = training_root / "annotations.coco.json"
    training_source_sha = "1" * 64
    write_json(
        training_coco,
        {
            "categories": categories(),
            "images": [
                image_record(
                    1,
                    "images/train.jpg",
                    scene_id="train-scene",
                    source_frame=0,
                    image_sha256=digest(training_image),
                    asset_id="train-asset",
                    source_sha256=training_source_sha,
                    normalized_frame=0,
                )
            ],
            "annotations": [],
        },
    )
    training_manifest = training_root / "manifest.json"
    write_json(
        training_manifest,
        {
            "schema": evaluator.TRAINING_SCHEMA,
            "files": [
                {
                    "path": "annotations.coco.json",
                    "sha256": digest(training_coco),
                    "size_bytes": training_coco.stat().st_size,
                },
                {
                    "path": "images/train.jpg",
                    "sha256": digest(training_image),
                    "size_bytes": training_image.stat().st_size,
                },
            ],
        },
    )
    dataset_manifest = tmp_path / "yolo" / "manifest.json"
    write_json(
        dataset_manifest,
        {
            "schema": evaluator.YOLO_DATASET_SCHEMA,
            "gate": {"passed": True, "checks": {"synthetic_fixture_verified": True}},
            "inputs": {
                "reference_manifest": {
                    "file_name": "manifest.json",
                    "schema": evaluator.TRAINING_SCHEMA,
                    "sha256": digest(training_manifest),
                },
                "coco": {
                    "file_name": "annotations.coco.json",
                    "semantic_sha256": "3" * 64,
                    "sha256": digest(training_coco),
                },
            },
            "source_statistics": {
                "assets": [
                    {
                        "asset_id": "train-asset",
                        "leakage_group_id": f"sha256:{training_source_sha}",
                    }
                ]
            },
        },
    )

    video = tmp_path / "holdout-scene.mp4"
    video.write_bytes(b"synthetic-video")
    holdout_root = tmp_path / "holdout"
    first_image = holdout_root / "images" / "holdout-0.jpg"
    second_image = holdout_root / "images" / "holdout-1.jpg"
    first_image.parent.mkdir(parents=True)
    first_image.write_bytes(b"holdout-image-0")
    second_image.write_bytes(b"holdout-image-1-empty")
    holdout_coco = holdout_root / "annotations.coco.json"
    holdout_source_sha = digest(video)
    write_json(
        holdout_coco,
        {
            "categories": categories(),
            "images": [
                image_record(
                    1,
                    "images/holdout-0.jpg",
                    scene_id="holdout-scene",
                    source_frame=0,
                    image_sha256=digest(first_image),
                    asset_id="holdout-asset",
                    source_sha256=holdout_source_sha,
                    normalized_frame=0,
                ),
                image_record(
                    2,
                    "images/holdout-1.jpg",
                    scene_id="holdout-scene",
                    source_frame=1,
                    image_sha256=digest(second_image),
                    asset_id="holdout-asset",
                    source_sha256=holdout_source_sha,
                    normalized_frame=1,
                ),
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [1.0, 1.0, 8.0, 6.0],
                    "area": 48.0,
                    "iscrowd": 0,
                }
            ],
        },
    )
    evidence_root = holdout_root / "evidence"
    final_decisions = evidence_root / "final-decisions.json"
    write_json(
        final_decisions,
        {
            "schema_version": "1.3",
            "scope": "full_review",
            "mutation_performed": False,
            "task_id": FINAL_HOLDOUT_TASK_ID,
            "frame_reviews": [
                {"frame": 0, "reviewed": True, "actions": [], "resolved_flag_ids": []},
                {"frame": 1, "reviewed": True, "actions": [], "resolved_flag_ids": []},
            ],
        },
    )
    source_annotation_sha = "a" * 64
    post_snapshot = evidence_root / "post-snapshot.json"
    write_json(
        post_snapshot,
        {
            "snapshot_schema": evaluator.CVAT_SNAPSHOT_SCHEMA,
            "task": {"id": FINAL_HOLDOUT_TASK_ID, "size": 2},
            "manifest": {"cvat_task_id": FINAL_HOLDOUT_TASK_ID, "sample_size": 2},
            "jobs": [
                {
                    "id": FINAL_HOLDOUT_JOB_ID,
                    "task_id": FINAL_HOLDOUT_TASK_ID,
                    "start_frame": 0,
                    "stop_frame": 1,
                    "frame_count": 2,
                    "type": "annotation",
                    "parent_job_id": None,
                    "issues": {"count": 0},
                    "stage": "annotation",
                    "state": "new",
                }
            ],
            "final_gate": {"passed": True, "blocking_reasons": []},
            "canonical_annotations_sha256": source_annotation_sha,
        },
    )
    completion_receipt = evidence_root / "completion-receipt.json"
    write_json(
        completion_receipt,
        {
            "schema": evaluator.COMPLETION_RECEIPT_SCHEMA,
            "task_id": FINAL_HOLDOUT_TASK_ID,
            "job_id": FINAL_HOLDOUT_JOB_ID,
            "dry_run": False,
            "mutation_performed": True,
            "job_state_after": "completed",
            "annotation_count": 1,
            "verified_post_completion_canonical_annotations_sha256": source_annotation_sha,
            "decision_validation": {
                "valid": True,
                "mutation_performed": False,
                "scope": "full_review",
                "task_id": FINAL_HOLDOUT_TASK_ID,
                "snapshot_frame_count": 2,
                "reviewed_frame_count": 2,
                "unresolved_manual_flag_count": 0,
            },
            "evidence": {
                "post_snapshot": {
                    "path": str(post_snapshot),
                    "sha256": digest(post_snapshot),
                },
                "decisions": {
                    "path": str(final_decisions),
                    "sha256": digest(final_decisions),
                },
            },
        },
    )
    holdout_manifest = holdout_root / "manifest.json"
    write_json(
        holdout_manifest,
        {
            "schema": evaluator.HOLDOUT_SCHEMA,
            "task_id": FINAL_HOLDOUT_TASK_ID,
            "review_completion": {
                "completion_receipt": bound(completion_receipt, holdout_manifest),
                "post_snapshot": bound(post_snapshot, holdout_manifest),
                "final_decisions": bound(final_decisions, holdout_manifest),
                "source_canonical_annotations_sha256": source_annotation_sha,
            },
            "files": [
                {
                    "path": "annotations.coco.json",
                    "sha256": digest(holdout_coco),
                    "size_bytes": holdout_coco.stat().st_size,
                },
                {
                    "path": "images/holdout-0.jpg",
                    "sha256": digest(first_image),
                    "size_bytes": first_image.stat().st_size,
                },
                {
                    "path": "images/holdout-1.jpg",
                    "sha256": digest(second_image),
                    "size_bytes": second_image.stat().st_size,
                },
                bound(completion_receipt, holdout_manifest),
                bound(post_snapshot, holdout_manifest),
                bound(final_decisions, holdout_manifest),
            ],
        },
    )

    freeze = tmp_path / "candidate-freeze"
    freeze.mkdir()
    candidate_weight = freeze / "best.pt"
    baseline_weight = tmp_path / "baseline.pt"
    warmup = tmp_path / "warmup.jpg"
    candidate_weight.write_bytes(b"candidate-weight")
    baseline_weight.write_bytes(b"baseline-weight")
    warmup.write_bytes(b"external-warmup")

    overlap = tmp_path / "overlap.json"
    frames = [
        {"scene_id": "holdout-scene", "source_frame": 0},
        {"scene_id": "holdout-scene", "source_frame": 1},
    ]
    protocol = tmp_path / "protocol.json"
    write_json(
        protocol,
        {
            "schema": evaluator.PROTOCOL_SCHEMA,
            "protocol_id": "synthetic-final-holdout-v1",
            "mode": "production_scene_videos",
            "candidate_freeze": {"sha256": "0" * 64, "size_bytes": 1},
            "training_dataset_manifest": bound(dataset_manifest, protocol),
            "training_reference_manifest": bound(training_manifest, protocol),
            "baseline": {
                "model_id": "baseline-yolo11n",
                "weight": bound(baseline_weight, protocol),
            },
            "holdout": {
                "task_id": FINAL_HOLDOUT_TASK_ID,
                "job_id": FINAL_HOLDOUT_JOB_ID,
                "manifest": bound(holdout_manifest, protocol),
                "annotations": bound(holdout_coco, protocol),
                "evaluation_frames": frames,
                "evaluation_frames_sha256": canonical_sha256(frames),
                "source_videos": [
                    {
                        "scene_id": "holdout-scene",
                        "frame_step": 1,
                        "file": bound(video, protocol),
                    }
                ],
            },
            "overlap_evidence": {"path": "overlap.json", "sha256": "0" * 64, "size_bytes": 1},
            "warmup_image": bound(warmup, protocol),
            "settings": {
                "confidence": 0.4,
                "image_size": 640,
                "device": "cpu",
                "nms_iou": 0.75,
                "rider_overlap": 0.25,
                "match_iou": 0.5,
            },
            "gates": evaluator.EXPECTED_GATES,
        },
    )
    result = Fixture(
        tmp_path,
        protocol,
        freeze,
        tmp_path / "result.json",
        training_coco,
        training_manifest,
        dataset_manifest,
        holdout_coco,
        holdout_manifest,
        completion_receipt,
        post_snapshot,
        final_decisions,
        overlap,
        candidate_weight,
        baseline_weight,
        warmup,
        (first_image, second_image),
        (video,),
    )
    result.rebuild_freeze()
    result.rebuild_overlap()
    result.rebuild_protocol()
    return result


def exact_runner(
    _weight: Path,
    _warmup: Path,
    videos: tuple[RuntimeSourceVideo, ...],
    _settings: dict,
    role: str,
) -> ModelRun:
    observed = frozenset(
        (video.scene_id, frame) for video in videos for frame in video.target_frames
    )
    prediction = {
        "prediction_id": f"{role}-prediction",
        "scene_id": "holdout-scene",
        "frame": 0,
        "label": "car" if role == "candidate" else "bus",
        "confidence": 0.9,
        "bbox": [1.0, 1.0, 9.0, 7.0],
        "source": "auto",
    }
    return ModelRun((prediction,), observed, 2, 0.01, {"test": True})


def test_default_is_read_only_preflight_and_counts_empty_frames(fixture: Fixture) -> None:
    called = False

    def forbidden_runner(*_args) -> ModelRun:
        nonlocal called
        called = True
        raise AssertionError("dry-run must not invoke a model")

    result = evaluate(
        fixture.protocol,
        fixture.freeze,
        fixture.output,
        runner=forbidden_runner,
    )

    assert result["dry_run"] is True
    assert result["inference_performed"] is False
    assert result["gate_result"] == "PASS"
    assert result["validation"]["evaluation_frame_count"] == 2
    assert result["validation"]["zero_annotation_frame_count"] == 1
    assert called is False
    assert not fixture.output.exists()
    assert not fixture.claim_path.exists()


def test_apply_uses_same_frozen_batch_and_publishes_passing_result(fixture: Fixture) -> None:
    calls: list[tuple[str, tuple[tuple[str, str], ...], bytes]] = []

    def recording_runner(weight, _warmup, videos, _settings, role):
        calls.append(
            (
                role,
                tuple((video.scene_id, digest(video.file.path)) for video in videos),
                weight.read_bytes(),
            )
        )
        return exact_runner(weight, _warmup, videos, _settings, role)

    result = evaluate(
        fixture.protocol,
        fixture.freeze,
        fixture.output,
        apply=True,
        runner=recording_runner,
    )

    assert [call[0] for call in calls] == ["baseline", "candidate"]
    assert calls[0][1] == calls[1][1]
    assert calls[0][2] == b"baseline-weight"
    assert calls[1][2] == b"candidate-weight"
    assert result["gate_result"] == "PASS"
    assert result["results"][1]["quality"]["clean_frame_rate"] == 1.0
    assert result["results"][1]["quality"]["evaluated_frame_count"] == 2
    assert tuple(result["results"][1]["quality"]["per_class"]) == evaluator.CANONICAL_LABELS
    assert result["promotion_gates"]["candidate_f1_strictly_greater_than_baseline"] is True
    assert fixture.output.is_file()
    assert fixture.claim_path.is_file()


def test_apply_is_single_use_even_after_success(fixture: Fixture) -> None:
    evaluate(fixture.protocol, fixture.freeze, fixture.output, apply=True, runner=exact_runner)

    with pytest.raises(FrozenHoldoutError, match="already consumed|output exists"):
        evaluate(fixture.protocol, fixture.freeze, fixture.output, apply=True, runner=exact_runner)
    with pytest.raises(FrozenHoldoutError, match="already consumed"):
        evaluate(
            fixture.protocol,
            fixture.freeze,
            fixture.root / "different-output.json",
            apply=True,
            runner=exact_runner,
        )


def test_relocating_candidate_freeze_cannot_bypass_single_use(fixture: Fixture) -> None:
    evaluate(fixture.protocol, fixture.freeze, fixture.output, apply=True, runner=exact_runner)
    relocated = fixture.root / "another-registry" / "candidate-freeze"
    relocated.parent.mkdir()
    shutil.copytree(fixture.freeze, relocated)

    with pytest.raises(FrozenHoldoutError, match="already consumed"):
        evaluate(
            fixture.protocol,
            relocated,
            fixture.root / "relocated-output.json",
            apply=True,
            runner=exact_runner,
        )


def test_failed_apply_keeps_claim_and_blocks_retry(fixture: Fixture) -> None:
    def incomplete_runner(_weight, _warmup, _videos, _settings, role):
        return ModelRun((), frozenset({("holdout-scene", 0)}), 1, 0.01, {"role": role})

    with pytest.raises(FrozenHoldoutError, match="fixed frame universe"):
        evaluate(
            fixture.protocol,
            fixture.freeze,
            fixture.output,
            apply=True,
            runner=incomplete_runner,
        )
    assert not fixture.output.exists()
    assert fixture.claim_path.is_file()

    with pytest.raises(FrozenHoldoutError, match="already consumed"):
        evaluate(fixture.protocol, fixture.freeze, fixture.output, apply=True, runner=exact_runner)


def test_private_batch_mutation_is_detected_before_candidate_run(fixture: Fixture) -> None:
    roles: list[str] = []

    def mutating_runner(weight, warmup, videos, settings, role):
        roles.append(role)
        if role == "baseline":
            videos[0].file.path.chmod(0o600)
            videos[0].file.path.write_bytes(b"mutated-private-batch")
        return exact_runner(weight, warmup, videos, settings, role)

    with pytest.raises(FrozenHoldoutError, match="frozen record"):
        evaluate(
            fixture.protocol,
            fixture.freeze,
            fixture.output,
            apply=True,
            runner=mutating_runner,
        )

    assert roles == ["baseline"]
    assert fixture.claim_path.is_file()
    assert not fixture.output.exists()


def test_existing_output_symlink_is_never_replaced(fixture: Fixture) -> None:
    target = fixture.root / "important.json"
    target.write_text("keep", encoding="utf-8")
    fixture.output.symlink_to(target)

    preflight = evaluate(fixture.protocol, fixture.freeze, fixture.output)
    assert preflight["gate_result"] == "FAIL"
    with pytest.raises(FrozenHoldoutError, match="already consumed|output exists"):
        evaluate(fixture.protocol, fixture.freeze, fixture.output, apply=True, runner=exact_runner)
    assert target.read_text(encoding="utf-8") == "keep"


def test_rejects_candidate_weight_drift(fixture: Fixture) -> None:
    fixture.candidate_weight.write_bytes(b"changed")
    with pytest.raises(FrozenHoldoutError, match="managed hash and size"):
        validate_protocol(fixture.protocol, fixture.freeze)


def test_rejects_candidate_selected_with_holdout_access(fixture: Fixture) -> None:
    receipt = fixture.freeze / "receipt.json"
    freeze = json.loads(receipt.read_text(encoding="utf-8"))
    freeze["holdout_input_read"] = True
    write_json(receipt, freeze)
    fixture.rebuild_protocol()

    with pytest.raises(FrozenHoldoutError, match="was not read"):
        validate_protocol(fixture.protocol, fixture.freeze)


def test_rejects_managed_holdout_image_drift(fixture: Fixture) -> None:
    fixture.holdout_images[1].write_bytes(b"changed")
    with pytest.raises(FrozenHoldoutError, match="frozen record"):
        validate_protocol(fixture.protocol, fixture.freeze)


def test_rejects_holdout_without_full_review_completion_evidence(fixture: Fixture) -> None:
    manifest = json.loads(fixture.holdout_manifest.read_text(encoding="utf-8"))
    del manifest["review_completion"]
    write_json(fixture.holdout_manifest, manifest)
    fixture.rebuild_overlap()
    fixture.rebuild_protocol()

    with pytest.raises(FrozenHoldoutError, match="review_completion"):
        validate_protocol(fixture.protocol, fixture.freeze)


def test_rejects_incomplete_holdout_frame_review(fixture: Fixture) -> None:
    decisions = json.loads(fixture.final_decisions.read_text(encoding="utf-8"))
    decisions["frame_reviews"] = decisions["frame_reviews"][:1]
    write_json(fixture.final_decisions, decisions)
    fixture.rebuild_review_evidence_bindings()
    fixture.rebuild_overlap()
    fixture.rebuild_protocol()

    with pytest.raises(FrozenHoldoutError, match="complete CVAT frame span"):
        validate_protocol(fixture.protocol, fixture.freeze)


def test_rejects_holdout_job_without_applied_completion(fixture: Fixture) -> None:
    receipt = json.loads(fixture.completion_receipt.read_text(encoding="utf-8"))
    receipt["job_state_after"] = "in_progress"
    write_json(fixture.completion_receipt, receipt)
    fixture.rebuild_review_evidence_bindings()
    fixture.rebuild_overlap()
    fixture.rebuild_protocol()

    with pytest.raises(FrozenHoldoutError, match="not proven completed"):
        validate_protocol(fixture.protocol, fixture.freeze)


def test_rejects_training_holdout_source_overlap(fixture: Fixture) -> None:
    training = json.loads(fixture.training_coco.read_text(encoding="utf-8"))
    holdout = json.loads(fixture.holdout_coco.read_text(encoding="utf-8"))
    training["images"][0]["source_asset_id"] = holdout["images"][0]["source_asset_id"]
    training["images"][0]["source_leakage_group_id"] = holdout["images"][0][
        "source_leakage_group_id"
    ]
    training["images"][0]["source_normalized_asset_frame"] = 0
    write_json(fixture.training_coco, training)
    fixture.rebuild_all_bindings()

    with pytest.raises(FrozenHoldoutError, match="training and holdout data overlap"):
        validate_protocol(fixture.protocol, fixture.freeze)


def test_rejects_identical_image_content_with_distinct_source_identity(fixture: Fixture) -> None:
    training_image = fixture.training_manifest.parent / "images" / "train.jpg"
    training_image.write_bytes(fixture.holdout_images[0].read_bytes())
    training = json.loads(fixture.training_coco.read_text(encoding="utf-8"))
    training["images"][0]["sha256"] = digest(training_image)
    write_json(fixture.training_coco, training)
    training_manifest = json.loads(fixture.training_manifest.read_text(encoding="utf-8"))
    training_manifest["files"][1] = {
        "path": "images/train.jpg",
        "sha256": digest(training_image),
        "size_bytes": training_image.stat().st_size,
    }
    write_json(fixture.training_manifest, training_manifest)
    fixture.rebuild_all_bindings()

    with pytest.raises(FrozenHoldoutError, match="training and holdout data overlap"):
        validate_protocol(fixture.protocol, fixture.freeze)


def test_rejects_forged_zero_overlap_evidence(fixture: Fixture) -> None:
    overlap = json.loads(fixture.overlap.read_text(encoding="utf-8"))
    overlap["computed"]["training_asset_ids_sha256"] = "f" * 64
    write_json(fixture.overlap, overlap)
    fixture.rebuild_protocol()

    with pytest.raises(FrozenHoldoutError, match="independently computed"):
        validate_protocol(fixture.protocol, fixture.freeze)


def test_rejects_warmup_content_from_holdout(fixture: Fixture) -> None:
    fixture.warmup.write_bytes(fixture.holdout_images[0].read_bytes())
    fixture.rebuild_protocol()
    with pytest.raises(FrozenHoldoutError, match="warmup image content"):
        validate_protocol(fixture.protocol, fixture.freeze)


def test_rejects_source_video_not_named_by_holdout_source_identity(fixture: Fixture) -> None:
    fixture.videos[0].write_bytes(b"different-but-protocol-bound-video")
    fixture.rebuild_protocol()

    with pytest.raises(FrozenHoldoutError, match="source video hashes"):
        validate_protocol(fixture.protocol, fixture.freeze)


def test_parse_coco_allows_training_scene_to_span_multiple_source_assets() -> None:
    first_sha = "8" * 64
    second_sha = "9" * 64
    payload = {
        "categories": categories(),
        "images": [
            image_record(
                1,
                "images/first.jpg",
                scene_id="shared-training-scene",
                source_frame=0,
                image_sha256="a" * 64,
                asset_id="asset-a",
                source_sha256=first_sha,
                normalized_frame=0,
            ),
            image_record(
                2,
                "images/second.jpg",
                scene_id="shared-training-scene",
                source_frame=1,
                image_sha256="b" * 64,
                asset_id="asset-b",
                source_sha256=second_sha,
                normalized_frame=0,
            ),
        ],
        "annotations": [],
    }

    parsed = evaluator._parse_coco(payload, location="training fixture")

    assert parsed[5] == {"shared-training-scene": frozenset({first_sha, second_sha})}


def test_rejects_source_videos_swapped_between_scenes(fixture: Fixture) -> None:
    second_video = fixture.root / "holdout-scene-2.mp4"
    second_video.write_bytes(b"synthetic-video-for-second-scene")
    coco = json.loads(fixture.holdout_coco.read_text(encoding="utf-8"))
    coco["images"][1].update(
        {
            "scene_id": "holdout-scene-2",
            "source_frame": 0,
            "source_asset_id": "holdout-asset-2",
            "source_leakage_group_id": f"sha256:{digest(second_video)}",
            "source_normalized_asset_frame": 0,
        }
    )
    write_json(fixture.holdout_coco, coco)
    fixture.rebuild_holdout_manifest()
    fixture.rebuild_overlap()
    fixture.rebuild_protocol()

    protocol = json.loads(fixture.protocol.read_text(encoding="utf-8"))
    frames = [
        {"scene_id": "holdout-scene", "source_frame": 0},
        {"scene_id": "holdout-scene-2", "source_frame": 0},
    ]
    protocol["holdout"]["evaluation_frames"] = frames
    protocol["holdout"]["evaluation_frames_sha256"] = canonical_sha256(frames)
    protocol["holdout"]["source_videos"] = [
        {
            "scene_id": "holdout-scene",
            "frame_step": 1,
            "file": bound(second_video, fixture.protocol),
        },
        {
            "scene_id": "holdout-scene-2",
            "frame_step": 1,
            "file": bound(fixture.videos[0], fixture.protocol),
        },
    ]
    write_json(fixture.protocol, protocol)

    with pytest.raises(FrozenHoldoutError, match="scene's holdout source identity"):
        validate_protocol(fixture.protocol, fixture.freeze)


def test_rejects_incomplete_explicit_frame_universe(fixture: Fixture) -> None:
    protocol = json.loads(fixture.protocol.read_text(encoding="utf-8"))
    protocol["holdout"]["evaluation_frames"] = protocol["holdout"]["evaluation_frames"][:1]
    protocol["holdout"]["evaluation_frames_sha256"] = canonical_sha256(
        protocol["holdout"]["evaluation_frames"]
    )
    write_json(fixture.protocol, protocol)
    with pytest.raises(FrozenHoldoutError, match="differs from holdout COCO"):
        validate_protocol(fixture.protocol, fixture.freeze)


def test_rejects_frames_unreachable_by_ultralytics_stride(fixture: Fixture) -> None:
    protocol = json.loads(fixture.protocol.read_text(encoding="utf-8"))
    protocol["holdout"]["source_videos"][0]["frame_step"] = 5
    write_json(fixture.protocol, protocol)
    with pytest.raises(FrozenHoldoutError, match="unreachable with vid_stride"):
        validate_protocol(fixture.protocol, fixture.freeze)


def test_rejects_noncanonical_category_mapping(fixture: Fixture) -> None:
    coco = json.loads(fixture.holdout_coco.read_text(encoding="utf-8"))
    coco["categories"] = coco["categories"][:-1]
    write_json(fixture.holdout_coco, coco)
    fixture.rebuild_holdout_manifest()
    fixture.rebuild_overlap()
    fixture.rebuild_protocol()
    with pytest.raises(FrozenHoldoutError, match="canonical eight-class"):
        validate_protocol(fixture.protocol, fixture.freeze)


def test_rejects_changed_absolute_gates(fixture: Fixture) -> None:
    protocol = json.loads(fixture.protocol.read_text(encoding="utf-8"))
    protocol["gates"]["recall_min"] = 0.80
    write_json(fixture.protocol, protocol)
    with pytest.raises(FrozenHoldoutError, match="fixed gates"):
        validate_protocol(fixture.protocol, fixture.freeze)


def test_equal_candidate_f1_fails_strict_promotion_gate(fixture: Fixture) -> None:
    def tied_runner(weight, warmup, videos, settings, _role):
        return exact_runner(weight, warmup, videos, settings, "candidate")

    result = evaluate(
        fixture.protocol,
        fixture.freeze,
        fixture.output,
        apply=True,
        runner=tied_runner,
    )
    assert result["results"][0]["quality"]["f1_score"] == 1.0
    assert result["results"][1]["quality"]["f1_score"] == 1.0
    assert result["promotion_gates"]["candidate_f1_strictly_greater_than_baseline"] is False
    assert result["gate_result"] == "FAIL"


def test_ultralytics_runner_uses_result_frame_index(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import ultralytics

    class FakeModel:
        def __init__(self, _weight: str) -> None:
            self.names = {index: name for index, name in enumerate(evaluator.CANONICAL_LABELS)}

        def predict(self, *, source: str, stream: bool = False, **_kwargs):
            if stream:
                return iter([SimpleNamespace(boxes=None) for _ in range(3)])
            return []

    monkeypatch.setattr(ultralytics, "YOLO", FakeModel)
    weight = tmp_path / "weight.pt"
    warmup = tmp_path / "warmup.jpg"
    video_path = tmp_path / "video.mp4"
    for path in (weight, warmup, video_path):
        path.write_bytes(b"x")
    video_file = evaluator.BoundFile(video_path, digest(video_path), 1)
    video = RuntimeSourceVideo("scene", 5, video_file, (4, 9, 14))

    result = evaluator.ultralytics_runner(
        weight,
        warmup,
        (video,),
        {
            "confidence": 0.4,
            "image_size": 640,
            "device": "cpu",
            "nms_iou": 0.75,
            "rider_overlap": 0.25,
            "match_iou": 0.5,
        },
        "candidate",
    )

    assert result.observed_frame_keys == frozenset({("scene", 4), ("scene", 9), ("scene", 14)})


def test_source_video_base_type_has_no_implicit_targets(tmp_path: Path) -> None:
    path = tmp_path / "video.mp4"
    path.write_bytes(b"x")
    video = SourceVideo("scene", 1, evaluator.BoundFile(path, digest(path), 1))
    assert not hasattr(video, "target_frames")
