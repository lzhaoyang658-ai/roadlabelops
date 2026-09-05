from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml
from PIL import Image

from scripts import analyze_training_recovery as analysis
from scripts import freeze_model_candidate as freeze
from scripts import prepare_yolo_dataset as prepare
from scripts import train_yolo_candidate as train

SYNTHETIC_TRAINING_TASK_ID = 41001
SYNTHETIC_TRAINING_JOB_ID = 51001
FINAL_HOLDOUT_TASK_ID = 91001
FINAL_HOLDOUT_JOB_ID = 92001


@pytest.fixture(autouse=True)
def _configure_final_holdout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROADLABELOPS_FINAL_HOLDOUT_TASK_IDS", str(FINAL_HOLDOUT_TASK_ID))
    monkeypatch.setenv("ROADLABELOPS_FINAL_HOLDOUT_JOB_IDS", str(FINAL_HOLDOUT_JOB_ID))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _artifact(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    return {
        "path": relative,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _aggregate(map50_95: float) -> dict[str, float]:
    return {
        "precision": 0.5,
        "recall": 0.4,
        "map50": 0.6,
        "map50_95": map50_95,
        "fitness": map50_95,
    }


def _training_protocol(*, legacy: bool) -> dict[str, Any]:
    training: dict[str, Any] = {
        "epochs": 2,
        "patience": 1,
        "imgsz": 640,
        "batch": 2,
        "device": "cpu",
        "workers": 0,
        "optimizer": "AdamW",
        "deterministic": True,
        "amp": False,
        "cache": "disk",
        "close_mosaic": 1,
        "freeze": 0,
    }
    if not legacy:
        training.update(
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            warmup_bias_lr=0.1,
            cls_pw=0.0,
        )
    return {
        "schema": freeze.LEGACY_PROTOCOL_SCHEMA if legacy else freeze.PROTOCOL_SCHEMA,
        "model_family": "YOLO11n",
        "taxonomy": (
            list(freeze.REQUIRED_LABELS)
            if legacy
            else {
                "canonical_names": list(freeze.REQUIRED_LABELS),
                "model_names": list(freeze.MODEL_LABELS),
                "model_to_canonical": freeze.MODEL_TO_CANONICAL,
            }
        ),
        "training": training,
        "validation_selection": {
            "primary": "mAP50-95",
            "tie_breakers": ["mAP50", "recall", "precision", "smaller_seed"],
        },
        "holdout_access": "prohibited",
    }


def _write_candidate(
    root: Path,
    *,
    seed: int,
    map50_95: float,
    dataset: train.ValidatedDataset,
    legacy: bool,
) -> Path:
    (root / "artifacts").mkdir(parents=True)
    (root / "weights").mkdir()
    protocol = _training_protocol(legacy=legacy)
    args = {
        **protocol["training"],
        "seed": seed,
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        "box": 7.5,
        "cls": 0.5,
        "cls_pw": 0.0,
        "dfl": 1.5,
        "mosaic": 1.0,
    }
    (root / "artifacts" / "args.yaml").write_text(
        yaml.safe_dump(args, allow_unicode=True, sort_keys=True), encoding="utf-8"
    )
    (root / "artifacts" / "results.csv").write_text(
        "epoch,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),"
        "metrics/mAP50-95(B),lr/pg0\n"
        f"1,0.4,0.3,0.5,{map50_95 / 2},0.004\n"
        f"2,0.5,0.4,0.6,{map50_95},0.009\n",
        encoding="utf-8",
    )
    (root / "weights" / "best.pt").write_bytes(f"best-{seed}".encode())
    (root / "weights" / "last.pt").write_bytes(f"last-{seed}".encode())

    per_class = {
        name: {
            "class_id": class_id,
            "precision": 0.5,
            "recall": 0.4,
            "map50": 0.6,
            "map50_95": max(0.0, map50_95 - class_id / 100),
        }
        for class_id, name in enumerate(freeze.REQUIRED_LABELS)
    }
    dataset_claim: dict[str, Any] = {
        "manifest": {
            "schema": dict(dataset.schema),
            "sha256": dataset.manifest_sha256,
            "size_bytes": dataset.manifest_size_bytes,
        },
        "dataset_yaml": {
            "sha256": dataset.dataset_yaml.sha256,
            "size_bytes": dataset.dataset_yaml.size_bytes,
        },
        "managed_files_sha256": dataset.managed_files_sha256,
        "managed_file_count": len(dataset.files),
        "counts": dataset.counts,
    }
    if not legacy:
        dataset_claim["taxonomy"] = dataset.taxonomy
    receipt = {
        "schema": freeze.LEGACY_CANDIDATE_SCHEMA if legacy else freeze.CANDIDATE_SCHEMA,
        "gate": {
            "passed": True,
            "checks": {
                name: True
                for name in (
                    freeze.LEGACY_REQUIRED_GATE_CHECKS
                    if legacy
                    else freeze.SUPPORT_AWARE_REQUIRED_GATE_CHECKS
                )
            },
        },
        "mutation_performed": True,
        "protocol": protocol,
        "protocol_sha256": _canonical_sha256(protocol),
        "resolved_args": {
            "model_family": "YOLO11n",
            "seed": seed,
            **protocol["training"],
        },
        "inputs": {
            "dataset": dataset_claim,
            "base_weights": {
                "file_name": "yolo11n.pt",
                "model_family": "YOLO11n",
                "sha256": "d" * 64,
                "size_bytes": 100,
            },
        },
        "timestamps": {
            "started_at": "2026-09-01T00:00:00.000000Z",
            "finished_at": "2026-09-01T00:01:00.000000Z",
            "duration_seconds": 60.0,
        },
        "environment": {
            "python": {"version": "3.12.0"},
            "packages": {"ultralytics": freeze.EXPECTED_ULTRALYTICS_VERSION},
        },
        "metrics": {"aggregate": _aggregate(map50_95), "per_class": per_class},
        "best_epoch": {
            "index": 1,
            "number": 2,
            "selection_fitness": map50_95,
            "epochs_recorded": 2,
        },
        "artifacts": {
            "args": _artifact(root, "artifacts/args.yaml"),
            "results": _artifact(root, "artifacts/results.csv"),
            "best_weights": _artifact(root, "weights/best.pt"),
            "last_weights": _artifact(root, "weights/last.pt"),
        },
        "holdout": {"input_read": False, "statement": freeze.NO_HOLDOUT_STATEMENT},
    }
    receipt_path = root / "receipt.json"
    _write_json(receipt_path, receipt)
    return receipt_path


def _reference_payload(root: Path) -> tuple[Path, Path]:
    image_specs = (
        (1, "images/task-41001/train-a.png", (255, 0, 0), 100, 0),
        (2, "images/task-41001/train-b.png", (0, 255, 0), 100, 1),
        (3, "images/task-41001/validation.png", (0, 0, 255), 200, 0),
    )
    images = []
    for image_id, file_name, color, asset_id, frame in image_specs:
        path = root / file_name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (128, 64), color).save(path)
        source_hash = "1" * 64 if asset_id == 100 else "2" * 64
        images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": 128,
                "height": 64,
                "sha256": _sha256(path),
                "scene_id": f"scene-{asset_id}",
                "task_id": SYNTHETIC_TRAINING_TASK_ID,
                "source_asset_id": asset_id,
                "source_leakage_group_id": f"sha256:{source_hash}",
                "source_normalized_asset_frame": frame,
            }
        )

    annotations: list[dict[str, Any]] = []

    def add(image_id: int, category_id: int, bbox: list[int]) -> None:
        annotations.append(
            {
                "id": len(annotations) + 1,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
            }
        )

    for index in range(12):
        add(1, 1, [2 + (index % 6) * 10, 2 + (index // 6) * 10, 6, 6])
    for category_id in range(2, 9):
        add(1, category_id, [4 + category_id * 8, 32, 8, 8])
    for category_id in range(1, 9):
        add(3, category_id, [4 + category_id * 10, 5, 3, 3])

    categories = [
        {"id": index, "name": name, "supercategory": "road_object"}
        for index, name in enumerate(prepare.REQUIRED_LABELS, start=1)
    ]
    coco = {
        "info": {
            "schema": prepare.REFERENCE_SCHEMA,
            "labels_sha256": "a" * 64,
            "source_map_sha256": "b" * 64,
        },
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    coco_path = root / "annotations.coco.json"
    _write_json(coco_path, coco)
    category_counts = {
        name: sum(annotation["category_id"] == index for annotation in annotations)
        for index, name in enumerate(prepare.REQUIRED_LABELS, start=1)
    }
    manifest = {
        "schema": prepare.REFERENCE_SCHEMA,
        "gate": {
            "passed": True,
            "blocking_reasons": [],
            "checks": {name: True for name in prepare.REFERENCE_REQUIRED_GATE_CHECKS},
        },
        "evidence": {
            "labels": {"sha256": "a" * 64},
            "source_map": {
                "sha256": "b" * 64,
                "assets": [
                    {
                        "asset_id": 100,
                        "sha256": "1" * 64,
                        "leakage_group_id": f"sha256:{'1' * 64}",
                    },
                    {
                        "asset_id": 200,
                        "sha256": "2" * 64,
                        "leakage_group_id": f"sha256:{'2' * 64}",
                    },
                ],
            },
        },
        "counts": {
            "images": len(images),
            "annotations": len(annotations),
            "annotations_by_category": category_counts,
        },
        "source_statistics": {
            "tasks": [
                {
                    "task_id": SYNTHETIC_TRAINING_TASK_ID,
                    "job_id": SYNTHETIC_TRAINING_JOB_ID,
                }
            ]
        },
        "files": [
            {
                "path": "annotations.coco.json",
                "sha256": _sha256(coco_path),
                "size_bytes": coco_path.stat().st_size,
            },
            *[
                {
                    "path": image["file_name"],
                    "sha256": image["sha256"],
                    "size_bytes": (root / image["file_name"]).stat().st_size,
                }
                for image in images
            ],
        ],
    }
    manifest_path = root / "manifest.json"
    _write_json(manifest_path, manifest)
    return coco_path, manifest_path


def _make_legacy_dataset(dataset_root: Path) -> None:
    yaml_path = dataset_root / "dataset.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "train": "images/train",
                "val": "images/val",
                "names": {index: name for index, name in enumerate(train.REQUIRED_LABELS)},
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    manifest_path = dataset_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema"] = train.LEGACY_DATASET_SCHEMA
    manifest.pop("taxonomy")
    manifest["gate"]["checks"].pop("canonical_and_model_taxonomies_bound")
    yaml_record = next(record for record in manifest["files"] if record["path"] == "dataset.yaml")
    yaml_record.update(sha256=_sha256(yaml_path), size_bytes=yaml_path.stat().st_size)
    _write_json(manifest_path, manifest)


@dataclass(frozen=True)
class WorkspaceEvidence:
    reference_manifest: Path
    dataset_manifest: Path
    candidate_receipts: tuple[Path, Path, Path]


def _build_workspace(root: Path, *, legacy: bool) -> WorkspaceEvidence:
    reference_root = root / "data" / "ground-truth" / "training-reference"
    reference_root.mkdir(parents=True)
    coco_path, reference_manifest = _reference_payload(reference_root)
    split_plan = root / "build-fixtures" / "split.json"
    _write_json(
        split_plan,
        {
            "schema": prepare.SPLIT_PLAN_SCHEMA,
            "train_asset_ids": [100],
            "val_asset_ids": [200],
        },
    )
    dataset_root = root / "data" / "training" / "training-yolo"
    prepare.prepare_yolo_dataset(
        coco_path,
        split_plan,
        dataset_root,
        reference_manifest_path=reference_manifest,
    )
    if legacy:
        _make_legacy_dataset(dataset_root)
    validated_dataset = train.validate_yolo_dataset(dataset_root)
    receipts = tuple(
        _write_candidate(
            root / "data" / "model-candidates" / f"seed-{seed}",
            seed=seed,
            map50_95=score,
            dataset=validated_dataset,
            legacy=legacy,
        )
        for seed, score in ((42, 0.05), (43, 0.08), (44, 0.11))
    )
    return WorkspaceEvidence(reference_manifest, dataset_root / "manifest.json", receipts)


@pytest.fixture
def evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> WorkspaceEvidence:
    monkeypatch.chdir(tmp_path)
    return _build_workspace(tmp_path, legacy=True)


def _analyze(evidence: WorkspaceEvidence, output: Path) -> dict[str, Any]:
    return analysis.analyze_training_recovery(
        training_reference_manifest=evidence.reference_manifest,
        dataset_manifest=evidence.dataset_manifest,
        candidate_receipts=evidence.candidate_receipts,
        output=output,
    )


def test_emits_deterministic_training_only_recovery_diagnostics(
    tmp_path: Path, evidence: WorkspaceEvidence
) -> None:
    first_path = tmp_path / "docs" / "evidence" / "analysis-one.json"
    second_path = tmp_path / "docs" / "evidence" / "analysis-two.json"

    payload = _analyze(evidence, first_path)
    second = _analyze(evidence, second_path)

    assert first_path.read_bytes() == second_path.read_bytes()
    assert json.loads(first_path.read_text(encoding="utf-8")) == payload == second
    assert payload["scope"] == {
        "input_scope": "training_only",
        "holdout_access": "prohibited",
        "final_holdout_access": "prohibited",
        "mutation_performed": False,
    }
    assert payload["verification"]["final_holdout_path_read"] is False
    assert payload["dataset"]["splits"]["train"]["annotations_by_class"]["car"] == 12
    assert payload["dataset"]["splits"]["val"]["annotations_by_class"]["car"] == 1
    assert payload["dataset"]["imbalance"] == {
        "train_max_to_min_nonzero_class_ratio": 12.0,
        "train_validation_class_distribution_total_variation": 0.506579,
        "validation_source_fraction": 0.5,
        "validation_image_fraction": 0.333333,
    }
    assert (
        payload["dataset"]["bounding_boxes"]["splits"]["val"]["overall"][
            "sqrt_area_below_32px_fraction"
        ]
        == 1.0
    )
    seed_stats = payload["training"]["seed_metrics"]["aggregate_statistics"]["map50_95"]
    assert seed_stats["mean"] == 0.08
    assert seed_stats["sample_standard_deviation"] == 0.03
    assert seed_stats["coefficient_of_variation"] == 0.375
    learning = payload["training"]["learning_arguments"]
    assert learning["common_args"]["lr0"] == 0.01
    assert learning["learning_rate_diagnostic"]["ratio_to_adamw_documented_reference"] == 10.0
    assert "lr0" in learning["observed_args_not_semantically_frozen_in_v1_protocol"]
    mapping = payload["training"]["pretrained_class_name_mapping"]
    assert mapping["exact_match_count"] == 5
    assert [row["canonical_name"] for row in mapping["classes"] if not row["exact_name_match"]] == [
        "pedestrian",
        "traffic_light",
        "traffic_sign",
    ]
    assert {
        "SINGLE_SOURCE_VALIDATION",
        "TRAIN_VALIDATION_CLASS_SHIFT",
        "TRAIN_CLASS_IMBALANCE",
        "VALIDATION_HAS_NO_NEGATIVE_IMAGES",
        "ADAMW_LR0_ABOVE_REFERENCE",
        "INCOMPLETE_PRETRAINED_CLASS_NAME_MAPPING",
        "HIGH_SEED_VARIANCE",
        "SMALL_OBJECT_DOMINATED_VALIDATION",
    }.issubset({flag["code"] for flag in payload["diagnosis"]["flags"]})


def test_accepts_current_dataset_and_candidate_contracts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    evidence = _build_workspace(tmp_path, legacy=False)

    payload = _analyze(evidence, tmp_path / "evidence" / "current.json")

    mapping = payload["training"]["pretrained_class_name_mapping"]
    assert mapping["exact_match_count"] == 8
    assert mapping["exact_match_fraction"] == 1.0
    learning = payload["training"]["learning_arguments"]
    assert "lr0" in learning["semantic_protocol_fields"]
    assert "lr0" not in learning["observed_args_not_semantically_frozen_in_v1_protocol"]


@pytest.mark.parametrize(
    "malicious",
    (
        Path("data/holdout/manifest.json"),
        Path("data/ground-truth/task-91001/manifest.json"),
        Path("data/ground-truth/task/91001/manifest.json"),
    ),
)
def test_rejects_final_holdout_before_opening_inputs(
    tmp_path: Path, evidence: WorkspaceEvidence, malicious: Path
) -> None:
    output = tmp_path / "evidence" / "forbidden.json"

    with pytest.raises(analysis.TrainingRecoveryAnalysisError, match="forbidden"):
        analysis.analyze_training_recovery(
            training_reference_manifest=malicious,
            dataset_manifest=evidence.dataset_manifest,
            candidate_receipts=evidence.candidate_receipts,
            output=output,
        )

    assert not output.exists()
    assert not output.parent.exists()


def test_rejects_paths_outside_exact_allowed_prefixes(
    tmp_path: Path, evidence: WorkspaceEvidence
) -> None:
    with pytest.raises(analysis.TrainingRecoveryAnalysisError, match="must be under"):
        analysis.analyze_training_recovery(
            training_reference_manifest=tmp_path / "data" / "ground-truthish" / "manifest.json",
            dataset_manifest=evidence.dataset_manifest,
            candidate_receipts=evidence.candidate_receipts,
            output=tmp_path / "evidence" / "outside.json",
        )


def test_rejects_symlinked_input_component(tmp_path: Path, evidence: WorkspaceEvidence) -> None:
    link = tmp_path / "data" / "ground-truth" / "reference-link"
    link.symlink_to(evidence.reference_manifest.parent, target_is_directory=True)

    with pytest.raises(analysis.TrainingRecoveryAnalysisError, match="symlink"):
        analysis.analyze_training_recovery(
            training_reference_manifest=link / "manifest.json",
            dataset_manifest=evidence.dataset_manifest,
            candidate_receipts=evidence.candidate_receipts,
            output=tmp_path / "evidence" / "symlink-input.json",
        )


def test_rejects_reference_hash_mismatch_without_publishing(
    tmp_path: Path, evidence: WorkspaceEvidence
) -> None:
    image = evidence.reference_manifest.parent / "images" / "task-41001" / "train-a.png"
    image.write_bytes(image.read_bytes() + b"tamper")
    output = tmp_path / "evidence" / "reference-mismatch.json"

    with pytest.raises(analysis.TrainingRecoveryAnalysisError, match="hash or size mismatch"):
        _analyze(evidence, output)

    assert not output.exists()


def test_rejects_candidate_artifact_hash_mismatch_without_publishing(
    tmp_path: Path, evidence: WorkspaceEvidence
) -> None:
    args_path = evidence.candidate_receipts[0].parent / "artifacts" / "args.yaml"
    args_path.write_text(args_path.read_text(encoding="utf-8") + "# tampered\n", encoding="utf-8")
    output = tmp_path / "evidence" / "candidate-mismatch.json"

    with pytest.raises(analysis.TrainingRecoveryAnalysisError, match="artifact args hash"):
        _analyze(evidence, output)

    assert not output.exists()


def test_no_overwrite_preserves_regular_file_and_broken_symlink(
    tmp_path: Path, evidence: WorkspaceEvidence
) -> None:
    regular = tmp_path / "evidence" / "existing.json"
    regular.parent.mkdir()
    regular.write_bytes(b"sentinel")
    with pytest.raises(FileExistsError, match="already exists"):
        _analyze(evidence, regular)
    assert regular.read_bytes() == b"sentinel"

    broken = tmp_path / "evidence" / "broken.json"
    broken.symlink_to("missing-target.json")
    with pytest.raises(FileExistsError, match="already exists"):
        _analyze(evidence, broken)
    assert broken.is_symlink()
    assert broken.readlink() == Path("missing-target.json")


def test_rejects_symlinked_output_parent_without_writing_through_it(
    tmp_path: Path, evidence: WorkspaceEvidence
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "evidence-link"
    link.symlink_to(outside, target_is_directory=True)

    with pytest.raises(analysis.TrainingRecoveryAnalysisError, match="non-symlink directories"):
        _analyze(evidence, link / "analysis.json")

    assert not (outside / "analysis.json").exists()
