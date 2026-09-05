from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml
from PIL import Image

import scripts.evaluate_training_validation as evaluation
from scripts.evaluate_training_validation import (
    EvaluationSettings,
    ModelRun,
    TrainingValidationError,
    canonical_sha256,
    evaluate_training_validation,
)

FINAL_HOLDOUT_TASK_ID = 91001
FINAL_HOLDOUT_JOB_ID = 92001


@pytest.fixture(autouse=True)
def _configure_final_holdout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROADLABELOPS_FINAL_HOLDOUT_TASK_IDS", str(FINAL_HOLDOUT_TASK_ID))
    monkeypatch.setenv("ROADLABELOPS_FINAL_HOLDOUT_JOB_IDS", str(FINAL_HOLDOUT_JOB_ID))


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    return {"path": relative, "sha256": _sha256(path), "size_bytes": path.stat().st_size}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _taxonomy(model_names: tuple[str, ...] = evaluation.MODEL_LABELS) -> dict[str, Any]:
    mapping = (
        evaluation.MODEL_TO_CANONICAL
        if model_names == evaluation.MODEL_LABELS
        else evaluation.LEGACY_MODEL_TO_CANONICAL
    )
    return {
        "canonical_names": list(evaluation.CANONICAL_LABELS),
        "model_names": list(model_names),
        "model_to_canonical": mapping,
    }


def _enable_transfer_evidence(fixture: EvaluationFixture) -> None:
    base_sha = "c" * 64
    message = "Remapped 8/8 cls head rows from pretrained weights by class name"
    fixture.receipt_payload["gate"]["checks"] = {
        name: True for name in evaluation.CURRENT_CANDIDATE_GATE_CHECKS
    }
    fixture.receipt_payload["inputs"]["base_weights"] = {
        "file_name": "yolo11n.pt",
        "model_family": "YOLO11n",
        "sha256": base_sha,
        "size_bytes": 123,
    }
    source_ids = [2, 5, 7, 3, 1, 0, 9, 11]
    fixture.receipt_payload["pretrained_transfer"] = {
        "schema": {"name": "roadlabelops.pretrained-class-head-transfer", "version": 1},
        "source_model": {
            "family": "YOLO11n",
            "base_weights_sha256": base_sha,
            "class_count": 80,
            "names_sha256": "d" * 64,
        },
        "target": {
            "class_count": 8,
            "model_names": list(evaluation.MODEL_LABELS),
            "canonical_names": list(evaluation.CANONICAL_LABELS),
        },
        "matched_rows": [
            {
                "target_id": target_id,
                "target_model_name": model_name,
                "canonical_name": canonical_name,
                "source_id": source_ids[target_id],
                "source_model_name": model_name,
            }
            for target_id, (model_name, canonical_name) in enumerate(
                zip(evaluation.MODEL_LABELS, evaluation.CANONICAL_LABELS, strict=True)
            )
        ],
        "matched_row_count": 8,
        "target_row_count": 8,
        "runtime_observation": {
            "verification_mode": "ultralytics_logger",
            "message": message,
            "message_sha256": hashlib.sha256(message.encode()).hexdigest(),
            "matched_row_count": 8,
            "target_row_count": 8,
        },
    }
    fixture.bind_candidate_to_dataset()


@dataclass
class EvaluationFixture:
    workspace: Path
    dataset: Path
    manifest: Path
    manifest_payload: dict[str, Any]
    candidate: Path
    receipt: Path
    receipt_payload: dict[str, Any]
    weight: Path
    output_parent: Path

    def write_manifest(self) -> None:
        _write_json(self.manifest, self.manifest_payload)

    def bind_candidate_to_dataset(self) -> None:
        manifest_bytes = self.manifest.read_bytes()
        files = sorted(self.manifest_payload["files"], key=lambda item: item["path"])
        yaml_record = next(item for item in files if item["path"] == "dataset.yaml")
        dataset_binding = self.receipt_payload["inputs"]["dataset"]
        dataset_binding.update(
            manifest={
                "schema": self.manifest_payload["schema"],
                "sha256": _sha256(self.manifest),
                "size_bytes": len(manifest_bytes),
            },
            dataset_yaml={
                "sha256": yaml_record["sha256"],
                "size_bytes": yaml_record["size_bytes"],
            },
            managed_files_sha256=canonical_sha256(files),
            managed_file_count=len(files),
            counts=self.manifest_payload["counts"],
        )
        self.receipt_payload["protocol_sha256"] = canonical_sha256(self.receipt_payload["protocol"])
        _write_json(self.receipt, self.receipt_payload)


@pytest.fixture
def validation_fixture(tmp_path: Path) -> EvaluationFixture:
    workspace = tmp_path / "workspace"
    dataset = workspace / "data" / "training" / "fold-01-yolo"
    for split in ("train", "val"):
        (dataset / "images" / split).mkdir(parents=True)
        (dataset / "labels" / split).mkdir(parents=True)
    Image.new("RGB", (100, 100), "white").save(dataset / "images" / "train" / "train.png")
    Image.new("RGB", (100, 100), "white").save(dataset / "images" / "val" / "a.png")
    Image.new("RGB", (100, 100), "black").save(dataset / "images" / "val" / "b.png")
    (dataset / "labels" / "train" / "train.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (dataset / "labels" / "val" / "a.txt").write_text("5 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    (dataset / "labels" / "val" / "b.txt").write_bytes(b"")
    (dataset / "dataset.yaml").write_text(
        yaml.safe_dump(
            {
                "train": "images/train",
                "val": "images/val",
                "names": {index: name for index, name in enumerate(evaluation.MODEL_LABELS)},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    relative_paths = [
        "dataset.yaml",
        "images/train/train.png",
        "images/val/a.png",
        "images/val/b.png",
        "labels/train/train.txt",
        "labels/val/a.txt",
        "labels/val/b.txt",
    ]
    train_counts = {
        label: int(index == 0) for index, label in enumerate(evaluation.CANONICAL_LABELS)
    }
    val_counts = {label: int(index == 5) for index, label in enumerate(evaluation.CANONICAL_LABELS)}
    total_counts = {
        label: train_counts[label] + val_counts[label] for label in evaluation.CANONICAL_LABELS
    }
    manifest_payload: dict[str, Any] = {
        "schema": evaluation.DATASET_SCHEMA,
        "taxonomy": _taxonomy(),
        "gate": {
            "passed": True,
            "checks": {name: True for name in evaluation.CURRENT_DATASET_GATE_CHECKS},
        },
        "inputs": {
            "split_plan": {
                "file_name": "fold-01.split.json",
                "sha256": "a" * 64,
                "semantic_sha256": "b" * 64,
                "schema": evaluation.SPLIT_PLAN_SCHEMA,
            }
        },
        "split": {"train_asset_ids": ["asset-train"], "val_asset_ids": ["asset-val"]},
        "counts": {
            "images": {"total": 3, "train": 1, "val": 2, "zero_annotations": 1},
            "annotations": {
                "total": 2,
                "train": 1,
                "val": 1,
                "by_category": total_counts,
                "by_split_and_category": {"train": train_counts, "val": val_counts},
            },
            "assets": {"total": 2, "train": 1, "val": 1},
        },
        "files": [_record(dataset, relative) for relative in relative_paths],
    }
    manifest = dataset / "manifest.json"
    _write_json(manifest, manifest_payload)

    candidate = workspace / "data" / "model-candidates" / "repaired_control" / "fold-01"
    (candidate / "weights").mkdir(parents=True)
    weight = candidate / "weights" / "best.pt"
    weight.write_bytes(b"candidate-best-weight")
    protocol = {
        "schema": evaluation.PROTOCOL_SCHEMA,
        "model_family": "YOLO11n",
        "taxonomy": _taxonomy(),
        "training": {"imgsz": 640, "device": "mps", "lr0": 0.001, "cls_pw": 0.0},
        "validation_selection": {
            "primary": "mAP50-95",
            "tie_breakers": ["mAP50", "recall", "precision", "smaller_seed"],
        },
        "holdout_access": "prohibited",
    }
    per_class = {
        label: {
            "class_id": class_id,
            "status": "evaluable" if class_id == 5 else "not_evaluable",
            "support_count": int(class_id == 5),
            "precision": 0.8 if class_id == 5 else None,
            "recall": 0.7 if class_id == 5 else None,
            "map50": 0.6 if class_id == 5 else None,
            "map50_95": 0.5 if class_id == 5 else None,
        }
        for class_id, label in enumerate(evaluation.CANONICAL_LABELS)
    }
    receipt_payload: dict[str, Any] = {
        "schema": evaluation.CANDIDATE_SCHEMA,
        "gate": {
            "passed": True,
            "checks": {name: True for name in evaluation.SUPPORT_AWARE_CANDIDATE_GATE_CHECKS},
        },
        "mutation_performed": True,
        "protocol": protocol,
        "protocol_sha256": canonical_sha256(protocol),
        "resolved_args": {
            "model_family": "YOLO11n",
            "seed": 42,
            "imgsz": 640,
            "device": "mps",
            "lr0": 0.001,
            "cls_pw": 0.0,
        },
        "inputs": {
            "dataset": {
                "manifest": {
                    "schema": evaluation.DATASET_SCHEMA,
                    "sha256": _sha256(manifest),
                    "size_bytes": manifest.stat().st_size,
                },
                "dataset_yaml": {
                    "sha256": _sha256(dataset / "dataset.yaml"),
                    "size_bytes": (dataset / "dataset.yaml").stat().st_size,
                },
                "managed_files_sha256": canonical_sha256(
                    sorted(manifest_payload["files"], key=lambda item: item["path"])
                ),
                "managed_file_count": len(manifest_payload["files"]),
                "counts": manifest_payload["counts"],
                "taxonomy": _taxonomy(),
            }
        },
        "metrics": {
            "aggregate": {"map50": 0.61, "map50_95": 0.51},
            "per_class": per_class,
        },
        "timestamps": {"duration_seconds": 12.5},
        "artifacts": {
            "best_weights": {
                "path": "weights/best.pt",
                "sha256": _sha256(weight),
                "size_bytes": weight.stat().st_size,
            }
        },
        "holdout": {"input_read": False, "statement": evaluation.NO_HOLDOUT_STATEMENT},
    }
    receipt = candidate / "receipt.json"
    _write_json(receipt, receipt_payload)
    output_parent = workspace / "data" / "model-candidates" / "evaluations"
    output_parent.mkdir(parents=True)
    return EvaluationFixture(
        workspace,
        dataset,
        manifest,
        manifest_payload,
        candidate,
        receipt,
        receipt_payload,
        weight,
        output_parent,
    )


class RecordingRunner:
    def __init__(self, *, complete: bool = True) -> None:
        self.complete = complete
        self.calls = 0
        self.frames: tuple[evaluation.ValidationFrame, ...] = ()
        self.settings: dict[str, Any] = {}

    def __call__(
        self,
        weight: Path,
        frames: tuple[evaluation.ValidationFrame, ...],
        settings: dict[str, Any],
    ) -> ModelRun:
        self.calls += 1
        self.frames = tuple(frames)
        self.settings = dict(settings)
        assert weight.name == "best.pt"
        assert all("/images/val/" in frame.image.path.as_posix() for frame in frames)
        predictions = (
            {
                "prediction_id": "person-match",
                "scene_id": frames[0].scene_id,
                "frame": 0,
                "model_label": "person",
                "confidence": 0.90,
                "bbox": [30.0, 30.0, 70.0, 70.0],
                "source": "auto",
            },
            {
                "prediction_id": "mapped-duplicate",
                "scene_id": frames[0].scene_id,
                "frame": 0,
                "model_label": "pedestrian",
                "confidence": 0.80,
                "bbox": [30.0, 30.0, 70.0, 70.0],
                "source": "auto",
            },
            {
                "prediction_id": "empty-frame-fp",
                "scene_id": frames[1].scene_id,
                "frame": 0,
                "model_label": "car",
                "confidence": 0.70,
                "bbox": [10.0, 10.0, 20.0, 20.0],
                "source": "auto",
            },
        )
        observed = {(frame.scene_id, frame.frame) for frame in frames}
        if not self.complete:
            observed.remove((frames[-1].scene_id, frames[-1].frame))
        return ModelRun(
            raw_predictions=predictions,
            observed_frame_keys=frozenset(observed),
            predict_call_count=1,
            inference_wall_seconds=0.25,
            model_load_seconds=0.05,
            model_metadata={"provider": "fake"},
        )


def _evaluate(
    fixture: EvaluationFixture,
    *,
    output_name: str = "fold-01.json",
    arm_id: str = "repaired_control",
    fold_id: str = "fold-01",
    runner: RecordingRunner | None = None,
    settings: EvaluationSettings | None = None,
) -> dict[str, Any]:
    return evaluate_training_validation(
        fixture.dataset,
        fixture.manifest,
        fixture.receipt,
        fixture.weight,
        fixture.output_parent / output_name,
        arm_id=arm_id,
        fold_id=fold_id,
        settings=settings,
        workspace_root=fixture.workspace,
        runner=runner or RecordingRunner(),
    )


def test_evaluates_complete_val_split_with_mapping_and_empty_frame(
    validation_fixture: EvaluationFixture,
) -> None:
    runner = RecordingRunner()

    report = _evaluate(validation_fixture, runner=runner)

    assert runner.calls == 1
    assert runner.settings == {
        "confidence": 0.4,
        "image_size": 640,
        "device": "mps",
        "nms_iou": 0.75,
        "rider_overlap": 0.25,
        "match_iou": 0.5,
    }
    assert report["schema"] == evaluation.OUTPUT_SCHEMA
    assert report["experiment"] == {
        "arm_id": "repaired_control",
        "seed": 42,
        "fold_id": "fold-01",
    }
    assert report["val_source"] == {
        "split": "val",
        "asset_ids": ["asset-val"],
        "frame_count": 2,
        "zero_annotation_frame_count": 1,
        "annotation_count": 1,
        "frames_sha256": report["val_source"]["frames_sha256"],
    }
    assert len(report["val_source"]["frames_sha256"]) == 64
    assert report["metrics"]["map50_95"] == 0.51
    assert report["metrics"]["overall"] == {
        "true_positive_count": 1,
        "false_positive_count": 1,
        "false_negative_count": 0,
        "prediction_count": 2,
        "ground_truth_count": 1,
        "precision": 0.5,
        "recall": 1.0,
        "f1_score": 2 / 3,
        "evaluated_frame_count": 2,
        "clean_frame_count": 1,
        "clean_frame_rate": 0.5,
        "complete_frame_coverage": True,
    }
    pedestrian = report["metrics"]["per_class"]["pedestrian"]
    assert pedestrian["status"] == "evaluable"
    assert pedestrian["true_positive_count"] == 1
    car = report["metrics"]["per_class"]["car"]
    assert car["status"] == "not_evaluable"
    assert car["false_positive_count"] == 1
    assert car["precision"] is None
    assert report["compute"] == {
        "training_duration_seconds": 12.5,
        "evaluation_inference_seconds": 0.25,
        "model_load_seconds": 0.05,
        "evaluated_frames_per_second": 8.0,
        "predict_call_count": 1,
    }
    saved = json.loads((validation_fixture.output_parent / "fold-01.json").read_text())
    assert saved == report


def test_zero_predictions_with_ground_truth_has_zero_f1_not_null(
    validation_fixture: EvaluationFixture,
) -> None:
    class EmptyRunner:
        def __call__(self, _weight: Path, frames: Any, _settings: Any) -> ModelRun:
            return ModelRun(
                raw_predictions=(),
                observed_frame_keys=frozenset((frame.scene_id, frame.frame) for frame in frames),
                predict_call_count=1,
                inference_wall_seconds=0.1,
                model_load_seconds=0.01,
                model_metadata={"provider": "fake"},
            )

    report = _evaluate(
        validation_fixture,
        output_name="zero-predictions.json",
        runner=EmptyRunner(),
    )

    overall = report["metrics"]["overall"]
    assert overall["true_positive_count"] == 0
    assert overall["false_positive_count"] == 0
    assert overall["false_negative_count"] == 1
    assert overall["precision"] is None
    assert overall["recall"] == 0.0
    assert overall["f1_score"] == 0.0
    pedestrian = report["metrics"]["per_class"]["pedestrian"]
    assert pedestrian["precision"] is None
    assert pedestrian["recall"] == 0.0
    assert pedestrian["f1_score"] == 0.0
    assert report["metrics"]["per_class"]["car"]["f1_score"] is None


def test_rejects_incomplete_validation_frame_coverage(
    validation_fixture: EvaluationFixture,
) -> None:
    with pytest.raises(TrainingValidationError, match="complete validation frame universe"):
        _evaluate(validation_fixture, runner=RecordingRunner(complete=False))

    assert not (validation_fixture.output_parent / "fold-01.json").exists()


@pytest.mark.parametrize("target", ["dataset", "weight"])
def test_rejects_dataset_or_weight_hash_drift(
    validation_fixture: EvaluationFixture, target: str
) -> None:
    if target == "dataset":
        (validation_fixture.dataset / "labels" / "val" / "a.txt").write_text(
            "5 0.5 0.5 0.2 0.2\n", encoding="utf-8"
        )
        message = "managed hash or size"
    else:
        validation_fixture.weight.write_bytes(b"drifted-best-weight")
        message = "receipt hash"

    with pytest.raises(TrainingValidationError, match=message):
        _evaluate(validation_fixture)


def test_revalidates_candidate_weight_after_inference(
    validation_fixture: EvaluationFixture,
) -> None:
    class MutatingRunner(RecordingRunner):
        def __call__(self, weight: Path, frames: Any, settings: Any) -> ModelRun:
            result = super().__call__(weight, frames, settings)
            weight.write_bytes(b"changed-during-inference")
            return result

    with pytest.raises(TrainingValidationError, match=r"candidate input\[1\] changed"):
        _evaluate(validation_fixture, runner=MutatingRunner())

    assert not (validation_fixture.output_parent / "fold-01.json").exists()


def test_rejects_wrong_fold_or_candidate_dataset_binding(
    validation_fixture: EvaluationFixture,
) -> None:
    with pytest.raises(TrainingValidationError, match="differs from dataset fold binding"):
        _evaluate(validation_fixture, fold_id="fold-02")

    validation_fixture.receipt_payload["inputs"]["dataset"]["manifest"]["sha256"] = "f" * 64
    _write_json(validation_fixture.receipt, validation_fixture.receipt_payload)
    with pytest.raises(TrainingValidationError, match="another dataset manifest"):
        _evaluate(validation_fixture, output_name="wrong-dataset.json")


def test_rejects_final_holdout_paths_before_opening(
    validation_fixture: EvaluationFixture,
) -> None:
    forbidden_dataset = validation_fixture.workspace / "data" / "holdout" / "never-open"
    with pytest.raises(TrainingValidationError, match="forbidden"):
        evaluate_training_validation(
            forbidden_dataset,
            forbidden_dataset / "manifest.json",
            validation_fixture.receipt,
            validation_fixture.weight,
            validation_fixture.output_parent / "holdout.json",
            arm_id="repaired_control",
            fold_id="fold-01",
            workspace_root=validation_fixture.workspace,
            runner=lambda *_args: pytest.fail("forbidden input reached inference"),
        )

    forbidden_candidate = (
        validation_fixture.workspace
        / "data"
        / "model-candidates"
        / f"Task_{FINAL_HOLDOUT_TASK_ID}"
        / "receipt.json"
    )
    with pytest.raises(TrainingValidationError, match="forbidden"):
        evaluate_training_validation(
            validation_fixture.dataset,
            validation_fixture.manifest,
            forbidden_candidate,
            forbidden_candidate.parent / "weights" / "best.pt",
            validation_fixture.output_parent / "configured-holdout-attempt.json",
            arm_id="repaired_control",
            fold_id="fold-01",
            workspace_root=validation_fixture.workspace,
            runner=lambda *_args: pytest.fail("forbidden input reached inference"),
        )


def test_output_is_no_replace_and_dataset_symlink_is_rejected(
    validation_fixture: EvaluationFixture,
) -> None:
    output = validation_fixture.output_parent / "fold-01.json"
    output.write_text("preserve-me", encoding="utf-8")
    with pytest.raises(TrainingValidationError, match="already exists"):
        _evaluate(validation_fixture, runner=RecordingRunner())
    assert output.read_text(encoding="utf-8") == "preserve-me"

    linked = validation_fixture.workspace / "data" / "training" / "linked-fold"
    linked.symlink_to(validation_fixture.dataset, target_is_directory=True)
    with pytest.raises(TrainingValidationError, match="symbolic links"):
        evaluate_training_validation(
            linked,
            linked / "manifest.json",
            validation_fixture.receipt,
            validation_fixture.weight,
            validation_fixture.output_parent / "linked.json",
            arm_id="repaired_control",
            fold_id="fold-01",
            workspace_root=validation_fixture.workspace,
            runner=lambda *_args: pytest.fail("symlink input reached inference"),
        )


def test_accepts_legacy_v2_dataset_with_identity_mapping(
    validation_fixture: EvaluationFixture,
) -> None:
    dataset_yaml = validation_fixture.dataset / "dataset.yaml"
    dataset_yaml.write_text(
        yaml.safe_dump(
            {
                "train": "images/train",
                "val": "images/val",
                "names": {index: name for index, name in enumerate(evaluation.CANONICAL_LABELS)},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    yaml_record = next(
        item
        for item in validation_fixture.manifest_payload["files"]
        if item["path"] == "dataset.yaml"
    )
    yaml_record.update(sha256=_sha256(dataset_yaml), size_bytes=dataset_yaml.stat().st_size)
    validation_fixture.manifest_payload["schema"] = evaluation.LEGACY_DATASET_SCHEMA
    validation_fixture.manifest_payload.pop("taxonomy")
    validation_fixture.manifest_payload["gate"]["checks"].pop(
        "canonical_and_model_taxonomies_bound"
    )
    validation_fixture.write_manifest()
    identity_taxonomy = _taxonomy(evaluation.CANONICAL_LABELS)
    validation_fixture.receipt_payload["protocol"]["taxonomy"] = identity_taxonomy
    validation_fixture.receipt_payload["inputs"]["dataset"]["taxonomy"] = identity_taxonomy
    validation_fixture.bind_candidate_to_dataset()

    report = _evaluate(validation_fixture)

    assert report["bindings"]["dataset"]["manifest"]["schema"] == (evaluation.LEGACY_DATASET_SCHEMA)
    assert report["bindings"]["dataset"]["taxonomy"] == identity_taxonomy


def test_image_size_is_derived_from_arm_and_receipt_and_confidence_is_fixed(
    validation_fixture: EvaluationFixture,
) -> None:
    validation_fixture.receipt_payload["resolved_args"]["imgsz"] = 960
    validation_fixture.receipt_payload["protocol"]["training"]["imgsz"] = 960
    validation_fixture.bind_candidate_to_dataset()
    runner = RecordingRunner()

    report = _evaluate(
        validation_fixture,
        output_name="small-target.json",
        arm_id="small_target_960",
        runner=runner,
    )

    assert report["settings"]["image_size"] == 960
    assert runner.settings["image_size"] == 960
    with pytest.raises(TrainingValidationError, match="fixed protocol value 0.40"):
        _evaluate(
            validation_fixture,
            output_name="wrong-confidence.json",
            arm_id="small_target_960",
            settings=EvaluationSettings(confidence=0.001, device="mps"),
        )


def test_rejects_arm_image_size_that_differs_from_candidate_receipt(
    validation_fixture: EvaluationFixture,
) -> None:
    with pytest.raises(TrainingValidationError, match="training signature differs at 'imgsz'"):
        _evaluate(validation_fixture, arm_id="small_target_960")


def test_rejects_same_size_arm_with_different_class_balance_signature(
    validation_fixture: EvaluationFixture,
) -> None:
    with pytest.raises(TrainingValidationError, match="training signature differs at 'cls_pw'"):
        _evaluate(validation_fixture, arm_id="class_balance_025")


def test_validates_runtime_eight_of_eight_pretrained_transfer_evidence(
    validation_fixture: EvaluationFixture,
) -> None:
    _enable_transfer_evidence(validation_fixture)
    report = _evaluate(validation_fixture, output_name="transfer-proof.json")
    assert report["gate"]["checks"]["recovery_arm_training_signature_verified"] is True

    validation_fixture.receipt_payload["pretrained_transfer"]["matched_row_count"] = 7
    validation_fixture.bind_candidate_to_dataset()
    with pytest.raises(TrainingValidationError, match="not complete"):
        _evaluate(
            validation_fixture,
            output_name="bad-transfer-proof.json",
            runner=lambda *_args: pytest.fail("bad transfer proof reached inference"),
        )


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        (EvaluationSettings(nms_iou=0.70), "settings.nms_iou"),
        (EvaluationSettings(rider_overlap=0.30), "settings.rider_overlap"),
    ],
)
def test_rejects_product_postprocess_setting_drift_before_inference(
    validation_fixture: EvaluationFixture,
    settings: EvaluationSettings,
    message: str,
) -> None:
    with pytest.raises(TrainingValidationError, match=message):
        _evaluate(
            validation_fixture,
            output_name="setting-drift.json",
            settings=settings,
            runner=lambda *_args: pytest.fail("setting drift reached inference"),
        )
