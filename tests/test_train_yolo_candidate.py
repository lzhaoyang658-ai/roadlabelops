from __future__ import annotations

import copy
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
import yaml

import scripts.train_yolo_candidate as candidate
from scripts.freeze_model_candidate import freeze_model_candidate
from scripts.train_yolo_candidate import (
    CandidateTrainingError,
    TrainingConfig,
    sha256,
    train_yolo_candidate,
    validate_yolo_dataset,
)


@dataclass
class DatasetFixture:
    root: Path
    weights: Path
    output_parent: Path
    manifest: dict[str, Any]

    def rewrite_manifest(self) -> None:
        (self.root / "manifest.json").write_text(
            json.dumps(self.manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def _file_record(root: Path, relative_path: str) -> dict[str, Any]:
    path = root / relative_path
    return {
        "path": relative_path,
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


@pytest.fixture
def dataset_fixture(tmp_path: Path) -> DatasetFixture:
    root = tmp_path / "published-yolo"
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
    (root / "images" / "train" / "train.jpg").write_bytes(b"fake-train-image")
    (root / "images" / "val" / "val.jpg").write_bytes(b"fake-val-image")
    train_lines = [
        f"{class_id} 0.5 0.5 0.2 0.2" for class_id in range(len(candidate.REQUIRED_LABELS))
    ]
    val_lines = [
        f"{class_id} 0.5 0.5 0.1 0.1" for class_id in range(len(candidate.REQUIRED_LABELS))
    ]
    (root / "labels" / "train" / "train.txt").write_text(
        "\n".join(train_lines) + "\n", encoding="utf-8"
    )
    (root / "labels" / "val" / "val.txt").write_text("\n".join(val_lines) + "\n", encoding="utf-8")
    dataset_yaml = {
        "train": "images/train",
        "val": "images/val",
        "names": {index: name for index, name in enumerate(candidate.MODEL_LABELS)},
    }
    (root / "dataset.yaml").write_text(
        yaml.safe_dump(dataset_yaml, sort_keys=False), encoding="utf-8"
    )
    relative_paths = [
        "dataset.yaml",
        "images/train/train.jpg",
        "images/val/val.jpg",
        "labels/train/train.txt",
        "labels/val/val.txt",
    ]
    per_category = {name: 2 for name in candidate.REQUIRED_LABELS}
    per_split = {name: 1 for name in candidate.REQUIRED_LABELS}
    manifest: dict[str, Any] = {
        "schema": candidate.DATASET_SCHEMA,
        "taxonomy": {
            "canonical_names": list(candidate.REQUIRED_LABELS),
            "model_names": list(candidate.MODEL_LABELS),
            "model_to_canonical": candidate.MODEL_TO_CANONICAL,
        },
        "gate": {
            "passed": True,
            "checks": {name: True for name in candidate.CURRENT_DATASET_GATE_CHECKS},
        },
        "counts": {
            "images": {"total": 2, "train": 1, "val": 1, "zero_annotations": 0},
            "annotations": {
                "total": 16,
                "train": 8,
                "val": 8,
                "by_category": per_category,
                "by_split_and_category": {"train": per_split, "val": per_split},
            },
            "assets": {"total": 2, "train": 1, "val": 1},
        },
        "files": [_file_record(root, relative_path) for relative_path in relative_paths],
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    weights = tmp_path / "yolo11n.pt"
    weights.write_bytes(b"trusted-yolo11n-fixture")
    output_parent = tmp_path / "candidates"
    output_parent.mkdir()
    return DatasetFixture(root, weights, output_parent, manifest)


class FakeBox:
    ap_class_index: ClassVar[list[int]] = list(range(8))
    p: ClassVar[list[float]] = [0.81 + index / 1000 for index in range(8)]
    r: ClassVar[list[float]] = [0.71 + index / 1000 for index in range(8)]
    ap50: ClassVar[list[float]] = [0.61 + index / 1000 for index in range(8)]
    ap: ClassVar[list[float]] = [0.51 + index / 1000 for index in range(8)]


class FakeMetrics:
    names: ClassVar[Any] = {index: name for index, name in enumerate(candidate.MODEL_LABELS)}
    results_dict: ClassVar[Any] = {
        "metrics/precision(B)": 0.82,
        "metrics/recall(B)": 0.72,
        "metrics/mAP50(B)": 0.62,
        "metrics/mAP50-95(B)": 0.52,
        "fitness": 0.52,
    }
    box: ClassVar[Any] = FakeBox()


class FakeTrainer:
    def __init__(
        self,
        weight_path: str,
        calls: list[dict[str, Any]],
        *,
        metrics: FakeMetrics | None = None,
        args_override: dict[str, Any] | None = None,
        results_override: str | None = None,
        mutate_source: Path | None = None,
        transfer_count: int = 8,
    ) -> None:
        self.weight_path = weight_path
        self.calls = calls
        self.metrics = metrics or FakeMetrics()
        self.args_override = args_override or {}
        self.results_override = results_override
        self.mutate_source = mutate_source
        self.names = {index: f"coco-class-{index}" for index in range(80)}
        self.names.update(
            {
                0: "person",
                1: "bicycle",
                2: "car",
                3: "motorcycle",
                5: "bus",
                7: "truck",
                9: "traffic light",
                11: "stop sign",
            }
        )
        self.pretrained_transfer_runtime = (
            f"Remapped {transfer_count}/8 cls head rows from pretrained weights by class name"
        )

    def train(self, **kwargs: Any) -> FakeMetrics:
        workspace = Path(kwargs["project"]).parent
        self.calls.append(
            {
                "weight_path": self.weight_path,
                "kwargs": kwargs,
                "environment": {
                    name: os.environ.get(name)
                    for name in (
                        "YOLO_CONFIG_DIR",
                        "XDG_CACHE_HOME",
                        "TORCH_HOME",
                        "MPLCONFIGDIR",
                        "TMPDIR",
                    )
                },
                "workspace": workspace,
            }
        )
        run = Path(kwargs["project"]) / kwargs["name"]
        (run / "weights").mkdir(parents=True)
        args = {"model": self.weight_path, **kwargs, **self.args_override}
        (run / "args.yaml").write_text(yaml.safe_dump(args), encoding="utf-8")
        results = self.results_override or (
            "epoch,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),"
            "metrics/mAP50-95(B),train/box_loss\n"
            "1,0.7,0.6,0.5,0.4,1.2\n"
            "2,0.8,0.7,0.6,0.5,1.0\n"
        )
        (run / "results.csv").write_text(results, encoding="utf-8")
        (run / "weights" / "best.pt").write_bytes(b"candidate-best-weights")
        (run / "weights" / "last.pt").write_bytes(b"candidate-last-weights")
        if self.mutate_source is not None:
            self.mutate_source.write_bytes(b"mutated-source")
        return self.metrics


def _config(seed: int = 7) -> TrainingConfig:
    return TrainingConfig(
        seed=seed,
        epochs=2,
        patience=1,
        imgsz=640,
        batch=2,
        device="cpu",
        workers=0,
        optimizer="AdamW",
        deterministic=True,
        amp=False,
        cache="disk",
        close_mosaic=1,
        freeze=0,
    )


def _run(
    fixture: DatasetFixture,
    output: Path,
    *,
    trainer_factory: Any,
    config: TrainingConfig | None = None,
    preflight: bool = False,
) -> dict[str, Any]:
    return train_yolo_candidate(
        fixture.root,
        fixture.weights,
        output,
        dataset_manifest_sha256=sha256(fixture.root / "manifest.json"),
        weights_sha256=sha256(fixture.weights),
        config=config or _config(),
        preflight=preflight,
        trainer_factory=trainer_factory,
        clock=lambda: datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc),
    )


def test_preflight_verifies_isolated_copy_without_training_or_output(
    dataset_fixture: DatasetFixture,
) -> None:
    output = dataset_fixture.output_parent / "seed-7"
    before = {
        path.relative_to(dataset_fixture.root).as_posix(): sha256(path)
        for path in dataset_fixture.root.rglob("*")
        if path.is_file()
    }

    result = _run(
        dataset_fixture,
        output,
        preflight=True,
        trainer_factory=lambda _path: pytest.fail("preflight instantiated trainer"),
    )

    assert result["mode"] == "preflight"
    assert result["mutation_performed"] is False
    assert result["gate"]["passed"] is True
    assert result["holdout"]["input_read"] is False
    assert not output.exists()
    assert not list(dataset_fixture.output_parent.glob(".seed-7.*"))
    assert before == {
        path.relative_to(dataset_fixture.root).as_posix(): sha256(path)
        for path in dataset_fixture.root.rglob("*")
        if path.is_file()
    }


def test_training_is_isolated_and_publishes_verified_receipt_no_replace(
    dataset_fixture: DatasetFixture,
) -> None:
    output = dataset_fixture.output_parent / "seed-7"
    calls: list[dict[str, Any]] = []

    result = _run(
        dataset_fixture,
        output,
        trainer_factory=lambda weight_path: FakeTrainer(weight_path, calls),
    )

    assert result["gate"]["passed"] is True
    assert len(calls) == 1
    call = calls[0]
    workspace = call["workspace"]
    assert not workspace.exists()
    assert Path(call["weight_path"]).name == "yolo11n.pt"
    assert Path(call["weight_path"]).parent.name == "inputs"
    assert Path(call["kwargs"]["data"]).parent.name == "dataset"
    assert all(
        value is not None and Path(value).is_relative_to(workspace)
        for value in call["environment"].values()
    )
    assert call["kwargs"]["deterministic"] is True
    assert call["kwargs"]["cache"] == "disk"
    assert call["kwargs"]["lr0"] == pytest.approx(0.001)
    assert call["kwargs"]["lrf"] == pytest.approx(0.01)
    assert call["kwargs"]["momentum"] == pytest.approx(0.9)
    assert call["kwargs"]["weight_decay"] == pytest.approx(0.0005)
    assert call["kwargs"]["warmup_epochs"] == pytest.approx(3.0)
    assert call["kwargs"]["warmup_bias_lr"] == pytest.approx(0.0)
    assert call["kwargs"]["cls_pw"] == pytest.approx(0.0)
    assert call["kwargs"]["plots"] is False
    assert call["kwargs"]["resume"] is False

    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["schema"] == candidate.RECEIPT_SCHEMA
    assert receipt["protocol"]["model_family"] == "YOLO11n"
    assert receipt["protocol"]["taxonomy"] == {
        "canonical_names": list(candidate.REQUIRED_LABELS),
        "model_names": list(candidate.MODEL_LABELS),
        "model_to_canonical": candidate.MODEL_TO_CANONICAL,
    }
    assert receipt["resolved_args"]["seed"] == 7
    assert receipt["inputs"]["dataset"]["manifest"]["sha256"] == sha256(
        dataset_fixture.root / "manifest.json"
    )
    assert receipt["inputs"]["dataset"]["dataset_yaml"]["sha256"] == sha256(
        dataset_fixture.root / "dataset.yaml"
    )
    assert receipt["inputs"]["dataset"]["manifest"]["schema"] == candidate.DATASET_SCHEMA
    assert receipt["inputs"]["dataset"]["taxonomy"] == receipt["protocol"]["taxonomy"]
    assert receipt["inputs"]["base_weights"]["sha256"] == sha256(dataset_fixture.weights)
    assert receipt["metrics"]["aggregate"]["map50_95"] == 0.52
    assert receipt["metrics"]["aggregate"]["fitness"] == 0.52
    assert set(receipt["metrics"]["per_class"]) == set(candidate.REQUIRED_LABELS)
    assert {
        name: receipt["metrics"]["per_class"][name]["class_id"]
        for name in candidate.REQUIRED_LABELS
    } == {name: index for index, name in enumerate(candidate.REQUIRED_LABELS)}
    assert all(
        record["status"] == "evaluable" and record["support_count"] == 1
        for record in receipt["metrics"]["per_class"].values()
    )
    assert receipt["gate"]["checks"]["support_aware_complete_eight_class_metrics_verified"] is True
    assert receipt["gate"]["checks"]["pretrained_class_head_transfer_verified"] is True
    assert receipt["pretrained_transfer"]["matched_row_count"] == 8
    assert receipt["pretrained_transfer"]["target_row_count"] == 8
    assert receipt["pretrained_transfer"]["runtime_observation"]["verification_mode"] == (
        "injected_test_double"
    )
    assert [row["canonical_name"] for row in receipt["pretrained_transfer"]["matched_rows"]] == (
        list(candidate.REQUIRED_LABELS)
    )
    assert receipt["best_epoch"] == {
        "epochs_recorded": 2,
        "index": 1,
        "number": 2,
        "selection_fitness": pytest.approx(0.5),
    }
    assert receipt["holdout"] == {
        "input_read": False,
        "statement": "No configured final holdout input was read.",
    }
    assert set(receipt["artifacts"]) == set(candidate.OUTPUT_ARTIFACT_PATHS)
    for record in receipt["artifacts"].values():
        artifact = output / record["path"]
        assert artifact.stat().st_size == record["size_bytes"]
        assert sha256(artifact) == record["sha256"]
    assert sorted(
        path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()
    ) == [
        "artifacts/args.yaml",
        "artifacts/results.csv",
        "receipt.json",
        "weights/best.pt",
        "weights/last.pt",
    ]

    with pytest.raises(CandidateTrainingError, match="already exists"):
        _run(
            dataset_fixture,
            output,
            trainer_factory=lambda _path: pytest.fail("existing output trained again"),
        )


def test_best_epoch_uses_ultralytics_map50_95_fitness(
    dataset_fixture: DatasetFixture,
) -> None:
    output = dataset_fixture.output_parent / "map50-95-best"
    results = (
        "epoch,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),"
        "metrics/mAP50-95(B),train/box_loss\n"
        "1,0.7,0.6,0.99,0.50,1.2\n"
        "2,0.8,0.7,0.00,0.51,1.0\n"
    )

    _run(
        dataset_fixture,
        output,
        trainer_factory=lambda path: FakeTrainer(path, [], results_override=results),
    )

    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["best_epoch"] == {
        "epochs_recorded": 2,
        "index": 1,
        "number": 2,
        "selection_fitness": pytest.approx(0.51),
    }


def _set_validation_class_support(fixture: DatasetFixture, supported_class_ids: list[int]) -> None:
    label_path = fixture.root / "labels" / "val" / "val.txt"
    label_path.write_text(
        "".join(f"{class_id} 0.5 0.5 0.1 0.1\n" for class_id in supported_class_ids),
        encoding="utf-8",
    )
    label_record = next(
        record for record in fixture.manifest["files"] if record["path"] == "labels/val/val.txt"
    )
    label_record.update(sha256=sha256(label_path), size_bytes=label_path.stat().st_size)
    by_split = fixture.manifest["counts"]["annotations"]["by_split_and_category"]
    by_split["val"] = {
        label: int(class_id in supported_class_ids)
        for class_id, label in enumerate(candidate.REQUIRED_LABELS)
    }
    fixture.manifest["counts"]["annotations"].update(
        val=len(supported_class_ids),
        total=8 + len(supported_class_ids),
        by_category={
            label: 1 + int(class_id in supported_class_ids)
            for class_id, label in enumerate(candidate.REQUIRED_LABELS)
        },
    )
    fixture.manifest["counts"]["images"]["zero_annotations"] = int(not supported_class_ids)
    fixture.rewrite_manifest()


def _metrics_for_supported_classes(class_ids: list[int]) -> FakeMetrics:
    metrics = FakeMetrics()
    metrics.box = SimpleNamespace(
        ap_class_index=class_ids,
        p=[0.80 + position / 100 for position in range(len(class_ids))],
        r=[0.70 + position / 100 for position in range(len(class_ids))],
        ap50=[0.60 + position / 100 for position in range(len(class_ids))],
        ap=[0.50 + position / 100 for position in range(len(class_ids))],
    )
    return metrics


def test_missing_validation_classes_are_not_evaluable_in_v2_receipt(
    dataset_fixture: DatasetFixture,
) -> None:
    supported_ids = [0, 1, 3, 6, 7]
    _set_validation_class_support(dataset_fixture, supported_ids)
    output = dataset_fixture.output_parent / "missing-val-classes"

    _run(
        dataset_fixture,
        output,
        trainer_factory=lambda path: FakeTrainer(
            path, [], metrics=_metrics_for_supported_classes(supported_ids)
        ),
    )

    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    per_class = receipt["metrics"]["per_class"]
    for class_id, label in enumerate(candidate.REQUIRED_LABELS):
        record = per_class[label]
        assert record["class_id"] == class_id
        if class_id in supported_ids:
            position = supported_ids.index(class_id)
            assert record["status"] == "evaluable"
            assert record["support_count"] == 1
            assert record["map50_95"] == pytest.approx(0.50 + position / 100)
        else:
            assert record == {
                "class_id": class_id,
                "status": "not_evaluable",
                "support_count": 0,
                "precision": None,
                "recall": None,
                "map50": None,
                "map50_95": None,
            }


@pytest.mark.parametrize("failure", ["missing-source-name", "missing-runtime-proof"])
def test_v3_training_requires_runtime_eight_of_eight_pretrained_transfer(
    dataset_fixture: DatasetFixture,
    failure: str,
) -> None:
    output = dataset_fixture.output_parent / failure

    def factory(path: str) -> FakeTrainer:
        trainer = FakeTrainer(path, [])
        if failure == "missing-source-name":
            trainer.names.pop(11)
        else:
            del trainer.pretrained_transfer_runtime
        return trainer

    expected = "does not cover every target|did not report"
    with pytest.raises(CandidateTrainingError, match=expected):
        _run(dataset_fixture, output, trainer_factory=factory)
    assert not output.exists()


def test_trainer_metric_classes_must_exactly_match_validation_support(
    dataset_fixture: DatasetFixture,
) -> None:
    supported_ids = [0, 1, 3, 6, 7]
    _set_validation_class_support(dataset_fixture, supported_ids)
    output = dataset_fixture.output_parent / "wrong-supported-class-metrics"

    with pytest.raises(CandidateTrainingError, match="exactly match validation-supported"):
        _run(
            dataset_fixture,
            output,
            trainer_factory=lambda path: FakeTrainer(
                path, [], metrics=_metrics_for_supported_classes([0, 1, 3, 6])
            ),
        )

    assert not output.exists()


def test_dataset_validation_rejects_unmanaged_or_tampered_files(
    dataset_fixture: DatasetFixture,
) -> None:
    (dataset_fixture.root / "labels" / "train.cache").write_bytes(b"cache")
    with pytest.raises(CandidateTrainingError, match="exactly manifest.json"):
        validate_yolo_dataset(dataset_fixture.root)
    (dataset_fixture.root / "labels" / "train.cache").unlink()
    (dataset_fixture.root / "labels" / "train" / "train.txt").write_text(
        "0 0.5 0.5 0.3 0.3\n", encoding="utf-8"
    )
    with pytest.raises(CandidateTrainingError, match="hash or size differs"):
        validate_yolo_dataset(dataset_fixture.root)


def test_dataset_validation_rejects_wrong_yaml_taxonomy_even_when_rehashed(
    dataset_fixture: DatasetFixture,
) -> None:
    dataset_yaml = dataset_fixture.root / "dataset.yaml"
    payload = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    payload["names"][0] = "vehicle"
    dataset_yaml.write_text(yaml.safe_dump(payload), encoding="utf-8")
    record = next(
        record for record in dataset_fixture.manifest["files"] if record["path"] == "dataset.yaml"
    )
    record.update(sha256=sha256(dataset_yaml), size_bytes=dataset_yaml.stat().st_size)
    dataset_fixture.rewrite_manifest()

    with pytest.raises(CandidateTrainingError, match="bound eight-class model taxonomy"):
        validate_yolo_dataset(dataset_fixture.root)


def test_training_accepts_legacy_v2_canonical_dataset_and_records_identity_mapping(
    dataset_fixture: DatasetFixture,
) -> None:
    dataset_yaml = dataset_fixture.root / "dataset.yaml"
    payload = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    payload["names"] = {index: name for index, name in enumerate(candidate.REQUIRED_LABELS)}
    dataset_yaml.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    record = next(
        record for record in dataset_fixture.manifest["files"] if record["path"] == "dataset.yaml"
    )
    record.update(sha256=sha256(dataset_yaml), size_bytes=dataset_yaml.stat().st_size)
    dataset_fixture.manifest["schema"] = candidate.LEGACY_DATASET_SCHEMA
    dataset_fixture.manifest.pop("taxonomy")
    dataset_fixture.manifest["gate"]["checks"].pop("canonical_and_model_taxonomies_bound")
    dataset_fixture.rewrite_manifest()
    metrics = FakeMetrics()
    metrics.names = {index: name for index, name in enumerate(candidate.REQUIRED_LABELS)}
    output = dataset_fixture.output_parent / "legacy-v2-dataset"

    _run(
        dataset_fixture,
        output,
        trainer_factory=lambda path: FakeTrainer(path, [], metrics=metrics, transfer_count=5),
    )

    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["schema"] == candidate.RECEIPT_SCHEMA
    assert receipt["inputs"]["dataset"]["manifest"]["schema"] == (candidate.LEGACY_DATASET_SCHEMA)
    assert receipt["protocol"]["taxonomy"] == {
        "canonical_names": list(candidate.REQUIRED_LABELS),
        "model_names": list(candidate.REQUIRED_LABELS),
        "model_to_canonical": candidate.LEGACY_MODEL_TO_CANONICAL,
    }
    assert set(receipt["metrics"]["per_class"]) == set(candidate.REQUIRED_LABELS)


def test_training_rejects_wrong_weight_hash_and_dataset_nested_output(
    dataset_fixture: DatasetFixture,
) -> None:
    with pytest.raises(CandidateTrainingError, match="dataset-manifest SHA-256 mismatch"):
        train_yolo_candidate(
            dataset_fixture.root,
            dataset_fixture.weights,
            dataset_fixture.output_parent / "candidate",
            dataset_manifest_sha256="0" * 64,
            weights_sha256=sha256(dataset_fixture.weights),
            config=_config(),
            preflight=True,
        )
    with pytest.raises(CandidateTrainingError, match="base-weight SHA-256 mismatch"):
        train_yolo_candidate(
            dataset_fixture.root,
            dataset_fixture.weights,
            dataset_fixture.output_parent / "candidate",
            dataset_manifest_sha256=sha256(dataset_fixture.root / "manifest.json"),
            weights_sha256="0" * 64,
            config=_config(),
            preflight=True,
        )
    with pytest.raises(CandidateTrainingError, match="outside the immutable dataset"):
        train_yolo_candidate(
            dataset_fixture.root,
            dataset_fixture.weights,
            dataset_fixture.root / "candidate",
            dataset_manifest_sha256=sha256(dataset_fixture.root / "manifest.json"),
            weights_sha256=sha256(dataset_fixture.weights),
            config=_config(),
            preflight=True,
        )


def test_training_rejects_wrong_metric_names_without_partial_output(
    dataset_fixture: DatasetFixture,
) -> None:
    output = dataset_fixture.output_parent / "wrong-names"
    calls: list[dict[str, Any]] = []
    metrics = FakeMetrics()
    metrics.names = {**FakeMetrics.names, 7: "road_sign"}

    with pytest.raises(CandidateTrainingError, match="wrong eight-class names"):
        _run(
            dataset_fixture,
            output,
            trainer_factory=lambda path: FakeTrainer(path, calls, metrics=metrics),
        )

    assert not output.exists()
    assert not list(dataset_fixture.output_parent.glob(".wrong-names.*"))


@pytest.mark.parametrize(
    ("metrics_mutation", "results", "message"),
    [
        (
            lambda metrics: metrics.results_dict.update({"metrics/mAP50-95(B)": math.nan}),
            None,
            "finite number",
        ),
        (
            lambda _metrics: None,
            "epoch,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),"
            + "metrics/mAP50-95(B)\n0,0.8,0.7,0.6,nan\n",
            "non-finite",
        ),
    ],
)
def test_training_rejects_nonfinite_metrics_or_results(
    dataset_fixture: DatasetFixture,
    metrics_mutation: Any,
    results: str | None,
    message: str,
) -> None:
    output = dataset_fixture.output_parent / "nonfinite"
    calls: list[dict[str, Any]] = []
    metrics = FakeMetrics()
    metrics.results_dict = copy.deepcopy(FakeMetrics.results_dict)
    metrics_mutation(metrics)

    with pytest.raises(CandidateTrainingError, match=message):
        _run(
            dataset_fixture,
            output,
            trainer_factory=lambda path: FakeTrainer(
                path, calls, metrics=metrics, results_override=results
            ),
        )

    assert not output.exists()


def test_training_rejects_trainer_argument_drift_and_source_mutation(
    dataset_fixture: DatasetFixture,
) -> None:
    drift_output = dataset_fixture.output_parent / "drift"
    calls: list[dict[str, Any]] = []
    with pytest.raises(CandidateTrainingError, match="immutable argument 'seed'"):
        _run(
            dataset_fixture,
            drift_output,
            trainer_factory=lambda path: FakeTrainer(path, calls, args_override={"seed": 99}),
        )
    assert not drift_output.exists()

    mutation_output = dataset_fixture.output_parent / "mutation"
    source_label = dataset_fixture.root / "labels" / "train" / "train.txt"
    with pytest.raises(CandidateTrainingError, match="managed dataset file changed"):
        _run(
            dataset_fixture,
            mutation_output,
            trainer_factory=lambda path: FakeTrainer(path, [], mutate_source=source_label),
        )
    assert not mutation_output.exists()


def test_invalid_config_fails_before_creating_workspace(dataset_fixture: DatasetFixture) -> None:
    output = dataset_fixture.output_parent / "bad-config"
    with pytest.raises(CandidateTrainingError, match="multiple of 32"):
        _run(
            dataset_fixture,
            output,
            config=TrainingConfig(seed=1, imgsz=641),
            trainer_factory=lambda _path: pytest.fail("invalid config instantiated trainer"),
        )
    assert not list(dataset_fixture.output_parent.glob(".bad-config.*"))


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"lr0": 0.0}, "lr0 must be within"),
        ({"lrf": 1.1}, "lrf must be within"),
        ({"momentum": 1.0}, "momentum must be within"),
        ({"weight_decay": -0.1}, "weight_decay must be within"),
        ({"warmup_epochs": -0.1}, "warmup_epochs must be non-negative"),
        ({"warmup_bias_lr": 1.1}, "warmup_bias_lr must be within"),
        ({"cls_pw": 1.1}, "cls_pw must be within"),
    ],
)
def test_explicit_optimizer_hyperparameters_are_range_checked_before_training(
    dataset_fixture: DatasetFixture,
    override: dict[str, float],
    message: str,
) -> None:
    output = dataset_fixture.output_parent / "bad-explicit-hyperparameter"

    with pytest.raises(CandidateTrainingError, match=message):
        _run(
            dataset_fixture,
            output,
            config=TrainingConfig(seed=1, **override),
            trainer_factory=lambda _path: pytest.fail("invalid config instantiated trainer"),
        )

    assert not list(dataset_fixture.output_parent.glob(".bad-explicit-hyperparameter.*"))


def test_three_published_training_receipts_feed_the_freeze_selector(
    dataset_fixture: DatasetFixture,
) -> None:
    candidate_directories: list[Path] = []
    scores = {
        42: (0.60, 0.50),
        43: (0.65, 0.55),
        44: (0.63, 0.53),
    }
    for seed, (map50, map50_95) in scores.items():
        metrics = FakeMetrics()
        metrics.results_dict = {
            **FakeMetrics.results_dict,
            "metrics/mAP50(B)": map50,
            "metrics/mAP50-95(B)": map50_95,
            "fitness": map50_95,
        }
        output = dataset_fixture.output_parent / f"seed-{seed}"
        _run(
            dataset_fixture,
            output,
            config=_config(seed),
            trainer_factory=lambda path, result=metrics: FakeTrainer(path, [], metrics=result),
        )
        candidate_directories.append(output)

    frozen = dataset_fixture.output_parent / "frozen"
    result = freeze_model_candidate(candidate_directories, frozen)

    assert result["selected_seed"] == 43
    selection = json.loads((frozen / "receipt.json").read_text(encoding="utf-8"))
    assert selection["holdout_input_read"] is False
    assert selection["selected_seed"] == 43
    assert selection["selected_weight"]["sha256"] == sha256(frozen / "best.pt")
