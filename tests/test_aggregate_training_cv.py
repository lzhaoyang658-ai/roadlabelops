from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.aggregate_training_cv import (
    CANDIDATE_PROTOCOL_SCHEMA,
    CANDIDATE_SCHEMA,
    CANONICAL_NAMES,
    COCO_SOURCE_CLASS_IDS,
    CURRENT_CANDIDATE_GATE_CHECKS,
    CV_PLAN_SCHEMA,
    DATASET_SCHEMA,
    EVALUATION_SCHEMA,
    OUTPUT_SCHEMA,
    PRETRAINED_TRANSFER_MESSAGE,
    PRETRAINED_TRANSFER_SCHEMA,
    PROTOCOL_SCHEMA,
    RANKING,
    SPLIT_PLAN_SCHEMA,
    TrainingCVAggregationError,
    _summarize_runs,
    aggregate_training_cv,
)

ARMS = (
    {"arm_id": "repaired_control", "imgsz": 640, "lr0": 0.001, "cls_pw": 0.0},
    {"arm_id": "small_target_960", "imgsz": 960, "lr0": 0.001, "cls_pw": 0.0},
    {"arm_id": "class_balance_025", "imgsz": 640, "lr0": 0.001, "cls_pw": 0.25},
)
COMMON = {
    "amp": True,
    "batch": 8,
    "cache": "none",
    "close_mosaic": 10,
    "deterministic": True,
    "device": "mps",
    "epochs": 150,
    "freeze": 0,
    "lrf": 0.01,
    "momentum": 0.9,
    "optimizer": "AdamW",
    "patience": 25,
    "warmup_bias_lr": 0.0,
    "warmup_epochs": 3.0,
    "weight_decay": 0.0005,
    "workers": 0,
}
MODEL_NAMES = (
    "car",
    "bus",
    "truck",
    "motorcycle",
    "bicycle",
    "person",
    "traffic light",
    "stop sign",
)
TAXONOMY = {
    "canonical_names": list(CANONICAL_NAMES),
    "model_names": list(MODEL_NAMES),
    "model_to_canonical": dict(zip(MODEL_NAMES, CANONICAL_NAMES, strict=True)),
}
READINESS = {
    "oof_precision_min": 0.9,
    "oof_recall_min": 0.85,
    "clean_frame_rate_min": 0.8,
    "every_seed_and_source_f1_noninferior_to_repaired_control": True,
    "supported_class_recall_drop_max": 0.05,
    "seed_oof_f1_sample_standard_deviation_max": 0.03,
    "all_planned_runs_required": True,
    "exact_contract_match_required": True,
}


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True).encode(
            "utf-8"
        )
        + b"\n"
    )


def _canonical_sha(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _binding(path: Path) -> dict[str, Any]:
    encoded = path.read_bytes()
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "size_bytes": len(encoded)}


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _count_f1(tp: int, fp: int, fn: int) -> float | None:
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else None


@dataclass
class EvidenceFixture:
    root: Path
    protocol: Path
    base_weights: Path
    folds: dict[str, dict[str, Any]]
    datasets: dict[str, dict[str, Any]]
    screening_reports: list[Path]

    def confirmation_reports(self, *, arms: tuple[str, ...]) -> list[Path]:
        paths: list[Path] = []
        for arm in arms:
            for seed in (43, 44):
                for fold_id in sorted(self.folds):
                    paths.append(_make_report(self, arm, seed, fold_id))
        return paths


def _source_counts(index: int) -> tuple[int, int, dict[str, int]]:
    frame_count = (2, 3, 4, 5, 10, 6)[index - 1]
    zero_count = 1 if index == 5 else 0
    support = {label: index for label in CANONICAL_NAMES}
    if index == 1:
        support["traffic_sign"] = 0
    return frame_count, zero_count, support


def _source_assets(fold_count: int = 5) -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []
    for index in range(1, fold_count + 1):
        frame_count, _zero_count, support = _source_counts(index)
        assets.append(
            {
                "asset_id": index,
                "leakage_group_id": f"sha256:{index:064x}",
                "image_count": frame_count,
                "annotation_count": sum(support.values()),
            }
        )
    return assets


def _make_fixture(root: Path, *, fold_count: int = 5) -> EvidenceFixture:
    evidence_root = root / "docs/evidence"
    plan_root = evidence_root / "loso-v1"
    reference_root = root / "data/ground-truth/training-reference-v2"
    coco_payload = {"schema": "synthetic-training-coco", "images": [], "annotations": []}
    coco_path = reference_root / "annotations.coco.json"
    _write_json(coco_path, coco_payload)
    reference_path = reference_root / "manifest.json"
    _write_json(
        reference_path,
        {
            "schema": {"name": "roadlabelops.training-coco-reference", "version": 2},
            "files": [{"path": "annotations.coco.json", **_binding(coco_path)}],
            "gate": {"passed": True, "blocking_reasons": [], "checks": {"valid": True}},
            "counts": {
                "images": 24,
                "zero_annotation_images": 1,
                "annotations": 119,
                "categories": 8,
                "annotations_by_category": {
                    label: 15 - (1 if label == "traffic_sign" else 0) for label in CANONICAL_NAMES
                },
            },
            "source_statistics": {
                "assets": [
                    {**asset, "scene_ids": [f"scene-{asset['asset_id']}"]}
                    for asset in _source_assets()
                ]
            },
        },
    )
    base_weights = root / "yolo11n.pt"
    base_weights.write_bytes(b"immutable-base-yolo11n-weight")
    implementation = root / "scripts/recovery_training_impl.py"
    implementation.parent.mkdir(parents=True, exist_ok=True)
    implementation.write_text("# synthetic immutable implementation\n", encoding="utf-8")
    source_assets = _source_assets(fold_count)
    total_images = sum(int(asset["image_count"]) for asset in source_assets)
    source_statistics = [_source_counts(index) for index in range(1, fold_count + 1)]
    total_zero_annotations = sum(zero_count for _, zero_count, _ in source_statistics)
    class_totals = {
        label: sum(support[label] for _, _, support in source_statistics)
        for label in CANONICAL_NAMES
    }
    total_annotations = sum(class_totals.values())
    reference_payload = json.loads(reference_path.read_text(encoding="utf-8"))
    reference_payload["counts"].update(
        {
            "images": total_images,
            "zero_annotation_images": total_zero_annotations,
            "annotations": total_annotations,
            "annotations_by_category": class_totals,
        }
    )
    reference_payload["source_statistics"]["assets"] = [
        {**asset, "scene_ids": [f"scene-{asset['asset_id']}"]} for asset in source_assets
    ]
    _write_json(reference_path, reference_payload)
    fold_records: list[dict[str, Any]] = []
    folds: dict[str, dict[str, Any]] = {}
    semantic_folds: list[dict[str, Any]] = []
    for index in range(1, fold_count + 1):
        fold_id = f"fold-{index:02d}"
        split_payload = {
            "schema": SPLIT_PLAN_SCHEMA,
            "train_asset_ids": [asset for asset in range(1, fold_count + 1) if asset != index],
            "val_asset_ids": [index],
        }
        split_path = plan_root / "folds" / f"{fold_id}.split.json"
        _write_json(split_path, split_payload)
        frame_count, zero_count, support = _source_counts(index)
        classes = {
            label: {
                "box_count": count,
                "positive_image_count": min(count, frame_count),
                "zero_image_count": max(0, frame_count - min(count, frame_count)),
            }
            for label, count in support.items()
        }
        fold_record = {
            "fold_id": fold_id,
            "val_asset_id": index,
            "train": {
                "asset_ids": [asset for asset in range(1, fold_count + 1) if asset != index],
                "source_count": fold_count - 1,
                "image_count": total_images - frame_count,
                "zero_annotation_image_count": total_zero_annotations - zero_count,
                "annotation_count": total_annotations - sum(support.values()),
                "classes": {
                    label: {
                        "box_count": class_totals[label] - count,
                        "positive_image_count": 1,
                        "zero_image_count": (total_images - frame_count) - 1,
                    }
                    for label, count in support.items()
                },
            },
            "val": {
                "asset_ids": [index],
                "source_count": 1,
                "image_count": frame_count,
                "zero_annotation_image_count": zero_count,
                "annotation_count": sum(support.values()),
                "classes": classes,
            },
            "validation_evaluability": {
                label: "evaluable" if count else "not_evaluable" for label, count in support.items()
            },
            "split_plan": {
                "path": f"folds/{fold_id}.split.json",
                **_binding(split_path),
                "schema": SPLIT_PLAN_SCHEMA,
            },
            "gate": {"passed": True, "checks": {"valid": True}},
        }
        fold_records.append(fold_record)
        semantic_folds.append({"fold_id": fold_id, "split_plan": split_payload})
        folds[fold_id] = fold_record
    plan_semantic = _canonical_sha(
        {
            "schema": CV_PLAN_SCHEMA,
            "method": "leave-one-source-asset-out",
            "folds": semantic_folds,
        }
    )
    cv_plan = {
        "schema": CV_PLAN_SCHEMA,
        "input_scope": "training_internal_only",
        "method": "deterministic leave-one-source-asset-out",
        "taxonomy": list(CANONICAL_NAMES),
        "inputs": {
            "reference_manifest": {
                "path": "data/ground-truth/training-reference-v2/manifest.json",
                **_binding(reference_path),
                "schema": {"name": "roadlabelops.training-coco-reference", "version": 2},
            },
            "coco": {
                "path": "data/ground-truth/training-reference-v2/annotations.coco.json",
                **_binding(coco_path),
                "schema": {"name": "roadlabelops.training-coco-reference", "version": 2},
            },
        },
        "plan_semantic_sha256": plan_semantic,
        "counts": {
            "source_assets": fold_count,
            "folds": fold_count,
            **({"source_groups": fold_count} if fold_count != 5 else {}),
            "images": total_images,
            "zero_annotation_images": total_zero_annotations,
            "annotations": total_annotations,
            "categories": 8,
        },
        "folds": fold_records,
        "holdout_firewall": {"final_holdout_input_read": False},
        "gate": {
            "passed": True,
            "blocking_reasons": [],
            "checks": {"minimum_unique_source_group_count_met": True},
        },
    }
    cv_path = plan_root / "manifest.json"
    _write_json(cv_path, cv_plan)

    mapping = [
        {"id": index, "canonical": canonical, "model": model}
        for index, (canonical, model) in enumerate(zip(CANONICAL_NAMES, MODEL_NAMES, strict=True))
    ]
    protocol = {
        "schema": PROTOCOL_SCHEMA,
        "protocol_id": "recovery-r1-test",
        "status": "frozen_before_recovery_training",
        "scope": {
            "phase": "Recovery R1",
            "input_scope": "immutable_training_reference_and_training_internal_validation_only",
            "final_holdout_status": "sealed_and_consumed",
            "new_final_holdout_allowed": False,
        },
        "inputs": {
            "training_reference": {
                "path": "data/ground-truth/training-reference-v2/manifest.json",
                **_binding(reference_path),
                "schema": {"name": "roadlabelops.training-coco-reference", "version": 2},
                "annotations": {
                    "path": "data/ground-truth/training-reference-v2/annotations.coco.json",
                    **_binding(coco_path),
                },
                "source_asset_count": fold_count,
                "source_assets": source_assets,
            },
            "loso_plan": {
                "path": "docs/evidence/loso-v1/manifest.json",
                **_binding(cv_path),
                "schema": CV_PLAN_SCHEMA,
                "plan_semantic_sha256": plan_semantic,
                "fold_count": fold_count,
            },
            "base_weights": {"path": "yolo11n.pt", **_binding(base_weights)},
        },
        "taxonomy": {
            "canonical_names": list(CANONICAL_NAMES),
            "model_names": list(MODEL_NAMES),
            "mapping": mapping,
            "output_namespace": "canonical",
            "id_order_changes_allowed": False,
        },
        "validation": {
            "method": "leave-one-source-out",
            "fold_count": fold_count,
        },
        "experiments": {
            "common_training": COMMON,
            "arms": list(ARMS),
            "screening": {"seed": 42, "all_arms": True, "all_folds": True},
            "confirmation": {
                "winner_and_paired_repaired_control": True,
                "control_may_self_compare_when_it_wins": True,
                "seeds": [42, 43, 44],
                "all_folds": True,
                "reuse_identical_seed_42_screening_runs": True,
                "frozen_screening_aggregate_binding_required": True,
            },
            "unregistered_experiments_allowed": False,
        },
        "evaluation": {
            "confidence": 0.4,
            "match_iou": 0.5,
            "nms_iou": 0.75,
            "rider_overlap": 0.25,
            "class_threshold_overrides": {},
            "complete_frame_universe_required": True,
            "image_size": "same_as_experiment_arm_imgsz",
            "threshold_tuning_allowed": False,
        },
        "selection": {
            "ranking": RANKING,
            "metrics_must_be_aggregated_from_raw_oof_counts": True,
        },
        "readiness_gates": READINESS,
        "implementation": {
            "revision": "synthetic-recovery-revision",
            "files": [{"path": "scripts/recovery_training_impl.py", **_binding(implementation)}],
        },
    }
    protocol_path = evidence_root / "recovery-protocol.json"
    _write_json(protocol_path, protocol)

    datasets: dict[str, dict[str, Any]] = {}
    for fold_id, fold in folds.items():
        dataset_root = root / "data/training" / fold_id
        yaml_path = dataset_root / "dataset.yaml"
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        yaml_path.write_text("path: .\ntrain: images/train\nval: images/val\n", encoding="utf-8")
        file_records = [{"path": "dataset.yaml", **_binding(yaml_path)}]
        for split_name in ("train", "val"):
            partition = fold[split_name]
            frame_count = partition["image_count"]
            zero_count = partition["zero_annotation_image_count"]
            nonzero_count = frame_count - zero_count
            buckets: list[list[str]] = [[] for _ in range(frame_count)]
            cursor = 0
            for class_id, label in enumerate(CANONICAL_NAMES):
                for _ in range(partition["classes"][label]["box_count"]):
                    buckets[cursor % nonzero_count].append(f"{class_id} 0.5 0.5 0.1 0.1")
                    cursor += 1
            for frame_index, lines in enumerate(buckets, start=1):
                stem = f"{split_name}-{frame_index:03d}"
                image_path = dataset_root / "images" / split_name / f"{stem}.jpg"
                label_path = dataset_root / "labels" / split_name / f"{stem}.txt"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                label_path.parent.mkdir(parents=True, exist_ok=True)
                image_path.write_bytes(f"image:{fold_id}:{stem}".encode())
                label_path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
                file_records.extend(
                    [
                        {
                            "path": image_path.relative_to(dataset_root).as_posix(),
                            **_binding(image_path),
                        },
                        {
                            "path": label_path.relative_to(dataset_root).as_posix(),
                            **_binding(label_path),
                        },
                    ]
                )
        file_records.sort(key=lambda item: item["path"])
        managed_sha = _canonical_sha(file_records)
        support = {label: fold["val"]["classes"][label]["box_count"] for label in CANONICAL_NAMES}
        counts = {
            "images": {
                "total": total_images,
                "train": total_images - fold["val"]["image_count"],
                "val": fold["val"]["image_count"],
                "zero_annotations": total_zero_annotations,
            },
            "annotations": {
                "total": total_annotations,
                "train": total_annotations - sum(support.values()),
                "val": sum(support.values()),
                "by_category": {label: class_totals[label] for label in CANONICAL_NAMES},
                "by_split_and_category": {
                    "train": {
                        label: class_totals[label] - count for label, count in support.items()
                    },
                    "val": support,
                },
            },
            "assets": {"total": fold_count, "train": fold_count - 1, "val": 1},
        }
        split_record = fold["split_plan"]
        manifest = {
            "schema": DATASET_SCHEMA,
            "taxonomy": TAXONOMY,
            "gate": {"passed": True, "blocking_reasons": [], "checks": {"valid": True}},
            "inputs": {
                "coco": {
                    "file_name": "annotations.coco.json",
                    "sha256": _binding(coco_path)["sha256"],
                    "semantic_sha256": _canonical_sha(coco_payload),
                },
                "reference_manifest": {
                    "file_name": "manifest.json",
                    "sha256": _binding(reference_path)["sha256"],
                    "schema": {"name": "roadlabelops.training-coco-reference", "version": 2},
                },
                "split_plan": {
                    "file_name": Path(split_record["path"]).name,
                    "sha256": split_record["sha256"],
                    "semantic_sha256": _canonical_sha(
                        json.loads((plan_root / split_record["path"]).read_text(encoding="utf-8"))
                    ),
                    "schema": SPLIT_PLAN_SCHEMA,
                },
            },
            "split": {
                "method": "explicit typed source_asset_id plan constrained by source leakage group",
                "train_asset_ids": fold["train"]["asset_ids"],
                "val_asset_ids": fold["val"]["asset_ids"],
            },
            "counts": counts,
            "files": file_records,
        }
        manifest_path = dataset_root / "manifest.json"
        _write_json(manifest_path, manifest)
        datasets[fold_id] = {
            "root": dataset_root,
            "manifest": manifest,
            "manifest_path": manifest_path,
            "yaml_path": yaml_path,
            "managed_sha": managed_sha,
            "files": file_records,
        }

    fixture = EvidenceFixture(
        root=root,
        protocol=protocol_path,
        base_weights=base_weights,
        folds=folds,
        datasets=datasets,
        screening_reports=[],
    )
    fixture.screening_reports = [
        _make_report(fixture, arm["arm_id"], 42, fold_id)
        for arm in ARMS
        for fold_id in sorted(folds)
    ]
    return fixture


def _quality_for(
    arm_id: str, fold_index: int, support: int, *, degraded: bool = False
) -> tuple[int, int, int]:
    if support == 0:
        return 0, 0, 0
    if degraded:
        return 0, 0, support
    if arm_id == "small_target_960":
        missed = 1 if fold_index == 1 else 0
    elif arm_id == "repaired_control":
        missed = min(1, support)
    else:
        missed = min(2, support)
    return support - missed, missed, missed


def _metrics(
    fixture: EvidenceFixture,
    arm_id: str,
    fold_id: str,
    *,
    degraded: bool = False,
) -> tuple[dict[str, Any], int]:
    fold = fixture.folds[fold_id]
    fold_index = int(fold_id[-2:])
    per_class: dict[str, Any] = {}
    total_tp = total_fp = total_fn = 0
    for label in CANONICAL_NAMES:
        support = fold["val"]["classes"][label]["box_count"]
        tp, fp, fn = _quality_for(arm_id, fold_index, support, degraded=degraded)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        precision = _ratio(tp, tp + fp)
        recall = _ratio(tp, tp + fn)
        per_class[label] = {
            "status": "evaluable" if support else "not_evaluable",
            "support_count": support,
            "true_positive_count": tp,
            "false_positive_count": fp,
            "false_negative_count": fn,
            "prediction_count": tp + fp,
            "precision": precision if support else None,
            "recall": recall if support else None,
            "f1_score": _count_f1(tp, fp, fn) if support else None,
        }
    frame_count = fold["val"]["image_count"]
    if degraded:
        clean = 0
    elif arm_id == "small_target_960":
        clean = frame_count - (1 if fold_index == 1 else 0)
    elif arm_id == "repaired_control":
        clean = max(0, frame_count - 1)
    else:
        clean = max(0, frame_count - 2)
    precision = _ratio(total_tp, total_tp + total_fp)
    recall = _ratio(total_tp, total_tp + total_fn)
    map_by_arm = {"small_target_960": 0.92, "repaired_control": 0.82, "class_balance_025": 0.72}
    return {
        "map50": map_by_arm[arm_id] + 0.02,
        "map50_95": map_by_arm[arm_id],
        "overall": {
            "true_positive_count": total_tp,
            "false_positive_count": total_fp,
            "false_negative_count": total_fn,
            "prediction_count": total_tp + total_fp,
            "ground_truth_count": total_tp + total_fn,
            "precision": precision,
            "recall": recall,
            "f1_score": _count_f1(total_tp, total_fp, total_fn),
            "evaluated_frame_count": frame_count,
            "clean_frame_count": clean,
            "clean_frame_rate": clean / frame_count,
            "complete_frame_coverage": True,
        },
        "per_class": per_class,
    }, clean


def _make_report(
    fixture: EvidenceFixture,
    arm_id: str,
    seed: int,
    fold_id: str,
    *,
    degraded: bool = False,
) -> Path:
    arm = next(item for item in ARMS if item["arm_id"] == arm_id)
    dataset = fixture.datasets[fold_id]
    candidate_root = fixture.root / "data/model-candidates" / f"{arm_id}-{seed}-{fold_id}"
    weight_path = candidate_root / "weights/best.pt"
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    weight_path.write_bytes(f"weight:{arm_id}:{seed}:{fold_id}".encode())
    last_weight_path = candidate_root / "weights/last.pt"
    last_weight_path.write_bytes(f"last-weight:{arm_id}:{seed}:{fold_id}".encode())
    resolved = {
        "model_family": "YOLO11n",
        **COMMON,
        **{key: value for key, value in arm.items() if key != "arm_id"},
        "seed": seed,
    }
    candidate_protocol = {
        "schema": CANDIDATE_PROTOCOL_SCHEMA,
        "model_family": "YOLO11n",
        "taxonomy": TAXONOMY,
        "training": {
            key: value for key, value in resolved.items() if key not in {"model_family", "seed"}
        },
        "validation_selection": {
            "primary": "mAP50-95",
            "tie_breakers": ["mAP50", "recall", "precision", "smaller_seed"],
        },
        "holdout_access": "prohibited",
    }
    metrics, _clean = _metrics(fixture, arm_id, fold_id, degraded=degraded)
    manifest_path = dataset["manifest_path"]
    yaml_path = dataset["yaml_path"]
    args_path = candidate_root / "artifacts/args.yaml"
    results_path = candidate_root / "artifacts/results.csv"
    args_path.parent.mkdir(parents=True, exist_ok=True)
    isolated_root = (
        fixture.root / "data/model-candidates" / f".{candidate_root.name}.workspace-synthetic"
    )
    trainer_args = {
        **{
            key: (False if key == "cache" and value == "none" else value)
            for key, value in resolved.items()
            if key != "model_family"
        },
        "task": "detect",
        "mode": "train",
        "model": str(isolated_root / "inputs/yolo11n.pt"),
        "data": str(isolated_root / "dataset/dataset.yaml"),
        "project": str(isolated_root / "trainer-output"),
        "name": "train",
        "exist_ok": False,
        "save": True,
        "save_period": -1,
        "val": True,
        "plots": False,
        "resume": False,
        "pretrained": True,
        "cls_remap": True,
        "single_cls": False,
        "rect": False,
        "cos_lr": False,
        "fraction": 1.0,
        "multi_scale": 0.0,
        "dropout": 0.0,
        "classes": None,
        "split": "val",
    }
    args_path.write_text(yaml.safe_dump(trainer_args, sort_keys=True), encoding="utf-8")
    results_path.write_text(
        (
            "epoch,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),"
            "metrics/mAP50-95(B)\n"
            f"1,0.95,0.95,{metrics['map50']},{metrics['map50_95']}\n"
        ),
        encoding="utf-8",
    )
    candidate_per_class = {
        label: {
            "class_id": class_id,
            "status": "evaluable" if support else "not_evaluable",
            "support_count": support,
            "precision": 0.95 if support else None,
            "recall": 0.95 if support else None,
            "map50": metrics["map50"] if support else None,
            "map50_95": metrics["map50_95"] if support else None,
        }
        for class_id, label in enumerate(CANONICAL_NAMES)
        for support in [fixture.folds[fold_id]["val"]["classes"][label]["box_count"]]
    }
    transfer_rows = [
        {
            "target_id": target_id,
            "target_model_name": model_name,
            "canonical_name": canonical_name,
            "source_id": COCO_SOURCE_CLASS_IDS[model_name],
            "source_model_name": model_name,
        }
        for target_id, (model_name, canonical_name) in enumerate(
            zip(MODEL_NAMES, CANONICAL_NAMES, strict=True)
        )
    ]
    artifact_records = {
        "args": {"path": "artifacts/args.yaml", **_binding(args_path)},
        "results": {"path": "artifacts/results.csv", **_binding(results_path)},
        "best_weights": {"path": "weights/best.pt", **_binding(weight_path)},
        "last_weights": {"path": "weights/last.pt", **_binding(last_weight_path)},
    }
    receipt = {
        "schema": CANDIDATE_SCHEMA,
        "gate": {
            "passed": True,
            "checks": {key: True for key in CURRENT_CANDIDATE_GATE_CHECKS},
        },
        "mutation_performed": True,
        "protocol": candidate_protocol,
        "protocol_sha256": _canonical_sha(candidate_protocol),
        "resolved_args": resolved,
        "inputs": {
            "dataset": {
                "manifest": {"schema": DATASET_SCHEMA, **_binding(manifest_path)},
                "dataset_yaml": _binding(yaml_path),
                "managed_files_sha256": dataset["managed_sha"],
                "managed_file_count": len(dataset["files"]),
                "counts": dataset["manifest"]["counts"],
                "taxonomy": TAXONOMY,
            },
            "base_weights": {
                "file_name": "yolo11n.pt",
                "model_family": "YOLO11n",
                **_binding(fixture.base_weights),
            },
        },
        "timestamps": {
            "started_at": "2026-09-02T00:00:00.000000Z",
            "finished_at": "2026-09-02T00:00:10.000000Z",
            "duration_seconds": 10.0,
        },
        "environment": {
            "python": {"version": "3.11.0", "implementation": "CPython"},
            "platform": {"system": "Darwin", "release": "test", "machine": "arm64"},
            "packages": {"ultralytics": "8.4.135"},
            "git": {"commit": "synthetic", "dirty": False},
        },
        "metrics": {
            "aggregate": {
                "precision": 0.95,
                "recall": 0.95,
                "map50": metrics["map50"],
                "map50_95": metrics["map50_95"],
                "fitness": metrics["map50_95"],
            },
            "per_class": candidate_per_class,
        },
        "best_epoch": {
            "index": 0,
            "number": 1,
            "selection_fitness": metrics["map50_95"],
            "epochs_recorded": 1,
        },
        "pretrained_transfer": {
            "schema": PRETRAINED_TRANSFER_SCHEMA,
            "source_model": {
                "family": "YOLO11n",
                "base_weights_sha256": _binding(fixture.base_weights)["sha256"],
                "class_count": 80,
                "names_sha256": "a" * 64,
            },
            "target": {
                "class_count": 8,
                "model_names": list(MODEL_NAMES),
                "canonical_names": list(CANONICAL_NAMES),
            },
            "matched_rows": transfer_rows,
            "matched_row_count": 8,
            "target_row_count": 8,
            "runtime_observation": {
                "verification_mode": "ultralytics_logger",
                "message": PRETRAINED_TRANSFER_MESSAGE,
                "message_sha256": hashlib.sha256(
                    PRETRAINED_TRANSFER_MESSAGE.encode("utf-8")
                ).hexdigest(),
                "matched_row_count": 8,
                "target_row_count": 8,
            },
        },
        "artifacts": artifact_records,
        "holdout": {
            "input_read": False,
            "statement": "No configured final holdout input was read.",
        },
    }
    receipt_path = candidate_root / "receipt.json"
    _write_json(receipt_path, receipt)
    fold = fixture.folds[fold_id]
    split_record = fold["split_plan"]
    settings = {
        "confidence": 0.4,
        "image_size": arm["imgsz"],
        "device": "mps",
        "nms_iou": 0.75,
        "rider_overlap": 0.25,
        "match_iou": 0.5,
    }
    frame_count = fold["val"]["image_count"]
    managed_by_path = {record["path"]: record for record in dataset["files"]}
    val_images = sorted(
        (
            record
            for record in dataset["files"]
            if Path(record["path"]).parts[:2] == ("images", "val")
        ),
        key=lambda record: (record["path"].casefold(), record["path"]),
    )
    frame_records = []
    for image_record in val_images:
        image_path = Path(image_record["path"])
        label_name = (Path("labels/val") / image_path.with_suffix(".txt").name).as_posix()
        label_record = managed_by_path[label_name]
        frame_records.append(
            {
                "scene_id": image_record["path"],
                "frame": 0,
                "image_path": image_record["path"],
                "image_sha256": image_record["sha256"],
                "label_path": label_record["path"],
                "label_sha256": label_record["sha256"],
            }
        )
    report = {
        "schema": EVALUATION_SCHEMA,
        "gate": {"passed": True, "checks": {"evaluated": True}},
        "experiment": {"arm_id": arm_id, "seed": seed, "fold_id": fold_id},
        "settings": settings,
        "bindings": {
            "dataset": {
                "root": f"data/training/{fold_id}",
                "manifest": {
                    "path": "manifest.json",
                    **_binding(manifest_path),
                    "schema": DATASET_SCHEMA,
                },
                "dataset_yaml": {"path": "dataset.yaml", **_binding(yaml_path)},
                "managed_files_sha256": dataset["managed_sha"],
                "managed_file_count": len(dataset["files"]),
                "taxonomy": TAXONOMY,
            },
            "candidate": {
                "receipt": {
                    "path": f"data/model-candidates/{candidate_root.name}/receipt.json",
                    **_binding(receipt_path),
                    "schema": CANDIDATE_SCHEMA,
                },
                "weight": {
                    "path": f"data/model-candidates/{candidate_root.name}/weights/best.pt",
                    **_binding(weight_path),
                    "artifact": "best_weights",
                },
                "protocol_sha256": _canonical_sha(candidate_protocol),
            },
            "fold": {
                "binding_mode": "dataset_manifest_split_plan_and_val_assets",
                "manifest_fold_id": None,
                "split_plan": {
                    "file_name": Path(split_record["path"]).name,
                    "sha256": split_record["sha256"],
                    "semantic_sha256": dataset["manifest"]["inputs"]["split_plan"][
                        "semantic_sha256"
                    ],
                    "schema": SPLIT_PLAN_SCHEMA,
                },
                "val_asset_ids": fold["val"]["asset_ids"],
            },
        },
        "val_source": {
            "split": "val",
            "asset_ids": fold["val"]["asset_ids"],
            "frame_count": frame_count,
            "zero_annotation_frame_count": fold["val"]["zero_annotation_image_count"],
            "annotation_count": fold["val"]["annotation_count"],
            "frames_sha256": _canonical_sha(frame_records),
        },
        "metrics": metrics,
        "compute": {
            "training_duration_seconds": 10.0,
            "evaluation_inference_seconds": float(frame_count),
            "model_load_seconds": 1.0,
            "evaluated_frames_per_second": 1.0,
            "predict_call_count": 1,
        },
        "holdout_firewall": {
            "input_read": False,
            "statement": "No configured final holdout input was read.",
            "allowed_scopes": ["data/training", "data/model-candidates"],
            "rejected_scopes": ["data/holdout", "configured-final-holdout"],
        },
    }
    report_path = fixture.root / "docs/evidence/reports" / f"{arm_id}-{seed}-{fold_id}.json"
    _write_json(report_path, report)
    return report_path


def _aggregate_screening(fixture: EvidenceFixture, output_name: str = "screening.json") -> Path:
    output = fixture.root / "docs/evidence" / output_name
    aggregate_training_cv(
        mode="screening",
        recovery_protocol_path=fixture.protocol,
        evaluation_report_paths=fixture.screening_reports,
        output=output,
        workspace_root=fixture.root,
    )
    return output


def _receipt_for_report(fixture: EvidenceFixture, report_path: Path) -> tuple[Path, dict[str, Any]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    relative = report["bindings"]["candidate"]["receipt"]["path"]
    return fixture.root / relative, report


def _refresh_report_receipt_binding(report_path: Path, receipt_path: Path) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["bindings"]["candidate"]["receipt"].update(_binding(receipt_path))
    _write_json(report_path, report)


def _assert_screening_rejected(
    fixture: EvidenceFixture,
    *,
    match: str,
    output_name: str = "rejected.json",
) -> None:
    with pytest.raises(TrainingCVAggregationError, match=match):
        aggregate_training_cv(
            mode="screening",
            recovery_protocol_path=fixture.protocol,
            evaluation_report_paths=fixture.screening_reports,
            output=fixture.root / "docs/evidence" / output_name,
            workspace_root=fixture.root,
        )


def test_screening_is_deterministic_micro_aggregated_and_ranked(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    first = _aggregate_screening(fixture, "screening-a.json")
    second = _aggregate_screening(fixture, "screening-b.json")

    assert first.read_bytes() == second.read_bytes()
    payload = json.loads(first.read_text(encoding="utf-8"))
    assert payload["schema"] == OUTPUT_SCHEMA
    assert payload["selection"]["winner_arm_id"] == "small_target_960"
    target = next(item for item in payload["aggregates"] if item["arm_id"] == "small_target_960")
    source_mean_recall = (
        sum(
            item["true_positive_count"]
            / (item["true_positive_count"] + item["false_negative_count"])
            for item in target["source_strata"]
        )
        / 5
    )
    assert target["oof"]["recall"] == pytest.approx(112 / 119)
    assert target["oof"]["recall"] != pytest.approx(source_mean_recall)
    assert target["oof"]["zero_annotation_frame_count"] == 1
    fold_one = target["source_strata"][0]
    assert fold_one["per_class"]["traffic_sign"] == {
        "status": "not_evaluable",
        "support_count": 0,
        "recall": None,
        "f1_score": None,
    }
    with pytest.raises(TrainingCVAggregationError, match="already exists"):
        aggregate_training_cv(
            mode="screening",
            recovery_protocol_path=fixture.protocol,
            evaluation_report_paths=fixture.screening_reports,
            output=first,
            workspace_root=fixture.root,
        )


def test_six_fold_screening_and_confirmation_cover_bound_fold_set(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path, fold_count=6)
    screening = _aggregate_screening(fixture)
    screening_payload = json.loads(screening.read_text(encoding="utf-8"))

    assert screening_payload["contract"]["fold_ids"] == [
        f"fold-{index:02d}" for index in range(1, 7)
    ]
    assert screening_payload["run_matrix"]["required_count"] == 18
    assert screening_payload["gate"]["checks"]["source_derived_loso_plan_verified"] is True

    confirmation_reports = fixture.confirmation_reports(
        arms=("small_target_960", "repaired_control")
    )
    output = tmp_path / "docs/evidence/confirmation-six-fold.json"
    summary = aggregate_training_cv(
        mode="confirmation",
        recovery_protocol_path=fixture.protocol,
        screening_aggregate_path=screening,
        winner_arm_id="small_target_960",
        evaluation_report_paths=confirmation_reports,
        output=output,
        workspace_root=tmp_path,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert summary["run_count"] == 36
    assert payload["run_matrix"]["required_count"] == 36
    assert len(payload["readiness"]["paired_source_comparisons"]) == 18
    assert sum(run["provenance"] == "reused_from_frozen_screening" for run in payload["runs"]) == 12


def test_rejects_fewer_than_three_source_groups(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path, fold_count=2)

    with pytest.raises(TrainingCVAggregationError, match="at least 3 unique source"):
        aggregate_training_cv(
            mode="screening",
            recovery_protocol_path=fixture.protocol,
            evaluation_report_paths=fixture.screening_reports,
            output=tmp_path / "docs/evidence/rejected-two-fold.json",
            workspace_root=tmp_path,
        )


def test_six_fold_screening_rejects_missing_and_extra_fold_reports(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path, fold_count=6)
    with pytest.raises(TrainingCVAggregationError, match="run matrix differs"):
        aggregate_training_cv(
            mode="screening",
            recovery_protocol_path=fixture.protocol,
            evaluation_report_paths=fixture.screening_reports[:-1],
            output=tmp_path / "docs/evidence/missing-six-fold.json",
            workspace_root=tmp_path,
        )

    extra = fixture.screening_reports[0]
    payload = json.loads(extra.read_text(encoding="utf-8"))
    payload["experiment"]["fold_id"] = "fold-07"
    _write_json(extra, payload)
    with pytest.raises(TrainingCVAggregationError, match="unregistered arm or fold"):
        aggregate_training_cv(
            mode="screening",
            recovery_protocol_path=fixture.protocol,
            evaluation_report_paths=fixture.screening_reports,
            output=tmp_path / "docs/evidence/extra-six-fold.json",
            workspace_root=tmp_path,
        )


def test_screening_rejects_missing_duplicate_and_settings_drift(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    with pytest.raises(TrainingCVAggregationError, match="run matrix differs"):
        aggregate_training_cv(
            mode="screening",
            recovery_protocol_path=fixture.protocol,
            evaluation_report_paths=fixture.screening_reports[:-1],
            output=tmp_path / "docs/evidence/missing.json",
            workspace_root=tmp_path,
        )
    with pytest.raises(TrainingCVAggregationError, match="duplicate arm/seed/fold"):
        aggregate_training_cv(
            mode="screening",
            recovery_protocol_path=fixture.protocol,
            evaluation_report_paths=[*fixture.screening_reports, fixture.screening_reports[0]],
            output=tmp_path / "docs/evidence/duplicate.json",
            workspace_root=tmp_path,
        )

    drifted = fixture.screening_reports[0]
    payload = json.loads(drifted.read_text(encoding="utf-8"))
    payload["settings"]["confidence"] = 0.39
    _write_json(drifted, payload)
    with pytest.raises(TrainingCVAggregationError, match="settings drifted"):
        aggregate_training_cv(
            mode="screening",
            recovery_protocol_path=fixture.protocol,
            evaluation_report_paths=fixture.screening_reports,
            output=tmp_path / "docs/evidence/drift.json",
            workspace_root=tmp_path,
        )


def test_firewall_and_symlink_inputs_are_rejected(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    forbidden = tmp_path / "data/holdout/report.json"
    with pytest.raises(TrainingCVAggregationError, match="firewall"):
        aggregate_training_cv(
            mode="screening",
            recovery_protocol_path=fixture.protocol,
            evaluation_report_paths=[forbidden, *fixture.screening_reports[1:]],
            output=tmp_path / "docs/evidence/firewall.json",
            workspace_root=tmp_path,
        )

    link = tmp_path / "docs/evidence/reports/symlink.json"
    os.symlink(fixture.screening_reports[0], link)
    with pytest.raises(TrainingCVAggregationError, match="symlink"):
        aggregate_training_cv(
            mode="screening",
            recovery_protocol_path=fixture.protocol,
            evaluation_report_paths=[link, *fixture.screening_reports[1:]],
            output=tmp_path / "docs/evidence/symlink-output.json",
            workspace_root=tmp_path,
        )


def test_confirmation_reuses_screening_and_applies_paired_readiness(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    screening = _aggregate_screening(fixture)
    confirmation_reports = fixture.confirmation_reports(
        arms=("small_target_960", "repaired_control")
    )
    output = tmp_path / "docs/evidence/confirmation.json"
    summary = aggregate_training_cv(
        mode="confirmation",
        recovery_protocol_path=fixture.protocol,
        screening_aggregate_path=screening,
        winner_arm_id="small_target_960",
        evaluation_report_paths=confirmation_reports,
        output=output,
        workspace_root=tmp_path,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert summary["run_count"] == 30
    assert payload["gate"]["passed"] is True
    assert payload["readiness"]["winner_seed_oof_f1_sample_standard_deviation"] == 0.0
    assert all(item["passed"] for item in payload["readiness"]["paired_source_comparisons"])
    assert sum(run["provenance"] == "reused_from_frozen_screening" for run in payload["runs"]) == 10
    assert (
        payload["bindings"]["screening_aggregate"]["sha256"]
        == hashlib.sha256(screening.read_bytes()).hexdigest()
    )


def test_confirmation_writes_auditable_failed_gate(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    screening = _aggregate_screening(fixture)
    reports = fixture.confirmation_reports(arms=("small_target_960", "repaired_control"))
    for fold_id in fixture.folds:
        degraded = _make_report(fixture, "small_target_960", 43, fold_id, degraded=True)
        assert degraded in reports
    output = tmp_path / "docs/evidence/confirmation-failed.json"
    summary = aggregate_training_cv(
        mode="confirmation",
        recovery_protocol_path=fixture.protocol,
        screening_aggregate_path=screening,
        winner_arm_id="small_target_960",
        evaluation_report_paths=reports,
        output=output,
        workspace_root=tmp_path,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert summary["gate"]["passed"] is False
    assert payload["strata"]["small_target_960"]["43"]["oof"]["precision"] is None
    assert payload["strata"]["small_target_960"]["43"]["oof"]["f1_score"] == 0.0
    assert payload["readiness"]["checks"]["winner_seed_thresholds_pass"] is False
    assert (
        payload["readiness"]["checks"]["every_seed_and_source_f1_noninferior_to_repaired_control"]
        is False
    )
    assert "winner_seed_thresholds_pass" in payload["gate"]["blocking_reasons"]
    assert payload["holdout_firewall"]["input_read"] is False


def test_confirmation_rejects_tampered_screening_report_binding(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    screening = _aggregate_screening(fixture)
    screening_payload = json.loads(screening.read_text(encoding="utf-8"))
    screening_payload["runs"][0]["report"]["sha256"] = "0" * 64
    tampered = tmp_path / "docs/evidence/tampered-screening.json"
    _write_json(tampered, screening_payload)
    reports = fixture.confirmation_reports(arms=("small_target_960", "repaired_control"))
    with pytest.raises(TrainingCVAggregationError, match="changed after"):
        aggregate_training_cv(
            mode="confirmation",
            recovery_protocol_path=fixture.protocol,
            screening_aggregate_path=tampered,
            winner_arm_id="small_target_960",
            evaluation_report_paths=reports,
            output=tmp_path / "docs/evidence/should-not-exist.json",
            workspace_root=tmp_path,
        )


def test_confirmation_rejects_seed42_duplicate(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    screening = _aggregate_screening(fixture)
    reports = fixture.confirmation_reports(arms=("small_target_960", "repaired_control"))
    reports.append(
        next(
            path for path in fixture.screening_reports if "small_target_960-42-fold-01" in path.name
        )
    )
    with pytest.raises(TrainingCVAggregationError, match="must be reused"):
        aggregate_training_cv(
            mode="confirmation",
            recovery_protocol_path=fixture.protocol,
            screening_aggregate_path=screening,
            winner_arm_id="small_target_960",
            evaluation_report_paths=reports,
            output=tmp_path / "docs/evidence/duplicate-seed42.json",
            workspace_root=tmp_path,
        )


def test_protocol_requires_screening_seed_42(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    protocol = json.loads(fixture.protocol.read_text(encoding="utf-8"))
    protocol["experiments"]["screening"]["seed"] = 41
    _write_json(fixture.protocol, protocol)

    _assert_screening_rejected(fixture, match="screening must cover")


def test_protocol_rejects_reference_source_leakage_alias(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    protocol = json.loads(fixture.protocol.read_text(encoding="utf-8"))
    reference_path = fixture.root / protocol["inputs"]["training_reference"]["path"]
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    duplicate_group = reference["source_statistics"]["assets"][0]["leakage_group_id"]
    reference["source_statistics"]["assets"][1]["leakage_group_id"] = duplicate_group
    _write_json(reference_path, reference)
    protocol["inputs"]["training_reference"].update(_binding(reference_path))
    protocol["inputs"]["training_reference"]["source_assets"][1]["leakage_group_id"] = (
        duplicate_group
    )
    _write_json(fixture.protocol, protocol)

    _assert_screening_rejected(fixture, match="leakage groups must be unique")


def test_protocol_rejects_split_plan_train_val_drift(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    protocol = json.loads(fixture.protocol.read_text(encoding="utf-8"))
    cv_path = fixture.root / protocol["inputs"]["loso_plan"]["path"]
    cv_plan = json.loads(cv_path.read_text(encoding="utf-8"))
    first_fold = cv_plan["folds"][0]
    split_path = cv_path.parent / first_fold["split_plan"]["path"]
    split = json.loads(split_path.read_text(encoding="utf-8"))
    split["train_asset_ids"] = [1, 2, 3, 4]
    _write_json(split_path, split)
    first_fold["split_plan"].update(_binding(split_path))
    semantic_folds = []
    for fold in cv_plan["folds"]:
        bound_split_path = cv_path.parent / fold["split_plan"]["path"]
        semantic_folds.append(
            {
                "fold_id": fold["fold_id"],
                "split_plan": json.loads(bound_split_path.read_text(encoding="utf-8")),
            }
        )
    cv_plan["plan_semantic_sha256"] = _canonical_sha(
        {
            "schema": CV_PLAN_SCHEMA,
            "method": "leave-one-source-asset-out",
            "folds": semantic_folds,
        }
    )
    _write_json(cv_path, cv_plan)
    protocol["inputs"]["loso_plan"].update(_binding(cv_path))
    protocol["inputs"]["loso_plan"]["plan_semantic_sha256"] = cv_plan["plan_semantic_sha256"]
    _write_json(fixture.protocol, protocol)

    _assert_screening_rejected(fixture, match="split train/val assets differ")


@pytest.mark.parametrize("dependency", ["base_weights", "implementation"])
def test_protocol_revalidates_frozen_dependency_bytes(tmp_path: Path, dependency: str) -> None:
    fixture = _make_fixture(tmp_path)
    target = (
        fixture.base_weights
        if dependency == "base_weights"
        else fixture.root / "scripts/recovery_training_impl.py"
    )
    target.write_bytes(target.read_bytes() + b"tampered")

    _assert_screening_rejected(fixture, match="binding differs")


def test_dataset_manifest_train_counts_are_bound_to_fold(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    report_path = fixture.screening_reports[0]
    receipt_path, report = _receipt_for_report(fixture, report_path)
    fold_id = report["experiment"]["fold_id"]
    dataset = fixture.datasets[fold_id]
    manifest_path = dataset["manifest_path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["counts"]["images"]["train"] += 1
    manifest["counts"]["images"]["total"] += 1
    _write_json(manifest_path, manifest)
    report["bindings"]["dataset"]["manifest"].update(_binding(manifest_path))
    _write_json(report_path, report)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["inputs"]["dataset"]["manifest"].update(_binding(manifest_path))
    receipt["inputs"]["dataset"]["counts"] = manifest["counts"]
    _write_json(receipt_path, receipt)
    _refresh_report_receipt_binding(report_path, receipt_path)

    _assert_screening_rejected(fixture, match="image/asset counts differ")


def test_dataset_manifest_split_preserves_typed_source_ids(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    report_path = fixture.screening_reports[0]
    receipt_path, report = _receipt_for_report(fixture, report_path)
    fold_id = report["experiment"]["fold_id"]
    dataset = fixture.datasets[fold_id]
    manifest_path = dataset["manifest_path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["split"]["train_asset_ids"][0] = str(manifest["split"]["train_asset_ids"][0])
    _write_json(manifest_path, manifest)
    report["bindings"]["dataset"]["manifest"].update(_binding(manifest_path))
    _write_json(report_path, report)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["inputs"]["dataset"]["manifest"].update(_binding(manifest_path))
    _write_json(receipt_path, receipt)
    _refresh_report_receipt_binding(report_path, receipt_path)

    _assert_screening_rejected(fixture, match="train/validation assets differ")


def test_dataset_managed_training_labels_are_recounted(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    report_path = fixture.screening_reports[0]
    receipt_path, report = _receipt_for_report(fixture, report_path)
    fold_id = report["experiment"]["fold_id"]
    dataset = fixture.datasets[fold_id]
    manifest_path = dataset["manifest_path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    label_record = next(
        record for record in manifest["files"] if record["path"].startswith("labels/train/")
    )
    label_path = dataset["root"] / label_record["path"]
    label_path.write_text(label_path.read_text(encoding="utf-8") + "0 0.5 0.5 0.1 0.1\n")
    label_record.update(_binding(label_path))
    manifest["files"].sort(key=lambda record: record["path"])
    managed_sha = _canonical_sha(manifest["files"])
    _write_json(manifest_path, manifest)
    report["bindings"]["dataset"]["manifest"].update(_binding(manifest_path))
    report["bindings"]["dataset"]["managed_files_sha256"] = managed_sha
    _write_json(report_path, report)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["inputs"]["dataset"]["manifest"].update(_binding(manifest_path))
    receipt["inputs"]["dataset"]["managed_files_sha256"] = managed_sha
    _write_json(receipt_path, receipt)
    _refresh_report_receipt_binding(report_path, receipt_path)

    _assert_screening_rejected(fixture, match="train label inventory differs")


def test_validation_frames_digest_is_recomputed_from_managed_inventory(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    report_path = fixture.screening_reports[0]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["val_source"]["frames_sha256"] = "0" * 64
    _write_json(report_path, report)

    _assert_screening_rejected(fixture, match="frames digest differs")


def test_candidate_rejects_arbitrary_gate_and_artifact_drift(tmp_path: Path) -> None:
    gate_fixture = _make_fixture(tmp_path / "gate")
    gate_report = gate_fixture.screening_reports[0]
    gate_receipt_path, _ = _receipt_for_report(gate_fixture, gate_report)
    gate_receipt = json.loads(gate_receipt_path.read_text(encoding="utf-8"))
    gate_receipt["gate"] = {"passed": True, "checks": {"trained": True}}
    _write_json(gate_receipt_path, gate_receipt)
    _refresh_report_receipt_binding(gate_report, gate_receipt_path)
    _assert_screening_rejected(gate_fixture, match="gate is incomplete")

    artifact_fixture = _make_fixture(tmp_path / "artifact")
    artifact_report = artifact_fixture.screening_reports[0]
    artifact_receipt_path, _ = _receipt_for_report(artifact_fixture, artifact_report)
    args_path = artifact_receipt_path.parent / "artifacts/args.yaml"
    args_path.write_bytes(args_path.read_bytes() + b"\n# drift\n")
    _assert_screening_rejected(artifact_fixture, match="artifact 'args' hash/size differs")


def test_candidate_base_weights_must_match_frozen_protocol(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    report_path = fixture.screening_reports[0]
    receipt_path, _ = _receipt_for_report(fixture, report_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["inputs"]["base_weights"]["sha256"] = "0" * 64
    _write_json(receipt_path, receipt)
    _refresh_report_receipt_binding(report_path, receipt_path)

    _assert_screening_rejected(fixture, match="base weights differ")


def test_candidate_requires_runtime_verified_eight_of_eight_pretrained_transfer(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    report_path = fixture.screening_reports[0]
    receipt_path, _ = _receipt_for_report(fixture, report_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["pretrained_transfer"]["runtime_observation"]["verification_mode"] = (
        "injected_test_double"
    )
    _write_json(receipt_path, receipt)
    _refresh_report_receipt_binding(report_path, receipt_path)

    _assert_screening_rejected(fixture, match="runtime transfer observation")


@pytest.mark.parametrize("artifact_name", ["args", "results"])
def test_candidate_crosschecks_trainer_artifacts(tmp_path: Path, artifact_name: str) -> None:
    fixture = _make_fixture(tmp_path)
    report_path = fixture.screening_reports[0]
    receipt_path, _ = _receipt_for_report(fixture, report_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    artifact_path = receipt_path.parent / receipt["artifacts"][artifact_name]["path"]
    if artifact_name == "args":
        args = yaml.safe_load(artifact_path.read_text(encoding="utf-8"))
        args["batch"] = 99
        artifact_path.write_text(yaml.safe_dump(args, sort_keys=True), encoding="utf-8")
        error = "differs from receipt.resolved_args"
    else:
        artifact_path.write_text(
            (
                "epoch,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),"
                "metrics/mAP50-95(B)\n1,0.1,0.1,0.1,0.1\n"
            ),
            encoding="utf-8",
        )
        error = "candidate best_epoch differs from results.csv"
    receipt["artifacts"][artifact_name].update(_binding(artifact_path))
    _write_json(receipt_path, receipt)
    _refresh_report_receipt_binding(report_path, receipt_path)

    _assert_screening_rejected(fixture, match=error)


@pytest.mark.parametrize("drift", ["class_filter", "unbound_data"])
def test_candidate_rejects_behavior_changing_or_unbound_trainer_args(
    tmp_path: Path, drift: str
) -> None:
    fixture = _make_fixture(tmp_path)
    report_path = fixture.screening_reports[0]
    receipt_path, report = _receipt_for_report(fixture, report_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    args_path = receipt_path.parent / receipt["artifacts"]["args"]["path"]
    args = yaml.safe_load(args_path.read_text(encoding="utf-8"))
    if drift == "class_filter":
        args["classes"] = [0]
        error = "immutable trainer argument 'classes'"
    else:
        fold_id = report["experiment"]["fold_id"]
        args["data"] = str(fixture.datasets[fold_id]["yaml_path"])
        error = "did not use the isolated training workspace"
    args_path.write_text(yaml.safe_dump(args, sort_keys=True), encoding="utf-8")
    receipt["artifacts"]["args"].update(_binding(args_path))
    _write_json(receipt_path, receipt)
    _refresh_report_receipt_binding(report_path, receipt_path)

    _assert_screening_rejected(fixture, match=error)


def test_checkpoint_revalidation_metrics_may_differ_when_best_epoch_is_bound(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    report_path = fixture.screening_reports[0]
    receipt_path, _ = _receipt_for_report(fixture, report_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    results_path = receipt_path.parent / receipt["artifacts"]["results"]["path"]
    results_path.write_text(
        (
            "epoch,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),"
            "metrics/mAP50-95(B)\n"
            "0,0.01,0.02,0.03,0.04\n"
            "1,0.11,0.12,0.13,0.14\n"
        ),
        encoding="utf-8",
    )
    receipt["best_epoch"] = {
        "index": 1,
        "number": 2,
        "selection_fitness": 0.14,
        "epochs_recorded": 2,
    }
    receipt["artifacts"]["results"].update(_binding(results_path))
    _write_json(receipt_path, receipt)
    _refresh_report_receipt_binding(report_path, receipt_path)

    output = _aggregate_screening(fixture, "checkpoint-revalidation.json")

    assert json.loads(output.read_text(encoding="utf-8"))["gate"]["passed"] is True


@pytest.mark.parametrize("metric", ["precision", "recall", "map50", "map50_95"])
@pytest.mark.parametrize("tampered_level", ["aggregate", "per_class"])
def test_candidate_aggregate_must_match_evaluable_per_class_mean(
    tmp_path: Path, tampered_level: str, metric: str
) -> None:
    fixture = _make_fixture(tmp_path)
    report_path = fixture.screening_reports[0]
    receipt_path, _ = _receipt_for_report(fixture, report_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if tampered_level == "aggregate":
        receipt["metrics"]["aggregate"][metric] = 0.94
        if metric == "map50_95":
            receipt["metrics"]["aggregate"]["fitness"] = 0.94
    else:
        record = next(
            record
            for record in receipt["metrics"]["per_class"].values()
            if record["status"] == "evaluable"
        )
        record[metric] = 0.94
    _write_json(receipt_path, receipt)
    _refresh_report_receipt_binding(report_path, receipt_path)

    _assert_screening_rejected(fixture, match="does not equal the non-weighted")


@pytest.mark.parametrize("metric", ["map50", "map50_95"])
def test_report_map_metrics_must_match_checkpoint_receipt(tmp_path: Path, metric: str) -> None:
    fixture = _make_fixture(tmp_path)
    report_path = fixture.screening_reports[0]
    _, report = _receipt_for_report(fixture, report_path)
    report["metrics"][metric] = 0.01
    _write_json(report_path, report)

    _assert_screening_rejected(fixture, match=rf"report {metric} differs from candidate")


@pytest.mark.parametrize(
    ("field", "tampered_value"),
    [
        ("index", 1),
        ("number", 2),
        ("selection_fitness", 0.5),
        ("epochs_recorded", 2),
    ],
)
def test_candidate_best_epoch_must_match_results_csv(
    tmp_path: Path, field: str, tampered_value: float
) -> None:
    fixture = _make_fixture(tmp_path)
    report_path = fixture.screening_reports[0]
    receipt_path, _ = _receipt_for_report(fixture, report_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["best_epoch"][field] = tampered_value
    _write_json(receipt_path, receipt)
    _refresh_report_receipt_binding(report_path, receipt_path)

    _assert_screening_rejected(fixture, match="candidate best_epoch differs from results.csv")


def test_empty_source_f1_is_null_without_predictions_and_zero_with_fp() -> None:
    def run(fold_id: str, *, tp: int, fp: int, fn: int) -> dict[str, Any]:
        support = tp + fn
        per_class = {
            label: {
                "status": "evaluable" if label == "car" and support else "not_evaluable",
                "support_count": support if label == "car" else 0,
                "true_positive_count": tp if label == "car" else 0,
                "false_positive_count": fp if label == "car" else 0,
                "false_negative_count": fn if label == "car" else 0,
                "prediction_count": tp + fp if label == "car" else 0,
                "precision": _ratio(tp, tp + fp) if label == "car" and support else None,
                "recall": _ratio(tp, support) if label == "car" and support else None,
                "f1_score": _count_f1(tp, fp, fn) if label == "car" and support else None,
            }
            for label in CANONICAL_NAMES
        }
        return {
            "experiment": {"arm_id": "arm", "seed": 42, "fold_id": fold_id},
            "val_source": {
                "asset_ids": [fold_id],
                "zero_annotation_frame_count": int(support == 0),
            },
            "metrics": {
                "map50": 0.5,
                "map50_95": 0.4,
                "overall": {
                    "true_positive_count": tp,
                    "false_positive_count": fp,
                    "false_negative_count": fn,
                    "evaluated_frame_count": 1,
                    "clean_frame_count": int(fp == 0 and fn == 0),
                },
                "per_class": per_class,
            },
            "compute": {
                "training_duration_seconds": 1.0,
                "evaluation_inference_seconds": 1.0,
                "model_load_seconds": 1.0,
            },
        }

    clean_empty = run("fold-empty", tp=0, fp=0, fn=0)
    positive = run("fold-positive", tp=4, fp=1, fn=1)
    summary = _summarize_runs([clean_empty, positive])
    empty_source = next(
        item for item in summary["source_strata"] if item["fold_id"] == "fold-empty"
    )
    assert empty_source["f1_score"] is None
    assert summary["worst_source"]["fold_id"] == "fold-positive"
    assert summary["oof"]["zero_annotation_frame_count"] == 1

    empty_with_fp = run("fold-empty", tp=0, fp=1, fn=0)
    summary_with_fp = _summarize_runs([empty_with_fp, positive])
    empty_source_with_fp = next(
        item for item in summary_with_fp["source_strata"] if item["fold_id"] == "fold-empty"
    )
    assert empty_source_with_fp["f1_score"] == 0.0
    assert summary_with_fp["worst_source"]["fold_id"] == "fold-empty"


def _make_reanalysis_fixture(
    root: Path,
    *,
    fold_count: int = 5,
) -> tuple[EvidenceFixture, Path, list[Path]]:
    fixture = _make_fixture(root, fold_count=fold_count)

    def relocate(paths: list[Path]) -> list[Path]:
        relocated: list[Path] = []
        destination_root = root / "data/model-candidates/reanalysis-reports"
        destination_root.mkdir(parents=True, exist_ok=True)
        for source in paths:
            destination = destination_root / source.name
            source.replace(destination)
            relocated.append(destination)
        return relocated

    fixture.screening_reports = relocate(fixture.screening_reports)
    source_protocol_path = fixture.protocol
    source_protocol = json.loads(source_protocol_path.read_text(encoding="utf-8"))
    source_protocol_binding = {
        "path": source_protocol_path.relative_to(root).as_posix(),
        **_binding(source_protocol_path),
        "schema": PROTOCOL_SCHEMA,
        "protocol_id": source_protocol["protocol_id"],
    }
    source_screening = _aggregate_screening(fixture, "source-screening.json")
    source_screening_payload = json.loads(source_screening.read_text(encoding="utf-8"))
    source_screening_binding = {
        "path": source_screening.relative_to(root).as_posix(),
        **_binding(source_screening),
        "schema": OUTPUT_SCHEMA,
        "winner_arm_id": source_screening_payload["selection"]["winner_arm_id"],
    }

    confirmation_reports = relocate(
        fixture.confirmation_reports(arms=("small_target_960", "repaired_control"))
    )
    all_reports = [*fixture.screening_reports, *confirmation_reports]
    report_bindings = []
    for report_path in all_reports:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report_bindings.append(
            {
                "path": report_path.relative_to(root).as_posix(),
                **_binding(report_path),
                "schema": EVALUATION_SCHEMA,
                "experiment": report["experiment"],
            }
        )

    failure_path = root / "docs/evidence/source-confirmation-failure.json"
    failure_schema = {
        "name": "roadlabelops.training-recovery-r1-aggregation-failure",
        "version": 1,
    }
    _write_json(
        failure_path,
        {
            "schema": failure_schema,
            "status": "failed",
            "stage": "confirmation",
            "protocol": source_protocol_binding,
            "aggregate_output": {
                "path": "docs/evidence/source-confirmation.aggregate.json",
                "written": False,
            },
        },
    )
    failure_binding = {
        "path": failure_path.relative_to(root).as_posix(),
        **_binding(failure_path),
        "schema": failure_schema,
    }

    reanalysis_protocol = dict(source_protocol)
    reanalysis_protocol["protocol_id"] = "recovery-r1-reanalysis-test"
    reanalysis_protocol["status"] = "frozen_before_reanalysis_after_collection"
    reanalysis_protocol["reanalysis"] = {
        "mode": "immutable_evidence_reanalysis",
        "source_protocol": source_protocol_binding,
        "source_screening_aggregate": source_screening_binding,
        "failure_evidence": failure_binding,
        "collection_status": "complete_before_reanalysis_protocol_freeze",
        "new_replacement_or_supplemental_runs_allowed": False,
        "report_count": len(report_bindings),
        "reports_manifest_sha256": _canonical_sha(report_bindings),
        "reports": report_bindings,
    }
    reanalysis_protocol_path = root / "docs/evidence/reanalysis-protocol.json"
    _write_json(reanalysis_protocol_path, reanalysis_protocol)
    fixture.protocol = reanalysis_protocol_path
    return fixture, source_screening, confirmation_reports


def test_reanalysis_protocol_aggregates_only_its_frozen_report_lineage(tmp_path: Path) -> None:
    fixture, _source_screening, confirmation_reports = _make_reanalysis_fixture(tmp_path)
    reanalysis_screening = _aggregate_screening(fixture, "reanalysis-screening.json")
    output = tmp_path / "docs/evidence/reanalysis-confirmation.json"

    summary = aggregate_training_cv(
        mode="confirmation",
        recovery_protocol_path=fixture.protocol,
        screening_aggregate_path=reanalysis_screening,
        winner_arm_id="small_target_960",
        evaluation_report_paths=confirmation_reports,
        output=output,
        workspace_root=tmp_path,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert summary["run_count"] == 30
    assert payload["gate"]["passed"] is True
    assert payload["bindings"]["recovery_protocol"]["protocol_id"] == (
        "recovery-r1-reanalysis-test"
    )


def test_six_fold_reanalysis_lineage_covers_every_bound_fold(tmp_path: Path) -> None:
    fixture, _source_screening, confirmation_reports = _make_reanalysis_fixture(
        tmp_path, fold_count=6
    )
    reanalysis_screening = _aggregate_screening(fixture, "reanalysis-screening-six-fold.json")
    output = tmp_path / "docs/evidence/reanalysis-confirmation-six-fold.json"

    summary = aggregate_training_cv(
        mode="confirmation",
        recovery_protocol_path=fixture.protocol,
        screening_aggregate_path=reanalysis_screening,
        winner_arm_id="small_target_960",
        evaluation_report_paths=confirmation_reports,
        output=output,
        workspace_root=tmp_path,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert summary["run_count"] == 36
    assert payload["bindings"]["reanalysis_lineage"]["report_count"] == 42
    assert payload["gate"]["checks"]["immutable_reanalysis_evidence_lineage_verified"] is True


@pytest.mark.parametrize(
    ("drift", "error"),
    [
        ("report_hash", "binding or experiment differs"),
        ("report_path", "binding or experiment differs"),
        ("report_experiment", "binding or experiment differs"),
        ("source_protocol", "source protocol is invalid"),
        ("source_screening", "source screening aggregate is invalid"),
        ("failure_evidence", "failure evidence is invalid"),
    ],
)
def test_reanalysis_protocol_rejects_any_frozen_lineage_drift(
    tmp_path: Path, drift: str, error: str
) -> None:
    fixture, _source_screening, _confirmation_reports = _make_reanalysis_fixture(tmp_path)
    protocol = json.loads(fixture.protocol.read_text(encoding="utf-8"))
    lineage = protocol["reanalysis"]
    if drift == "report_hash":
        lineage["reports"][0]["sha256"] = "0" * 64
    elif drift == "report_path":
        lineage["reports"][0]["path"] = lineage["reports"][1]["path"]
    elif drift == "report_experiment":
        lineage["reports"][0]["experiment"]["seed"] = 43
    elif drift == "source_protocol":
        lineage["source_protocol"]["sha256"] = "0" * 64
    elif drift == "source_screening":
        lineage["source_screening_aggregate"]["sha256"] = "0" * 64
    elif drift == "failure_evidence":
        lineage["failure_evidence"]["sha256"] = "0" * 64
    else:
        raise AssertionError(f"unhandled drift case: {drift}")
    if drift.startswith("report_"):
        lineage["reports_manifest_sha256"] = _canonical_sha(lineage["reports"])
    _write_json(fixture.protocol, protocol)

    _assert_screening_rejected(fixture, match=error)


def test_standard_training_protocol_must_not_contain_reanalysis_lineage(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    protocol = json.loads(fixture.protocol.read_text(encoding="utf-8"))
    protocol["reanalysis"] = {}
    _write_json(fixture.protocol, protocol)

    _assert_screening_rejected(
        fixture,
        match="training protocol must not contain reanalysis lineage",
    )
