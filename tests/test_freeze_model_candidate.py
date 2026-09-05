from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import scripts.freeze_model_candidate as freeze
from scripts.freeze_model_candidate import CandidateFreezeError, freeze_model_candidate


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def artifact(path: Path, relative_path: str) -> dict[str, Any]:
    return {
        "path": relative_path,
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def aggregate(
    *, precision: float, recall: float, map50: float, map50_95: float
) -> dict[str, float]:
    return {
        "precision": precision,
        "recall": recall,
        "map50": map50,
        "map50_95": map50_95,
        "fitness": map50_95,
    }


def protocol(*, legacy: bool = False) -> dict[str, Any]:
    training = {
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
            lr0=0.001,
            lrf=0.01,
            momentum=0.9,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            warmup_bias_lr=0.0,
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


def write_candidate(
    root: Path,
    *,
    seed: int,
    metrics: dict[str, float],
    legacy: bool = False,
) -> dict[str, Any]:
    (root / "artifacts").mkdir(parents=True)
    (root / "weights").mkdir()
    (root / "artifacts" / "args.yaml").write_text(f"seed: {seed}\n", encoding="utf-8")
    (root / "artifacts" / "results.csv").write_text(
        "epoch,metrics/mAP50(B),metrics/mAP50-95(B)\n0,0.7,0.6\n",
        encoding="utf-8",
    )
    (root / "weights" / "best.pt").write_bytes(f"best-weights-seed-{seed}".encode())
    (root / "weights" / "last.pt").write_bytes(f"last-weights-seed-{seed}".encode())
    frozen_protocol = protocol(legacy=legacy)
    per_class = {
        label: {
            "class_id": class_id,
            **({} if legacy else {"status": "evaluable", "support_count": 1}),
            "precision": 0.60 + class_id / 100,
            "recall": 0.50 + class_id / 100,
            "map50": 0.40 + class_id / 100,
            "map50_95": 0.30 + class_id / 100,
        }
        for class_id, label in enumerate(freeze.REQUIRED_LABELS)
    }
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
        "protocol": frozen_protocol,
        "protocol_sha256": canonical_sha256(frozen_protocol),
        "resolved_args": {
            "model_family": "YOLO11n",
            "seed": seed,
            **copy.deepcopy(frozen_protocol["training"]),
        },
        "inputs": {
            "dataset": {
                "manifest": {
                    "schema": (freeze.LEGACY_DATASET_SCHEMA if legacy else freeze.DATASET_SCHEMA),
                    "sha256": "a" * 64,
                    "size_bytes": 1234,
                },
                "dataset_yaml": {"sha256": "b" * 64, "size_bytes": 456},
                "managed_files_sha256": "c" * 64,
                "managed_file_count": 11,
                "counts": {
                    "images": {"total": 5, "train": 3, "val": 2},
                    "annotations": {"total": 40},
                },
                **(
                    {}
                    if legacy
                    else {
                        "taxonomy": {
                            "canonical_names": list(freeze.REQUIRED_LABELS),
                            "model_names": list(freeze.MODEL_LABELS),
                            "model_to_canonical": freeze.MODEL_TO_CANONICAL,
                        }
                    }
                ),
            },
            "base_weights": {
                "file_name": "yolo11n.pt",
                "model_family": "YOLO11n",
                "sha256": "d" * 64,
                "size_bytes": 654321,
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
        "metrics": {"aggregate": metrics, "per_class": per_class},
        "best_epoch": {
            "index": 1,
            "number": 2,
            "selection_fitness": 0.61,
            "epochs_recorded": 2,
        },
        "artifacts": {
            "args": artifact(root / "artifacts" / "args.yaml", "artifacts/args.yaml"),
            "results": artifact(root / "artifacts" / "results.csv", "artifacts/results.csv"),
            "best_weights": artifact(root / "weights" / "best.pt", "weights/best.pt"),
            "last_weights": artifact(root / "weights" / "last.pt", "weights/last.pt"),
        },
        "holdout": {
            "input_read": False,
            "statement": freeze.NO_HOLDOUT_STATEMENT,
        },
    }
    write_receipt(root, receipt)
    return receipt


def write_receipt(root: Path, payload: dict[str, Any], *, allow_nan: bool = False) -> None:
    (root / "receipt.json").write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=allow_nan, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def rewrite_receipt(root: Path, mutation: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    payload = json.loads((root / "receipt.json").read_text(encoding="utf-8"))
    mutation(payload)
    write_receipt(root, payload)
    return payload


@dataclass(frozen=True)
class Candidates:
    roots: tuple[Path, Path, Path]


@pytest.fixture
def candidates(tmp_path: Path) -> Candidates:
    roots = tuple(tmp_path / f"candidate-{index}" for index in range(3))
    write_candidate(
        roots[0],
        seed=44,
        metrics=aggregate(precision=0.90, recall=0.90, map50=0.80, map50_95=0.70),
    )
    write_candidate(
        roots[1],
        seed=42,
        metrics=aggregate(precision=0.85, recall=0.90, map50=0.80, map50_95=0.70),
    )
    write_candidate(
        roots[2],
        seed=43,
        metrics=aggregate(precision=0.85, recall=0.90, map50=0.80, map50_95=0.70),
    )
    return Candidates(roots)


def test_freezes_ranked_candidate_and_complete_audit_receipt(
    tmp_path: Path, candidates: Candidates
) -> None:
    output = tmp_path / "frozen"

    result = freeze_model_candidate(candidates.roots, output)

    assert result["selected_seed"] == 44
    assert (output / "best.pt").read_bytes() == (
        candidates.roots[0] / "weights/best.pt"
    ).read_bytes()
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["schema"] == freeze.FROZEN_SCHEMA
    assert receipt["selection_order"] == list(freeze.SELECTION_ORDER)
    assert [item["seed"] for item in receipt["candidate_rankings"]] == [44, 42, 43]
    assert [item["rank"] for item in receipt["candidate_rankings"]] == [1, 2, 3]
    assert receipt["contracts"]["candidate_count"] == 3
    assert receipt["contracts"]["taxonomy"] == list(freeze.REQUIRED_LABELS)
    assert len(receipt["candidate_rankings"]) == 3
    assert all("contract" in item for item in receipt["candidate_rankings"])
    assert receipt["selected_source"]["receipt"]["sha256"] == sha256(
        candidates.roots[0] / "receipt.json"
    )
    assert receipt["selected_weight"] == {
        "path": "best.pt",
        "sha256": sha256(output / "best.pt"),
        "size_bytes": (output / "best.pt").stat().st_size,
    }
    assert receipt["holdout_input_read"] is False
    assert receipt["holdout_statement"] == freeze.NO_HOLDOUT_STATEMENT


def test_freezes_three_legacy_v1_candidates_without_rewriting_them(tmp_path: Path) -> None:
    roots = tuple(tmp_path / f"legacy-candidate-{index}" for index in range(3))
    for root, seed, score in zip(roots, (42, 43, 44), (0.50, 0.55, 0.53), strict=True):
        write_candidate(
            root,
            seed=seed,
            metrics=aggregate(
                precision=0.80,
                recall=0.70,
                map50=score + 0.10,
                map50_95=score,
            ),
            legacy=True,
        )

    result = freeze_model_candidate(roots, tmp_path / "legacy-frozen")

    assert result["selected_seed"] == 43
    receipt = json.loads((tmp_path / "legacy-frozen" / "receipt.json").read_text())
    assert receipt["contracts"]["protocol"]["schema"] == freeze.LEGACY_PROTOCOL_SCHEMA
    assert receipt["contracts"]["dataset"]["manifest"]["schema"] == (freeze.LEGACY_DATASET_SCHEMA)


def test_ranking_prioritizes_map_then_recall_before_other_metrics(
    tmp_path: Path, candidates: Candidates
) -> None:
    def primary_winner(payload: dict[str, Any]) -> None:
        payload["metrics"]["aggregate"] = aggregate(
            precision=0.01, recall=0.01, map50=0.01, map50_95=0.71
        )

    def map_and_recall_winner(payload: dict[str, Any]) -> None:
        payload["metrics"]["aggregate"] = aggregate(
            precision=0.01, recall=0.20, map50=0.90, map50_95=0.70
        )

    rewrite_receipt(candidates.roots[0], primary_winner)
    rewrite_receipt(candidates.roots[1], map_and_recall_winner)
    rewrite_receipt(
        candidates.roots[2],
        lambda payload: payload["metrics"].update(
            aggregate=aggregate(precision=0.99, recall=0.10, map50=0.90, map50_95=0.70)
        ),
    )

    freeze_model_candidate(candidates.roots, tmp_path / "frozen-priority")

    receipt = json.loads((tmp_path / "frozen-priority" / "receipt.json").read_text())
    assert [item["seed"] for item in receipt["candidate_rankings"]] == [44, 42, 43]


def test_requires_exactly_three_distinct_candidates(tmp_path: Path, candidates: Candidates) -> None:
    with pytest.raises(CandidateFreezeError, match="exactly three"):
        freeze_model_candidate(candidates.roots[:2], tmp_path / "too-few")
    with pytest.raises(CandidateFreezeError, match="distinct"):
        freeze_model_candidate(
            [candidates.roots[0], candidates.roots[0], candidates.roots[1]],
            tmp_path / "duplicate",
        )


def test_requires_exact_frozen_seed_set(tmp_path: Path, candidates: Candidates) -> None:
    rewrite_receipt(
        candidates.roots[2],
        lambda payload: payload["resolved_args"].update(seed=45),
    )

    with pytest.raises(CandidateFreezeError, match=r"exactly \[42, 43, 44\]"):
        freeze_model_candidate(candidates.roots, tmp_path / "wrong-seeds")


@pytest.mark.parametrize("contract", ["protocol", "dataset", "base_weights"])
def test_rejects_cross_candidate_contract_drift(
    tmp_path: Path, candidates: Candidates, contract: str
) -> None:
    def mutate(payload: dict[str, Any]) -> None:
        if contract == "protocol":
            payload["protocol"]["training"]["epochs"] = 3
            payload["resolved_args"]["epochs"] = 3
            payload["protocol_sha256"] = canonical_sha256(payload["protocol"])
        elif contract == "dataset":
            payload["inputs"]["dataset"]["managed_files_sha256"] = "e" * 64
        else:
            payload["inputs"]["base_weights"]["sha256"] = "e" * 64

    rewrite_receipt(candidates.roots[2], mutate)

    with pytest.raises(CandidateFreezeError, match="different"):
        freeze_model_candidate(candidates.roots, tmp_path / f"drift-{contract}")


def test_seed_is_the_only_allowed_resolved_argument_difference(
    tmp_path: Path, candidates: Candidates
) -> None:
    rewrite_receipt(
        candidates.roots[1],
        lambda payload: payload["resolved_args"].update(batch=3),
    )

    with pytest.raises(CandidateFreezeError, match="resolved_args.batch differs"):
        freeze_model_candidate(candidates.roots, tmp_path / "config-drift")


def test_rejects_invalid_or_drifted_explicit_optimizer_hyperparameters(
    tmp_path: Path, candidates: Candidates
) -> None:
    def invalid_class_weight(payload: dict[str, Any]) -> None:
        payload["protocol"]["training"]["cls_pw"] = 1.1
        payload["resolved_args"]["cls_pw"] = 1.1
        payload["protocol_sha256"] = canonical_sha256(payload["protocol"])

    rewrite_receipt(candidates.roots[0], invalid_class_weight)

    with pytest.raises(CandidateFreezeError, match="cls_pw must be within"):
        freeze_model_candidate(candidates.roots, tmp_path / "invalid-cls-pw")


def test_direct_freeze_rejects_v2_candidate_with_not_evaluable_class(
    tmp_path: Path, candidates: Candidates
) -> None:
    def make_not_evaluable(payload: dict[str, Any]) -> None:
        record = payload["metrics"]["per_class"]["bicycle"]
        record.update(
            status="not_evaluable",
            support_count=0,
            precision=None,
            recall=None,
            map50=None,
            map50_95=None,
        )

    rewrite_receipt(candidates.roots[0], make_not_evaluable)

    with pytest.raises(CandidateFreezeError, match="not evaluable.*cannot enter direct"):
        freeze_model_candidate(candidates.roots, tmp_path / "not-evaluable")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload["gate"].update(passed=False), "gate.passed"),
        (
            lambda payload: payload["gate"]["checks"].update(source_inputs_unchanged=False),
            "source_inputs_unchanged",
        ),
        (lambda payload: payload["holdout"].update(input_read=True), "input_read"),
        (
            lambda payload: payload["protocol"].update(holdout_access="allowed"),
            "protocol_sha256|holdout_access",
        ),
    ],
)
def test_rejects_failed_gates_or_holdout_access(
    tmp_path: Path,
    candidates: Candidates,
    mutation: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    rewrite_receipt(candidates.roots[0], mutation)

    with pytest.raises(CandidateFreezeError, match=message):
        freeze_model_candidate(candidates.roots, tmp_path / "failed-gate")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1, True])
def test_rejects_nonfinite_or_wrong_metrics(
    tmp_path: Path, candidates: Candidates, value: Any
) -> None:
    payload = json.loads((candidates.roots[0] / "receipt.json").read_text())
    payload["metrics"]["aggregate"]["map50_95"] = value
    write_receipt(candidates.roots[0], payload, allow_nan=True)

    with pytest.raises(CandidateFreezeError, match="finite|number|within"):
        freeze_model_candidate(candidates.roots, tmp_path / "invalid-metric")


def test_rejects_inconsistent_fitness(tmp_path: Path, candidates: Candidates) -> None:
    rewrite_receipt(
        candidates.roots[0],
        lambda payload: payload["metrics"]["aggregate"].update(fitness=0.0),
    )
    with pytest.raises(CandidateFreezeError, match="fitness is inconsistent"):
        freeze_model_candidate(candidates.roots, tmp_path / "bad-fitness")


def test_rejects_legacy_weighted_fitness(tmp_path: Path, candidates: Candidates) -> None:
    def use_legacy_fitness(payload: dict[str, Any]) -> None:
        aggregate_metrics = payload["metrics"]["aggregate"]
        aggregate_metrics["fitness"] = (
            0.1 * aggregate_metrics["map50"] + 0.9 * aggregate_metrics["map50_95"]
        )

    rewrite_receipt(candidates.roots[0], use_legacy_fitness)
    with pytest.raises(CandidateFreezeError, match="fitness is inconsistent"):
        freeze_model_candidate(candidates.roots, tmp_path / "legacy-fitness")


def test_accepts_v1_legacy_best_epoch_fitness_when_rank_metrics_are_current(
    tmp_path: Path, candidates: Candidates
) -> None:
    payload = json.loads((candidates.roots[0] / "receipt.json").read_text())
    assert payload["best_epoch"]["selection_fitness"] != payload["metrics"]["aggregate"]["map50_95"]

    result = freeze_model_candidate(candidates.roots, tmp_path / "legacy-best-epoch")

    assert result["selected_seed"] == 44


def test_rejects_unfrozen_ultralytics_version(tmp_path: Path, candidates: Candidates) -> None:
    rewrite_receipt(
        candidates.roots[0],
        lambda payload: payload["environment"]["packages"].update(ultralytics="8.4.134"),
    )
    with pytest.raises(CandidateFreezeError, match="packages.ultralytics"):
        freeze_model_candidate(candidates.roots, tmp_path / "wrong-ultralytics")


def test_rejects_wrong_taxonomy_even_with_recomputed_protocol_hash(
    tmp_path: Path, candidates: Candidates
) -> None:
    def mutate(payload: dict[str, Any]) -> None:
        payload["protocol"]["taxonomy"]["canonical_names"][-1] = "not_traffic_sign"
        payload["protocol_sha256"] = canonical_sha256(payload["protocol"])

    rewrite_receipt(candidates.roots[0], mutate)

    with pytest.raises(CandidateFreezeError, match="eight-class"):
        freeze_model_candidate(candidates.roots, tmp_path / "bad-taxonomy")


@pytest.mark.parametrize("drift", ["hash", "size"])
def test_rejects_best_weight_hash_or_size_drift(
    tmp_path: Path, candidates: Candidates, drift: str
) -> None:
    weight = candidates.roots[0] / "weights/best.pt"
    if drift == "hash":
        weight.write_bytes(b"x" * weight.stat().st_size)
    else:
        weight.write_bytes(b"x")
        rewrite_receipt(
            candidates.roots[0],
            lambda payload: payload["artifacts"]["best_weights"].update(sha256=sha256(weight)),
        )

    with pytest.raises(CandidateFreezeError, match="best.pt (SHA-256|size) differs"):
        freeze_model_candidate(candidates.roots, tmp_path / f"weight-{drift}")


def test_rejects_artifact_path_traversal(tmp_path: Path, candidates: Candidates) -> None:
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"outside")

    def mutate(payload: dict[str, Any]) -> None:
        payload["artifacts"]["best_weights"] = artifact(outside, "../outside.pt")

    rewrite_receipt(candidates.roots[0], mutate)

    with pytest.raises(CandidateFreezeError, match="unsafe relative artifact path"):
        freeze_model_candidate(candidates.roots, tmp_path / "traversal")


@pytest.mark.parametrize("target", ["candidate", "receipt", "weight"])
def test_rejects_symlinked_candidate_inputs(
    tmp_path: Path, candidates: Candidates, target: str
) -> None:
    roots = list(candidates.roots)
    if target == "candidate":
        link = tmp_path / "candidate-link"
        link.symlink_to(roots[0], target_is_directory=True)
        roots[0] = link
    elif target == "receipt":
        receipt = roots[0] / "receipt.json"
        real_receipt = tmp_path / "external-receipt.json"
        receipt.replace(real_receipt)
        receipt.symlink_to(real_receipt)
    else:
        weight = roots[0] / "weights/best.pt"
        external = tmp_path / "external-best.pt"
        weight.replace(external)
        weight.symlink_to(external)

    with pytest.raises(CandidateFreezeError, match="symlink"):
        freeze_model_candidate(roots, tmp_path / f"symlink-{target}")


def test_existing_output_is_never_replaced(tmp_path: Path, candidates: Candidates) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(CandidateFreezeError, match="already exists"):
        freeze_model_candidate(candidates.roots, output)

    assert marker.read_text(encoding="utf-8") == "keep"


def test_atomic_publish_refuses_racing_output_and_cleans_staging(
    tmp_path: Path, candidates: Candidates, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "raced"
    real_publish = freeze._atomic_publish_directory_no_replace

    def race(staging: Path, target: Path) -> None:
        target.mkdir()
        (target / "winner.txt").write_text("keep", encoding="utf-8")
        real_publish(staging, target)

    monkeypatch.setattr(freeze, "_atomic_publish_directory_no_replace", race)

    with pytest.raises(CandidateFreezeError, match="already exists"):
        freeze_model_candidate(candidates.roots, output)

    assert (output / "winner.txt").read_text(encoding="utf-8") == "keep"
    assert not list(tmp_path.glob(".raced.staging-*"))


def test_cli_accepts_candidate_dir_alias(tmp_path: Path, candidates: Candidates) -> None:
    output = tmp_path / "cli-frozen"
    arguments: list[str] = []
    for root in candidates.roots:
        arguments.extend(["--candidate-dir", str(root)])
    arguments.extend(["--output", str(output)])

    assert freeze.main(arguments) == 0
    assert (output / "best.pt").is_file()
