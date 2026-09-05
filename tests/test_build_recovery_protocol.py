from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from scripts.build_recovery_protocol import (
    AGGREGATE_SCHEMA,
    AGGREGATION_FAILURE_SCHEMA,
    CANONICAL_NAMES,
    CV_PLAN_SCHEMA,
    EVALUATION_SCHEMA,
    EXPERIMENT_ARMS,
    MANDATORY_IMPLEMENTATION_PATHS,
    PROTOCOL_SCHEMA,
    READINESS_GATES,
    REFERENCE_SCHEMA,
    SPLIT_SCHEMA,
    RecoveryProtocolError,
    build_recovery_protocol,
)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def file_binding(path: Path) -> dict[str, object]:
    encoded = path.read_bytes()
    return {
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "size_bytes": len(encoded),
    }


def semantic_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def create_fixture(root: Path, *, fold_count: int = 5) -> dict[str, Any]:
    reference_root = root / "data/ground-truth/training-v1"
    annotations = reference_root / "annotations.coco.json"
    write_json(annotations, {"images": [], "annotations": [], "categories": []})
    annotation_binding = file_binding(annotations)
    assets = [
        {
            "asset_id": asset_id,
            "leakage_group_id": f"sha256:{asset_id:064x}",
            "image_count": 1,
            "annotation_count": 8,
            "scene_ids": [f"scene-{asset_id}"],
        }
        for asset_id in range(1, fold_count + 1)
    ]
    reference = reference_root / "manifest.json"
    write_json(
        reference,
        {
            "schema": REFERENCE_SCHEMA,
            "counts": {
                "annotations_by_category": {name: fold_count for name in CANONICAL_NAMES},
            },
            "source_statistics": {"assets": assets},
            "files": [{"path": "annotations.coco.json", **annotation_binding}],
            "gate": {"passed": True, "checks": {"immutable_reference": True}},
        },
    )

    folds = []
    for index, asset_id in enumerate(range(1, fold_count + 1), start=1):
        split_path = root / f"docs/evidence/training-cv-plan-v1/folds/fold-{index:02d}.split.json"
        write_json(
            split_path,
            {
                "schema": SPLIT_SCHEMA,
                "train_asset_ids": [
                    value for value in range(1, fold_count + 1) if value != asset_id
                ],
                "val_asset_ids": [asset_id],
            },
        )
        split_binding = file_binding(split_path)
        folds.append(
            {
                "fold_id": f"fold-{index:02d}",
                "val_asset_id": asset_id,
                "train": {
                    "asset_ids": [value for value in range(1, fold_count + 1) if value != asset_id],
                    "source_count": fold_count - 1,
                },
                "val": {"asset_ids": [asset_id], "source_count": 1},
                "split_plan": {
                    "path": f"folds/fold-{index:02d}.split.json",
                    **split_binding,
                    "schema": SPLIT_SCHEMA,
                },
                "gate": {"passed": True, "checks": {"source_disjoint": True}},
            }
        )
    cv_manifest = root / "docs/evidence/training-cv-plan-v1/manifest.json"
    write_json(
        cv_manifest,
        {
            "schema": CV_PLAN_SCHEMA,
            "taxonomy": list(CANONICAL_NAMES),
            "inputs": {
                "reference_manifest": {
                    "path": "data/ground-truth/training-v1/manifest.json",
                    **file_binding(reference),
                    "schema": REFERENCE_SCHEMA,
                }
            },
            "plan_semantic_sha256": "a" * 64,
            "counts": {
                "source_assets": fold_count,
                "folds": fold_count,
                **({"source_groups": fold_count} if fold_count != 5 else {}),
            },
            "folds": folds,
            "holdout_firewall": {"final_holdout_input_read": False},
            "gate": {
                "passed": True,
                "checks": {"minimum_unique_source_group_count_met": True},
            },
        },
    )
    weights = root / "yolo11n.pt"
    weights.write_bytes(b"base-weight")
    implementation_files: list[Path] = []
    for relative in sorted(MANDATORY_IMPLEMENTATION_PATHS):
        implementation = root / relative
        implementation.parent.mkdir(parents=True, exist_ok=True)
        implementation.write_text(f"fixture for {relative}\n", encoding="utf-8")
        implementation_files.append(implementation)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Recovery Test"], cwd=root, check=True)
    subprocess.run(
        ["git", "config", "user.email", "recovery-test@example.invalid"],
        cwd=root,
        check=True,
    )
    subprocess.run(
        ["git", "add", "--", *sorted(MANDATORY_IMPLEMENTATION_PATHS)], cwd=root, check=True
    )
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=root, check=True)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "reference": reference,
        "cv_manifest": cv_manifest,
        "weights": weights,
        "implementation_files": implementation_files,
        "revision": revision,
    }


def build(root: Path, fixture: dict[str, Any], output: Path) -> dict[str, object]:
    return build_recovery_protocol(
        protocol_id="training-recovery-r1-v1",
        training_reference_manifest=fixture["reference"],
        cv_plan_manifest=fixture["cv_manifest"],
        base_weights=fixture["weights"],
        implementation_revision=str(fixture["revision"]),
        implementation_files=list(fixture["implementation_files"]),
        output=output,
        workspace=root,
    )


def create_reanalysis_fixture(root: Path, fixture: dict[str, Any]) -> dict[str, Any]:
    source_protocol = root / "docs/evidence/recovery-r1-v2.protocol.json"
    build_recovery_protocol(
        protocol_id="training-recovery-r1-v2",
        training_reference_manifest=fixture["reference"],
        cv_plan_manifest=fixture["cv_manifest"],
        base_weights=fixture["weights"],
        implementation_revision=str(fixture["revision"]),
        implementation_files=list(fixture["implementation_files"]),
        output=source_protocol,
        workspace=root,
    )
    source_payload = json.loads(source_protocol.read_text(encoding="utf-8"))
    source_binding = {
        "path": source_protocol.relative_to(root).as_posix(),
        **file_binding(source_protocol),
        "schema": PROTOCOL_SCHEMA,
        "protocol_id": source_payload["protocol_id"],
    }

    winner = "small_target_960"
    cv_payload = json.loads(fixture["cv_manifest"].read_text(encoding="utf-8"))
    fold_ids = [str(record["fold_id"]) for record in cv_payload["folds"]]
    screening_experiments = [
        (str(arm["arm_id"]), 42, fold_id) for arm in EXPERIMENT_ARMS for fold_id in fold_ids
    ]
    confirmation_experiments = [
        (arm_id, seed, fold_id)
        for arm_id in ("repaired_control", winner)
        for seed in (43, 44)
        for fold_id in fold_ids
    ]
    report_paths: list[Path] = []
    report_by_experiment: dict[tuple[str, int, str], Path] = {}
    for arm_id, seed, fold_id in [*screening_experiments, *confirmation_experiments]:
        report_path = (
            root
            / "data/model-candidates"
            / f"recovery-r1-v2-{arm_id}-seed-{seed}-{fold_id}.evaluation.json"
        )
        write_json(
            report_path,
            {
                "schema": EVALUATION_SCHEMA,
                "experiment": {"arm_id": arm_id, "seed": seed, "fold_id": fold_id},
                "gate": {"passed": True, "checks": {"evaluated": True}},
                "holdout_firewall": {"input_read": False},
            },
        )
        report_paths.append(report_path)
        report_by_experiment[(arm_id, seed, fold_id)] = report_path

    source_screening = root / "docs/evidence/recovery-r1-v2.screening.aggregate.json"
    screening_runs = []
    for arm_id, seed, fold_id in screening_experiments:
        report_path = report_by_experiment[(arm_id, seed, fold_id)]
        screening_runs.append(
            {
                "experiment": {"arm_id": arm_id, "seed": seed, "fold_id": fold_id},
                "report": {
                    "path": report_path.relative_to(root).as_posix(),
                    **file_binding(report_path),
                    "schema": EVALUATION_SCHEMA,
                },
            }
        )
    write_json(
        source_screening,
        {
            "schema": AGGREGATE_SCHEMA,
            "mode": "screening",
            "gate": {"passed": True, "checks": {"complete": True}},
            "bindings": {"recovery_protocol": source_binding},
            "contract": {"fold_ids": fold_ids},
            "selection": {"winner_arm_id": winner},
            "runs": screening_runs,
        },
    )

    failure_evidence = root / "docs/evidence/recovery-r1-v2.confirmation-failure.json"
    write_json(
        failure_evidence,
        {
            "schema": AGGREGATION_FAILURE_SCHEMA,
            "status": "failed",
            "stage": "confirmation",
            "protocol": source_binding,
            "aggregate_output": {"written": False},
        },
    )
    return {
        "source_protocol": source_protocol,
        "source_screening": source_screening,
        "failure_evidence": failure_evidence,
        "reports": report_paths,
        "experiments": set(screening_experiments) | set(confirmation_experiments),
        "winner": winner,
    }


def build_reanalysis(
    root: Path,
    fixture: dict[str, Any],
    lineage: dict[str, Any],
    output: Path,
    *,
    reports: list[Path] | None = None,
) -> dict[str, object]:
    return build_recovery_protocol(
        protocol_id="training-recovery-r1-v3",
        training_reference_manifest=fixture["reference"],
        cv_plan_manifest=fixture["cv_manifest"],
        base_weights=fixture["weights"],
        implementation_revision=str(fixture["revision"]),
        implementation_files=list(fixture["implementation_files"]),
        output=output,
        workspace=root,
        reanalysis_source_protocol=lineage["source_protocol"],
        reanalysis_source_screening_aggregate=lineage["source_screening"],
        reanalysis_failure_evidence=lineage["failure_evidence"],
        reanalysis_evaluation_reports=(lineage["reports"] if reports is None else reports),
    )


def test_builds_frozen_training_only_protocol_and_refuses_overwrite(tmp_path: Path) -> None:
    fixture = create_fixture(tmp_path)
    output = tmp_path / "docs/evidence/recovery-r1.json"

    summary = build(tmp_path, fixture, output)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == PROTOCOL_SCHEMA
    assert payload["scope"]["final_holdout_status"] == "sealed_and_consumed"
    assert payload["scope"]["new_final_holdout_allowed"] is False
    assert payload["experiments"]["arms"] == list(EXPERIMENT_ARMS)
    assert payload["readiness_gates"] == READINESS_GATES
    assert payload["implementation"]["revision"] == fixture["revision"]
    assert payload["implementation"]["mandatory_files_verified"] is True
    assert {record["path"] for record in payload["implementation"]["files"]} == set(
        MANDATORY_IMPLEMENTATION_PATHS
    )
    assert payload["refit_contract"]["separate_protocol_frozen_before_refit"] is True
    assert payload["refit_contract"]["epochs"] == {
        "source": "best_epoch.number_from_all_15_confirmed_winner_runs",
        "statistic": "integer_median_of_15_values",
        "minimum": 1,
        "maximum": 150,
    }
    assert payload["refit_contract"]["selected_checkpoint"] == ("last.pt_at_the_exact_refit_epoch")
    assert payload["reproducibility"]["bitwise_reproducibility_required"] is False
    assert "MPS" in payload["reproducibility"]["accelerator_limitation"]
    assert payload["experiments"]["confirmation"] == {
        "winner_and_paired_repaired_control": True,
        "control_may_self_compare_when_it_wins": True,
        "seeds": [42, 43, 44],
        "all_folds": True,
        "reuse_identical_seed_42_screening_runs": True,
        "frozen_screening_aggregate_binding_required": True,
    }
    assert payload["taxonomy"]["mapping"][5] == {
        "id": 5,
        "canonical": "pedestrian",
        "model": "person",
    }
    assert summary["holdout_input_read"] is False
    assert summary["screening_run_count"] == 15
    assert summary["confirmation_run_count_if_challenger_wins"] == 30
    assert summary["maximum_unique_run_count_with_screening_reuse"] == 35
    assert hashlib.sha256(output.read_bytes()).hexdigest() == summary["protocol_sha256"]

    with pytest.raises(FileExistsError, match="already exists"):
        build(tmp_path, fixture, output)


def test_builds_six_fold_protocol_with_dynamic_run_contract(tmp_path: Path) -> None:
    fixture = create_fixture(tmp_path, fold_count=6)
    output = tmp_path / "docs/evidence/recovery-r1-six-fold.json"

    summary = build(tmp_path, fixture, output)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["validation"]["fold_count"] == 6
    assert payload["inputs"]["loso_plan"]["fold_count"] == 6
    assert payload["refit_contract"]["epochs"] == {
        "source": "best_epoch.number_from_all_18_confirmed_winner_runs",
        "statistic": "integer_median_of_18_values",
        "minimum": 1,
        "maximum": 150,
    }
    assert summary["fold_count"] == 6
    assert summary["screening_run_count"] == 18
    assert summary["confirmation_run_count_if_control_wins"] == 18
    assert summary["confirmation_run_count_if_challenger_wins"] == 36
    assert summary["maximum_unique_run_count_with_screening_reuse"] == 42


def test_rejects_fewer_than_three_source_groups(tmp_path: Path) -> None:
    fixture = create_fixture(tmp_path, fold_count=2)

    with pytest.raises(RecoveryProtocolError, match="at least 3 unique source"):
        build(tmp_path, fixture, tmp_path / "docs/evidence/rejected-two-fold.json")


@pytest.mark.parametrize("extra_fold", [False, True])
def test_six_fold_protocol_rejects_missing_or_extra_plan_fold(
    tmp_path: Path, extra_fold: bool
) -> None:
    fixture = create_fixture(tmp_path, fold_count=6)
    payload = json.loads(fixture["cv_manifest"].read_text(encoding="utf-8"))
    if extra_fold:
        extra = dict(payload["folds"][-1])
        extra["fold_id"] = "fold-07"
        payload["folds"].append(extra)
    else:
        payload["folds"].pop()
    write_json(fixture["cv_manifest"], payload)

    with pytest.raises(RecoveryProtocolError, match="counts and fold list"):
        build(tmp_path, fixture, tmp_path / "docs/evidence/rejected-fold-set.json")


def test_same_inputs_produce_identical_bytes(tmp_path: Path) -> None:
    fixture = create_fixture(tmp_path)
    first = tmp_path / "docs/evidence/recovery-r1-a.json"
    second = tmp_path / "docs/evidence/recovery-r1-b.json"

    build(tmp_path, fixture, first)
    build(tmp_path, fixture, second)

    assert first.read_bytes() == second.read_bytes()


def test_builds_hash_bound_reanalysis_protocol_without_replacement(
    tmp_path: Path,
) -> None:
    fixture = create_fixture(tmp_path)
    lineage = create_reanalysis_fixture(tmp_path, fixture)
    output = tmp_path / "docs/evidence/recovery-r1-v3.protocol.json"

    summary = build_reanalysis(tmp_path, fixture, lineage, output)

    payload = json.loads(output.read_text(encoding="utf-8"))
    reanalysis = payload["reanalysis"]
    assert payload["status"] == "frozen_before_reanalysis_after_collection"
    assert reanalysis["mode"] == "immutable_evidence_reanalysis"
    assert reanalysis["collection_status"] == ("complete_before_reanalysis_protocol_freeze")
    assert reanalysis["new_replacement_or_supplemental_runs_allowed"] is False
    assert reanalysis["report_count"] == 35
    assert len(reanalysis["reports"]) == 35
    assert {
        (
            record["experiment"]["arm_id"],
            record["experiment"]["seed"],
            record["experiment"]["fold_id"],
        )
        for record in reanalysis["reports"]
    } == lineage["experiments"]
    assert reanalysis["reports_manifest_sha256"] == semantic_sha256(reanalysis["reports"])
    for record in reanalysis["reports"]:
        report_path = tmp_path / record["path"]
        assert {key: record[key] for key in ("sha256", "size_bytes")} == file_binding(report_path)
        assert record["schema"] == EVALUATION_SCHEMA
    assert reanalysis["source_protocol"] == {
        "path": lineage["source_protocol"].relative_to(tmp_path).as_posix(),
        **file_binding(lineage["source_protocol"]),
        "schema": PROTOCOL_SCHEMA,
        "protocol_id": "training-recovery-r1-v2",
    }
    assert reanalysis["source_screening_aggregate"] == {
        "path": lineage["source_screening"].relative_to(tmp_path).as_posix(),
        **file_binding(lineage["source_screening"]),
        "schema": AGGREGATE_SCHEMA,
        "winner_arm_id": lineage["winner"],
    }
    assert reanalysis["failure_evidence"] == {
        "path": lineage["failure_evidence"].relative_to(tmp_path).as_posix(),
        **file_binding(lineage["failure_evidence"]),
        "schema": AGGREGATION_FAILURE_SCHEMA,
    }
    assert summary["reanalysis"] is True
    assert summary["reanalysis_report_count"] == 35

    with pytest.raises(FileExistsError, match="already exists"):
        build_reanalysis(tmp_path, fixture, lineage, output)


def test_six_fold_reanalysis_lineage_covers_dynamic_matrix(tmp_path: Path) -> None:
    fixture = create_fixture(tmp_path, fold_count=6)
    lineage = create_reanalysis_fixture(tmp_path, fixture)
    output = tmp_path / "docs/evidence/recovery-r1-v3-six-fold.protocol.json"

    summary = build_reanalysis(tmp_path, fixture, lineage, output)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["validation"]["fold_count"] == 6
    assert payload["reanalysis"]["report_count"] == 42
    assert {record["experiment"]["fold_id"] for record in payload["reanalysis"]["reports"]} == {
        f"fold-{index:02d}" for index in range(1, 7)
    }
    assert summary["reanalysis_report_count"] == 42


def test_reanalysis_rejects_missing_report(tmp_path: Path) -> None:
    fixture = create_fixture(tmp_path)
    lineage = create_reanalysis_fixture(tmp_path, fixture)

    with pytest.raises(RecoveryProtocolError, match="exact completed"):
        build_reanalysis(
            tmp_path,
            fixture,
            lineage,
            tmp_path / "docs/evidence/recovery-r1-v3.protocol.json",
            reports=lineage["reports"][:-1],
        )


def test_reanalysis_rejects_tampered_source_screening_binding(tmp_path: Path) -> None:
    fixture = create_fixture(tmp_path)
    lineage = create_reanalysis_fixture(tmp_path, fixture)
    screening = json.loads(lineage["source_screening"].read_text(encoding="utf-8"))
    screening["bindings"]["recovery_protocol"]["sha256"] = "0" * 64
    write_json(lineage["source_screening"], screening)

    with pytest.raises(RecoveryProtocolError, match="screening aggregate is inconsistent"):
        build_reanalysis(
            tmp_path,
            fixture,
            lineage,
            tmp_path / "docs/evidence/recovery-r1-v3.protocol.json",
        )


@pytest.mark.parametrize(
    ("key", "relative"),
    [
        ("reference", "data/holdout/synthetic-reference/manifest.json"),
        ("cv_manifest", "docs/evidence/final-holdout-plan/manifest.json"),
        ("reference", "docs/evidence/task-91001/manifest.json"),
        ("cv_manifest", "docs/evidence/job-id-92001/manifest.json"),
    ],
)
def test_rejects_every_final_holdout_input_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    relative: str,
) -> None:
    monkeypatch.setenv("ROADLABELOPS_FINAL_HOLDOUT_TASK_IDS", "91001")
    monkeypatch.setenv("ROADLABELOPS_FINAL_HOLDOUT_JOB_IDS", "92001")
    fixture = create_fixture(tmp_path)
    prohibited = tmp_path / relative
    prohibited.parent.mkdir(parents=True, exist_ok=True)
    prohibited.write_bytes(fixture[key].read_bytes())
    fixture[key] = prohibited

    with pytest.raises(RecoveryProtocolError, match="firewall"):
        build(tmp_path, fixture, tmp_path / "docs/evidence/recovery-r1.json")


def test_rejects_cv_plan_bound_to_another_reference(tmp_path: Path) -> None:
    fixture = create_fixture(tmp_path)
    payload = json.loads(fixture["cv_manifest"].read_text(encoding="utf-8"))
    payload["inputs"]["reference_manifest"]["sha256"] = "f" * 64
    write_json(fixture["cv_manifest"], payload)

    with pytest.raises(RecoveryProtocolError, match="not bound"):
        build(tmp_path, fixture, tmp_path / "docs/evidence/recovery-r1.json")


def test_rejects_symlinked_implementation_file(tmp_path: Path) -> None:
    fixture = create_fixture(tmp_path)
    link = tmp_path / "scripts/trainer-link.py"
    implementation_files = list(fixture["implementation_files"])
    link.symlink_to(implementation_files[0])
    implementation_files[0] = link
    fixture["implementation_files"] = implementation_files

    with pytest.raises(RecoveryProtocolError, match="symlink"):
        build(tmp_path, fixture, tmp_path / "docs/evidence/recovery-r1.json")


def test_rejects_reference_annotation_hash_drift(tmp_path: Path) -> None:
    fixture = create_fixture(tmp_path)
    annotations = fixture["reference"].parent / "annotations.coco.json"
    annotations.write_text("{}\n", encoding="utf-8")

    with pytest.raises(RecoveryProtocolError, match="do not match"):
        build(tmp_path, fixture, tmp_path / "docs/evidence/recovery-r1.json")


def test_rejects_dirty_or_missing_mandatory_implementation_file(tmp_path: Path) -> None:
    fixture = create_fixture(tmp_path)
    dirty = tmp_path / "scripts/train_yolo_candidate.py"
    dirty.write_text("dirty\n", encoding="utf-8")
    with pytest.raises(RecoveryProtocolError, match="must be clean"):
        build(tmp_path, fixture, tmp_path / "docs/evidence/recovery-r1.json")

    fixture = create_fixture(tmp_path / "missing")
    fixture["implementation_files"] = [
        path
        for path in fixture["implementation_files"]
        if path.relative_to(tmp_path / "missing").as_posix() != "uv.lock"
    ]
    with pytest.raises(RecoveryProtocolError, match="omit mandatory"):
        build(
            tmp_path / "missing",
            fixture,
            tmp_path / "missing/docs/evidence/recovery-r1.json",
        )


def test_rejects_revision_device_output_and_split_contract_drift(tmp_path: Path) -> None:
    fixture = create_fixture(tmp_path)
    common = {
        "protocol_id": "training-recovery-r1-v1",
        "training_reference_manifest": fixture["reference"],
        "cv_plan_manifest": fixture["cv_manifest"],
        "base_weights": fixture["weights"],
        "implementation_files": fixture["implementation_files"],
        "workspace": tmp_path,
    }
    with pytest.raises(RecoveryProtocolError, match="full lowercase Git SHA"):
        build_recovery_protocol(
            **common,
            implementation_revision="abc123",
            output=tmp_path / "docs/evidence/bad-revision.json",
        )
    with pytest.raises(RecoveryProtocolError, match="device must be 'mps'"):
        build_recovery_protocol(
            **common,
            implementation_revision=fixture["revision"],
            device="cpu",
            output=tmp_path / "docs/evidence/bad-device.json",
        )
    with pytest.raises(RecoveryProtocolError, match="below docs/evidence"):
        build_recovery_protocol(
            **common,
            implementation_revision=fixture["revision"],
            output=tmp_path / "recovery-r1.json",
        )

    cv_payload = json.loads(fixture["cv_manifest"].read_text(encoding="utf-8"))
    fold = cv_payload["folds"][0]
    split_path = fixture["cv_manifest"].parent / fold["split_plan"]["path"]
    split = json.loads(split_path.read_text(encoding="utf-8"))
    split["train_asset_ids"].append(fold["val_asset_id"])
    write_json(split_path, split)
    fold["split_plan"].update(file_binding(split_path))
    write_json(fixture["cv_manifest"], cv_payload)
    with pytest.raises(RecoveryProtocolError, match="split file differs|exact complement"):
        build(tmp_path, fixture, tmp_path / "docs/evidence/bad-split.json")
