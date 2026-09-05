#!/usr/bin/env python3
"""Freeze the training-only Recovery R1 experiment contract.

This builder has an intentionally narrow input surface.  It accepts the
immutable training reference, a source-level LOSO plan, the base weight and
the implementation files that will execute the experiment.  The configured
final holdout and every path below ``data/holdout`` are outside its admissible
input domain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from roadlabelops.holdout_policy import (
    NO_FINAL_HOLDOUT_STATEMENT,
    final_holdout_scope_reason,
)

PROTOCOL_SCHEMA = {"name": "roadlabelops.training-recovery-protocol", "version": 1}
AGGREGATE_SCHEMA = {"name": "roadlabelops.training-cv-aggregate", "version": 1}
EVALUATION_SCHEMA = {
    "name": "roadlabelops.training-validation-evaluation",
    "version": 1,
}
AGGREGATION_FAILURE_SCHEMA = {
    "name": "roadlabelops.training-recovery-r1-aggregation-failure",
    "version": 1,
}
REFERENCE_SCHEMA = {"name": "roadlabelops.training-coco-reference", "version": 2}
CV_PLAN_SCHEMA = {"name": "roadlabelops.training-loso-cv-plan", "version": 1}
SPLIT_SCHEMA = {"name": "roadlabelops.training-asset-split", "version": 1}

CANONICAL_NAMES = (
    "car",
    "bus",
    "truck",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_light",
    "traffic_sign",
)
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
EXPECTED_SEEDS = (42, 43, 44)
MINIMUM_SOURCE_GROUP_COUNT = 3
NO_HOLDOUT_STATEMENT = NO_FINAL_HOLDOUT_STATEMENT

COMMON_TRAINING = {
    "amp": True,
    "batch": 8,
    "cache": "none",
    "close_mosaic": 10,
    "deterministic": True,
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
EXPERIMENT_ARMS = (
    {"arm_id": "repaired_control", "imgsz": 640, "lr0": 0.001, "cls_pw": 0.0},
    {"arm_id": "small_target_960", "imgsz": 960, "lr0": 0.001, "cls_pw": 0.0},
    {"arm_id": "class_balance_025", "imgsz": 640, "lr0": 0.001, "cls_pw": 0.25},
)
EVALUATION_SETTINGS = {
    "confidence": 0.4,
    "match_iou": 0.5,
    "nms_iou": 0.75,
    "rider_overlap": 0.25,
    "class_threshold_overrides": {},
}
READINESS_GATES = {
    "oof_precision_min": 0.9,
    "oof_recall_min": 0.85,
    "clean_frame_rate_min": 0.8,
    "every_seed_and_source_f1_noninferior_to_repaired_control": True,
    "supported_class_recall_drop_max": 0.05,
    "seed_oof_f1_sample_standard_deviation_max": 0.03,
    "all_planned_runs_required": True,
    "exact_contract_match_required": True,
}
MANDATORY_IMPLEMENTATION_PATHS = frozenset(
    {
        "pyproject.toml",
        "roadlabelops/holdout_policy.py",
        "roadlabelops/models.py",
        "roadlabelops/tools/detection.py",
        "roadlabelops/tools/quality.py",
        "scripts/aggregate_training_cv.py",
        "scripts/analyze_training_recovery.py",
        "scripts/build_recovery_protocol.py",
        "scripts/build_training_cv_plan.py",
        "scripts/evaluate_training_validation.py",
        "scripts/freeze_model_candidate.py",
        "scripts/prepare_yolo_dataset.py",
        "scripts/train_yolo_candidate.py",
        "uv.lock",
    }
)
REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")


class RecoveryProtocolError(ValueError):
    """Raised when a Recovery R1 contract cannot be frozen safely."""


def _object(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RecoveryProtocolError(f"{location} must be an object")
    return value


def _list(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise RecoveryProtocolError(f"{location} must be a list")
    return value


def _text(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RecoveryProtocolError(f"{location} must be a non-empty string")
    if any(ord(character) < 32 for character in value):
        raise RecoveryProtocolError(f"{location} contains a control character")
    return value.strip()


def _sha256(value: Any, location: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RecoveryProtocolError(f"{location} must be a lowercase SHA-256 digest")
    return value


def _integer(value: Any, location: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RecoveryProtocolError(f"{location} must be an integer >= {minimum}")
    return value


def _asset_id(value: Any, location: str) -> int | str:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise RecoveryProtocolError(f"{location} must be an integer or non-empty string")
    if isinstance(value, str) and not value.strip():
        raise RecoveryProtocolError(f"{location} must be an integer or non-empty string")
    return value


def _asset_identity(value: int | str) -> tuple[str, str]:
    return type(value).__name__, str(value)


def _safe_relative(value: Any, location: str) -> PurePosixPath:
    raw = _text(value, location)
    posix = PurePosixPath(raw)
    windows = PureWindowsPath(raw)
    if (
        "\\" in raw
        or posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or posix.as_posix() != raw
        or not posix.name
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        raise RecoveryProtocolError(f"{location} must be a safe workspace-relative POSIX path")
    return posix


def _contains_prohibited_input(relative: PurePosixPath) -> bool:
    return final_holdout_scope_reason(relative) is not None


def _workspace_file(
    path: Path,
    *,
    workspace: Path,
    location: str,
    allowed_prefix: tuple[str, ...] | None = None,
) -> tuple[Path, PurePosixPath]:
    raw = os.fspath(path)
    if not raw or "\x00" in raw or any(part == ".." for part in Path(raw).parts):
        raise RecoveryProtocolError(f"{location} is unsafe")
    absolute = path if path.is_absolute() else workspace / path
    try:
        relative_path = absolute.relative_to(workspace)
    except ValueError as error:
        raise RecoveryProtocolError(f"{location} must be inside the workspace") from error
    relative = PurePosixPath(relative_path.as_posix())
    if _contains_prohibited_input(relative):
        raise RecoveryProtocolError(f"{location} crosses the final-holdout firewall")
    if allowed_prefix is not None and relative.parts[: len(allowed_prefix)] != allowed_prefix:
        prefix = "/".join(allowed_prefix)
        raise RecoveryProtocolError(f"{location} must be below {prefix}")

    current = workspace
    for part in relative.parts:
        current /= part
        try:
            details = os.lstat(current)
        except OSError as error:
            raise RecoveryProtocolError(f"{location} is unavailable: {error}") from error
        if stat.S_ISLNK(details.st_mode):
            raise RecoveryProtocolError(f"{location} must not traverse a symlink")
    if not stat.S_ISREG(details.st_mode):
        raise RecoveryProtocolError(f"{location} must be a regular file")
    return current, relative


def _stable_bytes(path: Path, location: str) -> tuple[bytes, os.stat_result]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RecoveryProtocolError(f"could not open {location}: {error}") from error
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    encoded = b"".join(chunks)
    if identity_before != identity_after or len(encoded) != before.st_size:
        raise RecoveryProtocolError(f"{location} changed while it was read")
    return encoded, after


def _read_json(path: Path, location: str) -> tuple[dict[str, Any], str, int]:
    encoded, details = _stable_bytes(path, location)
    try:
        payload = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RecoveryProtocolError(f"could not decode {location}: {error}") from error
    return _object(payload, location), hashlib.sha256(encoded).hexdigest(), details.st_size


def _binding(path: Path, relative: PurePosixPath, location: str) -> dict[str, Any]:
    encoded, details = _stable_bytes(path, location)
    return {
        "path": relative.as_posix(),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "size_bytes": details.st_size,
    }


def _git(
    workspace: Path,
    arguments: Sequence[str],
    *,
    text: bool = True,
) -> str | bytes:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=workspace,
            check=True,
            capture_output=True,
            text=text,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RecoveryProtocolError(
            f"could not verify the implementation Git revision: {error}"
        ) from error
    return completed.stdout


def _validate_implementation_revision(
    workspace: Path,
    revision: str,
    bindings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if REVISION_PATTERN.fullmatch(revision) is None:
        raise RecoveryProtocolError("implementation_revision must be a full lowercase Git SHA")
    repository_root = Path(str(_git(workspace, ["rev-parse", "--show-toplevel"])).strip()).resolve()
    if repository_root != workspace:
        raise RecoveryProtocolError(
            "workspace must be the root of the implementation Git repository"
        )
    head = str(_git(workspace, ["rev-parse", "HEAD"])).strip()
    if head != revision:
        raise RecoveryProtocolError("implementation_revision must equal the current Git HEAD")

    paths = [str(binding["path"]) for binding in bindings]
    missing = sorted(MANDATORY_IMPLEMENTATION_PATHS - set(paths))
    if missing:
        raise RecoveryProtocolError(
            "implementation_files omit mandatory Recovery files: " + ", ".join(missing)
        )
    status = str(
        _git(
            workspace,
            ["status", "--porcelain=v1", "--untracked-files=all", "--", *paths],
        )
    )
    if status.strip():
        raise RecoveryProtocolError("implementation files must be clean at implementation_revision")

    for binding in bindings:
        relative = str(binding["path"])
        try:
            committed = bytes(
                _git(
                    workspace,
                    ["show", "--no-ext-diff", "--no-textconv", f"{revision}:{relative}"],
                    text=False,
                )
            )
        except RecoveryProtocolError as error:
            raise RecoveryProtocolError(
                f"implementation file is not tracked at revision: {relative}"
            ) from error
        if (
            hashlib.sha256(committed).hexdigest() != binding["sha256"]
            or len(committed) != binding["size_bytes"]
        ):
            raise RecoveryProtocolError(
                f"implementation file differs from revision {revision}: {relative}"
            )
    return {
        "revision": revision,
        "repository_root": ".",
        "mandatory_files_verified": True,
        "selected_files_clean": True,
        "files": [dict(binding) for binding in bindings],
    }


def _validate_reference(
    path: Path, relative: PurePosixPath, workspace: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, digest, size_bytes = _read_json(path, "training reference manifest")
    if payload.get("schema") != REFERENCE_SCHEMA:
        raise RecoveryProtocolError("training reference manifest schema is unsupported")
    gate = _object(payload.get("gate"), "training reference manifest.gate")
    checks = _object(gate.get("checks"), "training reference manifest.gate.checks")
    if (
        gate.get("passed") is not True
        or not checks
        or not all(value is True for value in checks.values())
    ):
        raise RecoveryProtocolError("training reference gate must pass every declared check")
    categories = _object(payload.get("counts"), "training reference manifest.counts").get(
        "annotations_by_category"
    )
    if not isinstance(categories, dict) or set(categories) != set(CANONICAL_NAMES):
        raise RecoveryProtocolError("training reference must contain the canonical taxonomy")
    assets = _list(
        _object(payload.get("source_statistics"), "training reference source_statistics").get(
            "assets"
        ),
        "training reference source_statistics.assets",
    )
    source_assets: list[dict[str, Any]] = []
    for index, raw_asset in enumerate(assets):
        location = f"training reference asset[{index}]"
        asset = _object(raw_asset, location)
        source_assets.append(
            {
                "asset_id": _asset_id(asset.get("asset_id"), f"{location}.asset_id"),
                "leakage_group_id": _text(
                    asset.get("leakage_group_id"), f"{location}.leakage_group_id"
                ),
                "image_count": _integer(
                    asset.get("image_count"), f"{location}.image_count", minimum=1
                ),
                "annotation_count": _integer(
                    asset.get("annotation_count"), f"{location}.annotation_count"
                ),
            }
        )
    asset_ids = [record["asset_id"] for record in source_assets]
    leakage_groups = [record["leakage_group_id"] for record in source_assets]
    source_count = len(assets)
    if (
        source_count < MINIMUM_SOURCE_GROUP_COUNT
        or len({_asset_identity(value) for value in asset_ids}) != source_count
    ):
        raise RecoveryProtocolError(
            "Recovery R1 requires at least "
            f"{MINIMUM_SOURCE_GROUP_COUNT} unique source assets/source groups"
        )
    if len(set(leakage_groups)) != source_count:
        raise RecoveryProtocolError("training reference source leakage groups must be unique")

    annotation_records = [
        _object(record, f"training reference files[{index}]")
        for index, record in enumerate(_list(payload.get("files"), "training reference files"))
        if isinstance(record, dict) and record.get("path") == "annotations.coco.json"
    ]
    if len(annotation_records) != 1:
        raise RecoveryProtocolError(
            "training reference must bind annotations.coco.json exactly once"
        )
    annotation_relative = relative.parent / "annotations.coco.json"
    annotation_path, _ = _workspace_file(
        Path(annotation_relative.as_posix()),
        workspace=workspace,
        location="training annotations",
        allowed_prefix=("data", "ground-truth"),
    )
    annotation_binding = _binding(annotation_path, annotation_relative, "training annotations")
    record = annotation_records[0]
    if annotation_binding["sha256"] != _sha256(
        record.get("sha256"), "annotation sha256"
    ) or annotation_binding["size_bytes"] != record.get("size_bytes"):
        raise RecoveryProtocolError("training annotations do not match the reference manifest")
    return payload, {
        "path": relative.as_posix(),
        "sha256": digest,
        "size_bytes": size_bytes,
        "schema": REFERENCE_SCHEMA,
        "annotations": annotation_binding,
        "source_asset_count": len(assets),
        "source_assets": source_assets,
    }


def _validate_cv_plan(
    path: Path,
    relative: PurePosixPath,
    *,
    reference_binding: Mapping[str, Any],
    workspace: Path,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    payload, digest, size_bytes = _read_json(path, "LOSO plan manifest")
    if payload.get("schema") != CV_PLAN_SCHEMA:
        raise RecoveryProtocolError("LOSO plan manifest schema is unsupported")
    if payload.get("taxonomy") != list(CANONICAL_NAMES):
        raise RecoveryProtocolError("LOSO plan taxonomy does not match the canonical taxonomy")
    gate = _object(payload.get("gate"), "LOSO plan manifest.gate")
    checks = _object(gate.get("checks"), "LOSO plan manifest.gate.checks")
    if (
        gate.get("passed") is not True
        or not checks
        or not all(value is True for value in checks.values())
    ):
        raise RecoveryProtocolError("LOSO plan gate must pass every declared check")
    inputs = _object(payload.get("inputs"), "LOSO plan manifest.inputs")
    bound_reference = _object(inputs.get("reference_manifest"), "LOSO plan reference binding")
    if (
        _sha256(bound_reference.get("sha256"), "LOSO reference sha256")
        != reference_binding["sha256"]
    ):
        raise RecoveryProtocolError("LOSO plan is not bound to the supplied training reference")
    if bound_reference.get("path") != reference_binding["path"]:
        raise RecoveryProtocolError("LOSO plan reference path differs from the supplied reference")
    counts = _object(payload.get("counts"), "LOSO plan manifest.counts")
    folds = _list(payload.get("folds"), "LOSO plan manifest.folds")
    reference_asset_ids = [
        _asset_id(record.get("asset_id"), f"training reference source_assets[{index}].asset_id")
        for index, record in enumerate(
            _list(reference_binding.get("source_assets"), "training reference source_assets")
        )
    ]
    reference_identities = {_asset_identity(value) for value in reference_asset_ids}
    source_count = len(reference_asset_ids)
    if (
        counts.get("source_assets") != source_count
        or counts.get("folds") != source_count
        or len(folds) != source_count
        or ("source_groups" in counts and counts.get("source_groups") != source_count)
    ):
        raise RecoveryProtocolError(
            "LOSO plan counts and fold list must match the training reference source count"
        )
    train_source_count = source_count - 1
    val_assets: list[tuple[str, str]] = []
    fold_ids: list[str] = []
    for index, raw_fold in enumerate(folds):
        fold = _object(raw_fold, f"LOSO plan fold[{index}]")
        fold_id = _text(fold.get("fold_id"), f"LOSO plan fold[{index}].fold_id")
        fold_ids.append(fold_id)
        val_asset = _asset_id(fold.get("val_asset_id"), f"LOSO plan fold[{index}].val_asset_id")
        val_identity = _asset_identity(val_asset)
        val_assets.append(val_identity)
        train = _object(fold.get("train"), f"LOSO plan fold {fold_id}.train")
        val = _object(fold.get("val"), f"LOSO plan fold {fold_id}.val")
        train_asset_ids = [
            _asset_id(value, f"LOSO plan fold {fold_id}.train.asset_ids[{asset_index}]")
            for asset_index, value in enumerate(
                _list(train.get("asset_ids"), f"LOSO plan fold {fold_id}.train.asset_ids")
            )
        ]
        fold_val_asset_ids = [
            _asset_id(value, f"LOSO plan fold {fold_id}.val.asset_ids[{asset_index}]")
            for asset_index, value in enumerate(
                _list(val.get("asset_ids"), f"LOSO plan fold {fold_id}.val.asset_ids")
            )
        ]
        train_identities = {_asset_identity(value) for value in train_asset_ids}
        fold_val_identities = {_asset_identity(value) for value in fold_val_asset_ids}
        if (
            len(train_asset_ids) != train_source_count
            or len(train_identities) != train_source_count
            or fold_val_asset_ids != [val_asset]
            or train_identities & fold_val_identities
            or train_identities | fold_val_identities != reference_identities
            or train_identities != reference_identities - {val_identity}
            or _integer(train.get("source_count"), f"LOSO plan fold {fold_id}.train.source_count")
            != train_source_count
            or _integer(val.get("source_count"), f"LOSO plan fold {fold_id}.val.source_count") != 1
        ):
            raise RecoveryProtocolError(
                f"LOSO plan fold {fold_id} train/validation assets are not an exact complement"
            )
        fold_gate = _object(fold.get("gate"), f"LOSO plan fold[{index}].gate")
        fold_checks = _object(fold_gate.get("checks"), f"LOSO plan fold[{index}].gate.checks")
        if (
            fold_gate.get("passed") is not True
            or not fold_checks
            or not all(value is True for value in fold_checks.values())
        ):
            raise RecoveryProtocolError(f"LOSO plan fold {fold_id} did not pass its gate")
        split = _object(fold.get("split_plan"), f"LOSO plan fold[{index}].split_plan")
        if split.get("schema") != SPLIT_SCHEMA:
            raise RecoveryProtocolError(f"LOSO plan fold {fold_id} has an unsupported split schema")
        split_relative = _safe_relative(split.get("path"), f"LOSO plan fold {fold_id} split path")
        split_path, split_workspace_relative = _workspace_file(
            path.parent / Path(*split_relative.parts),
            workspace=workspace,
            location=f"LOSO plan fold {fold_id} split file",
            allowed_prefix=("docs", "evidence"),
        )
        actual_split, actual_digest, actual_size = _read_json(
            split_path, f"LOSO plan fold {fold_id} split file"
        )
        if actual_split.get("schema") != SPLIT_SCHEMA:
            raise RecoveryProtocolError(f"LOSO plan fold {fold_id} split file schema is invalid")
        if (
            actual_split.get("train_asset_ids") != train_asset_ids
            or actual_split.get("val_asset_ids") != fold_val_asset_ids
        ):
            raise RecoveryProtocolError(
                f"LOSO plan fold {fold_id} split file differs from the fold asset contract"
            )
        if actual_digest != _sha256(
            split.get("sha256"), f"LOSO plan fold {fold_id} split sha256"
        ) or actual_size != split.get("size_bytes"):
            raise RecoveryProtocolError(f"LOSO plan fold {fold_id} split file binding differs")
        expected_relative = relative.parent / split_relative
        if split_workspace_relative != expected_relative:
            raise RecoveryProtocolError(f"LOSO plan fold {fold_id} split path is inconsistent")
    if len(set(fold_ids)) != source_count or len(set(val_assets)) != source_count:
        raise RecoveryProtocolError("each LOSO fold and validation source must be unique")
    plan_semantic_sha256 = _sha256(payload.get("plan_semantic_sha256"), "LOSO plan semantic sha256")
    firewall = _object(payload.get("holdout_firewall"), "LOSO plan holdout_firewall")
    if firewall.get("final_holdout_input_read") is not False:
        raise RecoveryProtocolError("LOSO plan does not prove the final-holdout firewall")
    return (
        {
            "path": relative.as_posix(),
            "sha256": digest,
            "size_bytes": size_bytes,
            "schema": CV_PLAN_SCHEMA,
            "plan_semantic_sha256": plan_semantic_sha256,
            "fold_count": len(folds),
        },
        tuple(sorted(fold_ids)),
    )


def _json_bytes(payload: Any) -> bytes:
    try:
        return (
            json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError) as error:
        raise RecoveryProtocolError(f"protocol is not valid JSON: {error}") from error


def _semantic_sha256(payload: Any) -> str:
    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RecoveryProtocolError(f"reanalysis lineage is not valid JSON: {error}") from error
    return hashlib.sha256(encoded).hexdigest()


def _validate_reanalysis_inputs(
    *,
    source_protocol: Path,
    source_screening_aggregate: Path,
    failure_evidence: Path,
    evaluation_reports: Sequence[Path],
    cv_binding: Mapping[str, Any],
    fold_ids: Sequence[str],
    workspace: Path,
) -> dict[str, Any]:
    """Bind a completed evidence set for analysis-only protocol correction."""

    source_path, source_relative = _workspace_file(
        source_protocol,
        workspace=workspace,
        location="reanalysis source protocol",
        allowed_prefix=("docs", "evidence"),
    )
    source_payload, source_sha, source_size = _read_json(source_path, "reanalysis source protocol")
    if (
        source_payload.get("schema") != PROTOCOL_SCHEMA
        or source_payload.get("status") != "frozen_before_recovery_training"
    ):
        raise RecoveryProtocolError("reanalysis source must be a frozen training protocol")
    source_protocol_id = _text(source_payload.get("protocol_id"), "reanalysis source protocol_id")
    source_inputs = _object(source_payload.get("inputs"), "reanalysis source protocol.inputs")
    source_validation = _object(
        source_payload.get("validation"), "reanalysis source protocol.validation"
    )
    if _object(
        source_inputs.get("loso_plan"), "reanalysis source LOSO binding"
    ) != cv_binding or source_validation.get("fold_count") != len(fold_ids):
        raise RecoveryProtocolError("reanalysis source protocol binds another LOSO fold set")
    source_binding = {
        "path": source_relative.as_posix(),
        "sha256": source_sha,
        "size_bytes": source_size,
        "schema": PROTOCOL_SCHEMA,
        "protocol_id": source_protocol_id,
    }

    screening_path, screening_relative = _workspace_file(
        source_screening_aggregate,
        workspace=workspace,
        location="reanalysis source screening aggregate",
        allowed_prefix=("docs", "evidence"),
    )
    screening_payload, screening_sha, screening_size = _read_json(
        screening_path, "reanalysis source screening aggregate"
    )
    screening_gate = _object(
        screening_payload.get("gate"), "reanalysis source screening aggregate.gate"
    )
    screening_protocol = _object(
        _object(
            screening_payload.get("bindings"),
            "reanalysis source screening aggregate.bindings",
        ).get("recovery_protocol"),
        "reanalysis source screening aggregate recovery protocol",
    )
    selection = _object(screening_payload.get("selection"), "reanalysis source screening selection")
    winner = _text(selection.get("winner_arm_id"), "reanalysis source screening winner")
    screening_contract = _object(
        screening_payload.get("contract"), "reanalysis source screening contract"
    )
    arm_ids = {str(arm["arm_id"]) for arm in EXPERIMENT_ARMS}
    if (
        screening_payload.get("schema") != AGGREGATE_SCHEMA
        or screening_payload.get("mode") != "screening"
        or screening_gate.get("passed") is not True
        or screening_protocol != source_binding
        or winner not in arm_ids
        or screening_contract.get("fold_ids") != list(fold_ids)
    ):
        raise RecoveryProtocolError("reanalysis source screening aggregate is inconsistent")
    screening_binding = {
        "path": screening_relative.as_posix(),
        "sha256": screening_sha,
        "size_bytes": screening_size,
        "schema": AGGREGATE_SCHEMA,
        "winner_arm_id": winner,
    }

    failure_path, failure_relative = _workspace_file(
        failure_evidence,
        workspace=workspace,
        location="reanalysis failure evidence",
        allowed_prefix=("docs", "evidence"),
    )
    failure_payload, failure_sha, failure_size = _read_json(
        failure_path, "reanalysis failure evidence"
    )
    failure_protocol = _object(
        failure_payload.get("protocol"), "reanalysis failure evidence.protocol"
    )
    aggregate_output = _object(
        failure_payload.get("aggregate_output"),
        "reanalysis failure evidence.aggregate_output",
    )
    if (
        failure_payload.get("schema") != AGGREGATION_FAILURE_SCHEMA
        or failure_payload.get("status") != "failed"
        or failure_payload.get("stage") != "confirmation"
        or failure_protocol != source_binding
        or aggregate_output.get("written") is not False
    ):
        raise RecoveryProtocolError("reanalysis failure evidence is inconsistent")
    failure_binding = {
        "path": failure_relative.as_posix(),
        "sha256": failure_sha,
        "size_bytes": failure_size,
        "schema": AGGREGATION_FAILURE_SCHEMA,
    }

    report_bindings: list[dict[str, Any]] = []
    observed: set[tuple[str, int, str]] = set()
    for index, raw_report in enumerate(evaluation_reports):
        report_path, report_relative = _workspace_file(
            raw_report,
            workspace=workspace,
            location=f"reanalysis evaluation report[{index}]",
            allowed_prefix=("data", "model-candidates"),
        )
        report_payload, report_sha, report_size = _read_json(
            report_path, f"reanalysis evaluation report[{index}]"
        )
        experiment = _object(
            report_payload.get("experiment"),
            f"reanalysis evaluation report[{index}].experiment",
        )
        if set(experiment) != {"arm_id", "seed", "fold_id"}:
            raise RecoveryProtocolError(
                f"reanalysis evaluation report[{index}] experiment contract differs"
            )
        arm_id = _text(experiment.get("arm_id"), f"reanalysis report[{index}] arm_id")
        seed = _integer(experiment.get("seed"), f"reanalysis report[{index}] seed")
        fold_id = _text(experiment.get("fold_id"), f"reanalysis report[{index}] fold_id")
        identity = (arm_id, seed, fold_id)
        gate = _object(report_payload.get("gate"), f"reanalysis report[{index}].gate")
        firewall = _object(
            report_payload.get("holdout_firewall"),
            f"reanalysis report[{index}].holdout_firewall",
        )
        if (
            report_payload.get("schema") != EVALUATION_SCHEMA
            or gate.get("passed") is not True
            or firewall.get("input_read") is not False
            or identity in observed
        ):
            raise RecoveryProtocolError(
                f"reanalysis evaluation report[{index}] is unsafe or duplicated"
            )
        observed.add(identity)
        report_bindings.append(
            {
                "path": report_relative.as_posix(),
                "sha256": report_sha,
                "size_bytes": report_size,
                "schema": EVALUATION_SCHEMA,
                "experiment": {
                    "arm_id": arm_id,
                    "seed": seed,
                    "fold_id": fold_id,
                },
            }
        )

    screening_expected = {(arm_id, 42, fold_id) for arm_id in arm_ids for fold_id in fold_ids}
    confirmation_arms = {"repaired_control", winner}
    confirmation_expected = {
        (arm_id, seed, fold_id)
        for arm_id in confirmation_arms
        for seed in (43, 44)
        for fold_id in fold_ids
    }
    expected = screening_expected | confirmation_expected
    if observed != expected:
        raise RecoveryProtocolError(
            "reanalysis reports must be the exact completed screening/confirmation matrix"
        )

    binding_by_experiment = {
        (
            str(binding["experiment"]["arm_id"]),
            int(binding["experiment"]["seed"]),
            str(binding["experiment"]["fold_id"]),
        ): binding
        for binding in report_bindings
    }
    screening_runs = _list(
        screening_payload.get("runs"), "reanalysis source screening aggregate.runs"
    )
    if len(screening_runs) != len(screening_expected):
        raise RecoveryProtocolError("reanalysis source screening run count differs")
    seen_screening: set[tuple[str, int, str]] = set()
    for index, raw_run in enumerate(screening_runs):
        run = _object(raw_run, f"reanalysis source screening run[{index}]")
        experiment = _object(
            run.get("experiment"), f"reanalysis source screening run[{index}].experiment"
        )
        identity = (
            _text(experiment.get("arm_id"), f"source screening run[{index}] arm_id"),
            _integer(experiment.get("seed"), f"source screening run[{index}] seed"),
            _text(experiment.get("fold_id"), f"source screening run[{index}] fold_id"),
        )
        expected_binding = binding_by_experiment.get(identity)
        report = _object(run.get("report"), f"reanalysis source screening run[{index}].report")
        if (
            identity not in screening_expected
            or identity in seen_screening
            or expected_binding is None
            or report
            != {key: expected_binding[key] for key in ("path", "sha256", "size_bytes", "schema")}
        ):
            raise RecoveryProtocolError(
                "reanalysis reports differ from the frozen source screening evidence"
            )
        seen_screening.add(identity)
    if seen_screening != screening_expected:
        raise RecoveryProtocolError("reanalysis source screening matrix is incomplete")

    report_bindings.sort(
        key=lambda binding: (
            str(binding["experiment"]["arm_id"]),
            int(binding["experiment"]["seed"]),
            str(binding["experiment"]["fold_id"]),
        )
    )
    return {
        "mode": "immutable_evidence_reanalysis",
        "source_protocol": source_binding,
        "source_screening_aggregate": screening_binding,
        "failure_evidence": failure_binding,
        "collection_status": "complete_before_reanalysis_protocol_freeze",
        "new_replacement_or_supplemental_runs_allowed": False,
        "report_count": len(report_bindings),
        "reports_manifest_sha256": _semantic_sha256(report_bindings),
        "reports": report_bindings,
    }


def _publish_no_replace(output: Path, encoded: bytes, workspace: Path) -> None:
    raw = os.fspath(output)
    if not raw or "\x00" in raw or any(part == ".." for part in Path(raw).parts):
        raise RecoveryProtocolError("output path is unsafe")
    absolute = output if output.is_absolute() else workspace / output
    try:
        relative = absolute.relative_to(workspace)
    except ValueError as error:
        raise RecoveryProtocolError("output must be inside the workspace") from error
    if _contains_prohibited_input(PurePosixPath(relative.as_posix())):
        raise RecoveryProtocolError("output crosses the final-holdout firewall")
    if relative.parts[:2] != ("docs", "evidence"):
        raise RecoveryProtocolError("output must be below docs/evidence")
    parent = absolute.parent
    try:
        parent_relative = parent.relative_to(workspace)
    except ValueError as error:
        raise RecoveryProtocolError("output parent must be inside the workspace") from error
    current = workspace
    for part in parent_relative.parts:
        current /= part
        try:
            details = os.lstat(current)
        except OSError as error:
            raise RecoveryProtocolError(f"output parent is unavailable: {error}") from error
        if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
            raise RecoveryProtocolError("output parent must not traverse a symlink")
    try:
        os.lstat(absolute)
    except FileNotFoundError:
        pass
    except OSError as error:
        raise RecoveryProtocolError(f"could not inspect output: {error}") from error
    else:
        raise FileExistsError(f"output already exists: {absolute}")

    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{absolute.name}.", dir=parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, absolute, follow_symlinks=False)
        except FileExistsError as error:
            raise FileExistsError(f"output already exists: {absolute}") from error
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def build_recovery_protocol(
    *,
    protocol_id: str,
    training_reference_manifest: Path,
    cv_plan_manifest: Path,
    base_weights: Path,
    implementation_revision: str,
    implementation_files: Sequence[Path],
    output: Path,
    device: str = "mps",
    workspace: Path | None = None,
    reanalysis_source_protocol: Path | None = None,
    reanalysis_source_screening_aggregate: Path | None = None,
    reanalysis_failure_evidence: Path | None = None,
    reanalysis_evaluation_reports: Sequence[Path] = (),
) -> dict[str, Any]:
    """Validate, content-bind and exclusively publish Recovery R1."""

    root = (workspace or Path.cwd()).resolve()
    protocol_id = _text(protocol_id, "protocol_id")
    implementation_revision = _text(implementation_revision, "implementation_revision")
    device = _text(device, "device")
    if device != "mps":
        raise RecoveryProtocolError("Recovery R1 device must be 'mps'")
    if not implementation_files:
        raise RecoveryProtocolError("at least one implementation file is required")
    reanalysis_requested = any(
        value is not None
        for value in (
            reanalysis_source_protocol,
            reanalysis_source_screening_aggregate,
            reanalysis_failure_evidence,
        )
    ) or bool(reanalysis_evaluation_reports)
    if reanalysis_requested and (
        reanalysis_source_protocol is None
        or reanalysis_source_screening_aggregate is None
        or reanalysis_failure_evidence is None
        or not reanalysis_evaluation_reports
    ):
        raise RecoveryProtocolError(
            "reanalysis requires its source protocol, screening aggregate, failure evidence, "
            "and complete report matrix"
        )

    reference_path, reference_relative = _workspace_file(
        training_reference_manifest,
        workspace=root,
        location="training reference manifest",
        allowed_prefix=("data", "ground-truth"),
    )
    _, reference_binding = _validate_reference(reference_path, reference_relative, root)
    cv_path, cv_relative = _workspace_file(
        cv_plan_manifest,
        workspace=root,
        location="LOSO plan manifest",
        allowed_prefix=("docs", "evidence"),
    )
    cv_binding, fold_ids = _validate_cv_plan(
        cv_path,
        cv_relative,
        reference_binding=reference_binding,
        workspace=root,
    )
    weight_path, weight_relative = _workspace_file(
        base_weights,
        workspace=root,
        location="base weights",
    )
    weight_binding = _binding(weight_path, weight_relative, "base weights")

    implementation_bindings: list[dict[str, Any]] = []
    seen_implementation_paths: set[str] = set()
    for index, implementation_file in enumerate(implementation_files):
        path, relative = _workspace_file(
            implementation_file,
            workspace=root,
            location=f"implementation_files[{index}]",
        )
        if relative.as_posix() in seen_implementation_paths:
            raise RecoveryProtocolError("implementation files must be unique")
        seen_implementation_paths.add(relative.as_posix())
        implementation_bindings.append(_binding(path, relative, f"implementation_files[{index}]"))
    implementation_bindings.sort(key=lambda binding: binding["path"])
    implementation_contract = _validate_implementation_revision(
        root, implementation_revision, implementation_bindings
    )

    reanalysis_contract = None
    if reanalysis_requested:
        assert reanalysis_source_protocol is not None
        assert reanalysis_source_screening_aggregate is not None
        assert reanalysis_failure_evidence is not None
        reanalysis_contract = _validate_reanalysis_inputs(
            source_protocol=reanalysis_source_protocol,
            source_screening_aggregate=reanalysis_source_screening_aggregate,
            failure_evidence=reanalysis_failure_evidence,
            evaluation_reports=reanalysis_evaluation_reports,
            cv_binding=cv_binding,
            fold_ids=fold_ids,
            workspace=root,
        )

    common_training = dict(COMMON_TRAINING)
    common_training["device"] = device
    payload = {
        "schema": PROTOCOL_SCHEMA,
        "protocol_id": protocol_id,
        "status": (
            "frozen_before_reanalysis_after_collection"
            if reanalysis_contract is not None
            else "frozen_before_recovery_training"
        ),
        "scope": {
            "phase": "Recovery R1",
            "input_scope": "immutable_training_reference_and_training_internal_validation_only",
            "final_holdout_status": "sealed_and_consumed",
            "new_final_holdout_allowed": False,
            "statement": NO_HOLDOUT_STATEMENT,
        },
        "inputs": {
            "training_reference": reference_binding,
            "loso_plan": cv_binding,
            "base_weights": weight_binding,
        },
        "taxonomy": {
            "canonical_names": list(CANONICAL_NAMES),
            "model_names": list(MODEL_NAMES),
            "mapping": [
                {"id": index, "canonical": canonical, "model": model}
                for index, (canonical, model) in enumerate(zip(CANONICAL_NAMES, MODEL_NAMES))
            ],
            "output_namespace": "canonical",
            "id_order_changes_allowed": False,
        },
        "validation": {
            "method": "leave-one-source-out",
            "fold_count": len(fold_ids),
            "source_asset_and_leakage_group_disjoint": True,
            "every_source_validates_once": True,
            "training_fold_requires_all_classes": True,
            "absent_validation_class_status": "not_evaluable",
        },
        "reproducibility": {
            "fixed_seed_and_deterministic_algorithms_requested": True,
            "bitwise_reproducibility_required": False,
            "accelerator_limitation": (
                "Apple MPS reports non-deterministic scatter_reduce and "
                "index_put_with_accumulate kernels"
            ),
            "statistical_control": "paired folds plus three-seed variability gate",
        },
        "experiments": {
            "common_training": common_training,
            "arms": [dict(arm) for arm in EXPERIMENT_ARMS],
            "screening": {"seed": 42, "all_arms": True, "all_folds": True},
            "confirmation": {
                "winner_and_paired_repaired_control": True,
                "control_may_self_compare_when_it_wins": True,
                "seeds": list(EXPECTED_SEEDS),
                "all_folds": True,
                "reuse_identical_seed_42_screening_runs": True,
                "frozen_screening_aggregate_binding_required": True,
            },
            "unregistered_experiments_allowed": False,
        },
        "evaluation": {
            **EVALUATION_SETTINGS,
            "complete_frame_universe_required": True,
            "image_size": "same_as_experiment_arm_imgsz",
            "threshold_tuning_allowed": False,
        },
        "selection": {
            "ranking": [
                "oof_product_path_f1_desc",
                "worst_source_f1_desc",
                "clean_frame_rate_desc",
                "map50_95_desc",
                "compute_cost_asc",
                "arm_id_asc",
            ],
            "metrics_must_be_aggregated_from_raw_oof_counts": True,
        },
        "readiness_gates": READINESS_GATES,
        "freeze_requirements": {
            "recovery_r1_output": "validated_training_recipe_not_a_deployable_weight",
            "full_training_refit_required_before_external_acceptance": True,
            "weight_taxonomy_runtime_settings_protocol_cv_dependencies_and_code_bound": True,
            "new_unseen_final_acceptance_allowed_only_after_pass": True,
        },
        "refit_contract": {
            "stage": "Recovery R2",
            "separate_protocol_frozen_before_refit": True,
            "trigger": "passed_confirmation_aggregate",
            "dataset": "all_images_from_the_immutable_training_reference",
            "validation_during_refit": False,
            "winner_arm_parameters_reused_exactly": True,
            "seed": 42,
            "epochs": {
                "source": (
                    "best_epoch.number_from_all_"
                    f"{len(EXPECTED_SEEDS) * len(fold_ids)}_confirmed_winner_runs"
                ),
                "statistic": (f"integer_median_of_{len(EXPECTED_SEEDS) * len(fold_ids)}_values"),
                "minimum": 1,
                "maximum": 150,
            },
            "early_stopping": False,
            "patience": 0,
            "selected_checkpoint": "last.pt_at_the_exact_refit_epoch",
            "post_refit_weight_selection": "none",
        },
        "implementation": implementation_contract,
        "abort_policy": {
            "input_or_hash_drift": "abort",
            "configuration_or_implementation_drift": "abort",
            "missing_run_or_non_finite_metric": "abort",
            "failed_internal_gate": "do_not_open_or_replace_the_consumed_holdout",
        },
        **({"reanalysis": reanalysis_contract} if reanalysis_contract is not None else {}),
    }
    _, current_reference_binding = _validate_reference(reference_path, reference_relative, root)
    current_cv_binding, current_fold_ids = _validate_cv_plan(
        cv_path,
        cv_relative,
        reference_binding=current_reference_binding,
        workspace=root,
    )
    current_weight_binding = _binding(weight_path, weight_relative, "base weights")
    current_implementation_bindings = [
        _binding(
            root / Path(*PurePosixPath(str(binding["path"])).parts),
            PurePosixPath(str(binding["path"])),
            f"implementation file {binding['path']}",
        )
        for binding in implementation_bindings
    ]
    current_implementation_contract = _validate_implementation_revision(
        root, implementation_revision, current_implementation_bindings
    )
    current_reanalysis_contract = None
    if reanalysis_contract is not None:
        assert reanalysis_source_protocol is not None
        assert reanalysis_source_screening_aggregate is not None
        assert reanalysis_failure_evidence is not None
        current_reanalysis_contract = _validate_reanalysis_inputs(
            source_protocol=reanalysis_source_protocol,
            source_screening_aggregate=reanalysis_source_screening_aggregate,
            failure_evidence=reanalysis_failure_evidence,
            evaluation_reports=reanalysis_evaluation_reports,
            cv_binding=current_cv_binding,
            fold_ids=current_fold_ids,
            workspace=root,
        )
    if (
        current_reference_binding != reference_binding
        or current_cv_binding != cv_binding
        or current_fold_ids != fold_ids
        or current_weight_binding != weight_binding
        or current_implementation_contract != implementation_contract
        or current_reanalysis_contract != reanalysis_contract
    ):
        raise RecoveryProtocolError("a frozen protocol input changed before publication")
    encoded = _json_bytes(payload)
    _publish_no_replace(output, encoded, root)
    output_path = output if output.is_absolute() else root / output
    return {
        "output": str(output_path),
        "protocol_sha256": hashlib.sha256(encoded).hexdigest(),
        "protocol_id": protocol_id,
        "fold_count": len(fold_ids),
        "screening_run_count": len(EXPERIMENT_ARMS) * len(fold_ids),
        "confirmation_run_count_if_control_wins": len(EXPECTED_SEEDS) * len(fold_ids),
        "confirmation_run_count_if_challenger_wins": (len(EXPECTED_SEEDS) * len(fold_ids) * 2),
        "maximum_unique_run_count_with_screening_reuse": (
            len(EXPERIMENT_ARMS) * len(fold_ids) + 2 * (len(EXPECTED_SEEDS) - 1) * len(fold_ids)
        ),
        "reanalysis": reanalysis_contract is not None,
        "reanalysis_report_count": (
            int(reanalysis_contract["report_count"]) if reanalysis_contract is not None else 0
        ),
        "holdout_input_read": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--training-reference-manifest", type=Path, required=True)
    parser.add_argument("--cv-plan-manifest", type=Path, required=True)
    parser.add_argument("--base-weights", type=Path, required=True)
    parser.add_argument("--implementation-revision", required=True)
    parser.add_argument("--implementation-file", type=Path, action="append", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--reanalysis-source-protocol", type=Path)
    parser.add_argument("--reanalysis-source-screening-aggregate", type=Path)
    parser.add_argument("--reanalysis-failure-evidence", type=Path)
    parser.add_argument("--reanalysis-evaluation-report", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_recovery_protocol(
        protocol_id=args.protocol_id,
        training_reference_manifest=args.training_reference_manifest,
        cv_plan_manifest=args.cv_plan_manifest,
        base_weights=args.base_weights,
        implementation_revision=args.implementation_revision,
        implementation_files=args.implementation_file,
        output=args.output,
        device=args.device,
        reanalysis_source_protocol=args.reanalysis_source_protocol,
        reanalysis_source_screening_aggregate=args.reanalysis_source_screening_aggregate,
        reanalysis_failure_evidence=args.reanalysis_failure_evidence,
        reanalysis_evaluation_reports=args.reanalysis_evaluation_report,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
