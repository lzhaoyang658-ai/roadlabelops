#!/usr/bin/env python3
"""Aggregate immutable training-only LOSO evaluation evidence.

This command deliberately has no final-holdout input.  It validates a frozen
Recovery protocol, its source-derived LOSO plan, evaluator reports and every artifact
directly named by those reports before deriving out-of-fold metrics from raw
TP/FP/FN counts.  Screening selects one arm deterministically.  Confirmation
reuses the frozen seed-42 screening evidence and evaluates paired, same-source
comparisons across seeds 42, 43 and 44.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import numbers
import os
import re
import stat
import statistics
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import yaml

from roadlabelops.holdout_policy import (
    FINAL_HOLDOUT_REJECTED_SCOPES,
    NO_FINAL_HOLDOUT_STATEMENT,
    final_holdout_scope_reason,
)

OUTPUT_SCHEMA = {"name": "roadlabelops.training-cv-aggregate", "version": 1}
PROTOCOL_SCHEMA = {"name": "roadlabelops.training-recovery-protocol", "version": 1}
AGGREGATION_FAILURE_SCHEMA = {
    "name": "roadlabelops.training-recovery-r1-aggregation-failure",
    "version": 1,
}
CV_PLAN_SCHEMA = {"name": "roadlabelops.training-loso-cv-plan", "version": 1}
SPLIT_PLAN_SCHEMA = {"name": "roadlabelops.training-asset-split", "version": 1}
EVALUATION_SCHEMA = {
    "name": "roadlabelops.training-validation-evaluation",
    "version": 1,
}
DATASET_SCHEMA = {"name": "roadlabelops.yolo-dataset", "version": 3}
CANDIDATE_SCHEMA = {"name": "roadlabelops.yolo-candidate-training", "version": 2}
CANDIDATE_PROTOCOL_SCHEMA = {
    "name": "roadlabelops.yolo-candidate-training-protocol",
    "version": 2,
}
PRETRAINED_TRANSFER_SCHEMA = {
    "name": "roadlabelops.pretrained-class-head-transfer",
    "version": 1,
}
REFERENCE_SCHEMA = {"name": "roadlabelops.training-coco-reference", "version": 2}
PRETRAINED_TRANSFER_MESSAGE = "Remapped 8/8 cls head rows from pretrained weights by class name"
COCO_SOURCE_CLASS_IDS = {
    "person": 0,
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "bus": 5,
    "truck": 7,
    "traffic light": 9,
    "stop sign": 11,
}

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
EXPECTED_REPORT_KEYS = {
    "schema",
    "gate",
    "experiment",
    "settings",
    "bindings",
    "val_source",
    "metrics",
    "compute",
    "holdout_firewall",
}
OVERALL_KEYS = {
    "true_positive_count",
    "false_positive_count",
    "false_negative_count",
    "prediction_count",
    "ground_truth_count",
    "precision",
    "recall",
    "f1_score",
    "evaluated_frame_count",
    "clean_frame_count",
    "clean_frame_rate",
    "complete_frame_coverage",
}
PER_CLASS_KEYS = {
    "status",
    "support_count",
    "true_positive_count",
    "false_positive_count",
    "false_negative_count",
    "prediction_count",
    "precision",
    "recall",
    "f1_score",
}
COMPUTE_KEYS = {
    "training_duration_seconds",
    "evaluation_inference_seconds",
    "model_load_seconds",
    "evaluated_frames_per_second",
    "predict_call_count",
}
SETTINGS_KEYS = {
    "confidence",
    "image_size",
    "device",
    "nms_iou",
    "rider_overlap",
    "match_iou",
}
RANKING = [
    "oof_product_path_f1_desc",
    "worst_source_f1_desc",
    "clean_frame_rate_desc",
    "map50_95_desc",
    "compute_cost_asc",
    "arm_id_asc",
]
REPORT_SCOPES = (("docs", "evidence"), ("data", "model-candidates"))
ALLOWED_SCOPES = (("data", "training"), ("docs", "evidence"), ("data", "model-candidates"))
MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_MANAGED_BYTES = 4 * 1024 * 1024 * 1024
MINIMUM_SOURCE_GROUP_COUNT = 3
ARTIFACT_PATHS = {
    "args": "artifacts/args.yaml",
    "results": "artifacts/results.csv",
    "best_weights": "weights/best.pt",
    "last_weights": "weights/last.pt",
}
RESULT_METRIC_COLUMNS = (
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
)
RECEIPT_KEYS = {
    "schema",
    "gate",
    "protocol",
    "protocol_sha256",
    "inputs",
    "resolved_args",
    "metrics",
    "best_epoch",
    "artifacts",
    "holdout",
    "timestamps",
    "environment",
    "mutation_performed",
    "pretrained_transfer",
}
CURRENT_CANDIDATE_GATE_CHECKS = {
    "supported_dataset_manifest_schema_verified",
    "dataset_gate_verified",
    "all_managed_dataset_files_verified",
    "dataset_yaml_hash_and_taxonomy_verified",
    "base_yolo11n_weight_hash_verified",
    "isolated_workspace_training_verified",
    "trainer_args_match_frozen_protocol",
    "support_aware_complete_eight_class_metrics_verified",
    "required_training_artifacts_verified",
    "source_inputs_unchanged",
    "holdout_input_not_read",
    "pretrained_class_head_transfer_verified",
}


class TrainingCVAggregationError(ValueError):
    """Raised when training-CV evidence is unsafe, incomplete or inconsistent."""


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as error:
            raise TrainingCVAggregationError("YAML has an unhashable mapping key") from error
        if duplicate:
            raise TrainingCVAggregationError(f"YAML contains duplicate key {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


@dataclass(frozen=True)
class FileIdentity:
    device: int
    inode: int
    mode: int
    link_count: int
    size_bytes: int
    modified_ns: int
    changed_ns: int


@dataclass(frozen=True)
class BoundFile:
    path: Path
    relative: str
    sha256: str
    size_bytes: int
    identity: FileIdentity
    maximum_bytes: int


@dataclass(frozen=True)
class FoldContract:
    fold_id: str
    train_asset_ids: tuple[int | str, ...]
    val_asset_ids: tuple[int | str, ...]
    train_frame_count: int
    frame_count: int
    train_zero_annotation_frame_count: int
    zero_annotation_frame_count: int
    train_annotation_count: int
    annotation_count: int
    train_class_support: Mapping[str, int]
    class_support: Mapping[str, int]
    split_file_name: str
    split_sha256: str
    split_semantic_sha256: str


@dataclass(frozen=True)
class ProtocolContract:
    payload: Mapping[str, Any]
    binding: Mapping[str, Any]
    cv_binding: Mapping[str, Any]
    training_reference_binding: Mapping[str, Any]
    training_annotations_binding: Mapping[str, Any]
    base_weights_binding: Mapping[str, Any]
    implementation_bindings: tuple[Mapping[str, Any], ...]
    taxonomy: Mapping[str, Any]
    common_training: Mapping[str, Any]
    arms: Mapping[str, Mapping[str, Any]]
    fold_ids: tuple[str, ...]
    folds: Mapping[str, FoldContract]
    screening_seed: int
    confirmation_seeds: tuple[int, ...]
    settings: Mapping[str, Any]
    readiness: Mapping[str, Any]
    reanalysis: Mapping[str, Any] | None
    reanalysis_reports: Mapping[str, Mapping[str, Any]]


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
        raise TrainingCVAggregationError(f"aggregate is not valid JSON: {error}") from error


def _canonical_bytes(payload: Any) -> bytes:
    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise TrainingCVAggregationError(f"value is not canonical JSON: {error}") from error


def _semantic_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TrainingCVAggregationError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise TrainingCVAggregationError(f"JSON contains non-finite number {value}")


def _parse_json(encoded: bytes, location: str) -> dict[str, Any]:
    try:
        value = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TrainingCVAggregationError(f"could not decode {location}: {error}") from error
    return _object(value, location)


def _object(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TrainingCVAggregationError(f"{location} must be an object")
    return value


def _list(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise TrainingCVAggregationError(f"{location} must be a list")
    return value


def _text(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip() or any(ord(char) < 32 for char in value):
        raise TrainingCVAggregationError(f"{location} must be a non-empty safe string")
    return value.strip()


def _integer(value: Any, location: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TrainingCVAggregationError(f"{location} must be an integer >= {minimum}")
    return value


def _number(
    value: Any,
    location: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TrainingCVAggregationError(f"{location} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise TrainingCVAggregationError(f"{location} must be a finite number") from error
    if not math.isfinite(result):
        raise TrainingCVAggregationError(f"{location} must be a finite number")
    if minimum is not None and result < minimum:
        raise TrainingCVAggregationError(f"{location} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise TrainingCVAggregationError(f"{location} must be <= {maximum}")
    return 0.0 if result == 0.0 else result


def _nullable_number(value: Any, location: str) -> float | None:
    return None if value is None else _number(value, location, minimum=0.0, maximum=1.0)


def _sha256(value: Any, location: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise TrainingCVAggregationError(f"{location} must be a lowercase SHA-256 digest")
    return value


def _asset_id(value: Any, location: str) -> int | str:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise TrainingCVAggregationError(f"{location} must be an integer or string")
    if isinstance(value, str) and not value.strip():
        raise TrainingCVAggregationError(f"{location} must be non-empty")
    return value


def _asset_identity(value: int | str) -> tuple[str, str]:
    return type(value).__name__, str(value)


def _same_json(first: Any, second: Any) -> bool:
    return _canonical_bytes(first) == _canonical_bytes(second)


def _same_number(actual: Any, expected: float | None, location: str) -> None:
    if expected is None:
        if actual is not None:
            raise TrainingCVAggregationError(f"{location} must be null")
        return
    observed = _number(actual, location, minimum=0.0, maximum=1.0)
    if not math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise TrainingCVAggregationError(
            f"{location} differs from raw-count recomputation: {observed!r} != {expected!r}"
        )


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _count_f1(tp: int, fp: int, fn: int) -> float | None:
    denominator = 2 * tp + fp + fn
    return (2 * tp / denominator) if denominator else None


def _reported_f1(tp: int, fp: int, fn: int) -> float | None:
    denominator = 2 * tp + fp + fn
    return (2 * tp / denominator) if denominator else None


def _contains_final_holdout(parts: Sequence[str]) -> bool:
    return final_holdout_scope_reason(PurePosixPath(*parts)) is not None


def _contains_forbidden(relative: PurePosixPath) -> bool:
    return final_holdout_scope_reason(relative) is not None


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
        raise TrainingCVAggregationError(f"{location} must be a safe workspace-relative POSIX path")
    if _contains_forbidden(posix):
        raise TrainingCVAggregationError(f"{location} crosses the final-holdout firewall")
    return posix


def _absolute_lexical(path: Path, workspace: Path) -> Path:
    candidate = path if path.is_absolute() else workspace / path
    return Path(os.path.abspath(os.fspath(candidate)))


def _inside_workspace(path: Path, workspace: Path, location: str) -> PurePosixPath:
    try:
        relative_path = path.relative_to(workspace)
    except ValueError as error:
        raise TrainingCVAggregationError(f"{location} must be inside the workspace") from error
    relative = PurePosixPath(*relative_path.parts)
    if _contains_forbidden(relative):
        raise TrainingCVAggregationError(f"{location} crosses the final-holdout firewall")
    return relative


def _has_scope(relative: PurePosixPath, scopes: Sequence[tuple[str, ...]]) -> bool:
    return any(relative.parts[: len(scope)] == scope for scope in scopes)


def _inspect_chain(
    path: Path,
    workspace: Path,
    location: str,
    *,
    leaf_kind: str,
) -> os.stat_result | None:
    relative = _inside_workspace(path, workspace, location)
    current = workspace
    details: os.stat_result | None = None
    for index, part in enumerate(relative.parts):
        current /= part
        try:
            details = os.lstat(current)
        except FileNotFoundError:
            if leaf_kind == "missing" and index == len(relative.parts) - 1:
                return None
            raise TrainingCVAggregationError(f"{location} is unavailable: {current}") from None
        except OSError as error:
            raise TrainingCVAggregationError(f"could not inspect {location}: {error}") from error
        if stat.S_ISLNK(details.st_mode):
            raise TrainingCVAggregationError(f"{location} must not traverse a symlink")
        if index < len(relative.parts) - 1 and not stat.S_ISDIR(details.st_mode):
            raise TrainingCVAggregationError(f"{location} has a non-directory ancestor")
    if details is None:
        raise TrainingCVAggregationError(f"{location} must not be the workspace root")
    if leaf_kind == "file" and not stat.S_ISREG(details.st_mode):
        raise TrainingCVAggregationError(f"{location} must be a regular file")
    if leaf_kind == "directory" and not stat.S_ISDIR(details.st_mode):
        raise TrainingCVAggregationError(f"{location} must be a directory")
    if leaf_kind == "missing":
        raise TrainingCVAggregationError(f"{location} already exists")
    return details


def _scoped_path(
    value: Path | str,
    *,
    workspace: Path,
    scopes: Sequence[tuple[str, ...]],
    location: str,
    leaf_kind: str = "file",
) -> tuple[Path, PurePosixPath]:
    raw = os.fspath(value)
    if not raw or "\x00" in raw or any(part == ".." for part in Path(raw).parts):
        raise TrainingCVAggregationError(f"{location} is unsafe")
    path = _absolute_lexical(Path(raw), workspace)
    relative = _inside_workspace(path, workspace, location)
    if not _has_scope(relative, scopes):
        allowed = ", ".join("/".join(scope) for scope in scopes)
        raise TrainingCVAggregationError(f"{location} must be below one of: {allowed}")
    _inspect_chain(path, workspace, location, leaf_kind=leaf_kind)
    return path, relative


def _identity(details: os.stat_result) -> FileIdentity:
    return FileIdentity(
        device=details.st_dev,
        inode=details.st_ino,
        mode=details.st_mode,
        link_count=details.st_nlink,
        size_bytes=details.st_size,
        modified_ns=details.st_mtime_ns,
        changed_ns=details.st_ctime_ns,
    )


class _EvidenceReader:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.files: dict[Path, BoundFile] = {}
        self.contents: dict[Path, bytes] = {}

    def read(
        self,
        value: Path | str,
        *,
        scopes: Sequence[tuple[str, ...]],
        location: str,
        maximum_bytes: int = MAX_JSON_BYTES,
        retain: bool = True,
        force: bool = False,
    ) -> tuple[bytes, BoundFile]:
        path, relative = _scoped_path(
            value,
            workspace=self.workspace,
            scopes=scopes,
            location=location,
        )
        previous = self.files.get(path)
        if previous is not None and not force:
            try:
                current_identity = _identity(os.lstat(path))
            except OSError as error:
                raise TrainingCVAggregationError(
                    f"could not re-inspect {location}: {error}"
                ) from error
            if current_identity != previous.identity:
                raise TrainingCVAggregationError(f"{location} changed between reads")
            if retain:
                encoded = self.contents.get(path)
                if encoded is None:
                    # A large artifact first read without retention is never expected
                    # to become a JSON input, but re-read it safely if it does.
                    force = True
                else:
                    return encoded, previous
            else:
                return b"", previous
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(path, flags)
        except OSError as error:
            raise TrainingCVAggregationError(f"could not open {location}: {error}") from error
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_size > maximum_bytes:
                raise TrainingCVAggregationError(
                    f"{location} must be a regular file no larger than {maximum_bytes} bytes"
                )
            digest = hashlib.sha256()
            chunks: list[bytes] = []
            size = 0
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > maximum_bytes:
                    raise TrainingCVAggregationError(f"{location} exceeds its size limit")
                digest.update(chunk)
                if retain:
                    chunks.append(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        if _identity(before) != _identity(after) or size != before.st_size:
            raise TrainingCVAggregationError(f"{location} changed while it was read")
        bound = BoundFile(
            path=path,
            relative=relative.as_posix(),
            sha256=digest.hexdigest(),
            size_bytes=size,
            identity=_identity(after),
            maximum_bytes=maximum_bytes,
        )
        previous = self.files.get(path)
        if previous is not None and (
            previous.identity != bound.identity
            or previous.sha256 != bound.sha256
            or previous.size_bytes != bound.size_bytes
            or previous.relative != bound.relative
        ):
            raise TrainingCVAggregationError(f"{location} changed between reads")
        if previous is None:
            self.files[path] = bound
        else:
            bound = previous
        encoded = b"".join(chunks)
        if retain:
            self.contents[path] = encoded
        return encoded, bound

    def json(
        self,
        value: Path | str,
        *,
        scopes: Sequence[tuple[str, ...]],
        location: str,
    ) -> tuple[dict[str, Any], BoundFile]:
        encoded, bound = self.read(value, scopes=scopes, location=location)
        return _parse_json(encoded, location), bound

    def revalidate(self) -> None:
        expected = list(self.files.values())
        for bound in expected:
            _, actual = self.read(
                bound.path,
                # Every entry reached this registry through a narrower, explicit
                # scope check.  Protocol-bound source/code/base-weight files may
                # legitimately sit outside the three report scopes, so final
                # revalidation addresses those already-authorized exact paths.
                scopes=((),),
                location=f"bound input {bound.relative}",
                maximum_bytes=bound.maximum_bytes,
                retain=False,
                force=True,
            )
            if actual != bound:
                raise TrainingCVAggregationError(
                    f"bound input changed before publication: {bound.relative}"
                )


def _binding(bound: BoundFile, *, schema: Any | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": bound.relative,
        "sha256": bound.sha256,
        "size_bytes": bound.size_bytes,
    }
    if schema is not None:
        result["schema"] = schema
    return result


def _require_passed_gate(value: Any, location: str) -> None:
    gate = _object(value, location)
    checks = _object(gate.get("checks"), f"{location}.checks")
    if (
        gate.get("passed") is not True
        or not checks
        or any(item is not True for item in checks.values())
    ):
        raise TrainingCVAggregationError(f"{location} must pass every declared check")
    reasons = gate.get("blocking_reasons")
    if reasons is not None and _list(reasons, f"{location}.blocking_reasons"):
        raise TrainingCVAggregationError(f"{location} has blocking reasons")


def _scan_declared_paths(value: Any, location: str = "input") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{location}.{key}"
            if key in {"path", "root", "file_name"} and isinstance(item, str):
                raw = item.replace("\\", "/")
                parts = tuple(part for part in raw.split("/") if part)
                if _contains_final_holdout(parts):
                    raise TrainingCVAggregationError(
                        f"{child} references a forbidden final-holdout scope"
                    )
            _scan_declared_paths(item, child)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _scan_declared_paths(item, f"{location}[{index}]")


def _fold_identifier(value: Any, location: str) -> str:
    identifier = _text(value, location)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", identifier):
        raise TrainingCVAggregationError(f"{location} has unsafe characters")
    if _contains_final_holdout((identifier,)):
        raise TrainingCVAggregationError(f"{location} must not identify the final holdout")
    return identifier


def _validate_reanalysis_lineage(
    value: Any,
    *,
    reader: _EvidenceReader,
    arms: Mapping[str, Mapping[str, Any]],
    fold_ids: Sequence[str],
    screening_seed: int,
    cv_binding: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Mapping[str, Any]]]:
    lineage = _object(value, "Recovery protocol.reanalysis")
    expected_keys = {
        "mode",
        "source_protocol",
        "source_screening_aggregate",
        "failure_evidence",
        "collection_status",
        "new_replacement_or_supplemental_runs_allowed",
        "report_count",
        "reports_manifest_sha256",
        "reports",
    }
    if set(lineage) != expected_keys:
        raise TrainingCVAggregationError("Recovery reanalysis contract differs")
    if (
        lineage.get("mode") != "immutable_evidence_reanalysis"
        or lineage.get("collection_status") != "complete_before_reanalysis_protocol_freeze"
        or lineage.get("new_replacement_or_supplemental_runs_allowed") is not False
    ):
        raise TrainingCVAggregationError("Recovery reanalysis policy is unsupported")

    source_record = _object(lineage.get("source_protocol"), "Recovery reanalysis source_protocol")
    if set(source_record) != {"path", "sha256", "size_bytes", "schema", "protocol_id"}:
        raise TrainingCVAggregationError("Recovery reanalysis source protocol binding differs")
    source_relative = _safe_relative(
        source_record.get("path"), "Recovery reanalysis source protocol path"
    )
    source_payload, source_file = reader.json(
        reader.workspace / Path(*source_relative.parts),
        scopes=(("docs", "evidence"),),
        location="Recovery reanalysis source protocol",
    )
    expected_source_binding = {
        **_binding(source_file, schema=PROTOCOL_SCHEMA),
        "protocol_id": _text(
            source_payload.get("protocol_id"), "Recovery reanalysis source protocol_id"
        ),
    }
    if (
        source_record != expected_source_binding
        or source_payload.get("schema") != PROTOCOL_SCHEMA
        or source_payload.get("status") != "frozen_before_recovery_training"
    ):
        raise TrainingCVAggregationError("Recovery reanalysis source protocol is invalid")
    source_scope = _object(source_payload.get("scope"), "Recovery reanalysis source scope")
    if (
        source_scope.get("input_scope")
        != "immutable_training_reference_and_training_internal_validation_only"
        or source_scope.get("final_holdout_status") != "sealed_and_consumed"
        or source_scope.get("new_final_holdout_allowed") is not False
    ):
        raise TrainingCVAggregationError(
            "Recovery reanalysis source protocol does not preserve the holdout firewall"
        )
    source_inputs = _object(source_payload.get("inputs"), "Recovery reanalysis source inputs")
    source_validation = _object(
        source_payload.get("validation"), "Recovery reanalysis source validation"
    )
    if _object(
        source_inputs.get("loso_plan"), "Recovery reanalysis source LOSO plan"
    ) != cv_binding or source_validation.get("fold_count") != len(fold_ids):
        raise TrainingCVAggregationError(
            "Recovery reanalysis source protocol binds another LOSO fold set"
        )

    screening_record = _object(
        lineage.get("source_screening_aggregate"),
        "Recovery reanalysis source_screening_aggregate",
    )
    if set(screening_record) != {
        "path",
        "sha256",
        "size_bytes",
        "schema",
        "winner_arm_id",
    }:
        raise TrainingCVAggregationError("Recovery reanalysis source screening binding differs")
    screening_relative = _safe_relative(
        screening_record.get("path"), "Recovery reanalysis source screening path"
    )
    screening_payload, screening_file = reader.json(
        reader.workspace / Path(*screening_relative.parts),
        scopes=(("docs", "evidence"),),
        location="Recovery reanalysis source screening aggregate",
    )
    source_winner = _fold_identifier(
        _object(
            screening_payload.get("selection"),
            "Recovery reanalysis source screening selection",
        ).get("winner_arm_id"),
        "Recovery reanalysis source winner",
    )
    expected_screening_binding = {
        **_binding(screening_file, schema=OUTPUT_SCHEMA),
        "winner_arm_id": source_winner,
    }
    source_screening_protocol = _object(
        _object(
            screening_payload.get("bindings"),
            "Recovery reanalysis source screening bindings",
        ).get("recovery_protocol"),
        "Recovery reanalysis source screening protocol binding",
    )
    if (
        screening_record != expected_screening_binding
        or screening_payload.get("schema") != OUTPUT_SCHEMA
        or screening_payload.get("mode") != "screening"
        or source_screening_protocol != source_record
        or source_winner not in arms
    ):
        raise TrainingCVAggregationError(
            "Recovery reanalysis source screening aggregate is invalid"
        )
    _require_passed_gate(
        screening_payload.get("gate"), "Recovery reanalysis source screening aggregate.gate"
    )
    source_screening_contract = _object(
        screening_payload.get("contract"),
        "Recovery reanalysis source screening aggregate.contract",
    )
    if source_screening_contract.get("fold_ids") != list(fold_ids):
        raise TrainingCVAggregationError("Recovery reanalysis source screening fold set differs")

    failure_record = _object(
        lineage.get("failure_evidence"), "Recovery reanalysis failure_evidence"
    )
    if set(failure_record) != {"path", "sha256", "size_bytes", "schema"}:
        raise TrainingCVAggregationError("Recovery reanalysis failure binding differs")
    failure_relative = _safe_relative(
        failure_record.get("path"), "Recovery reanalysis failure evidence path"
    )
    failure_payload, failure_file = reader.json(
        reader.workspace / Path(*failure_relative.parts),
        scopes=(("docs", "evidence"),),
        location="Recovery reanalysis failure evidence",
    )
    if (
        failure_record != _binding(failure_file, schema=AGGREGATION_FAILURE_SCHEMA)
        or failure_payload.get("schema") != AGGREGATION_FAILURE_SCHEMA
        or failure_payload.get("status") != "failed"
        or failure_payload.get("stage") != "confirmation"
        or _object(failure_payload.get("protocol"), "Recovery reanalysis failure protocol")
        != source_record
        or _object(
            failure_payload.get("aggregate_output"),
            "Recovery reanalysis failure aggregate_output",
        ).get("written")
        is not False
    ):
        raise TrainingCVAggregationError("Recovery reanalysis failure evidence is invalid")

    raw_reports = _list(lineage.get("reports"), "Recovery reanalysis reports")
    report_count = _integer(lineage.get("report_count"), "Recovery reanalysis report_count")
    if report_count != len(raw_reports):
        raise TrainingCVAggregationError("Recovery reanalysis report count differs")
    if _semantic_sha256(raw_reports) != _sha256(
        lineage.get("reports_manifest_sha256"),
        "Recovery reanalysis reports_manifest_sha256",
    ):
        raise TrainingCVAggregationError("Recovery reanalysis report manifest hash differs")

    report_bindings: dict[str, Mapping[str, Any]] = {}
    binding_by_experiment: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for index, raw_binding in enumerate(raw_reports):
        location = f"Recovery reanalysis report[{index}]"
        binding = _object(raw_binding, location)
        if set(binding) != {"path", "sha256", "size_bytes", "schema", "experiment"}:
            raise TrainingCVAggregationError(f"{location} binding contract differs")
        relative = _safe_relative(binding.get("path"), f"{location}.path")
        report_payload, report_file = reader.json(
            reader.workspace / Path(*relative.parts),
            scopes=(("data", "model-candidates"),),
            location=location,
        )
        experiment = _object(binding.get("experiment"), f"{location}.experiment")
        if set(experiment) != {"arm_id", "seed", "fold_id"}:
            raise TrainingCVAggregationError(f"{location} experiment contract differs")
        identity = (
            _fold_identifier(experiment.get("arm_id"), f"{location} arm_id"),
            _integer(experiment.get("seed"), f"{location} seed"),
            _fold_identifier(experiment.get("fold_id"), f"{location} fold_id"),
        )
        expected_binding = {
            **_binding(report_file, schema=EVALUATION_SCHEMA),
            "experiment": {
                "arm_id": identity[0],
                "seed": identity[1],
                "fold_id": identity[2],
            },
        }
        if (
            binding != expected_binding
            or report_payload.get("schema") != EVALUATION_SCHEMA
            or report_payload.get("experiment") != expected_binding["experiment"]
            or relative.as_posix() in report_bindings
            or identity in binding_by_experiment
        ):
            raise TrainingCVAggregationError(f"{location} binding or experiment differs")
        report_bindings[relative.as_posix()] = binding
        binding_by_experiment[identity] = binding

    screening_expected = {
        (arm_id, screening_seed, fold_id) for arm_id in arms for fold_id in fold_ids
    }
    confirmation_expected = {
        (arm_id, seed, fold_id)
        for arm_id in {"repaired_control", source_winner}
        for seed in (43, 44)
        for fold_id in fold_ids
    }
    if set(binding_by_experiment) != screening_expected | confirmation_expected:
        raise TrainingCVAggregationError(
            "Recovery reanalysis reports are not the exact completed evidence matrix"
        )

    screening_runs = _list(
        screening_payload.get("runs"), "Recovery reanalysis source screening runs"
    )
    if len(screening_runs) != len(screening_expected):
        raise TrainingCVAggregationError("Recovery reanalysis source screening run count differs")
    seen_screening: set[tuple[str, int, str]] = set()
    for index, raw_run in enumerate(screening_runs):
        run = _object(raw_run, f"Recovery reanalysis source screening run[{index}]")
        experiment = _object(
            run.get("experiment"),
            f"Recovery reanalysis source screening run[{index}].experiment",
        )
        identity = (
            _fold_identifier(experiment.get("arm_id"), f"source run[{index}] arm_id"),
            _integer(experiment.get("seed"), f"source run[{index}] seed"),
            _fold_identifier(experiment.get("fold_id"), f"source run[{index}] fold_id"),
        )
        expected_binding = binding_by_experiment.get(identity)
        if (
            identity not in screening_expected
            or identity in seen_screening
            or expected_binding is None
            or _object(run.get("report"), f"source run[{index}].report")
            != {key: expected_binding[key] for key in ("path", "sha256", "size_bytes", "schema")}
        ):
            raise TrainingCVAggregationError(
                "Recovery reanalysis reports differ from frozen source screening evidence"
            )
        seen_screening.add(identity)
    if seen_screening != screening_expected:
        raise TrainingCVAggregationError(
            "Recovery reanalysis source screening evidence is incomplete"
        )
    return dict(lineage), report_bindings


def _read_protocol(
    protocol_path: Path,
    *,
    reader: _EvidenceReader,
) -> ProtocolContract:
    payload, protocol_file = reader.json(
        protocol_path,
        scopes=(("docs", "evidence"),),
        location="Recovery protocol",
    )
    _scan_declared_paths(payload, "Recovery protocol")
    if payload.get("schema") != PROTOCOL_SCHEMA:
        raise TrainingCVAggregationError("Recovery protocol schema is unsupported")
    protocol_status = payload.get("status")
    if protocol_status not in {
        "frozen_before_recovery_training",
        "frozen_before_reanalysis_after_collection",
    }:
        raise TrainingCVAggregationError("Recovery protocol is not frozen")
    if protocol_status == "frozen_before_recovery_training" and "reanalysis" in payload:
        raise TrainingCVAggregationError("a training protocol must not contain reanalysis lineage")
    if (
        protocol_status == "frozen_before_reanalysis_after_collection"
        and "reanalysis" not in payload
    ):
        raise TrainingCVAggregationError("a reanalysis protocol must bind its evidence lineage")
    scope = _object(payload.get("scope"), "Recovery protocol.scope")
    if (
        scope.get("input_scope")
        != "immutable_training_reference_and_training_internal_validation_only"
        or scope.get("final_holdout_status") != "sealed_and_consumed"
        or scope.get("new_final_holdout_allowed") is not False
    ):
        raise TrainingCVAggregationError("Recovery protocol does not preserve the holdout firewall")

    inputs = _object(payload.get("inputs"), "Recovery protocol.inputs")
    if set(inputs) != {"training_reference", "loso_plan", "base_weights"}:
        raise TrainingCVAggregationError("Recovery protocol input contract differs")
    raw_reference = _object(
        inputs.get("training_reference"), "Recovery protocol.inputs.training_reference"
    )
    required_reference_fields = {
        "path",
        "sha256",
        "size_bytes",
        "schema",
        "annotations",
        "source_asset_count",
    }
    if not required_reference_fields.issubset(raw_reference):
        raise TrainingCVAggregationError("Recovery training-reference binding contract differs")
    reference_relative = _safe_relative(
        raw_reference.get("path"), "Recovery training-reference path"
    )
    reference_payload, reference_file = reader.json(
        reader.workspace / Path(*reference_relative.parts),
        scopes=(("data", "ground-truth"),),
        location="immutable training-reference manifest",
    )
    if (
        reference_file.sha256
        != _sha256(raw_reference.get("sha256"), "Recovery training-reference sha256")
        or reference_file.size_bytes
        != _integer(raw_reference.get("size_bytes"), "Recovery training-reference size", minimum=1)
        or raw_reference.get("schema") != REFERENCE_SCHEMA
        or reference_payload.get("schema") != REFERENCE_SCHEMA
    ):
        raise TrainingCVAggregationError("Recovery training-reference binding differs")
    _require_passed_gate(reference_payload.get("gate"), "training-reference manifest.gate")
    raw_annotations = _object(
        raw_reference.get("annotations"), "Recovery training-reference annotations"
    )
    if set(raw_annotations) != {"path", "sha256", "size_bytes"}:
        raise TrainingCVAggregationError("Recovery annotations binding contract differs")
    annotations_relative = _safe_relative(
        raw_annotations.get("path"), "Recovery training annotations path"
    )
    _, annotations_file = reader.read(
        reader.workspace / Path(*annotations_relative.parts),
        scopes=(("data", "ground-truth"),),
        location="immutable training annotations",
        maximum_bytes=MAX_MANAGED_BYTES,
        retain=False,
    )
    if annotations_file.sha256 != _sha256(
        raw_annotations.get("sha256"), "Recovery training annotations sha256"
    ) or annotations_file.size_bytes != _integer(
        raw_annotations.get("size_bytes"), "Recovery annotations size", minimum=1
    ):
        raise TrainingCVAggregationError("Recovery training annotations binding differs")
    reference_records = _list(reference_payload.get("files"), "training-reference manifest.files")
    try:
        reference_annotation_path = annotations_file.path.relative_to(
            reference_file.path.parent
        ).as_posix()
    except ValueError as error:
        raise TrainingCVAggregationError(
            "Recovery training annotations must be below the training-reference root"
        ) from error
    matching_annotation_records = [
        record
        for raw_record in reference_records
        if (record := _object(raw_record, "training-reference managed file")).get("path")
        == reference_annotation_path
    ]
    if len(matching_annotation_records) != 1 or matching_annotation_records[0] != {
        "path": reference_annotation_path,
        "sha256": annotations_file.sha256,
        "size_bytes": annotations_file.size_bytes,
    }:
        raise TrainingCVAggregationError(
            "training-reference manifest does not manage the bound annotations file"
        )

    reference_statistics = _object(
        reference_payload.get("source_statistics"),
        "training-reference manifest.source_statistics",
    )
    normalized_source_assets: list[dict[str, Any]] = []
    source_by_identity: dict[tuple[str, str], dict[str, Any]] = {}
    seen_leakage_groups: set[str] = set()
    for index, raw_asset in enumerate(
        _list(
            reference_statistics.get("assets"),
            "training-reference manifest.source_statistics.assets",
        )
    ):
        location = f"training-reference source asset[{index}]"
        asset = _object(raw_asset, location)
        asset_id = _asset_id(asset.get("asset_id"), f"{location}.asset_id")
        identity = _asset_identity(asset_id)
        leakage_group = _text(asset.get("leakage_group_id"), f"{location}.leakage_group_id")
        image_count = _integer(asset.get("image_count"), f"{location}.image_count", minimum=1)
        annotation_count = _integer(asset.get("annotation_count"), f"{location}.annotation_count")
        if identity in source_by_identity:
            raise TrainingCVAggregationError(
                "training-reference source assets must have unique typed IDs"
            )
        if leakage_group in seen_leakage_groups:
            raise TrainingCVAggregationError(
                "training-reference source leakage groups must be unique"
            )
        normalized = {
            "asset_id": asset_id,
            "leakage_group_id": leakage_group,
            "image_count": image_count,
            "annotation_count": annotation_count,
        }
        normalized_source_assets.append(normalized)
        source_by_identity[identity] = normalized
        seen_leakage_groups.add(leakage_group)
    if len(normalized_source_assets) < MINIMUM_SOURCE_GROUP_COUNT:
        raise TrainingCVAggregationError(
            "training-reference manifest must describe at least "
            f"{MINIMUM_SOURCE_GROUP_COUNT} unique source assets/source groups"
        )
    if raw_reference.get("source_asset_count") != len(normalized_source_assets):
        raise TrainingCVAggregationError(
            "Recovery training-reference source count differs from its manifest"
        )
    if "source_assets" in raw_reference:
        protocol_source_assets = _list(
            raw_reference.get("source_assets"),
            "Recovery training-reference source_assets",
        )
        protocol_sources_by_identity: dict[tuple[str, str], Mapping[str, Any]] = {}
        for index, raw_source in enumerate(protocol_source_assets):
            location = f"Recovery training-reference source_assets[{index}]"
            source = _object(raw_source, location)
            source_id = _asset_id(source.get("asset_id"), f"{location}.asset_id")
            identity = _asset_identity(source_id)
            if identity in protocol_sources_by_identity:
                raise TrainingCVAggregationError(
                    "Recovery training-reference source assets contain a duplicate typed ID"
                )
            protocol_sources_by_identity[identity] = source
        if set(protocol_sources_by_identity) != set(source_by_identity):
            raise TrainingCVAggregationError(
                "Recovery training-reference source assets differ from its manifest"
            )
        for identity, actual_source in source_by_identity.items():
            protocol_source = protocol_sources_by_identity[identity]
            if protocol_source.get("leakage_group_id") != actual_source["leakage_group_id"]:
                raise TrainingCVAggregationError(
                    "Recovery training-reference source leakage groups differ from its manifest"
                )
            for optional_count in ("image_count", "annotation_count"):
                if (
                    optional_count in protocol_source
                    and protocol_source[optional_count] != actual_source[optional_count]
                ):
                    raise TrainingCVAggregationError(
                        "Recovery training-reference source counts differ from its manifest"
                    )

    raw_base_weights = _object(inputs.get("base_weights"), "Recovery protocol.inputs.base_weights")
    if set(raw_base_weights) != {"path", "sha256", "size_bytes"}:
        raise TrainingCVAggregationError("Recovery base-weight binding contract differs")
    base_relative = _safe_relative(raw_base_weights.get("path"), "Recovery base-weight path")
    _, base_weights_file = reader.read(
        reader.workspace / Path(*base_relative.parts),
        scopes=((),),
        location="Recovery base YOLO11n weight",
        maximum_bytes=MAX_MANAGED_BYTES,
        retain=False,
    )
    if (
        base_weights_file.path.name != "yolo11n.pt"
        or base_weights_file.sha256
        != _sha256(raw_base_weights.get("sha256"), "Recovery base-weight sha256")
        or base_weights_file.size_bytes
        != _integer(raw_base_weights.get("size_bytes"), "Recovery base-weight size", minimum=1)
    ):
        raise TrainingCVAggregationError("Recovery base-weight binding differs")

    implementation = _object(payload.get("implementation"), "Recovery protocol.implementation")
    if not {"revision", "files"}.issubset(implementation):
        raise TrainingCVAggregationError("Recovery implementation binding contract differs")
    _text(implementation.get("revision"), "Recovery implementation revision")
    implementation_bindings: list[dict[str, Any]] = []
    seen_implementation_paths: set[str] = set()
    for index, raw_file in enumerate(
        _list(implementation.get("files"), "Recovery implementation.files")
    ):
        record = _object(raw_file, f"Recovery implementation.files[{index}]")
        if set(record) != {"path", "sha256", "size_bytes"}:
            raise TrainingCVAggregationError(
                f"Recovery implementation.files[{index}] binding contract differs"
            )
        relative = _safe_relative(
            record.get("path"), f"Recovery implementation.files[{index}].path"
        )
        if relative.as_posix() in seen_implementation_paths:
            raise TrainingCVAggregationError("Recovery implementation paths must be unique")
        seen_implementation_paths.add(relative.as_posix())
        _, implementation_file = reader.read(
            reader.workspace / Path(*relative.parts),
            scopes=((),),
            location=f"Recovery implementation file {relative.as_posix()}",
            maximum_bytes=MAX_JSON_BYTES,
            retain=False,
        )
        if implementation_file.sha256 != _sha256(
            record.get("sha256"), f"Recovery implementation.files[{index}].sha256"
        ) or implementation_file.size_bytes != _integer(
            record.get("size_bytes"),
            f"Recovery implementation.files[{index}].size_bytes",
            minimum=1,
        ):
            raise TrainingCVAggregationError(
                f"Recovery implementation file binding differs: {relative.as_posix()}"
            )
        implementation_bindings.append(_binding(implementation_file))
    if not implementation_bindings:
        raise TrainingCVAggregationError("Recovery implementation.files must not be empty")

    taxonomy = _object(payload.get("taxonomy"), "Recovery protocol.taxonomy")
    if taxonomy.get("canonical_names") != list(CANONICAL_NAMES):
        raise TrainingCVAggregationError("Recovery protocol canonical taxonomy differs")
    model_names = _list(taxonomy.get("model_names"), "Recovery protocol.taxonomy.model_names")
    mapping = _list(taxonomy.get("mapping"), "Recovery protocol.taxonomy.mapping")
    expected_mapping = [
        {"id": index, "canonical": canonical, "model": model}
        for index, (canonical, model) in enumerate(zip(CANONICAL_NAMES, model_names, strict=True))
    ]
    if mapping != expected_mapping or taxonomy.get("output_namespace") != "canonical":
        raise TrainingCVAggregationError("Recovery protocol taxonomy mapping is inconsistent")
    dataset_taxonomy = {
        "canonical_names": list(CANONICAL_NAMES),
        "model_names": model_names,
        "model_to_canonical": {
            model: canonical for model, canonical in zip(model_names, CANONICAL_NAMES, strict=True)
        },
    }

    experiments = _object(payload.get("experiments"), "Recovery protocol.experiments")
    common = _object(experiments.get("common_training"), "Recovery common_training")
    raw_arms = _list(experiments.get("arms"), "Recovery protocol.experiments.arms")
    arms: dict[str, dict[str, Any]] = {}
    for index, raw_arm in enumerate(raw_arms):
        arm = _object(raw_arm, f"Recovery arm[{index}]")
        arm_id = _fold_identifier(arm.get("arm_id"), f"Recovery arm[{index}].arm_id")
        if arm_id in arms:
            raise TrainingCVAggregationError(f"duplicate Recovery arm_id {arm_id!r}")
        arms[arm_id] = dict(arm)
    if len(arms) != 3 or "repaired_control" not in arms:
        raise TrainingCVAggregationError(
            "screening requires exactly three arms including repaired_control"
        )
    screening = _object(experiments.get("screening"), "Recovery screening")
    screening_seed = _integer(screening.get("seed"), "Recovery screening seed")
    if (
        screening_seed != 42
        or screening.get("all_arms") is not True
        or screening.get("all_folds") is not True
    ):
        raise TrainingCVAggregationError("Recovery screening must cover all arms and folds")
    confirmation = _object(experiments.get("confirmation"), "Recovery confirmation")
    confirmation_seeds = tuple(
        _integer(value, f"Recovery confirmation seed[{index}]")
        for index, value in enumerate(
            _list(confirmation.get("seeds"), "Recovery confirmation.seeds")
        )
    )
    if (
        confirmation_seeds != (42, 43, 44)
        or confirmation.get("all_folds") is not True
        or confirmation.get("reuse_identical_seed_42_screening_runs") is not True
        or confirmation.get("frozen_screening_aggregate_binding_required") is not True
        or confirmation.get("winner_and_paired_repaired_control") is not True
    ):
        raise TrainingCVAggregationError("Recovery confirmation contract is unsupported")

    evaluation = _object(payload.get("evaluation"), "Recovery protocol.evaluation")
    if (
        evaluation.get("complete_frame_universe_required") is not True
        or evaluation.get("image_size") != "same_as_experiment_arm_imgsz"
        or evaluation.get("threshold_tuning_allowed") is not False
        or evaluation.get("class_threshold_overrides") != {}
    ):
        raise TrainingCVAggregationError("Recovery evaluation contract is unsupported")
    fixed_settings = {
        "confidence": _number(
            evaluation.get("confidence"), "evaluation.confidence", minimum=0, maximum=1
        ),
        "device": _text(common.get("device"), "common_training.device"),
        "nms_iou": _number(evaluation.get("nms_iou"), "evaluation.nms_iou", minimum=0, maximum=1),
        "rider_overlap": _number(
            evaluation.get("rider_overlap"), "evaluation.rider_overlap", minimum=0, maximum=1
        ),
        "match_iou": _number(
            evaluation.get("match_iou"), "evaluation.match_iou", minimum=0, maximum=1
        ),
    }

    selection = _object(payload.get("selection"), "Recovery protocol.selection")
    if (
        selection.get("ranking") != RANKING
        or selection.get("metrics_must_be_aggregated_from_raw_oof_counts") is not True
    ):
        raise TrainingCVAggregationError("Recovery selection ranking differs from v1")

    raw_cv = _object(inputs.get("loso_plan"), "Recovery protocol.inputs.loso_plan")
    cv_relative = _safe_relative(raw_cv.get("path"), "Recovery LOSO plan path")
    cv_payload, cv_file = reader.json(
        reader.workspace / Path(*cv_relative.parts),
        scopes=(("docs", "evidence"),),
        location="LOSO plan manifest",
    )
    if (
        cv_file.sha256 != _sha256(raw_cv.get("sha256"), "Recovery LOSO plan sha256")
        or cv_file.size_bytes
        != _integer(raw_cv.get("size_bytes"), "Recovery LOSO plan size", minimum=1)
        or raw_cv.get("schema") != CV_PLAN_SCHEMA
        or cv_payload.get("schema") != CV_PLAN_SCHEMA
    ):
        raise TrainingCVAggregationError("Recovery LOSO plan binding differs")
    _require_passed_gate(cv_payload.get("gate"), "LOSO plan.gate")
    cv_inputs = _object(cv_payload.get("inputs"), "LOSO plan.inputs")
    cv_reference = _object(cv_inputs.get("reference_manifest"), "LOSO plan reference binding")
    if cv_reference != {
        "path": reference_file.relative,
        "sha256": reference_file.sha256,
        "size_bytes": reference_file.size_bytes,
        "schema": REFERENCE_SCHEMA,
    }:
        raise TrainingCVAggregationError("LOSO plan binds another training-reference manifest")
    cv_coco = _object(cv_inputs.get("coco"), "LOSO plan COCO binding")
    if cv_coco != {
        "path": annotations_file.relative,
        "sha256": annotations_file.sha256,
        "size_bytes": annotations_file.size_bytes,
        "schema": REFERENCE_SCHEMA,
    }:
        raise TrainingCVAggregationError("LOSO plan binds another training COCO file")
    if cv_payload.get("taxonomy") != list(CANONICAL_NAMES):
        raise TrainingCVAggregationError("LOSO plan taxonomy differs")
    plan_semantic = _sha256(raw_cv.get("plan_semantic_sha256"), "Recovery LOSO semantic hash")
    if cv_payload.get("plan_semantic_sha256") != plan_semantic:
        raise TrainingCVAggregationError("LOSO plan semantic binding differs")
    counts = _object(cv_payload.get("counts"), "LOSO plan.counts")
    source_count = len(normalized_source_assets)
    declared_fold_count = _integer(
        counts.get("folds"),
        "LOSO plan.counts.folds",
        minimum=MINIMUM_SOURCE_GROUP_COUNT,
    )
    declared_source_count = _integer(
        counts.get("source_assets"),
        "LOSO plan.counts.source_assets",
        minimum=MINIMUM_SOURCE_GROUP_COUNT,
    )
    raw_folds = _list(cv_payload.get("folds"), "LOSO plan.folds")
    if (
        declared_source_count != source_count
        or declared_fold_count != source_count
        or len(raw_folds) != declared_fold_count
        or raw_reference.get("source_asset_count") != source_count
        or ("source_groups" in counts and counts.get("source_groups") != source_count)
    ):
        raise TrainingCVAggregationError(
            "LOSO source count, fold count, fold list, and training reference must agree"
        )
    if "fold_count" in raw_cv and raw_cv.get("fold_count") != declared_fold_count:
        raise TrainingCVAggregationError("Recovery LOSO fold count binding differs")
    cv_image_count = _integer(counts.get("images"), "LOSO plan.counts.images", minimum=1)
    cv_zero_count = _integer(
        counts.get("zero_annotation_images"),
        "LOSO plan.counts.zero_annotation_images",
    )
    cv_annotation_count = _integer(counts.get("annotations"), "LOSO plan.counts.annotations")
    if cv_zero_count > cv_image_count or counts.get("categories") != len(CANONICAL_NAMES):
        raise TrainingCVAggregationError("LOSO plan global counts are inconsistent")
    reference_counts = _object(
        reference_payload.get("counts"), "training-reference manifest.counts"
    )
    reference_class_counts = _object(
        reference_counts.get("annotations_by_category"),
        "training-reference manifest.counts.annotations_by_category",
    )
    if (
        reference_counts.get("images") != cv_image_count
        or reference_counts.get("zero_annotation_images") != cv_zero_count
        or reference_counts.get("annotations") != cv_annotation_count
        or reference_counts.get("categories") != len(CANONICAL_NAMES)
        or set(reference_class_counts) != set(CANONICAL_NAMES)
    ):
        raise TrainingCVAggregationError(
            "LOSO plan global counts differ from the training-reference manifest"
        )
    validated_reference_class_counts = {
        label: _integer(
            reference_class_counts[label],
            f"training-reference annotations_by_category.{label}",
        )
        for label in CANONICAL_NAMES
    }
    if sum(validated_reference_class_counts.values()) != cv_annotation_count:
        raise TrainingCVAggregationError(
            "training-reference per-class counts do not sum to its annotation count"
        )

    folds: dict[str, FoldContract] = {}
    semantic_folds: list[dict[str, Any]] = []
    seen_assets: set[tuple[str, str]] = set()
    train_source_count = source_count - 1
    for index, raw_fold in enumerate(raw_folds):
        fold = _object(raw_fold, f"LOSO fold[{index}]")
        if set(fold) != {
            "fold_id",
            "val_asset_id",
            "train",
            "val",
            "validation_evaluability",
            "split_plan",
            "gate",
        }:
            raise TrainingCVAggregationError(f"LOSO fold[{index}] has unexpected or missing fields")
        fold_id = _fold_identifier(fold.get("fold_id"), f"LOSO fold[{index}].fold_id")
        if fold_id in folds:
            raise TrainingCVAggregationError(f"duplicate LOSO fold {fold_id!r}")
        val_asset = _asset_id(fold.get("val_asset_id"), f"LOSO fold {fold_id}.val_asset_id")
        identity = _asset_identity(val_asset)
        if identity in seen_assets:
            raise TrainingCVAggregationError("a validation source appears in multiple folds")
        seen_assets.add(identity)
        _require_passed_gate(fold.get("gate"), f"LOSO fold {fold_id}.gate")
        train = _object(fold.get("train"), f"LOSO fold {fold_id}.train")
        val = _object(fold.get("val"), f"LOSO fold {fold_id}.val")
        partition_keys = {
            "asset_ids",
            "source_count",
            "image_count",
            "zero_annotation_image_count",
            "annotation_count",
            "classes",
        }
        if set(train) != partition_keys or set(val) != partition_keys:
            raise TrainingCVAggregationError(f"LOSO fold {fold_id} train/val contract differs")
        train_assets = tuple(
            _asset_id(value, f"LOSO fold {fold_id}.train.asset_ids[{item_index}]")
            for item_index, value in enumerate(
                _list(train.get("asset_ids"), f"LOSO fold {fold_id}.train.asset_ids")
            )
        )
        val_assets = tuple(
            _asset_id(value, f"LOSO fold {fold_id}.val.asset_ids[{item_index}]")
            for item_index, value in enumerate(
                _list(val.get("asset_ids"), f"LOSO fold {fold_id}.val.asset_ids")
            )
        )
        train_identities = {_asset_identity(value) for value in train_assets}
        val_identities = {_asset_identity(value) for value in val_assets}
        if (
            len(train_assets) != train_source_count
            or len(train_identities) != train_source_count
            or val_assets != (val_asset,)
            or len(val_identities) != 1
            or train_identities & val_identities
            or _integer(train.get("source_count"), f"LOSO fold {fold_id}.train.source_count")
            != train_source_count
            or _integer(val.get("source_count"), f"LOSO fold {fold_id}.val.source_count") != 1
        ):
            raise TrainingCVAggregationError(
                f"LOSO fold {fold_id} must have {train_source_count} typed train sources "
                "and one disjoint val source"
            )
        train_classes = _object(train.get("classes"), f"LOSO fold {fold_id}.train.classes")
        classes = _object(val.get("classes"), f"LOSO fold {fold_id}.val.classes")
        evaluability = _object(
            fold.get("validation_evaluability"), f"LOSO fold {fold_id}.validation_evaluability"
        )
        if (
            set(train_classes) != set(CANONICAL_NAMES)
            or set(classes) != set(CANONICAL_NAMES)
            or set(evaluability) != set(CANONICAL_NAMES)
        ):
            raise TrainingCVAggregationError(f"LOSO fold {fold_id} class contract differs")
        train_class_support: dict[str, int] = {}
        class_support: dict[str, int] = {}
        for label in CANONICAL_NAMES:
            train_record = _object(
                train_classes[label], f"LOSO fold {fold_id}.train.classes.{label}"
            )
            class_record = _object(classes[label], f"LOSO fold {fold_id}.val.classes.{label}")
            class_keys = {"box_count", "positive_image_count", "zero_image_count"}
            if set(train_record) != class_keys or set(class_record) != class_keys:
                raise TrainingCVAggregationError(
                    f"LOSO fold {fold_id} class-statistics contract differs for {label}"
                )
            train_support = _integer(
                train_record.get("box_count"),
                f"LOSO fold {fold_id}.train.{label}.box_count",
                minimum=1,
            )
            support = _integer(
                class_record.get("box_count"), f"LOSO fold {fold_id}.{label}.box_count"
            )
            expected_status = "evaluable" if support else "not_evaluable"
            if evaluability[label] != expected_status:
                raise TrainingCVAggregationError(
                    f"LOSO fold {fold_id} evaluability differs for {label}"
                )
            train_class_support[label] = train_support
            class_support[label] = support
        split = _object(fold.get("split_plan"), f"LOSO fold {fold_id}.split_plan")
        if set(split) != {"path", "sha256", "size_bytes", "schema"}:
            raise TrainingCVAggregationError(
                f"LOSO fold {fold_id} split-plan record contract differs"
            )
        split_relative = _safe_relative(split.get("path"), f"LOSO fold {fold_id} split path")
        split_payload, split_file = reader.json(
            cv_file.path.parent / Path(*split_relative.parts),
            scopes=(("docs", "evidence"),),
            location=f"LOSO fold {fold_id} split plan",
        )
        if (
            split.get("schema") != SPLIT_PLAN_SCHEMA
            or split_payload.get("schema") != SPLIT_PLAN_SCHEMA
            or split_file.sha256 != _sha256(split.get("sha256"), f"LOSO fold {fold_id} split sha")
            or split_file.size_bytes
            != _integer(split.get("size_bytes"), f"LOSO fold {fold_id} split size", minimum=1)
        ):
            raise TrainingCVAggregationError(f"LOSO fold {fold_id} split-plan binding differs")
        if set(split_payload) != {"schema", "train_asset_ids", "val_asset_ids"}:
            raise TrainingCVAggregationError(f"LOSO fold {fold_id} split-plan contract differs")
        split_train_assets = tuple(
            _asset_id(value, f"{fold_id} split train_asset_ids[{item_index}]")
            for item_index, value in enumerate(
                _list(split_payload.get("train_asset_ids"), f"{fold_id} split train assets")
            )
        )
        split_val_assets = tuple(
            _asset_id(value, f"{fold_id} split val_asset_ids[{item_index}]")
            for item_index, value in enumerate(
                _list(split_payload.get("val_asset_ids"), f"{fold_id} split val assets")
            )
        )
        if split_train_assets != train_assets or split_val_assets != val_assets:
            raise TrainingCVAggregationError(
                f"LOSO fold {fold_id} split train/val assets differ from its manifest fold"
            )
        train_frame_count = _integer(
            train.get("image_count"), f"LOSO fold {fold_id}.train.image_count", minimum=1
        )
        val_frame_count = _integer(
            val.get("image_count"), f"LOSO fold {fold_id}.val.image_count", minimum=1
        )
        train_zero_count = _integer(
            train.get("zero_annotation_image_count"),
            f"LOSO fold {fold_id}.train.zero_annotation_image_count",
        )
        val_zero_count = _integer(
            val.get("zero_annotation_image_count"),
            f"LOSO fold {fold_id}.val.zero_annotation_image_count",
        )
        train_annotation_count = _integer(
            train.get("annotation_count"), f"LOSO fold {fold_id}.train.annotation_count"
        )
        val_annotation_count = _integer(
            val.get("annotation_count"), f"LOSO fold {fold_id}.val.annotation_count"
        )
        for partition_name, partition_classes, partition_support, partition_frames in (
            ("train", train_classes, train_class_support, train_frame_count),
            ("val", classes, class_support, val_frame_count),
        ):
            for label in CANONICAL_NAMES:
                class_record = _object(
                    partition_classes[label],
                    f"LOSO fold {fold_id}.{partition_name}.classes.{label}",
                )
                positive_images = _integer(
                    class_record.get("positive_image_count"),
                    f"LOSO fold {fold_id}.{partition_name}.{label}.positive_image_count",
                )
                zero_images = _integer(
                    class_record.get("zero_image_count"),
                    f"LOSO fold {fold_id}.{partition_name}.{label}.zero_image_count",
                )
                if (
                    positive_images > partition_frames
                    or zero_images != partition_frames - positive_images
                    or positive_images > partition_support[label]
                    or (partition_support[label] == 0) != (positive_images == 0)
                ):
                    raise TrainingCVAggregationError(
                        f"LOSO fold {fold_id} {partition_name} class/image counts "
                        f"are inconsistent for {label}"
                    )
        if (
            train_zero_count > train_frame_count
            or val_zero_count > val_frame_count
            or train_annotation_count != sum(train_class_support.values())
            or val_annotation_count != sum(class_support.values())
        ):
            raise TrainingCVAggregationError(
                f"LOSO fold {fold_id} partition counts are inconsistent"
            )
        folds[fold_id] = FoldContract(
            fold_id=fold_id,
            train_asset_ids=train_assets,
            val_asset_ids=val_assets,
            train_frame_count=train_frame_count,
            frame_count=val_frame_count,
            train_zero_annotation_frame_count=train_zero_count,
            zero_annotation_frame_count=val_zero_count,
            train_annotation_count=train_annotation_count,
            annotation_count=val_annotation_count,
            train_class_support=train_class_support,
            class_support=class_support,
            split_file_name=split_file.path.name,
            split_sha256=split_file.sha256,
            split_semantic_sha256=_semantic_sha256(split_payload),
        )
        semantic_folds.append({"fold_id": fold_id, "split_plan": split_payload})
    if len(folds) != source_count or len(seen_assets) != source_count:
        raise TrainingCVAggregationError("LOSO plan must contain one unique fold per unique source")
    if "validation" in payload:
        protocol_validation = _object(payload.get("validation"), "Recovery protocol.validation")
        if protocol_validation.get("fold_count") != len(folds):
            raise TrainingCVAggregationError(
                "Recovery protocol validation fold count differs from its LOSO plan"
            )
    if seen_assets != set(source_by_identity):
        raise TrainingCVAggregationError(
            "LOSO plan source assets differ from the immutable training reference"
        )
    if (
        sum(int(source["image_count"]) for source in source_by_identity.values()) != cv_image_count
        or sum(int(source["annotation_count"]) for source in source_by_identity.values())
        != cv_annotation_count
    ):
        raise TrainingCVAggregationError(
            "training-reference source counts do not sum to its global counts"
        )
    for fold in folds.values():
        train_identities = {_asset_identity(value) for value in fold.train_asset_ids}
        val_identities = {_asset_identity(value) for value in fold.val_asset_ids}
        if train_identities | val_identities != seen_assets or train_identities != (
            seen_assets - val_identities
        ):
            raise TrainingCVAggregationError(
                f"LOSO fold {fold.fold_id} does not partition the complete typed source set"
            )
        val_source = source_by_identity[next(iter(val_identities))]
        train_sources = [source_by_identity[identity] for identity in train_identities]
        if (
            fold.frame_count != val_source["image_count"]
            or fold.annotation_count != val_source["annotation_count"]
            or fold.train_frame_count != sum(int(source["image_count"]) for source in train_sources)
            or fold.train_annotation_count
            != sum(int(source["annotation_count"]) for source in train_sources)
            or fold.train_frame_count + fold.frame_count != cv_image_count
            or fold.train_zero_annotation_frame_count + fold.zero_annotation_frame_count
            != cv_zero_count
            or fold.train_annotation_count + fold.annotation_count != cv_annotation_count
        ):
            raise TrainingCVAggregationError(
                f"LOSO fold {fold.fold_id} counts differ from its source partition"
            )
        if any(
            fold.train_class_support[label] + fold.class_support[label]
            != validated_reference_class_counts[label]
            for label in CANONICAL_NAMES
        ):
            raise TrainingCVAggregationError(
                f"LOSO fold {fold.fold_id} class counts differ from the training reference"
            )
    if (
        sum(fold.frame_count for fold in folds.values()) != cv_image_count
        or sum(fold.zero_annotation_frame_count for fold in folds.values()) != cv_zero_count
        or sum(fold.annotation_count for fold in folds.values()) != cv_annotation_count
        or any(
            sum(fold.class_support[label] for fold in folds.values())
            != validated_reference_class_counts[label]
            for label in CANONICAL_NAMES
        )
    ):
        raise TrainingCVAggregationError(
            "LOSO validation folds do not reconstruct the immutable training reference"
        )
    expected_plan_semantic = _semantic_sha256(
        {
            "schema": CV_PLAN_SCHEMA,
            "method": "leave-one-source-asset-out",
            "folds": semantic_folds,
        }
    )
    if expected_plan_semantic != plan_semantic:
        raise TrainingCVAggregationError("LOSO plan semantic hash does not match its fold payloads")

    readiness = _object(payload.get("readiness_gates"), "Recovery readiness_gates")
    required_readiness = {
        "oof_precision_min",
        "oof_recall_min",
        "clean_frame_rate_min",
        "every_seed_and_source_f1_noninferior_to_repaired_control",
        "supported_class_recall_drop_max",
        "seed_oof_f1_sample_standard_deviation_max",
        "all_planned_runs_required",
        "exact_contract_match_required",
    }
    if set(readiness) != required_readiness:
        raise TrainingCVAggregationError("Recovery readiness gate keys differ from v1")
    for key in (
        "oof_precision_min",
        "oof_recall_min",
        "clean_frame_rate_min",
        "supported_class_recall_drop_max",
        "seed_oof_f1_sample_standard_deviation_max",
    ):
        _number(readiness[key], f"Recovery readiness_gates.{key}", minimum=0, maximum=1)
    if any(
        readiness[key] is not True
        for key in (
            "every_seed_and_source_f1_noninferior_to_repaired_control",
            "all_planned_runs_required",
            "exact_contract_match_required",
        )
    ):
        raise TrainingCVAggregationError("Recovery readiness boolean gates must be true")

    reanalysis: Mapping[str, Any] | None = None
    reanalysis_reports: Mapping[str, Mapping[str, Any]] = {}
    if protocol_status == "frozen_before_reanalysis_after_collection":
        reanalysis, reanalysis_reports = _validate_reanalysis_lineage(
            payload.get("reanalysis"),
            reader=reader,
            arms=arms,
            fold_ids=tuple(sorted(folds)),
            screening_seed=screening_seed,
            cv_binding=raw_cv,
        )

    return ProtocolContract(
        payload=payload,
        binding={
            **_binding(protocol_file, schema=PROTOCOL_SCHEMA),
            "protocol_id": _text(payload.get("protocol_id"), "Recovery protocol_id"),
        },
        cv_binding={
            **_binding(cv_file, schema=CV_PLAN_SCHEMA),
            "plan_semantic_sha256": plan_semantic,
        },
        training_reference_binding=_binding(reference_file, schema=REFERENCE_SCHEMA),
        training_annotations_binding=_binding(annotations_file, schema=REFERENCE_SCHEMA),
        base_weights_binding=_binding(base_weights_file),
        implementation_bindings=tuple(implementation_bindings),
        taxonomy=dataset_taxonomy,
        common_training=common,
        arms=arms,
        fold_ids=tuple(sorted(folds)),
        folds=folds,
        screening_seed=screening_seed,
        confirmation_seeds=confirmation_seeds,
        settings=fixed_settings,
        readiness=readiness,
        reanalysis=reanalysis,
        reanalysis_reports=reanalysis_reports,
    )


def _validate_report_metric_counts(
    report: Mapping[str, Any], fold: FoldContract, location: str
) -> None:
    metrics = _object(report.get("metrics"), f"{location}.metrics")
    if set(metrics) != {"map50", "map50_95", "overall", "per_class"}:
        raise TrainingCVAggregationError(f"{location}.metrics has unexpected or missing fields")
    _number(metrics.get("map50"), f"{location}.metrics.map50", minimum=0, maximum=1)
    _number(metrics.get("map50_95"), f"{location}.metrics.map50_95", minimum=0, maximum=1)
    overall = _object(metrics.get("overall"), f"{location}.metrics.overall")
    if set(overall) != OVERALL_KEYS:
        raise TrainingCVAggregationError(f"{location}.metrics.overall contract differs")
    tp = _integer(overall.get("true_positive_count"), f"{location}.overall.tp")
    fp = _integer(overall.get("false_positive_count"), f"{location}.overall.fp")
    fn = _integer(overall.get("false_negative_count"), f"{location}.overall.fn")
    predictions = _integer(overall.get("prediction_count"), f"{location}.overall.predictions")
    ground_truth = _integer(overall.get("ground_truth_count"), f"{location}.overall.ground_truth")
    frames = _integer(overall.get("evaluated_frame_count"), f"{location}.overall.frames", minimum=1)
    clean = _integer(overall.get("clean_frame_count"), f"{location}.overall.clean")
    if predictions != tp + fp or ground_truth != tp + fn:
        raise TrainingCVAggregationError(f"{location} overall raw counts are inconsistent")
    if ground_truth != fold.annotation_count or frames != fold.frame_count or clean > frames:
        raise TrainingCVAggregationError(f"{location} overall counts differ from the LOSO fold")
    precision = _ratio(tp, predictions)
    recall = _ratio(tp, ground_truth)
    _same_number(overall.get("precision"), precision, f"{location}.overall.precision")
    _same_number(overall.get("recall"), recall, f"{location}.overall.recall")
    _same_number(
        overall.get("f1_score"),
        _reported_f1(tp, fp, fn),
        f"{location}.overall.f1_score",
    )
    _same_number(
        overall.get("clean_frame_rate"),
        _ratio(clean, frames),
        f"{location}.overall.clean_frame_rate",
    )
    if overall.get("complete_frame_coverage") is not True:
        raise TrainingCVAggregationError(f"{location} did not evaluate every validation frame")

    per_class = _object(metrics.get("per_class"), f"{location}.metrics.per_class")
    if set(per_class) != set(CANONICAL_NAMES):
        raise TrainingCVAggregationError(f"{location} per-class taxonomy differs")
    class_totals = {"tp": 0, "fp": 0, "fn": 0}
    for label in CANONICAL_NAMES:
        record = _object(per_class[label], f"{location}.per_class.{label}")
        if set(record) != PER_CLASS_KEYS:
            raise TrainingCVAggregationError(f"{location}.per_class.{label} contract differs")
        class_tp = _integer(record.get("true_positive_count"), f"{location}.{label}.tp")
        class_fp = _integer(record.get("false_positive_count"), f"{location}.{label}.fp")
        class_fn = _integer(record.get("false_negative_count"), f"{location}.{label}.fn")
        support = _integer(record.get("support_count"), f"{location}.{label}.support")
        predictions_for_class = _integer(
            record.get("prediction_count"), f"{location}.{label}.predictions"
        )
        if support != class_tp + class_fn or predictions_for_class != class_tp + class_fp:
            raise TrainingCVAggregationError(f"{location}.{label} raw counts are inconsistent")
        if support != fold.class_support[label]:
            raise TrainingCVAggregationError(f"{location}.{label} support differs from LOSO plan")
        expected_status = "evaluable" if support else "not_evaluable"
        if record.get("status") != expected_status:
            raise TrainingCVAggregationError(f"{location}.{label} evaluability is incorrect")
        if support == 0:
            if any(record.get(key) is not None for key in ("precision", "recall", "f1_score")):
                raise TrainingCVAggregationError(
                    f"{location}.{label} missing validation class must be not_evaluable/null"
                )
        else:
            class_precision = _ratio(class_tp, predictions_for_class)
            class_recall = _ratio(class_tp, support)
            _same_number(record.get("precision"), class_precision, f"{location}.{label}.precision")
            _same_number(record.get("recall"), class_recall, f"{location}.{label}.recall")
            _same_number(
                record.get("f1_score"),
                _reported_f1(class_tp, class_fp, class_fn),
                f"{location}.{label}.f1",
            )
        class_totals["tp"] += class_tp
        class_totals["fp"] += class_fp
        class_totals["fn"] += class_fn
    if class_totals != {"tp": tp, "fp": fp, "fn": fn}:
        raise TrainingCVAggregationError(f"{location} per-class counts do not sum to overall")


def _validate_dataset_binding(
    report: Mapping[str, Any],
    *,
    fold: FoldContract,
    contract: ProtocolContract,
    reader: _EvidenceReader,
    location: str,
) -> tuple[Mapping[str, Any], BoundFile, BoundFile]:
    bindings = _object(report.get("bindings"), f"{location}.bindings")
    if set(bindings) != {"dataset", "candidate", "fold"}:
        raise TrainingCVAggregationError(f"{location}.bindings contract differs")
    raw = _object(bindings.get("dataset"), f"{location}.bindings.dataset")
    if set(raw) != {
        "root",
        "manifest",
        "dataset_yaml",
        "managed_files_sha256",
        "managed_file_count",
        "taxonomy",
    }:
        raise TrainingCVAggregationError(f"{location}.bindings.dataset contract differs")
    root_relative = _safe_relative(raw.get("root"), f"{location}.dataset.root")
    root_path, _ = _scoped_path(
        reader.workspace / Path(*root_relative.parts),
        workspace=reader.workspace,
        scopes=(("data", "training"),),
        location=f"{location} dataset root",
        leaf_kind="directory",
    )
    manifest_record = _object(raw.get("manifest"), f"{location}.dataset.manifest")
    if (
        set(manifest_record) != {"path", "sha256", "size_bytes", "schema"}
        or manifest_record.get("path") != "manifest.json"
    ):
        raise TrainingCVAggregationError(f"{location} dataset manifest record differs")
    manifest, manifest_file = reader.json(
        root_path / "manifest.json",
        scopes=(("data", "training"),),
        location=f"{location} dataset manifest",
    )
    if (
        manifest_file.sha256 != _sha256(manifest_record.get("sha256"), f"{location} manifest sha")
        or manifest_file.size_bytes
        != _integer(manifest_record.get("size_bytes"), f"{location} manifest size", minimum=1)
        or manifest_record.get("schema") != DATASET_SCHEMA
        or manifest.get("schema") != DATASET_SCHEMA
    ):
        raise TrainingCVAggregationError(f"{location} dataset manifest binding differs")
    _require_passed_gate(manifest.get("gate"), f"{location} dataset manifest.gate")
    if raw.get("taxonomy") != contract.taxonomy or manifest.get("taxonomy") != contract.taxonomy:
        raise TrainingCVAggregationError(f"{location} dataset taxonomy differs from protocol")
    yaml_record = _object(raw.get("dataset_yaml"), f"{location}.dataset.dataset_yaml")
    if (
        set(yaml_record) != {"path", "sha256", "size_bytes"}
        or yaml_record.get("path") != "dataset.yaml"
    ):
        raise TrainingCVAggregationError(f"{location} dataset.yaml record differs")
    _, yaml_file = reader.read(
        root_path / "dataset.yaml",
        scopes=(("data", "training"),),
        location=f"{location} dataset.yaml",
        maximum_bytes=4 * 1024 * 1024,
    )
    if yaml_file.sha256 != _sha256(
        yaml_record.get("sha256"), f"{location} dataset yaml sha"
    ) or yaml_file.size_bytes != _integer(
        yaml_record.get("size_bytes"), f"{location} dataset yaml size", minimum=1
    ):
        raise TrainingCVAggregationError(f"{location} dataset.yaml binding differs")

    manifest_inputs = _object(manifest.get("inputs"), f"{location} dataset manifest.inputs")
    if set(manifest_inputs) != {"coco", "reference_manifest", "split_plan"}:
        raise TrainingCVAggregationError(f"{location} dataset input contract differs")
    dataset_reference = _object(
        manifest_inputs.get("reference_manifest"), f"{location} dataset reference binding"
    )
    if dataset_reference != {
        "file_name": Path(str(contract.training_reference_binding["path"])).name,
        "sha256": contract.training_reference_binding["sha256"],
        "schema": REFERENCE_SCHEMA,
    }:
        raise TrainingCVAggregationError(
            f"{location} dataset binds another training-reference manifest"
        )
    dataset_coco = _object(manifest_inputs.get("coco"), f"{location} dataset COCO binding")
    if set(dataset_coco) != {"file_name", "sha256", "semantic_sha256"} or (
        dataset_coco.get("file_name")
        != Path(str(contract.training_annotations_binding["path"])).name
        or dataset_coco.get("sha256") != contract.training_annotations_binding["sha256"]
    ):
        raise TrainingCVAggregationError(f"{location} dataset binds another training COCO file")
    _sha256(dataset_coco.get("semantic_sha256"), f"{location} dataset COCO semantic sha256")

    files = _list(manifest.get("files"), f"{location} dataset manifest.files")
    canonical_records: list[dict[str, Any]] = []
    managed_by_path: dict[str, dict[str, Any]] = {}
    label_bytes: dict[str, bytes] = {}
    seen: set[str] = set()
    for index, value in enumerate(files):
        record = _object(value, f"{location} dataset file[{index}]")
        if set(record) != {"path", "sha256", "size_bytes"}:
            raise TrainingCVAggregationError(f"{location} dataset file[{index}] contract differs")
        relative = _safe_relative(record.get("path"), f"{location} dataset file[{index}].path")
        name = relative.as_posix()
        if name in seen or name == "manifest.json":
            raise TrainingCVAggregationError(f"{location} dataset has duplicate managed path")
        seen.add(name)
        expected_sha = _sha256(record.get("sha256"), f"{location} dataset file[{index}].sha")
        expected_size = _integer(record.get("size_bytes"), f"{location} dataset file[{index}].size")
        is_label = (
            len(relative.parts) == 3
            and relative.parts[0] == "labels"
            and relative.parts[1] in {"train", "val"}
        )
        retain = is_label
        encoded, managed = reader.read(
            root_path / Path(*relative.parts),
            scopes=(("data", "training"),),
            location=f"{location} managed dataset file {name}",
            maximum_bytes=(
                4 * 1024 * 1024
                if name == "dataset.yaml"
                else 16 * 1024 * 1024
                if is_label
                else MAX_MANAGED_BYTES
            ),
            retain=retain,
        )
        if managed.sha256 != expected_sha or managed.size_bytes != expected_size:
            raise TrainingCVAggregationError(f"{location} managed file {name!r} differs")
        canonical_record = {"path": name, "sha256": expected_sha, "size_bytes": expected_size}
        canonical_records.append(canonical_record)
        managed_by_path[name] = canonical_record
        if retain:
            label_bytes[name] = encoded
    canonical_records.sort(key=lambda item: item["path"])
    if "dataset.yaml" not in seen:
        raise TrainingCVAggregationError(f"{location} dataset.yaml is not managed")
    if _semantic_sha256(canonical_records) != _sha256(
        raw.get("managed_files_sha256"), f"{location} managed files sha"
    ) or len(canonical_records) != _integer(
        raw.get("managed_file_count"), f"{location} managed file count", minimum=1
    ):
        raise TrainingCVAggregationError(f"{location} managed-file binding differs")

    split = _object(manifest.get("split"), f"{location} dataset manifest.split")
    if (
        set(split) != {"method", "train_asset_ids", "val_asset_ids"}
        or split.get("method")
        != "explicit typed source_asset_id plan constrained by source leakage group"
    ):
        raise TrainingCVAggregationError(f"{location} dataset split contract differs")
    manifest_train_assets = tuple(
        _asset_id(value, f"{location} dataset train_asset_ids[{index}]")
        for index, value in enumerate(
            _list(split.get("train_asset_ids"), f"{location} dataset train assets")
        )
    )
    manifest_val_assets = tuple(
        _asset_id(value, f"{location} dataset val_asset_ids[{index}]")
        for index, value in enumerate(
            _list(split.get("val_asset_ids"), f"{location} dataset val assets")
        )
    )
    if tuple(_asset_identity(value) for value in manifest_train_assets) != tuple(
        _asset_identity(value) for value in fold.train_asset_ids
    ) or tuple(_asset_identity(value) for value in manifest_val_assets) != tuple(
        _asset_identity(value) for value in fold.val_asset_ids
    ):
        raise TrainingCVAggregationError(f"{location} dataset train/validation assets differ")
    source_split = _object(manifest_inputs.get("split_plan"), f"{location} dataset split_plan")
    expected_split = {
        "file_name": fold.split_file_name,
        "sha256": fold.split_sha256,
        "semantic_sha256": fold.split_semantic_sha256,
        "schema": SPLIT_PLAN_SCHEMA,
    }
    if source_split != expected_split:
        raise TrainingCVAggregationError(f"{location} dataset split-plan binding differs")
    if manifest.get("fold_id") is not None and manifest.get("fold_id") != fold.fold_id:
        raise TrainingCVAggregationError(f"{location} dataset fold_id differs")
    dataset_counts = _object(manifest.get("counts"), f"{location} dataset counts")
    image_counts = _object(
        dataset_counts.get("images"),
        f"{location} dataset image counts",
    )
    annotation_counts = _object(
        dataset_counts.get("annotations"),
        f"{location} dataset annotation counts",
    )
    asset_counts = _object(dataset_counts.get("assets"), f"{location} dataset asset counts")
    if image_counts != {
        "total": fold.train_frame_count + fold.frame_count,
        "train": fold.train_frame_count,
        "val": fold.frame_count,
        "zero_annotations": (
            fold.train_zero_annotation_frame_count + fold.zero_annotation_frame_count
        ),
    } or asset_counts != {
        "total": len(fold.train_asset_ids) + len(fold.val_asset_ids),
        "train": len(fold.train_asset_ids),
        "val": len(fold.val_asset_ids),
    }:
        raise TrainingCVAggregationError(f"{location} dataset image/asset counts differ")
    if (
        annotation_counts.get("total") != fold.train_annotation_count + fold.annotation_count
        or annotation_counts.get("train") != fold.train_annotation_count
        or annotation_counts.get("val") != fold.annotation_count
    ):
        raise TrainingCVAggregationError(f"{location} dataset fold counts differ")
    by_split = _object(annotation_counts.get("by_split_and_category"), f"{location} class counts")
    if (
        set(by_split) != {"train", "val"}
        or _object(by_split.get("train"), f"{location} train class counts")
        != dict(fold.train_class_support)
        or _object(by_split.get("val"), f"{location} val class counts") != dict(fold.class_support)
        or _object(annotation_counts.get("by_category"), f"{location} total class counts")
        != {
            label: fold.train_class_support[label] + fold.class_support[label]
            for label in CANONICAL_NAMES
        }
    ):
        raise TrainingCVAggregationError(f"{location} dataset train/val class counts differ")

    images_by_split: dict[str, dict[str, dict[str, Any]]] = {"train": {}, "val": {}}
    labels_by_split: dict[str, dict[str, dict[str, Any]]] = {"train": {}, "val": {}}
    for name, record in managed_by_path.items():
        parts = PurePosixPath(name).parts
        if (
            len(parts) != 3
            or parts[1] not in {"train", "val"}
            or parts[0] not in {"images", "labels"}
        ):
            continue
        stem = PurePosixPath(name).stem.casefold()
        target = images_by_split[parts[1]] if parts[0] == "images" else labels_by_split[parts[1]]
        if stem in target:
            raise TrainingCVAggregationError(
                f"{location} dataset {parts[1]} inventory has duplicate stem {stem!r}"
            )
        target[stem] = record
    expected_inventory = {
        "train": (
            fold.train_frame_count,
            fold.train_zero_annotation_frame_count,
            fold.train_class_support,
        ),
        "val": (fold.frame_count, fold.zero_annotation_frame_count, fold.class_support),
    }
    for split_name, (
        expected_frames,
        expected_zero_frames,
        expected_support,
    ) in expected_inventory.items():
        split_images = images_by_split[split_name]
        split_labels = labels_by_split[split_name]
        if (
            not split_images
            or set(split_images) != set(split_labels)
            or len(split_images) != expected_frames
        ):
            raise TrainingCVAggregationError(
                f"{location} dataset {split_name} image/label inventory differs from the fold"
            )
        observed_support = {label: 0 for label in CANONICAL_NAMES}
        observed_zero_frames = 0
        for label_record in split_labels.values():
            raw_label = label_bytes[label_record["path"]]
            try:
                lines = [line for line in raw_label.decode("utf-8").splitlines() if line.strip()]
            except UnicodeDecodeError as error:
                raise TrainingCVAggregationError(
                    f"{location} {split_name} label {label_record['path']!r} is not UTF-8"
                ) from error
            if not lines:
                observed_zero_frames += 1
            for line_number, line in enumerate(lines, start=1):
                tokens = line.split()
                if len(tokens) != 5 or not re.fullmatch(r"[0-7]", tokens[0]):
                    raise TrainingCVAggregationError(
                        f"{location} {split_name} label {label_record['path']!r} "
                        f"line {line_number} is not a canonical YOLO row"
                    )
                for coordinate_index, raw_coordinate in enumerate(tokens[1:], start=1):
                    try:
                        parsed_coordinate = float(raw_coordinate)
                    except ValueError as error:
                        raise TrainingCVAggregationError(
                            f"{location} {split_name} label {label_record['path']!r} "
                            f"line {line_number} has a non-numeric coordinate"
                        ) from error
                    coordinate = _number(
                        parsed_coordinate,
                        (
                            f"{location} {split_name} label {label_record['path']!r} "
                            f"line {line_number} coordinate {coordinate_index}"
                        ),
                        minimum=0,
                        maximum=1,
                    )
                    if coordinate_index in {3, 4} and coordinate <= 0:
                        raise TrainingCVAggregationError(
                            f"{location} {split_name} label {label_record['path']!r} "
                            f"line {line_number} has a non-positive box dimension"
                        )
                observed_support[CANONICAL_NAMES[int(tokens[0])]] += 1
        if observed_zero_frames != expected_zero_frames or observed_support != dict(
            expected_support
        ):
            raise TrainingCVAggregationError(
                f"{location} {split_name} label inventory differs from fold class/zero counts"
            )
    frame_records = []
    for stem, image_record in sorted(
        images_by_split["val"].items(),
        key=lambda item: (item[1]["path"].casefold(), item[1]["path"]),
    ):
        label_record = labels_by_split["val"][stem]
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
    computed_frames_sha256 = _semantic_sha256(frame_records)
    report_frames_sha256 = _object(report.get("val_source"), f"{location}.val_source").get(
        "frames_sha256"
    )
    if computed_frames_sha256 != report_frames_sha256:
        raise TrainingCVAggregationError(
            f"{location} validation frames digest differs from managed v3 inventory"
        )
    return manifest, manifest_file, yaml_file


def _parse_args_yaml(encoded: bytes, location: str) -> dict[str, Any]:
    try:
        payload = yaml.load(encoded.decode("utf-8"), Loader=_UniqueKeyLoader)
    except (UnicodeDecodeError, yaml.YAMLError, TrainingCVAggregationError) as error:
        raise TrainingCVAggregationError(f"could not decode {location}: {error}") from error
    return _object(payload, location)


def _validate_args_artifact(
    encoded: bytes,
    *,
    resolved: Mapping[str, Any],
    workspace: Path,
    candidate_name: str,
    location: str,
) -> None:
    args = _parse_args_yaml(encoded, location)
    expected = {
        key: (False if key == "cache" and value == "none" else value)
        for key, value in resolved.items()
        if key != "model_family"
    }
    for key, value in expected.items():
        if not _same_json(args.get(key), value):
            raise TrainingCVAggregationError(
                f"{location} differs from receipt.resolved_args at {key!r}"
            )
    fixed = {
        "task": "detect",
        "mode": "train",
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
    for key, expected_value in fixed.items():
        if not _same_json(args.get(key), expected_value):
            raise TrainingCVAggregationError(
                f"{location} violates the immutable trainer argument {key!r}"
            )
    staging_roots: set[tuple[str, ...]] = set()
    for key, required_suffix in (
        ("model", ("inputs", "yolo11n.pt")),
        ("data", ("dataset", "dataset.yaml")),
        ("project", ("trainer-output",)),
    ):
        raw = _text(args.get(key), f"{location}.{key}")
        if "\\" in raw or not Path(raw).is_absolute():
            raise TrainingCVAggregationError(
                f"{location}.{key} must name the isolated absolute training workspace path"
            )
        lexical = Path(os.path.abspath(raw))
        relative = _inside_workspace(lexical, workspace, f"{location}.{key}")
        if (
            relative.parts[:2] != ("data", "model-candidates")
            or len(relative.parts) != 3 + len(required_suffix)
            or tuple(relative.parts[-len(required_suffix) :]) != required_suffix
        ):
            raise TrainingCVAggregationError(
                f"{location}.{key} did not use the isolated training workspace"
            )
        staging_name = relative.parts[2]
        required_prefix = f".{candidate_name}.workspace-"
        if not staging_name.startswith(required_prefix) or staging_name == required_prefix:
            raise TrainingCVAggregationError(
                f"{location}.{key} did not use the candidate's isolated training workspace"
            )
        staging_roots.add(relative.parts[:3])
    if len(staging_roots) != 1:
        raise TrainingCVAggregationError(
            f"{location} model, data and project paths name different isolated workspaces"
        )
    for key, value in args.items():
        if isinstance(value, numbers.Real) and not isinstance(value, bool):
            _number(value, f"{location}.{key}")
        if isinstance(value, str) and Path(value).is_absolute():
            _inside_workspace(
                Path(os.path.abspath(value)), workspace, f"{location} absolute argument {key!r}"
            )


def _parse_results_csv(encoded: bytes, location: str) -> tuple[dict[str, Any], dict[str, float]]:
    try:
        lines = encoded.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise TrainingCVAggregationError(f"{location} is not UTF-8") from error
    reader = csv.reader(lines)
    try:
        raw_header = next(reader)
    except StopIteration as error:
        raise TrainingCVAggregationError(f"{location} is empty") from error
    header = [item.strip() for item in raw_header]
    if (
        not header
        or len(header) != len(set(header))
        or "epoch" not in header
        or not set(RESULT_METRIC_COLUMNS).issubset(header)
    ):
        raise TrainingCVAggregationError(f"{location} has an invalid metric header")
    rows: list[dict[str, float]] = []
    for line_number, values in enumerate(reader, start=2):
        if len(values) != len(header):
            raise TrainingCVAggregationError(f"{location} row {line_number} has the wrong width")
        row: dict[str, float] = {}
        for key, raw_value in zip(header, values, strict=True):
            try:
                value = float(raw_value.strip())
            except ValueError as error:
                raise TrainingCVAggregationError(
                    f"{location} row {line_number} column {key!r} is not numeric"
                ) from error
            if not math.isfinite(value):
                raise TrainingCVAggregationError(
                    f"{location} row {line_number} column {key!r} is non-finite"
                )
            row[key] = value
        if not row["epoch"].is_integer() or row["epoch"] < 0:
            raise TrainingCVAggregationError(f"{location} row {line_number} has an invalid epoch")
        for key in RESULT_METRIC_COLUMNS:
            if not 0 <= row[key] <= 1:
                raise TrainingCVAggregationError(
                    f"{location} row {line_number} metric {key!r} is outside [0, 1]"
                )
        rows.append(row)
    if not rows:
        raise TrainingCVAggregationError(f"{location} contains no epochs")
    epochs = [int(row["epoch"]) for row in rows]
    first_epoch = epochs[0]
    if first_epoch not in {0, 1} or epochs != list(range(first_epoch, first_epoch + len(rows))):
        raise TrainingCVAggregationError(f"{location} epochs are not contiguous")
    best = max(rows, key=lambda row: (row["metrics/mAP50-95(B)"], row["epoch"]))
    best_epoch = {
        "index": int(best["epoch"]) - first_epoch,
        "number": int(best["epoch"]) - first_epoch + 1,
        "selection_fitness": best["metrics/mAP50-95(B)"],
        "epochs_recorded": len(rows),
    }
    best_metrics = {key: best[key] for key in RESULT_METRIC_COLUMNS}
    return best_epoch, best_metrics


def _validate_candidate_metrics(
    value: Any,
    *,
    fold: FoldContract,
    location: str,
) -> Mapping[str, float]:
    metrics = _object(value, location)
    if set(metrics) != {"aggregate", "per_class"}:
        raise TrainingCVAggregationError(f"{location} contract differs")
    aggregate = _object(metrics.get("aggregate"), f"{location}.aggregate")
    if set(aggregate) != {"precision", "recall", "map50", "map50_95", "fitness"}:
        raise TrainingCVAggregationError(f"{location}.aggregate contract differs")
    validated = {
        key: _number(aggregate.get(key), f"{location}.aggregate.{key}", minimum=0, maximum=1)
        for key in ("precision", "recall", "map50", "map50_95", "fitness")
    }
    if not math.isclose(validated["fitness"], validated["map50_95"], rel_tol=0, abs_tol=1e-12):
        raise TrainingCVAggregationError(f"{location}.aggregate fitness differs from mAP50-95")
    per_class = _object(metrics.get("per_class"), f"{location}.per_class")
    if set(per_class) != set(CANONICAL_NAMES):
        raise TrainingCVAggregationError(f"{location}.per_class taxonomy differs")
    evaluable_values: dict[str, list[float]] = {
        metric: [] for metric in ("precision", "recall", "map50", "map50_95")
    }
    for class_id, label in enumerate(CANONICAL_NAMES):
        record = _object(per_class[label], f"{location}.per_class.{label}")
        expected_keys = {
            "class_id",
            "status",
            "support_count",
            "precision",
            "recall",
            "map50",
            "map50_95",
        }
        if set(record) != expected_keys or record.get("class_id") != class_id:
            raise TrainingCVAggregationError(f"{location}.per_class.{label} contract differs")
        support = _integer(
            record.get("support_count"), f"{location}.per_class.{label}.support_count"
        )
        if support != fold.class_support[label]:
            raise TrainingCVAggregationError(
                f"{location}.per_class.{label} support differs from the LOSO fold"
            )
        if support == 0:
            if record.get("status") != "not_evaluable" or any(
                record.get(key) is not None for key in ("precision", "recall", "map50", "map50_95")
            ):
                raise TrainingCVAggregationError(
                    f"{location}.per_class.{label} must be not_evaluable/null"
                )
        else:
            if record.get("status") != "evaluable":
                raise TrainingCVAggregationError(f"{location}.per_class.{label} must be evaluable")
            for key in ("precision", "recall", "map50", "map50_95"):
                evaluable_values[key].append(
                    _number(
                        record.get(key),
                        f"{location}.per_class.{label}.{key}",
                        minimum=0,
                        maximum=1,
                    )
                )

    for metric, values in evaluable_values.items():
        if not values:
            raise TrainingCVAggregationError(
                f"{location} has no evaluable per-class {metric} values"
            )
        expected = math.fsum(values) / len(values)
        if not math.isclose(validated[metric], expected, rel_tol=0, abs_tol=1e-12):
            raise TrainingCVAggregationError(
                f"{location}.aggregate.{metric} does not equal the non-weighted "
                "evaluable per-class mean"
            )

    # Ultralytics writes results.csv during training, then reloads and
    # revalidates best.pt to produce the receipt metrics.  On Apple MPS those
    # are distinct observations because some kernels are non-deterministic.
    # The CSV is validated separately as the exact best-epoch/selection-fitness
    # authority; the checkpoint revalidation is bound here by its complete
    # per-class evidence and by the evaluation report's receipt/weight hashes.
    return validated


def _validate_pretrained_transfer(
    value: Any,
    *,
    contract: ProtocolContract,
    location: str,
) -> None:
    evidence = _object(value, location)
    if (
        set(evidence)
        != {
            "schema",
            "source_model",
            "target",
            "matched_rows",
            "matched_row_count",
            "target_row_count",
            "runtime_observation",
        }
        or evidence.get("schema") != PRETRAINED_TRANSFER_SCHEMA
    ):
        raise TrainingCVAggregationError(f"{location} contract differs")

    source = _object(evidence.get("source_model"), f"{location}.source_model")
    if set(source) != {
        "family",
        "base_weights_sha256",
        "class_count",
        "names_sha256",
    } or (
        source.get("family") != "YOLO11n"
        or source.get("base_weights_sha256") != contract.base_weights_binding["sha256"]
        or _integer(source.get("class_count"), f"{location}.source_model.class_count") != 80
    ):
        raise TrainingCVAggregationError(
            f"{location} source model differs from the frozen YOLO11n base weights"
        )
    _sha256(source.get("names_sha256"), f"{location}.source_model.names_sha256")

    target = _object(evidence.get("target"), f"{location}.target")
    expected_model_names = list(contract.taxonomy["model_names"])
    expected_canonical_names = list(contract.taxonomy["canonical_names"])
    if target != {
        "class_count": len(CANONICAL_NAMES),
        "model_names": expected_model_names,
        "canonical_names": expected_canonical_names,
    }:
        raise TrainingCVAggregationError(f"{location} target taxonomy differs")

    expected_rows = []
    for target_id, (model_name, canonical_name) in enumerate(
        zip(expected_model_names, expected_canonical_names, strict=True)
    ):
        source_id = COCO_SOURCE_CLASS_IDS.get(model_name)
        if source_id is None:
            raise TrainingCVAggregationError(
                f"{location} target model class {model_name!r} has no frozen COCO source row"
            )
        expected_rows.append(
            {
                "target_id": target_id,
                "target_model_name": model_name,
                "canonical_name": canonical_name,
                "source_id": source_id,
                "source_model_name": model_name,
            }
        )
    if (
        not _same_json(
            _list(evidence.get("matched_rows"), f"{location}.matched_rows"), expected_rows
        )
        or _integer(evidence.get("matched_row_count"), f"{location}.matched_row_count")
        != len(expected_rows)
        or _integer(evidence.get("target_row_count"), f"{location}.target_row_count")
        != len(expected_rows)
    ):
        raise TrainingCVAggregationError(f"{location} does not prove an exact 8/8 row transfer")

    runtime = _object(evidence.get("runtime_observation"), f"{location}.runtime_observation")
    expected_message_digest = hashlib.sha256(
        PRETRAINED_TRANSFER_MESSAGE.encode("utf-8")
    ).hexdigest()
    if runtime != {
        "verification_mode": "ultralytics_logger",
        "message": PRETRAINED_TRANSFER_MESSAGE,
        "message_sha256": expected_message_digest,
        "matched_row_count": len(expected_rows),
        "target_row_count": len(expected_rows),
    }:
        raise TrainingCVAggregationError(
            f"{location} lacks the required Ultralytics runtime transfer observation"
        )


def _validate_candidate_binding(
    report: Mapping[str, Any],
    *,
    experiment: tuple[str, int, str],
    dataset_manifest: Mapping[str, Any],
    dataset_file: BoundFile,
    dataset_yaml: BoundFile,
    fold: FoldContract,
    contract: ProtocolContract,
    reader: _EvidenceReader,
    location: str,
) -> None:
    arm_id, seed, _fold_id = experiment
    candidate_binding = _object(
        _object(report.get("bindings"), f"{location}.bindings").get("candidate"),
        f"{location}.bindings.candidate",
    )
    if set(candidate_binding) != {"receipt", "weight", "protocol_sha256"}:
        raise TrainingCVAggregationError(f"{location} candidate binding contract differs")
    receipt_record = _object(candidate_binding.get("receipt"), f"{location} candidate receipt")
    if set(receipt_record) != {"path", "sha256", "size_bytes", "schema"}:
        raise TrainingCVAggregationError(f"{location} candidate receipt record differs")
    receipt_relative = _safe_relative(receipt_record.get("path"), f"{location} receipt path")
    receipt, receipt_file = reader.json(
        reader.workspace / Path(*receipt_relative.parts),
        scopes=(("data", "model-candidates"),),
        location=f"{location} candidate receipt",
    )
    if (
        receipt_file.path.name != "receipt.json"
        or receipt_file.sha256 != _sha256(receipt_record.get("sha256"), f"{location} receipt sha")
        or receipt_file.size_bytes
        != _integer(receipt_record.get("size_bytes"), f"{location} receipt size", minimum=1)
        or receipt_record.get("schema") != CANDIDATE_SCHEMA
        or receipt.get("schema") != CANDIDATE_SCHEMA
    ):
        raise TrainingCVAggregationError(f"{location} candidate receipt binding differs")
    _scan_declared_paths(receipt, f"{location} candidate receipt")
    if set(receipt) != RECEIPT_KEYS:
        raise TrainingCVAggregationError(f"{location} candidate receipt top-level contract differs")
    receipt_gate = _object(receipt.get("gate"), f"{location} candidate receipt.gate")
    if set(receipt_gate) != {"passed", "checks"}:
        raise TrainingCVAggregationError(f"{location} candidate receipt gate contract differs")
    receipt_checks = _object(
        receipt_gate.get("checks"), f"{location} candidate receipt.gate.checks"
    )
    if (
        receipt_gate.get("passed") is not True
        or set(receipt_checks) != CURRENT_CANDIDATE_GATE_CHECKS
        or any(value is not True for value in receipt_checks.values())
    ):
        raise TrainingCVAggregationError(
            f"{location} candidate receipt gate is incomplete or unsupported"
        )
    if receipt.get("mutation_performed") is not True:
        raise TrainingCVAggregationError(f"{location} candidate was not a completed training run")
    holdout = _object(receipt.get("holdout"), f"{location} candidate holdout")
    if (
        set(holdout) != {"input_read", "statement"}
        or holdout.get("input_read") is not False
        or holdout.get("statement") != NO_FINAL_HOLDOUT_STATEMENT
    ):
        raise TrainingCVAggregationError(f"{location} candidate crossed the holdout firewall")
    expected_args = {
        "model_family": "YOLO11n",
        **dict(contract.common_training),
        **{key: value for key, value in contract.arms[arm_id].items() if key != "arm_id"},
        "seed": seed,
    }
    if not _same_json(
        _object(receipt.get("resolved_args"), f"{location} resolved_args"), expected_args
    ):
        raise TrainingCVAggregationError(
            f"{location} candidate args drifted from Recovery protocol"
        )
    protocol = _object(receipt.get("protocol"), f"{location} candidate protocol")
    protocol_sha = _sha256(receipt.get("protocol_sha256"), f"{location} candidate protocol sha")
    if (
        set(protocol)
        != {
            "schema",
            "model_family",
            "taxonomy",
            "training",
            "validation_selection",
            "holdout_access",
        }
        or protocol.get("schema") != CANDIDATE_PROTOCOL_SCHEMA
        or protocol.get("model_family") != "YOLO11n"
        or not _same_json(protocol.get("taxonomy"), contract.taxonomy)
        or not _same_json(
            protocol.get("training"),
            {
                key: value
                for key, value in expected_args.items()
                if key not in {"model_family", "seed"}
            },
        )
        or protocol.get("holdout_access") != "prohibited"
        or _semantic_sha256(protocol) != protocol_sha
        or candidate_binding.get("protocol_sha256") != protocol_sha
    ):
        raise TrainingCVAggregationError(f"{location} candidate protocol binding differs")
    if protocol.get("validation_selection") != {
        "primary": "mAP50-95",
        "tie_breakers": ["mAP50", "recall", "precision", "smaller_seed"],
    }:
        raise TrainingCVAggregationError(f"{location} candidate validation selection differs")

    receipt_inputs = _object(receipt.get("inputs"), f"{location} candidate inputs")
    if set(receipt_inputs) != {"dataset", "base_weights"}:
        raise TrainingCVAggregationError(f"{location} candidate inputs contract differs")
    receipt_dataset = _object(
        receipt_inputs.get("dataset"),
        f"{location} candidate dataset binding",
    )
    if set(receipt_dataset) != {
        "manifest",
        "dataset_yaml",
        "managed_files_sha256",
        "managed_file_count",
        "counts",
        "taxonomy",
    }:
        raise TrainingCVAggregationError(f"{location} candidate dataset contract differs")
    manifest_binding = _object(receipt_dataset.get("manifest"), f"{location} receipt manifest")
    yaml_binding = _object(receipt_dataset.get("dataset_yaml"), f"{location} receipt dataset yaml")
    report_dataset = _object(
        _object(report.get("bindings"), f"{location}.bindings").get("dataset"),
        f"{location} report dataset",
    )
    if (
        manifest_binding
        != {
            "schema": DATASET_SCHEMA,
            "sha256": dataset_file.sha256,
            "size_bytes": dataset_file.size_bytes,
        }
        or yaml_binding != {"sha256": dataset_yaml.sha256, "size_bytes": dataset_yaml.size_bytes}
        or receipt_dataset.get("managed_files_sha256") != report_dataset.get("managed_files_sha256")
        or receipt_dataset.get("managed_file_count") != report_dataset.get("managed_file_count")
        or not _same_json(receipt_dataset.get("counts"), dataset_manifest.get("counts"))
        or not _same_json(receipt_dataset.get("taxonomy"), contract.taxonomy)
    ):
        raise TrainingCVAggregationError(f"{location} candidate dataset binding differs")

    receipt_base = _object(receipt_inputs.get("base_weights"), f"{location} candidate base_weights")
    if receipt_base != {
        "file_name": Path(str(contract.base_weights_binding["path"])).name,
        "model_family": "YOLO11n",
        "sha256": contract.base_weights_binding["sha256"],
        "size_bytes": contract.base_weights_binding["size_bytes"],
    }:
        raise TrainingCVAggregationError(
            f"{location} candidate base weights differ from the Recovery protocol"
        )
    _validate_pretrained_transfer(
        receipt.get("pretrained_transfer"),
        contract=contract,
        location=f"{location} candidate pretrained_transfer",
    )

    weight_record = _object(candidate_binding.get("weight"), f"{location} candidate weight")
    if (
        set(weight_record) != {"path", "sha256", "size_bytes", "artifact"}
        or weight_record.get("artifact") != "best_weights"
    ):
        raise TrainingCVAggregationError(f"{location} candidate weight record differs")
    weight_relative = _safe_relative(weight_record.get("path"), f"{location} candidate weight path")
    artifacts = _object(receipt.get("artifacts"), f"{location} candidate artifacts")
    if set(artifacts) != set(ARTIFACT_PATHS):
        raise TrainingCVAggregationError(f"{location} candidate artifact contract differs")
    artifact_files: dict[str, BoundFile] = {}
    artifact_bytes: dict[str, bytes] = {}
    for artifact_name, required_path in ARTIFACT_PATHS.items():
        artifact = _object(
            artifacts.get(artifact_name), f"{location} candidate artifacts.{artifact_name}"
        )
        if set(artifact) != {"path", "sha256", "size_bytes"}:
            raise TrainingCVAggregationError(
                f"{location} candidate artifact {artifact_name!r} record differs"
            )
        relative = _safe_relative(
            artifact.get("path"), f"{location} candidate artifact {artifact_name}.path"
        )
        if relative.as_posix() != required_path:
            raise TrainingCVAggregationError(
                f"{location} candidate artifact {artifact_name!r} path differs"
            )
        retain = artifact_name in {"args", "results"}
        encoded, bound = reader.read(
            receipt_file.path.parent / Path(*relative.parts),
            scopes=(("data", "model-candidates"),),
            location=f"{location} candidate artifact {artifact_name}",
            maximum_bytes=MAX_MANAGED_BYTES,
            retain=retain,
        )
        if bound.sha256 != _sha256(
            artifact.get("sha256"), f"{location} artifact {artifact_name} sha256"
        ) or bound.size_bytes != _integer(
            artifact.get("size_bytes"),
            f"{location} artifact {artifact_name} size_bytes",
            minimum=1,
        ):
            raise TrainingCVAggregationError(
                f"{location} candidate artifact {artifact_name!r} hash/size differs"
            )
        artifact_files[artifact_name] = bound
        if retain:
            artifact_bytes[artifact_name] = encoded

    weight_file = artifact_files["best_weights"]
    if (
        reader.workspace / Path(*weight_relative.parts) != weight_file.path
        or weight_file.sha256 != _sha256(weight_record.get("sha256"), f"{location} weight sha")
        or weight_file.size_bytes
        != _integer(weight_record.get("size_bytes"), f"{location} weight size", minimum=1)
    ):
        raise TrainingCVAggregationError(f"{location} candidate best-weight binding differs")

    _validate_args_artifact(
        artifact_bytes["args"],
        resolved=expected_args,
        workspace=reader.workspace,
        candidate_name=receipt_file.path.parent.name,
        location=f"{location} candidate args.yaml",
    )
    expected_best_epoch, _ = _parse_results_csv(
        artifact_bytes["results"], f"{location} candidate results.csv"
    )
    best_epoch = _object(receipt.get("best_epoch"), f"{location} candidate best_epoch")
    if best_epoch != expected_best_epoch or best_epoch["epochs_recorded"] > expected_args["epochs"]:
        raise TrainingCVAggregationError(
            f"{location} candidate best_epoch differs from results.csv"
        )
    receipt_metrics = _validate_candidate_metrics(
        receipt.get("metrics"),
        fold=fold,
        location=f"{location} receipt metrics",
    )
    report_metrics = _object(report.get("metrics"), f"{location} report metrics")
    for key in ("map50", "map50_95"):
        actual = _number(report_metrics.get(key), f"{location} report {key}", minimum=0, maximum=1)
        expected = _number(
            receipt_metrics.get(key), f"{location} receipt {key}", minimum=0, maximum=1
        )
        if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise TrainingCVAggregationError(f"{location} report {key} differs from candidate")
    timestamps = _object(receipt.get("timestamps"), f"{location} receipt timestamps")
    if set(timestamps) != {"started_at", "finished_at", "duration_seconds"}:
        raise TrainingCVAggregationError(f"{location} candidate timestamps contract differs")
    _text(timestamps.get("started_at"), f"{location} candidate started_at")
    _text(timestamps.get("finished_at"), f"{location} candidate finished_at")
    duration = _number(
        timestamps.get("duration_seconds"),
        f"{location} receipt training duration",
        minimum=0,
    )
    environment = _object(receipt.get("environment"), f"{location} candidate environment")
    if set(environment) != {"python", "platform", "packages", "git"}:
        raise TrainingCVAggregationError(f"{location} candidate environment contract differs")
    packages = _object(environment.get("packages"), f"{location} candidate packages")
    if packages.get("ultralytics") != "8.4.135":
        raise TrainingCVAggregationError(f"{location} candidate does not bind Ultralytics 8.4.135")
    compute = _object(report.get("compute"), f"{location} compute")
    observed_duration = _number(
        compute.get("training_duration_seconds"), f"{location} compute training duration", minimum=0
    )
    if not math.isclose(duration, observed_duration, rel_tol=1e-12, abs_tol=1e-12):
        raise TrainingCVAggregationError(f"{location} training duration differs from receipt")


def _validate_report(
    path: Path,
    *,
    contract: ProtocolContract,
    reader: _EvidenceReader,
    provenance: str,
) -> dict[str, Any]:
    payload, report_file = reader.json(path, scopes=REPORT_SCOPES, location="evaluation report")
    _scan_declared_paths(payload, "evaluation report")
    if set(payload) != EXPECTED_REPORT_KEYS or payload.get("schema") != EVALUATION_SCHEMA:
        raise TrainingCVAggregationError("evaluation report top-level contract differs")
    _require_passed_gate(payload.get("gate"), "evaluation report.gate")
    firewall = _object(payload.get("holdout_firewall"), "evaluation report.holdout_firewall")
    if (
        firewall.get("input_read") is not False
        or firewall.get("allowed_scopes") != ["data/training", "data/model-candidates"]
        or firewall.get("rejected_scopes") != list(FINAL_HOLDOUT_REJECTED_SCOPES)
    ):
        raise TrainingCVAggregationError("evaluation report holdout firewall differs")
    experiment = _object(payload.get("experiment"), "evaluation report.experiment")
    if set(experiment) != {"arm_id", "seed", "fold_id"}:
        raise TrainingCVAggregationError("evaluation report experiment contract differs")
    arm_id = _fold_identifier(experiment.get("arm_id"), "evaluation report arm_id")
    seed = _integer(experiment.get("seed"), "evaluation report seed")
    fold_id = _fold_identifier(experiment.get("fold_id"), "evaluation report fold_id")
    if arm_id not in contract.arms or fold_id not in contract.folds:
        raise TrainingCVAggregationError("evaluation report names an unregistered arm or fold")
    if contract.reanalysis is not None:
        expected_reanalysis_binding = contract.reanalysis_reports.get(report_file.relative)
        actual_reanalysis_binding = {
            **_binding(report_file, schema=EVALUATION_SCHEMA),
            "experiment": {"arm_id": arm_id, "seed": seed, "fold_id": fold_id},
        }
        if expected_reanalysis_binding != actual_reanalysis_binding:
            raise TrainingCVAggregationError(
                "evaluation report is not the exact file frozen for reanalysis"
            )
    fold = contract.folds[fold_id]

    settings = _object(payload.get("settings"), "evaluation report.settings")
    if set(settings) != SETTINGS_KEYS:
        raise TrainingCVAggregationError("evaluation report settings contract differs")
    expected_settings = {
        **contract.settings,
        "image_size": contract.arms[arm_id].get("imgsz"),
    }
    if settings != expected_settings:
        raise TrainingCVAggregationError("evaluation settings drifted from Recovery protocol")

    val_source = _object(payload.get("val_source"), "evaluation report.val_source")
    if set(val_source) != {
        "split",
        "asset_ids",
        "frame_count",
        "zero_annotation_frame_count",
        "annotation_count",
        "frames_sha256",
    }:
        raise TrainingCVAggregationError("evaluation report val_source contract differs")
    if (
        val_source.get("split") != "val"
        or val_source.get("asset_ids") != list(fold.val_asset_ids)
        or val_source.get("frame_count") != fold.frame_count
        or val_source.get("zero_annotation_frame_count") != fold.zero_annotation_frame_count
        or val_source.get("annotation_count") != fold.annotation_count
    ):
        raise TrainingCVAggregationError(
            "evaluation report validation source differs from LOSO plan"
        )
    _sha256(val_source.get("frames_sha256"), "evaluation report val_source.frames_sha256")

    fold_binding = _object(
        _object(payload.get("bindings"), "evaluation report.bindings").get("fold"),
        "evaluation report.bindings.fold",
    )
    if set(fold_binding) != {
        "binding_mode",
        "manifest_fold_id",
        "split_plan",
        "val_asset_ids",
    }:
        raise TrainingCVAggregationError("evaluation fold binding contract differs")
    if (
        fold_binding.get("binding_mode") != "dataset_manifest_split_plan_and_val_assets"
        or fold_binding.get("manifest_fold_id") not in (None, fold_id)
        or fold_binding.get("val_asset_ids") != list(fold.val_asset_ids)
        or _object(fold_binding.get("split_plan"), "evaluation fold split_plan")
        != {
            "file_name": fold.split_file_name,
            "sha256": fold.split_sha256,
            "semantic_sha256": fold.split_semantic_sha256,
            "schema": SPLIT_PLAN_SCHEMA,
        }
    ):
        raise TrainingCVAggregationError("evaluation fold binding differs from LOSO plan")

    _validate_report_metric_counts(payload, fold, "evaluation report")
    compute = _object(payload.get("compute"), "evaluation report.compute")
    if set(compute) != COMPUTE_KEYS or compute.get("predict_call_count") != 1:
        raise TrainingCVAggregationError("evaluation compute contract differs")
    for key in (
        "training_duration_seconds",
        "evaluation_inference_seconds",
        "model_load_seconds",
    ):
        _number(compute.get(key), f"evaluation report.compute.{key}", minimum=0)
    inference = float(compute["evaluation_inference_seconds"])
    expected_fps = fold.frame_count / inference if inference else None
    fps = compute.get("evaluated_frames_per_second")
    if expected_fps is None:
        if fps is not None:
            raise TrainingCVAggregationError("evaluation FPS must be null for zero inference time")
    else:
        actual_fps = _number(fps, "evaluation report evaluated_frames_per_second", minimum=0)
        if not math.isclose(actual_fps, expected_fps, rel_tol=1e-12, abs_tol=1e-12):
            raise TrainingCVAggregationError("evaluation FPS differs from frame count and duration")

    dataset, dataset_file, dataset_yaml = _validate_dataset_binding(
        payload,
        fold=fold,
        contract=contract,
        reader=reader,
        location="evaluation report",
    )
    _validate_candidate_binding(
        payload,
        experiment=(arm_id, seed, fold_id),
        dataset_manifest=dataset,
        dataset_file=dataset_file,
        dataset_yaml=dataset_yaml,
        fold=fold,
        contract=contract,
        reader=reader,
        location="evaluation report",
    )
    return {
        "report": _binding(report_file, schema=EVALUATION_SCHEMA),
        "provenance": provenance,
        "experiment": {"arm_id": arm_id, "seed": seed, "fold_id": fold_id},
        "settings": payload["settings"],
        "bindings": payload["bindings"],
        "val_source": payload["val_source"],
        "metrics": payload["metrics"],
        "compute": payload["compute"],
    }


def _summarize_runs(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not runs:
        raise TrainingCVAggregationError("cannot summarize an empty run set")
    tp = fp = fn = evaluated = clean = zero_annotation_frames = 0
    map50: list[float] = []
    map50_95: list[float] = []
    class_counts = {label: {"tp": 0, "fp": 0, "fn": 0} for label in CANONICAL_NAMES}
    source_strata: list[dict[str, Any]] = []
    training_seconds = inference_seconds = load_seconds = 0.0
    for run in runs:
        metrics = _object(run["metrics"], "run.metrics")
        overall = _object(metrics["overall"], "run.metrics.overall")
        run_tp = int(overall["true_positive_count"])
        run_fp = int(overall["false_positive_count"])
        run_fn = int(overall["false_negative_count"])
        tp += run_tp
        fp += run_fp
        fn += run_fn
        evaluated += int(overall["evaluated_frame_count"])
        clean += int(overall["clean_frame_count"])
        zero_annotation_frames += int(run["val_source"]["zero_annotation_frame_count"])
        map50.append(float(metrics["map50"]))
        map50_95.append(float(metrics["map50_95"]))
        per_class_source: dict[str, Any] = {}
        for label in CANONICAL_NAMES:
            item = _object(metrics["per_class"][label], f"run {label}")
            class_counts[label]["tp"] += int(item["true_positive_count"])
            class_counts[label]["fp"] += int(item["false_positive_count"])
            class_counts[label]["fn"] += int(item["false_negative_count"])
            per_class_source[label] = {
                "status": item["status"],
                "support_count": item["support_count"],
                "recall": item["recall"],
                "f1_score": (
                    None
                    if item["status"] == "not_evaluable"
                    else _count_f1(
                        int(item["true_positive_count"]),
                        int(item["false_positive_count"]),
                        int(item["false_negative_count"]),
                    )
                ),
            }
        source_strata.append(
            {
                "fold_id": run["experiment"]["fold_id"],
                "asset_ids": run["val_source"]["asset_ids"],
                "frame_count": overall["evaluated_frame_count"],
                "true_positive_count": run_tp,
                "false_positive_count": run_fp,
                "false_negative_count": run_fn,
                "f1_score": _count_f1(run_tp, run_fp, run_fn),
                "clean_frame_rate": _ratio(
                    int(overall["clean_frame_count"]), int(overall["evaluated_frame_count"])
                ),
                "per_class": per_class_source,
            }
        )
        compute = _object(run["compute"], "run.compute")
        training_seconds += float(compute["training_duration_seconds"])
        inference_seconds += float(compute["evaluation_inference_seconds"])
        load_seconds += float(compute["model_load_seconds"])
    source_strata.sort(key=lambda item: item["fold_id"])
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    per_class: dict[str, Any] = {}
    for label in CANONICAL_NAMES:
        counts = class_counts[label]
        support = counts["tp"] + counts["fn"]
        predictions = counts["tp"] + counts["fp"]
        per_class[label] = {
            "status": "evaluable" if support else "not_evaluable",
            "support_count": support,
            "true_positive_count": counts["tp"],
            "false_positive_count": counts["fp"],
            "false_negative_count": counts["fn"],
            "prediction_count": predictions,
            "precision": _ratio(counts["tp"], predictions) if support else None,
            "recall": _ratio(counts["tp"], support),
            "f1_score": _count_f1(counts["tp"], counts["fp"], counts["fn"]) if support else None,
            "evaluable_source_count": sum(
                item["per_class"][label]["status"] == "evaluable" for item in source_strata
            ),
            "not_evaluable_source_count": sum(
                item["per_class"][label]["status"] == "not_evaluable" for item in source_strata
            ),
        }
    evaluable_sources = [item for item in source_strata if item["f1_score"] is not None]
    worst_source = (
        min(evaluable_sources, key=lambda item: (item["f1_score"], item["fold_id"]))
        if evaluable_sources
        else None
    )
    return {
        "run_count": len(runs),
        "oof": {
            "true_positive_count": tp,
            "false_positive_count": fp,
            "false_negative_count": fn,
            "prediction_count": tp + fp,
            "ground_truth_count": tp + fn,
            "precision": precision,
            "recall": recall,
            "f1_score": _count_f1(tp, fp, fn),
            "evaluated_frame_count": evaluated,
            "zero_annotation_frame_count": zero_annotation_frames,
            "clean_frame_count": clean,
            "clean_frame_rate": _ratio(clean, evaluated),
        },
        "per_class": per_class,
        "source_strata": source_strata,
        "worst_source": (
            {
                "status": "evaluable",
                "fold_id": worst_source["fold_id"],
                "asset_ids": worst_source["asset_ids"],
                "f1_score": worst_source["f1_score"],
            }
            if worst_source is not None
            else {
                "status": "not_evaluable",
                "fold_id": None,
                "asset_ids": [],
                "f1_score": None,
            }
        ),
        "validation_map": {
            "aggregation": "unweighted_source_macro_from_candidate_receipts",
            "map50": sum(map50) / len(map50),
            "map50_95": sum(map50_95) / len(map50_95),
        },
        "compute": {
            "training_duration_seconds": training_seconds,
            "evaluation_inference_seconds": inference_seconds,
            "model_load_seconds": load_seconds,
            "total_seconds": training_seconds + inference_seconds + load_seconds,
        },
    }


def _matrix(
    arms: Sequence[str], seeds: Sequence[int], folds: Sequence[str]
) -> list[dict[str, Any]]:
    return [
        {"arm_id": arm_id, "seed": seed, "fold_id": fold_id}
        for arm_id in sorted(arms)
        for seed in sorted(seeds)
        for fold_id in sorted(folds)
    ]


def _run_key(run: Mapping[str, Any]) -> tuple[str, int, str]:
    experiment = _object(run.get("experiment"), "run.experiment")
    return str(experiment["arm_id"]), int(experiment["seed"]), str(experiment["fold_id"])


def _validate_exact_matrix(
    runs: Sequence[Mapping[str, Any]], expected: Sequence[Mapping[str, Any]], location: str
) -> None:
    keys = [_run_key(run) for run in runs]
    if len(keys) != len(set(keys)):
        raise TrainingCVAggregationError(f"{location} contains duplicate arm/seed/fold reports")
    expected_keys = {
        (str(item["arm_id"]), int(item["seed"]), str(item["fold_id"])) for item in expected
    }
    actual_keys = set(keys)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        raise TrainingCVAggregationError(
            f"{location} run matrix differs; missing={missing!r}, unexpected={unexpected!r}"
        )


def _group_summaries(
    runs: Sequence[Mapping[str, Any]], arms: Sequence[str], seeds: Sequence[int]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    strata: dict[str, Any] = {}
    arm_summaries: list[dict[str, Any]] = []
    for arm_id in sorted(arms):
        strata[arm_id] = {}
        all_arm_runs: list[Mapping[str, Any]] = []
        for seed in sorted(seeds):
            selected = [
                run
                for run in runs
                if run["experiment"]["arm_id"] == arm_id and run["experiment"]["seed"] == seed
            ]
            summary = _summarize_runs(selected)
            strata[arm_id][str(seed)] = summary
            all_arm_runs.extend(selected)
        arm_summaries.append(
            {
                "arm_id": arm_id,
                "seed_count": len(seeds),
                **_summarize_runs(all_arm_runs),
            }
        )
    return strata, arm_summaries


def _screening_selection(arm_summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def descending(value: Any) -> float:
        return math.inf if value is None else -float(value)

    def key(item: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            descending(item["oof"]["f1_score"]),
            descending(item["worst_source"]["f1_score"]),
            descending(item["oof"]["clean_frame_rate"]),
            descending(item["validation_map"]["map50_95"]),
            float(item["compute"]["total_seconds"]),
            str(item["arm_id"]),
        )

    ordered = sorted(arm_summaries, key=key)
    ranking: list[dict[str, Any]] = []
    for index, item in enumerate(ordered, start=1):
        ranking.append(
            {
                "rank": index,
                "arm_id": item["arm_id"],
                "oof_f1_score": item["oof"]["f1_score"],
                "worst_source_f1_score": item["worst_source"]["f1_score"],
                "clean_frame_rate": item["oof"]["clean_frame_rate"],
                "map50_95": item["validation_map"]["map50_95"],
                "compute_cost_seconds": item["compute"]["total_seconds"],
            }
        )
    return {
        "ranking_contract": RANKING,
        "ranked_arms": ranking,
        "winner_arm_id": ranking[0]["arm_id"],
    }


def _read_screening_report_paths(
    screening_path: Path,
    *,
    contract: ProtocolContract,
    reader: _EvidenceReader,
    winner_arm_id: str,
) -> tuple[list[tuple[Path, Mapping[str, Any]]], Mapping[str, Any], Mapping[str, Any]]:
    payload, bound = reader.json(
        screening_path,
        scopes=(("docs", "evidence"),),
        location="frozen screening aggregate",
    )
    _scan_declared_paths(payload, "frozen screening aggregate")
    if payload.get("schema") != OUTPUT_SCHEMA or payload.get("mode") != "screening":
        raise TrainingCVAggregationError("confirmation requires a screening aggregate v1")
    _require_passed_gate(payload.get("gate"), "frozen screening aggregate.gate")
    bindings = _object(payload.get("bindings"), "frozen screening aggregate.bindings")
    if (
        _object(bindings.get("recovery_protocol"), "screening recovery binding").get("sha256")
        != contract.binding["sha256"]
    ):
        raise TrainingCVAggregationError("screening aggregate binds another Recovery protocol")
    selection = _object(payload.get("selection"), "frozen screening aggregate.selection")
    if selection.get("winner_arm_id") != winner_arm_id:
        raise TrainingCVAggregationError("specified winner differs from frozen screening winner")
    screening_contract = _object(payload.get("contract"), "frozen screening aggregate.contract")
    if screening_contract.get("fold_ids") != list(contract.fold_ids):
        raise TrainingCVAggregationError("frozen screening aggregate binds another LOSO fold set")
    runs = _list(payload.get("runs"), "frozen screening aggregate.runs")
    expected = _matrix(list(contract.arms), [contract.screening_seed], contract.fold_ids)
    _validate_exact_matrix(runs, expected, "frozen screening aggregate")
    paths: list[tuple[Path, Mapping[str, Any]]] = []
    for index, run in enumerate(runs):
        report = _object(run.get("report"), f"screening run[{index}].report")
        if set(report) != {"path", "sha256", "size_bytes", "schema"}:
            raise TrainingCVAggregationError(
                f"screening run[{index}].report binding contract differs"
            )
        if report.get("schema") != EVALUATION_SCHEMA:
            raise TrainingCVAggregationError(
                f"screening run[{index}] report schema binding differs"
            )
        _sha256(report.get("sha256"), f"screening run[{index}].report.sha256")
        _integer(report.get("size_bytes"), f"screening run[{index}].report.size_bytes", minimum=1)
        relative = _safe_relative(report.get("path"), f"screening run[{index}].report.path")
        paths.append((reader.workspace / Path(*relative.parts), report))
    return (
        paths,
        {
            **_binding(bound, schema=OUTPUT_SCHEMA),
            "winner_arm_id": winner_arm_id,
        },
        payload,
    )


def _confirmation_readiness(
    *,
    winner: str,
    strata: Mapping[str, Any],
    contract: ProtocolContract,
) -> dict[str, Any]:
    gates = contract.readiness
    control = "repaired_control"
    threshold_checks: list[dict[str, Any]] = []
    paired_sources: list[dict[str, Any]] = []
    paired_classes: list[dict[str, Any]] = []
    winner_seed_f1: list[float] = []
    for seed in contract.confirmation_seeds:
        seed_key = str(seed)
        winner_summary = strata[winner][seed_key]
        winner_seed_f1.append(float(winner_summary["oof"]["f1_score"]))
        threshold_checks.append(
            {
                "seed": seed,
                "oof_precision": winner_summary["oof"]["precision"],
                "oof_recall": winner_summary["oof"]["recall"],
                "clean_frame_rate": winner_summary["oof"]["clean_frame_rate"],
                "passed": (
                    winner_summary["oof"]["precision"] is not None
                    and winner_summary["oof"]["precision"] >= gates["oof_precision_min"]
                    and winner_summary["oof"]["recall"] is not None
                    and winner_summary["oof"]["recall"] >= gates["oof_recall_min"]
                    and winner_summary["oof"]["clean_frame_rate"] is not None
                    and winner_summary["oof"]["clean_frame_rate"] >= gates["clean_frame_rate_min"]
                ),
            }
        )
        control_summary = strata[control][seed_key]
        winner_sources = {item["fold_id"]: item for item in winner_summary["source_strata"]}
        control_sources = {item["fold_id"]: item for item in control_summary["source_strata"]}
        for fold_id in contract.fold_ids:
            winner_source = winner_sources[fold_id]
            control_source = control_sources[fold_id]
            winner_f1 = winner_source["f1_score"]
            control_f1 = control_source["f1_score"]
            if winner_f1 is None or control_f1 is None:
                source_status = "not_evaluable"
                delta = None
                # With no ground truth and no predictions F1 is undefined.  A
                # clean winner is noninferior to a control that emitted false
                # positives; the inverse is conservatively a failure.
                source_passed = winner_f1 is None
            else:
                source_status = "evaluable"
                delta = winner_f1 - control_f1
                source_passed = winner_f1 >= control_f1
            paired_sources.append(
                {
                    "seed": seed,
                    "fold_id": fold_id,
                    "status": source_status,
                    "winner_f1_score": winner_f1,
                    "control_f1_score": control_f1,
                    "delta": delta,
                    "passed": source_passed,
                }
            )
            for label in CANONICAL_NAMES:
                winner_class = winner_source["per_class"][label]
                control_class = control_source["per_class"][label]
                if winner_class["status"] == "not_evaluable":
                    comparison = {
                        "seed": seed,
                        "fold_id": fold_id,
                        "class_name": label,
                        "status": "not_evaluable",
                        "winner_recall": None,
                        "control_recall": None,
                        "delta": None,
                        "passed": True,
                    }
                else:
                    winner_recall = float(winner_class["recall"])
                    control_recall = float(control_class["recall"])
                    delta = winner_recall - control_recall
                    comparison = {
                        "seed": seed,
                        "fold_id": fold_id,
                        "class_name": label,
                        "status": "evaluable",
                        "winner_recall": winner_recall,
                        "control_recall": control_recall,
                        "delta": delta,
                        "passed": delta >= -float(gates["supported_class_recall_drop_max"]),
                    }
                paired_classes.append(comparison)
    sample_std = statistics.stdev(winner_seed_f1)
    checks = {
        "all_planned_runs_present": True,
        "exact_contract_match": True,
        "winner_seed_thresholds_pass": all(item["passed"] for item in threshold_checks),
        "every_seed_and_source_f1_noninferior_to_repaired_control": all(
            item["passed"] for item in paired_sources
        ),
        "paired_supported_class_recall_drop_within_limit": all(
            item["passed"] for item in paired_classes
        ),
        "seed_oof_f1_sample_standard_deviation_within_limit": sample_std
        <= gates["seed_oof_f1_sample_standard_deviation_max"],
    }
    blocking_reasons = [key for key, passed in checks.items() if not passed]
    return {
        "winner_arm_id": winner,
        "baseline_arm_id": control,
        "thresholds": dict(gates),
        "winner_seed_metrics": threshold_checks,
        "paired_source_comparisons": paired_sources,
        "paired_class_recall_comparisons": paired_classes,
        "winner_seed_oof_f1": winner_seed_f1,
        "winner_seed_oof_f1_sample_standard_deviation": sample_std,
        "checks": checks,
        "blocking_reasons": blocking_reasons,
        "passed": not blocking_reasons,
    }


def _publish_no_replace(output: Path, encoded: bytes, workspace: Path) -> Path:
    output_path, _ = _scoped_path(
        output,
        workspace=workspace,
        scopes=(("docs", "evidence"),),
        location="output",
        leaf_kind="missing",
    )
    if output_path.suffix != ".json":
        raise TrainingCVAggregationError("output must be a new JSON file")
    parent = output_path.parent
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(
            prefix=f".{output_path.name}.", suffix=".tmp", dir=parent
        )
        temporary = Path(name)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, output_path, follow_symlinks=False)
        except FileExistsError as error:
            raise TrainingCVAggregationError(f"output already exists: {output_path}") from error
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output_path


def aggregate_training_cv(
    *,
    mode: str,
    recovery_protocol_path: Path,
    evaluation_report_paths: Sequence[Path],
    output: Path,
    screening_aggregate_path: Path | None = None,
    winner_arm_id: str | None = None,
    workspace_root: Path | None = None,
) -> dict[str, Any]:
    """Validate and aggregate screening or paired-confirmation LOSO evidence."""

    if mode not in {"screening", "confirmation"}:
        raise TrainingCVAggregationError("mode must be screening or confirmation")
    workspace = (workspace_root or Path.cwd()).resolve(strict=True)
    reader = _EvidenceReader(workspace)
    contract = _read_protocol(Path(recovery_protocol_path), reader=reader)
    runs: list[dict[str, Any]] = []
    screening_binding: Mapping[str, Any] | None = None
    if mode == "screening":
        if screening_aggregate_path is not None or winner_arm_id is not None:
            raise TrainingCVAggregationError(
                "screening must not receive a screening aggregate or winner override"
            )
        for path in evaluation_report_paths:
            runs.append(
                _validate_report(
                    Path(path), contract=contract, reader=reader, provenance="provided"
                )
            )
        expected = _matrix(list(contract.arms), [contract.screening_seed], contract.fold_ids)
        _validate_exact_matrix(runs, expected, "screening")
        arms = list(contract.arms)
        seeds = [contract.screening_seed]
    else:
        if screening_aggregate_path is None or winner_arm_id is None:
            raise TrainingCVAggregationError(
                "confirmation requires --screening-aggregate and --winner-arm-id"
            )
        winner = _fold_identifier(winner_arm_id, "winner_arm_id")
        if winner not in contract.arms:
            raise TrainingCVAggregationError("winner_arm_id is not a Recovery arm")
        screening_paths, screening_binding, screening_payload = _read_screening_report_paths(
            Path(screening_aggregate_path),
            contract=contract,
            reader=reader,
            winner_arm_id=winner,
        )
        screened_runs: list[dict[str, Any]] = []
        for path, expected_report_binding in screening_paths:
            screened = _validate_report(
                path,
                contract=contract,
                reader=reader,
                provenance="provided",
            )
            if screened["report"] != expected_report_binding:
                raise TrainingCVAggregationError(
                    "a report changed after the screening aggregate was frozen"
                )
            screened_runs.append(screened)
        screening_strata, screening_summaries = _group_summaries(
            screened_runs, list(contract.arms), [contract.screening_seed]
        )
        recomputed_selection = _screening_selection(screening_summaries)
        if recomputed_selection["winner_arm_id"] != winner:
            raise TrainingCVAggregationError(
                "frozen screening winner does not match recomputed report evidence"
            )
        screened_runs.sort(key=_run_key)
        if (
            screening_payload.get("runs") != screened_runs
            or screening_payload.get("strata") != screening_strata
            or screening_payload.get("aggregates") != screening_summaries
            or screening_payload.get("selection") != recomputed_selection
        ):
            raise TrainingCVAggregationError(
                "frozen screening aggregate differs from recomputed report evidence"
            )
        required_arms = {winner, "repaired_control"}
        runs.extend(
            {**run, "provenance": "reused_from_frozen_screening"}
            for run in screened_runs
            if run["experiment"]["arm_id"] in required_arms
        )
        for path in evaluation_report_paths:
            run = _validate_report(
                Path(path), contract=contract, reader=reader, provenance="provided_confirmation"
            )
            if run["experiment"]["seed"] == contract.screening_seed:
                raise TrainingCVAggregationError(
                    "confirmation seed-42 reports must be reused from frozen screening"
                )
            runs.append(run)
        arms = [winner] if winner == "repaired_control" else [winner, "repaired_control"]
        seeds = list(contract.confirmation_seeds)
        expected = _matrix(arms, seeds, contract.fold_ids)
        _validate_exact_matrix(runs, expected, "confirmation")

    # A source inventory digest must be identical for the same fold across every
    # arm/seed, otherwise paired source comparisons are not meaningful.
    frame_bindings: dict[str, str] = {}
    for run in runs:
        fold_id = run["experiment"]["fold_id"]
        frames_sha = run["val_source"]["frames_sha256"]
        previous = frame_bindings.setdefault(fold_id, frames_sha)
        if previous != frames_sha:
            raise TrainingCVAggregationError(
                f"validation frame inventory drifted across reports for {fold_id}"
            )

    runs.sort(key=_run_key)
    strata, arm_summaries = _group_summaries(runs, arms, seeds)
    selection: Mapping[str, Any] | None
    readiness: Mapping[str, Any] | None
    if mode == "screening":
        selection = _screening_selection(arm_summaries)
        readiness = None
        gate_passed = True
        blocking_reasons: list[str] = []
        gate_checks = {
            "frozen_recovery_protocol_verified": True,
            "source_derived_loso_plan_verified": True,
            "exact_screening_matrix_present": True,
            "all_reports_and_artifact_bindings_verified": True,
            "metrics_recomputed_from_raw_counts": True,
            "deterministic_protocol_ranking_applied": True,
            "final_holdout_inputs_rejected": True,
        }
    else:
        selection = {
            "winner_arm_id": winner_arm_id,
            "source": "frozen_screening_aggregate",
        }
        readiness = _confirmation_readiness(
            winner=str(winner_arm_id), strata=strata, contract=contract
        )
        gate_passed = bool(readiness["passed"])
        blocking_reasons = list(readiness["blocking_reasons"])
        gate_checks = {
            "frozen_recovery_protocol_verified": True,
            "frozen_screening_aggregate_verified": True,
            "seed_42_reused_from_screening": True,
            "exact_confirmation_matrix_present": True,
            "all_reports_and_artifact_bindings_verified": True,
            "metrics_recomputed_from_raw_counts": True,
            "paired_same_seed_same_fold_comparisons_applied": True,
            "readiness_gates_passed": gate_passed,
            "final_holdout_inputs_rejected": True,
        }

    if contract.reanalysis is not None:
        gate_checks["immutable_reanalysis_evidence_lineage_verified"] = True

    payload: dict[str, Any] = {
        "schema": OUTPUT_SCHEMA,
        "mode": mode,
        "bindings": {
            "recovery_protocol": dict(contract.binding),
            "loso_plan": dict(contract.cv_binding),
            "training_reference": dict(contract.training_reference_binding),
            "training_annotations": dict(contract.training_annotations_binding),
            "base_weights": dict(contract.base_weights_binding),
            "implementation_files": [dict(binding) for binding in contract.implementation_bindings],
            **(
                {"screening_aggregate": dict(screening_binding)}
                if screening_binding is not None
                else {}
            ),
            **(
                {
                    "reanalysis_lineage": {
                        "source_protocol": dict(contract.reanalysis["source_protocol"]),
                        "source_screening_aggregate": dict(
                            contract.reanalysis["source_screening_aggregate"]
                        ),
                        "failure_evidence": dict(contract.reanalysis["failure_evidence"]),
                        "report_count": contract.reanalysis["report_count"],
                        "reports_manifest_sha256": contract.reanalysis["reports_manifest_sha256"],
                    }
                }
                if contract.reanalysis is not None
                else {}
            ),
        },
        "contract": {
            "taxonomy": dict(contract.taxonomy),
            "arm_ids": sorted(arms),
            "seeds": sorted(seeds),
            "fold_ids": list(contract.fold_ids),
            "evaluation_settings": dict(contract.settings),
            "ranking": RANKING,
            "readiness_gates": dict(contract.readiness),
        },
        "run_matrix": {
            "required": expected,
            "required_count": len(expected),
            "observed_count": len(runs),
            "complete": True,
        },
        "runs": runs,
        "strata": strata,
        "aggregates": arm_summaries,
        "selection": selection,
        "readiness": readiness,
        "holdout_firewall": {
            "input_read": False,
            "allowed_scopes": ["data/training", "docs/evidence", "data/model-candidates"],
            "rejected_scopes": list(FINAL_HOLDOUT_REJECTED_SCOPES),
        },
        "gate": {
            "passed": gate_passed,
            "blocking_reasons": blocking_reasons,
            "checks": gate_checks,
        },
    }
    encoded = _json_bytes(payload)
    reader.revalidate()
    output_path = _publish_no_replace(Path(output), encoded, workspace)
    return {
        "output": str(output_path),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "mode": mode,
        "run_count": len(runs),
        "winner_arm_id": selection["winner_arm_id"],
        "gate": payload["gate"],
        "holdout_input_read": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("screening", "confirmation"), required=True)
    parser.add_argument("--recovery-protocol", type=Path, required=True)
    parser.add_argument("--evaluation-report", type=Path, action="append", default=[])
    parser.add_argument("--screening-aggregate", type=Path)
    parser.add_argument("--winner-arm-id")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = aggregate_training_cv(
            mode=args.mode,
            recovery_protocol_path=args.recovery_protocol,
            evaluation_report_paths=args.evaluation_report,
            screening_aggregate_path=args.screening_aggregate,
            winner_arm_id=args.winner_arm_id,
            output=args.output,
            workspace_root=args.workspace_root,
        )
    except (TrainingCVAggregationError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
