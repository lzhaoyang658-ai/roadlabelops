"""Freeze the best of exactly three evidence-bound YOLO11n training candidates.

This selector deliberately has no training, CVAT, network, or holdout inputs.  It
only accepts three completed candidate output directories.  Each directory must
contain ``receipt.json`` and the receipt-bound ``weights/best.pt``.  Candidate
receipts are validated as one common experiment contract with seed as the sole
allowed configuration difference, then ranked by aggregate mAP50-95, mAP50,
recall, precision, and finally the smaller seed.

The output is a new, atomically published directory containing ``best.pt`` and
``receipt.json``.  Existing outputs, including empty directories and symlinks,
are never replaced.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import math
import os
import re
import shutil
import stat
import sys
import tempfile
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from roadlabelops.holdout_policy import NO_FINAL_HOLDOUT_STATEMENT

LEGACY_CANDIDATE_SCHEMA = {"name": "roadlabelops.yolo-candidate-training", "version": 1}
CANDIDATE_SCHEMA = {"name": "roadlabelops.yolo-candidate-training", "version": 2}
LEGACY_PROTOCOL_SCHEMA = {
    "name": "roadlabelops.yolo-candidate-training-protocol",
    "version": 1,
}
PROTOCOL_SCHEMA = {
    "name": "roadlabelops.yolo-candidate-training-protocol",
    "version": 2,
}
LEGACY_DATASET_SCHEMA = {"name": "roadlabelops.yolo-dataset", "version": 2}
DATASET_SCHEMA = {"name": "roadlabelops.yolo-dataset", "version": 3}
FROZEN_SCHEMA = {"name": "roadlabelops.yolo-frozen-candidate", "version": 1}

REQUIRED_LABELS = (
    "car",
    "bus",
    "truck",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_light",
    "traffic_sign",
)
MODEL_LABELS = (
    "car",
    "bus",
    "truck",
    "motorcycle",
    "bicycle",
    "person",
    "traffic light",
    "stop sign",
)
MODEL_TO_CANONICAL = dict(zip(MODEL_LABELS, REQUIRED_LABELS, strict=True))
LEGACY_MODEL_TO_CANONICAL = {name: name for name in REQUIRED_LABELS}
LEGACY_TRAINING_FIELDS = (
    "epochs",
    "patience",
    "imgsz",
    "batch",
    "device",
    "workers",
    "optimizer",
    "deterministic",
    "amp",
    "cache",
    "close_mosaic",
    "freeze",
)
TRAINING_FIELDS = (
    *LEGACY_TRAINING_FIELDS,
    "lr0",
    "lrf",
    "momentum",
    "weight_decay",
    "warmup_epochs",
    "warmup_bias_lr",
    "cls_pw",
)
AGGREGATE_METRICS = ("precision", "recall", "map50", "map50_95", "fitness")
RANKING_METRICS = ("map50_95", "map50", "recall", "precision")
ARTIFACT_PATHS = {
    "args": "artifacts/args.yaml",
    "results": "artifacts/results.csv",
    "best_weights": "weights/best.pt",
    "last_weights": "weights/last.pt",
}
LEGACY_REQUIRED_GATE_CHECKS = frozenset(
    {
        "schema_v2_dataset_manifest_verified",
        "dataset_gate_verified",
        "all_managed_dataset_files_verified",
        "dataset_yaml_hash_and_taxonomy_verified",
        "base_yolo11n_weight_hash_verified",
        "isolated_workspace_training_verified",
        "trainer_args_match_frozen_protocol",
        "finite_complete_eight_class_metrics_verified",
        "required_training_artifacts_verified",
        "source_inputs_unchanged",
        "holdout_input_not_read",
    }
)
REQUIRED_GATE_CHECKS = (LEGACY_REQUIRED_GATE_CHECKS - {"schema_v2_dataset_manifest_verified"}) | {
    "supported_dataset_manifest_schema_verified"
}
PRE_SUPPORT_AWARE_REQUIRED_GATE_CHECKS = REQUIRED_GATE_CHECKS
REQUIRED_GATE_CHECKS = (REQUIRED_GATE_CHECKS - {"finite_complete_eight_class_metrics_verified"}) | {
    "support_aware_complete_eight_class_metrics_verified"
}
SUPPORT_AWARE_REQUIRED_GATE_CHECKS = REQUIRED_GATE_CHECKS
REQUIRED_GATE_CHECKS = REQUIRED_GATE_CHECKS | {"pretrained_class_head_transfer_verified"}
NO_HOLDOUT_STATEMENT = NO_FINAL_HOLDOUT_STATEMENT
SELECTION_ORDER = ("mAP50-95", "mAP50", "recall", "precision", "smaller_seed")
EXPECTED_SEEDS = frozenset({42, 43, 44})
EXPECTED_ULTRALYTICS_VERSION = "8.4.135"
MAX_RECEIPT_BYTES = 4 * 1024 * 1024


class CandidateFreezeError(ValueError):
    """Raised when candidate evidence cannot be frozen safely."""


# A descriptive alias for callers that name errors after the module operation.
FreezeModelCandidateError = CandidateFreezeError


@dataclass(frozen=True)
class ValidatedCandidate:
    root: Path
    root_identity: tuple[int, int]
    seed: int
    receipt_sha256: str
    receipt_size_bytes: int
    protocol: dict[str, Any]
    protocol_sha256: str
    resolved_args_without_seed: dict[str, Any]
    dataset: dict[str, Any]
    base_weights: dict[str, Any]
    metrics: dict[str, float]
    best_weights_path: str
    best_weights_sha256: str
    best_weights_size_bytes: int

    @property
    def rank_key(self) -> tuple[float, float, float, float, int]:
        return (
            -self.metrics["map50_95"],
            -self.metrics["map50"],
            -self.metrics["recall"],
            -self.metrics["precision"],
            self.seed,
        )


def _expect_keys(value: Mapping[str, Any], expected: set[str], location: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise CandidateFreezeError(
            f"{location} has unexpected or missing keys "
            f"(missing={missing}, unexpected={unexpected})"
        )


def _object(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CandidateFreezeError(f"{location} must be an object")
    return value


def _list(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise CandidateFreezeError(f"{location} must be an array")
    return value


def _strict_bool(value: Any, location: str) -> bool:
    if not isinstance(value, bool):
        raise CandidateFreezeError(f"{location} must be a boolean")
    return value


def _strict_int(
    value: Any,
    location: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CandidateFreezeError(f"{location} must be an integer")
    if minimum is not None and value < minimum:
        raise CandidateFreezeError(f"{location} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise CandidateFreezeError(f"{location} must be at most {maximum}")
    return value


def _finite_metric(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CandidateFreezeError(f"{location} must be a number")
    try:
        result = float(value)
    except OverflowError as error:
        raise CandidateFreezeError(f"{location} must be finite") from error
    if not math.isfinite(result):
        raise CandidateFreezeError(f"{location} must be finite")
    if not 0.0 <= result <= 1.0:
        raise CandidateFreezeError(f"{location} must be within [0, 1]")
    return 0.0 if result == 0.0 else result


def _finite_number(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CandidateFreezeError(f"{location} must be a number")
    try:
        result = float(value)
    except OverflowError as error:
        raise CandidateFreezeError(f"{location} must be finite") from error
    if not math.isfinite(result):
        raise CandidateFreezeError(f"{location} must be finite")
    return 0.0 if result == 0.0 else result


def _text(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CandidateFreezeError(f"{location} must be a non-empty string")
    if any(ord(character) < 32 for character in value):
        raise CandidateFreezeError(f"{location} contains a control character")
    return value


def _sha256(value: Any, location: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CandidateFreezeError(f"{location} must be a lowercase SHA-256 digest")
    return value


def _canonical_bytes(payload: Any) -> bytes:
    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise CandidateFreezeError(f"contract is not canonical JSON: {error}") from error


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


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
        raise CandidateFreezeError(f"selection receipt is not valid JSON: {error}") from error


def _relative_artifact_path(value: Any, location: str) -> PurePosixPath:
    text = _text(value, location)
    posix = PurePosixPath(text)
    windows = PureWindowsPath(text)
    if (
        "\\" in text
        or text.lower().startswith("file:")
        or posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or posix.as_posix() != text
        or not posix.name
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        raise CandidateFreezeError(f"{location} is an unsafe relative artifact path")
    return posix


def _reject_lexical_traversal(path: Path, location: str) -> None:
    raw = os.fspath(path)
    if not raw or "\x00" in raw or any(part == ".." for part in Path(raw).parts):
        raise CandidateFreezeError(f"{location} is unsafe")


def _absolute_lexical(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _reject_symlink_components(path: Path, location: str, *, allow_missing_leaf: bool) -> None:
    absolute = _absolute_lexical(path)
    current = Path(absolute.anchor)
    parts = absolute.parts[1:] if absolute.anchor else absolute.parts
    for index, part in enumerate(parts):
        current /= part
        is_leaf = index == len(parts) - 1
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            if allow_missing_leaf and is_leaf:
                return
            raise CandidateFreezeError(f"{location} does not exist: {path}") from None
        except OSError as error:
            raise CandidateFreezeError(f"could not inspect {location}: {error}") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise CandidateFreezeError(f"{location} must not contain or be a symlink: {current}")


def _reject_existing_symlink_prefix(path: Path, location: str) -> None:
    """Reject symlinks in the existing prefix of a path that may not exist yet."""

    absolute = _absolute_lexical(path)
    current = Path(absolute.anchor)
    parts = absolute.parts[1:] if absolute.anchor else absolute.parts
    for part in parts:
        current /= part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            return
        except OSError as error:
            raise CandidateFreezeError(f"could not inspect {location}: {error}") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise CandidateFreezeError(f"{location} must not contain a symlink: {current}")


def _open_candidate_root(path: Path) -> tuple[Path, int, tuple[int, int]]:
    _reject_lexical_traversal(path, "candidate directory")
    root = _absolute_lexical(path)
    _reject_symlink_components(root, "candidate directory", allow_missing_leaf=False)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(root, flags)
    except OSError as error:
        raise CandidateFreezeError(f"could not open candidate directory {path}: {error}") from error
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        os.close(descriptor)
        raise CandidateFreezeError(f"candidate directory is not a directory: {path}")
    return root, descriptor, (metadata.st_dev, metadata.st_ino)


def _open_relative_regular(root_fd: int, relative: PurePosixPath, location: str) -> int:
    current_fd = os.dup(root_fd)
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    file_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        for part in relative.parts[:-1]:
            next_fd = os.open(part, directory_flags, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        result = os.open(relative.name, file_flags, dir_fd=current_fd)
    except OSError as error:
        if error.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise CandidateFreezeError(f"{location} must not contain or be a symlink") from error
        raise CandidateFreezeError(f"could not open {location}: {error}") from error
    finally:
        os.close(current_fd)
    metadata = os.fstat(result)
    if not stat.S_ISREG(metadata.st_mode):
        os.close(result)
        raise CandidateFreezeError(f"{location} must be a regular file")
    return result


def _read_all_stable(
    descriptor: int, location: str, *, maximum: int
) -> tuple[bytes, os.stat_result]:
    before = os.fstat(descriptor)
    if before.st_size > maximum:
        raise CandidateFreezeError(f"{location} exceeds {maximum} bytes")
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = os.read(descriptor, min(1024 * 1024, maximum + 1 - total))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if total > maximum:
            raise CandidateFreezeError(f"{location} exceeds {maximum} bytes")
    after = os.fstat(descriptor)
    if _stat_identity(before) != _stat_identity(after) or total != before.st_size:
        raise CandidateFreezeError(f"{location} changed while it was being read")
    return b"".join(chunks), after


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _hash_open_file(descriptor: int, location: str) -> tuple[str, int]:
    before = os.fstat(descriptor)
    digest = hashlib.sha256()
    total = 0
    while chunk := os.read(descriptor, 1024 * 1024):
        digest.update(chunk)
        total += len(chunk)
    after = os.fstat(descriptor)
    if _stat_identity(before) != _stat_identity(after) or total != before.st_size:
        raise CandidateFreezeError(f"{location} changed while it was being hashed")
    return digest.hexdigest(), total


def _reject_json_constant(value: str) -> None:
    raise CandidateFreezeError(f"receipt contains non-finite JSON number {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CandidateFreezeError(f"receipt contains duplicate JSON key {key!r}")
        result[key] = value
    return result


def _parse_receipt(encoded: bytes, location: str) -> dict[str, Any]:
    try:
        text = encoded.decode("utf-8")
        payload = json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_object,
        )
    except CandidateFreezeError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CandidateFreezeError(f"could not parse {location}: {error}") from error
    result = _object(payload, location)
    _reject_nonfinite_json(result, location)
    return result


def _reject_nonfinite_json(value: Any, location: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_nonfinite_json(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_nonfinite_json(item, f"{location}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise CandidateFreezeError(f"{location} must be finite")


def _validate_schema(value: Any, expected: Mapping[str, Any], location: str) -> None:
    schema = _object(value, location)
    _expect_keys(schema, set(expected), location)
    _text(schema["name"], f"{location}.name")
    _strict_int(schema["version"], f"{location}.version", minimum=1)
    if not _same_contract(schema, expected):
        raise CandidateFreezeError(f"{location} is not the supported schema {dict(expected)!r}")


def _supported_schema(
    value: Any, supported: Sequence[Mapping[str, Any]], location: str
) -> dict[str, Any]:
    schema = _object(value, location)
    _expect_keys(schema, {"name", "version"}, location)
    _text(schema["name"], f"{location}.name")
    _strict_int(schema["version"], f"{location}.version", minimum=1)
    for expected in supported:
        if _same_contract(schema, expected):
            return dict(expected)
    raise CandidateFreezeError(f"{location} is unsupported")


def _validate_training_config(value: Any, location: str, fields: Sequence[str]) -> dict[str, Any]:
    training = _object(value, location)
    _expect_keys(training, set(fields), location)
    epochs = _strict_int(training["epochs"], f"{location}.epochs", minimum=1)
    patience = _strict_int(training["patience"], f"{location}.patience", minimum=0)
    if patience > epochs:
        raise CandidateFreezeError(f"{location}.patience must not exceed epochs")
    imgsz = _strict_int(training["imgsz"], f"{location}.imgsz", minimum=32)
    if imgsz % 32:
        raise CandidateFreezeError(f"{location}.imgsz must be a multiple of 32")
    _strict_int(training["batch"], f"{location}.batch", minimum=1)
    device = _text(training["device"], f"{location}.device")
    if device not in {"cpu", "mps"} and not (device.isascii() and device.isdecimal()):
        raise CandidateFreezeError(f"{location}.device is not a supported fixed device")
    _strict_int(training["workers"], f"{location}.workers", minimum=0)
    optimizer = _text(training["optimizer"], f"{location}.optimizer")
    if optimizer not in {"SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp"}:
        raise CandidateFreezeError(f"{location}.optimizer is unsupported")
    if _strict_bool(training["deterministic"], f"{location}.deterministic") is not True:
        raise CandidateFreezeError(f"{location}.deterministic must be true")
    _strict_bool(training["amp"], f"{location}.amp")
    cache = _text(training["cache"], f"{location}.cache")
    if cache not in {"disk", "ram", "none"}:
        raise CandidateFreezeError(f"{location}.cache is unsupported")
    close_mosaic = _strict_int(training["close_mosaic"], f"{location}.close_mosaic", minimum=0)
    if close_mosaic > epochs:
        raise CandidateFreezeError(f"{location}.close_mosaic must not exceed epochs")
    _strict_int(training["freeze"], f"{location}.freeze", minimum=0)
    if tuple(fields) == TRAINING_FIELDS:
        lr0 = _finite_number(training["lr0"], f"{location}.lr0")
        if not 0.0 < lr0 <= 1.0:
            raise CandidateFreezeError(f"{location}.lr0 must be within (0, 1]")
        lrf = _finite_number(training["lrf"], f"{location}.lrf")
        if not 0.0 < lrf <= 1.0:
            raise CandidateFreezeError(f"{location}.lrf must be within (0, 1]")
        momentum = _finite_number(training["momentum"], f"{location}.momentum")
        if not 0.0 <= momentum < 1.0:
            raise CandidateFreezeError(f"{location}.momentum must be within [0, 1)")
        weight_decay = _finite_number(training["weight_decay"], f"{location}.weight_decay")
        if not 0.0 <= weight_decay <= 1.0:
            raise CandidateFreezeError(f"{location}.weight_decay must be within [0, 1]")
        warmup_epochs = _finite_number(training["warmup_epochs"], f"{location}.warmup_epochs")
        if warmup_epochs < 0.0:
            raise CandidateFreezeError(f"{location}.warmup_epochs must be non-negative")
        warmup_bias_lr = _finite_number(training["warmup_bias_lr"], f"{location}.warmup_bias_lr")
        if not 0.0 <= warmup_bias_lr <= 1.0:
            raise CandidateFreezeError(f"{location}.warmup_bias_lr must be within [0, 1]")
        cls_pw = _finite_number(training["cls_pw"], f"{location}.cls_pw")
        if not 0.0 <= cls_pw <= 1.0:
            raise CandidateFreezeError(f"{location}.cls_pw must be within [0, 1]")
    return training


def _taxonomy_contract(model_names: Sequence[str]) -> dict[str, Any]:
    names = tuple(model_names)
    if names == REQUIRED_LABELS:
        mapping = LEGACY_MODEL_TO_CANONICAL
    elif names == MODEL_LABELS:
        mapping = MODEL_TO_CANONICAL
    else:
        raise CandidateFreezeError("training protocol uses an unsupported model taxonomy")
    return {
        "canonical_names": list(REQUIRED_LABELS),
        "model_names": list(names),
        "model_to_canonical": dict(mapping),
    }


def _validate_taxonomy_contract(value: Any, location: str) -> dict[str, Any]:
    taxonomy = _object(value, location)
    _expect_keys(
        taxonomy,
        {"canonical_names", "model_names", "model_to_canonical"},
        location,
    )
    canonical_names = taxonomy["canonical_names"]
    model_names = taxonomy["model_names"]
    if not isinstance(canonical_names, list) or tuple(canonical_names) != REQUIRED_LABELS:
        raise CandidateFreezeError(f"{location}.canonical_names violates the eight-class contract")
    if not isinstance(model_names, list):
        raise CandidateFreezeError(f"{location}.model_names must be a list")
    expected = _taxonomy_contract(tuple(model_names))
    if not _same_contract(taxonomy, expected):
        raise CandidateFreezeError(f"{location} violates the canonical/model mapping contract")
    return expected


def _validate_protocol(
    value: Any, claimed_sha256: Any, location: str
) -> tuple[dict[str, Any], str]:
    protocol = _object(value, location)
    _expect_keys(
        protocol,
        {
            "schema",
            "model_family",
            "taxonomy",
            "training",
            "validation_selection",
            "holdout_access",
        },
        location,
    )
    protocol_schema = _supported_schema(
        protocol["schema"],
        (LEGACY_PROTOCOL_SCHEMA, PROTOCOL_SCHEMA),
        f"{location}.schema",
    )
    if protocol["model_family"] != "YOLO11n":
        raise CandidateFreezeError(f"{location}.model_family must be 'YOLO11n'")
    taxonomy = protocol["taxonomy"]
    if protocol_schema == LEGACY_PROTOCOL_SCHEMA:
        if not isinstance(taxonomy, list) or tuple(taxonomy) != REQUIRED_LABELS:
            raise CandidateFreezeError(
                f"{location}.taxonomy must be the fixed eight-class contract"
            )
        training_fields = LEGACY_TRAINING_FIELDS
    else:
        _validate_taxonomy_contract(taxonomy, f"{location}.taxonomy")
        training_fields = TRAINING_FIELDS
    _validate_training_config(protocol["training"], f"{location}.training", training_fields)
    selection = _object(protocol["validation_selection"], f"{location}.validation_selection")
    _expect_keys(selection, {"primary", "tie_breakers"}, f"{location}.validation_selection")
    if selection != {
        "primary": "mAP50-95",
        "tie_breakers": ["mAP50", "recall", "precision", "smaller_seed"],
    }:
        raise CandidateFreezeError(f"{location}.validation_selection is not the fixed contract")
    if protocol["holdout_access"] != "prohibited":
        raise CandidateFreezeError(f"{location}.holdout_access must be 'prohibited'")
    claimed = _sha256(claimed_sha256, f"{location}_sha256")
    actual = _canonical_sha256(protocol)
    if claimed != actual:
        raise CandidateFreezeError(f"{location}_sha256 does not match the protocol")
    return protocol, actual


def _validate_gate(value: Any, location: str, candidate_schema: Mapping[str, Any]) -> None:
    gate = _object(value, location)
    _expect_keys(gate, {"passed", "checks"}, location)
    if _strict_bool(gate["passed"], f"{location}.passed") is not True:
        raise CandidateFreezeError(f"{location}.passed must be true")
    checks = _object(gate["checks"], f"{location}.checks")
    supported_check_sets = (
        (LEGACY_REQUIRED_GATE_CHECKS,)
        if candidate_schema == LEGACY_CANDIDATE_SCHEMA
        else (
            PRE_SUPPORT_AWARE_REQUIRED_GATE_CHECKS,
            SUPPORT_AWARE_REQUIRED_GATE_CHECKS,
            REQUIRED_GATE_CHECKS,
        )
    )
    if set(checks) not in tuple(set(required) for required in supported_check_sets):
        raise CandidateFreezeError(f"{location}.checks has an unsupported compatibility contract")
    for name, result in checks.items():
        _text(name, f"{location}.checks key")
        if _strict_bool(result, f"{location}.checks.{name}") is not True:
            raise CandidateFreezeError(f"{location}.checks.{name} must pass")


def _validate_dataset(
    value: Any, location: str, candidate_schema: Mapping[str, Any]
) -> dict[str, Any]:
    dataset = _object(value, location)
    expected_keys = {
        "manifest",
        "dataset_yaml",
        "managed_files_sha256",
        "managed_file_count",
        "counts",
    }
    if candidate_schema == CANDIDATE_SCHEMA:
        expected_keys.add("taxonomy")
    _expect_keys(
        dataset,
        expected_keys,
        location,
    )
    manifest = _object(dataset["manifest"], f"{location}.manifest")
    _expect_keys(manifest, {"sha256", "size_bytes", "schema"}, f"{location}.manifest")
    _sha256(manifest["sha256"], f"{location}.manifest.sha256")
    _strict_int(manifest["size_bytes"], f"{location}.manifest.size_bytes", minimum=1)
    dataset_schema = _supported_schema(
        manifest["schema"],
        (LEGACY_DATASET_SCHEMA, DATASET_SCHEMA),
        f"{location}.manifest.schema",
    )
    if candidate_schema == LEGACY_CANDIDATE_SCHEMA and dataset_schema != LEGACY_DATASET_SCHEMA:
        raise CandidateFreezeError("legacy candidate receipt must use a schema-v2 dataset")
    dataset_yaml = _object(dataset["dataset_yaml"], f"{location}.dataset_yaml")
    _expect_keys(dataset_yaml, {"sha256", "size_bytes"}, f"{location}.dataset_yaml")
    _sha256(dataset_yaml["sha256"], f"{location}.dataset_yaml.sha256")
    _strict_int(dataset_yaml["size_bytes"], f"{location}.dataset_yaml.size_bytes", minimum=1)
    _sha256(dataset["managed_files_sha256"], f"{location}.managed_files_sha256")
    _strict_int(dataset["managed_file_count"], f"{location}.managed_file_count", minimum=1)
    _object(dataset["counts"], f"{location}.counts")
    if candidate_schema == CANDIDATE_SCHEMA:
        taxonomy = _validate_taxonomy_contract(dataset["taxonomy"], f"{location}.taxonomy")
        expected_taxonomy = _taxonomy_contract(
            REQUIRED_LABELS if dataset_schema == LEGACY_DATASET_SCHEMA else MODEL_LABELS
        )
        if not _same_contract(taxonomy, expected_taxonomy):
            raise CandidateFreezeError(
                f"{location}.taxonomy differs from the dataset schema contract"
            )
    return dataset


def _validate_base_weights(value: Any, location: str) -> dict[str, Any]:
    weights = _object(value, location)
    _expect_keys(weights, {"file_name", "model_family", "sha256", "size_bytes"}, location)
    if weights["file_name"] != "yolo11n.pt":
        raise CandidateFreezeError(f"{location}.file_name must be 'yolo11n.pt'")
    if weights["model_family"] != "YOLO11n":
        raise CandidateFreezeError(f"{location}.model_family must be 'YOLO11n'")
    _sha256(weights["sha256"], f"{location}.sha256")
    _strict_int(weights["size_bytes"], f"{location}.size_bytes", minimum=1)
    return weights


def _validate_pretrained_transfer(
    value: Any,
    *,
    base_weights: Mapping[str, Any],
    taxonomy: Mapping[str, Any],
    location: str,
) -> None:
    evidence = _object(value, location)
    _expect_keys(
        evidence,
        {
            "schema",
            "source_model",
            "target",
            "matched_rows",
            "matched_row_count",
            "target_row_count",
            "runtime_observation",
        },
        location,
    )
    if evidence["schema"] != {
        "name": "roadlabelops.pretrained-class-head-transfer",
        "version": 1,
    }:
        raise CandidateFreezeError(f"{location}.schema is unsupported")
    source = _object(evidence["source_model"], f"{location}.source_model")
    _expect_keys(
        source,
        {"family", "base_weights_sha256", "class_count", "names_sha256"},
        f"{location}.source_model",
    )
    if source["family"] != "YOLO11n" or source["base_weights_sha256"] != base_weights["sha256"]:
        raise CandidateFreezeError(f"{location}.source_model differs from base weights")
    source_count = _strict_int(
        source["class_count"], f"{location}.source_model.class_count", minimum=8
    )
    _sha256(source["names_sha256"], f"{location}.source_model.names_sha256")
    target = _object(evidence["target"], f"{location}.target")
    _expect_keys(
        target,
        {"class_count", "model_names", "canonical_names"},
        f"{location}.target",
    )
    if target != {
        "class_count": 8,
        "model_names": taxonomy["model_names"],
        "canonical_names": taxonomy["canonical_names"],
    }:
        raise CandidateFreezeError(f"{location}.target differs from the training taxonomy")
    if (
        _strict_int(evidence["matched_row_count"], f"{location}.matched_row_count") != 8
        or _strict_int(evidence["target_row_count"], f"{location}.target_row_count") != 8
    ):
        raise CandidateFreezeError(f"{location} must prove an 8/8 classifier-row transfer")
    rows = _list(evidence["matched_rows"], f"{location}.matched_rows")
    if len(rows) != 8:
        raise CandidateFreezeError(f"{location}.matched_rows must contain eight rows")
    source_ids: set[int] = set()
    for target_id, (model_name, canonical_name) in enumerate(
        zip(taxonomy["model_names"], taxonomy["canonical_names"], strict=True)
    ):
        row = _object(rows[target_id], f"{location}.matched_rows[{target_id}]")
        _expect_keys(
            row,
            {
                "target_id",
                "target_model_name",
                "canonical_name",
                "source_id",
                "source_model_name",
            },
            f"{location}.matched_rows[{target_id}]",
        )
        source_id = _strict_int(
            row["source_id"], f"{location}.matched_rows[{target_id}].source_id", minimum=0
        )
        if (
            row["target_id"] != target_id
            or row["target_model_name"] != model_name
            or row["canonical_name"] != canonical_name
            or row["source_model_name"] != model_name
            or source_id >= source_count
            or source_id in source_ids
        ):
            raise CandidateFreezeError(f"{location}.matched_rows[{target_id}] is inconsistent")
        source_ids.add(source_id)
    runtime = _object(evidence["runtime_observation"], f"{location}.runtime_observation")
    _expect_keys(
        runtime,
        {
            "verification_mode",
            "message",
            "message_sha256",
            "matched_row_count",
            "target_row_count",
        },
        f"{location}.runtime_observation",
    )
    if runtime["verification_mode"] not in {"ultralytics_logger", "injected_test_double"}:
        raise CandidateFreezeError(f"{location}.runtime_observation mode is unsupported")
    message = _text(runtime["message"], f"{location}.runtime_observation.message")
    if (
        re.fullmatch(
            r"Remapped 8/8 (?:decoder )?cls head rows from pretrained weights by class name",
            message,
        )
        is None
    ):
        raise CandidateFreezeError(f"{location}.runtime_observation message is inconsistent")
    if (
        hashlib.sha256(message.encode("utf-8")).hexdigest()
        != _sha256(runtime["message_sha256"], f"{location}.runtime_observation.message_sha256")
        or runtime["matched_row_count"] != 8
        or runtime["target_row_count"] != 8
    ):
        raise CandidateFreezeError(f"{location}.runtime_observation binding differs")


def _validate_resolved_args(
    value: Any, protocol: Mapping[str, Any], location: str
) -> tuple[int, dict[str, Any]]:
    resolved = _object(value, location)
    fields = (
        LEGACY_TRAINING_FIELDS if protocol["schema"] == LEGACY_PROTOCOL_SCHEMA else TRAINING_FIELDS
    )
    expected = {"model_family", "seed", *fields}
    _expect_keys(resolved, expected, location)
    if resolved["model_family"] != protocol["model_family"]:
        raise CandidateFreezeError(f"{location}.model_family differs from protocol")
    seed = _strict_int(resolved["seed"], f"{location}.seed", minimum=0, maximum=2**31 - 1)
    for field in fields:
        if _canonical_bytes(resolved[field]) != _canonical_bytes(protocol["training"][field]):
            raise CandidateFreezeError(f"{location}.{field} differs from protocol.training")
    without_seed = {key: deepcopy(item) for key, item in resolved.items() if key != "seed"}
    return seed, without_seed


def _validate_metrics(
    value: Any, location: str, candidate_schema: Mapping[str, Any]
) -> dict[str, float]:
    metrics = _object(value, location)
    _expect_keys(metrics, {"aggregate", "per_class"}, location)
    aggregate = _object(metrics["aggregate"], f"{location}.aggregate")
    _expect_keys(aggregate, set(AGGREGATE_METRICS), f"{location}.aggregate")
    validated = {
        metric: _finite_metric(aggregate[metric], f"{location}.aggregate.{metric}")
        for metric in AGGREGATE_METRICS
    }
    # Ultralytics 8.4.135 defines detection fitness as mAP50-95 (weights
    # [0, 0, 0, 1] over precision, recall, mAP50, and mAP50-95).  Keep this
    # receipt check aligned with the dependency version frozen by the protocol.
    expected_fitness = validated["map50_95"]
    if not math.isclose(validated["fitness"], expected_fitness, rel_tol=0.0, abs_tol=1e-12):
        raise CandidateFreezeError(f"{location}.aggregate.fitness is inconsistent")
    per_class = _object(metrics["per_class"], f"{location}.per_class")
    _expect_keys(per_class, set(REQUIRED_LABELS), f"{location}.per_class")
    support_aware: bool | None = None
    for class_id, label in enumerate(REQUIRED_LABELS):
        record_location = f"{location}.per_class.{label}"
        record = _object(per_class[label], record_location)
        legacy_fields = {"class_id", *RANKING_METRICS}
        current_fields = legacy_fields | {"status", "support_count"}
        if candidate_schema == LEGACY_CANDIDATE_SCHEMA:
            _expect_keys(record, legacy_fields, record_location)
            record_is_support_aware = False
        elif set(record) == current_fields:
            record_is_support_aware = True
        elif set(record) == legacy_fields:
            record_is_support_aware = False
        else:
            _expect_keys(record, current_fields, record_location)
            raise AssertionError("unreachable")
        if support_aware is None:
            support_aware = record_is_support_aware
        elif support_aware is not record_is_support_aware:
            raise CandidateFreezeError(f"{location}.per_class mixes metric record contracts")
        if _strict_int(record["class_id"], f"{record_location}.class_id") != class_id:
            raise CandidateFreezeError(f"{record_location}.class_id violates the class contract")
        if record_is_support_aware:
            status = _text(record["status"], f"{record_location}.status")
            support_count = _strict_int(
                record["support_count"], f"{record_location}.support_count", minimum=0
            )
            if status != "evaluable" or support_count == 0:
                raise CandidateFreezeError(
                    f"{record_location} is not evaluable and cannot enter direct candidate freeze"
                )
        for metric in RANKING_METRICS:
            _finite_metric(record[metric], f"{record_location}.{metric}")
    return validated


def _validate_best_epoch(
    value: Any,
    *,
    epochs: int,
    location: str,
) -> None:
    best = _object(value, location)
    _expect_keys(best, {"index", "number", "selection_fitness", "epochs_recorded"}, location)
    index = _strict_int(best["index"], f"{location}.index", minimum=0)
    number = _strict_int(best["number"], f"{location}.number", minimum=1)
    if number != index + 1 or number > epochs:
        raise CandidateFreezeError(f"{location} has an invalid epoch index/number")
    _finite_metric(best["selection_fitness"], f"{location}.selection_fitness")
    epochs_recorded = _strict_int(best["epochs_recorded"], f"{location}.epochs_recorded", minimum=1)
    if epochs_recorded > epochs or number > epochs_recorded:
        raise CandidateFreezeError(f"{location}.epochs_recorded is inconsistent")


def _validate_artifacts(value: Any, location: str) -> dict[str, dict[str, Any]]:
    artifacts = _object(value, location)
    _expect_keys(artifacts, set(ARTIFACT_PATHS), location)
    for name, required_path in ARTIFACT_PATHS.items():
        record_location = f"{location}.{name}"
        record = _object(artifacts[name], record_location)
        _expect_keys(record, {"path", "sha256", "size_bytes"}, record_location)
        relative = _relative_artifact_path(record["path"], f"{record_location}.path")
        if relative.as_posix() != required_path:
            raise CandidateFreezeError(
                f"{record_location}.path must be the managed path {required_path!r}"
            )
        _sha256(record["sha256"], f"{record_location}.sha256")
        _strict_int(record["size_bytes"], f"{record_location}.size_bytes", minimum=1)
    return artifacts


def _validate_holdout(value: Any, location: str) -> None:
    holdout = _object(value, location)
    _expect_keys(holdout, {"input_read", "statement"}, location)
    if _strict_bool(holdout["input_read"], f"{location}.input_read") is not False:
        raise CandidateFreezeError(f"{location}.input_read must be false")
    if holdout["statement"] != NO_HOLDOUT_STATEMENT:
        raise CandidateFreezeError(f"{location}.statement is not the required no-holdout statement")


def _validate_environment(value: Any, location: str) -> None:
    environment = _object(value, location)
    packages = _object(environment.get("packages"), f"{location}.packages")
    ultralytics_version = _text(packages.get("ultralytics"), f"{location}.packages.ultralytics")
    if ultralytics_version != EXPECTED_ULTRALYTICS_VERSION:
        raise CandidateFreezeError(
            f"{location}.packages.ultralytics must be {EXPECTED_ULTRALYTICS_VERSION!r}"
        )


def _read_receipt(root_fd: int, root: Path) -> tuple[dict[str, Any], str, int]:
    relative = PurePosixPath("receipt.json")
    descriptor = _open_relative_regular(root_fd, relative, f"{root}/receipt.json")
    try:
        encoded, metadata = _read_all_stable(
            descriptor, f"{root}/receipt.json", maximum=MAX_RECEIPT_BYTES
        )
    finally:
        os.close(descriptor)
    return (
        _parse_receipt(encoded, f"{root}/receipt.json"),
        hashlib.sha256(encoded).hexdigest(),
        metadata.st_size,
    )


def _validate_candidate(path: Path) -> ValidatedCandidate:
    root, root_fd, root_identity = _open_candidate_root(path)
    try:
        receipt, receipt_sha, receipt_size = _read_receipt(root_fd, root)
        required_receipt_keys = {
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
        }
        has_pretrained_transfer = "pretrained_transfer" in receipt
        if has_pretrained_transfer:
            required_receipt_keys.add("pretrained_transfer")
        _expect_keys(receipt, required_receipt_keys, f"{root}/receipt.json")
        candidate_schema = _supported_schema(
            receipt["schema"],
            (LEGACY_CANDIDATE_SCHEMA, CANDIDATE_SCHEMA),
            "receipt.schema",
        )
        _validate_gate(receipt["gate"], "receipt.gate", candidate_schema)
        transfer_gate_present = (
            "pretrained_class_head_transfer_verified"
            in _object(receipt["gate"], "receipt.gate")["checks"]
        )
        if transfer_gate_present != has_pretrained_transfer:
            raise CandidateFreezeError(
                "receipt pretrained-transfer evidence and gate check must appear together"
            )
        protocol, protocol_sha = _validate_protocol(
            receipt["protocol"], receipt["protocol_sha256"], "receipt.protocol"
        )
        expected_protocol_schema = (
            LEGACY_PROTOCOL_SCHEMA
            if candidate_schema == LEGACY_CANDIDATE_SCHEMA
            else PROTOCOL_SCHEMA
        )
        if protocol["schema"] != expected_protocol_schema:
            raise CandidateFreezeError(
                "candidate receipt and training protocol schema versions are incompatible"
            )
        inputs = _object(receipt["inputs"], "receipt.inputs")
        _expect_keys(inputs, {"dataset", "base_weights"}, "receipt.inputs")
        dataset = _validate_dataset(inputs["dataset"], "receipt.inputs.dataset", candidate_schema)
        if candidate_schema == CANDIDATE_SCHEMA and not _same_contract(
            protocol["taxonomy"], dataset["taxonomy"]
        ):
            raise CandidateFreezeError(
                "training protocol taxonomy differs from the dataset taxonomy"
            )
        base_weights = _validate_base_weights(inputs["base_weights"], "receipt.inputs.base_weights")
        if has_pretrained_transfer:
            _validate_pretrained_transfer(
                receipt["pretrained_transfer"],
                base_weights=base_weights,
                taxonomy=protocol["taxonomy"],
                location="receipt.pretrained_transfer",
            )
        seed, args_without_seed = _validate_resolved_args(
            receipt["resolved_args"], protocol, "receipt.resolved_args"
        )
        metrics = _validate_metrics(receipt["metrics"], "receipt.metrics", candidate_schema)
        _validate_best_epoch(
            receipt["best_epoch"],
            epochs=protocol["training"]["epochs"],
            location="receipt.best_epoch",
        )
        artifacts = _validate_artifacts(receipt["artifacts"], "receipt.artifacts")
        _validate_holdout(receipt["holdout"], "receipt.holdout")
        _object(receipt["timestamps"], "receipt.timestamps")
        _validate_environment(receipt["environment"], "receipt.environment")
        if _strict_bool(receipt["mutation_performed"], "receipt.mutation_performed") is not True:
            raise CandidateFreezeError("receipt.mutation_performed must be true")

        best = artifacts["best_weights"]
        best_path = _relative_artifact_path(best["path"], "receipt.artifacts.best_weights.path")
        best_fd = _open_relative_regular(root_fd, best_path, f"{root}/{best_path.as_posix()}")
        try:
            actual_sha, actual_size = _hash_open_file(best_fd, f"{root}/{best_path.as_posix()}")
        finally:
            os.close(best_fd)
        expected_sha = _sha256(best["sha256"], "receipt.artifacts.best_weights.sha256")
        expected_size = _strict_int(
            best["size_bytes"], "receipt.artifacts.best_weights.size_bytes", minimum=1
        )
        if actual_sha != expected_sha:
            raise CandidateFreezeError(f"{root}/weights/best.pt SHA-256 differs from receipt")
        if actual_size != expected_size:
            raise CandidateFreezeError(f"{root}/weights/best.pt size differs from receipt")
    finally:
        os.close(root_fd)

    return ValidatedCandidate(
        root=root,
        root_identity=root_identity,
        seed=seed,
        receipt_sha256=receipt_sha,
        receipt_size_bytes=receipt_size,
        protocol=deepcopy(protocol),
        protocol_sha256=protocol_sha,
        resolved_args_without_seed=deepcopy(args_without_seed),
        dataset=deepcopy(dataset),
        base_weights=deepcopy(base_weights),
        metrics=metrics,
        best_weights_path=best_path.as_posix(),
        best_weights_sha256=actual_sha,
        best_weights_size_bytes=actual_size,
    )


def _same_contract(first: Any, second: Any) -> bool:
    return _canonical_bytes(first) == _canonical_bytes(second)


def _validate_common_contract(candidates: Sequence[ValidatedCandidate]) -> None:
    seeds = [candidate.seed for candidate in candidates]
    if len(set(seeds)) != len(seeds):
        raise CandidateFreezeError("candidate seeds must be distinct")
    if set(seeds) != EXPECTED_SEEDS:
        raise CandidateFreezeError(
            f"candidate seeds must be exactly {sorted(EXPECTED_SEEDS)}; got {sorted(seeds)}"
        )
    baseline = candidates[0]
    comparisons = (
        ("protocol", baseline.protocol, lambda item: item.protocol),
        (
            "resolved configuration except seed",
            baseline.resolved_args_without_seed,
            lambda item: item.resolved_args_without_seed,
        ),
        ("dataset hashes", baseline.dataset, lambda item: item.dataset),
        ("base YOLO11n weights", baseline.base_weights, lambda item: item.base_weights),
    )
    for location, expected, accessor in comparisons:
        for candidate in candidates[1:]:
            if not _same_contract(expected, accessor(candidate)):
                raise CandidateFreezeError(
                    f"candidate seed {candidate.seed} has a different {location} contract"
                )


def _assert_candidate_unchanged(candidate: ValidatedCandidate) -> None:
    root, root_fd, root_identity = _open_candidate_root(candidate.root)
    try:
        if root_identity != candidate.root_identity:
            raise CandidateFreezeError(f"candidate directory changed during selection: {root}")
        _, receipt_sha, receipt_size = _read_receipt(root_fd, root)
        if receipt_sha != candidate.receipt_sha256 or receipt_size != candidate.receipt_size_bytes:
            raise CandidateFreezeError(f"candidate receipt changed during selection: {root}")
        relative = _relative_artifact_path(
            candidate.best_weights_path, "candidate best_weights_path"
        )
        descriptor = _open_relative_regular(root_fd, relative, f"{root}/{relative.as_posix()}")
        try:
            digest, size_bytes = _hash_open_file(descriptor, f"{root}/{relative.as_posix()}")
        finally:
            os.close(descriptor)
        if (
            digest != candidate.best_weights_sha256
            or size_bytes != candidate.best_weights_size_bytes
        ):
            raise CandidateFreezeError(f"candidate best.pt changed during selection: {root}")
    finally:
        os.close(root_fd)


def _source_record(candidate: ValidatedCandidate) -> dict[str, Any]:
    return {
        "kind": "content_bound_candidate_directory",
        "directory_name": candidate.root.name,
        "seed": candidate.seed,
        "receipt": {
            "path": "receipt.json",
            "sha256": candidate.receipt_sha256,
            "size_bytes": candidate.receipt_size_bytes,
        },
    }


def _ranking_record(candidate: ValidatedCandidate, rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "seed": candidate.seed,
        "metrics": {metric: candidate.metrics[metric] for metric in AGGREGATE_METRICS},
        "source": _source_record(candidate),
        "best_weights": {
            "path": candidate.best_weights_path,
            "sha256": candidate.best_weights_sha256,
            "size_bytes": candidate.best_weights_size_bytes,
        },
        "contract": {
            "protocol_sha256": candidate.protocol_sha256,
            "dataset": deepcopy(candidate.dataset),
            "base_weights": deepcopy(candidate.base_weights),
            "taxonomy": list(REQUIRED_LABELS),
        },
    }


def _selection_receipt(
    ranked: Sequence[ValidatedCandidate], copied_sha256: str, copied_size_bytes: int
) -> dict[str, Any]:
    selected = ranked[0]
    rankings = [_ranking_record(candidate, rank) for rank, candidate in enumerate(ranked, 1)]
    return {
        "schema": deepcopy(FROZEN_SCHEMA),
        "selection_order": list(SELECTION_ORDER),
        "selected_seed": selected.seed,
        "selected_source": _source_record(selected),
        "selected_metrics": {metric: selected.metrics[metric] for metric in AGGREGATE_METRICS},
        "selected_weight": {
            "path": "best.pt",
            "sha256": copied_sha256,
            "size_bytes": copied_size_bytes,
        },
        "candidate_rankings": rankings,
        "contracts": {
            "candidate_count": 3,
            "protocol": deepcopy(selected.protocol),
            "protocol_sha256": selected.protocol_sha256,
            "resolved_args_except_seed": deepcopy(selected.resolved_args_without_seed),
            "dataset": deepcopy(selected.dataset),
            "base_weights": deepcopy(selected.base_weights),
            "taxonomy": list(REQUIRED_LABELS),
        },
        "holdout_input_read": False,
        "holdout_statement": NO_HOLDOUT_STATEMENT,
    }


def _write_bytes_new(path: Path, encoded: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _copy_selected_weight(candidate: ValidatedCandidate, target: Path) -> tuple[str, int]:
    root, root_fd, root_identity = _open_candidate_root(candidate.root)
    try:
        if root_identity != candidate.root_identity:
            raise CandidateFreezeError(f"selected candidate directory changed: {root}")
        relative = _relative_artifact_path(
            candidate.best_weights_path, "selected best_weights_path"
        )
        source_fd = _open_relative_regular(root_fd, relative, f"{root}/{relative.as_posix()}")
        before = os.fstat(source_fd)
        digest = hashlib.sha256()
        total = 0
        try:
            with target.open("xb") as destination:
                while chunk := os.read(source_fd, 1024 * 1024):
                    destination.write(chunk)
                    digest.update(chunk)
                    total += len(chunk)
                destination.flush()
                os.fsync(destination.fileno())
        finally:
            after = os.fstat(source_fd)
            os.close(source_fd)
        if _stat_identity(before) != _stat_identity(after) or total != before.st_size:
            raise CandidateFreezeError("selected best.pt changed while it was being copied")
    finally:
        os.close(root_fd)
    copied_sha = digest.hexdigest()
    if copied_sha != candidate.best_weights_sha256:
        raise CandidateFreezeError("copied best.pt SHA-256 differs from selected candidate")
    if total != candidate.best_weights_size_bytes:
        raise CandidateFreezeError("copied best.pt size differs from selected candidate")
    return copied_sha, total


def _atomic_publish_directory_no_replace(staging: Path, output: Path) -> None:
    """Atomically publish a directory while refusing every existing target."""

    source_bytes = os.fsencode(staging)
    target_bytes = os.fsencode(output)
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        rename_exclusive = libc.renamex_np
        rename_exclusive.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename_exclusive.restype = ctypes.c_int
        result = rename_exclusive(source_bytes, target_bytes, 0x00000004)  # RENAME_EXCL
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        rename_no_replace = libc.renameat2
        rename_no_replace.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename_no_replace.restype = ctypes.c_int
        result = rename_no_replace(
            -100,
            source_bytes,
            -100,
            target_bytes,
            0x00000001,  # AT_FDCWD, RENAME_NOREPLACE
        )
    else:
        raise CandidateFreezeError(
            "atomic no-replace directory publication is unsupported on this platform"
        )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise CandidateFreezeError(f"output directory already exists: {output}")
    raise OSError(error_number, os.strerror(error_number), str(output))


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def freeze_model_candidate(
    candidate_directories: Sequence[Path | str], output_directory: Path | str
) -> dict[str, Any]:
    """Validate, rank, and immutably publish one of three training candidates."""

    if isinstance(candidate_directories, (str, bytes, Path)):
        raise CandidateFreezeError("candidate_directories must contain exactly three paths")
    paths = [Path(path) for path in candidate_directories]
    if len(paths) != 3:
        raise CandidateFreezeError("exactly three candidate directories are required")

    candidates = [_validate_candidate(path) for path in paths]
    if len({candidate.root_identity for candidate in candidates}) != 3:
        raise CandidateFreezeError("candidate directories must be three distinct directories")
    _validate_common_contract(candidates)
    ranked = sorted(candidates, key=lambda candidate: candidate.rank_key)

    # Re-check every content-bound source immediately before staging the result.
    for candidate in candidates:
        _assert_candidate_unchanged(candidate)

    output_input = Path(output_directory)
    _reject_lexical_traversal(output_input, "output directory")
    output = _absolute_lexical(output_input)
    if output == output.parent:
        raise CandidateFreezeError("output directory must not be a filesystem root")
    for first_index, first in enumerate(candidates):
        if output.is_relative_to(first.root) or first.root.is_relative_to(output):
            raise CandidateFreezeError("output must not overlap a candidate directory")
        for second in candidates[first_index + 1 :]:
            if first.root.is_relative_to(second.root) or second.root.is_relative_to(first.root):
                raise CandidateFreezeError("candidate directories must not overlap")
    _reject_existing_symlink_prefix(output.parent, "output parent")
    output.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(output.parent, "output parent", allow_missing_leaf=False)
    try:
        os.lstat(output)
    except FileNotFoundError:
        pass
    else:
        raise CandidateFreezeError(f"output directory already exists: {output}")

    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=str(output.parent)))
    published = False
    try:
        copied_sha, copied_size = _copy_selected_weight(ranked[0], staging / "best.pt")
        receipt = _selection_receipt(ranked, copied_sha, copied_size)
        _write_bytes_new(staging / "receipt.json", _json_bytes(receipt))
        _fsync_directory(staging)
        _atomic_publish_directory_no_replace(staging, output)
        published = True
        _fsync_directory(output.parent)
    finally:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)

    return {
        "output": str(output),
        "receipt": str(output / "receipt.json"),
        "best_weights": str(output / "best.pt"),
        "selected_seed": ranked[0].seed,
        "selected_metrics": {metric: ranked[0].metrics[metric] for metric in AGGREGATE_METRICS},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        "--candidate-dir",
        dest="candidates",
        type=Path,
        action="append",
        required=True,
        help="candidate output directory; provide exactly three times",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = freeze_model_candidate(args.candidates, args.output)
    except (CandidateFreezeError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
