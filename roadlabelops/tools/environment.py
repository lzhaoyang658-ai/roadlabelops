from __future__ import annotations

import os
import platform
import shutil
import sys
import tempfile
import zipfile
from collections.abc import Callable
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from ..models import ToolResult

SUPPORTED_CVAT_MAJOR_MINOR = (2, 74)


def _probe_writable_directory(path: Path) -> tuple[bool, str]:
    """Prove that the runtime can create, flush, and remove a file in the store."""

    if not path.exists():
        return False, "directory does not exist"
    if not path.is_dir():
        return False, "path is not a directory"
    probe_path: Path | None = None
    try:
        descriptor, raw_path = tempfile.mkstemp(prefix=".roadlabelops-ready-", dir=path)
        probe_path = Path(raw_path)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(b"ready")
            handle.flush()
            os.fsync(handle.fileno())
        probe_path.unlink()
        return True, str(path.resolve())
    except OSError as exc:
        if probe_path is not None:
            probe_path.unlink(missing_ok=True)
        return False, f"{type(exc).__name__}: store is not writable"


def _detector_check(
    detection_provider: str,
    detection_model: Path | str | None,
) -> dict[str, Any]:
    provider = detection_provider.strip().lower()
    if provider == "mock":
        return {
            "ok": True,
            "provider": "mock",
            "model": "deterministic-fixture",
            "real_flow_capable": False,
            "value": "available for Demo only",
        }
    if provider not in {"yolo", "ultralytics"}:
        return {
            "ok": False,
            "provider": provider,
            "model": str(detection_model or ""),
            "real_flow_capable": False,
            "value": "unsupported provider; use yolo, ultralytics, or mock",
        }

    try:
        package_available = find_spec("ultralytics") is not None
    except (ImportError, ValueError):
        package_available = False
    model_path = Path(detection_model).expanduser().resolve() if detection_model else None
    model_error: str | None = None
    try:
        model_available = bool(
            model_path
            and model_path.is_file()
            and model_path.suffix.lower() == ".pt"
            and model_path.stat().st_size > 0
            and zipfile.is_zipfile(model_path)
        )
        if model_available and model_path is not None:
            with zipfile.ZipFile(model_path) as archive:
                names = archive.namelist()
                corrupt_member = archive.testzip()
            has_pickle = any(name.endswith("/data.pkl") or name == "data.pkl" for name in names)
            has_tensors = any("/data/" in name or name.startswith("data/") for name in names)
            if corrupt_member or not has_pickle or not has_tensors:
                model_available = False
                model_error = "configured .pt is not a valid PyTorch checkpoint archive"
    except (OSError, zipfile.BadZipFile):
        model_available = False
        model_error = "configured .pt could not be validated"
    reasons = []
    if not package_available:
        reasons.append("ultralytics package is not installed")
    if not model_available:
        reasons.append(
            model_error
            or "configured local .pt weights are missing or not a valid checkpoint"
        )
    return {
        "ok": package_available and model_available,
        "provider": provider,
        "model": str(model_path) if model_path else None,
        "package_available": package_available,
        "model_available": model_available,
        "real_flow_capable": package_available and model_available,
        "value": "ready" if not reasons else "; ".join(reasons),
    }


def _cvat_connection_check(
    configured: bool,
    health_check: Callable[[], ToolResult] | None,
) -> dict[str, Any]:
    if not configured or health_check is None:
        return {
            "ok": False,
            "value": "not checked because CVAT is not fully configured",
        }
    try:
        health = health_check()
    except Exception as exc:
        return {
            "ok": False,
            "value": f"{type(exc).__name__}: CVAT health check failed",
        }
    if not health.ok:
        return {
            "ok": False,
            "value": str((health.error or {}).get("message", "CVAT health check failed")),
        }
    raw_version = health.data.get("version")
    try:
        version_parts = tuple(int(part) for part in str(raw_version).split(".")[:2])
    except ValueError:
        version_parts = ()
    compatible = version_parts == SUPPORTED_CVAT_MAJOR_MINOR
    return {
        "ok": compatible,
        "value": health.data,
        "compatible": compatible,
        "required": "2.74.x",
        **(
            {}
            if compatible
            else {"reason": "CVAT Server must be the verified 2.74.x line"}
        ),
    }


def check_environment(
    data_root: Path | str = "data",
    cvat_host: str | None = None,
    cvat_credentials_configured: bool = False,
    detection_provider: str = "mock",
    detection_model: Path | str | None = None,
    *,
    runtime_root: Path | str | None = None,
    cvat_health_check: Callable[[], ToolResult] | None = None,
) -> ToolResult:
    """Return separate Demo and dependency-complete real-flow readiness reports."""

    store_ok, store_value = _probe_writable_directory(Path(data_root))
    runtime_ok, runtime_value = _probe_writable_directory(
        Path(runtime_root) if runtime_root is not None else Path(data_root)
    )
    cvat_configured = bool(cvat_host and cvat_credentials_configured)
    checks: dict[str, dict[str, Any]] = {
        "python": {
            "ok": sys.version_info >= (3, 10),
            "value": platform.python_version(),
            "required": ">=3.10",
        },
        "store_writable": {"ok": store_ok, "value": store_value},
        "runtime_writable": {"ok": runtime_ok, "value": runtime_value},
        "ffmpeg": {
            "ok": shutil.which("ffmpeg") is not None,
            "value": shutil.which("ffmpeg"),
        },
        "ffprobe": {
            "ok": shutil.which("ffprobe") is not None,
            "value": shutil.which("ffprobe"),
        },
        "cvat_configured": {
            "ok": cvat_configured,
            "value": cvat_host if cvat_configured else "host or credentials not configured",
        },
        "cvat_connection": _cvat_connection_check(cvat_configured, cvat_health_check),
        "detector": _detector_check(detection_provider, detection_model),
    }

    demo_requirements = ("python", "store_writable", "runtime_writable")
    real_requirements = (
        "python",
        "store_writable",
        "runtime_writable",
        "ffmpeg",
        "ffprobe",
        "cvat_configured",
        "cvat_connection",
    )
    demo_blocking = [name for name in demo_requirements if not checks[name]["ok"]]
    real_blocking = [name for name in real_requirements if not checks[name]["ok"]]
    if not checks["detector"]["ok"] or not checks["detector"]["real_flow_capable"]:
        real_blocking.append("detector")

    demo_ready = not demo_blocking
    real_flow_ready = not real_blocking
    mode = "real" if real_flow_ready else "demo_only" if demo_ready else "unavailable"
    return ToolResult.success(
        {
            "checks": checks,
            "ready": real_flow_ready,
            "real_flow_ready": real_flow_ready,
            "demo_ready": demo_ready,
            "mode": mode,
            "blocking": real_blocking,
            "demo_blocking": demo_blocking,
        }
    )
