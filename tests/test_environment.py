from pathlib import Path

from roadlabelops.models import ToolResult
from roadlabelops.tools import environment


def _configure_system_tools(monkeypatch) -> None:
    monkeypatch.setattr(
        environment.shutil,
        "which",
        lambda name: f"/usr/local/bin/{name}" if name in {"ffmpeg", "ffprobe"} else None,
    )


def _write_checkpoint(path: Path) -> None:
    import zipfile

    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("model/data.pkl", b"metadata")
        archive.writestr("model/data/0", b"tensor")


def test_mock_configuration_is_demo_ready_but_not_real_ready(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _configure_system_tools(monkeypatch)

    result = environment.check_environment(tmp_path, detection_provider="mock")

    assert result.ok
    assert result.data["demo_ready"] is True
    assert result.data["real_flow_ready"] is False
    assert result.data["mode"] == "demo_only"
    assert result.data["blocking"] == [
        "cvat_configured",
        "cvat_connection",
        "detector",
    ]
    assert result.data["checks"]["detector"]["ok"] is True
    assert result.data["checks"]["detector"]["real_flow_capable"] is False


def test_real_flow_requires_writable_store_cvat_tools_package_and_local_model(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _configure_system_tools(monkeypatch)
    monkeypatch.setattr(environment, "find_spec", lambda name: object())
    model = tmp_path / "road.pt"
    _write_checkpoint(model)

    result = environment.check_environment(
        tmp_path,
        "https://cvat.example",
        True,
        "yolo",
        model,
        cvat_health_check=lambda: ToolResult.success({"version": "2.74.0"}),
    )

    assert result.data["ready"] is True
    assert result.data["real_flow_ready"] is True
    assert result.data["demo_ready"] is True
    assert result.data["mode"] == "real"
    assert result.data["blocking"] == []
    assert result.data["checks"]["cvat_connection"]["value"] == {
        "version": "2.74.0"
    }


def test_missing_local_model_blocks_real_readiness(tmp_path: Path, monkeypatch) -> None:
    _configure_system_tools(monkeypatch)
    monkeypatch.setattr(environment, "find_spec", lambda name: object())

    result = environment.check_environment(
        tmp_path,
        "http://localhost:8080",
        True,
        "yolo",
        tmp_path / "missing.pt",
        cvat_health_check=lambda: ToolResult.success({"version": "2.74.0"}),
    )

    assert result.data["real_flow_ready"] is False
    assert result.data["blocking"] == ["detector"]
    assert result.data["checks"]["detector"]["model_available"] is False


def test_failed_cvat_health_blocks_real_readiness(tmp_path: Path, monkeypatch) -> None:
    _configure_system_tools(monkeypatch)
    monkeypatch.setattr(environment, "find_spec", lambda name: object())
    model = tmp_path / "road.pt"
    _write_checkpoint(model)

    result = environment.check_environment(
        tmp_path,
        "http://localhost:8080",
        True,
        "yolo",
        model,
        cvat_health_check=lambda: ToolResult.failure(
            "CVAT_UNAVAILABLE",
            "CVAT is unavailable or authentication failed",
        ),
    )

    assert result.data["real_flow_ready"] is False
    assert result.data["blocking"] == ["cvat_connection"]


def test_corrupt_checkpoint_blocks_real_readiness(tmp_path: Path, monkeypatch) -> None:
    _configure_system_tools(monkeypatch)
    monkeypatch.setattr(environment, "find_spec", lambda name: object())
    model = tmp_path / "road.pt"
    model.write_bytes(b"not a torch checkpoint")

    result = environment.check_environment(
        tmp_path,
        "http://localhost:8080",
        True,
        "yolo",
        model,
        cvat_health_check=lambda: ToolResult.success({"version": "2.74.0"}),
    )

    assert result.data["real_flow_ready"] is False
    assert result.data["blocking"] == ["detector"]
    assert result.data["checks"]["detector"]["model_available"] is False


def test_unsupported_cvat_version_blocks_real_readiness(
    tmp_path: Path, monkeypatch
) -> None:
    _configure_system_tools(monkeypatch)
    monkeypatch.setattr(environment, "find_spec", lambda name: object())
    model = tmp_path / "road.pt"
    _write_checkpoint(model)

    result = environment.check_environment(
        tmp_path,
        "http://localhost:8080",
        True,
        "yolo",
        model,
        cvat_health_check=lambda: ToolResult.success({"version": "2.75.0"}),
    )

    assert result.data["real_flow_ready"] is False
    assert result.data["blocking"] == ["cvat_connection"]
    assert result.data["checks"]["cvat_connection"]["compatible"] is False


def test_non_writable_store_blocks_demo_and_real_readiness(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _configure_system_tools(monkeypatch)
    monkeypatch.setattr(
        environment,
        "_probe_writable_directory",
        lambda path: (False, "PermissionError: store is not writable"),
    )

    result = environment.check_environment(tmp_path, detection_provider="mock")

    assert result.data["demo_ready"] is False
    assert result.data["real_flow_ready"] is False
    assert result.data["mode"] == "unavailable"
    assert "store_writable" in result.data["demo_blocking"]
