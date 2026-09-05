from pathlib import Path

import pytest
from pydantic import SecretStr

from roadlabelops.settings import Settings, build_cvat_adapter, load_settings


def test_cvat_adapter_requires_credentials() -> None:
    settings = Settings(_env_file=None, cvat_host="http://localhost:8080")
    assert not settings.cvat_credentials_configured
    assert build_cvat_adapter(settings) is None


def test_cvat_adapter_uses_local_credentials() -> None:
    settings = Settings(
        _env_file=None,
        cvat_host="http://localhost:8080",
        cvat_username="annotator",
        cvat_password=SecretStr("local-secret"),
    )
    adapter = build_cvat_adapter(settings)
    assert adapter is not None
    assert adapter.config.username == "annotator"
    assert adapter.config.password == "local-secret"


def test_cors_origins_are_normalized() -> None:
    settings = Settings(
        _env_file=None,
        roadlabelops_cors_origins="http://localhost:3100/, https://roadlabel.example ",
    )

    assert settings.cors_origins == [
        "http://localhost:3100",
        "https://roadlabel.example",
    ]


def test_relative_runtime_paths_are_anchored_to_project_directory(tmp_path: Path) -> None:
    project_dir = tmp_path / "checkout"
    settings = Settings(
        _env_file=None,
        roadlabelops_project_dir=project_dir,
        roadlabelops_data_dir="var/data",
        roadlabelops_runtime_dir="var/runtime",
        detection_model="models/road.pt",
    )

    assert settings.roadlabelops_data_dir == project_dir / "var/data"
    assert settings.roadlabelops_runtime_dir == project_dir / "var/runtime"
    assert settings.detection_model_path == project_dir / "models/road.pt"


def test_load_settings_uses_explicit_absolute_env_file_from_any_cwd(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_dir = tmp_path / "project"
    env_file = tmp_path / "production.env"
    env_file.write_text(
        f"ROADLABELOPS_PROJECT_DIR={project_dir}\n"
        "ROADLABELOPS_DATA_DIR=state\n"
        "ROADLABELOPS_RUNTIME_DIR=run-state\n"
        "DETECTION_MODEL=models/yolo11n.pt\n",
        encoding="utf-8",
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.setenv("ROADLABELOPS_ENV_FILE", str(env_file))

    settings = load_settings()

    assert settings.roadlabelops_data_dir == project_dir / "state"
    assert settings.roadlabelops_runtime_dir == project_dir / "run-state"
    assert settings.detection_model_path == project_dir / "models/yolo11n.pt"


def test_load_settings_rejects_relative_env_file(monkeypatch) -> None:
    monkeypatch.setenv("ROADLABELOPS_ENV_FILE", "relative.env")

    with pytest.raises(ValueError, match="must be an absolute path"):
        load_settings()


def test_load_settings_rejects_exposed_credentials_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    env_file = tmp_path / "production.env"
    env_file.write_text(
        "CVAT_HOST=https://cvat.example.com\nCVAT_ACCESS_TOKEN=local-secret\n",
        encoding="utf-8",
    )
    env_file.chmod(0o644)
    monkeypatch.setenv("ROADLABELOPS_ENV_FILE", str(env_file))

    with pytest.raises(ValueError, match="must have mode 0600"):
        load_settings()


def test_load_settings_accepts_private_credentials_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    env_file = tmp_path / "production.env"
    env_file.write_text(
        "CVAT_HOST=https://cvat.example.com\nCVAT_ACCESS_TOKEN=local-secret\n",
        encoding="utf-8",
    )
    env_file.chmod(0o600)
    monkeypatch.setenv("ROADLABELOPS_ENV_FILE", str(env_file))

    settings = load_settings()

    assert settings.cvat_credentials_configured


def test_load_settings_rejects_exposed_password_even_without_username(
    tmp_path: Path,
    monkeypatch,
) -> None:
    env_file = tmp_path / "incomplete.env"
    env_file.write_text("CVAT_PASSWORD=local-secret\n", encoding="utf-8")
    env_file.chmod(0o644)
    monkeypatch.setenv("ROADLABELOPS_ENV_FILE", str(env_file))

    with pytest.raises(ValueError, match="must have mode 0600"):
        load_settings()
