from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .tools.cvat import CvatAdapter, CvatConfig
from .tools.detection import run_mock_detection, run_yolo_detection

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    """Runtime configuration with paths anchored to one explicit project directory."""

    model_config = SettingsConfigDict(
        env_file=DEFAULT_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    roadlabelops_env: str = "development"
    roadlabelops_project_dir: Path = PROJECT_ROOT
    roadlabelops_data_dir: Path = Path("data")
    roadlabelops_runtime_dir: Path = Path("runtime")
    roadlabelops_cors_origins: str = "http://localhost:3100"
    roadlabelops_max_upload_bytes: int = Field(default=2 * 1024 * 1024 * 1024, ge=1)
    roadlabelops_upload_overhead_bytes: int = Field(default=1024 * 1024, ge=0)
    roadlabelops_max_video_duration_seconds: float = Field(default=5 * 60, gt=0)
    roadlabelops_absolute_max_video_duration_seconds: float = Field(
        default=2 * 60 * 60, gt=0
    )
    roadlabelops_max_scene_count: int = Field(default=720, ge=1)
    roadlabelops_max_split_output_bytes: int = Field(
        default=8 * 1024 * 1024 * 1024, ge=1
    )
    roadlabelops_max_video_width: int = Field(default=3840, ge=1)
    roadlabelops_max_video_height: int = Field(default=2160, ge=1)
    roadlabelops_max_video_fps: float = Field(default=60.0, gt=0)
    cvat_host: str | None = None
    cvat_username: str | None = None
    cvat_password: SecretStr | None = None
    cvat_access_token: SecretStr | None = None
    detection_provider: str = "mock"
    detection_model: str = "yolo11n"
    detection_confidence: float = 0.4
    detection_nms_iou: float = Field(default=0.75, ge=0.0, le=1.0)
    detection_rider_overlap: float = Field(default=0.25, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def resolve_runtime_paths(self) -> Settings:
        """Resolve application paths independently from the process working directory."""

        project_dir = self.roadlabelops_project_dir.expanduser()
        if not project_dir.is_absolute():
            project_dir = PROJECT_ROOT / project_dir
        self.roadlabelops_project_dir = project_dir.resolve()

        data_dir = self.roadlabelops_data_dir.expanduser()
        if not data_dir.is_absolute():
            data_dir = self.roadlabelops_project_dir / data_dir
        self.roadlabelops_data_dir = data_dir.resolve()

        runtime_dir = self.roadlabelops_runtime_dir.expanduser()
        if not runtime_dir.is_absolute():
            runtime_dir = self.roadlabelops_project_dir / runtime_dir
        self.roadlabelops_runtime_dir = runtime_dir.resolve()
        if (
            self.roadlabelops_max_video_duration_seconds
            > self.roadlabelops_absolute_max_video_duration_seconds
        ):
            raise ValueError(
                "ROADLABELOPS_MAX_VIDEO_DURATION_SECONDS cannot exceed the absolute hard limit"
            )
        return self

    @property
    def cvat_credentials_configured(self) -> bool:
        return bool(self.cvat_access_token or (self.cvat_username and self.cvat_password))

    @property
    def cors_origins(self) -> list[str]:
        """Return normalized, explicit browser origins from a comma-separated setting."""

        return [
            origin.strip().rstrip("/")
            for origin in self.roadlabelops_cors_origins.split(",")
            if origin.strip()
        ]

    @property
    def detection_model_path(self) -> Path:
        """Return the explicit local detector path; inference never resolves model aliases."""

        model_path = Path(self.detection_model).expanduser()
        if model_path.suffix.lower() != ".pt":
            model_path = model_path.with_suffix(".pt")
        if not model_path.is_absolute():
            model_path = self.roadlabelops_project_dir / model_path
        return model_path.resolve()


def load_settings() -> Settings:
    """Load the project .env, or one absolute file selected by the process environment."""

    configured = os.environ.get("ROADLABELOPS_ENV_FILE")
    if configured:
        env_file = Path(configured).expanduser()
        if not env_file.is_absolute():
            raise ValueError("ROADLABELOPS_ENV_FILE must be an absolute path")
    else:
        env_file = DEFAULT_ENV_FILE
    settings = Settings(_env_file=env_file)
    _require_private_env_file(env_file, settings)
    return settings


def _require_private_env_file(env_file: Path, settings: Settings) -> None:
    """Refuse credential-bearing configuration readable by another local account."""

    has_secret = bool(settings.cvat_password or settings.cvat_access_token)
    if os.name == "nt" or not has_secret or not env_file.exists():
        return
    try:
        exposed_bits = env_file.stat().st_mode & 0o077
    except OSError as exc:
        raise ValueError("Could not verify ROADLABELOPS_ENV_FILE permissions") from exc
    if exposed_bits:
        raise ValueError(
            "ROADLABELOPS_ENV_FILE contains CVAT credentials and must have mode 0600; "
            f"run: chmod 600 {env_file}"
        )


def build_cvat_adapter(settings: Settings) -> CvatAdapter | None:
    if not settings.cvat_host or not settings.cvat_credentials_configured:
        return None
    return CvatAdapter(
        CvatConfig(
            host=settings.cvat_host,
            username=settings.cvat_username,
            password=(
                settings.cvat_password.get_secret_value() if settings.cvat_password else None
            ),
            access_token=(
                settings.cvat_access_token.get_secret_value()
                if settings.cvat_access_token
                else None
            ),
        )
    )


def build_detection_runner(settings: Settings):
    def detect(scene_path: Path | str, frame_step: int = 5):
        provider = settings.detection_provider.strip().lower()
        if provider == "mock":
            return run_mock_detection(scene_path, settings.detection_confidence)
        if provider not in {"yolo", "ultralytics"}:
            from .models import ToolResult

            return ToolResult.failure(
                "DETECTOR_PROVIDER_UNSUPPORTED",
                f"Detection provider {settings.detection_provider!r} is not supported",
            )
        return run_yolo_detection(
            scene_path,
            model_name=str(settings.detection_model_path),
            confidence=settings.detection_confidence,
            frame_step=frame_step,
            nms_iou_threshold=settings.detection_nms_iou,
            rider_overlap_threshold=settings.detection_rider_overlap,
        )

    return detect
