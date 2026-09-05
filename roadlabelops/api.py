from __future__ import annotations

import uuid
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from .ingest import ingest_video
from .runtime import WorkflowRuntime
from .settings import build_cvat_adapter, build_detection_runner, load_settings
from .storage import LocalStore, StorageError
from .tools.environment import check_environment

SETTINGS = load_settings()
DATA_ROOT = Path(SETTINGS.roadlabelops_data_dir)
STORE = LocalStore(DATA_ROOT, SETTINGS.roadlabelops_runtime_dir)
RUNTIME = WorkflowRuntime(
    STORE,
    cvat=build_cvat_adapter(SETTINGS),
    detector=build_detection_runner(SETTINGS),
)

app = FastAPI(title="RoadLabelOps API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=SETTINGS.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

UPLOAD_CHUNK_BYTES = 1024 * 1024
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/quicktime", "video/x-m4v"}
ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v"}


class ApiRequestError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        status_code: int,
        *,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.retryable = retryable


class UploadBodyTooLarge(RuntimeError):
    """Raised before multipart parsing can spool an oversized request body."""


class UploadBodyLimitMiddleware:
    """Enforce the upload request ceiling at the ASGI receive boundary."""

    def __init__(self, inner_app: Any) -> None:
        self.inner_app = inner_app

    async def __call__(self, scope, receive, send) -> None:
        if not (
            scope.get("type") == "http"
            and scope.get("method") == "POST"
            and scope.get("path") == "/api/v1/sessions/upload"
        ):
            await self.inner_app(scope, receive, send)
            return

        request_limit = (
            SETTINGS.roadlabelops_max_upload_bytes
            + SETTINGS.roadlabelops_upload_overhead_bytes
        )
        received_bytes = 0
        limit_exceeded = False
        buffered_messages: list[dict[str, Any]] = []

        async def limited_receive():
            nonlocal limit_exceeded, received_bytes
            message = await receive()
            if message.get("type") == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > request_limit:
                    limit_exceeded = True
                    raise UploadBodyTooLarge
            return message

        async def tracked_send(message) -> None:
            buffered_messages.append(message)

        try:
            await self.inner_app(scope, limited_receive, tracked_send)
        except UploadBodyTooLarge:
            limit_exceeded = True
        if limit_exceeded:
            raw_headers = {
                key.lower(): value
                for key, value in scope.get("headers", [])
            }
            request_id = raw_headers.get(b"x-request-id", b"").decode(
                "ascii", errors="ignore"
            ) or uuid.uuid4().hex[:12]
            response = JSONResponse(
                status_code=413,
                content={
                    "error": {
                        "code": "UPLOAD_TOO_LARGE",
                        "message": (
                            "Upload request exceeds the configured video size limit "
                            "including multipart overhead"
                        ),
                        "retryable": False,
                        "request_id": request_id,
                    }
                },
                headers={"X-Request-ID": request_id},
            )
            await response(scope, receive, send)
            return
        for message in buffered_messages:
            await send(message)


def _request_id(request: Request) -> str:
    return str(
        getattr(request.state, "request_id", None)
        or request.headers.get("x-request-id")
        or uuid.uuid4().hex[:12]
    )


def _error_response(
    request: Request,
    *,
    code: str,
    message: str,
    status_code: int,
    retryable: bool = False,
) -> JSONResponse:
    request_id = _request_id(request)
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "retryable": retryable,
                "request_id": request_id,
            }
        },
        headers={"X-Request-ID": request_id},
    )


@app.middleware("http")
async def attach_request_id(request: Request, call_next):
    request.state.request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    if request.method == "POST" and request.url.path == "/api/v1/sessions/upload":
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                request_bytes = int(content_length)
            except ValueError:
                return _error_response(
                    request,
                    code="INVALID_CONTENT_LENGTH",
                    message="Content-Length must be a non-negative integer",
                    status_code=400,
                )
            if request_bytes < 0:
                return _error_response(
                    request,
                    code="INVALID_CONTENT_LENGTH",
                    message="Content-Length must be a non-negative integer",
                    status_code=400,
                )
            request_limit = (
                SETTINGS.roadlabelops_max_upload_bytes
                + SETTINGS.roadlabelops_upload_overhead_bytes
            )
            if request_bytes > request_limit:
                return _error_response(
                    request,
                    code="UPLOAD_TOO_LARGE",
                    message=(
                        "Upload request exceeds the configured video size limit "
                        "including multipart overhead"
                    ),
                    status_code=413,
                )
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response


app.add_middleware(UploadBodyLimitMiddleware)


class ActionRequest(BaseModel):
    version: str = Field(default="1.0.0", pattern=r"^\d+\.\d+\.\d+$")
    approved: bool = False


@app.exception_handler(ApiRequestError)
async def handle_api_request_error(request: Request, exc: ApiRequestError) -> JSONResponse:
    return _error_response(
        request,
        code=exc.code,
        message=exc.message,
        status_code=exc.status_code,
        retryable=exc.retryable,
    )


@app.exception_handler(FileNotFoundError)
async def handle_not_found(request: Request, exc: FileNotFoundError) -> JSONResponse:
    return _error_response(
        request,
        code="NOT_FOUND",
        message="The requested resource does not exist",
        status_code=404,
    )


@app.exception_handler(StorageError)
async def handle_storage_error(request: Request, exc: StorageError) -> JSONResponse:
    return _error_response(
        request,
        code="INVALID_PATH",
        message=str(exc),
        status_code=400,
    )


@app.exception_handler(Exception)
async def handle_exception(request: Request, exc: Exception) -> JSONResponse:
    return _error_response(
        request,
        code="INTERNAL_ERROR",
        message="The operation could not be completed",
        status_code=500,
        retryable=True,
    )


@app.exception_handler(RequestValidationError)
async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    issue = exc.errors()[0] if exc.errors() else {}
    location = ".".join(str(part) for part in issue.get("loc", ()) if part != "body")
    message = "Request parameters are invalid"
    if location:
        message = f"Request parameter {location} is invalid"
    return _error_response(
        request,
        code="VALIDATION_ERROR",
        message=message,
        status_code=422,
    )


@app.get("/api/v1/health")
def health() -> dict[str, object]:
    """Backward-compatible liveness endpoint; it never checks downstream services."""

    return liveness()


@app.get("/api/v1/health/live")
def liveness() -> dict[str, object]:
    return {
        "status": "ok",
        "service": "roadlabelops",
        "version": "0.1.0",
        "probe": "liveness",
    }


@app.get("/api/v1/health/ready", response_model=None)
def readiness():
    """Check every dependency required by the real video-to-Release workflow."""

    cvat = getattr(RUNTIME, "cvat", None)
    result = check_environment(
        STORE.root,
        SETTINGS.cvat_host,
        SETTINGS.cvat_credentials_configured,
        SETTINGS.detection_provider,
        SETTINGS.detection_model_path,
        runtime_root=SETTINGS.roadlabelops_runtime_dir,
        cvat_health_check=cvat.health if cvat else None,
    )
    payload = {
        "status": "ready" if result.data["real_flow_ready"] else "not_ready",
        "service": "roadlabelops",
        "version": "0.1.0",
        "probe": "readiness",
        **result.data,
    }
    return JSONResponse(
        status_code=200 if result.data["real_flow_ready"] else 503,
        content=payload,
    )


@app.get("/api/v1/dashboard")
def dashboard() -> dict[str, object]:
    return RUNTIME.dashboard()


@app.get("/api/v1/sessions/{session_id}")
def get_session(session_id: str) -> dict[str, object]:
    return {"session": STORE.get_session(session_id).to_dict()}


@app.post("/api/v1/demo", status_code=201)
def create_demo() -> dict[str, object]:
    return {"session": RUNTIME.create_demo().to_dict()}


@app.post("/api/v1/sessions/upload", status_code=201)
def upload_video(
    video: Annotated[UploadFile, File(...)],
    scene_seconds: int = Query(default=15, ge=10, le=30),
    allow_long_video: bool = Query(default=False),
) -> dict[str, object]:
    suffix = Path(video.filename or "upload.mp4").suffix.lower()
    content_type = (video.content_type or "").split(";", 1)[0].strip().lower()
    if content_type not in ALLOWED_VIDEO_TYPES or suffix not in ALLOWED_VIDEO_SUFFIXES:
        raise ApiRequestError(
            "UNSUPPORTED_MEDIA",
            "Use an MP4, MOV, or M4V road video",
            415,
        )
    destination = STORE.raw_dir / f"upload_{uuid.uuid4().hex[:10]}{suffix}"
    total_bytes = 0
    try:
        with destination.open("xb") as handle:
            while chunk := video.file.read(UPLOAD_CHUNK_BYTES):
                total_bytes += len(chunk)
                if total_bytes > SETTINGS.roadlabelops_max_upload_bytes:
                    raise ApiRequestError(
                        "UPLOAD_TOO_LARGE",
                        f"Video exceeds the {SETTINGS.roadlabelops_max_upload_bytes} byte upload limit",
                        413,
                    )
                handle.write(chunk)
        if total_bytes == 0:
            raise ApiRequestError("EMPTY_UPLOAD", "The uploaded video is empty", 422)
        result = ingest_video(
            STORE,
            destination,
            scene_seconds=scene_seconds,
            session_name=Path(video.filename or destination.name).stem,
            max_duration_seconds=SETTINGS.roadlabelops_max_video_duration_seconds,
            allow_long_video=allow_long_video,
            absolute_max_duration_seconds=(
                SETTINGS.roadlabelops_absolute_max_video_duration_seconds
            ),
            max_scene_count=SETTINGS.roadlabelops_max_scene_count,
            max_split_output_bytes=SETTINGS.roadlabelops_max_split_output_bytes,
            max_width=SETTINGS.roadlabelops_max_video_width,
            max_height=SETTINGS.roadlabelops_max_video_height,
            max_fps=SETTINGS.roadlabelops_max_video_fps,
        )
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    if not result.ok:
        destination.unlink(missing_ok=True)
        raise ApiRequestError(
            str((result.error or {}).get("code", "INGEST_FAILED")),
            str((result.error or {}).get("message", "The video could not be ingested")),
            422,
            retryable=result.retryable,
        )
    if result.data.get("existing") and not result.data.get("source_recovered"):
        destination.unlink(missing_ok=True)
    return result.data


@app.get("/api/v1/scenes/{scene_id}/thumbnail")
def get_thumbnail(scene_id: str):
    for session in STORE.list_sessions():
        for scene in session.scenes:
            if scene.scene_id != scene_id:
                continue
            if not scene.thumbnail_path:
                raise FileNotFoundError(scene_id)
            thumbnail = Path(scene.thumbnail_path)
            if thumbnail.is_symlink():
                raise StorageError("scene thumbnail must not be a symbolic link")
            try:
                resolved = thumbnail.resolve(strict=True)
            except (OSError, RuntimeError):
                raise FileNotFoundError(scene_id) from None
            scenes_root = STORE.scenes_dir.resolve()
            if not resolved.is_relative_to(scenes_root):
                raise StorageError("scene thumbnail escapes the managed scenes directory")
            if not resolved.is_file() or resolved.is_symlink():
                raise FileNotFoundError(scene_id)
            return FileResponse(resolved, media_type="image/jpeg")
    raise FileNotFoundError(scene_id)


@app.post("/api/v1/sessions/{session_id}/actions/{action}", response_model=None)
def advance(session_id: str, action: str, body: ActionRequest):
    result = RUNTIME.advance(
        session_id,
        action,
        version=body.version,
        approved=body.approved,
    )
    if not result.ok:
        raise ApiRequestError(
            str((result.error or {}).get("code", "ACTION_FAILED")),
            str((result.error or {}).get("message", "The workflow action failed")),
            409,
            retryable=result.retryable,
        )
    return result.data


@app.post("/api/v1/sessions/{session_id}/release/verify", response_model=None)
def verify_release(session_id: str):
    result = RUNTIME.verify_release(session_id)
    if not result.ok:
        raise ApiRequestError(
            str((result.error or {}).get("code", "RELEASE_INVALID")),
            str((result.error or {}).get("message", "The release could not be verified")),
            409,
        )
    return result.data
