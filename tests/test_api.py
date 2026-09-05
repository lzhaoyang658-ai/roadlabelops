from pathlib import Path

from fastapi.testclient import TestClient

from roadlabelops import api
from roadlabelops.models import Scene, Session, Stage, ToolResult
from roadlabelops.settings import Settings
from roadlabelops.storage import LocalStore


class StubRuntime:
    def __init__(self, result: ToolResult | None = None) -> None:
        self.result = result or ToolResult.success({"session": {"session_id": "session_ok"}})

    def advance(
        self,
        session_id: str,
        action: str,
        *,
        version: str,
        approved: bool = False,
    ) -> ToolResult:
        return self.result

    def verify_release(self, session_id: str) -> ToolResult:
        return self.result


def configure_api(
    tmp_path: Path,
    monkeypatch,
    *,
    max_upload_bytes: int = 1024,
    upload_overhead_bytes: int = 1024 * 1024,
) -> LocalStore:
    store = LocalStore(tmp_path / "data", tmp_path / "runtime")
    settings = Settings(
        _env_file=None,
        roadlabelops_data_dir=store.root,
        roadlabelops_max_upload_bytes=max_upload_bytes,
        roadlabelops_upload_overhead_bytes=upload_overhead_bytes,
    )
    monkeypatch.setattr(api, "STORE", store)
    monkeypatch.setattr(api, "SETTINGS", settings)
    monkeypatch.setattr(api, "RUNTIME", StubRuntime())
    return store


def test_request_id_is_returned_on_success_and_error(tmp_path: Path, monkeypatch) -> None:
    configure_api(tmp_path, monkeypatch)
    client = TestClient(api.app)

    health = client.get("/api/v1/health", headers={"x-request-id": "trace-test"})
    error = client.get("/api/v1/sessions/missing", headers={"x-request-id": "trace-test"})

    assert health.headers["x-request-id"] == "trace-test"
    assert error.status_code == 404
    assert error.json()["error"] == {
        "code": "NOT_FOUND",
        "message": "The requested resource does not exist",
        "retryable": False,
        "request_id": "trace-test",
    }


def test_liveness_does_not_call_dependency_checks(tmp_path: Path, monkeypatch) -> None:
    configure_api(tmp_path, monkeypatch)
    monkeypatch.setattr(
        api,
        "check_environment",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not run")),
    )
    client = TestClient(api.app)

    response = client.get("/api/v1/health/live")

    assert response.status_code == 200
    assert response.json()["probe"] == "liveness"


def test_readiness_returns_503_when_real_dependencies_are_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    configure_api(tmp_path, monkeypatch)
    monkeypatch.setattr(
        api,
        "check_environment",
        lambda *_args, **_kwargs: ToolResult.success(
            {
                "real_flow_ready": False,
                "demo_ready": True,
                "mode": "demo_only",
                "blocking": ["cvat_connection", "detector"],
                "checks": {},
            }
        ),
    )
    client = TestClient(api.app)

    response = client.get("/api/v1/health/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"
    assert response.json()["mode"] == "demo_only"


def test_readiness_returns_200_for_complete_real_flow(
    tmp_path: Path,
    monkeypatch,
) -> None:
    configure_api(tmp_path, monkeypatch)
    monkeypatch.setattr(
        api,
        "check_environment",
        lambda *_args, **_kwargs: ToolResult.success(
            {
                "real_flow_ready": True,
                "demo_ready": True,
                "mode": "real",
                "blocking": [],
                "checks": {},
            }
        ),
    )
    client = TestClient(api.app)

    response = client.get("/api/v1/health/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_upload_rejects_extension_even_with_video_mime(tmp_path: Path, monkeypatch) -> None:
    store = configure_api(tmp_path, monkeypatch)
    client = TestClient(api.app)

    response = client.post(
        "/api/v1/sessions/upload",
        files={"video": ("road.txt", b"not-video", "video/mp4")},
    )

    assert response.status_code == 415
    assert response.json()["error"]["code"] == "UNSUPPORTED_MEDIA"
    assert list(store.raw_dir.iterdir()) == []


def test_oversized_upload_removes_partial_file(tmp_path: Path, monkeypatch) -> None:
    store = configure_api(tmp_path, monkeypatch, max_upload_bytes=4)
    client = TestClient(api.app)

    response = client.post(
        "/api/v1/sessions/upload",
        files={"video": ("road.mp4", b"12345", "video/mp4")},
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "UPLOAD_TOO_LARGE"
    assert list(store.raw_dir.iterdir()) == []


def test_content_length_guard_rejects_before_multipart_parsing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = configure_api(
        tmp_path,
        monkeypatch,
        max_upload_bytes=4,
        upload_overhead_bytes=16,
    )
    client = TestClient(api.app)

    response = client.post(
        "/api/v1/sessions/upload",
        content=b"",
        headers={"content-length": "21", "content-type": "application/octet-stream"},
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "UPLOAD_TOO_LARGE"
    assert list(store.raw_dir.iterdir()) == []


def test_content_length_guard_allows_configured_multipart_overhead(
    tmp_path: Path,
    monkeypatch,
) -> None:
    configure_api(
        tmp_path,
        monkeypatch,
        max_upload_bytes=4,
        upload_overhead_bytes=1024,
    )
    monkeypatch.setattr(
        api,
        "ingest_video",
        lambda *_args, **_kwargs: ToolResult.success({
            "session": {"session_id": "session_ok"},
            "existing": False,
        }),
    )
    client = TestClient(api.app)

    response = client.post(
        "/api/v1/sessions/upload",
        files={"video": ("road.mp4", b"1234", "video/mp4")},
    )

    assert response.status_code == 201


def test_chunked_upload_is_limited_before_multipart_spooling(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = configure_api(
        tmp_path,
        monkeypatch,
        max_upload_bytes=4,
        upload_overhead_bytes=48,
    )
    boundary = "roadlabelops-boundary"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="video"; filename="road.mp4"\r\n'
        "Content-Type: video/mp4\r\n\r\n"
        "12345\r\n"
        f"--{boundary}--\r\n"
    ).encode()

    def body_chunks():
        yield body[:40]
        yield body[40:80]
        yield body[80:]

    client = TestClient(api.app)
    response = client.post(
        "/api/v1/sessions/upload",
        content=body_chunks(),
        headers={
            "content-type": f"multipart/form-data; boundary={boundary}",
            "x-request-id": "chunk-limit-test",
        },
    )

    assert response.status_code == 413
    assert response.headers["x-request-id"] == "chunk-limit-test"
    assert response.json()["error"] == {
        "code": "UPLOAD_TOO_LARGE",
        "message": (
            "Upload request exceeds the configured video size limit "
            "including multipart overhead"
        ),
        "retryable": False,
        "request_id": "chunk-limit-test",
    }
    assert list(store.raw_dir.iterdir()) == []


def test_upload_passes_media_limits_and_explicit_long_video_confirmation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    configure_api(tmp_path, monkeypatch)
    captured: dict[str, object] = {}

    def fake_ingest(*_args, **kwargs):
        captured.update(kwargs)
        return ToolResult.success({"session": {"session_id": "session_ok"}, "existing": False})

    monkeypatch.setattr(api, "ingest_video", fake_ingest)
    client = TestClient(api.app)

    response = client.post(
        "/api/v1/sessions/upload?allow_long_video=true",
        files={"video": ("road.mp4", b"1234", "video/mp4")},
    )

    assert response.status_code == 201
    assert captured == {
        "scene_seconds": 15,
        "session_name": "road",
        "max_duration_seconds": 300.0,
        "allow_long_video": True,
        "absolute_max_duration_seconds": 7200.0,
        "max_scene_count": 720,
        "max_split_output_bytes": 8 * 1024 * 1024 * 1024,
        "max_width": 3840,
        "max_height": 2160,
        "max_fps": 60.0,
    }


def test_upload_keeps_replacement_source_when_dedupe_repairs_missing_lineage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = configure_api(tmp_path, monkeypatch)
    uploaded_path: Path | None = None

    def fake_ingest(_store, destination, **_kwargs):
        nonlocal uploaded_path
        uploaded_path = Path(destination)
        return ToolResult.success(
            {
                "session": {"session_id": "session_existing"},
                "existing": True,
                "source_recovered": True,
            }
        )

    monkeypatch.setattr(api, "ingest_video", fake_ingest)
    client = TestClient(api.app)

    response = client.post(
        "/api/v1/sessions/upload",
        files={"video": ("road.mp4", b"replacement", "video/mp4")},
    )

    assert response.status_code == 201
    assert uploaded_path is not None
    assert uploaded_path.read_bytes() == b"replacement"
    assert list(store.raw_dir.iterdir()) == [uploaded_path]


def test_failed_ingest_removes_upload_and_preserves_retryability(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = configure_api(tmp_path, monkeypatch)
    monkeypatch.setattr(
        api,
        "ingest_video",
        lambda *_args, **_kwargs: ToolResult.failure(
            "VIDEO_PROBE_FAILED",
            "ffprobe could not read the video",
            retryable=True,
        ),
    )
    client = TestClient(api.app)

    response = client.post(
        "/api/v1/sessions/upload",
        files={"video": ("road.mp4", b"1234", "video/mp4")},
    )

    assert response.status_code == 422
    assert response.json()["error"]["retryable"] is True
    assert list(store.raw_dir.iterdir()) == []


def test_action_failure_uses_consistent_error_envelope(tmp_path: Path, monkeypatch) -> None:
    configure_api(tmp_path, monkeypatch)
    monkeypatch.setattr(
        api,
        "RUNTIME",
        StubRuntime(ToolResult.failure("CVAT_OFFLINE", "CVAT is unavailable", retryable=True)),
    )
    client = TestClient(api.app)

    response = client.post(
        "/api/v1/sessions/session_ok/actions/create_tasks",
        json={"version": "1.0.0"},
    )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "CVAT_OFFLINE"
    assert response.json()["error"]["retryable"] is True


def test_validation_errors_use_consistent_error_envelope(tmp_path: Path, monkeypatch) -> None:
    configure_api(tmp_path, monkeypatch)
    client = TestClient(api.app)

    response = client.post(
        "/api/v1/sessions/session_ok/actions/release",
        json={"version": "not-semver"},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"
    assert response.json()["error"]["retryable"] is False


def test_release_verification_returns_runtime_receipt(tmp_path: Path, monkeypatch) -> None:
    configure_api(tmp_path, monkeypatch)
    monkeypatch.setattr(
        api,
        "RUNTIME",
        StubRuntime(ToolResult.success({"receipt": {"verified": True}})),
    )
    client = TestClient(api.app)

    response = client.post("/api/v1/sessions/session_ok/release/verify")

    assert response.status_code == 200
    assert response.json() == {"receipt": {"verified": True}}


def _session_with_thumbnail(store: LocalStore, thumbnail: Path) -> Session:
    session = Session(
        session_id="session_thumbnail",
        name="thumbnail fixture",
        source_path=str(store.raw_dir / "source.mp4"),
        source_sha256="a" * 64,
        duration_seconds=10,
        fps=25,
        width=640,
        height=360,
        status=Stage.SLICED,
        scenes=[
            Scene(
                scene_id="session_thumbnail_scene_001",
                session_id="session_thumbnail",
                start_seconds=0,
                end_seconds=10,
                video_path=str(store.scenes_dir / "scene.mp4"),
                thumbnail_path=str(thumbnail),
            )
        ],
    )
    store.save_session(session)
    return session


def test_thumbnail_rejects_stored_path_outside_managed_scenes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = configure_api(tmp_path, monkeypatch)
    outside = tmp_path / "private.jpg"
    outside.write_bytes(b"private")
    session = _session_with_thumbnail(store, outside)
    client = TestClient(api.app)

    response = client.get(f"/api/v1/scenes/{session.scenes[0].scene_id}/thumbnail")

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_PATH"
    assert response.content != outside.read_bytes()


def test_thumbnail_serves_regular_file_inside_managed_scenes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = configure_api(tmp_path, monkeypatch)
    thumbnail = store.scenes_dir / "session_thumbnail" / "thumb.jpg"
    thumbnail.parent.mkdir()
    thumbnail.write_bytes(b"jpeg-fixture")
    session = _session_with_thumbnail(store, thumbnail)
    client = TestClient(api.app)

    response = client.get(f"/api/v1/scenes/{session.scenes[0].scene_id}/thumbnail")

    assert response.status_code == 200
    assert response.content == b"jpeg-fixture"


def test_thumbnail_rejects_symbolic_link_even_when_target_is_managed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = configure_api(tmp_path, monkeypatch)
    target = store.scenes_dir / "session_thumbnail" / "target.jpg"
    target.parent.mkdir()
    target.write_bytes(b"jpeg-fixture")
    thumbnail = target.with_name("link.jpg")
    thumbnail.symlink_to(target)
    session = _session_with_thumbnail(store, thumbnail)
    client = TestClient(api.app)

    response = client.get(f"/api/v1/scenes/{session.scenes[0].scene_id}/thumbnail")

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_PATH"
