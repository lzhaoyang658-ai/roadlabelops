import json
from pathlib import Path
from types import SimpleNamespace

from roadlabelops import cli
from roadlabelops.models import ToolResult


def _doctor_settings(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        roadlabelops_data_dir=tmp_path / "data",
        roadlabelops_runtime_dir=tmp_path / "runtime",
        cvat_host=None,
        cvat_credentials_configured=False,
        detection_provider="mock",
        detection_model_path=tmp_path / "missing.pt",
    )


def test_doctor_exits_nonzero_when_only_demo_is_ready(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(cli, "load_settings", lambda: _doctor_settings(tmp_path))
    monkeypatch.setattr(cli, "build_cvat_adapter", lambda settings: None)
    monkeypatch.setattr(cli, "build_detection_runner", lambda settings: lambda *_args: None)
    monkeypatch.setattr(
        cli,
        "check_environment",
        lambda *_args, **_kwargs: ToolResult.success(
            {
                "real_flow_ready": False,
                "demo_ready": True,
                "mode": "demo_only",
                "blocking": ["cvat_configured", "detector"],
            }
        ),
    )

    exit_code = cli.main(["doctor"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert payload["requested_readiness"] == "real"
    assert payload["mode"] == "demo_only"


def test_doctor_demo_only_flag_accepts_demo_readiness(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(cli, "load_settings", lambda: _doctor_settings(tmp_path))
    monkeypatch.setattr(cli, "build_cvat_adapter", lambda settings: None)
    monkeypatch.setattr(cli, "build_detection_runner", lambda settings: lambda *_args: None)
    monkeypatch.setattr(
        cli,
        "check_environment",
        lambda *_args, **_kwargs: ToolResult.success(
            {
                "real_flow_ready": False,
                "demo_ready": True,
                "mode": "demo_only",
                "blocking": ["cvat_configured", "detector"],
            }
        ),
    )

    exit_code = cli.main(["doctor", "--demo-only"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["requested_readiness"] == "demo"


def test_advance_forwards_explicit_approval_to_runtime(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    settings = _doctor_settings(tmp_path)
    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "build_cvat_adapter", lambda configured: None)
    monkeypatch.setattr(
        cli,
        "build_detection_runner",
        lambda configured: lambda *_args: ToolResult.success(),
    )
    captured: dict[str, object] = {}

    class RuntimeFixture:
        def advance(
            self,
            session_id: str,
            action: str,
            *,
            version: str,
            approved: bool,
        ) -> ToolResult:
            captured.update(
                session_id=session_id,
                action=action,
                version=version,
                approved=approved,
            )
            return ToolResult.success({"status": "ok"})

    monkeypatch.setattr(
        cli,
        "WorkflowRuntime",
        lambda *_args, **_kwargs: RuntimeFixture(),
    )

    exit_code = cli.main(
        ["advance", "session_permission", "preannotate", "--approve", "--version", "2.1.0"]
    )

    assert exit_code == 0
    assert captured == {
        "session_id": "session_permission",
        "action": "preannotate",
        "version": "2.1.0",
        "approved": True,
    }
    assert json.loads(capsys.readouterr().out) == {"status": "ok"}
