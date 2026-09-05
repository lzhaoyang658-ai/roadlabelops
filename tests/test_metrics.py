from roadlabelops.metrics import calculate_operational_metrics
from roadlabelops.models import Scene, Session
from roadlabelops.runtime import WorkflowRuntime
from roadlabelops.storage import LocalStore


def _session(
    *,
    tasks: int = 0,
    session_id: str = "session_metrics",
    demo: bool = False,
) -> Session:
    session = Session(
        session_id=session_id,
        name="metrics",
        source_path="source.mp4",
        source_sha256="a" * 64,
        duration_seconds=31,
        fps=25,
        width=640,
        height=360,
        scene_seconds=15,
        demo=demo,
    )
    session.scenes = [
        Scene(
            scene_id=f"{session_id}_scene_{index:03d}",
            session_id=session.session_id,
            start_seconds=(index - 1) * 15,
            end_seconds=min(index * 15, 31),
            video_path=f"scene_{index:03d}.mp4",
            cvat_task_id=700 + index if index <= tasks else None,
        )
        for index in range(1, 4)
    ]
    return session


def test_operational_metrics_follow_prd_denominators() -> None:
    journal = [
        {
            "event": "stage.started",
            "action": "release",
            "session_id": "session_metrics",
            "trace_id": "first",
        },
        {
            "event": "stage.started",
            "tool_name": "release",
            "session_id": "session_metrics",
            "trace_id": "second",
        },
    ]

    result = calculate_operational_metrics(
        [_session(tasks=2)],
        {"session_metrics": {"verified": True}},
        journal,
    )

    assert result["video_slice_success_rate"]["value"] == 1.0
    assert result["task_creation_success_rate"]["value"] == 0.6667
    assert result["release_verification_success_rate"]["value"] == 0.5


def test_operational_metrics_explain_empty_denominators() -> None:
    result = calculate_operational_metrics([], {}, [])

    assert all(metric["value"] is None for metric in result.values())
    assert all(metric["reason"] for metric in result.values())


def test_dashboard_exposes_operational_metrics(tmp_path) -> None:
    runtime = WorkflowRuntime(LocalStore(tmp_path / "data", tmp_path / "runtime"))
    runtime.create_demo()

    dashboard = runtime.dashboard()

    assert all(
        metric["value"] is None
        for metric in dashboard["operational_metrics"].values()
    )


def test_operational_metrics_exclude_demo_sessions_tasks_and_releases() -> None:
    real = _session(tasks=2)
    demo = _session(tasks=3, session_id="session_demo_metrics", demo=True)
    journal = [
        {
            "event": "stage.started",
            "action": "release",
            "session_id": real.session_id,
            "trace_id": "real-release",
        },
        {
            "event": "stage.started",
            "action": "release",
            "session_id": demo.session_id,
            "trace_id": "demo-release",
        },
    ]

    result = calculate_operational_metrics(
        [real, demo],
        {
            real.session_id: {"verified": True},
            demo.session_id: {"verified": True},
        },
        journal,
    )

    assert result["video_slice_success_rate"] == {
        "value": 1.0,
        "numerator": 3,
        "denominator": 3,
        "reason": None,
    }
    assert result["task_creation_success_rate"]["value"] == 0.6667
    assert result["release_verification_success_rate"] == {
        "value": 1.0,
        "numerator": 1,
        "denominator": 1,
        "reason": None,
    }


def test_ingest_events_add_failures_and_do_not_double_count_legacy_sessions() -> None:
    session = _session(tasks=0)
    journal = [
        {
            "event": "ingest.completed",
            "session_id": session.session_id,
            "trace_id": "successful-attempt",
            "success": True,
            "demo": False,
            "planned_scene_count": 3,
            "successful_scene_count": 3,
        },
        # A duplicate upload for the same persisted Session is not a second slice.
        {
            "event": "ingest.completed",
            "session_id": session.session_id,
            "trace_id": "deduplicated-upload",
            "success": True,
            "demo": False,
            "planned_scene_count": 3,
            "successful_scene_count": 3,
            "existing": True,
        },
        {
            "event": "ingest.completed",
            "session_id": "session_failed",
            "trace_id": "failed-attempt",
            "success": False,
            "demo": False,
            "planned_scene_count": 2,
            "successful_scene_count": 0,
        },
        # Probe failures are durable, but have no planned Scenes to add.
        {
            "event": "ingest.completed",
            "session_id": None,
            "trace_id": "probe-failure",
            "success": False,
            "demo": False,
            "planned_scene_count": None,
            "successful_scene_count": 0,
        },
    ]

    result = calculate_operational_metrics([session], {}, journal)

    assert result["video_slice_success_rate"] == {
        "value": 0.6,
        "numerator": 3,
        "denominator": 5,
        "reason": None,
    }
