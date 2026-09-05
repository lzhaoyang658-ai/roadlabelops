"""Re-run detection and safely replace only untouched automatic CVAT annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from roadlabelops.models import Stage, utc_now
from roadlabelops.settings import Settings, build_cvat_adapter, build_detection_runner
from roadlabelops.storage import LocalStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_id")
    parser.add_argument(
        "--approve-replace-auto",
        action="store_true",
        help="explicitly allow replacement when a CVAT task already contains auto annotations",
    )
    args = parser.parse_args()

    settings = Settings()
    store = LocalStore(settings.roadlabelops_data_dir)
    session = store.get_session(args.session_id)
    allowed = {
        Stage.TASKS_CREATED,
        Stage.PREANNOTATED,
        Stage.WAITING_FOR_HUMAN_REVIEW,
    }
    if session.status not in allowed:
        raise SystemExit(f"Cannot refresh annotations while session is {session.status.value}")
    cvat = build_cvat_adapter(settings)
    if cvat is None:
        raise SystemExit("CVAT is not configured")
    detect = build_detection_runner(settings)
    summary: list[dict[str, int | bool | str]] = []

    for scene in session.scenes:
        if not scene.cvat_task_id:
            raise SystemExit(f"Scene has no CVAT task: {scene.scene_id}")
        result = detect(Path(scene.video_path), session.frame_step)
        if not result.ok:
            raise SystemExit(f"Detection failed for {scene.scene_id}: {result.error}")
        predictions = [
            {**item, "scene_id": scene.scene_id}
            for item in result.data["predictions"]
        ]
        prediction_path = Path(scene.video_path).with_suffix(".predictions.json")
        store.write_json_atomic(prediction_path, predictions)
        imported = cvat.import_predictions(
            scene.cvat_task_id,
            predictions,
            allow_replace_auto=args.approve_replace_auto,
        )
        if not imported.ok:
            raise SystemExit(f"CVAT refresh refused for {scene.scene_id}: {imported.error}")
        scene.prediction_count = len(predictions)
        summary.append({
            "scene_id": scene.scene_id,
            "task_id": scene.cvat_task_id,
            "prediction_count": len(predictions),
            "replaced_auto": bool(imported.data.get("replaced_auto")),
        })
        session.updated_at = utc_now()
        store.save_session(session)
        if imported.data.get("replaced_auto"):
            store.append_journal(
                {
                    "run_id": f"run_{session.session_id}",
                    "session_id": session.session_id,
                    "stage": session.status.value,
                    "event": "permission.approved",
                    "tool_name": "replace_auto_annotations",
                    "timestamp": utc_now(),
                    "scene_id": scene.scene_id,
                    "cvat_task_id": scene.cvat_task_id,
                }
            )

    print(json.dumps({
        "session_id": session.session_id,
        "status": session.status.value,
        "scenes": summary,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
