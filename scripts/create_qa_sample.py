"""Create a deterministic annotation sample and a separate CVAT image task."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
from cvat_sdk import models
from cvat_sdk.core.proxies.tasks import ResourceType
from PIL import Image, ImageDraw, ImageFont

from roadlabelops.runtime import WorkflowRuntime
from roadlabelops.settings import Settings, build_cvat_adapter
from roadlabelops.storage import LocalStore

COLORS = {
    "car": "#22c55e",
    "bus": "#f59e0b",
    "truck": "#ef4444",
    "motorcycle": "#3b82f6",
    "bicycle": "#8b5cf6",
    "pedestrian": "#ec4899",
    "traffic_light": "#06b6d4",
    "traffic_sign": "#a855f7",
}


def distribute(total: int, bucket_count: int) -> list[int]:
    base, remainder = divmod(total, bucket_count)
    return [base + (1 if index < remainder else 0) for index in range(bucket_count)]


def select_evenly(available: list[int], count: int) -> list[int]:
    if count <= 0 or not available:
        return []
    if count >= len(available):
        return available
    if count == 1:
        return [available[len(available) // 2]]
    indices = [round(index * (len(available) - 1) / (count - 1)) for index in range(count)]
    return [available[index] for index in indices]


def sampled_video_frames(video_path: Path, frame_step: int) -> list[int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open scene video: {video_path}")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if frame_count <= 0:
        raise RuntimeError(f"Could not determine scene frame count: {video_path}")
    return list(range(frame_step - 1, frame_count, frame_step))


def extract_frames(video_path: Path, targets: dict[int, Path]) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open scene video: {video_path}")
    pending = set(targets)
    frame_index = 0
    try:
        while pending:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index in pending:
                output = targets[frame_index]
                output.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(output), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise RuntimeError(f"Could not write sample image: {output}")
                pending.remove(frame_index)
            frame_index += 1
    finally:
        capture.release()
    if pending:
        raise RuntimeError(f"Scene ended before sampled frames were found: {sorted(pending)}")


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def make_overlay(sample: dict[str, Any], image_dir: Path, overlay_dir: Path, purpose: str) -> Path:
    source = image_dir / sample["file_name"]
    image = Image.open(source).convert("RGB")
    draw = ImageDraw.Draw(image)
    label_font = font(20)
    for annotation in sample["annotations"]:
        x1, y1, x2, y2 = annotation["bbox_xyxy"]
        label = annotation["label"]
        confidence = annotation.get("confidence")
        color = COLORS.get(label, "#ffffff")
        draw.rectangle((x1, y1, x2, y2), outline=color, width=4)
        caption = label if confidence is None else f"{label} {confidence:.2f}"
        text_box = draw.textbbox((x1, y1), caption, font=label_font, stroke_width=1)
        text_height = text_box[3] - text_box[1] + 8
        top = max(0, y1 - text_height)
        draw.rectangle((x1, top, text_box[2] + 8, y1), fill=color)
        draw.text((x1 + 4, top + 3), caption, fill="white", font=label_font, stroke_width=1)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    output = overlay_dir / f"{purpose}-{sample['sample_index']:03d}-{source.name}"
    image.save(output, quality=92)
    return output


def make_contact_sheets(
    samples: list[dict[str, Any]],
    overlay_paths: list[Path],
    output_dir: Path,
    purpose: str,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    title_font = font(24)
    paths: list[Path] = []
    per_sheet = 6
    tile_width, image_height, header_height = 640, 360, 40
    for start in range(0, len(samples), per_sheet):
        subset = samples[start : start + per_sheet]
        subset_paths = overlay_paths[start : start + per_sheet]
        sheet = Image.new("RGB", (tile_width * 2, (image_height + header_height) * 3), "#111827")
        draw = ImageDraw.Draw(sheet)
        for local_index, (sample, overlay_path) in enumerate(zip(subset, subset_paths)):
            column, row = local_index % 2, local_index // 2
            x = column * tile_width
            y = row * (image_height + header_height)
            title = (
                f"{purpose.upper()} {sample['sample_index']:03d} · "
                f"{sample['scene_id'].rsplit('_', 1)[-1]}"
                f" · frame {sample['source_frame']} · {len(sample['annotations'])} boxes"
            )
            draw.text((x + 10, y + 6), title, fill="white", font=title_font)
            tile = Image.open(overlay_path).convert("RGB").resize((tile_width, image_height))
            sheet.paste(tile, (x, y + header_height))
        output = output_dir / f"contact-sheet-{len(paths) + 1:02d}.jpg"
        sheet.save(output, quality=92)
        paths.append(output)
    return paths


def create_cvat_task(
    settings: Settings,
    session_name: str,
    session_id: str,
    revision: str,
    purpose: str,
    sample_size: int,
    image_paths: list[Path],
    cvat_shapes: list[dict[str, Any]],
) -> dict[str, Any]:
    adapter = build_cvat_adapter(settings)
    if adapter is None:
        raise RuntimeError("CVAT is not configured")
    project_kind = "QA" if purpose == "qa" else "Training"
    project_result = adapter.ensure_project(
        f"RoadLabelOps {project_kind} · {session_name}", WorkflowRuntime._labels()
    )
    if not project_result.ok:
        raise RuntimeError(str(project_result.error))
    project_id = int(project_result.data["project_id"])
    task_name = f"{session_id} · deterministic-{sample_size}-frame-{purpose.upper()} · {revision}"

    with adapter._client() as client:
        matches = [
            task
            for task in client.tasks.list()
            if task.name == task_name and task.project_id == project_id
        ]
        created = not matches
        if matches:
            task = client.tasks.retrieve(matches[0].id)
        else:
            task = client.tasks.create_from_data(
                spec=models.TaskWriteRequest(name=task_name, project_id=project_id),
                resource_type=ResourceType.LOCAL,
                resources=[str(path.resolve()) for path in image_paths],
                data_params={"image_quality": 100, "sorting_method": "lexicographical"},
            )

        existing = task.get_annotations()
        existing_objects = [*existing.tags, *existing.shapes, *existing.tracks]
        if existing_objects and not all(
            str(getattr(item, "source", "manual")) == "auto" for item in existing_objects
        ):
            raise RuntimeError(
                f"{project_kind} task contains human annotations; automatic overwrite denied"
            )

        if not existing_objects:
            label_ids = {label.name: label.id for label in task.get_labels()}
            shapes = [
                models.LabeledShapeRequest(
                    type="rectangle",
                    label_id=label_ids[item["label"]],
                    frame=int(item["frame"]),
                    points=item["bbox"],
                    source="auto",
                    score=float(item["confidence"]),
                )
                for item in cvat_shapes
                if item["label"] in label_ids
            ]
            task.set_annotations(models.LabeledDataRequest(shapes=shapes))

        jobs = task.get_jobs()
        return {
            "project_id": project_id,
            "project_created": bool(project_result.data.get("created")),
            "task_id": int(task.id),
            "task_created": created,
            "job_ids": [int(job.id) for job in jobs],
            "annotation_count": len(existing_objects) or len(cvat_shapes),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_id")
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--revision", default="frame-aligned-v1")
    parser.add_argument("--purpose", choices=("qa", "training"), default="qa")
    args = parser.parse_args()

    settings = Settings()
    store = LocalStore(settings.roadlabelops_data_dir)
    session = store.get_session(args.session_id)
    if args.sample_size < len(session.scenes):
        raise SystemExit("sample size must be at least the scene count")

    safe_revision = args.revision.replace("/", "-").replace("..", "-")
    output_root = store.sessions_dir / (f"{session.session_id}.{args.purpose}-{safe_revision}")
    image_dir = output_root / "images"
    overlay_dir = output_root / "overlays"
    contact_sheet_dir = output_root / "contact-sheets"
    scene_counts = distribute(args.sample_size, len(session.scenes))
    samples: list[dict[str, Any]] = []
    cvat_shapes: list[dict[str, Any]] = []

    for scene, count in zip(session.scenes, scene_counts):
        prediction_path = Path(scene.video_path).with_suffix(".predictions.json")
        predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
        predictions_by_frame: dict[int, list[dict[str, Any]]] = {}
        for prediction in predictions:
            predictions_by_frame.setdefault(int(prediction["frame"]), []).append(prediction)
        if args.purpose == "training":
            available = sampled_video_frames(Path(scene.video_path), session.frame_step)
        else:
            available = sorted(predictions_by_frame)
        selected = select_evenly(available, count)
        targets: dict[int, Path] = {}
        for source_frame in selected:
            file_name = f"{scene.scene_id}_frame_{source_frame:06d}.jpg"
            targets[source_frame] = image_dir / file_name
        extract_frames(Path(scene.video_path), targets)

        for source_frame in selected:
            sample_index = len(samples) + 1
            annotations: list[dict[str, Any]] = []
            for prediction in predictions_by_frame.get(source_frame, []):
                x1, y1, x2, y2 = [round(float(value), 2) for value in prediction["bbox"]]
                annotation = {
                    "prediction_id": prediction.get("prediction_id"),
                    "label": str(prediction["label"]),
                    "confidence": round(float(prediction.get("confidence", 1.0)), 4),
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "bbox": [x1, y1, round(x2 - x1, 2), round(y2 - y1, 2)],
                }
                annotations.append(annotation)
                cvat_shapes.append(
                    {
                        "frame": sample_index - 1,
                        "label": annotation["label"],
                        "confidence": annotation["confidence"],
                        "bbox": annotation["bbox_xyxy"],
                    }
                )
            samples.append(
                {
                    "sample_index": sample_index,
                    "scene_id": scene.scene_id,
                    "source_frame": source_frame,
                    "file_name": targets[source_frame].name,
                    "annotations": annotations,
                }
            )

    overlay_paths = [
        make_overlay(sample, image_dir, overlay_dir, args.purpose) for sample in samples
    ]
    contact_sheets = make_contact_sheets(samples, overlay_paths, contact_sheet_dir, args.purpose)
    cvat = create_cvat_task(
        settings,
        session.name,
        session.session_id,
        safe_revision,
        args.purpose,
        len(samples),
        [image_dir / sample["file_name"] for sample in samples],
        cvat_shapes,
    )
    manifest = {
        "session_id": session.session_id,
        "source_sha256": session.source_sha256,
        "purpose": args.purpose,
        "sampling_revision": safe_revision,
        "method": "Deterministic, evenly spaced sampling across all scenes",
        "frame_pool": (
            "all detector-stride frames, including zero-prediction negatives"
            if args.purpose == "training"
            else "frames containing at least one prediction"
        ),
        "sample_size": len(samples),
        "sample_counts_by_scene": dict(Counter(item["scene_id"] for item in samples)),
        "predicted_box_count": len(cvat_shapes),
        "samples": samples,
        "cvat": cvat,
        "contact_sheets": [str(path.resolve()) for path in contact_sheets],
    }
    manifest_path = output_root / "sample-manifest.json"
    store.write_json_atomic(manifest_path, manifest)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "sample_size": len(samples),
                "predicted_box_count": len(cvat_shapes),
                "cvat": cvat,
                "contact_sheet_count": len(contact_sheets),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
