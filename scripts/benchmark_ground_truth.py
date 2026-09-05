#!/usr/bin/env python3
"""Benchmark Ultralytics detection models against a RoadLabelOps COCO reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO

from roadlabelops.tools.detection import MODEL_MAPPING, postprocess_predictions, result_frame_index
from roadlabelops.tools.quality import calculate_quality


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_reference(
    ground_truth_dir: Path,
) -> tuple[list[Path], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    payload = json.loads((ground_truth_dir / "annotations.coco.json").read_text())
    categories = {item["id"]: item["name"] for item in payload["categories"]}
    images_by_id = {item["id"]: item for item in payload["images"]}
    metadata_by_path: dict[str, dict[str, Any]] = {}
    image_paths: list[Path] = []
    for image in payload["images"]:
        path = (ground_truth_dir / image["file_name"]).resolve()
        image_paths.append(path)
        metadata_by_path[str(path)] = image

    annotations: list[dict[str, Any]] = []
    for item in payload["annotations"]:
        image = images_by_id[item["image_id"]]
        x, y, width, height = item["bbox"]
        annotations.append(
            {
                "scene_id": image["scene_id"],
                "frame": image["source_frame"],
                "label": categories[item["category_id"]],
                "bbox": [x, y, x + width, y + height],
            }
        )
    return image_paths, metadata_by_path, annotations


def to_predictions(
    results: list[Any],
    metadata_by_path: dict[str, dict[str, Any]],
    model_stem: str,
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for result in results:
        image = metadata_by_path[str(Path(result.path).resolve())]
        if result.boxes is None:
            continue
        for box_index, box in enumerate(result.boxes):
            model_label = str(result.names[int(box.cls.item())])
            label = MODEL_MAPPING.get(model_label)
            if not label:
                continue
            predictions.append(
                {
                    "prediction_id": (
                        f"{model_stem}_{image['scene_id']}_{image['source_frame']}_{box_index}"
                    ),
                    "scene_id": image["scene_id"],
                    "frame": image["source_frame"],
                    "label": label,
                    "confidence": round(float(box.conf.item()), 4),
                    "bbox": [round(float(value), 2) for value in box.xyxy[0].tolist()],
                    "source": "auto",
                }
            )
    return predictions


def to_video_predictions(
    model: YOLO,
    scene_paths: list[tuple[str, Path]],
    target_frames: set[tuple[str, int]],
    *,
    model_stem: str,
    confidence: float,
    image_size: int,
    device: str,
    frame_step: int,
) -> tuple[list[dict[str, Any]], list[dict[str, float]], int]:
    predictions: list[dict[str, Any]] = []
    speeds: list[dict[str, float]] = []
    processed_frame_count = 0
    for scene_id, scene_path in scene_paths:
        results = model.predict(
            source=str(scene_path),
            conf=confidence,
            imgsz=image_size,
            device=device,
            stream=True,
            vid_stride=frame_step,
            verbose=False,
        )
        for result_index, result in enumerate(results):
            frame = result_frame_index(result_index, frame_step)
            processed_frame_count += 1
            speeds.append({key: float(value) for key, value in result.speed.items()})
            if (scene_id, frame) not in target_frames or result.boxes is None:
                continue
            for box_index, box in enumerate(result.boxes):
                model_label = str(result.names[int(box.cls.item())])
                label = MODEL_MAPPING.get(model_label)
                if not label:
                    continue
                predictions.append(
                    {
                        "prediction_id": f"{model_stem}_{scene_id}_{frame}_{box_index}",
                        "scene_id": scene_id,
                        "frame": frame,
                        "label": label,
                        "confidence": round(float(box.conf.item()), 4),
                        "bbox": [round(float(value), 2) for value in box.xyxy[0].tolist()],
                        "source": "auto",
                    }
                )
    return predictions, speeds, processed_frame_count


def benchmark_model(
    model_name: str,
    image_paths: list[Path],
    metadata_by_path: dict[str, dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    *,
    confidence: float,
    image_size: int,
    device: str,
    nms_iou: float,
    rider_overlap: float,
    scenes_dir: Path | None,
    frame_step: int,
    include_predictions: bool,
) -> dict[str, Any]:
    load_started = time.perf_counter()
    model = YOLO(model_name)
    load_seconds = time.perf_counter() - load_started

    model.predict(
        source=str(image_paths[0]),
        conf=confidence,
        imgsz=image_size,
        device=device,
        verbose=False,
    )
    inference_started = time.perf_counter()
    if scenes_dir is None:
        results = model.predict(
            source=[str(path) for path in image_paths],
            conf=confidence,
            imgsz=image_size,
            device=device,
            verbose=False,
        )
        raw_predictions = to_predictions(results, metadata_by_path, Path(model_name).stem)
        speeds = [{key: float(value) for key, value in result.speed.items()} for result in results]
        processed_frame_count = len(results)
        source_mode = "ground_truth_images"
    else:
        scene_ids = list(
            dict.fromkeys(str(metadata["scene_id"]) for metadata in metadata_by_path.values())
        )
        scene_paths = [
            (scene_id, scenes_dir / f"{'_'.join(scene_id.split('_')[-2:])}.mp4")
            for scene_id in scene_ids
        ]
        missing = [str(path) for _, path in scene_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing source scenes: {missing}")
        target_frames = {
            (str(metadata["scene_id"]), int(metadata["source_frame"]))
            for metadata in metadata_by_path.values()
        }
        raw_predictions, speeds, processed_frame_count = to_video_predictions(
            model,
            scene_paths,
            target_frames,
            model_stem=Path(model_name).stem,
            confidence=confidence,
            image_size=image_size,
            device=device,
            frame_step=frame_step,
        )
        source_mode = "production_scene_videos"
    inference_seconds = time.perf_counter() - inference_started
    predictions, postprocessing = postprocess_predictions(
        raw_predictions,
        nms_iou_threshold=nms_iou,
        rider_overlap_threshold=rider_overlap,
    )
    evaluation_frame_keys = {
        (str(metadata["scene_id"]), int(metadata["source_frame"]))
        for metadata in metadata_by_path.values()
    }
    quality = calculate_quality(
        predictions,
        ground_truth,
        evaluated_frame_keys=evaluation_frame_keys,
    ).data
    speed_keys = ("preprocess", "inference", "postprocess")
    average_ultralytics_ms = {
        key: round(sum(speed[key] for speed in speeds) / len(speeds), 3) for key in speed_keys
    }

    weight_path = Path(model_name).resolve()
    gates = {
        "precision_at_least_0_90": (quality["precision"] or 0.0) >= 0.90,
        "recall_at_least_0_85": (quality["recall"] or 0.0) >= 0.85,
        "clean_frame_rate_at_least_0_80": (quality["clean_frame_rate"] or 0.0) >= 0.80,
    }
    benchmark = {
        "model": model_name,
        "weight_path": str(weight_path),
        "weight_sha256": sha256(weight_path),
        "weight_size_mb": round(weight_path.stat().st_size / 1_000_000, 2),
        "parameter_count": sum(parameter.numel() for parameter in model.model.parameters()),
        "load_seconds": round(load_seconds, 3),
        "warmup_images": 1,
        "source_mode": source_mode,
        "evaluation_image_count": len(image_paths),
        "timed_frame_count": processed_frame_count,
        "inference_wall_seconds": round(inference_seconds, 3),
        "frames_per_second": round(processed_frame_count / inference_seconds, 2),
        "wall_ms_per_frame": round(inference_seconds * 1000 / processed_frame_count, 2),
        "average_ultralytics_ms": average_ultralytics_ms,
        "postprocessing": postprocessing,
        "quality": quality,
        "gates": gates,
        "gate_result": "PASS" if all(gates.values()) else "FAIL",
    }
    if include_predictions:
        benchmark["raw_predictions"] = raw_predictions
    return benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"])
    parser.add_argument("--confidence", type=float, default=0.40)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--nms-iou", type=float, default=0.75)
    parser.add_argument("--rider-overlap", type=float, default=0.25)
    parser.add_argument("--scenes-dir", type=Path)
    parser.add_argument("--frame-step", type=int, default=5)
    parser.add_argument("--include-predictions", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ground_truth_dir = args.ground_truth.resolve()
    image_paths, metadata_by_path, ground_truth = load_reference(ground_truth_dir)
    results = []
    for model_name in args.models:
        print(
            f"Benchmarking {model_name} on {len(image_paths)} images ({args.device})...", flush=True
        )
        results.append(
            benchmark_model(
                model_name,
                image_paths,
                metadata_by_path,
                ground_truth,
                confidence=args.confidence,
                image_size=args.image_size,
                device=args.device,
                nms_iou=args.nms_iou,
                rider_overlap=args.rider_overlap,
                scenes_dir=args.scenes_dir.resolve() if args.scenes_dir else None,
                frame_step=args.frame_step,
                include_predictions=args.include_predictions,
            )
        )
        latest = results[-1]
        print(
            json.dumps(
                {
                    "model": latest["model"],
                    "quality": latest["quality"],
                    "fps": latest["frames_per_second"],
                    "gate_result": latest["gate_result"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    payload = {
        "benchmark_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ground_truth_dir": str(ground_truth_dir),
        "ground_truth_sha256": sha256(ground_truth_dir / "annotations.coco.json"),
        "ground_truth_image_count": len(image_paths),
        "ground_truth_annotation_count": len(ground_truth),
        "environment": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": args.device,
        },
        "settings": {
            "confidence": args.confidence,
            "image_size": args.image_size,
            "nms_iou": args.nms_iou,
            "rider_overlap": args.rider_overlap,
            "match_iou": 0.5,
            "frame_step": args.frame_step,
            "source_mode": (
                "production_scene_videos" if args.scenes_dir else "ground_truth_images"
            ),
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
