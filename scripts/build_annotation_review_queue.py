"""Build a risk-ranked human review queue without mutating CVAT annotations."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from roadlabelops.tools.detection import MODEL_MAPPING, postprocess_predictions

COLORS = {
    "car": "#22c55e",
    "bus": "#f59e0b",
    "truck": "#ef4444",
    "motorcycle": "#3b82f6",
    "bicycle": "#8b5cf6",
    "pedestrian": "#ec4899",
}
PRIORITY = {
    "motorcycle": 5.0,
    "bus": 5.0,
    "pedestrian": 2.0,
    "bicycle": 2.0,
    "truck": 1.5,
    "car": 1.0,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def box_iou(first: list[float], second: list[float]) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def find_unmatched_candidates(
    baseline: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    match_iou: float,
) -> list[dict[str, Any]]:
    unmatched: list[dict[str, Any]] = []
    for candidate in candidates:
        if any(
            candidate["label"] == item["label"]
            and box_iou(candidate["bbox_xyxy"], item["bbox_xyxy"]) >= match_iou
            for item in baseline
        ):
            continue
        unmatched.append(candidate)
    return unmatched


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def draw_label(
    draw: ImageDraw.ImageDraw,
    box: list[float],
    caption: str,
    color: str,
    label_font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    *,
    width: int,
) -> None:
    x1, y1, x2, y2 = box
    draw.rectangle((x1, y1, x2, y2), outline=color, width=width)
    bounds = draw.textbbox((x1, y1), caption, font=label_font, stroke_width=1)
    label_height = bounds[3] - bounds[1] + 8
    top = max(0, y1 - label_height)
    draw.rectangle((x1, top, bounds[2] + 8, y1), fill=color)
    draw.text((x1 + 4, top + 3), caption, fill="white", font=label_font, stroke_width=1)


def make_overlay(
    image_path: Path,
    baseline: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    label_font = font(18)
    for item in baseline:
        draw_label(
            draw,
            item["bbox_xyxy"],
            item["label"],
            COLORS.get(item["label"], "#ffffff"),
            label_font,
            width=2,
        )
    for item in candidates:
        draw_label(
            draw,
            item["bbox_xyxy"],
            f"CHECK {item['label']} {item['confidence']:.2f}",
            "#facc15",
            label_font,
            width=5,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=92)
    return output_path


def make_contact_sheets(
    ranked: list[dict[str, Any]],
    overlay_paths: list[Path],
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    title_font = font(22)
    tile_width, image_height, header_height = 640, 360, 42
    for start in range(0, len(ranked), 6):
        subset = ranked[start : start + 6]
        subset_paths = overlay_paths[start : start + 6]
        sheet = Image.new("RGB", (1280, (image_height + header_height) * 3), "#111827")
        draw = ImageDraw.Draw(sheet)
        for local_index, (item, overlay_path) in enumerate(zip(subset, subset_paths)):
            column, row = local_index % 2, local_index // 2
            x, y = column * tile_width, row * (image_height + header_height)
            title = (
                f"RANK {start + local_index + 1:03d} · SAMPLE {item['sample_index']:03d}"
                f" · risk {item['risk_score']:.1f} · +{item['candidate_count']}"
            )
            draw.text((x + 10, y + 7), title, fill="white", font=title_font)
            tile = Image.open(overlay_path).convert("RGB").resize((tile_width, image_height))
            sheet.paste(tile, (x, y + header_height))
        path = output_dir / f"review-contact-sheet-{len(paths) + 1:02d}.jpg"
        sheet.save(path, quality=92)
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--model", type=Path, default=Path("yolo11n.pt"))
    parser.add_argument("--confidence", type=float, default=0.15)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--candidate-nms-iou", type=float, default=0.50)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--review-labels", nargs="+", choices=sorted(PRIORITY))
    parser.add_argument("--top", type=int, default=60)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    image_dir = manifest_path.parent / "images"
    output_root = args.output or (manifest_path.parent / "review-queue-lowconf-v1")
    output_root = output_root.resolve()
    image_paths = [image_dir / item["file_name"] for item in manifest["samples"]]
    missing = [str(path) for path in image_paths if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing sample images: {missing[:3]}")

    sample_by_path = {
        str(path.resolve()): sample for path, sample in zip(image_paths, manifest["samples"])
    }
    model = YOLO(str(args.model))
    raw_by_sample: dict[int, list[dict[str, Any]]] = {}
    raw_predictions: list[dict[str, Any]] = []
    batch_size = max(1, args.batch_size)
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        results = model.predict(
            source=[str(path) for path in batch_paths],
            conf=args.confidence,
            imgsz=args.image_size,
            device=args.device,
            batch=len(batch_paths),
            verbose=False,
        )
        for result in results:
            sample = sample_by_path[str(Path(result.path).resolve())]
            sample_index = int(sample["sample_index"])
            if result.boxes is None:
                continue
            for box_index, box in enumerate(result.boxes):
                model_label = str(result.names[int(box.cls.item())])
                label = MODEL_MAPPING.get(model_label)
                if not label:
                    continue
                prediction = {
                    "prediction_id": f"review_{sample_index}_{box_index}",
                    "scene_id": sample["scene_id"],
                    "frame": int(sample["source_frame"]),
                    "sample_index": sample_index,
                    "label": label,
                    "confidence": round(float(box.conf.item()), 4),
                    "bbox": [round(float(value), 2) for value in box.xyxy[0].tolist()],
                    "source": "auto",
                }
                raw_predictions.append(prediction)

    processed, postprocessing = postprocess_predictions(
        raw_predictions, nms_iou_threshold=args.candidate_nms_iou
    )
    for prediction in processed:
        raw_by_sample.setdefault(int(prediction["sample_index"]), []).append(
            {
                "label": prediction["label"],
                "confidence": prediction["confidence"],
                "bbox_xyxy": prediction["bbox"],
            }
        )

    ranked: list[dict[str, Any]] = []
    candidate_classes: Counter[str] = Counter()
    for sample in manifest["samples"]:
        baseline = sample["annotations"]
        candidates = find_unmatched_candidates(
            baseline,
            raw_by_sample.get(int(sample["sample_index"]), []),
            args.match_iou,
        )
        if args.review_labels:
            candidates = [item for item in candidates if item["label"] in args.review_labels]
        candidate_classes.update(item["label"] for item in candidates)
        risk_score = sum(PRIORITY.get(item["label"], 1.0) for item in candidates)
        risk_score += sum(0.5 for item in baseline if float(item.get("confidence", 1.0)) < 0.50)
        ranked.append(
            {
                "sample_index": int(sample["sample_index"]),
                "scene_id": sample["scene_id"],
                "source_frame": int(sample["source_frame"]),
                "file_name": sample["file_name"],
                "baseline_count": len(baseline),
                "candidate_count": len(candidates),
                "candidate_class_counts": dict(Counter(item["label"] for item in candidates)),
                "risk_score": round(risk_score, 2),
                "candidates": candidates,
            }
        )
    ranked.sort(key=lambda item: (-item["risk_score"], item["sample_index"]))

    top = ranked[: max(0, min(args.top, len(ranked)))]
    overlays: list[Path] = []
    for item in top:
        sample = manifest["samples"][item["sample_index"] - 1]
        overlays.append(
            make_overlay(
                image_dir / item["file_name"],
                sample["annotations"],
                item["candidates"],
                output_root / "overlays" / f"review-{item['sample_index']:03d}.jpg",
            )
        )
    contact_sheets = make_contact_sheets(top, overlays, output_root / "contact-sheets")

    payload = {
        "source_manifest": str(manifest_path),
        "model": str(args.model.resolve()),
        "model_sha256": sha256(args.model.resolve()),
        "confidence": args.confidence,
        "match_iou": args.match_iou,
        "candidate_nms_iou": args.candidate_nms_iou,
        "image_size": args.image_size,
        "device": args.device,
        "batch_size": batch_size,
        "review_labels": args.review_labels or sorted(PRIORITY),
        "sample_count": len(ranked),
        "frames_with_candidates": sum(item["candidate_count"] > 0 for item in ranked),
        "candidate_count": sum(item["candidate_count"] for item in ranked),
        "candidate_class_counts": dict(candidate_classes),
        "postprocessing": postprocessing,
        "reviewed_by_human": False,
        "mutation_performed": False,
        "ranked_frames": ranked,
        "contact_sheets": [str(path) for path in contact_sheets],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "review-queue.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(output_path),
                "frames_with_candidates": payload["frames_with_candidates"],
                "candidate_count": payload["candidate_count"],
                "candidate_class_counts": payload["candidate_class_counts"],
                "contact_sheet_count": len(contact_sheets),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
