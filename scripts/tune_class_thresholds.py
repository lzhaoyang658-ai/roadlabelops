#!/usr/bin/env python3
"""Evaluate class-specific confidence thresholds from a low-confidence benchmark."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

from benchmark_ground_truth import load_reference

from roadlabelops.tools.detection import postprocess_predictions
from roadlabelops.tools.quality import calculate_quality


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nms-iou", type=float, default=0.75)
    parser.add_argument("--rider-overlap", type=float, default=0.25)
    return parser.parse_args()


def threshold_predictions(
    predictions: list[dict[str, Any]],
    thresholds: dict[str, float],
) -> list[dict[str, Any]]:
    return [
        item for item in predictions
        if float(item["confidence"]) >= thresholds.get(item["label"], thresholds["default"])
    ]


def main() -> None:
    args = parse_args()
    benchmark = json.loads(args.benchmark.read_text())
    raw_predictions = benchmark["results"][0]["raw_predictions"]
    _, _, ground_truth = load_reference(args.ground_truth.resolve())

    results = []
    default_thresholds = (0.35, 0.40, 0.45, 0.50)
    minority_thresholds = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
    for default, bus, motorcycle in itertools.product(
        default_thresholds, minority_thresholds, minority_thresholds
    ):
        thresholds = {"default": default, "bus": bus, "motorcycle": motorcycle}
        candidates = threshold_predictions(raw_predictions, thresholds)
        predictions, postprocessing = postprocess_predictions(
            candidates,
            nms_iou_threshold=args.nms_iou,
            rider_overlap_threshold=args.rider_overlap,
        )
        quality = calculate_quality(predictions, ground_truth).data
        gates = {
            "precision_at_least_0_90": (quality["precision"] or 0.0) >= 0.90,
            "recall_at_least_0_85": (quality["recall"] or 0.0) >= 0.85,
            "clean_frame_rate_at_least_0_80": (quality["clean_frame_rate"] or 0.0) >= 0.80,
        }
        results.append({
            "thresholds": thresholds,
            "quality": quality,
            "postprocessing": postprocessing,
            "gate_result": "PASS" if all(gates.values()) else "FAIL",
        })

    ranked = sorted(results, key=lambda item: (
        item["gate_result"] == "PASS",
        item["quality"]["clean_frame_rate"] or 0.0,
        item["quality"]["f1_score"] or 0.0,
        item["quality"]["precision"] or 0.0,
    ), reverse=True)
    payload = {
        "source_benchmark": str(args.benchmark.resolve()),
        "combination_count": len(results),
        "pass_count": sum(item["gate_result"] == "PASS" for item in results),
        "best": ranked[0],
        "top_10": ranked[:10],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
