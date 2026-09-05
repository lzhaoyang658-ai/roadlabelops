from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from scripts.build_manual_class_candidate_pack import (
    _run_yolo,
    box_iou,
    box_overlap_metrics,
    build_candidate_pack,
    normalize_candidates,
)

SYNTHETIC_TASK_A_ID = 41001
SYNTHETIC_TASK_B_ID = 41002


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_normalize_candidates_is_class_aware_hash_bound_and_matches_existing() -> None:
    raw = [
        {
            "label": "traffic_light",
            "model_label": "traffic light",
            "confidence": 0.90,
            "bbox": [10, 10, 30, 40],
        },
        {
            "label": "traffic_light",
            "model_label": "traffic light",
            "confidence": 0.80,
            "bbox": [10, 10, 30, 40],
        },
        {
            "label": "traffic_sign",
            "model_label": "stop sign",
            "confidence": 0.70,
            "bbox": [10, 10, 30, 40],
        },
        {
            "label": "car",
            "model_label": "car",
            "confidence": 0.99,
            "bbox": [40, 10, 80, 40],
        },
    ]

    candidates = normalize_candidates(
        raw,
        task_id=SYNTHETIC_TASK_A_ID,
        frame=3,
        width=100,
        height=80,
        model_sha256="a" * 64,
        source_image_sha256="d" * 64,
        confidence_threshold=0.12,
        nms_iou_threshold=0.30,
        existing_shapes=[{"label": "traffic_light", "points": [10, 10, 30, 40]}],
        existing_match_iou=0.30,
    )

    assert [item["label"] for item in candidates] == ["traffic_light", "traffic_sign"]
    assert candidates[0]["status"] == "already_annotated"
    assert candidates[0]["review_reason"] == "same_label_match"
    assert candidates[1]["status"] == "needs_human_review"
    assert candidates[1]["review_reason"] == "cross_label_overlap"
    assert candidates[1]["existing_overlaps"][0]["candidate_coverage"] == 1.0
    assert candidates[0]["candidate_id"].startswith(
        f"task-{SYNTHETIC_TASK_A_ID}-frame-000003-traffic-light-"
    )
    assert len(candidates[0]["candidate_id"].rsplit("-", 1)[1]) == 20
    assert box_iou(candidates[0]["bbox"], candidates[1]["bbox"]) == 1.0
    assert box_overlap_metrics(candidates[0]["bbox"], candidates[1]["bbox"]) == (
        1.0,
        1.0,
        1.0,
    )


def test_normalize_candidates_supports_explicit_label_selection() -> None:
    candidates = normalize_candidates(
        [
            {
                "label": "car",
                "model_label": "car",
                "confidence": 0.95,
                "bbox": [5, 5, 40, 35],
            },
            {
                "label": "pedestrian",
                "model_label": "person",
                "confidence": 0.90,
                "bbox": [50, 5, 70, 50],
            },
            {
                "label": "traffic_light",
                "model_label": "traffic light",
                "confidence": 0.99,
                "bbox": [80, 5, 90, 30],
            },
        ],
        task_id=SYNTHETIC_TASK_B_ID,
        frame=7,
        width=100,
        height=80,
        model_sha256="b" * 64,
        source_image_sha256="e" * 64,
        confidence_threshold=0.40,
        nms_iou_threshold=0.50,
        existing_shapes=[{"label": "car", "points": [5, 5, 40, 35]}],
        existing_match_iou=0.30,
        review_labels=frozenset({"car", "pedestrian"}),
    )

    assert [(item["label"], item["status"]) for item in candidates] == [
        ("car", "already_annotated"),
        ("pedestrian", "needs_human_review"),
    ]


def _fixture_pack(tmp_path: Path) -> tuple[Path, Path]:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    first = image_dir / "first.jpg"
    second = image_dir / "second.jpg"
    Image.new("RGB", (100, 80), "#334155").save(first, quality=95)
    Image.new("RGB", (100, 80), "#475569").save(second, quality=95)
    pack = {
        "schema_version": "1.0",
        "pack_type": "full_annotation_review",
        "read_only": True,
        "mutation_performed": False,
        "all_frames_included": True,
        "frame_count": 2,
        "task_id": 42,
        "source_snapshot_sha256": "c" * 64,
        "annotation_sha256": "d" * 64,
        "frames": [
            {
                "frame": 0,
                "sample_index": 1,
                "source_path": str(first),
                "actual_sha256": sha256(first),
                "width": 100,
                "height": 80,
                "shapes": [],
            },
            {
                "frame": 5,
                "sample_index": 9,
                "source_path": str(second),
                "actual_sha256": sha256(second),
                "width": 100,
                "height": 80,
                "shapes": [
                    {
                        "id": 1,
                        "label": "traffic_sign",
                        "points": [50, 10, 70, 30],
                    }
                ],
            },
        ],
    }
    pack_path = tmp_path / "review-pack.json"
    pack_path.write_text(json.dumps(pack), encoding="utf-8")
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"fixed-model")
    return pack_path, model_path


def test_build_candidate_pack_is_read_only_complete_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    pack_path, model_path = _fixture_pack(tmp_path)
    pack_before = pack_path.read_bytes()
    model_before = model_path.read_bytes()

    def fake_predictor(*_args):
        return [
            [
                {
                    "label": "traffic_light",
                    "model_label": "traffic light",
                    "confidence": 0.91,
                    "bbox": [10, 10, 30, 40],
                }
            ],
            [
                {
                    "label": "traffic_sign",
                    "model_label": "stop sign",
                    "confidence": 0.88,
                    "bbox": [50, 10, 70, 30],
                }
            ],
        ]

    output_dir = tmp_path / "candidate-pack"
    result = build_candidate_pack(
        pack_path,
        model_path,
        output_dir,
        predictor=fake_predictor,
    )
    payload = json.loads(result.read_text(encoding="utf-8"))

    assert pack_path.read_bytes() == pack_before
    assert model_path.read_bytes() == model_before
    assert payload["read_only"] is True
    assert payload["mutation_performed"] is False
    assert payload["frame_count"] == 2
    assert payload["frames_with_candidates"] == 2
    assert payload["frames_needing_human_review"] == 1
    assert payload["candidate_count"] == 2
    assert payload["needs_human_review_count"] == 1
    assert payload["needs_human_review_counts_by_label"] == {"traffic_light": 1}
    assert payload["needs_human_review_counts_by_reason"] == {"no_same_label_match": 1}
    assert payload["frames"][0]["overlay"] == "overlays/frame-000000.jpg"
    assert "overlay" not in payload["frames"][1]
    assert (output_dir / payload["frames"][0]["overlay"]).is_file()
    assert len(payload["contact_sheets"]) == 1
    assert (output_dir / payload["contact_sheets"][0]).is_file()
    assert not (output_dir / ".incomplete").exists()
    assert not (output_dir / ".inputs").exists()
    assert not list(tmp_path.glob(".candidate-pack.*"))

    original_output = result.read_bytes()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        build_candidate_pack(
            pack_path,
            model_path,
            output_dir,
            predictor=fake_predictor,
        )
    assert result.read_bytes() == original_output


@pytest.mark.parametrize("drift", ["sha", "dimensions"])
def test_build_candidate_pack_rejects_source_drift_without_partial_output(
    tmp_path: Path, drift: str
) -> None:
    pack_path, model_path = _fixture_pack(tmp_path)
    pack = json.loads(pack_path.read_text(encoding="utf-8"))
    if drift == "sha":
        pack["frames"][0]["actual_sha256"] = "0" * 64
    else:
        pack["frames"][0]["width"] = 99
    pack_path.write_text(json.dumps(pack), encoding="utf-8")
    output_dir = tmp_path / "candidate-pack"

    with pytest.raises(ValueError, match="differ.*review pack"):
        build_candidate_pack(
            pack_path,
            model_path,
            output_dir,
            predictor=lambda *_args: [[], []],
        )

    assert not output_dir.exists()
    assert not list(tmp_path.glob(".candidate-pack.*"))


@pytest.mark.parametrize("mutated_input", ["image", "model"])
def test_build_candidate_pack_rejects_inputs_changed_during_prediction(
    tmp_path: Path, mutated_input: str
) -> None:
    pack_path, model_path = _fixture_pack(tmp_path)
    pack = json.loads(pack_path.read_text(encoding="utf-8"))
    output_dir = tmp_path / "candidate-pack"

    def mutating_predictor(*_args):
        if mutated_input == "image":
            Path(pack["frames"][0]["source_path"]).write_bytes(b"changed")
        else:
            model_path.write_bytes(b"changed-model")
        return [[], []]

    with pytest.raises(ValueError, match="changed while"):
        build_candidate_pack(
            pack_path,
            model_path,
            output_dir,
            predictor=mutating_predictor,
        )

    assert not output_dir.exists()
    assert not list(tmp_path.glob(".candidate-pack.*"))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("frame", True, "frame 0 number must be an integer"),
        ("sample_index", "1", "sample_index must be an integer"),
        ("width", 100.0, "width must be an integer"),
    ],
)
def test_build_candidate_pack_rejects_ambiguous_integer_fields(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    pack_path, model_path = _fixture_pack(tmp_path)
    pack = json.loads(pack_path.read_text(encoding="utf-8"))
    pack["frames"][0][field] = value
    pack_path.write_text(json.dumps(pack), encoding="utf-8")

    with pytest.raises(TypeError, match=message):
        build_candidate_pack(
            pack_path,
            model_path,
            tmp_path / "candidate-pack",
            predictor=lambda *_args: [[], []],
        )


def test_build_candidate_pack_uses_frozen_inputs_during_replace_restore(tmp_path: Path) -> None:
    pack_path, model_path = _fixture_pack(tmp_path)
    pack = json.loads(pack_path.read_text(encoding="utf-8"))
    original_image = Path(pack["frames"][0]["source_path"])
    original_image_bytes = original_image.read_bytes()
    original_model_bytes = model_path.read_bytes()

    def aba_predictor(image_paths, frozen_model, *_args):
        assert all(path.parent.name == ".inputs" for path in image_paths)
        assert frozen_model.parent.name == ".inputs"
        assert image_paths[0].read_bytes() == original_image_bytes
        assert frozen_model.read_bytes() == original_model_bytes
        assert image_paths[0].stat().st_ino != original_image.stat().st_ino
        assert frozen_model.stat().st_ino != model_path.stat().st_ino
        original_image.write_bytes(b"temporary replacement")
        model_path.write_bytes(b"temporary model")
        assert image_paths[0].read_bytes() == original_image_bytes
        assert frozen_model.read_bytes() == original_model_bytes
        original_image.write_bytes(original_image_bytes)
        model_path.write_bytes(original_model_bytes)
        return [[], []]

    result = build_candidate_pack(
        pack_path,
        model_path,
        tmp_path / "candidate-pack",
        predictor=aba_predictor,
    )

    assert result.is_file()
    assert original_image.read_bytes() == original_image_bytes
    assert model_path.read_bytes() == original_model_bytes


@pytest.mark.parametrize("mutated_input", ["image", "model"])
def test_build_candidate_pack_rejects_mutated_frozen_inputs(
    tmp_path: Path, mutated_input: str
) -> None:
    pack_path, model_path = _fixture_pack(tmp_path)

    def mutating_predictor(image_paths, frozen_model, *_args):
        if mutated_input == "image":
            image_paths[0].write_bytes(b"changed frozen image")
        else:
            frozen_model.write_bytes(b"changed frozen model")
        return [[], []]

    with pytest.raises(ValueError, match="Frozen .* SHA-256 changed"):
        build_candidate_pack(
            pack_path,
            model_path,
            tmp_path / "candidate-pack",
            predictor=mutating_predictor,
        )

    assert not (tmp_path / "candidate-pack").exists()
    assert not list(tmp_path.glob(".candidate-pack.*"))


def test_build_candidate_pack_does_not_replace_concurrently_created_empty_directory(
    tmp_path: Path,
) -> None:
    pack_path, model_path = _fixture_pack(tmp_path)
    output_dir = tmp_path / "candidate-pack"

    def racing_predictor(*_args):
        output_dir.mkdir()
        return [[], []]

    with pytest.raises(FileExistsError):
        build_candidate_pack(
            pack_path,
            model_path,
            output_dir,
            predictor=racing_predictor,
        )

    assert output_dir.is_dir()
    assert not list(output_dir.iterdir())
    assert not list(tmp_path.glob(".candidate-pack.*"))


@pytest.mark.parametrize("field", ["nms_iou", "existing_match_iou"])
def test_build_candidate_pack_rejects_zero_overlap_threshold(tmp_path: Path, field: str) -> None:
    pack_path, model_path = _fixture_pack(tmp_path)

    with pytest.raises(ValueError, match="greater than 0"):
        build_candidate_pack(
            pack_path,
            model_path,
            tmp_path / "candidate-pack",
            predictor=lambda *_args: [[], []],
            **{field: 0.0},
        )


def test_candidate_identity_normalizes_negative_zero() -> None:
    common = {
        "task_id": 1,
        "frame": 0,
        "width": 100,
        "height": 80,
        "model_sha256": "a" * 64,
        "source_image_sha256": "b" * 64,
        "confidence_threshold": 0.1,
        "nms_iou_threshold": 0.5,
        "existing_shapes": [],
        "existing_match_iou": 0.3,
        "review_labels": frozenset({"car"}),
    }
    negative_zero = normalize_candidates(
        [{"label": "car", "confidence": 0.9, "bbox": [-0.001, 0, 20, 20]}],
        **common,
    )
    positive_zero = normalize_candidates(
        [{"label": "car", "confidence": 0.9, "bbox": [0, 0, 20, 20]}],
        **common,
    )

    assert negative_zero[0]["bbox"] == positive_zero[0]["bbox"]
    assert negative_zero[0]["candidate_id"] == positive_zero[0]["candidate_id"]
    assert "-0.0" not in json.dumps(negative_zero)

    changed_source = normalize_candidates(
        [{"label": "car", "confidence": 0.9, "bbox": [0, 0, 20, 20]}],
        **{**common, "source_image_sha256": "c" * 64},
    )
    changed_model = normalize_candidates(
        [{"label": "car", "confidence": 0.9, "bbox": [0, 0, 20, 20]}],
        **{**common, "model_sha256": "d" * 64},
    )
    assert changed_source[0]["candidate_id"] != positive_zero[0]["candidate_id"]
    assert changed_model[0]["candidate_id"] != positive_zero[0]["candidate_id"]


def test_run_yolo_accepts_normalized_aliases_and_binds_result_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path = tmp_path / "frame.jpg"
    model_path = tmp_path / "model.pt"
    image_path.write_bytes(b"image")
    model_path.write_bytes(b"model")

    class FakeScalar:
        def __init__(self, value: float):
            self.value = value

        def item(self) -> float:
            return self.value

    class FakeYOLO:
        def __init__(self, _path: str):
            self.names = {0: "Person", 1: "traffic_sign"}

        def predict(self, *, source, **_kwargs):
            box = SimpleNamespace(
                cls=FakeScalar(1),
                conf=FakeScalar(0.9),
                xyxy=[SimpleNamespace(tolist=lambda: [1.0, 2.0, 20.0, 30.0])],
            )
            return [SimpleNamespace(path=source[0], boxes=[box], names=self.names)]

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))

    predictions = _run_yolo(
        [image_path],
        model_path,
        0.4,
        0.5,
        960,
        "cpu",
        1,
        frozenset({"pedestrian", "traffic_sign"}),
    )

    assert predictions == [
        [
            {
                "label": "traffic_sign",
                "model_label": "traffic_sign",
                "confidence": 0.9,
                "bbox": [1.0, 2.0, 20.0, 30.0],
            }
        ]
    ]


def test_run_yolo_rejects_missing_class_and_result_path_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path = tmp_path / "frame.jpg"
    other_path = tmp_path / "other.jpg"
    model_path = tmp_path / "model.pt"
    image_path.write_bytes(b"image")
    other_path.write_bytes(b"other")
    model_path.write_bytes(b"model")

    class FakeYOLO:
        def __init__(self, _path: str):
            self.names = ["car"]

        def predict(self, *, source, **_kwargs):
            return [SimpleNamespace(path=str(other_path), boxes=None, names=self.names)]

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))

    with pytest.raises(ValueError, match="no usable class"):
        _run_yolo(
            [image_path],
            model_path,
            0.4,
            0.5,
            960,
            "cpu",
            1,
            frozenset({"traffic_sign"}),
        )

    with pytest.raises(RuntimeError, match="order/path"):
        _run_yolo(
            [image_path],
            model_path,
            0.4,
            0.5,
            960,
            "cpu",
            1,
            frozenset({"car"}),
        )
