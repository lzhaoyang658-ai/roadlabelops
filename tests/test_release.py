import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import pytest

import roadlabelops.tools.release as release_module
from roadlabelops.models import Scene, Session, Stage
from roadlabelops.storage import LocalStore
from roadlabelops.tools.quality import calculate_quality
from roadlabelops.tools.release import (
    CATEGORY_IDS,
    ROAD_LABEL_TAXONOMY,
    build_coco_release,
    verify_coco_release,
)


def _session(*, demo: bool = True) -> Session:
    return Session(
        session_id="session_test",
        name="test",
        source_path="data/raw/test.mp4",
        source_sha256="a" * 64,
        duration_seconds=15,
        fps=25,
        width=1920,
        height=1080,
        status=Stage.QUALITY_CALCULATED,
        demo=demo,
        scenes=[
            Scene(
                "scene_1",
                "session_test",
                0,
                15,
                "scene.mp4",
                cvat_task_id=99 if not demo else None,
                cvat_job_ids=[199] if not demo else [],
                status="completed",
                final_count=1,
            )
        ],
    )


def _build_release(
    store: LocalStore,
    session: Session,
    version: str,
    annotations: list[dict],
    **kwargs,
):
    normalised, error = release_module._normalise_annotations(session, annotations)
    predictions = []
    if error is None:
        predictions = [
            {
                "prediction_id": f"prediction-{index}",
                "scene_id": item["scene_id"],
                "frame": item["frame"],
                "label": item["label"],
                "confidence": 0.9,
                "bbox": item["bbox"],
                "source": "auto",
            }
            for index, item in enumerate(normalised, start=1)
        ]
    evaluated = {(scene.scene_id, 0) for scene in session.scenes}
    reviewed_jobs = 1 if session.demo else 0
    reason = None if session.demo else "Historical rejection data is unavailable."
    quality = calculate_quality(
        predictions,
        normalised if error is None else [],
        reviewed_jobs,
        reviewed_jobs,
        evaluated_frame_keys=evaluated,
        first_pass_acceptance_reason=reason,
    ).data
    return build_coco_release(
        store,
        session,
        version,
        annotations,
        quality_report=kwargs.pop("quality_report", quality),
        predictions=kwargs.pop("predictions", predictions),
        evaluated_frame_keys=kwargs.pop("evaluated_frame_keys", evaluated),
        accepted_jobs=kwargs.pop("accepted_jobs", reviewed_jobs),
        reviewed_jobs=kwargs.pop("reviewed_jobs", reviewed_jobs),
        first_pass_acceptance_reason=kwargs.pop(
            "first_pass_acceptance_reason", reason
        ),
        **kwargs,
    )


def _rewrite_release_integrity(store: LocalStore, release: Path) -> None:
    manifest_path = release / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload_sha256 = release_module._payload_digests(release)
    manifest["payload_sha256"] = payload_sha256
    manifest["file_sha256"] = payload_sha256
    manifest["export_sha256"] = payload_sha256["annotations.coco.json"]
    manifest["quality_sha256"] = payload_sha256["quality.json"]
    manifest["predictions_sha256"] = payload_sha256["predictions.json"]
    manifest["yolo_metadata_sha256"] = payload_sha256["dataset.yaml"]
    store.write_json_atomic(manifest_path, manifest)
    store.write_json_atomic(
        release / "receipt.json",
        release_module._receipt(manifest["release_id"], manifest, manifest_path),
    )


def test_release_is_immutable_and_has_verifiable_receipt(tmp_path: Path) -> None:
    store = LocalStore(tmp_path / "data", tmp_path / "runtime")
    first = _build_release(store, _session(), "1.0.0", [{"id": 1, "label": "car"}])
    second = _build_release(store, _session(), "1.0.0", [{"id": 1, "label": "car"}])

    assert first.ok, first.error
    assert first.data["receipt"]["valid"] is True
    assert first.data["receipt"]["release_id"] == "session_test-v1.0.0"
    assert not second.ok
    assert second.error and second.error["code"] == "RELEASE_EXISTS"
    release = Path(first.data["path"])
    manifest = json.loads((release / "manifest.json").read_text())
    assert manifest["taxonomy"] == list(ROAD_LABEL_TAXONOMY)
    assert manifest["file_sha256"] == manifest["payload_sha256"]
    assert "annotations.coco.json" in manifest["payload_sha256"]
    assert (release / "images" / "scene_1_frame_000000.jpg").is_file()


def test_release_rejects_unknown_category_bad_box_and_unknown_scene(tmp_path: Path) -> None:
    store = LocalStore(tmp_path / "data", tmp_path / "runtime")
    bad_category = _build_release(store, _session(), "1.0.0", [{"label": "scooter"}])
    bad_box = _build_release(
        store, _session(), "1.0.1", [{"label": "car", "bbox": [5, 2, 5, 6]}]
    )
    unknown_scene = _build_release(
        store, _session(), "1.0.2", [{"label": "car", "scene_id": "other"}]
    )

    assert bad_category.error and bad_category.error["code"] == "UNKNOWN_CATEGORY"
    assert bad_box.error and bad_box.error["code"] == "BBOX_OUT_OF_BOUNDS"
    assert unknown_scene.error and unknown_scene.error["code"] == "UNKNOWN_IMAGE_REFERENCE"


def test_verifier_detects_tampering_missing_and_extra_payloads(tmp_path: Path) -> None:
    store = LocalStore(tmp_path / "data", tmp_path / "runtime")
    result = _build_release(store, _session(), "1.0.0", [{"label": "car"}])
    assert result.ok
    release = Path(result.data["path"])
    (release / "annotations.coco.json").write_text("tampered")
    (release / "untracked.txt").write_text("unexpected")
    (release / "images" / "scene_1_frame_000000.jpg").unlink()

    verified = verify_coco_release(release)
    codes = {issue["code"] for issue in verified.data["receipt"]["issues"]}
    assert not verified.ok
    assert {"PAYLOAD_SHA256_MISMATCH", "RELEASE_FILE_EXTRA", "RELEASE_FILE_MISSING"} <= codes


def test_verifier_rejects_symlinks_without_hashing_their_targets(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = LocalStore(tmp_path / "data", tmp_path / "runtime")
    result = _build_release(store, _session(), "1.0.0", [{"label": "car"}])
    release = Path(result.data["path"])
    outside = tmp_path / "outside.txt"
    outside.write_text("must not enter a release", encoding="utf-8")
    (release / "linked.txt").symlink_to(outside)

    def fail_if_hashed(_path: Path) -> str:
        raise AssertionError("no release content may be hashed after a symlink is found")

    monkeypatch.setattr(release_module, "_digest", fail_if_hashed)

    verified = verify_coco_release(release)

    codes = {issue["code"] for issue in verified.data["receipt"]["issues"]}
    assert "RELEASE_SYMLINK_FORBIDDEN" in codes


def test_verifier_rejects_receipt_rewrite_and_extra_fields(tmp_path: Path) -> None:
    store = LocalStore(tmp_path / "data", tmp_path / "runtime")
    result = _build_release(store, _session(), "1.0.0", [{"label": "car"}])
    release = Path(result.data["path"])
    receipt_path = release / "receipt.json"
    trusted_receipt_sha256 = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["manifest_file"] = "rewritten.json"
    receipt["untrusted_note"] = "tampered"
    store.write_json_atomic(receipt_path, receipt)

    verified = verify_coco_release(
        release,
        expected_receipt_sha256=trusted_receipt_sha256,
    )

    codes = {issue["code"] for issue in verified.data["receipt"]["issues"]}
    assert not verified.ok
    assert {
        "RECEIPT_SHA256_MISMATCH",
        "RECEIPT_MANIFEST_FILE_INVALID",
        "RECEIPT_DOCUMENT_INVALID",
    } <= codes


def test_verifier_checks_an_external_manifest_anchor(tmp_path: Path) -> None:
    store = LocalStore(tmp_path / "data", tmp_path / "runtime")
    result = _build_release(store, _session(), "1.0.0", [{"label": "car"}])
    release = Path(result.data["path"])
    trusted_hash = json.loads((release / "receipt.json").read_text())["manifest_sha256"]
    manifest_path = release / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["created_at"] = "tampered"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    verified = verify_coco_release(
        release,
        expected_manifest_sha256=trusted_hash,
    )

    codes = {issue["code"] for issue in verified.data["receipt"]["issues"]}
    assert "MANIFEST_SHA256_MISMATCH" in codes


def test_real_release_requires_materialized_source_frames(tmp_path: Path) -> None:
    store = LocalStore(tmp_path / "data", tmp_path / "runtime")
    session = _session(demo=False)
    session.scenes[0].final_count = 1
    result = _build_release(
        store,
        session,
        "1.0.0",
        [{"label": "car", "scene_id": "scene_1", "frame": 0, "bbox": [0, 0, 1, 1]}],
    )
    assert not result.ok
    assert result.error and result.error["code"] == "RELEASE_SOURCE_MISSING"
    assert not (store.releases_dir / "session_test-v1.0.0").exists()


def test_real_release_requires_recorded_scene_sha256(tmp_path: Path) -> None:
    store = LocalStore(tmp_path / "data", tmp_path / "runtime")
    source = store.raw_dir / "source.mp4"
    scene_video = store.scenes_dir / "scene.mp4"
    source.write_bytes(b"source")
    scene_video.write_bytes(b"scene")
    session = _session(demo=False)
    session.source_path = str(source)
    session.source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    session.scenes[0].video_path = str(scene_video)
    session.scenes[0].video_sha256 = None

    result = _build_release(
        store,
        session,
        "1.0.0",
        [{"label": "car", "scene_id": "scene_1", "frame": 0, "bbox": [0, 0, 1, 1]}],
    )

    assert not result.ok
    assert result.error and result.error["code"] == "SCENE_SHA256_INVALID"
    assert not (store.releases_dir / "session_test-v1.0.0").exists()


def test_fixed_categories_are_in_export(tmp_path: Path) -> None:
    store = LocalStore(tmp_path / "data", tmp_path / "runtime")
    result = _build_release(store, _session(), "1.0.0", [{"label": "car"}])
    coco = json.loads((Path(result.data["path"]) / "annotations.coco.json").read_text())
    assert coco["categories"] == [
        {"id": CATEGORY_IDS[label], "name": label} for label in ROAD_LABEL_TAXONOMY
    ]


def test_release_rejects_scene_id_path_traversal_before_writing(tmp_path: Path) -> None:
    store = LocalStore(tmp_path / "data", tmp_path / "runtime")
    session = _session()
    session.scenes[0].scene_id = "../../escaped"
    sentinel = tmp_path / "escaped_frame_000000.jpg"
    sentinel.write_text("keep", encoding="utf-8")

    result = _build_release(store, session, "1.0.0", [{"label": "car"}])

    assert not result.ok
    assert result.error and result.error["code"] == "INVALID_SCENE_ID"
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert not (store.releases_dir / "session_test-v1.0.0").exists()


def test_verifier_rejects_release_root_symlink_without_reading_target(tmp_path: Path) -> None:
    store = LocalStore(tmp_path / "data", tmp_path / "runtime")
    result = _build_release(store, _session(), "1.0.0", [{"label": "car"}])
    alias = tmp_path / "release-alias"
    alias.symlink_to(Path(result.data["path"]), target_is_directory=True)

    verified = verify_coco_release(alias)

    assert not verified.ok
    assert verified.error and verified.error["code"] == "RELEASE_SYMLINK_FORBIDDEN"


def test_real_release_rehashes_the_original_source(tmp_path: Path) -> None:
    store = LocalStore(tmp_path / "data", tmp_path / "runtime")
    source = store.raw_dir / "source.mp4"
    scene_video = store.scenes_dir / "scene.mp4"
    source.write_bytes(b"changed source")
    scene_video.write_bytes(b"scene")
    session = _session(demo=False)
    session.source_path = str(source)
    session.scenes[0].video_path = str(scene_video)
    session.scenes[0].final_count = 1

    result = _build_release(
        store,
        session,
        "1.0.0",
        [{"label": "car", "scene_id": "scene_1", "frame": 0, "bbox": [0, 0, 1, 1]}],
    )

    assert not result.ok
    assert result.error and result.error["code"] == "SOURCE_SHA256_MISMATCH"


def test_failed_publication_never_exposes_a_partial_release(tmp_path: Path, monkeypatch) -> None:
    store = LocalStore(tmp_path / "data", tmp_path / "runtime")

    def fail_rename(_source: Path, _target: Path) -> None:
        raise OSError("simulated filesystem failure")

    monkeypatch.setattr(release_module.os, "rename", fail_rename)
    result = _build_release(store, _session(), "1.0.0", [{"label": "car"}])

    assert not result.ok
    assert result.error and result.error["code"] == "RELEASE_BUILD_FAILED"
    assert not (store.releases_dir / "session_test-v1.0.0").exists()
    assert not list(store.releases_dir.glob(".*.tmp"))


def test_concurrent_publication_has_exactly_one_winner(tmp_path: Path) -> None:
    store = LocalStore(tmp_path / "data", tmp_path / "runtime")

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda _: _build_release(
                    store, _session(), "1.0.0", [{"label": "car"}]
                ),
                range(2),
            )
        )

    assert sum(result.ok for result in results) == 1
    assert sum(
        bool(result.error and result.error["code"] == "RELEASE_EXISTS")
        for result in results
    ) == 1
    verified = verify_coco_release(store.releases_dir / "session_test-v1.0.0")
    assert verified.ok, verified.error


def test_new_manifest_version_and_quality_are_self_verified(tmp_path: Path) -> None:
    store = LocalStore(tmp_path / "data", tmp_path / "runtime")
    result = _build_release(
        store,
        _session(),
        "1.2.3",
        [{"label": "car"}],
    )

    assert result.ok, result.error
    release = Path(result.data["path"])
    manifest = json.loads((release / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "2.0.0"
    assert manifest["version"] == "1.2.3"
    assert manifest["formats"] == ["COCO", "YOLO"]
    assert manifest["quality_sha256"] == manifest["payload_sha256"]["quality.json"]
    assert manifest["predictions_sha256"] == manifest["payload_sha256"]["predictions.json"]
    dataset_path = release / "dataset.yaml"
    assert dataset_path.is_file()
    metadata = dataset_path.read_text(encoding="utf-8")
    assert "path:" not in metadata
    assert "train: images\nval: images\n" in metadata
    check_det_dataset = pytest.importorskip("ultralytics.data.utils").check_det_dataset
    checked = check_det_dataset(str(dataset_path), autodownload=False)
    assert checked["train"] == str(release / "images")
    assert checked["val"] == str(release / "images")
    assert (release / "labels" / "scene_1_frame_000000.txt").is_file()
    assert verify_coco_release(
        release,
        expected_session_id="session_test",
        expected_version="1.2.3",
        expected_source_sha256="a" * 64,
    ).ok


def test_demo_release_jpeg_matches_coco_geometry_and_loads_as_yolo(
    tmp_path: Path,
) -> None:
    store = LocalStore(tmp_path / "data", tmp_path / "runtime")
    session = _session()
    session.width = 37
    session.height = 23
    result = _build_release(store, session, "1.2.4", [{"label": "car"}])

    assert result.ok, result.error
    release = Path(result.data["path"])
    coco = json.loads((release / "annotations.coco.json").read_text(encoding="utf-8"))
    image = coco["images"][0]
    image_path = release / image["file_name"]
    assert release_module._jpeg_dimensions(image_path) == (37, 23)
    assert (image["width"], image["height"]) == (37, 23)

    # Ultralytics writes labels.cache while scanning, so load a disposable copy
    # rather than mutating the already-published immutable release.
    check_det_dataset = pytest.importorskip("ultralytics.data.utils").check_det_dataset
    yolo_dataset = pytest.importorskip("ultralytics.data.dataset").YOLODataset
    dataset_copy = tmp_path / "yolo-copy"
    shutil.copytree(release, dataset_copy)
    checked = check_det_dataset(str(dataset_copy / "dataset.yaml"), autodownload=False)
    dataset = yolo_dataset(
        img_path=checked["train"],
        data=checked,
        task="detect",
        imgsz=64,
        augment=False,
        cache=False,
    )
    assert len(dataset) == 1


def test_demo_jpeg_uses_libjpeg_safe_dimension_boundary(tmp_path: Path) -> None:
    image_module = pytest.importorskip("PIL.Image")
    boundary_image = image_module.open(BytesIO(release_module._demo_jpeg(65_500, 10)))
    boundary_image.load()

    assert boundary_image.size == (65_500, 10)
    progressive_bytes = BytesIO()
    image_module.new("RGB", (32, 24), "gray").save(
        progressive_bytes, "JPEG", progressive=True
    )
    progressive_path = tmp_path / "progressive.jpg"
    progressive_path.write_bytes(progressive_bytes.getvalue())
    assert release_module._jpeg_dimensions(progressive_path) == (32, 24)
    with pytest.raises(ValueError, match="65500"):
        release_module._demo_jpeg(65_501, 10)
    store = LocalStore(tmp_path / "data", tmp_path / "runtime")
    oversized = _session()
    oversized.width = 65_501
    result = _build_release(store, oversized, "1.2.6", [{"label": "car"}])
    assert not result.ok
    assert result.error and result.error["code"] == "INVALID_SESSION_GEOMETRY"


@pytest.mark.parametrize(
    ("version", "corrupt"),
    [
        pytest.param(
            "1.3.0",
            lambda _image: (
                b"\xff\xd8\xff\xc0\x00\x07\x08"
                + (1080).to_bytes(2, "big")
                + (1920).to_bytes(2, "big")
            ),
            id="same-size-sof-only",
        ),
        pytest.param("1.3.1", lambda image: image[:-2], id="missing-eoi"),
        pytest.param("1.3.2", lambda image: image[:95], id="truncated-segment"),
        pytest.param("1.3.3", lambda image: image + b"trailing", id="trailing-garbage"),
    ],
)
def test_verifier_rejects_structurally_invalid_demo_jpeg_after_coordinated_rehash(
    tmp_path: Path,
    version: str,
    corrupt,
) -> None:
    store = LocalStore(tmp_path / "data", tmp_path / "runtime")
    result = _build_release(store, _session(), version, [{"label": "car"}])
    assert result.ok, result.error
    release = Path(result.data["path"])
    image_path = release / "images" / "scene_1_frame_000000.jpg"
    image_path.write_bytes(corrupt(image_path.read_bytes()))
    _rewrite_release_integrity(store, release)

    verified = verify_coco_release(release)

    codes = {issue["code"] for issue in verified.data["receipt"]["issues"]}
    assert not verified.ok
    assert {"COCO_IMAGE_FILE_INVALID", "DEMO_IMAGE_CONTENT_INVALID"} <= codes


def test_verifier_rejects_jpeg_geometry_that_differs_from_coco(tmp_path: Path) -> None:
    store = LocalStore(tmp_path / "data", tmp_path / "runtime")
    result = _build_release(store, _session(), "1.2.5", [{"label": "car"}])
    assert result.ok, result.error
    release = Path(result.data["path"])
    image_path = release / "images" / "scene_1_frame_000000.jpg"
    image_path.write_bytes(release_module._demo_jpeg(32, 32))
    _rewrite_release_integrity(store, release)

    verified = verify_coco_release(release)

    codes = {issue["code"] for issue in verified.data["receipt"]["issues"]}
    assert not verified.ok
    assert "COCO_IMAGE_DIMENSIONS_MISMATCH" in codes


def test_verifier_requires_canonical_flat_image_paths_after_coordinated_rehash(
    tmp_path: Path,
) -> None:
    store = LocalStore(tmp_path / "data", tmp_path / "runtime")
    result = _build_release(store, _session(), "1.3.5", [{"label": "car"}])
    assert result.ok, result.error
    release = Path(result.data["path"])
    original = release / "images" / "scene_1_frame_000000.jpg"
    nested = release / "images" / "nested" / original.name
    nested.parent.mkdir()
    original.rename(nested)
    coco_path = release / "annotations.coco.json"
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    coco["images"][0]["file_name"] = f"images/nested/{original.name}"
    store.write_json_atomic(coco_path, coco)
    _rewrite_release_integrity(store, release)

    verified = verify_coco_release(release)

    codes = {issue["code"] for issue in verified.data["receipt"]["issues"]}
    assert not verified.ok
    assert "COCO_IMAGE_PATH_INVALID" in codes


@pytest.mark.parametrize(
    ("demo", "extra_relative", "version"),
    [
        pytest.param(True, "images/unreferenced.jpg", "1.3.6", id="demo-flat"),
        pytest.param(
            True,
            "images/nested/unreferenced.jpg",
            "1.3.7",
            id="demo-nested",
        ),
        pytest.param(False, "images/unreferenced.jpg", "1.3.8", id="real-flat"),
        pytest.param(
            False,
            "images/nested/unreferenced.jpg",
            "1.3.9",
            id="real-nested",
        ),
    ],
)
def test_verifier_rejects_unreferenced_manifested_images_after_coordinated_rehash(
    tmp_path: Path,
    monkeypatch,
    demo: bool,
    extra_relative: str,
    version: str,
) -> None:
    store = LocalStore(tmp_path / "data", tmp_path / "runtime")
    session = _session(demo=demo)
    if not demo:
        source = store.raw_dir / "source.mp4"
        scene_video = store.scenes_dir / "scene.mp4"
        source.write_bytes(b"source")
        scene_video.write_bytes(b"verified-scene")
        session.source_path = str(source)
        session.source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        session.scenes[0].video_path = str(scene_video)
        session.scenes[0].video_sha256 = hashlib.sha256(scene_video.read_bytes()).hexdigest()

        def materialize_valid_jpeg(
            output: Path,
            _scene: Scene,
            _frame: int,
            *,
            demo: bool,
            width: int,
            height: int,
            source_path: Path | None = None,
        ) -> None:
            assert not demo
            assert source_path is not None
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(release_module._demo_jpeg(width, height))

        monkeypatch.setattr(
            release_module, "_materialize_image", materialize_valid_jpeg
        )

    result = _build_release(
        store,
        session,
        version,
        [{"label": "car", "scene_id": "scene_1", "frame": 0, "bbox": [0, 0, 1, 1]}],
    )
    assert result.ok, result.error
    release = Path(result.data["path"])
    extra = release / extra_relative
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_bytes(b"unreferenced garbage")
    _rewrite_release_integrity(store, release)

    verified = verify_coco_release(release)

    codes = {issue["code"] for issue in verified.data["receipt"]["issues"]}
    assert not verified.ok
    assert "COCO_IMAGE_SET_INVALID" in codes


def test_real_verifier_fully_decodes_images_after_coordinated_dht_rehash(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = LocalStore(tmp_path / "data", tmp_path / "runtime")
    source = store.raw_dir / "source.mp4"
    scene_video = store.scenes_dir / "scene.mp4"
    source.write_bytes(b"source")
    scene_video.write_bytes(b"verified-scene")
    session = _session(demo=False)
    session.source_path = str(source)
    session.source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    session.scenes[0].video_path = str(scene_video)
    session.scenes[0].video_sha256 = hashlib.sha256(scene_video.read_bytes()).hexdigest()

    def materialize_valid_jpeg(
        output: Path,
        _scene: Scene,
        _frame: int,
        *,
        demo: bool,
        width: int,
        height: int,
        source_path: Path | None = None,
    ) -> None:
        assert not demo
        assert source_path is not None
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(release_module._demo_jpeg(width, height))

    monkeypatch.setattr(release_module, "_materialize_image", materialize_valid_jpeg)
    result = _build_release(
        store,
        session,
        "1.3.4",
        [{"label": "car", "scene_id": "scene_1", "frame": 0, "bbox": [0, 0, 1, 1]}],
    )
    assert result.ok, result.error
    release = Path(result.data["path"])
    image_path = release / "images" / "scene_1_frame_000000.jpg"
    corrupted = bytearray(image_path.read_bytes())
    dht_marker = corrupted.index(b"\xff\xc4")
    corrupted[dht_marker + 21] = 15  # Invalid DC coefficient category.
    image_path.write_bytes(corrupted)
    assert release_module._jpeg_dimensions(image_path) == (1920, 1080)
    _rewrite_release_integrity(store, release)

    verified = verify_coco_release(release)

    codes = {issue["code"] for issue in verified.data["receipt"]["issues"]}
    assert not verified.ok
    assert "COCO_IMAGE_DECODE_FAILED" in codes


def test_release_preserves_object_attributes_and_scene_tags(tmp_path: Path) -> None:
    store = LocalStore(tmp_path / "data", tmp_path / "runtime")
    session = _session()
    session.scenes[0].scene_tags = [
        {
            "scene_id": "scene_1",
            "frame": 0,
            "label": "lighting",
            "source": "manual",
            "attributes": [{"name": "lighting", "value": "night"}],
        }
    ]
    result = _build_release(
        store,
        session,
        "2.0.0",
        [
            {
                "label": "car",
                "bbox": [0, 0, 192, 108],
                "attributes": {
                    "occlusion": "partial",
                    "motion": "moving",
                    "direction": "same",
                },
            }
        ],
    )

    assert result.ok, result.error
    release = Path(result.data["path"])
    coco = json.loads((release / "annotations.coco.json").read_text(encoding="utf-8"))
    manifest = json.loads((release / "manifest.json").read_text(encoding="utf-8"))
    assert coco["annotations"][0]["attributes"] == {
        "occlusion": "partial",
        "motion": "moving",
        "direction": "same",
    }
    assert manifest["scene_lineage"][0]["scene_tags"] == [
        {"frame": 0, "label": "lighting", "source": "manual", "value": "night"}
    ]
    assert (release / "labels" / "scene_1_frame_000000.txt").read_text().startswith("0 ")


def test_release_treats_blank_cvat_select_values_as_absent(tmp_path: Path) -> None:
    store = LocalStore(tmp_path / "data", tmp_path / "runtime")
    result = _build_release(
        store,
        _session(),
        "1.0.0",
        [
            {
                "label": "car",
                "attributes": [
                    {"name": "occlusion", "value": ""},
                    {"name": "motion", "value": "   "},
                    {"name": "direction", "value": "same"},
                ],
            }
        ],
    )

    assert result.ok, result.error
    coco = json.loads(
        (Path(result.data["path"]) / "annotations.coco.json").read_text(
            encoding="utf-8"
        )
    )
    assert coco["annotations"][0]["attributes"] == {"direction": "same"}


def test_release_rejects_values_outside_the_v1_attribute_schema(tmp_path: Path) -> None:
    store = LocalStore(tmp_path / "data", tmp_path / "runtime")
    result = _build_release(
        store,
        _session(),
        "1.0.0",
        [{"label": "car", "attributes": {"occlusion": "invisible"}}],
    )

    assert not result.ok
    assert result.error and result.error["code"] == "INVALID_ATTRIBUTES"


def test_verifier_rejects_semantic_tamper_even_when_internal_hashes_are_rewritten(
    tmp_path: Path,
) -> None:
    store = LocalStore(tmp_path / "data", tmp_path / "runtime")
    result = _build_release(store, _session(), "1.0.0", [{"label": "car"}])
    release = Path(result.data["path"])
    coco_path = release / "annotations.coco.json"
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    coco["annotations"][0]["bbox"][0] = -1
    store.write_json_atomic(coco_path, coco)

    manifest_path = release / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload_sha256 = release_module._payload_digests(release)
    manifest["payload_sha256"] = payload_sha256
    manifest["file_sha256"] = payload_sha256
    manifest["export_sha256"] = payload_sha256["annotations.coco.json"]
    store.write_json_atomic(manifest_path, manifest)
    store.write_json_atomic(
        release / "receipt.json",
        release_module._receipt(manifest["release_id"], manifest, manifest_path),
    )

    verified = verify_coco_release(release)

    codes = {issue["code"] for issue in verified.data["receipt"]["issues"]}
    assert not verified.ok
    assert "COCO_BBOX_INVALID" in codes


def test_verifier_rejects_forged_or_duplicated_scene_and_cvat_lineage(
    tmp_path: Path,
) -> None:
    store = LocalStore(tmp_path / "data", tmp_path / "runtime")
    session = _session()
    session.scenes[0].cvat_task_id = 99
    session.scenes[0].cvat_job_ids = [199]
    result = _build_release(store, session, "1.0.0", [{"label": "car"}])
    assert result.ok, result.error
    release = Path(result.data["path"])
    manifest_path = release / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    forged = dict(manifest["scene_lineage"][0])
    forged["cvat_task_id"] = "forged-task"
    forged["cvat_job_ids"] = [199, 199]
    forged["final_count"] = 999_999
    manifest["scene_lineage"] = [forged, dict(forged)]
    manifest["cvat_task_ids"] = ["forged-task", "forged-task"]
    store.write_json_atomic(manifest_path, manifest)
    store.write_json_atomic(
        release / "receipt.json",
        release_module._receipt(manifest["release_id"], manifest, manifest_path),
    )

    verified = verify_coco_release(release)

    codes = {issue["code"] for issue in verified.data["receipt"]["issues"]}
    assert not verified.ok
    assert {"SCENE_LINEAGE_INVALID", "CVAT_LINEAGE_INVALID"} <= codes
    assert "SCENE_FINAL_COUNT_MISMATCH" in codes


def test_verifier_recomputes_quality_from_frozen_predictions(tmp_path: Path) -> None:
    store = LocalStore(tmp_path / "data", tmp_path / "runtime")
    session = _session()
    annotations = [{"label": "car"}]
    normalised, error = release_module._normalise_annotations(session, annotations)
    assert error is None
    evaluated = {("scene_1", 0)}
    original_quality = calculate_quality(
        [],
        normalised,
        1,
        1,
        evaluated_frame_keys=evaluated,
        first_pass_acceptance_reason=None,
    ).data
    result = _build_release(
        store,
        session,
        "1.0.0",
        annotations,
        predictions=[],
        quality_report=original_quality,
    )
    assert result.ok, result.error
    release = Path(result.data["path"])

    fabricated_predictions = [
        {
            "prediction_id": "fabricated",
            "scene_id": "scene_1",
            "frame": 0,
            "label": "car",
            "confidence": 1.0,
            "bbox": [0.0, 0.0, 1.0, 1.0],
            "source": "auto",
        }
    ]
    fabricated_quality = calculate_quality(
        fabricated_predictions,
        normalised,
        1,
        1,
        evaluated_frame_keys=evaluated,
        first_pass_acceptance_reason=None,
    ).data
    store.write_json_atomic(release / "quality.json", fabricated_quality)
    manifest_path = release / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload_sha256 = release_module._payload_digests(release)
    manifest["payload_sha256"] = payload_sha256
    manifest["file_sha256"] = payload_sha256
    manifest["quality_sha256"] = payload_sha256["quality.json"]
    store.write_json_atomic(manifest_path, manifest)
    store.write_json_atomic(
        release / "receipt.json",
        release_module._receipt(manifest["release_id"], manifest, manifest_path),
    )

    verified = verify_coco_release(release)

    codes = {issue["code"] for issue in verified.data["receipt"]["issues"]}
    assert not verified.ok
    assert "QUALITY_RECOMPUTATION_MISMATCH" in codes


def test_real_release_extracts_only_from_verified_private_scene_snapshot(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = LocalStore(tmp_path / "data", tmp_path / "runtime")
    source = store.raw_dir / "source.mp4"
    scene_video = store.scenes_dir / "scene.mp4"
    source.write_bytes(b"source")
    scene_video.write_bytes(b"verified-scene-a")
    session = _session(demo=False)
    session.source_path = str(source)
    session.source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    session.scenes[0].video_path = str(scene_video)
    session.scenes[0].video_sha256 = hashlib.sha256(scene_video.read_bytes()).hexdigest()
    observed_sources: list[Path] = []

    def fake_materialize(
        output: Path,
        _scene: Scene,
        _frame: int,
        *,
        demo: bool,
        width: int,
        height: int,
        source_path: Path | None = None,
    ) -> None:
        assert not demo
        assert (width, height) == (1920, 1080)
        assert source_path is not None
        scene_video.write_bytes(b"mutable-scene-b")
        assert source_path != scene_video
        assert source_path.read_bytes() == b"verified-scene-a"
        observed_sources.append(source_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(release_module._DEMO_JPEG)

    monkeypatch.setattr(release_module, "_materialize_image", fake_materialize)
    result = _build_release(
        store,
        session,
        "1.0.0",
        [{"label": "car", "scene_id": "scene_1", "frame": 0, "bbox": [0, 0, 1, 1]}],
    )

    assert result.ok, result.error
    assert observed_sources
    assert not (Path(result.data["path"]) / ".source-snapshots").exists()
