from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from roadlabelops.tools.cvat import (
    CvatAdapter,
    CvatConfig,
    FakeCvatAdapter,
    RoadLabelSchema,
    load_road_label_schema,
    road_label_schema_from_mapping,
)


class ClientContext:
    def __init__(self, client: Any) -> None:
        self.client = client

    def __enter__(self) -> Any:
        return self.client

    def __exit__(self, *_args: object) -> None:
        return None


class CapturingProjects:
    def __init__(self, existing: list[Any] | None = None) -> None:
        self.existing = existing or []
        self.requests: list[Any] = []

    def list(self) -> list[Any]:
        return self.existing

    def create(self, request: Any) -> Any:
        self.requests.append(request)
        return SimpleNamespace(id=81)


class ExistingProjects:
    def __init__(self, labels: list[Any]) -> None:
        self.project = SimpleNamespace(id=82, get_labels=lambda: labels)
        self.requests: list[Any] = []

    def list(self) -> list[Any]:
        return [SimpleNamespace(id=self.project.id, name="RoadLabelOps")]

    def retrieve(self, project_id: int) -> Any:
        assert project_id == self.project.id
        return self.project

    def create(self, request: Any) -> Any:
        self.requests.append(request)
        return SimpleNamespace(id=83)


def schema_mapping() -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "labels": [{"name": "car"}, {"name": "pedestrian"}],
        "attributes": {
            "occlusion": ["none", "partial", "heavy"],
            "motion": ["moving", "stopped", "unknown"],
        },
        "scene_tags": {
            "lighting": ["day", "night"],
            "weather": ["clear", "rain", "fog"],
        },
    }


def existing_schema_labels() -> list[Any]:
    def select_attribute(name: str, values: list[str]) -> Any:
        return SimpleNamespace(
            name=name,
            mutable=True,
            input_type="select",
            values=values,
        )

    rectangle_attributes = [
        select_attribute("occlusion", ["none", "partial", "heavy"]),
        select_attribute("motion", ["moving", "stopped", "unknown"]),
    ]
    return [
        SimpleNamespace(
            name="car",
            type="rectangle",
            attributes=list(rectangle_attributes),
        ),
        SimpleNamespace(
            name="pedestrian",
            type="rectangle",
            attributes=list(rectangle_attributes),
        ),
        SimpleNamespace(
            name="lighting",
            type="tag",
            attributes=[select_attribute("lighting", ["day", "night"])],
        ),
        SimpleNamespace(
            name="weather",
            type="tag",
            attributes=[select_attribute("weather", ["clear", "rain", "fog"])],
        ),
    ]


def test_loads_repository_road_label_schema() -> None:
    schema = load_road_label_schema()

    assert schema.schema_version == "1.0.0"
    assert schema.object_labels == (
        "car",
        "bus",
        "truck",
        "motorcycle",
        "bicycle",
        "pedestrian",
        "traffic_light",
        "traffic_sign",
    )
    assert {item.name: item.values for item in schema.rectangle_attributes} == {
        "occlusion": ("none", "partial", "heavy"),
        "motion": ("moving", "stopped", "parked", "unknown"),
        "direction": ("same", "opposite", "crossing", "unknown"),
    }
    assert {item.name: item.values for item in schema.scene_tags} == {
        "lighting": ("day", "night"),
        "weather": ("clear", "rain", "fog"),
        "road_type": ("urban", "highway", "intersection"),
        "traffic_density": ("low", "medium", "high"),
    }


@pytest.mark.parametrize(
    "mutation, match",
    [
        (lambda payload: payload.update(labels=[]), "labels must be a non-empty list"),
        (
            lambda payload: payload.update(labels=[{"name": "car"}, {"name": "car"}]),
            "label names must be unique",
        ),
        (
            lambda payload: payload.update(attributes={"motion": ["moving", "moving"]}),
            "values must be unique",
        ),
        (
            lambda payload: payload.update(scene_tags={"car": ["day", "night"]}),
            "object labels and scene tag labels must be distinct",
        ),
    ],
)
def test_rejects_invalid_road_label_schema(mutation, match: str) -> None:
    payload = schema_mapping()
    mutation(payload)

    with pytest.raises(ValueError, match=match):
        road_label_schema_from_mapping(payload)


def test_ensure_project_creates_rectangle_attributes_and_scene_tag_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = CvatAdapter(CvatConfig("http://cvat.test"), label_schema=schema_mapping())
    projects = CapturingProjects()
    client = SimpleNamespace(projects=projects)
    monkeypatch.setattr(adapter, "_client", lambda: ClientContext(client))

    result = adapter.ensure_project("RoadLabelOps", ["car", "pedestrian"])

    assert result.ok, result.error
    assert result.data == {"project_id": 81, "created": True}
    request = projects.requests[0].to_dict()
    labels = {label["name"]: label for label in request["labels"]}
    assert request["name"] == "RoadLabelOps"
    assert set(labels) == {"car", "pedestrian", "lighting", "weather"}
    assert labels["car"]["type"] == "rectangle"
    assert labels["pedestrian"]["type"] == "rectangle"
    assert labels["car"]["attributes"] == [
        {
            "name": "occlusion",
            "mutable": True,
            "input_type": "select",
            "values": ["none", "partial", "heavy"],
        },
        {
            "name": "motion",
            "mutable": True,
            "input_type": "select",
            "values": ["moving", "stopped", "unknown"],
        },
    ]
    assert labels["lighting"] == {
        "name": "lighting",
        "type": "tag",
        "attributes": [
            {
                "name": "lighting",
                "mutable": True,
                "input_type": "select",
                "values": ["day", "night"],
            }
        ],
    }


def test_ensure_project_keeps_custom_label_list_backward_compatible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = CvatAdapter(CvatConfig("http://cvat.test"), label_schema=schema_mapping())
    projects = CapturingProjects()
    monkeypatch.setattr(
        adapter,
        "_client",
        lambda: ClientContext(SimpleNamespace(projects=projects)),
    )

    result = adapter.ensure_project("Legacy", ["custom_object"])

    assert result.ok, result.error
    assert projects.requests[0].to_dict()["labels"] == [
        {"name": "custom_object", "type": "rectangle", "attributes": []}
    ]


def test_ensure_project_reuses_only_an_exact_schema_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = CvatAdapter(CvatConfig("http://cvat.test"), label_schema=schema_mapping())
    projects = ExistingProjects(existing_schema_labels())
    monkeypatch.setattr(
        adapter,
        "_client",
        lambda: ClientContext(SimpleNamespace(projects=projects)),
    )

    result = adapter.ensure_project("RoadLabelOps", ["car", "pedestrian"])

    assert result.ok, result.error
    assert result.data == {"project_id": 82, "created": False}
    assert projects.requests == []


@pytest.mark.parametrize("mismatch", ["object_label", "attribute", "scene_tag"])
def test_ensure_project_fails_closed_when_existing_schema_differs(
    monkeypatch: pytest.MonkeyPatch,
    mismatch: str,
) -> None:
    labels = existing_schema_labels()
    if mismatch == "object_label":
        labels[0].name = "truck"
    elif mismatch == "attribute":
        labels[0].attributes[0].values = ["none", "partial"]
    else:
        labels.pop()
    adapter = CvatAdapter(CvatConfig("http://cvat.test"), label_schema=schema_mapping())
    projects = ExistingProjects(labels)
    monkeypatch.setattr(
        adapter,
        "_client",
        lambda: ClientContext(SimpleNamespace(projects=projects)),
    )

    result = adapter.ensure_project("RoadLabelOps", ["car", "pedestrian"])

    assert not result.ok
    assert result.error and result.error["code"] == "CVAT_PROJECT_SCHEMA_MISMATCH"
    assert projects.requests == []


class PredictionTask:
    def __init__(self, sources: list[str]) -> None:
        self.annotations = SimpleNamespace(
            tags=[],
            shapes=[SimpleNamespace(source=source) for source in sources],
            tracks=[],
        )
        self.writes: list[Any] = []

    def get_annotations(self) -> Any:
        return self.annotations

    def get_labels(self) -> list[Any]:
        return [SimpleNamespace(id=12, name="car")]

    def set_annotations(self, value: Any) -> None:
        self.writes.append(value)


def prediction_adapter(
    monkeypatch: pytest.MonkeyPatch, sources: list[str]
) -> tuple[CvatAdapter, PredictionTask]:
    task = PredictionTask(sources)
    adapter = CvatAdapter(CvatConfig("http://cvat.test"), label_schema=schema_mapping())
    client = SimpleNamespace(tasks=SimpleNamespace(retrieve=lambda _task_id: task))
    monkeypatch.setattr(adapter, "_client", lambda: ClientContext(client))
    return adapter, task


def test_import_predictions_never_replaces_manual_annotations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, task = prediction_adapter(monkeypatch, ["auto", "manual"])

    result = adapter.import_predictions(
        44,
        [{"label": "car", "frame": 0, "bbox": [1, 2, 10, 20]}],
        allow_replace_auto=True,
    )

    assert not result.ok
    assert result.error and result.error["code"] == "HUMAN_ANNOTATIONS_EXIST"
    assert task.writes == []


def test_import_predictions_requires_approval_before_replacing_auto_annotations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, task = prediction_adapter(monkeypatch, ["auto"])
    prediction = [{"label": "car", "frame": 0, "bbox": [1, 2, 10, 20]}]

    denied = adapter.import_predictions(44, prediction)

    assert not denied.ok
    assert denied.error and denied.error["code"] == "AUTO_ANNOTATIONS_EXIST"
    assert task.writes == []

    approved = adapter.import_predictions(44, prediction, allow_replace_auto=True)

    assert approved.ok, approved.error
    assert approved.data["replaced_auto"] is True
    assert len(task.writes) == 1
    assert task.writes[0].to_dict()["shapes"][0]["source"] == "auto"


def test_import_predictions_rejects_missing_label_without_partial_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, task = prediction_adapter(monkeypatch, [])

    result = adapter.import_predictions(
        44,
        [
            {"label": "car", "frame": 0, "bbox": [1, 2, 10, 20]},
            {"label": "bus", "frame": 1, "bbox": [2, 3, 11, 21]},
        ],
    )

    assert not result.ok
    assert result.error and result.error["code"] == "CVAT_LABEL_SCHEMA_MISMATCH"
    assert "bus" in result.error["message"]
    assert task.writes == []


def attribute(spec_id: int, value: str) -> Any:
    return SimpleNamespace(spec_id=spec_id, value=value)


def rectangle(
    identifier: int,
    *,
    frame: int,
    label_id: int | None = None,
    attributes: list[Any] | None = None,
    outside: bool = False,
) -> Any:
    return SimpleNamespace(
        id=identifier,
        frame=frame,
        label_id=label_id,
        type="rectangle",
        points=[1.125, 2, 20, 30.999],
        source="manual",
        score=None,
        outside=outside,
        attributes=attributes or [],
    )


def test_get_review_result_round_trips_rectangle_attributes_and_scene_tags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = [
        SimpleNamespace(
            id=1,
            name="car",
            attributes=[
                SimpleNamespace(id=101, name="occlusion"),
                SimpleNamespace(id=102, name="motion"),
                SimpleNamespace(id=103, name="direction"),
            ],
        ),
        SimpleNamespace(
            id=9,
            name="lighting",
            attributes=[SimpleNamespace(id=901, name="lighting")],
        ),
    ]
    annotations = SimpleNamespace(
        shapes=[
            rectangle(
                500,
                frame=2,
                label_id=1,
                attributes=[
                    attribute(101, "partial"),
                    attribute(102, "moving"),
                    attribute(103, ""),
                ],
            )
        ],
        tracks=[
            SimpleNamespace(
                label_id=1,
                attributes=[attribute(102, "moving"), attribute(103, "same")],
                shapes=[
                    rectangle(
                        501,
                        frame=3,
                        attributes=[attribute(102, "stopped")],
                    ),
                    rectangle(502, frame=4, outside=True),
                ],
            )
        ],
        tags=[
            SimpleNamespace(
                id=700,
                frame=0,
                label_id=9,
                source="manual",
                attributes=[attribute(901, "night")],
            )
        ],
    )
    task = SimpleNamespace(
        get_labels=lambda: labels,
        get_jobs=lambda: [
            SimpleNamespace(id=88, status=None, stage="acceptance", state="completed")
        ],
        get_annotations=lambda: annotations,
    )
    adapter = CvatAdapter(CvatConfig("http://cvat.test"), label_schema=schema_mapping())
    client = SimpleNamespace(tasks=SimpleNamespace(retrieve=lambda _task_id: task))
    monkeypatch.setattr(adapter, "_client", lambda: ClientContext(client))

    result = adapter.get_review_result(44)

    assert result.ok, result.error
    assert result.data["completed"] is True
    assert result.data["annotations"] == [
        {
            "annotation_id": 500,
            "frame": 2,
            "label": "car",
            "bbox": [1.12, 2.0, 20.0, 31.0],
            "source": "manual",
            "confidence": 1.0,
            "attributes": [
                {"spec_id": 101, "name": "occlusion", "value": "partial"},
                {"spec_id": 102, "name": "motion", "value": "moving"},
            ],
        },
        {
            "annotation_id": 501,
            "frame": 3,
            "label": "car",
            "bbox": [1.12, 2.0, 20.0, 31.0],
            "source": "manual",
            "confidence": 1.0,
            "attributes": [
                {"spec_id": 102, "name": "motion", "value": "stopped"},
                {"spec_id": 103, "name": "direction", "value": "same"},
            ],
        },
    ]
    assert result.data["scene_tags"] == [
        {
            "annotation_id": 700,
            "frame": 0,
            "label": "lighting",
            "source": "manual",
            "attributes": [{"spec_id": 901, "name": "lighting", "value": "night"}],
        }
    ]


def test_get_review_result_does_not_complete_an_annotation_stage_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = SimpleNamespace(
        get_labels=list,
        get_jobs=lambda: [
            SimpleNamespace(
                id=88,
                status="completed",
                stage="annotation",
                state="completed",
            )
        ],
        get_annotations=lambda: SimpleNamespace(tags=[], shapes=[], tracks=[]),
    )
    adapter = CvatAdapter(CvatConfig("http://cvat.test"), label_schema=schema_mapping())
    client = SimpleNamespace(tasks=SimpleNamespace(retrieve=lambda _task_id: task))
    monkeypatch.setattr(adapter, "_client", lambda: ClientContext(client))

    result = adapter.get_review_result(44)

    assert result.ok, result.error
    assert result.data["completed"] is False


def test_fake_adapter_accepts_replace_approval_keyword() -> None:
    result = FakeCvatAdapter().import_predictions(701, [], allow_replace_auto=True)

    assert result.ok
    assert result.data["replaced_auto"] is False


def test_adapter_accepts_schema_path(tmp_path: Path) -> None:
    path = tmp_path / "labels.yaml"
    path.write_text(
        "schema_version: 1.2.3\nlabels:\n  - name: cone\nattributes: {}\nscene_tags: {}\n",
        encoding="utf-8",
    )

    adapter = CvatAdapter(CvatConfig("http://cvat.test", label_schema_path=path))

    assert isinstance(adapter.label_schema, RoadLabelSchema)
    assert adapter.label_schema.object_labels == ("cone",)
