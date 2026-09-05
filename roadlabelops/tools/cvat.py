from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..models import ToolResult

_PACKAGE_ROAD_LABEL_SCHEMA = (
    Path(__file__).resolve().parents[1] / "config" / "road_labels.yaml"
)
_SOURCE_ROAD_LABEL_SCHEMA = (
    Path(__file__).resolve().parents[2] / "config" / "road_labels.yaml"
)
DEFAULT_ROAD_LABEL_SCHEMA = (
    _PACKAGE_ROAD_LABEL_SCHEMA
    if _PACKAGE_ROAD_LABEL_SCHEMA.is_file()
    else _SOURCE_ROAD_LABEL_SCHEMA
)


@dataclass(frozen=True, slots=True)
class SelectAttribute:
    name: str
    values: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RoadLabelSchema:
    """Validated subset of ``road_labels.yaml`` used by the CVAT adapter."""

    schema_version: str
    object_labels: tuple[str, ...]
    rectangle_attributes: tuple[SelectAttribute, ...] = ()
    scene_tags: tuple[SelectAttribute, ...] = ()


def _non_empty_name(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a non-empty string")
    return value.strip()


def _select_attributes(value: Any, location: str) -> tuple[SelectAttribute, ...]:
    if value is None:
        return ()
    if not isinstance(value, Mapping):
        raise TypeError(f"{location} must be a mapping")
    attributes: list[SelectAttribute] = []
    for raw_name, raw_values in value.items():
        name = _non_empty_name(raw_name, f"{location} name")
        if (
            not isinstance(raw_values, Sequence)
            or isinstance(raw_values, (str, bytes, bytearray))
            or not raw_values
        ):
            raise ValueError(f"{location}.{name} must be a non-empty list")
        values = tuple(_non_empty_name(item, f"{location}.{name} value") for item in raw_values)
        if len(values) != len(set(values)):
            raise ValueError(f"{location}.{name} values must be unique")
        attributes.append(SelectAttribute(name=name, values=values))
    return tuple(attributes)


def road_label_schema_from_mapping(payload: Mapping[str, Any]) -> RoadLabelSchema:
    """Validate and normalize the CVAT-facing fields of a road-label schema."""

    version = _non_empty_name(payload.get("schema_version"), "schema_version")
    raw_labels = payload.get("labels")
    if (
        not isinstance(raw_labels, Sequence)
        or isinstance(raw_labels, (str, bytes, bytearray))
        or not raw_labels
    ):
        raise ValueError("labels must be a non-empty list")
    labels: list[str] = []
    for index, raw_label in enumerate(raw_labels):
        if not isinstance(raw_label, Mapping):
            raise TypeError(f"labels[{index}] must be a mapping")
        labels.append(_non_empty_name(raw_label.get("name"), f"labels[{index}].name"))
    if len(labels) != len(set(labels)):
        raise ValueError("label names must be unique")

    rectangle_attributes = _select_attributes(payload.get("attributes"), "attributes")
    scene_tags = _select_attributes(payload.get("scene_tags"), "scene_tags")
    tag_names = {tag.name for tag in scene_tags}
    collisions = sorted(set(labels) & tag_names)
    if collisions:
        raise ValueError(
            "object labels and scene tag labels must be distinct: " + ", ".join(collisions)
        )
    return RoadLabelSchema(
        schema_version=version,
        object_labels=tuple(labels),
        rectangle_attributes=rectangle_attributes,
        scene_tags=scene_tags,
    )


def load_road_label_schema(path: Path | str = DEFAULT_ROAD_LABEL_SCHEMA) -> RoadLabelSchema:
    """Load the YAML source of truth used to create CVAT label specifications."""

    schema_path = Path(path)
    try:
        payload = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f"Could not load road label schema: {schema_path}") from error
    if not isinstance(payload, Mapping):
        raise TypeError("road label schema must be a mapping")
    return road_label_schema_from_mapping(payload)


@dataclass(slots=True)
class CvatConfig:
    host: str
    username: str | None = None
    password: str | None = None
    access_token: str | None = None
    label_schema_path: Path | str | None = None


class CvatAdapter:
    """Thin high-level SDK adapter. No delete operation is intentionally exposed."""

    def __init__(
        self,
        config: CvatConfig,
        label_schema: RoadLabelSchema | Mapping[str, Any] | Path | str | None = None,
    ) -> None:
        self.config = config
        schema_source = label_schema
        if schema_source is None:
            configured_path = config.label_schema_path
            default_path = Path(configured_path) if configured_path else DEFAULT_ROAD_LABEL_SCHEMA
            schema_source = default_path if default_path.is_file() else None
        self.label_schema = self._coerce_schema(schema_source)

    @staticmethod
    def _coerce_schema(
        value: RoadLabelSchema | Mapping[str, Any] | Path | str | None,
    ) -> RoadLabelSchema | None:
        if value is None or isinstance(value, RoadLabelSchema):
            return value
        if isinstance(value, Mapping):
            return road_label_schema_from_mapping(value)
        return load_road_label_schema(value)

    def _client(self):
        from cvat_sdk import make_client

        client = make_client(host=self.config.host)
        if self.config.access_token:
            client.api_client.configuration.access_token = self.config.access_token
        elif self.config.username and self.config.password:
            client.login((self.config.username, self.config.password))
        else:
            raise RuntimeError("CVAT credentials are not configured")
        return client

    def health(self) -> ToolResult:
        try:
            with self._client() as client:
                about = client.api_client.server_api.retrieve_about()[0]
                return ToolResult.success(
                    {"host": self.config.host, "version": getattr(about, "version", None)}
                )
        except Exception:
            return ToolResult.failure(
                "CVAT_UNAVAILABLE", "CVAT is unavailable or authentication failed", retryable=True
            )

    def ensure_project(
        self,
        name: str,
        labels: list[str] | RoadLabelSchema | Mapping[str, Any] | Path | str | None = None,
    ) -> ToolResult:
        try:
            from cvat_sdk import models

            schema = self._schema_for(labels)
            with self._client() as client:
                matches = [item for item in client.projects.list() if item.name == name]
                if matches:
                    project = client.projects.retrieve(matches[0].id)
                    if not self._project_schema_matches(project.get_labels(), schema):
                        return ToolResult.failure(
                            "CVAT_PROJECT_SCHEMA_MISMATCH",
                            (
                                f"Existing CVAT project {matches[0].id} does not exactly match "
                                "the configured object labels, attributes, and scene tags"
                            ),
                        )
                    return ToolResult.success(
                        {
                            "project_id": project.id,
                            "created": False,
                        }
                    )
                project = client.projects.create(
                    models.ProjectWriteRequest(
                        name=name,
                        labels=self._project_label_requests(models, schema),
                    )
                )
                return ToolResult.success(
                    {
                        "project_id": project.id,
                        "created": True,
                    },
                    side_effects=[f"cvat:project:{project.id}"],
                )
        except Exception:
            return ToolResult.failure(
                "CVAT_PROJECT_FAILED", "CVAT project could not be created", retryable=True
            )

    def _schema_for(
        self,
        labels: list[str] | RoadLabelSchema | Mapping[str, Any] | Path | str | None,
    ) -> RoadLabelSchema:
        if labels is None:
            if self.label_schema is None:
                raise ValueError("A CVAT label schema or label list is required")
            return self.label_schema
        if isinstance(labels, RoadLabelSchema):
            return labels
        if isinstance(labels, (Mapping, Path, str)):
            schema = self._coerce_schema(labels)
            if schema is None:  # pragma: no cover - narrowed by the branch above
                raise ValueError("A CVAT label schema is required")
            return schema

        normalized = tuple(
            _non_empty_name(label, f"labels[{index}]") for index, label in enumerate(labels)
        )
        if not normalized or len(normalized) != len(set(normalized)):
            raise ValueError("labels must be non-empty and unique")
        # Runtime V1 still passes the historical list[str]. Enrich that exact
        # taxonomy from YAML while retaining legacy behavior for custom lists.
        if self.label_schema and normalized == self.label_schema.object_labels:
            return self.label_schema
        return RoadLabelSchema(schema_version="legacy", object_labels=normalized)

    @staticmethod
    def _project_label_requests(models: Any, schema: RoadLabelSchema) -> list[Any]:
        def attribute(spec: SelectAttribute) -> Any:
            return models.AttributeRequest(
                name=spec.name,
                mutable=True,
                input_type="select",
                values=list(spec.values),
            )

        requests = [
            models.PatchedLabelRequest(
                name=label,
                type="rectangle",
                attributes=[attribute(spec) for spec in schema.rectangle_attributes],
            )
            for label in schema.object_labels
        ]
        requests.extend(
            models.PatchedLabelRequest(
                name=tag.name,
                type="tag",
                attributes=[attribute(tag)],
            )
            for tag in schema.scene_tags
        )
        return requests

    @staticmethod
    def _field(value: Any, name: str, default: Any = None) -> Any:
        if isinstance(value, Mapping):
            return value.get(name, default)
        return getattr(value, name, default)

    @classmethod
    def _attribute_signature(cls, value: Any) -> tuple[str, bool, str, tuple[str, ...]] | None:
        name = cls._field(value, "name")
        mutable = cls._field(value, "mutable")
        input_type = cls._enum_value(cls._field(value, "input_type"))
        raw_values = cls._field(value, "values")
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(mutable, bool)
            or not isinstance(input_type, str)
            or not isinstance(raw_values, Sequence)
            or isinstance(raw_values, (str, bytes, bytearray))
            or not all(isinstance(item, str) for item in raw_values)
        ):
            return None
        return name, mutable, input_type, tuple(raw_values)

    @classmethod
    def _project_schema_signature(
        cls, labels: Sequence[Any]
    ) -> tuple[tuple[str, str, tuple[tuple[str, bool, str, tuple[str, ...]], ...]], ...] | None:
        normalized: list[tuple[str, str, tuple[tuple[str, bool, str, tuple[str, ...]], ...]]] = []
        for label in labels:
            name = cls._field(label, "name")
            label_type = cls._enum_value(cls._field(label, "type"))
            raw_attributes = cls._field(label, "attributes", ()) or ()
            if (
                not isinstance(name, str)
                or not name
                or not isinstance(label_type, str)
                or not isinstance(raw_attributes, Sequence)
                or isinstance(raw_attributes, (str, bytes, bytearray))
            ):
                return None
            attributes: list[tuple[str, bool, str, tuple[str, ...]]] = []
            for attribute in raw_attributes:
                signature = cls._attribute_signature(attribute)
                if signature is None:
                    return None
                attributes.append(signature)
            normalized.append((name, label_type, tuple(sorted(attributes))))
        if len({item[0] for item in normalized}) != len(normalized):
            return None
        return tuple(sorted(normalized))

    @classmethod
    def _project_schema_matches(
        cls, actual_labels: Sequence[Any], expected: RoadLabelSchema
    ) -> bool:
        rectangle_attributes = tuple(
            sorted(
                (attribute.name, True, "select", attribute.values)
                for attribute in expected.rectangle_attributes
            )
        )
        expected_signature = tuple(
            sorted(
                [(name, "rectangle", rectangle_attributes) for name in expected.object_labels]
                + [
                    (
                        tag.name,
                        "tag",
                        ((tag.name, True, "select", tag.values),),
                    )
                    for tag in expected.scene_tags
                ]
            )
        )
        return cls._project_schema_signature(actual_labels) == expected_signature

    def create_task(self, name: str, project_id: int, video_path: Path | str) -> ToolResult:
        try:
            from cvat_sdk import models
            from cvat_sdk.core.proxies.tasks import ResourceType

            with self._client() as client:
                matches = [
                    task
                    for task in client.tasks.list()
                    if task.name == name and task.project_id == project_id
                ]
                if matches:
                    task = client.tasks.retrieve(matches[0].id)
                    return ToolResult.success(
                        {
                            "task_id": task.id,
                            "job_ids": [job.id for job in task.get_jobs()],
                            "created": False,
                        }
                    )
                task = client.tasks.create_from_data(
                    spec=models.TaskWriteRequest(name=name, project_id=project_id),
                    resource_type=ResourceType.LOCAL,
                    resources=[str(Path(video_path).resolve())],
                    data_params={"image_quality": 70},
                )
                return ToolResult.success(
                    {
                        "task_id": task.id,
                        "job_ids": [job.id for job in task.get_jobs()],
                        "created": True,
                    },
                    side_effects=[f"cvat:task:{task.id}"],
                )
        except Exception:
            return ToolResult.failure(
                "CVAT_TASK_FAILED", "CVAT task could not be created", retryable=True
            )

    def import_predictions(
        self,
        task_id: int,
        predictions: list[dict[str, Any]],
        *,
        allow_replace_auto: bool = False,
    ) -> ToolResult:
        try:
            from cvat_sdk import models

            with self._client() as client:
                task = client.tasks.retrieve(task_id)
                label_ids = {label.name: label.id for label in task.get_labels()}
                missing_labels = sorted(
                    {
                        str(prediction.get("label", ""))
                        for prediction in predictions
                        if str(prediction.get("label", "")) not in label_ids
                    }
                )
                if missing_labels:
                    return ToolResult.failure(
                        "CVAT_LABEL_SCHEMA_MISMATCH",
                        "CVAT task is missing prediction label(s): " + ", ".join(missing_labels),
                    )
                existing = task.get_annotations()
                existing_objects = [*existing.tags, *existing.shapes, *existing.tracks]
                if existing_objects and not all(
                    self._enum_value(getattr(item, "source", "manual")) == "auto"
                    for item in existing_objects
                ):
                    return ToolResult.failure(
                        "HUMAN_ANNOTATIONS_EXIST",
                        "CVAT already contains manual annotations; automatic overwrite is denied",
                    )
                if existing_objects and not allow_replace_auto:
                    return ToolResult.failure(
                        "AUTO_ANNOTATIONS_EXIST",
                        "CVAT already contains automatic annotations; explicit approval is required",
                    )

                shapes = []
                for prediction in predictions:
                    label_id = label_ids[str(prediction["label"])]
                    shapes.append(
                        models.LabeledShapeRequest(
                            type="rectangle",
                            label_id=label_id,
                            frame=int(prediction["frame"]),
                            points=[float(value) for value in prediction["bbox"]],
                            source="auto",
                            score=float(prediction.get("confidence", 1.0)),
                        )
                    )
                task.set_annotations(models.LabeledDataRequest(shapes=shapes))
                return ToolResult.success(
                    {
                        "task_id": task_id,
                        "annotation_count": len(shapes),
                        "imported": bool(shapes),
                        "replaced_auto": bool(existing_objects),
                    }
                )
        except Exception:
            return ToolResult.failure(
                "CVAT_ANNOTATION_IMPORT_FAILED",
                "Predictions could not be written to CVAT",
                retryable=True,
            )

    def get_review_result(self, task_id: int) -> ToolResult:
        try:
            with self._client() as client:
                task = client.tasks.retrieve(task_id)
                raw_labels = list(task.get_labels())
                labels = {label.id: label.name for label in raw_labels}
                attribute_names = {
                    label.id: {
                        attribute.id: attribute.name
                        for attribute in (getattr(label, "attributes", None) or [])
                    }
                    for label in raw_labels
                }
                jobs = task.get_jobs()
                job_states = [
                    {
                        "job_id": job.id,
                        "status": self._enum_value(getattr(job, "status", None)),
                        "stage": self._enum_value(getattr(job, "stage", None)),
                        "state": self._enum_value(getattr(job, "state", None)),
                    }
                    for job in jobs
                ]
                completed = bool(job_states) and all(
                    item["stage"] == "acceptance"
                    and (item["status"] == "completed" or item["state"] == "completed")
                    for item in job_states
                )
                annotations = task.get_annotations()
                final: list[dict[str, Any]] = []
                for shape in annotations.shapes:
                    if self._enum_value(shape.type) != "rectangle":
                        continue
                    final.append(
                        self._shape_to_dict(
                            shape,
                            labels.get(shape.label_id, "unknown"),
                            attribute_names.get(shape.label_id, {}),
                        )
                    )
                for track in annotations.tracks:
                    label = labels.get(track.label_id, "unknown")
                    for shape in track.shapes:
                        if (
                            getattr(shape, "outside", False)
                            or self._enum_value(shape.type) != "rectangle"
                        ):
                            continue
                        final.append(
                            self._shape_to_dict(
                                shape,
                                label,
                                attribute_names.get(track.label_id, {}),
                                inherited_attributes=getattr(track, "attributes", None),
                            )
                        )
                scene_tags = [
                    self._tag_to_dict(
                        tag,
                        labels.get(tag.label_id, "unknown"),
                        attribute_names.get(tag.label_id, {}),
                    )
                    for tag in annotations.tags
                ]
                return ToolResult.success(
                    {
                        "task_id": task_id,
                        "completed": completed,
                        "jobs": job_states,
                        "annotations": final,
                        "scene_tags": scene_tags,
                    }
                )
        except Exception:
            return ToolResult.failure(
                "CVAT_REVIEW_SYNC_FAILED",
                "CVAT review status or annotations could not be read",
                retryable=True,
            )

    @staticmethod
    def _enum_value(value: Any) -> str | None:
        if value is None:
            return None
        return str(getattr(value, "value", value))

    @classmethod
    def _shape_to_dict(
        cls,
        shape: Any,
        label: str,
        attribute_names: Mapping[int, str] | None = None,
        *,
        inherited_attributes: Sequence[Any] | None = None,
    ) -> dict[str, Any]:
        score = getattr(shape, "score", None)
        return {
            "annotation_id": getattr(shape, "id", None),
            "frame": int(shape.frame),
            "label": label,
            "bbox": [round(float(value), 2) for value in shape.points],
            "source": cls._enum_value(getattr(shape, "source", "manual")),
            "confidence": float(score) if score is not None else 1.0,
            "attributes": cls._attribute_values(
                inherited_attributes,
                getattr(shape, "attributes", None),
                names=attribute_names or {},
            ),
        }

    @classmethod
    def _tag_to_dict(
        cls,
        tag: Any,
        label: str,
        attribute_names: Mapping[int, str] | None = None,
    ) -> dict[str, Any]:
        return {
            "annotation_id": getattr(tag, "id", None),
            "frame": int(tag.frame),
            "label": label,
            "source": cls._enum_value(getattr(tag, "source", "manual")),
            "attributes": cls._attribute_values(
                getattr(tag, "attributes", None),
                names=attribute_names or {},
            ),
        }

    @staticmethod
    def _attribute_values(
        *collections: Sequence[Any] | None,
        names: Mapping[int, str],
    ) -> list[dict[str, Any]]:
        # Track-level immutable attributes are inherited by each rectangle;
        # frame-level values with the same spec id take precedence.
        values: dict[int, dict[str, Any]] = {}
        order: list[int] = []
        for collection in collections:
            for attribute in collection or []:
                raw_spec_id = (
                    attribute.get("spec_id")
                    if isinstance(attribute, Mapping)
                    else getattr(attribute, "spec_id", None)
                )
                try:
                    spec_id = int(raw_spec_id)
                except (TypeError, ValueError):
                    continue
                raw_value = (
                    attribute.get("value")
                    if isinstance(attribute, Mapping)
                    else getattr(attribute, "value", None)
                )
                # CVAT materializes every select attribute on imported shapes and
                # represents an untouched selection as an empty string.  Empty
                # selections are absence, not a taxonomy value, so do not leak
                # them into the reviewed annotation contract or Release payload.
                if raw_value is None or (isinstance(raw_value, str) and not raw_value.strip()):
                    continue
                if spec_id not in values:
                    order.append(spec_id)
                values[spec_id] = {
                    "spec_id": spec_id,
                    "name": names.get(spec_id),
                    "value": raw_value,
                }
        return [values[spec_id] for spec_id in order]


class FakeCvatAdapter:
    def __init__(self, start_id: int = 700) -> None:
        self._next = start_id

    def health(self) -> ToolResult:
        return ToolResult.success({"host": "fake://cvat", "version": "test"})

    def ensure_project(
        self,
        name: str,
        labels: list[str] | RoadLabelSchema | Mapping[str, Any] | Path | str | None = None,
    ) -> ToolResult:
        if isinstance(labels, RoadLabelSchema):
            payload: Any = list(labels.object_labels)
        else:
            payload = labels
        return ToolResult.success({"project_id": 70, "created": True, "labels": payload})

    def create_task(self, name: str, project_id: int, video_path: Path | str) -> ToolResult:
        self._next += 1
        return ToolResult.success(
            {"task_id": self._next, "job_ids": [self._next + 1000], "created": True},
            side_effects=[f"cvat:task:{self._next}"],
        )

    def import_predictions(
        self,
        task_id: int,
        predictions: list[dict[str, Any]],
        *,
        allow_replace_auto: bool = False,
    ) -> ToolResult:
        return ToolResult.success(
            {
                "task_id": task_id,
                "annotation_count": len(predictions),
                "imported": bool(predictions),
                "replaced_auto": False,
            }
        )

    def get_review_result(self, task_id: int) -> ToolResult:
        return ToolResult.success(
            {
                "task_id": task_id,
                "completed": True,
                "jobs": [{"job_id": task_id + 1000, "status": "completed"}],
                "annotations": [],
                "scene_tags": [],
            }
        )
