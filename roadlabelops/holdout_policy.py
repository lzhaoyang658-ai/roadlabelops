"""Configuration helpers for keeping a deployment's final holdout isolated."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from pathlib import PurePath

FINAL_HOLDOUT_TASK_IDS_ENV = "ROADLABELOPS_FINAL_HOLDOUT_TASK_IDS"
FINAL_HOLDOUT_JOB_IDS_ENV = "ROADLABELOPS_FINAL_HOLDOUT_JOB_IDS"
NO_FINAL_HOLDOUT_STATEMENT = "No configured final holdout input was read."
FINAL_HOLDOUT_REJECTED_SCOPES = ("data/holdout", "configured-final-holdout")


class FinalHoldoutConfigError(ValueError):
    """Raised when final-holdout identity configuration is unsafe or ambiguous."""


@dataclass(frozen=True)
class FinalHoldoutIdentity:
    task_id: int
    job_id: int


def _positive_identifier(value: object, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise FinalHoldoutConfigError(f"{location} must be a positive integer")
    return value


def parse_final_holdout_ids(raw: str | None, *, variable: str) -> frozenset[int]:
    """Parse a comma-separated set of positive identifiers without hidden defaults."""

    if raw is None or not raw.strip():
        return frozenset()
    parts = raw.split(",")
    if any(not part.strip() or not re.fullmatch(r"[1-9]\d*", part.strip()) for part in parts):
        raise FinalHoldoutConfigError(
            f"{variable} must be a comma-separated list of positive integers"
        )
    return frozenset(int(part.strip()) for part in parts)


def configured_final_holdout_ids(
    environ: Mapping[str, str] | None = None,
) -> tuple[frozenset[int], frozenset[int]]:
    environment = os.environ if environ is None else environ
    return (
        parse_final_holdout_ids(
            environment.get(FINAL_HOLDOUT_TASK_IDS_ENV),
            variable=FINAL_HOLDOUT_TASK_IDS_ENV,
        ),
        parse_final_holdout_ids(
            environment.get(FINAL_HOLDOUT_JOB_IDS_ENV),
            variable=FINAL_HOLDOUT_JOB_IDS_ENV,
        ),
    )


def resolve_final_holdout_identity(
    *,
    task_id: int | None = None,
    job_id: int | None = None,
    environ: Mapping[str, str] | None = None,
) -> FinalHoldoutIdentity:
    """Resolve one bound holdout identity from explicit arguments or environment."""

    task_ids, job_ids = configured_final_holdout_ids(environ)
    if task_id is None:
        if len(task_ids) != 1:
            raise FinalHoldoutConfigError(
                f"set exactly one id in {FINAL_HOLDOUT_TASK_IDS_ENV} or pass task_id"
            )
        task_id = next(iter(task_ids))
    else:
        task_id = _positive_identifier(task_id, "final holdout task_id")
        if task_ids and task_id not in task_ids:
            raise FinalHoldoutConfigError(
                f"final holdout task_id is not declared by {FINAL_HOLDOUT_TASK_IDS_ENV}"
            )
    if job_id is None:
        if len(job_ids) != 1:
            raise FinalHoldoutConfigError(
                f"set exactly one id in {FINAL_HOLDOUT_JOB_IDS_ENV} or pass job_id"
            )
        job_id = next(iter(job_ids))
    else:
        job_id = _positive_identifier(job_id, "final holdout job_id")
        if job_ids and job_id not in job_ids:
            raise FinalHoldoutConfigError(
                f"final holdout job_id is not declared by {FINAL_HOLDOUT_JOB_IDS_ENV}"
            )
    return FinalHoldoutIdentity(task_id=task_id, job_id=job_id)


def is_configured_final_holdout_task(value: object) -> bool:
    task_ids, _ = configured_final_holdout_ids()
    return not isinstance(value, bool) and isinstance(value, int) and value in task_ids


def is_configured_final_holdout_job(value: object) -> bool:
    _, job_ids = configured_final_holdout_ids()
    return not isinstance(value, bool) and isinstance(value, int) and value in job_ids


def final_holdout_scope_reason(value: str | PurePath) -> str | None:
    """Identify path/name references that cross the configured holdout firewall."""

    raw = str(value)
    normalized = [
        re.sub(r"[^a-z0-9]", "", part.casefold())
        for part in re.split(r"[\\/]", raw)
        if part
    ]
    if any(part == "dataholdout" for part in normalized) or any(
        left == "data" and right == "holdout"
        for left, right in pairwise(normalized)
    ):
        return "data/holdout"
    if any("finalholdout" in part for part in normalized) or any(
        left == "final" and right == "holdout"
        for left, right in pairwise(normalized)
    ):
        return "configured final holdout"
    task_ids, job_ids = configured_final_holdout_ids()

    def contains_identifier(kind: str, identifiers: frozenset[int]) -> bool:
        patterns = [
            re.compile(rf"{kind}(?:id)?0*{identifier}(?!\d)")
            for identifier in identifiers
        ]
        if any(pattern.search(part) for pattern in patterns for part in normalized):
            return True
        prefixes = {kind, f"{kind}s", f"{kind}id"}
        for index, part in enumerate(normalized):
            if part not in prefixes:
                continue
            following = normalized[index + 1 : index + 3]
            if following and following[0].isdigit() and int(following[0]) in identifiers:
                return True
            if (
                len(following) == 2
                and following[0] == "id"
                and following[1].isdigit()
                and int(following[1]) in identifiers
            ):
                return True
        return False

    if contains_identifier("task", task_ids) or contains_identifier("job", job_ids):
        return "configured final holdout"
    return None
