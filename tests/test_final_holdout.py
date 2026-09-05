from pathlib import Path

import pytest

from roadlabelops.holdout_policy import (
    FINAL_HOLDOUT_JOB_IDS_ENV,
    FINAL_HOLDOUT_TASK_IDS_ENV,
    FinalHoldoutConfigError,
    configured_final_holdout_ids,
    final_holdout_scope_reason,
    parse_final_holdout_ids,
    resolve_final_holdout_identity,
)


def test_configured_ids_have_no_real_or_implicit_default() -> None:
    assert configured_final_holdout_ids({}) == (frozenset(), frozenset())


@pytest.mark.parametrize("raw", ["0", "-1", "1,,2", "one", "1.5"])
def test_invalid_final_holdout_id_configuration_fails_closed(raw: str) -> None:
    with pytest.raises(FinalHoldoutConfigError, match="positive integers"):
        parse_final_holdout_ids(raw, variable=FINAL_HOLDOUT_TASK_IDS_ENV)


def test_explicit_or_environment_identity_is_validated() -> None:
    environment = {
        FINAL_HOLDOUT_TASK_IDS_ENV: "91001",
        FINAL_HOLDOUT_JOB_IDS_ENV: "92001",
    }
    assert resolve_final_holdout_identity(environ=environment) == (
        resolve_final_holdout_identity(
            task_id=91001,
            job_id=92001,
            environ=environment,
        )
    )
    with pytest.raises(FinalHoldoutConfigError, match="not declared"):
        resolve_final_holdout_identity(
            task_id=91002,
            job_id=92001,
            environ=environment,
        )


def test_firewall_rejects_default_semantics_and_configured_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(FINAL_HOLDOUT_TASK_IDS_ENV, "91001")
    monkeypatch.setenv(FINAL_HOLDOUT_JOB_IDS_ENV, "92001")

    assert final_holdout_scope_reason(Path("data/holdout/manifest.json")) == "data/holdout"
    assert final_holdout_scope_reason("artifacts/final-holdout/reference.json")
    assert final_holdout_scope_reason("artifacts/final/holdout/reference.json")
    assert final_holdout_scope_reason("exports/task-91001/annotations.json")
    assert final_holdout_scope_reason("exports/task/91001/annotations.json")
    assert final_holdout_scope_reason("exports/job_id_92001/snapshot.json")
    assert final_holdout_scope_reason("exports/jobs/id/92001/snapshot.json")
    assert final_holdout_scope_reason("exports/task-91002/training.json") is None
