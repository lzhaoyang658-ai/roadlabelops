from pathlib import Path

from roadlabelops.permissions import Decision, PermissionPolicy


def test_denies_writes_outside_data(tmp_path: Path) -> None:
    policy = PermissionPolicy(tmp_path / "data")
    result = policy.check("write_file", tmp_path / "elsewhere" / "result.json")
    assert result.decision is Decision.DENY


def test_release_overwrite_is_never_allowed(tmp_path: Path) -> None:
    policy = PermissionPolicy(tmp_path / "data")
    assert policy.check("overwrite_release").decision is Decision.DENY


def test_unknown_action_is_denied_by_default(tmp_path: Path) -> None:
    policy = PermissionPolicy(tmp_path / "data")

    result = policy.check("future_unreviewed_action")

    assert result.decision is Decision.DENY
    assert "not covered" in result.reason


def test_known_workflow_action_remains_allowed(tmp_path: Path) -> None:
    policy = PermissionPolicy(tmp_path / "data")

    assert policy.check("preannotate").decision is Decision.ALLOW


def test_new_file_inside_managed_data_is_allowed(tmp_path: Path) -> None:
    policy = PermissionPolicy(tmp_path / "data")

    result = policy.check("write_file", tmp_path / "data" / "scenes" / "new.json")

    assert result.decision is Decision.ALLOW
