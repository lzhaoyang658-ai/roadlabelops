from __future__ import annotations

import fcntl
import json
import os
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .models import Session


class StorageError(RuntimeError):
    pass


class LocalStore:
    def __init__(self, root: Path | str = "data", runtime_root: Path | str = "runtime") -> None:
        self.root = Path(root).resolve()
        self.runtime_root = Path(runtime_root).resolve()
        self.sessions_dir = self.root / "sessions"
        self.scenes_dir = self.root / "scenes"
        self.raw_dir = self.root / "raw"
        self.releases_dir = self.root / "releases"
        self._lock = threading.RLock()
        for path in (
            self.sessions_dir,
            self.scenes_dir,
            self.raw_dir,
            self.releases_dir,
            self.runtime_root,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def safe_data_path(self, relative: str | Path) -> Path:
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise StorageError("path escapes the managed data directory")
        return path

    @contextmanager
    def operation_lock(self, name: str):
        """Serialize a state transition across threads and local processes."""

        if not name or not all(character.isalnum() or character in "._-" for character in name):
            raise StorageError("invalid operation lock name")
        locks_dir = self.runtime_root / "locks"
        locks_dir.mkdir(parents=True, exist_ok=True)
        lock_path = locks_dir / f"{name}.lock"
        with self._lock, lock_path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def save_session(self, session: Session) -> Path:
        path = self.sessions_dir / f"{session.session_id}.json"
        self.write_json_atomic(path, session.to_dict())
        return path

    def get_session(self, session_id: str) -> Session:
        if not session_id.replace("_", "").replace("-", "").isalnum():
            raise StorageError("invalid session id")
        path = self.sessions_dir / f"{session_id}.json"
        if not path.exists():
            raise FileNotFoundError(session_id)
        return Session.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list_sessions(self) -> list[Session]:
        sessions: list[Session] = []
        for path in self.sessions_dir.glob("*.json"):
            if path.name.endswith((".quality.json", ".release.json")):
                continue
            try:
                sessions.append(Session.from_dict(json.loads(path.read_text(encoding="utf-8"))))
            except (KeyError, ValueError, TypeError, json.JSONDecodeError):
                continue
        return sorted(sessions, key=lambda item: item.updated_at, reverse=True)

    def append_journal(self, event: dict[str, Any]) -> None:
        path = self.runtime_root / "journal.jsonl"
        line = json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n"
        with self._lock, path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    def read_journal(self, limit: int | None = 50) -> list[dict[str, Any]]:
        path = self.runtime_root / "journal.jsonl"
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8").splitlines()
        if limit is not None:
            lines = lines[-limit:]
        events: list[dict[str, Any]] = []
        for line in lines:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return events

    def write_json_atomic(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
