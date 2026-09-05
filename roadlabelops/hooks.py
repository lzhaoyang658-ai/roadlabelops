from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

Hook = Callable[[dict[str, Any]], None]


class HookBus:
    def __init__(self, log_path: Path | str = "logs/roadlabelops.jsonl") -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._hooks: dict[str, list[Hook]] = {"before": [], "after": [], "failed": []}

    def register(self, event: str, callback: Hook) -> None:
        self._hooks.setdefault(event, []).append(callback)

    def emit(self, event: str, payload: dict[str, Any]) -> None:
        safe = self._redact(payload)
        for callback in self._hooks.get(event, []):
            try:
                callback(safe)
            except Exception:
                continue
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"event": event, **safe}, ensure_ascii=False) + "\n")

    def trace(self, tool: str, session_id: str | None = None) -> dict[str, Any]:
        return {
            "trace_id": uuid.uuid4().hex,
            "tool": tool,
            "session_id": session_id,
            "started_at": time.time(),
        }

    @staticmethod
    def _redact(value: Any) -> Any:
        secret_words = {"password", "token", "secret", "authorization", "api_key"}
        if isinstance(value, dict):
            return {
                key: "[REDACTED]" if key.lower() in secret_words else HookBus._redact(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [HookBus._redact(item) for item in value]
        return value
