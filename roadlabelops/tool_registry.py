from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from .hooks import HookBus
from .models import ToolResult
from .permissions import Decision, PermissionPolicy

Tool = Callable[..., ToolResult]


class ToolRegistry:
    def __init__(self, policy: PermissionPolicy, hooks: HookBus) -> None:
        self.policy = policy
        self.hooks = hooks
        self._tools: dict[str, Tool] = {}

    def register(self, name: str, tool: Tool) -> None:
        if name in self._tools:
            raise ValueError(f"tool already registered: {name}")
        self._tools[name] = tool

    def execute(
        self,
        name: str,
        *,
        session_id: str | None = None,
        permission_action: str | None = None,
        permission_target: Any | None = None,
        permission_exists: bool = False,
        approved: bool = False,
        event_trace_id: str | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        trace = self.hooks.trace(name, session_id)
        if event_trace_id:
            trace["trace_id"] = event_trace_id
        self.hooks.emit("before", {**trace, "input_summary": list(kwargs)})
        started = time.perf_counter()
        if name not in self._tools:
            result = ToolResult.failure("TOOL_NOT_FOUND", f"Unknown tool: {name}")
        elif permission_action:
            target = permission_target if permission_target is not None else kwargs.get("target")
            if permission_exists:
                permission = self.policy.check(permission_action, target, exists=True)
            else:
                permission = self.policy.check(permission_action, target)
            if permission.decision is Decision.DENY:
                result = ToolResult.failure("PERMISSION_DENIED", permission.reason)
            elif permission.decision is Decision.ASK and not approved:
                result = ToolResult.failure("PERMISSION_REQUIRED", permission.reason)
            else:
                result = self._invoke(name, kwargs)
        else:
            result = self._invoke(name, kwargs)
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        result.metrics.setdefault("duration_ms", duration_ms)
        if not result.ok:
            self.hooks.emit(
                "failed",
                {
                    **trace,
                    "duration_ms": duration_ms,
                    "error": result.error,
                    "retryable": result.retryable,
                },
            )
        self.hooks.emit(
            "after",
            {**trace, "ok": result.ok, "duration_ms": duration_ms, "error": result.error},
        )
        return result

    def _invoke(self, name: str, kwargs: dict[str, Any]) -> ToolResult:
        try:
            return self._tools[name](**kwargs)
        except Exception as exc:
            return ToolResult.failure("TOOL_FAILED", str(exc), retryable=False)
