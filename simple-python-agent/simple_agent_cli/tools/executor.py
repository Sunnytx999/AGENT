from __future__ import annotations

import json

from ..messages import APIMessage
from .registry import TOOL_HANDLERS, ToolHandler, ToolResult


def call_local_tool(
    name: str,
    arguments: dict[str, object],
    handlers: dict[str, ToolHandler] | None = None,
) -> ToolResult:
    registry = handlers or TOOL_HANDLERS
    handler = registry.get(name)
    if handler is None:
        return {"ok": False, "error": f"Unknown tool: {name}"}
    try:
        return handler(arguments)
    except (OSError, UnicodeError, ValueError) as exc:
        return {"ok": False, "error": str(exc)}


class ToolExecutor:
    """负责解析模型的 tool_call，并调用注册表中的本地工具。"""

    def __init__(self, handlers: dict[str, ToolHandler] | None = None) -> None:
        self.handlers = handlers or TOOL_HANDLERS

    def execute(self, tool_call: object) -> APIMessage:
        if not isinstance(tool_call, dict):
            return self._tool_message("", "", {"ok": False, "error": "Invalid tool call"})

        tool_call_id = str(tool_call.get("id") or "")
        function = tool_call.get("function") or {}
        if not isinstance(function, dict):
            function = {}

        name = str(function.get("name") or "")
        raw_arguments = function.get("arguments") or "{}"
        try:
            arguments = json.loads(str(raw_arguments))
        except json.JSONDecodeError:
            arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}

        result = call_local_tool(name, arguments, self.handlers)
        return self._tool_message(tool_call_id, name, result)

    @staticmethod
    def _tool_message(
        tool_call_id: str,
        name: str,
        result: ToolResult,
    ) -> APIMessage:
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": name,
            "content": json.dumps(result, ensure_ascii=False),
        }
