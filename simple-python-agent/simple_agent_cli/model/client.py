from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Iterable, Iterator, Protocol
from urllib import error, request

from ..messages import (
    MessageList,
    create_assistant_message,
    create_stream_event,
    create_system_api_error_message,
)
from .types import ModelMessage, ToolDefinition


@dataclass(frozen=True)
class LLMConfig:
    api_key: str | None
    base_url: str
    model: str | None
    timeout_seconds: float = 60.0
    max_tool_turns: int = 10

    # @classmethod
    # def from_env(cls) -> LLMConfig:
    #     base_url = (
    #         os.getenv("SIMPLE_AGENT_BASE_URL")
    #         or os.getenv("OPENAI_BASE_URL")
    #         or "https://api.openai.com/v1"
    #     )
    #     return cls(
    #         api_key=os.getenv("SIMPLE_AGENT_API_KEY") or os.getenv("OPENAI_API_KEY"),
    #         base_url=base_url.rstrip("/"),
    #         model=os.getenv("SIMPLE_AGENT_MODEL") or os.getenv("OPENAI_MODEL"),
    #         timeout_seconds=float(os.getenv("SIMPLE_AGENT_TIMEOUT", "60")),
    #         max_tool_turns=int(os.getenv("SIMPLE_AGENT_MAX_TOOL_TURNS", "10")),
    #     )
    @classmethod
    def from_env(cls) -> LLMConfig:
        return cls(
            api_key="sk-24b86a67cd48417092412b5a1da99252",
            base_url="https://api.deepseek.com",
            model="deepseek-v4-pro",
            timeout_seconds=60.0,
            max_tool_turns=10,
        )


class LLMError(Exception):
    pass


class ModelClient(Protocol):
    def call_model(
        self,
        messages: MessageList,
        tools: list[ToolDefinition],
    ) -> Iterable[ModelMessage]:
        """逐个返回 StreamEvent，最后返回 AssistantMessage 或 API 错误。"""


class OpenAIChatClient:
    """OpenAI 兼容的 Chat Completions HTTP 客户端。"""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def call_model(
        self,
        messages: MessageList,
        tools: list[ToolDefinition],
    ) -> Iterator[ModelMessage]:
        payload = {
            "model": self.config.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "stream": True,
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        url = f"{self.config.base_url}/chat/completions"
        req = request.Request(url, data=body, headers=headers, method="POST")

        started_at = time.perf_counter()
        try:
            with request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                request_id = response.headers.get("x-request-id")
                yield from self._read_stream(response, request_id, started_at)
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            yield create_system_api_error_message(
                f"HTTP {exc.code}",
                kind="http_error",
                status_code=exc.code,
                detail=detail,
                cause=exc,
            )
        except error.URLError as exc:
            yield create_system_api_error_message(
                str(exc.reason),
                kind="network_error",
                cause=exc,
            )
        except TimeoutError as exc:
            yield create_system_api_error_message(
                "request timed out",
                kind="timeout",
                cause=exc,
            )

    def _read_stream(
        self,
        response: object,
        request_id: str | None,
        started_at: float,
    ) -> Iterator[ModelMessage]:
        message: dict[str, object] = {"role": "assistant", "content": ""}
        tool_calls: dict[int, dict[str, object]] = {}
        response_id = ""
        model = self.config.model or ""
        usage: dict[str, object] = {}
        finish_reason: str | None = None
        saw_chunk = False
        first_event = True
        fallback_lines: list[str] = []

        for raw_line in response:  # type: ignore[union-attr]
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            if not line.startswith("data:"):
                fallback_lines.append(line)
                continue

            data_text = line[5:].strip()
            if data_text == "[DONE]":
                break
            try:
                chunk = json.loads(data_text)
            except json.JSONDecodeError as exc:
                yield create_system_api_error_message(
                    "invalid stream event",
                    kind="invalid_response",
                    detail=data_text,
                    cause=exc,
                )
                return
            if not isinstance(chunk, dict):
                continue
            if isinstance(chunk.get("error"), dict):
                error_data = chunk["error"]
                yield create_system_api_error_message(
                    str(error_data.get("message") or "stream API error"),
                    kind=str(error_data.get("type") or "api_error"),
                    detail=json.dumps(error_data, ensure_ascii=False),
                )
                return

            saw_chunk = True
            ttft_ms = (
                (time.perf_counter() - started_at) * 1000
                if first_event
                else None
            )
            first_event = False
            yield create_stream_event(chunk, ttft_ms=ttft_ms)

            response_id = str(chunk.get("id") or response_id)
            model = str(chunk.get("model") or model)
            chunk_usage = chunk.get("usage")
            if isinstance(chunk_usage, dict):
                usage = chunk_usage

            choices = chunk.get("choices") or []
            if not isinstance(choices, list) or not choices:
                continue
            choice = choices[0]
            if not isinstance(choice, dict):
                continue
            if choice.get("finish_reason") is not None:
                finish_reason = str(choice["finish_reason"])
            delta = choice.get("delta") or {}
            if isinstance(delta, dict):
                _merge_delta(message, tool_calls, delta)

        if not saw_chunk and fallback_lines:
            yield self._parse_non_stream_response(
                "\n".join(fallback_lines),
                request_id,
            )
            return
        if not saw_chunk:
            yield create_system_api_error_message(
                "stream ended without data",
                kind="invalid_response",
            )
            return

        if tool_calls:
            message["tool_calls"] = [
                tool_calls[index] for index in sorted(tool_calls)
            ]
        yield create_assistant_message(
            message,
            response_id=response_id,
            model=model,
            usage=usage,
            finish_reason=finish_reason,
            request_id=request_id,
        )

    def _parse_non_stream_response(
        self,
        response_body: str,
        request_id: str | None,
    ) -> ModelMessage:
        """兼容忽略 stream=true、仍返回普通 JSON 的服务。"""
        try:
            data = json.loads(response_body)
            choice = data["choices"][0]
            message = choice["message"]
            if not isinstance(message, dict):
                raise TypeError("message is not an object")
            usage = data.get("usage") or {}
            if not isinstance(usage, dict):
                usage = {}
            return create_assistant_message(
                message,
                response_id=str(data.get("id") or ""),
                model=str(data.get("model") or self.config.model or ""),
                usage=usage,
                finish_reason=(
                    str(choice.get("finish_reason"))
                    if choice.get("finish_reason") is not None
                    else None
                ),
                request_id=request_id,
            )
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            return create_system_api_error_message(
                "unexpected response",
                kind="invalid_response",
                detail=response_body,
                cause=exc,
            )


def _merge_delta(
    message: dict[str, object],
    tool_calls: dict[int, dict[str, object]],
    delta: dict[str, object],
) -> None:
    """把一个 OpenAI 流式 delta 合并到最终 assistant message。"""
    role = delta.get("role")
    if isinstance(role, str):
        message["role"] = role

    for key, value in delta.items():
        if key in {"role", "tool_calls"} or value is None:
            continue
        if isinstance(value, str):
            previous = message.get(key)
            message[key] = f"{previous or ''}{value}"
        else:
            message[key] = value

    calls = delta.get("tool_calls") or []
    if not isinstance(calls, list):
        return
    for position, call_delta in enumerate(calls):
        if not isinstance(call_delta, dict):
            continue
        index_value = call_delta.get("index", position)
        index = index_value if isinstance(index_value, int) else position
        current = tool_calls.setdefault(
            index,
            {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
        )
        if call_delta.get("id"):
            current["id"] = call_delta["id"]
        if call_delta.get("type"):
            current["type"] = call_delta["type"]

        function_delta = call_delta.get("function") or {}
        function = current["function"]
        if not isinstance(function_delta, dict) or not isinstance(function, dict):
            continue
        for key in ("name", "arguments"):
            value = function_delta.get(key)
            if isinstance(value, str):
                function[key] = f"{function.get(key) or ''}{value}"
