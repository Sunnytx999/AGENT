from __future__ import annotations

import json
import time
from collections.abc import Generator, Iterator

from .hooks import QueryHooks
from .messages import (
    APIMessage,
    AssistantMessage,
    EventList,
    Message,
    MessageList,
    SystemAPIErrorMessage,
    assistant_api_message,
    create_stream_event,
    user_message,
)
from .model.client import LLMError, ModelClient
from .model.types import ModelMessage, ToolDefinition
from .tools.executor import ToolExecutor


def query(
    prompt: str,
    messages: MessageList,
    model_client: ModelClient,
    tool_executor: ToolExecutor,
    tools: list[ToolDefinition],
    max_tool_turns: int,
    hooks: QueryHooks | None = None,
    events: EventList | None = None,
) -> Generator[Message, None, str]:
    """添加用户消息并运行查询；失败时回滚本次对话产生的所有消息。"""
    snapshot_len = len(messages)
    messages.append(user_message(prompt))
    try:
        return (
            yield from query_loop(
                messages,
                model_client,
                tool_executor,
                tools,
                max_tool_turns,
                hooks,
                events,
            )
        )
    except LLMError:
        del messages[snapshot_len:]
        raise


def query_loop(
    messages: MessageList,
    model_client: ModelClient,
    tool_executor: ToolExecutor,
    tools: list[ToolDefinition],
    max_tool_turns: int,
    hooks: QueryHooks | None = None,
    events: EventList | None = None,
) -> Generator[Message, None, str]:
    """运行模型/工具循环，直到模型返回不含 tool_calls 的最终答案。"""
    active_hooks = hooks or QueryHooks()
    event_log = events if events is not None else []
    repeated_failure: tuple[str, str] | None = None

    # 多出的一轮用于让模型根据最后一次工具结果生成最终答案。
    for turn in range(1, max_tool_turns + 2):
        assistant: AssistantMessage | None = None
        api_error: SystemAPIErrorMessage | None = None

        # 模型事件到达一条就立即向 Agent/CLI 继续 yield，不等待请求结束。
        for event in call_model(model_client, messages, tools, active_hooks):
            event_log.append(event)
            yield event
            if event.get("type") == "assistant":
                assistant = event  # type: ignore[assignment]
            elif event.get("type") == "system" and event.get("subtype") == "api_error":
                api_error = event  # type: ignore[assignment]

        if api_error is not None:
            raise LLMError(_format_api_error(api_error))

        if assistant is None:
            raise LLMError("model returned no final assistant message")

        collected = collect_assistant_message(assistant)
        tool_calls = collected.get("tool_calls") or []

        if not tool_calls:
            messages.append(collected)
            return str(collected.get("content") or "")

        if turn > max_tool_turns:
            raise LLMError(f"tool call limit reached ({max_tool_turns})")

        messages.append(collected)
        tool_results: list[APIMessage] = []
        calls = tool_calls if isinstance(tool_calls, list) else [tool_calls]
        for tool_call in calls:
            active_hooks.before_tool(tool_call)
            start_event = create_stream_event(
                {"type": "tool_call_start", "tool_call": tool_call}
            )
            event_log.append(start_event)
            yield start_event

            started_at = time.perf_counter()
            result = tool_executor.execute(tool_call)
            elapsed_seconds = time.perf_counter() - started_at
            active_hooks.after_tool(result)
            tool_results.append(result)
            messages.append(result)

            result_event = create_stream_event(
                {
                    "type": "tool_result",
                    "result": result,
                    "elapsed_seconds": elapsed_seconds,
                }
            )
            event_log.append(result_event)
            yield result_event

        repeated_failure = check_repeated_failures(tool_results, repeated_failure)

    raise LLMError("tool loop ended unexpectedly")


def call_model(
    model_client: ModelClient,
    messages: MessageList,
    tools: list[ToolDefinition],
    hooks: QueryHooks,
) -> Iterator[ModelMessage]:
    """执行一次模型请求，并触发请求前后的 Hook。"""
    hooks.before_model(messages)
    for event in model_client.call_model(messages, tools):
        hooks.after_model(event)
        yield event


def collect_assistant_message(message: AssistantMessage) -> APIMessage:
    """从 AssistantMessage 信封提取模型可见的 API 上下文消息。"""
    return assistant_api_message(message)


def _format_api_error(event: SystemAPIErrorMessage) -> str:
    error = event["error"]
    message = error["message"]
    detail = error.get("detail")
    return f"{message}: {detail}" if detail else message


def check_repeated_failures(
    tool_results: list[APIMessage],
    previous_failure: tuple[str, str] | None,
) -> tuple[str, str] | None:
    """检测连续相同的工具错误，防止模型陷入无效重试。"""
    repeated_failure = previous_failure
    for tool_result in tool_results:
        try:
            result = json.loads(str(tool_result.get("content") or "{}"))
        except json.JSONDecodeError:
            result = {"ok": False, "error": "Invalid tool result"}

        if result.get("ok") is False:
            failure = (
                str(tool_result.get("name") or ""),
                str(result.get("error") or "unknown error"),
            )
            if failure == repeated_failure:
                raise LLMError(
                    f"repeated tool failure: {failure[0]}: {failure[1]}"
                )
            repeated_failure = failure
        else:
            repeated_failure = None

    return repeated_failure
