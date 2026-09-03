from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, NotRequired, TypeAlias, TypedDict
from uuid import uuid4


# 发送给 OpenAI 兼容接口的原始上下文消息。
APIMessage: TypeAlias = dict[str, object]
MessageList: TypeAlias = list[APIMessage]


class AssistantMessage(TypedDict):
    """一次完整模型响应的信封。"""

    type: Literal["assistant"]
    uuid: str
    timestamp: str
    message: APIMessage
    response_id: str
    model: str
    usage: dict[str, object]
    finish_reason: str | None
    request_id: NotRequired[str]


class StreamEvent(TypedDict):
    """流式响应中的一个增量事件；当前非流式客户端暂时不会产生它。"""

    type: Literal["stream_event"]
    event: dict[str, object]
    ttft_ms: NotRequired[float]


class APIErrorDetail(TypedDict):
    message: str
    kind: str
    status_code: int | None
    detail: NotRequired[str]


class SystemAPIErrorMessage(TypedDict):
    """API 请求失败事件，不会被加入下一轮模型上下文。"""

    type: Literal["system"]
    subtype: Literal["api_error"]
    level: Literal["error"]
    uuid: str
    timestamp: str
    error: APIErrorDetail
    retry_in_ms: int
    retry_attempt: int
    max_retries: int
    cause: NotRequired[str]


Message: TypeAlias = StreamEvent | AssistantMessage | SystemAPIErrorMessage
EventList: TypeAlias = list[Message]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def system_message(content: str) -> APIMessage:
    return {"role": "system", "content": content}


def user_message(content: str) -> APIMessage:
    return {"role": "user", "content": content}


def assistant_api_message(event: AssistantMessage) -> APIMessage:
    """从 AssistantMessage 信封中取出需要写回 API 上下文的消息。"""
    message = event["message"]
    collected: APIMessage = {
        "role": "assistant",
        "content": message.get("content") or "",
    }
    tool_calls = message.get("tool_calls")
    if tool_calls:
        collected["tool_calls"] = tool_calls
    return collected


def create_assistant_message(
    message: APIMessage,
    *,
    response_id: str = "",
    model: str = "",
    usage: dict[str, object] | None = None,
    finish_reason: str | None = None,
    request_id: str | None = None,
) -> AssistantMessage:
    event: AssistantMessage = {
        "type": "assistant",
        "uuid": str(uuid4()),
        "timestamp": _now(),
        "message": message,
        "response_id": response_id,
        "model": model,
        "usage": usage or {},
        "finish_reason": finish_reason,
    }
    if request_id:
        event["request_id"] = request_id
    return event


def create_stream_event(
    event: dict[str, object],
    *,
    ttft_ms: float | None = None,
) -> StreamEvent:
    stream_event: StreamEvent = {"type": "stream_event", "event": event}
    if ttft_ms is not None:
        stream_event["ttft_ms"] = ttft_ms
    return stream_event


def create_system_api_error_message(
    message: str,
    *,
    kind: str = "unknown",
    status_code: int | None = None,
    detail: str | None = None,
    cause: BaseException | None = None,
    retry_in_ms: int = 0,
    retry_attempt: int = 1,
    max_retries: int = 1,
) -> SystemAPIErrorMessage:
    error: APIErrorDetail = {
        "message": message,
        "kind": kind,
        "status_code": status_code,
    }
    if detail:
        error["detail"] = detail

    event: SystemAPIErrorMessage = {
        "type": "system",
        "subtype": "api_error",
        "level": "error",
        "uuid": str(uuid4()),
        "timestamp": _now(),
        "error": error,
        "retry_in_ms": retry_in_ms,
        "retry_attempt": retry_attempt,
        "max_retries": max_retries,
    }
    if cause:
        event["cause"] = repr(cause)
    return event
