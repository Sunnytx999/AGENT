from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Callable

from .hooks import QueryHooks
from .messages import EventList, Message, MessageList, system_message
from .model.client import LLMConfig, LLMError, ModelClient, OpenAIChatClient
from .query import query
from .tools.definitions import LOCAL_TOOLS
from .tools.executor import ToolExecutor
from .tools.registry import get_local_time as _get_local_time_tool


@dataclass
class AgentState:
    history: list[str] = field(default_factory=list)
    running: bool = True


def get_local_time() -> dict[str, str | int]:
    """兼容旧导入：返回本机时间数据。"""
    result = _get_local_time_tool({}).get("result") or {}
    return dict(result) if isinstance(result, dict) else {}


class SimpleAgent:
    """Agent 对外接口；查询循环、模型请求和工具执行由独立模块负责。"""

    def __init__(
        self,
        config: LLMConfig | None = None,
        model_client: ModelClient | None = None,
        tool_executor: ToolExecutor | None = None,
        hooks: QueryHooks | None = None,
    ) -> None:
        self.turn_count = 0
        self.config = config or LLMConfig.from_env()
        self.model_client = model_client or OpenAIChatClient(self.config)
        self.tool_executor = tool_executor or ToolExecutor()
        self.hooks = hooks or QueryHooks()
        # events 保存 AssistantMessage / StreamEvent / SystemAPIErrorMessage，
        # messages 只保存下一轮需要发给模型的 OpenAI 格式上下文。
        self.events: EventList = []
        self.messages: MessageList = [
            system_message(
                "You are a helpful CLI coding assistant. Keep answers concise. "
                "When the user asks for the current time or date, use the "
                "get_local_time tool instead of guessing. Use read_file, "
                "list_files, and search_text to inspect local files when needed. "
                "Do not invent file contents."
            )
        ]

    def respond(self, prompt: str) -> str:
        """兼容同步调用：消费完整事件流，只返回最终文本。"""
        return _consume_stream(self.respond_stream(prompt))

    def respond_stream(self, prompt: str) -> Generator[Message, None, str]:
        """端到端透传模型、工具和错误事件，并返回最终文本。"""
        self.turn_count += 1
        if not self.config.model:
            return (
                "LLM is not configured. Set SIMPLE_AGENT_MODEL and "
                "SIMPLE_AGENT_API_KEY, or use SIMPLE_AGENT_BASE_URL for an "
                "OpenAI-compatible local/server model."
            )

        try:
            return (
                yield from query(
                    prompt=prompt,
                    messages=self.messages,
                    model_client=self.model_client,
                    tool_executor=self.tool_executor,
                    tools=LOCAL_TOOLS,
                    max_tool_turns=self.config.max_tool_turns,
                    hooks=self.hooks,
                    events=self.events,
                )
            )
        except LLMError as exc:
            return f"LLM request failed: {exc}"


def _consume_stream(stream: Generator[Message, None, str]) -> str:
    while True:
        try:
            next(stream)
        except StopIteration as stop:
            return stop.value


CommandHandler = Callable[[str, AgentState], str | None]


def command_help(_args: str, _state: AgentState) -> str:
    return "\n".join(
        [
            "Available commands:",
            "  /help       Show commands",
            "  /history    Show submitted prompts",
            "  /clear      Clear prompt history",
            "  /exit       Exit interactive mode",
        ]
    )


def command_history(_args: str, state: AgentState) -> str:
    if not state.history:
        return "History is empty."
    return "\n".join(f"{i + 1}. {item}" for i, item in enumerate(state.history))


def command_clear(_args: str, state: AgentState) -> str:
    state.history.clear()
    return "History cleared."


def command_exit(_args: str, state: AgentState) -> str:
    state.running = False
    return "Bye."


COMMANDS: dict[str, CommandHandler] = {
    "help": command_help,
    "history": command_history,
    "clear": command_clear,
    "exit": command_exit,
    "quit": command_exit,
}


def parse_slash_command(user_input: str) -> tuple[str, str] | None:
    text = user_input.strip()
    if not text.startswith("/"):
        return None

    body = text[1:]
    if not body:
        return "", ""

    name, _, args = body.partition(" ")
    return name, args.strip()


def handle_slash_command(user_input: str, state: AgentState) -> str | None:
    parsed = parse_slash_command(user_input)
    if parsed is None:
        return None

    name, args = parsed
    handler = COMMANDS.get(name)
    if handler is None:
        return f"Unknown command: /{name}. Try /help."

    return handler(args, state)


def submit_input(user_input: str, agent: SimpleAgent, state: AgentState) -> str | None:
    if not user_input.strip():
        return None

    command_result = handle_slash_command(user_input, state)
    if command_result is not None:
        return command_result

    state.history.append(user_input)
    return agent.respond(user_input)


def submit_input_stream(
    user_input: str,
    agent: SimpleAgent,
    state: AgentState,
) -> Generator[Message, None, str | None]:
    """流式版本的 submit_input；本地命令直接作为生成器返回值返回。"""
    if not user_input.strip():
        return None

    command_result = handle_slash_command(user_input, state)
    if command_result is not None:
        return command_result

    state.history.append(user_input)
    return (yield from agent.respond_stream(user_input))
