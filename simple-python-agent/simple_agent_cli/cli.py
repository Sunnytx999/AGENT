from __future__ import annotations

import argparse
import json
import time
from collections.abc import Generator

from .agent import AgentState, SimpleAgent, submit_input_stream
from .messages import APIMessage, Message


class StreamRenderer:
    """把 StreamEvent 和工具执行状态实时显示在终端。"""

    def __init__(self) -> None:
        self.current_streamed_content = ""
        self.saw_model_event = False
        self.process_section_open = False
        self.model_round = 0
        self.tool_started_at: dict[str, float] = {}

    def reset(self) -> None:
        self.current_streamed_content = ""
        self.saw_model_event = False
        self.process_section_open = False
        self.model_round = 0
        self.tool_started_at.clear()

    def after_model(self, event: Message) -> None:
        self.saw_model_event = True
        if event.get("type") == "assistant":
            self.model_round += 1
            message = event.get("message")
            tool_calls = (
                message.get("tool_calls")
                if isinstance(message, dict)
                else None
            )
            # 只有还要调用工具的模型响应才属于中间过程。
            if isinstance(tool_calls, list) and tool_calls:
                self._open_process_section()
                print(
                    f"\n第 {self.model_round} 轮："
                    f"模型准备调用 {len(tool_calls)} 个工具",
                    flush=True,
                )
                if self.current_streamed_content:
                    print(f"模型说明：{self.current_streamed_content}", flush=True)
            self.current_streamed_content = ""
            return
        if event.get("type") != "stream_event":
            return
        raw_event = event.get("event")
        if not isinstance(raw_event, dict):
            return
        choices = raw_event.get("choices") or []
        if not isinstance(choices, list) or not choices:
            return
        choice = choices[0]
        if not isinstance(choice, dict):
            return
        delta = choice.get("delta") or {}
        if not isinstance(delta, dict):
            return
        self._open_process_section()
        print(
            f"模型 delta："
            f"{json.dumps(delta, ensure_ascii=False, separators=(',', ':'))}",
            flush=True,
        )
        content = delta.get("content")
        if isinstance(content, str) and content:
            # 先缓存本轮增量。收到完整 AssistantMessage 后才能准确知道
            # 它是带 tool_calls 的中间过程，还是不带 tool_calls 的最终结果。
            self.current_streamed_content += content

    def handle_event(self, event: Message) -> None:
        """消费 query_loop 逐条透传的模型或工具事件。"""
        if event.get("type") != "stream_event":
            self.after_model(event)
            return
        raw_event = event.get("event")
        if not isinstance(raw_event, dict):
            return
        event_type = raw_event.get("type")
        if event_type == "tool_call_start":
            self.before_tool(raw_event.get("tool_call"))
            return
        if event_type == "tool_result":
            result = raw_event.get("result")
            if isinstance(result, dict):
                elapsed = raw_event.get("elapsed_seconds")
                self.after_tool(
                    result,
                    elapsed if isinstance(elapsed, (int, float)) else None,
                )
            return
        self.after_model(event)

    def before_tool(self, tool_call: object) -> None:
        name = _tool_call_name(tool_call)
        call_id = _tool_call_id(tool_call)
        timer_key = call_id or name or "unknown"
        self.tool_started_at[timer_key] = time.perf_counter()
        self._open_process_section()
        id_text = f"（{call_id}）" if call_id else ""
        print(f"\n调用工具：{name or 'unknown'}{id_text}", flush=True)
        arguments = _format_tool_arguments(tool_call)
        if arguments:
            print(f"参数：{arguments}", flush=True)

    def after_tool(
        self,
        result: APIMessage,
        elapsed_seconds: float | None = None,
    ) -> None:
        name = str(result.get("name") or "unknown")
        call_id = str(result.get("tool_call_id") or "")
        timer_key = call_id or name
        started_at = self.tool_started_at.pop(timer_key, None)
        measured_elapsed = (
            elapsed_seconds
            if elapsed_seconds is not None
            else time.perf_counter() - started_at
            if started_at is not None
            else None
        )
        elapsed = (
            f"，耗时 {measured_elapsed:.3f} 秒"
            if measured_elapsed is not None
            else ""
        )
        try:
            content = json.loads(str(result.get("content") or "{}"))
            status = "完成" if content.get("ok") is not False else "失败"
        except json.JSONDecodeError:
            content = {"ok": False, "error": "Invalid tool result"}
            status = "失败"
        print(f"工具{status}：{name}{elapsed}", flush=True)
        print(f"结果摘要：{_format_result_preview(content)}", flush=True)

    def finish(self, output: str | None) -> None:
        if not output:
            return
        if not self.saw_model_event:
            # /help、/history 等本地命令不是模型结果，保持原样输出。
            print(output)
            return
        print("\n[最终结果]")
        print(output)

    def _open_process_section(self) -> None:
        if not self.process_section_open:
            print("\n[中间过程]")
            self.process_section_open = True


def _tool_call_name(tool_call: object) -> str:
    if not isinstance(tool_call, dict):
        return ""
    function = tool_call.get("function") or {}
    if not isinstance(function, dict):
        return ""
    return str(function.get("name") or "")


def _tool_call_id(tool_call: object) -> str:
    if not isinstance(tool_call, dict):
        return ""
    return str(tool_call.get("id") or "")


def _format_tool_arguments(tool_call: object) -> str:
    if not isinstance(tool_call, dict):
        return ""
    function = tool_call.get("function") or {}
    if not isinstance(function, dict):
        return ""
    raw_arguments = str(function.get("arguments") or "")
    if not raw_arguments:
        return ""
    try:
        parsed = json.loads(raw_arguments)
        return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
    except json.JSONDecodeError:
        return raw_arguments


def _format_result_preview(content: object, max_chars: int = 800) -> str:
    try:
        rendered = json.dumps(content, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        rendered = str(content)
    if len(rendered) <= max_chars:
        return rendered
    return f"{rendered[:max_chars]}…（结果已截断）"


def _create_agent_and_renderer() -> tuple[SimpleAgent, StreamRenderer]:
    renderer = StreamRenderer()
    return SimpleAgent(), renderer


def _consume_events(
    stream: Generator[Message, None, str | None],
    renderer: StreamRenderer,
) -> str | None:
    while True:
        try:
            renderer.handle_event(next(stream))
        except StopIteration as stop:
            return stop.value


def run_interactive() -> None:
    agent, renderer = _create_agent_and_renderer()
    state = AgentState()

    print("Simple Python Agent")
    print("Type /help for commands, /exit to quit.")

    while state.running:
        try:
            user_input = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        renderer.reset()
        output = _consume_events(
            submit_input_stream(user_input, agent, state),
            renderer,
        )
        renderer.finish(output)


def run_headless(prompt: str) -> None:
    agent, renderer = _create_agent_and_renderer()
    state = AgentState()
    renderer.reset()
    output = _consume_events(
        submit_input_stream(prompt, agent, state),
        renderer,
    )
    renderer.finish(output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A minimal Python agent CLI.")
    parser.add_argument(
        "-p",
        "--print",
        dest="prompt",
        help="Run one prompt and exit.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.prompt is not None:
        run_headless(args.prompt)
        return

    run_interactive()
