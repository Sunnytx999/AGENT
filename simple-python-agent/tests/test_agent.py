from __future__ import annotations

import json
import io
import os
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from simple_agent_cli.agent import LLMConfig, SimpleAgent
from simple_agent_cli.cli import StreamRenderer
from simple_agent_cli.messages import (
    AssistantMessage,
    MessageList,
    create_assistant_message,
    create_stream_event,
    create_system_api_error_message,
)
from simple_agent_cli.model.types import ModelMessage, ToolDefinition
from simple_agent_cli.model.client import OpenAIChatClient
from simple_agent_cli.tools import call_local_tool


class StubModelClient:
    def __init__(self, responses: list[list[ModelMessage]]) -> None:
        self.responses = iter(responses)
        self.requests: list[MessageList] = []

    def call_model(
        self,
        messages: MessageList,
        _tools: list[ToolDefinition],
    ) -> list[ModelMessage]:
        self.requests.append(list(messages))
        return next(self.responses)


def stub_agent(
    responses: list[list[ModelMessage]],
    max_tool_turns: int = 3,
) -> tuple[SimpleAgent, StubModelClient]:
    client = StubModelClient(responses)
    agent = SimpleAgent(
        config=LLMConfig(
            api_key=None,
            base_url="http://example.invalid",
            model="test-model",
            max_tool_turns=max_tool_turns,
        ),
        model_client=client,
    )
    return agent, client


def assistant_response(
    content: str = "",
    tool_calls: list[dict[str, object]] | None = None,
) -> AssistantMessage:
    message: dict[str, object] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return create_assistant_message(
        message,
        response_id="response-test",
        model="test-model",
        finish_reason="tool_calls" if tool_calls else "stop",
    )


def tool_call(name: str = "get_local_time", call_id: str = "call-1") -> dict[str, object]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": "{}"},
    }


class QueryLoopTests(unittest.TestCase):
    def test_tool_result_is_sent_before_final_answer(self) -> None:
        agent, client = stub_agent(
            [
                [assistant_response(tool_calls=[tool_call()])],
                [assistant_response("It is noon.")],
            ]
        )

        self.assertEqual(agent.respond("What time is it?"), "It is noon.")
        self.assertEqual([message["role"] for message in agent.messages[-4:]], [
            "user", "assistant", "tool", "assistant"
        ])
        self.assertTrue(json.loads(str(agent.messages[-2]["content"]))["ok"])
        self.assertEqual(client.requests[1][-1]["role"], "tool")
        self.assertEqual([event["type"] for event in agent.events], [
            "assistant", "stream_event", "stream_event", "assistant"
        ])
        self.assertEqual(
            agent.events[1]["event"]["type"],
            "tool_call_start",
        )
        self.assertEqual(
            agent.events[2]["event"]["type"],
            "tool_result",
        )

    def test_respond_stream_yields_before_model_generator_finishes(self) -> None:
        progress: list[str] = []

        class StreamingStub:
            def call_model(
                self,
                _messages: MessageList,
                _tools: list[ToolDefinition],
            ):
                progress.append("started")
                yield create_stream_event(
                    {
                        "choices": [
                            {"delta": {"role": "assistant", "content": "Hi"}}
                        ]
                    }
                )
                progress.append("continued")
                yield assistant_response("Hi")

        agent = SimpleAgent(
            config=LLMConfig(
                api_key=None,
                base_url="http://example.invalid",
                model="test-model",
            ),
            model_client=StreamingStub(),
        )

        stream = agent.respond_stream("hello")
        first_event = next(stream)

        self.assertEqual(first_event["type"], "stream_event")
        self.assertEqual(progress, ["started"])

        with self.assertRaises(StopIteration) as stopped:
            while True:
                next(stream)
        self.assertEqual(stopped.exception.value, "Hi")
        self.assertEqual(progress, ["started", "continued"])

    def test_tool_turn_limit_rolls_back_the_whole_user_turn(self) -> None:
        agent, _client = stub_agent(
            [
                [assistant_response(tool_calls=[tool_call(call_id="call-1")])],
                [assistant_response(tool_calls=[tool_call(call_id="call-2")])],
            ],
            max_tool_turns=1,
        )
        original_messages = list(agent.messages)

        result = agent.respond("keep calling tools")

        self.assertIn("tool call limit reached (1)", result)
        self.assertEqual(agent.messages, original_messages)

    def test_repeated_tool_failure_is_stopped(self) -> None:
        agent, _client = stub_agent(
            [
                [assistant_response(tool_calls=[tool_call("missing", "call-1")])],
                [assistant_response(tool_calls=[tool_call("missing", "call-2")])],
            ]
        )
        original_messages = list(agent.messages)

        result = agent.respond("fail")

        self.assertIn("repeated tool failure", result)
        self.assertEqual(agent.messages, original_messages)

    def test_stream_event_is_logged_before_assistant_message(self) -> None:
        stream = create_stream_event({"type": "content.delta", "delta": "Hi"})
        agent, _client = stub_agent(
            [[stream, assistant_response("Hi")]]
        )

        self.assertEqual(agent.respond("hello"), "Hi")
        self.assertEqual([event["type"] for event in agent.events], [
            "stream_event", "assistant"
        ])

    def test_api_error_is_logged_but_not_added_to_model_context(self) -> None:
        api_error = create_system_api_error_message(
            "HTTP 429",
            kind="http_error",
            status_code=429,
            detail="rate limited",
        )
        agent, _client = stub_agent([[api_error]])
        original_messages = list(agent.messages)

        result = agent.respond("hello")

        self.assertIn("HTTP 429: rate limited", result)
        self.assertEqual(agent.messages, original_messages)
        self.assertEqual(agent.events[-1]["subtype"], "api_error")


class StreamingClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = OpenAIChatClient(
            LLMConfig(
                api_key=None,
                base_url="http://example.invalid",
                model="test-model",
            )
        )

    def test_sse_text_chunks_are_merged_into_assistant_message(self) -> None:
        response = [
            b'data: {"id":"r1","model":"m1","choices":[{"delta":{"role":"assistant","content":"Hel"},"finish_reason":null}]}\n',
            b'data: {"id":"r1","model":"m1","choices":[{"delta":{"content":"lo"},"finish_reason":"stop"}]}\n',
            b"data: [DONE]\n",
        ]

        events = list(self.client._read_stream(response, "request-1", time.perf_counter()))

        self.assertEqual([event["type"] for event in events], [
            "stream_event", "stream_event", "assistant"
        ])
        self.assertEqual(events[-1]["message"]["content"], "Hello")
        self.assertEqual(events[-1]["finish_reason"], "stop")

    def test_sse_tool_call_chunks_are_merged(self) -> None:
        response = [
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call-1","type":"function","function":{"name":"read_","arguments":"{\\\"path\\\":"}}]},"finish_reason":null}]}\n',
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"file","arguments":"\\\"x.py\\\"}"}}]},"finish_reason":"tool_calls"}]}\n',
            b"data: [DONE]\n",
        ]

        events = list(self.client._read_stream(response, None, time.perf_counter()))
        tool_call = events[-1]["message"]["tool_calls"][0]

        self.assertEqual(tool_call["id"], "call-1")
        self.assertEqual(tool_call["function"]["name"], "read_file")
        self.assertEqual(tool_call["function"]["arguments"], '{"path":"x.py"}')

    def test_renderer_prints_final_stream_only_once(self) -> None:
        renderer = StreamRenderer()
        event = create_stream_event(
            {"choices": [{"delta": {"content": "Hello"}}]}
        )
        output = io.StringIO()

        with redirect_stdout(output):
            renderer.after_model(event)
            renderer.after_model(assistant_response("Hello"))
            renderer.finish("Hello")

        rendered = output.getvalue()
        self.assertIn('[中间过程]\n模型 delta：{"content":"Hello"}', rendered)
        self.assertEqual(rendered.count("[最终结果]"), 1)
        self.assertTrue(rendered.endswith("[最终结果]\nHello\n"))

    def test_renderer_separates_multiple_streamed_turns_from_final_result(self) -> None:
        renderer = StreamRenderer()
        first = create_stream_event(
            {"choices": [{"delta": {"content": "Checking..."}}]}
        )
        final = create_stream_event(
            {"choices": [{"delta": {"content": "Done"}}]}
        )
        output = io.StringIO()

        with redirect_stdout(output):
            renderer.after_model(first)
            renderer.after_model(
                assistant_response("Checking...", tool_calls=[tool_call()])
            )
            renderer.after_model(final)
            renderer.after_model(assistant_response("Done"))
            renderer.finish("Done")

        rendered = output.getvalue()
        self.assertIn("[中间过程]", rendered)
        self.assertIn("第 1 轮：模型准备调用 1 个工具", rendered)
        self.assertIn("模型说明：Checking...", rendered)
        self.assertNotIn("Checking...Done", rendered)
        self.assertTrue(rendered.endswith("[最终结果]\nDone\n"))

    def test_renderer_shows_tool_arguments_status_and_result_preview(self) -> None:
        renderer = StreamRenderer()
        call = tool_call("read_file", "call-42")
        call["function"]["arguments"] = '{"path":"main.py"}'
        result = {
            "role": "tool",
            "tool_call_id": "call-42",
            "name": "read_file",
            "content": '{"ok":true,"result":{"content":"hello"}}',
        }
        output = io.StringIO()

        with redirect_stdout(output):
            renderer.before_tool(call)
            renderer.after_tool(result)

        rendered = output.getvalue()
        self.assertIn("调用工具：read_file（call-42）", rendered)
        self.assertIn('参数：{"path":"main.py"}', rendered)
        self.assertIn("工具完成：read_file，耗时 ", rendered)
        self.assertIn('结果摘要：{"ok":true', rendered)


class LocalToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_cwd = Path.cwd()
        os.chdir(self.temp_dir.name)
        Path("sample.py").write_text("first\nHello Agent\nlast\n", encoding="utf-8")

    def tearDown(self) -> None:
        os.chdir(self.original_cwd)
        self.temp_dir.cleanup()

    def test_read_file_returns_numbered_range(self) -> None:
        result = call_local_tool("read_file", {"path": "sample.py", "offset": 2, "limit": 1})
        self.assertTrue(result["ok"])
        self.assertEqual(result["result"]["content"], "2: Hello Agent")

    def test_list_files_filters_with_glob(self) -> None:
        result = call_local_tool("list_files", {"pattern": "**/*.py"})
        self.assertEqual(result["result"]["files"], [str(Path("sample.py").resolve())])

    def test_search_text_returns_file_and_line(self) -> None:
        result = call_local_tool(
            "search_text",
            {"pattern": "hello", "glob": "**/*.py", "case_sensitive": False},
        )
        self.assertEqual(result["result"]["matches"][0]["line"], 2)

    def test_read_file_accepts_absolute_path(self) -> None:
        path = Path("sample.py").resolve()
        result = call_local_tool("read_file", {"path": str(path), "limit": 1})
        self.assertTrue(result["ok"])
        self.assertEqual(result["result"]["path"], str(path))

    def test_write_file_creates_parent_directories(self) -> None:
        result = call_local_tool(
            "write_file",
            {"path": "new/nested.txt", "content": "hello\nworld\n"},
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["result"]["operation"], "created")
        self.assertEqual(Path("new/nested.txt").read_text(encoding="utf-8"), "hello\nworld\n")

    def test_write_file_overwrites_existing_file(self) -> None:
        result = call_local_tool(
            "write_file",
            {"path": "sample.py", "content": "replacement\n"},
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["result"]["operation"], "updated")
        self.assertEqual(Path("sample.py").read_text(encoding="utf-8"), "replacement\n")

    def test_edit_file_replaces_unique_text(self) -> None:
        result = call_local_tool(
            "edit_file",
            {
                "path": "sample.py",
                "old_string": "Hello Agent",
                "new_string": "Hello World",
            },
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["result"]["replacements"], 1)
        self.assertIn("Hello World", Path("sample.py").read_text(encoding="utf-8"))

    def test_edit_file_rejects_ambiguous_match(self) -> None:
        Path("repeated.txt").write_text("same\nsame\n", encoding="utf-8")
        result = call_local_tool(
            "edit_file",
            {
                "path": "repeated.txt",
                "old_string": "same",
                "new_string": "changed",
            },
        )

        self.assertFalse(result["ok"])
        self.assertIn("occurs 2 times", result["error"])
        self.assertEqual(Path("repeated.txt").read_text(encoding="utf-8"), "same\nsame\n")

    def test_edit_file_can_replace_all_matches(self) -> None:
        Path("repeated.txt").write_text("same\nsame\n", encoding="utf-8")
        result = call_local_tool(
            "edit_file",
            {
                "path": "repeated.txt",
                "old_string": "same",
                "new_string": "changed",
                "replace_all": True,
            },
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["result"]["replacements"], 2)
        self.assertEqual(
            Path("repeated.txt").read_text(encoding="utf-8"),
            "changed\nchanged\n",
        )


if __name__ == "__main__":
    unittest.main()
