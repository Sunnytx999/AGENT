from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from langchain_agent_cli.agent import AgentConfig, LangChainAgent, SessionState
from langchain_agent_cli.cli import handle_command
from langchain_agent_cli.config import AgentConfig as Config
from langchain_agent_cli.tools import (
    TodoItem,
    edit_file,
    list_files,
    make_todo_write_tool,
    read_file,
    search_text,
    write_file,
)


class StubGraph:
    def __init__(self, reply: str = "done") -> None:
        self.reply = reply
        self.received = None
        self.last_messages = []

    def invoke(self, value, config):
        self.received = (value, config)
        self.last_messages = [*value["messages"], AIMessage(content=self.reply)]
        return {"messages": self.last_messages}

    def get_state(self, config):
        return type("Snapshot", (), {"values": {"messages": self.last_messages}})()


class AgentTests(unittest.TestCase):
    def test_normal_respond_uses_executor(self) -> None:
        graph = StubGraph("normal")
        agent = LangChainAgent(
            config=AgentConfig(None, "http://example.invalid", "test", max_tool_turns=3),
            graph=graph,
        )
        self.assertEqual(agent.respond("hello"), "normal")
        self.assertEqual(
            graph.received[1]["configurable"]["thread_id"],
            "cli-session",
        )

    def test_plan_task_uses_explore_and_plan_agents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = LangChainAgent(
                config=AgentConfig(None, "http://example.invalid", "test"),
                graph=StubGraph("executor"),
            )
            agent.explore_graph = StubGraph("exploration result")
            agent.plan_graph = StubGraph("implementation plan")
            agent.plan_file_path = Path(tmp) / "plan.md"

            result = agent.plan_task("Add a new CLI command")

            self.assertIn("implementation plan", result)
            self.assertEqual(agent.pending_plan, "implementation plan")
            self.assertTrue(agent.plan_file_path.exists())
            self.assertEqual(
                agent.plan_file_path.read_text(encoding="utf-8"),
                "implementation plan",
            )

    def test_approve_plan_executes(self) -> None:
        agent = LangChainAgent(
            config=AgentConfig(None, "http://example.invalid", "test"),
            graph=StubGraph("executed"),
        )
        agent.pending_plan = "do the thing"

        result = agent.approve_plan()

        self.assertEqual(result, "executed")
        self.assertEqual(agent.approved_plan, "do the thing")
        self.assertFalse(agent.plan_mode)

    def test_reject_plan_clears_pending(self) -> None:
        agent = LangChainAgent(
            config=AgentConfig(None, "http://example.invalid", "test"),
            graph=StubGraph(),
        )
        agent.pending_plan = "do the thing"
        agent.plan_mode = True

        result = agent.reject_plan()

        self.assertIn("rejected", result)
        self.assertIsNone(agent.pending_plan)
        self.assertTrue(agent.plan_mode)

    def test_todo_display(self) -> None:
        agent = LangChainAgent(
            config=AgentConfig(None, "http://example.invalid", "test"),
            graph=StubGraph(),
        )
        agent.todos = [
            TodoItem(content="Read files", status="completed", activeForm="Reading files"),
            TodoItem(content="Write code", status="in_progress", activeForm="Writing code"),
        ]

        output = agent.show_todos()

        self.assertIn("[completed] Read files", output)
        self.assertIn("[in_progress] Write code", output)

    def test_plan_commands(self) -> None:
        agent = LangChainAgent(
            config=AgentConfig(None, "http://example.invalid", "test"),
            graph=StubGraph(),
        )
        state = SessionState()

        self.assertIn("enabled", handle_command("/plan", state, agent))
        self.assertTrue(agent.plan_mode)
        self.assertIn("exited", handle_command("/exitplan", state, agent))
        self.assertFalse(agent.plan_mode)

    def test_plan_mode_answers_simple_request_directly(self) -> None:
        agent = LangChainAgent(
            config=AgentConfig(None, "http://example.invalid", "test"),
            graph=StubGraph("executor"),
        )
        agent.chat_graph = StubGraph("hello")
        agent.plan_mode = True

        self.assertEqual(agent.respond("你好"), "hello")

    def test_missing_model_has_clear_message(self) -> None:
        agent = LangChainAgent(config=AgentConfig(None, "http://example.invalid", None))
        self.assertIn("SIMPLE_AGENT_MODEL", agent.respond("hello"))

    def test_config_matches_original_fixed_settings(self) -> None:
        config = Config.from_env()
        self.assertEqual(config.model, "deepseek-v4-pro")
        self.assertEqual(config.base_url, "https://api.deepseek.com")
        self.assertTrue(config.api_key)


class ToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.previous = Path.cwd()
        os.chdir(self.temp.name)
        Path("sample.py").write_text("first\nHello Agent\nlast\n", encoding="utf-8")

    def tearDown(self) -> None:
        os.chdir(self.previous)
        self.temp.cleanup()

    def test_read_and_search(self) -> None:
        read = read_file.invoke({"path": "sample.py", "offset": 2, "limit": 1})
        self.assertEqual(read["content"], "2: Hello Agent")
        found = search_text.invoke({"pattern": "hello", "glob": "**/*.py", "case_sensitive": False})
        self.assertEqual(found["matches"][0]["line"], 2)

    def test_list_write_and_edit(self) -> None:
        self.assertEqual(len(list_files.invoke({"pattern": "**/*.py"})["files"]), 1)
        write_file.invoke({"path": "new/file.txt", "content": "old"})
        edit_file.invoke({"path": "new/file.txt", "old_string": "old", "new_string": "new"})
        self.assertEqual(Path("new/file.txt").read_text(encoding="utf-8"), "new")

    def test_todo_write_updates_owner(self) -> None:
        received = []
        todo_write = make_todo_write_tool(lambda todos: received.extend(todos))

        result = todo_write.invoke({
            "todos": [{
                "content": "Read files",
                "status": "in_progress",
                "activeForm": "Reading files",
            }]
        })

        self.assertIn("updated", result.lower())
        self.assertEqual(received, [
            TodoItem(
                content="Read files",
                status="in_progress",
                activeForm="Reading files",
            )
        ])


if __name__ == "__main__":
    unittest.main()


