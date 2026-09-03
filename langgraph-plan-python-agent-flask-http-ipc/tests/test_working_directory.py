from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from langchain_agent_cli import LangChainAgent
from langchain_agent_cli.config import AgentConfig
from langchain_agent_cli.shared_prompt import prompt_with_working_directory
from langchain_agent_cli.subagent_tools import (
    make_execute_tool,
    make_explore_tool,
    make_plan_tool,
)


NO_MODEL = AgentConfig(api_key=None, base_url="", model=None)


class WorkingDirectoryTests(unittest.TestCase):
    def test_main_agent_saves_working_directory_at_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            expected = Path(temporary).resolve()
            agent = LangChainAgent(config=NO_MODEL, working_directory=expected)
            self.assertEqual(agent.working_directory, expected)
            self.assertEqual(agent._working_directory_for_subagent(None), expected)

    def test_subagent_override_requires_existing_absolute_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            agent = LangChainAgent(config=NO_MODEL, working_directory=Path(temporary))
            with self.assertRaises(ValueError):
                agent._working_directory_for_subagent("relative-directory")
            with self.assertRaises(ValueError):
                agent._working_directory_for_subagent(
                    str(Path(temporary).resolve() / "missing")
                )

    def test_working_directory_is_injected_into_system_prompt(self) -> None:
        prompt = prompt_with_working_directory("BASE", r"E:\project")
        self.assertIn("## Working Directory", prompt)
        self.assertIn(r"E:\project", prompt)
        self.assertIn("absolute path", prompt)

    def test_subagent_tools_default_or_forward_working_directory(self) -> None:
        calls: list[tuple[object, ...]] = []
        explore = make_explore_tool(
            lambda request, directory: calls.append((request, directory)) or "explored"
        )
        plan = make_plan_tool(
            lambda task, request, directory: calls.append((task, request, directory))
            or "planned"
        )
        execute = make_execute_tool(
            lambda steps, directory: calls.append((steps, directory)) or "executed"
        )

        self.assertEqual(explore.invoke({"request": "inspect"}), "explored")
        self.assertEqual(
            plan.invoke(
                {
                    "task": "change it",
                    "request": "make a plan",
                    "working_directory": r"E:\other",
                }
            ),
            "planned",
        )
        self.assertEqual(execute.invoke({"steps": ["implement"]}), "executed")
        self.assertEqual(calls[0], ("inspect", None))
        self.assertEqual(calls[1], ("change it", "make a plan", r"E:\other"))
        self.assertEqual(calls[2], (["implement"], None))

        schema = explore.args_schema.model_json_schema()
        description = schema["properties"]["working_directory"]["description"]
        self.assertIn("absolute directory", description)


if __name__ == "__main__":
    unittest.main()
