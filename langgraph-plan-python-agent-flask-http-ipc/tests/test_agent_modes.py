from __future__ import annotations

import unittest
from pathlib import Path

from langchain_agent_cli.agent import LangChainAgent, MAIN_AGENT_PROMPT


class AgentModeTests(unittest.TestCase):
    def test_plan_prompt_requires_plan_subagent_for_game_creation(self):
        self.assertIn("MUST call `plan_codebase`", MAIN_AGENT_PROMPT)
        self.assertIn("Creating a game", MAIN_AGENT_PROMPT)
        self.assertIn("If the task warrants writing a multi-step plan.md", MAIN_AGENT_PROMPT)

    def test_approve_uses_isolated_thread_and_accepts_main_written_plan(self):
        agent = LangChainAgent.__new__(LangChainAgent)
        agent.pending_plan = "# Plan\n\n1. Build the game."
        agent.approved_plan = None
        agent.current_task = "Build a game"
        agent.plan_mode = True
        agent.working_directory = Path.cwd().resolve()
        agent.approval_thread_id = "session:approval"
        agent._execution_authorized = False
        agent._execution_called = False
        agent._log = lambda _message: None
        captured: dict[str, str] = {}

        def invoke_main(prompt: str, thread_id: str | None = None) -> str:
            captured["prompt"] = prompt
            captured["thread_id"] = thread_id or ""
            agent._execution_called = True
            return "execution started"

        agent._invoke_main = invoke_main

        result = agent.approve_plan()

        self.assertEqual(result, "execution started")
        self.assertFalse(agent.plan_mode)
        self.assertTrue(agent._execution_authorized)
        self.assertEqual(captured["thread_id"], "session:approval")
        self.assertIn("RUNTIME MODE: PLAN_APPROVED", captured["prompt"])
        self.assertIn("Do not claim that approval is still required", captured["prompt"])
        self.assertIn("# Plan", captured["prompt"])

    def test_typed_approve_command_switches_mode_before_invoking_main(self):
        agent = LangChainAgent.__new__(LangChainAgent)
        agent.pending_plan = "# Plan\n\n1. Build the game."
        agent.approved_plan = None
        agent.current_task = "Build a game"
        agent.plan_mode = True
        agent.working_directory = Path.cwd().resolve()
        agent.approval_thread_id = "session:approval"
        agent._execution_authorized = False
        agent._execution_called = False
        agent._log = lambda _message: None
        captured: dict[str, str] = {}

        def invoke_main(prompt: str, thread_id: str | None = None) -> str:
            captured["prompt"] = prompt
            captured["thread_id"] = thread_id or ""
            agent._execution_called = True
            return "execution started"

        agent._invoke_main = invoke_main

        result = agent.respond("/approve")

        self.assertEqual(result, "execution started")
        self.assertFalse(agent.plan_mode)
        self.assertTrue(agent._execution_authorized)
        self.assertTrue(captured["prompt"].startswith("RUNTIME MODE: PLAN_APPROVED"))
        self.assertEqual(captured["thread_id"], "session:approval")


if __name__ == "__main__":
    unittest.main()
