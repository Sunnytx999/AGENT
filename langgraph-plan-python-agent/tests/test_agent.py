from __future__ import annotations

import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from prompt_toolkit.document import Document

from langchain_agent_cli.agent import (
    MAIN_AGENT_PROMPT,
    AgentConfig,
    LangChainAgent,
    SessionState,
)
from langchain_agent_cli.cli import CommandCompleter, handle_command
from langchain_agent_cli.config import AgentConfig as Config
from langchain_agent_cli.execution_workflow import ExecutionWorkflow
from langchain_agent_cli.execution_workflow import EXECUTION_AGENT_PROMPT
from langchain_agent_cli.explore_agent import EXPLORE_PROMPT
from langchain_agent_cli.plan_agent import PLAN_PROMPT
from langchain_agent_cli.shared_prompt import SHARED_FILE_INSPECTION_PROMPT
from langchain_agent_cli.tools import (
    edit_file,
    list_files,
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


class StubExecutionWorkflow:
    def __init__(self) -> None:
        self.received = None

    def invoke(self, task, plan, recursion_limit):
        self.received = (task, plan, recursion_limit)
        return {
            "past_steps": [(plan[0], "implemented")],
            "response": "all done",
        }


class StreamingToolGraph:
    def __init__(self) -> None:
        self.last_messages = []

    def stream(self, value, config, stream_mode):
        call = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_text",
                    "args": {"pattern": "plan", "glob": "**/*.py"},
                    "id": "call-123",
                    "type": "tool_call",
                }
            ],
        )
        result = ToolMessage(
            content={"matches": [{"path": "agent.py", "line": 1}]},
            name="search_text",
            tool_call_id="call-123",
        )
        final = AIMessage(content="done")
        self.last_messages = [*value["messages"], call, result, final]
        yield {"model": {"messages": [call]}}
        yield {"tools": {"messages": [result]}}
        yield {"model": {"messages": [final]}}

    def get_state(self, config):
        return type("Snapshot", (), {"values": {"messages": self.last_messages}})()


class AgentTests(unittest.TestCase):
    def test_all_agents_share_local_file_inspection_policy(self) -> None:
        marker = "## Local File Inspection Policy"

        self.assertIn(marker, SHARED_FILE_INSPECTION_PROMPT)
        for prompt in (
            MAIN_AGENT_PROMPT,
            EXPLORE_PROMPT,
            PLAN_PROMPT,
            EXECUTION_AGENT_PROMPT,
        ):
            self.assertIn(marker, prompt)
            self.assertIn("If the result contains no files plausibly related", prompt)

    def test_verbose_stream_logs_tool_arguments_results_and_nodes(self) -> None:
        agent = LangChainAgent(
            config=AgentConfig(None, "http://example.invalid", "test"),
            graph=StreamingToolGraph(),
            verbose=True,
        )
        output = io.StringIO()

        with redirect_stdout(output):
            result = agent.respond("inspect planning")

        logs = output.getvalue()
        self.assertEqual(result, "done")
        self.assertIn("[main] graph node: model", logs)
        self.assertIn("[main] graph node: tools", logs)
        self.assertIn("tool call: search_text (id=call-123)", logs)
        self.assertIn('\"pattern\": \"plan\"', logs)
        self.assertIn("tool result: search_text (id=call-123)", logs)
        self.assertIn("'path': 'agent.py'", logs)

    def test_main_prompt_defines_complete_plan_mode_phases(self) -> None:
        expected_sections = [
            "## PLAN Mode: Non-Negotiable Restrictions",
            "## PLAN Mode Phase 1: Understand and Explore",
            "## PLAN Mode Phase 2: Design",
            "## PLAN Mode Phase 3: Main-Agent Review",
            "## PLAN Mode Phase 4: Write the Final Plan",
            "## PLAN Mode Phase 5: Wait for Approval",
            "## PLAN Mode: Revising an Existing Plan",
            "## PLAN_APPROVED Mode: Mandatory Execution",
        ]

        for section in expected_sections:
            self.assertIn(section, MAIN_AGENT_PROMPT)
        self.assertIn(
            "try to call at least one `plan_codebase` subagent",
            MAIN_AGENT_PROMPT,
        )
        self.assertIn("You MUST call `execute_plan`", MAIN_AGENT_PROMPT)

    def test_main_agent_is_created_during_initialization(self) -> None:
        agent = LangChainAgent(
            config=AgentConfig(None, "http://example.invalid", "test"),
        )

        self.assertIsNotNone(agent.main_graph)
        self.assertIsNone(agent.explore_graph)
        self.assertIsNone(agent.plan_graph)
        self.assertIsNone(agent.execution_graph)

    def test_slash_command_completion_includes_description(self) -> None:
        completions = list(
            CommandCompleter().get_completions(Document("/pl"), None)
        )

        self.assertEqual([item.text for item in completions], ["/plan"])
        self.assertEqual(completions[0].display_meta_text, "进入 Plan 模式")

    def test_normal_respond_uses_main_agent(self) -> None:
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
        self.assertIn("RUNTIME MODE: NORMAL", graph.received[0]["messages"][0].content)
        self.assertIsNone(agent.execution_graph)

    def test_plan_mode_uses_same_main_agent(self) -> None:
        graph = StubGraph("reviewed plan")
        agent = LangChainAgent(
            config=AgentConfig(None, "http://example.invalid", "test"),
            graph=graph,
        )
        agent.enter_plan_mode()

        result = agent.respond("Add a new CLI command")

        self.assertEqual(result, "reviewed plan")
        prompt = graph.received[0]["messages"][0].content
        self.assertIn("RUNTIME MODE: PLAN", prompt)
        self.assertIn("Original task:\nAdd a new CLI command", prompt)
        self.assertEqual(agent.current_task, "Add a new CLI command")
        self.assertFalse(hasattr(agent, "chat_graph"))

    def test_main_agent_tools_expose_three_independent_components(self) -> None:
        agent = LangChainAgent(
            config=AgentConfig(None, "http://example.invalid", "test"),
            graph=StubGraph(),
        )

        names = {tool.name for tool in agent._get_main_tools()}

        self.assertTrue({"explore_codebase", "plan_codebase", "execute_plan"} <= names)

    def test_plan_mode_can_only_write_active_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = LangChainAgent(
                config=AgentConfig(None, "http://example.invalid", "test"),
                graph=StubGraph(),
            )
            agent.plan_file_path = Path(tmp) / "plan.md"
            agent.enter_plan_mode()
            tools = {tool.name: tool for tool in agent._get_main_tools()}

            with self.assertRaises(PermissionError):
                tools["write_file"].invoke(
                    {"path": str(Path(tmp) / "code.py"), "content": "blocked"}
                )

            tools["write_file"].invoke(
                {"path": str(agent.plan_file_path), "content": "1. change code"}
            )
            self.assertEqual(agent.pending_plan, "1. change code")
            self.assertFalse(hasattr(agent, "pending_steps"))

    def test_explore_and_plan_tools_return_results_to_main(self) -> None:
        agent = LangChainAgent(
            config=AgentConfig(None, "http://example.invalid", "test"),
            graph=StubGraph(),
        )
        agent.current_task = "add feature"
        agent.explore_graph = StubGraph("files found")
        agent.plan_graph = StubGraph("candidate plan")
        tools = {tool.name: tool for tool in agent._get_main_tools()}

        self.assertEqual(
            tools["explore_codebase"].invoke({"request": "inspect routing"}),
            "files found",
        )
        self.assertEqual(
            tools["plan_codebase"].invoke(
                {
                    "task": "add feature",
                    "request": "design the change",
                }
            ),
            "candidate plan",
        )
        self.assertEqual(agent.current_task, "add feature")

    def test_plan_subagent_call_establishes_task_only_once(self) -> None:
        agent = LangChainAgent(
            config=AgentConfig(None, "http://example.invalid", "test"),
            graph=StubGraph(),
        )
        agent.plan_mode = True
        agent.plan_graph = StubGraph("candidate plan")
        tool = next(tool for tool in agent._get_main_tools() if tool.name == "plan_codebase")

        tool.invoke({"task": "build feature", "request": "create initial plan"})
        tool.invoke({"task": "change the plan", "request": "revise one detail"})

        self.assertEqual(agent.current_task, "build feature")

    def test_execute_tool_uses_combined_execution_component_after_approval(self) -> None:
        agent = LangChainAgent(
            config=AgentConfig(None, "http://example.invalid", "test"),
            graph=StubGraph(),
        )
        workflow = StubExecutionWorkflow()
        agent.execution_graph = StubGraph()
        agent.execution_workflow = workflow
        agent.current_task = "add feature"
        agent.pending_plan = "1. implement it"
        agent.approved_plan = agent.pending_plan
        agent._execution_authorized = True
        tools = {tool.name: tool for tool in agent._get_main_tools()}

        result = tools["execute_plan"].invoke(
            {"steps": ["implement it", "run focused tests"]}
        )

        self.assertIn("Execute component finished", result)
        self.assertIn("Step: implement it", result)
        self.assertIn("Final response:\nall done", result)
        self.assertEqual(
            workflow.received[0:2],
            ("add feature", ["implement it", "run focused tests"]),
        )
        self.assertFalse(agent._execution_authorized)
        self.assertIsNone(agent.pending_plan)

    def test_approve_plan_routes_back_through_main_with_execute_requirement(self) -> None:
        agent = LangChainAgent(
            config=AgentConfig(None, "http://example.invalid", "test"),
            graph=StubGraph("executed"),
        )
        agent.pending_plan = "do the thing"
        agent.current_task = "test task"

        result = agent.approve_plan()

        self.assertEqual(result, "executed")
        self.assertEqual(agent.approved_plan, "do the thing")
        self.assertFalse(agent.plan_mode)
        approval_prompt = agent.main_graph.received[0]["messages"][0].content
        self.assertIn("RUNTIME MODE: PLAN_APPROVED", approval_prompt)
        self.assertIn("MUST now call execute_plan", approval_prompt)
        self.assertIn("steps list derived semantically", approval_prompt)

    def test_reject_plan_keeps_plan_for_revision(self) -> None:
        agent = LangChainAgent(
            config=AgentConfig(None, "http://example.invalid", "test"),
            graph=StubGraph(),
        )
        agent.pending_plan = "do the thing"
        agent.plan_mode = True

        result = agent.reject_plan()

        self.assertIn("not approved", result)
        self.assertEqual(agent.pending_plan, "do the thing")
        self.assertTrue(agent.plan_mode)

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

    def test_plan_mode_answers_simple_request_with_main_agent(self) -> None:
        agent = LangChainAgent(
            config=AgentConfig(None, "http://example.invalid", "test"),
            graph=StubGraph("hello"),
        )
        agent.plan_mode = True

        self.assertEqual(agent.respond("你好"), "hello")
        self.assertEqual(agent.current_task, "你好")

        agent.respond("实现一个扫雷游戏")
        self.assertEqual(agent.current_task, "实现一个扫雷游戏")

    def test_saved_plan_prevents_revision_feedback_from_replacing_task(self) -> None:
        agent = LangChainAgent(
            config=AgentConfig(None, "http://example.invalid", "test"),
            graph=StubGraph("revised"),
        )
        agent.plan_mode = True
        agent.current_task = "实现一个扫雷游戏"
        agent.pending_plan = "1. 实现棋盘"

        agent.respond("增加三个难度等级")

        self.assertEqual(agent.current_task, "实现一个扫雷游戏")

    def test_approve_rejects_plan_without_task_established_by_plan_agent(self) -> None:
        agent = LangChainAgent(
            config=AgentConfig(None, "http://example.invalid", "test"),
            graph=StubGraph(),
        )
        agent.pending_plan = "a plan written without calling Plan"

        result = agent.approve_plan()

        self.assertIn("no established original task", result)
        self.assertIsNone(agent.approved_plan)

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

class ExecutionWorkflowTests(unittest.TestCase):
    def test_graph_executes_one_step_and_replans_until_complete(self) -> None:
        executor_prompts = []
        replanner_prompts = []
        events = []
        replan_responses = iter([
            '{"steps": ["replacement verification"], "response": null}',
            '{"steps": [], "response": "all done"}',
        ])

        def executor_agent(prompt):
            executor_prompts.append(prompt)
            if "replacement verification" in prompt:
                return "finished replacement verification"
            return "finished inspect"

        def replanner_agent(prompt):
            replanner_prompts.append(prompt)
            return next(replan_responses)

        workflow = ExecutionWorkflow(
            execution_agent=lambda prompt: (
                replanner_agent(prompt)
                if "Current remaining plan:" in prompt
                else executor_agent(prompt)
            ),
            on_event=events.append,
        )
        result = workflow.invoke("build feature", ["inspect", "old verification"])

        self.assertIn("Execute only this current step:\ninspect", executor_prompts[0])
        self.assertIn(
            "Execute only this current step:\nreplacement verification",
            executor_prompts[1],
        )
        self.assertIn(
            'Current remaining plan:\n["old verification"]',
            replanner_prompts[0],
        )
        self.assertIn("will replace the current remaining plan", replanner_prompts[0])
        self.assertIn("Result: finished inspect", executor_prompts[1])
        self.assertEqual(result["response"], "all done")
        self.assertEqual(len(result["past_steps"]), 2)
        self.assertIn("[replan] plan changed: yes", events)
        self.assertIn("[replan] task complete: all done", events)

    def test_llm_failure_stops_without_replanning_loop(self) -> None:
        prompts = []

        def failing_execution_agent(prompt):
            prompts.append(prompt)
            return "LLM request failed: insufficient balance"

        workflow = ExecutionWorkflow(execution_agent=failing_execution_agent)
        result = workflow.invoke("build feature", ["change code", "run tests"])

        self.assertEqual(len(prompts), 1)
        self.assertEqual(result["response"], "LLM request failed: insufficient balance")


if __name__ == "__main__":
    unittest.main()


