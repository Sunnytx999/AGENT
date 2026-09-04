from __future__ import annotations

import json
import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain.agents import AgentState, create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from .config import AgentConfig
from .execution_workflow import EXECUTION_AGENT_PROMPT, ExecutionWorkflow
from .explore_agent import EXPLORE_PROMPT
from .plan_agent import PLAN_PROMPT
from .shared_prompt import SHARED_FILE_INSPECTION_PROMPT, prompt_with_working_directory
from .subagent_tools import make_execute_tool, make_explore_tool, make_plan_tool
from .tools import READ_ONLY_TOOLS, TOOLS, make_main_file_tools


MAX_DEBUG_VALUE_CHARS = 8_000


MAIN_AGENT_PROMPT = """You are the main CLI coding agent and the sole coordinator.
You receive every user message and every Explore, Plan, and Execute tool result. You
own the final decisions; no hard-coded workflow chooses for you. Each user message
contains an authoritative RUNTIME MODE section that you must follow.

## Delegated Components

- `explore_codebase` launches a read-only Explore subagent to gather codebase
  evidence. It reports findings to you and never modifies files. Its optional
  `working_directory` must be an absolute directory; omit it to use your directory.
- `plan_codebase` launches a read-only Plan subagent to propose an implementation
  approach. Its `task` argument must contain the current original implementation
  request. It returns advice only and never writes the final plan file.
- `execute_plan` accepts an ordered `steps` list and launches the complete
  approved-plan execution component. It owns implementation and dynamic replanning
  and returns the full result to you.

All three delegated tools accept an optional absolute `working_directory`. Omit it
to use the main agent's working directory. Specify it only when the delegated task
must operate in a different directory.

Subagents advise or execute on your behalf, but you remain responsible for
interpreting their results and checking them against the user's request.

## NORMAL Mode

When RUNTIME MODE is NORMAL, answer ordinary questions yourself and use file tools as
needed for normal coding work. Explore and Plan are optional helpers. Never call
Execute unless the runtime explicitly says that a plan has been approved.

## PLAN Mode: Non-Negotiable Restrictions

When RUNTIME MODE is PLAN, the user does not want implementation yet.

- Do not modify project files, configuration, dependencies, or repository state.
- Do not perform implementation work or call `execute_plan`.
- You may read files and search the codebase.
- The exact active plan path supplied in the runtime message is the only file you may
  create or edit. This restriction is also enforced by the file tools.
- Writing the plan file is planning, not permission to implement the plan.

## PLAN Mode Phase 1: Understand and Explore

1. Identify the user's goal, constraints, expected behavior, and unresolved questions.
2. Inspect relevant code instead of inventing file contents, APIs, or architecture.
3. Search for existing functions, utilities, conventions, and tests that should be
   reused before proposing new abstractions.
4. Call `explore_codebase` when the affected area is unclear, the task spans multiple
   modules, important implementation details are missing, or a focused investigation
   would improve confidence. Give it a concrete search objective.
5. Skip Explore for greetings, conceptual questions, truly trivial changes, or when
   exact files and sufficient evidence are already available. Use the minimum number
   of Explore calls needed; usually one focused call is enough.

## PLAN Mode Phase 2: Design

1. Form an implementation approach from the original request and verified evidence.
2. You MUST call `plan_codebase` at least once before writing plan.md for every
   non-trivial implementation task. Creating a game, application, feature, service,
   library, integration, or multi-file project is always non-trivial for this rule,
   even when the domain is familiar and an initial approach seems obvious.
3. Do not replace the required Plan call with your own internal planning. Use Plan's
   independent pass to validate assumptions, expose edge cases, and improve the work
   sequence. Explore remains optional when sufficient evidence is already available.
4. When calling Plan, pass the current original implementation request in `task`.
   Do not use a greeting, planning instruction, exploration summary, or revision
   request as the original task.
5. Give Plan the important exploration findings, relevant paths, requirements,
   constraints, and the specific design question in `request`.
6. Skip Plan only for greetings, conceptual answers requiring no project change, or
   a truly trivial edit such as a typo, an obvious one-line change, or a simple
   rename. If the task warrants writing a multi-step plan.md, it warrants calling
   Plan first. Its output is a candidate plan; you still own final review and saving.

## PLAN Mode Phase 3: Main-Agent Review

You must personally review the proposed approach before saving it.

1. Check every step against the original request and latest feedback.
2. Verify critical claims by reading relevant files when necessary.
3. Remove unnecessary work, correct inaccurate assumptions, and include missing edge
   cases, compatibility concerns, and validation work.
4. Ensure the approach reuses project patterns and identifies concrete files or
   components expected to change.
5. Ask the user about important ambiguity instead of silently making a large product
   or architecture decision.

## PLAN Mode Phase 4: Write the Final Plan

After review, you—not a subagent—must write the complete final plan to the exact
active plan path using `write_file` or `edit_file`.

The final plan should:

- state the goal and important constraints;
- clearly distinguish implementation work from background, requirements, constraints,
  and acceptance criteria;
- use a concise ordered implementation sequence that another agent can follow;
- name relevant files, components, and reusable functions where known;
- describe expected behavior rather than merely saying "make changes";
- include tests or other concrete verification;
- contain only work that remains to be done.

Never present an unsaved Plan-subagent response as the final plan.

## PLAN Mode Phase 5: Wait for Approval

After saving the reviewed plan, summarize it and tell the user to use `/approve` to
execute it or provide feedback to revise it. Do not begin implementation and do not
call `execute_plan` while still in PLAN mode.

## PLAN Mode: Revising an Existing Plan

When the runtime includes an existing plan and the user asks for changes:

1. Treat the latest message as feedback on that plan, not as an unrelated new task.
2. Preserve correct parts and make focused incremental edits for small requests.
3. For a substantial redesign, changed scope, or newly discovered uncertainty, call
   Explore and/or Plan again as needed.
4. Personally review the revision against both the original task and latest feedback,
   then update the active plan file.
5. Present the revised saved plan and wait for approval again.

## PLAN_APPROVED Mode: Mandatory Execution

The RUNTIME MODE in the latest user message is authoritative and replaces every
earlier mode marker in conversation history. When it is PLAN_APPROVED, the program
has already left PLAN Mode and the user has explicitly approved the displayed plan.
Never claim that the runtime is still PLAN merely because earlier messages contain
that marker. Approval is equally valid whether the saved plan was proposed by the
Plan subagent or composed directly by the main agent.

Read the complete approved plan and semantically derive a concise, complete,
ordered list containing only executable tasks. Requirements, context, constraints,
file inventories, and acceptance criteria are supporting information, not separate
tasks. You MUST call `execute_plan` with that list in its `steps` argument. Do not
implement the plan directly with `write_file` or `edit_file`, and do not silently
replace the approved plan. After the Execute tool returns, use its complete result to
give the user the final answer.

Always ask before destructive actions outside an already approved plan.""" + (
    SHARED_FILE_INSPECTION_PROMPT
)


@dataclass
class SessionState:
    history: list[str] = field(default_factory=list)
    running: bool = True


class LangChainAgent:
    def __init__(
        self,
        config: AgentConfig | None = None,
        graph: Any | None = None,
        verbose: bool = False,
        session_id: str = "cli-session",
        plan_file_path: Path | None = None,
        working_directory: Path | None = None,
    ) -> None:
        self.config = config or AgentConfig.from_env()
        self.verbose = verbose
        requested_working_directory = working_directory or Path.cwd()
        self.working_directory = requested_working_directory.expanduser().resolve()
        if not self.working_directory.is_dir():
            raise ValueError(
                f"Working directory does not exist or is not a directory: "
                f"{self.working_directory}"
            )
        self.thread_id = f"{session_id}:main"
        self.approval_thread_id = f"{session_id}:approval"
        self.explore_thread_id = f"{session_id}:explore"
        self.plan_thread_id = f"{session_id}:plan"
        self.execution_thread_id = f"{session_id}:execution"

        self.plan_mode = False
        self.pending_plan: str | None = None
        self.approved_plan: str | None = None
        self.current_task: str | None = None
        self.plan_file_path = (
            plan_file_path or self.working_directory / "plan.md"
        ).expanduser().resolve()

        self._execution_authorized = False
        self._execution_called = False
        self._checkpointer: MemorySaver | None = None
        self.model: ChatOpenAI | None = None

        self.main_graph = graph
        self.explore_graphs: dict[Path, Any] = {}
        self.plan_graphs: dict[Path, Any] = {}
        self.execution_graphs: dict[Path, Any] = {}
        self.main_tools: list[Any] | None = None

        if self.config.model:
            self._checkpointer = MemorySaver()
            self.model = ChatOpenAI(
                model=self.config.model,
                api_key=self.config.api_key or "not-needed",
                base_url=self.config.base_url,
                timeout=self.config.timeout_seconds,
                max_retries=2,
            )

        # The main agent is the persistent coordinator for the whole session.
        # Explore, Plan, and Execute remain lazy because they are optional tools.
        self._ensure_main_graph()

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message, flush=True)

    def _build_agent(self, system_prompt: str, tools: list[Any]) -> Any:
        return create_agent(
            model=self.model,
            tools=tools,
            system_prompt=system_prompt,
            state_schema=AgentState,
            checkpointer=self._checkpointer,
        )

    def _get_main_tools(self) -> list[Any]:
        if self.main_tools is None:
            file_tools = make_main_file_tools(
                can_write=self._main_agent_can_write,
                on_write=self._on_main_agent_write,
            )
            self.main_tools = [
                *file_tools,
                make_explore_tool(self._run_explore),
                make_plan_tool(self._run_plan),
                make_execute_tool(self._run_approved_execution),
            ]
        return self.main_tools

    def _ensure_main_graph(self) -> None:
        if self.main_graph is not None or self.model is None:
            return
        self._log("[main] creating main agent...")
        prompt = prompt_with_working_directory(
            MAIN_AGENT_PROMPT,
            str(self.working_directory),
        )
        self.main_graph = self._build_agent(prompt, self._get_main_tools())
        self._log("[main] main agent created with Explore, Plan, and Execute tools.")

    def _subagent_graph(
        self,
        graphs: dict[Path, Any],
        working_directory: Path,
        system_prompt: str,
        tools: list[Any],
        label: str,
    ) -> Any | None:
        graph = graphs.get(working_directory)
        if graph is not None or self.model is None:
            return graph
        self._log(
            f"[{label}] creating subagent for working directory "
            f"{working_directory}..."
        )
        graph = self._build_agent(
            prompt_with_working_directory(system_prompt, str(working_directory)),
            tools,
        )
        graphs[working_directory] = graph
        return graph

    def _working_directory_for_subagent(self, value: str | None) -> Path:
        if value is None or not value.strip():
            return self.working_directory
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            raise ValueError(
                "Subagent working_directory must be an absolute directory path."
            )
        resolved = candidate.resolve()
        if not resolved.is_dir():
            raise ValueError(
                f"Subagent working directory does not exist or is not a directory: "
                f"{resolved}"
            )
        return resolved

    @staticmethod
    def _thread_for_directory(base_thread_id: str, working_directory: Path) -> str:
        digest = hashlib.sha256(
            str(working_directory).casefold().encode("utf-8")
        ).hexdigest()[:12]
        return f"{base_thread_id}:{digest}"

    def _config_for(self, thread_id: str) -> dict[str, Any]:
        return {
            "recursion_limit": max(80, self.config.max_tool_turns * 2 + 2),
            "configurable": {"thread_id": thread_id},
        }

    def _invoke_agent(
        self,
        graph: Any,
        thread_id: str,
        prompt: str,
        label: str,
    ) -> str:
        if graph is None:
            self._log(f"[{label}] LLM is not configured.")
            return "LLM is not configured. Set SIMPLE_AGENT_MODEL and SIMPLE_AGENT_API_KEY."
        self._log(f"[{label}] invoking agent on thread {thread_id}...")
        self._log(f"[{label}] waiting for the first graph update...")
        started_at = time.monotonic()
        last_update_at = started_at
        try:
            if hasattr(graph, "stream"):
                for update in graph.stream(
                    {"messages": [HumanMessage(content=prompt)]},
                    config=self._config_for(thread_id),
                    stream_mode="updates",
                ):
                    now = time.monotonic()
                    self._log(
                        f"[{label}] graph update received after "
                        f"{now - last_update_at:.2f}s."
                    )
                    last_update_at = now
                    for step, data in update.items():
                        self._log(f"[{label}] graph node: {step}")
                        messages = data.get("messages", []) if isinstance(data, dict) else []
                        for message in messages:
                            if isinstance(message, AIMessage) and message.tool_calls:
                                for call in message.tool_calls:
                                    name = str(call.get("name", "unknown"))
                                    call_id = str(call.get("id", "unknown"))
                                    self._log(
                                        f"[{label}] tool call: {name} (id={call_id})"
                                    )
                                    self._log(
                                        f"[{label}] tool arguments:\n"
                                        f"{_format_debug_value(call.get('args', {}))}"
                                    )
                            elif isinstance(message, ToolMessage):
                                self._log(
                                    f"[{label}] tool result: {message.name or 'unknown'} "
                                    f"(id={message.tool_call_id or 'unknown'})"
                                )
                                self._log(
                                    f"[{label}] tool output:\n"
                                    f"{_format_debug_value(message.content)}"
                                )

                get_state = getattr(graph, "get_state", None)
                if callable(get_state):
                    snapshot = get_state(self._config_for(thread_id))
                    result_messages = list(snapshot.values.get("messages", []))
                else:
                    result_messages = []
            else:
                result = graph.invoke(
                    {"messages": [HumanMessage(content=prompt)]},
                    config=self._config_for(thread_id),
                )
                result_messages = list(result.get("messages", []))
        except Exception as exc:
            self._log(
                f"[{label}] agent failed after "
                f"{time.monotonic() - started_at:.2f}s: {exc}"
            )
            return f"LLM request failed: {exc}"

        self._log(
            f"[{label}] agent returned {len(result_messages)} messages in "
            f"{time.monotonic() - started_at:.2f}s."
        )
        return _message_text(result_messages[-1]) if result_messages else ""

    def _run_explore(
        self,
        request: str,
        working_directory: str | None = None,
    ) -> str:
        active_directory = self._working_directory_for_subagent(working_directory)
        graph = self._subagent_graph(
            self.explore_graphs,
            active_directory,
            EXPLORE_PROMPT,
            READ_ONLY_TOOLS,
            "explore",
        )
        task = self.current_task or "No task has been recorded."
        prompt = (
            f"Working directory:\n{active_directory}\n\n"
            f"Original user task:\n{task}\n\n"
            f"Main agent request:\n{request}"
        )
        return self._invoke_agent(
            graph,
            self._thread_for_directory(self.explore_thread_id, active_directory),
            prompt,
            label="explore",
        )

    def _run_plan(
        self,
        task: str,
        request: str,
        working_directory: str | None = None,
    ) -> str:
        normalized_task = task.strip()
        if not normalized_task:
            return "Plan denied: task must contain the original implementation request."
        if self.plan_mode and self.current_task is None:
            self.current_task = normalized_task
            self._log(f"[plan] original task established: {self.current_task}")

        active_directory = self._working_directory_for_subagent(working_directory)
        graph = self._subagent_graph(
            self.plan_graphs,
            active_directory,
            PLAN_PROMPT,
            READ_ONLY_TOOLS,
            "plan",
        )
        current_plan = self.pending_plan or "No plan has been written yet."
        effective_task = self.current_task or normalized_task
        prompt = (
            f"Working directory:\n{active_directory}\n\n"
            f"Original user task:\n{effective_task}\n\n"
            f"Current plan:\n{current_plan}\n\n"
            f"Main agent planning request:\n{request}"
        )
        return self._invoke_agent(
            graph,
            self._thread_for_directory(self.plan_thread_id, active_directory),
            prompt,
            label="plan",
        )

    def _run_approved_execution(
        self,
        steps: list[str],
        working_directory: str | None = None,
    ) -> str:
        self._execution_called = True
        if not self._execution_authorized or self.approved_plan is None:
            return "Execution denied: the user has not approved a plan."

        normalized_steps = [str(step).strip() for step in steps if str(step).strip()]
        if not normalized_steps:
            return (
                "Execution denied: execute_plan requires at least one executable "
                "step derived from the approved plan."
            )

        active_directory = self._working_directory_for_subagent(working_directory)
        graph = self._subagent_graph(
            self.execution_graphs,
            active_directory,
            EXECUTION_AGENT_PROMPT,
            TOOLS,
            "execute",
        )
        if graph is None:
            return "LLM is not configured. Set SIMPLE_AGENT_MODEL and SIMPLE_AGENT_API_KEY."

        execution_thread_id = self._thread_for_directory(
            self.execution_thread_id,
            active_directory,
        )
        execution_workflow = ExecutionWorkflow(
            execution_agent=lambda prompt: self._invoke_agent(
                graph,
                execution_thread_id,
                f"Working directory:\n{active_directory}\n\n{prompt}",
                label="execute",
            ),
            on_event=self._log,
        )

        result = execution_workflow.invoke(
            task=self.current_task or "Approved plan execution",
            plan=normalized_steps,
            recursion_limit=max(80, self.config.max_tool_turns * 4),
        )
        response = str(result.get("response", "Task completed."))
        history = result.get("past_steps", [])
        completed = "\n\n".join(
            f"Step: {step}\nResult: {step_result}"
            for step, step_result in history
        ) or "No executable steps were run."

        if not response.startswith(("LLM request failed:", "LLM is not configured.")):
            self.pending_plan = None
            self._execution_authorized = False
        return (
            "Execute component finished.\n\n"
            f"Completed steps:\n{completed}\n\n"
            f"Final response:\n{response}"
        )

    def _main_agent_can_write(self, path: str) -> bool:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            return False
        if not self.plan_mode:
            return True
        return candidate.resolve() == self.plan_file_path.resolve()

    def _on_main_agent_write(self, path: str) -> None:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            return
        resolved = candidate.resolve()
        if not self.plan_mode or resolved != self.plan_file_path.resolve():
            return
        plan = resolved.read_text(encoding="utf-8")
        self.pending_plan = plan
        self._log(f"[plan] reviewed plan saved to {resolved}.")

    def enter_plan_mode(self) -> str:
        self.plan_mode = True
        self.pending_plan = None
        self.approved_plan = None
        self.current_task = None
        self._execution_authorized = False
        self._log("[plan] plan mode enabled.")
        return "Plan mode enabled. Describe your task, then use /approve when satisfied."

    def plan_task(self, prompt: str) -> str:
        if not self.plan_mode:
            self.enter_plan_mode()
        return self.respond(prompt)

    def approve_plan(self) -> str:
        if self.pending_plan is None:
            self._log("[approve] no pending plan.")
            return "No pending plan to approve. Ask the main agent to write plan.md first."
        if self.current_task is None:
            self._log("[approve] pending plan has no established original task.")
            return (
                "The plan has no established original task. Provide the original "
                "implementation request before approving it."
            )

        self.approved_plan = self.pending_plan
        self.plan_mode = False
        self._execution_authorized = True
        self._execution_called = False
        self._log("[approve] plan approved; main agent must call Execute.")

        result = self._invoke_main(
            self._approved_mode_prompt(),
            thread_id=self.approval_thread_id,
        )
        if not self._execution_called:
            self._log("[approve] warning: main agent did not call execute_plan.")
        return result

    def _approved_mode_prompt(self, user_message: str | None = None) -> str:
        follow_up = f"\n\nLatest user message:\n{user_message}" if user_message else ""
        return (
            "RUNTIME MODE: PLAN_APPROVED\n"
            f"Main working directory: {self.working_directory}\n"
            "THIS LATEST RUNTIME MODE IS AUTHORITATIVE AND REPLACES ALL EARLIER "
            "PLAN MODE MARKERS. The program has already switched out of PLAN Mode. "
            "Do not claim that approval is still required and do not ask the user "
            "to approve again. The saved plan is valid whether it was produced with "
            "the Plan subagent or directly reviewed and written by the main agent.\n\n"
            "The user explicitly approved the current plan. You MUST now call "
            "execute_plan with a complete ordered steps list derived semantically "
            "from the approved plan. Do not implement it directly. After that tool "
            "returns, report its result to the user.\n\n"
            f"Original task:\n{self.current_task or 'Unknown'}\n\n"
            f"Approved plan:\n{self.approved_plan}{follow_up}"
        )

    def reject_plan(self) -> str:
        if self.pending_plan is None:
            self._log("[reject] no pending plan.")
            return "No pending plan to revise."
        self.plan_mode = True
        self.approved_plan = None
        self._execution_authorized = False
        self._log("[plan] plan not approved; waiting for revision feedback.")
        return (
            "Plan was not approved. Describe what should change. The main agent will "
            "revise it directly or call the Plan subagent for a larger redesign."
        )

    def exit_plan_mode(self) -> str:
        self.pending_plan = None
        self.approved_plan = None
        self.current_task = None
        self.plan_mode = False
        self._execution_authorized = False
        self._log("[plan] plan mode exited.")
        return "Plan mode exited."

    def _invoke_main(self, prompt: str, thread_id: str | None = None) -> str:
        self._ensure_main_graph()
        self._log("[main] running main agent...")
        result = self._invoke_agent(
            self.main_graph,
            thread_id or self.thread_id,
            prompt,
            label="main",
        )
        self._log("[main] main agent finished.")
        return result

    def respond_normally(self, prompt: str) -> str:
        return self._invoke_main(
            "RUNTIME MODE: NORMAL\n"
            f"Main working directory: {self.working_directory}\n\n"
            f"User request:\n{prompt}"
        )

    def respond(self, prompt: str) -> str:
        command = prompt.strip().casefold()
        if command == "/approve":
            return self.approve_plan()
        if command == "/reject":
            return self.reject_plan()
        if command == "/exit-plan":
            return self.exit_plan_mode()
        if command == "/plan":
            if self.plan_mode:
                return "Plan mode is already enabled. Describe the task to plan."
            return self.enter_plan_mode()

        if not self.plan_mode:
            if self._execution_authorized:
                self._execution_called = False
                return self._invoke_main(
                    self._approved_mode_prompt(prompt),
                    thread_id=self.approval_thread_id,
                )
            if self.pending_plan is not None and not self._execution_authorized:
                return "A plan is pending. Use /approve or /reject."
            return self.respond_normally(prompt)

        normalized_prompt = prompt.strip()
        if self.pending_plan is None and normalized_prompt:
            previous_task = self.current_task
            self.current_task = normalized_prompt
            action = "initialized" if previous_task is None else "updated"
            self._log(f"[plan] current task {action}: {self.current_task}")

        current_plan = self.pending_plan or "No plan has been written yet."
        original_task = self.current_task or "Not established yet."
        return self._invoke_main(
            "RUNTIME MODE: PLAN\n"
            f"Main working directory: {self.working_directory}\n"
            f"Active plan path: {self.plan_file_path.resolve()}\n\n"
            f"Original task:\n{original_task}\n\n"
            f"Current plan:\n{current_plan}\n\n"
            f"Current user message:\n{prompt}\n\n"
            "Follow the Plan Mode process from your system instructions. Review the "
            "final plan yourself and save it to the exact active plan path."
        )


def _message_text(message: BaseMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    return "".join(
        str(block.get("text", ""))
        for block in message.content
        if isinstance(block, dict)
    )


def _format_debug_value(value: object) -> str:
    if isinstance(value, str):
        rendered = value
    else:
        try:
            rendered = json.dumps(
                value,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        except (TypeError, ValueError):
            rendered = repr(value)
    if len(rendered) <= MAX_DEBUG_VALUE_CHARS:
        return rendered
    omitted = len(rendered) - MAX_DEBUG_VALUE_CHARS
    return (
        f"{rendered[:MAX_DEBUG_VALUE_CHARS]}\n"
        f"... [truncated {omitted} characters]"
    )


def final_text(events: list[tuple[str, BaseMessage]]) -> str:
    for _step, message in reversed(events):
        if isinstance(message, AIMessage) and not message.tool_calls:
            return _message_text(message)
    return ""


def render_event(step: str, message: BaseMessage) -> str | None:
    if step == "error":
        return None
    if isinstance(message, AIMessage) and message.tool_calls:
        names = ", ".join(str(call.get("name", "unknown")) for call in message.tool_calls)
        return f"[工具调用] {names}"
    if isinstance(message, ToolMessage):
        return f"[工具结果] {message.name or 'unknown'}: {_message_text(message)}"
    return None
