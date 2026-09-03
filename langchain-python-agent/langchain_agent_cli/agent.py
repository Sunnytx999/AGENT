from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain.agents import AgentState, create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

from .config import AgentConfig
from .tools import TOOLS, TodoItem, make_todo_write_tool

SYSTEM_PROMPT = (
    "You are a helpful CLI coding assistant. Keep answers concise. "
    "Use get_local_time for the current time or date. Use the file tools to inspect "
    "local files and never invent file contents. Ask before making destructive changes."
)

CHAT_PROMPT = SYSTEM_PROMPT

EXPLORE_PROMPT = (
    "You are a fast read-only codebase explorer. "
    "Find relevant files, search for patterns, and report useful implementation details. "
    "Do not modify anything."
)

PLAN_PROMPT = (
    "You are a software architect. "
    "Given the original task and exploration results, design a concrete implementation plan. "
    "Include files to create or modify, reusable functions, and verification steps. "
    "Do not modify anything; only produce the plan."
)

EXECUTOR_PROMPT = (
    "You are an execution agent. "
    "Use the provided tools to implement the approved plan. "
    "Use todo_write to track tasks, mark tasks in_progress before working on them, "
    "mark them completed when done, and add new tasks when you discover follow-up work. "
    "Continue until the task is complete."
)

READ_ONLY_TOOL_NAMES = {
    "get_local_time",
    "read_file",
    "list_files",
    "search_text",
}

READ_ONLY_TOOLS = [
    candidate
    for candidate in TOOLS
    if getattr(candidate, "name", None) in READ_ONLY_TOOL_NAMES
]


class PlanDecision(BaseModel):
    needs_plan: bool


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
    ) -> None:
        self.config = config or AgentConfig.from_env()
        self.verbose = verbose
        self.thread_id = "cli-session"
        self.explore_thread_id = "explore-session"
        self.plan_thread_id = "plan-session"
        self.chat_thread_id = "chat-session"

        self.plan_mode = False
        self.pending_plan: str | None = None
        self.approved_plan: str | None = None
        self.current_task: str | None = None
        self.todos: list[TodoItem] = []
        self.plan_file_path = Path("plan.md")

        self._checkpointer: MemorySaver | None = None
        self.model: ChatOpenAI | None = None

        self.chat_graph = None
        self.explore_graph = None
        self.plan_graph = None
        self.exec_graph = graph

        if self.config.model:
            self._checkpointer = MemorySaver()
            self.model = ChatOpenAI(
                model=self.config.model,
                api_key=self.config.api_key or "not-needed",
                base_url=self.config.base_url,
                timeout=self.config.timeout_seconds,
                max_retries=2,
            )
            self.todo_write_tool = make_todo_write_tool(self._update_todos)

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message, flush=True)

    def _update_todos(self, todos: list[TodoItem]) -> None:
        self.todos = todos
        self._log("[todo_write] task list updated.")

    def _build_agent(self, system_prompt: str, tools: list[Any]) -> Any:
        return create_agent(
            model=self.model,
            tools=tools,
            system_prompt=system_prompt,
            state_schema=AgentState,
            checkpointer=self._checkpointer,
        )

    def _ensure_chat_graph(self) -> None:
        if self.chat_graph is not None or self.model is None:
            return
        self._log("[agent] creating lightweight chat agent...")
        self.chat_graph = self._build_agent(CHAT_PROMPT, [])
        self._log("[agent] chat agent created.")

    def _ensure_explore_graph(self) -> None:
        if self.explore_graph is not None or self.model is None:
            return
        self._log("[agent] creating Explore subagent...")
        self.explore_graph = self._build_agent(EXPLORE_PROMPT, READ_ONLY_TOOLS)
        self._log(
            "[agent] Explore subagent created with tools: "
            + ", ".join(getattr(tool, "name", "?") for tool in READ_ONLY_TOOLS)
        )

    def _ensure_plan_graph(self) -> None:
        if self.plan_graph is not None or self.model is None:
            return
        self._log("[agent] creating Plan subagent...")
        self.plan_graph = self._build_agent(PLAN_PROMPT, READ_ONLY_TOOLS)
        self._log("[agent] Plan subagent created.")

    def _ensure_exec_graph(self) -> None:
        if self.exec_graph is not None or self.model is None:
            return
        self._log("[agent] creating Executor subagent...")
        self.exec_graph = self._build_agent(
            EXECUTOR_PROMPT,
            [*TOOLS, self.todo_write_tool],
        )
        self._log(
            "[agent] Executor subagent created with todo_write and all file tools."
        )

    def _should_plan(self, prompt: str) -> bool:
        if self.model is None:
            return False

        self._log("[plan] asking model whether this task needs full planning...")
        try:
            response = self.model.invoke([
                SystemMessage(
                    content=(
                        "Decide whether this request needs the full plan workflow with codebase exploration, "
                        "architecture planning, plan-file creation, approval, and todo-tracked execution. "
                        "Reply with only true or false. "
                        "False is for greetings, simple questions, explanations, or trivial requests. "
                        "True is for multi-step coding, refactoring, feature implementation, or ambiguous implementation tasks."
                    )
                ),
                HumanMessage(content=prompt),
            ])
            text = str(response.content).strip().lower()
            result = text.startswith("true") or "true" in text.splitlines()[0] if text else False
        except Exception as exc:
            self._log(f"[plan] planning decision failed, treating as simple request: {exc}")
            result = False

        self._log(f"[plan] planning decision: needs_plan={result}")
        return result

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
        self._log(f"[{label}] invoking subagent on thread {thread_id}...")
        try:
            if hasattr(graph, "stream"):
                for update in graph.stream(
                    {"messages": [HumanMessage(content=prompt)]},
                    config=self._config_for(thread_id),
                    stream_mode="updates",
                ):
                    for step, data in update.items():
                        step_messages = data.get("messages", []) if isinstance(data, dict) else []
                        for message in step_messages:
                            if isinstance(message, AIMessage) and message.tool_calls:
                                names = ", ".join(
                                    str(call.get("name", "unknown"))
                                    for call in message.tool_calls
                                )
                                self._log(f"[{label}] model requested tools: {names}")
                            elif isinstance(message, ToolMessage):
                                self._log(f"[{label}] tool result: {message.name or 'unknown'}")
                            else:
                                preview = _message_text(message)
                                preview = preview.replace("\n", " ").strip()
                                if len(preview) > 120:
                                    preview = preview[:120] + "..."
                                self._log(
                                    f"[{label}] message received: "
                                    f"{type(message).__name__}: {preview}"
                                )

                get_state = getattr(graph, "get_state", None)
                if callable(get_state):
                    snapshot = get_state(self._config_for(thread_id))
                    messages = list(snapshot.values.get("messages", []))
                else:
                    messages = []
            else:
                result = graph.invoke(
                    {"messages": [HumanMessage(content=prompt)]},
                    config=self._config_for(thread_id),
                )
                messages = list(result.get("messages", []))
        except Exception as exc:
            self._log(f"[{label}] subagent failed: {exc}")
            return f"LLM request failed: {exc}"

        self._log(f"[{label}] subagent returned {len(messages)} messages.")
        return _message_text(messages[-1]) if messages else ""

    def plan_task(self, prompt: str) -> str:
        self._ensure_explore_graph()
        self._ensure_plan_graph()

        if self.explore_graph is None or self.plan_graph is None:
            self._log("[plan] Explore or Plan subagent is not configured.")
            return "LLM is not configured. Set SIMPLE_AGENT_MODEL and SIMPLE_AGENT_API_KEY."

        self.current_task = prompt

        self._log("[plan] running Explore subagent...")
        exploration = self._invoke_agent(
            self.explore_graph,
            self.explore_thread_id,
            (
                "Explore the current codebase for this task. "
                "Read relevant files, search for existing patterns, and report useful details.\n\n"
                f"Task: {prompt}"
            ),
            label="explore",
        )
        self._log("[plan] Explore subagent finished.")

        self._log("[plan] running Plan subagent...")
        plan = self._invoke_agent(
            self.plan_graph,
            self.plan_thread_id,
            (
                "Create an implementation plan for the task below.\n\n"
                f"Task:\n{prompt}\n\n"
                f"Exploration result:\n{exploration}\n\n"
                "Return the plan only. Do not modify files."
            ),
            label="plan",
        )
        self._log("[plan] Plan subagent finished.")

        self.pending_plan = plan
        self._log(f"[plan] writing plan file to {self.plan_file_path}...")
        self.plan_file_path.write_text(plan, encoding="utf-8")
        self._log("[plan] plan file written.")
        return f"Plan written to {self.plan_file_path}\n\n{plan}"

    def approve_plan(self) -> str:
        if self.pending_plan is None:
            self._log("[approve] no pending plan.")
            return "No pending plan to approve."

        plan = self.pending_plan
        self.pending_plan = None
        self.approved_plan = plan
        self.plan_mode = False

        self._ensure_exec_graph()
        self._log("[approve] plan approved. switching to execution.")
        execution_prompt = (
            "You have an approved implementation plan:\n\n"
            f"{plan}\n\n"
            "Start by creating a todo list with todo_write, then execute the plan "
            "step by step. Update todo_write after each step. If you discover a "
            "blocker or new task, update the todo list. Continue until the work "
            "is complete, then return a final summary."
        )
        self._log("[executor] running executor subagent...")
        result = self._invoke_agent(
            self.exec_graph,
            self.thread_id,
            execution_prompt,
            label="executor",
        )
        self._log("[executor] execution finished.")
        return result

    def reject_plan(self) -> str:
        if self.pending_plan is None:
            self._log("[reject] no pending plan.")
            return "No pending plan to reject."
        self.pending_plan = None
        self.plan_mode = True
        self._log("[reject] pending plan cleared.")
        return "Plan rejected. You can provide more details or ask me to plan again."

    def exit_plan_mode(self) -> str:
        self.pending_plan = None
        self.plan_mode = False
        self._log("[plan] plan mode exited.")
        return "Plan mode exited."

    def execute_prompt(self, prompt: str) -> str:
        self._ensure_exec_graph()
        self._log("[executor] running executor subagent...")
        result = self._invoke_agent(
            self.exec_graph,
            self.thread_id,
            prompt,
            label="executor",
        )
        self._log("[executor] execution finished.")
        return result

    def respond(self, prompt: str) -> str:
        if self.plan_mode:
            if self._should_plan(prompt):
                return self.plan_task(prompt)

            self._ensure_chat_graph()
            self._log("[chat] simple plan-mode request, answering directly.")
            return self._invoke_agent(
                self.chat_graph,
                self.chat_thread_id,
                prompt,
                label="chat",
            )

        if self.pending_plan is not None:
            self._log("[agent] plan is pending approval.")
            return (
                "A plan is pending approval. "
                "Use /approve to execute it or /reject to discard it."
            )
        return self.execute_prompt(prompt)

    def show_todos(self) -> str:
        if not self.todos:
            return "Todo list is empty."
        lines = []
        for index, todo in enumerate(self.todos, 1):
            status = todo.status
            content = todo.content
            active_form = todo.activeForm
            line = f"{index}. [{status}] {content}"
            if active_form:
                line += f" | {active_form}"
            lines.append(line)
        return "\n".join(lines)


def _message_text(message: BaseMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    return "".join(
        str(block.get("text", ""))
        for block in message.content
        if isinstance(block, dict)
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









