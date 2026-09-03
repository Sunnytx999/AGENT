from __future__ import annotations

import json
import operator
from typing import Annotated, Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from .shared_prompt import SHARED_FILE_INSPECTION_PROMPT


EXECUTION_AGENT_PROMPT = (
    "You are the execution component supporting a main coding agent. You handle "
    "both implementation and replanning for an approved plan. For an execution "
    "prompt, use tools to complete only the requested current step and report its "
    "concrete result. For a replanning prompt, do not modify files; return only the "
    "requested JSON containing the complete revised remaining plan or final response."
) + SHARED_FILE_INSPECTION_PROMPT


class ExecutionState(TypedDict, total=False):
    """State for the approved plan's execute-and-replan loop."""

    task: str
    plan: list[str]
    past_steps: Annotated[list[tuple[str, str]], operator.add]
    response: str


class ExecutionWorkflow:
    """Own the complete execute-and-replan component for an approved plan.

    One execution agent handles both implementation and replanning prompts. From
    the main agent's perspective this entire loop is one execute_plan tool call.
    """

    def __init__(
        self,
        execution_agent: Callable[[str], str],
        on_event: Callable[[str], None] | None = None,
    ) -> None:
        self._execution_agent = execution_agent
        self._on_event = on_event or (lambda _message: None)

        # One invoke() may loop through these two nodes many times.
        builder = StateGraph(ExecutionState)
        builder.add_node("execute", self._execute_node)
        builder.add_node("replan", self._replan_node)
        builder.add_edge(START, "execute")
        builder.add_edge("execute", "replan")
        builder.add_conditional_edges(
            "replan",
            self._route_after_replan,
            {"execute": "execute", "end": END},
        )
        self.graph = builder.compile()

    def _execute_node(self, state: ExecutionState) -> dict[str, object]:
        """Execute only the first remaining step and append its result."""
        current_step = state["plan"][0]
        result = self._execution_agent(
            _execution_prompt(
                task=state["task"],
                step=current_step,
                past_steps=state.get("past_steps", []),
            )
        )
        if _is_llm_failure(result):
            return {
                "plan": state["plan"],
                "past_steps": [(current_step, result)],
                "response": result,
            }
        return {
            "plan": state["plan"][1:],
            "past_steps": [(current_step, result)],
        }

    def _replan_node(self, state: ExecutionState) -> dict[str, object]:
        """Ask the replanner to update remaining steps or return a final answer."""
        if state.get("response"):
            return {"plan": []}
        current_plan = state.get("plan", [])
        raw = self._execution_agent(
            _replan_prompt(
                task=state["task"],
                remaining_steps=current_plan,
                past_steps=state.get("past_steps", []),
            )
        )
        if _is_llm_failure(raw):
            return {"plan": [], "response": raw}
        decision = _parse_replan_decision(
            raw=raw,
            remaining_steps=current_plan,
            past_steps=state.get("past_steps", []),
        )
        self._report_replan(current_plan, decision)
        update: dict[str, object] = {"plan": decision["steps"]}
        if decision.get("response"):
            update["response"] = decision["response"]
        return update

    def _report_replan(
        self,
        current_plan: list[str],
        decision: dict[str, object],
    ) -> None:
        revised_plan = decision["steps"]
        self._on_event(f"[replan] current remaining plan: {current_plan}")
        self._on_event(f"[replan] revised remaining plan: {revised_plan}")
        self._on_event(
            "[replan] plan changed: "
            + ("yes" if revised_plan != current_plan else "no")
        )
        if decision.get("response"):
            self._on_event(f"[replan] task complete: {decision['response']}")

    @staticmethod
    def _route_after_replan(state: ExecutionState) -> str:
        return "end" if state.get("response") or not state.get("plan") else "execute"

    def invoke(
        self,
        task: str,
        plan: list[str],
        recursion_limit: int = 50,
    ) -> ExecutionState:
        if not plan:
            return {
                "task": task,
                "plan": [],
                "past_steps": [],
                "response": "The approved plan contains no executable steps.",
            }
        return self.graph.invoke(
            {"task": task, "plan": plan, "past_steps": []},
            config={"recursion_limit": recursion_limit},
        )


def _execution_prompt(
    task: str,
    step: str,
    past_steps: list[tuple[str, str]],
) -> str:
    history = _format_history(past_steps) or "None"
    return (
        f"Original task:\n{task}\n\n"
        f"Completed steps:\n{history}\n\n"
        f"Execute only this current step:\n{step}\n\n"
        "Return the concrete result of this step."
    )


def _replan_prompt(
    task: str,
    remaining_steps: list[str],
    past_steps: list[tuple[str, str]],
) -> str:
    return (
        f"Original task:\n{task}\n\n"
        f"Completed steps and results:\n"
        f"{_format_history(past_steps)}\n\n"
        f"Current remaining plan:\n"
        f"{json.dumps(remaining_steps, ensure_ascii=False)}\n\n"

        "Review the execution progress and decide whether additional work is "
        "required to fully satisfy the original task.\n\n"

        "Return only one strict JSON object. Do not include Markdown fences, "
        "explanations, or any text outside the JSON object.\n\n"

        "Use exactly this schema:\n"
        '{"steps": ["step 1", "step 2"], "response": null}\n\n'

        "Rules:\n"
        "1. If more work is required, `steps` must contain the COMPLETE "
        "ordered list of ALL remaining executable steps.\n"
        "2. The returned `steps` array completely replaces the current "
        "remaining plan. Do not return only newly added steps, changed "
        "steps, or only the next step.\n"
        "3. If the current remaining plan is still correct and no changes "
        "are needed, return the entire current remaining plan unchanged.\n"
        "4. If changes are needed, return the entire revised remaining "
        "plan, including unchanged steps and changed or newly added steps.\n"
        "5. Do not include completed steps in `steps`.\n"
        "6. While any work remains, `response` must be null.\n\n"

        "Example when the current remaining plan is still correct:\n"
        "Current remaining plan:\n"
        '["implement login", "run login tests"]\n'
        "Return:\n"
        '{"steps": ["implement login", "run login tests"], '
        '"response": null}\n\n'

        "Example when the remaining plan must be revised:\n"
        "Current remaining plan:\n"
        '["implement login", "run login tests"]\n'
        "Suppose a database migration is now required before "
        "implementation.\n"
        "Return:\n"
        '{"steps": ["create database migration", "implement login", '
        '"run login tests"], "response": null}\n\n'

        "If the original task is fully complete and no additional work is "
        "required, return:\n"
        '{"steps": [], "response": "final result for the user"}\n\n'

        "Only return a completion response when the original task is "
        "actually satisfied. An empty current remaining plan does not by "
        "itself prove completion. Inspect the completed-step results and "
        "add any necessary follow-up steps when appropriate."
    )


def _format_history(past_steps: list[tuple[str, str]]) -> str:
    return "\n\n".join(
        f"Step: {step}\nResult: {result}"
        for step, result in past_steps
    )


def _parse_replan_decision(
    raw: str,
    remaining_steps: list[str],
    past_steps: list[tuple[str, str]],
) -> dict[str, object]:
    """Parse replanner JSON and fall back without losing unfinished work."""
    try:
        payload = json.loads(_strip_json_fence(raw))
        steps = [
            str(step)
            for step in payload.get("steps", [])
            if str(step).strip()
        ]
        response = payload.get("response")
        return {
            "steps": steps,
            "response": str(response) if response else None,
        }
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
        if remaining_steps:
            return {"steps": remaining_steps, "response": None}
        fallback = past_steps[-1][1] if past_steps else "Task completed."
        return {"steps": [], "response": fallback}


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _is_llm_failure(result: str) -> bool:
    return result.startswith("LLM request failed:") or result.startswith(
        "LLM is not configured."
    )
