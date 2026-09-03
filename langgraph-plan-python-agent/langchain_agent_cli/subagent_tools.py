from __future__ import annotations

from typing import Callable

from langchain.tools import tool


def make_explore_tool(run_explore: Callable[[str], str]):
    @tool(
        description=(
            "Launch the read-only Explore subagent to inspect the codebase and "
            "return relevant files, patterns, constraints, and implementation details. "
            "Use it when the main agent needs more codebase evidence."
        )
    )
    def explore_codebase(request: str) -> str:
        return run_explore(request)

    return explore_codebase


def make_plan_tool(run_plan: Callable[[str, str], str]):
    @tool(
        description=(
            "Launch the read-only Plan subagent for architectural analysis or a "
            "candidate implementation plan. It returns advice only and cannot write "
            "plan.md. For task, pass the exact original implementation request; this "
            "establishes the session's original task. For request, pass exploration "
            "findings, constraints, and the specific planning question. The main agent "
            "must review the result and write the final plan."
        )
    )
    def plan_codebase(task: str, request: str) -> str:
        return run_plan(task, request)

    return plan_codebase


def make_execute_tool(run_execution: Callable[[list[str]], str]):
    @tool(
        description=(
            "Execute the user-approved plan. This is the only approved-plan execution "
            "entry point and internally owns both implementation and replanning. Read "
            "the full approved plan, distinguish executable work from context and "
            "constraints, and pass the complete ordered executable task list in steps. "
            "Call it only after the user explicitly approves the plan."
        )
    )
    def execute_plan(steps: list[str]) -> str:
        return run_execution(steps)

    return execute_plan
