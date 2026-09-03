from __future__ import annotations

from typing import Annotated, Callable

from langchain.tools import tool


def make_explore_tool(run_explore: Callable[[str, str | None], str]):
    @tool(
        description=(
            "Launch the read-only Explore subagent to inspect the codebase and "
            "return relevant files, patterns, constraints, and implementation details. "
            "Use it when the main agent needs more codebase evidence. The optional "
            "working_directory overrides the main agent's working directory."
        )
    )
    def explore_codebase(
        request: Annotated[
            str,
            "Concrete read-only investigation for the Explore subagent.",
        ],
        working_directory: Annotated[
            str | None,
            "Optional absolute directory for the subagent. Omit it to inherit the main agent's working directory.",
        ] = None,
    ) -> str:
        return run_explore(request, working_directory)

    return explore_codebase


def make_plan_tool(run_plan: Callable[[str, str, str | None], str]):
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
    def plan_codebase(
        task: Annotated[
            str,
            "Exact original implementation request being planned.",
        ],
        request: Annotated[
            str,
            "Planning question with relevant evidence, paths, constraints, and requested changes.",
        ],
        working_directory: Annotated[
            str | None,
            "Optional absolute directory for the subagent. Omit it to inherit the main agent's working directory.",
        ] = None,
    ) -> str:
        return run_plan(task, request, working_directory)

    return plan_codebase


def make_execute_tool(run_execution: Callable[[list[str], str | None], str]):
    @tool(
        description=(
            "Execute the user-approved plan. This is the only approved-plan execution "
            "entry point and internally owns both implementation and replanning. Read "
            "the full approved plan, distinguish executable work from context and "
            "constraints, and pass the complete ordered executable task list in steps. "
            "Call it only after the user explicitly approves the plan."
        )
    )
    def execute_plan(
        steps: Annotated[
            list[str],
            "Complete ordered list of executable steps derived from the approved plan.",
        ],
        working_directory: Annotated[
            str | None,
            "Optional absolute directory in which to execute the plan. Omit it to inherit the main agent's working directory.",
        ] = None,
    ) -> str:
        return run_execution(steps, working_directory)

    return execute_plan
