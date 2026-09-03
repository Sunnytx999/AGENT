from __future__ import annotations

import argparse

from .agent import LangChainAgent, SessionState


def handle_command(text: str, state: SessionState, agent: LangChainAgent | None = None) -> str | None:
    command = text.strip().lower()
    if not command.startswith("/"):
        return None

    if command == "/help":
        return (
            "Available commands: /help, /history, /clear, /plan, "
            "/approve, /reject, /exitplan, /todo, /exit"
        )
    if command == "/history":
        return "\n".join(f"{i}. {item}" for i, item in enumerate(state.history, 1)) or "History is empty."
    if command == "/clear":
        state.history.clear()
        return "History cleared."
    if command == "/plan":
        if agent is None:
            return "Plan mode is not available in this context."
        agent.plan_mode = True
        return "Plan mode enabled. Describe your task, then use /approve when satisfied."
    if command == "/approve":
        if agent is None:
            return "Plan approval is not available in this context."
        return agent.approve_plan()
    if command == "/reject":
        if agent is None:
            return "Plan rejection is not available in this context."
        return agent.reject_plan()
    if command == "/exitplan":
        if agent is None:
            return "Plan mode is not available in this context."
        return agent.exit_plan_mode()
    if command == "/todo":
        if agent is None:
            return "Todo is not available in this context."
        return agent.show_todos()
    if command in {"/exit", "/quit"}:
        state.running = False
        return "Bye."
    return f"Unknown command: {text.strip()}. Try /help."


def run_prompt(agent: LangChainAgent, prompt: str) -> str:
    output = agent.respond(prompt)
    if output:
        print(f"\n[最终结果]\n{output}")
    return output


def run_interactive() -> None:
    agent = LangChainAgent(verbose=True)
    state = SessionState()
    print("LangChain Python Agent")
    print("Type /help for commands, /plan to enter plan mode, /exit to quit.")

    while state.running:
        prefix = "[plan] " if agent.plan_mode else ""
        try:
            prompt = input(f"{prefix}> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not prompt.strip():
            continue

        command_result = handle_command(prompt, state, agent)
        if command_result is not None:
            print(command_result)
            continue

        state.history.append(prompt)
        run_prompt(agent, prompt)


def main() -> None:
    parser = argparse.ArgumentParser(description="A LangChain-based coding agent CLI.")
    parser.add_argument("-p", "--print", dest="prompt", help="Run one prompt and exit.")
    args = parser.parse_args()

    if args.prompt is None:
        run_interactive()
        return

    output = run_prompt(LangChainAgent(verbose=True), args.prompt)
    if output:
        print(output)


if __name__ == "__main__":
    main()


