from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Callable
from uuid import uuid4

from flask import Flask, jsonify, render_template, request

from langchain_agent_cli import LangChainAgent


@dataclass
class AgentSession:
    agent: LangChainAgent
    lock: Lock = field(default_factory=Lock)


class SessionManager:
    def __init__(
        self,
        session_root: Path,
        agent_factory: Callable[..., LangChainAgent] = LangChainAgent,
    ) -> None:
        self._session_root = session_root.resolve()
        self._agent_factory = agent_factory
        self._sessions: dict[str, AgentSession] = {}
        self._lock = Lock()

    def create(self) -> tuple[str, AgentSession]:
        session_id = uuid4().hex
        plan_path = self._session_root / session_id / "plan.md"
        session = AgentSession(
            self._agent_factory(
                verbose=True,
                session_id=session_id,
                plan_file_path=plan_path,
            )
        )
        with self._lock:
            self._sessions[session_id] = session
        return session_id, session

    def get(self, session_id: str) -> AgentSession | None:
        with self._lock:
            return self._sessions.get(session_id)


def create_app(
    agent_factory: Callable[..., LangChainAgent] = LangChainAgent,
    session_root: Path | None = None,
) -> Flask:
    app = Flask(__name__)
    app.json.ensure_ascii = False
    root = session_root or Path(__file__).resolve().parent / ".sessions"
    sessions = SessionManager(root, agent_factory)

    def require_session() -> tuple[str, AgentSession] | tuple[None, None]:
        payload = request.get_json(silent=True) or {}
        session_id = str(payload.get("session_id", "")).strip()
        return session_id, sessions.get(session_id)

    def run_session_action(action: Callable[[LangChainAgent], str]):
        session_id, session = require_session()
        if session is None:
            return jsonify({"error": "Unknown or missing session_id."}), 404
        with session.lock:
            response = action(session.agent)
            return jsonify(_snapshot(session_id, session.agent, response))

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/health")
    def health():
        return jsonify({"ok": True, "process_id": os.getpid()})

    @app.post("/api/sessions")
    def create_session():
        session_id, session = sessions.create()
        return jsonify(_snapshot(session_id, session.agent)), 201

    @app.post("/api/messages")
    def send_message():
        payload = request.get_json(silent=True) or {}
        message = str(payload.get("message", "")).strip()
        if not message:
            return jsonify({"error": "message is required."}), 400
        return run_session_action(lambda agent: agent.respond(message))

    @app.post("/api/plan")
    def enter_plan():
        return run_session_action(lambda agent: agent.enter_plan_mode())

    @app.post("/api/approve")
    def approve_plan():
        return run_session_action(lambda agent: agent.approve_plan())

    @app.post("/api/reject")
    def reject_plan():
        return run_session_action(lambda agent: agent.reject_plan())

    @app.post("/api/exit-plan")
    def exit_plan():
        return run_session_action(lambda agent: agent.exit_plan_mode())

    @app.get("/api/sessions/<session_id>")
    def get_session(session_id: str):
        session = sessions.get(session_id)
        if session is None:
            return jsonify({"error": "Unknown session_id."}), 404
        with session.lock:
            return jsonify(_snapshot(session_id, session.agent))

    return app


def _snapshot(
    session_id: str,
    agent: LangChainAgent,
    response: str | None = None,
) -> dict[str, object]:
    return {
        "session_id": session_id,
        "response": response,
        "plan_mode": agent.plan_mode,
        "current_task": agent.current_task,
        "pending_plan": agent.pending_plan,
        "plan_file": str(agent.plan_file_path),
        "working_directory": str(agent.working_directory),
        "execution_authorized": agent._execution_authorized,
    }


app = create_app()


if __name__ == "__main__":
    if os.getenv("SIMPLE_AGENT_FLASK_CHILD") != "1":
        raise RuntimeError(
            "Do not start app.py directly. Start the parent process with: "
            "python native_host.py"
        )
    app.run(
        host="127.0.0.1",
        port=int(os.getenv("SIMPLE_AGENT_FLASK_PORT", "5000")),
        debug=False,
    )
