from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from app import create_app
from native_host import select_flask_port


class StubAgent:
    def __init__(self, verbose, session_id, plan_file_path, working_directory=None):
        self.plan_mode = False
        self.current_task = None
        self.pending_plan = None
        self.plan_file_path = Path(plan_file_path)
        self.working_directory = Path(working_directory or Path.cwd()).resolve()
        self._execution_authorized = False

    def respond(self, message):
        self.current_task = message
        return f"reply: {message}"

    def enter_plan_mode(self):
        self.plan_mode = True
        return "plan enabled"

    def approve_plan(self):
        return "approved"

    def reject_plan(self):
        return "rejected"

    def exit_plan_mode(self):
        self.plan_mode = False
        return "plan exited"


class FlaskAppTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        app = create_app(StubAgent, Path(self.temp.name))
        app.testing = True
        self.client = app.test_client()

    def tearDown(self):
        self.temp.cleanup()

    def test_session_plan_and_message_flow(self):
        created = self.client.post("/api/sessions")
        self.assertEqual(created.status_code, 201)
        session_id = created.get_json()["session_id"]

        planned = self.client.post("/api/plan", json={"session_id": session_id})
        self.assertTrue(planned.get_json()["plan_mode"])

        replied = self.client.post(
            "/api/messages",
            json={"session_id": session_id, "message": "build a game"},
        )
        self.assertEqual(replied.get_json()["response"], "reply: build a game")
        self.assertEqual(replied.get_json()["current_task"], "build a game")

    def test_index_serves_web_interface(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"LangGraph Plan Agent", response.data)
        self.assertIn(b'id="messageForm"', response.data)
        self.assertIn(b'id="planButton"', response.data)
        self.assertIn(b'id="workingDirectoryInput"', response.data)

    def test_session_accepts_selected_working_directory(self):
        selected = Path(self.temp.name).resolve()
        response = self.client.post(
            "/api/sessions",
            json={"working_directory": str(selected)},
        )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.get_json()["working_directory"], str(selected))

    def test_session_rejects_invalid_working_directory(self):
        relative = self.client.post(
            "/api/sessions",
            json={"working_directory": "relative/path"},
        )
        missing = self.client.post(
            "/api/sessions",
            json={"working_directory": str(Path(self.temp.name) / "missing")},
        )

        self.assertEqual(relative.status_code, 400)
        self.assertEqual(missing.status_code, 400)

    def test_health_reports_the_serving_process(self):
        response = self.client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json(),
            {"ok": True, "process_id": os.getpid()},
        )

    def test_unconfigured_flask_port_is_selected_dynamically(self):
        previous = os.environ.pop("SIMPLE_AGENT_FLASK_PORT", None)
        try:
            port = select_flask_port()
        finally:
            if previous is not None:
                os.environ["SIMPLE_AGENT_FLASK_PORT"] = previous

        self.assertGreater(port, 0)
        self.assertLessEqual(port, 65535)

    def test_unknown_session_and_missing_message(self):
        missing = self.client.post("/api/messages", json={"session_id": "missing"})
        self.assertEqual(missing.status_code, 400)

        unknown = self.client.post(
            "/api/messages",
            json={"session_id": "missing", "message": "hello"},
        )
        self.assertEqual(unknown.status_code, 404)


if __name__ == "__main__":
    unittest.main()
