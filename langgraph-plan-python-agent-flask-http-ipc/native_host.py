from __future__ import annotations

import ctypes
import hmac
import json
import os
import secrets
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from langchain_agent_cli.direct_native import DirectNativeTools


MAX_REQUEST_BYTES = 64 * 1024 * 1024


def _dispatch(tools: DirectNativeTools, method: str, params: dict[str, Any]) -> dict[str, Any]:
    if method == "get_local_time":
        return tools.get_local_time()
    if method == "read_file":
        return tools.read_file(str(params.get("path", "")), int(params.get("offset", 1)), int(params.get("limit", 500)))
    if method == "list_files":
        return tools.list_files(str(params.get("pattern", "")), int(params.get("max_results", 200)))
    if method == "search_text":
        return tools.search_text(str(params.get("pattern", "")), str(params.get("glob", "")), bool(params.get("case_sensitive", True)), int(params.get("max_results", 200)))
    if method == "write_file":
        return tools.write_file(str(params.get("path", "")), str(params.get("content", "")))
    if method == "edit_file":
        return tools.edit_file(str(params.get("path", "")), str(params.get("old_string", "")), str(params.get("new_string", "")), bool(params.get("replace_all", False)))
    raise ValueError(f"Unknown native tool method: {method}")


def create_tool_server(
    tools: DirectNativeTools,
    token: str,
    host: str = "127.0.0.1",
    port: int = 0,
) -> ThreadingHTTPServer:
    class ToolRequestHandler(BaseHTTPRequestHandler):
        server_version = "AgentNativeToolHost/1.0"

        def _write_json(self, status: int, payload: dict[str, object]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self) -> bool:
            return hmac.compare_digest(
                self.headers.get("Authorization", ""), f"Bearer {token}"
            )

        def do_GET(self) -> None:
            if self.path != "/health":
                self._write_json(404, {"ok": False, "error": "Not found"})
            elif not self._authorized():
                self._write_json(401, {"ok": False, "error": "Unauthorized"})
            else:
                self._write_json(200, {"ok": True, "process_id": os.getpid()})

        def do_POST(self) -> None:
            if self.path != "/tools/call":
                self._write_json(404, {"ok": False, "error": "Not found"})
                return
            if not self._authorized():
                self._write_json(401, {"ok": False, "error": "Unauthorized"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 1 or length > MAX_REQUEST_BYTES:
                    raise ValueError("Invalid Content-Length")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("Request body must be a JSON object")
                method = str(payload.get("method", "")).strip()
                params = payload.get("params", {})
                if not method or not isinstance(params, dict):
                    raise ValueError("method and object params are required")
                result = _dispatch(tools, method, params)
            except (ValueError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                self._write_json(400, {"ok": False, "error": str(exc)})
                return
            except Exception as exc:
                self._write_json(500, {"ok": False, "error": str(exc)})
                return
            self._write_json(200, {"ok": True, "result": result})

        def log_message(self, format: str, *args: object) -> None:
            print(f"[native-http] {self.address_string()} {format % args}", flush=True)

    return ThreadingHTTPServer((host, port), ToolRequestHandler)


def _wait_for_windows_process(process_id: int) -> int:
    synchronize = 0x00100000
    query_limited_information = 0x1000
    wait_timeout = 0x00000102
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    kernel32.WaitForSingleObject.restype = ctypes.c_ulong
    kernel32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    kernel32.GetExitCodeProcess.restype = ctypes.c_int
    kernel32.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    kernel32.TerminateProcess.restype = ctypes.c_int
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel32.OpenProcess(synchronize | query_limited_information, False, process_id)
    if not handle:
        raise OSError(ctypes.get_last_error(), "Unable to open Flask child process")
    try:
        while kernel32.WaitForSingleObject(handle, 500) == wait_timeout:
            pass
        exit_code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            raise OSError(ctypes.get_last_error(), "Unable to read Flask exit code")
        return int(exit_code.value)
    except KeyboardInterrupt:
        kernel32.TerminateProcess(handle, 130)
        kernel32.WaitForSingleObject(handle, 5000)
        return 130
    finally:
        kernel32.CloseHandle(handle)


def _terminate_windows_process(process_id: int) -> None:
    terminate = 0x0001
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    kernel32.TerminateProcess.restype = ctypes.c_int
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel32.OpenProcess(terminate, False, process_id)
    if not handle:
        return
    try:
        kernel32.TerminateProcess(handle, 1)
    finally:
        kernel32.CloseHandle(handle)


def wait_for_flask(
    host: str,
    port: int,
    child_pid: int,
    timeout: float = 15.0,
) -> None:
    """Wait until the expected Flask child answers its health endpoint."""
    url = f"http://{host}:{port}/api/health"
    deadline = time.monotonic() + timeout
    last_error = "no response"
    while time.monotonic() < deadline:
        try:
            request = Request(url, method="GET")
            with urlopen(request, timeout=1.0) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, dict) or payload.get("ok") is not True:
                last_error = "health endpoint returned an invalid response"
            elif int(payload.get("process_id", -1)) != child_pid:
                last_error = (
                    "health endpoint belongs to another process "
                    f"(expected PID {child_pid}, got {payload.get('process_id')})"
                )
            else:
                return
        except (HTTPError, URLError, TimeoutError, ValueError, TypeError,
                UnicodeDecodeError, json.JSONDecodeError) as exc:
            last_error = str(exc)
        time.sleep(0.2)
    raise RuntimeError(
        f"Flask child PID {child_pid} did not become ready at {url} "
        f"within {timeout:.1f}s: {last_error}"
    )


def main() -> int:
    if sys.platform != "win32":
        raise RuntimeError("This launcher currently requires Windows.")
    project_root = Path(__file__).resolve().parent
    tools = DirectNativeTools()
    token = secrets.token_urlsafe(32)
    server = create_tool_server(tools, token)
    server_thread = Thread(target=server.serve_forever, name="native-http", daemon=True)
    server_thread.start()
    host, port = server.server_address
    endpoint = f"http://{host}:{port}"
    os.environ["SIMPLE_AGENT_TOOL_URL"] = endpoint
    os.environ["SIMPLE_AGENT_TOOL_TOKEN"] = token
    os.environ["SIMPLE_AGENT_FLASK_CHILD"] = "1"
    os.environ["PYTHONUNBUFFERED"] = "1"
    print(f"[native-host] PID {os.getpid()} loaded {tools.path}", flush=True)
    print(f"[native-host] tool API listening on {endpoint}", flush=True)
    started = tools.start_process(str(Path(sys.executable).resolve()), str((project_root / "app.py").resolve()), str(project_root))
    if started.get("error"):
        server.shutdown()
        server.server_close()
        raise RuntimeError(str(started["error"]))
    child_pid = int(started["process_id"])
    print(f"[native-host] Flask child started with PID {child_pid}", flush=True)
    flask_port = int(os.getenv("SIMPLE_AGENT_FLASK_PORT", "5000"))
    try:
        wait_for_flask("127.0.0.1", flask_port, child_pid)
        print(
            f"[native-host] Flask child PID {child_pid} is ready",
            flush=True,
        )
        print(f"[native-host] open http://127.0.0.1:{flask_port}", flush=True)
        return _wait_for_windows_process(child_pid)
    except (Exception, KeyboardInterrupt):
        _terminate_windows_process(child_pid)
        raise
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)
        print("[native-host] tool API stopped", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
