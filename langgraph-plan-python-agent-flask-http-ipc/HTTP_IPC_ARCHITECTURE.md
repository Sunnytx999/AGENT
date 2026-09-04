# HTTP IPC architecture

This is a separate variant of `langgraph-plan-python-agent-flask`. Agent
behavior, Plan Mode, the Web UI, and the public Flask API stay unchanged. The
native tool boundary is now process-isolated.

```text
native_host.py (parent Python process)
├── loads native/bin/agent_tools_http_ipc_v2.dll exactly once
├── exposes an authenticated 127.0.0.1 HTTP/JSON tool API
└── calls the DLL's agent_start_process function to create app.py

app.py (Flask child process)
├── serves http://127.0.0.1:5000
├── runs LangChain and LangGraph
└── calls native tools through HTTP/JSON; it never loads the DLL
```

The parent binds the native tool API to a random localhost port and generates
a random bearer token. The DLL-created Flask child inherits the endpoint and
token through `SIMPLE_AGENT_TOOL_URL` and `SIMPLE_AGENT_TOOL_TOKEN`.

## Build and run

```powershell
python -m pip install -r requirements.txt
powershell -ExecutionPolicy Bypass -File native\build.ps1
python native_host.py
```

Then open `http://127.0.0.1:5000`. Do not launch `app.py` directly.

After the DLL returns the child PID, the parent repeatedly requests
`GET /api/health`. It prints the Web address only after the response PID
matches the DLL-created child PID. A port occupied by another process therefore
does not produce a false ready message.

## Tool protocol

The Flask child sends:

```json
{
  "method": "read_file",
  "params": {
    "path": "E:\\project\\main.py",
    "offset": 1,
    "limit": 100
  }
}
```

The parent calls the DLL in its own process and returns:

```json
{
  "ok": true,
  "result": {
    "path": "E:\\project\\main.py",
    "content": "1: print('hello')"
  }
}
```

LangChain tool schemas and the Plan Mode write guard remain in Python. Only
the low-level time/file implementation executes inside the parent-loaded DLL.
