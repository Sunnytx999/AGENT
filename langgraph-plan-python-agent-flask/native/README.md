# Native C++ tools

`agent_tools` implements the parts of the project that do not depend on an
agent runtime:

- local date and time
- UTF-8 file reading
- glob-based file listing
- regular-expression text search
- UTF-8 file creation/overwrite
- exact string replacement

All file paths and file-selection globs passed across the native ABI must be
absolute. The library never reads or derives the process working directory;
the model receives the active working directory in its agent prompt and must
construct absolute tool arguments from it.

The library exposes a stable C ABI from `include/agent_tools.h`. Python loads it
with `ctypes` in `langchain_agent_cli/cpp_bridge.py`; LangChain tool schemas and
agent-state checks remain in `langchain_agent_cli/tools.py`.

Explore, Plan, and Execute remain Python tools because they invoke LangGraph and
depend on the active Agent session. The Plan Mode write guard also remains in
Python, while its allowed write/edit operation is performed by this DLL.

## Build on Windows

The project includes a prebuilt 64-bit `bin/agent_tools_v2.dll`. To rebuild it,
install either CMake plus Visual Studio Build Tools, or Zig:

```powershell
python -m pip install ziglang
powershell -ExecutionPolicy Bypass -File native\build.ps1
```

By default the bridge loads `native/bin/agent_tools_v2.dll`. Set
`SIMPLE_AGENT_TOOLS_LIBRARY` to load a library from another location.
