# Simple Python Agent

This is a tiny learning project that shows the basic shape of a coding-agent CLI.

## Files

- `simple_agent_cli/cli.py`: command-line entry, decides interactive vs non-interactive mode.
- `simple_agent_cli/agent.py`: public Agent interface, state, slash-command handling.
- `simple_agent_cli/query.py`: query transaction and model/tool loop.
- `simple_agent_cli/messages.py`: API context messages plus the
  `StreamEvent | AssistantMessage | SystemAPIErrorMessage` event union.
- `simple_agent_cli/model/client.py`: OpenAI-compatible HTTP model client.
- `simple_agent_cli/model/types.py`: model-facing message and tool types.
- `simple_agent_cli/tools/definitions.py`: tool schemas sent to the model.
- `simple_agent_cli/tools/registry.py`: built-in tool handlers and registration.
- `simple_agent_cli/tools/executor.py`: tool-call parsing and execution.
- `simple_agent_cli/hooks.py`: query-loop extension points.
- `simple_agent_cli/__main__.py`: allows `python -m simple_agent_cli`.
- `main.py`: compatibility entry, calls the CLI.
- `simple-agent.bat`: Windows launcher script.

## Run

Configure an OpenAI-compatible chat completions endpoint first:

```powershell
$env:SIMPLE_AGENT_API_KEY = "your-api-key"
$env:SIMPLE_AGENT_MODEL = "deepseek-v4-flash"
```

Optional custom endpoint:

```powershell
$env:SIMPLE_AGENT_BASE_URL = "https://api.deepseek.com"
```

The agent also accepts `OPENAI_API_KEY`, `OPENAI_MODEL`, and `OPENAI_BASE_URL`.

```bash
python -m simple_agent_cli
```

```bash
python -m simple_agent_cli -p "hello agent"
```

Or on Windows:

```bat
D:\SUNNY\simple-python-agent\simple-agent.bat
```

```bat
D:\SUNNY\simple-python-agent\simple-agent.bat -p "hello agent"
```

If `D:\SUNNY\simple-python-agent` is in PATH, you can run:

```bat
simple-agent
simple-agent -p "hello agent"
```

## Slash Commands

```text
/help
/history
/clear
/exit
```

## Local Tools

The model can request local tools through OpenAI-compatible tool calls:

```text
get_local_time
read_file
list_files
search_text
write_file
edit_file
```

`get_local_time` reads the current date, time, timezone, and Unix timestamp from
the local computer, then returns that result to the model.

The file tools accept paths that are absolute
or relative to the directory where the agent was started:

- `read_file`: reads up to 200 numbered UTF-8 lines from a file.
- `list_files`: lists up to 100 files matching a glob pattern.
- `search_text`: returns up to 100 regular-expression matches.
- `write_file`: creates or completely overwrites a UTF-8 text file and creates
  missing parent directories.
- `edit_file`: performs an exact string replacement; ambiguous matches are
  rejected unless `replace_all` is enabled.

Files must be UTF-8 text. Result limits keep individual tool responses from
filling the model context.

The agent runs a small query loop: after a model requests a tool, the tool result
is added to the conversation and the model is called again for a final answer.
The loop stops after the configured tool-turn limit (10 by default), and also
stops if the same tool failure is repeated twice.

## Message flow

`agent.messages` contains only OpenAI-compatible context sent back to the model:
system, user, assistant, and tool messages.

`agent.events` contains query lifecycle events:

- `AssistantMessage`: a completed model response with response metadata.
- `StreamEvent`: an SSE response delta, rendered immediately by the CLI.
- `SystemAPIErrorMessage`: an API/network/timeout error envelope.

Stream and error events are deliberately not added to the next model request.

The client reads SSE deltas as they arrive. The CLI buffers each model turn
until its completed `AssistantMessage` reveals whether it contains tool calls:
tool-call turns are shown as intermediate work, while the tool-free turn is
printed once as the final result.

```text
[中间过程]

第 1 轮：模型准备调用 1 个工具
模型说明：I will inspect the file first.

调用工具：read_file（call_123）
参数：{"path":"main.py"}
工具完成：read_file，耗时 0.002 秒
结果摘要：{"ok":true,"result":{"content":"..."}}

[最终结果]
The file starts the CLI.
```

The completed `AssistantMessage` is still assembled after the stream ends, so
tool calls and multi-turn context work the same way as non-streaming responses.
