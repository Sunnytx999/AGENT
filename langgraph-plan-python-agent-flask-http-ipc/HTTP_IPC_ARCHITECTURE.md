# HTTP 进程间通信架构

这个项目是 `langgraph-plan-python-agent-flask` 的 HTTP 进程间通信版本。Agent 行为、Plan Mode、Web 界面和 Flask API 保持在 Flask 子进程中，C++ 原生工具则隔离在父进程中运行。

## 进程关系

```text
native_host.py（父 Python 进程）
├── 通过 direct_native.py 加载 agent_tools_http_ipc_v2.dll
├── 在随机本机端口提供带认证的 HTTP/JSON 工具服务
├── 生成工具服务地址和随机 Bearer Token
├── 调用 DLL 的 agent_start_process 创建 app.py 子进程
└── 检查 Flask 健康状态并等待子进程结束

app.py（Flask 子进程）
├── 在动态选择的本机端口提供 Web 页面和 JSON API
├── 运行 LangChain、LangGraph 与 Agent 会话
├── 运行主 Agent、Explore、Plan 和 Execute
└── 通过 HTTP/JSON 调用父进程中的原生工具，不加载 DLL

浏览器进程
├── 从 Flask 获取 HTML、JavaScript 和 CSS
└── 使用 fetch() 向 Flask JSON API 发送请求
```

C++ DLL 不是一个独立进程。它被加载到父 Python 进程的地址空间中，DLL 函数也在父进程内执行。

## 启动过程

运行：

```powershell
python native_host.py
```

启动过程如下：

1. 父进程通过 `ctypes.CDLL` 加载 DLL。
2. 父进程启动私有工具 HTTP 服务，操作系统为其选择随机端口。
3. 父进程生成随机 Bearer Token。
4. 父进程为 Flask 选择一个可用 Web 端口。
5. 父进程设置以下环境变量：

```text
SIMPLE_AGENT_TOOL_URL
SIMPLE_AGENT_TOOL_TOKEN
SIMPLE_AGENT_FLASK_PORT
SIMPLE_AGENT_FLASK_CHILD=1
```

6. 父进程调用 DLL 的 `agent_start_process`。
7. DLL 使用 Windows `CreateProcessW` 创建 Flask 子进程。
8. Flask 子进程继承环境变量，因此知道工具服务地址、认证 Token 和自己的 Web 端口。
9. 父进程反复请求 Flask 的 `GET /api/health`。
10. 只有健康接口返回的 PID 与 DLL 创建的子进程 PID 一致时，父进程才打印可访问地址。

动态端口避免旧 Flask 进程占用固定端口导致启动失败。浏览器应打开启动日志最后输出的地址，不应假定端口始终为 `5000`。

## 工具调用协议

Flask 子进程通过以下接口请求原生工具：

```text
POST /tools/call
Authorization: Bearer <随机令牌>
Content-Type: application/json; charset=utf-8
```

例如读取文件时，Flask 子进程发送：

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

父进程中的 `native_host.py` 执行以下步骤：

1. 检查请求路径是否为 `/tools/call`。
2. 使用固定时序比较验证 Bearer Token。
3. 读取并解析 JSON 请求体。
4. 通过 `_dispatch()` 将 `method` 映射到 `DirectNativeTools` 方法。
5. `direct_native.py` 使用 `ctypes` 调用 DLL 的 C ABI。
6. DLL 返回 UTF-8 JSON。
7. 父进程将结果包装成 HTTP JSON 响应。

成功响应示例：

```json
{
  "ok": true,
  "result": {
    "path": "E:\\project\\main.py",
    "content": "1: print('hello')"
  }
}
```

错误响应示例：

```json
{
  "ok": false,
  "error": "File does not exist"
}
```

## 完整工具调用链

```text
大模型生成工具调用
    ↓
langchain_agent_cli/tools.py
    ↓
langchain_agent_cli/cpp_bridge.py
    ↓ HTTP POST /tools/call
native_host.py
    ↓ _dispatch()
direct_native.py
    ↓ ctypes/C ABI
agent_tools_http_ipc_v2.dll
    ↓ UTF-8 JSON结果
native_host.py 返回 HTTP响应
    ↓
cpp_bridge.py 解析响应
    ↓
LangGraph 生成 ToolMessage
    ↓
模型继续处理
```

## 两个 HTTP 服务

项目同时运行两个 HTTP 服务：

| 服务 | 所在进程 | 使用者 | 端口 |
| --- | --- | --- | --- |
| Flask Web 服务 | Flask 子进程 | 浏览器或 curl | 动态选择 |
| 原生工具服务 | 父 Python 进程 | Flask 子进程 | 随机选择 |

两个服务都只监听 `127.0.0.1`，但用途不同。浏览器不能直接调用原生工具接口。

## 不经过 IPC 的功能

以下功能都在 Flask 子进程内部运行，不经过父进程工具 HTTP 服务：

- 主 Agent 的模型调用；
- Explore 子 Agent；
- Plan 子 Agent；
- Execute 子 Agent；
- Execute/Replan LangGraph；
- Plan Mode 权限判断；
- 会话状态管理；
- `plan.md` 状态同步。

只有由 C++ 实现的时间和文件等底层工具会跨进程调用。

## 构建和运行

```powershell
python -m pip install -r requirements.txt
powershell -ExecutionPolicy Bypass -File native\build.ps1
python native_host.py
```

不要直接运行 `python app.py`。直接启动的 Flask 进程不会获得父进程生成的私有工具地址和认证 Token。
