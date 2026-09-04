# LangGraph Plan Python Agent - Flask HTTP IPC

这是一个使用 Flask 提供 Web 界面、使用 LangChain/LangGraph 组织 Agent，并通过本机 HTTP/JSON 跨进程调用 C++ 动态库工具的示例项目。

项目最重要的设计目标是：**C++ DLL 只加载到父 Python 进程一次，运行 Agent 的 Flask 子进程不再重复加载 DLL。**

## 一、整体架构

一次正常运行主要涉及三个进程：

```text
浏览器进程
    │
    │ HTTP/JSON：聊天、创建会话、Plan Mode 操作
    ▼
Flask 子进程（app.py）
    │
    │ HTTP/JSON：调用本地原生工具
    ▼
父 Python 进程（native_host.py）
    │
    │ ctypes：同一进程内调用 C ABI
    ▼
agent_tools_http_ipc_v2.dll
```

PowerShell 负责启动父 Python 进程并显示其标准输出，但它不运行 Agent 或工具逻辑。

### 1. 父 Python 进程

入口文件是 `native_host.py`。它负责：

1. 通过根目录的 `direct_native.py` 和 `ctypes.CDLL` 加载 `native/bin/agent_tools_http_ipc_v2.dll`。
2. 在 `127.0.0.1` 的随机可用端口上启动私有工具 HTTP 服务。
3. 生成随机 Bearer Token，保护私有工具接口。
4. 选择一个可用的 Flask Web 端口。
5. 调用 DLL 导出的 `agent_start_process`，由 C++ `CreateProcessW` 创建 Flask 子进程。
6. 请求 Flask 的 `GET /api/health`，验证响应 PID 与刚创建的子进程 PID 一致。
7. 等待 Flask 子进程结束，并在启动失败或中断时进行清理。

工具 HTTP 服务使用 `ThreadingHTTPServer`。服务线程仍属于父 Python 进程，并不是额外进程。

### 2. C++ 动态库

动态库位于：

```text
native/bin/agent_tools_http_ipc_v2.dll
```

它在父 Python 进程内实现：

- 获取本地时间；
- 读取 UTF-8 文件；
- 按 glob 列出文件；
- 使用正则表达式搜索文本；
- 创建或覆盖 UTF-8 文件；
- 精确替换文件内容；
- 创建 Flask 子进程。

DLL 不是独立进程。`ctypes` 加载 DLL 后，DLL 函数直接运行在父 Python 进程中。

### 3. Flask 子进程

入口文件是 `app.py`。它负责：

- 提供 Web 页面与 JSON API；
- 为每个 Web 会话创建独立的 `LangChainAgent` 对象；
- 运行主 Agent、Explore、Plan、Execute 和 Execute/Replan LangGraph；
- 管理 Plan Mode、计划批准和会话状态；
- 通过 `langchain_agent_cli/cpp_bridge.py` 请求父进程中的原生工具服务。

Explore、Plan 和 Execute 是 Flask 子进程中的 Python/LangGraph 组件，不是新的操作系统进程。创建多个 Agent 会话也不会为每个会话再启动一个 Python 进程。

### 4. 浏览器进程

浏览器从 Flask 获取：

```text
GET /
GET /static/app.js
GET /static/style.css
```

页面中的 `fetch()` 再以 JSON 请求 Flask API。浏览器不直接连接父进程的工具服务，也不知道工具服务的 Bearer Token。

## 二、两条 HTTP 通信链路

项目中存在两套相互独立的 HTTP 服务。

### 1. 浏览器与 Flask

父进程会为 Flask 选择一个可用端口，例如：

```text
http://127.0.0.1:60123
```

浏览器通过这个地址访问页面，并调用公开的应用 API：

```text
POST /api/sessions
POST /api/messages
POST /api/plan
POST /api/approve
POST /api/reject
POST /api/exit-plan
GET  /api/sessions/<session_id>
GET  /api/health
```

例如，浏览器发送消息时，`static/app.js` 会发出类似请求：

```http
POST /api/messages HTTP/1.1
Content-Type: application/json

{
  "session_id": "会话ID",
  "message": "你好"
}
```

Flask 沿同一条 TCP 连接返回 JSON 响应。

### 2. Flask 与父进程工具服务

父进程还会为原生工具 HTTP 服务选择另一个随机端口，例如：

```text
http://127.0.0.1:60122
```

父进程在创建 Flask 子进程前设置：

```text
SIMPLE_AGENT_TOOL_URL
SIMPLE_AGENT_TOOL_TOKEN
SIMPLE_AGENT_FLASK_PORT
SIMPLE_AGENT_FLASK_CHILD=1
```

子进程继承这些环境变量，因此知道私有工具服务地址和认证 Token。

当 Agent 调用 `read_file` 时，调用链如下：

```text
大模型返回 read_file 工具调用
    ↓
langchain_agent_cli/tools.py
    ↓
langchain_agent_cli/cpp_bridge.py
    ↓ POST /tools/call + Bearer Token
父进程 native_host.py
    ↓ _dispatch()
根目录 direct_native.py
    ↓ ctypes/C ABI
C++ DLL agent_read_file()
    ↓ 返回 UTF-8 JSON
父进程返回 HTTP JSON
    ↓
Flask 子进程获得工具结果
    ↓
LangGraph 将 ToolMessage 交回模型
```

请求体示例：

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

这条 HTTP 链路只监听 `127.0.0.1`，并要求随机 Token。它不是提供给浏览器或外部机器使用的公共 API。

## 三、哪些调用不经过 HTTP IPC

以下逻辑都在 Flask 子进程内部完成：

- 主 Agent 调用 Explore 子 Agent；
- 主 Agent 调用 Plan 子 Agent；
- 主 Agent 调用 Execute 组件；
- Execute/Replan LangGraph 节点切换；
- Plan Mode 权限判断；
- 会话状态与 `plan.md` 状态同步。

只有时间和文件等底层原生工具需要通过 HTTP 请求父进程中的 DLL。

## 四、工作目录与会话

Web 页面允许为新会话选择一个已存在的绝对工作目录，例如：

```text
E:\my-project
```

点击“新会话”后，该目录会作为 `LangChainAgent.working_directory`。主 Agent 会在提示词中看到它，Explore、Plan 和 Execute 默认继承该目录。传给本地文件工具的路径与 glob 必须是绝对路径。

每个会话拥有独立的 Agent 状态和计划文件：

```text
.sessions/<session_id>/plan.md
```

`.sessions` 是运行时数据，不应提交到 Git。

## 五、启动顺序

当前子进程启动逻辑使用 Windows `CreateProcessW`，因此启动器目前要求 Windows。

### 1. 安装依赖

```powershell
cd E:\simple-python-agent\langgraph-plan-python-agent-flask-http-ipc
python -m pip install -r requirements.txt
```

### 2. 编译 DLL

项目已经包含预编译 DLL。修改 C++ 源码后可以重新编译：

```powershell
powershell -ExecutionPolicy Bypass -File native\build.ps1
```

### 3. 启动父进程

```powershell
python native_host.py
```

不要直接运行：

```powershell
python app.py
```

因为直接启动不会获得父进程生成的工具服务地址和认证 Token。

### 4. 打开实际输出地址

启动日志类似：

```text
[native-host] tool API listening on http://127.0.0.1:60122
[native-host] Flask child started with PID 12345
 * Running on http://127.0.0.1:60123
[native-host] Flask child PID 12345 is ready
[native-host] open http://127.0.0.1:60123
```

浏览器应该打开最后一行地址：

```text
http://127.0.0.1:60123
```

端口默认动态选择，不要假定始终是 `5000`。

## 六、目录结构

```text
langgraph-plan-python-agent-flask-http-ipc/
├── native_host.py              # 父进程入口、工具 HTTP 服务、子进程管理
├── direct_native.py            # 父进程专用 ctypes/C ABI 桥接
├── app.py                      # Flask 子进程入口与 Web API
├── langchain_agent_cli/
│   ├── agent.py                # 主 Agent、模式状态和子 Agent 调度
│   ├── cpp_bridge.py           # Flask 子进程使用的工具 HTTP 客户端
│   ├── tools.py                # LangChain 工具定义与 Plan Mode 写入保护
│   ├── explore_agent.py        # Explore 子 Agent 提示词
│   ├── plan_agent.py           # Plan 子 Agent 提示词
│   ├── execution_workflow.py   # Execute/Replan LangGraph
│   └── subagent_tools.py       # Explore、Plan、Execute 工具包装
├── native/
│   ├── include/agent_tools.h   # 稳定 C ABI 声明
│   ├── src/agent_tools.cpp     # C++ 工具与子进程启动实现
│   ├── bin/                    # 编译后的 DLL
│   └── build.ps1               # Windows 编译脚本
├── templates/index.html        # Web 页面
├── static/app.js               # 浏览器请求与状态更新
├── static/style.css            # 页面样式
└── tests/                      # Flask、Agent、工作目录和 IPC 测试
```

## 七、测试

运行核心测试：

```powershell
python -m unittest tests.test_agent_modes tests.test_app tests.test_native_tools tests.test_working_directory
```

其中 `test_native_tools` 会真实经过：

```text
Python HTTP 客户端 → 父进程测试服务器 → ctypes → C++ DLL
```

## 八、补充架构文档

更精简的 IPC 协议说明见：

- [HTTP_IPC_ARCHITECTURE.md](HTTP_IPC_ARCHITECTURE.md)
- [native/README.md](native/README.md)
