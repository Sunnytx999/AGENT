# C++ 原生工具

`agent_tools` 实现项目中不依赖 Agent 运行时的底层功能：

- 获取本地日期和时间；
- 读取 UTF-8 文本文件；
- 使用 glob 列出文件；
- 使用正则表达式搜索文本；
- 创建或完全覆盖 UTF-8 文件；
- 使用精确字符串匹配编辑文件；
- 启动 Flask 子进程。

## 路径规则

所有跨越 C ABI 传入的文件路径和文件选择 glob 都必须是绝对路径。

动态库不会为文件工具读取、推导或自动拼接进程工作目录。Agent 会在提示词中获得当前工作目录，并负责生成绝对工具参数。

例如：

```text
工作目录：E:\project
相对目标：src\main.py
传给工具：E:\project\src\main.py
```

## C ABI

公开接口声明位于：

```text
include/agent_tools.h
```

实现位于：

```text
src/agent_tools.cpp
```

接口使用稳定的 C ABI，避免直接向 Python 暴露 C++ 名字修饰、C++ 对象布局和异常。

每个工具函数通过调用方提供的缓冲区返回 UTF-8 JSON，并使用以下返回码：

| 返回值 | 含义 |
| --- | --- |
| `0` | 调用成功，结果已写入缓冲区 |
| `1` | 缓冲区为空或过小，调用方应根据 `required_size` 重新分配 |
| `-1` | ABI 参数无效 |

`required_size` 包含字符串末尾的 NUL 字节。

## DLL 在哪里加载

只有父 Python 进程加载 DLL：

```text
native_host.py
    ↓
根目录 direct_native.py
    ↓ ctypes.CDLL(...)
native/bin/agent_tools_http_ipc_v2.dll
```

Flask 子进程不会加载 DLL。它通过：

```text
langchain_agent_cli/cpp_bridge.py
```

向父进程发送 HTTP/JSON 工具请求。

这样的设计保证同一次应用运行中只有父进程加载一次 DLL。

## Python 与 C ABI 的桥接

根目录 `direct_native.py` 负责：

1. 确定 DLL 默认路径或读取 `SIMPLE_AGENT_TOOLS_LIBRARY`。
2. 使用 `ctypes.CDLL` 加载动态库。
3. 设置每个导出函数的 `argtypes` 和 `restype`。
4. 首次调用函数，获取所需输出缓冲区大小。
5. 分配缓冲区并再次调用函数。
6. 将返回的 UTF-8 JSON 解析为 Python 字典。

这个桥接只供父进程使用，不属于 Flask 子进程中的 Agent 工具客户端。

## 子进程启动

DLL 导出：

```text
agent_start_process
```

父进程调用它启动 Flask。Windows 实现使用 `CreateProcessW`，并继承父进程环境变量。

参数包括：

- Python 可执行文件绝对路径；
- `app.py` 绝对路径；
- 可选的子进程工作目录。

当前父进程不指定子进程工作目录，因此 Flask 继承父进程当前目录。每个 Agent 会话随后可以从 Web 页面或 `POST /api/sessions` 单独选择自己的绝对工作目录。

## 哪些内容仍在 Python 中

以下内容依赖 LangChain、LangGraph 或具体 Agent 会话，因此不会移入 DLL：

- LangChain 工具 Schema 与参数描述；
- Plan Mode 文件写入保护；
- 主 Agent、Explore、Plan 和 Execute；
- Execute/Replan 工作流；
- HTTP 客户端与服务端协议；
- 会话和工作目录状态。

写文件和编辑文件的底层操作由 DLL 完成，但“当前模式是否允许写这个路径”的判断仍由 Python 执行。

## Windows 构建

项目包含预编译的 64 位文件：

```text
bin/agent_tools_http_ipc_v2.dll
```

重新构建需要以下任意一组工具：

- CMake 和 Visual Studio Build Tools；
- Zig。

使用 Zig 时可以执行：

```powershell
python -m pip install ziglang
powershell -ExecutionPolicy Bypass -File native\build.ps1
```

构建结果写入：

```text
native/bin/agent_tools_http_ipc_v2.dll
```

默认情况下，根目录 `direct_native.py` 加载这个文件。也可以通过环境变量指定其他动态库：

```powershell
$env:SIMPLE_AGENT_TOOLS_LIBRARY = "E:\path\to\agent_tools_http_ipc_v2.dll"
python native_host.py
```

## 安全边界

- 原生工具服务只监听 `127.0.0.1`。
- Flask 子进程必须携带父进程生成的随机 Bearer Token。
- 文件工具要求绝对路径，但绝对路径不等于沙箱；调用者仍需负责限制允许访问的目录。
- Plan Mode 的只读限制由 Python 工具包装器执行，不由 DLL 判断。
