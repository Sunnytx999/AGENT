# LangChain Python Agent

这是原 `simple-python-agent` 的 LangChain 版本，位于独立目录，不修改原项目。
模型、消息类型、工具 schema、工具调用循环和流式进度由 LangChain 1.x / LangGraph
负责；本项目保留原有 CLI、会话历史、斜杠命令以及 6 个本地工具。

## 安装

建议使用独立虚拟环境：

```powershell
cd langchain-python-agent
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 模型配置

为了与原项目当前行为保持一致，API Key、DeepSeek 地址、模型名称、超时和最大工具
轮数固定写在 `langchain_agent_cli/config.py` 的 `AgentConfig.from_env()` 中，运行前
不需要设置环境变量。修改模型配置时直接编辑该方法。

`ChatOpenAI` 支持符合 OpenAI Chat Completions 规范的自定义地址；若服务商有专用
LangChain 集成，且需要其非标准响应字段，应改用对应的模型类。

## 运行

```powershell
python -m langchain_agent_cli
python -m langchain_agent_cli -p "读取 README.md 并概括"
```

Windows 也可以运行 `langchain-agent.bat`。将项目目录加入 `PATH` 后，可在任意
命令行中输入 `langchain-python-agent` 启动同名脚本。

命令：`/help`、`/history`、`/clear`、`/exit`。

工具：`get_local_time`、`read_file`、`list_files`、`search_text`、
`write_file`、`edit_file`。

## 测试

```powershell
python -m unittest discover -s tests -v
```
