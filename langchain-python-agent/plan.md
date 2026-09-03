# LangChain 聊天 Agent 项目实现计划：`E:\persona-chatbot`

## 1. 目标与范围

- 支持多个人物角色，首批包含海盗 `/pirate` 和女巫 `/witch`。
- 通过斜杠命令切换人物，切换后对话风格和 system prompt 立即变化。
- 每个人物拥有独立的 system prompt、语言风格提示和开场白。
- 每个人物拥有独立的短期对话记忆，互不混合。
- 使用 LangGraph 的 `StateGraph`、`AgentState` 和 `MemorySaver` 管理短期记忆。
- 提供 CLI 交互入口和完整单元测试。

---

## 2. 项目目录结构

```
E:\persona-chatbot\
├── pyproject.toml
├── README.md
├── .env.example
├── src\
│   └── persona_chatbot\
│       ├── __init__.py
│       ├── config.py
│       ├── state.py
│       ├── llm.py
│       ├── memory.py
│       ├── graph.py
│       ├── chatbot.py
│       ├── commands.py
│       ├── cli.py
│       └── personas\
│           ├── __init__.py
│           ├── base.py
│           ├── pirate.py
│           └── witch.py
└── tests\
    ├── conftest.py
    ├── test_personas.py
    ├── test_commands.py
    ├── test_graph.py
    └── test_memory.py
```

---

## 3. 各文件职责与设计

### 3.1 `pyproject.toml`

- 定义项目元信息，Python 版本要求 `>=3.11`。
- 运行依赖：
  - `langgraph`
  - `langchain`
  - `langchain-core`
  - `langchain-openai`
  - `python-dotenv`
- 开发依赖：
  - `pytest`
  - `pytest-asyncio`
  - `ruff`（可选）
- 注册 CLI 入口：
  - `persona-chatbot = persona_chatbot.cli:main`

---

### 3.2 `.env.example`

- 提供环境变量模板：
  - `OPENAI_API_KEY`
  - `LLM_MODEL`，默认 `gpt-4o-mini`
  - `LLM_BASE_URL`，可选
  - `LLM_TEMPERATURE`，默认 `0.7`

---

### 3.3 `src/persona_chatbot/config.py`

- 定义 `Settings` 数据类，负责从环境变量读取配置。
- 提供可复用函数：
  - `get_settings()`：读取并缓存配置。
- 配置项用于创建 LLM 和决定运行参数。

---

### 3.4 `src/persona_chatbot/state.py`

- 定义 LangGraph 使用的 `AgentState`，结构为 `TypedDict`：
  - `messages`：`Annotated[list[BaseMessage], add_messages]`
  - `persona_id`：当前人物 id
- 设计说明：
  - `messages` 使用 LangGraph 的 `add_messages` reducer 自动追加新消息。
  - `persona_id` 不参与消息累加，只在每次调用时决定注入哪个 system prompt。

---

### 3.5 `src/persona_chatbot/personas/base.py`

- 定义 `Persona` 数据类，字段包括：
  - `id`
  - `display_name`
  - `system_prompt`
  - `style_hints`
  - `opening_line`
- 定义人物注册表：
  - `PERSONA_REGISTRY`
- 提供可复用函数：
  - `register_persona(persona)`
  - `get_persona(persona_id)`
  - `list_personas()`
  - 未知人物 id 抛出 `PersonaNotFoundError`

---

### 3.6 `src/persona_chatbot/personas/pirate.py`

- 定义海盗人物：
  - id：`pirate`
  - 显示名：`Pirate Captain`
  - system prompt：明确“你是海盗船长”，要求使用航海黑话、粗犷语气。
  - style hints：例如频繁使用 `Arrr`、航海比喻、称呼对方为 `matey`。
  - 开场白：符合海盗口吻。

---

### 3.7 `src/persona_chatbot/personas/witch.py`

- 定义女巫人物：
  - id：`witch`
  - 显示名：`Forest Witch`
  - system prompt：明确“你是神秘森林女巫”，要求语气神秘、喜欢 cackle、提及魔药和咒语。
  - style hints：例如使用 `cackle`、神秘暗示、自然元素比喻。
  - 开场白：符合女巫口吻。

---

### 3.8 `src/persona_chatbot/personas/__init__.py`

- 导入 `pirate` 和 `witch`。
- 将两个人物注册到 `PERSONA_REGISTRY`。
- 对外暴露：
  - `get_persona`
  - `list_personas`
  - `PERSONA_REGISTRY`

---

### 3.9 `src/persona_chatbot/llm.py`

- 提供可复用函数：
  - `create_chat_model(settings)`：返回 `BaseChatModel`。
- 默认使用 `ChatOpenAI`，从配置读取模型名、温度和可选 `base_url`。
- 为单元测试预留注入 fake LLM 的接口。

---

### 3.10 `src/persona_chatbot/memory.py`

- 提供可复用函数：
  - `create_memory()`：返回 `MemorySaver` 实例。
  - `build_thread_id(persona_id, user_id=None)`：
    - 默认返回 `persona:{persona_id}`
    - 若未来支持多用户，则返回 `{user_id}:{persona_id}`
- 设计说明：
  - LangGraph 通过 `thread_id` 区分 checkpoint。
  - 每个人物使用独立 `thread_id`，因此 `MemorySaver` 会为每个人物保存独立历史。

---

### 3.11 `src/persona_chatbot/graph.py`

- 提供核心可复用函数：
  - `build_chat_graph(llm, persona_registry) -> CompiledStateGraph`
- 构建 `StateGraph(AgentState)`。
- 节点 `chatbot` 的处理流程：
  1. 根据 `state["persona_id"]` 获取对应 `Persona`。
  2. 在 `state["messages"]` 前注入 `SystemMessage(persona.system_prompt)`。
  3. 调用 LLM。
  4. 返回 `{"messages": [ai_response]}`，由 `add_messages` 自动追加。
- 图结构：
  - `START -> chatbot -> END`
- 使用 `MemorySaver` 编译图。
- 设计说明：
  - system prompt 不写入持久化消息，只在每次调用时动态注入，避免重复和污染。

---

### 3.12 `src/persona_chatbot/commands.py`

- 定义命令解析结果结构。
- 提供可复用函数：
  - `parse_input(raw: str) -> ParsedInput`
  - `extract_command_name(raw: str)`
  - `resolve_switch_target(command: str)`
- 支持命令：
  - `/pirate`：切换到海盗
  - `/witch`：切换到女巫
  - `/help`：显示帮助
  - `/exit`：退出
- 设计说明：
  - 命令由 CLI 层优先处理，不进入 LLM 和记忆。
  - 未知 `/xxx` 默认给出提示，不调用 LLM。

---

### 3.13 `src/persona_chatbot/chatbot.py`

- 定义 `PersonaChatbot` 编排类：
  - `__init__(llm=None, checkpointer=None)`
  - `async handle_message(raw_text, current_persona_id) -> ChatResult`
  - `switch_persona(persona_id) -> Persona`
- 职责：
  - 管理当前人物。
  - 判断输入是命令还是普通消息。
  - 对普通消息调用编译后的 LangGraph。
  - 每次调用传入：
    - `state`: `{"messages": [HumanMessage(...)], "persona_id": current_persona_id}`
    - `config`: `{"configurable": {"thread_id": build_thread_id(persona_id)}}`
- 返回结果包含：
  - AI 回复文本
  - 新的当前人物 id
  - 是否为命令

---

### 3.14 `src/persona_chatbot/cli.py`

- 提供 CLI 入口：
  - `main()`
  - `async_main()`
- 启动流程：
  1. 加载 `.env`。
  2. 创建 LLM、MemorySaver、LangGraph。
  3. 创建 `PersonaChatbot`。
  4. 默认人物可设为 `pirate`。
- 交互循环：
  - 显示当前人物前缀，例如 `[pirate] You:`。
  - 输入 `/pirate` 或 `/witch` 时切换人物，并打印新人物开场白。
  - 输入普通文本时调用 `PersonaChatbot` 并打印回复。
  - 输入 `/help` 显示命令列表。
  - 输入 `/exit` 或 `Ctrl+C` 退出。

---

### 3.15 `README.md`

- 项目简介。
- 安装步骤。
- 环境变量配置说明。
- 运行 CLI 的方法。
- 运行测试的方法。
- 架构简要说明。

---

## 4. 关键数据流

1. 用户在 CLI 输入文本。
2. `commands.parse_input` 判断是命令还是普通消息。
3. 如果是 `/pirate` 或 `/witch`：
   - 更新当前人物 id。
   - 不调用 LLM，不写入记忆。
   - 打印新人物开场白。
4. 如果是普通消息：
   - 构造 `AgentState`，包含 `messages` 和当前 `persona_id`。
   - 使用 `thread_id = persona:{persona_id}` 调用图。
   - 图节点根据 `persona_id` 注入 system prompt。
   - LLM 生成回复，LangGraph 将回复追加到该人物的 checkpoint。
5. 每个人物的历史只存在于自己的 `thread_id` 下，因此记忆天然隔离。

---

## 5. 单元测试计划

### 5.1 `tests/conftest.py`

- 提供共享 fixture：
  - 测试配置。
  - 确定性 fake LLM，避免真实 API 调用。
  - 每次测试新建的 `MemorySaver`。
  - 每次测试新建的 `PersonaChatbot`。

---

### 5.2 `tests/test_personas.py`

- 验证注册表包含 `pirate` 和 `witch`。
- 验证每个人物 id 唯一。
- 验证每个人物 `system_prompt` 非空且互不相同。
- 验证 `opening_line` 非空且互不相同。
- 验证获取未知人物时抛出异常。

---

### 5.3 `tests/test_commands.py`

- `/pirate` 解析为切换到 `pirate`。
- `/witch` 解析为切换到 `witch`。
- 普通文本解析为普通消息。
- 未知斜杠命令安全处理。
- `/help` 和 `/exit` 被正确识别。

---

### 5.4 `tests/test_graph.py`

- 使用 fake LLM 编译图。
- 验证 `persona_id=pirate` 时注入海盗 system prompt。
- 验证 `persona_id=witch` 时注入女巫 system prompt。
- 验证 AI 回复被正确追加到 `messages`。

---

### 5.5 `tests/test_memory.py`

- 使用真实 `MemorySaver` 和 fake LLM。
- 同一海盗 `thread_id` 连续调用两次：
  - 海盗历史包含两条人类消息和两条 AI 消息。
- 切换到女巫 `thread_id` 后调用一次：
  - 女巫历史只包含女巫自己的消息，不包含海盗消息。
- 再次切回海盗：
  - 海盗历史仍保留之前的内容。
- 证明每个人物记忆独立。

---

## 6. 验证步骤

1. 打开终端并进入项目目录：
   - `cd /d E:\persona-chatbot`
2. 创建并激活虚拟环境：
   - `python -m venv .venv`
   - `.venv\Scripts\activate`
3. 安装项目及开发依赖：
   - `pip install -e ".[dev]"`
4. 运行测试：
   - `pytest -q`
   - 预期所有测试通过。
5. 启动 CLI：
   - `persona-chatbot`
   - 或 `python -m persona_chatbot.cli`
6. 手动验证：
   - 输入 `/pirate`，确认海盗开场白和提示符变化。
   - 输入几轮海盗对话。
   - 输入 `/witch`，确认女巫开场白和风格变化。
   - 输入几轮女巫对话。
   - 输入 `/pirate` 切回海盗，询问之前海盗对话内容，确认仍记得海盗历史。
   - 输入 `/witch` 并询问之前海盗对话内容，确认不记得海盗历史。
   - 测试 `/help` 和 `/exit`。
7. 可选静态检查：
   - `ruff check src tests`

---

## 7. 建议实施顺序

1. 初始化项目骨架：`pyproject.toml`、包目录、`.env.example`、`config.py`。
2. 实现 `state.py`、`personas` 模块和 `commands.py`。
3. 实现 `llm.py`、`memory.py`、`graph.py`。
4. 实现 `chatbot.py` 和 `cli.py`。
5. 编写测试并运行 `pytest`。
6. 编写 `README.md` 并完成手动验证。