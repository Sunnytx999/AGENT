# LangGraph Plan Python Agent 架构总结

## 1. 项目定位

`langgraph-plan-python-agent` 是一个由单一主 Agent 统一调度的 CLI 编程 Agent。

普通对话、Plan Mode、计划修改和计划批准都进入同一个主 Agent 会话。Explore、Plan 和 Execute 不直接面对用户，而是作为三个工具提供给主 Agent，由主 Agent 根据运行模式、用户要求和当前上下文决定是否调用。

```text
用户 / CLI
    ↓
主 Agent
    ├── 普通文件工具
    ├── explore_codebase → Explore 子 Agent（只读）
    ├── plan_codebase    → Plan 子 Agent（只读）
    └── execute_plan     → ExecutionWorkflow
                              ↓
                       execute → replan
                           ↑        ↓
                           └────────┘
```

流程选择主要由 `MAIN_AGENT_PROMPT` 约束，并没有在代码中固定成 `Explore → Plan → Execute`。普通模式下 Explore 和 Plan 都是可选组件；计划得到用户批准后，主 Agent则必须调用 Execute。


## 2. 主 Agent

主 Agent通过 LangChain的 `create_agent()`创建：

```python
self.main_graph = self._build_agent(
    MAIN_AGENT_PROMPT,
    self._get_main_tools(),
)
```

底层构造为：

```python
create_agent(
    model=self.model,
    tools=tools,
    system_prompt=system_prompt,
    state_schema=AgentState,
    checkpointer=self._checkpointer,
)
```

`create_agent()`返回的本身就是一个编译后的 LangGraph Agent。其内部完成大模型调用、工具请求、工具结果回传以及继续推理，直到模型不再请求工具。

```text
HumanMessage
    ↓
模型推理
    ↓
AIMessage（可能包含 tool_calls）
    ↓
执行工具并生成 ToolMessage
    ↓
模型继续推理
    ↓
最终 AIMessage
```

主 Agent是持久化的协调者，负责：

- 接收全部用户消息；
- 接收 Explore、Plan和 Execute工具结果；
- 判断是否需要委托子 Agent；
- 审查 Plan子 Agent返回的候选方案；
- 保存最终 `plan.md`；
- 在用户批准计划后调用 Execute；
- 将最终结果返回给用户。

## 3. 子 Agent的调用方式

### 3.1 子 Agent被包装为工具

子 Agent并不是由 CLI直接调用，而是通过工具工厂包装为主 Agent可见的 LangChain工具。

以 Explore为例：

```python
def make_explore_tool(run_explore: Callable[[str], str]):
    @tool(...)
    def explore_codebase(request: str) -> str:
        return run_explore(request)

    return explore_codebase
```

主 Agent创建工具时传入绑定方法：

```python
make_explore_tool(self._run_explore)
```

最终关系为：

```text
explore_codebase(request)
    ↓
self._run_explore(request)
```

Plan和 Execute完全相同：

```text
plan_codebase(task, request)
    ↓
self._run_plan(task, request)

execute_plan(steps)
    ↓
self._run_approved_execution(steps)
```

这里使用嵌套函数和工具工厂，是为了把当前 `LangChainAgent`实例的方法和状态绑定到工具中。工具执行时可以访问当前任务、当前计划、批准状态、子 Agent graph。

### 3.2 主 Agent看到的工具调用

模型看到的是标准工具定义，例如：

```json
{
  "name": "explore_codebase",
  "args": {
    "request": "检查认证模块的结构、现有实现和测试"
  }
}
```

LangChain执行 `explore_codebase()`，工具内部调用 `_run_explore()`。子 Agent的返回字符串会成为 `ToolMessage`，重新放回主 Agent消息列表，主 Agent随后根据结果继续推理。

```text
主 Agent AIMessage（请求 Explore工具）
    ↓
Explore子 Agent运行
    ↓
Explore返回调查结果
    ↓
主 Agent收到 ToolMessage
    ↓
主 Agent继续决策
```

### 3.3 子 Agent按需懒加载

主 Agent在初始化时立即创建，但 Explore、Plan和 Execute只在第一次使用时创建：

```python
def _ensure_explore_graph(self) -> None:
    if self.explore_graph is not None or self.model is None:
        return
    self.explore_graph = self._build_agent(
        EXPLORE_PROMPT,
        READ_ONLY_TOOLS,
    )
```

这样简单问候或普通问题不会提前创建所有子 Agent。

每个 Agent使用独立的 LangGraph thread ID：

```text
cli-session
explore-session
plan-session
execution-session
```

它们可以共享同一个 `MemorySaver`，但不同 thread ID的消息状态彼此隔离。

### 3.4 Explore子 Agent

Explore负责快速、只读地调查代码库：

```python
def _run_explore(self, request: str) -> str:
    self._ensure_explore_graph()
    task = self.current_task or "No task has been recorded."
    prompt = (
        f"Original user task:\n{task}\n\n"
        f"Main agent request:\n{request}"
    )
    return self._invoke_agent(
        self.explore_graph,
        self.explore_thread_id,
        prompt,
        label="explore",
    )
```

调用链：

```text
主 Agent调用 explore_codebase
    ↓
_run_explore()
    ↓
组合原始任务与调查要求
    ↓
调用 Explore graph
    ↓
Explore使用只读工具检查代码
    ↓
返回文件、模式、约束和实现证据
```

Explore只拥有 `READ_ONLY_TOOLS`：

- `get_local_time`
- `read_file`
- `list_files`
- `search_text`

所以它不能修改项目。

### 3.5 Plan子 Agent

Plan工具接收：

```python
plan_codebase(task: str, request: str)
```

- `task`：原始用户实现需求；
- `request`：探索结果、当前约束和具体规划问题。

`_run_plan()`将以下内容组合后交给 Plan子 Agent：

```text
原始用户任务
当前计划
主 Agent的规划要求
```

Plan同样只拥有 `READ_ONLY_TOOLS`。它只能返回候选计划或规划建议，不能写入 `plan.md`。

```text
Plan子 Agent
    → 分析并返回候选方案

主 Agent
    → 审查候选方案
    → 修正错误或遗漏
    → 写入最终 plan.md
```

因此 Plan子 Agent不是计划的最终所有者，主 Agent才是。

### 3.6 Execute组件

计划批准后，主 Agent调用：

```json
{
  "name": "execute_plan",
  "args": {
    "steps": [
      "实现数据模型",
      "实现业务逻辑",
      "补充测试并运行"
    ]
  }
}
```

这里的 `steps` 不是通过固定解析器从 Markdown机械提取出来的。主 Agent读取完整批准计划后，在语义上区分背景、约束、验收标准和实际任务，再生成有序 `steps`。

```text
approved_plan
    ↓ 主 Agent语义理解
结构化 steps
    ↓
execute_plan(steps)
    ↓
_run_approved_execution(steps)
    ↓
ExecutionWorkflow.invoke(...)
```

## 4. Plan Mode流程

### 4.1 进入计划模式

用户输入 `/plan` 后调用：

```python
agent.enter_plan_mode()
```

并初始化状态：

```python
self.plan_mode = True
self.pending_plan = None
self.approved_plan = None
self.current_task = None
self._execution_authorized = False
```

### 4.2 生成并保存计划

用户提交需求后，主 Agent进入 `RUNTIME MODE: PLAN`。它可以按需调用 Explore和 Plan，但必须亲自审查最终方案，然后使用 `write_file`或 `edit_file`保存到指定的 `plan.md`。

文件写入成功后：

```python
def _on_main_agent_write(self, path: str) -> None:
    plan = resolved.read_text(encoding="utf-8")
    self.pending_plan = plan
```

所以：

```text
磁盘 plan.md
    ↓
pending_plan
```

`pending_plan` 表示等待用户批准的计划。

### 4.3 修改计划

用户执行 `/reject` 后，当前计划不会被删除。后续输入被视为计划修改意见：

- 小修改可以由主 Agent直接完成；
- 大范围调整可以再次调用 Plan；
- 新的不确定性可以再次调用 Explore；
- 修改后主 Agent重新保存 `plan.md`并等待批准。

### 4.4 批准计划

用户执行 `/approve` 后：

```python
self.approved_plan = self.pending_plan
self.plan_mode = False
self._execution_authorized = True
```

代码不会直接绕过主 Agent调用 ExecutionWorkflow，而是将 `PLAN_APPROVED`事件重新交给同一个主 Agent：

```python
self._invoke_main(self._approved_mode_prompt())
```

提示词要求主 Agent必须调用 `execute_plan`。同时还有两项代码状态：

```python
self._execution_authorized
self._execution_called
```

- `_execution_authorized`：用户是否真的批准计划；
- `_execution_called`：主 Agent是否按要求调用 Execute。

即使模型在未批准时错误调用 Execute，`_run_approved_execution()`也会拒绝执行。

## 5. Replan实现

Replan不是主 Agent每完成一步后重新改写 `plan.md`，也不是一个独立的 Replan子 Agent。它是 `ExecutionWorkflow` 内部的 LangGraph循环，并与 Execute共用同一个 execution agent。

### 5.1 ExecutionState

```python
class ExecutionState(TypedDict, total=False):
    task: str
    plan: list[str]
    past_steps: Annotated[
        list[tuple[str, str]],
        operator.add,
    ]
    response: str
```

字段含义：

| 字段 | 含义 |
| --- | --- |
| `task` | 原始用户任务 |
| `plan` | 当前剩余的可执行步骤 |
| `past_steps` | 已执行步骤及其结果 |
| `response` | 最终结果；存在时通常应该结束 |

`past_steps`使用 `operator.add`作为 reducer。节点返回新的记录时，LangGraph会追加到原列表，而不是覆盖。

```python
旧值：[("A", "A完成")]
节点返回：[("B", "B完成")]
合并后：[("A", "A完成"), ("B", "B完成")]
```

`plan`没有 reducer，因此节点返回的新 `plan`会直接替换旧值。这正是 Replan能够替换剩余计划的基础。

### 5.2 LangGraph结构

```python
builder = StateGraph(ExecutionState)
builder.add_node("execute", self._execute_node)
builder.add_node("replan", self._replan_node)
builder.add_edge(START, "execute")
builder.add_edge("execute", "replan")
builder.add_conditional_edges(
    "replan",
    self._route_after_replan,
    {"execute": "execute", "end": END},
)
```

图结构为：

```text
START
  ↓
execute
  ↓
replan
  ├── 还有步骤 → execute
  └── 已完成   → END
```

一次 `workflow.invoke()`可能在内部循环多次。

### 5.3 初始状态

```python
self.graph.invoke(
    {
        "task": task,
        "plan": plan,
        "past_steps": [],
    },
    config={"recursion_limit": recursion_limit},
)
```

例如：

```python
{
    "task": "实现用户系统",
    "plan": [
        "实现用户模型",
        "实现登录接口",
        "补充测试",
    ],
    "past_steps": [],
}
```

### 5.4 Execute节点

Execute每次只执行第一条剩余步骤：

```python
current_step = state["plan"][0]
```

提示词包含：

```text
原始任务
已经完成的步骤和结果
本轮唯一需要执行的当前步骤
```

execution agent拥有完整 `TOOLS`，可以读写项目文件。成功后节点返回：

```python
{
    "plan": state["plan"][1:],
    "past_steps": [(current_step, result)],
}
```

也就是：

```text
从 plan中移除第一步
+
将执行结果追加到 past_steps
```

### 5.5 Replan节点

每执行完一个步骤，LangGraph都会根据固定边进入 `_replan_node()`。Replan不会只检查“原计划还剩几步”，而是把以下完整上下文交给同一个 execution agent：

```text
原始任务：最终需要满足什么需求
已完成步骤及其结果：实际做了什么、成功还是失败、发现了什么
当前剩余计划：尚未执行的原计划步骤
```

execution agent需要根据这些信息判断：原始任务是否已经真正完成，以及当前剩余计划是否仍然正确。它只能返回一个严格的 JSON对象，不能添加 Markdown代码块、解释或 JSON之外的文字。返回结构为：

```json
{
  "steps": ["步骤1", "步骤2"],
  "response": null
}
```

其中最重要的语义约定是：

> `steps` 表示从现在开始仍需执行的完整、有序计划。返回值会整体替换当前 `ExecutionState.plan`，因此它既不是差异列表，也不是只包含新增步骤、修改步骤或下一个步骤的增量结果。

Replan有以下三种正常决定。

#### 情况一：仍有工作，但不需要修改计划

如果当前剩余计划仍然正确，必须将其完整、原样返回。不能只返回下一步。

例如当前剩余计划是：

```json
["实现登录接口", "补充认证失败测试"]
```

则应返回：

```json
{
  "steps": [
    "实现登录接口",
    "补充认证失败测试"
  ],
  "response": null
}
```

#### 情况二：仍有工作，并且需要修改计划

如果执行结果暴露了新的依赖、错误或遗漏，就返回完整的调整后计划。新数组中既要包括新增或修改的步骤，也要保留仍然有效的旧步骤，但不能再次包含已经完成的步骤。

例如原剩余计划是：

```json
["实现登录接口", "补充认证失败测试"]
```

执行历史表明登录接口依赖尚未建立的数据库表，则可以返回：

```json
{
  "steps": [
    "创建用户表数据库迁移",
    "实现登录接口",
    "补充认证失败测试"
  ],
  "response": null
}
```

这三个步骤会整体替换旧的剩余计划。只要还有任何工作，`response` 就必须为 `null`，路由随后回到 Execute继续执行新计划的第一步。

#### 情况三：原始任务已经完成

只有结合原始任务和全部执行结果，确认不再需要任何后续工作时，才返回：

```json
{
  "steps": [],
  "response": "用户系统已经实现并通过测试"
}
```

此时 `response` 是提供给用户的最终结果，路由函数会选择 `END`。

需要特别注意：

> 当前剩余计划为 `[]`，只表示原计划中的步骤已经取完，并不能单独证明原始任务已经完成。

例如最后一个原定步骤是“运行测试”，但执行结果显示测试失败。即使传入 Replan时当前计划已经是 `[]`，Replan仍应补充后续工作：

```json
{
  "steps": [
    "定位测试失败原因",
    "修复实现",
    "重新运行测试"
  ],
  "response": null
}
```

因此，Replan判断的是“原始任务是否已满足”，而不只是“原计划是否已经执行完”。

### 5.6 Replan结果解析

模型输出由 `_parse_replan_decision()`处理：

```python
payload = json.loads(_strip_json_fence(raw))

steps = [
    str(step)
    for step in payload.get("steps", [])
    if str(step).strip()
]

response = payload.get("response")
```

处理内容包括：

1. 去掉可能存在的 Markdown JSON代码围栏；
2. 解析 JSON；
3. 读取 `steps`；
4. 将步骤转换成字符串；
5. 过滤空步骤；
6. 读取可选的最终 `response`。

解析成功后：

```python
update = {"plan": decision["steps"]}

if decision.get("response"):
    update["response"] = decision["response"]
```

因为 `plan`没有 reducer，新 `steps`会替换旧的剩余计划。

### 5.7 解析失败回退

如果 Replan没有返回合法 JSON，并且仍有剩余任务：

```python
return {
    "steps": remaining_steps,
    "response": None,
}
```

也就是保留原计划，不丢失未完成工作。

如果已经没有剩余步骤，则使用最后一次执行结果作为最终响应：

```python
fallback = (
    past_steps[-1][1]
    if past_steps
    else "Task completed."
)
```

### 5.8 条件路由

```python
def _route_after_replan(state: ExecutionState) -> str:
    return (
        "end"
        if state.get("response") or not state.get("plan")
        else "execute"
    )
```

路由规则：

```text
response有值
→ END

plan为空
→ END

response为空且plan仍有步骤
→ execute
```

### 5.9 状态变化示例

初始状态：

```python
{
    "plan": ["A", "B", "C"],
    "past_steps": [],
}
```

执行 A：

```python
{
    "plan": ["B", "C"],
    "past_steps": [("A", "A完成")],
}
```

Replan发现 B需要拆分：

```json
{
  "steps": ["B1", "B2", "C"],
  "response": null
}
```

新的状态：

```python
{
    "plan": ["B1", "B2", "C"],
    "past_steps": [("A", "A完成")],
}
```

执行 B1后，Replan发现 B2不再需要：

```json
{
  "steps": ["C"],
  "response": null
}
```

执行 C后返回：

```json
{
  "steps": [],
  "response": "任务完成"
}
```

LangGraph路由到 `END`并返回最终 `ExecutionState`。

## 6. Execute和 Replan为什么共用一个 Agent

项目没有单独创建 `replan_graph`。Execute和 Replan都调用：

```python
execution_agent=lambda prompt: self._invoke_agent(
    self.execution_graph,
    self.execution_thread_id,
    prompt,
    label="execute",
)
```

两种阶段通过不同提示词区分：

```text
Execute提示词
→ 使用工具完成当前步骤

Replan提示词
→ 不修改文件，只返回严格 JSON
```

优点包括：

- Execute知道之前完成了什么；
- Replan可以根据真实执行结果调整剩余计划；
- 使用同一个 thread保持上下文连续；
- 主 Agent只需要调用一次 `execute_plan`；
- Execute/Replan循环完全封装在一个组件中。

需要注意：Replan阶段虽然提示模型不要修改文件，但 execution agent仍然拥有完整写工具。因此这一限制主要由提示词保证，并不是通过切换到只读工具在代码层强制隔离。

## 7. 错误处理

### 7.1 子 Agent调用失败

`_invoke_agent()`捕获异常并返回：

```text
LLM request failed: ...
```

错误作为工具结果回到主 Agent，避免 CLI直接崩溃。

### 7.2 Execute失败

如果当前执行步骤返回 LLM失败信息，Execute节点会设置 `response`。随后 Replan检测到已有 `response`并结束工作流。

### 7.3 Replan失败

如果 Replan阶段的 LLM请求失败，工作流清空计划、把失败信息写入 `response`并结束。

### 7.4 JSON解析失败

解析器保留原始剩余步骤，防止因为一次格式错误而错误丢弃未完成工作。

## 8. 权限设计

Plan Mode中主 Agent仍然拥有写工具，但通过 `make_main_file_tools()`增加运行时权限检查：

```python
if not self.plan_mode:
    return True

return resolved_path == self.plan_file_path.resolve()
```

最终权限边界：

```text
NORMAL模式
→ 主 Agent可以写普通项目文件

PLAN模式
→ 主 Agent只能写当前 plan.md

Explore / Plan子 Agent
→ 永远只有只读工具

PLAN_APPROVED
→ 主 Agent必须通过 Execute实施计划
```

这形成了“提示词约束 + 工具权限校验”的双重保护。


## 9. 核心结论


1. **单一主 Agent统一调度**：用户始终与同一个主 Agent交互；
2. **子 Agent工具化**：Explore、Plan和 Execute通过 LangChain工具由主 Agent按需调用；
3. **计划建议与计划所有权分离**：Plan只提供候选方案，主 Agent审查并写入最终计划；
4. **批准后强制委托 Execute**：主 Agent把 Markdown计划语义转换成结构化 `steps`；
5. **LangGraph管理动态 Replan**：每执行一步就评估一次剩余工作，并用完整的新 `steps`替换旧计划；
6. **状态驱动而非重新解析 Markdown**：执行阶段围绕 `ExecutionState.plan`、`past_steps`和 `response`运行。


> 主 Agent负责决策和用户交互，Explore负责查证，Plan负责候选设计，Execute负责实施；Execute内部使用 LangGraph在每一步之后执行 Replan，直到任务完成。
