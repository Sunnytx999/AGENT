# LangGraph Plan Python Agent

这是一个由单一主 Agent 调度的 CLI 编码 Agent。普通对话、Plan Mode、计划修改和
计划批准事件都进入同一个主 Agent 会话，不再区分 Main Agent 与 Chat Agent。

```text
                         ┌─ explore_codebase（可选、只读）
用户 / CLI → 主 Agent ──┼─ plan_codebase（可选、只读）
                         └─ execute_plan（计划批准后必须调用）
                                   └─ execute ↔ replan → 完成
```

流程选择主要由 `MAIN_AGENT_PROMPT` 约束，而不是用代码固定
`Explore → Plan → Execute`：

- 普通模式下，主 Agent 可以直接回答或修改代码，也可以按需调用 Explore/Plan。
- Plan Mode 下，主 Agent 只能读取项目文件，唯一允许写入的是当前 `plan.md`。
- Explore 和 Plan 子 Agent 都是只读的，只向主 Agent 返回调查结果或候选方案。
- 主 Agent 必须亲自审查候选方案，并把最终计划写入 `plan.md`。
- Plan Mode 会先用当前用户输入初始化 `current_task`。在 `plan.md` 尚未生成时，后续
  输入会更新它，因此问候不会永久占用原始任务；计划写入后，后续输入作为修改反馈，
  不再覆盖原始任务。Plan 子 Agent 是可选的，不负责初始化该状态。
- 用户要求修改计划时，小改动由主 Agent 直接完成；大改动可再次调用 Plan 或 Explore。
- 用户执行 `/approve` 后，同一个主 Agent 收到 `PLAN_APPROVED` 事件，并被要求必须调用
  `execute_plan`。主 Agent 根据完整 `plan.md` 的语义生成结构化 `steps` 参数，Execute
  组件把它写入 `ExecutionState.plan`，再统一负责逐步执行和动态 replan。

## 代码导航

- `langchain_agent_cli/agent.py`：主 Agent、运行模式、权限状态和三个子组件的调度。
- `langchain_agent_cli/subagent_tools.py`：暴露给主 Agent 的 Explore、Plan、Execute 工具。
- `langchain_agent_cli/explore_agent.py`：只读 Explore 子 Agent 提示词。
- `langchain_agent_cli/plan_agent.py`：只读 Plan 子 Agent 提示词。
- `langchain_agent_cli/execution_workflow.py`：合并后的 Execute 部分，内部包含
  execute/replan LangGraph 循环。
- `langchain_agent_cli/tools.py`：本地文件工具及 Plan Mode 写权限保护。
- `langchain_agent_cli/cli.py`：终端循环和斜杠命令。

## 运行

```powershell
cd E:\simple-python-agent\langgraph-plan-python-agent
python -m pip install -r requirements.txt
python -m langchain_agent_cli
```

项目目录加入 `PATH` 后，也可以直接运行：

```powershell
langgraph-plan-python-agent
```

Plan Mode 示例：

```text
/plan
描述任务
/approve
```

如果计划需要修改，执行 `/reject`，然后直接输入修改意见。当前计划会被保留给主
Agent 作为修改基础。

## 测试

```powershell
python -m unittest discover -s tests -v
```

## 节点分类 GNN（TensorFlow）

`gnn_node_classifier` 是一个独立、自包含的图神经网络节点分类模块，使用
TensorFlow 2 + Keras 自定义层实现经典 2 层 GCN：

```text
GCNLayer(16, ReLU) -> Dropout(0.5) -> GCNLayer(num_classes) -> log_softmax
```

核心实现不依赖 PyTorch/PyG 或 Spektral 等第三方 GNN 库，消息传递
`A_hat @ X @ W` 在 Keras 自定义层 `GCNLayer` 中完成。

### 安装

```powershell
python -m pip install -r requirements-gnn.txt
```

默认安装 CPU 版 TensorFlow 即可完成 Cora 训练。如需 GPU 加速，请按 TensorFlow
官方文档安装对应的 CUDA/cuDNN 环境。

### 运行

```powershell
python -m gnn_node_classifier
```

首次运行会自动下载 Cora 数据集（约几十 MB）并解压到 `data/cora/`；若网络不可用，
也可手动从 `https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz` 下载并放置
`cora.content` 与 `cora.cites` 到 `data/cora/`。

常用参数：

```powershell
python -m gnn_node_classifier --epochs 200 --hidden 16 --lr 0.01 --seed 42 --data-dir data/cora
```

训练结束后会打印最终训练/验证/测试准确率，以及验证集最优 epoch 对应的测试准确率。
在 Cora 数据集上测试准确率通常约为 0.8。

### GNN 测试

GNN 测试使用小型合成图，完全离线、无需下载 Cora；未安装 TensorFlow 时这些测试会
自动跳过：

```powershell
python -m unittest tests.test_gnn_node_classifier -v
```
