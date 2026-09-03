# 计划：将 GNN 节点分类代码从 TensorFlow/Keras 迁移到 PyTorch

## 目标与范围

当前目录下唯一涉及神经网络的代码是 `gnn_node_classifier` 模块（2 层 GCN 节点分类器），
目前使用 TensorFlow 2 + Keras 自定义层实现。本计划将其完整迁移为 PyTorch 实现，
保持现有 CLI 接口、训练流程、数据加载逻辑与测试目标不变。

迁移范围：

- `gnn_node_classifier/model.py`：Keras 自定义层/模型 → `torch.nn.Module`
- `gnn_node_classifier/train.py`：TensorFlow 训练/评估 → PyTorch autograd / optimizer
- `gnn_node_classifier/data.py`：数据逻辑本身不依赖 TF，仅更新文档注释
- `gnn_node_classifier/__main__.py`：更新命令行标题/说明文本
- `gnn_node_classifier/__init__.py`：更新模块 docstring
- `requirements-gnn.txt`：将 `tensorflow` 替换为 `torch`
- `tests/test_gnn_node_classifier.py`：改写为 PyTorch 版本
- `README.md`：将 GNN 章节从 TensorFlow 描述改为 PyTorch

不修改：`main.py`、`langchain_agent_cli/*`（这些是 Agent 框架，不属于神经网络代码）。

---

## 1. `gnn_node_classifier/model.py`

全部重写为 PyTorch：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, activation=None, use_bias=True):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.use_bias = bool(use_bias)
        self.activation = activation

        self.weight = nn.Parameter(torch.empty(self.in_features, self.out_features))
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(self.out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)   # 等价于 Keras glorot_uniform
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adjacency):
        if adjacency.is_sparse:
            support = torch.sparse.mm(adjacency, x)
        else:
            support = adjacency @ x
        output = support @ self.weight
        if self.bias is not None:
            output = output + self.bias
        if self.activation is not None:
            output = self.activation(output)
        return output


class GCN(nn.Module):
    def __init__(self, in_features, num_classes, hidden_channels=16, dropout=0.5):
        super().__init__()
        self.conv1 = GCNLayer(in_features, hidden_channels, activation=F.relu)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = GCNLayer(hidden_channels, num_classes)

    def forward(self, x, adjacency):
        x = self.conv1(x, adjacency)
        x = self.dropout(x)
        x = self.conv2(x, adjacency)
        return F.log_softmax(x, dim=-1)
```

要点：

- PyTorch 需要在构造时明确 `in_features`，因此 `GCN` 签名新增 `in_features` 参数。
- `nn.Dropout` 随 `model.train()` / `model.eval()` 自动开关，不再传入 `training` 参数。
- 稀疏邻接矩阵用 `torch.sparse.mm` 做 `A_hat @ X`，稠密矩阵直接用 `@`。
- 激活函数用 `F.relu`；初始化用 `xavier_uniform_` 对齐原 `glorot_uniform`。

---

## 2. `gnn_node_classifier/train.py`

改为 PyTorch 训练逻辑，保持 `train_one_epoch` / `evaluate` / `train_model` 三个公开函数签名不变。

### 张量转换

```python
def _to_feature_tensor(features):
    if isinstance(features, torch.Tensor):
        return features.float()
    return torch.as_tensor(features, dtype=torch.float32)


def _to_adjacency_tensor(adjacency):
    if isinstance(adjacency, torch.Tensor):
        return adjacency.coalesce().float() if adjacency.is_sparse else adjacency.float()
    if sp.issparse(adjacency):
        adjacency = adjacency.tocoo()
        indices = torch.from_numpy(
            np.vstack([adjacency.row, adjacency.col]).astype(np.int64)
        )
        values = torch.from_numpy(adjacency.data.astype(np.float32))
        return torch.sparse_coo_tensor(
            indices, values, size=tuple(adjacency.shape)
        ).coalesce()
    return torch.as_tensor(adjacency, dtype=torch.float32)
```

### 训练一步

```python
def train_one_epoch(model, data, optimizer, loss_fn, weight_decay=0.0):
    model.train()
    features = _to_feature_tensor(data["features"])
    adjacency = _to_adjacency_tensor(data["adjacency"])
    labels = torch.as_tensor(data["labels"], dtype=torch.long)
    train_mask = torch.as_tensor(data["train_mask"], dtype=torch.bool)

    optimizer.zero_grad()
    logits = model(features, adjacency)
    loss = loss_fn(logits[train_mask], labels[train_mask])

    if weight_decay > 0.0:
        # 对齐原实现：仅对核权重（2 维参数）施加 L2 惩罚，tf.nn.l2_loss = 0.5 * sum(w^2)
        l2_penalty = 0.5 * sum(p.square().sum() for p in model.parameters() if p.dim() >= 2)
        loss = loss + weight_decay * l2_penalty

    loss.backward()
    optimizer.step()
    return float(loss.item())
```

### 评估

```python
@torch.no_grad()
def evaluate(model, data):
    model.eval()
    features = _to_feature_tensor(data["features"])
    adjacency = _to_adjacency_tensor(data["adjacency"])
    labels = torch.as_tensor(data["labels"], dtype=torch.long)

    logits = model(features, adjacency)
    predictions = logits.argmax(dim=-1)

    accuracies = []
    for mask_name in ("train_mask", "val_mask", "test_mask"):
        mask = torch.as_tensor(data[mask_name], dtype=torch.bool)
        acc = (predictions[mask] == labels[mask]).float().mean().item()
        accuracies.append(acc)
    return tuple(accuracies)
```

### 训练入口

```python
def train_model(data, *, hidden_channels=16, lr=0.01, weight_decay=5e-4,
                epochs=200, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    in_features = int(np.asarray(data["features"]).shape[-1])
    num_classes = int(data["num_classes"])
    model = GCN(in_features=in_features, num_classes=num_classes,
                hidden_channels=hidden_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = F.cross_entropy   # 等价于 SparseCategoricalCrossentropy(from_logits=True)
    # ... 其余 epoch 循环、best_epoch 选择与 metrics 结构保持不变
```

其余 `train_model` 的历史记录、`best_val_acc`/`best_test_acc`、返回 `(model, metrics)` 逻辑保持原样。

---

## 3. `gnn_node_classifier/data.py`

数据解析与邻接矩阵归一化逻辑完全不变。仅更新文件顶部 docstring 和 `load_cora` 的
docstring：把“可转换为 `tf.sparse.SparseTensor`”改为“可转换为
`torch.sparse_coo_tensor`”，去掉 TF 相关措辞。

---

## 4. `gnn_node_classifier/__main__.py`

- 顶部 docstring 保持运行示例不变。
- 将第 60 行标题 `GCN Node Classifier (TensorFlow 2 + Keras)` 改为
  `GCN Node Classifier (PyTorch)`。
- 其余 argparse、数据统计输出、训练调用与结果打印逻辑不变。

---

## 5. `gnn_node_classifier/__init__.py`

将模块 docstring 从 `TensorFlow 2 + Keras` 改为 `PyTorch`；导出的
`GCN`、`GCNLayer`、`load_cora`、`train_one_epoch`、`evaluate`、`train_model`
保持不变。

---

## 6. `requirements-gnn.txt`

改为：

```
-r requirements.txt
torch>=2.0
numpy>=1.23
scipy>=1.10
```

（移除 `tensorflow>=2.12`。）

---

## 7. `tests/test_gnn_node_classifier.py`

改写为 PyTorch 版本，测试场景保持不变：

- `try/except ImportError` 改为导入 `torch` 与
  `from gnn_node_classifier import GCN, evaluate, train_one_epoch`，以及
  `from gnn_node_classifier.data import normalize_adjacency`。
- `_make_synthetic_data()` 保持不变（仍返回 5 节点、8 特征、7 类的稠密邻接矩阵）。
- `setUp`：`torch.manual_seed(0)`。
- `test_forward_output_shape`：`model = GCN(in_features=8, num_classes=7)`，
  `model.eval()` 后调用 `model(features, adjacency)`，断言 `log_probs.shape == (5, 7)`。
- `test_output_is_log_probability`：`model.eval()`，
  `probabilities = torch.sum(torch.exp(log_probs), dim=1)`，
  `np.testing.assert_allclose(probabilities.numpy(), np.ones(5), atol=1e-5)`。
- `test_train_step_loss_is_finite`：`optimizer = torch.optim.Adam(model.parameters(), lr=0.01)`，
  `loss_fn = F.cross_entropy`，循环 3 次调用 `train_one_epoch(model, self.data, optimizer, loss_fn)`，
  用 `math.isfinite(loss)` 断言。
- `test_evaluate_returns_accuracy_range`：`GCN(in_features=8, num_classes=7)`，
  断言三个准确率都在 `[0, 1]`。
- `test_normalize_adjacency_shape_and_nonzero_rows`：保持不变（`data.py` 仍返回 scipy CSR）。
- skip 文案改为 `"PyTorch is not installed; skipping GNN tests."`。

---

## 8. `README.md`

更新 “节点分类 GNN（TensorFlow）” 章节：

- 标题改为 “节点分类 GNN（PyTorch）”。
- 正文把 “TensorFlow 2 + Keras 自定义层” 改为 “PyTorch `nn.Module`”。
- 安装说明改为 `pip install -r requirements-gnn.txt`，说明默认安装 CPU 版 PyTorch。
- 运行命令、参数示例、Cora 下载说明与测试命令保持不变；测试说明改为
  “未安装 PyTorch 时这些测试会自动跳过”。

---

## API 变更说明

- `GCN(num_classes=..., hidden_channels=...)` →
  `GCN(in_features, num_classes, hidden_channels=..., dropout=...)`。
  因为 PyTorch 层需要显式输入维度，调用方需提供 `in_features`（`train_model` 与测试均已同步更新）。
- 模型调用从 `model((x, adjacency), training=...)` →
  `model(x, adjacency)`，dropout 通过 `model.train()` / `model.eval()` 控制。
- `train_one_epoch` / `evaluate` / `train_model` 的公开签名与返回结构不变，CLI 无需改动。

---

## 验证方式

1. 安装依赖：

   ```powershell
   python -m pip install -r requirements-gnn.txt
   ```

2. 运行 GNN 单元测试（离线、无需下载 Cora）：

   ```powershell
   python -m unittest tests.test_gnn_node_classifier -v
   ```

   预期 5 个测试全部通过。

3. 运行完整测试集确认无回归：

   ```powershell
   python -m unittest discover -s tests -v
   ```

4. 端到端训练（首次运行需联网下载 Cora）：

   ```powershell
   python -m gnn_node_classifier --epochs 10 --hidden 16 --lr 0.01 --seed 42
   ```

   预期打印数据集统计、训练过程，并输出约 0.8 左右的测试准确率（少量 epoch 仅作冒烟验证）。

5. 确认代码中不再残留 TensorFlow/Keras 导入（除文档外）：

   ```powershell
   python -c "import ast, pathlib; print([p for p in pathlib.Path('gnn_node_classifier').glob('*.py') if 'tensorflow' in p.read_text() or 'keras' in p.read_text()])"
   ```

   预期输出为空列表。
