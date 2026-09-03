# LangGraph Plan Python Agent - Flask

这是 `langgraph-plan-python-agent` 的独立最小 Flask Web版本，原项目不需要修改。

## 启动

```powershell
pip install -r requirements.txt
python app.py
```

浏览器打开：

```text
http://127.0.0.1:5000
```

页面支持普通聊天、进入 Plan Mode、批准或拒绝计划、退出 Plan Mode，以及查看当前会话状态和 `plan.md`。

每个浏览器会话拥有独立的 Agent、LangGraph thread ID和：

```text
.sessions/<session_id>/plan.md
```

当前版本使用同步请求。复杂计划执行期间，页面会显示“处理中”，直到 Agent完整返回。

## JSON API

```text
POST /api/sessions
POST /api/messages
POST /api/plan
POST /api/approve
POST /api/reject
POST /api/exit-plan
GET  /api/sessions/<session_id>
```

## 横板格斗游戏

仓库内置了一个使用 `pygame-ce` 的横板格斗小游戏，纯逻辑与渲染分离：

- 纯逻辑层（`game/config.py`、`game/models.py`、`game/rules.py`、`game/world.py`、`game/ai.py`）不依赖 pygame。
- 渲染与主循环（`game/renderer.py`、`game/run.py`、`game/__main__.py`）负责绘制与输入。

### 安装与启动

```powershell
pip install -r requirements-game.txt
python -m game
```

### 键位说明

| 动作 | 按键 |
| --- | --- |
| 左移 | `A` / `←` |
| 右移 | `D` / `→` |
| 跳跃 | `W` / `↑` / `空格` |
| 攻击 | `J` / `Z` |
| 重新开始 | `R` / `回车` |
| 退出 | `ESC` |

### Python 版本说明

`pygame-ce 2.5.x` 需要 Python 3.9–3.13。若当前解释器为 Python 3.14 且没有可用的 pygame wheel，请使用 Python 3.11–3.13 创建独立 venv：

Windows（PowerShell）：

```powershell
py -0p                # 查看可用 Python 版本
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements-game.txt
python -m game
```

macOS / Linux：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-game.txt
python -m game
```

### 测试

```powershell
python -m unittest tests.test_game_logic -v
python -m unittest tests.test_ai -v
python -m unittest discover -s tests -v
```

纯逻辑与 AI 测试不依赖 pygame；运行游戏前请确认已安装 `pygame-ce`。
