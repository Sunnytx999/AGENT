"""横板格斗游戏核心包。

该包将游戏拆分为纯逻辑层（不依赖 pygame）与渲染/输入层：

- config: 游戏常量
- models: 纯数据模型与枚举
- rules: 纯函数规则（碰撞、伤害、镜头、胜负）
- world: 游戏世界状态更新编排
- ai: 敌人 AI 决策
- renderer / run: Pygame 渲染与主循环
"""

from .config import FPS, SCREEN_HEIGHT, SCREEN_WIDTH

__all__ = ["FPS", "SCREEN_HEIGHT", "SCREEN_WIDTH"]
