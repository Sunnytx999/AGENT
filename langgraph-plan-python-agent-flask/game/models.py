"""横板格斗游戏的纯数据模型。

本模块不 import pygame，所有类型可直接用于单元测试。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from . import config


class Facing(Enum):
    """朝向，其值可直接作为水平方向乘数使用。"""

    LEFT = -1
    RIGHT = 1


class FighterState(str, Enum):
    """角色当前所处的逻辑状态。"""

    IDLE = "idle"
    WALKING = "walking"
    JUMPING = "jumping"
    ATTACKING = "attacking"
    HITSTUN = "hitstun"
    DEAD = "dead"


class GameStatus(str, Enum):
    """整局游戏状态。"""

    RUNNING = "running"
    WIN = "win"
    LOSE = "lose"


@dataclass
class AABB:
    """轴对齐包围盒。"""

    x: float
    y: float
    w: float
    h: float

    @property
    def left(self) -> float:
        return self.x

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def top(self) -> float:
        return self.y

    @property
    def bottom(self) -> float:
        return self.y + self.h


@dataclass
class Fighter:
    """可战斗角色（玩家或敌人）。

    x/y 表示角色碰撞盒左上角位置；y 轴向下为正。
    """

    x: float
    y: float
    width: float
    height: float
    max_hp: int

    vx: float = 0.0
    vy: float = 0.0
    facing: Facing = Facing.RIGHT
    state: FighterState = FighterState.IDLE
    hp: int | None = None
    on_ground: bool = False

    attack_timer: float = 0.0     # 当前挥击伤害窗口剩余时间
    attack_cooldown: float = 0.0  # 距离下次可攻击的剩余时间
    stun_timer: float = 0.0       # 受击硬直剩余时间
    swing_id: int = 0             # 当前挥击唯一标识
    hit_targets: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        if self.hp is None:
            self.hp = self.max_hp

    @property
    def is_dead(self) -> bool:
        return self.hp is not None and self.hp <= 0

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2

    @property
    def aabb(self) -> AABB:
        return AABB(self.x, self.y, self.width, self.height)


@dataclass
class PlayerInput:
    """一帧内的玩家输入快照。

    jump 与 attack 表示“按下”动作，应由输入层在按下事件的当帧置真；
    是否消费该边缘触发由 world 层决定。
    """

    left: bool = False
    right: bool = False
    jump: bool = False
    attack: bool = False


@dataclass
class WorldState:
    """可序列化/可断言的游戏世界快照。"""

    player: Fighter
    enemies: list[Fighter] = field(default_factory=list)
    camera_x: float = 0.0
    status: GameStatus = GameStatus.RUNNING
    time: float = 0.0


def new_player() -> Fighter:
    """按配置创建位于出生点的玩家。"""

    return Fighter(
        x=config.PLAYER_START_X,
        y=config.GROUND_HEIGHT - config.PLAYER_HEIGHT,
        width=config.PLAYER_WIDTH,
        height=config.PLAYER_HEIGHT,
        max_hp=config.PLAYER_MAX_HP,
        facing=Facing.RIGHT,
    )


def new_enemy(x: float, facing: Facing = Facing.LEFT) -> Fighter:
    """按配置在指定 x 位置创建站在地面上的敌人。"""

    return Fighter(
        x=x,
        y=config.GROUND_HEIGHT - config.ENEMY_HEIGHT,
        width=config.ENEMY_WIDTH,
        height=config.ENEMY_HEIGHT,
        max_hp=config.ENEMY_MAX_HP,
        facing=facing,
    )
