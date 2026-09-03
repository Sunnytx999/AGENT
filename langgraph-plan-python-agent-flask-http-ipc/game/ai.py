"""敌人 AI 决策层。

decide 是纯函数：不修改任何角色状态，只根据当前局势返回意图。
实际把意图写回 enemy 的动作由 world.Game 负责。
"""

from __future__ import annotations

from dataclasses import dataclass

from . import config
from .models import Fighter, FighterState


@dataclass(frozen=True)
class EnemyIntent:
    """一个敌人的行动意图。

    move:   -1 向左、0 待机、1 向右
    attack: 是否发起攻击
    """

    move: int = 0
    attack: bool = False


def decide(enemy: Fighter, player: Fighter, dt: float = 0.0) -> EnemyIntent:
    """根据敌我状态决定敌人本帧意图。

    - 敌人死亡、玩家死亡或敌人处于硬直 → 待机
    - 进入攻击范围且冷却完成 → 攻击
    - 进入仇恨范围 → 向玩家移动
    - 否则 → 待机
    """

    del dt  # 当前规则与时间步长无关，保留参数以稳定接口

    if enemy.is_dead or player.is_dead:
        return EnemyIntent()
    if enemy.state == FighterState.HITSTUN:
        return EnemyIntent()

    dx = player.center_x - enemy.center_x
    distance = abs(dx)
    direction = 1 if dx > 0 else -1

    if distance <= config.ENEMY_ATTACK_RANGE and enemy.attack_cooldown <= 0.0:
        return EnemyIntent(move=0, attack=True)

    if distance <= config.ENEMY_AGGRO_RANGE:
        return EnemyIntent(move=direction, attack=False)

    return EnemyIntent()
