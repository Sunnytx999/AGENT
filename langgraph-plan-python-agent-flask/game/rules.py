"""横板格斗游戏的纯函数规则层。

本模块不 import pygame，所有函数保持无副作用（除 apply_damage 会修改目标
Fighter 外），便于单元测试。
"""

from __future__ import annotations

from .models import AABB, Fighter, Facing, FighterState, GameStatus


def clamp(value: float, lo: float, hi: float) -> float:
    """将 value 限制在 [lo, hi] 区间内。"""

    return max(lo, min(hi, value))


def aabb_overlap(a: AABB, b: AABB) -> bool:
    """判断两个轴对齐包围盒是否重叠（仅接触不视为命中）。"""

    return (
        a.left < b.right
        and a.right > b.left
        and a.top < b.bottom
        and a.bottom > b.top
    )


def active_attack_box(
    fighter: Fighter,
    attack_range: float,
    attack_height: float,
) -> AABB | None:
    """按朝向生成角色前方的攻击盒。

    攻击窗口结束（attack_timer <= 0）时返回 None。
    """

    if fighter.attack_timer <= 0.0:
        return None

    x = (
        fighter.x + fighter.width
        if fighter.facing == Facing.RIGHT
        else fighter.x - attack_range
    )
    y = fighter.y + (fighter.height - attack_height) / 2.0
    return AABB(x, y, attack_range, attack_height)


def apply_damage(
    target: Fighter,
    damage: int,
    stun: float,
    knockback: float,
    direction: float,
) -> bool:
    """对目标结算一次伤害，返回是否实际造成伤害。

    direction 为击退速度的方向乘数（正值为向右）。目标死亡或已死亡时
    不重复结算并返回 False。
    """

    if target.is_dead:
        return False

    target.hp = max(0, (target.hp or 0) - damage)
    target.vx = knockback * direction
    target.attack_timer = 0.0  # 受击打断当前挥击

    if target.hp <= 0:
        target.state = FighterState.DEAD
        target.stun_timer = 0.0
    else:
        target.stun_timer = stun
        target.state = FighterState.HITSTUN

    return True


def update_camera_x(
    player_x: float,
    screen_width: int,
    level_width: float,
) -> float:
    """根据玩家跟踪点（通常传角色中心 x）计算 clamp 后的镜头 x。"""

    target = player_x - screen_width / 2.0
    return clamp(target, 0.0, max(0.0, level_width - screen_width))


def check_win_lose(
    player: Fighter,
    enemies: list[Fighter],
) -> GameStatus:
    """根据玩家与敌人血量返回当前胜负状态。"""

    if player.is_dead:
        return GameStatus.LOSE
    if enemies and all(enemy.is_dead for enemy in enemies):
        return GameStatus.WIN
    return GameStatus.RUNNING
