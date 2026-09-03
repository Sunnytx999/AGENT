"""横板格斗游戏的世界状态更新编排。

Game.update 是纯逻辑核心：读取 PlayerInput，推进物理、攻击判定、镜头与
胜负状态。本模块同样不 import pygame。
"""

from __future__ import annotations

from . import config
from .ai import decide
from .models import (
    Facing,
    Fighter,
    FighterState,
    GameStatus,
    PlayerInput,
    WorldState,
    new_enemy,
    new_player,
)
from .rules import (
    active_attack_box,
    apply_damage,
    aabb_overlap,
    check_win_lose,
    clamp,
    update_camera_x,
)

# 击退/硬直期间水平速度的衰减系数（1/秒）。
_KNOCKBACK_DAMPING = 8.0


class Game:
    """一局横板格斗游戏的完整逻辑状态。"""

    def __init__(self, enemy_positions: list[float] | None = None) -> None:
        if enemy_positions is None:
            enemy_positions = [700.0, 1150.0, 1700.0]
        self._enemy_positions = [float(x) for x in enemy_positions]

        self.player: Fighter
        self.enemies: list[Fighter]
        self.camera_x: float
        self.status: GameStatus
        self.time: float
        self._next_swing_id: int

        self.reset()

    def reset(self) -> None:
        """恢复初始玩家、敌人、镜头与胜负状态。"""

        self.player = new_player()
        self.enemies = [new_enemy(x) for x in self._enemy_positions]
        self.camera_x = 0.0
        self.status = GameStatus.RUNNING
        self.time = 0.0
        self._next_swing_id = 1

    def snapshot(self) -> WorldState:
        """返回当前世界快照（渲染与测试均可用）。"""

        return WorldState(
            player=self.player,
            enemies=list(self.enemies),
            camera_x=self.camera_x,
            status=self.status,
            time=self.time,
        )

    def update(
        self,
        dt: float,
        player_input: PlayerInput | None = None,
    ) -> None:
        """按给定时间步长推进一帧。

        游戏结束后（WIN/LOSE）直接返回，不再推进。
        """

        dt = max(0.0, dt)
        player_input = player_input or PlayerInput()

        if self.status != GameStatus.RUNNING:
            return

        self.time += dt
        self._tick_timers(dt)
        self._handle_player_input(player_input)
        self._update_enemy_ai()

        self._apply_physics(self.player, dt)
        for enemy in self.enemies:
            self._apply_physics(enemy, dt)

        self._resolve_attacks()
        self._update_states()

        self.camera_x = update_camera_x(
            self.player.center_x,
            config.SCREEN_WIDTH,
            config.LEVEL_WIDTH,
        )
        self.status = check_win_lose(self.player, self.enemies)

    # ------------------------------------------------------------------
    # 内部步骤
    # ------------------------------------------------------------------

    def _tick_timers(self, dt: float) -> None:
        for fighter in [self.player, *self.enemies]:
            fighter.attack_timer = max(0.0, fighter.attack_timer - dt)
            fighter.attack_cooldown = max(0.0, fighter.attack_cooldown - dt)
            fighter.stun_timer = max(0.0, fighter.stun_timer - dt)

    def _handle_player_input(self, player_input: PlayerInput) -> None:
        player = self.player

        if player.state in (FighterState.HITSTUN, FighterState.DEAD):
            return

        if player_input.attack and player.attack_cooldown <= 0.0:
            self._start_attack(player, is_player=True)
            return

        if player.state == FighterState.ATTACKING:
            player.vx = 0.0
            return

        if player_input.left and not player_input.right:
            player.vx = -config.MOVE_SPEED
            player.facing = Facing.LEFT
        elif player_input.right and not player_input.left:
            player.vx = config.MOVE_SPEED
            player.facing = Facing.RIGHT
        else:
            player.vx = 0.0

        player.state = (
            FighterState.WALKING if player.on_ground else FighterState.JUMPING
        )

        if player_input.jump and player.on_ground:
            player.vy = config.JUMP_VELOCITY
            player.on_ground = False
            player.state = FighterState.JUMPING

    def _update_enemy_ai(self) -> None:
        """把每个存活敌人的 AI 意图写到 vx、朝向与攻击状态。"""

        for enemy in self.enemies:
            if enemy.is_dead:
                continue

            # 始终面向玩家，保证移动方向与攻击盒方向正确。
            enemy.facing = (
                Facing.RIGHT
                if enemy.center_x <= self.player.center_x
                else Facing.LEFT
            )

            intent = decide(enemy, self.player)

            if intent.attack:
                if enemy.attack_cooldown <= 0.0:
                    self._start_attack(enemy, is_player=False)
                continue

            # 硬直与正在挥击的敌人不响应移动意图。
            if enemy.state in (FighterState.HITSTUN, FighterState.ATTACKING):
                continue

            enemy.vx = intent.move * config.MOVE_SPEED

            if enemy.on_ground:
                enemy.state = (
                    FighterState.WALKING
                    if intent.move != 0
                    else FighterState.IDLE
                )
            else:
                enemy.state = FighterState.JUMPING

    def _start_attack(self, fighter: Fighter, *, is_player: bool) -> None:
        if is_player:
            active = config.PLAYER_ATTACK_ACTIVE
            cooldown = config.PLAYER_ATTACK_COOLDOWN
        else:
            active = config.ENEMY_ATTACK_ACTIVE
            cooldown = config.ENEMY_ATTACK_COOLDOWN

        fighter.attack_timer = active
        fighter.attack_cooldown = cooldown
        fighter.swing_id = self._next_swing_id
        self._next_swing_id += 1
        fighter.hit_targets.clear()
        fighter.state = FighterState.ATTACKING
        fighter.vx = 0.0

    def _apply_physics(self, fighter: Fighter, dt: float) -> None:
        # 硬直与死亡状态没有主动控制，水平速度逐渐衰减到 0。
        if fighter.state in (FighterState.HITSTUN, FighterState.DEAD):
            fighter.vx *= max(0.0, 1.0 - _KNOCKBACK_DAMPING * dt)

        fighter.vy += config.GRAVITY * dt
        if fighter.vy > config.MAX_FALL_SPEED:
            fighter.vy = config.MAX_FALL_SPEED

        fighter.x += fighter.vx * dt
        fighter.y += fighter.vy * dt

        ground_y = config.GROUND_HEIGHT - fighter.height
        if fighter.y >= ground_y:
            fighter.y = ground_y
            fighter.vy = 0.0
            fighter.on_ground = True
        else:
            fighter.on_ground = False

        fighter.x = clamp(fighter.x, 0.0, config.LEVEL_WIDTH - fighter.width)

    def _resolve_attacks(self) -> None:
        fighters: list[Fighter] = [self.player, *self.enemies]

        for attacker in fighters:
            if attacker.is_dead or attacker.attack_timer <= 0.0:
                continue

            if attacker is self.player:
                attack_range = config.PLAYER_ATTACK_RANGE
                attack_height = config.PLAYER_HEIGHT
                damage = config.PLAYER_ATTACK_DAMAGE
                stun = config.PLAYER_ATTACK_STUN
                knockback = config.PLAYER_ATTACK_KNOCKBACK
                targets = self.enemies
            else:
                attack_range = config.ENEMY_ATTACK_RANGE
                attack_height = config.ENEMY_HEIGHT
                damage = config.ENEMY_ATTACK_DAMAGE
                stun = config.ENEMY_ATTACK_STUN
                knockback = config.ENEMY_ATTACK_KNOCKBACK
                targets = [self.player]

            box = active_attack_box(attacker, attack_range, attack_height)
            if box is None:
                continue

            for target in targets:
                target_key = id(target)
                if target.is_dead or target_key in attacker.hit_targets:
                    continue
                if aabb_overlap(box, target.aabb):
                    apply_damage(
                        target,
                        damage,
                        stun,
                        knockback,
                        direction=attacker.facing.value,
                    )
                    attacker.hit_targets.add(target_key)

    def _update_states(self) -> None:
        for fighter in [self.player, *self.enemies]:
            if fighter.state == FighterState.DEAD:
                continue

            if (
                fighter.state == FighterState.HITSTUN
                and fighter.stun_timer <= 0.0
            ):
                fighter.state = (
                    FighterState.IDLE
                    if fighter.on_ground
                    else FighterState.JUMPING
                )
                fighter.vx = 0.0
            elif (
                fighter.state == FighterState.ATTACKING
                and fighter.attack_timer <= 0.0
            ):
                fighter.state = (
                    FighterState.IDLE
                    if fighter.on_ground
                    else FighterState.JUMPING
                )
            elif fighter.state == FighterState.JUMPING and fighter.on_ground:
                fighter.state = FighterState.IDLE
