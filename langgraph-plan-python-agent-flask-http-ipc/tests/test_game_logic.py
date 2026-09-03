"""纯逻辑层单元测试。

覆盖玩家移动/跳跃、攻击与冷却/去重、镜头 clamp 以及胜负判定。
这些测试不依赖 pygame，可在任何标准 Python 环境运行。
"""

from __future__ import annotations

import unittest

from game import config
from game.models import Facing, FighterState, GameStatus, PlayerInput
from game.rules import check_win_lose, update_camera_x
from game.world import Game

DT = 1.0 / config.FPS


class GameLogicTests(unittest.TestCase):
    def test_move_left_and_right_stays_in_bounds(self):
        game = Game(enemy_positions=[])

        for _ in range(600):
            game.update(DT, PlayerInput(right=True))
        self.assertEqual(
            game.player.x,
            config.LEVEL_WIDTH - config.PLAYER_WIDTH,
        )
        self.assertEqual(game.player.facing, Facing.RIGHT)

        for _ in range(600):
            game.update(DT, PlayerInput(left=True))
        self.assertEqual(game.player.x, 0.0)
        self.assertEqual(game.player.facing, Facing.LEFT)

    def test_jump_sets_upward_velocity_and_lands_on_ground(self):
        game = Game(enemy_positions=[])
        start_y = game.player.y

        game.update(DT, PlayerInput(jump=True))

        self.assertLess(game.player.vy, 0.0)
        self.assertFalse(game.player.on_ground)
        self.assertEqual(game.player.state, FighterState.JUMPING)

        for _ in range(120):
            game.update(DT)
            if game.player.on_ground:
                break

        self.assertTrue(game.player.on_ground)
        self.assertEqual(game.player.vy, 0.0)
        self.assertEqual(game.player.y, start_y)

    def test_attack_damages_enemy_and_respects_cooldown(self):
        game = Game(enemy_positions=[200.0])
        enemy = game.enemies[0]

        game.update(DT, PlayerInput(attack=True))

        self.assertEqual(
            enemy.hp,
            config.ENEMY_MAX_HP - config.PLAYER_ATTACK_DAMAGE,
        )
        self.assertEqual(game.player.swing_id, 1)
        self.assertEqual(enemy.state, FighterState.HITSTUN)

        # 让攻击窗口结束但冷却尚未结束。
        for _ in range(10):
            game.update(DT)

        swing_before = game.player.swing_id
        hp_before = enemy.hp
        self.assertGreater(game.player.attack_cooldown, 0.0)

        game.update(DT, PlayerInput(attack=True))

        # 冷却期内无法再次挥击，也不会额外扣血。
        self.assertEqual(game.player.swing_id, swing_before)
        self.assertEqual(enemy.hp, hp_before)

    def test_single_swing_hits_enemy_only_once(self):
        game = Game(enemy_positions=[200.0])
        enemy = game.enemies[0]

        game.update(DT, PlayerInput(attack=True))
        hp_after_first_hit = enemy.hp

        # 同一挥击窗口仍有效且攻击盒仍重叠，但去重后不再扣血。
        game.update(DT)
        self.assertEqual(enemy.hp, hp_after_first_hit)

    def test_camera_follows_right_and_clamps_to_level_end(self):
        self.assertEqual(
            update_camera_x(0.0, config.SCREEN_WIDTH, config.LEVEL_WIDTH),
            0.0,
        )
        self.assertEqual(
            update_camera_x(
                config.LEVEL_WIDTH,
                config.SCREEN_WIDTH,
                config.LEVEL_WIDTH,
            ),
            config.LEVEL_WIDTH - config.SCREEN_WIDTH,
        )

        game = Game(enemy_positions=[])
        for _ in range(600):
            game.update(DT, PlayerInput(right=True))

        self.assertEqual(
            game.camera_x,
            config.LEVEL_WIDTH - config.SCREEN_WIDTH,
        )

    def test_win_when_all_enemies_are_dead(self):
        game = Game(enemy_positions=[200.0])
        game.enemies[0].hp = 0

        game.update(0.0)

        self.assertTrue(game.enemies[0].is_dead)
        self.assertEqual(game.status, GameStatus.WIN)

    def test_lose_when_player_hp_is_zero(self):
        game = Game(enemy_positions=[200.0])
        game.player.hp = 0

        game.update(0.0)

        self.assertEqual(game.status, GameStatus.LOSE)
        self.assertEqual(check_win_lose(game.player, game.enemies), GameStatus.LOSE)


if __name__ == "__main__":
    unittest.main()
