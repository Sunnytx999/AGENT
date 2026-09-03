"""敌人 AI 单元测试。"""

from __future__ import annotations

import unittest

from game import config
from game.ai import EnemyIntent, decide
from game.models import FighterState, new_enemy, new_player
from game.world import Game

DT = 1.0 / config.FPS


class EnemyAITests(unittest.TestCase):
    def _enemy(self, x: float = 500.0):
        return new_enemy(x)

    def _player(self, x: float = 200.0):
        player = new_player()
        player.x = x
        return player

    def test_dead_enemy_returns_idle(self):
        enemy = self._enemy()
        enemy.hp = 0

        self.assertEqual(decide(enemy, self._player(), DT), EnemyIntent())

    def test_hitstun_enemy_returns_idle(self):
        enemy = self._enemy()
        enemy.state = FighterState.HITSTUN

        self.assertEqual(decide(enemy, self._player(), DT), EnemyIntent())

    def test_dead_player_makes_enemy_idle(self):
        player = self._player()
        player.hp = 0

        self.assertEqual(decide(self._enemy(), player, DT), EnemyIntent())

    def test_attack_when_in_range_and_cooldown_ready(self):
        enemy = self._enemy(300.0)
        player = self._player()
        player.x = (
            enemy.center_x
            + config.ENEMY_ATTACK_RANGE
            - 5.0
            - player.width / 2.0
        )
        enemy.attack_cooldown = 0.0

        intent = decide(enemy, player, DT)

        self.assertTrue(intent.attack)
        self.assertEqual(intent.move, 0)

    def test_no_attack_when_on_cooldown(self):
        enemy = self._enemy(300.0)
        player = self._player()
        player.x = (
            enemy.center_x
            + config.ENEMY_ATTACK_RANGE
            - 5.0
            - player.width / 2.0
        )
        enemy.attack_cooldown = 0.2

        intent = decide(enemy, player, DT)

        self.assertFalse(intent.attack)
        self.assertEqual(intent.move, 1)  # 仍在仇恨范围内，朝玩家移动

    def test_move_toward_player_in_aggro_range(self):
        enemy = self._enemy(600.0)

        to_right = decide(enemy, self._player(700.0), DT)
        self.assertEqual(to_right.move, 1)
        self.assertFalse(to_right.attack)

        to_left = decide(enemy, self._player(400.0), DT)
        self.assertEqual(to_left.move, -1)
        self.assertFalse(to_left.attack)

    def test_idle_outside_aggro_range(self):
        enemy = self._enemy(1000.0)

        self.assertEqual(decide(enemy, self._player(100.0), DT), EnemyIntent())

    def test_enemy_closes_distance_and_damages_player(self):
        game = Game(enemy_positions=[400.0])
        start_hp = game.player.hp

        for _ in range(600):
            game.update(DT)
            if game.player.hp < start_hp:
                break

        self.assertLess(game.player.hp, start_hp)

    def test_dead_enemy_does_not_move_or_attack(self):
        game = Game(enemy_positions=[400.0])
        enemy = game.enemies[0]
        enemy.hp = 0
        enemy_x = enemy.x

        game.update(DT)

        self.assertTrue(enemy.is_dead)
        self.assertEqual(enemy.x, enemy_x)
        self.assertEqual(enemy.attack_timer, 0.0)


if __name__ == "__main__":
    unittest.main()
