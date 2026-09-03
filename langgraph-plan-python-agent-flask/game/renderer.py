"""Pygame 渲染层。

使用色块绘制角色、地面、血条 HUD 以及 WIN/LOSE 覆盖层。
本模块是唯一直接依赖 pygame 的渲染入口之一。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pygame

from . import config
from .models import Fighter, FighterState, Facing, GameStatus

if TYPE_CHECKING:
    from .world import Game

# 颜色常量
BACKGROUND_COLOR = (30, 30, 42)
GROUND_COLOR = (58, 110, 62)
GROUND_EDGE_COLOR = (120, 200, 120)
PLAYER_COLOR = (64, 120, 255)
ENEMY_COLOR = (232, 64, 64)
HIT_COLOR = (255, 255, 255)
OUTLINE_COLOR = (15, 15, 18)
HP_BACK_COLOR = (40, 40, 40)
HP_FRAME_COLOR = (90, 90, 90)
HP_PLAYER_COLOR = (90, 210, 90)
HP_ENEMY_COLOR = (230, 90, 90)
TEXT_COLOR = (255, 255, 255)
OVERLAY_COLOR = (0, 0, 0, 150)


class Renderer:
    """负责把一局 Game 的世界状态绘制到 pygame Surface 上。"""

    def __init__(self, screen: pygame.Surface) -> None:
        self.screen = screen
        self.small_font = pygame.font.Font(None, 24)
        self.font = pygame.font.Font(None, 36)
        self.big_font = pygame.font.Font(None, 96)

    def draw(self, game: "Game") -> None:
        """绘制当前帧。"""

        self.screen.fill(BACKGROUND_COLOR)
        self._draw_ground(game.camera_x)
        for enemy in game.enemies:
            self._draw_fighter(enemy, ENEMY_COLOR, game.camera_x)
        self._draw_fighter(game.player, PLAYER_COLOR, game.camera_x)
        self._draw_hud(game)

        if game.status != GameStatus.RUNNING:
            self._draw_overlay(game.status)

    # ------------------------------------------------------------------
    # 内部绘制
    # ------------------------------------------------------------------

    def _draw_ground(self, camera_x: float) -> None:
        ground_rect = pygame.Rect(
            0,
            int(config.GROUND_HEIGHT),
            config.SCREEN_WIDTH,
            config.SCREEN_HEIGHT - int(config.GROUND_HEIGHT),
        )
        pygame.draw.rect(self.screen, GROUND_COLOR, ground_rect)
        pygame.draw.line(
            self.screen,
            GROUND_EDGE_COLOR,
            (0, int(config.GROUND_HEIGHT)),
            (config.SCREEN_WIDTH, int(config.GROUND_HEIGHT)),
            3,
        )

    def _draw_fighter(
        self,
        fighter: Fighter,
        base_color: tuple[int, int, int],
        camera_x: float,
    ) -> None:
        screen_x = fighter.x - camera_x
        rect = pygame.Rect(
            int(screen_x),
            int(fighter.y),
            int(fighter.width),
            int(fighter.height),
        )

        color = HIT_COLOR if fighter.state == FighterState.HITSTUN else base_color
        pygame.draw.rect(self.screen, color, rect)
        pygame.draw.rect(self.screen, OUTLINE_COLOR, rect, 2)

        # 朝向指示条，方便观察角色面朝方向。
        eye_w = max(6, int(fighter.width * 0.16))
        eye_h = max(6, int(fighter.height * 0.10))
        eye_y = int(fighter.y + fighter.height * 0.28)
        if fighter.facing == Facing.RIGHT:
            eye_x = int(screen_x + fighter.width - eye_w - 4)
        else:
            eye_x = int(screen_x + 4)
        pygame.draw.rect(
            self.screen,
            OUTLINE_COLOR,
            pygame.Rect(eye_x, eye_y, eye_w, eye_h),
        )

    def _draw_hud(self, game: "Game") -> None:
        self._draw_hp_bar(
            20,
            20,
            280,
            18,
            game.player.hp or 0,
            game.player.max_hp,
            HP_PLAYER_COLOR,
            "PLAYER",
        )

        for index, enemy in enumerate(game.enemies):
            y = 20 + index * 34
            self._draw_hp_bar(
                config.SCREEN_WIDTH - 240,
                y,
                220,
                14,
                enemy.hp or 0,
                enemy.max_hp,
                HP_ENEMY_COLOR,
                f"ENEMY {index + 1}",
            )

    def _draw_hp_bar(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        value: int,
        max_value: int,
        color: tuple[int, int, int],
        label: str,
    ) -> None:
        label_surface = self.small_font.render(label, True, TEXT_COLOR)
        self.screen.blit(label_surface, (x, y - 20))

        frame = pygame.Rect(x - 2, y - 2, width + 4, height + 4)
        pygame.draw.rect(self.screen, HP_FRAME_COLOR, frame)

        back = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, HP_BACK_COLOR, back)

        ratio = max(0.0, min(1.0, value / max_value)) if max_value > 0 else 0.0
        fill = pygame.Rect(x, y, int(width * ratio), height)
        pygame.draw.rect(self.screen, color, fill)

    def _draw_overlay(self, status: GameStatus) -> None:
        overlay = pygame.Surface(
            (config.SCREEN_WIDTH, config.SCREEN_HEIGHT),
            pygame.SRCALPHA,
        )
        overlay.fill(OVERLAY_COLOR)
        self.screen.blit(overlay, (0, 0))

        text = "WIN" if status == GameStatus.WIN else "LOSE"
        title_surface = self.big_font.render(text, True, (255, 220, 60))
        title_rect = title_surface.get_rect(
            center=(config.SCREEN_WIDTH // 2, config.SCREEN_HEIGHT // 2 - 30)
        )
        self.screen.blit(title_surface, title_rect)

        hint_surface = self.font.render(
            "Press R to restart / ESC to quit",
            True,
            TEXT_COLOR,
        )
        hint_rect = hint_surface.get_rect(
            center=(config.SCREEN_WIDTH // 2, config.SCREEN_HEIGHT // 2 + 40)
        )
        self.screen.blit(hint_surface, hint_rect)
