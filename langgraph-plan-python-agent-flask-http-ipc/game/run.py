"""Pygame 主循环与输入映射。"""

from __future__ import annotations

import pygame

from . import config
from .models import PlayerInput
from .renderer import Renderer
from .world import Game

LEFT_KEYS = {pygame.K_a, pygame.K_LEFT}
RIGHT_KEYS = {pygame.K_d, pygame.K_RIGHT}
JUMP_KEYS = {pygame.K_w, pygame.K_UP, pygame.K_SPACE}
ATTACK_KEYS = {pygame.K_j, pygame.K_z}
RESTART_KEYS = {pygame.K_r, pygame.K_RETURN}
QUIT_KEYS = {pygame.K_ESCAPE}

MAX_DT = 0.05  # 限制单帧最大步长，避免窗口拖动后物理跳变。


def main() -> None:
    """初始化 pygame 并运行游戏主循环。"""

    pygame.init()
    screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
    pygame.display.set_caption("横板格斗游戏")
    clock = pygame.time.Clock()

    renderer = Renderer(screen)
    game = Game()

    running = True
    while running:
        dt = min(clock.tick(config.FPS) / 1000.0, MAX_DT)

        left = False
        right = False
        jump_pressed = False
        attack_pressed = False
        restart = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in QUIT_KEYS:
                    running = False
                elif event.key in RESTART_KEYS:
                    restart = True
                elif event.key in JUMP_KEYS:
                    jump_pressed = True
                elif event.key in ATTACK_KEYS:
                    attack_pressed = True

        held = pygame.key.get_pressed()
        left = held[pygame.K_a] or held[pygame.K_LEFT]
        right = held[pygame.K_d] or held[pygame.K_RIGHT]

        if restart:
            game.reset()
        else:
            game.update(
                dt,
                PlayerInput(
                    left=left,
                    right=right,
                    jump=jump_pressed,
                    attack=attack_pressed,
                ),
            )

        renderer.draw(game)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
