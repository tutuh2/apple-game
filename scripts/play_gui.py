"""フルーツボックス GUI 시연 — pygame 기반.

우리 시뮬레이터(rejection sampling 포함)를 사이트와 비슷한 비주얼로 그리고
greedy_smallest 정책이 자동으로 플레이하는 모습을 보여준다.

사용법:
    python3 scripts/play_gui.py                       # 사람이 보면서 자동 플레이
    python3 scripts/play_gui.py --seed 42 --delay 50  # 50ms 간격으로 빠르게
    python3 scripts/play_gui.py --no-auto             # 사람이 직접 마우스 드래그

키:
    SPACE  pause/resume
    R      reset (새 보드, seed 증가)
    ESC    종료

게임 끝나면 화면에 "Try Again" 버튼 표시 — 클릭 또는 R 키로 다음 보드.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pygame  # noqa: E402

from agent.heuristics import (  # noqa: E402
    find_valid_actions,
    greedy_smallest_policy,
)
from env.fruit_box import BOARD_COLS, BOARD_ROWS, Action, FruitBox  # noqa: E402


CELL_PX = 50
MARGIN = 40
SCORE_AREA_W = 120
WIDTH = MARGIN * 2 + CELL_PX * BOARD_COLS + SCORE_AREA_W
HEIGHT = MARGIN * 2 + CELL_PX * BOARD_ROWS + 60

BG = (210, 240, 200)
GRID_BG = (240, 250, 235)
APPLE_RED = (224, 76, 60)
APPLE_STEM = (90, 60, 30)
TEXT_WHITE = (255, 255, 255)
TEXT_BLACK = (40, 40, 40)
HIGHLIGHT = (255, 220, 100, 120)
TIMER_BG = (220, 230, 220)
TIMER_FILL = (140, 200, 110)
GAME_DURATION_SEC = 120.0


def _draw_apple(
    surface: pygame.Surface,
    cx: int,
    cy: int,
    radius: int,
    digit: int,
    font: pygame.font.Font,
) -> None:
    """한 셀에 빨간 사과 + 흰 숫자."""
    pygame.draw.rect(surface, APPLE_STEM, (cx - 2, cy - radius - 4, 4, 6))
    pygame.draw.circle(surface, APPLE_RED, (cx, cy), radius)
    pygame.draw.circle(
        surface,
        (255, 150, 130),
        (cx - radius // 3, cy - radius // 3),
        max(2, radius // 5),
    )
    text = font.render(str(digit), True, TEXT_WHITE)
    rect = text.get_rect(center=(cx, cy + 2))
    surface.blit(text, rect)


def _cell_to_px(r: int, c: int) -> tuple[int, int]:
    x = MARGIN + c * CELL_PX + CELL_PX // 2
    y = MARGIN + r * CELL_PX + CELL_PX // 2
    return x, y


def _px_to_cell(x: int, y: int) -> tuple[int, int] | None:
    if x < MARGIN or y < MARGIN:
        return None
    c = (x - MARGIN) // CELL_PX
    r = (y - MARGIN) // CELL_PX
    if 0 <= r < BOARD_ROWS and 0 <= c < BOARD_COLS:
        return int(r), int(c)
    return None


def _draw_board(
    surface: pygame.Surface,
    game: FruitBox,
    apple_font: pygame.font.Font,
    highlight: tuple[int, int, int, int] | None = None,
    selection: tuple[int, int, int, int] | None = None,
) -> None:
    surface.fill(BG)

    grid_rect = pygame.Rect(
        MARGIN - 10,
        MARGIN - 10,
        CELL_PX * BOARD_COLS + 20,
        CELL_PX * BOARD_ROWS + 20,
    )
    pygame.draw.rect(surface, GRID_BG, grid_rect, border_radius=8)

    for r in range(BOARD_ROWS + 1):
        y = MARGIN + r * CELL_PX
        pygame.draw.line(
            surface, (200, 220, 200),
            (MARGIN, y),
            (MARGIN + CELL_PX * BOARD_COLS, y), 1,
        )
    for c in range(BOARD_COLS + 1):
        x = MARGIN + c * CELL_PX
        pygame.draw.line(
            surface, (200, 220, 200),
            (x, MARGIN),
            (x, MARGIN + CELL_PX * BOARD_ROWS), 1,
        )

    if highlight is not None:
        r1, c1, r2, c2 = highlight
        x1, y1 = _cell_to_px(r1, c1)
        x2, y2 = _cell_to_px(r2, c2)
        rect = pygame.Rect(
            x1 - CELL_PX // 2,
            y1 - CELL_PX // 2,
            x2 - x1 + CELL_PX,
            y2 - y1 + CELL_PX,
        )
        s = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        s.fill(HIGHLIGHT)
        surface.blit(s, rect.topleft)
        pygame.draw.rect(surface, (220, 150, 30), rect, 3, border_radius=4)

    radius = CELL_PX // 2 - 4
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            v = int(game.board[r, c])
            if v == 0:
                continue
            cx, cy = _cell_to_px(r, c)
            _draw_apple(surface, cx, cy, radius, v, apple_font)

    if selection is not None:
        sr1, sc1, sr2, sc2 = selection
        rr1, rr2 = sorted((sr1, sr2))
        cc1, cc2 = sorted((sc1, sc2))
        x1, y1 = _cell_to_px(rr1, cc1)
        x2, y2 = _cell_to_px(rr2, cc2)
        rect = pygame.Rect(
            x1 - CELL_PX // 2,
            y1 - CELL_PX // 2,
            x2 - x1 + CELL_PX,
            y2 - y1 + CELL_PX,
        )
        s = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        s.fill((100, 150, 255, 80))
        surface.blit(s, rect.topleft)
        pygame.draw.rect(surface, (60, 100, 220), rect, 2)


def _draw_side_panel(
    surface: pygame.Surface,
    game: FruitBox,
    elapsed_sec: float,
    steps: int,
    auto: bool,
    paused: bool,
    big_font: pygame.font.Font,
    small_font: pygame.font.Font,
) -> None:
    panel_x = MARGIN + CELL_PX * BOARD_COLS + 20

    label = small_font.render("SCORE", True, TEXT_BLACK)
    surface.blit(label, (panel_x, MARGIN))
    score_text = big_font.render(str(game.score), True, (200, 80, 40))
    surface.blit(score_text, (panel_x, MARGIN + 20))

    bar_x = panel_x + 10
    bar_y = MARGIN + 100
    bar_w = 20
    bar_h = CELL_PX * BOARD_ROWS - 100
    pygame.draw.rect(surface, TIMER_BG, (bar_x, bar_y, bar_w, bar_h))
    remaining = max(0.0, GAME_DURATION_SEC - elapsed_sec)
    fill_h = int(bar_h * remaining / GAME_DURATION_SEC)
    pygame.draw.rect(
        surface,
        TIMER_FILL,
        (bar_x, bar_y + (bar_h - fill_h), bar_w, fill_h),
    )
    t_text = small_font.render(f"{int(remaining):3d}s", True, TEXT_BLACK)
    surface.blit(t_text, (panel_x, bar_y + bar_h + 5))

    bottom_y = HEIGHT - 50
    mode = "AUTO" if auto else "MANUAL"
    if paused:
        mode += " (PAUSED)"
    surface.blit(small_font.render(mode, True, TEXT_BLACK), (MARGIN, bottom_y))
    surface.blit(
        small_font.render(f"step {steps}", True, TEXT_BLACK),
        (MARGIN + 200, bottom_y),
    )


def _draw_game_over(
    surface: pygame.Surface,
    final_score: int,
    font: pygame.font.Font,
    button_font: pygame.font.Font,
) -> pygame.Rect:
    """Game Over 오버레이 + Try Again 버튼. 버튼 Rect 반환 (클릭 감지용)."""
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 140))
    surface.blit(overlay, (0, 0))

    text = font.render(f"Game Over — {final_score}", True, (255, 255, 255))
    rect = text.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 40))
    surface.blit(text, rect)

    # Try Again 버튼
    btn_w, btn_h = 200, 56
    btn_rect = pygame.Rect(
        WIDTH // 2 - btn_w // 2,
        HEIGHT // 2 + 20,
        btn_w,
        btn_h,
    )
    pygame.draw.rect(surface, (220, 80, 60), btn_rect, border_radius=12)
    pygame.draw.rect(surface, (255, 255, 255), btn_rect, 2, border_radius=12)
    btn_text = button_font.render("Try Again", True, (255, 255, 255))
    surface.blit(btn_text, btn_text.get_rect(center=btn_rect.center))

    # 키 안내
    hint = button_font.render(
        "click button or press R", True, (220, 220, 220)
    )
    surface.blit(
        hint,
        hint.get_rect(center=(WIDTH // 2, btn_rect.bottom + 24)),
    )
    return btn_rect


class GameState:
    """한 판의 상태를 묶어둠. reset()으로 새 판 시작."""

    def __init__(self, seed: int):
        self.seed = seed
        self.game = FruitBox(seed=seed)
        self.game.reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.highlight: tuple[int, int, int, int] | None = None
        self.highlight_until = 0
        self.start_time = time.time()
        self.last_action_time = pygame.time.get_ticks()


def run(seed: int, auto: bool, delay_ms: int) -> None:
    pygame.init()
    pygame.display.set_caption("Fruit Box — greedy_smallest demo")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    apple_font = pygame.font.SysFont("Arial", 22, bold=True)
    small_font = pygame.font.SysFont("Arial", 16)
    big_font = pygame.font.SysFont("Arial", 36, bold=True)
    over_font = pygame.font.SysFont("Arial", 42, bold=True)
    btn_font = pygame.font.SysFont("Arial", 22, bold=True)

    state = GameState(seed)
    paused = False
    drag_start: tuple[int, int] | None = None
    drag_now: tuple[int, int] | None = None
    try_again_btn: pygame.Rect | None = None  # game-over 시에만 활성

    def reset_to_next() -> None:
        nonlocal state, drag_start, drag_now
        next_seed = state.seed + 1
        print(f"[gui] reset → seed={next_seed}")
        state = GameState(next_seed)
        drag_start = None
        drag_now = None

    clock = pygame.time.Clock()
    running = True
    while running:
        elapsed = time.time() - state.start_time
        game_over = (
            not find_valid_actions(state.game)
            or elapsed >= GAME_DURATION_SEC
        )

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    reset_to_next()
                    continue
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # 게임 끝 + 버튼 영역 클릭 → 리셋 (auto/manual 무관)
                if game_over and try_again_btn is not None and try_again_btn.collidepoint(event.pos):
                    reset_to_next()
                    continue
                # manual 모드 드래그 시작
                if not auto and not game_over:
                    cell = _px_to_cell(*event.pos)
                    if cell is not None:
                        drag_start = cell
                        drag_now = cell
            elif not auto and event.type == pygame.MOUSEMOTION and drag_start:
                cell = _px_to_cell(*event.pos)
                if cell is not None:
                    drag_now = cell
            elif not auto and event.type == pygame.MOUSEBUTTONUP and drag_start:
                if drag_now is not None:
                    r1, c1 = drag_start
                    r2, c2 = drag_now
                    r1, r2 = sorted((r1, r2))
                    c1, c2 = sorted((c1, c2))
                    try:
                        action = Action(r1, c1, r2, c2)
                        reward = state.game.step(action)
                        if reward > 0:
                            state.highlight = (r1, c1, r2, c2)
                            state.highlight_until = pygame.time.get_ticks() + 200
                            state.steps += 1
                    except ValueError:
                        pass
                drag_start = None
                drag_now = None

        now_ms = pygame.time.get_ticks()
        if (
            auto
            and not paused
            and not game_over
            and now_ms - state.last_action_time >= delay_ms
        ):
            action = greedy_smallest_policy(state.game, state.rng)
            if action is not None:
                state.game.step(action)
                state.highlight = (action.r1, action.c1, action.r2, action.c2)
                state.highlight_until = now_ms + max(delay_ms - 10, 50)
                state.steps += 1
            state.last_action_time = now_ms

        if state.highlight is not None and now_ms > state.highlight_until:
            state.highlight = None

        selection_view = None
        if drag_start and drag_now:
            selection_view = (
                drag_start[0], drag_start[1],
                drag_now[0], drag_now[1],
            )

        _draw_board(
            screen, state.game, apple_font,
            highlight=state.highlight,
            selection=selection_view,
        )
        _draw_side_panel(
            screen, state.game, elapsed, state.steps, auto, paused,
            big_font, small_font,
        )

        if game_over:
            try_again_btn = _draw_game_over(
                screen, state.game.score, over_font, btn_font
            )
        else:
            try_again_btn = None

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


def main() -> None:
    p = argparse.ArgumentParser(description="フルーツボックス GUI 시연")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--no-auto",
        dest="auto",
        action="store_false",
        help="끄면 사람이 마우스 드래그로 직접 플레이",
    )
    p.add_argument(
        "--delay",
        type=int,
        default=200,
        help="AUTO 모드에서 한 수 사이 간격(ms)",
    )
    args = p.parse_args()
    run(seed=args.seed, auto=args.auto, delay_ms=args.delay)


if __name__ == "__main__":
    main()
