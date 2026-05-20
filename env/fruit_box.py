"""フルーツボックス 게임 시뮬레이터 (순수 룰).

게임 룰:
- 10행 × 17열 격자, 각 칸에 1~9 사과
- 직사각형 영역 선택 → 영역 내 사과 합이 정확히 10이면 제거 + 점수 += 제거된 개수
- 제거된 사과는 0으로 표시되며 복구 불가
- 더 이상 합=10인 직사각형이 없으면 종료

RL 인터페이스(Gymnasium)는 Step 2에서 별도 wrapper로 추가한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


BOARD_ROWS = 10
BOARD_COLS = 17
TARGET_SUM = 10
MIN_APPLE = 1
MAX_APPLE = 9


@dataclass(frozen=True)
class Action:
    """직사각형 영역 선택 액션.

    좌표는 0-indexed, 양쪽 포함 (r1 <= r2, c1 <= c2).
    """

    r1: int
    c1: int
    r2: int
    c2: int

    def __post_init__(self) -> None:
        if not (0 <= self.r1 <= self.r2 < BOARD_ROWS):
            raise ValueError(f"행 좌표가 잘못됨: r1={self.r1}, r2={self.r2}")
        if not (0 <= self.c1 <= self.c2 < BOARD_COLS):
            raise ValueError(f"열 좌표가 잘못됨: c1={self.c1}, c2={self.c2}")

    @property
    def size(self) -> int:
        return (self.r2 - self.r1 + 1) * (self.c2 - self.c1 + 1)


class FruitBox:
    """フルーツボックス 게임 환경.

    사용법:
        env = FruitBox(seed=42)
        env.reset()
        while not env.is_done():
            reward = env.step(Action(0, 0, 1, 2))
            print(f"점수: {env.score}")
    """

    def __init__(self, seed: Optional[int] = None):
        self._rng = np.random.default_rng(seed)
        self.board: np.ndarray = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)
        self.score: int = 0
        self._initialized = False

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """새 게임 시작. 보드를 1~9 랜덤으로 채우고 점수 0.

        진짜 ゲーム菜園 게임은 rejection sampling — 보드 합이 10의 배수가
        될 때까지 재생성. 출처: https://oshizi.com/ko/research/analyzing-board-generation/
        (500개 보드 100% 체크섬 통과).
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        # 합이 10의 배수가 될 때까지 재생성. 약 10% 확률이라 평균 ~10번 안에 끝남.
        while True:
            board = self._rng.integers(
                MIN_APPLE, MAX_APPLE + 1, size=(BOARD_ROWS, BOARD_COLS), dtype=np.int8
            )
            if int(board.sum()) % TARGET_SUM == 0:
                break
        self.board = board
        self.score = 0
        self._initialized = True
        return self.board.copy()

    def rectangle_sum(self, r1: int, c1: int, r2: int, c2: int) -> int:
        """직사각형 영역의 사과 합."""
        return int(self.board[r1 : r2 + 1, c1 : c2 + 1].sum())

    def rectangle_apple_count(self, r1: int, c1: int, r2: int, c2: int) -> int:
        """직사각형 영역의 사과(0이 아닌 칸) 개수."""
        region = self.board[r1 : r2 + 1, c1 : c2 + 1]
        return int(np.count_nonzero(region))

    def step(self, action: Action) -> int:
        """액션 실행.

        합이 정확히 10이면 영역을 0으로 만들고 보상 = 제거된 사과 수.
        아니면 보상 0, 보드 변경 없음.
        """
        if not self._initialized:
            raise RuntimeError("reset()을 먼저 호출하세요.")

        total = self.rectangle_sum(action.r1, action.c1, action.r2, action.c2)
        if total != TARGET_SUM:
            return 0

        removed = self.rectangle_apple_count(action.r1, action.c1, action.r2, action.c2)
        self.board[action.r1 : action.r2 + 1, action.c1 : action.c2 + 1] = 0
        self.score += removed
        return removed

    def is_done(self) -> bool:
        """더 이상 합=10인 직사각형이 존재하지 않으면 True.

        2D prefix sum으로 모든 직사각형의 합을 빠르게 검사한다.
        """
        if not self._initialized:
            raise RuntimeError("reset()을 먼저 호출하세요.")

        if self.board.sum() < TARGET_SUM:
            return True

        # psum[r+1][c+1] = 보드[0:r+1, 0:c+1]의 합
        psum = np.zeros((BOARD_ROWS + 1, BOARD_COLS + 1), dtype=np.int32)
        psum[1:, 1:] = self.board.cumsum(axis=0).cumsum(axis=1)

        for r1 in range(BOARD_ROWS):
            for r2 in range(r1, BOARD_ROWS):
                # 행 r1..r2 고정 시, 각 열 c에 대한 부분합 = psum[r2+1][c] - psum[r1][c]
                col_psum = psum[r2 + 1] - psum[r1]  # shape: (COLS+1,)
                # 직사각형 (r1, c1, r2, c2)의 합 = col_psum[c2+1] - col_psum[c1]
                # 즉 col_psum의 두 인덱스 차이가 TARGET_SUM이면 found
                for c1 in range(BOARD_COLS):
                    rect_sums = col_psum[c1 + 1 :] - col_psum[c1]
                    if (rect_sums == TARGET_SUM).any():
                        return False
        return True

    def render(self) -> str:
        """터미널 출력용 문자열. 0은 '.'으로 표시."""
        lines = ["   " + " ".join(f"{c:2d}" for c in range(BOARD_COLS))]
        for r in range(BOARD_ROWS):
            row_str = " ".join(
                f"{int(v):2d}" if v != 0 else " ." for v in self.board[r]
            )
            lines.append(f"{r:2d} {row_str}")
        lines.append(f"\n점수: {self.score}")
        return "\n".join(lines)
