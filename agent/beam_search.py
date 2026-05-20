"""Beam search — deterministic + perfect info 게임의 정공법.

이 게임은 RL이 본질적으로 안 맞는 도메인일 수 있다 (상대 없음, 무작위 없음).
빔서치는 매 step마다 "현재 누적 점수 상위 W개"만 유지하며 게임 끝까지 펼친다.

알고리즘:
  beam = [(board_0, score_0=0)]
  while 빔에 non-terminal이 있고:
      candidates = []
      for (board, score) in beam:
          for valid_rect in find_valid_actions(board):
              new_board, removed = step(board, valid_rect)
              if new_board not in seen:
                  candidates.append((new_board, score + removed))
      beam = top_W(candidates, key=score)
  return max(score for (_, score) in beam)

W=1이면 사실상 greedy. W가 클수록 천장 가까워짐. 시간 제한으로 컷.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from env.fruit_box import BOARD_COLS, BOARD_ROWS, TARGET_SUM


@dataclass
class BeamResult:
    """빔서치 한 판 결과."""

    best_score: int
    steps: int
    elapsed_sec: float
    timed_out: bool
    beam_final_size: int


def _find_valid_actions_array(board: np.ndarray) -> list[tuple[int, int, int, int]]:
    """현재 보드의 합=10 직사각형을 모두 반환 (2D prefix sum)."""
    psum = np.zeros((BOARD_ROWS + 1, BOARD_COLS + 1), dtype=np.int32)
    psum[1:, 1:] = board.cumsum(axis=0).cumsum(axis=1)

    results: list[tuple[int, int, int, int]] = []
    for r1 in range(BOARD_ROWS):
        for r2 in range(r1, BOARD_ROWS):
            col_psum = psum[r2 + 1] - psum[r1]
            for c1 in range(BOARD_COLS):
                rect_sums = col_psum[c1 + 1 :] - col_psum[c1]
                matching = np.where(rect_sums == TARGET_SUM)[0]
                for offset in matching:
                    c2 = c1 + int(offset)
                    results.append((r1, c1, r2, c2))
    return results


def _apply(
    board: np.ndarray, rect: tuple[int, int, int, int]
) -> tuple[np.ndarray, int]:
    """board 복사본에 직사각형 클리어. (new_board, removed) 반환."""
    r1, c1, r2, c2 = rect
    region = board[r1 : r2 + 1, c1 : c2 + 1]
    removed = int(np.count_nonzero(region))
    new_board = board.copy()
    new_board[r1 : r2 + 1, c1 : c2 + 1] = 0
    return new_board, removed


def beam_search(
    initial_board: np.ndarray,
    beam_width: int,
    time_limit_sec: Optional[float] = None,
    max_steps: int = 200,
) -> BeamResult:
    """초기 보드에서 빔 폭 W로 검색. 게임 끝나거나 시간 초과까지.

    동률은 안정 정렬: 먼저 만들어진 후보가 우선 (재현성).
    중복 보드 상태는 set으로 제거 — 같은 보드에 여러 경로로 도달하는 경우
    한 번만 유지 (이미 더 좋거나 같은 점수일 가능성 높음).
    """
    start = time.time()
    beam: list[tuple[np.ndarray, int]] = [(initial_board.copy(), 0)]
    seen: set[bytes] = {initial_board.tobytes()}
    best_score = 0
    timed_out = False
    steps = 0

    for step in range(max_steps):
        if time_limit_sec is not None and time.time() - start > time_limit_sec:
            timed_out = True
            break

        candidates: list[tuple[np.ndarray, int]] = []
        any_extended = False
        for board, score in beam:
            actions = _find_valid_actions_array(board)
            if not actions:
                if score > best_score:
                    best_score = score
                continue
            any_extended = True
            for rect in actions:
                new_board, removed = _apply(board, rect)
                key = new_board.tobytes()
                if key in seen:
                    continue
                seen.add(key)
                candidates.append((new_board, score + removed))

        if not any_extended:
            break

        if not candidates:
            break

        # 점수 내림차순 top W. 안정 정렬.
        candidates.sort(key=lambda x: -x[1])
        beam = candidates[:beam_width]
        steps = step + 1

        beam_max = beam[0][1]
        if beam_max > best_score:
            best_score = beam_max

    for _, score in beam:
        if score > best_score:
            best_score = score

    return BeamResult(
        best_score=best_score,
        steps=steps,
        elapsed_sec=time.time() - start,
        timed_out=timed_out,
        beam_final_size=len(beam),
    )
