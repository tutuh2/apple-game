"""フルーツボックス 시뮬레이터 단위 테스트.

핵심 룰을 lock-in한다:
  - 직사각형 합 계산
  - 합=10이면 제거 + 점수, 아니면 무변동
  - 빈 칸(0)은 합에 포함되지만 사과 개수에는 안 들어감
  - is_done 정확성
  - Action 좌표 유효성
"""

from __future__ import annotations

import numpy as np
import pytest

from env import FruitBox, Action, BOARD_ROWS, BOARD_COLS


def _make_env_with_board(board: np.ndarray) -> FruitBox:
    """테스트용: 보드를 명시적으로 주입한 환경 생성."""
    env = FruitBox(seed=0)
    env.reset()
    env.board = board.astype(np.int8).copy()
    return env


class TestAction:
    def test_valid_action(self):
        a = Action(0, 0, 1, 2)
        assert a.size == 6

    def test_single_cell(self):
        a = Action(3, 4, 3, 4)
        assert a.size == 1

    def test_invalid_row_order(self):
        with pytest.raises(ValueError):
            Action(2, 0, 1, 0)

    def test_invalid_col_order(self):
        with pytest.raises(ValueError):
            Action(0, 5, 0, 3)

    def test_out_of_bounds_row(self):
        with pytest.raises(ValueError):
            Action(0, 0, BOARD_ROWS, 0)

    def test_out_of_bounds_col(self):
        with pytest.raises(ValueError):
            Action(0, 0, 0, BOARD_COLS)


class TestReset:
    def test_board_shape(self):
        env = FruitBox(seed=42)
        board = env.reset()
        assert board.shape == (BOARD_ROWS, BOARD_COLS)

    def test_values_in_range(self):
        env = FruitBox(seed=42)
        env.reset()
        assert env.board.min() >= 1
        assert env.board.max() <= 9

    def test_score_is_zero_after_reset(self):
        env = FruitBox(seed=42)
        env.reset()
        assert env.score == 0

    def test_seed_reproducible(self):
        env1 = FruitBox(seed=42)
        env1.reset()
        env2 = FruitBox(seed=42)
        env2.reset()
        assert np.array_equal(env1.board, env2.board)

    def test_different_seeds_differ(self):
        env1 = FruitBox(seed=42)
        env1.reset()
        env2 = FruitBox(seed=43)
        env2.reset()
        assert not np.array_equal(env1.board, env2.board)

    def test_board_sum_is_multiple_of_10(self):
        """진짜 ゲーム菜園 게임의 rejection sampling 재현 검증.
        100판 모두 보드 합이 10의 배수여야 한다."""
        for seed in range(100):
            env = FruitBox(seed=seed)
            env.reset()
            total = int(env.board.sum())
            assert total % 10 == 0, f"seed={seed}: 보드 합 {total}은 10의 배수가 아님"


class TestRectangleSum:
    def test_single_cell(self):
        board = np.full((BOARD_ROWS, BOARD_COLS), 3, dtype=np.int8)
        env = _make_env_with_board(board)
        assert env.rectangle_sum(0, 0, 0, 0) == 3

    def test_full_row(self):
        board = np.full((BOARD_ROWS, BOARD_COLS), 1, dtype=np.int8)
        env = _make_env_with_board(board)
        assert env.rectangle_sum(0, 0, 0, BOARD_COLS - 1) == BOARD_COLS

    def test_includes_zeros(self):
        board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)
        board[0, 0] = 5
        board[0, 1] = 5
        env = _make_env_with_board(board)
        assert env.rectangle_sum(0, 0, 0, 3) == 10


class TestStep:
    def test_sum_equals_10_removes_and_scores(self):
        board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)
        board[0, 0] = 7
        board[0, 1] = 3
        env = _make_env_with_board(board)

        reward = env.step(Action(0, 0, 0, 1))

        assert reward == 2
        assert env.score == 2
        assert env.board[0, 0] == 0
        assert env.board[0, 1] == 0

    def test_sum_not_10_no_change(self):
        board = np.full((BOARD_ROWS, BOARD_COLS), 5, dtype=np.int8)
        env = _make_env_with_board(board)
        original = env.board.copy()

        reward = env.step(Action(0, 0, 0, 0))

        assert reward == 0
        assert env.score == 0
        assert np.array_equal(env.board, original)

    def test_zeros_count_toward_sum_but_not_score(self):
        board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)
        board[0, 0] = 4
        board[0, 1] = 6
        env = _make_env_with_board(board)

        reward = env.step(Action(0, 0, 0, 3))

        assert reward == 2
        assert env.score == 2

    def test_large_rectangle_sum_10(self):
        board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)
        board[2, 3] = 3
        board[3, 5] = 7
        env = _make_env_with_board(board)

        reward = env.step(Action(0, 0, 4, 8))

        assert reward == 2
        assert env.board[2, 3] == 0
        assert env.board[3, 5] == 0

    def test_step_before_reset_raises(self):
        env = FruitBox()
        with pytest.raises(RuntimeError):
            env.step(Action(0, 0, 0, 0))


class TestIsDone:
    def test_empty_board_is_done(self):
        board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)
        env = _make_env_with_board(board)
        assert env.is_done() is True

    def test_board_with_sum_10_not_done(self):
        board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)
        board[5, 5] = 4
        board[5, 6] = 6
        env = _make_env_with_board(board)
        assert env.is_done() is False

    def test_board_no_sum_10_is_done(self):
        # 모든 칸이 9 → 1칸=9, 2칸 이상=18+ → 합=10 불가능
        board = np.full((BOARD_ROWS, BOARD_COLS), 9, dtype=np.int8)
        env = _make_env_with_board(board)
        assert env.is_done() is True

    def test_done_after_clearing_all(self):
        board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)
        board[0, 0] = 4
        board[0, 1] = 6
        env = _make_env_with_board(board)
        env.step(Action(0, 0, 0, 1))
        assert env.is_done() is True

    def test_done_before_reset_raises(self):
        env = FruitBox()
        with pytest.raises(RuntimeError):
            env.is_done()


class TestRender:
    def test_render_returns_string(self):
        env = FruitBox(seed=42)
        env.reset()
        out = env.render()
        assert isinstance(out, str)
        assert "점수: 0" in out

    def test_render_shows_dot_for_zero(self):
        board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)
        env = _make_env_with_board(board)
        out = env.render()
        assert "." in out
