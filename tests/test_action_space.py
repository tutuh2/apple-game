"""action_space 모듈 단위 테스트."""

from __future__ import annotations

import numpy as np
import pytest

from env.action_space import (
    NUM_ACTIONS,
    NUM_COL_PAIRS,
    NUM_ROW_PAIRS,
    compute_action_mask,
    decode,
    encode,
)
from env.fruit_box import BOARD_COLS, BOARD_ROWS, TARGET_SUM


class TestConstants:
    def test_row_pair_count(self):
        assert NUM_ROW_PAIRS == BOARD_ROWS * (BOARD_ROWS + 1) // 2 == 55

    def test_col_pair_count(self):
        assert NUM_COL_PAIRS == BOARD_COLS * (BOARD_COLS + 1) // 2 == 153

    def test_num_actions(self):
        assert NUM_ACTIONS == NUM_ROW_PAIRS * NUM_COL_PAIRS == 8415


class TestEncodeDecode:
    def test_round_trip_all(self):
        for r1 in range(BOARD_ROWS):
            for r2 in range(r1, BOARD_ROWS):
                for c1 in range(BOARD_COLS):
                    for c2 in range(c1, BOARD_COLS):
                        idx = encode(r1, c1, r2, c2)
                        assert 0 <= idx < NUM_ACTIONS
                        assert decode(idx) == (r1, c1, r2, c2)

    def test_unique_indices(self):
        seen = set()
        for r1 in range(BOARD_ROWS):
            for r2 in range(r1, BOARD_ROWS):
                for c1 in range(BOARD_COLS):
                    for c2 in range(c1, BOARD_COLS):
                        idx = encode(r1, c1, r2, c2)
                        assert idx not in seen
                        seen.add(idx)
        assert len(seen) == NUM_ACTIONS

    def test_encode_validates_bounds(self):
        with pytest.raises(ValueError):
            encode(-1, 0, 0, 0)
        with pytest.raises(ValueError):
            encode(0, 0, BOARD_ROWS, 0)
        with pytest.raises(ValueError):
            encode(2, 0, 1, 0)
        with pytest.raises(ValueError):
            encode(0, 5, 0, 2)

    def test_decode_validates_bounds(self):
        with pytest.raises(ValueError):
            decode(-1)
        with pytest.raises(ValueError):
            decode(NUM_ACTIONS)


class TestActionMask:
    def test_empty_board_no_valid(self):
        board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)
        mask = compute_action_mask(board)
        assert mask.shape == (NUM_ACTIONS,)
        assert mask.dtype == bool
        assert not mask.any()

    def test_simple_5_plus_5(self):
        board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)
        board[0, 0] = 5
        board[0, 1] = 5
        mask = compute_action_mask(board)
        assert mask[encode(0, 0, 0, 1)]

    def test_mask_matches_brute_force(self):
        rng = np.random.default_rng(42)
        board = rng.integers(1, 10, size=(BOARD_ROWS, BOARD_COLS), dtype=np.int8)
        mask = compute_action_mask(board)

        for idx in range(NUM_ACTIONS):
            r1, c1, r2, c2 = decode(idx)
            actual_sum = int(board[r1 : r2 + 1, c1 : c2 + 1].sum())
            expected = actual_sum == TARGET_SUM
            assert mask[idx] == expected

    def test_mask_consistent_with_is_done(self):
        from env.fruit_box import FruitBox

        env = FruitBox(seed=1)
        env.reset()
        mask = compute_action_mask(env.board)
        assert mask.any()
        assert not env.is_done()
