"""FruitBoxEnv (Gymnasium wrapper) 단위 테스트."""

from __future__ import annotations

import numpy as np

from env.action_space import NUM_ACTIONS, encode
from env.fruit_box import BOARD_COLS, BOARD_ROWS
from env.fruit_box_gym import FruitBoxEnv


class TestSpaces:
    def test_observation_space(self):
        env = FruitBoxEnv(seed=0)
        assert env.observation_space.shape == (1, BOARD_ROWS, BOARD_COLS)
        assert env.observation_space.dtype == np.float32

    def test_action_space(self):
        env = FruitBoxEnv(seed=0)
        assert env.action_space.n == NUM_ACTIONS


class TestReset:
    def test_reset_returns_obs_and_info(self):
        env = FruitBoxEnv()
        obs, info = env.reset(seed=42)
        assert obs.shape == (1, BOARD_ROWS, BOARD_COLS)
        assert obs.dtype == np.float32
        assert 0.0 <= obs.min() and obs.max() <= 1.0
        assert info["score"] == 0
        assert info["valid_actions"] > 0

    def test_reset_is_deterministic_with_seed(self):
        env1 = FruitBoxEnv()
        obs1, _ = env1.reset(seed=123)
        env2 = FruitBoxEnv()
        obs2, _ = env2.reset(seed=123)
        np.testing.assert_array_equal(obs1, obs2)


class TestStep:
    def test_valid_step_returns_positive_reward(self):
        env = FruitBoxEnv()
        env.reset(seed=7)
        mask = env.action_masks()
        valid_idx = int(np.argmax(mask))
        obs, reward, terminated, truncated, info = env.step(valid_idx)

        assert reward > 0
        assert obs.shape == (1, BOARD_ROWS, BOARD_COLS)
        assert isinstance(terminated, bool)
        assert truncated is False
        assert info["score"] == reward

    def test_invalid_step_zero_reward(self):
        env = FruitBoxEnv()
        env.reset(seed=7)

        env._game.board = np.full((BOARD_ROWS, BOARD_COLS), 9, dtype=np.int8)
        env._refresh_mask()

        before_board = env._game.board.copy()
        before_score = env._game.score
        idx = encode(0, 0, 0, 0)  # 단일 칸, 합=9
        _, reward, _, _, info = env.step(idx)

        assert reward == 0
        assert info["score"] == before_score
        np.testing.assert_array_equal(env._game.board, before_board)


class TestActionMasks:
    def test_action_masks_shape(self):
        env = FruitBoxEnv()
        env.reset(seed=0)
        mask = env.action_masks()
        assert mask.shape == (NUM_ACTIONS,)
        assert mask.dtype == bool

    def test_action_masks_refreshed_after_step(self):
        env = FruitBoxEnv()
        env.reset(seed=0)
        mask_before = env.action_masks().copy()
        valid_idx = int(np.argmax(mask_before))
        env.step(valid_idx)
        mask_after = env.action_masks()
        assert not np.array_equal(mask_before, mask_after)


class TestTermination:
    def test_terminates_when_no_valid_actions(self):
        env = FruitBoxEnv()
        env.reset(seed=0)
        env._game.board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)
        env._refresh_mask()
        _, _, terminated, _, _ = env.step(0)
        assert terminated is True


class TestRewardMode:
    """reward_mode='raw' (기본) vs 'shaped' 동작 검증."""

    def _set_board_with_5_plus_5(self, env: FruitBoxEnv) -> None:
        env._game.board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)
        env._game.board[0, 0] = 5
        env._game.board[0, 1] = 5
        env._refresh_mask()

    def test_invalid_reward_mode_raises(self):
        import pytest

        with pytest.raises(ValueError):
            FruitBoxEnv(reward_mode="invalid")

    def test_raw_mode_returns_apple_count(self):
        env = FruitBoxEnv(reward_mode="raw")
        env.reset(seed=0)
        self._set_board_with_5_plus_5(env)
        idx = encode(0, 0, 0, 1)
        _, reward, _, _, info = env.step(idx)
        assert reward == 2.0
        assert info["raw_reward"] == 2

    def test_shaped_mode_small_region_bonus(self):
        env = FruitBoxEnv(reward_mode="shaped")
        env.reset(seed=0)
        self._set_board_with_5_plus_5(env)
        idx = encode(0, 0, 0, 1)
        _, reward, _, _, info = env.step(idx)
        # raw=2 + small bonus (2*0.5) + terminal bonus (score=2 * 0.05)
        # = 2 + 1 + 0.1 = 3.1
        assert abs(reward - 3.1) < 1e-6
        assert info["raw_reward"] == 2

    def test_shaped_mode_large_region_no_small_bonus(self):
        env = FruitBoxEnv(reward_mode="shaped")
        env.reset(seed=0)
        env._game.board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)
        env._game.board[0, 0:5] = [1, 2, 3, 1, 3]  # size=5, raw=5
        env._refresh_mask()
        idx = encode(0, 0, 0, 4)
        _, reward, _, _, info = env.step(idx)
        assert info["raw_reward"] == 5
        # small bonus 적용됐다면 reward >= 7.5. 미적용이면 < 7.5.
        # 종료 보너스(score=5 * 0.05 = 0.25)는 별개로 붙음.
        assert reward < 7.5

    def test_raw_mode_no_bonuses(self):
        env = FruitBoxEnv(reward_mode="raw")
        env.reset(seed=0)
        self._set_board_with_5_plus_5(env)
        idx = encode(0, 0, 0, 1)
        _, reward, terminated, _, _ = env.step(idx)
        assert terminated is True
        assert reward == 2.0
