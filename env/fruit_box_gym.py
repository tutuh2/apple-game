"""フルーツボックス 게임을 Gymnasium Env로 wrap.

Observation: shape (1, 10, 17) float32, 값은 board / 9.0 (정규화).
  채널 차원을 유지해 CNN 정책으로 확장하기 쉽게 함.

Action: Discrete(NUM_ACTIONS) = 8415. 직사각형 (r1, c1, r2, c2)를
  flat int로 인코딩.

Reward modes:
  - "raw": 제거된 사과 수 (합 != 10이면 0). 게임 점수와 직결.
  - "shaped": raw + 도메인 지식 보너스.
      * small-region 보너스: 직사각형 크기 ≤ 4 칸이면 reward × 1.5
        (greedy_smallest가 random보다 강했던 패턴을 RL에 명시 신호로 줌)
      * 종료 시 보너스: 게임 끝나면 최종 score × 0.05 추가
        (긴 게임을 끌고 가는 정책을 선호하게 함)

Termination: 더 이상 합=10 직사각형이 없을 때.
"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env.action_space import NUM_ACTIONS, compute_action_mask, decode
from env.fruit_box import BOARD_COLS, BOARD_ROWS, MAX_APPLE, Action, FruitBox


SMALL_REGION_THRESHOLD = 4
SMALL_REGION_BONUS_MULT = 0.5
TERMINAL_SCORE_BONUS = 0.05


class FruitBoxEnv(gym.Env):
    """Gymnasium-호환 フルーツボックス 환경."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        seed: Optional[int] = None,
        reward_mode: str = "raw",
    ):
        super().__init__()
        if reward_mode not in ("raw", "shaped"):
            raise ValueError(f"reward_mode는 'raw' 또는 'shaped'여야 함: {reward_mode}")
        self._reward_mode = reward_mode
        self._game = FruitBox(seed=seed)
        self._cached_mask: Optional[np.ndarray] = None

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(1, BOARD_ROWS, BOARD_COLS), dtype=np.float32
        )
        self.action_space = spaces.Discrete(NUM_ACTIONS)

    def _obs(self) -> np.ndarray:
        return (self._game.board.astype(np.float32) / float(MAX_APPLE))[None, :, :]

    def _refresh_mask(self) -> np.ndarray:
        self._cached_mask = compute_action_mask(self._game.board)
        return self._cached_mask

    def action_masks(self) -> np.ndarray:
        """MaskablePPO가 호출. 현재 보드의 valid action mask."""
        if self._cached_mask is None:
            self._refresh_mask()
        return self._cached_mask

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._game.reset(seed=seed)
        self._refresh_mask()
        return self._obs(), {"score": 0, "valid_actions": int(self._cached_mask.sum())}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action_int = int(action)
        r1, c1, r2, c2 = decode(action_int)
        region_size = (r2 - r1 + 1) * (c2 - c1 + 1)
        raw_reward = self._game.step(Action(r1, c1, r2, c2))

        self._refresh_mask()
        terminated = not bool(self._cached_mask.any())
        truncated = False

        reward = float(raw_reward)
        if self._reward_mode == "shaped":
            if raw_reward > 0 and region_size <= SMALL_REGION_THRESHOLD:
                reward += raw_reward * SMALL_REGION_BONUS_MULT
            if terminated:
                reward += self._game.score * TERMINAL_SCORE_BONUS

        info = {
            "score": self._game.score,
            "valid_actions": int(self._cached_mask.sum()),
            "raw_reward": int(raw_reward),
        }
        return self._obs(), reward, terminated, truncated, info

    def render(self) -> str:
        return self._game.render()
