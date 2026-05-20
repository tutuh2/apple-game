"""베이스라인 휴리스틱 정책.

세 가지 단순 정책으로 RL 학습 목표 기준선을 제공한다:
  - random_policy:  유효 액션 중 랜덤 선택
  - greedy_largest: 사과 개수가 가장 많은 합=10 영역 선택
  - greedy_smallest: 사과 개수가 가장 적은 합=10 영역 선택

핵심 헬퍼:
  - find_valid_actions: 현재 보드에서 합=10인 모든 직사각형을 찾는다 (2D prefix sum)
  - run_episode: 정책을 한 게임에 적용해 최종 점수 반환
"""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np

from env import FruitBox, Action, BOARD_ROWS, BOARD_COLS, TARGET_SUM


Policy = Callable[[FruitBox, np.random.Generator], Optional[Action]]


def find_valid_actions(env: FruitBox) -> List[Action]:
    """현재 보드에서 합=10인 모든 직사각형 액션을 반환.

    2D prefix sum으로 O(R²C²)에 모든 직사각형 합을 계산한다.
    """
    board = env.board
    psum = np.zeros((BOARD_ROWS + 1, BOARD_COLS + 1), dtype=np.int32)
    psum[1:, 1:] = board.cumsum(axis=0).cumsum(axis=1)

    actions: List[Action] = []
    for r1 in range(BOARD_ROWS):
        for r2 in range(r1, BOARD_ROWS):
            col_psum = psum[r2 + 1] - psum[r1]
            for c1 in range(BOARD_COLS):
                rect_sums = col_psum[c1 + 1 :] - col_psum[c1]
                matching_c2_offsets = np.where(rect_sums == TARGET_SUM)[0]
                for offset in matching_c2_offsets:
                    c2 = c1 + int(offset)
                    actions.append(Action(r1, c1, r2, c2))
    return actions


def _apple_count(env: FruitBox, action: Action) -> int:
    return env.rectangle_apple_count(action.r1, action.c1, action.r2, action.c2)


def random_policy(
    env: FruitBox, rng: np.random.Generator
) -> Optional[Action]:
    """유효 액션 중에서 균등 랜덤 선택."""
    actions = find_valid_actions(env)
    if not actions:
        return None
    idx = int(rng.integers(0, len(actions)))
    return actions[idx]


def greedy_largest_policy(
    env: FruitBox, rng: np.random.Generator
) -> Optional[Action]:
    """제거되는 사과 개수가 가장 많은 영역 선택. 동률이면 랜덤."""
    actions = find_valid_actions(env)
    if not actions:
        return None

    best_apples = -1
    best: List[Action] = []
    for a in actions:
        apples = _apple_count(env, a)
        if apples > best_apples:
            best_apples = apples
            best = [a]
        elif apples == best_apples:
            best.append(a)

    idx = int(rng.integers(0, len(best)))
    return best[idx]


def greedy_smallest_policy(
    env: FruitBox, rng: np.random.Generator
) -> Optional[Action]:
    """제거되는 사과 개수가 가장 적은 영역 선택. 보드 모양 보존 전략."""
    actions = find_valid_actions(env)
    if not actions:
        return None

    best_apples = float("inf")
    best: List[Action] = []
    for a in actions:
        apples = _apple_count(env, a)
        if apples < best_apples:
            best_apples = apples
            best = [a]
        elif apples == best_apples:
            best.append(a)

    idx = int(rng.integers(0, len(best)))
    return best[idx]


def lookahead_greedy_policy(
    env: FruitBox, rng: np.random.Generator
) -> Optional[Action]:
    """다음 step에 valid 후보가 가장 많이 남는 영역 선택.

    greedy_smallest가 "보드 모양 보존"을 간접적으로 노렸다면, 이건 그걸
    직접 측정한다 — 후보별로 "이걸 두면 다음 step에 valid 후보가 몇 개
    남는가"를 계산하고 최대를 고른다.

    동률은 랜덤. K개 후보에 대해 각각 임시 보드 변경 + valid count 계산이라
    step당 O(K · R²C²) — 비싸지만 휴리스틱 한 번 측정용이라 OK.
    """
    actions = find_valid_actions(env)
    if not actions:
        return None

    board = env.board

    best_lookahead = -1
    best: List[Action] = []
    for a in actions:
        patch = board[a.r1 : a.r2 + 1, a.c1 : a.c2 + 1]
        saved = patch.copy()
        patch[:] = 0
        next_k = len(find_valid_actions(env))
        patch[:] = saved  # 복구

        if next_k > best_lookahead:
            best_lookahead = next_k
            best = [a]
        elif next_k == best_lookahead:
            best.append(a)

    idx = int(rng.integers(0, len(best)))
    return best[idx]


def run_episode(
    policy: Policy,
    seed: Optional[int] = None,
    policy_rng: Optional[np.random.Generator] = None,
    max_steps: int = 500,
) -> int:
    """주어진 정책으로 한 게임을 끝까지 플레이하고 최종 점수 반환."""
    env = FruitBox(seed=seed)
    env.reset()

    if policy_rng is None:
        policy_rng = np.random.default_rng(seed)

    for _ in range(max_steps):
        action = policy(env, policy_rng)
        if action is None:
            break
        env.step(action)

    return env.score
