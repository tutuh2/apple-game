"""Monte Carlo Tree Search (MCTS) with PUCT — AlphaZero style.

PointerNet (또는 임의의 정책 + value 함수)을 prior로 받아 트리 탐색.
한 턴마다 N번 시뮬레이션 → 가장 많이 방문한 자식을 최종 액션으로 선택.

핵심 인터페이스:
  run_search(root_board, model_fn, n_simulations, c_puct) -> action_idx
  play_one_game(seed, model_fn, ...) -> final_score

model_fn(board) -> (priors: (K,), value: float)
  K는 보드의 valid 후보 수. priors는 합 1.

MCTS 본체는 모델 종류 모름 → 재사용성 ↑.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from env.action_space import compute_candidates
from env.fruit_box import Action, FruitBox


ModelFn = Callable[[np.ndarray], tuple[np.ndarray, float]]
# leaf 보드를 받아 게임 끝까지 시뮬레이션해서 누적 점수를 반환.
# AlphaZero 이전 고전 MCTS 방식의 value 추정.
RolloutFn = Callable[[np.ndarray], float]


@dataclass
class MCTSNode:
    """탐색 트리 노드. 한 노드 = 하나의 보드 상태."""

    board: np.ndarray  # (10, 17) int8
    coords: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), dtype=np.int16))
    priors: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    children: dict[int, "MCTSNode"] = field(default_factory=dict)
    visits: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    q_values: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    n_total: int = 0
    expanded: bool = False
    terminal: bool = False

    def __post_init__(self) -> None:
        if not self.expanded and not self.terminal:
            coords, _ = compute_candidates(self.board)
            if len(coords) == 0:
                self.terminal = True
            self.coords = coords

    @property
    def k(self) -> int:
        return len(self.coords)


def _puct_select(node: MCTSNode, c_puct: float) -> int:
    """PUCT 공식으로 자식 인덱스 선택."""
    sqrt_total = math.sqrt(max(node.n_total, 1))
    u = c_puct * node.priors * sqrt_total / (1 + node.visits)
    return int(np.argmax(node.q_values + u))


def _expand(node: MCTSNode, model_fn: ModelFn) -> float:
    """노드 expand + 모델로 prior + value 계산. value 반환."""
    if node.terminal:
        node.expanded = True
        return 0.0

    priors, value = model_fn(node.board)
    if len(priors) != node.k:
        raise RuntimeError(
            f"model_fn returned {len(priors)} priors but node has {node.k} candidates"
        )

    node.priors = priors.astype(np.float32)
    node.visits = np.zeros(node.k, dtype=np.int32)
    node.q_values = np.zeros(node.k, dtype=np.float32)
    node.expanded = True
    return value


def _step_board(
    board: np.ndarray, r1: int, c1: int, r2: int, c2: int
) -> tuple[np.ndarray, int]:
    """보드에 액션 적용. (new_board, removed_count) 반환."""
    region = board[r1 : r2 + 1, c1 : c2 + 1]
    rect_sum = int(region.sum())
    if rect_sum != 10:
        return board, 0
    removed = int(np.count_nonzero(region))
    new_board = board.copy()
    new_board[r1 : r2 + 1, c1 : c2 + 1] = 0
    return new_board, removed


def _simulate(
    root: MCTSNode,
    model_fn: ModelFn,
    c_puct: float,
    rollout_fn: Optional[RolloutFn] = None,
) -> None:
    """한 번의 시뮬레이션: selection → expansion → (rollout) → backup.

    rollout_fn이 주어지면 leaf 노드에서 게임 끝까지 시뮬레이션해 얻은
    점수를 leaf value로 사용. AlphaZero 이전 고전 MCTS 방식.
    None이면 model_fn의 value만 사용.
    """
    path: list[tuple[MCTSNode, int, int]] = []  # (parent, action_idx, reward)
    node = root

    while node.expanded and not node.terminal:
        action_idx = _puct_select(node, c_puct)
        r1, c1, r2, c2 = (int(x) for x in node.coords[action_idx])
        new_board, removed = _step_board(node.board, r1, c1, r2, c2)

        if action_idx in node.children:
            child = node.children[action_idx]
        else:
            child = MCTSNode(board=new_board)
            node.children[action_idx] = child
            path.append((node, action_idx, removed))
            node = child
            break

        path.append((node, action_idx, removed))
        node = child

    if not node.expanded:
        model_value = _expand(node, model_fn)
        # rollout_fn이 있으면 그쪽이 우선 — 학습된 value보다 실제 게임
        # 끝까지의 점수가 신호로 강함. 없으면 model value 사용.
        if rollout_fn is not None and not node.terminal:
            leaf_value = rollout_fn(node.board)
        else:
            leaf_value = model_value
    else:
        leaf_value = 0.0

    # Backup: leaf value + 경로의 즉시 reward
    cumulative = leaf_value
    for parent, action_idx, reward in reversed(path):
        cumulative += reward
        parent.visits[action_idx] += 1
        parent.n_total += 1
        n = parent.visits[action_idx]
        parent.q_values[action_idx] += (cumulative - parent.q_values[action_idx]) / n


def run_search(
    root_board: np.ndarray,
    model_fn: ModelFn,
    n_simulations: int,
    c_puct: float = 1.4,
    rollout_fn: Optional[RolloutFn] = None,
) -> Optional[int]:
    """루트 보드에서 N번 시뮬레이션 후 가장 많이 방문한 action_idx 반환.

    rollout_fn이 주어지면 각 simulation의 leaf에서 게임 끝까지 시뮬해
    얻은 점수를 value로 사용. valid 액션 없으면 None.
    """
    root = MCTSNode(board=root_board.copy())
    if root.terminal:
        return None

    _expand(root, model_fn)

    for _ in range(n_simulations):
        _simulate(root, model_fn, c_puct, rollout_fn=rollout_fn)

    return int(np.argmax(root.visits))


def play_one_game(
    seed: int,
    model_fn: ModelFn,
    n_simulations: int = 200,
    c_puct: float = 1.4,
    max_steps: int = 200,
    rollout_fn: Optional[RolloutFn] = None,
) -> int:
    """seed 보드 한 판을 MCTS로 끝까지 플레이 → 최종 점수 반환."""
    game = FruitBox(seed=seed)
    game.reset(seed=seed)
    steps = 0
    while steps < max_steps:
        action_idx = run_search(
            game.board, model_fn, n_simulations, c_puct, rollout_fn=rollout_fn
        )
        if action_idx is None:
            break
        coords, _ = compute_candidates(game.board)
        if action_idx >= len(coords):
            break
        r1, c1, r2, c2 = (int(x) for x in coords[action_idx])
        game.step(Action(r1, c1, r2, c2))
        steps += 1
    return game.score
