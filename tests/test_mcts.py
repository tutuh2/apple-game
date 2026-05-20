"""MCTS 노드 + 탐색 단위 테스트."""

from __future__ import annotations

import numpy as np

from agent.mcts import MCTSNode, _step_board, play_one_game, run_search
from env.action_space import compute_candidates
from env.fruit_box import BOARD_COLS, BOARD_ROWS


def _uniform_model_fn(board: np.ndarray) -> tuple[np.ndarray, float]:
    """모든 후보에 균등 prior, value=0인 더미 model_fn."""
    coords, _ = compute_candidates(board)
    k = len(coords)
    if k == 0:
        return np.zeros(0, dtype=np.float32), 0.0
    priors = np.full(k, 1.0 / k, dtype=np.float32)
    return priors, 0.0


class TestStepBoard:
    def test_sum_10_removes(self):
        board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)
        board[0, 0] = 5
        board[0, 1] = 5
        new_board, removed = _step_board(board, 0, 0, 0, 1)
        assert removed == 2
        assert new_board[0, 0] == 0
        assert new_board[0, 1] == 0
        assert board[0, 0] == 5  # 원본 보존

    def test_sum_not_10_no_change(self):
        board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)
        board[0, 0] = 5
        board[0, 1] = 3
        new_board, removed = _step_board(board, 0, 0, 0, 1)
        assert removed == 0
        assert new_board is board


class TestMCTSNode:
    def test_empty_board_terminal(self):
        board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)
        node = MCTSNode(board=board)
        assert node.terminal
        assert node.k == 0

    def test_5_plus_5_not_terminal(self):
        board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)
        board[0, 0] = 5
        board[0, 1] = 5
        node = MCTSNode(board=board)
        assert not node.terminal
        assert node.k >= 1


class TestRunSearch:
    def test_empty_board_returns_none(self):
        board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)
        result = run_search(board, _uniform_model_fn, n_simulations=10)
        assert result is None

    def test_returns_valid_action_index(self):
        board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)
        board[0, 0] = 5
        board[0, 1] = 5
        result = run_search(board, _uniform_model_fn, n_simulations=20)
        assert result is not None
        assert isinstance(result, int)
        assert result >= 0


class TestPlayOneGame:
    def test_completes_a_game(self):
        score = play_one_game(
            seed=0,
            model_fn=_uniform_model_fn,
            n_simulations=10,
            max_steps=200,
        )
        assert score >= 0
        assert score <= 170

    def test_deterministic_with_seed(self):
        s1 = play_one_game(seed=42, model_fn=_uniform_model_fn, n_simulations=5)
        s2 = play_one_game(seed=42, model_fn=_uniform_model_fn, n_simulations=5)
        assert s1 == s2


class TestRollout:
    """rollout_fn 호출 여부와 value backup 동작 검증."""

    def test_rollout_fn_invoked(self):
        """non-terminal leaf expand 시 rollout_fn이 호출되는지.
        보드에 여러 합=10 패턴을 둬서 한 액션을 둬도 leaf가 terminal이 아니게."""
        calls: list[int] = []

        def rollout_fn(board: np.ndarray) -> float:
            calls.append(int(board.sum()))
            return 0.0

        from env.fruit_box import FruitBox

        env = FruitBox(seed=42)
        env.reset()
        run_search(env.board, _uniform_model_fn, n_simulations=5, rollout_fn=rollout_fn)
        # 최소 한 번은 non-terminal leaf에서 호출돼야 함.
        assert len(calls) >= 1

    def test_rollout_value_propagates_to_q(self):
        """rollout이 큰 양수 반환하면 q_values에 신호가 누적되는지."""
        from agent.mcts import MCTSNode, _expand, _simulate

        board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)
        board[0, 0] = 5
        board[0, 1] = 5
        board[5, 5] = 4
        board[5, 6] = 6

        def rollout_high(b: np.ndarray) -> float:
            return 100.0

        root = MCTSNode(board=board.copy())
        _expand(root, _uniform_model_fn)
        for _ in range(20):
            _simulate(root, _uniform_model_fn, c_puct=1.4, rollout_fn=rollout_high)

        # rollout=100이 backup되면 q_values 합/평균이 0보다 의미있게 큼.
        assert root.q_values.max() > 1.0

    def test_no_rollout_keeps_value_small(self):
        """rollout_fn=None이면 leaf value 0 — 즉시 reward만 누적."""
        from agent.mcts import MCTSNode, _expand, _simulate

        board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)
        board[0, 0] = 5
        board[0, 1] = 5
        root = MCTSNode(board=board.copy())
        _expand(root, _uniform_model_fn)
        for _ in range(10):
            _simulate(root, _uniform_model_fn, c_puct=1.4, rollout_fn=None)

        # 즉시 reward(2 사과 제거)만 backup → q_values 평균이 작은 양수.
        assert root.q_values.max() <= 10.0
