"""Uniform prior + 0 value로 pure MCTS 평가 (모델 없음).

가설: 단순 1-step 휴리스틱(greedy_smallest 113.50, lookahead_greedy 109.30)
이 천장이라면 MCTS의 다단 lookahead가 거기를 뚫을 수 있는가?

PointerNet prior 없이 uniform — 즉 "어느 후보든 동등하게 신경 쓰며 N번
시뮬레이션" 형식. PUCT 식에서 prior가 1/K로 평탄해진다.

사용법:
    python3 scripts/eval_mcts_uniform.py --episodes 10 --simulations 200
    python3 scripts/eval_mcts_uniform.py --episodes 10 --simulations 1000
    python3 scripts/eval_mcts_uniform.py --episodes 100 --simulations 5000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from agent.heuristics import greedy_smallest_policy, run_episode  # noqa: E402
from agent.mcts import play_one_game  # noqa: E402
from env.action_space import compute_candidates  # noqa: E402
from env.fruit_box import FruitBox  # noqa: E402


def uniform_model_fn(board: np.ndarray) -> tuple[np.ndarray, float]:
    """모델 없이 prior 균등 + value 0. MCTS가 순수 simulation에 의존."""
    coords, _ = compute_candidates(board)
    k = len(coords)
    if k == 0:
        return np.zeros(0, dtype=np.float32), 0.0
    return np.full(k, 1.0 / k, dtype=np.float32), 0.0


def greedy_smallest_rollout(
    board: np.ndarray, rng: np.random.Generator
) -> float:
    """주어진 board에서 greedy_smallest로 게임 끝까지 시뮬해 누적 점수 반환.

    MCTS leaf value 추정용. board를 그대로 변형하지 않도록 FruitBox 인스턴스
    하나에 copy해서 넘김.
    """
    game = FruitBox()
    game.board = board.copy()
    game.score = 0
    game._initialized = True

    score = 0
    for _ in range(200):
        action = greedy_smallest_policy(game, rng)
        if action is None:
            break
        reward = game.step(action)
        score += int(reward)
    return float(score)


def main() -> int:
    p = argparse.ArgumentParser(description="Uniform-prior MCTS 평가")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--simulations", type=int, default=200, help="MCTS 시뮬레이션 N")
    p.add_argument("--c-puct", type=float, default=1.4)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument(
        "--rollout",
        action="store_true",
        help="leaf에서 greedy_smallest로 끝까지 시뮬해 value 추정",
    )
    args = p.parse_args()

    rollout_fn = None
    if args.rollout:
        rollout_rng = np.random.default_rng(args.seed_start + 999)

        def rollout_fn(board: np.ndarray) -> float:
            return greedy_smallest_rollout(board, rollout_rng)

    mode = "rollout=greedy_smallest" if args.rollout else "no rollout (value=0)"
    print(
        f"=== Uniform-prior MCTS 평가 ({args.episodes}판, "
        f"N={args.simulations}, c_puct={args.c_puct}, {mode}) ==="
    )
    print("  최대 가능 점수: 170\n")

    scores: list[int] = []
    start = time.time()
    for i in range(args.episodes):
        seed = args.seed_start + i
        s = play_one_game(
            seed=seed,
            model_fn=uniform_model_fn,
            n_simulations=args.simulations,
            c_puct=args.c_puct,
            rollout_fn=rollout_fn,
        )
        scores.append(s)
        elapsed = time.time() - start
        avg = float(np.mean(scores))
        print(
            f"  [{i+1:3d}/{args.episodes}] seed={seed:3d} score={s:3d}  "
            f"running_avg={avg:6.2f}  ({elapsed:.0f}s)"
        )

    arr = np.array(scores)
    print(
        f"\n  평균 {arr.mean():6.2f}  중앙값 {np.median(arr):6.1f}  "
        f"표편 {arr.std():5.2f}  min {arr.min():3d}  max {arr.max():3d}"
    )

    print("\n=== 비교 ===")
    print("  greedy_smallest    평균 113.50  ← 천장")
    print("  lookahead_greedy   평균 109.30")
    print("  v4 + MCTS (N=200)  평균 111.20  (PointerNet prior)")
    print(f"  uniform MCTS       평균 {arr.mean():6.2f}  (N={args.simulations})")
    delta = arr.mean() - 113.50
    sgn = ("+" if delta >= 0 else "") + f"{delta:.2f}"
    print(f"\n  greedy_smallest 대비: {sgn}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
