"""빔서치 평가 — 각 보드의 이론적 천장에 얼마나 가까운지 측정.

이 게임은 deterministic + perfect info. RL/MCTS의 113점 벽이 알고리즘
한계인지 게임 자체 한계인지 모름. 빔서치(폭 W, 시간 제한 T)로 직접 측정.

사용법:
    python3 scripts/eval_beam.py --episodes 5 --beam-width 50
    python3 scripts/eval_beam.py --episodes 5 --beam-width 200 --time-limit 60
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from agent.beam_search import beam_search  # noqa: E402
from env.fruit_box import FruitBox  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="빔서치 평가 — 각 보드 천장 측정")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--beam-width", type=int, default=50)
    p.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="판당 시간 제한(초). None이면 게임 끝까지.",
    )
    p.add_argument("--seed-start", type=int, default=0)
    args = p.parse_args()

    print(
        f"=== 빔서치 평가 ({args.episodes}판, W={args.beam_width}, "
        f"time_limit={args.time_limit}s/판) ==="
    )
    print("  최대 가능 점수: 170\n")

    scores: list[int] = []
    timed_outs: list[bool] = []
    start = time.time()
    for i in range(args.episodes):
        seed = args.seed_start + i
        env = FruitBox(seed=seed)
        env.reset()
        result = beam_search(
            env.board,
            beam_width=args.beam_width,
            time_limit_sec=args.time_limit,
        )
        scores.append(result.best_score)
        timed_outs.append(result.timed_out)
        elapsed_total = time.time() - start
        avg = float(np.mean(scores))
        flag = " [TIMEOUT]" if result.timed_out else ""
        print(
            f"  [{i+1:3d}/{args.episodes}] seed={seed:3d} "
            f"score={result.best_score:3d}  "
            f"steps={result.steps:3d} "
            f"beam_final={result.beam_final_size:4d} "
            f"({result.elapsed_sec:.1f}s){flag}  "
            f"running_avg={avg:6.2f}  total={elapsed_total:.0f}s"
        )

    arr = np.array(scores)
    print(
        f"\n  평균 {arr.mean():6.2f}  중앙값 {np.median(arr):6.1f}  "
        f"표편 {arr.std():5.2f}  min {arr.min():3d}  max {arr.max():3d}"
    )
    n_timeout = sum(timed_outs)
    if n_timeout > 0:
        print(f"  ⚠ timed_out: {n_timeout}/{args.episodes}")

    print("\n=== 천장 비교 ===")
    print("  random            평균 103.43")
    print("  greedy_smallest   평균 113.50  ← 휴리스틱 천장")
    print(f"  beam_W={args.beam_width:<3d}        평균 {arr.mean():6.2f}")
    delta = arr.mean() - 113.50
    sgn = ("+" if delta >= 0 else "") + f"{delta:.2f}"
    print(f"\n  greedy_smallest 대비: {sgn}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
