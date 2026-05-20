"""휴리스틱 정책 벤치마크 — N판 평균/최대/최소/표준편차.

이 결과가 Step 2 RL 학습의 목표 기준선이 된다.
RL이 greedy_largest 평균을 충분히 넘기지 못하면 RL 도입 의미가 없다.

사용법:
    python3 scripts/benchmark_heuristics.py [--episodes 100]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from agent.heuristics import (  # noqa: E402
    random_policy,
    greedy_largest_policy,
    greedy_smallest_policy,
    lookahead_greedy_policy,
    run_episode,
)


POLICIES = {
    "random": random_policy,
    "greedy_largest": greedy_largest_policy,
    "greedy_smallest": greedy_smallest_policy,
    "lookahead_greedy": lookahead_greedy_policy,
}


def benchmark(policy_name: str, n_episodes: int) -> dict:
    policy = POLICIES[policy_name]
    scores = []
    start = time.time()
    for seed in range(n_episodes):
        score = run_episode(policy, seed=seed)
        scores.append(score)
    elapsed = time.time() - start

    arr = np.array(scores)
    return {
        "name": policy_name,
        "n": n_episodes,
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "median": float(np.median(arr)),
        "elapsed_sec": elapsed,
    }


def print_result(r: dict) -> None:
    print(
        f"  {r['name']:18s} "
        f"평균 {r['mean']:6.2f}  "
        f"중앙값 {r['median']:6.1f}  "
        f"표편 {r['std']:5.2f}  "
        f"min {r['min']:3d}  max {r['max']:3d}  "
        f"({r['elapsed_sec']:.1f}초)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="휴리스틱 벤치마크")
    parser.add_argument("--episodes", type=int, default=100, help="에피소드 수")
    args = parser.parse_args()

    print(f"=== 휴리스틱 벤치마크 ({args.episodes}판, seed=0..{args.episodes - 1}) ===")
    print(f"  최대 가능 점수: 170 (모든 사과 제거 시)\n")

    results = []
    for name in ["random", "greedy_smallest", "greedy_largest", "lookahead_greedy"]:
        r = benchmark(name, args.episodes)
        results.append(r)
        print_result(r)

    print("\n=== 비교 ===")
    base = next(r for r in results if r["name"] == "random")
    for r in results:
        delta = r["mean"] - base["mean"]
        sign = "+" if delta >= 0 else ""
        print(f"  {r['name']:18s} 평균 {r['mean']:6.2f}  (random 대비 {sign}{delta:.2f})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
