"""BC로 사전학습된 정책의 실게임 성능 평가.

agent/train_bc.py 가 저장한 zip은 sb3 MaskablePPO 호환이라
MaskablePPO.load()로 그대로 읽을 수 있다. 평가 절차는 eval_ppo.py와
동일하지만 BC 컨텍스트를 명시한다 — greedy_smallest(113.50)과 직접 비교.

사용법:
    python3 scripts/eval_bc.py models/bc_greedy_smallest_1000_e10.zip
    python3 scripts/eval_bc.py models/bc_greedy_smallest_1000_e10.zip --episodes 100

성공 기준:
    >= 110  : 성공 — greedy_smallest 거의 복제, PPO fine-tune 진입 가능
   100~110  : 부분 성공 — 데이터 더, 모델 더 깊게
    < 100  : 실패 — 표현/손실 재설계
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from sb3_contrib import MaskablePPO  # noqa: E402

from env.fruit_box_gym import FruitBoxEnv  # noqa: E402


def run_episode(model: MaskablePPO, seed: int, deterministic: bool = True) -> int:
    env = FruitBoxEnv()
    obs, _ = env.reset(seed=seed)
    terminated = False
    info = {"score": 0}
    while not terminated:
        mask = env.action_masks()
        if not mask.any():
            break
        action, _ = model.predict(obs, action_masks=mask, deterministic=deterministic)
        obs, _, terminated, _, info = env.step(int(action))
    return info["score"]


def benchmark(model_path: Path, n_episodes: int, deterministic: bool) -> dict:
    print(f"[eval-bc] loading {model_path}")
    model = MaskablePPO.load(model_path, device="cpu")

    scores: list[int] = []
    start = time.time()
    for seed in range(n_episodes):
        scores.append(run_episode(model, seed=seed, deterministic=deterministic))
    elapsed = time.time() - start

    arr = np.array(scores)
    return {
        "n": n_episodes,
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "median": float(np.median(arr)),
        "elapsed_sec": elapsed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="BC 정책 평가")
    parser.add_argument("model_path", type=Path, help="저장된 BC .zip 모델 경로")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="deterministic=False (정책 분포에서 샘플링)",
    )
    args = parser.parse_args()

    if not args.model_path.exists():
        print(f"[eval-bc] model not found: {args.model_path}", file=sys.stderr)
        return 1

    print(
        f"=== BC 정책 평가 ({args.episodes}판, seed=0..{args.episodes - 1}) ==="
    )
    print("  최대 가능 점수: 170")
    print(f"  deterministic: {not args.stochastic}\n")

    r = benchmark(args.model_path, args.episodes, deterministic=not args.stochastic)
    print(
        f"  평균 {r['mean']:6.2f}  "
        f"중앙값 {r['median']:6.1f}  "
        f"표편 {r['std']:5.2f}  "
        f"min {r['min']:3d}  max {r['max']:3d}  "
        f"({r['elapsed_sec']:.1f}초)"
    )

    print("\n=== 휴리스틱 비교 (100판 기준) ===")
    print("  random            평균 103.43")
    print("  greedy_largest    평균  97.24")
    print("  greedy_smallest   평균 113.50  ← BC가 복제하려는 정책")
    print(f"  BC (현재)         평균 {r['mean']:6.2f}")
    delta = r["mean"] - 113.50
    sign = "+" if delta >= 0 else ""
    print(f"\n  greedy_smallest 대비: {sign}{delta:.2f}")

    if r["mean"] >= 110:
        print("  → 성공: PPO fine-tune 진입 가능")
    elif r["mean"] >= 100:
        print("  → 부분 성공: 데이터 더, 또는 모델 더 깊게")
    else:
        print("  → 실패: 표현/손실 재설계 필요")

    return 0


if __name__ == "__main__":
    sys.exit(main())
