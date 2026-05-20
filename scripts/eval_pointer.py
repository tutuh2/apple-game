"""학습된 PointerNet 모델 평가 (v4).

사용법:
    python3 scripts/eval_pointer.py models/pointer_v1_shaped_200000.pt
    python3 scripts/eval_pointer.py models/pointer_v1_shaped_200000.pt --episodes 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from agent.pointer_net import PointerNet  # noqa: E402
from agent.ppo_pointer import evaluate_model  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="PointerNet 평가 (raw 게임 점수)")
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument(
        "--stochastic", action="store_true", help="deterministic=False 샘플링"
    )
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if not args.model_path.exists():
        print(f"[eval] model not found: {args.model_path}", file=sys.stderr)
        return 1

    print(f"[eval] loading {args.model_path}")
    device = torch.device(args.device)
    model = PointerNet().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    print(f"=== PointerNet 평가 ({args.episodes}판, seed={args.seed_start}..) ===")
    print("  최대 가능 점수: 170\n")

    mean, std, scores = evaluate_model(
        model,
        n_episodes=args.episodes,
        seed_start=args.seed_start,
        device=device,
        deterministic=not args.stochastic,
    )
    arr = np.array(scores)
    print(
        f"  평균 {mean:6.2f}  중앙값 {np.median(arr):6.1f}  표편 {std:5.2f}  "
        f"min {arr.min():3d}  max {arr.max():3d}"
    )

    print("\n=== Step 1 휴리스틱 비교 ===")
    print("  random            평균 103.43")
    print("  greedy_largest    평균  97.24")
    print("  greedy_smallest   평균 113.50  ← 베이스라인 최강")
    print(f"  pointer (v4)      평균 {mean:6.2f}")
    delta = mean - 113.50
    sign = "+" if delta >= 0 else ""
    print(f"\n  greedy_smallest 대비: {sign}{delta:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
