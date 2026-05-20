"""PointerNet + MCTS 평가 (옵션 C — AlphaZero 스타일).

사용법:
    python3 scripts/eval_mcts.py models/pointer_v1_shaped_200000.pt
    python3 scripts/eval_mcts.py models/pointer_v1_shaped_200000.pt --episodes 100 --simulations 200
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from agent.mcts import play_one_game  # noqa: E402
from agent.pointer_mcts import make_model_fn  # noqa: E402
from agent.pointer_net import PointerNet  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="PointerNet+MCTS 평가 (raw 게임 점수)")
    p.add_argument("model_path", type=Path)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--simulations", type=int, default=200, help="MCTS 시뮬레이션 N")
    p.add_argument("--c-puct", type=float, default=1.4)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    if not args.model_path.exists():
        print(f"[eval] model not found: {args.model_path}", file=sys.stderr)
        return 1

    print(f"[eval] loading {args.model_path}")
    device = torch.device(args.device)
    model = PointerNet().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    model_fn = make_model_fn(model, device)

    print(
        f"=== MCTS+PointerNet 평가 ({args.episodes}판, "
        f"N={args.simulations}, c_puct={args.c_puct}) ===\n"
        f"  최대 가능 점수: 170\n"
    )

    scores: list[int] = []
    start = time.time()
    for i in range(args.episodes):
        seed = args.seed_start + i
        s = play_one_game(
            seed=seed,
            model_fn=model_fn,
            n_simulations=args.simulations,
            c_puct=args.c_puct,
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

    print("\n=== Step 1 휴리스틱 + RL 시도 비교 ===")
    print("  random            평균 103.43")
    print("  greedy_largest    평균  97.24")
    print("  greedy_smallest   평균 113.50  ← 베이스라인 최강")
    print("  v4 PointerNet     평균 110.21")
    print(f"  v4 + MCTS         평균 {arr.mean():6.2f}")
    delta_g = arr.mean() - 113.50
    delta_v4 = arr.mean() - 110.21
    sgn_g = ("+" if delta_g >= 0 else "") + f"{delta_g:.2f}"
    sgn_v = ("+" if delta_v4 >= 0 else "") + f"{delta_v4:.2f}"
    print(f"\n  greedy_smallest 대비: {sgn_g}")
    print(f"  v4 단독 대비:         {sgn_v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
