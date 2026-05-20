"""greedy_smallest로 N판을 플레이하면서 BC 학습용 데이터를 .npz로 저장.

각 step마다 다음 세 가지를 모은다:
  - obs: 정규화된 보드 (1, 10, 17) float32 — sb3 정책 입력 형식과 동일
  - action: encode(r1, c1, r2, c2)로 만든 flat int64 인덱스 (0..8414)
  - mask: 그 상태의 valid action bool 마스크 (8415,) — masked CE 용

종료 상태(액션이 None)는 저장하지 않는다.

사용법:
    python -m scripts.generate_imitation_data --episodes 1000 --seed 0
    → data/imitation_greedy_smallest_1000.npz
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from agent.heuristics import find_valid_actions, greedy_smallest_policy
from env.action_space import NUM_ACTIONS, compute_action_mask, encode
from env.fruit_box import BOARD_COLS, BOARD_ROWS, MAX_APPLE, FruitBox


DEFAULT_OUTPUT_DIR = Path("data")


def _normalize(board: np.ndarray) -> np.ndarray:
    return (board.astype(np.float32) / float(MAX_APPLE))[None, :, :]


def generate(
    episodes: int,
    seed: int,
    output_dir: Path,
    max_steps: int = 500,
) -> Path:
    """N판 시뮬레이션 후 .npz 저장하고 경로 반환."""
    output_dir.mkdir(parents=True, exist_ok=True)

    obs_list: list[np.ndarray] = []
    action_list: list[int] = []
    mask_list: list[np.ndarray] = []
    episode_id_list: list[int] = []
    scores: list[int] = []

    # 데이터 생성용 재현성: 같은 seed면 같은 데이터.
    # 판마다 다른 보드를 쓰되 재현 가능하도록 episode_seed = seed + ep_idx.
    t0 = time.time()
    for ep in range(episodes):
        episode_seed = seed + ep
        env = FruitBox(seed=episode_seed)
        env.reset()
        policy_rng = np.random.default_rng(episode_seed)

        for _ in range(max_steps):
            # 종료 체크: 더 둘 게 없으면 이 step은 저장하지 않고 종료.
            valid = find_valid_actions(env)
            if not valid:
                break

            board = env.board.copy()
            mask = compute_action_mask(board)
            action = greedy_smallest_policy(env, policy_rng)
            assert action is not None  # valid가 있는데 None이면 버그
            flat = encode(action.r1, action.c1, action.r2, action.c2)
            assert mask[flat], "선택된 action이 mask에서 invalid — 버그"

            obs_list.append(_normalize(board))
            action_list.append(flat)
            mask_list.append(mask)
            episode_id_list.append(ep)

            env.step(action)

        scores.append(env.score)

        if (ep + 1) % 100 == 0:
            elapsed = time.time() - t0
            mean_score = float(np.mean(scores))
            print(
                f"[gen] {ep + 1}/{episodes} eps | "
                f"samples={len(obs_list):,} | "
                f"mean_score={mean_score:.2f} | "
                f"elapsed={elapsed:.1f}s"
            )

    obs_arr = np.stack(obs_list, axis=0).astype(np.float32)
    action_arr = np.asarray(action_list, dtype=np.int64)
    mask_arr = np.stack(mask_list, axis=0).astype(bool)
    episode_id_arr = np.asarray(episode_id_list, dtype=np.int32)
    scores_arr = np.asarray(scores, dtype=np.int32)

    assert obs_arr.shape == (len(obs_list), 1, BOARD_ROWS, BOARD_COLS)
    assert mask_arr.shape == (len(obs_list), NUM_ACTIONS)

    output_path = output_dir / f"imitation_greedy_smallest_{episodes}.npz"
    np.savez_compressed(
        output_path,
        obs=obs_arr,
        action=action_arr,
        mask=mask_arr,
        episode_id=episode_id_arr,
        scores=scores_arr,
    )

    print(
        f"[gen] saved → {output_path} "
        f"(obs {obs_arr.nbytes / 1e6:.1f} MB, "
        f"mask {mask_arr.nbytes / 1e6:.1f} MB; npz는 압축됨)"
    )
    print(
        f"[gen] 점수 분포: mean={scores_arr.mean():.2f} "
        f"median={float(np.median(scores_arr)):.1f} "
        f"min={int(scores_arr.min())} max={int(scores_arr.max())}"
    )
    return output_path


def main() -> None:
    p = argparse.ArgumentParser(description="greedy_smallest로 BC 학습 데이터 생성.")
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = p.parse_args()

    generate(
        episodes=args.episodes,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
