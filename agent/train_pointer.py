"""Pointer Network PPO 학습 진입점 (v4).

사용법:
    python -m agent.train_pointer --steps 200000 --seed 0
    python -m agent.train_pointer --steps 50000 --device cpu  # sanity check

v1~v3와 분리된 저장 포맷: models/pointer_v1_{mode}_{N}.pt (PyTorch state_dict).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from agent.ppo_pointer import PPOConfig, evaluate_model, train_ppo


DEFAULT_OUTPUT_DIR = Path("models")


def _resolve_device(arg: str) -> str:
    if arg == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return arg


def main() -> None:
    p = argparse.ArgumentParser(description="Train PointerNet+PPO on FruitBox (v4).")
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"]
    )
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument(
        "--reward-mode", type=str, default="shaped", choices=["raw", "shaped"]
    )
    p.add_argument("--n-steps", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-4)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)

    cfg = PPOConfig(
        total_steps=args.steps,
        n_steps=args.n_steps,
        learning_rate=args.lr,
        seed=args.seed,
        device=device,
        reward_mode=args.reward_mode,
    )

    model, _ = train_ppo(cfg)

    save_path = args.output_dir / f"pointer_v1_{args.reward_mode}_{args.steps}.pt"
    torch.save(model.state_dict(), save_path)
    print(f"[train] saved → {save_path}")

    mean, std, scores = evaluate_model(
        model,
        n_episodes=args.eval_episodes,
        seed_start=args.seed + 10_000,
        device=torch.device(device),
        deterministic=True,
    )
    print(
        f"[train] eval over {args.eval_episodes} episodes (raw game score): "
        f"mean={mean:.2f} std={std:.2f}  min={min(scores)} max={max(scores)}"
    )


if __name__ == "__main__":
    main()
