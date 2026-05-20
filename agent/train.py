"""MaskablePPO + 작은 CNN으로 フルーツボックス 정책 학습 (v2).

v1(MlpPolicy) 결과: 200k 학습해도 ep_rew_mean ~ 103에서 정체. value head는
explained_variance 0.89까지 학습됐지만 policy가 보드의 공간 구조를
표현하지 못해 좋은 액션을 못 찾음. → v2에서 10×17 보드 전용 CNN 도입.

사용법:
    python -m agent.train --steps 200000 --seed 0
    python -m agent.train --steps 10000 --device cpu  # sanity check

설계:
- 보드를 (1, 10, 17) 채널 이미지로 보고 Conv 2층 + Linear로 압축.
- MaskablePPO + custom features_extractor 조합. policy="MlpPolicy"는
  유지하되 features_extractor만 SmallBoardCNN으로 교체.
- v1 대비 변경 하이퍼파라미터:
    learning_rate: 3e-4 → 1e-4   (CNN은 더 천천히 안정적으로)
    n_steps:       2048  → 4096   (한 번에 더 많이 모아서 안정적 업데이트)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from env.fruit_box import BOARD_COLS, BOARD_ROWS
from env.fruit_box_gym import FruitBoxEnv


DEFAULT_OUTPUT_DIR = Path("models")


class SmallBoardCNN(BaseFeaturesExtractor):
    """10×17 보드 전용 작은 CNN feature extractor.

    SB3 기본 CnnPolicy는 84×84 이미지(Atari)를 가정해서 작은 보드에 안 맞음.
    여기선 padding=1로 공간 크기를 유지하고 2층 Conv → flatten → Linear.

    Output: features_dim 차원 벡터 (기본 128). MaskablePPO의 정책/가치
    heads가 이 벡터를 입력으로 받음.
    """

    def __init__(self, observation_space, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]  # 1

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample = torch.zeros(1, n_input_channels, BOARD_ROWS, BOARD_COLS)
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


def _mask_fn(env: FruitBoxEnv):
    return env.action_masks()


def _resolve_device(arg: str) -> str:
    if arg == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return arg


def _make_env(seed: int, reward_mode: str = "raw") -> ActionMasker:
    env = FruitBoxEnv(seed=seed, reward_mode=reward_mode)
    return ActionMasker(env, _mask_fn)


def train(
    total_steps: int,
    seed: int,
    device: str,
    output_dir: Path,
    eval_episodes: int = 20,
    reward_mode: str = "shaped",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_device = _resolve_device(device)
    print(
        f"[train] device = {resolved_device}, "
        f"total_steps = {total_steps:,}, reward_mode = {reward_mode}"
    )

    env = _make_env(seed=seed, reward_mode=reward_mode)
    eval_env = _make_env(seed=seed + 10_000, reward_mode="raw")

    policy_kwargs = dict(
        features_extractor_class=SmallBoardCNN,
        features_extractor_kwargs=dict(features_dim=128),
    )

    model = MaskablePPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-4,
        n_steps=4096,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=seed,
        device=resolved_device,
    )

    model.learn(total_timesteps=total_steps, progress_bar=False)

    final_path = output_dir / f"ppo_fruitbox_cnn_{reward_mode}_{total_steps}.zip"
    model.save(final_path)
    print(f"[train] saved model → {final_path}")

    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=eval_episodes,
        deterministic=True,
        use_masking=True,
    )
    print(
        f"[train] eval over {eval_episodes} episodes: "
        f"mean={mean_reward:.2f} std={std_reward:.2f}"
    )
    return final_path


def main() -> None:
    p = argparse.ArgumentParser(description="Train MaskablePPO+CNN on FruitBox.")
    p.add_argument("--steps", type=int, default=200_000, help="총 학습 timestep")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
    )
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument(
        "--reward-mode",
        type=str,
        default="shaped",
        choices=["raw", "shaped"],
        help="raw=게임 점수, shaped=small-region/종료 보너스 추가",
    )
    args = p.parse_args()

    train(
        total_steps=args.steps,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
        eval_episodes=args.eval_episodes,
        reward_mode=args.reward_mode,
    )


if __name__ == "__main__":
    main()
