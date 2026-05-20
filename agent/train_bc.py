"""Behavior Cloning: greedy_smallest 데이터로 sb3 MaskablePPO 정책을 사전학습.

핵심 아이디어:
  v1~v4 PPO는 모두 102점 천장에서 시작해 못 빠져나왔다. greedy_smallest를
  먼저 흉내내서 정책을 113점 수준에서 시작하게 만들면 그 위에 PPO fine-tune
  을 얹어 113점을 깨는 게 목표.

설계:
  - sb3 MaskablePPO를 그대로 인스턴스화 (train.py와 동일 SmallBoardCNN)
  - .policy의 forward path를 직접 호출해서 logits → masked cross-entropy
  - 학습 후 model.save()로 .zip 저장 → 추후 PPO fine-tune에 그대로 load 가능
  - value_net은 무시 (fine-tune 단계에서 0부터 학습됨)

데이터 구조 (scripts/generate_imitation_data.py가 만든 .npz):
  obs    (N, 1, 10, 17) float32
  action (N,)            int64    target (정답 액션의 flat index)
  mask   (N, 8415)       bool     valid action 마스크
  episode_id (N,)        int32    판 단위 train/val 분할에 사용

사용법:
    python -m agent.train_bc --data data/imitation_greedy_smallest_1000.npz \\
        --epochs 10 --batch-size 256

출력:
    models/bc_greedy_smallest_1000_e10.zip   # MaskablePPO.load()로 읽기 가능
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sb3_contrib import MaskablePPO
from torch.utils.data import DataLoader, TensorDataset

from agent.train import SmallBoardCNN, _make_env


DEFAULT_OUTPUT_DIR = Path("models")
NEG_INF = -1e9  # masked-out logits에 더해 사실상 0 확률로 만들 값


@dataclass
class Split:
    obs: torch.Tensor
    action: torch.Tensor
    mask: torch.Tensor


def _resolve_device(arg: str) -> str:
    if arg == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return arg


def _split_by_episode(
    obs: np.ndarray,
    action: np.ndarray,
    mask: np.ndarray,
    episode_id: np.ndarray,
    val_frac: float,
    rng: np.random.Generator,
) -> tuple[Split, Split]:
    """판 단위로 train/val 분할 (같은 판 step이 양쪽에 섞이지 않도록)."""
    unique_eps = np.unique(episode_id)
    rng.shuffle(unique_eps)
    n_val = max(1, int(len(unique_eps) * val_frac))
    val_eps = set(int(e) for e in unique_eps[:n_val])
    val_idx = np.array(
        [i for i, e in enumerate(episode_id) if int(e) in val_eps], dtype=np.int64
    )
    train_idx = np.array(
        [i for i, e in enumerate(episode_id) if int(e) not in val_eps], dtype=np.int64
    )

    def to_split(idx: np.ndarray) -> Split:
        return Split(
            obs=torch.from_numpy(obs[idx]),
            action=torch.from_numpy(action[idx]),
            mask=torch.from_numpy(mask[idx]),
        )

    return to_split(train_idx), to_split(val_idx)


def _policy_logits(policy, obs: torch.Tensor) -> torch.Tensor:
    """MaskableActorCriticPolicy의 forward 일부를 직접 호출해 raw logits만 얻는다.

    sb3의 forward()는 액션 샘플링까지 포함하므로 학습에는 부적합.
    여기서는 cross-entropy 계산에 필요한 logits만 추출한다.
    """
    features = policy.extract_features(obs)
    if policy.share_features_extractor:
        latent_pi, _ = policy.mlp_extractor(features)
    else:
        pi_features, _ = features
        latent_pi = policy.mlp_extractor.forward_actor(pi_features)
    return policy.action_net(latent_pi)


def _masked_ce_and_acc(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    """valid 액션 안에서만 softmax하는 cross-entropy + top-1 정확도."""
    # invalid 위치를 -inf로 보내 softmax 분모에서 사실상 빠지게 함.
    masked_logits = logits.masked_fill(~mask, NEG_INF)
    loss = F.cross_entropy(masked_logits, target)
    pred = masked_logits.argmax(dim=1)
    acc = (pred == target).float().mean().item()
    return loss, acc


def _eval(
    policy,
    split: Split,
    device: str,
    batch_size: int,
) -> tuple[float, float]:
    policy.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_n = 0
    loader = DataLoader(
        TensorDataset(split.obs, split.action, split.mask),
        batch_size=batch_size,
        shuffle=False,
    )
    with torch.no_grad():
        for obs_b, act_b, mask_b in loader:
            obs_b = obs_b.to(device)
            act_b = act_b.to(device)
            mask_b = mask_b.to(device)
            logits = _policy_logits(policy, obs_b)
            loss, acc = _masked_ce_and_acc(logits, act_b, mask_b)
            n = obs_b.shape[0]
            total_loss += loss.item() * n
            total_acc += acc * n
            total_n += n
    return total_loss / total_n, total_acc / total_n


def train_bc(
    data_path: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    val_frac: float,
    seed: int,
    device_arg: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(device_arg)
    print(f"[bc] device={device} data={data_path}")

    # 1) 데이터 로드 + train/val 분할
    with np.load(data_path) as npz:
        obs = npz["obs"]
        action = npz["action"]
        mask = npz["mask"]
        episode_id = npz["episode_id"]
        scores = npz["scores"]
    print(
        f"[bc] loaded N={obs.shape[0]:,} samples, "
        f"{int(episode_id.max()) + 1} episodes, "
        f"score mean={scores.mean():.2f}"
    )
    rng = np.random.default_rng(seed)
    train_split, val_split = _split_by_episode(
        obs, action, mask, episode_id, val_frac=val_frac, rng=rng
    )
    print(
        f"[bc] split: train={train_split.obs.shape[0]:,} | "
        f"val={val_split.obs.shape[0]:,}"
    )

    # 2) sb3 MaskablePPO를 train.py와 같은 설정으로 만들고 .policy만 떼서 학습.
    #    learn()은 호출하지 않음 — PPO RL은 안 돌고, 우리가 직접 BC.
    env = _make_env(seed=seed, reward_mode="raw")
    policy_kwargs = dict(
        features_extractor_class=SmallBoardCNN,
        features_extractor_kwargs=dict(features_dim=128),
    )
    model = MaskablePPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=4096,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=seed,
        device=device,
    )
    policy = model.policy
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    # 3) 학습 루프
    train_loader = DataLoader(
        TensorDataset(train_split.obs, train_split.action, train_split.mask),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    for epoch in range(1, epochs + 1):
        policy.train()
        t0 = time.time()
        running_loss = 0.0
        running_acc = 0.0
        running_n = 0

        for obs_b, act_b, mask_b in train_loader:
            obs_b = obs_b.to(device)
            act_b = act_b.to(device)
            mask_b = mask_b.to(device)

            logits = _policy_logits(policy, obs_b)
            loss, acc = _masked_ce_and_acc(logits, act_b, mask_b)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()

            n = obs_b.shape[0]
            running_loss += loss.item() * n
            running_acc += acc * n
            running_n += n

        train_loss = running_loss / running_n
        train_acc = running_acc / running_n
        val_loss, val_acc = _eval(policy, val_split, device, batch_size)
        dt = time.time() - t0
        print(
            f"[bc] epoch {epoch:>2}/{epochs} | "
            f"train loss={train_loss:.4f} acc={train_acc:.3f} | "
            f"val loss={val_loss:.4f} acc={val_acc:.3f} | "
            f"{dt:.1f}s"
        )

    # 4) sb3 호환 zip 저장 — 추후 MaskablePPO.load(path)로 fine-tune 진입.
    stem = data_path.stem.replace("imitation_greedy_smallest_", "bc_greedy_smallest_")
    out_path = output_dir / f"{stem}_e{epochs}.zip"
    model.save(out_path)
    print(f"[bc] saved → {out_path}")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="Behavior cloning on greedy_smallest data.")
    p.add_argument("--data", type=Path, required=True, help="imitation .npz 경로")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--val-frac", type=float, default=0.1, help="판 단위 val 비율")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"]
    )
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = p.parse_args()

    train_bc(
        data_path=args.data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        val_frac=args.val_frac,
        seed=args.seed,
        device_arg=args.device,
    )


if __name__ == "__main__":
    main()
