"""PointerNet용 PPO 학습 루프 (SB3 없이 PyTorch 직접 구현).

가변 candidate 수 K를 다루기 위해 SB3 표준 인터페이스를 못 씀.
표준 PPO + clipped objective를 그대로 구현하되 batch마다 K_max로 패딩.

핵심 흐름:
  1. Rollout 수집 (n_steps): env와 상호작용하며 (board, cands, action,
     log_prob, value, reward, done) 누적.
  2. GAE advantage 계산 (gamma, lam).
  3. Update (n_epochs × 미니배치): clipped policy loss + value loss +
     entropy bonus.
  4. 로그 출력 후 다음 iteration.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from agent.pointer_net import PointerNet, pad_candidates
from env.action_space import compute_candidates
from env.fruit_box import MAX_APPLE, Action, FruitBox
from env.fruit_box_gym import (
    SMALL_REGION_BONUS_MULT,
    SMALL_REGION_THRESHOLD,
    TERMINAL_SCORE_BONUS,
)


@dataclass
class PPOConfig:
    total_steps: int = 200_000
    n_steps: int = 2048
    n_epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    reward_mode: str = "shaped"  # "raw" | "shaped"
    seed: int = 0
    device: str = "cpu"


@dataclass
class RolloutBuffer:
    """가변 K rollout 저장소. 학습 시 패딩해서 미니배치 생성."""

    boards: list[np.ndarray] = field(default_factory=list)
    cand_feats: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)

    def clear(self) -> None:
        for lst in (
            self.boards,
            self.cand_feats,
            self.actions,
            self.log_probs,
            self.values,
            self.rewards,
            self.dones,
        ):
            lst.clear()

    def __len__(self) -> int:
        return len(self.actions)


def _board_to_obs(board: np.ndarray) -> np.ndarray:
    return (board.astype(np.float32) / float(MAX_APPLE))[None, :, :]


def _shape_reward(
    raw_reward: int,
    region_size: int,
    terminated: bool,
    final_score: int,
    mode: str,
) -> float:
    r = float(raw_reward)
    if mode == "shaped":
        if raw_reward > 0 and region_size <= SMALL_REGION_THRESHOLD:
            r += raw_reward * SMALL_REGION_BONUS_MULT
        if terminated:
            r += final_score * TERMINAL_SCORE_BONUS
    return r


def _collect_rollout(
    game: FruitBox,
    model: PointerNet,
    buffer: RolloutBuffer,
    n_steps: int,
    cfg: PPOConfig,
    device: torch.device,
    rng: np.random.Generator,
    ep_returns_window: list[float],
    ep_lengths_window: list[int],
    state: dict,
) -> None:
    model.eval()
    with torch.no_grad():
        for _ in range(n_steps):
            board_copy = game.board.copy()
            coords, feats = compute_candidates(board_copy)

            if len(coords) == 0:
                seed = int(rng.integers(0, 1_000_000_000))
                game.reset(seed=seed)
                ep_returns_window.append(state["ep_return"])
                ep_lengths_window.append(state["ep_len"])
                state["ep_return"] = 0.0
                state["ep_len"] = 0
                continue

            board_t = torch.from_numpy(_board_to_obs(board_copy)).unsqueeze(0).to(device)
            feats_t = torch.from_numpy(feats).unsqueeze(0).to(device)
            mask_t = torch.ones(1, feats.shape[0], dtype=torch.bool, device=device)

            logits, value = model(board_t, feats_t, mask_t)
            dist = Categorical(logits=logits[0])
            action_idx = int(dist.sample().item())
            log_prob = float(
                dist.log_prob(torch.tensor(action_idx, device=device)).item()
            )
            value_f = float(value.item())

            r1, c1, r2, c2 = (int(x) for x in coords[action_idx])
            region_size = (r2 - r1 + 1) * (c2 - c1 + 1)
            raw_reward = game.step(Action(r1, c1, r2, c2))
            new_coords, _ = compute_candidates(game.board)
            terminated = len(new_coords) == 0
            shaped = _shape_reward(
                raw_reward,
                region_size,
                terminated,
                game.score,
                cfg.reward_mode,
            )

            buffer.boards.append(_board_to_obs(board_copy))
            buffer.cand_feats.append(feats)
            buffer.actions.append(action_idx)
            buffer.log_probs.append(log_prob)
            buffer.values.append(value_f)
            buffer.rewards.append(shaped)
            buffer.dones.append(terminated)

            state["ep_return"] += shaped
            state["ep_len"] += 1

            if terminated:
                seed = int(rng.integers(0, 1_000_000_000))
                game.reset(seed=seed)
                ep_returns_window.append(state["ep_return"])
                ep_lengths_window.append(state["ep_len"])
                state["ep_return"] = 0.0
                state["ep_len"] = 0


def _compute_gae(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    last_value: float,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(n)):
        next_value = last_value if t == n - 1 else values[t + 1]
        next_nonterminal = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        gae = delta + gamma * lam * next_nonterminal * gae
        advantages[t] = gae
    returns = advantages + np.array(values, dtype=np.float32)
    return advantages, returns


def _update(
    model: PointerNet,
    optimizer: torch.optim.Optimizer,
    buffer: RolloutBuffer,
    advantages: np.ndarray,
    returns: np.ndarray,
    cfg: PPOConfig,
    device: torch.device,
) -> dict:
    model.train()
    n = len(buffer)
    indices = np.arange(n)

    adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    old_log_probs = np.array(buffer.log_probs, dtype=np.float32)

    stats = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
    }
    n_batches = 0

    for _ in range(cfg.n_epochs):
        np.random.shuffle(indices)
        for start in range(0, n, cfg.batch_size):
            batch_idx = indices[start : start + cfg.batch_size]
            if len(batch_idx) == 0:
                continue

            boards = torch.from_numpy(
                np.stack([buffer.boards[i] for i in batch_idx])
            ).to(device)
            feat_list = [torch.from_numpy(buffer.cand_feats[i]) for i in batch_idx]
            feats_padded, mask_padded = pad_candidates(feat_list)
            feats_padded = feats_padded.to(device)
            mask_padded = mask_padded.to(device)

            actions_t = torch.tensor(
                [buffer.actions[i] for i in batch_idx], dtype=torch.long, device=device
            )
            old_lp_t = torch.from_numpy(old_log_probs[batch_idx]).to(device)
            adv_t = torch.from_numpy(adv_norm[batch_idx]).to(device)
            ret_t = torch.from_numpy(returns[batch_idx]).to(device)

            logits, values = model(boards, feats_padded, mask_padded)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions_t)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_lp_t)
            unclipped = ratio * adv_t
            clipped = torch.clamp(ratio, 1 - cfg.clip_range, 1 + cfg.clip_range) * adv_t
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = F.mse_loss(values, ret_t)

            loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                log_ratio = new_log_probs - old_lp_t
                approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean().item()
                clip_frac = ((ratio - 1).abs() > cfg.clip_range).float().mean().item()

            stats["policy_loss"] += policy_loss.item()
            stats["value_loss"] += value_loss.item()
            stats["entropy"] += entropy.item()
            stats["approx_kl"] += approx_kl
            stats["clip_fraction"] += clip_frac
            n_batches += 1

    if n_batches > 0:
        for k in stats:
            stats[k] /= n_batches
    return stats


def train_ppo(cfg: PPOConfig, log_interval: int = 1) -> tuple[PointerNet, dict]:
    """PPO 학습 루프. (model, last_stats) 반환."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    device = torch.device(cfg.device)
    model = PointerNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    game = FruitBox(seed=cfg.seed)
    game.reset(seed=cfg.seed)

    buffer = RolloutBuffer()
    ep_returns_window: list[float] = []
    ep_lengths_window: list[int] = []
    state = {"ep_return": 0.0, "ep_len": 0}

    total_iterations = max(1, cfg.total_steps // cfg.n_steps)
    start_time = time.time()
    total_collected = 0

    print(
        f"[ppo] device={cfg.device}, total_steps={cfg.total_steps:,}, "
        f"n_steps={cfg.n_steps}, iterations={total_iterations}, "
        f"reward_mode={cfg.reward_mode}"
    )

    last_stats: dict = {}
    for it in range(1, total_iterations + 1):
        buffer.clear()
        _collect_rollout(
            game=game,
            model=model,
            buffer=buffer,
            n_steps=cfg.n_steps,
            cfg=cfg,
            device=device,
            rng=rng,
            ep_returns_window=ep_returns_window,
            ep_lengths_window=ep_lengths_window,
            state=state,
        )

        # Bootstrap last value
        model.eval()
        with torch.no_grad():
            coords, feats = compute_candidates(game.board)
            if len(coords) == 0:
                last_value = 0.0
            else:
                board_t = (
                    torch.from_numpy(_board_to_obs(game.board)).unsqueeze(0).to(device)
                )
                feats_t = torch.from_numpy(feats).unsqueeze(0).to(device)
                mask_t = torch.ones(1, feats.shape[0], dtype=torch.bool, device=device)
                _, v = model(board_t, feats_t, mask_t)
                last_value = float(v.item())

        advantages, returns = _compute_gae(
            buffer.rewards,
            buffer.values,
            buffer.dones,
            last_value,
            cfg.gamma,
            cfg.gae_lambda,
        )
        stats = _update(model, optimizer, buffer, advantages, returns, cfg, device)
        last_stats = stats

        total_collected += len(buffer)

        if it % log_interval == 0:
            elapsed = time.time() - start_time
            fps = int(total_collected / max(elapsed, 1e-6))
            ep_rew_mean = (
                float(np.mean(ep_returns_window[-100:])) if ep_returns_window else 0.0
            )
            ep_len_mean = (
                float(np.mean(ep_lengths_window[-100:])) if ep_lengths_window else 0.0
            )
            print(
                "-" * 60 + "\n"
                f"| iter {it:4d}/{total_iterations} | total_steps {total_collected:>7,} | "
                f"fps {fps:4d} | elapsed {elapsed:6.0f}s\n"
                f"| ep_rew_mean {ep_rew_mean:7.2f} | ep_len_mean {ep_len_mean:5.1f} | "
                f"episodes {len(ep_returns_window):4d}\n"
                f"| policy_loss {stats['policy_loss']:+.4f} | value_loss {stats['value_loss']:.4f} | "
                f"entropy {stats['entropy']:+.4f}\n"
                f"| approx_kl {stats['approx_kl']:.4f} | clip_frac {stats['clip_fraction']:.3f}\n"
                + "-" * 60
            )

    return model, last_stats


def evaluate_model(
    model: PointerNet,
    n_episodes: int,
    seed_start: int,
    device: torch.device,
    deterministic: bool = True,
) -> tuple[float, float, list[int]]:
    """raw 게임 점수로 평가. (mean, std, scores) 반환."""
    model.eval()
    scores: list[int] = []
    with torch.no_grad():
        for s in range(n_episodes):
            game = FruitBox(seed=seed_start + s)
            game.reset(seed=seed_start + s)
            while True:
                coords, feats = compute_candidates(game.board)
                if len(coords) == 0:
                    break
                board_t = (
                    torch.from_numpy(_board_to_obs(game.board)).unsqueeze(0).to(device)
                )
                feats_t = torch.from_numpy(feats).unsqueeze(0).to(device)
                mask_t = torch.ones(1, feats.shape[0], dtype=torch.bool, device=device)
                logits, _ = model(board_t, feats_t, mask_t)
                if deterministic:
                    action_idx = int(torch.argmax(logits[0]).item())
                else:
                    action_idx = int(Categorical(logits=logits[0]).sample().item())
                r1, c1, r2, c2 = (int(x) for x in coords[action_idx])
                game.step(Action(r1, c1, r2, c2))
            scores.append(game.score)
    arr = np.array(scores)
    return float(arr.mean()), float(arr.std()), scores
