"""PointerNet을 MCTS의 prior + value 함수로 연결.

PointerNet.forward는 (logits, value) 반환. MCTS는 (priors_sum1, value)를
원하므로 softmax 변환 + scalar 변환.
"""

from __future__ import annotations

import numpy as np
import torch

from agent.mcts import ModelFn
from agent.pointer_net import PointerNet
from env.action_space import compute_candidates
from env.fruit_box import MAX_APPLE


def _board_to_obs(board: np.ndarray) -> np.ndarray:
    return (board.astype(np.float32) / float(MAX_APPLE))[None, :, :]


def make_model_fn(model: PointerNet, device: torch.device) -> ModelFn:
    """PointerNet 인스턴스를 MCTS용 model_fn으로 래핑.

    반환된 함수는 board 받아 (priors, value) 반환.
    매 호출마다 inference 1번. 그래서 MCTS 한 턴에 N번 호출됨.
    """
    model.eval()

    def model_fn(board: np.ndarray) -> tuple[np.ndarray, float]:
        coords, feats = compute_candidates(board)
        k = len(coords)
        if k == 0:
            return np.zeros(0, dtype=np.float32), 0.0

        with torch.no_grad():
            board_t = torch.from_numpy(_board_to_obs(board)).unsqueeze(0).to(device)
            feats_t = torch.from_numpy(feats).unsqueeze(0).to(device)
            mask_t = torch.ones(1, k, dtype=torch.bool, device=device)
            logits, value = model(board_t, feats_t, mask_t)
            priors = torch.softmax(logits[0], dim=-1).cpu().numpy().astype(np.float32)
            v = float(value.item())
        return priors, v

    return model_fn
