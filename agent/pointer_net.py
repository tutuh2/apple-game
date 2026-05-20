"""Pointer Network 정책 + value head (옵션 C — v4).

v1~v3의 고정 Discrete(8415) 액션 공간 한계를 우회하기 위해, 그때그때
valid한 K개 후보 직사각형 중에서 attention score로 하나를 선택한다.
K는 가변. AlphaZero 계열의 정책 표현 방식.

forward는 (logits, values) 반환.
  - logits: (B, K_max). padded 위치는 NEG_INF.
  - values: (B,). state value 예측.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from env.action_space import CANDIDATE_FEATURE_DIM
from env.fruit_box import BOARD_COLS, BOARD_ROWS

NEG_INF = -1e9


class PointerNet(nn.Module):
    """Board + variable candidate set → logits over candidates + value."""

    def __init__(
        self,
        board_dim: int = 128,
        cand_dim: int = 64,
        score_hidden: int = 64,
        value_hidden: int = 64,
    ):
        super().__init__()
        self.board_dim = board_dim
        self.cand_dim = cand_dim

        self.board_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * BOARD_ROWS * BOARD_COLS, board_dim),
            nn.ReLU(),
        )

        self.cand_mlp = nn.Sequential(
            nn.Linear(CANDIDATE_FEATURE_DIM, cand_dim),
            nn.ReLU(),
            nn.Linear(cand_dim, cand_dim),
            nn.ReLU(),
        )

        self.score_mlp = nn.Sequential(
            nn.Linear(board_dim + cand_dim, score_hidden),
            nn.ReLU(),
            nn.Linear(score_hidden, 1),
        )

        self.value_mlp = nn.Sequential(
            nn.Linear(board_dim, value_hidden),
            nn.ReLU(),
            nn.Linear(value_hidden, 1),
        )

    def forward(
        self,
        board: torch.Tensor,
        cand_features: torch.Tensor,
        cand_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            board: (B, 1, BOARD_ROWS, BOARD_COLS) float32
            cand_features: (B, K_max, CANDIDATE_FEATURE_DIM) float32
            cand_mask: (B, K_max) bool, True = 실제 후보, False = padding

        Returns:
            logits: (B, K_max). padding 위치는 NEG_INF.
            values: (B,).
        """
        _, k_max, _ = cand_features.shape

        board_feat = self.board_cnn(board)  # (B, board_dim)
        cand_feat = self.cand_mlp(cand_features)  # (B, K_max, cand_dim)

        board_feat_expand = board_feat.unsqueeze(1).expand(-1, k_max, -1)
        joint = torch.cat([board_feat_expand, cand_feat], dim=-1)
        scores = self.score_mlp(joint).squeeze(-1)  # (B, K_max)

        logits = torch.where(cand_mask, scores, torch.full_like(scores, NEG_INF))
        values = self.value_mlp(board_feat).squeeze(-1)
        return logits, values


def pad_candidates(
    feat_list: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """가변 K의 candidate feature list를 패딩해 (B, K_max, F) 배치로 만듦.

    Args:
        feat_list: 길이 B의 리스트. 각 원소는 (K_i, F) float32 tensor.

    Returns:
        padded: (B, K_max, F)
        mask:   (B, K_max) bool. True = 실제 후보.
    """
    b = len(feat_list)
    if b == 0:
        raise ValueError("빈 리스트")
    f_dim = feat_list[0].shape[-1]
    k_max = max(max((t.shape[0] for t in feat_list), default=1), 1)

    padded = torch.zeros(b, k_max, f_dim, dtype=torch.float32)
    mask = torch.zeros(b, k_max, dtype=torch.bool)
    for i, t in enumerate(feat_list):
        k_i = t.shape[0]
        if k_i > 0:
            padded[i, :k_i] = t
            mask[i, :k_i] = True
    return padded, mask
