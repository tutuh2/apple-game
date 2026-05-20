"""PointerNet 모델 + compute_candidates 단위 테스트."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from agent.pointer_net import NEG_INF, PointerNet, pad_candidates
from env.action_space import (
    CANDIDATE_FEATURE_DIM,
    compute_action_mask,
    compute_candidates,
)
from env.fruit_box import BOARD_COLS, BOARD_ROWS, FruitBox


class TestComputeCandidates:
    def test_empty_board(self):
        board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)
        coords, feats = compute_candidates(board)
        assert coords.shape == (0, 4)
        assert feats.shape == (0, CANDIDATE_FEATURE_DIM)

    def test_single_4_plus_6(self):
        # 9로 가득 찬 보드에 (0,0)=4, (0,1)=6 한 쌍만 — 9는 합=10 만들지
        # 못하므로 유일한 합=10 영역은 (0,0,0,1).
        board = np.full((BOARD_ROWS, BOARD_COLS), 9, dtype=np.int8)
        board[0, 0] = 4
        board[0, 1] = 6
        coords, feats = compute_candidates(board)
        assert coords.shape == (1, 4)
        assert tuple(coords[0].tolist()) == (0, 0, 0, 1)
        # 내부 통계: 4와 6 두 종류, max=6, min=4
        assert feats[0, 7] == pytest.approx(2 / 9)
        assert feats[0, 8] == pytest.approx(6 / 9)
        assert feats[0, 9] == pytest.approx(4 / 9)
        # lookahead: 이 후보 제거 후 보드에는 9만 남아 valid 후보 없음.
        assert feats[0, 10] == pytest.approx(0.0)
        # delta_K: (0 - 1) / 100 = -0.01
        assert feats[0, 11] == pytest.approx(-1 / 100)

    def test_count_matches_mask(self):
        env = FruitBox(seed=7)
        env.reset()
        mask = compute_action_mask(env.board)
        coords, feats = compute_candidates(env.board)
        assert coords.shape[0] == int(mask.sum())
        assert feats.shape == (coords.shape[0], CANDIDATE_FEATURE_DIM)

    def test_features_normalized(self):
        """좌표/크기/내부통계는 [0, 1] 범위. lookahead 10번은 ~[0, 1+],
        11번 delta는 부호 있음 — 비교 시 그쪽 두 열은 별도 처리."""
        env = FruitBox(seed=11)
        env.reset()
        _, feats = compute_candidates(env.board)
        assert feats.dtype == np.float32
        # 0~9 컬럼: [0, 1] 정규화
        assert (feats[:, :10] >= 0.0).all()
        assert (feats[:, :10] <= 1.0 + 1e-6).all()
        # 10번 lookahead_K/200: 음수는 절대 안 됨, 합리적 범위 안.
        assert (feats[:, 10] >= 0.0).all()
        assert (feats[:, 10] <= 2.0).all()
        # 11번 delta_K/100: 부호 있음, 절대값은 작아야 함.
        assert (np.abs(feats[:, 11]) <= 2.0).all()

    def test_lookahead_signal_present(self):
        """랜덤 보드에서 lookahead 특성이 후보 간 차이를 만들어내는지.
        모든 후보의 lookahead가 동일하면 사실상 신호가 없는 것."""
        env = FruitBox(seed=13)
        env.reset()
        _, feats = compute_candidates(env.board)
        if feats.shape[0] >= 2:
            lookahead_values = feats[:, 10]
            assert lookahead_values.std() > 0.0


class TestPadCandidates:
    def test_pads_to_max_k(self):
        a = torch.zeros(3, CANDIDATE_FEATURE_DIM)
        b = torch.zeros(5, CANDIDATE_FEATURE_DIM)
        c = torch.zeros(1, CANDIDATE_FEATURE_DIM)
        padded, mask = pad_candidates([a, b, c])
        assert padded.shape == (3, 5, CANDIDATE_FEATURE_DIM)
        assert mask.shape == (3, 5)
        assert mask[0, :3].all() and not mask[0, 3:].any()
        assert mask[1, :].all()
        assert mask[2, 0] and not mask[2, 1:].any()

    def test_empty_candidate_handled(self):
        a = torch.zeros(0, CANDIDATE_FEATURE_DIM)
        b = torch.zeros(2, CANDIDATE_FEATURE_DIM)
        padded, mask = pad_candidates([a, b])
        assert padded.shape == (2, 2, CANDIDATE_FEATURE_DIM)
        assert not mask[0].any()
        assert mask[1, :2].all()


class TestPointerNetForward:
    def test_forward_shapes(self):
        model = PointerNet()
        b, k = 4, 7
        board = torch.zeros(b, 1, BOARD_ROWS, BOARD_COLS)
        feats = torch.zeros(b, k, CANDIDATE_FEATURE_DIM)
        mask = torch.ones(b, k, dtype=torch.bool)
        logits, values = model(board, feats, mask)
        assert logits.shape == (b, k)
        assert values.shape == (b,)

    def test_padded_logits_are_neg_inf(self):
        model = PointerNet()
        b, k = 2, 4
        board = torch.zeros(b, 1, BOARD_ROWS, BOARD_COLS)
        feats = torch.zeros(b, k, CANDIDATE_FEATURE_DIM)
        mask = torch.tensor(
            [[True, True, False, False], [True, True, True, False]], dtype=torch.bool
        )
        logits, _ = model(board, feats, mask)
        assert logits[0, 2].item() <= NEG_INF / 2
        assert logits[0, 3].item() <= NEG_INF / 2
        assert logits[1, 3].item() <= NEG_INF / 2
        assert logits[0, 0].item() > NEG_INF / 2

    def test_softmax_ignores_padding(self):
        model = PointerNet()
        b, k = 1, 5
        board = torch.zeros(b, 1, BOARD_ROWS, BOARD_COLS)
        feats = torch.zeros(b, k, CANDIDATE_FEATURE_DIM)
        mask = torch.tensor([[True, True, False, False, False]], dtype=torch.bool)
        logits, _ = model(board, feats, mask)
        probs = torch.softmax(logits, dim=-1)
        assert abs(probs[0, 2].item()) < 1e-6
        assert abs(probs[0, 3].item()) < 1e-6
        assert abs(probs[0, 4].item()) < 1e-6
        assert abs(probs[0, :2].sum().item() - 1.0) < 1e-5

    def test_variable_k_across_calls(self):
        model = PointerNet()
        for k in [1, 3, 100]:
            board = torch.zeros(2, 1, BOARD_ROWS, BOARD_COLS)
            feats = torch.zeros(2, k, CANDIDATE_FEATURE_DIM)
            mask = torch.ones(2, k, dtype=torch.bool)
            logits, values = model(board, feats, mask)
            assert logits.shape == (2, k)
            assert values.shape == (2,)
