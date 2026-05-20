"""직사각형 액션 ↔ flat 인덱스 인코딩 + action mask 생성.

PPO 같은 정책 모델은 단일 정수 액션을 요구하므로 (r1, c1, r2, c2)를
하나의 정수로 매핑한다. 행 쌍과 열 쌍을 각각 인덱싱한 뒤 곱해서 결합한다.

행 쌍 인덱스: r1 <= r2 를 만족하는 (r1, r2) 조합을 사전식으로 0부터 매김.
열 쌍 인덱스도 동일.

flat_index = row_pair_index * NUM_COL_PAIRS + col_pair_index
"""

from __future__ import annotations

import numpy as np

from env.fruit_box import BOARD_COLS, BOARD_ROWS, MAX_APPLE, TARGET_SUM


def _build_pair_table(n: int) -> tuple[np.ndarray, np.ndarray]:
    """크기 n 축에서 i <= j 쌍 (i, j)을 사전식으로 나열한 테이블 생성.

    Returns:
        pairs: shape (num_pairs, 2), 각 행이 (i, j)
        index_of: shape (n, n), index_of[i, j] = 사전식 인덱스 (i > j이면 -1)
    """
    pairs = [(i, j) for i in range(n) for j in range(i, n)]
    pair_arr = np.array(pairs, dtype=np.int16)
    index_of = -np.ones((n, n), dtype=np.int32)
    for idx, (i, j) in enumerate(pairs):
        index_of[i, j] = idx
    return pair_arr, index_of


_ROW_PAIRS, _ROW_PAIR_INDEX = _build_pair_table(BOARD_ROWS)
_COL_PAIRS, _COL_PAIR_INDEX = _build_pair_table(BOARD_COLS)

NUM_ROW_PAIRS = len(_ROW_PAIRS)  # 55
NUM_COL_PAIRS = len(_COL_PAIRS)  # 153
NUM_ACTIONS = NUM_ROW_PAIRS * NUM_COL_PAIRS  # 8415


def encode(r1: int, c1: int, r2: int, c2: int) -> int:
    """직사각형 좌표를 flat 인덱스로 변환."""
    if not (0 <= r1 <= r2 < BOARD_ROWS):
        raise ValueError(f"잘못된 행 좌표: r1={r1}, r2={r2}")
    if not (0 <= c1 <= c2 < BOARD_COLS):
        raise ValueError(f"잘못된 열 좌표: c1={c1}, c2={c2}")
    row_idx = int(_ROW_PAIR_INDEX[r1, r2])
    col_idx = int(_COL_PAIR_INDEX[c1, c2])
    return row_idx * NUM_COL_PAIRS + col_idx


def decode(flat_index: int) -> tuple[int, int, int, int]:
    """flat 인덱스를 (r1, c1, r2, c2)로 변환."""
    if not (0 <= flat_index < NUM_ACTIONS):
        raise ValueError(f"잘못된 액션 인덱스: {flat_index}")
    row_idx, col_idx = divmod(flat_index, NUM_COL_PAIRS)
    r1, r2 = _ROW_PAIRS[row_idx]
    c1, c2 = _COL_PAIRS[col_idx]
    return int(r1), int(c1), int(r2), int(c2)


def compute_action_mask(board: np.ndarray) -> np.ndarray:
    """현재 보드에서 valid한 액션(합 == 10)의 boolean mask 반환.

    2D prefix sum으로 모든 직사각형의 합을 O(R²·C²)에 계산.

    Returns:
        shape (NUM_ACTIONS,) bool array. True = 합이 정확히 10.
    """
    psum = np.zeros((BOARD_ROWS + 1, BOARD_COLS + 1), dtype=np.int32)
    psum[1:, 1:] = board.cumsum(axis=0).cumsum(axis=1)

    mask = np.zeros(NUM_ACTIONS, dtype=bool)

    for row_idx in range(NUM_ROW_PAIRS):
        r1, r2 = _ROW_PAIRS[row_idx]
        col_psum = psum[r2 + 1] - psum[r1]
        base = row_idx * NUM_COL_PAIRS
        for col_idx in range(NUM_COL_PAIRS):
            c1, c2 = _COL_PAIRS[col_idx]
            rect_sum = int(col_psum[c2 + 1] - col_psum[c1])
            if rect_sum == TARGET_SUM:
                mask[base + col_idx] = True

    return mask


CANDIDATE_FEATURE_DIM = 12

# lookahead count 정규화 상수. 실제 K는 보드에 따라 ~0..200 변동.
# 정확한 상한이 아니라 입력 스케일 [0, 1] 근처 유지가 목적.
_LOOKAHEAD_NORM = 200.0
_LOOKAHEAD_DELTA_NORM = 100.0


def _count_valid_rects(board: np.ndarray) -> int:
    """보드의 합=10 직사각형 개수만 세기 (좌표 저장 없이 빠르게).

    compute_action_mask와 거의 같은 로직이지만 개수만 세고 끝.
    lookahead feature 계산에서 후보 K번 호출되므로 핫패스다.
    """
    psum = np.zeros((BOARD_ROWS + 1, BOARD_COLS + 1), dtype=np.int32)
    psum[1:, 1:] = board.cumsum(axis=0).cumsum(axis=1)
    col_starts = _COL_PAIRS[:, 0]
    col_ends_p1 = _COL_PAIRS[:, 1] + 1
    count = 0
    for row_idx in range(NUM_ROW_PAIRS):
        r1, r2 = _ROW_PAIRS[row_idx]
        col_psum = psum[r2 + 1] - psum[r1]
        diffs = col_psum[col_ends_p1] - col_psum[col_starts]
        count += int((diffs == TARGET_SUM).sum())
    return count


def compute_candidates(board: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """현재 보드의 valid 직사각형 후보 K개를 좌표 + 12개 특성으로 반환.

    v5에서 8 → 12 특성으로 확장. 추가된 4개:
        7  unique_apples / 9     내부에 등장한 사과 종류 수 (다양성)
        8  max_apple / 9         내부 최대 사과 (큰 사과를 포함하는가)
        9  min_apple / 9         내부 최소 사과
       10  lookahead_K / 200     이 액션 후 valid 후보 수 (절대값)
       11  delta_K / 100         이 액션 후 K 변화량 (현재 K 대비)

    핵심은 10, 11 — greedy_smallest가 강했던 "보드 모양 보존" 신호를
    명시적으로 정책에 주입. lookahead가 클수록 좋은 액션.

    Returns:
        coords: shape (K, 4) int16. 각 행이 (r1, c1, r2, c2).
        features: shape (K, CANDIDATE_FEATURE_DIM=12) float32.
        K=0 가능 (게임 종료 직전).
    """
    psum = np.zeros((BOARD_ROWS + 1, BOARD_COLS + 1), dtype=np.int32)
    psum[1:, 1:] = board.cumsum(axis=0).cumsum(axis=1)

    coords_list: list[tuple[int, int, int, int]] = []

    for row_idx in range(NUM_ROW_PAIRS):
        r1, r2 = _ROW_PAIRS[row_idx]
        col_psum = psum[r2 + 1] - psum[r1]
        for col_idx in range(NUM_COL_PAIRS):
            c1, c2 = _COL_PAIRS[col_idx]
            rect_sum = int(col_psum[c2 + 1] - col_psum[c1])
            if rect_sum == TARGET_SUM:
                coords_list.append((int(r1), int(c1), int(r2), int(c2)))

    k = len(coords_list)
    if k == 0:
        return (
            np.zeros((0, 4), dtype=np.int16),
            np.zeros((0, CANDIDATE_FEATURE_DIM), dtype=np.float32),
        )

    coords = np.array(coords_list, dtype=np.int16)
    features = np.zeros((k, CANDIDATE_FEATURE_DIM), dtype=np.float32)
    r1s = coords[:, 0].astype(np.float32)
    c1s = coords[:, 1].astype(np.float32)
    r2s = coords[:, 2].astype(np.float32)
    c2s = coords[:, 3].astype(np.float32)
    h = r2s - r1s + 1
    w = c2s - c1s + 1

    # 0~6: 좌표 + 크기 (기존)
    features[:, 0] = r1s / BOARD_ROWS
    features[:, 1] = c1s / BOARD_COLS
    features[:, 2] = r2s / BOARD_ROWS
    features[:, 3] = c2s / BOARD_COLS
    features[:, 4] = h / BOARD_ROWS
    features[:, 5] = w / BOARD_COLS
    features[:, 6] = (h * w) / (BOARD_ROWS * BOARD_COLS)

    # 7~9: 내부 통계  |  10~11: lookahead
    # 후보당 board 슬라이스 1회 + 임시 변경 후 _count_valid_rects 1회.
    # "이걸 두면 다음에 valid 후보가 몇 개 남는가" = greedy_smallest의
    # 핵심 휴리스틱을 정책에 명시적으로 노출.
    current_k_norm = float(k) / _LOOKAHEAD_DELTA_NORM
    for i in range(k):
        r1i = int(coords[i, 0])
        c1i = int(coords[i, 1])
        r2i = int(coords[i, 2])
        c2i = int(coords[i, 3])
        patch = board[r1i : r2i + 1, c1i : c2i + 1]

        # 내부 통계 — 합=10인 영역에 0 셀이 섞일 수 있으므로 nonzero만 본다.
        nonzero = patch[patch > 0]
        if nonzero.size == 0:
            unique_n = 0
            max_v = 0
            min_v = 0
        else:
            unique_n = int(np.unique(nonzero).size)
            max_v = int(nonzero.max())
            min_v = int(nonzero.min())
        features[i, 7] = unique_n / float(MAX_APPLE)
        features[i, 8] = max_v / float(MAX_APPLE)
        features[i, 9] = min_v / float(MAX_APPLE)

        # lookahead — 직사각형을 0으로 만든 가상 보드의 valid 후보 수.
        saved = patch.copy()
        patch[:] = 0
        next_k = _count_valid_rects(board)
        patch[:] = saved  # 복구

        features[i, 10] = float(next_k) / _LOOKAHEAD_NORM
        features[i, 11] = (float(next_k) / _LOOKAHEAD_DELTA_NORM) - current_k_norm

    return coords, features
