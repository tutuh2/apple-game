"""화면 캡처에서 사과 격자를 자동 검출 → Calibration 자동 생성.

사용자가 마우스로 격자 모서리를 지정할 필요 없다.

알고리즘:
  1. HSV 빨간색 마스크
  2. connectedComponentsWithStats로 사과 컴포넌트 추출
  3. 비슷한 크기 컴포넌트만 남김 (광고/UI 빨간 요소 제거)
  4. x 좌표 정렬 → cluster 간격으로 cell_width 추정
  5. y 좌표 동일하게 cell_height
  6. 격자 origin (좌상단 사과 중심) → Calibration 객체

게임 시작 직후(170 사과 모두 있는 보드)에서 호출하면 가장 정확.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from agent.board_recognizer import Calibration
from env.fruit_box import BOARD_COLS, BOARD_ROWS


@dataclass
class DetectionResult:
    calibration: Calibration
    n_apples_detected: int
    confidence: float  # 0~1, 격자 정합도


def _red_mask(image_rgb: np.ndarray) -> np.ndarray:
    """이미지에서 빨간색 픽셀만 마스킹 (사이트 사과 색)."""
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    mask = ((h < 15) | (h > 165)) & (s > 100) & (v > 80)
    return mask.astype(np.uint8) * 255


def _find_apple_components(
    red_mask: np.ndarray,
    min_area: int = 100,
    max_area: int = 5000,
) -> list[tuple[int, int, int, int, int]]:
    """빨간 컴포넌트 추출 → (cx, cy, w, h, area) 리스트.

    크기로 1차 필터링해서 광고/배경의 빨간 요소를 제거.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

    n_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(
        cleaned, connectivity=8
    )
    results = []
    for i in range(1, n_labels):  # 0 = 배경
        x, y, w, h, area = stats[i]
        if not (min_area <= area <= max_area):
            continue
        ar = w / max(h, 1)
        if ar < 0.5 or ar > 2.0:
            continue
        cx, cy = int(centroids[i][0]), int(centroids[i][1])
        results.append((cx, cy, int(w), int(h), int(area)))
    return results


def _filter_by_grid_size(
    components: list[tuple[int, int, int, int, int]],
    target_count: int = BOARD_ROWS * BOARD_COLS,
) -> list[tuple[int, int, int, int, int]]:
    """target_count에 맞는 가장 자주 등장하는 크기 컴포넌트만 남김.

    사과는 모두 같은 크기. 다른 크기는 광고/UI.
    """
    if len(components) < target_count // 2:
        return components

    areas = np.array([c[4] for c in components])
    median = np.median(areas)
    lo, hi = median * 0.65, median * 1.5
    filtered = [c for c in components if lo <= c[4] <= hi]
    return filtered


def _cluster_axis(values: list[int], expected_n: int) -> Optional[list[int]]:
    """1D 좌표들을 expected_n 개의 클러스터로 나눠 중심값 반환.

    예: x 좌표들 → 17개 열 중심. 단순 정렬 + 간격 측정.
    Returns: 정렬된 N개 중심값 또는 검출 실패 시 None.
    """
    if len(values) < expected_n:
        return None

    sorted_v = sorted(values)
    cell_w = (sorted_v[-1] - sorted_v[0]) / (expected_n - 1)
    if cell_w <= 0:
        return None

    centers: list[list[int]] = [[] for _ in range(expected_n)]
    origin = sorted_v[0]
    for v in sorted_v:
        idx = int(round((v - origin) / cell_w))
        idx = max(0, min(expected_n - 1, idx))
        centers[idx].append(v)

    result = []
    for bucket in centers:
        if not bucket:
            return None  # 빈 클러스터 → 패턴 불일치
        result.append(int(np.median(bucket)))
    return result


def detect_grid(image_rgb: np.ndarray) -> Optional[DetectionResult]:
    """전체 화면(또는 큰 영역) 이미지에서 격자 자동 검출.

    Returns:
        DetectionResult 또는 검출 실패 시 None.
    """
    mask = _red_mask(image_rgb)
    components = _find_apple_components(mask)
    if len(components) < BOARD_ROWS * BOARD_COLS // 2:
        return None
    components = _filter_by_grid_size(components)
    if len(components) < BOARD_ROWS * BOARD_COLS // 2:
        return None

    xs = [c[0] for c in components]
    ys = [c[1] for c in components]
    col_centers = _cluster_axis(xs, BOARD_COLS)
    row_centers = _cluster_axis(ys, BOARD_ROWS)
    if col_centers is None or row_centers is None:
        return None

    cell_w = (col_centers[-1] - col_centers[0]) / (BOARD_COLS - 1)
    cell_h = (row_centers[-1] - row_centers[0]) / (BOARD_ROWS - 1)
    ratio = cell_w / max(cell_h, 1)
    if ratio < 0.7 or ratio > 1.3:
        return None

    matched = 0
    for cx, cy, _, _, _ in components:
        col_idx = min(
            range(BOARD_COLS), key=lambda i: abs(col_centers[i] - cx)
        )
        row_idx = min(
            range(BOARD_ROWS), key=lambda i: abs(row_centers[i] - cy)
        )
        dx = abs(col_centers[col_idx] - cx)
        dy = abs(row_centers[row_idx] - cy)
        if dx < cell_w / 3 and dy < cell_h / 3:
            matched += 1

    confidence = matched / (BOARD_ROWS * BOARD_COLS)
    if confidence < 0.5:
        return None

    calibration = Calibration(
        top_left_x=col_centers[0],
        top_left_y=row_centers[0],
        bottom_right_x=col_centers[-1],
        bottom_right_y=row_centers[-1],
        cell_width=float(cell_w),
        cell_height=float(cell_h),
        rows=BOARD_ROWS,
        cols=BOARD_COLS,
    )
    return DetectionResult(
        calibration=calibration,
        n_apples_detected=len(components),
        confidence=confidence,
    )
