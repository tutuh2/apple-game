"""사이트 화면 캡처 → 보드 (10, 17) int8 변환.

전략:
  1. calibration 좌표로 격자 영역을 셀별로 자르기 (170개 셀)
  2. 셀별로 "사과가 있는가" 판정 — HSV로 빨간색 픽셀 비율 측정 (빈 셀은 0)
  3. 사과가 있는 셀 → 1~9 숫자 인식
     - 1단계 (자동 학습): 첫 보드에서 셀끼리 픽셀 유사도로 9개 클러스터 추출
                         흰 픽셀 카운트로 라벨 추정 → 보드 합이 10의 배수면 채택
     - 2단계 (운영): 템플릿 매칭으로 매 수마다 빠르게 인식
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from env.fruit_box import BOARD_COLS, BOARD_ROWS


@dataclass
class Calibration:
    top_left_x: int
    top_left_y: int
    bottom_right_x: int
    bottom_right_y: int
    cell_width: float
    cell_height: float
    rows: int = BOARD_ROWS
    cols: int = BOARD_COLS

    @classmethod
    def load(cls, path: Path) -> "Calibration":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            top_left_x=int(data["top_left_x"]),
            top_left_y=int(data["top_left_y"]),
            bottom_right_x=int(data["bottom_right_x"]),
            bottom_right_y=int(data["bottom_right_y"]),
            cell_width=float(data["cell_width"]),
            cell_height=float(data["cell_height"]),
            rows=int(data.get("rows", BOARD_ROWS)),
            cols=int(data.get("cols", BOARD_COLS)),
        )

    def cell_center(self, r: int, c: int) -> tuple[int, int]:
        x = self.top_left_x + int(round(c * self.cell_width))
        y = self.top_left_y + int(round(r * self.cell_height))
        return x, y

    def screen_region(self, padding: int = 10) -> dict:
        """전체 격자가 들어가는 화면 영역 (mss grab용)."""
        left = self.top_left_x - int(self.cell_width // 2) - padding
        top = self.top_left_y - int(self.cell_height // 2) - padding
        right = self.bottom_right_x + int(self.cell_width // 2) + padding
        bottom = self.bottom_right_y + int(self.cell_height // 2) + padding
        return {
            "left": left,
            "top": top,
            "width": right - left,
            "height": bottom - top,
        }


def extract_cells(
    image: np.ndarray,
    calibration: Calibration,
    cell_size: int = 32,
) -> np.ndarray:
    """전체 화면 캡처에서 셀별 작은 이미지 170개를 잘라 반환.

    image: HxWx3 RGB. 화면 좌표계 기준.
    Returns: (rows, cols, cell_size, cell_size, 3) uint8
    """
    cells = np.zeros(
        (calibration.rows, calibration.cols, cell_size, cell_size, 3),
        dtype=np.uint8,
    )
    half = int(min(calibration.cell_width, calibration.cell_height) // 2) - 2
    half = max(half, 4)
    for r in range(calibration.rows):
        for c in range(calibration.cols):
            cx, cy = calibration.cell_center(r, c)
            x1 = max(0, cx - half)
            y1 = max(0, cy - half)
            x2 = min(image.shape[1], cx + half)
            y2 = min(image.shape[0], cy + half)
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            cells[r, c] = cv2.resize(crop, (cell_size, cell_size))
    return cells


def _has_apple(cell_rgb: np.ndarray, red_threshold: float = 0.05) -> bool:
    """셀에 빨간 사과가 있는지. HSV의 hue로 빨간색 픽셀 비율 측정."""
    hsv = cv2.cvtColor(cell_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    red_mask = ((h < 15) | (h > 165)) & (s > 80) & (v > 80)
    ratio = float(red_mask.sum()) / red_mask.size
    return ratio > red_threshold


def _white_digit_mask(cell_rgb: np.ndarray) -> np.ndarray:
    """셀에서 흰색 숫자 픽셀만 마스킹 (HSV)."""
    hsv = cv2.cvtColor(cell_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    mask = (s < 80) & (v > 180)
    return mask.astype(np.uint8) * 255


def _normalize_digit_mask(mask: np.ndarray, size: int = 32) -> np.ndarray:
    """흰 픽셀 bbox로 crop → 정사각 패딩 → size×size로 resize.

    셀 크기/줌이 달라도 숫자 자체의 형태만 비교하도록 정규화.
    흰 픽셀이 너무 적으면 영(0) 마스크 반환.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) < 4:
        return np.zeros((size, size), dtype=np.float32)
    y1, y2 = int(ys.min()), int(ys.max())
    x1, x2 = int(xs.min()), int(xs.max())
    cropped = mask[y1 : y2 + 1, x1 : x2 + 1]
    h, w = cropped.shape
    side = max(h, w)
    pad_y = (side - h) // 2
    pad_x = (side - w) // 2
    square = np.zeros((side, side), dtype=np.uint8)
    square[pad_y : pad_y + h, pad_x : pad_x + w] = cropped
    resized = cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32)


def _cluster_cells_by_similarity(
    cell_masks: list[np.ndarray],
    n_clusters: int = 9,
) -> list[int]:
    """셀의 흰색 숫자 마스크들을 비교해 9개 그룹으로 클러스터링 (k-means)."""
    flat = np.stack([m.flatten().astype(np.float32) for m in cell_masks])
    norms = np.linalg.norm(flat, axis=1, keepdims=True) + 1e-6
    flat = flat / norms
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, labels, _ = cv2.kmeans(
        flat, n_clusters, None, criteria, 5, cv2.KMEANS_PP_CENTERS
    )
    return [int(x) for x in labels.flatten()]


def _label_clusters_by_white_count(
    cell_masks: list[np.ndarray],
    cluster_ids: list[int],
    n_clusters: int = 9,
) -> dict[int, int]:
    """각 클러스터의 평균 흰 픽셀 수로 1..9 라벨 매핑.

    가장 흰 픽셀이 적은 클러스터 → 1, 가장 많은 → 큰 숫자.
    1과 8은 잘 분리되지만 5/6 등은 헷갈릴 수 있음. 임시 추정.
    """
    sums_per_cluster: dict[int, list[int]] = {i: [] for i in range(n_clusters)}
    for mask, cid in zip(cell_masks, cluster_ids):
        sums_per_cluster[cid].append(int(mask.sum()))
    avg = [(cid, np.mean(s)) for cid, s in sums_per_cluster.items() if s]
    avg.sort(key=lambda x: x[1])
    return {cid: digit + 1 for digit, (cid, _) in enumerate(avg)}


def auto_recognize_first_board(
    image: np.ndarray, calibration: Calibration
) -> Optional[np.ndarray]:
    """첫 보드 자동 인식 시도. 검증 실패 시 None.

    검증:
      - 보드 합이 10의 배수 (사이트의 rejection sampling)
      - 보드 합이 600~1000 범위 (oshizi 조사: 평균 851 ± 35)
    """
    cells = extract_cells(image, calibration)
    rows, cols = calibration.rows, calibration.cols

    cell_masks: list[np.ndarray] = []
    apple_present: list[bool] = []
    for r in range(rows):
        for c in range(cols):
            cell = cells[r, c]
            has = _has_apple(cell)
            apple_present.append(has)
            if has:
                cell_masks.append(_white_digit_mask(cell))

    if len(cell_masks) < 9:
        return None

    cluster_ids = _cluster_cells_by_similarity(cell_masks, n_clusters=9)
    cid_to_digit = _label_clusters_by_white_count(cell_masks, cluster_ids)

    board = np.zeros((rows, cols), dtype=np.int8)
    apple_idx = 0
    for r in range(rows):
        for c in range(cols):
            if apple_present[r * cols + c]:
                board[r, c] = cid_to_digit[cluster_ids[apple_idx]]
                apple_idx += 1

    total = int(board.sum())
    if total % 10 != 0:
        return None
    if total < 600 or total > 1000:
        return None
    return board


def build_templates(
    image: np.ndarray, calibration: Calibration, labels: np.ndarray
) -> dict[int, np.ndarray]:
    """알려진 보드(labels)로 1~9 템플릿(평균 정규화 마스크) 생성.

    labels: (rows, cols) int, 정답 보드 (0 = 빈 셀).
    Returns: {digit: 32x32 float32 정규화 마스크}.
    각 셀에서 흰 픽셀 bbox crop → 정사각 패딩 → 32×32 resize → 평균.
    """
    cells = extract_cells(image, calibration)
    by_digit: dict[int, list[np.ndarray]] = {d: [] for d in range(1, 10)}
    for r in range(calibration.rows):
        for c in range(calibration.cols):
            d = int(labels[r, c])
            if d == 0:
                continue
            mask = _white_digit_mask(cells[r, c])
            norm = _normalize_digit_mask(mask)
            by_digit[d].append(norm)

    return {
        d: (
            np.stack(masks).mean(axis=0)
            if masks
            else np.zeros((32, 32), dtype=np.float32)
        )
        for d, masks in by_digit.items()
    }


def _match_digit(
    norm_mask: np.ndarray,
    template_arrs: np.ndarray,
    template_keys: list[int],
) -> tuple[int, float]:
    """정규화 마스크와 템플릿 비교. (best_digit, margin) 반환.

    margin = (second_best_dist - best_dist) / second_best_dist
    margin이 클수록 확신 ↑. 0에 가까우면 헷갈리는 셀.
    norm_mask가 영(0)이면 (0, 0.0) 반환.
    """
    if norm_mask.sum() < 1.0:
        return 0, 0.0
    a_norm = norm_mask / (np.linalg.norm(norm_mask) + 1e-6)
    t_norms = template_arrs / (
        np.linalg.norm(template_arrs.reshape(len(template_arrs), -1), axis=1).reshape(-1, 1, 1)
        + 1e-6
    )
    diffs = ((t_norms - a_norm) ** 2).sum(axis=(1, 2))
    order = np.argsort(diffs)
    best, second = int(order[0]), int(order[1])
    best_d = float(diffs[best])
    second_d = float(diffs[second])
    margin = (second_d - best_d) / (second_d + 1e-6)
    return template_keys[best], margin


def recognize_board(
    image: np.ndarray,
    calibration: Calibration,
    templates: dict[int, np.ndarray],
    log_low_confidence: bool = False,
) -> np.ndarray:
    """매 step 호출. 화면 → 보드 (10, 17) int8.

    빈 셀(사과 없음)은 0, 숫자 셀은 정규화 마스크로 cosine-SSD 최소인 1~9.
    """
    cells = extract_cells(image, calibration)
    board = np.zeros((calibration.rows, calibration.cols), dtype=np.int8)

    template_keys = sorted(templates.keys())
    template_arrs = np.stack([templates[d] for d in template_keys])

    for r in range(calibration.rows):
        for c in range(calibration.cols):
            cell = cells[r, c]
            if not _has_apple(cell):
                continue
            mask = _white_digit_mask(cell)
            norm = _normalize_digit_mask(mask)
            digit, margin = _match_digit(norm, template_arrs, template_keys)
            board[r, c] = digit
            if log_low_confidence and margin < 0.05 and digit > 0:
                print(f"  low-confidence r={r} c={c} → {digit} (margin={margin:.3f})")
    return board


def save_templates(
    templates: dict[int, np.ndarray],
    path: Path,
    trained_cell_size: float | None = None,
) -> None:
    """templates를 npz로 저장. trained_cell_size를 함께 기록하면

    이후 recognize_board에서 입력 이미지의 셀 크기와 비교해 자동 업스케일.
    """
    payload = {f"digit_{d}": t for d, t in templates.items()}
    if trained_cell_size is not None:
        payload["_trained_cell_size"] = np.float32(trained_cell_size)
    np.savez_compressed(path, **payload)


def load_templates(path: Path) -> dict[int, np.ndarray]:
    """digit_1..digit_9 마스크 dict. _trained_cell_size는 무시 (별도 API).

    이 함수는 기존 호환 유지 — 셀 크기 메타가 필요하면 load_templates_meta 사용.
    """
    with np.load(path) as npz:
        return {
            int(k.split("_")[1]): npz[k].copy()
            for k in npz.files
            if k.startswith("digit_")
        }


def load_templates_meta(path: Path) -> dict:
    """templates + trained_cell_size 함께 로드.

    Returns: {"templates": {...}, "trained_cell_size": float | None}.
    """
    with np.load(path) as npz:
        tpl = {
            int(k.split("_")[1]): npz[k].copy()
            for k in npz.files
            if k.startswith("digit_")
        }
        size = (
            float(npz["_trained_cell_size"])
            if "_trained_cell_size" in npz.files
            else None
        )
    return {"templates": tpl, "trained_cell_size": size}


def _prepare_cell_for_ocr(cell_rgb: np.ndarray, upscale: int = 6) -> np.ndarray:
    """OCR 입력으로 좋은 형태: 흰 배경 + 검정 숫자 + 큰 사이즈.

    1) 흰 픽셀 마스크 (숫자만)
    2) bounding box로 글자 영역만 crop (사과 빨강 영역 제거)
    3) 정사각 패딩 추가 (OCR이 비율 일정한 거 선호)
    4) 반전 (흰 배경, 검정 숫자) + 큰 사이즈로 확대
    """
    mask = _white_digit_mask(cell_rgb)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.full((upscale * 32, upscale * 32), 255, dtype=np.uint8)

    y1, y2 = int(ys.min()), int(ys.max())
    x1, x2 = int(xs.min()), int(xs.max())
    cropped = mask[y1 : y2 + 1, x1 : x2 + 1]

    # 정사각으로 패딩 (글자 비율 보존)
    h, w = cropped.shape
    side = max(h, w)
    pad_y = (side - h) // 2
    pad_x = (side - w) // 2
    square = np.zeros((side, side), dtype=np.uint8)
    square[pad_y : pad_y + h, pad_x : pad_x + w] = cropped

    # 여유 패딩 + 반전
    pad = side // 4
    padded = np.full((side + pad * 2, side + pad * 2), 0, dtype=np.uint8)
    padded[pad : pad + side, pad : pad + side] = square
    inv = 255 - padded

    big = cv2.resize(
        inv,
        (inv.shape[1] * upscale, inv.shape[0] * upscale),
        interpolation=cv2.INTER_CUBIC,
    )
    return big


def _extract_cell_at_native_size(
    image: np.ndarray, calibration: Calibration, r: int, c: int
) -> np.ndarray:
    """단일 셀을 원본 픽셀 크기 그대로 잘라 반환 (resize 없음). OCR용."""
    half = int(min(calibration.cell_width, calibration.cell_height) // 2) - 2
    half = max(half, 4)
    cx, cy = calibration.cell_center(r, c)
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(image.shape[1], cx + half)
    y2 = min(image.shape[0], cy + half)
    return image[y1:y2, x1:x2]


def ocr_recognize_first_board(
    image: np.ndarray,
    calibration: Calibration,
    log_fn=None,
) -> Optional[np.ndarray]:
    """Tesseract OCR로 첫 보드 인식 — 셀별로 숫자 한 글자씩.

    셀은 원본 픽셀 크기로 추출(resize 없음) → OCR 정확도 ↑.
    검증: 보드 합 % 10 == 0 (사이트는 rejection sampling으로 항상 10의 배수).
    """
    import pytesseract

    board = np.zeros((calibration.rows, calibration.cols), dtype=np.int8)
    config = "--psm 10 -c tessedit_char_whitelist=123456789"

    n_apples = 0
    failed = 0
    for r in range(calibration.rows):
        for c in range(calibration.cols):
            cell = _extract_cell_at_native_size(image, calibration, r, c)
            if cell.size == 0 or not _has_apple(cell):
                continue
            n_apples += 1
            prepared = _prepare_cell_for_ocr(cell)
            text = pytesseract.image_to_string(prepared, config=config).strip()
            if len(text) == 1 and text in "123456789":
                board[r, c] = int(text)
            else:
                failed += 1

    if log_fn:
        log_fn(
            f"OCR: 사과 {n_apples}개 중 실패 {failed}개, "
            f"보드 합 {int(board.sum())}"
        )

    if failed > 0:
        return None

    total = int(board.sum())
    if total % 10 != 0:
        return None
    if total < 600 or total > 1000:
        return None
    return board
