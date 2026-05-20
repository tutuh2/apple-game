"""사이트 점수 표시 자동 검출 + OCR.

전략:
  1. 격자 calibration 기준 우측 상단 영역 검색
  2. 짙은 초록 픽셀(점수 폰트 색)의 bounding box로 점수 영역 확정
  3. 매 캡처마다 그 영역을 잘라 pytesseract로 점수 읽기
  4. 점수는 단조 증가만 가능 — 감소하면 OCR 오류로 간주
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from agent.board_recognizer import Calibration


@dataclass
class ScoreRegion:
    """점수 표시 화면 영역 (캡처 좌표계, recognize_scale 적용 후 기준)."""

    x1: int
    y1: int
    x2: int
    y2: int

    def crop(self, image: np.ndarray) -> np.ndarray:
        return image[self.y1 : self.y2, self.x1 : self.x2]


def find_score_region(
    image_rgb: np.ndarray, calibration: Calibration, margin: int = 8
) -> Optional[ScoreRegion]:
    """캡처 이미지에서 점수 표시 위치 자동 검출.

    격자 우측 상단의 짙은 초록 픽셀을 connectedComponents로 분리해
    "작은 글자" 후보만 골라낸다.
    - 게이지바: h/w > 4 (매우 세로 긴) → 제외
    - 점수 배경 박스: 너무 큰 영역 → 제외 (영역의 30% 이상이면)
    - 점수 글자: 보통 cell_height의 0.5~1.5배 높이, h/w 1~3

    DETECT 시점에 한 번 호출하면 됨. 이후 매 step 같은 영역을 재사용
    (점수가 자릿수 늘어나도 expand_right margin이 흡수).
    """
    if image_rgb.size == 0:
        return None

    cw, ch = calibration.cell_width, calibration.cell_height
    sx1 = int(calibration.bottom_right_x + cw / 2)
    sy1 = int(max(0, calibration.top_left_y - ch))
    sx2 = int(min(image_rgb.shape[1], sx1 + 5 * cw))
    sy2 = int(calibration.top_left_y + ch * 2)
    if sx2 <= sx1 or sy2 <= sy1:
        return None

    roi = image_rgb[sy1:sy2, sx1:sx2]
    roi_area = roi.shape[0] * roi.shape[1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    green_mask = (h > 35) & (h < 90) & (s > 80) & (v > 60) & (v < 200)
    if int(green_mask.sum()) < 30:
        return None

    n_labels, _labels, stats, _cent = cv2.connectedComponentsWithStats(
        green_mask.astype(np.uint8) * 255, connectivity=8
    )
    # 글자 후보 = 너무 크지도 작지도 않고, 게이지바도 배경 박스도 아닌 것
    digits: list[tuple[int, int, int, int]] = []
    min_h = max(int(ch * 0.3), 8)
    max_h = max(int(ch * 1.5), 50)
    for i in range(1, n_labels):
        x, y, w, hh, area = stats[i]
        if hh < min_h or hh > max_h:
            continue
        if area < 30:
            continue
        if hh / max(w, 1) > 4.0:  # 게이지바
            continue
        if area > roi_area * 0.3:  # 배경 박스
            continue
        digits.append((int(x), int(y), int(w), int(hh)))

    if not digits:
        return None

    # 같은 가로 줄에 있는 글자들만 묶기
    digits.sort(key=lambda c: c[0])
    ref_y = digits[0][1]
    same_row = [c for c in digits if abs(c[1] - ref_y) < max(ch * 0.5, 10)]
    if not same_row:
        same_row = digits

    xs1 = min(c[0] for c in same_row)
    ys1 = min(c[1] for c in same_row)
    xs2 = max(c[0] + c[2] for c in same_row)
    ys2 = max(c[1] + c[3] for c in same_row)

    expand_right = max(margin, int(cw * 2))
    gx1 = max(0, sx1 + xs1 - margin)
    gy1 = max(0, sy1 + ys1 - margin)
    gx2 = min(image_rgb.shape[1], sx1 + xs2 + expand_right)
    gy2 = min(image_rgb.shape[0], sy1 + ys2 + margin)

    return ScoreRegion(x1=gx1, y1=gy1, x2=gx2, y2=gy2)


def _binarize_score_image(roi_rgb: np.ndarray) -> np.ndarray:
    """점수 영역에서 OCR 입력용 이진 이미지 생성.

    초록 픽셀 = 글자(검정), 나머지 = 흰 배경. 게이지바(거의 모든 행이
    초록인 세로 컬럼)는 마스크에서 제거 후 큰 사이즈로 upscale.
    """
    hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    green_mask = (h > 35) & (h < 90) & (s > 80) & (v > 60) & (v < 200)
    # 게이지바 제거: 70% 이상의 행이 초록인 세로 컬럼은 게이지바
    if green_mask.shape[0] > 0:
        col_density = green_mask.sum(axis=0) / green_mask.shape[0]
        bar_cols = col_density > 0.7
        green_mask[:, bar_cols] = False
    img = np.where(green_mask, 0, 255).astype(np.uint8)
    big = cv2.resize(
        img, (img.shape[1] * 6, img.shape[0] * 6), interpolation=cv2.INTER_CUBIC
    )
    big = cv2.medianBlur(big, 3)
    _, bw = cv2.threshold(big, 127, 255, cv2.THRESH_BINARY)
    return bw


def read_score(roi_rgb: np.ndarray) -> Optional[int]:
    """ROI 이미지에서 점수 정수 읽기. 실패 시 None.

    여러 psm 모드 시도 후 첫 성공값 반환. 한 자리(0~9)일 때 psm 7/8/10이
    안 되고, 자릿수 많을 때는 6/7이 더 잘 됨.
    """
    try:
        import pytesseract
    except ImportError:
        return None

    if roi_rgb.size == 0:
        return None
    prepared = _binarize_score_image(roi_rgb)
    for psm in (7, 6, 8, 13, 10):
        config = f"--psm {psm} -c tessedit_char_whitelist=0123456789"
        text = pytesseract.image_to_string(prepared, config=config).strip()
        digits = "".join(c for c in text if c.isdigit())
        if digits:
            try:
                return int(digits)
            except ValueError:
                continue
    return None
