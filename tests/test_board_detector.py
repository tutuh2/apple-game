"""board_detector 단위 테스트 — 합성 이미지에서 격자 자동 검출."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from agent.board_detector import detect_grid
from env.fruit_box import BOARD_COLS, BOARD_ROWS


def _make_full_board_image(
    offset_x: int = 50,
    offset_y: int = 80,
    cell_px: int = 40,
    canvas_w: int = 1200,
    canvas_h: int = 700,
    apple_color: tuple[int, int, int] = (220, 70, 60),
    bg_color: tuple[int, int, int] = (230, 245, 220),
) -> np.ndarray:
    """170개 빨간 사과가 격자로 배치된 합성 이미지."""
    img = Image.new("RGB", (canvas_w, canvas_h), bg_color)
    draw = ImageDraw.Draw(img)
    radius = cell_px // 2 - 3
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            cx = offset_x + c * cell_px + cell_px // 2
            cy = offset_y + r * cell_px + cell_px // 2
            draw.ellipse(
                (cx - radius, cy - radius, cx + radius, cy + radius),
                fill=apple_color,
            )
    return np.array(img)


class TestDetectGrid:
    def test_clean_full_board(self):
        img = _make_full_board_image()
        result = detect_grid(img)
        assert result is not None
        assert result.n_apples_detected >= 160
        assert result.confidence > 0.9
        assert abs(result.calibration.top_left_x - 70) <= 3
        assert abs(result.calibration.top_left_y - 100) <= 3

    def test_different_offset(self):
        img = _make_full_board_image(offset_x=200, offset_y=150)
        result = detect_grid(img)
        assert result is not None
        assert abs(result.calibration.top_left_x - 220) <= 3
        assert abs(result.calibration.top_left_y - 170) <= 3

    def test_cell_width_height(self):
        img = _make_full_board_image(cell_px=40)
        result = detect_grid(img)
        assert result is not None
        assert abs(result.calibration.cell_width - 40) < 2
        assert abs(result.calibration.cell_height - 40) < 2

    def test_no_apples_returns_none(self):
        img = Image.new("RGB", (800, 600), (230, 245, 220))
        result = detect_grid(np.array(img))
        assert result is None

    def test_too_few_apples_returns_none(self):
        img = Image.new("RGB", (800, 600), (230, 245, 220))
        draw = ImageDraw.Draw(img)
        for i in range(5):
            draw.ellipse(
                (100 + i * 50, 100, 130 + i * 50, 130), fill=(220, 70, 60)
            )
        result = detect_grid(np.array(img))
        assert result is None

    def test_distractor_red_blob_ignored(self):
        img_arr = _make_full_board_image()
        img = Image.fromarray(img_arr)
        draw = ImageDraw.Draw(img)
        # 격자 옆에 큰 빨간 사각형 (광고 영역 가정)
        draw.rectangle((900, 50, 1150, 200), fill=(220, 70, 60))
        result = detect_grid(np.array(img))
        assert result is not None
        assert abs(result.calibration.top_left_x - 70) <= 3
