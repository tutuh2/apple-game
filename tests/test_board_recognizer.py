"""board_recognizer 단위 테스트 — 합성 이미지로 round-trip 검증."""

from __future__ import annotations

import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from agent.board_recognizer import (
    Calibration,
    build_templates,
    extract_cells,
    recognize_board,
    save_templates,
    load_templates,
)
from env.fruit_box import BOARD_COLS, BOARD_ROWS, FruitBox


CELL_PX = 48
PAD = 30


def _make_calibration() -> Calibration:
    return Calibration(
        top_left_x=PAD + CELL_PX // 2,
        top_left_y=PAD + CELL_PX // 2,
        bottom_right_x=PAD + CELL_PX // 2 + (BOARD_COLS - 1) * CELL_PX,
        bottom_right_y=PAD + CELL_PX // 2 + (BOARD_ROWS - 1) * CELL_PX,
        cell_width=float(CELL_PX),
        cell_height=float(CELL_PX),
    )


def _render_board(board: np.ndarray) -> np.ndarray:
    w = PAD * 2 + CELL_PX * BOARD_COLS
    h = PAD * 2 + CELL_PX * BOARD_ROWS
    img = Image.new("RGB", (w, h), (230, 245, 220))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
    except Exception:
        font = ImageFont.load_default()

    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            d = int(board[r, c])
            if d == 0:
                continue
            cx = PAD + CELL_PX // 2 + c * CELL_PX
            cy = PAD + CELL_PX // 2 + r * CELL_PX
            radius = CELL_PX // 2 - 3
            draw.ellipse(
                (cx - radius, cy - radius, cx + radius, cy + radius),
                fill=(220, 70, 60),
            )
            text = str(d)
            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text(
                (cx - tw // 2 - bbox[0], cy - th // 2 - bbox[1]),
                text,
                font=font,
                fill=(255, 255, 255),
            )

    return np.array(img)


class TestCalibrationLoad:
    def test_roundtrip(self, tmp_path):
        calib_path = tmp_path / "calib.json"
        data = {
            "top_left_x": 100,
            "top_left_y": 200,
            "bottom_right_x": 900,
            "bottom_right_y": 700,
            "cell_width": 50.0,
            "cell_height": 55.5,
            "rows": BOARD_ROWS,
            "cols": BOARD_COLS,
        }
        calib_path.write_text(json.dumps(data))
        c = Calibration.load(calib_path)
        assert c.top_left_x == 100
        assert c.cell_width == 50.0
        assert c.rows == BOARD_ROWS


class TestExtractCells:
    def test_shape(self):
        env = FruitBox(seed=0)
        env.reset()
        img = _render_board(env.board)
        c = _make_calibration()
        cells = extract_cells(img, c, cell_size=32)
        assert cells.shape == (BOARD_ROWS, BOARD_COLS, 32, 32, 3)
        assert cells.dtype == np.uint8


class TestRecognizeBoard:
    def test_round_trip_with_known_template(self):
        """동일 보드로 템플릿 만들고 다시 인식 → 100% 일치."""
        env = FruitBox(seed=7)
        env.reset()
        board = env.board.copy()
        img = _render_board(board)
        calib = _make_calibration()

        templates = build_templates(img, calib, board)
        recognized = recognize_board(img, calib, templates)
        assert np.array_equal(
            recognized.astype(np.int8), board.astype(np.int8)
        )

    def test_empty_cells_detected_as_zero(self):
        """일부 셀을 0(빈 셀)으로 만들면 인식도 0이 나와야 함."""
        env = FruitBox(seed=11)
        env.reset()
        board = env.board.copy()
        board[0:2, :] = 0
        img = _render_board(board)
        calib = _make_calibration()

        # 템플릿은 다른 꽉 찬 보드로
        env2 = FruitBox(seed=99)
        env2.reset()
        img2 = _render_board(env2.board)
        templates = build_templates(img2, calib, env2.board)

        recognized = recognize_board(img, calib, templates)
        assert (recognized[0:2, :] == 0).all()


class TestTemplatesIO:
    def test_save_load_roundtrip(self, tmp_path):
        env = FruitBox(seed=0)
        env.reset()
        img = _render_board(env.board)
        calib = _make_calibration()
        templates = build_templates(img, calib, env.board)

        path = tmp_path / "templates.npz"
        save_templates(templates, path)
        loaded = load_templates(path)

        for d in range(1, 10):
            assert d in loaded
            assert np.allclose(templates[d], loaded[d])
