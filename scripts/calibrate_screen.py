"""화면 좌표 보정 — 사이트 보드의 픽셀 위치를 측정해 저장.

사용자가 게임 페이지에서 Play를 눌러 10×17 사과 격자가 보이는 상태로 만든 뒤,
이 스크립트를 실행. 카운트다운 후 마우스 위치를 두 번 캡처:
  1. 좌상단 사과 (행=0, 열=0)의 중심
  2. 우하단 사과 (행=9, 열=16)의 중심

결과는 calibration.json으로 저장. 자동 플레이가 이 좌표를 기반으로 화면에서
보드 영역을 잘라내고, 직사각형 (r1,c1,r2,c2)를 픽셀 좌표로 환산한다.

사용법:
    python3 scripts/calibrate_screen.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mss  # noqa: E402
import pyautogui  # noqa: E402
from PIL import Image  # noqa: E402

from env.fruit_box import BOARD_COLS, BOARD_ROWS  # noqa: E402


DEFAULT_OUT = Path("calibration.json")
PREVIEW_PATH = Path("/tmp/calibration_preview.png")


def _countdown_capture(label: str, seconds: int = 5) -> tuple[int, int]:
    """카운트다운 후 마우스 위치 반환."""
    print(f"\n{label}")
    print(f"  마우스를 그 위치에 가만히 두세요. {seconds}초 후 캡처합니다.")
    for s in range(seconds, 0, -1):
        print(f"  {s}...", end="\r", flush=True)
        time.sleep(1)
    pos = pyautogui.position()
    x, y = int(pos.x), int(pos.y)
    print(f"  캡처됨: ({x}, {y})       ")
    return x, y


def save_preview(top_left: tuple[int, int], bottom_right: tuple[int, int]) -> None:
    """격자 영역을 캡처해 미리보기 PNG로 저장. 사용자가 영역이 맞는지 확인용."""
    pad = 20
    x1, y1 = top_left
    x2, y2 = bottom_right
    region = {
        "left": min(x1, x2) - pad,
        "top": min(y1, y2) - pad,
        "width": abs(x2 - x1) + 2 * pad,
        "height": abs(y2 - y1) + 2 * pad,
    }
    with mss.mss() as sct:
        shot = sct.grab(region)
        img = Image.frombytes("RGB", shot.size, shot.rgb)
        img.save(PREVIEW_PATH)
    print(f"\n  미리보기 → {PREVIEW_PATH}")


def main() -> int:
    p = argparse.ArgumentParser(description="화면 좌표 보정")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--countdown", type=int, default=5)
    args = p.parse_args()

    print("=== 화면 좌표 보정 ===")
    print("1) 브라우저에서 fruit_box 페이지를 띄우고 Play를 눌러 격자가 보이게 하세요.")
    print("2) 카운트다운 동안 마우스를 지시한 위치에 두세요.")
    print("3) 두 위치는 첫 사과 (0,0) 중심과 마지막 사과 (9,16) 중심입니다.")
    input("\n준비 됐으면 Enter: ")

    tl = _countdown_capture(
        "[STEP 1/2] 첫 사과(행 0, 열 0) 중심에 마우스를 두세요",
        seconds=args.countdown,
    )
    br = _countdown_capture(
        f"[STEP 2/2] 마지막 사과(행 {BOARD_ROWS - 1}, 열 {BOARD_COLS - 1}) 중심에 마우스를 두세요",
        seconds=args.countdown,
    )

    if tl[0] >= br[0] or tl[1] >= br[1]:
        print("\n[!] 좌표 순서가 이상합니다. 좌상단이 우하단보다 작아야 합니다.")
        print(f"    top_left={tl}, bottom_right={br}")
        return 1

    cell_w = (br[0] - tl[0]) / (BOARD_COLS - 1)
    cell_h = (br[1] - tl[1]) / (BOARD_ROWS - 1)

    data = {
        "top_left_x": tl[0],
        "top_left_y": tl[1],
        "bottom_right_x": br[0],
        "bottom_right_y": br[1],
        "cell_width": cell_w,
        "cell_height": cell_h,
        "rows": BOARD_ROWS,
        "cols": BOARD_COLS,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    args.out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"\n[OK] 저장됨 → {args.out}")
    print(f"     cell 크기: {cell_w:.1f} × {cell_h:.1f} px")

    save_preview(tl, br)
    print("\n미리보기 이미지를 확인해서 격자가 잘 잡혔는지 보세요.")
    print("어긋났으면 다시 실행하세요.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
