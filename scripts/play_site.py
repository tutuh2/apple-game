"""실제 ゲーム菜園 사이트 자동 플레이.

전제:
  - 사용자가 사이트 띄워서 Play 누른 상태 (10×17 격자 보이는 상태)
  - calibration.json이 그 화면에 대해 만들어져 있음 (scripts/calibrate_screen.py)

흐름:
  1. calibration 로드
  2. templates.npz 있으면 로드, 없으면 첫 화면 캡처해서 자동 학습 시도
  3. 매 step: 캡처 → recognize → greedy_smallest → 드래그
  4. 게임 끝(valid 액션 없음/타이머) 또는 보드 변화 없음 N회 → 종료
  5. ESC 또는 마우스를 화면 좌상단 코너로 → pyautogui FAILSAFE로 abort

사용법:
    python3 scripts/calibrate_screen.py            # 좌표 보정 한 번
    python3 scripts/play_site.py                   # 첫 실행 — 자동 학습
    python3 scripts/play_site.py --delay 0.3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import mss  # noqa: E402
import pyautogui  # noqa: E402

from agent.board_recognizer import (  # noqa: E402
    Calibration,
    auto_recognize_first_board,
    build_templates,
    load_templates,
    recognize_board,
    save_templates,
)
from agent.heuristics import (  # noqa: E402
    find_valid_actions,
    greedy_smallest_policy,
)
from env.fruit_box import Action, FruitBox  # noqa: E402


pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.0


def _capture_screen() -> np.ndarray:
    """전체 화면 캡처 → RGB ndarray."""
    with mss.mss() as sct:
        monitor = sct.monitors[0]
        shot = sct.grab(monitor)
        img = np.array(shot)[:, :, :3]  # BGRA → BGR
        return img[:, :, ::-1].copy()  # BGR → RGB


def _drag(
    start_xy: tuple[int, int],
    end_xy: tuple[int, int],
    duration_sec: float = 0.15,
) -> None:
    pyautogui.moveTo(start_xy[0], start_xy[1], duration=0.05)
    pyautogui.mouseDown()
    pyautogui.moveTo(end_xy[0], end_xy[1], duration=duration_sec)
    pyautogui.mouseUp()


def _action_to_pixels(
    action: Action, calibration: Calibration, inset: int = 4
) -> tuple[tuple[int, int], tuple[int, int]]:
    """직사각형 (r1,c1,r2,c2) → 드래그 시작/끝 픽셀.

    각 셀 중심에서 inset px 안쪽으로 — 사과 클릭 판정 안전.
    """
    x1, y1 = calibration.cell_center(action.r1, action.c1)
    x2, y2 = calibration.cell_center(action.r2, action.c2)
    x1 -= inset
    y1 -= inset
    x2 += inset
    y2 += inset
    return (x1, y1), (x2, y2)


def _initialize_templates(
    calibration: Calibration,
    templates_path: Path,
    force_relearn: bool,
) -> dict[int, np.ndarray]:
    """templates.npz 로드 또는 첫 화면 자동 학습."""
    if templates_path.exists() and not force_relearn:
        print(f"[site] templates 로드: {templates_path}")
        return load_templates(templates_path)

    print("[site] 첫 화면 캡처해서 자동 학습 시도 — 3초 후 시작")
    for s in range(3, 0, -1):
        print(f"  {s}...", end="\r", flush=True)
        time.sleep(1)
    print("  캡처!         ")

    img = _capture_screen()
    board = auto_recognize_first_board(img, calibration)
    if board is None:
        print(
            "[!] 자동 인식 실패. 가능한 원인:\n"
            "    - calibration 좌표가 틀림 (scripts/calibrate_screen.py 다시 실행)\n"
            "    - 게임이 Play 후 보드 상태가 아님\n"
            "    - 화면 캡처 권한 부족"
        )
        sys.exit(1)

    total = int(board.sum())
    n_apples = int((board > 0).sum())
    print(
        f"[site] 자동 인식 OK — 사과 {n_apples}개, 합 {total} (10의 배수 ✓)"
    )

    templates = build_templates(img, calibration, board)
    templates_path.parent.mkdir(parents=True, exist_ok=True)
    save_templates(templates, templates_path)
    print(f"[site] templates 저장: {templates_path}")
    return templates


def play(
    calibration: Calibration,
    templates: dict[int, np.ndarray],
    delay_sec: float,
    drag_duration: float,
    max_steps: int,
    stale_limit: int,
) -> int:
    """메인 루프. 반환: 시뮬 추정 점수 (실제 사이트 점수는 OCR 안 함)."""
    game = FruitBox()
    game.board = np.zeros_like(game.board)
    game.score = 0
    game._initialized = True
    rng = np.random.default_rng(0)

    score = 0
    last_apple_count = -1
    stale = 0

    for step in range(max_steps):
        img = _capture_screen()
        board = recognize_board(img, calibration, templates)
        game.board = board

        apples_now = int((board > 0).sum())
        if last_apple_count == apples_now:
            stale += 1
        else:
            stale = 0
            last_apple_count = apples_now

        if stale >= stale_limit:
            print(f"[site] {stale_limit}회 연속 보드 변화 없음 → 종료")
            break

        actions = find_valid_actions(game)
        if not actions:
            print("[site] valid 액션 없음 → 종료")
            break

        action = greedy_smallest_policy(game, rng)
        assert action is not None

        reward = game.step(action)
        score += int(reward)

        start, end = _action_to_pixels(action, calibration)
        print(
            f"[step {step+1:3d}] r=({action.r1},{action.c1})-"
            f"({action.r2},{action.c2}) +{reward} → "
            f"score={score}  apples={apples_now}"
        )
        _drag(start, end, duration_sec=drag_duration)
        time.sleep(delay_sec)

    return score


def main() -> int:
    p = argparse.ArgumentParser(description="실제 사이트 자동 플레이")
    p.add_argument("--calibration", type=Path, default=Path("calibration.json"))
    p.add_argument(
        "--templates",
        type=Path,
        default=Path("models/site_templates.npz"),
    )
    p.add_argument(
        "--relearn-templates",
        action="store_true",
        help="기존 templates 무시하고 첫 화면에서 다시 학습",
    )
    p.add_argument("--delay", type=float, default=0.4, help="한 수 사이 간격(초)")
    p.add_argument(
        "--drag-duration",
        type=float,
        default=0.15,
        help="드래그 시간(초) — 너무 짧으면 사이트가 못 잡을 수 있음",
    )
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument(
        "--stale-limit",
        type=int,
        default=3,
        help="연속 N회 보드 변화 없으면 종료",
    )
    p.add_argument(
        "--countdown",
        type=int,
        default=5,
        help="시작 카운트다운 초 — 사용자가 사이트 창을 활성화할 시간",
    )
    args = p.parse_args()

    if not args.calibration.exists():
        print(
            f"[!] calibration 없음: {args.calibration}\n"
            f"    먼저 실행: python3 scripts/calibrate_screen.py",
            file=sys.stderr,
        )
        return 1

    calibration = Calibration.load(args.calibration)
    templates = _initialize_templates(
        calibration, args.templates, args.relearn_templates
    )

    print(
        f"\n[site] 자동 플레이 시작 — {args.countdown}초 후. "
        f"중단하려면 마우스를 화면 좌상단 코너로 보내세요."
    )
    for s in range(args.countdown, 0, -1):
        print(f"  {s}...", end="\r", flush=True)
        time.sleep(1)
    print("  GO!          ")

    try:
        final = play(
            calibration=calibration,
            templates=templates,
            delay_sec=args.delay,
            drag_duration=args.drag_duration,
            max_steps=args.max_steps,
            stale_limit=args.stale_limit,
        )
    except pyautogui.FailSafeException:
        print("\n[site] FAILSAFE — 사용자가 마우스를 코너로 보내 중단")
        return 130

    print(f"\n[site] 종료. 시뮬 추정 점수: {final}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
