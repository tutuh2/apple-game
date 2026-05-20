"""콘솔에서 フルーツボックス 시뮬레이터를 직접 플레이.

사용법:
    python3 scripts/play_console.py [--seed N]

명령:
    r1 c1 r2 c2  — 직사각형 영역 선택 (0-indexed, 양쪽 포함)
    r            — 새 게임 시작
    q            — 종료
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 프로젝트 루트를 import path에 추가 (스크립트 단독 실행 지원)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env import FruitBox, Action  # noqa: E402


def parse_input(line: str) -> Action | str | None:
    """입력 한 줄 파싱. Action 또는 명령 문자열 또는 None(잘못된 입력)."""
    line = line.strip().lower()
    if line in {"q", "quit", "exit"}:
        return "quit"
    if line in {"r", "reset"}:
        return "reset"
    parts = line.split()
    if len(parts) != 4:
        return None
    try:
        r1, c1, r2, c2 = map(int, parts)
        return Action(r1, c1, r2, c2)
    except (ValueError, TypeError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="フルーツボックス 콘솔 플레이")
    parser.add_argument("--seed", type=int, default=None, help="보드 랜덤 시드")
    args = parser.parse_args()

    env = FruitBox(seed=args.seed)
    env.reset()

    print("=" * 60)
    print(" フルーツボックス — 콘솔 플레이")
    print(" 입력: r1 c1 r2 c2  (예: 0 1 0 2)")
    print(" 명령: r=리셋, q=종료")
    print("=" * 60)
    print(env.render())

    while True:
        if env.is_done():
            print("\n게임 종료 — 합=10인 영역이 더 이상 없습니다.")
            print(f"최종 점수: {env.score}")
            break

        try:
            line = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            print("\n종료.")
            return 0

        parsed = parse_input(line)
        if parsed is None:
            print("입력 형식: r1 c1 r2 c2  (또는 r=리셋, q=종료)")
            continue
        if parsed == "quit":
            print("종료.")
            return 0
        if parsed == "reset":
            env.reset()
            print("새 게임 시작.")
            print(env.render())
            continue

        action: Action = parsed
        try:
            reward = env.step(action)
        except ValueError as e:
            print(f"잘못된 좌표: {e}")
            continue

        if reward > 0:
            print(f"✓ 합=10! 사과 {reward}개 제거. (점수 +{reward} → {env.score})")
        else:
            total = env.rectangle_sum(action.r1, action.c1, action.r2, action.c2)
            print(f"✗ 합={total} (10이 아님). 점수 변동 없음.")
        print(env.render())

    return 0


if __name__ == "__main__":
    sys.exit(main())
