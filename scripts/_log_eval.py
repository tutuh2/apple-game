"""평가 스크립트 출력에서 결과 라인을 추출해 docs/EVAL_LOG.md에 append.

.claude/settings.json의 PostToolUse(Bash) 훅이 호출. eval_*.py 의
stdout을 stdin으로 받아 timestamp + summary 를 누적 기록.

사용:
    python3 scripts/eval_mcts.py models/X.pt | python3 scripts/_log_eval.py
"""

from __future__ import annotations

import datetime as dt
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = PROJECT_ROOT / "docs" / "EVAL_LOG.md"


def _extract_summary(text: str) -> str | None:
    for line in text.splitlines():
        if re.search(r"평균\s+\d+\.\d+\s+중앙값", line):
            return line.strip()
    return None


def _extract_context(text: str) -> dict[str, str]:
    ctx: dict[str, str] = {}
    for line in text.splitlines():
        m = re.search(r"loading\s+(\S+)", line)
        if m:
            ctx["model"] = m.group(1)
        m = re.search(r"\((\d+)판", line)
        if m:
            ctx["episodes"] = m.group(1)
        m = re.search(r"N=(\d+)", line)
        if m:
            ctx["simulations"] = m.group(1)
        m = re.search(r"c_puct=([\d.]+)", line)
        if m:
            ctx["c_puct"] = m.group(1)
    return ctx


def main() -> int:
    text = sys.stdin.read()
    summary = _extract_summary(text)
    if not summary:
        return 0

    ctx = _extract_context(text)
    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not LOG_PATH.exists():
        LOG_PATH.write_text(
            "# Evaluation Log\n\n"
            "평가 스크립트 실행 결과 자동 누적. 직접 편집 자제(덧붙이는 건 OK).\n\n"
        )

    cmd = os.environ.get("CLAUDE_TOOL_BASH_COMMAND", "").strip()
    cmd_short = cmd[:80] + ("..." if len(cmd) > 80 else "")

    block_lines = [f"## {timestamp}"]
    if ctx.get("model"):
        block_lines.append(f"- model: `{ctx['model']}`")
    if ctx.get("episodes"):
        block_lines.append(f"- episodes: {ctx['episodes']}")
    if ctx.get("simulations"):
        block_lines.append(f"- MCTS simulations: {ctx['simulations']}")
    if ctx.get("c_puct"):
        block_lines.append(f"- c_puct: {ctx['c_puct']}")
    if cmd_short:
        block_lines.append(f"- cmd: `{cmd_short}`")
    block_lines.append(f"- **결과**: {summary}")
    block_lines.append("")

    with LOG_PATH.open("a") as f:
        f.write("\n".join(block_lines) + "\n")
    print(
        f"[eval-log] appended to {LOG_PATH.relative_to(PROJECT_ROOT)}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
