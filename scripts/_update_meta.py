"""README.md / docs/PROGRESS.md 의 자동 갱신 영역을 업데이트.

.claude/settings.json의 PostToolUse 훅이 호출. 직접도 호출 가능:
    python3 scripts/_update_meta.py

갱신 영역은 마커로 구분:
  <!-- AUTO:TREE -->     ... <!-- /AUTO:TREE -->
  <!-- AUTO:STATS -->    ... <!-- /AUTO:STATS -->

마커가 없는 파일은 그냥 통과 (안전).
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TARGETS = [
    PROJECT_ROOT / "README.md",
    PROJECT_ROOT / "docs" / "PROGRESS.md",
]

TREE_EXCLUDE = {
    "__pycache__",
    ".pytest_cache",
    ".git",
    "models",
    ".venv",
    "venv",
}


def _build_tree(
    root: Path, prefix: str = "", depth: int = 0, max_depth: int = 3
) -> list[str]:
    if depth > max_depth:
        return []
    entries = []
    for p in sorted(root.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
        if p.name.startswith(".") and p.name != ".claude":
            continue
        if p.name in TREE_EXCLUDE:
            continue
        entries.append(p)

    lines: list[str] = []
    for i, p in enumerate(entries):
        is_last = i == len(entries) - 1
        connector = "└── " if is_last else "├── "
        suffix = "/" if p.is_dir() else ""
        lines.append(f"{prefix}{connector}{p.name}{suffix}")
        if p.is_dir() and p.name not in TREE_EXCLUDE and p.name != ".claude":
            extension = "    " if is_last else "│   "
            lines.extend(_build_tree(p, prefix + extension, depth + 1, max_depth))
    return lines


def _count_tests() -> int:
    try:
        result = subprocess.run(
            ["python3", "-m", "pytest", "--collect-only", "-q"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        for line in reversed(result.stdout.splitlines()):
            m = re.match(r"^\s*(\d+)\s+tests?\s+collected", line)
            if m:
                return int(m.group(1))
    except Exception:
        pass
    count = 0
    tests_dir = PROJECT_ROOT / "tests"
    if tests_dir.exists():
        for f in tests_dir.glob("test_*.py"):
            count += len(re.findall(r"^\s*def test_", f.read_text(), re.MULTILINE))
    return count


def _list_models() -> list[str]:
    models_dir = PROJECT_ROOT / "models"
    if not models_dir.exists():
        return []
    return sorted(p.name for p in models_dir.iterdir() if p.is_file())


def _build_stats() -> str:
    n_tests = _count_tests()
    models = _list_models()
    lines = [
        f"- 테스트: **{n_tests}개** (`pytest`)",
        f"- 학습 산출물: **{len(models)}개** in `models/`",
    ]
    if models:
        lines.append("")
        lines.append("  | 파일 |")
        lines.append("  |---|")
        for m in models:
            lines.append(f"  | `{m}` |")
    return "\n".join(lines)


def _replace_block(text: str, marker: str, new_content: str) -> tuple[str, bool]:
    pattern = re.compile(
        rf"(<!-- AUTO:{marker} -->)(.*?)(<!-- /AUTO:{marker} -->)",
        re.DOTALL,
    )
    replacement = rf"\1\n{new_content}\n\3"
    new_text, n = pattern.subn(replacement, text)
    return new_text, n > 0


def update_file(path: Path) -> bool:
    if not path.exists():
        return False
    original = path.read_text()
    text = original

    tree_content = (
        "```\napple/\n"
        + "\n".join(_build_tree(PROJECT_ROOT))
        + "\n```"
    )
    text, _ = _replace_block(text, "TREE", tree_content)

    stats_content = _build_stats()
    text, _ = _replace_block(text, "STATS", stats_content)

    if text != original:
        path.write_text(text)
        return True
    return False


def main() -> int:
    changed: list[str] = []
    for target in TARGETS:
        if update_file(target):
            changed.append(str(target.relative_to(PROJECT_ROOT)))
    if changed:
        print(f"[meta] updated: {', '.join(changed)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
