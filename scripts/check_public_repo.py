from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_TRACKED_BYTES = 5 * 1024 * 1024
TEXT_SUFFIXES = {".cfg", ".csv", ".json", ".jsonl", ".md", ".ps1", ".py", ".toml", ".txt", ".yaml", ".yml"}
SKIP_DIRS = {".git", ".venv", "outputs", "__pycache__"}
PATTERNS = {
    "private API key": re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{16,}"),
    "developer machine path": re.compile(r"C:\\Users\\ych13", re.IGNORECASE),
}


def iter_public_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in SKIP_DIRS for part in path.parts):
            continue
        files.append(path)
    return files


def main() -> int:
    findings: list[str] = []
    for path in iter_public_files():
        relative = path.relative_to(ROOT)
        if path.stat().st_size > MAX_TRACKED_BYTES:
            findings.append(f"large file: {relative} ({path.stat().st_size} bytes)")
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name != ".env.example":
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for label, pattern in PATTERNS.items():
            if pattern.search(text):
                findings.append(f"{label}: {relative}")

    if findings:
        print("Public-repository checks failed:")
        for finding in findings:
            print(f"- {finding}")
        return 1

    print("Public-repository checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
