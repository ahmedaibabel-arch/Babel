#!/usr/bin/env python3
"""
Scan the repository for tokens that violate the clean-room boundary.

The scan intentionally skips generated data, logs, outputs, and the git metadata
directory. It is meant to catch private paths, source-like names, and other
restricted references before a release.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SKIP_DIRS = {".git", ".venv", "data", "logs", "outputs"}
TEXT_SUFFIXES = {
    "", ".md", ".txt", ".py", ".sh", ".yaml", ".yml", ".json",
    ".jsonl", ".toml", ".ini", ".cfg", ".mk", ".rst",
}

FORBIDDEN_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("private_path", re.compile(r"/Users/" + "ai/")),
    ("private_project", re.compile(r"solid-" + "outlook-498013-s6")),
    ("private_corpus", re.compile(r"Mendel" + "Linguistics" + "Builder")),
    ("private_binary", re.compile(r"gramma" + "lecte_fr\.zip")),
    ("private_probe", re.compile(r"tome3" + "_probe")),
    ("commercial_product", re.compile(r"\b" + "An" + "tidote\b", re.IGNORECASE)),
    ("vendor_name", re.compile(r"\b" + "Drui" + "de\b", re.IGNORECASE)),
]


def is_text_file(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES or path.name in {"Makefile", "LICENSE"}


def iter_files(root: Path):
    self_path = Path(__file__).resolve()
    for path in root.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.resolve() == self_path:
            continue
        if path.is_file() and is_text_file(path):
            yield path


def main() -> int:
    hits: list[str] = []
    for path in iter_files(ROOT):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for label, pattern in FORBIDDEN_PATTERNS:
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                hits.append(f"{path.relative_to(ROOT)}:{line}: {label}: {match.group(0)}")

    if hits:
        print("Forbidden tokens found:")
        for hit in hits:
            print(f"  {hit}")
        return 1

    print("Clean-room scan passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
