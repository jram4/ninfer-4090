#!/usr/bin/env python3
"""Reject private/generated material and broken local Markdown links."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_PREFIXES = ("build/", "models/", "profiles/", "vcpkg_installed/")
FORBIDDEN_SUFFIXES = (".jsonl", ".pem", ".key")
FORBIDDEN_PATTERNS = {
    "private host name": re.compile(r"\b" + "fn" + "n" + r"\b", re.IGNORECASE),
    "private host username": re.compile("ara" + "mirezfamily", re.IGNORECASE),
    "private overlay-network address": re.compile(r"100\.119" + r"\.241\.20"),
    "absolute Linux home path": re.compile(r"/home/[A-Za-z0-9._-]+/"),
    "absolute macOS home path": re.compile(r"/Users/[A-Za-z0-9._-]+/"),
    "private key material": re.compile(r"BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY"),
    "GitHub token": re.compile(r"gh[opsu]_[A-Za-z0-9]{20,}"),
    "OpenAI-style secret": re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
}
MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def tracked_files() -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-files", "-co", "--exclude-standard", "-z"], cwd=ROOT
    )
    paths = [ROOT / item.decode() for item in output.split(b"\0") if item]
    return [path for path in paths if path.exists()]


def check_paths(files: list[Path]) -> list[str]:
    errors: list[str] = []
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        if relative.startswith(FORBIDDEN_PREFIXES):
            errors.append(f"forbidden generated path: {relative}")
        if relative.endswith(FORBIDDEN_SUFFIXES):
            errors.append(f"forbidden sensitive suffix: {relative}")
    return errors


def check_contents(files: list[Path]) -> list[str]:
    errors: list[str] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        relative = path.relative_to(ROOT).as_posix()
        for label, pattern in FORBIDDEN_PATTERNS.items():
            if pattern.search(text):
                errors.append(f"{label}: {relative}")
    return errors


def check_markdown_links(files: list[Path]) -> list[str]:
    errors: list[str] = []
    for path in files:
        if path.suffix.lower() != ".md":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for raw_target in MARKDOWN_LINK.findall(text):
            target = raw_target.strip().strip("<>")
            if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            if "(" in target or any(character.isspace() for character in target):
                continue
            local = target.split("#", 1)[0]
            if not local:
                continue
            resolved = (path.parent / local).resolve()
            try:
                resolved.relative_to(ROOT)
            except ValueError:
                errors.append(f"link escapes repository: {path.relative_to(ROOT)} -> {target}")
                continue
            if not resolved.exists():
                errors.append(f"broken local link: {path.relative_to(ROOT)} -> {target}")
    return errors


def main() -> int:
    files = tracked_files()
    errors = check_paths(files) + check_contents(files) + check_markdown_links(files)
    if errors:
        for error in sorted(set(errors)):
            print(f"ERROR: {error}")
        return 1
    print(f"OK: validated {len(files)} public files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
