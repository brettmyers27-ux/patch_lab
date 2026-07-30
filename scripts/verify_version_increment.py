#!/usr/bin/env python3
"""Reject commits that reuse, skip, or automatically cross PatchLab versions."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from version_policy import (
    VERSION_FILE,
    format_version,
    next_version,
    parse_version_text,
)


ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str) -> str:
    return subprocess.check_output(
        ("git", *args),
        cwd=ROOT,
        text=True,
        stderr=subprocess.STDOUT,
    )


def _git_file(revision: str) -> str:
    return _git("show", f"{revision}:{VERSION_FILE.as_posix()}")


def _verify_pair(old_text: str, new_text: str, label: str) -> None:
    old = parse_version_text(old_text)
    new = parse_version_text(new_text)
    expected = next_version(old)
    if new != expected:
        raise ValueError(
            f"{label}: version must advance exactly "
            f"{format_version(old)} -> {format_version(expected)}, "
            f"not {format_version(new)}"
        )
    print(f"PASS {label}: {format_version(old)} -> {format_version(new)}")


def _verify_staged() -> None:
    old_text = _git_file("HEAD")
    new_text = _git("show", f":{VERSION_FILE.as_posix()}")
    _verify_pair(old_text, new_text, "staged commit")


def _verify_range(base: str, head: str) -> None:
    commits = [
        line
        for line in _git(
            "rev-list",
            "--reverse",
            "--no-merges",
            f"{base}..{head}",
        ).splitlines()
        if line
    ]
    if not commits:
        print("PASS version sequence: no non-merge commits to inspect")
        return
    for commit in commits:
        parents = _git("rev-list", "--parents", "-n", "1", commit).split()
        if len(parents) < 2:
            continue
        _verify_pair(
            _git_file(parents[1]),
            _git_file(commit),
            commit[:12],
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staged", action="store_true")
    parser.add_argument("--range", nargs=2, metavar=("BASE", "HEAD"))
    args = parser.parse_args()

    try:
        if args.staged:
            _verify_staged()
        elif args.range:
            _verify_range(*args.range)
        else:
            _verify_pair(_git_file("HEAD^"), _git_file("HEAD"), "HEAD")
    except (ValueError, subprocess.CalledProcessError) as exc:
        print(f"FAIL version policy: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
