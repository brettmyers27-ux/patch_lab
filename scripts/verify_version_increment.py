#!/usr/bin/env python3
"""Reject commits that reuse, skip, or automatically cross PatchLab versions."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from version_policy import (
    VERSION_FILE,
    format_version,
    is_valid_transition,
    parse_version_text,
)


#: Refs tried, in order, to find the last version PatchLab actually shipped
#: from -- the baseline a release-prep branch is allowed to sit above.
RELEASED_REFS = ("origin/main", "main")


def _git(*args: str) -> str:
    # Both real invocations (the pre-commit hook, and CI's checkout step) run
    # this from the repository's own working directory, so that -- not this
    # script's on-disk location -- is the git repo being validated. Using
    # the caller's cwd also lets tests point this at a disposable repo.
    return subprocess.check_output(
        ("git", *args),
        cwd=Path.cwd(),
        text=True,
        stderr=subprocess.STDOUT,
    )


def _git_file(revision: str) -> str:
    return _git("show", f"{revision}:{VERSION_FILE.as_posix()}")


def _released_base_version(refs: tuple[str, ...] = RELEASED_REFS) -> tuple[int, int, int] | None:
    """Best-effort version at the last ref that represents what shipped.

    Returns ``None`` when no such ref is reachable (a shallow clone, a
    sandbox with no remote, ...); callers then fall back to the strict
    every-commit-advances rule rather than silently skipping validation.
    """

    for ref in refs:
        try:
            return parse_version_text(_git_file(ref))
        except (subprocess.CalledProcessError, ValueError):
            continue
    return None


def _verify_transition(
    old: tuple[int, int, int],
    new: tuple[int, int, int],
    *,
    base: tuple[int, int, int] | None,
    label: str,
) -> None:
    if not is_valid_transition(old, new, base=base):
        if new == old:
            raise ValueError(
                f"{label}: still at {format_version(old)}, the last released "
                "version -- bump app/__version__.py before this commit to "
                "start preparing the next release"
            )
        raise ValueError(
            f"{label}: version must stay at {format_version(old)} (same "
            f"release) or advance exactly one step, not {format_version(new)}"
        )
    if new == old:
        print(f"PASS {label}: {format_version(old)} unchanged (same release)")
    else:
        print(f"PASS {label}: {format_version(old)} -> {format_version(new)}")


def _verify_staged() -> None:
    old = parse_version_text(_git_file("HEAD"))
    new = parse_version_text(_git("show", f":{VERSION_FILE.as_posix()}"))
    _verify_transition(old, new, base=_released_base_version(), label="staged commit")


def _verify_range(base: str, head: str) -> None:
    base_version = parse_version_text(_git_file(base))
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
        old = parse_version_text(_git_file(parents[1]))
        new = parse_version_text(_git_file(commit))
        _verify_transition(old, new, base=base_version, label=commit[:12])


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
            old = parse_version_text(_git_file("HEAD^"))
            new = parse_version_text(_git_file("HEAD"))
            _verify_transition(old, new, base=_released_base_version(), label="HEAD")
    except (ValueError, subprocess.CalledProcessError) as exc:
        print(f"FAIL version policy: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
