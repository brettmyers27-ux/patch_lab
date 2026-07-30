#!/usr/bin/env python3
"""Advance PatchLab to the next owner-approved 1.x version."""

from __future__ import annotations

import argparse
from pathlib import Path

from version_policy import (
    VERSION_FILE,
    format_version,
    next_version,
    parse_version_text,
    replace_version,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the next version without changing app/__version__.py.",
    )
    args = parser.parse_args()

    path = ROOT / VERSION_FILE
    original = path.read_text(encoding="utf-8")
    current = parse_version_text(original)
    updated = next_version(current)
    if not args.dry_run:
        path.write_text(replace_version(original, updated), encoding="utf-8")
    print(f"{format_version(current)} -> {format_version(updated)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
