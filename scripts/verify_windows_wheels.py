#!/usr/bin/env python3
"""Verify every direct pin has a CPython 3.11 Windows-compatible PyPI wheel."""

from __future__ import annotations

import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = PROJECT_ROOT / "requirements.txt"
PIN = re.compile(r"^([A-Za-z0-9_.-]+)==([^;\s]+)")


def pinned_requirements() -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    for raw in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        match = PIN.match(raw.strip())
        if match:
            result.append((match.group(1), match.group(2)))
    return result


def compatible(filename: str) -> bool:
    value = filename.casefold()
    if not value.endswith(".whl"):
        return False
    if (
        value.endswith("-py3-none-any.whl")
        or value.endswith("-py2.py3-none-any.whl")
        or value.endswith("-py3-none-win_amd64.whl")
        or value.endswith("-py2.py3-none-win_amd64.whl")
    ):
        return True
    if not value.endswith("-win_amd64.whl"):
        return False
    return (
        "-cp311-cp311-" in value
        or "-cp311-abi3-" in value
        or any(f"-cp{minor}-abi3-" in value for minor in range(38, 312))
    )


def main() -> int:
    rows: list[tuple[str, str, str]] = []
    failed = False
    for name, version in pinned_requirements():
        url = (
            "https://pypi.org/pypi/"
            f"{urllib.parse.quote(name)}/{urllib.parse.quote(version)}/json"
        )
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                payload = json.load(response)
            files = [
                str(item["filename"])
                for item in payload.get("urls", [])
                if compatible(str(item.get("filename", "")))
            ]
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            files = []
            detail = f"lookup failed: {exc}"
        else:
            detail = ", ".join(files) if files else "no CPython 3.11 win_amd64/any wheel"
        status = "PASS" if files else "FAIL"
        failed = failed or not files
        rows.append((status, f"{name}=={version}", detail))
    width = max(len(row[1]) for row in rows)
    print(f"{'STATUS':<6}  {'PIN':<{width}}  COMPATIBLE PYPI FILE")
    print(f"{'-' * 6}  {'-' * width}  {'-' * 60}")
    for status, pin, detail in rows:
        print(f"{status:<6}  {pin:<{width}}  {detail}")
    print(f"\nWINDOWS_WHEEL_GATE={'FAIL' if failed else 'PASS'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
