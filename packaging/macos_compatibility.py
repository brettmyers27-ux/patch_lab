#!/usr/bin/env python3
"""Audit a frozen macOS app and enforce PatchLab's native compatibility contract."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import re
import subprocess
import sys
from pathlib import Path

from platform_compatibility import (
    MACOS_ALLOWED_ARCHITECTURES,
    MACOS_MINIMUM,
    MACOS_REQUIRED_ARCHITECTURE,
)

_VERSION = re.compile(r"^(?:minos|version)\s+(\d+(?:\.\d+)*)$")
_SDK = re.compile(r"^sdk\s+(\d+(?:\.\d+)*)$")
_FORBIDDEN = (b"-march=native", b"-mcpu=native")


def _version(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split("."))


def _run(*command: str) -> str:
    return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL)


def _owner(path: Path) -> str:
    value = str(path)
    if "Python.framework" in value or "/python" in value.lower():
        return "Python runtime"
    if "PySide6" in value or "/Qt" in value:
        return "Qt / PySide6"
    if "/torch" in value:
        return "Torch"
    if "dawdreamer" in value.lower():
        return "DawDreamer"
    if "pedalboard" in value.lower():
        return "Pedalboard"
    if "ffmpeg" in value.lower():
        return "ffmpeg"
    if "/MacOS/PatchLab" in value:
        return "PatchLab launcher"
    return "other bundled dependency"


def _macho(path: Path) -> bool:
    try:
        # Avoid a subprocess for every asset in a multi-gigabyte frozen app.
        # These are Mach-O 32/64-endian and universal/fat magic values.
        return path.open("rb").read(4) in {
            b"\xfe\xed\xfa\xce", b"\xce\xfa\xed\xfe", b"\xfe\xed\xfa\xcf", b"\xcf\xfa\xed\xfe",
            b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca", b"\xca\xfe\xba\xbf", b"\xbf\xba\xfe\xca",
        }
    except OSError:
        return False


def _inspect(path: Path) -> dict[str, object]:
    output = _run("otool", "-l", str(path))
    minima: list[str] = []
    sdks: list[str] = []
    command = ""
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("cmd "):
            command = stripped.removeprefix("cmd ")
        elif command in {"LC_BUILD_VERSION", "LC_VERSION_MIN_MACOSX"}:
            if command == "LC_BUILD_VERSION" and stripped.startswith("minos "):
                if match := _VERSION.search(stripped):
                    minima.append(match.group(1))
            elif command == "LC_VERSION_MIN_MACOSX" and stripped.startswith("version "):
                if match := _VERSION.search(stripped):
                    minima.append(match.group(1))
            if match := _SDK.search(stripped):
                sdks.append(match.group(1))
    architectures = _run("lipo", "-archs", str(path)).split()
    raw = path.read_bytes()
    developer_roots = (
        str(Path.home()).encode(),
        str(Path(__file__).resolve().parents[1]).encode(),
    )
    return {
        "path": str(path),
        "owner": _owner(path),
        "architectures": architectures,
        "minimum_macos": max(minima, key=_version) if minima else None,
        "sdk": max(sdks, key=_version) if sdks else None,
        # Third-party wheels legitimately carry their vendor CI source paths.
        # Reject paths identifying this build machine or PatchLab checkout,
        # rather than treating upstream debug strings as user-data leakage.
        "developer_path": any(root in raw for root in developer_roots),
        "native_cpu_flag": any(flag in raw for flag in _FORBIDDEN),
    }


def audit(app: Path) -> dict[str, object]:
    app = app.resolve()
    paths = {
        path.resolve() for path in app.rglob("*") if path.is_file() and _macho(path)
    }
    # otool/lipo are independent per file. A bounded pool keeps this gate quick
    # enough for release builds without overwhelming the local machine.
    with ThreadPoolExecutor(max_workers=12) as executor:
        rows = list(executor.map(_inspect, paths))
    rows.sort(key=lambda row: _version(str(row["minimum_macos"] or "0")), reverse=True)
    failures: list[str] = []
    for row in rows:
        path = str(row["path"])
        architectures = set(row["architectures"])
        if MACOS_REQUIRED_ARCHITECTURE not in architectures:
            failures.append(f"missing arm64: {path}")
        extra = architectures - MACOS_ALLOWED_ARCHITECTURES
        if extra:
            failures.append(f"unexpected architecture {sorted(extra)}: {path}")
        minimum = row["minimum_macos"]
        if minimum and _version(str(minimum)) > _version(MACOS_MINIMUM):
            failures.append(f"minimum macOS {minimum} exceeds {MACOS_MINIMUM}: {path}")
        if row["developer_path"]:
            failures.append(f"developer absolute path embedded: {path}")
        if row["native_cpu_flag"]:
            failures.append(f"machine-native CPU flag embedded: {path}")
    return {
        "app": str(app),
        "declared_minimum_macos": MACOS_MINIMUM,
        "required_architecture": MACOS_REQUIRED_ARCHITECTURE,
        "macho_count": len(rows),
        "binaries": rows,
        "failures": failures,
        "passed": not failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("app", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = audit(args.app)
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
