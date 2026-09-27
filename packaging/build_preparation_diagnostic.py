#!/usr/bin/env python3
"""Build the small, unsigned, self-contained preparation diagnostic app."""
from __future__ import annotations

import os
import argparse
import plistlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "packaging" / "preparation_diagnostic.py"
CORE = ROOT / "core" / "preparation_diagnostic.py"
OUTPUT_NAME = "PatchLab Preparation Diagnostic.app"
from platform_compatibility import MACOS_MINIMUM

MINIMUM_MACOS = MACOS_MINIMUM
RUNTIME_TOOL = ROOT / "packaging" / "macos_build_runtime.py"
COMPATIBILITY_AUDIT = ROOT / "packaging" / "macos_compatibility.py"
RUNTIME_REQUIREMENTS = ROOT / "packaging" / "requirements-macos-diagnostic.txt"


def _build_python() -> str:
    cache = Path(
        os.environ.get(
            "PATCHLAB_MACOS_RUNTIME_CACHE",
            str(Path.home() / ".cache" / "PatchLab" / "macos-build-runtime"),
        )
    )
    return subprocess.check_output(
        [
            sys.executable,
            str(RUNTIME_TOOL),
            "--cache",
            str(cache / "diagnostic"),
            "--requirements",
            str(RUNTIME_REQUIREMENTS),
        ],
        text=True,
    ).strip()


def build(output_dir: Path, *, error_details: bool = False) -> Path:
    if sys.platform != "darwin":
        raise RuntimeError("build the macOS diagnostic on macOS")
    if __import__("platform").machine() != "arm64":
        raise RuntimeError("build the macOS diagnostic on Apple Silicon")
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    launcher = ROOT / "packaging" / (
        "preparation_error_diagnostic.py" if error_details else "preparation_diagnostic.py"
    )
    cores = [ROOT / "core" / "preparation_diagnostic.py"]
    if error_details:
        cores.append(ROOT / "core" / "preparation_error_diagnostic.py")
    name = "PatchLab Preparation Error Diagnostic" if error_details else "PatchLab Preparation Diagnostic"
    identifier = "com.patchlab.preparation-error-diagnostic" if error_details else "com.patchlab.preparation-diagnostic"
    target = output_dir / f"{name}.app"
    if target.exists():
        shutil.rmtree(target)
    with tempfile.TemporaryDirectory(prefix="patchlab-preparation-diagnostic-") as temporary:
        work = Path(temporary)
        staged = work / "source"
        (staged / "core").mkdir(parents=True)
        shutil.copy2(launcher, staged / "preparation_diagnostic.py")
        for core in cores:
            shutil.copy2(core, staged / "core" / core.name)
        environment = dict(os.environ)
        environment["MACOSX_DEPLOYMENT_TARGET"] = MINIMUM_MACOS
        environment["PYINSTALLER_CONFIG_DIR"] = str(work / "pyinstaller-cache")
        subprocess.run(
            [
                _build_python(),
                "-m",
                "PyInstaller",
                "--clean",
                "--noconfirm",
                "--windowed",
                "--name",
                name,
                "--osx-bundle-identifier",
                identifier,
                "--target-architecture",
                "arm64",
                "--paths",
                str(staged),
                "--distpath",
                str(work / "dist"),
                "--workpath",
                str(work / "work"),
                str(staged / "preparation_diagnostic.py"),
            ],
            check=True,
            cwd=staged,
            env=environment,
        )
        app = work / "dist" / f"{name}.app"
        info_path = app / "Contents" / "Info.plist"
        with info_path.open("rb") as handle:
            info = plistlib.load(handle)
        info["LSMinimumSystemVersion"] = MINIMUM_MACOS
        info["CFBundleDisplayName"] = name
        info["CFBundleShortVersionString"] = "1.0.0"
        info["CFBundleVersion"] = "1.0.0"
        with info_path.open("wb") as handle:
            plistlib.dump(info, handle)
        subprocess.run(
            [
                sys.executable,
                str(COMPATIBILITY_AUDIT),
                str(app),
                "--report",
                str(work / "macos-compatibility-report.json"),
            ],
            check=True,
        )
        shutil.copytree(app, target)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--error-details", action="store_true")
    parser.add_argument("destination", nargs="?", type=Path, default=ROOT / "release")
    args = parser.parse_args()
    print(build(args.destination, error_details=args.error_details))
