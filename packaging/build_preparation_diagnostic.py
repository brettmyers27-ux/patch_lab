#!/usr/bin/env python3
"""Build the small, unsigned, self-contained preparation diagnostic app."""
from __future__ import annotations

import os
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
# PyInstaller's arm64 bootloader itself supports older macOS releases, but the
# Python 3.11 framework currently used by PatchLab is built with minos 26.0.
# Declaring anything lower would let Finder offer an app the embedded runtime
# cannot launch.  A later platform-compatibility project can lower this only by
# rebuilding the runtime stack against an older deployment target.
MINIMUM_MACOS = "26.0"


def build(output_dir: Path) -> Path:
    if sys.platform != "darwin":
        raise RuntimeError("build the macOS diagnostic on macOS")
    if __import__("platform").machine() != "arm64":
        raise RuntimeError("build the macOS diagnostic on Apple Silicon")
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / OUTPUT_NAME
    if target.exists():
        shutil.rmtree(target)
    with tempfile.TemporaryDirectory(prefix="patchlab-preparation-diagnostic-") as temporary:
        work = Path(temporary)
        staged = work / "source"
        (staged / "core").mkdir(parents=True)
        shutil.copy2(LAUNCHER, staged / "preparation_diagnostic.py")
        shutil.copy2(CORE, staged / "core" / "preparation_diagnostic.py")
        environment = dict(os.environ)
        environment["MACOSX_DEPLOYMENT_TARGET"] = MINIMUM_MACOS
        environment["PYINSTALLER_CONFIG_DIR"] = str(work / "pyinstaller-cache")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "PyInstaller",
                "--clean",
                "--noconfirm",
                "--windowed",
                "--name",
                "PatchLab Preparation Diagnostic",
                "--osx-bundle-identifier",
                "com.patchlab.preparation-diagnostic",
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
        app = work / "dist" / OUTPUT_NAME
        info_path = app / "Contents" / "Info.plist"
        with info_path.open("rb") as handle:
            info = plistlib.load(handle)
        info["LSMinimumSystemVersion"] = MINIMUM_MACOS
        info["CFBundleDisplayName"] = "PatchLab Preparation Diagnostic"
        info["CFBundleShortVersionString"] = "1.0.0"
        info["CFBundleVersion"] = "1.0.0"
        with info_path.open("wb") as handle:
            plistlib.dump(info, handle)
        shutil.copytree(app, target)
    return target


if __name__ == "__main__":
    destination = Path(sys.argv[1]) if len(sys.argv) == 2 else ROOT / "release"
    print(build(destination))
