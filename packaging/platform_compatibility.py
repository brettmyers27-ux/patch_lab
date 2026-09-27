"""The single compatibility contract used by PatchLab packaging and release gates."""

from __future__ import annotations

# The portable Python runtime itself supports macOS 11.  The complete pinned
# product stack does not: the arm64 slices in the current PySide6 6.11.1
# bundle require macOS 15.0. Torch and SciPy require macOS 14.0. Do not lower
# this declaration without rebuilding and testing every native dependency on
# the proposed older OS.
MACOS_DESIRED_FLOOR = "11.0"
MACOS_MINIMUM = "15.0"
MACOS_REQUIRED_ARCHITECTURE = "arm64"
MACOS_ALLOWED_ARCHITECTURES = frozenset({"arm64", "x86_64"})

# This remains an intended target until a native Windows frozen build is run.
WINDOWS_INTENDED_MINIMUM = "Windows 10 x64 (pending native verification)"
WINDOWS_REQUIRED_ARCHITECTURE = "x86_64"

PYTHON_RUNTIME_VERSION = "3.11.16"
PYTHON_RUNTIME_ARCHIVE = "cpython-3.11.16+20260924-aarch64-apple-darwin-install_only.tar.gz"
PYTHON_RUNTIME_URL = (
    "https://github.com/astral-sh/python-build-standalone/releases/download/20260924/"
    "cpython-3.11.16%2B20260924-aarch64-apple-darwin-install_only.tar.gz"
)
PYTHON_RUNTIME_SHA256 = "d718e3c5c6f4b225ed25f88bf65e4c5d314e0dea0d716ea50bc9d038630c502b"
