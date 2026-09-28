"""Regression coverage for the hermetic macOS checkout-installer gate."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS installer verification")
def test_macos_installer_fixture_gate() -> None:
    """The fixture must create the same CLAP-cache parent the installer marks."""

    completed = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "verify_macos_installer.py")],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert '"gate_pass": true' in completed.stdout.lower()
