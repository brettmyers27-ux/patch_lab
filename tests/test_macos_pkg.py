from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS package verification")
def test_macos_pkg_fixture_gate() -> None:
    """The PKG targets /Applications and upgrades without touching user data."""

    completed = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "verify_macos_pkg.py")],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert '"gate_pass": true' in completed.stdout.lower()
