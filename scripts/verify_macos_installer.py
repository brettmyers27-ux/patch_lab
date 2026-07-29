#!/usr/bin/env python3
"""Hermetic end-to-end orchestration check for the unchanged macOS installer."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if completed.returncode:
        raise RuntimeError(
            f"{' '.join(command)} failed ({completed.returncode}):\n"
            f"{completed.stdout}\n{completed.stderr}"
        )


def _write(path: Path, value: str, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    if executable:
        path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _dependency_marker(requirements: bytes) -> str:
    requirements_hash = hashlib.sha256(requirements).hexdigest()
    shasum_line = f"{requirements_hash}  requirements.txt\n"
    value = (shasum_line + "macos-torch-torchaudio-v1\n").encode()
    return hashlib.sha256(value).hexdigest()


def main() -> int:
    if sys.platform != "darwin":
        print("MACOS_INSTALLER_GATE=SKIP (requires macOS)")
        return 0
    with tempfile.TemporaryDirectory(prefix="patchlab-macos-installer-") as directory:
        root = Path(directory)
        source = root / "fixture-source"
        install = root / "installed" / "soundmatch"
        applications = root / "Applications"
        passcode = root / "passcode.txt"
        passcode.write_text("test-passcode", encoding="utf-8")
        requirements = b""
        _write(source / "requirements.txt", requirements.decode())
        _write(source / "app" / "__version__.py", '__version__ = "test"\n')
        (source / "app" / "icons").mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            PROJECT_ROOT / "app" / "icons" / "PatchLab.icns",
            source / "app" / "icons" / "PatchLab.icns",
        )
        _write(
            source / "scripts" / "install_support.py",
            """#!/usr/bin/env python3
import pathlib, sys
root = pathlib.Path(__file__).resolve().parents[1]
command = sys.argv[1]
marker = root / ".fixture-auth"
if command == "auth-status":
    raise SystemExit(0 if marker.is_file() else 1)
if command == "auth":
    if sys.stdin.read() != "test-passcode":
        raise SystemExit(2)
    marker.write_text("ok")
elif command == "clap":
    (root / "data/models/huggingface").mkdir(parents=True, exist_ok=True)
print("FIXTURE_" + command.upper().replace("-", "_") + "=PASS")
""",
            executable=True,
        )
        _write(
            source / "scripts" / "cache_clap.py",
            '#!/usr/bin/env python3\nprint("FIXTURE_CLAP_CACHE=PASS")\n',
            executable=True,
        )
        venv_bin = source / ".venv" / "bin"
        venv_bin.mkdir(parents=True)
        (venv_bin / "python").symlink_to(Path(sys.executable).resolve())
        marker = source / ".venv" / (
            ".patchlab-dependencies-" + _dependency_marker(requirements)
        )
        marker.touch()

        _run(["git", "init"], cwd=source)
        _run(["git", "config", "user.email", "installer-gate@patchlab.invalid"], cwd=source)
        _run(["git", "config", "user.name", "PatchLab Installer Gate"], cwd=source)
        _run(["git", "add", "-f", "."], cwd=source)
        _run(["git", "commit", "-m", "fixture"], cwd=source)

        environment = dict(os.environ)
        environment.update(
            {
                "PATCHLAB_REPO_URL": str(source),
                "PATCHLAB_INSTALL_ROOT": str(install),
                "PATCHLAB_USER_APPLICATIONS": str(applications),
                "PATCHLAB_INSTALL_APPLICATIONS": "skip",
                "PATCHLAB_INSTALL_TEST_MODE": "1",
                "PATCHLAB_PASSCODE_FILE": str(passcode),
                "PATCHLAB_RELAY_URL": "https://fixture.invalid",
            }
        )
        installer = PROJECT_ROOT / "install.sh"
        _run(["bash", str(installer)], cwd=PROJECT_ROOT, env=environment)
        # A second complete pass proves update/resume/idempotency orchestration.
        _run(["bash", str(installer)], cwd=PROJECT_ROOT, env=environment)

        app = applications / "PatchLab.app"
        launcher = app / "Contents" / "MacOS" / "PatchLab"
        info = app / "Contents" / "Info.plist"
        launcher_text = launcher.read_text(encoding="utf-8")
        payload = {
            "app_exists": app.is_dir(),
            "launcher_executable": launcher.is_file() and os.access(launcher, os.X_OK),
            "info_plist_exists": info.is_file(),
            "distribution_mode": "PATCHLAB_DISTRIBUTION_MODE=1" in launcher_text,
            "relay_configured": "https://fixture.invalid" in launcher_text,
            "rerun_completed": True,
        }
        payload["gate_pass"] = all(payload.values())
        print("MACOS_INSTALLER_REPORT=" + json.dumps(payload, sort_keys=True))
        return 0 if payload["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
