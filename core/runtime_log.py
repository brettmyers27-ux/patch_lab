"""Small rotating runtime log retained locally for support diagnostics."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from core.platform_env import ENV


MAX_LOG_BYTES = 2 * 1024 * 1024
LOG_FILENAME = "patchlab-runtime.log"


def runtime_log_path() -> Path:
    root = ENV.app_data_dir / "diagnostics"
    root.mkdir(parents=True, exist_ok=True)
    return root / LOG_FILENAME


def append_runtime_log(message: str) -> None:
    """Best-effort logging that can never interrupt normal PatchLab work."""

    try:
        path = runtime_log_path()
        if path.is_file() and path.stat().st_size >= MAX_LOG_BYTES:
            backup = path.with_suffix(".previous.log")
            backup.unlink(missing_ok=True)
            path.replace(backup)
        stamp = datetime.now(timezone.utc).isoformat()
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{stamp}] {message.rstrip()}\n")
    except OSError:
        # A full or disconnected drive must never make the GUI unusable.
        pass
