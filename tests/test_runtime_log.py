from __future__ import annotations

from pathlib import Path

from core import runtime_log


def test_runtime_log_rotates_without_raising(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "patchlab-runtime.log"
    monkeypatch.setattr(runtime_log, "runtime_log_path", lambda: path)
    monkeypatch.setattr(runtime_log, "MAX_LOG_BYTES", 12)

    runtime_log.append_runtime_log("first")
    runtime_log.append_runtime_log("second")

    assert path.is_file()
    assert "second" in path.read_text(encoding="utf-8")
    assert path.with_suffix(".previous.log").is_file()
