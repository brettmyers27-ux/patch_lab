"""Match-page product actions, worker pipe safety, and the 1.5.4 -> 1.5.5 update.

Covers the rest of the reported beta failures: exporting a closest match at all,
a cancelled library job that used to die with BrokenPipeError, and an update that
must refuse to start rather than fill a nearly-full disk.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from core.update_check import (
    UPDATE_SPACE_HEADROOM_BYTES,
    macos_package_releases,
    newest_macos_package,
    update_available,
    update_space_preflight,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_BYTES = 3_408_203_409  # the real 1.5.5 installer size class


# ---------------------------------------------------------------------------
# Exporting a closest match (C) reuses the one verified implementation
# ---------------------------------------------------------------------------


def _result_with_matches(tmp_path: Path, *, installed: Path | None) -> Path:
    result = {
        "status": "complete",
        "factory_only": True,
        "existing_matches": [
            {
                "preset_id": 7, "content_hash": "a" * 40, "name": "Installed Pad",
                "synth": "serum2", "source_path": str(installed) if installed else "",
                "local_source_available": bool(installed), "similarity": 0.91,
            },
            {
                "preset_id": 8, "content_hash": "b" * 40, "name": "Absent Lead",
                "synth": "serum2", "source_path": "/nope/Absent Lead.SerumPreset",
                "local_source_available": False, "similarity": 0.83,
            },
        ],
        "recommendation": {"synth": "serum2", "content_hash": "a" * 40},
    }
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    return path


def test_a_closest_match_maps_onto_the_verified_exact_copy_export(tmp_path: Path) -> None:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from export_match import existing_match_as_recommendation

    installed = tmp_path / "Installed Pad.SerumPreset"
    installed.write_bytes(b"{}")
    result = json.loads(_result_with_matches(tmp_path, installed=installed).read_text())

    mapped = existing_match_as_recommendation(result, 0)
    assert mapped["factory_source_path"] == str(installed)
    assert mapped["synth"] == "serum2" and mapped["content_hash"] == "a" * 40

    with pytest.raises(RuntimeError) as raised:
        existing_match_as_recommendation(result, 1)
    assert "not installed on this Mac" in str(raised.value)
    with pytest.raises(RuntimeError):
        existing_match_as_recommendation(result, 99)


def test_the_export_worker_accepts_the_closest_match_selector() -> None:
    proc = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts/export_match.py"), "--help"],
        capture_output=True, text=True, cwd=PROJECT_ROOT,
    )
    assert proc.returncode == 0 and "--existing-match" in proc.stdout


def test_the_export_runner_passes_the_selector_through(tmp_path: Path) -> None:
    from unittest.mock import MagicMock

    from PySide6.QtCore import QProcess

    from app.workers import ExportProcessRunner

    runner = ExportProcessRunner.__new__(ExportProcessRunner)
    runner._buffer = ""
    runner._result = None
    runner._error = None
    runner.process = MagicMock()
    runner.process.state.return_value = QProcess.ProcessState.NotRunning
    runner._start_worker = MagicMock()
    ExportProcessRunner.start(runner, tmp_path / "r.json", tmp_path / "o.SerumPreset", existing_match=3)
    worker, arguments = runner._start_worker.call_args.args
    assert worker == "export" and arguments[-2:] == ["--existing-match", "3"]

    runner._start_worker.reset_mock()
    ExportProcessRunner.start(runner, tmp_path / "r.json", tmp_path / "o.SerumPreset")
    assert "--existing-match" not in runner._start_worker.call_args.args[1]


# ---------------------------------------------------------------------------
# A cancelled worker: the parent closing the pipe is not a crash
# ---------------------------------------------------------------------------


def test_a_closed_pipe_ends_a_worker_quietly(tmp_path: Path) -> None:
    """Regression: progress printing raised BrokenPipeError when the GUI cancelled."""

    script = tmp_path / "run.py"
    script.write_text(
        "import sys, time\n"
        f"sys.path.insert(0, {str(PROJECT_ROOT)!r})\n"
        "from core.worker_runtime import emit\n"
        "sys.stderr.write('READY\\n'); sys.stderr.flush()\n"
        "time.sleep(0.5)\n"
        "results = [emit('LOCAL_LIBRARY_PROGRESS=' + 'x' * 100000) for _ in range(40)]\n"
        "sys.stderr.write('STOPPED=%s\\n' % results.count(False))\n"
        "print('after the pipe died')\n"
        "sys.stderr.write('SURVIVED\\n')\n",
        encoding="utf-8",
    )
    proc = subprocess.Popen(
        [sys.executable, str(script)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    assert proc.stderr.readline().strip() == "READY"
    proc.stdout.close()  # exactly what the GUI does when it cancels the job
    _out, err = proc.communicate(timeout=60)
    assert proc.returncode == 0, f"exit {proc.returncode}: {err}"
    assert "SURVIVED" in err, err
    assert "STOPPED=" in err and "STOPPED=0" not in err, "emit must report the dead pipe"
    assert "BrokenPipeError" not in err and "Traceback" not in err
    assert "Exception ignored" not in err, "interpreter shutdown must stay clean too"


def test_the_dispatcher_treats_a_dead_parent_as_a_clean_end() -> None:
    source = (PROJECT_ROOT / "app/worker_dispatch.py").read_text(encoding="utf-8")
    assert "except BrokenPipeError:" in source
    assert "return PARENT_GONE_EXIT" in source
    index = source.index("except BrokenPipeError:")
    assert source.index("except BaseException") > index, "handled before the generic catch"


def test_the_library_worker_prints_progress_through_the_safe_emitter() -> None:
    source = (PROJECT_ROOT / "scripts/process_local_library.py").read_text(encoding="utf-8")
    assert 'emit("LOCAL_LIBRARY_PROGRESS=' in source
    assert 'print(\n            "LOCAL_LIBRARY_PROGRESS=' not in source


# ---------------------------------------------------------------------------
# J. the installed 1.5.4 updater sees 1.5.5
# ---------------------------------------------------------------------------


def test_J_a_1_5_4_client_sees_1_5_5_as_newer() -> None:
    rows = [{
        "name": "PatchLab-1.5.5-macOS.pkg", "version": "1.5.5", "size": PACKAGE_BYTES,
        "sha256": "c" * 64, "kind": "macos-package",
    }]
    assert update_available("1.5.4", "1.5.5") is True
    assert update_available("1.5.5", "1.5.5") is False
    release = newest_macos_package(rows, "1.5.4")
    assert release is not None and release.version == "1.5.5" and release.size == PACKAGE_BYTES
    assert newest_macos_package(rows, "1.5.5") is None, "the new build offers no update to itself"
    assert [r.name for r in macos_package_releases(rows)] == ["PatchLab-1.5.5-macOS.pkg"]


def test_J_a_malformed_catalog_row_never_becomes_an_update_prompt() -> None:
    for bad in (
        {"name": "PatchLab-1.5.5-macOS.pkg", "version": "1.5.6", "size": 10, "sha256": "c" * 64, "kind": "macos-package"},
        {"name": "PatchLab-1.5.5-macOS.pkg", "version": "1.5.5", "size": 0, "sha256": "c" * 64, "kind": "macos-package"},
        {"name": "PatchLab-1.5.5-macOS.pkg", "version": "1.5.5", "size": 10, "sha256": "short", "kind": "macos-package"},
        {"name": "evil.pkg", "version": "1.5.5", "size": 10, "sha256": "c" * 64, "kind": "macos-package"},
    ):
        assert macos_package_releases([bad]) == []


# ---------------------------------------------------------------------------
# K. insufficient disk space refuses BEFORE downloading
# ---------------------------------------------------------------------------


def test_K_the_reporting_testers_free_space_is_refused(tmp_path: Path, monkeypatch) -> None:
    """Their Mac had ~4.4 GiB free for a 3.4 GB package: enough to store it and
    nothing else, so an update that started could not have completed."""

    import shutil

    monkeypatch.setattr(shutil, "disk_usage", lambda _p: type("U", (), {"free": int(4.7e9)})())
    check = update_space_preflight(PACKAGE_BYTES, destination=tmp_path)
    assert not check.sufficient
    message = check.user_message()
    assert "4.7 GB" in message and "3.4 GB" in message
    assert "free up some space" in message.casefold()
    assert check.as_dict()["sufficient"] is False


def test_K_ample_space_is_allowed(tmp_path: Path, monkeypatch) -> None:
    import shutil

    monkeypatch.setattr(shutil, "disk_usage", lambda _p: type("U", (), {"free": int(40e9)})())
    assert update_space_preflight(PACKAGE_BYTES, destination=tmp_path).sufficient


def test_K_the_requirement_exceeds_the_package_with_real_headroom(tmp_path: Path, monkeypatch) -> None:
    import shutil

    monkeypatch.setattr(shutil, "disk_usage", lambda _p: type("U", (), {"free": 0})())
    check = update_space_preflight(PACKAGE_BYTES, destination=tmp_path)
    assert check.required_bytes >= PACKAGE_BYTES * 2 + UPDATE_SPACE_HEADROOM_BYTES - 1
    # Room for the download, the expanded payload, and the replacement app.
    assert check.required_bytes > 2 * PACKAGE_BYTES


def test_K_a_missing_destination_still_measures_its_nearest_existing_parent(tmp_path: Path) -> None:
    check = update_space_preflight(1024, destination=tmp_path / "a" / "b" / "updates")
    assert check.available_bytes > 0 and check.sufficient


def test_K_the_ui_checks_space_before_starting_a_download() -> None:
    source = (PROJECT_ROOT / "app/ui.py").read_text(encoding="utf-8")
    body = source.split("def _download_package_update(")[1].split("def _update_download_progress(")[0]
    assert "update_space_preflight(" in body
    assert body.index("update_space_preflight(") < body.index("self.update_download_runner.start")
    assert "Not enough free space" in body


def test_the_update_ready_dialog_says_installer_must_be_completed() -> None:
    source = (PROJECT_ROOT / "app/ui.py").read_text(encoding="utf-8")
    body = source.split("def _update_download_completed(")[1].split("def append_log(")[0]
    assert "macOS Installer" in body and "click through it" in body
