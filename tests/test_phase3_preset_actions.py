"""Phase 3: Open Preset File Location, Save Copy, and the Generated Preset
Folder setting, driven against the real ``app.ui.MainWindow``.

Covers the remaining items of the Phase 3 test matrix not already exercised
by ``tests/test_preset_save_gui.py`` (A/E/F/N), ``tests/test_preset_output.py``
(D/O/P) or ``tests/test_serum2_only_machine.py`` (B's export-verifier half):

B - Finder reveal uses the exact stored path, never a recomputed one
C - a folder change only affects presets generated after it
G - a saved path whose file is gone gets a clear, specific message
H - a batch's export folder is built from the one shared resolver
I - one failed file in a batch never blocks the others
J - a closest match backed by a real user file reveals that original
K - a closest match with no local file is blocked with a clear reason
L - no "Load in Serum" reference remains anywhere in the product
M - a normal generated result never shows a permanent Export Preset button
Q - the disable-keychain env var actually stops keyring from being touched
"""

from __future__ import annotations

import errno
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import core.bug_report as bug_report
import core.preset_save as preset_save
from core.db import Database
from core.preset_output import PresetOutputPreferences, save_preset_output_preferences
from core.preset_save import IncidentStatus, SaveIncident
from tests.test_gui_release_flows import Gui, gui  # noqa: F401
from tests.test_preset_save_gui import MATCH_UID, _fail, _incident, _mkdir, _worker_failed


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def save_gui(gui: Gui, tmp_path: Path, monkeypatch):  # noqa: F811
    """Same rig as tests/test_preset_save_gui.py's fixture of the same name."""

    monkeypatch.setattr(preset_save, "incidents_root", lambda app_data_dir=None: tmp_path / "incidents")
    monkeypatch.setattr(bug_report, "reports_root", lambda: _mkdir(tmp_path / "reports"))
    window = gui.window
    window.export_runner.start = MagicMock(name="export_runner.start")
    window.incident_report_runner.start = MagicMock(name="incident_report_runner.start")
    database = Database(window._match_database_path())
    database.migrate()
    with database.connect() as connection:
        connection.execute(
            "INSERT INTO match_library(match_uid,source_name,source_audio_path,source_content_hash,"
            "result_json_path,target_synth,budget,similarity_percent,base_name,recommendation_synth,"
            "no_confident_match,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (MATCH_UID, "Bass.wav", str(tmp_path / "Bass.wav"), "h", str(tmp_path / "result.json"),
             "serum2", "best", 91.0, "Base", "serum1", 0, "2026-09-22 10:00:00"),
        )
    (tmp_path / "Bass.wav").write_bytes(b"RIFF....WAVE")
    (tmp_path / "result.json").write_text(json.dumps({"recommendation": {"synth": "serum1"}}))
    window._current_match_uid = MATCH_UID
    gui.database = database
    return gui


# --- B. reveal uses the exact stored path, never recomputed -----------------


def test_B_reveal_uses_the_stored_path_not_a_recomputed_one(save_gui, tmp_path: Path) -> None:
    window = save_gui.window
    final = tmp_path / "Presets" / "PatchLab - Bass.fxp"
    final.parent.mkdir(parents=True)
    final.write_bytes(b"preset")
    save_gui.database.set_match_exported_path(MATCH_UID, final)

    with patch("app.ui.subprocess.run") as run:
        window.open_preset_file_location(MATCH_UID)
    run.assert_called_once_with(["open", "-R", str(final)], check=False)


def test_B_missing_saved_file_shows_clear_message_not_a_silent_no_op(save_gui, tmp_path: Path) -> None:
    window = save_gui.window
    ghost = tmp_path / "Presets" / "gone.fxp"  # recorded but never written / deleted since
    save_gui.database.set_match_exported_path(MATCH_UID, ghost)

    with patch("app.ui.subprocess.run") as run:
        window.open_preset_file_location(MATCH_UID)
    run.assert_not_called()
    assert window.statusBar().currentMessage() == "PatchLab can't find this preset file anymore."


# --- C. changing the folder only affects presets saved after the change -----


def test_C_folder_change_leaves_an_old_saved_result_pointing_at_its_old_path(
    save_gui, tmp_path: Path
) -> None:
    window = save_gui.window
    old_root = tmp_path / "Old Root"
    old_final = old_root / "PatchLab - Bass.fxp"
    old_final.parent.mkdir(parents=True)
    old_final.write_bytes(b"preset")
    save_gui.database.set_match_exported_path(MATCH_UID, old_final)

    new_root = tmp_path / "New Root"
    save_preset_output_preferences(PresetOutputPreferences(folder=str(new_root)))

    with patch("app.ui.subprocess.run") as run:
        window.open_preset_file_location(MATCH_UID)
    run.assert_called_once_with(["open", "-R", str(old_final)], check=False)

    # A newly-generated result, by contrast, must resolve into the new root.
    assert window._patchlab_export_folder("serum2") == new_root


# --- G. a saved path whose file is gone -------------------------------------


def test_G_no_saved_preset_at_all_shows_the_generic_message(save_gui) -> None:
    window = save_gui.window
    with patch("app.ui.QMessageBox") as message_box:
        window.open_preset_file_location(MATCH_UID)
    message_box.information.assert_called_once()
    args = message_box.information.call_args[0]
    assert args[1] == "No saved preset"


# --- H. a batch's export folder comes from the one shared resolver ----------


def test_H_batch_export_folder_is_built_from_the_shared_resolver(save_gui, tmp_path: Path) -> None:
    window = save_gui.window
    custom = tmp_path / "Custom Output"
    save_preset_output_preferences(PresetOutputPreferences(folder=str(custom)))

    resolved = window._patchlab_export_folder("serum2") / "My Batch"
    assert resolved == custom / "My Batch"

    source = (PROJECT_ROOT / "app" / "ui.py").read_text(encoding="utf-8")
    assert 'export_folder = self._patchlab_export_folder(target_synth) / folder_name' in source


# --- I. one failed file in a batch never blocks the others ------------------


def test_I_a_failed_batch_file_advances_to_the_next_one(save_gui, tmp_path: Path) -> None:
    window = save_gui.window
    window._batch_state = {
        "batch_id": 1, "folder_name": "Batch", "source_folder": tmp_path,
        "export_folder": tmp_path / "out", "target_synth": "serum2", "budget": "quick",
        "files": [], "index": 0, "total": 3, "completed": 0, "failed": 0, "skipped": 0,
        "phase": "export", "current_path": tmp_path / "one.wav", "started": 0.0,
        "cancel_requested": False, "prerender": [],
    }
    with patch.object(window, "_start_next_batch_file") as start_next, \
         patch.object(window, "_persist_batch_progress"):
        window._batch_file_failed("disk full")
        # _start_next_batch_file is scheduled via QTimer.singleShot(0, ...),
        # so it has not run yet -- what matters is the batch survives.
        assert window._batch_state is not None, "one failure must not tear down the batch"
        assert window._batch_state["failed"] == 1
        assert window._batch_state["phase"] == "idle", "ready to pick up the next file"


# --- J/K. closest matches: reveal the real file, or explain why not ---------


def test_J_closest_match_with_a_real_local_file_reveals_the_original(save_gui, tmp_path: Path) -> None:
    window = save_gui.window
    original = tmp_path / "My Presets" / "Warm Pad.SerumPreset"
    original.parent.mkdir(parents=True)
    original.write_bytes(b"preset")
    item = {
        "name": "Warm Pad", "synth": "serum2",
        "local_source_available": True, "source_path": str(original),
    }
    with patch("app.ui.subprocess.run") as run:
        window.open_existing_match_location(item)
    run.assert_called_once_with(["open", "-R", str(original)], check=False)


def test_K_closest_match_with_no_local_file_is_blocked_with_a_reason(save_gui) -> None:
    window = save_gui.window
    item = {"name": "Factory Init", "synth": "serum2", "local_source_available": False, "source_path": None}
    with patch("app.ui.subprocess.run") as run, patch("app.ui.QMessageBox") as message_box:
        window.open_existing_match_location(item)
    run.assert_not_called()
    message_box.information.assert_called_once()
    title, message = message_box.information.call_args[0][1:3]
    assert title == "This preset cannot be located"
    assert "not installed on this Mac" in message


# --- L. no "Load in Serum" reference remains anywhere ------------------------


def test_L_no_load_in_serum_reference_remains_in_product_code_or_docs() -> None:
    import subprocess as sp

    result = sp.run(
        ["grep", "-rIln", "Load in Serum\\|load_in_serum",
         "app", "core", "docs", "scripts"],
        cwd=PROJECT_ROOT, capture_output=True, text=True,
    )
    hits = [line for line in result.stdout.splitlines() if line.strip()]
    assert hits == [], f"stale 'Load in Serum' reference(s): {hits}"


# --- M. no permanent Export Preset button for a normal generated result -----


def test_M_no_permanent_export_preset_button_exists(save_gui) -> None:
    window = save_gui.window
    for attribute in ("export_preset_button", "load_in_serum_button"):
        assert not hasattr(window, attribute), f"stale button attribute: {attribute}"
    # The two legitimate save-related actions for a fresh, unsaved result are
    # "Save Preset" (legacy/never-attempted) and "Retry Saving Preset"
    # (recovery) -- never a generic, always-present "Export Preset".
    assert hasattr(window, "save_preset_now_button")
    assert hasattr(window, "retry_save_button")
    assert hasattr(window, "open_preset_location_button")


def test_M_saved_result_shows_open_location_not_export(save_gui, tmp_path: Path) -> None:
    window = save_gui.window
    final = tmp_path / "Presets" / "PatchLab - Bass.fxp"
    final.parent.mkdir(parents=True)
    final.write_bytes(b"preset")
    save_gui.database.set_match_exported_path(MATCH_UID, final)
    window._refresh_save_state()

    assert not window.open_preset_location_button.isHidden()
    assert window.save_preset_now_button.isHidden()
    assert window.retry_save_button.isHidden()


# --- Q. the disable-keychain env var actually stops keyring from being touched


def test_Q_disable_keychain_env_var_is_set_for_every_test_in_this_suite(monkeypatch) -> None:
    import os

    assert os.environ.get("PATCHLAB_DISABLE_KEYCHAIN", "").strip() == "1", (
        "tests/conftest.py must export PATCHLAB_DISABLE_KEYCHAIN=1 so no test "
        "run ever touches the real macOS keychain and triggers an auth prompt"
    )


def test_Q_access_store_never_imports_keyring_under_the_test_env(tmp_path: Path, monkeypatch) -> None:
    import sys

    from core.access_gate import AccessStore

    monkeypatch.setenv("PATCHLAB_DISABLE_KEYCHAIN", "1")
    monkeypatch.setitem(sys.modules, "keyring", None)  # importing it would now raise
    store = AccessStore(marker_path=tmp_path / "access-state.json")
    assert store.keyring is None
