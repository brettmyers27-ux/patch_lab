"""The real MainWindow's side of the generated-preset save lifecycle.

Drives ``app.ui.MainWindow`` offscreen with the real signals its export runner
emits.  Only the worker processes and dialogs are replaced (see
``tests/test_gui_release_flows``); incidents, reports, the Library database and
the widgets are all real.
"""

from __future__ import annotations

import errno
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import core.bug_report as bug_report
import core.preset_save as preset_save
from core.db import Database
from core.fxp import build_fxp
from core.preset_save import IncidentStatus, SaveIncident
from tests.test_gui_release_flows import Gui, gui  # noqa: F401  (the fixture this module builds on)


MATCH_UID = "match-under-test"


@pytest.fixture
def save_gui(gui: Gui, tmp_path: Path, monkeypatch):  # noqa: F811
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


def _mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _incident(tmp_path: Path, **run_again) -> SaveIncident:
    return SaveIncident.create(
        match_uid=MATCH_UID, result_path=tmp_path / "result.json",
        target_path=tmp_path / "Presets" / "PatchLab - Bass.fxp", synth="serum1",
        run_again=run_again or {
            "source_audio_path": str(tmp_path / "Bass.wav"), "target_synth": "serum1",
            "budget": "quick", "start_offset_s": 2.25,
        },
    )


def _fail(incident: SaveIncident, *, trigger: str, error: OSError | Exception, stage: str = "commit",
          status: str = IncidentStatus.FAILED.value) -> dict:
    failure = preset_save.classify_save_failure(error, stage=stage)
    incident.record_attempt(trigger=trigger, stage=stage, target_path=incident.target_path,
                            ok=False, started=0.0, exc=error, failure=failure)
    incident.status = status
    incident.save()
    return {"incident_id": incident.incident_id, "kind": failure.kind.value, "reason": failure.reason,
            "attempts": len(incident.attempts)}


def _worker_failed(window, incident, failure: dict, *, trigger: str) -> None:
    window._save_incident_context = {"incident_id": incident.incident_id, "match_uid": MATCH_UID,
                                     "trigger": trigger}
    window.export_runner.failure = failure
    window.export_runner.failed.emit("PatchLab created your preset, but couldn't save the file.")


def _visible(widget) -> bool:
    return not widget.isHidden()


# --- D/E. a save that succeeds (first time or after recovering) shows nothing extra --


def test_E_recovered_save_records_the_path_and_shows_no_retry_or_report(save_gui, tmp_path) -> None:
    window = save_gui.window
    incident = _incident(tmp_path)
    final = tmp_path / "Presets" / "PatchLab - Bass.fxp"
    final.parent.mkdir(parents=True)
    final.write_bytes(build_fxp(b"state", plugin_id=b"XfsX", program_name="PatchLab"))
    _fail(incident, trigger="auto", error=OSError(errno.EBUSY, "busy"))
    incident.staged = {"sha256": preset_save.sha256_file(final), "verified": True}
    incident.status = IncidentStatus.COMMITTED.value
    incident.save()
    window._save_incident_context = {"incident_id": incident.incident_id, "match_uid": MATCH_UID, "trigger": "auto"}
    window.export_runner.completed.emit({"incident_id": incident.incident_id, "path": str(final), "attempts": 2})

    assert save_gui.database.get_match_library(MATCH_UID).exported_preset_path == final
    assert SaveIncident.load(incident.incident_id).status == "saved"
    assert not _visible(window.retry_save_button) and not _visible(window.save_failure_label)
    window.incident_report_runner.start.assert_not_called()
    assert not SaveIncident.load(incident.incident_id).report, "recovery succeeded: nothing to report"


# --- F. automatic recovery exhausted ---------------------------------------------------


def test_F_exhausted_auto_save_shows_retry_and_sends_exactly_one_report(save_gui, tmp_path) -> None:
    window = save_gui.window
    incident = _incident(tmp_path)
    for _ in range(2):
        _fail(incident, trigger="auto", error=OSError(errno.EIO, "I/O error"))
    failure = _fail(incident, trigger="auto", error=OSError(errno.EIO, "I/O error"))
    _worker_failed(window, incident, failure, trigger="auto")

    assert _visible(window.retry_save_button) and window.retry_save_button.text() == "Retry Saving Preset"
    assert window.retry_save_button.isEnabled()
    assert not _visible(window.run_again_button)
    assert window.save_failure_label.text() == preset_save.AUTOMATIC_FAILURE_MESSAGE
    assert "Traceback" not in window.save_failure_label.text()

    window.incident_report_runner.start.assert_called_once()
    request = Path(window.incident_report_runner.start.call_args.args[0])
    text = request.read_text(encoding="utf-8")
    assert f"Ticket ID: {incident.incident_id}" in text, "the ticket IS the incident"
    assert '"failure_kind": "transient_io"' in text and '"attempt": 3' in text
    stored = SaveIncident.load(incident.incident_id)
    assert stored.report["ticket_id"] == incident.incident_id and stored.report["status"] == "saved_locally"

    # The same failure again (another result, same cause) is not uploaded twice.
    other = _incident(tmp_path)
    failure = _fail(other, trigger="auto", error=OSError(errno.EIO, "I/O error"))
    window._incident_report_active = None
    _worker_failed(window, other, failure, trigger="auto")
    window.incident_report_runner.start.assert_called_once()
    assert SaveIncident.load(other.incident_id).report["ticket_id"] == incident.incident_id


def test_F_the_library_row_offers_retry_for_a_failed_save(save_gui, tmp_path) -> None:
    window = save_gui.window
    incident = _incident(tmp_path)
    _worker_failed(window, incident, _fail(incident, trigger="auto", error=OSError(errno.EIO, "x")), trigger="auto")
    window.refresh_match_library()
    record = save_gui.database.get_match_library(MATCH_UID)
    row = window._build_library_row(record)
    labels = [button.text() for button in row.findChildren(type(window.retry_save_button))]
    assert "Retry Saving Preset" in labels and "Rename Preset" not in labels


# --- G. manual retry ----------------------------------------------------------------


def test_G_retry_saves_the_same_result_without_rerunning_match(save_gui, tmp_path) -> None:
    window = save_gui.window
    incident = _incident(tmp_path)
    _worker_failed(window, incident, _fail(incident, trigger="auto", error=OSError(errno.EIO, "x")), trigger="auto")
    save_gui.click(window.retry_save_button)

    window.export_runner.start.assert_called_once()
    call = window.export_runner.start.call_args
    assert call.kwargs == {"incident_id": incident.incident_id, "trigger": "manual", "attempts": 1}
    assert Path(call.args[1]) == Path(incident.target_path), "same destination, same result"
    window.match_runner.start.assert_not_called()
    assert window.retry_save_button.text() == "Saving…" and not window.retry_save_button.isEnabled()

    final = Path(incident.target_path)
    final.parent.mkdir(parents=True, exist_ok=True)
    final.write_bytes(build_fxp(b"state", plugin_id=b"XfsX", program_name="PatchLab"))
    stored = SaveIncident.load(incident.incident_id)
    stored.staged = {"sha256": preset_save.sha256_file(final), "verified": True}
    stored.status = IncidentStatus.COMMITTED.value
    stored.save()
    window.export_runner.completed.emit({"incident_id": incident.incident_id, "path": str(final), "attempts": 2})

    assert save_gui.database.get_match_library(MATCH_UID).exported_preset_path == final
    assert not _visible(window.retry_save_button) and not _visible(window.save_failure_label)


# --- H. manual retry fails ----------------------------------------------------------------


def test_H_failed_manual_retry_explains_and_files_one_linked_follow_up(save_gui, tmp_path) -> None:
    window = save_gui.window
    incident = _incident(tmp_path)
    _worker_failed(window, incident, _fail(incident, trigger="auto", error=OSError(errno.EACCES, "denied")),
                   trigger="auto")
    window._incident_report_active = None
    incident = SaveIncident.load(incident.incident_id)  # as the retry worker would
    _worker_failed(window, incident, _fail(incident, trigger="manual", error=OSError(errno.EACCES, "denied")),
                   trigger="manual")

    assert window.save_failure_label.text().startswith(
        "PatchLab still can't save this preset because this folder isn't writable."
    )
    assert _visible(window.retry_save_button), "the user can retry after fixing the folder"
    stored = SaveIncident.load(incident.incident_id)
    assert stored.followup["original_ticket_id"] == incident.incident_id
    assert stored.followup["ticket_id"] != incident.incident_id
    followup_text = Path(stored.followup["request_path"]).read_text(encoding="utf-8")
    assert "manual recovery also failed" in followup_text
    assert f"Original ticket: {incident.incident_id}" in followup_text
    assert window.incident_report_runner.start.call_count == 2

    window._incident_report_active = None
    incident = SaveIncident.load(incident.incident_id)
    _worker_failed(window, incident, _fail(incident, trigger="manual", error=OSError(errno.EACCES, "denied")),
                   trigger="manual")
    assert window.incident_report_runner.start.call_count == 2, "one follow-up, not one per click"


# --- J. unrecoverable result: Run Again with the original setup ------------------------------


def test_J_unrecoverable_result_offers_run_again_with_the_same_setup(save_gui, tmp_path) -> None:
    window = save_gui.window
    save_gui.select_target("serum2")
    window.match_budget.setCurrentIndex(2)
    incident = _incident(tmp_path)
    failure = _fail(incident, trigger="rebuild", error=RuntimeError("graph mismatch"), stage="validate",
                    status=IncidentStatus.UNRECOVERABLE.value)
    _worker_failed(window, incident, failure, trigger="manual")

    assert _visible(window.run_again_button) and not _visible(window.retry_save_button)
    assert window.save_failure_label.text() == preset_save.UNRECOVERABLE_MESSAGE
    window.match_runner.start.assert_not_called(), "never launched without the user asking"

    save_gui.click(window.run_again_button)
    window.match_runner.start.assert_called_once()
    call = window.match_runner.start.call_args
    assert Path(call.args[0]) == (tmp_path / "Bass.wav").resolve()
    assert call.kwargs["target_synth"] == "serum1"
    assert call.kwargs["budget"] == "quick"
    assert call.kwargs["offset"] == pytest.approx(2.25)


# --- the auto-save entry point opens one incident and asks for three attempts ---------------


def test_auto_save_opens_an_incident_with_three_attempts(save_gui, tmp_path) -> None:
    window = save_gui.window
    window._match_result = {
        "recommendation": {"synth": "serum2"},
        "source": {"start_offset_s": 0.5},
    }
    archived = MagicMock()
    archived.record.match_uid = MATCH_UID
    archived.record.target_synth = "serum2"
    archived.record.budget = "balanced"
    archived.result_json_path = tmp_path / "result.json"
    archived.source_audio_path = tmp_path / "Bass.wav"
    window._auto_save_generated_preset(archived, tmp_path / "Bass.wav")

    call = window.export_runner.start.call_args
    assert call.kwargs["trigger"] == "auto" and call.kwargs["attempts"] == 3
    incident = SaveIncident.load(call.kwargs["incident_id"])
    assert incident.match_uid == MATCH_UID and incident.synth == "serum2"
    assert incident.run_again == {
        "source_audio_path": str(tmp_path / "Bass.wav"), "target_synth": "serum2",
        "budget": "balanced", "start_offset_s": 0.5,
    }
    assert not _visible(window.retry_save_button), "nothing shows while a normal save runs"
